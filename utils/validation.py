"""
    training, valida, predict tools

"""
import torch
import sys
from utils.config import Config
import gc
from data.BasisDataset import BaseDataset
from utils.message import Msg
from torch.utils.data import DataLoader, random_split
from torch import optim
from torch import nn
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils.crop import read_image_json
import logging
import os
import torch.nn.functional as F
import numpy as np
from utils.metrics import compute_metrics_for_label, compute_disc_for_slice
from configuration.labels import KITS_HEC_LABEL_MAPPING, HEC_NAME_LIST, HEC_SD_TOLERANCES_MM, GT_SEGM_FNAME
from utils.pre_pro import preprocess, data_crop
sys.path.append("../")
from threading import Thread
import cv2
from data.kits_3D_dataset import KitsDataset3D
import math


class Validation:
    def __init__(self):
        pass

    def training(self, model, config: Config):
        """
        :param model: training model
        :param config: user config
        :return: None
        """
        msg = Msg()
        msg.training_conf(config)
        # dataset
        dataset = BaseDataset(config.image_root, config.mask_root, transform=3, pre_pro=config.pro_pre)
        if config.second_network:
            var_dataset = BaseDataset(config.val_image_crop, config.val_mask_crop,
                                      pre_pro=config.pro_pre)
        else:
            var_dataset = BaseDataset(config.val_image_path, config.val_mask_path,
                                      is_val=True)
        # n_val = "30 cases"
        n_train = len(dataset)
        Msg.num_dataset(n_train, len(var_dataset))

        train = dataset
        train_loader = DataLoader(train, batch_size=config.batch_size,
                                  shuffle=True, num_workers=config.num_workers)

        # training
        # optimizer
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)
        # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=1)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, config.epochs-10], gamma=0.1)
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        # visualization
        # writer = SummaryWriter()
        summary_title = f'{config.network}_{config.optimizer}_{config.loss}'
        if config.weight:
            summary_title = f'{config.network}_{config.optimizer}_weight'
        writer = SummaryWriter(comment=summary_title)

        if config.loss == 0:
            # weights = torch.FloatTensor(config.entropy_weight).to(device=config.device)
            # criterion = nn.CrossEntropyLoss(weight=weights)
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.BCEWithLogitsLoss()
        # begin training
        global_step = 0

        for epoch in range(config.epochs):
            model.train()
            epoch_loss = 0
            with tqdm(total=n_train, desc=f'Epoch{epoch+1}/{config.epochs}', unit="img") as pbar:
                for batch in train_loader:
                    # read image
                    imgs = batch['image']
                    masks = batch['mask']

                    imgs = imgs.to(device=config.device, dtype=torch.float32)
                    # when use softmax type have to
                    mask_type = torch.float32 if config.loss == 1 else torch.long
                    true_masks = masks.to(device=config.device, dtype=mask_type)
                    if config.loss == 0:
                        true_masks = true_masks.squeeze(dim=1)
                    masks_pred = model(imgs)
                    # print("train", true_masks.shape, masks_pred.shape)
                    # print(imgs.max())
                    # focal loss or other
                    # masks_pred = masks_pred.argmax(dim=1)
                    loss = criterion(masks_pred, true_masks)
                    epoch_loss += loss
                    # print(loss)
                    pbar.set_postfix(**{'loss (batch)': loss.item()})

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_value_(model.parameters(), 0.1)
                    optimizer.step()

                    pbar.update(imgs.shape[0])
                    del imgs, masks, true_masks, masks_pred
                    gc.collect()
                global_step += 1
                # evaluation ,write data
                # if global_step % (n_train // (5 * config.batch_size)) == 0:
                #     for tag, value in model.named_parameters():
                #         tag = tag.replace('.', '/')
                #         writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                #         writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                # evaluation
                # valuation Loss
                # print("val")
                val_score, val_dice_sds = self.eval_net(model, config)
                writer.add_scalar(f'{summary_title}/learning_rate', optimizer.param_groups[0]['lr'], global_step)
                disc_mean, sds_mean = np.mean(val_dice_sds, axis=0)
                writer.add_scalar(f'{summary_title}/Loss/train', epoch_loss.item()/len(train_loader), global_step)

                writer.add_scalars(f'{summary_title}/disc_sds', {"disc": disc_mean, "sds": sds_mean},
                                   global_step)
                writer.add_scalars(f'{summary_title}/per_disc_sds',
                                   {"class1_disc": val_dice_sds[0][0], "class1_sds": val_dice_sds[0][1],
                                    "class2_disc": val_dice_sds[1][0], "class2_sds": val_dice_sds[1][1],
                                    "class3_disc": val_dice_sds[2][0], "class3_sds": val_dice_sds[2][1]},
                                   global_step)
                scheduler.step(val_score)
                # get kits ds and sd
                # val_ds_sd = self.kits_valuate(model, val_loader, config)
                logging.info('Validation cross entropy: {}'.format(val_score))
                logging.info('Validation mean disc: \n {}'.format(val_dice_sds))

                writer.add_scalar(f'{summary_title}/Loss/test', val_score, global_step)
            # save model
            try:
                os.mkdir(summary_title)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(model.state_dict(),
                       f"{summary_title}/CP_epoch{epoch + 1}.pth")
            logging.info(f'Checkpoint {epoch + 1} saved !')

        writer.close()

    def training_3d(self, model, config: Config):
        """
        :param model: training model
        :param config: user config
        :return: None
        """
        msg = Msg()
        msg.training_conf(config)
        # dataset
        train = KitsDataset3D(config)
        val = KitsDataset3D(config, is_train=False)
        # train = train[:1]
        n_train = len(train)
        n_val = len(val)
        msg.num_dataset(n_train, n_val)
        train_loader = DataLoader(train, batch_size=config.batch_size,
                                  shuffle=True, num_workers=config.num_workers)
        val_loader = DataLoader(val, batch_size=config.batch_size,
                                shuffle=False, num_workers=config.num_workers)
        # training
        # optimizer
        # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=1)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, config.epochs-10], gamma=0.1)
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        # visualization
        # writer = SummaryWriter()
        summary_title = f'{config.network}_{config.optimizer}_{config.loss}'
        if config.weight:
            summary_title = f'{config.network}_{config.optimizer}_weight'
        writer = SummaryWriter(comment=summary_title)

        if config.loss == 0:
            # weights = torch.FloatTensor(config.entropy_weight).to(device=config.device)
            # criterion = nn.CrossEntropyLoss(weight=weights)
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.BCEWithLogitsLoss()
        # begin training
        global_step = 0

        for epoch in range(config.epochs):
            model.train()
            epoch_loss = 0
            with tqdm(total=n_train, desc=f'Epoch{epoch+1}/{config.epochs}', unit="img") as pbar:
                for batch in train_loader:
                    # read image
                    imgs = batch['image']
                    masks = batch['mask']

                    imgs = imgs.to(device=config.device, dtype=torch.float32)
                    # when use softmax type have to
                    mask_type = torch.float32 if config.loss == 1 else torch.long
                    true_masks = masks.to(device=config.device, dtype=mask_type)
                    if config.loss == 0:
                        true_masks = true_masks.squeeze(dim=1)
                    masks_pred = model(imgs)
                    # print("train", true_masks.shape, masks_pred.shape)
                    # print(imgs.max())
                    # focal loss or other
                    # masks_pred = masks_pred.argmax(dim=1)
                    # print(masks_pred.shape, true_masks.max())
                    loss = criterion(masks_pred, true_masks)
                    epoch_loss += loss
                    # print(loss)
                    pbar.set_postfix(**{'loss (batch)': loss.item()})

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_value_(model.parameters(), 0.1)
                    optimizer.step()

                    pbar.update(imgs.shape[0])
                    del imgs, masks, true_masks, masks_pred
                    gc.collect()
                global_step += 1

                val_score, val_dice_sds = self.eval_net(model, config, val_loader)
                writer.add_scalar(f'{summary_title}/learning_rate', optimizer.param_groups[0]['lr'], global_step)
                disc_mean, sds_mean = np.mean(val_dice_sds, axis=0)
                writer.add_scalar(f'{summary_title}/Loss/train', epoch_loss.item()/len(train_loader), global_step)

                writer.add_scalars(f'{summary_title}/disc_sds', {"disc": disc_mean, "sds": sds_mean},
                                   global_step)
                writer.add_scalars(f'{summary_title}/per_disc_sds',
                                   {"class1_disc": val_dice_sds[0][0], "class1_sds": val_dice_sds[0][1],
                                    "class2_disc": val_dice_sds[1][0], "class2_sds": val_dice_sds[1][1],
                                    "class3_disc": val_dice_sds[2][0], "class3_sds": val_dice_sds[2][1]},
                                   global_step)
                scheduler.step()
                # get kits ds and sd
                # val_ds_sd = self.kits_valuate(model, val_loader, config)
                logging.info('Validation cross entropy: {}'.format(val_score))
                logging.info('Validation mean disc: \n {}'.format(val_dice_sds))

                writer.add_scalar(f'{summary_title}/Loss/test', val_score, global_step)
            # save model
            try:
                os.mkdir(summary_title)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(model.state_dict(),
                       f"{summary_title}/CP_epoch{epoch + 1}.pth")
            logging.info(f'Checkpoint {epoch + 1} saved !')

        writer.close()

    def eval_net(self, model, config: Config, val_loader, is_training=True) -> float:
        """
        evaluate the train network
        :param self:
        :param model:
        :param val_loader:
        :param config: cpu or cuda
        :param is_training:training evaluate
        :return: score loss and kits disc and sds(3(disc)+3(sds)+2(mean))
        """
        kits_metrics = np.zeros((len(HEC_NAME_LIST), 2), dtype=float)
        model.eval()
        mask_type = torch.long

        # with tqdm(total=n_val, desc='Validation round', unit='case', leave=False) as pbar:
        tot = 0
        for case in val_loader:
            imgs, true_masks = case['image'], case['mask']
            imgs = imgs.to(device=config.device, dtype=torch.float32)
            true_masks = true_masks.to(device=config.device, dtype=mask_type)
            spacing = case['spacing']

            with torch.no_grad():
                case_pred = model(imgs)

            true_masks = true_masks.squeeze(dim=1)

            tot += F.cross_entropy(case_pred, true_masks).item()
            # print(true_masks.shape, case_pred.shape)
            case_pred = case_pred.argmax(dim=1)
            # 将整个case合并
            # images, masks = crop_images()
            # print(case_pred.shape, true_masks.shape, spacing.shape)
            kits_metrics += self.calculate_metrics(case_pred.cpu().numpy(),
                                                   true_masks.cpu().numpy(),
                                                   spacing.cpu().numpy())

            gc.collect()
            # pbar.update()
        if is_training:
            model.train()
        return tot / len(val_loader), kits_metrics / len(val_loader)

    def predict(self, model, image: np.ndarray, config: Config):
        """
        predict one image
        :param model: predict model
        :param image: a image numpy:shape[c,h,w]
        :param config: user configure
        :return: predict image shape [h,w]
        """
        model.eval()
        # print(image.shape)
        image = torch.FloatTensor(np.expand_dims(image, 0))
        image = image.to(device=config.device)
        # print(image.shape)
        with torch.no_grad():
            predict_img = model(image)
            if config.loss == 0:
                image = predict_img.argmax(dim=1)
            # 4D s*c*h*w
        output = torch.squeeze(image)
        output = output.float()

        if config.device == "cpu":
            return output.numpy()
        # print(output.cpu().numpy())
        return output.cpu().numpy()

    def predict_all(self, model, images, config, old_spacing=None):
        """
        predict a patient CT image
        :param model:
        :param images: [s,h,w]
        :param config:
        :param old_spacing:
        :return: [slices or cube]: [n,32,384,384] or [n,64,128,128]
        """
        # crop slice to 32*384*384 or 64*128*128
        new_spacing = []
        if config.coarse:
            images, new_spacing = preprocess(config, images, old_spacing)
            images = data_crop(images, config.crop_xy)
        if type(images) == np.ndarray:
            images = torch.from_numpy(images).float()
        if type(images) != torch.FloatTensor:
            images = images.float()
        # print(images.shape)
        # reshape to target shape [1,1,s,w,h]
        z_thick = images.shape[0]
        target_gen_size = config.layer_thick
        range_val = int(math.ceil((z_thick - target_gen_size) / config.layer_thick) + 1)
        images_slice = torch.zeros([range_val, config.layer_thick, config.crop_xy[0],  config.crop_xy[1]])
        # print(images_slice.shape, images.shape)
        stride = config.layer_thick
        for i in range(range_val):
            start_num = i * stride
            end_num = start_num + target_gen_size
            if end_num <= z_thick:
                # 数据块长度没有超出x轴的范围,正常取块
                images_slice[i] = images[start_num:end_num, :, :]
            else:
                # 数据块长度超出x轴的范围, 从最后一层往前取一个 batch_gen_size 大小的块作为本次获取的数据块

                p = end_num - z_thick
                # print(p, z_thick)
                # images_slice[i] = images[(z_thick - target_gen_size):z_thick, :, :]
                last_slices = images[start_num:z_thick, :, :]
                images_slice[i] = torch.from_numpy(np.pad(np.array(last_slices), ((0, p), (0, 0), (0, 0)), 'minimum'))

        if len(images_slice.shape) != 5:
            images_slice = torch.unsqueeze(images_slice, dim=1)
        # predict target images cube
        model.eval()
        images_slice = images_slice.to(device=config.device)
        # print(images_slice.dtype, config.device)
        with torch.no_grad():
            case_pred = model(images_slice)

        case_pred = case_pred.argmax(dim=1)
        # 将整个case合并
        case_pred = case_pred.squeeze(dim=1)
        return torch.tensor(case_pred, dtype=torch.int8), images, new_spacing, z_thick

    @staticmethod
    def calculate_metrics(masks, ground_trues, spacing=None, second_network=False):
        """
        :param ground_trues: [b,d,w,h]
        :param masks: [b,d,w,h]
        :param spacing: [b,d,w,h]
        :return: ndarray:[3,2]
        """
        # calculate per metrics
        # 数据量大，计算慢（多线程）：多个线程一起结束
        metrics_thread = list()
        bs_metrics = np.zeros((len(HEC_NAME_LIST), 2), dtype=float)
        # 遍历每个batch size的矩阵
        # print("gt_len: "+str(len(ground_trues)))
        # for per_case in range(len(ground_trues)):
        metrics_case = np.zeros((len(HEC_NAME_LIST), 2), dtype=float)
        for cube in range(len(masks)):
            # 可以每个cube创建一个线程计算指标：一共有batch_size 个线程
            metrics_thread.append(Metrics_thread(masks[cube],
                                                 ground_trues[cube],
                                                 tuple(spacing[cube])))
        for i in range(len(masks)):
            metrics_thread[i].start()
        for i in range(len(masks)):
            metrics_thread[i].join()
            # print("end")
        dice_1_counter = 0
        for i in range(len(masks)):
            result = metrics_thread[i].get_result()
            if result[2][0] == 1:
                result[2][0] = 0
                dice_1_counter += 1
            bs_metrics += result
        if len(masks) == dice_1_counter:
            dice_mean = 0
        else:
            dice_mean = bs_metrics[2][0] / (len(masks) - dice_1_counter)
        bs_metrics = bs_metrics / len(masks)
        bs_metrics[2][0] = dice_mean
        # print(dice_1_counter)
        return bs_metrics

    def network_optimizer(self):
        pass

    @staticmethod
    def choose_network(config: Config):
        if config.optimizer == "SGD":
            pass


class Metrics_thread(Thread):
    def __init__(self, ground_trues, masks, spacing):
        Thread.__init__(self)
        self.result = 0
        self.ground_trues = ground_trues
        self.masks = masks
        self.spacing = spacing

    def run(self):
        metrics_case = np.zeros((len(HEC_NAME_LIST), 2), dtype=float)
        for i, hec in enumerate(HEC_NAME_LIST):
            metrics_case[i] = compute_metrics_for_label(self.masks, self.ground_trues,
                                                        KITS_HEC_LABEL_MAPPING[hec],
                                                        tuple(self.spacing),
                                                        sd_tolerance_mm=HEC_SD_TOLERANCES_MM[hec])
        self.result = metrics_case

    def get_result(self):
        return self.result


def get_cube(image, mask, crop_info: dict):
    """
    get crop image cube
    :param image: [n,1,h,w]
    :param mask: [n,1,h,w]
    :param crop_info: {slice,i,bbox}
    :return: image， mask([n,1,h,w])
    """
    config = Config()
    image_crop = torch.zeros((len(crop_info), 1, 256, 256)).to(config.device)
    mask_crop = torch.zeros((len(crop_info), 1, 256, 256)).to(config.device)
    # print(image.shape, mask.shape, image_crop.shape)

    for index in range(len(crop_info)):
        info = crop_info[index]
        bbox = info["bbox"]
        # print(info["slice"], bbox)
        # print(image_crop[index][0].shape, image[int(info["slice"])][0][bbox[0]:bbox[1], bbox[2]:bbox[3]].shape, index)
        image_crop[index][0].add_(image[int(info["slice"])][0][bbox[0]:bbox[1], bbox[2]:bbox[3]])
        mask_crop[index][0].add_(mask[int(info["slice"])][0][bbox[0]:bbox[1], bbox[2]:bbox[3]])

    return image_crop, mask_crop.type(torch.long)


def combine_image(mask, origin_shape, crop_info):
    """
    :param mask: [n,h1,w1]
    :param origin_shape: [origin_n,1,h,w]
    :param crop_info: [{slice index bbox}{}]
    :return:
    """
    config = Config()
    origin_shape = list(origin_shape)
    if len(origin_shape) == 3:
        origin_shape.insert(1, 1)
    print(origin_shape)
    combine_mask = np.zeros(origin_shape, np.uint8)
    print(combine_mask.shape)
    # print(mask.shape, origin_shape)
    assert len(mask.shape) == 3, "mask shape is not 3"
    # print(len(crop_info), len(mask))

    for index in range(len(mask)):
        info = crop_info[index]
        bbox = info["bbox"]
        # mask to 512*512 mask
        mask_temp = np.zeros_like(combine_mask[info["slice"]][0], dtype=np.uint8)
        # print(info["slice"], bbox)
        # show_views(image[index][0], mask[index][0])
        # print(bbox,mask_temp.shape, mask.shape, mask_temp[bbox[0]:bbox[1], bbox[2]:bbox[3]].shape)
        mask_temp[bbox[0]:bbox[1], bbox[2]:bbox[3]] = mask[index].cpu()
        combine_mask[info["slice"]][0] = cv2.bitwise_or(combine_mask[info["slice"]][0], mask_temp)
        # show_views(combine_mask[info["slice"]][0], mask_temp,
        #            image_title=["combine_mask", "mask_temp"])
#         break
    return torch.from_numpy(combine_mask).to(config.device)


if __name__ == "__main__":
    config = Config()
    var_dataset = BaseDataset(config.kits_val_image_path, config.kits_val_mask_path,
                              is_val=True)
    crop_infos = read_image_json(config.image_json)
    print(var_dataset.ids[1])
    from utils.visualization import show_views
    crop_info = crop_infos[var_dataset.ids[1]]
    # print(crop_info)
    image = var_dataset[1]["image"]
    mask = var_dataset[1]["mask"]
    show_views(image[0][0], mask[0][0])
    crop_image, crop_mask = get_cube(image, mask, crop_info)

    show_views(crop_mask[0][0], crop_image[0][0])
    combine_pre = combine_image(crop_mask, image.shape, crop_info)
    # show_views(mask[1][0], combine_pre[1][0])
    # print((mask[1][0] != combine_pre[1][0]).sum())
    print(mask.shape, combine_pre.shape)
    for i in range(combine_pre.shape[0]):
        if (mask[i][0] != combine_pre[i][0]).sum() != 0:
            print(i)
            show_views(mask[i][0], combine_pre[i][0])
    print((mask != combine_pre).sum())
