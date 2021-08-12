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
from data.kits_dataset import KitsDataset
from utils.crop import read_image_json
import logging
import os
import torch.nn.functional as F
import numpy as np
from utils.metrics import compute_metrics_for_label, compute_disc_for_slice
from configuration.labels import KITS_HEC_LABEL_MAPPING, HEC_NAME_LIST, HEC_SD_TOLERANCES_MM, GT_SEGM_FNAME
sys.path.append("../")
from threading import Thread
import cv2
from loss.loss_function import Focal_loss, DiceLoss
from generalized_wasserstein_dice_loss.loss import GeneralizedWassersteinDiceLoss


class Validation:
    def __init__(self):
        pass

    def training(self, model,
                 config: Config):
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

    def eval_net(self, model, config: Config, is_training=True) -> float:
        """
        evaluate the train network
        :param self:
        :param model:
        :param config: cpu or cuda
        :param is_training:training evaluate
        :return: score loss and kits disc and sds(3(disc)+3(sds)+2(mean))
        """
        kits_metrics = np.zeros((len(HEC_NAME_LIST), 2), dtype=float)
        model.eval()
        mask_type = torch.long
        if config.second_network:
            var_dataset = BaseDataset(config.val_image_path, config.val_mask_path,
                                      pre_pro=True, is_val=True)
            # var_loader = DataLoader(var_dataset, batch_size=config.batch_size,
            #                         shuffle=True, num_workers=config.num_workers)
        else:
            var_dataset = BaseDataset(config.val_image_path, config.val_mask_path,
                                      is_val=True)
        n_val = len(var_dataset)  # the number of batch

        tot = 0
        counter = len(var_dataset)
        # with tqdm(total=n_val, desc='Validation round', unit='case', leave=False) as pbar:
        if not config.second_network:
            counter = 0
            for case in var_dataset:
                imgs, true_masks = case['image'], case['mask']
                imgs = imgs.to(device=config.device, dtype=torch.float32)
                true_masks = true_masks.to(device=config.device, dtype=mask_type)
                spacing = case['spacing']
                imgs = imgs.to(device=config.device, dtype=torch.float32)

                case_pred = 0
                batch_size = 16
                # print(imgs.shape[0])
                if imgs.shape[0] > 600:
                    # break
                    continue
                counter += 1

                # 理论上分批次与全部批次预测结果一样：分成64批次
                for batch in range(imgs.shape[0]//batch_size+1):
                    with torch.no_grad():
                        if imgs.shape[0] <= batch*batch_size:
                            break
                        if batch == len(imgs) // batch_size:
                            mask_pred = model(imgs[batch*batch_size:])
                        else:
                            mask_pred = model(imgs[batch*batch_size:(batch+1)*batch_size])
                        if batch == 0:
                            case_pred = mask_pred
                        else:
                            case_pred = torch.cat([case_pred, mask_pred], dim=0)
                        del mask_pred
                        gc.collect()
                if config.loss == 0:
                    true_masks = true_masks.squeeze(dim=1)

                    tot += F.cross_entropy(case_pred, true_masks).item()
                    # print(true_masks.shape, case_pred.shape)
                    case_pred = case_pred.argmax(dim=1)
                    # 将整个case合并
                    case_pred = case_pred.squeeze(dim=1)
                    # images, masks = crop_images()
                    kits_metrics += self.calculate_metrics(case_pred.cpu().numpy(),
                                                           true_masks.cpu().numpy(),
                                                           spacing.cpu().numpy())
                    # kits_metrics = 0
                del case_pred, imgs, true_masks
                gc.collect()
                # pbar.update()
        else:
            counter = 0
            for index, case in enumerate(var_dataset):
                imgs, true_masks = case['image'], case['mask']
                imgs = imgs.to(device=config.device, dtype=torch.float32)
                true_masks = true_masks.to(device=config.device, dtype=mask_type)
                # print(mask_type)
                spacing = case['spacing']

                case_pred = 0
                batch_size = 16
                # print(imgs.shape[0])
                # if imgs.shape[0] > 600:
                #     # break
                #     continue
                counter += 1
                # case slice patch_count bbox[y y x x]
                crop_infos = read_image_json(config.image_json)
                crop_info = crop_infos[var_dataset.ids[index]]
                # print(index)
                crop_image, crop_mask = get_cube(imgs, true_masks, crop_info)
                # print(crop_image.shape, crop_mask.shape)

                # 理论上分批次与全部批次预测结果一样：分成64批次
                for batch in range(crop_image.shape[0] // batch_size + 1):
                    # print(batch)
                    with torch.no_grad():
                        if crop_image.shape[0] <= batch * batch_size:
                            break
                        if batch == len(crop_image) // batch_size:
                            mask_pred = model(crop_image[batch * batch_size:])
                        else:
                            mask_pred = model(crop_image[batch * batch_size:(batch + 1) * batch_size])
                        if batch == 0:
                            case_pred = mask_pred
                        else:
                            case_pred = torch.cat([case_pred, mask_pred], dim=0)
                        del mask_pred
                        gc.collect()
                # print(case_pred.shape)
                # 将整个case合并
                # print(case_pred.shape, crop_mask.shape)
                crop_mask = crop_mask.squeeze(dim=1)
                # print(case_pred.dtype, crop_mask.dtype)
                tot += F.cross_entropy(case_pred, crop_mask).item()
                case_pred = case_pred.argmax(dim=1)
                # print(crop_image.shape)
                combine_pred = combine_image(case_pred, imgs.shape, crop_info)
                combine_pred = combine_pred.squeeze(dim=1)
                true_masks = true_masks.squeeze(dim=1)
                # print(crop_mask.shape, case_pred.shape, crop_mask.shape)
                # print(true_masks.shape, case_pred.shape)
                kits_metrics += self.calculate_metrics(combine_pred.cpu().numpy(),
                                                       true_masks.cpu().numpy(),
                                                       spacing.cpu().numpy())
                    # kits_metrics = 0
                del case_pred, imgs, true_masks
                gc.collect()
        if is_training:
            model.train()
        return tot / counter, kits_metrics / counter

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

    def predict_all(self, model, images, config, batch_size=64):
        """
        predict a patient CT image
        :param model:
        :param images: return a patient predict
        :param config:
        :param batch_size
        :return:
        """
        if len(images.shape) != 4:
            images = torch.FloatTensor(images).unsqueeze(dim=1)
        # for i in range(len(images)):
        #     # lung shape [h,w] to [c,h,w]
        #     lung = np.expand_dims(images[i], 1)
        #     print(lung.shape)
        #
        #     mask = self.predict(model, lung, config)
        #     masks.append(mask)
        model.eval()
        for batch in range(images.shape[0] // batch_size + 1):
            with torch.no_grad():
                # print(batch * batch_size, (batch + 1) * batch_size)
                if images.shape[0] <= batch * batch_size:
                    break
                if batch == len(images) // batch_size:
                    mask_pred = model(images[batch * batch_size:])
                else:
                    mask_pred = model(images[batch * batch_size:(batch + 1) * batch_size])
                if batch == 0:
                    case_pred = mask_pred
                else:
                    case_pred = torch.cat([case_pred, mask_pred], dim=0)
        case_pred = case_pred.argmax(dim=1)
        # 将整个case合并
        case_pred = case_pred.squeeze(dim=1)
        return case_pred

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
        for i, hec in enumerate(HEC_NAME_LIST):

            metrics_case[i] = compute_metrics_for_label(masks, ground_trues,
                                                        KITS_HEC_LABEL_MAPPING[hec],
                                                        tuple(spacing),
                                                        sd_tolerance_mm=HEC_SD_TOLERANCES_MM[hec])
            #     metrics_thread.append(Metrics_thread(ground_trues[per_case],
            #                                          masks[per_case],
            #                                          tuple(spacing[per_case]),
            #                                          hec))
            # for i in range(len(HEC_NAME_LIST)):
            #     metrics_thread[i].start()
            # for i in range(len(HEC_NAME_LIST)):
            #     metrics_thread[i].join()
            #     print("end")
            #
            # for i in range(len(HEC_NAME_LIST)):
            #     metrics_case[i] = metrics_thread[i].get_result()
            # bs_metrics += metrics_case
        return metrics_case

    def network_optimizer(self):
        pass

    @staticmethod
    def choose_network(config: Config):
        if config.optimizer == "SGD":
            pass


class Metrics_thread(Thread):
    def __init__(self, ground_trues, masks, spacing, hec):
        Thread.__init__(self)
        self.result = 0
        self.ground_trues = ground_trues
        self.masks = masks
        self.spacing = spacing
        self.hec = hec

    def run(self):
        # print("start")
        self.result = compute_metrics_for_label(self.ground_trues, self.masks,
                                                KITS_HEC_LABEL_MAPPING[self.hec],
                                                self.spacing,
                                                sd_tolerance_mm=HEC_SD_TOLERANCES_MM[self.hec])

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
