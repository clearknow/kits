"""
1. predict patient nodule
"""

import torch
from utils.config import Config, CoarseConfig
from utils.message import Msg
from utils.validation import Validation, get_cube, combine_image
from net.ResUnet import ResNetUNet
from utils.save import Save
from utils.convert import Convert
from utils.visualization import show_views
import numpy as np
from net.ResUnet_3D import UNet3D
import math
import cv2
from utils.data_utils import get_area, crop_area, z_resample
from utils.metrics import compute_metrics_for_label
from configuration.labels import KITS_HEC_LABEL_MAPPING, HEC_NAME_LIST, HEC_SD_TOLERANCES_MM, GT_SEGM_FNAME
import pandas as pd
import os


def main():
    msg = Msg()
    config = Config()
    coarse_config = CoarseConfig()
    validation = Validation()
    convert_file = Convert()
    save = Save()

    # model = UNet(n_channels=1, classes=2)
    config.first_model = f"./ResNetUNet-3D_conv_coarse_True_Adam_0/CP_epoch7.pth"
    model1 = UNet3D()
    model1.load_state_dict(torch.load(config.first_model,
                                      map_location=config.device))
    model1.to(device=config.device)

    # raw image root
    image_root = config.image_3d_path
    mask_root = config.mask_3d_path
    # mask_root = "/public/home/cxiao/Study/data/kits21/mask/case_00277.nii.gz"
    cases_name = pd.read_json(config.case_name_json)
    all_metrics = np.zeros((len(HEC_NAME_LIST), 2), dtype=float)
    case_counter = 0
    for case_name in cases_name["cases_name"]:
        image_filepath = os.path.join(image_root, case_name+".nii.gz")
        mask_filepath = os.path.join(mask_root, case_name+".nii.gz")
        # print(image_filepath)
        if image_filepath.endswith(".gz"):
            case_id = int(case_name[7:10])
            # print(case_id)
            if case_id < 270:
                continue
        case_counter += 1
        images_numpy, origin_spacing = convert_file.select(image_filepath)
        ground_t, origin_spacing = convert_file.select(mask_filepath)

        # coarse network-----------------------------------------
        first_masks, crop_image, new_spacing, z_thick = validation.predict_all(model1, images_numpy, config, origin_spacing)
        msg.norm("masks len", len(first_masks))
        # combine slices mask to n,384,384(re_sample):
        masks_list = list()

        for i in range(len(first_masks)):
            masks_list.append(first_masks[i])
        first_pre_masks = torch.vstack(masks_list)[:len(crop_image), :, :]
        # coarse network-----------------------------------------

        # fine network-----------------------------------------
        config.second_model = f"./ResNetUNet-3D_coarse_False_aug3_Adam_0/CP_epoch6.pth"
        model2 = UNet3D(is_conv=False)
        model2.load_state_dict(torch.load(config.second_model,
                                          map_location=config.device))
        model2.to(device=config.device)
        # ______________________________ get kidney area
        first_pre_masks_copy = first_pre_masks.clone()
        first_pre_masks_copy[first_pre_masks_copy > 1] = 1
        kidney_area = get_area(first_pre_masks_copy)
        if len(kidney_area) >= 2:
            kidney_area = kidney_area[:2]
        else:
            kidney_area = kidney_area[0]
        # predict
        final_mask = torch.zeros_like(crop_image, dtype=torch.int8)

        for i in range(len(kidney_area)):
            crop_img, crop_mask, _, crop_edge = crop_area(crop_image, first_pre_masks, kidney_area[i],
                                                          coarse_config.area_size)
            second_masks, _, spacing, _ = validation.predict_all(model2, crop_img, coarse_config)
            # b*64*128*128 -> n*128*128
            # second_pre_masks = combine_image(second_masks, crop_img.shape, coarse_config)
            second_pre_masks = list()
            for i in range(len(second_masks)):
                second_pre_masks.append(second_masks[i])
            second_pre_masks = torch.vstack(second_pre_masks)[:len(crop_img), :, :]

            assert crop_img.shape != second_pre_masks, "difference shape between crop img and pre mask"
            # combine 64*128*128 to s*384*384
            final_mask[crop_edge[0]:crop_edge[3], crop_edge[1]:crop_edge[4], crop_edge[2]:crop_edge[5]] = second_pre_masks

        # n*384*384 to n1*512*512

        pad_x = padding(config.resize_xy[0], config.crop_xy[0])
        pad_y = padding(config.resize_xy[1], config.crop_xy[1])
        final_mask = np.pad(final_mask, ((0, 0), pad_x, pad_y), 'minimum')
        final_mask_resample, _ = z_resample(final_mask, new_spacing, re_thickness=origin_spacing[0], re_xy=config.resize_xy)
        if final_mask_resample.shape[0] < ground_t.shape[0]:
            p = ground_t.shape[0] - final_mask_resample.shape[0]
            final_mask_resample = np.pad(final_mask_resample, ((0, p), (0, 0), (0, 0)), 'minimum')
        else:
            final_mask_resample = final_mask_resample[:ground_t.shape[0], :, :]
        # assert final_mask_resample.shape == ground_t.shape, f"fina———{final_mask_resample.shape},gt_{ground_t.shape}"
        metrics_case = np.zeros((len(HEC_NAME_LIST), 2), dtype=float)
        for i, hec in enumerate(HEC_NAME_LIST):
            metrics_case[i] = compute_metrics_for_label(final_mask_resample, ground_t,
                                                        KITS_HEC_LABEL_MAPPING[hec],
                                                        tuple(origin_spacing),
                                                        sd_tolerance_mm=HEC_SD_TOLERANCES_MM[hec])
        msg.norm(case_name, metrics_case)
        msg.norm(case_name, origin_spacing)
        all_metrics += metrics_case
    msg.norm("dice and sds", all_metrics / case_counter)


def combine_predict(input_mask, origin_shape, config):
    """ combine and resample mask
    :param input_mask: [n,s,h,w] (n,32,384,384) or (n,64,128,128)
    :param config: train config
    :param z_thick: [s,h,w]
    :return: origin mask
    """
    voxel_z = origin_shape[0]
    target_gen_size = config.layer_thick
    range_val = int(math.ceil((voxel_z - target_gen_size) / config.stride) + 1)
    combine_masks = torch.zeros(origin_shape, dtype=torch.int8)

    for i in range(range_val):
        start_num = i * config.stride
        end_num = start_num + target_gen_size
        if end_num <= voxel_z:
            # 数据块长度没有超出x轴的范围,正常取块
            combine_masks[start_num:end_num, :, :] = torch.from_numpy(
                cv2.bitwise_or(np.array(combine_masks[start_num:end_num, :, :]),
                               np.array(input_mask[range_val])))
        else:
            # 数据块长度超出x轴的范围, 从最后一层往前取一个 batch_gen_size 大小的块作为本次获取的数据块
            combine_masks[(voxel_z - target_gen_size):voxel_z, :, :] = torch.from_numpy(cv2.bitwise_or(
                np.array(combine_masks[(voxel_z - target_gen_size):voxel_z, :, :]),
                np.array(input_mask[range_val])))
    return combine_masks


def padding(target_size, input_size):
    pad_1 = (target_size - input_size) // 2
    pad_2 = (target_size - input_size) / 2
    if pad_1 != pad_2:
        pad_2 = pad_1 + 1
    else:
        pad_2 = pad_1

    return (pad_1, pad_2)


if __name__ == "__main__":
    main()

