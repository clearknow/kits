#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   data_utils.py
@Time    :   2021/07/08 11:56:49
@Author  :   Dio Chen
@Version :   1.0.0
@Contact :   zhuo.chen@wuerzburg-dynamics.com
@License :   None
@Desc    :   None
'''

# here put the import lib
import os
import math
import random

import cv2
import torch
import SimpleITK as sitk
from torchvision import transforms
import numpy as np
from skimage import measure
from utils.config import Config
from data.BasisDataset import BaseDataset
from data.kits_dataset import KitsDataset
from utils.visualization import show_views
from utils.save import Save


def z_resample(img, spacing, re_thickness=2, re_xy=[512, 512], interpolation=cv2.INTER_NEAREST):
    """resample img to target thickness and target xy size (64, 256, 256)[x:1, y:1, z:1]-> (32, 512, 512)[x:0.5, y:0.5, z:2]

    Args:
        img : image, [z, x, y]
        spacing : spacing, [x_spacing, y_spacing, z_spacing]
        re_thickness (int, optional): target z thickness. Defaults to 2.
        re_xy (list, optional): target xy size. Defaults to [512, 512].
        interpolation ([type], optional): interpolation method. Defaults to cv2.INTER_NEAREST.

    Returns:
        resample_image, new_spacing
    """
    # RESCALE ALONG Z-AXIS.
    resize_x = 1.0
    resize_z = float(spacing[2]) / float(re_thickness)
    print(resize_z, img.shape)
    img = cv2.resize(img, dsize=None, fx=resize_x, fy=resize_z, interpolation=interpolation)

    # CHANGE ORDER: (z, x, y) -> (x, y, z).
    img = img.swapaxes(0, 2)
    img = img.swapaxes(0, 1)

    # RESCALE ALONG XY-AXIS.
    resize_x = float(re_xy[0] / float(img.shape[0]))
    resize_y = float(re_xy[1] / float(img.shape[1]))
    img = cv2.resize(img, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)

    # CHANGE ORDER: (x, y, z) -> (z, x, y).
    img = img.swapaxes(0, 2)
    img = img.swapaxes(2, 1)

    new_spacing = [spacing[0] / resize_x, spacing[1] / resize_y, re_thickness]

    return img, new_spacing


def data_windowing(img, windowing):
    """windowing data

    Args:
        img : image
        windowing: [bottom, top]

    Returns:
        image_windowing
    """
    img[img < windowing[0]] = windowing[0]
    img[img > windowing[1]] = windowing[1]
    return img


def data_crop(img, crop_xy):
    """crop data xy to target size

    Args:
        img : image, [c,z,x,y] or [c, x, y]
        crop_xy : target size, [x, y]

    Returns:
        img_crop
    """
    img_shape = img.shape

    if len(img_shape) == 3:
        if type(img) == np.ndarray:
            img = np.expand_dims(img, 0)
        else:
            img = torch.unsqueeze(img, dim=0)

    center = (int(img_shape[-2] / 2), int(img_shape[-1] / 2))

    x_edge = [center[0] - int(crop_xy[0] / 2), center[0] + int(crop_xy[0] / 2)]
    y_edge = [center[1] - int(crop_xy[1] / 2), center[1] + int(crop_xy[1] / 2)]

    img = img[:, :, x_edge[0]:x_edge[1], y_edge[0]:y_edge[1]]

    if len(img_shape) == 3:
        if type(img) == np.ndarray:
            img = np.squeeze(img, 0)
        else:
            img = torch.squeeze(img, dim=0)

    return img


def get_trans_compose(trans_type, fill=0, data_type='image'):
    """get transform compose

    Args:
        trans_type : type of data transforms. Defaults to None.
        fill (int, optional): fill value. Defaults to 0.
        data_type (str, optional): image or mask. Defaults to 'image'.

    Returns:
        transform_compose
    """

    if data_type == 'image':
        interpolation = transforms.InterpolationMode.NEAREST
    elif data_type == 'mask':
        interpolation = transforms.InterpolationMode.NEAREST

    trans = []

    if trans_type == 'Affine':
        tran = transforms.RandomAffine(degrees=(-15, 15),
                                       translate=(-0.1, 0.1),
                                       scale=None,
                                       shear=(-15, 15),
                                       interpolation=interpolation,
                                       fill=fill)
        trans.append(tran)

    if trans_type == 'AffineNoTranslate':
        tran = transforms.RandomAffine(degrees=(-20, 20),
                                       translate=None,
                                       scale=None,
                                       shear=(-20, 20),
                                       interpolation=interpolation,
                                       fill=fill)
        trans.append(tran)

    if trans_type == 'HorizontalFlip':
        tran = transforms.RandomHorizontalFlip(p=1)
        trans.append(tran)

    if trans_type == 'VerticalFlip':
        tran = transforms.RandomVerticalFlip(p=1)
        trans.append(tran)

    if trans_type == 'HorizontalFlip_Affine':
        tran_1 = transforms.RandomHorizontalFlip(p=1)
        trans.append(tran_1)

        tran_2 = transforms.RandomAffine(degrees=(-25, 25),
                                         translate=None,
                                         scale=None,
                                         shear=(-25, 25),
                                         interpolation=interpolation,
                                         fill=fill)
        trans.append(tran_2)
    transform = transforms.Compose(trans)

    return transform


def crop_area(mask_arr, img_arr, kidney_info, area_size):
    """use the kidney info to crop image shape like area_size

    :param mask_arr: target mask[n,h,w]
    :param img_arr: target image[n,h,w]
    :param kidney_info: kidney info:{'voxel': 72240, 'loc': (10, 120, 157, 107, 166, 205),
            'centroid': (62.516486710963456, 142.19957087486156, 181.06000830564784)}
    :param area_size: target shape (n,h,w)
    :return: return cube
            mask, img, normal, crop_edge
    """
    centroid = kidney_info['centroid']
    loc = kidney_info['loc']
    cent_x = int(math.ceil(centroid[1]))
    cent_y = int(math.ceil(centroid[2]))

    crop_x = int(area_size[1] / 2)
    crop_y = int(area_size[2] / 2)
    min_thickness = area_size[0]

    thickness = img_arr.shape[0]
    # print(img_arr.shape, mask_arr.shape)
    x_max = loc[4]
    x_min = loc[1]

    y_max = loc[5]
    y_min = loc[2]

    z_min = loc[0]
    z_max = loc[3]
    # print(z_min, z_max)
    if x_max - x_min <= 2 * crop_x and y_max - y_min <= 2 * crop_y:
        x_max = cent_x + crop_x
        x_min = cent_x - crop_x
        y_max = cent_y + crop_y
        y_min = cent_y - crop_y

        normal_xy = True
    else:
        x_max = x_max
        x_min = x_min
        y_max = y_max
        y_min = y_min

        normal_xy = False

    if thickness >= min_thickness:

        z_min = z_min - 5 if z_min - 5 >= 0 else 0
        z_max = z_max + 5 if z_max + 5 <= thickness else thickness

        normal_z = True
        # print(z_min, z_max)
        if z_max - z_min < min_thickness:
            if z_min == 0:
                z_max = min_thickness
            elif z_max == thickness:
                z_min = thickness - min_thickness
            elif z_min != 0 and z_max != thickness:
                # print('z_min != 0 and z_max != thickness', z_min, z_max)
                value = min_thickness - (z_max - z_min)
                if z_max + value <= thickness:
                    z_max = z_max + value
                else:
                    z_max = thickness
                    z_min = thickness - min_thickness

            normal_z = True
        else:
            # print('z_max - z_min > min_thickness  ', z_min, z_max)
            z_max = z_max
            z_min = z_min
            normal = True
    else:
        normal_z = False

    crop_edge = (z_min, x_min, y_min, z_max, x_max, y_max)
    mask = mask_arr[crop_edge[0]:crop_edge[3], crop_edge[1]:crop_edge[4], crop_edge[2]:crop_edge[5]]
    img = img_arr[crop_edge[0]:crop_edge[3], crop_edge[1]:crop_edge[4], crop_edge[2]:crop_edge[5]]

    normal = True if normal_xy and normal_z else False
    # print(z_min, z_max, thickness, min_thickness)
    # print(img.shape, mask.shape)
    # print('----------- crop')
    assert img.shape[0] >= min_thickness and mask.shape[0] >= min_thickness, "image or mask thickness is error"

    return mask, img, normal, crop_edge


def save_to_nii(filepath: str, case_name: str, img, loc_info: dict, suffix):
    """save numpy or tensor to nii

    :param filepath:
    :param case_name:
    :param img:
    :param loc_info:
    :param suffix: file name
    :return:
    """
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    case_name = os.path.join(filepath, case_name)
    if not os.path.exists(case_name):
        os.mkdir(case_name)

    save_nii(case_name, img, loc_info, suffix)


def save_nii(case_path: str, img, loc_info: dict, suffix):
    """ save nii helper

    :param case_path:
    :param img:
    :param loc_info: kidney infos
    :param suffix: save case file name
    :return:
    """
    if type(img) is not np.ndarray:
        img = img.cpu().numpy()

    img = sitk.GetImageFromArray(img)
    img.SetOrigin(loc_info['origin'])
    img.SetDirection(loc_info['direction'])
    img.SetSpacing(loc_info['space'])
    crop_img_path = os.path.join(case_path, suffix)
    sitk.WriteImage(img, crop_img_path)


def load_nii(filepath: str, channels_last) -> dict:

    image = sitk.ReadImage(filepath)
    img_arr = sitk.GetArrayFromImage(image).astype(np.float32)

    if channels_last:
        img_arr = img_arr.transpose(2, 1, 0)

    origin = image.GetOrigin()
    direction = image.GetDirection()
    space = image.GetSpacing()
    cases_dic = {'img_arr': img_arr, 'origin': origin, 'direction': direction, 'space': space}
    return cases_dic


def get_area(img_arr) -> list:
    """get the kidney contour info

    :param img_arr: image shape [h],[h,w],[n,h,w]
    :return:  the kidney contour infos:
            voxel : counter of area voxel,  loc:bbox, centroid: the value of axis mean
            eg:list[{'voxel': 72240, 'loc': (10, 120, 157, 107, 166, 205),
            'centroid': (62.516486710963456, 142.19957087486156, 181.06000830564784)},...]
            the area[0],area[1] may be the two kidney area
    """
    if type(img_arr) != np.ndarray:
        img_arr = img_arr.cpu().numpy()
    label = measure.label(img_arr, connectivity=2)
    props = measure.regionprops(label)

    area_info = list()
    for i in range(len(props)):
        area_info.append({"voxel": props[i].area, "loc": props[i].bbox, "centroid": props[i].centroid})
    sorted_area = sorted(area_info, key=lambda area_info: area_info['voxel'], reverse=True)
    return sorted_area


def to_distribution_dict(data_frame, analyzing, group_by='bef_spacing_z'):

    analyzing_dcit = {}

    for c in analyzing.keys():
        group_dict = {}

        data_analy = data_frame[analyzing[c]]
        if group_by:
            groups = data_analy.groupby(group_by).groups

            for t in groups:
                data_part = data_analy[data_analy[group_by] == t]
                group_dict[t] = data_part
                analyzing_dcit[c] = group_dict
        else:
            analyzing_dcit[c] = data_analy

    return analyzing_dcit


def split_dataset(name_list, ratio=[0.2, 0.1]):
    """
    split dataset to test,val,train set
    :param name_list:
    :param ratio: the ratio of test,val,train set
    :return: name_list:after shuffle, dataset_list:['train','train','train,'valid','test']
    """
    random.shuffle(name_list)
    length = len(name_list)

    test_num = math.ceil(length * ratio[0])
    valid_num = math.ceil(length * ratio[1])
    train_num = length - test_num - valid_num

    dataset_list = []
    dataset_list += ['train'] * train_num
    dataset_list += ['valid'] * valid_num
    dataset_list += ['test'] * test_num

    # print(test_num, valid_num, train_num, len(dataset_list), l)
    # print('-------')
    return name_list, dataset_list


def to_dataset_dict(analyzing_dcit, keys=[]):
    name_list = []
    analyzing_list = []
    group_list = []
    dataset_list = []
    print(analyzing_dcit)
    for k in analyzing_dcit.keys():
        group_dict = analyzing_dcit[k]
        for k_g in group_dict.keys():
            case_name = group_dict[k_g]['case_name'].values.tolist()
            case_name, dataset = split_dataset(case_name, ratio=[0.2, 0.1])

            name_list += case_name
            analyzing_list += [k] * len(case_name)
            group_list += [str(k_g)] * len(case_name)
            dataset_list += dataset
    df_dict = {keys[0]: name_list, keys[1]: analyzing_list, keys[2]: group_list, keys[3]: dataset_list}
    return df_dict


def to_dataset_dict_nothick(analyzing_dcit, keys=[]):
    name_list = []
    analyzing_list = []
    dataset_list = []
    print(analyzing_dcit)
    for k in analyzing_dcit.keys():
        group_dict = analyzing_dcit[k]
        case_name = group_dict['case_name'].values.tolist()
        case_name, dataset = split_dataset(case_name, ratio=[0.2, 0.1])
        name_list += case_name
        analyzing_list += [k] * len(case_name)
        dataset_list += dataset

    df_dict = {keys[0]: name_list, keys[1]: analyzing_list, keys[2]: dataset_list}

    return df_dict


def get_distribution_list(data_df, analyzing, thickness):

    distribution_list = []
    for a in analyzing:
        for t in thickness:
            classify = data_df[(data_df['size'] == a) & (data_df['thinkness'] == t)]
            distribution_list.append(classify)
    return distribution_list


def get_distribution_list_nothick(data_df, analyzing):

    distribution_list = []
    for a in analyzing:
        classify = data_df[(data_df['size'] == a)]
        distribution_list.append(classify)
    return distribution_list


def change_label(arr: np.ndarray, label) -> np.ndarray:
    """change array value to target label

    :param arr: target array
    :param label: change label
    :return: change label array
    """
    keys = label.keys()

    for k in keys:
        arr[arr == int(k)] = label[k]

    return arr


if __name__ == "__main__":
    config = Config()
    image_root = config.image_3d_path
    mask_root = config.mask_3d_path
    dataset = KitsDataset(image_root, mask_root,pre_process=False)
    # print(len(dataset))
    index = 1
    masks = dataset[index]["mask"]
    images = dataset[index]["image"]
    # print(masks.shape)

    areas = get_area(masks[0])
    if len(areas) >= 2:
        areas = areas[:2]
    else:
        areas = areas[0]
    area_size = (80, 80, 80)
    for i in range(len(areas)):
        mask, img, normal, crop_edge = crop_area(masks[0], images[0], areas[i], area_size)
        # print(mask.shape, img.shape)
        show_views(img[20], mask[20])

        Save.save_2_nii(img.float(), config.save_path, "kidney_cube")
        Save.save_2_nii(mask.float(), config.save_path, "mask_cube")
