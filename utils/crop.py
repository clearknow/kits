"""


"""
import json
import sys
sys.path.append("..")
import cv2
import numpy as np
from data.BasisDataset import BaseDataset
from utils.visualization import show_views, show_single_view
import torch
from utils.pre_pro import normalize
from utils.config import config
import os


# 裁剪ROI
def crop(image, mask, contour) -> list:
    """
    :param image: tensor
    :param mask:
    save image and mask to 128 * 128
    """
    x_min, y_min = contour.min(axis=0)[0]
    x_max, y_max = contour.max(axis=0)[0]
    # exit(-1)
    hight_center = (y_min + y_max) // 2
    width_center = (x_min + x_max) // 2
    # print(hight_center, width_center)
    hight_start = hight_center - 128
    hight_end = hight_center + 128
    width_start = width_center - 128
    width_end = width_center + 128

    if width_center < 128:
        width_start = 0
        width_end = 256
    if width_center > 384:
        width_end = 512
        width_start = 256

    if hight_center < 128:
        hight_start = 0
        hight_end = 256
    if hight_center > 384:
        hight_end = 512
        hight_start = 256
    return hight_start, hight_end, width_start, width_end


# 存为json格式
def crop_images(image, mask):
    contours = get_boundary


# 保存裁剪图片
def save_image(image, file_name, image_root):
    """
    save the crop image to npy type
    :param image:
    :param file_name:
    :param image_root:
    :return:
    """
    if not os.path.exists(image_root):
        os.makedirs(image_root)
    np.save(os.path.join(image_root, file_name), image)


# 保存边框向量
def save_boundary(boundary, file_name, fin):
    fin.write(file_name+" "+boundary)


# slice y, x min,max
def get_boundary(mask):
    if len(mask.shape) == 3:
        mask = mask.squeeze()
    mask = np.array(mask).astype(np.uint8)
    mask[mask > 1] = 1
    mask_copy = mask.copy()
    cnts, contours = cv2.findContours(mask_copy, mode=cv2.RETR_EXTERNAL,
                                      method=cv2.CHAIN_APPROX_NONE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)
    # print(len(cnts))  # [349 238]
    return c


def get_patch(image, mask, contour):
    mask = np.array(mask).astype(np.uint8)
    mask[mask > 1] = 1
    mask_copy = mask.copy()
    cnts, contours = cv2.findContours(mask_copy, mode=cv2.RETR_EXTERNAL,
                                      method=cv2.CHAIN_APPROX_NONE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)
    # print(len(cnts))  # [349 238]
    return c


def combine_pre(images, mask, ):
    pass


def save_json(data: list, file_name: str):
    with open(file_name, 'w') as file_obj:
        data = json.dumps(data, cls=MyEncoder)
        json.dump(data, file_obj)


def read_image_json(file_name:str):
    """
    :param file_name:
    :return: case slice  patch_count bbox
    """
    with open(file_name, 'r') as f:
        cases = json.load(f)
    cases = json.loads(cases)
    return cases


class MyEncoder(json.JSONEncoder):
    """
    重写json模块JSONEncoder类中的default方法
    """
    def default(self, obj):
        # np整数转为内置int
        if isinstance(obj, np.integer):
            return int(obj)
        else:
            return super(json.JetEncoder, self).default(obj)


def get_crop_info(image, mask):
    crop_infos = list()
    for ct_slice in range(len(mask)):
        contours = get_boundary(mask[ct_slice])
        # print(idx, len(contours))
        for i in range(len(contours)):
            crop_info = dict()
            bbox = crop(image[ct_slice], mask[ct_slice], contours[i])
            # print(i)
            crop_info["slice"] = ct_slice
            crop_info["patch_count"] = i
            crop_info["bbox"] = list(bbox)
            # print(crop_info)
            crop_infos.append(crop_info)

    return crop_infos


def get_bbox():
    base_dataset = BaseDataset(image_root, mask_root, is_val=is_val)
    print((len(base_dataset)))
    file_name = config.image_json
    cases = dict()
    for index in range(len(base_dataset)):
        idx = base_dataset.ids[index]
        if is_val:
            # 1. 2维数据时会存在多个patch，无法合并为一个3d cube
            mask = base_dataset[index]['mask'].squeeze()
            image = base_dataset[index]['image'].squeeze()
            crop_infos = get_crop_info(idx, image, mask)
            cases[idx] = crop_infos

        else:
            mask = base_dataset[index]['mask'].squeeze()
            image = base_dataset[index]['image'].squeeze()
            # print(idx)
            contours = get_boundary(mask)
            # print(len(contours))
            for i in range(len(contours)):
                crop_image, crop_mask = crop(image, mask, contours[i])
                file_name = str(i) + idx
                save_image(crop_image, file_name, image_save_root)
                save_image(crop_mask, file_name, mask_save_root)
    save_json(cases, file_name)


if __name__ == "__main__":
    # test()
    is_val = True
    file_name = config.image_json
    if is_val:
        image_save_root = config.val_image_crop
        mask_save_root = config.val_mask_crop
        image_root = config.val_image_path
        mask_root = config.val_mask_path
    else:
        image_save_root = config.image_crop
        mask_save_root = config.mask_crop
        image_root = config.image_root
        mask_root = config.mask_root

    get_bbox()
    #
    # with open(file_name, 'r') as f:
    #     cases = json.load(f)
    # cases = json.loads(cases)
    # l = dict()
    #
    # print(cases.keys())
    # print(len(cases[""]))