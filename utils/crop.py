"""


"""

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
def crop(image, mask, contour):
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
    crop_img = image[hight_start:hight_end, width_start:width_end]
    crop_mask = mask[hight_start:hight_end, width_start:width_end]
    # show_views(crop_mask, crop_img)
    return crop_img, crop_mask


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


def get_boundary(mask):
    mask = np.array(mask).astype(np.uint8)
    mask[mask > 1] = 1
    mask_copy = mask.copy()
    cnts, contours = cv2.findContours(mask_copy, mode=cv2.RETR_EXTERNAL,
                                      method=cv2.CHAIN_APPROX_NONE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)
    # print(len(cnts))  # [349 238]
    return c

    # pre_point = c[0][0]
    # new_contours = list()
    # contour = list()
    # contour.append(pre_point)
    # for i in range(1, len(c)):
    #     if any(abs(pre_point - c[i][0]) == [True, True]):
    #         contour.append(c[i][0])
    #         # print(contour)
    #     else:
    #         new_contours.append(contour)
    #         contour.clear()
    #         contour.append(pre_point)
    #     pre_point = c[i][0]
    #     if i == len(c) - 1:
    #         new_contours.append(np.array(contour))
    #
    # print(len(new_contours))

    # #     compute the rotated bounding box of the largest contour
    # rect = cv2.minAreaRect(c)
    # box = np.int0(cv2.boxPoints(rect))
    # new_mask = np.zeros_like(mask)
    # # c = np.array([[[50,10],[100,20],[180,200],[150,200],[160,200]]])
    # # # print(c)
    # cv2.drawContours(new_mask, c, -1, 1, -1)
    # # thresh = cv2.threshold(mask, 0, 255,
    # #                        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #
    # # cv2.drawContours(new_mask, c, -1, (255, 255, 255), 3)
    # new_mask = fill_hole(new_mask, contour)
    # #     new_mask = cv2.fillPoly(new_mask, contours ,5)
    # show_views(mask_copy, new_mask,cmap="gray")


if __name__ == "__main__":
    # test()
    is_val = True
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

    base_dataset = BaseDataset(image_root, mask_root, is_val=is_val)
    print((len(base_dataset)))
    for index in range(len(base_dataset)):
        idx = base_dataset.ids[index]
        if is_val:
            # 1. 2维数据时会存在多个patch，无法合并为一个3d cube
            mask = base_dataset[index]['mask'].squeeze()
            image = base_dataset[index]['image'].squeeze()
            for ct_slice in range(len(mask)):
                contours = get_boundary(mask[ct_slice])
                # print(len(contours))
                for i in range(len(contours)):
                    crop_image, crop_mask = crop(image[ct_slice], mask[ct_slice],
                                                 contours[i])
                    file_name = f"{ct_slice}_{i}_{idx.split('.')[0]}_.npy"
                    # show_views(crop_image,crop_mask)
                    # print(file_name)
                    # exit(-1)

                    save_image(crop_image, file_name, image_save_root)
                    save_image(crop_mask, file_name, mask_save_root)

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

