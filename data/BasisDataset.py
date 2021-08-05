
from torch.utils.data import Dataset
import torch
import numpy as np
from os.path import splitext
import os
from glob import glob
import sys
import warnings
from utils.config import config
from utils.convert import  convert
from utils.visualization import show_views
sys.path.append("..")
import cv2
# warnings.filterwarnings("ignore")


class BaseDataset(Dataset):

    def __init__(self, image_root, mask_root, is_val=False, pre_pro=True):
        """

        :param image_root:
        :param mask_root:
        :param is_val:  Tre：evaluation the performance
        """
        super(BaseDataset, self).__init__()
        self.image_root = image_root
        self.mask_root = mask_root
        # file name
        self.is_val = is_val
        self.ids = self.get_file(image_root) # [:1]if not is_val else self.get_file(image_root)
        self.pro_pre = pre_pro

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        """
        :return 1. [c,s,h,w]3D    2. [s,c,h,w] 2D
        """
        idx = self.ids[item]
        image_root = self.image_root
        mask_root = self.mask_root
        image_file = os.path.join(image_root, idx)
        mask_file = os.path.join(mask_root, idx)
        # print(mask_file, image_file)
        assert len(glob(image_file)) == 1, "image: no or multiple " + image_file
        assert len(glob(mask_file)) == 1, "mask: no or multiple " + mask_file
        if not self.is_val:
            image_np = np.load(image_file)
            mask_np = np.load(mask_file)
            mask_numpy = np.expand_dims(mask_np, axis=0)
            if self.pro_pre:
                image_np = self.preprocess(image_np)
            image_numpy = np.expand_dims(image_np, axis=0)
            return {
                "image": torch.from_numpy(image_numpy.copy()).type(torch.FloatTensor),
                "mask": torch.from_numpy(mask_numpy.copy()).type(torch.FloatTensor),
            }
        else:
            image_np, spacing = convert.nii_2_np(image_file)
            mask_np, _ = convert.nii_2_np(mask_file)
            # image_np = self.preprocess(image_np, 1000,0)
            # print(image_np.shape)
            # if mask_np.shape[1] != mask_np.shape[0]:
            #     print(mask_np.shape, image_np.shape, image_file)
            non_zero = np.nonzero(mask_np)[0]
            max_index = non_zero.max()
            min_index = non_zero.min()
            mask_numpy = np.expand_dims(mask_np[min_index:max_index], axis=1)
            if self.pro_pre:
                image_np = self.preprocess(image_np)
            image_numpy = np.expand_dims(image_np[min_index:max_index],
                                         axis=1)

            return {
                "image": torch.from_numpy(image_numpy.copy()).type(torch.FloatTensor),
                "mask": torch.from_numpy(mask_numpy.copy()).type(torch.FloatTensor),
                "spacing": torch.from_numpy(spacing).type(torch.FloatTensor)
            }

    # 获取目录下的文件路径名
    def get_file(self, image_root):
        ids = list()
        if not self.is_val:
            for file in os.listdir(image_root):
                if file.endswith(".npy"):
                    ids.append(file)
        else:
            for file in os.listdir(image_root):
                if file.endswith(".gz"):
                    ids.append(file)
        return ids

    @staticmethod
    def preprocess(image):
        """
        input a image to normalization argument the region:[h,w]
        :param image: a image metric
        :return: preprocess
        """
        image[image < -79] = -79
        image[image > 304] = 304
        image = (image - 101) / 76.9
        return image


if __name__ == "__main__":
    image_root = config.val_image_crop
    mask_root = config.val_mask_crop
    base_dataset = BaseDataset(image_root, mask_root)
    print(len(base_dataset))
    index = 0
    images = base_dataset[index]
    print(images["image"].shape, images["mask"].shape)
    # show_views(images["image"][0], images["mask"][0], cmap="gray")

