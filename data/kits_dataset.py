
from torch.utils.data import Dataset
import torch
import numpy as np
from os.path import splitext
import os
from glob import glob
import sys
import warnings
sys.path.append("../")
from utils.config import config
from utils.visualization import show_views
import cv2
# warnings.filterwarnings("ignore")
from utils.convert import convert
from sklearn.model_selection import KFold
import torchvision.transforms as transforms


class KitsDataset(Dataset):
    def __init__(self, image_root, mask_root, pre_process=True):
        super(KitsDataset, self).__init__()
        self.image_root = image_root
        self.mask_root = mask_root
        # file name
        self.ids = self.get_file(image_root)
        self.pre_process = pre_process

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        """
        :param item:
        :return: 4D (c,s,h,w)
        """
        idx = self.ids[item]
        image_root = self.image_root
        mask_root = self.mask_root
        image_file = os.path.join(image_root, idx)
        mask_file = os.path.join(mask_root, idx)
        # print(image_file,mask_file)
        assert len(glob(image_file)) == 1, "image: no or multiple " + image_file
        assert len(glob(mask_file)) == 1, "mask: no or multiple " + mask_file
        # (s,h,w)??
        image_np, space = convert.nii_2_np(image_file)
        mask_np, space = convert.nii_2_np(mask_file)
        # print(space)
        # non_zero = np.nonzero(mask_np)[0]
        # max_index = non_zero.max()
        # min_index = non_zero.min()
        mask_numpy = np.expand_dims(mask_np, axis=0)
        if self.pre_process:
            image_np = self.preprocess(image_np)
        image_numpy = np.expand_dims(image_np, axis=0)

        # bug xcd the shape w and h dont same
        return {
            "image": torch.from_numpy(image_numpy.copy()).type(torch.FloatTensor),
            "mask": torch.from_numpy(mask_numpy.copy()).type(torch.FloatTensor),
            "spacing": space
        }

    @staticmethod
    # ?????????????????????????????????
    def get_file(image_root):
        ids = list()
        for file in os.listdir(image_root):
            image_path = os.path.join(image_root, file)
            for image in os.listdir(image_path):
                 if image.endswith(".gz"):
                    ids.append(os.path.join(file, image))
        # print(ids)
        return ids

    @staticmethod
    def preprocess(image):
        """
        CT value to [-79,306], and normalization
        :param image: a image metric
        :return: preprocess
        """
        image[image < -79] = -79
        image[image > 304] = 304
        image = (image - 101) / 76.9
        return image

    def transforms(self):
        flip = transforms.Compose([
            transforms.RandomHorizontalFlip(1),
        ])


def main():
    image_root = config.image_3d_path
    mask_root = config.mask_3d_path
    base_dataset = KitsDataset(image_root, mask_root)
    # print(len(base_dataset.ids))
    index = 10
    images = base_dataset[index]
    show_views(images["image"][0][index], images["mask"][0][index], cmap="gray")

    # print(images["image"].max(), len(base_dataset))


if __name__ == "__main__":
    main()

