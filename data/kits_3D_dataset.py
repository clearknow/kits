from torch.utils.data import Dataset
import torch
import numpy as np
from os.path import splitext
import os
from glob import glob
import sys
import warnings
sys.path.append("../")
from utils.config import Config, CoarseConfig
from utils.visualization import show_views
import cv2
# warnings.filterwarnings("ignore")
from utils.convert import convert
from sklearn.model_selection import KFold
import torchvision.transforms as transforms
import pandas as pd
import json


class KitsDataset3D(Dataset):
    def __init__(self, config: Config, is_train=True):
        super(KitsDataset3D, self).__init__()
        if config.coarse:
            self.image_root = config.crop_origin_image
            self.mask_root = config.crop_origin_mask
        else:
            self.image_root = config.crop_64cube_image
            self.mask_root = config.crop_64cube_mask
        self.kidney_name_json = config.kidney_info_json
        self.cube_name_json = config.cube_name_json
        self.case_name_json = config.case_name_json
        self.coarse = config.coarse
        self.is_train = is_train
        #   name
        self.ids = self.get_kidney_info()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        """
        :param item:
        :return: 4D (c,s,h,w)
        """
        idx = self.ids[item]
        # print(len(self.ids))
        file_name = idx[0]
        spacing = idx[1]
        # print(file_name)
        image_root = self.image_root
        mask_root = self.mask_root
        image_file = os.path.join(image_root, file_name)
        mask_file = os.path.join(mask_root, file_name)
        # print(image_file, mask_file)
        assert len(glob(image_file)) == 1, "image: no or multiple " + image_file
        assert len(glob(mask_file)) == 1, "mask: no or multiple " + mask_file
        # (1, s,h,w)??
        image = torch.load(image_file)
        mask = torch.load(mask_file)
        # print(image.max(), mask.max())
        # print(spacing)
        return {
            "image": image,
            "mask": mask,
            "spacing": torch.tensor(spacing)
        }

    def get_kidney_info(self):
        """ get the cube name

        :param cube_name:json file cube :{"case_000xxx":{‘kidney':xx,'new_spacing’：}}
        :return:  case_name list()
        """
        with open(self.kidney_name_json) as f:
            kidney_infos = json.load(f)
            kidney_infos = json.loads(kidney_infos)
        with open(self.cube_name_json) as f:
            cubes_name = json.load(f)
            cubes_name = json.loads(cubes_name)
        with open(self.case_name_json) as f:
            cases_name = json.load(f)["cases_name"]
        if self.coarse:
            target_file_name = cases_name
        else:
            target_file_name = cubes_name["cases_name"]

        ids = list()
        for filename in os.listdir(self.mask_root):
            # print(filename)
            if filename.endswith(".pth"):
                case_name = filename[:10]
                if self.is_train and (int(case_name[-3:]) < 270):
                    ids.append([filename, kidney_infos[case_name]["new_spacing"]])
                elif not self.is_train and (int(case_name[-3:]) > 270):
                    ids.append([filename, kidney_infos[case_name]["new_spacing"]])
        # print(len(ids))
        return ids


def main():
    config = Config()
    # print(config.)
    base_dataset = KitsDataset3D(config)
    print(len(base_dataset))
    index = 10
    images = base_dataset[index]
    print(base_dataset.ids[index])

    # show_views(images["image"][0][index], images["mask"][0][index], cmap="gray")

    print(images["mask"].max(), len(base_dataset), images["image"].shape)


if __name__ == "__main__":
    main()

