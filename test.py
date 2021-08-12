"""
1. test model accurate
"""
from utils.visualization import show_single_view,show_views
import numpy as np
import gc
import torch
from utils.config import config
from torch import nn
from loss.loss_function import DiceLoss
import torchvision.transforms as transforms


def read_npy():
    image = np.load(f"{config.val_mask_crop}/0_0_case_00288_.npy")
    show_single_view(image, cmap="gray")
    image = np.load(f"{config.val_image_crop}/0_0_case_00288_.npy")
    show_single_view(image, cmap="gray")


def get_memory(i):
    del i
    gc.collect()


if __name__ == "__main__":
    read_npy()
    # c = torch.cat([a,b],dim=1)
    # print(id(c))
    # b=torch.squeeze(a, dim=0)
    # print(a.shape,b.shape)
    i = 10
    print("%03d"%i)
    print(torch.__version__)
    tran = transforms.RandomAffine(degrees=(-15, 15),
                                   translate=(-0.1, 0.1),
                                   scale=None,
                                   shear=(-15, 15),
                                   interpolation=0.1,
                                   fill=0)
