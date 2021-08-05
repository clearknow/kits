"""
1. test model accurate
"""
from utils.visualization import show_single_view,show_views
import numpy as np
import gc
import torch
from utils.config import config


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
