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
    import torch
    import numpy as np
    from generalized_wasserstein_dice_loss.loss import GeneralizedWassersteinDiceLoss

    # Example with 3 classes (including the background: label 0).
    # The distance between the background (class 0) and the other classes is the maximum, equal to 1.
    # The distance between class 1 and class 2 is 0.5.
    dist_mat = np.array([
        [0., 1., 1.],
        [1., 0., 0.5],
        [1., 0.5, 0.]
    ])
    wass_loss = GeneralizedWassersteinDiceLoss(dist_matrix=dist_mat)

    pred = torch.tensor([[[[1, 1, 10],
                         [1, 0, 10],
                         [1, 0, 0]],

                         [[0, 0, 0],
                          [0, 1, 0],
                          [0, 1, 0]],

                         [[0, 0, 0],
                          [0, 0, 0],
                          [0, 0, 2]]]], dtype=torch.float32)
    arg_pre = pred.softmax(dim=1)
    # print(arg_pre)
    grnd = torch.tensor([[[[0, 0, 0],
                         [0, 1, 0],
                         [0, 1, 2]]]], dtype=torch.int64)
    i = DiceLoss()(pred, grnd)
    print(i)
