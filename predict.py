"""
1. predict patient nodule
"""

import torch
from utils.config import Config
from utils.message import Msg
from utils.validation import Validation, get_cube, combine_image
from net.ResUnet import ResNetUNet
from utils.save import Save
from utils.convert import Convert
from utils.visualization import show_views
import segmentation_models_pytorch as smp
from net.baseNet import UNet_3d
from utils.crop import get_crop_info
import numpy as np
from data.BasisDataset import BaseDataset
from utils.metrics import compute_metrics_for_label, compute_disc_for_slice


def main():
    msg = Msg()
    config = Config()
    validation = Validation()
    convert_file = Convert()
    save = Save()

    # model = UNet(n_channels=1, classes=2)
    config.first_model = f"./ResNetUNet_sec_False_Adam_0/CP_epoch11.pth"
    # model = smp.Unet('resnet34', encoder_weights='imagenet', classes=2,in_channels=1)
    model1 = ResNetUNet(in_channel=1, classes=4)
    model1.load_state_dict(torch.load(config.first_model,
                                      map_location=config.device))
    model1.to(device=config.device)

    # raw image root
    image_root = "/public/home/cxiao/Study/data/kits21/image/4/case_00277.nii.gz"
    mask_root = "/public/home/cxiao/Study/data/kits21/mask//4/case_00277.nii.gz"
    # xcd fix
    patient_name = ""
    images_numpy, spacing = convert_file.select(image_root)
    ground_t,spacing = convert_file.select(mask_root)

    # nodule segment
    # masks = validation.predict_all(model, images_numpy, config)
    print(images_numpy.shape)
    first_masks = validation.predict_all(model1, BaseDataset.preprocess(images_numpy), config)
    # save
    show_views(images_numpy[39], images_numpy[39], first_masks[39], cmap="gray")
    msg.norm("masks len", len(first_masks))
    crop_info = get_crop_info(images_numpy, first_masks)

    crop_images, crop_masks = get_cube(images_numpy, first_masks, crop_info)
    print(crop_images.shape)
    config.second_model = f"./ResNetUNet_sec_True_aug_Adam_0/CP_epoch16.pth"
    model2 = ResNetUNet(in_channel=1, classes=4)
    model2.load_state_dict(torch.load(config.first_model,
                                      map_location=config.device))
    model2.to(device=config.device)
    second_masks = validation.predict_all(model2, BaseDataset.preprocess(crop_images), config)
    fina_masks = combine_image(second_masks, images_numpy.shape, crop_info)
    show_views(images_numpy[39], ground_t[39], fina_masks[39], cmap="gray")
    save.save_2_nii(fina_masks, save_path=config.save_path,
                    patient_name="kidney")


if __name__ == "__main__":
    main()

