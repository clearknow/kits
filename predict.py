"""
1. predict patient nodule
"""

import torch
from utils.config import Config
from utils.message import Msg
from utils.validation import Validation
from net.Unet import UNet
from utils.save import Save
from utils.convert import Convert
from utils.visualization import show_views
import segmentation_models_pytorch as smp
from net.baseNet import UNet_3d

if __name__ == "__main__":
    msg = Msg()
    config = Config()
    validation = Validation()
    convert_file = Convert()
    save = Save()

    # model = UNet(n_channels=1, classes=2)
    config.pre_model_path = f"./{config.network}/CP_epoch10.pth"
    # model = smp.Unet('resnet34', encoder_weights='imagenet', classes=2,in_channels=1)
    model = UNet_3d(in_channels=1, classes=4)
    model.load_state_dict(torch.load(config.pre_model_path,
                                     map_location=config.device))
    model.to(device=config.device)

    # raw image root
    image_root = "/public/home/cxiao/Study/data/kits21/image/0/case_00000.nii.gz"
    # xcd fix
    patient_name = ""
    images_numpy,_ = convert_file.select(image_root)

    # nodule segment
    # masks = validation.predict_all(model, images_numpy, config)
    masks = validation.predict(model, images_numpy, config)
    # save
    show_views(images_numpy[39], images_numpy[39],masks[39], cmap="gray")
    msg.norm("masks len", len(masks))

    save.save_2_nii(masks, save_path=config.save_path,
                    patient_name="kidney")
