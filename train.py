import sys
import torch
from utils.config import Config
from utils.validation import Validation
from net.Unet import Res_UNet
from utils.message import Msg
import segmentation_models_pytorch as smp
from net.Unet import Res_UNet, UNet_2_skip
from net.baseNet import UNet_3d, choose_net
from net.ResUnet import ResNetUNet

import time
sys.path.append("./")


if __name__ == "__main__":
    start = time.time()
    # user configure
    config = Config()
    validation = Validation()
    msg = Msg()
    # Unet_smp
    config.network = "ResNetUNet"
    # config.weight = True
    config.loss = 0 # WD

    config.second_network = True
    config.optimizer = f"sec_{config.second_network}_aug3_Adam"

    if config.second_network:
        config.image_root = config.image_crop
        config.mask_root = config.mask_crop
        config.pro_pre = False
        size = 256
    else:
        config.image_root = config.kits_image_root
        config.mask_root = config.kits_mask_root
        size = 512
    # model = UNet_2_skip(n_channels=1, classes=4)
    model = choose_net(config.network, img_size=size)
    # model = UNet_3d(in_channels=1, classes=4)
    # model = UNet3D(1, 4)
    model.to(device=config.device)
    validation.training(model, config)
    msg.end()
    stop = time.time()-start

    msg.norm("time", stop)

