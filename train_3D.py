import sys
import torch
from utils.config import Config, CoarseConfig
from utils.validation import Validation
from net.Unet import Res_UNet
from utils.message import Msg
from net.Unet import Res_UNet, UNet_2_skip
from net.baseNet import UNet_3d, choose_net
from net.ResUnet import ResNetUNet
import os

import time
from net.ResUnet_3D import UNet3D

sys.path.append("./")


if __name__ == "__main__":
    start = time.time()
    # user configure
    coarse = False
    if coarse:
        config = Config()
    else:
        config = CoarseConfig()
    validation = Validation()
    msg = Msg()
    config.network = "ResNetUNet-3D"
    config.loss = 0 # WD
    config.data_parallel = False

    config.optimizer = f"conv_coarse_{config.coarse}_SGD" + str(start)

    # model = UNet_2_skip(n_channels=1, classes=4)
    model = choose_net(config.network)
    if config.data_parallel:
        # os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
        model = torch.nn.DataParallel(model, device_ids=config.device_ids)
        model.to(device=config.device_ids[0])

    else:
        model.to(device=config.device)
    validation.training_3d(model, config)
    msg.end()
    stop = time.time()-start

    msg.norm("time", stop)

