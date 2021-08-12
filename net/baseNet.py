import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
from tensorboardX import SummaryWriter
from utils.config import Config
import segmentation_models_pytorch as smp
from net.Unet import Res_UNet, UNet_2_skip
from net.ResUnet import ResNetUNet
from net.vision_transformer import SwinUnet


# 保持slice的维度不变
class UNet_3d(nn.Module):
    # n_class 输出维度
    def __init__(self, in_channels, classes):
        super(UNet_3d, self).__init__()
        self.n_channels = in_channels
        self.n_classes = in_channels
        # self.in_c = in_c
        # network
        self.in_layer = DoubleConv(in_channels, 32)

        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)

        self.up1 = Up(256, 128)
        self.up2 = Up(128, 64)
        self.up3 = Up(64, 32)

        self.out = OutConv(32, classes)

    def forward(self, x):
        # print("0:{}".format(x.shape))
        x1 = self.in_layer(x)
        # print("1:{}".format(x1.shape))
        x2 = self.down1(x1)
        # print("2:{}".format(x2.shape))
        x3 = self.down2(x2)
        # print("3:{}".format(x3.shape))
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        x = self.out(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_ch, n_class):
        super(DoubleConv, self).__init__()
        self.in_ch = in_ch
        self.n_class = n_class

        self.feature = nn.Sequential(
            # same 卷积
            nn.Conv3d(in_ch, n_class, kernel_size=3, padding=1),
            nn.InstanceNorm3d(n_class),
            nn.ReLU(),

            nn.Conv3d(n_class, n_class, kernel_size=3, padding=1),
            nn.InstanceNorm3d(n_class),
            nn.ReLU(), )

    def forward(self, x):
        # print("dou",x.shape)
        return self.feature(x)


# 下采样
class Down(nn.Module):
    def __init__(self, in_ch, n_class):
        super(Down, self).__init__()

        self.feature = nn.Sequential(
            nn.Conv3d(in_ch, n_class, kernel_size=3, stride=(1,2,2),
                      padding=(1, 1, 1)),)

    def forward(self, x):
        # print("d",x.shape)
        return self.feature(x)


# 上采样  反卷积
class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch//2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            DoubleConv(in_ch, out_ch))

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # torch.Size([1, 320, 2, 14, 14])
        # torch.Size([1, 320, 4, 15, 15])
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffZ = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffZ // 2, diffZ - diffZ // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffX // 2, diffX - diffX // 2])
        # 维度从高到底扩充
        x = torch.cat([x2, x1], dim=1)
        # print("up",x.shape)
        return self.conv(x)


# 输出  1*1 卷积 + softmax
class OutConv(nn.Module):
    def __init__(self, in_ch, n_class):
        super(OutConv, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv3d(in_ch, n_class, kernel_size=1, stride=1),
            nn.InstanceNorm3d(n_class),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.feature(x)


def choose_net(network, in_channel=1, classes=4, img_size=256):
    if network == "Unet_smp":
        return smp.Unet('resnet34', classes=classes, in_channels=in_channel)
    elif network == "Unet":
        # cant use
        return Res_UNet(n_channels=in_channel, classes=classes)
    elif network == "SwinUnet":
        return SwinUnet(num_classes=classes, img_size=img_size)
    elif network == "ResNetUNet" or network == "ResNetUNet_no_weight" or\
            "ResNetUNet_2_skip" or network == "ResNetUNet_second" or \
            network == "ResNetUNet_second_no_weight":
        return ResNetUNet(in_channel=in_channel, classes=classes)



if __name__ == "__main__":
    model = UNet_3d(1, 3)
    # (1,128,256,256)
    print(summary(model, (1, 128, 128, 128), device="cpu"))
    writer = SummaryWriter()
    writer.add_graph(model,torch.randn((1, 1,64, 64, 64)))

    print(model)
