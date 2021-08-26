""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from tensorboardX import SummaryWriter


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, res_block=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.res_block = res_block
        self.residual = BasicBlock(in_channels, in_channels)
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # print(x.shape)
        if self.res_block:
            x = self.residual(x)
        return self.double_conv(x)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, res_block=False):
        """
        inplanes代表输入通道数，planes代表输出通道数。
        :param inplanes:
        :param planes:
        :param stride:
        :param downsample:
        """
        super(BasicBlock, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.conv1(x)
        residual = self.shortcut(x)
        out += residual
        out = self.relu(out)

        return out


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            BasicBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = BasicBlock(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        #         print("up", x1.shape)
        # input is CHW
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]
        #
        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        #         print("afterup", x.shape)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Softmax(dim=1),
        )

    #         self.out = torch.argmax(self.softmax,dim=1)

    def forward(self, x):
        x = self.conv(x)
        #         x = torch.argmax(x,dim=1)
        #         x = self.softmax(x)
        #         x = self.out(x)
        return x


class Up_2_skip(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 3)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 3, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        #         print("up", x1.shape)
        # input is CHW
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]
        #
        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        # print("afterup", x1.shape)
        x = torch.cat([x2,x2, x1], dim=1)
        # print("afterup", x.shape)
        return self.conv(x)


class Res_UNet(nn.Module):
    def __init__(self, n_channels, classes, bilinear=True):
        super(Res_UNet, self).__init__()
        self.n_channels = n_channels
        self.classes = classes
        self.bilinear = bilinear

        self.inc = BasicBlock(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1

        self.down4 = Down(256, 512 // factor)

        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # print(x4.shape, x5.shape)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits



class UNet_2_skip(nn.Module):
    def __init__(self, n_channels, classes, bilinear=True):
        super(UNet_2_skip, self).__init__()
        self.n_channels = n_channels
        self.classes = classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        factor = 2 if bilinear else 1

        self.down4 = Down(128, 256 // factor)

        self.up1 = Up_2_skip(128*3, 128 // factor, bilinear)
        self.up2 = Up_2_skip(64*3, 64 // factor, bilinear)
        self.up3 = Up_2_skip(32*3, 32 // factor, bilinear)
        self.up4 = Up_2_skip(16*3, 16, bilinear)
        self.outc = OutConv(16, classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # print(x4.shape, x5.shape)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == "__main__":
    model = Res_UNet(1, 4)
    # print(model)
    summary(model, (1, 512, 512))
    writer = SummaryWriter()
    writer.add_graph(model, torch.randn((1, 1,512,512)))
    # print( summary(model, (1,512,512)))

    # model = BasicBlock(inplanes=1, planes=1)
    # print(summary(model, (1, 512, 512)))