import torch
import torch.nn as nn
from .modules import *

# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
base_channel = int(64 * 0.25)


def VGG():
    layers = []
    layers += [BasicConv(3, base_channel)]

    layers += [BasicConv(base_channel, base_channel, kernel_size=3, stride=2, padding=1)]  # 150 * 150
    layers += [BasicConv(base_channel, base_channel * 2, kernel_size=3, stride=1, padding=1)]

    layers += [BasicConv(base_channel * 2, base_channel * 2, stride=2, kernel_size=3, padding=1)]  # 75 * 75
    layers += [BasicConv(base_channel * 2, base_channel * 4, stride=1, kernel_size=3, padding=1)]
    layers += [BasicConv(base_channel * 4, base_channel * 4, stride=1, kernel_size=3, padding=1)]

    layers += [BasicConv(base_channel * 4, base_channel * 4, stride=2, kernel_size=3, padding=1)]  # 38 * 38
    layers += [BasicConv(base_channel * 4, base_channel * 8, stride=1, kernel_size=3, padding=1)]
    layers += [BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1)]

    layers += [BasicConv(base_channel * 8, base_channel * 8, stride=2, kernel_size=3, padding=1)]  # 19 * 19
    layers += [BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1)]
    layers += [BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1)]
    layers += [BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1)]
    layers += [BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1)]
    layers += [BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1)]
    layers += [BasicConv(base_channel * 8, 128, kernel_size=1, stride=1, padding=0)]

    layers += [BasicConv(128, 128, kernel_size=3, stride=2, padding=1)]  # 10*10

    return layers


class NewVGG(nn.Module):
    def __init__(self):
        self.stage1 = nn.Sequential(
            BasicConv(3, base_channel),
            BasicConv(base_channel, base_channel, kernel_size=3, stride=2, padding=1),  # 150 * 150
            BasicConv(base_channel, base_channel * 2, kernel_size=3, stride=1, padding=1)
        )

        self.stage2 = nn.Sequential(
            BasicConv(base_channel * 2, base_channel * 2, stride=2, kernel_size=3, padding=1),  # 75 * 75
            BasicConv(base_channel * 2, base_channel * 4, stride=1, kernel_size=3, padding=1),
            BasicConv(base_channel * 4, base_channel * 4, stride=1, kernel_size=3, padding=1)
        )

        self.stage3 = nn.Sequential(
            BasicConv(base_channel * 4, base_channel * 4, stride=2, kernel_size=3, padding=1),  # 38 * 38
            BasicConv(base_channel * 4, base_channel * 8, stride=1, kernel_size=3, padding=1),
            BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1)
        )

        self.stage4 = nn.Sequential(
            BasicConv(base_channel * 8, base_channel * 8, stride=2, kernel_size=3, padding=1),  # 19 * 19
            BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1),
            BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1),
            BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1),
            BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1),
            BasicConv(base_channel * 8, base_channel * 8, stride=1, kernel_size=3, padding=1)

        )

        self.stage5 = nn.Sequential(
            BasicConv(base_channel * 8, 128, kernel_size=1, stride=1, padding=0),
            BasicConv(128, 128, kernel_size=3, stride=2, padding=1)  # 10*10
        )

        def forward(self, x):
            x1 = self.stage1(x)
            x2 = self.stage2(x1)
            x3 = self.stage3(x2)
            x4 = self.stage4(x3)
            x5 = self.stage5(x4)

            return [x3, x4, x5]


def VGG_Mobile():
    layers = []
    layers += [BasicConv(3, base_channel * 2, kernel_size=3, stride=2, padding=1)]
    layers += [DWConv(base_channel * 2, base_channel * 2, stride=1)]  # 150 * 150 conv1

    layers += [DWConv(base_channel * 2, base_channel * 4, stride=1)]  # conv2
    layers += [DWConv(base_channel * 4, base_channel * 4, stride=2)]  # 75 * 75  conv3

    layers += [DWConv(base_channel * 4, base_channel * 8, stride=1)]  # conv4
    layers += [DWConv(base_channel * 8, base_channel * 8, stride=1)]  # conv5
    layers += [DWConv(base_channel * 8, base_channel * 8, stride=2)]  # 38 * 38 conv6

    layers += [DWConv(base_channel * 8, base_channel * 16, stride=1)]  # conv7
    layers += [DWConv(base_channel * 16, base_channel * 16, stride=1)]  # conv8
    layers += [DWConv(base_channel * 16, base_channel * 16, stride=2)]  # 19 * 19 conv9

    layers += [DWConv(base_channel * 16, base_channel * 16, stride=1)]  # conv10
    layers += [DWConv(base_channel * 16, base_channel * 16, stride=1)]  # conv11
    layers += [DWConv(base_channel * 16, base_channel * 16, stride=1)]  # conv12

    layers += [DWConv(base_channel * 16, base_channel * 16, stride=1)]  # conv13
    layers += [DWConv(base_channel * 16, base_channel * 16, stride=1)]  # conv14

    layers += [DWConv(base_channel * 16, 256, stride=1)]  # conv15
    layers += [DWConv(256, 256, stride=2)]  # 10*10 16

    layers += [BasicConv(256, 128, kernel_size=1, stride=1, padding=0)]
    layers += [BasicConv(128, 256, kernel_size=3, stride=2, padding=1)]  # 5 * 5

    return layers


def VGG_MobileLittle():
    layers = []
    layers += [BasicConv(3, base_channel, kernel_size=3, stride=2, padding=1)]
    layers += [DWConv(base_channel, base_channel, stride=1)]  # 150 * 150 conv1

    layers += [DWConv(base_channel, base_channel * 2, stride=1)]  # conv2
    layers += [DWConv(base_channel * 2, base_channel * 2, stride=2)]  # 75 * 75  conv3

    layers += [DWConv(base_channel * 2, base_channel * 4, stride=1)]  # conv4
    layers += [DWConv(base_channel * 4, base_channel * 4, stride=1)]  # conv5
    layers += [DWConv(base_channel * 4, base_channel * 4, stride=2)]  # 38 * 38 conv6

    layers += [DWConv(base_channel * 4, base_channel * 8, stride=1)]  # conv7
    layers += [DWConv(base_channel * 8, base_channel * 8, stride=1)]  # conv8
    layers += [DWConv(base_channel * 8, base_channel * 8, stride=2)]  # 19 * 19 conv9

    layers += [DWConv(base_channel * 8, base_channel * 8, stride=1)]  # conv10
    layers += [DWConv(base_channel * 8, base_channel * 8, stride=1)]  # conv11
    layers += [DWConv(base_channel * 8, base_channel * 8, stride=1)]  # conv12

    layers += [DWConv(base_channel * 8, base_channel * 8, stride=1)]  # conv13
    layers += [DWConv(base_channel * 8, base_channel * 8, stride=1)]  # conv14

    layers += [DWConv(base_channel * 8, 128, stride=1)]  # conv15
    layers += [DWConv(128, 128, stride=2)]  # 10*10 16

    # layers += [BasicConv(128,128,kernel_size=1,stride=1,padding=0)]
    # layers += [BasicConv(128,128,kernel_size=3,stride=2, padding=1)] # 5 * 5

    return layers


def VGG_MobileLittle_v3():
    layers = []
    layers += [BasicConv(3, base_channel, kernel_size=3, stride=2, padding=1)]
    layers += [DWConv(base_channel, base_channel * 2, stride=1)]  # conv2

    layers += [DWConv(base_channel * 2, base_channel * 2, stride=2)]  # 75 * 75  conv3
    layers += [DWConv(base_channel * 2, base_channel * 2, stride=1)]  # conv4

    layers += [DWConv(base_channel * 2, base_channel * 4, stride=2)]  # 38 * 38 conv6
    layers += [DWConv(base_channel * 4, base_channel * 4, stride=1)]  # conv7
    layers += [DWConv(base_channel * 4, base_channel * 4, stride=1)]  # conv8
    layers += [DWConv(base_channel * 4, base_channel * 4, stride=1)]  # conv8

    layers += [DWConv(base_channel * 4, base_channel * 8, stride=2)]  # 19 * 19 conv9
    layers += [DWConv(base_channel * 8, base_channel * 8, stride=1)]  # conv10
    layers += [DWConv(base_channel * 8, base_channel * 8, stride=1)]  # conv11

    layers += [DWConv(base_channel * 8, base_channel * 16, stride=2)]  # conv12
    layers += [DWConv(base_channel * 16, base_channel * 16, stride=1)]  # conv13

    return layers


if __name__ == "__main__":
    from ptflops import get_model_complexity_info


    class model(nn.Module):
        def __init__(self):
            super(model, self).__init__()
            self.feature = nn.ModuleList(VGG_MobileLittle_v3())

        def forward(self, x):
            for m in self.feature:
                x = m(x)
            return x


    net = model()
    flops, params = get_model_complexity_info(net, (260, 260))
