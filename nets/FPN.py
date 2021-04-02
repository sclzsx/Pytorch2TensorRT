import torch
import torch.nn as nn
from .modules import *
import torch.nn.functional as F


class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=128, img_dim=300):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        # self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        if img_dim == 300:
            self.P5_upsampled = nn.ConvTranspose2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)
        elif img_dim == 512:
            self.P5_upsampled = nn.ConvTranspose2d(feature_size, feature_size, kernel_size=2, stride=2)
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.ConvTranspose2d(feature_size, feature_size, kernel_size=2, stride=2)
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        return [P3_x, P4_x, P5_x]


class DetNet_FPN(nn.Module):
    def __init__(self, in_channels, feature_size=128):
        super(DetNet_FPN, self).__init__()

        self.latern_convs = nn.ModuleList()

        self.num = len(in_channels)
        for i in range(self.num):
            conv = nn.Conv2d(in_channels[i], feature_size, kernel_size=1)
            self.latern_convs.append(conv)

        self.up_samples = nn.Sequential(
            nn.ConvTranspose2d(feature_size, feature_size, kernel_size=2, stride=2),
            nn.ConvTranspose2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)
        )

        self.last_convs = nn.ModuleList()
        for i in range(self.num):
            conv = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
            self.last_convs.append(conv)

    def forward(self, inputs):
        assert self.num == len(inputs)

        for i in range(self.num):
            inputs[i] = self.latern_convs[i](inputs[i])

        inputs[3] = inputs[3] + inputs[4]
        inputs[2] = inputs[3] + inputs[2]
        inputs[1] = self.up_samples[1](inputs[2]) + inputs[1]
        inputs[0] = self.up_samples[0](inputs[1]) + inputs[0]

        for i in range(self.num):
            inputs[i] = self.last_convs[i](inputs[i])

        return inputs


class CEM_block(nn.Module):
    def __init__(self, in_planes, down_planes, out_planes, dilation_rate, Bn_or_Drop="Bn"):
        super(CEM_block, self).__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_planes, down_planes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(down_planes),
            nn.ReLU(inplace=True),
        )
        if (Bn_or_Drop == "Bn"):
            self.dilation_conv = nn.Sequential(
                nn.Conv2d(down_planes, out_planes, kernel_size=3, stride=1, padding=dilation_rate,
                          dilation=dilation_rate, bias=False),
                nn.BatchNorm2d(out_planes),
                nn.ReLU(inplace=True),
            )
        elif (Bn_or_Drop == "Drop"):
            self.dilation_conv = nn.Sequential(
                nn.Conv2d(down_planes, out_planes, kernel_size=3, stride=1, padding=dilation_rate,
                          dilation=dilation_rate, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            )

    def forward(self, x):
        x = self.down_conv(x)
        x = self.dilation_conv(x)
        return x


class CEM(nn.Module):
    def __init__(self, fsize_in=128, fsize_down=24, fsize_out=32, dilation_rates=(1, 2, 3)):
        super(CEM, self).__init__()
        self.fsize_in = fsize_in
        self.fsize_out = fsize_out
        self.fsize_down = fsize_down
        self.conv1 = BasicConv(self.fsize_in, self.fsize_in, kernel_size=1, stride=1, relu=False)
        self.dilation_rates = dilation_rates
        self.CEM_block1 = CEM_block(self.fsize_in, self.fsize_down, self.fsize_out, self.dilation_rates[0])
        self.CEM_block2 = CEM_block(self.fsize_in + self.fsize_out * 1, self.fsize_down, self.fsize_out,
                                    self.dilation_rates[1])
        self.CEM_block3 = CEM_block(self.fsize_in + self.fsize_out * 2, self.fsize_down, self.fsize_out,
                                    self.dilation_rates[2])
        self.refine_conv = BasicConv(3 * self.fsize_out, self.fsize_in, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x_ = self.conv1(x)
        x0 = self.CEM_block1(x)
        # print("x0:",list(x0.size()))
        f0 = torch.cat((x, x0), 1)
        # print("f0:",list(f0.size()))
        x1 = self.CEM_block2(f0)
        # print("x1:",list(x1.size()))
        f1 = torch.cat((x1, f0), 1)
        # print("f1:",list(f1.size()))
        x2 = self.CEM_block3(f1)
        # print("x2:",list(x2.size()))
        f2 = torch.cat((x0, x1, x2), 1)
        # print("f2:",list(f2.size()))
        x3 = self.refine_conv(f2)
        # print("x3:",list(x3.size()))
        x4 = x_ + x3
        # print("x4:",list(x4.size()))
        return x4


class AC_FPN(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=128):
        super(AC_FPN, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        # self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_upsampled = nn.ConvTranspose2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.ConvTranspose2d(feature_size, feature_size, kernel_size=2, stride=2)
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.CEM = CEM(fsize_in=C5_size, fsize_down=24, fsize_out=32, dilation_rates=(1, 2, 3))

    def forward(self, inputs):
        C3, C4, C5 = inputs
        C5 = self.CEM(C5)
        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        return [P3_x, P4_x, P5_x]


class ABFPN(nn.Module):
    def __init__(self, in_channels, out_channels, step):
        super(ABFPN, self).__init__()
        self.levels = len(in_channels)
        self.step = step
        self.lateral_convs = nn.ModuleList()
        self.bifpn_convs = nn.ModuleList()

        for i in range(0, self.levels):
            l_conv = BasicConv(in_channels[i], out_channels, kernel_size=1)
            self.lateral_convs.append(l_conv)

        for jj in range(2):
            for i in range(self.levels - 1):
                fpn_conv = BasicDwConv(out_channels, out_channels, kernel_size=3, padding=1)
                self.bifpn_convs.append(fpn_conv)

        self.up_sample_convs = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
        )
        self.CEM = CEM(fsize_in=in_channels[-1], fsize_down=24, fsize_out=32, dilation_rates=(1, 2, 3))

    def forward(self, inputs):
        levels = self.levels
        inputs[levels - 1] = self.CEM(inputs[levels - 1])
        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down
        idx_bifpn = 0
        pathtd = laterals
        inputs_clone = []
        for in_tensor in laterals:
            inputs_clone.append(in_tensor.clone())

        for i in range(levels - 1, 0, -1):
            pathtd[i - 1] = pathtd[i - 1] + self.up_sample_convs[i - 1](pathtd[i])
            pathtd[i - 1] = self.bifpn_convs[idx_bifpn](pathtd[i - 1])
            idx_bifpn = idx_bifpn + 1

        # build down-top
        for i in range(0, levels - 2, 1):
            if self.step is 'test':
                pathtd[i + 1] = pathtd[i + 1] + F.max_pool2d(pathtd[i], kernel_size=3, stride=2, padding=0,
                                                             ceil_mode=True) + inputs_clone[i + 1]
            elif self.step is 'train':
                pathtd[i + 1] = pathtd[i + 1] + F.max_pool2d(pathtd[i], kernel_size=3, stride=2, padding=1) + \
                                inputs_clone[i + 1]
            pathtd[i + 1] = self.bifpn_convs[idx_bifpn](pathtd[i + 1])
            idx_bifpn = idx_bifpn + 1
        if self.step is 'test':
            pathtd[levels - 1] = pathtd[levels - 1] + F.max_pool2d(pathtd[levels - 2], kernel_size=3, stride=2,
                                                                   padding=1, ceil_mode=True)
        elif self.step is 'train':
            pathtd[levels - 1] = pathtd[levels - 1] + F.max_pool2d(pathtd[levels - 2], kernel_size=3, stride=2,
                                                                   padding=1)
        pathtd[levels - 1] = self.bifpn_convs[idx_bifpn](pathtd[levels - 1])
        # print("pathtd[0].size: ",list(pathtd[0].size()))
        # print("pathtd[1].size: ",list(pathtd[1].size()))
        # print("pathtd[2].size: ",list(pathtd[2].size()))
        return pathtd


class BiFPN(nn.Module):
    def __init__(self, in_channels, out_channels, step):
        super(BiFPN, self).__init__()
        self.levels = len(in_channels)
        self.step = step
        self.lateral_convs = nn.ModuleList()
        self.bifpn_convs = nn.ModuleList()

        for i in range(0, self.levels):
            l_conv = BasicConv(in_channels[i], out_channels, kernel_size=1)
            self.lateral_convs.append(l_conv)

        for jj in range(2):
            for i in range(self.levels - 1):
                fpn_conv = BasicDwConv(out_channels, out_channels, kernel_size=3, padding=1)
                self.bifpn_convs.append(fpn_conv)

        self.up_sample_convs = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
        )

    def forward(self, inputs):
        levels = self.levels

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down
        idx_bifpn = 0
        pathtd = laterals
        inputs_clone = []
        for in_tensor in laterals:
            inputs_clone.append(in_tensor.clone())

        for i in range(levels - 1, 0, -1):
            pathtd[i - 1] = pathtd[i - 1] + self.up_sample_convs[i - 1](pathtd[i])
            pathtd[i - 1] = self.bifpn_convs[idx_bifpn](pathtd[i - 1])
            idx_bifpn = idx_bifpn + 1

        # build down-top
        for i in range(0, levels - 2, 1):
            if self.step is 'test':
                pathtd[i + 1] = pathtd[i + 1] + F.max_pool2d(pathtd[i], kernel_size=3, stride=2, padding=0,
                                                             ceil_mode=True) + inputs_clone[i + 1]
            elif self.step is 'train':
                pathtd[i + 1] = pathtd[i + 1] + F.max_pool2d(pathtd[i], kernel_size=3, stride=2, padding=1) + \
                                inputs_clone[i + 1]
            pathtd[i + 1] = self.bifpn_convs[idx_bifpn](pathtd[i + 1])
            idx_bifpn = idx_bifpn + 1
        if self.step is 'test':
            pathtd[levels - 1] = pathtd[levels - 1] + F.max_pool2d(pathtd[levels - 2], kernel_size=3, stride=2,
                                                                   padding=1, ceil_mode=True)
        elif self.step is 'train':
            pathtd[levels - 1] = pathtd[levels - 1] + F.max_pool2d(pathtd[levels - 2], kernel_size=3, stride=2,
                                                                   padding=1)
        pathtd[levels - 1] = self.bifpn_convs[idx_bifpn](pathtd[levels - 1])
        # print("pathtd[0].size: ",list(pathtd[0].size()))
        # print("pathtd[1].size: ",list(pathtd[1].size()))
        # print("pathtd[2].size: ",list(pathtd[2].size()))
        return pathtd
