import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.models as models
import torch.backends.cudnn as cudnn
import os
import sys
import torch.nn.init as init
from .modules import *
from .FPN import PyramidFeatures
from .VGG import base_channel, VGG_MobileLittle_v3


class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=64, img_dim=260):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        # self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        if img_dim == 260:
            self.P5_upsampled = nn.ConvTranspose2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.ConvTranspose2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs
        # print(C5.shape)
        # print(C4.shape)
        # print(C3.shape)
        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        # print(P5_upsampled_x.shape)
        # print(P4_x.shape)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        return [P3_x, P4_x, P5_x]


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, backbone, neck, head, num_classes, img_dim=260):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes

        # SSD network
        self.base = nn.ModuleList(backbone)
        # Layer learns to scale the l2 normalized features from conv4_3
        # self.L2Norm = L2Norm(512, 20)
        # self.Norm = BasicRFB(128,128,stride = 1,scale=1.0)
        self.fpn = neck
        self.extras = nn.ModuleList(add_extras(img_dim))

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        fpn_sources = list()
        loc = list()
        conf = list()

        for k in range(8):
            x = self.base[k](x)
            # print(x.shape)

        # s = self.L2Norm(x)
        # sources.append(x)
        sources.append(x)

        for k in range(8, 11):
            x = self.base[k](x)
        sources.append(x)

        # apply vgg up to fc7
        for k in range(11, len(self.base)):
            x = self.base[k](x)
            # print(x.shape)
        # sources.append(x)
        sources.append(x)

        # features = self.fpn(fpn_sources)

        # features[0] = self.Norm(features[0])
        # print(features[0].shape)
        # print(features[0].cpu().data.numpy()[0,0:10,0,0])
        # sys.exit()

        for k, v in enumerate(self.extras):
            x = v(x)
            # print(x.shape)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            # print(x.shape)

            # tmploc = l(x).permute(0, 2, 3, 1).contiguous()
            tmploc = l(x)
            # print(tmploc.shape)
            # print(tmploc.cpu().data.numpy()[0,0:10,0,0])
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            # break
        # sys.exit()

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            # output = (
            #     loc.view(loc.size(0), -1, 4),  # loc preds
            #     self.softmax(conf.view(-1, self.num_classes)),  # conf preds
            # )
            output = torch.cat([loc.view(loc.size(0), -1, 4),
                                self.softmax(conf.view(-1, self.num_classes)).unsqueeze(0)], 2)
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                            map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


def add_extras(img_dim=240):
    # Extra layers added to VGG for feature scaling
    layers = []

    if img_dim == 240:
        layers += [BasicConv(base_channel * 16, base_channel * 4, kernel_size=1, stride=1, padding=0)]
        layers += [BasicConv(base_channel * 4, base_channel * 16, kernel_size=3, stride=2, padding=1)]  # 3 * 3

    elif img_dim == 512:
        layers += [BasicConv(128, 128, kernel_size=1, stride=1, padding=0)]
        layers += [BasicConv(128, 256, kernel_size=3, stride=2, padding=1)]  # 8 * 8

        layers += [BasicConv(256, 128, kernel_size=1, stride=1, padding=0)]
        layers += [BasicConv(128, 256, kernel_size=3, stride=2, padding=1)]  # 4 * 4

        layers += [BasicConv(256, 128, kernel_size=1, stride=1, padding=0)]
        layers += [BasicConv(128, 256, kernel_size=3, stride=2, padding=1)]  # 2 * 2

    return layers


def build_head(cfg, num_classes):
    loc_layers = []
    conf_layers = []
    # print(cfg[0])
    # 38*38  512
    loc_layers += [nn.Conv2d(base_channel * 4, cfg[0] * 4, kernel_size=1, padding=0)]
    conf_layers += [nn.Conv2d(base_channel * 4, cfg[0] * num_classes, kernel_size=1, padding=0)]

    # 19*19  512
    loc_layers += [nn.Conv2d(base_channel * 8, cfg[1] * 4, kernel_size=1, padding=0)]
    conf_layers += [nn.Conv2d(base_channel * 8, cfg[1] * num_classes, kernel_size=1, padding=0)]

    # 10*10  256
    loc_layers += [nn.Conv2d(base_channel * 16, cfg[2] * 4, kernel_size=1, padding=0)]
    conf_layers += [nn.Conv2d(base_channel * 16, cfg[2] * num_classes, kernel_size=1, padding=0)]

    # 5*5  256
    loc_layers += [nn.Conv2d(base_channel * 16, cfg[3] * 4, kernel_size=1, padding=0)]
    conf_layers += [nn.Conv2d(base_channel * 16, cfg[3] * num_classes, kernel_size=1, padding=0)]

    return (loc_layers, conf_layers)


mbox = {
    '240': [6, 6, 6, 6],  # number of boxes per feature map location
    '512': [6, 6, 6, 6],
}


def build_net(phase, size=240, num_classes=21, neck_type='FPN'):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size != 240 and size != 512:
        print("Error: Sorry only RFBNet300 and RFBNet512 are supported!")
        return

    backbone = VGG_MobileLittle_v3()
    neck = PyramidFeatures(base_channel * 4, base_channel * 4, 64, img_dim=size)
    head = build_head(mbox[str(size)], num_classes)
    return SSD(phase, backbone, neck, head, num_classes, img_dim=size)


if __name__ == "__main__":
    net = build_net('train', num_classes=2)

    # print(net)
    # print(x.shape)

    from ptflops import get_model_complexity_info

    img_dim = 240
    flops, params = get_model_complexity_info(net, (3, img_dim, img_dim), as_strings=True, print_per_layer_stat=True)
    print('Flops: ' + flops)
    print('Params: ' + params)

    # def hook(self, input, output):
    #     print(output.data.cpu().numpy().shape)
    #
    # for m in net.modules():
    #     if isinstance(m, nn.Conv2d):
    #         m.register_forward_hook(hook)

    x = torch.randn(10, 3, 240, 240)
    y = net(x)
