import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.models as models
import torch.backends.cudnn as cudnn
import os
import torch.nn.init as init
from .modules import *
from .FPN import PyramidFeatures, BiFPN, ABFPN, AC_FPN
from .VGG import base_channel, VGG, VGG_MobileLittle


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

    def __init__(self, phase, backbone, neck, head, num_classes, img_dim=300):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes

        # SSD network
        self.base = nn.ModuleList(backbone)
        # Layer learns to scale the l2 normalized features from conv4_3
        # self.L2Norm = L2Norm(512, 20)
        self.Norm = BasicRFB(128, 128, stride=1, scale=1.0)
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

        # apply vgg up to conv4_3 relu
        for k in range(9):
            x = self.base[k](x)

        fpn_sources.append(x)

        for k in range(9, 15):
            x = self.base[k](x)
        fpn_sources.append(x)

        for k in range(15, len(self.base)):
            x = self.base[k](x)

        fpn_sources.append(x)

        features = self.fpn(fpn_sources)

        features[0] = self.Norm(features[0])

        for k, v in enumerate(self.extras):
            x = v(x)
            # print(x.shape)
            if k % 2 == 1:
                features.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(features, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

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


def add_extras(img_dim=300):
    # Extra layers added to VGG for feature scaling
    layers = []

    if img_dim == 300:
        layers += [BasicConv(128, 128, kernel_size=1, stride=1, padding=0)]
        layers += [BasicConv(128, 256, kernel_size=3, stride=2, padding=1)]  # 5 * 5

        layers += [BasicConv(256, 128, kernel_size=1, stride=1, padding=0)]
        layers += [BasicConv(128, 256, kernel_size=3, stride=1, padding=0)]  # 3 * 3

        layers += [BasicConv(256, 128, kernel_size=1, stride=1, padding=0)]
        layers += [BasicConv(128, 256, kernel_size=3, stride=1, padding=0)]  # 1 * 1
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

    # 38*38  512
    loc_layers += [nn.Conv2d(128, cfg[0] * 4, kernel_size=1, padding=0)]
    conf_layers += [nn.Conv2d(128, cfg[0] * num_classes, kernel_size=1, padding=0)]

    # 19*19  512
    loc_layers += [nn.Conv2d(128, cfg[1] * 4, kernel_size=1, padding=0)]
    conf_layers += [nn.Conv2d(128, cfg[1] * num_classes, kernel_size=1, padding=0)]

    # 10*10  256
    loc_layers += [nn.Conv2d(128, cfg[2] * 4, kernel_size=1, padding=0)]
    conf_layers += [nn.Conv2d(128, cfg[2] * num_classes, kernel_size=1, padding=0)]

    # 5*5  256
    loc_layers += [nn.Conv2d(256, cfg[3] * 4, kernel_size=1, padding=0)]
    conf_layers += [nn.Conv2d(256, cfg[3] * num_classes, kernel_size=1, padding=0)]

    # 3*3  256
    loc_layers += [nn.Conv2d(256, cfg[4] * 4, kernel_size=1, padding=0)]
    conf_layers += [nn.Conv2d(256, cfg[4] * num_classes, kernel_size=1, padding=0)]

    # 1*1  256
    loc_layers += [nn.Conv2d(256, cfg[5] * 4, kernel_size=1, padding=0)]
    conf_layers += [nn.Conv2d(256, cfg[5] * num_classes, kernel_size=1, padding=0)]

    return (loc_layers, conf_layers)


mbox = {
    '300': [6, 6, 6, 6, 6, 6],  # number of boxes per feature map location
    '512': [6, 6, 6, 6, 6, 6],
}


def build_net(phase, size=300, num_classes=21, neck_type='FPN'):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size != 300 and size != 512:
        print("Error: Sorry only RFBNet300 and RFBNet512 are supported!")
        return

    backbone = VGG()
    if neck_type == 'BIFPN':
        BiFPN_inputs = [128, 128, 128]
        neck = BiFPN(BiFPN_inputs, 128, phase)
    elif neck_type == 'FPN':
        neck = PyramidFeatures(base_channel * 8, base_channel * 8, 128, img_dim=size)
    elif neck_type == 'ABFPN':
        ABFPN_inputs = [128, 128, 128]
        neck = ABFPN(ABFPN_inputs, 128, phase)
    elif neck_type == 'ACFPN':
        neck = AC_FPN(128, 128, 128)
    head = build_head(mbox[str(size)], num_classes)
    return SSD(phase, backbone, neck, head, num_classes, img_dim=size)


if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    import os
    import time

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    net = build_net('test', 300, 7)

    net.eval()
    net = net.cuda()
    cudnn.benchmark = True

    f, p = get_model_complexity_info(net, (3, 300, 300), as_strings=True, print_per_layer_stat=False)
    print('FLOPs:', f, 'Parms:', p)

    net.cuda()
    x = torch.randn(1, 3, 300, 300).cuda()
    with torch.no_grad():
        s = time.clock()
        y = net(x)
        print('fps', 1 / (time.clock() - s))