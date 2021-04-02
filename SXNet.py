import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, visual=1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual,
                      relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=1),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=visual + 1,
                      dilation=visual + 1, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=2 * visual + 1,
                      dilation=2 * visual + 1, relu=False)
        )

        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short

        return out


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


class conv_norm_act(nn.Module):
    def __init__(self, in_c, out_c, k, pad=0, dil=1, bias=True, act_name='relu', asy=False):
        super(conv_norm_act, self).__init__()
        if asy and k == 3 and in_c == out_c and dil == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=(3, 1), padding=(dil, 0), dilation=dil, bias=bias),
                nn.Conv2d(out_c, out_c, kernel_size=(1, 3), padding=(0, dil), dilation=dil, bias=bias))
        elif asy and k == 5 and in_c == out_c and dil == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=(5, 1), padding=(dil + 1, 0), dilation=dil, bias=bias),
                nn.Conv2d(out_c, out_c, kernel_size=(1, 5), padding=(0, dil + 1), dilation=dil, bias=bias))
        else:
            self.conv = nn.Conv2d(in_c, out_c, kernel_size=k, padding=pad, dilation=dil, bias=bias)

        if dil == 1:
            self.norm = nn.BatchNorm2d(out_c)
        else:
            self.norm = None

        if act_name == 'mish':
            self.act = Mish()
        elif act_name == 'lelu':
            self.act = nn.LeakyReLU()
        elif act_name == 'pelu':
            self.act = nn.PReLU()
        else:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.act(x)
        return x


class mkm_bone(nn.Module):
    def __init__(self, act_name, asy, bias, basic_ch):
        super(mkm_bone, self).__init__()
        self.conv_1_2 = conv_norm_act(basic_ch, basic_ch, k=1, pad=0, bias=bias, act_name=act_name, asy=asy)
        self.conv_1_3 = conv_norm_act(basic_ch, basic_ch, k=3, pad=1, bias=bias, act_name=act_name, asy=asy)
        self.conv_1_4 = conv_norm_act(basic_ch, basic_ch, k=5, pad=2, bias=bias, act_name=act_name, asy=asy)
        self.conv_1_5 = conv_norm_act(basic_ch, basic_ch, k=3, pad=2, dil=2, bias=bias, act_name=act_name, asy=asy)
        self.conv_1_6 = conv_norm_act(basic_ch, basic_ch, k=1, pad=0, bias=bias, act_name=act_name, asy=asy)
        self.conv_1_7 = conv_norm_act(basic_ch, basic_ch, k=3, pad=1, bias=bias, act_name=act_name, asy=asy)
        self.conv_1_8 = conv_norm_act(basic_ch, basic_ch, k=5, pad=2, bias=bias, act_name=act_name, asy=asy)
        self.conv_1_9 = conv_norm_act(basic_ch, basic_ch, k=3, pad=2, dil=2, bias=bias, act_name=act_name, asy=asy)
        self.conv_1_10 = conv_norm_act(basic_ch, basic_ch, k=1, pad=0, bias=bias, act_name=act_name, asy=asy)
        self.conv_1_11 = conv_norm_act(basic_ch, basic_ch, k=3, pad=1, bias=bias, act_name=act_name, asy=asy)
        self.conv_1_12 = conv_norm_act(basic_ch, basic_ch, k=5, pad=2, bias=bias, act_name=act_name, asy=asy)
        self.conv_1_13 = conv_norm_act(basic_ch, basic_ch, k=3, pad=2, dil=2, bias=bias, act_name=act_name, asy=asy)
        self.conv_1_14 = conv_norm_act(basic_ch, basic_ch, k=1, pad=0, bias=bias, act_name=act_name, asy=asy)
        self.conv_1_15 = conv_norm_act(basic_ch, basic_ch, k=3, pad=1, bias=bias, act_name=act_name, asy=asy)
        self.conv_1_16 = conv_norm_act(basic_ch, basic_ch, k=5, pad=2, bias=bias, act_name=act_name, asy=asy)

    def forward(self, x):
        x11 = self.conv_1_2(x)
        x12 = self.conv_1_3(x)
        x13 = self.conv_1_4(x)
        x1 = x11 + x12 + x13
        x1 = self.conv_1_5(x1)

        x21 = self.conv_1_6(x1)
        x22 = self.conv_1_7(x1)
        x23 = self.conv_1_8(x1)
        x2 = x21 + x22 + x23
        x2 = self.conv_1_9(x2)

        x31 = self.conv_1_10(x2)
        x32 = self.conv_1_11(x2)
        x33 = self.conv_1_12(x2)
        x3 = x31 + x32 + x33
        x3 = self.conv_1_13(x3)

        x41 = self.conv_1_14(x3)
        x42 = self.conv_1_15(x3)
        x43 = self.conv_1_16(x3)
        x4 = x41 + x42 + x43
        return x4


class rm_bone(nn.Module):
    def __init__(self, act_name, asy, bias, basic_ch):
        super(rm_bone, self).__init__()
        self.conv_2_2 = conv_norm_act(basic_ch, basic_ch, k=3, pad=1, bias=bias, act_name=act_name, asy=asy)
        self.conv_2_3 = conv_norm_act(basic_ch, basic_ch, k=3, pad=1, bias=bias, act_name=act_name, asy=asy)
        self.conv_2_4 = conv_norm_act(basic_ch, basic_ch, k=3, pad=1, bias=bias, act_name=act_name, asy=asy)
        self.conv_2_5 = conv_norm_act(basic_ch, basic_ch, k=3, pad=2, dil=2, bias=bias, act_name=act_name, asy=asy)
        self.conv_2_6 = conv_norm_act(basic_ch, basic_ch, k=3, pad=1, bias=bias, act_name=act_name, asy=asy)
        self.conv_2_7 = conv_norm_act(basic_ch, basic_ch, k=3, pad=1, bias=bias, act_name=act_name, asy=asy)
        self.conv_2_8 = conv_norm_act(basic_ch, basic_ch, k=3, pad=1, bias=bias, act_name=act_name, asy=asy)
        self.conv_2_9 = conv_norm_act(basic_ch, basic_ch, k=3, pad=2, dil=2, bias=bias, act_name=act_name, asy=asy)
        self.conv_2_10 = conv_norm_act(basic_ch, basic_ch, k=3, pad=1, bias=bias, act_name=act_name, asy=asy)
        self.conv_2_11 = conv_norm_act(basic_ch, basic_ch, k=3, pad=1, bias=bias, act_name=act_name, asy=asy)
        self.conv_2_12 = conv_norm_act(basic_ch, basic_ch, k=3, pad=1, bias=bias, act_name=act_name, asy=asy)
        self.conv_2_13 = conv_norm_act(basic_ch, basic_ch, k=3, pad=2, dil=2, bias=bias, act_name=act_name, asy=asy)
        self.conv_2_14 = conv_norm_act(basic_ch, basic_ch, k=3, pad=1, bias=bias, act_name=act_name, asy=asy)
        self.conv_2_15 = conv_norm_act(basic_ch, basic_ch, k=3, pad=1, bias=bias, act_name=act_name, asy=asy)
        self.conv_2_16 = conv_norm_act(basic_ch, basic_ch, k=3, pad=1, bias=bias, act_name=act_name, asy=asy)

    def forward(self, x):
        y1 = self.conv_2_2(x)
        y1 = self.conv_2_3(y1)
        y1 = self.conv_2_4(y1)
        y1 = x + y1
        dil1 = self.conv_2_5(y1)

        y1 = self.conv_2_6(dil1)
        y1 = self.conv_2_7(y1)
        y1 = self.conv_2_8(y1)
        y1 = dil1 + y1
        dil2 = self.conv_2_9(y1)

        y1 = self.conv_2_10(dil2)
        y1 = self.conv_2_11(y1)
        y1 = self.conv_2_12(y1)
        y1 = dil2 + y1
        dil3 = self.conv_2_13(y1)

        y1 = self.conv_2_14(dil3)
        y1 = self.conv_2_15(y1)
        y1 = self.conv_2_16(y1)
        y1 = dil3 + y1
        return y1


class TLU(nn.Module):
    def __init__(self, in_c, basic_ch, act_name, asy, bias):
        super(TLU, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=in_c, out_channels=basic_ch, kernel_size=3, padding=1, bias=True)
        self.conv_2 = conv_norm_act(basic_ch, basic_ch, k=3, pad=1, bias=bias, act_name=act_name, asy=asy)
        self.conv_3 = nn.Conv2d(in_channels=basic_ch, out_channels=in_c, kernel_size=3, padding=1, bias=bias)

    def forward(self, x):
        x1 = self.conv_1(x)
        x1 = self.conv_2(x1)
        x1 = self.conv_3(x1)
        return x + x1


class SXNet(nn.Module):
    def __init__(self, basic_ch, rfb_ch, in_c=3, act_name='mish', asy=False, bias=False, phase='test', tlu=True):
        super(SXNet, self).__init__()
        print('basic_ch:{}, rfb_ch:{}, in_c:{}, act_name:{}, asy:{}, bias:{}, phase:{}, tlu:{}'
              .format(basic_ch, rfb_ch, in_c, act_name, asy, bias, phase, tlu))

        self.conv_in = nn.Conv2d(in_channels=in_c, out_channels=basic_ch, kernel_size=3, padding=1, bias=True)

        self.bone1 = mkm_bone(act_name=act_name, asy=asy, bias=bias, basic_ch=basic_ch)
        self.bone2 = rm_bone(act_name=act_name, asy=asy, bias=bias, basic_ch=basic_ch)

        self.phase = phase

        if basic_ch == rfb_ch:
            self.up_flag = False
        else:
            self.up_flag = True
        if self.up_flag:
            self.conv_up1 = nn.Conv2d(in_channels=basic_ch, out_channels=rfb_ch, kernel_size=3, padding=1, bias=bias)
            self.conv_up2 = nn.Conv2d(in_channels=basic_ch, out_channels=rfb_ch, kernel_size=3, padding=1, bias=bias)

        self.conv_rfb_1 = nn.Sequential(BasicRFB(rfb_ch, rfb_ch), nn.BatchNorm2d(rfb_ch), Mish())
        self.conv_rfb_2 = nn.Sequential(BasicRFB(rfb_ch, rfb_ch), nn.BatchNorm2d(rfb_ch), Mish())

        self.conv_to_inc_1_1 = nn.Conv2d(in_channels=rfb_ch, out_channels=in_c, kernel_size=3, padding=1, bias=bias)
        self.conv_to_inc_1_2 = nn.Conv2d(in_channels=in_c * 2, out_channels=in_c, kernel_size=3, padding=1, bias=bias)
        self.conv_to_inc_2_1 = nn.Conv2d(in_channels=rfb_ch, out_channels=in_c, kernel_size=3, padding=1, bias=bias)
        self.conv_to_inc_2_2 = nn.Conv2d(in_channels=in_c * 2, out_channels=in_c, kernel_size=3, padding=1, bias=bias)

        self.act_1 = nn.Tanh()
        self.act_2 = nn.Tanh()

        self.conv_out = nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=3, padding=1, bias=bias)

        self.with_tlu = tlu
        if self.with_tlu:
            self.TLU = TLU(in_c=in_c, basic_ch=basic_ch, act_name=act_name, asy=asy, bias=bias)

    def forward(self, x):
        x_in = self.conv_in(x)

        x1 = self.bone1(x_in)
        if self.up_flag:
            x1 = self.conv_up1(x1)
        x1 = self.conv_rfb_1(x1)
        att1 = self.conv_to_inc_1_1(x1)
        x1_tmp = torch.cat([x, att1], 1)
        x1_tmp = self.act_1(x1_tmp)
        x1_tmp = self.conv_to_inc_1_2(x1_tmp)
        out1 = att1 * x1_tmp
        out1 = x - out1

        x2 = self.bone2(x_in)
        if self.up_flag:
            x2 = self.conv_up2(x2)
        x2 = self.conv_rfb_2(x2)
        att2 = self.conv_to_inc_2_1(x2)
        x2_tmp = torch.cat([x, att2], 1)
        x2_tmp = self.act_2(x2_tmp)
        x2_tmp = self.conv_to_inc_2_2(x2_tmp)
        out2 = att2 * x2_tmp
        out2 = x - out2

        z = out1 + out2
        z = self.conv_out(z)
        out_nlu = x - z

        if self.with_tlu:
            out_tlu = self.TLU(out_nlu)
            if self.phase == 'train':
                return out_nlu, out_tlu
            if self.phase == 'test':
                return out_nlu
        else:
            if self.phase == 'train':
                return out_nlu, None
            if self.phase == 'test':
                return out_nlu


if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    import os
    import time

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    ###########
    # net = SXNet(8, 16, asy=False, tlu=True)  # FLOPs: 8.41 GMac Parms: 27.34 k
    # net = SXNet(8, 16, asy=False, tlu=False)  # FLOPs: 8.09 GMac Parms: 26.31 k
    # net = SXNet(8, 16, asy=True, tlu=True)  # FLOPs: 6.23 GMac Parms: 20.24 k
    # net = SXNet(8, 16, asy=True, tlu=False)  # FLOPs: 5.97 GMac Parms: 19.4 k

    # net = SXNet(2, 16, asy=False, tlu=True, bias=False)  # FLOPs: 1.9 GMac Parms: 6.16 k
    # net = SXNet(4, 16, asy=False, tlu=True, bias=False)  # FLOPs: 3.31 GMac Parms: 10.73 k
    # net = SXNet(4, 16, asy=False, tlu=True, bias=True)  # FLOPs: 3.36 GMac Parms: 10.9 k
    # net = SXNet(4, 16, asy=True, tlu=True, bias=False)  # FLOPs: 2.76 GMac Parms: 8.95 k
    ###########

    net = SXNet(8, 16, asy=True, tlu=False, bias=False)  # FLOPs: 2.76 GMac Parms: 8.95 k

    f, p = get_model_complexity_info(net, (3, 480, 640), as_strings=True, print_per_layer_stat=False)
    print('FLOPs:', f, 'Parms:', p)

    net.cuda()
    x = torch.randn(1, 3, 480, 640).cuda()
    with torch.no_grad():
        s = time.clock()
        y = net(x)
        print('fps', 1 / (time.clock() - s))
