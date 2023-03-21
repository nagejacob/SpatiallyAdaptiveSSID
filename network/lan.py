import torch
import torch.nn as nn

class CALayer(nn.Module):
    def __init__(self, channel=64, reduction=16):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class RB(nn.Module):
    def __init__(self, filters):
        super(RB, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, 1)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(filters, filters, 1)
        self.cuca = CALayer(channel=filters)

    def forward(self, x):
        c0 = x
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        out = self.cuca(x)
        return out + c0

class NRB(nn.Module):
    def __init__(self, n, filters):
        super(NRB, self).__init__()
        nets = []
        for i in range(n):
            nets.append(RB(filters))
        self.body = nn.Sequential(*nets)
        self.tail = nn.Conv2d(filters, filters, 1)

    def forward(self, x):
        return x + self.tail(self.body(x))


class LAN(nn.Module):
    def __init__(self, blindspot, in_ch=3, out_ch=None, rbs=6):
        super(LAN, self).__init__()
        self.receptive_feild = blindspot
        assert self.receptive_feild % 2 == 1
        self.in_ch = in_ch
        self.out_ch = self.in_ch if out_ch is None else out_ch
        self.mid_ch = 64
        self.rbs = rbs

        layers = []
        layers.append(nn.Conv2d(self.in_ch, self.mid_ch, 1))
        layers.append(nn.ReLU())

        for i in range(self.receptive_feild // 2):
            layers.append(nn.Conv2d(self.mid_ch, self.mid_ch, 3, 1, 1))
            layers.append(nn.ReLU())

        layers.append(NRB(self.rbs, self.mid_ch))
        layers.append(nn.Conv2d(self.mid_ch, self.out_ch, 1))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)
