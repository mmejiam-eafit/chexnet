import torch.nn as nn
from torch.nn import functional as F
import torch

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()
        outputs = []
        calc_output = out_channels
        n = 1 + calc_output // 5 if calc_output % 5 != 0 else calc_output // 5
        for i in range(5):
            outputs.append(min(calc_output, n))
            calc_output = calc_output - n

        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, outputs[0], 1, bias=False),
            nn.BatchNorm2d(outputs[0]),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, outputs[1], rate1))
        modules.append(ASPPConv(in_channels, outputs[2], rate2))
        modules.append(ASPPConv(in_channels, outputs[3], rate3))
        modules.append(ASPPPooling(in_channels, outputs[4]))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(sum(outputs), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1), )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)
