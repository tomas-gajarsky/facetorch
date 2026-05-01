import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def _conv_bn(in_c, out_c, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU(negative_slope=0, inplace=True),
    )


def _conv_bn_no_relu(in_c, out_c, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_c),
    )


class _Body(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=None)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c3, c4, c5


class _FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.output1 = _conv_bn(in_channels_list[0], out_channels, 1)
        self.output2 = _conv_bn(in_channels_list[1], out_channels, 1)
        self.output3 = _conv_bn(in_channels_list[2], out_channels, 1)
        self.merge1 = _conv_bn(out_channels, out_channels, 3, padding=1)
        self.merge2 = _conv_bn(out_channels, out_channels, 3, padding=1)

    def forward(self, c3, c4, c5):
        p5 = self.output3(c5)
        p4 = self.output2(c4)
        p3 = self.output1(c3)

        up5 = F.interpolate(p5, size=p4.shape[2:], mode="nearest")
        p4 = self.merge2(p4 + up5)

        up4 = F.interpolate(p4, size=p3.shape[2:], mode="nearest")
        p3 = self.merge1(p3 + up4)

        return p3, p4, p5


class _SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv3X3 = _conv_bn_no_relu(in_channel, out_channel // 2, 3, padding=1)
        self.conv5X5_1 = _conv_bn(in_channel, out_channel // 4, 3, padding=1)
        self.conv5X5_2 = _conv_bn_no_relu(out_channel // 4, out_channel // 4, 3, padding=1)
        self.conv7X7_2 = _conv_bn(out_channel // 4, out_channel // 4, 3, padding=1)
        self.conv7x7_3 = _conv_bn_no_relu(out_channel // 4, out_channel // 4, 3, padding=1)

    def forward(self, x):
        c3 = self.conv3X3(x)
        c5_1 = self.conv5X5_1(x)
        c5 = self.conv5X5_2(c5_1)
        c7 = self.conv7x7_3(self.conv7X7_2(c5_1))
        return F.relu(torch.cat([c3, c5, c7], dim=1))


class _ClassHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        out = self.conv1x1(x).permute(0, 2, 3, 1).contiguous()
        return out.view(out.size(0), -1, 2)


class _BboxHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        out = self.conv1x1(x).permute(0, 2, 3, 1).contiguous()
        return out.view(out.size(0), -1, 4)


class _LandmarkHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        out = self.conv1x1(x).permute(0, 2, 3, 1).contiguous()
        return out.view(out.size(0), -1, 10)


class RetinaFaceResNet50(nn.Module):
    """RetinaFace detector with ResNet-50 backbone.

    Source: biubug6/Pytorch_Retinaface
    Returns: (bbox_regressions, classifications, landmark_regressions)
    """

    def __init__(self):
        super().__init__()
        self.body = _Body()
        self.fpn = _FPN([512, 1024, 2048], 256)
        self.ssh1 = _SSH(256, 256)
        self.ssh2 = _SSH(256, 256)
        self.ssh3 = _SSH(256, 256)
        self.ClassHead = nn.ModuleList([
            _ClassHead(256, 4) for _ in range(3)
        ])
        self.BboxHead = nn.ModuleList([
            _BboxHead(256, 8) for _ in range(3)
        ])
        self.LandmarkHead = nn.ModuleList([
            _LandmarkHead(256, 20) for _ in range(3)
        ])

    def forward(self, x):
        c3, c4, c5 = self.body(x)
        p3, p4, p5 = self.fpn(c3, c4, c5)
        f1 = self.ssh1(p3)
        f2 = self.ssh2(p4)
        f3 = self.ssh3(p5)

        features = [f1, f2, f3]
        bbox = torch.cat([self.BboxHead[i](features[i]) for i in range(3)], dim=1)
        cls = torch.cat([self.ClassHead[i](features[i]) for i in range(3)], dim=1)
        ldm = torch.cat([self.LandmarkHead[i](features[i]) for i in range(3)], dim=1)
        return bbox, F.softmax(cls, dim=-1), ldm
