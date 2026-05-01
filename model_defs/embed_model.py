import torch
import torch.nn as nn
import torchvision.models as models


class _BaseNet(nn.Module):
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
        self.avgpool = resnet.avgpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


class _ProjectionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 128),
        )

    def forward(self, x):
        return self.net(x)


class EmbedResNet50(nn.Module):
    """ResNet-50 face embedding with projection head.

    Source: 1adrianb/unsupervised-face-representation
    Output: (normalized_embedding[128], prototype_logits[3000])
    """

    def __init__(self):
        super().__init__()
        self.base_net = _BaseNet()
        self.projection_net = _ProjectionNet()
        self.prototypes = nn.Linear(128, 3000, bias=False)

    def forward(self, inputs):
        x = torch.cat([inputs])
        x = self.base_net(x)
        x = self.projection_net(x)
        x = x / torch.clamp_min(x.norm(2, dim=1, keepdim=True), 1e-12)
        return (x, self.prototypes(x))
