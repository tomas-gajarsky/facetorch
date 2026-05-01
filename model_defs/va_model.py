import torch
import torch.nn as nn


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

    def forward(self, avgpool, x):
        x = self.features(x)
        x = avgpool(x)
        return torch.flatten(x, 1)


class _Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.drop0 = nn.Dropout(p=0.5)
        self.lin0 = nn.Linear(9216, 256)
        self.relu0 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(p=0.5)
        self.lin1 = nn.Linear(256, 64)
        self.bn = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.drop0(x)
        x = self.lin0(x)
        x = self.relu0(x)
        x = self.drop0(x)
        x = self.lin1(x)
        x = self.bn(x)
        x = self.relu1(x)
        return x


class _TaskHeader(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(64, 1280)
        self.bn1 = nn.BatchNorm1d(1280)
        self.linear2 = nn.Linear(1280, 2)
        self.layer_blocks = nn.Sequential(
            nn.Linear(64, 1280),
            nn.ReLU(inplace=True),
            nn.Linear(1280, 2),
        )

    def forward(self, x):
        x = self.layer_blocks(x)
        return x


class ELIMALAlexNet(nn.Module):
    """ELIM Affective Learning AlexNet for Valence-Arousal prediction.

    Source: kdhht2334/ELIM_FER
    """

    def __init__(self):
        super().__init__()
        self.encoder = _Encoder()
        self.regressor = _Regressor()
        self.task_header = _TaskHeader()

    def forward(self, x):
        x = self.encoder(self.regressor.avgpool, x)
        x = self.regressor(x)
        x = self.task_header(x)
        return x
