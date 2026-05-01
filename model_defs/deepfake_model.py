import timm
import torch
import torch.nn as nn


class DeepfakeEfficientNetB7(nn.Module):
    """EfficientNet-B7 for deepfake detection (binary classification).

    Source: selimsef/dfdc_deepfake_challenge
    Architecture: TF-style EfficientNet-B7 feature extractor + avg pool + linear head.
    """

    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model(
            "tf_efficientnet_b7",
            num_classes=1000,
            pretrained=False,
            global_pool="avg",
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.0)
        self.fc = nn.Linear(2560, 1)

    def forward(self, x):
        x = self.encoder.conv_stem(x)
        x = self.encoder.bn1(x)
        x = self.encoder.blocks(x)
        x = self.encoder.conv_head(x)
        x = self.encoder.bn2(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
