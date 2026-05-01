import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SynergyNetMobileNetV2(nn.Module):
    """MobileNetV2 backbone for 3D face alignment (SynergyNet).

    Source: choyingw/SynergyNet
    Output: 62-dim vector [orientation(12), shape(40), expression(10)]
    """

    def __init__(self):
        super().__init__()
        mob = models.mobilenet_v2(weights=None)
        self.features = mob.features
        self.classifier_ori = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280, 12),
        )
        self.classifier_shape = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280, 40),
        )
        self.classifier_exp = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, [1, 1])
        x = x.reshape(x.size(0), -1)
        dropped_ori = self.classifier_ori[0](x)
        ori_out = self.classifier_ori[1](dropped_ori)
        dropped_shape = self.classifier_shape[0](dropped_ori)
        shape_out = self.classifier_shape[1](dropped_shape)
        exp_out = self.classifier_exp[1](self.classifier_exp[0](dropped_shape))
        return torch.cat([ori_out, shape_out, exp_out], dim=1)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        mapped = {}
        prefix = "I2P.backbone."
        skip = ("forwardDirection.", "reverseDirection.")
        for k, v in state_dict.items():
            if k.startswith(prefix):
                mapped[k[len(prefix):]] = v
            elif not any(k.startswith(s) for s in skip) and "." not in k:
                pass
        return super().load_state_dict(mapped, strict=strict, assign=assign)
