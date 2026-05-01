import timm
import torch.nn as nn


class EfficientNetB2FER(nn.Module):
    """EfficientNet-B2 for Facial Expression Recognition (8 classes).

    Source: HSE-asavchenko/face-emotion-recognition
    Uses TF-style EfficientNet-B2 with max pooling and 260x260 input.
    """

    def __init__(self, num_classes: int = 8):
        super().__init__()
        self.model = timm.create_model(
            "tf_efficientnet_b2",
            num_classes=num_classes,
            pretrained=False,
            global_pool="max",
        )

    def forward(self, x):
        return self.model(x)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if any(k.startswith("model.") for k in state_dict):
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        return self.model.load_state_dict(state_dict, strict=strict, assign=assign)


class EfficientNetB0FER(nn.Module):
    """EfficientNet-B0 for Facial Expression Recognition (7 classes).

    Source: HSE-asavchenko/face-emotion-recognition
    Uses TF-style EfficientNet-B0 with avg pooling and 244x244 input.
    """

    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.model = timm.create_model(
            "tf_efficientnet_b0",
            num_classes=num_classes,
            pretrained=False,
            global_pool="avg",
        )

    def forward(self, x):
        return self.model(x)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if any(k.startswith("model.") for k in state_dict):
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        mapped = {}
        for k, v in state_dict.items():
            mapped[k.replace("classifier.0.", "classifier.")] = v
        return self.model.load_state_dict(mapped, strict=strict, assign=assign)
