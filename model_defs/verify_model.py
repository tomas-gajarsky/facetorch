import torch
import torch.nn as nn


class _MagFaceIBasicBlock(nn.Module):
    """Residual block used by the converted MagFace IResNet-100 weights."""

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.prelu = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return out + identity


class MagFaceIResNet100(nn.Module):
    """IResNet-100 topology matching the hosted MagFace state dictionary."""

    def __init__(self):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU(64)
        self.layer1 = self._make_layer(64, 3, stride=2)
        self.layer2 = self._make_layer(128, 13, stride=2)
        self.layer3 = self._make_layer(256, 30, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.bn2 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512 * 7 * 7, 512)
        self.features = nn.BatchNorm1d(512)

    def _make_layer(self, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes),
            )

        layers = [
            _MagFaceIBasicBlock(self.inplanes, planes, stride, downsample)
        ]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(_MagFaceIBasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return self.features(x)


class _Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=1, padding=1, bias=True
        )
        self.prelu = nn.PReLU(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            stride=stride,
            padding=1,
            bias=True,
        )

        self.shortcut = None
        if stride > 1:
            if in_channels != out_channels:
                self.shortcut = nn.Conv2d(
                    in_channels, out_channels, 1, stride=stride, bias=True
                )
            else:
                self.shortcut = nn.MaxPool2d(1, stride=stride)

    def forward(self, x):
        identity = self.shortcut(x) if self.shortcut is not None else x
        out = self.bn(x)
        out = self.conv1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        return out + identity


class VerifyIResNet100(nn.Module):
    """IResNet-100 face verification model.

    Output: (normalized_embedding[512], norm)
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=True)
        self.prelu1 = nn.PReLU(64)
        self.layer1 = self._make_layer(64, 64, 3, stride=2)
        self.layer2 = self._make_layer(64, 128, 13, stride=2)
        self.layer3 = self._make_layer(128, 256, 30, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        self.bn_final = nn.BatchNorm2d(512)
        self.fc = nn.Linear(25088, 512, bias=True)
        self.bn_output = nn.BatchNorm1d(512, affine=False)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [_Block(in_channels, out_channels, stride=stride)]
        for _ in range(1, num_blocks):
            layers.append(_Block(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn_final(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn_output(x)
        norm = torch.linalg.norm(x, ord=2, dim=1, keepdim=True)
        return x / norm, norm

    def load_from_torchscript(self, ts_model):
        """Load parameters from a traced TorchScript model with CONSTANTS."""
        raw = []
        for node in ts_model.graph.nodes():
            if node.kind() == "prim::Constant":
                for out in node.outputs():
                    try:
                        t = out.toIValue()
                        if isinstance(t, torch.Tensor):
                            raw.append((int(out.debugName()), t))
                    except (RuntimeError, AttributeError, ValueError):
                        pass
        raw.sort(key=lambda x: x[0])
        constants = [t for _, t in raw]

        idx = 0

        def consume(param):
            nonlocal idx
            param.data.copy_(constants[idx])
            idx += 1

        consume(self.conv1.weight)
        consume(self.conv1.bias)
        consume(self.prelu1.weight)

        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                if isinstance(block.shortcut, nn.Conv2d):
                    consume(block.shortcut.weight)
                    consume(block.shortcut.bias)
                consume(block.bn.weight)
                consume(block.bn.bias)
                consume(block.bn.running_mean)
                consume(block.bn.running_var)
                consume(block.conv1.weight)
                consume(block.conv1.bias)
                consume(block.prelu.weight)
                consume(block.conv2.weight)
                consume(block.conv2.bias)

        consume(self.bn_final.weight)
        consume(self.bn_final.bias)
        consume(self.bn_final.running_mean)
        consume(self.bn_final.running_var)

        fc_weight_t = constants[idx]
        idx += 1
        self.fc.weight.data.copy_(fc_weight_t.T)
        consume(self.fc.bias)
        consume(self.bn_output.running_mean)
        consume(self.bn_output.running_var)

        if idx != len(constants):
            raise RuntimeError(
                f"AdaFace constant reconstruction consumed {idx} of "
                f"{len(constants)} tensors"
            )
