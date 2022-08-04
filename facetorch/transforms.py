from typing import Union

import torch
import torchvision
from torchvision import transforms


def script_transform(
    transform: transforms.Compose,
) -> Union[torch.jit.ScriptModule, torch.jit.ScriptFunction]:
    """Convert the composed transform to a TorchScript module.

    Args:
        transform (transforms.Compose): Transform compose object to be scripted.

    Returns:
        Union[torch.jit.ScriptModule, torch.jit.ScriptFunction]: Scripted transform.
    """

    transform_seq = torch.nn.Sequential(*transform.transforms)
    transform_jit = torch.jit.script(transform_seq)
    return transform_jit


class SquarePad(torch.nn.Module):
    """SquarePad is a transform that pads the image to a square shape."""

    def __init__(self) -> None:
        """It is initialized as a torch.nn.Module."""
        super().__init__()

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Pads a tensor to a square.

        Args:
            tensor (torch.Tensor): tensor to pad.

        Returns:
            torch.Tensor: Padded tensor.
        """
        height, width = tensor.shape[-2:]
        img_size = [width, height]

        max_wh = max(img_size)
        p_left, p_top = [(max_wh - s) // 2 for s in img_size]
        p_right, p_bottom = [
            max_wh - (s + pad) for s, pad in zip(img_size, [p_left, p_top])
        ]
        padding = (p_left, p_top, p_right, p_bottom)
        tensor_padded = torchvision.transforms.functional.pad(
            tensor, padding, 0, "constant"
        )
        return tensor_padded

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Pads a tensor to a square.

        Args:
            tensor (torch.Tensor): tensor to pad.

        Returns:
            torch.Tensor: Padded tensor.

        """
        return self.__call__(tensor)
