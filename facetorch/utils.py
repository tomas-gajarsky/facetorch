import omegaconf
import torch
import torchvision


def rgb2bgr(tensor: torch.Tensor) -> torch.Tensor:
    """Converts a batch of RGB tensors to BGR tensors or vice versa.

    Args:
        tensor (torch.Tensor): Batch of RGB (or BGR) channeled tensors
        with shape (dim0, channels, dim2, dim3)

    Returns:
        torch.Tensor: Batch of BGR (or RGB) tensors with shape (dim0, channels, dim2, dim3).
    """
    assert tensor.shape[1] == 3, "Tensor must have 3 channels."
    return tensor[:, [2, 1, 0]]


def fix_transform_list_attr(
    transform: torchvision.transforms.Compose,
) -> torchvision.transforms.Compose:
    """Fix the transform attributes by converting the listconfig to a list.
    This enables to optimize the transform using TorchScript.

    Args:
        transform (torchvision.transforms.Compose): Transform to be fixed.

    Returns:
        torchvision.transforms.Compose: Fixed transform.
    """
    for transform_x in transform.transforms:
        for key, value in transform_x.__dict__.items():
            if isinstance(value, omegaconf.listconfig.ListConfig):
                transform_x.__dict__[key] = list(value)
    return transform
