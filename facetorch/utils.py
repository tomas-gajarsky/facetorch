import numpy as np
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


def numpy_to_chw_tensor(array: np.ndarray) -> torch.Tensor:
    """Convert a numpy array to a channel-first (C, H, W) torch tensor.

    Args:
        array (np.ndarray): Image array with shape (H, W), (H, W, C), or (C, H, W).

    Returns:
        torch.Tensor: Tensor with shape (C, H, W).
    """
    tensor = torch.from_numpy(array.copy())
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3:
        if tensor.shape[2] in (1, 3, 4):
            tensor = tensor.permute(2, 0, 1).contiguous()
        elif tensor.shape[0] not in (1, 3, 4):
            raise ValueError(
                f"Ambiguous numpy array shape: {array.shape}. "
                "Expected (H, W), (H, W, C), or (C, H, W) where C is 1, 3, or 4."
            )
    else:
        raise ValueError(
            f"Unsupported numpy array with {array.ndim} dimensions. Expected 2 or 3."
        )
    return tensor


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
