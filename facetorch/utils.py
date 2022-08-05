import os

import omegaconf
import torch
import torchvision

from facetorch.datastruct import ImageData


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


def draw_boxes_and_save(data: ImageData, path_output: str) -> None:
    """Draws boxes on an image and saves it to a file.

    Args:
        data (ImageData): ImageData object containing the image tensor, detections, and faces.
        path_output (str): Path to the output file.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(path_output), exist_ok=True)
    loc_tensor = data.aggregate_loc_tensor()
    labels = [str(face.indx) for face in data.faces]
    data.img = torchvision.utils.draw_bounding_boxes(
        image=data.img,
        boxes=loc_tensor,
        labels=labels,
        colors="green",
        width=3,
    )
    pil_image = torchvision.transforms.functional.to_pil_image(data.img)
    pil_image.save(path_output)


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
