import os
import torch
import torchvision
from codetiming import Timer
from facetorch.base import BaseUtilizer
from facetorch.datastruct import ImageData
from facetorch.logger import LoggerJsonFile
from torchvision import transforms

logger = LoggerJsonFile().logger


class ImageSaver(BaseUtilizer):
    def __init__(
        self,
        transform: transforms.Compose,
        device: torch.device,
        optimize_transform: bool,
    ):
        """Initializes the ImageSaver class. This class is used to save the image tensor to an image file.

        Args:
            transform (Compose): Composed Torch transform object.
            device (torch.device): Torch device cpu or cuda object.
            optimize_transform (bool): Whether to optimize the transform.

        """
        super().__init__(transform, device, optimize_transform)

    @Timer("ImageSaver.run", "{name}: {milliseconds:.2f} ms", logger.debug)
    def run(self, data: ImageData) -> ImageData:
        """Saves the image tensor to an image file, if the path_output attribute of ImageData is not None.

        Args:
            data (ImageData): ImageData object containing the img tensor.

        Returns:
            ImageData: ImageData object containing the same data as the input.
        """
        if data.path_output is not None:
            os.makedirs(os.path.dirname(data.path_output), exist_ok=True)
            pil_image = torchvision.transforms.functional.to_pil_image(data.img)
            pil_image.save(data.path_output)

        return data
