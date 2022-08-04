from abc import abstractmethod

import torch
from codetiming import Timer
from facetorch.base import BaseProcessor
from facetorch.datastruct import ImageData
from facetorch.logger import LoggerJsonFile
from facetorch.utils import rgb2bgr
from torchvision import transforms

logger = LoggerJsonFile().logger


class BaseDetPreProcessor(BaseProcessor):
    @Timer(
        "BaseDetPreProcessor.__init__", "{name}: {milliseconds:.2f} ms", logger.debug
    )
    def __init__(
        self,
        transform: transforms.Compose,
        device: torch.device,
        optimize_transform: bool,
    ):
        """Base class for detector pre processors.

        All detector pre processors should subclass it.
        All subclass should overwrite:

        - Methods:``run``, used for running the processing

        Args:
            device (torch.device): Torch device cpu or cuda.
            transform (transforms.Compose): Transform compose object to be applied to the image.
            optimize_transform (bool): Whether to optimize the transform.

        """
        super().__init__(transform, device, optimize_transform)

    @abstractmethod
    def run(self, data: ImageData) -> ImageData:
        """Abstract method that runs the detector pre processing functionality.
        Returns a batch of preprocessed face tensors.

        Args:
            data (ImageData): ImageData object containing the image tensor.

        Returns:
            ImageData: ImageData object containing the image tensor preprocessed for the detector.

        """


class DetectorPreProcessor(BaseDetPreProcessor):
    @Timer(
        "DetectorPreProcessor.__init__", "{name}: {milliseconds:.2f} ms", logger.debug
    )
    def __init__(
        self,
        transform: transforms.Compose,
        device: torch.device,
        optimize_transform: bool,
        reverse_colors: bool,
    ):
        """Initialize the detector preprocessor.

        Args:
            transform (Compose): Composed Torch transform object.
            device (torch.device): Torch device cpu or cuda.
            optimize_transform (bool): Whether to optimize the transform.
            reverse_colors (bool): Whether to reverse the colors of the image tensor from RGB to BGR or vice versa. If False, the colors remain unchanged.

        """
        super().__init__(transform, device, optimize_transform)
        self.reverse_colors = reverse_colors

    @Timer("DetectorPreProcessor.run", "{name}: {milliseconds:.2f} ms", logger.debug)
    def run(self, data: ImageData) -> ImageData:
        """Run the detector preprocessor on the image tensor in BGR format and return the transformed image tensor.

        Args:
            data (ImageData): ImageData object containing the image tensor.

        Returns:
            ImageData: ImageData object containing the preprocessed image tensor.
        """
        if data.tensor.device != self.device:
            data.tensor = data.tensor.to(self.device)

        data.tensor = self.transform(data.tensor)

        if self.reverse_colors:
            data.tensor = rgb2bgr(data.tensor)

        return data
