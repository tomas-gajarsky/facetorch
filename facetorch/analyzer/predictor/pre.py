from abc import abstractmethod

import torch
from codetiming import Timer
from facetorch.base import BaseProcessor
from facetorch.logger import LoggerJsonFile
from facetorch.utils import rgb2bgr
from torchvision import transforms

logger = LoggerJsonFile().logger


class BasePredPreProcessor(BaseProcessor):
    @Timer(
        "BasePredPreProcessor.__init__", "{name}: {milliseconds:.2f} ms", logger.debug
    )
    def __init__(
        self,
        transform: transforms.Compose,
        device: torch.device,
        optimize_transform: bool,
    ):
        """Base class for predictor pre processors.

        All predictor pre processors should subclass it.
        All subclass should overwrite:

        - Methods:``run``, used for running the processing

        Args:
            device (torch.device): Torch device cpu or cuda.
            transform (transforms.Compose): Transform compose object to be applied to the  image.
            optimize_transform (bool): Whether to optimize the transform.

        """
        super().__init__(transform, device, optimize_transform)

    @abstractmethod
    def run(self, faces: torch.Tensor) -> torch.Tensor:
        """Abstract method that runs the predictor pre processing functionality and returns a batch of preprocessed face tensors.

        Args:
            faces (torch.Tensor): Batch of face tensors with shape (batch, channels, height, width).

        Returns:
            torch.Tensor: Batch of preprocessed face tensors with shape (batch, channels, height, width).

        """


class PredictorPreProcessor(BasePredPreProcessor):
    def __init__(
        self,
        transform: transforms.Compose,
        device: torch.device,
        optimize_transform: bool,
        reverse_colors: bool = False,
    ):
        """Torch transform based pre-processor that is applied to face tensors before they are passed to the predictor model.

        Args:
            transform (transforms.Compose): Composed Torch transform object.
            device (torch.device): Torch device cpu or cuda.
            optimize_transform (bool): Whether to optimize the transform.
            reverse_colors (bool): Whether to reverse the colors of the image tensor
        """
        super().__init__(transform, device, optimize_transform)
        self.reverse_colors = reverse_colors

    @Timer("PredictorPreProcessor.run", "{name}: {milliseconds:.2f} ms", logger.debug)
    def run(self, faces: torch.Tensor) -> torch.Tensor:
        """Runs the trasform on a batch of face tensors.

        Args:
            faces (torch.Tensor): Batch of face tensors.

        Returns:
            torch.Tensor: Batch of preprocessed face tensors.
        """
        if faces.device != self.device:
            faces = faces.to(self.device)

        faces = self.transform(faces)
        if self.reverse_colors:
            faces = rgb2bgr(faces)
        return faces
