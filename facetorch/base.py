import os
from abc import ABCMeta, abstractmethod
from typing import Optional, Tuple, Union

import torch
from codetiming import Timer
from torchvision import transforms

from facetorch import utils
from facetorch.datastruct import ImageData
from facetorch.logger import LoggerJsonFile
from facetorch.transforms import script_transform

logger = LoggerJsonFile().logger


class BaseProcessor(object, metaclass=ABCMeta):
    @Timer("BaseProcessor.__init__", "{name}: {milliseconds:.2f} ms", logger.debug)
    def __init__(
        self,
        transform: Optional[transforms.Compose],
        device: torch.device,
        optimize_transform: bool,
    ):
        """Base class for processors.

        All data pre and post processors should subclass it.
        All subclass should overwrite:

        - Methods:``run``, used for running the processing functionality.

        Args:
            device (torch.device): Torch device cpu or cuda.
            transform (transforms.Compose): Transform compose object to be applied to the image.
            optimize_transform (bool): Whether to optimize the transform.

        """
        super().__init__()
        self.device = device
        self.transform = transform if transform != "None" else None
        self.optimize_transform = optimize_transform

        if self.transform is not None:
            self.transform = utils.fix_transform_list_attr(self.transform)

        if self.optimize_transform is True:
            self.optimize()

    def optimize(self):
        """Optimizes the transform using torch.jit and deploys it to the device."""
        if self.transform is not None:
            self.transform = script_transform(self.transform)
            self.transform = self.transform.to(self.device)

    @abstractmethod
    def run(self):
        """Abstract method that should implement a tensor processing functionality"""


class BaseReader(BaseProcessor):
    @Timer("BaseReader.__init__", "{name}: {milliseconds:.2f} ms", logger.debug)
    def __init__(
        self,
        transform: transforms.Compose,
        device: torch.device,
        optimize_transform: bool,
    ):
        """Base class for image reader.

        All image readers should subclass it.
        All subclass should overwrite:

        - Methods:``run``, used for running the reading process and return a tensor.

        Args:
            transform (transforms.Compose): Transform to be applied to the image.
            device (torch.device): Torch device cpu or cuda.
            optimize_transform (bool): Whether to optimize the transforms that are resizing
            the image to a fixed size.

        """
        super().__init__(transform, device, optimize_transform)
        self.device = device
        self.optimize_transform = optimize_transform

    @abstractmethod
    def run(self, path: str) -> ImageData:
        """Abstract method that reads an image from a path and returns a data object containing
        a tensor of the image with
         shape (batch, channels, height, width).

        Args:
            path (str): Path to the image.

        Returns:
            ImageData: ImageData object with the image tensor.
        """


class BaseDownloader(object, metaclass=ABCMeta):
    @Timer("BaseDownloader.__init__", "{name}: {milliseconds:.2f} ms", logger.debug)
    def __init__(
        self,
        file_id: str,
        path_local: str,
    ):
        """Base class for downloaders.

        All downloaders should subclass it.
        All subclass should overwrite:

        - Methods:``run``, supporting to run the download functionality.

        Args:
            file_id (str): ID of the hosted file (e.g. Google Drive File ID).
            path_local (str): The file is downloaded to this local path.

        """
        super().__init__()
        self.file_id = file_id
        self.path_local = path_local

    @abstractmethod
    def run(self) -> None:
        """Abstract method that should implement the download functionality"""


class BaseModel(object, metaclass=ABCMeta):
    @Timer("BaseModel.__init__", "{name}: {milliseconds:.2f} ms", logger.debug)
    def __init__(self, downloader: BaseDownloader, device: torch.device):
        """Base class for torch models.

        All detectors and predictors should subclass it.
        All subclass should overwrite:

        - Methods:``run``, supporting to make detections and predictions with the model.

        Args:
            downloader (BaseDownloader): Downloader for the model.
            device (torch.device): Torch device cpu or cuda.

        Attributes:
            model (torch.jit.ScriptModule or torch.jit.TracedModule): Loaded TorchScript model.

        """
        super().__init__()
        self.downloader = downloader
        self.path_local = self.downloader.path_local
        self.device = device

        self.model = self.load_model()

    @Timer("BaseModel.load_model", "{name}: {milliseconds:.2f} ms", logger.debug)
    def load_model(self) -> Union[torch.jit.ScriptModule, torch.jit.TracedModule]:
        """Loads the TorchScript model.

        Returns:
            Union[torch.jit.ScriptModule, torch.jit.TracedModule]: Loaded TorchScript model.
        """
        if not os.path.exists(self.path_local):
            dir_local = os.path.dirname(self.path_local)
            os.makedirs(dir_local, exist_ok=True)
            self.downloader.run()
        model = torch.jit.load(self.path_local, map_location=self.device)
        model.eval()

        return model

    @Timer("BaseModel.inference", "{name}: {milliseconds:.2f} ms", logger.debug)
    def inference(
        self, tensor: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """Inference the model with the given tensor.

        Args:
            tensor (torch.Tensor): Input tensor for the model.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor]]: Output tensor or tuple of tensors.
        """
        with torch.no_grad():

            if tensor.device != self.device:
                tensor = tensor.to(self.device)

            logits = self.model(tensor)

        return logits

    @abstractmethod
    def run(self):
        """Abstract method for making the predictions. Example pipeline:

        - self.preprocessor.run
        - self.inference
        - self.postprocessor.run

        """


class BaseUtilizer(BaseProcessor):
    def __init__(
        self,
        transform: transforms.Compose,
        device: torch.device,
        optimize_transform: bool,
    ):
        """BaseUtilizer is a processor that takes ImageData as input to do any kind of work that requires model predictions for example, drawing, summarizing, etc.

        Args:
            transform (Compose): Composed Torch transform object.
            device (torch.device): Torch device cpu or cuda object.
            optimize_transform (bool): Whether to optimize the transform.
        """
        super().__init__(transform, device, optimize_transform)

    @abstractmethod
    def run(self, data: ImageData) -> ImageData:
        """Runs utility function on the ImageData object.

        Args:
            data (ImageData): ImageData object containing most of the data including the predictions.

        Returns:
            ImageData: ImageData object containing the same data as input or modified object.
        """

        return data
