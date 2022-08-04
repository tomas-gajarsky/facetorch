import torch
from codetiming import Timer
from facetorch.base import BaseDownloader, BaseModel
from facetorch.datastruct import ImageData
from facetorch.logger import LoggerJsonFile

from .post import BaseDetPostProcessor
from .pre import BaseDetPreProcessor

logger = LoggerJsonFile().logger


class FaceDetector(BaseModel):
    @Timer("FaceDetector.__init__", "{name}: {milliseconds:.2f} ms", logger.debug)
    def __init__(
        self,
        downloader: BaseDownloader,
        device: torch.device,
        preprocessor: BaseDetPreProcessor,
        postprocessor: BaseDetPostProcessor,
        **kwargs
    ):
        """FaceDetector is a wrapper around a neural network model that is trained to detect faces.

        Args:
            downloader (BaseDownloader): Downloader that downloads the model.
            device (torch.device): Torch device cpu or cuda for the model.
            preprocessor (BaseDetPreProcessor): Preprocessor that runs before the model.
            postprocessor (BaseDetPostProcessor): Postprocessor that runs after the model.
        """
        self.__dict__.update(kwargs)
        super().__init__(downloader, device)

        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

    @Timer("FaceDetector.run", "{name}: {milliseconds:.2f} ms", logger.debug)
    def run(self, data: ImageData) -> ImageData:
        """Detect all faces in the image.

        Args:
            ImageData: ImageData object containing the image tensor with values between 0 - 255 and shape (batch_size, channels, height, width).

        Returns:
            ImageData: Image data object with Detection tensors and detected Face objects.
        """
        data = self.preprocessor.run(data)
        logits = self.inference(data.tensor)
        data = self.postprocessor.run(data, logits)

        return data
