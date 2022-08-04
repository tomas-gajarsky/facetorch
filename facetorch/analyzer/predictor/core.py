from typing import List

import torch
from codetiming import Timer
from facetorch.base import BaseDownloader, BaseModel
from facetorch.datastruct import Prediction
from facetorch.logger import LoggerJsonFile

from .post import BasePredPostProcessor
from .pre import BasePredPreProcessor

logger = LoggerJsonFile().logger


class FacePredictor(BaseModel):
    @Timer("FacePredictor.__init__", "{name}: {milliseconds:.2f} ms", logger.debug)
    def __init__(
        self,
        downloader: BaseDownloader,
        device: torch.device,
        preprocessor: BasePredPreProcessor,
        postprocessor: BasePredPostProcessor,
        **kwargs
    ):
        """FacePredictor is a wrapper around a neural network model that is trained to predict facial features.

        Args:
            downloader (BaseDownloader): Downloader that downloads the model.
            device (torch.device): Torch device cpu or cuda for the model.
            preprocessor (BasePredPostProcessor): Preprocessor that runs before the model.
            postprocessor (BasePredPostProcessor): Postprocessor that runs after the model.
        """
        self.__dict__.update(kwargs)
        super().__init__(downloader, device)

        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

    @Timer("FacePredictor.run", "{name}: {milliseconds:.2f} ms", logger.debug)
    def run(self, faces: torch.Tensor) -> List[Prediction]:
        """Predicts facial features.

        Args:
            faces (torch.Tensor): Torch tensor containing a batch of faces with values between 0-1 and shape (batch_size, channels, height, width).

        Returns:
            (List[Prediction]): List of Prediction data objects. One for each face in the batch.
        """
        faces = self.preprocessor.run(faces)
        preds = self.inference(faces)
        preds_list = self.postprocessor.run(preds)

        return preds_list
