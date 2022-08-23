from abc import abstractmethod
from typing import List, Tuple, Union

import torch
from codetiming import Timer
from facetorch.base import BaseProcessor
from facetorch.datastruct import Prediction
from facetorch.logger import LoggerJsonFile
from torchvision import transforms

logger = LoggerJsonFile().logger


class BasePredPostProcessor(BaseProcessor):
    @Timer(
        "BasePredPostProcessor.__init__", "{name}: {milliseconds:.2f} ms", logger.debug
    )
    def __init__(
        self,
        transform: transforms.Compose,
        device: torch.device,
        optimize_transform: bool,
        labels: List[str],
    ):
        """Base class for predictor post processors.

        All predictor post processors should subclass it.
        All subclass should overwrite:

        - Methods:``run``, used for running the processing

        Args:
            device (torch.device): Torch device cpu or cuda.
            transform (transforms.Compose): Transform compose object to be applied to the image.
            optimize_transform (bool): Whether to optimize the transform.
            labels (List[str]): List of labels.

        """
        super().__init__(transform, device, optimize_transform)
        self.labels = labels

    def create_pred_list(
        self, preds: torch.Tensor, indices: List[int]
    ) -> List[Prediction]:
        """Create a list of predictions.

        Args:
            preds (torch.Tensor): Tensor of predictions, shape (batch, _).
            indices (List[int]): List of label indices, one for each sample.

        Returns:
            List[Prediction]: List of predictions.

        """
        assert (
            len(indices) == preds.shape[0]
        ), "Predictions and indices must have the same length."

        pred_labels = [self.labels[indx] for indx in indices]

        pred_list = []
        for i, label in enumerate(pred_labels):
            pred = Prediction(label, preds[i])
            pred_list.append(pred)
        return pred_list

    @abstractmethod
    def run(self, preds: Union[torch.Tensor, Tuple[torch.Tensor]]) -> List[Prediction]:
        """Abstract method that runs the predictor post processing functionality and returns a list of prediction data structures, one for each face in the batch.

        Args:
            preds (Union[torch.Tensor, Tuple[torch.Tensor]]): Output of the predictor model.

        Returns:
            List[Prediction]: List of predictions.

        """


class PostArgMax(BasePredPostProcessor):
    @Timer("PostArgMax.__init__", "{name}: {milliseconds:.2f} ms", logger.debug)
    def __init__(
        self,
        transform: transforms.Compose,
        device: torch.device,
        optimize_transform: bool,
        labels: List[str],
        dim: int,
    ):
        """Initialize the predictor postprocessor that runs argmax on the prediction tensor and returns a list of prediction data structures.

        Args:
            transform (Compose): Composed Torch transform object.
            device (torch.device): Torch device cpu or cuda.
            optimize_transform (bool): Whether to optimize the transform using TorchScript.
            labels (List[str]): List of labels.
            dim (int): Dimension of the prediction.
        """
        super().__init__(transform, device, optimize_transform, labels)
        self.dim = dim

    @Timer("PostArgMax.run", "{name}: {milliseconds:.2f} ms", logger.debug)
    def run(self, preds: torch.Tensor) -> List[Prediction]:
        """Post-processes the prediction tensor using argmax and returns a list of prediction data structures, one for each face.

        Args:
            preds (torch.Tensor): Batch prediction tensor.

        Returns:
            List[Prediction]: List of prediction data structures containing the predicted labels and confidence scores for each face in the batch.
        """
        indices = torch.argmax(preds, dim=self.dim).cpu().numpy().tolist()
        pred_list = self.create_pred_list(preds, indices)

        return pred_list


class PostSigmoidBinary(BasePredPostProcessor):
    @Timer("PostSigmoidBinary.__init__", "{name}: {milliseconds:.2f} ms", logger.debug)
    def __init__(
        self,
        transform: transforms.Compose,
        device: torch.device,
        optimize_transform: bool,
        labels: List[str],
        threshold: float = 0.5,
    ):
        """Initialize the predictor postprocessor that runs sigmoid on the prediction tensor and returns a list of prediction data structures.

        Args:
            transform (Compose): Composed Torch transform object.
            device (torch.device): Torch device cpu or cuda.
            optimize_transform (bool): Whether to optimize the transform using TorchScript.
            labels (List[str]): List of labels.
            threshold (float): Probability threshold for positive class.
        """
        super().__init__(transform, device, optimize_transform, labels)
        self.threshold = threshold

    @Timer("PostSigmoidBinary.run", "{name}: {milliseconds:.2f} ms", logger.debug)
    def run(self, preds: torch.Tensor) -> List[Prediction]:
        """Post-processes the prediction tensor using argmax and returns a list of prediction data structures, one for each face.

        Args:
            preds (torch.Tensor): Batch prediction tensor.

        Returns:
            List[Prediction]: List of prediction data structures containing the predicted labelsand confidence scores for each face in the batch.
        """
        preds = torch.sigmoid(preds.squeeze(1))
        preds_thresh = preds.where(preds >= self.threshold, torch.zeros_like(preds))
        indices = torch.round(preds_thresh)
        indices = indices.cpu().numpy().astype(int).tolist()
        pred_list = self.create_pred_list(preds, indices)

        return pred_list


class PostEmbedder(BasePredPostProcessor):
    def __init__(
        self,
        transform: transforms.Compose,
        device: torch.device,
        optimize_transform: bool,
        labels: List[str],
    ):
        """Initialize the predictor postprocessor that extracts the embedding from the prediction tensor and returns a list of prediction data structures.

        Args:
            transform (Compose): Composed Torch transform object.
            device (torch.device): Torch device cpu or cuda.
            optimize_transform (bool): Whether to optimize the transform using TorchScript.
            labels (List[str]): List of labels.
        """
        super().__init__(transform, device, optimize_transform, labels)

    @Timer("PostEmbedder.run", "{name}: {milliseconds:.2f} ms", logger.debug)
    def run(self, preds: torch.Tensor) -> List[Prediction]:
        """Post-processes the prediction tensor using argmax and returns a list of prediction data structures, one for each face.

        Args:
            preds (torch.Tensor): Batch prediction tensor.

        Returns:
            List[Prediction]: List of prediction data structures containing the predicted labels and confidence scores for each face in the batch.
        """
        if isinstance(preds, tuple):
            preds = preds[0]

        indices = [0] * preds.shape[0]
        pred_list = self.create_pred_list(preds, indices)

        return pred_list
