from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from codetiming import Timer

from facetorch.logger import LoggerJsonFile

logger = LoggerJsonFile().logger


@dataclass
class Dimensions:
    """Data class for image dimensions.

    Attributes:
        height (int): Image height.
        width (int): Image width.
    """

    height: int = field(default=0)
    width: int = field(default=0)


@dataclass
class Location:
    """Data class for face location.

    Attributes:
        x1 (int): x1 coordinate
        x2 (int): x2 coordinate
        y1 (int): y1 coordinate
        y2 (int): y2 coordinate
    """

    x1: int = field(default=0)
    x2: int = field(default=0)
    y1: int = field(default=0)
    y2: int = field(default=0)

    def form_square(self) -> None:
        """Form a square from the location.

        Returns:
            None
        """
        height = self.y2 - self.y1
        width = self.x2 - self.x1

        if height > width:
            diff = height - width
            self.x1 = self.x1 - int(diff / 2)
            self.x2 = self.x2 + int(diff / 2)
        elif height < width:
            diff = width - height
            self.y1 = self.y1 - int(diff / 2)
            self.y2 = self.y2 + int(diff / 2)
        else:
            pass

    def expand(self, amount: int) -> None:
        """Expand the location while keeping the center.

        Args:
            amount (int): Amount of pixels to expand the location by.

        Returns:
            None
        """
        if amount != 0:
            self.x1 = self.x1 - amount
            self.y1 = self.y1 - amount
            self.x2 = self.x2 + amount
            self.y2 = self.y2 + amount


@dataclass
class Prediction:
    """Data class for face prediction results and derivatives.

    Attributes:
        label (str): Label of the face given by predictor.
        logits (torch.Tensor): Output of the predictor model for the face.
        other (Dict): Any other predictions and derivatives for the face.
    """

    label: str = field(default_factory=str)
    logits: torch.Tensor = field(default_factory=torch.Tensor)
    other: Dict = field(default_factory=dict)


@dataclass
class Detection:
    """Data class for detector output.

    Attributes:
        loc (torch.Tensor): Locations of faces
        conf (torch.Tensor): Confidences of faces
        landmarks (torch.Tensor): Landmarks of faces
        boxes (torch.Tensor): Bounding boxes of faces
        dets (torch.Tensor): Detections of faces

    """

    loc: torch.Tensor = field(default_factory=torch.Tensor)
    conf: torch.Tensor = field(default_factory=torch.Tensor)
    landmarks: torch.Tensor = field(default_factory=torch.Tensor)
    boxes: torch.Tensor = field(default_factory=torch.Tensor)
    dets: torch.Tensor = field(default_factory=torch.Tensor)


@dataclass
class Face:
    """Data class for face attributes.

    Attributes:
        indx (int): Index of the face.
        loc (Location): Location of the face in the image.
        dims (Dimensions): Dimensions of the face (height, width).
        tensor (torch.Tensor): Face tensor.
        ratio (float): Ratio of the face area to the image area.
        preds (Dict[str, Prediction]): Predictions of the face given by predictor set.
    """

    indx: int = field(default_factory=int)
    loc: Location = field(default_factory=Location)
    dims: Dimensions = field(default_factory=Dimensions)
    tensor: torch.Tensor = field(default_factory=torch.Tensor)
    ratio: float = field(default_factory=float)
    preds: Dict[str, Prediction] = field(default_factory=dict)


@dataclass
class ImageData:
    """The main data class used for passing data between the different facetorch modules.

    Attributes:
        path_input (str): Path to the input image.
        path_output (str): Path to the output image where the resulting image is saved.
        img (torch.Tensor): Original image tensor used for drawing purposes.
        tensor (torch.Tensor): Processed image tensor.
        dims (Dimensions): Dimensions of the image (height, width).
        det (Detection): Detection data given by the detector.
        faces (List[Face]): List of faces in the image.
        version (int): Version of the facetorch library.

    """

    path_input: str = field(default_factory=str)
    path_output: Optional[str] = field(default_factory=str)
    img: torch.Tensor = field(default_factory=torch.Tensor)
    tensor: torch.Tensor = field(default_factory=torch.Tensor)
    dims: Dimensions = field(default_factory=Dimensions)
    det: Detection = field(default_factory=Detection)
    faces: List[Face] = field(default_factory=list)
    version: str = field(default_factory=str)

    def add_preds(
        self,
        preds_list: List[Prediction],
        predictor_name: str,
        face_offset: int = 0,
    ) -> None:
        """Adds a list of predictions to the data object.

        Args:
            preds_list (List[Prediction]): List of predictions.
            predictor_name (str): Name of the predictor.
            face_offset (int): Offset of the face index where the predictions are added.

        Returns:
            None

        """
        j = 0
        for i in range(face_offset, face_offset + len(preds_list)):
            self.faces[i].preds[predictor_name] = preds_list[j]
            j += 1

    def reset_img(self) -> None:
        """Reset the original image tensor to empty state."""
        self.img = torch.tensor([])

    def reset_tensor(self) -> None:
        """Reset the processed image tensor to empty state."""
        self.tensor = torch.tensor([])

    def reset_face_tensors(self) -> None:
        """Reset the face tensors to empty state."""
        for i in range(0, len(self.faces)):
            self.faces[i].tensor = torch.tensor([])

    def reset_face_pred_tensors(self) -> None:
        """Reset the face prediction tensors to empty state."""
        for i in range(0, len(self.faces)):
            for key in self.faces[i].preds:
                self.faces[i].preds[key].logits = torch.tensor([])
                self.faces[i].preds[key].other = {}

    def reset_det_tensors(self) -> None:
        """Reset the detection object to empty state."""
        self.det = Detection()

    @Timer("ImageData.reset_faces", "{name}: {milliseconds:.2f} ms", logger.debug)
    def reset_tensors(self) -> None:
        """Reset the tensors to empty state."""
        self.reset_img()
        self.reset_tensor()
        self.reset_face_tensors()
        self.reset_face_pred_tensors()
        self.reset_det_tensors()

    def set_dims(self) -> None:
        """Set the dimensions attribute from the tensor attribute."""
        self.dims.height = self.tensor.shape[2]
        self.dims.width = self.tensor.shape[3]

    def aggregate_loc_tensor(self) -> torch.Tensor:
        """Aggregates the location tensor from all faces.

        Returns:
            torch.Tensor: Aggregated location tensor for drawing purposes.
        """
        loc_tensor = torch.zeros((len(self.faces), 4), dtype=torch.float32)
        for i in range(0, len(self.faces)):
            loc_tensor[i] = torch.tensor(
                [
                    self.faces[i].loc.x1,
                    self.faces[i].loc.y1,
                    self.faces[i].loc.x2,
                    self.faces[i].loc.y2,
                ]
            )
        return loc_tensor


@dataclass
class Response:
    """Data class for response data, which is a subset of ImageData.

    Attributes:
        faces (List[Face]): List of faces in the image.
        version (int): Version of the facetorch library.

    """

    faces: List[Face] = field(default_factory=list)
    version: str = field(default_factory=str)
