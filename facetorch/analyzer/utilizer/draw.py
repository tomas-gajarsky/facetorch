import torch
import torchvision
from codetiming import Timer
from facetorch.base import BaseUtilizer
from facetorch.datastruct import ImageData
from facetorch.logger import LoggerJsonFile
from torchvision import transforms

logger = LoggerJsonFile().logger


class BoxDrawer(BaseUtilizer):
    def __init__(
        self,
        transform: transforms.Compose,
        device: torch.device,
        optimize_transform: bool,
        **kwargs
    ):
        """Initializes the BoxDrawer class. This class is used to draw the face boxes to the image tensor.

        Args:
            transform (Compose): Composed Torch transform object.
            device (torch.device): Torch device cpu or cuda object.
            optimize_transform (bool): Whether to optimize the transform.


        """
        self.__dict__.update(kwargs)
        super().__init__(transform, device, optimize_transform)

    @Timer("BoxDrawer.run", "{name}: {milliseconds:.2f} ms", logger.debug)
    def run(self, data: ImageData) -> ImageData:
        """Draws face boxes to the image tensor.

        Args:
            data (ImageData): ImageData object containing the image tensor and face locations.
        Returns:
            ImageData: ImageData object
        """
        data = self.draw_boxes(data)

        return data

    def draw_boxes(self, data: ImageData) -> ImageData:
        """Draws face boxes to the image tensor.

        Args:
            data (ImageData): ImageData object containing the image tensor, detections, and faces.

        Returns:
            None
        """
        loc_tensor = data.aggregate_loc_tensor()
        labels = [str(face.indx) for face in data.faces]
        data.img = torchvision.utils.draw_bounding_boxes(
            image=data.img,
            boxes=loc_tensor,
            labels=labels,
            colors="green",
            width=3,
        )

        return data
