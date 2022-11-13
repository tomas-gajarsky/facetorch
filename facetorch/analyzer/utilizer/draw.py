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
        color: str,
        line_width: int,
    ):
        """Initializes the BoxDrawer class. This class is used to draw the face boxes to the image tensor.

        Args:
            transform (Compose): Composed Torch transform object.
            device (torch.device): Torch device cpu or cuda object.
            optimize_transform (bool): Whether to optimize the transform.
            color (str): Color of the boxes.
            line_width (int): Line width of the boxes.

        """
        super().__init__(transform, device, optimize_transform)
        self.color = color
        self.line_width = line_width

    @Timer("BoxDrawer.run", "{name}: {milliseconds:.2f} ms", logger=logger.debug)
    def run(self, data: ImageData) -> ImageData:
        """Draws face boxes to the image tensor.

        Args:
            data (ImageData): ImageData object containing the image tensor and face locations.
        Returns:
            ImageData: ImageData object containing the image tensor with face boxes.
        """
        loc_tensor = data.aggregate_loc_tensor()
        labels = [str(face.indx) for face in data.faces]
        data.img = torchvision.utils.draw_bounding_boxes(
            image=data.img,
            boxes=loc_tensor,
            labels=labels,
            colors=self.color,
            width=self.line_width,
        )

        return data


class LandmarkDrawerTorch(BaseUtilizer):
    def __init__(
        self,
        transform: transforms.Compose,
        device: torch.device,
        optimize_transform: bool,
        width: int,
        color: str,
    ):
        """Initializes the LandmarkDrawer class. This class is used to draw the 3D face landmarks to the image tensor.

        Args:
            transform (Compose): Composed Torch transform object.
            device (torch.device): Torch device cpu or cuda object.
            optimize_transform (bool): Whether to optimize the transform.
            width (int): Marker keypoint width.
            color (str): Marker color.

        """
        super().__init__(transform, device, optimize_transform)
        self.width = width
        self.color = color

    @Timer("LandmarkDrawer.run", "{name}: {milliseconds:.2f} ms", logger=logger.debug)
    def run(self, data: ImageData) -> ImageData:
        """Draws 3D face landmarks to the image tensor.

        Args:
            data (ImageData): ImageData object containing the image tensor and 3D face landmarks.
        Returns:
            ImageData: ImageData object containing the image tensor with 3D face landmarks.
        """
        data = self._draw_landmarks(data)

        return data

    def _draw_landmarks(self, data: ImageData) -> ImageData:
        """Draws 3D face landmarks to the image tensor.

        Args:
            data (ImageData): ImageData object containing the image tensor, 3D face landmarks, and faces.

        Returns:
            (ImageData): ImageData object containing the image tensor with 3D face landmarks.
        """

        if len(data.faces) > 0:
            pts = [face.preds["align"].other["lmk3d"].cpu() for face in data.faces]

            img_in = data.img.clone()
            pts = torch.stack(pts)
            pts = torch.swapaxes(pts, 2, 1)

            img_out = torchvision.utils.draw_keypoints(
                img_in,
                pts,
                colors=self.color,
                radius=self.width,
            )
            data.img = img_out

        return data
