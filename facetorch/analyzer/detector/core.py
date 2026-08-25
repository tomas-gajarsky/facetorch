from typing import Optional

import torch
from codetiming import Timer
from facetorch.base import BaseDownloader, BaseModel
from facetorch.datastruct import Dimensions, Face, ImageData, Location
from facetorch.logger import LoggerJsonFile

from .post import BaseDetPostProcessor
from .pre import BaseDetPreProcessor

logger = LoggerJsonFile().logger


class FaceDetector(BaseModel):
    @Timer(
        "FaceDetector.__init__", "{name}: {milliseconds:.2f} ms", logger=logger.debug
    )
    def __init__(
        self,
        downloader: BaseDownloader,
        device: torch.device,
        preprocessor: BaseDetPreProcessor,
        postprocessor: BaseDetPostProcessor,
        native_model_class: Optional[str] = None,
        compile_model: bool = False,
        compile_options: Optional[dict] = None,
    ):
        """FaceDetector is a wrapper around a neural network model that is trained to detect faces.

        Args:
            downloader (BaseDownloader): Downloader that downloads the model.
            device (torch.device): Torch device cpu or cuda for the model.
            preprocessor (BaseDetPreProcessor): Preprocessor that runs before the model.
            postprocessor (BaseDetPostProcessor): Postprocessor that runs after the model.
            native_model_class (Optional[str]): Fully qualified native model class.
            compile_model (bool): If True, compile the loaded model. Default: False.
            compile_options (Optional[dict]): Keyword arguments forwarded to
                ``torch.compile``. Default: None.
        """
        super().__init__(
            downloader,
            device,
            native_model_class=native_model_class,
            compile_model=compile_model,
            compile_options=compile_options,
        )

        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

    @Timer("FaceDetector.run", "{name}: {milliseconds:.2f} ms", logger=logger.debug)
    def run(self, data: ImageData) -> ImageData:
        """Detect all faces in the image.

        Args:
            ImageData: ImageData object containing the image tensor with values between 0 - 255 and shape (batch_size, channels, height, width).

        Returns:
            ImageData: Image data object with Detection tensors and detected Face objects.
        """
        preserves_input = (
            getattr(self.preprocessor, "preserves_input_tensor", False) is True
        )
        raw_tensor = data.tensor if preserves_input else data.tensor.clone()
        img_h, img_w = raw_tensor.shape[-2:]
        data = self.preprocessor.run(data)
        logits = self.inference(data.tensor)
        data = self.postprocessor.run(data, logits)

        data.tensor = raw_tensor
        data.set_dims()

        extract_faces = getattr(self.postprocessor, "extract_faces", None)
        if callable(extract_faces):
            data.faces = []
            data = extract_faces(data)
        else:
            data.faces = self._restore_custom_faces(data, img_w, img_h)
        self._clamp_detection_geometry(data, img_w, img_h)

        return data

    @staticmethod
    def _clamp_detection_geometry(
        data: ImageData, image_width: int, image_height: int
    ) -> None:
        """Keep all public detector geometry in original-image coordinates."""
        if data.det.dets.numel() > 0 and data.det.dets.ndim == 2:
            data.det.dets[:, 0].clamp_(0, image_width)
            data.det.dets[:, 2].clamp_(0, image_width)
            data.det.dets[:, 1].clamp_(0, image_height)
            data.det.dets[:, 3].clamp_(0, image_height)
        if data.det.boxes.numel() > 0 and data.det.boxes.ndim == 2:
            data.det.boxes[:, 0].clamp_(0, image_width)
            data.det.boxes[:, 2].clamp_(0, image_width)
            data.det.boxes[:, 1].clamp_(0, image_height)
            data.det.boxes[:, 3].clamp_(0, image_height)
        if data.det.landmarks.numel() > 0 and data.det.landmarks.ndim == 2:
            data.det.landmarks[:, 0::2].clamp_(0, image_width)
            data.det.landmarks[:, 1::2].clamp_(0, image_height)

    @staticmethod
    def _restore_custom_faces(
        data: ImageData, image_width: int, image_height: int
    ) -> list:
        """Validate and recrop faces produced directly by a custom postprocessor."""
        restored = []
        image_area = image_width * image_height
        for face in data.faces:
            loc = Location(
                x1=face.loc.x1,
                y1=face.loc.y1,
                x2=face.loc.x2,
                y2=face.loc.y2,
            )
            loc.clamp(image_width, image_height)
            face_tensor = data.tensor[0, :, loc.y1 : loc.y2, loc.x1 : loc.x2]
            if face_tensor.numel() == 0:
                continue
            dims = Dimensions(
                height=int(face_tensor.shape[-2]),
                width=int(face_tensor.shape[-1]),
            )
            restored.append(
                Face(
                    indx=len(restored),
                    loc=loc,
                    dims=dims,
                    tensor=face_tensor,
                    ratio=(dims.height * dims.width) / image_area,
                    preds=face.preds,
                )
            )
        return restored
