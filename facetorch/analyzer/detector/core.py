import torch
from codetiming import Timer
from facetorch.base import BaseDownloader, BaseModel
from facetorch.datastruct import ImageData
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

    @Timer("FaceDetector.run", "{name}: {milliseconds:.2f} ms", logger=logger.debug)
    def run(self, data: ImageData) -> ImageData:
        """Detect all faces in the image.

        Args:
            ImageData: ImageData object containing the image tensor with values between 0 - 255 and shape (batch_size, channels, height, width).

        Returns:
            ImageData: Image data object with Detection tensors and detected Face objects.
        """
        orig_tensor = data.tensor
        img_h, img_w = orig_tensor.shape[-2], orig_tensor.shape[-1]
        data = self.preprocessor.run(data)
        logits = self.inference(data.tensor)
        data = self.postprocessor.run(data, logits)

        if data.tensor.shape[-2] != img_h or data.tensor.shape[-1] != img_w:
            data.tensor = orig_tensor
            data.set_dims()

            if hasattr(data.det, "dets") and data.det.dets.numel() > 0:
                data.det.dets[:, 0].clamp_(0, img_w)
                data.det.dets[:, 2].clamp_(0, img_w)
                data.det.dets[:, 1].clamp_(0, img_h)
                data.det.dets[:, 3].clamp_(0, img_h)

            data.faces = []
            if hasattr(self.postprocessor, "_extract_faces"):
                data = self.postprocessor._extract_faces(data)

        return data
