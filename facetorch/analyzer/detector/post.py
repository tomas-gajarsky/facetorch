from abc import abstractmethod
from itertools import product as product
from math import ceil
from typing import List, Tuple, Union

import torch
from codetiming import Timer
from facetorch.base import BaseProcessor
from facetorch.datastruct import Detection, Dimensions, Face, ImageData, Location
from facetorch.logger import LoggerJsonFile
from facetorch.utils import rgb2bgr
from torchvision import transforms

logger = LoggerJsonFile().logger


class BaseDetPostProcessor(BaseProcessor):
    @Timer(
        "BaseDetPostProcessor.__init__", "{name}: {milliseconds:.2f} ms", logger.debug
    )
    def __init__(
        self,
        transform: transforms.Compose,
        device: torch.device,
        optimize_transform: bool,
    ):
        """Base class for detector post processors.

        All detector post processors should subclass it.
        All subclass should overwrite:

        - Methods:``run``, used for running the processing

        Args:
            device (torch.device): Torch device cpu or cuda.
            transform (transforms.Compose): Transform compose object to be applied to the image.
            optimize_transform (bool): Whether to optimize the transform.

        """
        super().__init__(transform, device, optimize_transform)

    @abstractmethod
    def run(
        self, data: ImageData, logits: Union[torch.Tensor, Tuple[torch.Tensor]]
    ) -> ImageData:
        """Abstract method that runs the detector post processing functionality
        and returns the data object.

        Args:
            data (ImageData): ImageData object containing the image tensor.
            logits (Union[torch.Tensor, Tuple[torch.Tensor]]): Output of the detector model.

        Returns:
            ImageData: Image data object with Detection tensors and detected Face objects.


        """


class PriorBox:
    """
    PriorBox class for generating prior boxes.

    Args:
        min_sizes (List[List[int]]): List of list of minimum sizes for each feature map.
        steps (List[int]): List of steps for each feature map.
        clip (bool): Whether to clip the prior boxes to the image boundaries.
    """

    def __init__(self, min_sizes: List[List[int]], steps: List[int], clip: bool):
        self.min_sizes = [list(min_size) for min_size in min_sizes]
        self.steps = list(steps)
        self.clip = clip

    def forward(self, dims: Dimensions) -> torch.Tensor:
        """Generate prior boxes for each feature map.

        Args:
            dims (Dimensions): Dimensions of the image.

        Returns:
            torch.Tensor: Tensor of prior boxes.
        """
        feature_maps = [
            [ceil(dims.height / step), ceil(dims.width / step)] for step in self.steps
        ]
        anchors = []
        for k, f in enumerate(feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / dims.width
                    s_ky = min_size / dims.height
                    dense_cx = [x * self.steps[k] / dims.width for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / dims.height for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors.append([cx, cy, s_kx, s_ky])

        output = torch.Tensor(anchors)
        if self.clip:
            output.clamp_(min=0, max=1)
        return output


class PostRetFace(BaseDetPostProcessor):
    @Timer("PostRetFace.__init__", "{name}: {milliseconds:.2f} ms", logger.debug)
    def __init__(
        self,
        transform: transforms.Compose,
        device: torch.device,
        optimize_transform: bool,
        confidence_threshold: float,
        top_k: int,
        nms_threshold: float,
        keep_top_k: int,
        score_threshold: float,
        prior_box: PriorBox,
        variance: List[float],
        reverse_colors: bool = False,
        expand_pixels: int = 0,
    ):
        """Initialize the detector postprocessor. Modified from https://github.com/biubug6/Pytorch_Retinaface.

        Args:
            transform (Compose): Composed Torch transform object.
            device (torch.device): Torch device cpu or cuda.
            optimize_transform (bool): Whether to optimize the transform.
            confidence_threshold (float): Confidence threshold for face detection.
            top_k (int): Top K faces to keep before NMS.
            nms_threshold (float): NMS threshold.
            keep_top_k (int): Keep top K faces after NMS.
            score_threshold (float): Score threshold for face detection.
            prior_box (PriorBox): PriorBox object.
            variance (List[float]): Prior box variance.
            reverse_colors (bool): Whether to reverse the colors of the image tensor from RGB to BGR or vice versa. If False, the colors remain unchanged. Default: False.
            expand_pixels (int): Number of pixels to expand the face location and tensor by. Default: 0.
        """
        super().__init__(transform, device, optimize_transform)
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.nms_threshold = nms_threshold
        self.keep_top_k = keep_top_k
        self.score_threshold = score_threshold
        self.prior_box = prior_box
        self.variance = list(variance)
        self.reverse_colors = reverse_colors
        self.expand_pixels = expand_pixels

    @Timer("PostRetFace.run", "{name}: {milliseconds:.2f} ms", logger.debug)
    def run(
        self,
        data: ImageData,
        logits: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> ImageData:
        """Run the detector postprocessor.

        Args:
            data (ImageData): ImageData object containing the image tensor.
            logits (Union[torch.Tensor, Tuple[torch.Tensor]]): Output of the detector model.

        Returns:
            ImageData: Image data object with detection tensors and detected Face objects.
        """
        data.det = Detection(loc=logits[0], conf=logits[1], landmarks=logits[2])

        if self.reverse_colors:
            data.tensor = rgb2bgr(data.tensor)

        data = self._process_dets(data)
        data = self._extract_faces(data)
        return data

    def _process_dets(self, data: ImageData) -> ImageData:
        """Compute the detections and add them to the data detector.

        Args:
            data (ImageData): Image data with with locations and confidences from detector.

        Returns:
            ImageData: Image data object with detections.
        """

        def _decode(
            _loc: torch.Tensor, _priors: torch.Tensor, variances: List[float]
        ) -> torch.Tensor:
            _boxes = torch.cat(
                (
                    _priors[:, :2] + _loc[:, :2] * variances[0] * _priors[:, 2:],
                    _priors[:, 2:] * torch.exp(_loc[:, 2:] * variances[1]),
                ),
                1,
            )
            _boxes[:, :2] -= _boxes[:, 2:] / 2
            _boxes[:, 2:] += _boxes[:, :2]
            return _boxes

        def _extract_boxes(_loc: torch.Tensor) -> torch.Tensor:
            priors = self.prior_box.forward(data.dims)
            priors = priors.to(self.device)
            prior_data = priors.data
            _boxes = _decode(_loc.data.squeeze(0), prior_data, self.variance)
            img_scale = torch.Tensor([data.dims.width, data.dims.height]).repeat(2)
            _boxes = _boxes * img_scale.to(self.device)
            return _boxes

        def _nms(dets: torch.Tensor, thresh: float) -> torch.Tensor:
            """Non-maximum suppression."""
            x1 = dets[:, 0]
            y1 = dets[:, 1]
            x2 = dets[:, 2]
            y2 = dets[:, 3]

            areas = (x2 - x1 + 1) * (y2 - y1 + 1)
            order = torch.arange(dets.shape[0], device=self.device)

            zero_tensor = torch.tensor(0.0).to(self.device)
            keep = []
            while order.size()[0] > 0:
                i = order[0]
                keep.append(i)
                xx1 = torch.maximum(x1[i], x1[order[1:]])
                yy1 = torch.maximum(y1[i], y1[order[1:]])
                xx2 = torch.minimum(x2[i], x2[order[1:]])
                yy2 = torch.minimum(y2[i], y2[order[1:]])

                w = torch.maximum(zero_tensor, xx2 - xx1 + 1)
                h = torch.maximum(zero_tensor, yy2 - yy1 + 1)
                inter = torch.multiply(w, h)
                ovr = inter / (areas[i] + areas[order[1:]] - inter)

                inds = ovr <= thresh
                order = order[1:][inds]

            if len(keep) > 0:
                keep = torch.stack(keep)
            else:
                keep = torch.tensor([])

            return keep

        def _extract_dets(_conf: torch.Tensor, _boxes: torch.Tensor) -> torch.Tensor:
            scores = _conf.squeeze(0).data[:, 1]
            # ignore low scores
            inds = scores > self.confidence_threshold
            _boxes = _boxes[inds]
            scores = scores[inds]
            # keep top-K before NMS
            order = torch.argsort(scores, descending=True)[: self.top_k]
            _boxes = _boxes[order]
            scores = scores[order]
            # do NMS
            _dets = torch.hstack((_boxes, scores.unsqueeze(1)))
            keep = _nms(_dets, self.nms_threshold)

            if not keep.shape[0] == 0:
                _dets = _dets[keep, :]
                # keep top-K after NMS
                _dets = _dets[: self.keep_top_k, :]
                # keep dets with score > score_threshold
                _dets = _dets[_dets[:, 4] > self.score_threshold]

            return _dets

        data.det.boxes = _extract_boxes(data.det.loc)
        data.det.dets = _extract_dets(data.det.conf, data.det.boxes)
        return data

    def _extract_faces(self, data: ImageData) -> ImageData:
        """Extracts the faces from the original image using the detections.

        Args:
            data (ImageData): Image data with image tensor and detections.

        Returns:
            ImageData: Image data object with extracted faces.

        """

        def _get_coordinates(_det: torch.Tensor) -> Location:
            _det = torch.round(_det).int()
            loc = Location(
                x1=int(_det[0]),
                y1=int(_det[1]),
                x2=int(_det[2]),
                y2=int(_det[3]),
            )

            loc.expand(amount=self.expand_pixels)
            loc.form_square()

            return loc

        for indx, det in enumerate(data.det.dets):
            loc = _get_coordinates(det)
            face_tensor = data.tensor[0, :, loc.y1 : loc.y2, loc.x1 : loc.x2]
            dims = Dimensions(face_tensor.shape[-2], face_tensor.shape[-1])
            size_img = data.tensor.shape[-2] * data.tensor.shape[-1]
            size_ratio = (dims.height * dims.width) / size_img

            if not any([dim == 0 for dim in face_tensor.shape]):
                face = Face(
                    indx=indx, loc=loc, tensor=face_tensor, dims=dims, ratio=size_ratio
                )
                data.faces.append(face)

        return data
