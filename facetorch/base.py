import os
import copy
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
    @Timer(
        "BaseProcessor.__init__", "{name}: {milliseconds:.2f} ms", logger=logger.debug
    )
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

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    @abstractmethod
    def run(self):
        """Abstract method that should implement a tensor processing functionality"""


class BaseReader(BaseProcessor):
    @Timer("BaseReader.__init__", "{name}: {milliseconds:.2f} ms", logger=logger.debug)
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
        pass

    def process_tensor(self, tensor: torch.Tensor, fix_img_size: bool) -> ImageData:
        """Read an input tensor and normalize it to shape (B, C, H, W).

        Args:
            tensor (torch.Tensor): Image tensor with values between 0-255. Accepted
                shapes are (H, W), (C, H, W), (H, W, C), or (B, C, H, W), where
                C is in {1, 3, 4}. Unambiguous HWC tensors are converted to CHW.
                Batched tensors currently support only B=1.
            fix_img_size (bool): Whether to resize the image to a fixed size. If
                False, size_portrait and size_landscape are ignored.
        """

        data = ImageData(path_input=None)
        data.tensor = copy.deepcopy(tensor)

        if data.tensor.dim() == 2:
            data.tensor = data.tensor.unsqueeze(0)

        if data.tensor.dim() == 3:
            c0 = data.tensor.shape[0]
            c2 = data.tensor.shape[2]
            chw_like = c0 in (1, 3, 4)
            hwc_like = c2 in (1, 3, 4)
            if hwc_like and not chw_like:
                data.tensor = data.tensor.permute(2, 0, 1)
            elif not chw_like and not hwc_like:
                raise ValueError(
                    "Invalid 3D tensor shape. Expected CHW with C in {1,3,4} or "
                    "HWC with channels in the last dimension."
                )
            elif chw_like and hwc_like:
                raise ValueError(
                    "Ambiguous 3D tensor layout: both first and last dimensions "
                    "look like channel dimensions. Please pass CHW explicitly."
                )
            data.tensor = data.tensor.unsqueeze(0)

        if data.tensor.dim() != 4:
            raise ValueError(
                f"Unsupported tensor rank {data.tensor.dim()}. Expected 2D, 3D, or 4D input."
            )

        if data.tensor.shape[0] != 1:
            raise ValueError(
                f"Batched tensor input is not supported yet. Expected B=1, got B={data.tensor.shape[0]}."
            )

        channels = data.tensor.shape[1]
        if channels not in (1, 3, 4):
            raise ValueError(
                f"Unsupported channel count: {channels}. Expected channels in {{1,3,4}}."
            )

        if channels == 1:
            data.tensor = data.tensor.repeat(1, 3, 1, 1)
        elif channels == 4:
            data.tensor = data.tensor[:, :3, :, :]

        data.tensor = data.tensor.to(self.device)

        if fix_img_size:
            data.tensor = self.transform(data.tensor)

        data.img = data.tensor.squeeze(0).cpu()
        data.tensor = data.tensor.type(torch.float32)
        data.set_dims()

        return data


class BaseDownloader(object, metaclass=ABCMeta):
    @Timer(
        "BaseDownloader.__init__", "{name}: {milliseconds:.2f} ms", logger=logger.debug
    )
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

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    @abstractmethod
    def run(self) -> None:
        """Abstract method that should implement the download functionality"""


class BaseModel(object, metaclass=ABCMeta):
    @Timer("BaseModel.__init__", "{name}: {milliseconds:.2f} ms", logger=logger.debug)
    def __init__(
        self,
        downloader: BaseDownloader,
        device: torch.device,
        native_model_class: Optional[str] = None,
        compile_model: bool = False,
        compile_options: Optional[dict] = None,
    ):
        """Base class for torch models.

        All detectors and predictors should subclass it.
        All subclass should overwrite:

        - Methods:``run``, supporting to make detections and predictions with the model.

        Supports three model formats:

        - **TorchScript** (.pt): Legacy format loaded via ``torch.jit.load()``. Deprecated.
        - **Exported Program** (.pt2): Modern portable format via ``torch.export``.
          Loaded with ``torch.export.load()`` — no model source code needed.
        - **Native + state_dict**: ``native_model_class`` specifies the ``nn.Module``
          class; weights are extracted from the TorchScript file's state_dict.

        Args:
            downloader (BaseDownloader): Downloader for the model.
            device (torch.device): Torch device cpu or cuda.
            native_model_class (Optional[str]): Fully qualified class name of a native
                PyTorch nn.Module to use instead of TorchScript. The TorchScript file is
                loaded to extract the state_dict, which is then loaded into an instance of
                this class. Ignored when loading .pt2 files. Default: None.
            compile_model (bool): If True, wraps the model with ``torch.compile()``
                for optimized inference. Default: False.
            compile_options (Optional[dict]): Keyword arguments passed to
                ``torch.compile()`` (e.g. mode, backend, fullgraph). Default: None.

        Attributes:
            model (torch.nn.Module): Loaded model.

        """
        super().__init__()
        self.downloader = downloader
        self.path_local = self.downloader.path_local
        self.device = device
        self.native_model_class = native_model_class
        self.compile_model = compile_model
        self.compile_options = compile_options or {}

        self.model = self.load_model()

    @Timer("BaseModel.load_model", "{name}: {milliseconds:.2f} ms", logger=logger.debug)
    def load_model(self) -> torch.nn.Module:
        """Loads the model from the local file.

        Loading strategy by file extension:

        - ``.pt2``: ``torch.export.load()`` — portable exported program
        - ``.pt`` with ``native_model_class``: native nn.Module + state_dict from TorchScript
        - ``.pt`` without ``native_model_class``: ``torch.jit.load()`` (legacy TorchScript)

        After loading, optionally wraps with ``torch.compile()`` if enabled.

        Returns:
            torch.nn.Module: Loaded model in eval mode.
        """
        if not os.path.exists(self.path_local):
            dir_local = os.path.dirname(self.path_local)
            os.makedirs(dir_local, exist_ok=True)
            self.downloader.run()

        if self.path_local.endswith(".pt2"):
            model = self._load_exported_model()
        elif self.native_model_class is not None:
            model = self._load_native_model()
            model.eval()
        else:
            model = torch.jit.load(self.path_local, map_location=self.device)
            model.eval()

        if self.compile_model:
            model = torch.compile(model, **self.compile_options)

        return model

    def _load_exported_model(self) -> torch.nn.Module:
        """Loads a torch.export .pt2 model."""
        try:
            ep = torch.export.load(self.path_local)
        except (RuntimeError, AssertionError) as e:
            err_msg = str(e).lower()
            if any(k in err_msg for k in ("schema version", "serialized version", "example_inputs")):
                raise RuntimeError(
                    f"Cannot load {self.path_local}: the .pt2 model was exported with a "
                    f"different PyTorch version. The bundled models require torch >=2.3.0,<2.5.0. "
                    f"Current version: {torch.__version__}. Install a compatible version or "
                    f"re-export the model with your current PyTorch."
                ) from e
            raise
        model = ep.module()
        model.to(self.device)
        return model

    def _load_native_model(self) -> torch.nn.Module:
        """Loads a native PyTorch model using weights from a .pth or TorchScript file."""
        import importlib

        module_path, class_name = self.native_model_class.rsplit(".", 1)
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        model = model_class()

        if self.path_local.endswith(".pth"):
            state_dict = torch.load(
                self.path_local, map_location="cpu", weights_only=True
            )
            model.load_state_dict(state_dict, strict=True)
        else:
            ts_model = torch.jit.load(self.path_local, map_location="cpu")
            state_dict = dict(ts_model.state_dict())
            for name, mod in ts_model.named_modules():
                for buf in ("running_mean", "running_var"):
                    key = f"{name}.{buf}" if name else buf
                    if key not in state_dict:
                        try:
                            state_dict[key] = getattr(mod, buf)
                        except AttributeError:
                            pass
            if state_dict:
                model.load_state_dict(state_dict, strict=True)
            elif hasattr(model, "load_from_torchscript"):
                model.load_from_torchscript(ts_model)

        model.to(self.device)
        return model

    @Timer("BaseModel.inference", "{name}: {milliseconds:.2f} ms", logger=logger.debug)
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

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

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
