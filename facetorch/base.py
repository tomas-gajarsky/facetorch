import os
from abc import ABCMeta, abstractmethod
from typing import Any, Optional, Tuple, Union

import torch
from codetiming import Timer
from torchvision import transforms

from facetorch import utils
from facetorch.datastruct import ImageData
from facetorch.exceptions import ConfigurationError, ModelCompatibilityError
from facetorch.input import InputSpec, canonicalize_image_tensor
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
    def run(self, image_source: Any, fix_img_size: bool = False, **kwargs) -> ImageData:
        """Read one configured source and return canonical image data.

        Args:
            image_source (Any): Source accepted by the concrete reader.
            fix_img_size (bool): Apply the configured size transform.

        Returns:
            ImageData: ImageData object with the image tensor.
        """
        pass

    def process_tensor(
        self,
        tensor: torch.Tensor,
        fix_img_size: bool,
        *,
        input_policy: str = "coerce",
        input_spec: Optional[InputSpec] = None,
        source_kind: str = "torch",
        path_input: Optional[str] = None,
    ) -> ImageData:
        """Canonicalize one image tensor to RGB float32 ``BCHW`` in ``0..255``.

        Args:
            tensor (torch.Tensor): Source tensor. Torch inputs default to CHW/BCHW;
                NumPy and decoded inputs default to HWC/BHWC. Use ``InputSpec`` to
                declare another supported layout.
            fix_img_size (bool): Whether to resize the image to a fixed size. If
                False, size_portrait and size_landscape are ignored.
            input_policy (str): ``coerce`` or ``strict``.
            input_spec (Optional[InputSpec]): Explicit source representation.
            source_kind (str): Source convention used for deterministic defaults.
            path_input (Optional[str]): Local source path retained as metadata.
        """

        canonical = canonicalize_image_tensor(
            tensor,
            source_kind=source_kind,
            input_policy=input_policy,
            input_spec=input_spec,
        )
        data = ImageData(path_input=path_input, warnings=canonical.warnings)
        data.tensor = canonical.tensor.to(self.device)

        if fix_img_size:
            if self.transform is None:
                raise ConfigurationError(
                    "fix_img_size=True requires a configured reader transform."
                )
            data.tensor = self.transform(data.tensor)

        data.tensor = data.tensor.to(dtype=torch.float32)
        if data.tensor.ndim != 4 or data.tensor.shape[:2] != (1, 3):
            raise ConfigurationError(
                "Reader transform must preserve the canonical B=1, RGB BCHW layout."
            )
        if fix_img_size:
            if not torch.isfinite(data.tensor).all():
                raise ConfigurationError("Reader transform produced NaN or Inf values.")
            if data.tensor.numel() and (
                float(data.tensor.min()) < 0.0 or float(data.tensor.max()) > 255.0
            ):
                raise ConfigurationError(
                    "Reader transform must preserve canonical values within 0..255."
                )
        data.img = data.tensor[0].round().clamp(0, 255).to(torch.uint8).cpu()
        data.set_dims()
        data._facetorch_canonical = True

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
        self.verify_on_use = False

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    @abstractmethod
    def run(self) -> Optional[str]:
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
        self._verify_artifacts = (
            getattr(self.downloader, "verify_on_use", False) is True
        )
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
        should_verify = self._verify_artifacts
        if should_verify or not os.path.exists(self.path_local):
            dir_local = os.path.dirname(self.path_local)
            if dir_local:
                try:
                    os.makedirs(dir_local, exist_ok=True)
                except OSError as exc:
                    raise ConfigurationError(
                        f"Cannot create model cache directory {dir_local!r}. "
                        "Set FACETORCH_CACHE_DIR to a writable directory or "
                        "override downloader.path_local."
                    ) from exc
            resolved_path = self.downloader.run()
            if resolved_path is not None:
                self.path_local = os.fspath(resolved_path)
            else:
                self.path_local = self.downloader.path_local

        active_format = getattr(self.downloader, "active_format", None)
        if active_format not in {"pt2", "torchscript", "torch_data"}:
            active_format = None
        if active_format == "pt2" or (
            active_format is None and self.path_local.endswith(".pt2")
        ):
            model = self._load_exported_model_with_fallback()
        elif active_format == "torch_data":
            raise ConfigurationError(
                f"Artifact {self.path_local} contains data, not an executable model."
            )
        elif self.native_model_class is not None:
            model = self._load_native_model()
            model.eval()
        else:
            model = torch.jit.load(self.path_local, map_location=self.device)
            model.eval()

        if self.compile_model:
            model = torch.compile(model, **self.compile_options)

        return model

    @staticmethod
    def _is_export_schema_mismatch_error(exc: Exception) -> bool:
        err_msg = str(exc).lower()
        return any(
            key in err_msg
            for key in ("schema version", "serialized version", "example_inputs")
        ) or (
            "no item named 'version' in the archive" in err_msg
            or "serialized_exported_program.json" in err_msg
        )

    def _build_export_schema_mismatch_message(self) -> str:
        descriptor = getattr(self.downloader, "active_descriptor", None)
        active_filename = getattr(descriptor, "filename", None)
        tried = getattr(self.downloader, "_last_candidates", None)
        tried_msg = ""
        if isinstance(tried, list) and tried:
            tried_msg = f" Tried candidates: {', '.join(tried)}."

        active_msg = ""
        if active_filename:
            active_msg = f" Last downloaded candidate: {active_filename}."

        return (
            f"Cannot load {self.path_local}: the exported .pt2 model appears to be "
            f"incompatible with current PyTorch ({torch.__version__})."
            f"{active_msg}{tried_msg} "
            "Upload/export a compatible model artifact for your current torch "
            "major.minor version, or use a torch version compatible with one of the "
            "published artifacts."
        )

    def _load_exported_model_with_fallback(self) -> torch.nn.Module:
        """Load .pt2 and use only the next explicit manifest candidate on mismatch."""
        while True:
            try:
                return self._load_exported_model()
            except (RuntimeError, AssertionError, KeyError) as e:
                if not self._is_export_schema_mismatch_error(e):
                    raise

                mark_incompatible = (
                    getattr(self.downloader, "mark_incompatible", None)
                    if self._verify_artifacts
                    else None
                )
                if callable(mark_incompatible):
                    mark_incompatible()

                try_next = getattr(self.downloader, "try_next", None)
                if callable(try_next):
                    logger.warning(
                        f"Exported model load mismatch for {self.path_local} with "
                        f"torch={torch.__version__}. Trying next downloader candidate."
                    )
                    try:
                        has_next = try_next(force_download=False)
                    except TypeError:
                        try:
                            has_next = try_next()
                        except ModelCompatibilityError:
                            has_next = False
                    except ModelCompatibilityError:
                        has_next = False
                    if has_next:
                        self.path_local = self.downloader.path_local
                        continue

                raise ModelCompatibilityError(
                    self._build_export_schema_mismatch_message()
                ) from e

    def _load_exported_model(self) -> torch.nn.Module:
        """Loads a torch.export .pt2 model or an active legacy .pt fallback."""
        active_format = getattr(self.downloader, "active_format", None)
        if active_format == "torchscript":
            model = torch.jit.load(self.path_local, map_location=self.device)
            model.eval()
            return model

        ep = torch.export.load(self.path_local)
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
