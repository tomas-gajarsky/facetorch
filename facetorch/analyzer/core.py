import inspect
import logging
import os
import threading
import warnings
from collections.abc import Iterable, Iterator, Mapping, MutableMapping
from typing import Any, List, Optional, Union

import numpy as np
import torch
from codetiming import Timer
from PIL import Image

from facetorch.analyzer.predictor.core import FacePredictor
from facetorch.datastruct import (
    AnalysisResult,
    Detection,
    Dimensions,
    Face,
    ImageData,
    Location,
    Response,
)
from facetorch.exceptions import (
    ConfigurationError,
    FacetorchError,
    InferenceError,
    InputError,
)
from facetorch.input import InputSpec
from facetorch.logger import LoggerJsonFile
from importlib.metadata import version
from hydra.utils import instantiate
from omegaconf import OmegaConf

logger = LoggerJsonFile().logger

_UNLOADED = object()


class _LazyComponentRegistry(MutableMapping[str, Any]):
    """Mapping-compatible component registry that constructs values on access."""

    def __init__(
        self,
        configs: Optional[Mapping[str, Any]] = None,
        loaded: Optional[Mapping[str, Any]] = None,
        *,
        loader=None,
        lock: Optional[Any] = None,
    ) -> None:
        self._configs = dict(configs or {})
        self._loaded = dict(loaded or {})
        self._loader = loader
        self._lock = lock or threading.RLock()

    def __getitem__(self, name: str) -> Any:
        if name in self._loaded:
            return self._loaded[name]
        if name not in self._configs:
            raise KeyError(name)
        if self._loader is None:
            raise ConfigurationError(
                f"Component {name!r} has configuration but no component loader."
            )

        with self._lock:
            if name not in self._loaded:
                self._loaded[name] = self._loader(name, self._configs[name])
        return self._loaded[name]

    def __setitem__(self, name: str, component: Any) -> None:
        with self._lock:
            self._configs.pop(name, None)
            self._loaded[name] = component

    def __delitem__(self, name: str) -> None:
        with self._lock:
            if name not in self._configs and name not in self._loaded:
                raise KeyError(name)
            self._configs.pop(name, None)
            self._loaded.pop(name, None)

    def __iter__(self) -> Iterator[str]:
        names = list(self._configs)
        names.extend(name for name in self._loaded if name not in self._configs)
        return iter(names)

    def __len__(self) -> int:
        return len(set(self._configs).union(self._loaded))

    def __contains__(self, name: object) -> bool:
        """Check configured or loaded names without constructing a component."""
        return name in self._configs or name in self._loaded

    def copy(self) -> dict[str, Any]:
        """Return a regular mapping snapshot, loading the requested values."""
        return {name: self[name] for name in self}

    @property
    def loaded_names(self) -> tuple[str, ...]:
        """Names whose model wrappers have already been constructed."""
        return tuple(name for name in self if name in self._loaded)


class FaceAnalyzer(object):
    @Timer(
        "FaceAnalyzer.__init__", "{name}: {milliseconds:.2f} ms", logger=logger.debug
    )
    def __init__(self, cfg: OmegaConf):
        """FaceAnalyzer is the main class that reads images, runs face detection, tensor unification and facial feature prediction.
        It also draws bounding boxes and facial landmarks over the image.

        The following components are used:

        1. Reader - reads the image and returns an ImageData object containing the image tensor.
        2. Detector - wrapper around a neural network that detects faces.
        3. Unifier - processor that unifies sizes of all faces and normalizes them between 0 and 1.
        4. Predictor dict - dict of wrappers around neural networks trained to analyze facial features.
        5. Utilizer dict - dict of utilizer processors that can for example extract 3D face landmarks or draw boxes over the image.

        Args:
            cfg (OmegaConf): Config object with image reader, face detector, unifier and predictor configurations.

        Attributes:
            cfg (OmegaConf): Config object with image reader, face detector, unifier and predictor configurations.
            reader (BaseReader): Reader object that reads the image and returns an ImageData object containing the image tensor.
            detector (FaceDetector): Lazily loaded and cached FaceDetector object.
            unifier (FaceUnifier): FaceUnifier object that unifies sizes of all faces and normalizes them between 0 and 1.
            predictors (MutableMapping[str, FacePredictor]): Mapping of lazily loaded
                and cached predictors. Iterating names does not load models; accessing
                a value does.
            utilizers (MutableMapping[str, FaceUtilizer]): Mapping of lazily loaded
                utilizer objects. Selection-linked utilizers load only when their
                predictor ran.
            logger (logging.Logger): Logger object that logs messages to the console or to a file.

        """
        self.cfg = cfg
        self._component_lock = threading.RLock()

        if hasattr(self.cfg, "logger") and self.cfg.logger is not None:
            self.logger = instantiate(self.cfg.logger).logger
        else:
            self.logger = logging.getLogger("facetorch")
            if not self.logger.handlers:
                self.logger.setLevel(logging.INFO)
                self.logger.addHandler(logging.StreamHandler())

        self.logger.info("Initializing FaceAnalyzer")
        self.logger.debug("Config", extra=self.cfg.__dict__["_content"])

        self.logger.info("Initializing BaseReader")
        self.reader = instantiate(self.cfg.reader)
        self._reader_signature_owner = None
        self._reader_signature_parameters = None

        self.logger.info("Registering lazy FaceDetector")
        self._detector_config = self.cfg.detector if "detector" in self.cfg else None
        self._detector = _UNLOADED

        self.logger.info("Initializing FaceUnifier")
        if "unifier" in self.cfg:
            self.unifier = instantiate(self.cfg.unifier)
        else:
            self.unifier = None

        self.logger.info("Registering lazy FacePredictor objects")
        predictor_configs = {}
        if "predictor" in self.cfg:
            for predictor_name in self.cfg.predictor:
                self.logger.info(f"Registering FacePredictor {predictor_name}")
                predictor_configs[predictor_name] = self.cfg.predictor[predictor_name]
        self._predictors = _LazyComponentRegistry(
            predictor_configs,
            loader=self._load_predictor,
            lock=self._component_lock,
        )

        utilizer_configs = {}
        if "utilizer" in self.cfg:
            self.logger.info("Registering lazy BaseUtilizer objects")
            for utilizer_name in self.cfg.utilizer:
                self.logger.info(f"Registering BaseUtilizer {utilizer_name}")
                utilizer_configs[utilizer_name] = self.cfg.utilizer[utilizer_name]
        self._utilizers = _LazyComponentRegistry(
            utilizer_configs,
            loader=self._load_utilizer,
            lock=self._component_lock,
        )

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    @property
    def detector(self):
        """Return the configured detector, constructing and caching it on demand."""
        detector = self.__dict__.get("_detector", _UNLOADED)
        if detector is not _UNLOADED:
            return detector

        detector_config = self.__dict__.get("_detector_config")
        if detector_config is None:
            raise ConfigurationError("No face detector is configured.")

        lock = self._get_component_lock()
        with lock:
            if self._detector is _UNLOADED:
                self.logger.info("Initializing FaceDetector")
                self._detector = instantiate(detector_config)
        return self._detector

    @detector.setter
    def detector(self, detector) -> None:
        """Install an already-constructed detector, primarily for extensions/tests."""
        self._detector_config = None
        self._detector = detector

    @property
    def predictors(self) -> MutableMapping[str, FacePredictor]:
        """Return the lazy predictor mapping without constructing its values."""
        registry = self.__dict__.get("_predictors")
        if registry is None:
            registry = _LazyComponentRegistry(lock=self._get_component_lock())
            self._predictors = registry
        return registry

    @predictors.setter
    def predictors(self, predictors: Mapping[str, FacePredictor]) -> None:
        """Replace configured predictors with already-constructed components."""
        if not isinstance(predictors, Mapping):
            raise TypeError("predictors must be a mapping from names to predictors.")
        self._predictors = _LazyComponentRegistry(
            loaded=predictors,
            lock=self._get_component_lock(),
        )

    @property
    def configured_predictors(self) -> tuple[str, ...]:
        """Predictor names in deterministic configuration order, without loading."""
        return tuple(self.predictors)

    @property
    def loaded_predictors(self) -> tuple[str, ...]:
        """Predictor names whose wrappers and models are already cached."""
        registry = self.predictors
        if isinstance(registry, _LazyComponentRegistry):
            return registry.loaded_names
        return tuple(registry)

    @property
    def utilizers(self) -> MutableMapping[str, Any]:
        """Return the lazy utilizer mapping without constructing its values."""
        registry = self.__dict__.get("_utilizers")
        if registry is None:
            registry = _LazyComponentRegistry(lock=self._get_component_lock())
            self._utilizers = registry
        return registry

    @utilizers.setter
    def utilizers(self, utilizers: Mapping[str, Any]) -> None:
        """Replace configured utilizers with already-constructed components."""
        if not isinstance(utilizers, Mapping):
            raise TypeError("utilizers must be a mapping from names to utilizers.")
        self._utilizers = _LazyComponentRegistry(
            loaded=utilizers,
            lock=self._get_component_lock(),
        )

    @property
    def configured_utilizers(self) -> tuple[str, ...]:
        """Utilizer names in deterministic configuration order, without loading."""
        return tuple(self.utilizers)

    @property
    def loaded_utilizers(self) -> tuple[str, ...]:
        """Utilizer names whose objects are already cached."""
        registry = self.utilizers
        if isinstance(registry, _LazyComponentRegistry):
            return registry.loaded_names
        return tuple(registry)

    @property
    def detector_loaded(self) -> bool:
        """Whether the detector wrapper and model are already cached."""
        return self.__dict__.get("_detector", _UNLOADED) is not _UNLOADED

    def _get_component_lock(self):
        lock = self.__dict__.get("_component_lock")
        if lock is None:
            lock = threading.RLock()
            self._component_lock = lock
        return lock

    def _load_predictor(self, name: str, predictor_config) -> FacePredictor:
        self.logger.info(f"Initializing FacePredictor {name}")
        return instantiate(predictor_config)

    def _load_utilizer(self, name: str, utilizer_config):
        self.logger.info(f"Initializing BaseUtilizer {name}")
        return instantiate(utilizer_config)

    @staticmethod
    def _normalize_predictor_selection(
        selection: Optional[Iterable[str]], option_name: str
    ) -> Optional[tuple[str, ...]]:
        if selection is None:
            return None
        if isinstance(selection, (str, bytes)):
            raise ConfigurationError(
                f"{option_name} must be a collection of predictor names, not a string."
            )
        try:
            names = tuple(selection)
        except TypeError as exc:
            raise ConfigurationError(
                f"{option_name} must be a collection of predictor names."
            ) from exc

        invalid = [name for name in names if not isinstance(name, str) or not name]
        if invalid:
            raise ConfigurationError(
                f"{option_name} must contain only non-empty predictor names."
            )

        seen = set()
        duplicates = []
        for name in names:
            if name in seen and name not in duplicates:
                duplicates.append(name)
            seen.add(name)
        if duplicates:
            raise ConfigurationError(
                f"{option_name} contains duplicate predictor names: "
                + ", ".join(duplicates)
                + "."
            )
        return names

    def _select_predictor_names(
        self,
        include_predictors: Optional[Iterable[str]],
        exclude_predictors: Optional[Iterable[str]],
    ) -> tuple[str, ...]:
        """Validate selection and return names in configuration order."""
        include = self._normalize_predictor_selection(
            include_predictors, "include_predictors"
        )
        exclude = self._normalize_predictor_selection(
            exclude_predictors, "exclude_predictors"
        )
        if include is not None and exclude is not None:
            raise ConfigurationError(
                "Cannot specify both include_predictors and exclude_predictors. "
                "Use one or the other."
            )

        configured = self.configured_predictors
        configured_set = set(configured)
        requested = include if include is not None else exclude
        unknown = (
            [name for name in requested if name not in configured_set]
            if requested is not None
            else []
        )
        if unknown:
            raise ConfigurationError(
                "Unknown predictor name(s): "
                + ", ".join(unknown)
                + ". Configured predictors: "
                + (", ".join(configured) if configured else "none")
                + "."
            )

        if include is not None:
            included = set(include)
            return tuple(name for name in configured if name in included)
        if exclude is not None:
            excluded = set(exclude)
            return tuple(name for name in configured if name not in excluded)
        return configured

    @Timer("FaceAnalyzer.run", "{name}: {milliseconds:.2f} ms", logger=logger.debug)
    def run(
        self,
        image_source: Optional[
            Union[str, os.PathLike, torch.Tensor, np.ndarray, bytes, Image.Image]
        ] = None,
        path_image: Optional[str] = None,
        face_batch_size: Optional[int] = None,
        fix_img_size: bool = False,
        return_img_data: Optional[bool] = None,
        include_tensors: bool = False,
        path_output: Optional[str] = None,
        tensor: Optional[torch.Tensor] = None,
        include_predictors: Optional[List[str]] = None,
        exclude_predictors: Optional[List[str]] = None,
        skip_detector: bool = False,
        *,
        batch_size: Optional[int] = None,
        input_policy: str = "coerce",
        input_spec: Optional[InputSpec] = None,
    ) -> AnalysisResult:
        """Analyze exactly one source image and return one stable result type.

        Args:
            image_source: Input accepted by the configured reader. The default
                reader accepts local paths, tensors, NumPy arrays, bytes, and PIL
                images. URLs require an explicit URLReader configuration.
            path_image (Optional[str]): Deprecated. Use image_source instead.
            face_batch_size (Optional[int]): Number of faces from this image sent to
                each predictor at once. Default: 8.
            fix_img_size (bool): If True, resizes the image to the size specified in reader. Default is False.
            return_img_data (Optional[bool]): Deprecated no-op. Use
                ``include_tensors`` and the fields on ``AnalysisResult`` or call
                ``run_legacy`` for the former flag-dependent return type.
            include_tensors (bool): If True, includes tensors in the returned data object. If False, tensors are removed. Default is False.
            path_output (Optional[str]): Path where to save the image with detected faces. If None, the image is not saved. Default: None.
            tensor (Optional[torch.Tensor]): Deprecated. Use image_source instead.
            include_predictors (Optional[List[str]]): Names to run. None runs all
                configured predictors and an empty collection runs none.
            exclude_predictors (Optional[List[str]]): Names to omit. None and an
                empty collection omit none. Cannot be combined with an include.
            skip_detector (bool): If True, skip face detection, avoid constructing
                its model, and treat the input as a pre-cropped face. Default: False.
            batch_size (Optional[int]): Deprecated warning alias for
                ``face_batch_size`` throughout v1.x.
            input_policy (str): ``coerce`` (default) or ``strict``.
            input_spec (Optional[InputSpec]): Explicit source layout/range/color
                description, especially for strict-mode conversions.

        Returns:
            AnalysisResult: Stable result for the one source image.

        """

        compatibility_warnings = []

        if face_batch_size is not None and batch_size is not None:
            raise ConfigurationError(
                "Specify only face_batch_size; batch_size is its deprecated alias."
            )
        if batch_size is not None:
            message = (
                "batch_size is deprecated and will be removed after v1.x; "
                "use face_batch_size."
            )
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            compatibility_warnings.append(message)
            effective_face_batch_size = batch_size
        elif face_batch_size is None:
            effective_face_batch_size = 8
        else:
            effective_face_batch_size = face_batch_size

        if (
            isinstance(effective_face_batch_size, bool)
            or not isinstance(effective_face_batch_size, int)
            or effective_face_batch_size < 1
        ):
            raise ConfigurationError(
                "face_batch_size must be an integer greater than or equal to 1, "
                f"got {effective_face_batch_size!r}."
            )

        if return_img_data is not None:
            message = (
                "return_img_data no longer changes FaceAnalyzer.run's return type; "
                "use include_tensors or the explicit run_legacy adapter."
            )
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            compatibility_warnings.append(message)

        def _run_component(label, operation):
            try:
                return operation()
            except FacetorchError:
                raise
            except Exception as exc:
                raise InferenceError(f"{label} failed during analysis.") from exc

        def _predict_batch(
            data: ImageData, predictor: FacePredictor, predictor_name: str
        ) -> ImageData:
            n_faces = len(data.faces)

            for face_indx_start in range(0, n_faces, effective_face_batch_size):
                face_indx_end = min(
                    face_indx_start + effective_face_batch_size, n_faces
                )

                face_batch_tensor = torch.stack(
                    [face.tensor for face in data.faces[face_indx_start:face_indx_end]]
                )
                preds = predictor.run(face_batch_tensor)
                data.add_preds(preds, predictor_name, face_indx_start)

            return data

        self.logger.info("Running FaceAnalyzer")
        selected_predictors = self._select_predictor_names(
            include_predictors, exclude_predictors
        )
        configured_predictors = set(self.configured_predictors)

        supplied_sources = [
            name
            for name, value in (
                ("image_source", image_source),
                ("path_image", path_image),
                ("tensor", tensor),
            )
            if value is not None
        ]
        if len(supplied_sources) > 1:
            raise InputError(
                "Supply exactly one input source using image_source. Received: "
                + ", ".join(supplied_sources)
                + "."
            )
        if not supplied_sources:
            raise InputError(
                "image_source is required. Pass a file path, URL, tensor, numpy array, "
                "bytes, or PIL Image."
            )
        if path_image is not None:
            message = "path_image is deprecated; use image_source."
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            compatibility_warnings.append(message)
            image_source = path_image
        elif tensor is not None:
            message = "tensor is deprecated; use image_source."
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            compatibility_warnings.append(message)
            image_source = tensor

        self.logger.info("Reading image")
        data = self._read_input(
            image_source,
            fix_img_size,
            input_policy=input_policy,
            input_spec=input_spec,
        )
        data.warnings.extend(compatibility_warnings)

        path_output = None if path_output == "None" else path_output
        data.path_output = path_output

        try:
            data.version = version("facetorch")
        except Exception as e:
            self.logger.warning("Could not get version number", extra={"error": e})

        if skip_detector:
            self.logger.info("Skipping detector (skip_detector=True)")
            face_tensor = data.tensor[0]
            face = Face(
                indx=0,
                loc=Location(
                    x1=0, y1=0, x2=data.dims.width, y2=data.dims.height
                ),
                dims=Dimensions(
                    height=data.dims.height, width=data.dims.width
                ),
                tensor=face_tensor,
                ratio=1.0,
            )
            data.faces = [face]
            n_faces = 1
        else:
            self.logger.info("Detecting faces")
            data = _run_component("Face detector", lambda: self.detector.run(data))
            n_faces = len(data.faces)

        self.logger.info(f"Number of faces: {n_faces}")

        if n_faces > 0 and self.unifier is not None:
            self.logger.info("Unifying faces")
            data = _run_component("Face unifier", lambda: self.unifier.run(data))

            self.logger.info("Predicting facial features")
            for predictor_name in selected_predictors:
                predictor = self.predictors[predictor_name]
                self.logger.info(f"Running FacePredictor: {predictor_name}")
                data = _run_component(
                    f"Face predictor {predictor_name!r}",
                    lambda: _predict_batch(data, predictor, predictor_name),
                )

            self.logger.info("Utilizing facial features")
            ran_predictors = (
                set(data.faces[0].preds.keys()) if data.faces else set()
            )
            for utilizer_name in self.utilizers:
                if (
                    utilizer_name in configured_predictors
                    and utilizer_name not in ran_predictors
                ):
                    self.logger.info(
                        f"Skipping BaseUtilizer: {utilizer_name} (predictor not run)"
                    )
                    continue
                utilizer = self.utilizers[utilizer_name]
                self.logger.info(f"Running BaseUtilizer: {utilizer_name}")
                data = _run_component(
                    f"Face utilizer {utilizer_name!r}", lambda: utilizer.run(data)
                )
        else:
            if "save" in self.utilizers:
                _run_component(
                    "Face utilizer 'save'", lambda: self.utilizers["save"].run(data)
                )

        if not include_tensors:
            self.logger.debug(
                "Removing tensors from response as include_tensors is False"
            )
            data.reset_tensors()

        result = AnalysisResult.from_image_data(data, include_tensors=include_tensors)
        self.logger.debug(
            "Returning analysis result",
            extra={"face_count": len(result.faces), "version": result.version},
        )
        return result

    def run_legacy(
        self,
        *args,
        return_img_data: bool = False,
        **kwargs,
    ) -> Union[Response, ImageData]:
        """Run the canonical pipeline and adapt its result to the v0.x return union."""
        warnings.warn(
            "FaceAnalyzer.run_legacy is a v1.x compatibility adapter; migrate to run().",
            DeprecationWarning,
            stacklevel=2,
        )
        result = self.run(*args, **kwargs)
        if not return_img_data:
            return Response(faces=result.faces, version=result.version)

        return ImageData(
            path_input=result.path_input,
            path_output=result.path_output,
            img=result.image if result.image is not None else torch.tensor([]),
            tensor=result.tensor if result.tensor is not None else torch.tensor([]),
            dims=result.dimensions,
            det=result.detection if result.detection is not None else Detection(),
            faces=result.faces,
            version=result.version,
            warnings=list(result.warnings),
        )

    def _read_input(
        self,
        image_source: Union[
            str, os.PathLike, torch.Tensor, np.ndarray, bytes, Image.Image
        ],
        fix_img_size: bool,
        *,
        input_policy: str = "coerce",
        input_spec: Optional[InputSpec] = None,
    ) -> ImageData:
        """Delegate every source type to the configured public reader entry point."""
        run = self.reader.run
        if self.__dict__.get("_reader_signature_owner") is self.reader:
            parameters = self.__dict__.get("_reader_signature_parameters")
        else:
            try:
                parameters = inspect.signature(run).parameters
            except (TypeError, ValueError) as exc:
                raise ConfigurationError(
                    "Configured reader.run must expose an inspectable public signature."
                ) from exc
            self._reader_signature_owner = self.reader
            self._reader_signature_parameters = parameters

        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        reader_kwargs = {}
        if "fix_img_size" in parameters or accepts_kwargs:
            reader_kwargs["fix_img_size"] = fix_img_size

        supports_policy = "input_policy" in parameters or accepts_kwargs
        supports_spec = "input_spec" in parameters or accepts_kwargs
        if not supports_policy and input_policy != "coerce":
            raise ConfigurationError(
                "Configured reader uses the legacy protocol and cannot honor strict mode."
            )
        if not supports_spec and input_spec is not None:
            raise ConfigurationError(
                "Configured reader uses the legacy protocol and cannot honor InputSpec."
            )
        if supports_policy:
            reader_kwargs["input_policy"] = input_policy
        if supports_spec:
            reader_kwargs["input_spec"] = input_spec

        data = run(image_source, **reader_kwargs)
        if not isinstance(data, ImageData):
            raise ConfigurationError(
                "Configured reader.run must return facetorch.datastruct.ImageData."
            )
        if not supports_policy or not supports_spec:
            message = (
                "Configured reader uses the deprecated v0.x protocol; add keyword-only "
                "input_policy and input_spec parameters."
            )
            warnings.warn(message, DeprecationWarning, stacklevel=3)
            data.warnings.append(message)

        self._validate_reader_output(data)
        return data

    @staticmethod
    def _validate_reader_output(data: ImageData) -> None:
        tensor = data.tensor
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
            raise ConfigurationError("Reader output tensor must have BCHW rank 4.")
        if tensor.shape[0] != 1:
            raise InputError(
                "Batched image input is not supported. Expected B=1, "
                f"got B={tensor.shape[0]}."
            )
        if tensor.shape[1] != 3:
            raise ConfigurationError(
                "Reader output must use the canonical three-channel RGB representation."
            )
        if tensor.dtype != torch.float32:
            raise ConfigurationError("Reader output tensor must use float32 values.")
        if not getattr(data, "_facetorch_canonical", False):
            if not torch.isfinite(tensor).all():
                raise InputError("Reader output contains NaN or Inf values.")
            if tensor.numel() and (
                float(tensor.min()) < 0.0 or float(tensor.max()) > 255.0
            ):
                raise ConfigurationError("Reader output values must stay within 0..255.")
