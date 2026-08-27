import io
import json
import logging
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import numpy as np
import pytest
import torch
from PIL import Image

import facetorch
from facetorch.analyzer.core import FaceAnalyzer
from facetorch.analyzer.detector.core import FaceDetector
from facetorch.analyzer.detector.post import PostRetFace
from facetorch.analyzer.detector.pre import DetectorPreProcessor
from facetorch.analyzer.reader import core as reader_core
from facetorch.analyzer.reader import TensorReader, UniversalReader, URLReader
from facetorch.analyzer.utilizer.save import ImageSaver
from facetorch.datastruct import (
    Detection,
    Dimensions,
    Face,
    ImageData,
    Location,
    Prediction,
)
from facetorch.exceptions import ConfigurationError, InputCoercionWarning, InputError
from facetorch.input import InputSpec, canonicalize_image_tensor
from facetorch.logger import LoggerJsonFile
from torchvision import transforms

pytestmark = pytest.mark.release_blocker


def _rgb_array(height=8, width=9):
    values = np.arange(height * width * 3, dtype=np.uint8)
    return values.reshape(height, width, 3)


def _png_bytes(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def _minimal_analyzer(*, reader=None, detector=None, unifier=None, predictors=None):
    analyzer = object.__new__(FaceAnalyzer)
    analyzer.logger = logging.getLogger(f"facetorch-b02-{uuid4()}")
    analyzer.reader = reader or TensorReader(None, torch.device("cpu"), False)
    analyzer.detector = detector or SimpleNamespace(
        run=lambda _data: pytest.fail("detector should not run")
    )
    analyzer.unifier = unifier
    analyzer.predictors = predictors or {}
    analyzer.utilizers = {}
    return analyzer


@pytest.mark.parametrize(
    ("source", "strict_spec"),
    [
        (
            torch.linspace(0, 1, 3 * 8 * 9).reshape(3, 8, 9),
            InputSpec(layout="CHW", value_range="0_1", color_space="RGB"),
        ),
        (
            torch.linspace(0, 255, 3 * 8 * 9).reshape(3, 8, 9),
            InputSpec(layout="CHW", value_range="0_255", color_space="RGB"),
        ),
        (
            np.linspace(0, 1, 8 * 9 * 3, dtype=np.float32).reshape(8, 9, 3),
            InputSpec(layout="HWC", value_range="0_1", color_space="RGB"),
        ),
        (
            torch.arange(8 * 9, dtype=torch.uint8).reshape(8, 9),
            InputSpec(layout="HW", value_range="0_255", color_space="GRAY"),
        ),
        (
            np.full((8, 9, 4), (5, 10, 15, 20), dtype=np.uint8),
            InputSpec(
                layout="HWC",
                value_range="0_255",
                color_space="RGBA",
                alpha_mode="drop",
            ),
        ),
        (
            torch.from_numpy(_rgb_array()).permute(2, 0, 1).unsqueeze(0),
            InputSpec(layout="BCHW", value_range="0_255", color_space="RGB"),
        ),
        (
            _rgb_array()[None, ...],
            InputSpec(layout="BHWC", value_range="0_255", color_space="RGB"),
        ),
    ],
)
def test_coerce_and_explicit_strict_share_one_canonical_pipeline(source, strict_spec):
    reader = UniversalReader(None, torch.device("cpu"), False)
    expects_warning = strict_spec.color_space != "RGB" or (
        isinstance(source, (torch.Tensor, np.ndarray))
        and (source.dtype == torch.float32 or source.dtype == np.float32)
    )
    warning_context = (
        pytest.warns(InputCoercionWarning) if expects_warning else nullcontext()
    )
    with warning_context:
        coerced = reader.run(source, input_policy="coerce")
    strict = reader.run(source, input_policy="strict", input_spec=strict_spec)

    assert coerced.tensor.dtype == torch.float32
    assert coerced.tensor.shape[0:2] == (1, 3)
    assert torch.equal(coerced.tensor, strict.tensor)
    assert torch.equal(coerced.img, strict.img)


def test_path_bytes_and_pil_strict_inputs_are_equivalent(tmp_path):
    array = _rgb_array()
    path = tmp_path / "probe.png"
    Image.fromarray(array).save(path)
    payload = _png_bytes(array)
    reader = UniversalReader(None, torch.device("cpu"), False)

    with Image.open(io.BytesIO(payload)) as pil_image:
        results = [
            reader.run(path, input_policy="strict"),
            reader.run(payload, input_policy="strict"),
            reader.run(pil_image, input_policy="strict"),
        ]
        assert pil_image.getpixel((0, 0)) == tuple(array[0, 0])

    assert all(torch.equal(results[0].tensor, result.tensor) for result in results[1:])
    assert results[0].path_input == str(path)


@pytest.mark.parametrize(
    "source",
    [
        torch.ones((3, 8, 9), dtype=torch.float32),
        torch.ones((8, 9), dtype=torch.uint8),
        torch.ones((4, 8, 9), dtype=torch.uint8),
    ],
)
def test_strict_rejects_undeclared_conversions(source):
    reader = TensorReader(None, torch.device("cpu"), False)
    with pytest.raises(InputError):
        reader.run(source, input_policy="strict")


def test_torch_hwc_requires_an_explicit_layout():
    reader = TensorReader(None, torch.device("cpu"), False)
    source = torch.zeros((8, 9, 3), dtype=torch.uint8)
    with pytest.raises(InputError, match='InputSpec\\(layout="HWC"\\)'):
        reader.run(source)

    result = reader.run(source, input_spec=InputSpec(layout="HWC"))
    assert result.tensor.shape == (1, 3, 8, 9)


@pytest.mark.parametrize(
    "source",
    [
        torch.zeros((2, 3, 8, 9), dtype=torch.uint8),
        np.zeros((2, 8, 9, 3), dtype=np.uint8),
    ],
)
def test_multi_image_batches_are_rejected(source):
    reader = UniversalReader(None, torch.device("cpu"), False)
    with pytest.raises(InputError, match="B=1"):
        reader.run(source)


@pytest.mark.parametrize(
    "source",
    [
        torch.full((3, 8, 9), -0.01),
        torch.full((3, 8, 9), 256.0),
        torch.full((3, 8, 9), float("nan")),
        torch.full((3, 8, 9), float("inf")),
    ],
)
def test_invalid_numeric_inputs_are_rejected(source):
    reader = TensorReader(None, torch.device("cpu"), False)
    with pytest.raises(InputError):
        reader.run(source)


@pytest.mark.parametrize("width", [1, 3, 4])
def test_small_spatial_dimensions_do_not_change_source_layout_conventions(width):
    reader = UniversalReader(None, torch.device("cpu"), False)
    torch_chw = torch.zeros((3, 4, width), dtype=torch.uint8)
    numpy_hwc = np.zeros((4, width, 3), dtype=np.uint8)

    assert reader.run(torch_chw).tensor.shape == (1, 3, 4, width)
    assert reader.run(numpy_hwc).tensor.shape == (1, 3, 4, width)


def test_contradictory_input_spec_is_actionable():
    reader = TensorReader(None, torch.device("cpu"), False)
    with pytest.raises(InputError, match="alpha_mode"):
        reader.run(
            torch.zeros((3, 8, 9), dtype=torch.uint8),
            input_spec=InputSpec(alpha_mode="drop"),
        )
    with pytest.raises(InputError, match="input_policy"):
        reader.run(torch.zeros((3, 8, 9), dtype=torch.uint8), input_policy="guess")


def test_every_source_type_reaches_the_configured_reader(tmp_path):
    array = _rgb_array()
    path = tmp_path / "probe.png"
    Image.fromarray(array).save(path)

    class RecordingReader:
        def __init__(self):
            self.delegate = UniversalReader(None, torch.device("cpu"), False)
            self.sources = []

        def run(
            self,
            image_source,
            fix_img_size=False,
            *,
            input_policy="coerce",
            input_spec=None,
        ):
            self.sources.append(type(image_source))
            return self.delegate.run(
                image_source,
                fix_img_size,
                input_policy=input_policy,
                input_spec=input_spec,
            )

    recording = RecordingReader()
    analyzer = _minimal_analyzer(reader=recording)
    sources = [
        path,
        _png_bytes(array),
        Image.fromarray(array),
        array,
        torch.from_numpy(array).permute(2, 0, 1),
    ]
    try:
        for source in sources:
            analyzer._read_input(source, False)
    finally:
        sources[2].close()

    assert recording.sources == [type(source) for source in sources]


@pytest.mark.parametrize("source_kind", ["tensor", "numpy", "bytes", "pil", "path"])
def test_analysis_result_is_stable_across_source_types(source_kind, tmp_path):
    array = _rgb_array()
    path = tmp_path / "probe.png"
    Image.fromarray(array).save(path)
    source = {
        "tensor": torch.from_numpy(array).permute(2, 0, 1),
        "numpy": array,
        "bytes": _png_bytes(array),
        "pil": Image.fromarray(array),
        "path": path,
    }[source_kind]
    analyzer = _minimal_analyzer(
        reader=UniversalReader(None, torch.device("cpu"), False)
    )
    try:
        result = analyzer.run(
            image_source=source,
            skip_detector=True,
            include_tensors=True,
        )
    finally:
        if isinstance(source, Image.Image):
            source.close()

    assert isinstance(result, facetorch.AnalysisResult)
    assert result.tensor.shape == (1, 3, 8, 9)
    assert result.image.shape == (3, 8, 9)
    assert len(result.faces) == 1


def test_result_retention_and_legacy_adapter_are_explicit():
    analyzer = _minimal_analyzer()
    source = torch.zeros((3, 8, 9), dtype=torch.uint8)

    result = analyzer.run(image_source=source, skip_detector=True)
    assert isinstance(result, facetorch.AnalysisResult)
    assert result.tensor is None and result.image is None and result.detection is None

    with pytest.warns(DeprecationWarning, match="no longer changes"):
        same_type = analyzer.run(
            image_source=source,
            skip_detector=True,
            return_img_data=True,
        )
    assert isinstance(same_type, facetorch.AnalysisResult)

    with pytest.warns(DeprecationWarning, match="compatibility adapter"):
        legacy = analyzer.run_legacy(image_source=source, skip_detector=True)
    assert isinstance(legacy, facetorch.Response)

    with pytest.warns(DeprecationWarning, match="compatibility adapter"):
        legacy_data = analyzer.run_legacy(
            image_source=source,
            skip_detector=True,
            include_tensors=True,
            return_img_data=True,
        )
    assert isinstance(legacy_data, facetorch.ImageData)
    assert legacy_data.tensor.shape == (1, 3, 8, 9)


@pytest.mark.parametrize(
    ("return_img_data", "expected_type"),
    [(False, facetorch.Response), (True, facetorch.ImageData)],
)
def test_legacy_adapter_preserves_the_v0_positional_signature(
    return_img_data,
    expected_type,
):
    analyzer = _minimal_analyzer()
    source = torch.zeros((3, 8, 9), dtype=torch.uint8)

    with pytest.warns(DeprecationWarning, match="compatibility adapter"):
        result = analyzer.run_legacy(
            source,
            None,
            8,
            False,
            return_img_data,
            True,
            skip_detector=True,
        )

    assert isinstance(result, expected_type)
    if return_img_data:
        assert result.tensor.shape == (1, 3, 8, 9)


class _FaceDetectorStub:
    def __init__(self, count):
        self.count = count

    def run(self, data):
        height, width = data.tensor.shape[-2:]
        data.faces = [
            Face(
                indx=index,
                loc=Location(x1=0, y1=0, x2=width, y2=height),
                dims=Dimensions(height=height, width=width),
                tensor=torch.full_like(data.tensor[0], index),
                ratio=1.0,
            )
            for index in range(self.count)
        ]
        return data


class _BatchRecorder:
    def __init__(self):
        self.sizes = []

    def run(self, batch):
        self.sizes.append(len(batch))
        return [Prediction(label=str(int(face[0, 0, 0]))) for face in batch]


class _IdentityUnifier:
    def run(self, data):
        return data


@pytest.mark.parametrize("face_count", [0, 1, 2, 4, 8, 9])
def test_within_image_face_batching_preserves_order(face_count):
    predictor = _BatchRecorder()
    analyzer = _minimal_analyzer(
        detector=_FaceDetectorStub(face_count),
        unifier=_IdentityUnifier(),
        predictors={"probe": predictor},
    )
    result = analyzer.run(
        image_source=torch.zeros((3, 8, 9), dtype=torch.uint8),
        face_batch_size=4,
    )

    expected_sizes = [4] * (face_count // 4)
    if face_count % 4:
        expected_sizes.append(face_count % 4)
    assert predictor.sizes == expected_sizes
    assert [face.preds["probe"].label for face in result.faces] == [
        str(index) for index in range(face_count)
    ]


def test_shipped_predictor_batch_limit_caps_requested_chunks():
    predictor = _BatchRecorder()
    predictor.max_batch_size = 64
    analyzer = _minimal_analyzer(
        detector=_FaceDetectorStub(65),
        unifier=_IdentityUnifier(),
        predictors={"probe": predictor},
    )

    result = analyzer.run(
        image_source=torch.zeros((3, 8, 9), dtype=torch.uint8),
        face_batch_size=65,
    )

    assert predictor.sizes == [64, 1]
    assert len(result.faces) == 65


@pytest.mark.parametrize("invalid_limit", [0, True, 1.5, "64"])
def test_invalid_predictor_batch_limit_fails_before_inference(invalid_limit):
    predictor = _BatchRecorder()
    predictor.max_batch_size = invalid_limit
    analyzer = _minimal_analyzer(
        detector=_FaceDetectorStub(1),
        unifier=_IdentityUnifier(),
        predictors={"probe": predictor},
    )

    with pytest.raises(ConfigurationError, match="max_batch_size"):
        analyzer.run(
            image_source=torch.zeros((3, 8, 9), dtype=torch.uint8),
            face_batch_size=8,
        )

    assert predictor.sizes == []


def test_default_result_preserves_va_and_au_metadata_without_tensors():
    class MetadataPredictor:
        def __init__(self, prediction):
            self.prediction = prediction

        def run(self, batch):
            return [
                Prediction(
                    label=self.prediction.label,
                    logits=self.prediction.logits.clone(),
                    other={
                        key: value.copy() if isinstance(value, dict) else value
                        for key, value in self.prediction.other.items()
                    },
                )
                for _ in batch
            ]

    analyzer = _minimal_analyzer(
        detector=_FaceDetectorStub(1),
        unifier=_IdentityUnifier(),
        predictors={
            "va": MetadataPredictor(
                Prediction(
                    label="other",
                    logits=torch.tensor([0.25, -0.5]),
                    other={
                        "valence": 0.25,
                        "arousal": -0.5,
                        "diagnostic": {
                            "source": "va",
                            "tensor": torch.ones(1),
                            "values": [
                                1.0,
                                torch.ones(1),
                                {"name": "nested", "tensor": torch.ones(1)},
                            ],
                            "pair": ("kept", torch.ones(1)),
                        },
                    },
                )
            ),
            "au": MetadataPredictor(
                Prediction(
                    label="AU1",
                    logits=torch.tensor([0.8, 0.7]),
                    other={"multi": ["AU1", "AU2"]},
                )
            ),
        },
    )

    result = analyzer.run(
        image_source=torch.zeros((3, 8, 9), dtype=torch.uint8),
    )

    assert result.faces[0].preds["va"].logits.numel() == 0
    assert result.faces[0].preds["va"].other == {
        "valence": 0.25,
        "arousal": -0.5,
        "diagnostic": {
            "source": "va",
            "values": [1.0, {"name": "nested"}],
            "pair": ("kept",),
        },
    }
    assert result.faces[0].preds["au"].logits.numel() == 0
    assert result.faces[0].preds["au"].other == {"multi": ["AU1", "AU2"]}


def test_selected_predictor_requires_a_unifier_with_skip_detector():
    predictor = _BatchRecorder()
    analyzer = _minimal_analyzer(predictors={"probe": predictor})

    with pytest.raises(ConfigurationError, match="requires a face unifier"):
        analyzer.run(
            image_source=torch.zeros((3, 8, 9), dtype=torch.uint8),
            skip_detector=True,
        )

    assert predictor.sizes == []


def test_predictor_must_return_one_prediction_per_input_face():
    class ShortPredictor:
        def run(self, batch):
            return [Prediction(label="short") for _ in batch[:-1]]

    analyzer = _minimal_analyzer(
        detector=_FaceDetectorStub(2),
        unifier=_IdentityUnifier(),
        predictors={"probe": ShortPredictor()},
    )

    with pytest.raises(
        facetorch.InferenceError,
        match=r"returned 1 prediction\(s\) for 2 input face\(s\)",
    ):
        analyzer.run(
            image_source=torch.zeros((3, 8, 9), dtype=torch.uint8),
            face_batch_size=2,
        )


def test_face_batch_alias_warns_and_conflicts_fail():
    predictor = _BatchRecorder()
    analyzer = _minimal_analyzer(
        detector=_FaceDetectorStub(3),
        unifier=_IdentityUnifier(),
        predictors={"probe": predictor},
    )
    with pytest.warns(DeprecationWarning, match="batch_size"):
        result = analyzer.run(
            image_source=torch.zeros((3, 8, 9), dtype=torch.uint8),
            batch_size=2,
        )
    assert predictor.sizes == [2, 1]
    assert len(result.faces) == 3

    with pytest.raises(ConfigurationError, match="only face_batch_size"):
        analyzer.run(
            image_source=torch.zeros((3, 8, 9), dtype=torch.uint8),
            face_batch_size=2,
            batch_size=2,
        )


@pytest.mark.parametrize("invalid_size", [0, -1, 1.5, True])
def test_invalid_face_batch_size_is_actionable(invalid_size):
    analyzer = _minimal_analyzer()
    with pytest.raises(ConfigurationError, match="face_batch_size"):
        analyzer.run(
            image_source=torch.zeros((3, 8, 9), dtype=torch.uint8),
            face_batch_size=invalid_size,
            skip_detector=True,
        )


def test_detector_restores_raw_tensor_and_clamps_public_geometry():
    class PadPreprocessor:
        def run(self, data):
            data.tensor = torch.nn.functional.pad(data.tensor, (0, 2, 0, 2))
            data.set_dims()
            return data

    class GeometryPostprocessor:
        def run(self, data, _logits):
            data.det = Detection(
                dets=torch.tensor([[-2.0, -1.0, 12.0, 10.0, 0.9]]),
                boxes=torch.tensor([[-2.0, -1.0, 12.0, 10.0]]),
                landmarks=torch.tensor([[-1.0, -1.0, 12.0, 10.0]]),
            )
            data.faces = [
                Face(
                    indx=0,
                    loc=Location(x1=1, y1=2, x2=6, y2=7),
                    tensor=data.tensor[0, :, 2:7, 1:6],
                )
            ]
            return data

    detector = object.__new__(FaceDetector)
    detector.device = torch.device("cpu")
    detector.model = torch.nn.Identity()
    detector.preprocessor = PadPreprocessor()
    detector.postprocessor = GeometryPostprocessor()
    raw = torch.arange(3 * 8 * 9, dtype=torch.float32).reshape(1, 3, 8, 9)
    data = ImageData(tensor=raw.clone())
    data.set_dims()

    result = detector.run(data)

    assert torch.equal(result.tensor, raw)
    assert result.dims == Dimensions(height=8, width=9)
    assert result.faces[0].loc == Location(x1=1, y1=2, x2=6, y2=7)
    assert result.faces[0].dims == Dimensions(height=5, width=5)
    assert torch.equal(result.faces[0].tensor, raw[0, :, 2:7, 1:6])
    assert result.det.dets[0, :4].tolist() == [0.0, 0.0, 9.0, 8.0]
    assert result.det.boxes[0].tolist() == [0.0, 0.0, 9.0, 8.0]
    assert result.det.landmarks[0].tolist() == [0.0, 0.0, 9.0, 8.0]


def test_detector_reuses_raw_tensor_for_declared_out_of_place_preprocessor():
    class EmptyPostprocessor:
        def run(self, data, _logits):
            return data

    detector = object.__new__(FaceDetector)
    detector.device = torch.device("cpu")
    detector.model = torch.nn.Identity()
    detector.preprocessor = DetectorPreProcessor(
        transform=transforms.Compose(
            [transforms.Normalize(mean=[1.0, 1.0, 1.0], std=[1.0, 1.0, 1.0])]
        ),
        device=torch.device("cpu"),
        optimize_transform=False,
        reverse_colors=False,
    )
    detector.postprocessor = EmptyPostprocessor()
    raw = torch.zeros((1, 3, 32, 32))
    data = ImageData(tensor=raw)
    data.set_dims()

    result = detector.run(data)

    assert result.tensor is raw
    assert torch.count_nonzero(result.tensor) == 0


def test_default_detector_bounds_large_input_and_restores_source_coordinates():
    observed = {}

    class ShapeRecorder(torch.nn.Module):
        def forward(self, tensor):
            observed["shape"] = tuple(tensor.shape)
            return tensor

    class DetectorCoordinatePostprocessor:
        def run(self, data, _logits):
            data.det = Detection(
                dets=torch.tensor([[10.0, 20.0, 50.0, 100.0, 0.9]]),
                boxes=torch.tensor([[10.0, 20.0, 50.0, 100.0]]),
                landmarks=torch.tensor(
                    [[10.0, 20.0, 20.0, 30.0, 30.0, 40.0, 40.0, 50.0, 50.0, 60.0]]
                ),
            )
            data.faces = [
                Face(
                    indx=0,
                    loc=Location(x1=10, y1=20, x2=50, y2=100),
                    tensor=data.tensor[0, :, 20:100, 10:50],
                )
            ]
            return data

    detector = object.__new__(FaceDetector)
    detector.device = torch.device("cpu")
    detector.model = ShapeRecorder()
    detector.preprocessor = DetectorPreProcessor(
        transform=transforms.Compose([]),
        device=torch.device("cpu"),
        optimize_transform=False,
        reverse_colors=False,
    )
    detector.postprocessor = DetectorCoordinatePostprocessor()
    raw = torch.zeros((1, 3, 4096, 200))
    data = ImageData(tensor=raw)
    data.set_dims()

    result = detector.run(data)

    assert observed["shape"] == (1, 3, 2048, 128)
    assert result.tensor is raw
    assert result.dims == Dimensions(height=4096, width=200)
    assert result.det.dets[0, :4].tolist() == [20.0, 40.0, 100.0, 200.0]
    assert result.det.boxes[0].tolist() == [20.0, 40.0, 100.0, 200.0]
    assert result.det.landmarks[0].tolist() == [
        20.0,
        40.0,
        40.0,
        60.0,
        60.0,
        80.0,
        80.0,
        100.0,
        100.0,
        120.0,
    ]
    assert result.faces[0].loc == Location(x1=20, y1=40, x2=100, y2=200)
    assert torch.equal(result.faces[0].tensor, raw[0, :, 40:200, 20:100])


def test_detector_defensively_clones_for_custom_preprocessor():
    class InPlacePreprocessor:
        def run(self, data):
            data.tensor.add_(1)
            return data

    class EmptyPostprocessor:
        def run(self, data, _logits):
            return data

    detector = object.__new__(FaceDetector)
    detector.device = torch.device("cpu")
    detector.model = torch.nn.Identity()
    detector.preprocessor = InPlacePreprocessor()
    detector.postprocessor = EmptyPostprocessor()
    data = ImageData(tensor=torch.zeros((1, 3, 8, 9)))
    data.set_dims()

    result = detector.run(data)

    assert torch.count_nonzero(result.tensor) == 0


def test_custom_public_face_extractor_is_used_with_and_without_padding():
    class Preprocessor:
        def __init__(self, padding):
            self.padding = padding

        def run(self, data):
            data.tensor = torch.nn.functional.pad(
                data.tensor, (0, self.padding, 0, self.padding)
            )
            data.set_dims()
            return data

    class PublicExtractor:
        def __init__(self):
            self.extract_calls = 0

        def run(self, data, _logits):
            data.det = Detection(dets=torch.tensor([[1.0, 2.0, 6.0, 7.0, 0.9]]))
            return data

        def extract_faces(self, data):
            self.extract_calls += 1
            loc = Location(x1=1, y1=2, x2=6, y2=7)
            crop = data.tensor[0, :, 2:7, 1:6]
            data.faces = [
                Face(
                    indx=0,
                    loc=loc,
                    dims=Dimensions(height=5, width=5),
                    tensor=crop,
                )
            ]
            return data

    def run_with_padding(padding):
        postprocessor = PublicExtractor()
        detector = object.__new__(FaceDetector)
        detector.device = torch.device("cpu")
        detector.model = torch.nn.Identity()
        detector.preprocessor = Preprocessor(padding)
        detector.postprocessor = postprocessor
        data = ImageData(tensor=torch.arange(216).reshape(1, 3, 8, 9).float())
        data.set_dims()
        result = detector.run(data)
        assert postprocessor.extract_calls == 1
        return result

    unpadded = run_with_padding(0)
    padded = run_with_padding(2)
    assert padded.faces[0].loc == unpadded.faces[0].loc
    assert torch.equal(padded.faces[0].tensor, unpadded.faces[0].tensor)


def test_builtin_face_extraction_expands_squares_then_clamps():
    postprocessor = PostRetFace(
        transform=None,
        device=torch.device("cpu"),
        optimize_transform=False,
        confidence_threshold=0.0,
        top_k=10,
        nms_threshold=0.4,
        keep_top_k=10,
        score_threshold=0.0,
        prior_box=SimpleNamespace(),
        variance=[0.1, 0.2],
        expand_box_ratio=0.5,
    )
    data = ImageData(
        tensor=torch.zeros((1, 3, 8, 9)),
        det=Detection(
            dets=torch.tensor(
                [
                    [-2.0, -1.0, 4.0, 5.0, 0.9],
                    [7.0, 5.0, 12.0, 10.0, 0.8],
                    [11.0, 1.0, 14.0, 4.0, 0.7],
                ]
            )
        ),
    )
    data.set_dims()

    result = postprocessor.extract_faces(data)

    assert len(result.faces) == 2
    for face in result.faces:
        assert 0 <= face.loc.x1 < face.loc.x2 <= 9
        assert 0 <= face.loc.y1 < face.loc.y2 <= 8
        assert face.dims.height == face.dims.width
        assert face.tensor.shape[-2:] == (face.dims.height, face.dims.width)


def test_retinaface_boxes_landmarks_and_crops_share_image_coordinates():
    class OnePrior:
        def forward(self, _dims):
            return torch.tensor([[0.5, 0.5, 0.5, 0.5]])

    postprocessor = PostRetFace(
        transform=None,
        device=torch.device("cpu"),
        optimize_transform=False,
        confidence_threshold=0.1,
        top_k=10,
        nms_threshold=0.4,
        keep_top_k=10,
        score_threshold=0.1,
        prior_box=OnePrior(),
        variance=[0.1, 0.2],
    )
    data = ImageData(tensor=torch.zeros((1, 3, 8, 10)))
    data.set_dims()
    logits = (
        torch.zeros((1, 1, 4)),
        torch.tensor([[[0.1, 0.9]]]),
        torch.zeros((1, 1, 10)),
    )

    result = postprocessor.extract_faces(postprocessor.run(data, logits))

    assert result.det.dets.shape == (1, 5)
    assert torch.equal(result.det.boxes, result.det.dets[:, :4])
    assert result.det.landmarks.shape == (1, 10)
    assert result.det.landmarks[0].tolist() == [5.0, 4.0] * 5
    assert len(result.faces) == 1
    face = result.faces[0]
    assert face.loc == Location(x1=2, y1=1, x2=8, y2=7)
    assert face.tensor.shape[-2:] == (6, 6)
    dets_before = result.det.dets.clone()
    result.det.boxes.zero_()
    assert torch.equal(result.det.dets, dets_before)


def test_retinaface_postprocessing_follows_incoming_tensor_device():
    class OnePrior:
        def forward(self, _dims):
            return torch.tensor([[0.5, 0.5, 0.5, 0.5]])

    postprocessor = PostRetFace(
        transform=None,
        device=torch.device("meta"),
        optimize_transform=False,
        confidence_threshold=0.1,
        top_k=10,
        nms_threshold=0.4,
        keep_top_k=10,
        score_threshold=0.1,
        prior_box=OnePrior(),
        variance=[0.1, 0.2],
    )
    data = ImageData(tensor=torch.zeros((1, 3, 8, 10)))
    data.set_dims()
    logits = (
        torch.zeros((1, 1, 4)),
        torch.tensor([[[0.1, 0.9]]]),
        torch.zeros((1, 1, 10)),
    )

    result = postprocessor.run(data, logits)

    assert result.det.dets.device.type == "cpu"
    assert result.det.boxes.device.type == "cpu"
    assert result.det.landmarks.device.type == "cpu"


class _FakeConnection:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _FakeResponse:
    def __init__(self, body=b"", status_code=200, headers=None):
        self.body = body
        self.status = status_code
        self.headers = headers or {}
        self.closed = False
        self._offset = 0

    def read(self, chunk_size):
        chunk = self.body[self._offset : self._offset + chunk_size]
        self._offset += len(chunk)
        return chunk

    def close(self):
        self.closed = True


def _fake_url_open(*responses):
    pending = list(responses)

    def open_response(*_args, **_kwargs):
        return _FakeConnection(), pending.pop(0)

    return open_response


def test_url_reader_is_explicit_bounded_and_removes_query_metadata(monkeypatch):
    response = _FakeResponse(_png_bytes(_rgb_array()))
    monkeypatch.setattr(
        "facetorch.analyzer.reader.core.socket.getaddrinfo",
        lambda _host, port, **_kwargs: [(2, 1, 6, "", ("93.184.216.34", port))],
    )
    monkeypatch.setattr(
        "facetorch.analyzer.reader.core._open_pinned_response",
        _fake_url_open(response),
    )
    reader = URLReader(
        None,
        torch.device("cpu"),
        False,
        allowed_schemes=("https",),
        timeout=1,
        max_redirects=0,
        max_bytes=1024 * 1024,
    )

    result = reader.run("https://example.test/image.png?secret=value")

    assert result.tensor.shape == (1, 3, 8, 9)
    assert result.path_input == "https://example.test/image.png"
    assert response.closed


def test_url_reader_shares_one_deadline_across_redirects(monkeypatch):
    now = [100.0]
    observed_timeouts = []
    responses = [
        _FakeResponse(status_code=302, headers={"Location": "/final.png"}),
        _FakeResponse(_png_bytes(_rgb_array())),
    ]

    monkeypatch.setattr("facetorch.analyzer.reader.core.time.monotonic", lambda: now[0])
    monkeypatch.setattr(
        "facetorch.analyzer.reader.core.socket.getaddrinfo",
        lambda _host, port, **_kwargs: [(2, 1, 6, "", ("93.184.216.34", port))],
    )

    def open_response(_parsed, _address, timeout):
        observed_timeouts.append(timeout)
        if len(observed_timeouts) == 1:
            now[0] += 0.4
        return _FakeConnection(), responses.pop(0)

    monkeypatch.setattr(
        "facetorch.analyzer.reader.core._open_pinned_response", open_response
    )
    reader = URLReader(
        None,
        torch.device("cpu"),
        False,
        timeout=1.0,
        max_redirects=1,
    )

    result = reader.run("https://example.test/start.png")

    assert result.tensor.shape == (1, 3, 8, 9)
    assert observed_timeouts == pytest.approx([1.0, 0.6])


def test_url_reader_stops_slow_stream_at_total_deadline(monkeypatch):
    now = [0.0]
    connection = _FakeConnection()

    class _SlowResponse(_FakeResponse):
        def read(self, _chunk_size):
            now[0] += 0.4
            return b"x"

    response = _SlowResponse()
    monkeypatch.setattr("facetorch.analyzer.reader.core.time.monotonic", lambda: now[0])
    monkeypatch.setattr(
        "facetorch.analyzer.reader.core.socket.getaddrinfo",
        lambda _host, port, **_kwargs: [(2, 1, 6, "", ("93.184.216.34", port))],
    )
    monkeypatch.setattr(
        "facetorch.analyzer.reader.core._open_pinned_response",
        lambda *_args, **_kwargs: (connection, response),
    )
    reader = URLReader(
        None,
        torch.device("cpu"),
        False,
        timeout=1.0,
        max_bytes=1024,
    )

    with pytest.raises(InputError, match="timed out"):
        reader.run("https://example.test/image.png")

    assert response.closed
    assert connection.closed


def test_url_reader_closes_response_when_connect_exhausts_deadline(monkeypatch):
    now = [0.0]
    connection = _FakeConnection()
    response = _FakeResponse()
    monkeypatch.setattr("facetorch.analyzer.reader.core.time.monotonic", lambda: now[0])
    monkeypatch.setattr(
        "facetorch.analyzer.reader.core.socket.getaddrinfo",
        lambda _host, port, **_kwargs: [(2, 1, 6, "", ("93.184.216.34", port))],
    )

    def open_response(*_args, **_kwargs):
        now[0] = 1.0
        return connection, response

    monkeypatch.setattr(
        "facetorch.analyzer.reader.core._open_pinned_response", open_response
    )
    reader = URLReader(None, torch.device("cpu"), False, timeout=1.0)

    with pytest.raises(InputError, match="timed out"):
        reader.run("https://example.test/image.png")

    assert response.closed
    assert connection.closed


def test_url_reader_reports_connect_timeout_at_total_deadline(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("facetorch.analyzer.reader.core.time.monotonic", lambda: now[0])
    monkeypatch.setattr(
        "facetorch.analyzer.reader.core.socket.getaddrinfo",
        lambda _host, port, **_kwargs: [(2, 1, 6, "", ("93.184.216.34", port))],
    )

    def raise_timeout(*_args, **_kwargs):
        now[0] = 1.0
        raise TimeoutError()

    monkeypatch.setattr(
        "facetorch.analyzer.reader.core._open_pinned_response", raise_timeout
    )
    reader = URLReader(None, torch.device("cpu"), False, timeout=1.0)

    with pytest.raises(InputError, match="timed out"):
        reader.run("https://example.test/image.png")


def test_url_reader_wraps_connection_failure(monkeypatch):
    monkeypatch.setattr(
        "facetorch.analyzer.reader.core.socket.getaddrinfo",
        lambda _host, port, **_kwargs: [(2, 1, 6, "", ("93.184.216.34", port))],
    )
    monkeypatch.setattr(
        "facetorch.analyzer.reader.core._open_pinned_response",
        MagicMock(side_effect=OSError("connection refused")),
    )
    reader = URLReader(None, torch.device("cpu"), False, timeout=1.0)

    with pytest.raises(InputError, match="failed or timed out"):
        reader.run("https://example.test/image.png")


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (TimeoutError("read timed out"), "timed out"),
        (OSError("connection reset"), "unsuccessful response"),
    ],
)
def test_url_reader_wraps_body_read_failures(monkeypatch, failure, message):
    connection = _FakeConnection()

    class _FailingResponse(_FakeResponse):
        def read(self, _chunk_size):
            raise failure

    response = _FailingResponse()
    monkeypatch.setattr(
        "facetorch.analyzer.reader.core.socket.getaddrinfo",
        lambda _host, port, **_kwargs: [(2, 1, 6, "", ("93.184.216.34", port))],
    )
    monkeypatch.setattr(
        "facetorch.analyzer.reader.core._open_pinned_response",
        lambda *_args, **_kwargs: (connection, response),
    )
    reader = URLReader(None, torch.device("cpu"), False, timeout=1.0)

    with pytest.raises(InputError, match=message):
        reader.run("https://example.test/image.png")

    assert response.closed
    assert connection.closed


def test_url_reader_rejects_scheme_size_redirects_and_timeouts(monkeypatch):
    monkeypatch.setattr(
        "facetorch.analyzer.reader.core.socket.getaddrinfo",
        lambda _host, port, **_kwargs: [(2, 1, 6, "", ("93.184.216.34", port))],
    )
    reader = URLReader(
        None,
        torch.device("cpu"),
        False,
        allowed_schemes=("https",),
        timeout=1,
        max_redirects=0,
        max_bytes=4,
    )
    with pytest.raises(InputError, match="scheme"):
        reader.run("http://example.test/image.png")

    oversized = _FakeResponse(b"12345", headers={"Content-Length": "5"})
    monkeypatch.setattr(
        "facetorch.analyzer.reader.core._open_pinned_response",
        _fake_url_open(oversized),
    )
    with pytest.raises(InputError, match="size limit"):
        reader.run("https://example.test/image.png")
    assert oversized.closed

    redirect = _FakeResponse(status_code=302, headers={"Location": "/other.png"})
    monkeypatch.setattr(
        "facetorch.analyzer.reader.core._open_pinned_response",
        _fake_url_open(redirect),
    )
    with pytest.raises(InputError, match="redirect limit"):
        reader.run("https://example.test/image.png")
    assert redirect.closed

    def raise_timeout(*_args, **_kwargs):
        raise TimeoutError()

    monkeypatch.setattr(
        "facetorch.analyzer.reader.core._open_pinned_response", raise_timeout
    )
    with pytest.raises(InputError, match="timed out"):
        reader.run("https://example.test/image.png")


def test_url_reader_rejects_private_network_targets_before_request(monkeypatch):
    called = False

    def record_request(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("request must not be attempted")

    monkeypatch.setattr(
        "facetorch.analyzer.reader.core.socket.getaddrinfo",
        lambda _host, port, **_kwargs: [(2, 1, 6, "", ("127.0.0.1", port))],
    )
    monkeypatch.setattr(
        "facetorch.analyzer.reader.core._open_pinned_response", record_request
    )
    reader = URLReader(None, torch.device("cpu"), False)

    with pytest.raises(InputError, match="public network"):
        reader.run("https://internal.example/image.png")
    assert called is False


def test_pinned_https_connection_uses_numeric_ip_and_original_tls_hostname(
    monkeypatch,
):
    observed = {}

    class RawSocket:
        def close(self):
            observed["raw_closed"] = True

    class TLSContext:
        def wrap_socket(self, raw_socket, *, server_hostname):
            observed["raw_socket"] = raw_socket
            observed["server_hostname"] = server_hostname
            return "tls-socket"

    raw_socket = RawSocket()

    def create_connection(target, timeout, source_address):
        observed["target"] = target
        observed["timeout"] = timeout
        observed["source_address"] = source_address
        return raw_socket

    monkeypatch.setattr(reader_core.socket, "create_connection", create_connection)
    connection = reader_core._PinnedHTTPSConnection(
        "example.test",
        "93.184.216.34",
        443,
        2.5,
    )
    connection._context = TLSContext()

    connection.connect()

    assert observed["target"] == ("93.184.216.34", 443)
    assert observed["server_hostname"] == "example.test"
    assert observed["timeout"] == 2.5
    assert connection.sock == "tls-socket"


def test_pinned_request_preserves_host_and_query_without_reresolving(monkeypatch):
    observed = {}
    response = _FakeResponse()

    class Connection:
        def __init__(self, hostname, address, port, timeout):
            observed["constructor"] = (hostname, address, port, timeout)

        def request(self, method, target, headers):
            observed["request"] = (method, target, headers)

        def getresponse(self):
            return response

        def close(self):
            observed["closed"] = True

    monkeypatch.setattr(reader_core, "_PinnedHTTPSConnection", Connection)
    parsed = reader_core.urlsplit("https://example.test:444/image.png?signature=opaque")

    connection, result = reader_core._open_pinned_response(
        parsed,
        "93.184.216.34",
        3.0,
    )

    assert result is response
    assert observed["constructor"] == (
        "example.test",
        "93.184.216.34",
        444,
        3.0,
    )
    assert observed["request"][0:2] == (
        "GET",
        "/image.png?signature=opaque",
    )
    assert observed["request"][2]["Host"] == "example.test:444"
    connection.close()
    assert observed["closed"] is True


def test_valid_bytes_preserve_configuration_errors():
    reader = UniversalReader(None, torch.device("cpu"), False)
    with pytest.raises(ConfigurationError, match="configured reader transform"):
        reader.read_image_from_bytes(_png_bytes(_rgb_array()), fix_img_size=True)


def _clear_logger(name):
    target = logging.getLogger(name)
    for handler in list(target.handlers):
        target.removeHandler(handler)
        handler.close()


def test_logging_is_json_nested_and_idempotent(tmp_path):
    name = f"facetorch-b02-{uuid4()}"
    path = tmp_path / "nested" / "facetorch.log"
    try:
        first = LoggerJsonFile(name=name, level=logging.INFO, path_file=str(path))
        second = LoggerJsonFile(name=name, level=logging.INFO, path_file=str(path))
        assert first.logger is second.logger
        second.logger.info("one-record")
        for handler in second.logger.handlers:
            handler.flush()

        records = [json.loads(line) for line in path.read_text().splitlines()]
        assert [record["message"] for record in records] == ["one-record"]
    finally:
        _clear_logger(name)


def test_file_logging_rotates_at_the_configured_bound(tmp_path):
    name = f"facetorch-b02-rotation-{uuid4()}"
    path = tmp_path / "facetorch.log"
    try:
        configured = LoggerJsonFile(
            name=name,
            level=logging.INFO,
            path_file=str(path),
            max_bytes=160,
            backup_count=1,
        )
        for index in range(20):
            configured.logger.info("rotation-record-%s-%s", index, "x" * 40)
        for handler in configured.logger.handlers:
            handler.flush()
        assert path.is_file()
        assert path.with_name("facetorch.log.1").is_file()
    finally:
        _clear_logger(name)


def test_file_logging_reconfiguration_retires_the_previous_managed_path(tmp_path):
    name = f"facetorch-b02-reconfigure-{uuid4()}"
    first_path = tmp_path / "first.log"
    second_path = tmp_path / "second.log"
    try:
        first = LoggerJsonFile(
            name=name,
            level=logging.INFO,
            path_file=str(first_path),
        )
        first.logger.info("first-only")
        second = LoggerJsonFile(
            name=name,
            level=logging.ERROR,
            path_file=str(second_path),
        )
        second.logger.error("second-only")
        for handler in second.logger.handlers:
            handler.flush()

        assert second.logger.level == logging.ERROR
        assert "first-only" in first_path.read_text(encoding="utf-8")
        assert "second-only" not in first_path.read_text(encoding="utf-8")
        assert "second-only" in second_path.read_text(encoding="utf-8")
        managed_files = [
            handler
            for handler in second.logger.handlers
            if getattr(handler, "_facetorch_file_handler", False)
        ]
        assert [handler.baseFilename for handler in managed_files] == [
            str(second_path.resolve())
        ]
    finally:
        _clear_logger(name)


def test_file_logging_reconfiguration_updates_rotation_in_place(tmp_path):
    name = f"facetorch-b02-rotation-update-{uuid4()}"
    path = tmp_path / "facetorch.log"
    try:
        first = LoggerJsonFile(
            name=name,
            level=logging.INFO,
            path_file=str(path),
            max_bytes=1024,
            backup_count=3,
        )
        original = next(
            handler
            for handler in first.logger.handlers
            if getattr(handler, "_facetorch_file_handler", False)
        )
        second = LoggerJsonFile(
            name=name,
            level=logging.WARNING,
            path_file=str(path),
            max_bytes=256,
            backup_count=1,
        )
        managed_files = [
            handler
            for handler in second.logger.handlers
            if getattr(handler, "_facetorch_file_handler", False)
        ]

        assert managed_files == [original]
        assert original.maxBytes == 256
        assert original.backupCount == 1
        assert original.level == logging.WARNING
        assert second.logger.level == logging.WARNING
    finally:
        _clear_logger(name)


def test_image_saver_supports_nested_output(tmp_path):
    path = tmp_path / "nested" / "face.png"
    data = ImageData(
        path_output=str(path),
        img=torch.zeros((3, 8, 9), dtype=torch.uint8),
    )
    ImageSaver(None, torch.device("cpu"), False).run(data)
    assert path.is_file()


def test_public_exception_categories_are_distinct():
    assert issubclass(facetorch.InputError, facetorch.FacetorchError)
    assert issubclass(facetorch.ConfigurationError, facetorch.FacetorchError)
    assert issubclass(facetorch.CacheLockError, facetorch.FacetorchError)
    assert issubclass(facetorch.ModelCompatibilityError, facetorch.FacetorchError)
    assert issubclass(facetorch.OfflineCacheError, facetorch.FacetorchError)
    assert issubclass(facetorch.ArtifactIntegrityError, facetorch.FacetorchError)
    assert issubclass(facetorch.InferenceError, facetorch.FacetorchError)


def test_inference_failures_use_a_payload_safe_public_error():
    class FailingPredictor:
        def run(self, _batch):
            raise RuntimeError("private tensor payload: [1, 2, 3]")

    analyzer = _minimal_analyzer(
        detector=_FaceDetectorStub(1),
        unifier=_IdentityUnifier(),
        predictors={"probe": FailingPredictor()},
    )
    with pytest.raises(facetorch.InferenceError) as caught:
        analyzer.run(
            image_source=torch.zeros((3, 8, 9), dtype=torch.uint8),
            face_batch_size=1,
        )

    assert "probe" in str(caught.value)
    assert "payload" not in str(caught.value)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"layout": 3}, "must be a string"),
        ({"layout": "sideways"}, "Invalid InputSpec.layout"),
        ({"value_range": "wide"}, "Invalid InputSpec.value_range"),
        ({"color_space": "YUV"}, "Invalid InputSpec.color_space"),
        ({"alpha_mode": "blend"}, "Invalid InputSpec.alpha_mode"),
    ],
)
def test_input_spec_rejects_invalid_declarations(kwargs, message):
    with pytest.raises(InputError, match=message):
        InputSpec(**kwargs)


@pytest.mark.parametrize(
    ("source", "kwargs", "message"),
    [
        (torch.zeros(3), {}, "Unsupported image rank"),
        (
            torch.zeros((3, 4, 5), dtype=torch.uint8),
            {"input_spec": InputSpec(layout="BCHW")},
            "requires rank 4",
        ),
        (
            torch.zeros((3, 0, 5), dtype=torch.uint8),
            {},
            "spatial dimensions must be positive",
        ),
        (torch.zeros((2, 4, 5), dtype=torch.uint8), {}, "channel count"),
        (
            torch.zeros((3, 4, 5), dtype=torch.uint8),
            {"input_spec": InputSpec(color_space="GRAY")},
            "requires 1 channels",
        ),
        (
            torch.full((3, 4, 5), 2.0),
            {"input_spec": InputSpec(value_range="0_1")},
            "declares 0..1",
        ),
        (
            torch.full((3, 4, 5), 256.0),
            {"input_spec": InputSpec(value_range="0_255")},
            "declares 0..255",
        ),
    ],
)
def test_canonical_input_rejects_inconsistent_shape_color_and_range(
    source, kwargs, message
):
    with pytest.raises(InputError, match=message):
        canonicalize_image_tensor(source, source_kind="torch", **kwargs)


@pytest.mark.parametrize(
    ("source", "kwargs", "message"),
    [
        ("not-a-tensor", {}, "Expected a Torch tensor"),
        (torch.zeros((3, 4, 5), dtype=torch.bool), {}, "Unsupported image dtype"),
        (
            torch.zeros((3, 4, 5), dtype=torch.complex64),
            {},
            "Unsupported image dtype",
        ),
        (
            torch.zeros((3, 4, 5), dtype=torch.uint8),
            {"input_spec": object()},
            "input_spec must be an InputSpec",
        ),
    ],
)
def test_canonical_input_rejects_invalid_types(source, kwargs, message):
    with pytest.raises(InputError, match=message):
        canonicalize_image_tensor(source, source_kind="torch", **kwargs)


def test_canonical_input_handles_declared_bgr_and_integer_coercion():
    bgr = torch.tensor([10, 20, 30], dtype=torch.uint8).reshape(3, 1, 1)
    converted = canonicalize_image_tensor(
        bgr,
        source_kind="torch",
        input_policy="strict",
        input_spec=InputSpec(layout="CHW", value_range="0_255", color_space="BGR"),
    )
    assert converted.tensor[:, :, 0, 0].tolist() == [[30.0, 20.0, 10.0]]

    with pytest.warns(InputCoercionWarning, match="integer image dtype"):
        integer = canonicalize_image_tensor(
            torch.ones((3, 2, 2), dtype=torch.int16),
            source_kind="torch",
        )
    assert integer.tensor.dtype == torch.float32


@pytest.mark.parametrize("dtype", (torch.uint16, torch.uint32, torch.uint64))
def test_canonical_input_handles_unsigned_torch_dtypes(dtype):
    source = torch.tensor(range(12), dtype=dtype).reshape(3, 2, 2)
    with pytest.warns(InputCoercionWarning, match="integer image dtype"):
        converted = canonicalize_image_tensor(source, source_kind="torch")

    assert converted.tensor.dtype == torch.float32
    assert converted.tensor.min().item() == 0
    assert converted.tensor.max().item() == 11


@pytest.mark.parametrize("dtype", (np.uint16, np.uint32, np.uint64))
def test_canonical_input_handles_unsigned_numpy_dtypes(dtype):
    source = np.arange(12, dtype=dtype).reshape(2, 2, 3)
    reader = UniversalReader(None, torch.device("cpu"), False)
    with pytest.warns(InputCoercionWarning, match="integer image dtype"):
        converted = reader.run(source)

    assert converted.tensor.dtype == torch.float32
    assert converted.tensor.min().item() == 0
    assert converted.tensor.max().item() == 11


@pytest.mark.parametrize(
    "source",
    (
        torch.full((3, 2, 2), 256, dtype=torch.uint16),
        np.full((2, 2, 3), 256, dtype=np.uint16),
    ),
)
def test_unsigned_input_range_errors_remain_inside_the_public_boundary(source):
    reader = UniversalReader(None, torch.device("cpu"), False)
    with pytest.warns(InputCoercionWarning, match="integer image dtype"):
        with pytest.raises(InputError, match="0..255"):
            reader.run(source)


def test_float64_range_validation_does_not_round_into_the_supported_boundary():
    source = torch.full(
        (3, 2, 2),
        torch.nextafter(
            torch.tensor(255.0, dtype=torch.float64),
            torch.tensor(float("inf"), dtype=torch.float64),
        ).item(),
        dtype=torch.float64,
    )

    with pytest.raises(InputError, match="0..255"):
        canonicalize_image_tensor(
            source,
            source_kind="torch",
            input_spec=InputSpec(value_range="0_255"),
        )


def test_strict_rgba_requires_an_explicit_alpha_policy():
    with pytest.raises(InputError, match="alpha_mode"):
        canonicalize_image_tensor(
            torch.zeros((4, 2, 2), dtype=torch.uint8),
            source_kind="torch",
            input_policy="strict",
            input_spec=InputSpec(layout="CHW", value_range="0_255", color_space="RGBA"),
        )


def test_reader_rejects_unsupported_numpy_and_local_path_inputs(tmp_path):
    universal = UniversalReader(None, torch.device("cpu"), False)
    with pytest.raises(InputError, match="Expected a NumPy array"):
        universal.read_numpy_array("not-an-array", False)
    with pytest.raises(InputError, match="Unsupported NumPy image dtype"):
        universal.read_numpy_array(np.array([[object()]], dtype=object), False)
    with pytest.raises(InputError, match="Could not read local image"):
        universal.read_image_from_path(str(tmp_path / "missing.png"), False)

    image_reader = facetorch.analyzer.reader.ImageReader(
        None, torch.device("cpu"), False
    )
    with pytest.raises(InputError, match="only a local path"):
        image_reader.run(torch.zeros((3, 2, 2)))
    with pytest.raises(InputError, match="does not permit remote URLs"):
        image_reader.run("https://example.test/image.png")


def test_decoded_unusual_pil_modes_are_explicitly_coerced_or_rejected():
    reader = UniversalReader(None, torch.device("cpu"), False)
    image = Image.new("CMYK", (3, 2), (0, 20, 40, 0))
    with pytest.raises(InputError, match="Strict mode"):
        reader.read_pil_image(image, False, input_policy="strict")
    with pytest.warns(InputCoercionWarning, match="PIL mode"):
        result = reader.read_pil_image(image, False)
    assert result.tensor.shape == (1, 3, 2, 3)
    assert result.warnings[0].startswith("Converted decoded PIL mode")
    image.close()


def test_decoded_images_are_bounded_before_materialization():
    image = Image.new("RGB", (11, 10), color="red")
    reader = UniversalReader(
        None,
        torch.device("cpu"),
        False,
        max_decoded_pixels=109,
    )
    try:
        with pytest.raises(InputError, match="configured limit is 109"):
            reader.read_pil_image(image, False)
        with pytest.raises(InputError, match="configured limit is 109"):
            reader.read_image_from_bytes(_png_bytes(np.array(image)), False)
    finally:
        image.close()


def test_pillow_decompression_bomb_warnings_use_the_public_error(monkeypatch):
    payload = _png_bytes(_rgb_array())
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 50)
    reader = UniversalReader(None, torch.device("cpu"), False)

    with pytest.raises(InputError, match="Pillow's safety limit"):
        reader.read_image_from_bytes(payload, False)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"allowed_schemes": ()}, "allowed_schemes"),
        ({"allowed_schemes": ("ftp",)}, "allowed_schemes"),
        ({"timeout": 0}, "timeout"),
        ({"max_redirects": -1}, "max_redirects"),
        ({"max_bytes": 0}, "max_bytes"),
        ({"max_decoded_pixels": 0}, "max_decoded_pixels"),
        ({"max_decoded_pixels": True}, "max_decoded_pixels"),
    ],
)
def test_url_reader_rejects_invalid_limits(kwargs, message):
    with pytest.raises(InputError, match=message):
        URLReader(None, torch.device("cpu"), False, **kwargs)


@pytest.mark.parametrize(
    ("resolver", "message"),
    [
        (lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError()), "resolved safely"),
        (lambda *_args, **_kwargs: [], "did not resolve"),
        (
            lambda _host, port, **_kwargs: [(2, 1, 6, "", ("not-an-ip", port))],
            "resolved unexpectedly",
        ),
    ],
)
def test_url_reader_fails_closed_on_unsafe_dns_results(monkeypatch, resolver, message):
    monkeypatch.setattr("facetorch.analyzer.reader.core.socket.getaddrinfo", resolver)
    reader = URLReader(None, torch.device("cpu"), False)
    with pytest.raises(InputError, match=message):
        reader.run("https://example.test/image.png")


def test_url_reader_bounds_decoded_pixels_after_bounded_download(monkeypatch):
    monkeypatch.setattr(
        "facetorch.analyzer.reader.core.socket.getaddrinfo",
        lambda _host, port, **_kwargs: [(2, 1, 6, "", ("93.184.216.34", port))],
    )
    response = _FakeResponse(_png_bytes(_rgb_array()))
    monkeypatch.setattr(
        "facetorch.analyzer.reader.core._open_pinned_response",
        _fake_url_open(response),
    )
    reader = URLReader(
        None,
        torch.device("cpu"),
        False,
        max_bytes=1024,
        max_decoded_pixels=71,
    )

    with pytest.raises(InputError, match="configured limit is 71"):
        reader.run("https://example.test/image.png")
    assert response.closed


def test_url_reader_rejects_credentials_missing_hosts_and_non_strings(monkeypatch):
    reader = URLReader(None, torch.device("cpu"), False)
    with pytest.raises(InputError, match="only a URL string"):
        reader.run(Path("image.png"))
    with pytest.raises(InputError, match="no host"):
        reader.run("https:///image.png")
    with pytest.raises(InputError, match="credentials"):
        reader.run("https://user:secret@example.test/image.png")


def test_url_reader_handles_redirect_and_stream_protocol_errors(monkeypatch):
    monkeypatch.setattr(
        "facetorch.analyzer.reader.core.socket.getaddrinfo",
        lambda _host, port, **_kwargs: [(2, 1, 6, "", ("93.184.216.34", port))],
    )
    reader = URLReader(
        None,
        torch.device("cpu"),
        False,
        max_redirects=1,
        max_bytes=4,
    )

    missing_target = _FakeResponse(status_code=302)
    monkeypatch.setattr(
        "facetorch.analyzer.reader.core._open_pinned_response",
        _fake_url_open(missing_target),
    )
    with pytest.raises(InputError, match="omitted its target"):
        reader.run("https://example.test/image.png")
    assert missing_target.closed

    invalid_length = _FakeResponse(headers={"Content-Length": "many"})
    monkeypatch.setattr(
        "facetorch.analyzer.reader.core._open_pinned_response",
        _fake_url_open(invalid_length),
    )
    with pytest.raises(InputError, match="invalid Content-Length"):
        reader.run("https://example.test/image.png")
    assert invalid_length.closed

    streamed = _FakeResponse(body=b"12345")
    monkeypatch.setattr(
        "facetorch.analyzer.reader.core._open_pinned_response",
        _fake_url_open(streamed),
    )
    with pytest.raises(InputError, match="size limit"):
        reader.run("https://example.test/image.png")
    assert streamed.closed

    failed = _FakeResponse(status_code=500)
    monkeypatch.setattr(
        "facetorch.analyzer.reader.core._open_pinned_response",
        _fake_url_open(failed),
    )
    with pytest.raises(InputError, match="unsuccessful response"):
        reader.run("https://example.test/image.png")
    assert failed.closed


def test_url_reader_follows_one_redirect_and_sanitizes_ipv6_metadata(monkeypatch):
    responses = [
        _FakeResponse(status_code=302, headers={"Location": "/final.png"}),
        _FakeResponse(_png_bytes(_rgb_array())),
    ]
    monkeypatch.setattr(
        "facetorch.analyzer.reader.core.socket.getaddrinfo",
        lambda _host, port, **_kwargs: [
            (10, 1, 6, "", ("2606:2800:220:1:248:1893:25c8:1946", port, 0, 0))
        ],
    )
    monkeypatch.setattr(
        "facetorch.analyzer.reader.core._open_pinned_response",
        _fake_url_open(*responses),
    )
    reader = URLReader(None, torch.device("cpu"), False, max_redirects=1)
    result = reader.run(
        "https://[2606:2800:220:1:248:1893:25c8:1946]:444/start.png?token=x"
    )
    assert result.path_input == (
        "https://[2606:2800:220:1:248:1893:25c8:1946]:444/final.png"
    )
