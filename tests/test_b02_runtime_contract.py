import io
import json
import logging
from contextlib import nullcontext
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
import requests
import torch
from PIL import Image

import facetorch
from facetorch.analyzer.core import FaceAnalyzer
from facetorch.analyzer.detector.core import FaceDetector
from facetorch.analyzer.detector.post import PostRetFace
from facetorch.analyzer.reader import TensorReader, UniversalReader, URLReader
from facetorch.analyzer.utilizer.save import ImageSaver
from facetorch.datastruct import Detection, Dimensions, Face, ImageData, Location, Prediction
from facetorch.exceptions import ConfigurationError, InputCoercionWarning, InputError
from facetorch.input import InputSpec
from facetorch.logger import LoggerJsonFile


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
    expects_warning = (
        strict_spec.color_space != "RGB"
        or (
            isinstance(source, (torch.Tensor, np.ndarray))
            and (source.dtype == torch.float32 or source.dtype == np.float32)
        )
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


class _FakeResponse:
    def __init__(self, body=b"", status_code=200, headers=None):
        self.body = body
        self.status_code = status_code
        self.headers = headers or {}
        self.closed = False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError("failed")

    def iter_content(self, chunk_size):
        yield from (
            self.body[index : index + chunk_size]
            for index in range(0, len(self.body), chunk_size)
        )

    def close(self):
        self.closed = True


def test_url_reader_is_explicit_bounded_and_removes_query_metadata(monkeypatch):
    response = _FakeResponse(_png_bytes(_rgb_array()))
    monkeypatch.setattr(
        "facetorch.analyzer.reader.core.socket.getaddrinfo",
        lambda _host, port, **_kwargs: [
            (2, 1, 6, "", ("93.184.216.34", port))
        ],
    )
    monkeypatch.setattr(
        "facetorch.analyzer.reader.core.requests.get", lambda *_args, **_kwargs: response
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


def test_url_reader_rejects_scheme_size_redirects_and_timeouts(monkeypatch):
    monkeypatch.setattr(
        "facetorch.analyzer.reader.core.socket.getaddrinfo",
        lambda _host, port, **_kwargs: [
            (2, 1, 6, "", ("93.184.216.34", port))
        ],
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
        "facetorch.analyzer.reader.core.requests.get",
        lambda *_args, **_kwargs: oversized,
    )
    with pytest.raises(InputError, match="size limit"):
        reader.run("https://example.test/image.png")
    assert oversized.closed

    redirect = _FakeResponse(status_code=302, headers={"Location": "/other.png"})
    monkeypatch.setattr(
        "facetorch.analyzer.reader.core.requests.get",
        lambda *_args, **_kwargs: redirect,
    )
    with pytest.raises(InputError, match="redirect limit"):
        reader.run("https://example.test/image.png")
    assert redirect.closed

    def raise_timeout(*_args, **_kwargs):
        raise requests.Timeout()

    monkeypatch.setattr("facetorch.analyzer.reader.core.requests.get", raise_timeout)
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
        "facetorch.analyzer.reader.core.requests.get", record_request
    )
    reader = URLReader(None, torch.device("cpu"), False)

    with pytest.raises(InputError, match="public network"):
        reader.run("https://internal.example/image.png")
    assert called is False


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
