import inspect
import json
import logging
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import pytest
import torch
from torchvision import transforms

import facetorch
import facetorch.analyzer.reader as reader_api
from facetorch.analyzer.core import FaceAnalyzer
from facetorch.analyzer.detector.core import FaceDetector
from facetorch.analyzer.predictor.core import FacePredictor
from facetorch.analyzer.reader import TensorReader, UniversalReader
from facetorch.analyzer.unifier import FaceUnifier
from facetorch.analyzer.utilizer.save import ImageSaver
from facetorch.datastruct import Detection, Dimensions, Face, ImageData, Location, Prediction
from facetorch.logger import LoggerJsonFile
from omegaconf import OmegaConf


class _RecordingPredictor:
    def __init__(self):
        self.batches = []

    def run(self, faces):
        self.batches.append(faces.detach().clone())
        return [
            Prediction(label=str(int(round(float(face.flatten()[0])))))
            for face in faces
        ]


class _IdentityUnifier:
    def run(self, data):
        return data


class _FailingDetector:
    calls = 0

    def run(self, data):
        self.calls += 1
        raise AssertionError("detector must not run")


class _FaceProducingDetector:
    def __init__(self, count):
        self.count = count

    def run(self, data):
        height, width = data.tensor.shape[-2:]
        data.faces = [
            Face(
                indx=index,
                loc=Location(x1=0, y1=0, x2=width, y2=height),
                dims=Dimensions(height=height, width=width),
                tensor=torch.full_like(data.tensor[0], float(index)),
                ratio=1.0,
            )
            for index in range(self.count)
        ]
        return data


def _make_analyzer(*, detector=None, unifier=None, predictors=None):
    analyzer = object.__new__(FaceAnalyzer)
    analyzer.logger = logging.getLogger(f"facetorch-b01-{uuid4()}")
    analyzer.reader = TensorReader(
        transform=None,
        device=torch.device("cpu"),
        optimize_transform=False,
    )
    analyzer.detector = detector if detector is not None else _FailingDetector()
    analyzer.unifier = unifier if unifier is not None else _IdentityUnifier()
    analyzer.predictors = predictors if predictors is not None else {}
    analyzer.utilizers = {}
    return analyzer


@pytest.mark.release_blocker
def test_run_exposes_approved_hybrid_input_policy_contract():
    parameters = inspect.signature(FaceAnalyzer.run).parameters
    assert "input_policy" in parameters
    assert parameters["input_policy"].default == "coerce"
    assert "input_spec" in parameters


@pytest.mark.release_blocker
def test_face_batch_size_is_explicit_and_legacy_alias_remains_available():
    parameters = inspect.signature(FaceAnalyzer.run).parameters
    assert "face_batch_size" in parameters
    assert "batch_size" in parameters


@pytest.mark.release_blocker
def test_analysis_result_is_the_single_primary_return_type():
    result_type = getattr(facetorch, "AnalysisResult", None)
    assert result_type is not None

    analyzer = _make_analyzer()
    result = analyzer.run(
        image_source=torch.zeros((3, 8, 9)),
        skip_detector=True,
    )
    assert isinstance(result, result_type)


@pytest.mark.release_blocker
def test_model_components_are_not_eagerly_constructed():
    cfg = OmegaConf.create(
        {
            "reader": {"component": "reader"},
            "detector": {"component": "detector"},
            "unifier": {"component": "unifier"},
            "predictor": {"probe": {"component": "predictor"}},
        }
    )
    constructed = []

    def instantiate_probe(component):
        constructed.append(component["component"])
        return object()

    with patch("facetorch.analyzer.core.instantiate", side_effect=instantiate_probe):
        FaceAnalyzer(cfg)

    assert "detector" not in constructed
    assert "predictor" not in constructed


@pytest.mark.release_blocker
def test_optional_analyzer_logger_restores_info_diagnostics():
    cfg = OmegaConf.create({"reader": {"component": "reader"}})
    target = logging.getLogger("facetorch")
    original_level = target.level
    original_handler_levels = {
        handler: handler.level for handler in target.handlers
    }
    try:
        target.setLevel(logging.CRITICAL)
        for handler in target.handlers:
            if getattr(handler, "_facetorch_stream_handler", False):
                handler.setLevel(logging.CRITICAL)
        with patch(
            "facetorch.analyzer.core.instantiate", return_value=object()
        ):
            analyzer = FaceAnalyzer(cfg)

        managed = [
            handler
            for handler in target.handlers
            if getattr(handler, "_facetorch_stream_handler", False)
        ]
        assert analyzer.logger is target
        assert target.level == logging.INFO
        assert managed and all(handler.level == logging.INFO for handler in managed)
    finally:
        target.setLevel(original_level)
        for handler, level in original_handler_levels.items():
            handler.setLevel(level)


@pytest.mark.release_blocker
def test_remote_input_requires_an_explicit_bounded_url_reader():
    url_reader_type = getattr(reader_api, "URLReader", None)
    assert url_reader_type is not None

    parameters = inspect.signature(url_reader_type).parameters
    assert {
        "allowed_schemes",
        "timeout",
        "max_redirects",
        "max_bytes",
        "max_decoded_pixels",
    }.issubset(parameters)


@pytest.mark.release_blocker
def test_source_image_batch_is_rejected_before_detector_execution():
    detector = _FailingDetector()
    analyzer = _make_analyzer(detector=detector)

    with pytest.raises(ValueError, match="B=1"):
        analyzer.run(
            image_source=torch.zeros((2, 3, 8, 9)),
            skip_detector=True,
        )

    assert detector.calls == 0


@pytest.mark.release_blocker
def test_face_batching_stays_within_one_image_and_preserves_order():
    predictor = _RecordingPredictor()
    analyzer = _make_analyzer(
        detector=_FaceProducingDetector(count=3),
        predictors={"probe": predictor},
    )

    result = analyzer.run(
        image_source=torch.zeros((3, 8, 9)),
        batch_size=2,
    )

    assert [batch.shape[0] for batch in predictor.batches] == [2, 1]
    assert [face.preds["probe"].label for face in result.faces] == ["0", "1", "2"]


@pytest.mark.release_blocker
def test_skip_detector_uses_canonical_face_range():
    predictor = _RecordingPredictor()
    shipped_unifier = FaceUnifier(
        transform=transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[255.0, 255.0, 255.0],
                )
            ]
        ),
        device=torch.device("cpu"),
        optimize_transform=False,
    )
    analyzer = _make_analyzer(
        unifier=shipped_unifier,
        predictors={"probe": predictor},
    )

    analyzer.run(
        image_source=torch.full((3, 8, 9), 255, dtype=torch.uint8),
        skip_detector=True,
        include_tensors=True,
    )

    observed = predictor.batches[0]
    assert torch.isfinite(observed).all()
    assert float(observed.min()) >= 0.0
    assert float(observed.max()) <= 1.0


@pytest.mark.release_blocker
def test_empty_predictor_include_runs_nothing():
    predictor = _RecordingPredictor()
    analyzer = _make_analyzer(predictors={"probe": predictor})

    analyzer.run(
        image_source=torch.zeros((3, 8, 9)),
        skip_detector=True,
        include_predictors=[],
    )

    assert predictor.batches == []


@pytest.mark.release_blocker
def test_unknown_predictor_selection_fails_before_inference():
    predictor = _RecordingPredictor()
    analyzer = _make_analyzer(predictors={"probe": predictor})

    with pytest.raises(ValueError, match="unknown|Unknown"):
        analyzer.run(
            image_source=torch.zeros((3, 8, 9)),
            skip_detector=True,
            include_predictors=["missing"],
        )

    assert predictor.batches == []


@pytest.mark.release_blocker
def test_multiple_input_sources_are_rejected():
    analyzer = _make_analyzer()

    with pytest.raises(ValueError, match="input source|image_source"):
        analyzer.run(
            image_source=torch.zeros((3, 8, 9)),
            tensor=torch.zeros((3, 8, 9)),
            skip_detector=True,
        )


@pytest.mark.release_blocker
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_image_input_is_rejected(nonfinite):
    reader = TensorReader(None, torch.device("cpu"), False)
    with pytest.raises(ValueError, match="finite|NaN|Inf"):
        reader.run(torch.full((3, 8, 9), nonfinite))


@pytest.mark.release_blocker
def test_configured_reader_observes_tensor_input():
    """Regression contract for direct image input issue #82."""

    class ContractReader:
        def __init__(self):
            self.run_calls = 0
            self.process_calls = 0
            self.delegate = TensorReader(None, torch.device("cpu"), False)

        def run(self, image_source, fix_img_size=False):
            self.run_calls += 1
            return self.delegate.run(image_source, fix_img_size)

        def process_tensor(self, image_source, fix_img_size=False):
            self.process_calls += 1
            return self.delegate.process_tensor(image_source, fix_img_size)

    analyzer = _make_analyzer()
    analyzer.reader = ContractReader()
    analyzer._read_input(torch.zeros((3, 8, 9)), fix_img_size=False)

    assert analyzer.reader.run_calls == 1
    assert analyzer.reader.process_calls == 0


@pytest.mark.release_blocker
def test_grayscale_tensor_has_three_canonical_channels():
    """Regression contract for grayscale input issue #83."""

    reader = TensorReader(None, torch.device("cpu"), False)
    data = reader.run(torch.zeros((8, 9), dtype=torch.uint8))
    assert data.tensor.shape == (1, 3, 8, 9)


@pytest.mark.release_blocker
def test_direct_bytes_reader_closes_opened_image(monkeypatch):
    class TrackingImage:
        def __init__(self):
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            self.close()

        def close(self):
            self.closed = True

    opened = TrackingImage()
    reader = UniversalReader(None, torch.device("cpu"), False)
    sentinel = ImageData()
    monkeypatch.setattr(
        "facetorch.analyzer.reader.core.Image.open", lambda *_args, **_kwargs: opened
    )
    monkeypatch.setattr(reader, "read_pil_image", lambda *_args, **_kwargs: sentinel)

    assert reader.read_image_from_bytes(b"image", False) is sentinel
    assert opened.closed is True


@pytest.mark.release_blocker
@pytest.mark.parametrize("component_class", [FacePredictor, FaceDetector])
def test_model_wrappers_forward_compile_options(component_class, tmp_path):
    path_model = tmp_path / "model.pt"
    scripted = torch.jit.trace(torch.nn.Identity(), torch.zeros((1, 3)))
    torch.jit.save(scripted, str(path_model))
    downloader = SimpleNamespace(path_local=str(path_model))
    component_kwargs = {
        "downloader": downloader,
        "device": torch.device("cpu"),
        "preprocessor": object(),
        "postprocessor": object(),
        "compile_model": True,
        "compile_options": {"backend": "eager"},
    }

    with patch("torch.compile", side_effect=lambda model, **_kwargs: model) as compile_spy:
        component_class(**component_kwargs)

    compile_spy.assert_called_once()
    assert compile_spy.call_args.kwargs == {"backend": "eager"}


@pytest.mark.release_blocker
def test_custom_detector_postprocessor_retains_faces_after_padding():
    class PadPreprocessor:
        def run(self, data):
            data.tensor = torch.nn.functional.pad(data.tensor, (0, 2, 0, 2))
            data.set_dims()
            return data

    class CustomPostprocessor:
        def run(self, data, logits):
            data.det = Detection(dets=torch.tensor([[0.0, 0.0, 4.0, 4.0, 0.9]]))
            data.faces = [
                Face(
                    indx=0,
                    loc=Location(x1=0, y1=0, x2=4, y2=4),
                    dims=Dimensions(height=4, width=4),
                    tensor=data.tensor[0, :, :4, :4],
                    ratio=0.25,
                )
            ]
            return data

    detector = object.__new__(FaceDetector)
    detector.device = torch.device("cpu")
    detector.model = torch.nn.Identity()
    detector.preprocessor = PadPreprocessor()
    detector.postprocessor = CustomPostprocessor()
    data = ImageData(tensor=torch.zeros((1, 3, 8, 8)))
    data.set_dims()

    result = detector.run(data)

    assert len(result.faces) == 1
    assert result.faces[0].loc == Location(x1=0, y1=0, x2=4, y2=4)


def _clear_logger(name):
    target = logging.getLogger(name)
    for handler in list(target.handlers):
        target.removeHandler(handler)
        handler.close()


@pytest.mark.release_blocker
def test_file_logging_can_be_enabled_after_package_import(tmp_path):
    """Regression contract for logger issue #88."""

    name = f"facetorch-b01-{uuid4()}"
    path_log = tmp_path / "facetorch.log"
    try:
        LoggerJsonFile(name=name, level=logging.INFO)
        configured = LoggerJsonFile(
            name=name,
            level=logging.INFO,
            path_file=str(path_log),
        )
        configured.logger.info("release-blocker-probe")
        for handler in configured.logger.handlers:
            handler.flush()

        assert "release-blocker-probe" in path_log.read_text(encoding="utf-8")
    finally:
        _clear_logger(name)


@pytest.mark.release_blocker
def test_cold_component_imports_preserve_explicit_analyzer_logging(tmp_path):
    """Lazy Hydra imports must not reconfigure the shared package logger."""

    path_log = tmp_path / "cold-import.log"
    probe = r"""
import logging
from pathlib import Path
import sys

from facetorch import FaceAnalyzer, load_config

path_log = Path(sys.argv[1]).resolve()
cfg = load_config(profile="cpu", offline=True)
cfg.analyzer.logger.level = logging.INFO
cfg.analyzer.logger.path_file = str(path_log)
analyzer = FaceAnalyzer(cfg.analyzer)
managed_files = [
    handler
    for handler in analyzer.logger.handlers
    if getattr(handler, "_facetorch_file_handler", False)
]
if analyzer.logger.level != logging.INFO:
    raise SystemExit(f"logger level changed to {analyzer.logger.level}")
if len(managed_files) != 1 or Path(managed_files[0].baseFilename) != path_log:
    raise SystemExit("configured file handler was not retained")
analyzer.logger.info("after-lazy-component-imports")
for handler in analyzer.logger.handlers:
    handler.flush()
"""
    result = subprocess.run(
        [sys.executable, "-c", probe, str(path_log)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "after-lazy-component-imports" in path_log.read_text(encoding="utf-8")


@pytest.mark.release_blocker
def test_basename_log_path_and_json_format_are_supported(tmp_path, monkeypatch):
    name = f"facetorch-b01-{uuid4()}"
    monkeypatch.chdir(tmp_path)
    try:
        configured = LoggerJsonFile(
            name=name,
            level=logging.INFO,
            path_file="facetorch.log",
        )
        configured.logger.info("json-probe")
        for handler in configured.logger.handlers:
            handler.flush()

        records = Path("facetorch.log").read_text(encoding="utf-8").splitlines()
        assert any(json.loads(record)["message"] == "json-probe" for record in records)
    finally:
        _clear_logger(name)


@pytest.mark.release_blocker
def test_basename_image_output_path_is_supported(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    saver = ImageSaver(None, torch.device("cpu"), False)
    data = ImageData(
        path_output="face.png",
        img=torch.zeros((3, 8, 9), dtype=torch.uint8),
    )

    saver.run(data)

    assert Path("face.png").is_file()
