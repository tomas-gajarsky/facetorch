"""Public-boundary regressions from the final v1 readiness review."""

import logging
from types import SimpleNamespace

import pytest
import torch
from omegaconf import open_dict

from facetorch import FaceAnalyzer, load_config
from facetorch.base import BaseModel
from facetorch.datastruct import Prediction
from facetorch.exceptions import (
    ConfigurationError,
    InferenceError,
    ModelCompatibilityError,
    OfflineCacheError,
)

pytestmark = pytest.mark.release_blocker


class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, tensor):
        return self.linear(tensor)


class ParameterlessModel(torch.nn.Module):
    def forward(self, tensor):
        return tensor * 2


class ReconstructedModel(LinearModel):
    def load_from_torchscript(self, scripted):
        # A controlled hook can recover constants through its known contract.
        with torch.no_grad():
            self.linear.bias.copy_(scripted(torch.zeros(1, 3))[0])
            self.linear.weight.copy_((scripted(torch.eye(3)) - self.linear.bias).T)


class ConcreteModel(BaseModel):
    def run(self, tensor):
        return self.inference(tensor)


def _load_native(path, cls):
    return ConcreteModel(
        SimpleNamespace(path_local=str(path), verify_on_use=False),
        torch.device("cpu"),
        native_model_class=f"{__name__}.{cls.__name__}",
    )


def test_frozen_empty_state_cannot_silently_initialize_native_weights(tmp_path):
    source = LinearModel().eval()
    with torch.no_grad():
        source.linear.weight.fill_(2)
        source.linear.bias.zero_()
    frozen = torch.jit.freeze(torch.jit.trace(source, torch.ones(1, 3)))
    assert not frozen.state_dict()
    path = tmp_path / "frozen.pt"
    frozen.save(str(path))
    with pytest.raises(RuntimeError, match="Missing key"):
        _load_native(path, LinearModel)
    reconstructed = _load_native(path, ReconstructedModel)
    torch.testing.assert_close(
        reconstructed.run(torch.ones(1, 3)), frozen(torch.ones(1, 3))
    )


def test_parameterless_frozen_native_model_remains_valid(tmp_path):
    source = ParameterlessModel().eval()
    path = tmp_path / "parameterless.pt"
    torch.jit.freeze(torch.jit.trace(source, torch.ones(1, 3))).save(str(path))
    loaded = _load_native(path, ParameterlessModel)
    torch.testing.assert_close(loaded.run(torch.ones(1, 3)), torch.full((1, 3), 2.0))


def test_partial_native_state_is_rejected_before_inference(tmp_path):
    path = tmp_path / "partial.pth"
    torch.save({"linear.weight": torch.ones(3, 3)}, path)
    with pytest.raises(RuntimeError, match="linear.bias"):
        _load_native(path, LinearModel)


def test_offline_cache_error_survives_lazy_hydra_construction(tmp_path, monkeypatch):
    monkeypatch.setenv("FACETORCH_CACHE_DIR", str(tmp_path))
    analyzer = FaceAnalyzer(load_config(offline=True).analyzer)
    with pytest.raises(OfflineCacheError):
        analyzer.run(torch.zeros(3, 32, 32, dtype=torch.uint8), include_predictors=[])


def incompatible_component():
    raise ModelCompatibilityError("Unsupported test runtime")


@pytest.mark.parametrize("component", ["detector", "predictor"])
def test_lazy_compatibility_errors_keep_the_public_type(component):
    cfg = load_config(offline=True).analyzer
    target = {"_target_": f"{__name__}.incompatible_component"}
    if component == "detector":
        cfg.detector = target
    else:
        cfg.predictor = {"probe": target}
        cfg.utilizer = {}
        cfg.utilizer_dependencies = {}
    analyzer = FaceAnalyzer(cfg)
    with pytest.raises(ModelCompatibilityError, match="Unsupported test runtime"):
        analyzer.run(
            torch.zeros(3, 32, 32, dtype=torch.uint8),
            skip_detector=component == "predictor",
            include_predictors=["probe"] if component == "predictor" else [],
        )


def test_unknown_lazy_constructor_errors_are_configuration_errors():
    cfg = load_config(offline=True).analyzer
    cfg.detector = {"_target_": "builtins.int", "invalid_argument": True}
    analyzer = FaceAnalyzer(cfg)
    with pytest.raises(ConfigurationError, match="face detector"):
        analyzer.run(torch.zeros(3, 32, 32, dtype=torch.uint8), include_predictors=[])


@pytest.mark.parametrize("missing", [False, True])
def test_unconfigured_logger_preserves_application_state(monkeypatch, missing):
    cfg = load_config(offline=True).analyzer
    if missing:
        with open_dict(cfg):
            del cfg.logger
    else:
        cfg.logger = None
    app_logger = logging.getLogger("facetorch")
    handler = logging.NullHandler()
    monkeypatch.setattr(app_logger, "propagate", True)
    monkeypatch.setattr(app_logger, "handlers", [handler])
    original_level = app_logger.level
    try:
        app_logger.setLevel(logging.ERROR)
        for _ in range(2):
            FaceAnalyzer(cfg)
        assert app_logger.level == logging.ERROR
        assert app_logger.propagate is True
        assert app_logger.handlers == [handler]
    finally:
        app_logger.setLevel(original_level)


@pytest.mark.parametrize("include_tensors", [False, True])
@pytest.mark.parametrize(
    "prediction",
    [42, Prediction(logits=None), Prediction(other=None), Prediction(label=3)],
)
def test_malformed_predictor_values_fail_at_the_public_boundary(
    include_tensors, prediction
):
    cfg = load_config(offline=True).analyzer
    cfg.utilizer = {}
    cfg.utilizer_dependencies = {}
    analyzer = FaceAnalyzer(cfg)
    analyzer.predictors["invalid"] = SimpleNamespace(
        run=lambda batch: [prediction] * len(batch)
    )
    with pytest.raises(InferenceError, match="'invalid'.*face 0"):
        analyzer.run(
            torch.zeros(3, 8, 8, dtype=torch.uint8),
            skip_detector=True,
            include_predictors=["invalid"],
            include_tensors=include_tensors,
        )
