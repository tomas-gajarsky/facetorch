"""Edge behavior for the lazy analyzer and custom-reader extension contracts."""

import logging
from types import SimpleNamespace

import pytest
import torch

from facetorch.analyzer.core import FaceAnalyzer, _LazyComponentRegistry
from facetorch.datastruct import ImageData
from facetorch.exceptions import ConfigurationError, InputError
from facetorch.input import InputSpec

pytestmark = pytest.mark.release_blocker


def _analyzer_with_reader(reader):
    analyzer = object.__new__(FaceAnalyzer)
    analyzer.logger = logging.getLogger("facetorch-analyzer-contract-edges")
    analyzer.reader = reader
    analyzer._reader_signature_owner = None
    analyzer._reader_signature_parameters = None
    return analyzer


def _valid_data(*, canonical=True):
    data = ImageData(tensor=torch.zeros((1, 3, 2, 3), dtype=torch.float32))
    data.set_dims()
    if canonical:
        data._facetorch_canonical = True
    return data


def test_lazy_registry_mutation_copy_and_errors_preserve_mapping_semantics():
    loaded = []
    registry = _LazyComponentRegistry(
        {"configured": 1},
        loader=lambda name, value: loaded.append(name) or value + 1,
    )
    with pytest.raises(KeyError):
        _ = registry["missing"]
    assert registry.copy() == {"configured": 2}
    assert loaded == ["configured"]

    registry["installed"] = 3
    assert registry["installed"] == 3
    del registry["configured"]
    assert tuple(registry) == ("installed",)
    with pytest.raises(KeyError):
        del registry["missing"]

    without_loader = _LazyComponentRegistry({"configured": object()})
    with pytest.raises(ConfigurationError, match="no component loader"):
        _ = without_loader["configured"]


def test_analyzer_fallback_registries_setters_and_detector_contract():
    analyzer = object.__new__(FaceAnalyzer)
    analyzer.logger = logging.getLogger("facetorch-analyzer-fallbacks")

    assert tuple(analyzer.predictors) == ()
    assert tuple(analyzer.utilizers) == ()
    assert analyzer.detector_loaded is False
    with pytest.raises(ConfigurationError, match="No face detector"):
        _ = analyzer.detector

    detector = object()
    analyzer.detector = detector
    assert analyzer.detector is detector
    assert analyzer.detector_loaded is True

    with pytest.raises(TypeError, match="predictors must be a mapping"):
        analyzer.predictors = []
    with pytest.raises(TypeError, match="utilizers must be a mapping"):
        analyzer.utilizers = []

    analyzer.predictors = {"one": object()}
    analyzer.utilizers = {"save": object()}
    assert analyzer.loaded_predictors == ("one",)
    assert analyzer.loaded_utilizers == ("save",)

    analyzer.__dict__["_predictors"] = {"plain": object()}
    analyzer.__dict__["_utilizers"] = {"plain": object()}
    assert analyzer.loaded_predictors == ("plain",)
    assert analyzer.loaded_utilizers == ("plain",)


@pytest.mark.parametrize(
    ("selection", "message"),
    [
        ("one", "not a string"),
        (3, "must be a collection"),
        ([""], "non-empty"),
        ([3], "non-empty"),
        (["one", "one"], "duplicate"),
    ],
)
def test_predictor_selection_rejects_ambiguous_values(selection, message):
    with pytest.raises(ConfigurationError, match=message):
        FaceAnalyzer._normalize_predictor_selection(selection, "include_predictors")


def test_predictor_selection_is_validated_and_keeps_configuration_order():
    analyzer = object.__new__(FaceAnalyzer)
    analyzer.predictors = {"one": object(), "two": object(), "three": object()}
    with pytest.raises(ConfigurationError, match="Cannot specify both"):
        analyzer._select_predictor_names(["one"], ["two"])
    with pytest.raises(ConfigurationError, match="Unknown predictor"):
        analyzer._select_predictor_names(["missing"], None)
    assert analyzer._select_predictor_names(["three", "one"], None) == (
        "one",
        "three",
    )
    assert analyzer._select_predictor_names(None, ["two"]) == ("one", "three")
    assert analyzer._select_predictor_names(None, None) == ("one", "two", "three")


def test_custom_reader_signature_and_return_value_failures_are_actionable():
    uninspectable = _analyzer_with_reader(SimpleNamespace(run=object()))
    with pytest.raises(ConfigurationError, match="inspectable public signature"):
        uninspectable._read_input(torch.zeros((3, 2, 3)), False)

    class WrongReturnReader:
        def run(self, image_source, **kwargs):
            return image_source

    wrong_return = _analyzer_with_reader(WrongReturnReader())
    with pytest.raises(ConfigurationError, match="must return"):
        wrong_return._read_input(torch.zeros((3, 2, 3)), False)


def test_legacy_reader_cannot_claim_strict_or_explicit_input_support():
    class LegacyReader:
        def run(self, _source, fix_img_size=False):
            return _valid_data()

    analyzer = _analyzer_with_reader(LegacyReader())
    with pytest.raises(ConfigurationError, match="cannot honor strict mode"):
        analyzer._read_input(torch.zeros((3, 2, 3)), False, input_policy="strict")
    with pytest.raises(ConfigurationError, match="cannot honor InputSpec"):
        analyzer._read_input(
            torch.zeros((3, 2, 3)),
            False,
            input_spec=InputSpec(layout="CHW"),
        )

    with pytest.warns(DeprecationWarning, match="deprecated v0.x protocol"):
        data = analyzer._read_input(torch.zeros((3, 2, 3)), False)
    assert data.warnings and "deprecated v0.x protocol" in data.warnings[-1]
    # The signature is cached after first inspection and reused safely.
    assert analyzer._read_input(torch.zeros((3, 2, 3)), False).dims.width == 3


@pytest.mark.parametrize(
    ("tensor", "message"),
    [
        ("not-a-tensor", "BCHW rank 4"),
        (torch.zeros((3, 2, 3)), "BCHW rank 4"),
        (torch.zeros((2, 3, 2, 3)), "B=1"),
        (torch.zeros((1, 1, 2, 3)), "three-channel RGB"),
        (torch.zeros((1, 3, 2, 3), dtype=torch.uint8), "float32"),
        (torch.full((1, 3, 2, 3), float("nan")), "NaN or Inf"),
        (torch.full((1, 3, 2, 3), 256.0), "within 0..255"),
    ],
)
def test_custom_reader_output_validation_rejects_noncanonical_data(tensor, message):
    data = ImageData(tensor=tensor)
    with pytest.raises((ConfigurationError, InputError), match=message):
        FaceAnalyzer._validate_reader_output(data)
