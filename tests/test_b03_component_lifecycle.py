import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate as hydra_instantiate
from omegaconf import OmegaConf

from facetorch.analyzer.core import FaceAnalyzer, _LazyComponentRegistry
from facetorch.analyzer.detector.core import FaceDetector
from facetorch.analyzer.predictor.core import FacePredictor
from facetorch.analyzer.reader import TensorReader
from facetorch.datastruct import Prediction
from facetorch.exceptions import ConfigurationError, InferenceError


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_lazy_registry_membership_never_constructs_a_component():
    loaded = []
    registry = _LazyComponentRegistry(
        {"au": object()},
        loader=lambda name, _cfg: loaded.append(name),
    )

    assert "au" in registry
    assert "missing" not in registry
    assert loaded == []
    assert registry.loaded_names == ()


class _RecordingReader(TensorReader):
    def __init__(self):
        super().__init__(None, torch.device("cpu"), False)
        self.calls = 0

    def run(self, *args, **kwargs):
        self.calls += 1
        return super().run(*args, **kwargs)


class _IdentityUnifier:
    def run(self, data):
        return data


class _NoFaceDetector:
    def __init__(self):
        self.calls = 0

    def run(self, data):
        self.calls += 1
        data.faces = []
        return data


class _RecordingPredictor:
    def __init__(self, label):
        self.label = label
        self.calls = 0

    def run(self, faces):
        self.calls += 1
        return [Prediction(label=self.label) for _ in faces]


class _RecordingUtilizer:
    def __init__(self):
        self.calls = 0

    def run(self, data):
        self.calls += 1
        return data


def _component_graph(*, unavailable=()):
    reader = _RecordingReader()
    detector = _NoFaceDetector()
    predictors = {
        "first": _RecordingPredictor("first"),
        "second": _RecordingPredictor("second"),
    }
    utilizers = {"consumer": _RecordingUtilizer()}
    cfg = OmegaConf.create(
        {
            "reader": {"component": "reader"},
            "detector": {"component": "detector"},
            "unifier": {"component": "unifier"},
            "predictor": {
                name: {"component": f"predictor:{name}"} for name in predictors
            },
            "utilizer": {
                name: {"component": f"utilizer:{name}"} for name in utilizers
            },
            "utilizer_dependencies": {"consumer": ["second"]},
        }
    )
    constructed = []

    def instantiate_component(component):
        name = component["component"]
        constructed.append(name)
        if name in unavailable:
            raise RuntimeError(f"{name} is unavailable")
        if name == "reader":
            return reader
        if name == "detector":
            return detector
        if name == "unifier":
            return _IdentityUnifier()
        kind, component_name = name.split(":", 1)
        if kind == "predictor":
            return predictors[component_name]
        return utilizers[component_name]

    return SimpleNamespace(
        cfg=cfg,
        reader=reader,
        detector=detector,
        predictors=predictors,
        utilizers=utilizers,
        constructed=constructed,
        instantiate=instantiate_component,
    )


@pytest.mark.release_blocker
def test_component_names_are_inspectable_without_loading_models():
    graph = _component_graph()

    with patch(
        "facetorch.analyzer.core.instantiate", side_effect=graph.instantiate
    ):
        analyzer = FaceAnalyzer(graph.cfg)

        assert analyzer.configured_predictors == ("first", "second")
        assert tuple(analyzer.predictors.keys()) == ("first", "second")
        assert analyzer.loaded_predictors == ()
        assert analyzer.loaded_utilizers == ()
        assert analyzer.detector_loaded is False

    assert graph.constructed == ["reader", "unifier"]


@pytest.mark.release_blocker
def test_shipped_component_graph_initializes_without_artifact_components():
    with initialize_config_dir(
        config_dir=str(REPO_ROOT / "conf"), version_base=None
    ):
        cfg = compose(config_name="config")
    cfg.analyzer.logger = None

    forbidden_targets = {
        "facetorch.analyzer.detector.FaceDetector",
        "facetorch.analyzer.predictor.FacePredictor",
        "facetorch.analyzer.utilizer.align.Lmk3DMeshPose",
        "facetorch.analyzer.utilizer.draw.BoxDrawer",
        "facetorch.analyzer.utilizer.draw.LandmarkDrawerTorch",
        "facetorch.analyzer.utilizer.save.ImageSaver",
    }

    def reject_artifact_component(component, *args, **kwargs):
        target = component.get("_target_")
        if target in forbidden_targets:
            raise AssertionError(f"eager artifact component construction: {target}")
        return hydra_instantiate(component, *args, **kwargs)

    with patch(
        "facetorch.analyzer.core.instantiate",
        side_effect=reject_artifact_component,
    ):
        analyzer = FaceAnalyzer(cfg.analyzer)

    assert analyzer.configured_predictors == (
        "embed",
        "verify",
        "fer",
        "au",
        "va",
        "deepfake",
        "align",
    )
    assert analyzer.configured_utilizers == (
        "align",
        "draw_boxes",
        "draw_landmarks",
        "save",
    )
    assert analyzer.loaded_predictors == ()
    assert analyzer.loaded_utilizers == ()
    assert analyzer.detector_loaded is False
    assert analyzer.utilizer_dependencies == {"align": ("align",)}


@pytest.mark.release_blocker
def test_selection_constructs_only_requested_components_and_caches_them():
    graph = _component_graph(
        unavailable={"predictor:second", "utilizer:consumer"}
    )

    with patch(
        "facetorch.analyzer.core.instantiate", side_effect=graph.instantiate
    ):
        analyzer = FaceAnalyzer(graph.cfg)
        for _ in range(2):
            result = analyzer.run(
                image_source=torch.zeros((3, 4, 5), dtype=torch.uint8),
                skip_detector=True,
                include_predictors=["first"],
            )
            assert tuple(result.faces[0].preds) == ("first",)
        analyzer.run(
            image_source=torch.zeros((3, 4, 5), dtype=torch.uint8),
            skip_detector=True,
            include_predictors=[],
        )

    assert graph.constructed.count("predictor:first") == 1
    assert "predictor:second" not in graph.constructed
    assert "utilizer:consumer" not in graph.constructed
    assert "detector" not in graph.constructed
    assert analyzer.loaded_predictors == ("first",)
    assert analyzer.loaded_utilizers == ()
    assert analyzer.detector_loaded is False


@pytest.mark.release_blocker
@pytest.mark.parametrize(
    ("unavailable", "selection", "message"),
    [
        ("predictor:first", ["first"], "Face predictor 'first'"),
        ("utilizer:consumer", ["second"], "Face utilizer 'consumer'"),
    ],
)
def test_lazy_component_construction_failures_use_public_inference_error(
    unavailable, selection, message
):
    graph = _component_graph(unavailable={unavailable})

    with patch(
        "facetorch.analyzer.core.instantiate", side_effect=graph.instantiate
    ):
        analyzer = FaceAnalyzer(graph.cfg)
        with pytest.raises(InferenceError, match=message) as caught:
            analyzer.run(
                image_source=torch.zeros((3, 4, 5), dtype=torch.uint8),
                skip_detector=True,
                include_predictors=selection,
            )

    assert isinstance(caught.value.__cause__, RuntimeError)


@pytest.mark.release_blocker
def test_lazy_component_construction_preserves_public_errors():
    graph = _component_graph()

    def instantiate_component(component):
        if component["component"] == "predictor:first":
            raise ConfigurationError("predictor configuration is invalid")
        return graph.instantiate(component)

    with patch(
        "facetorch.analyzer.core.instantiate", side_effect=instantiate_component
    ):
        analyzer = FaceAnalyzer(graph.cfg)
        with pytest.raises(
            ConfigurationError, match="predictor configuration is invalid"
        ):
            analyzer.run(
                image_source=torch.zeros((3, 4, 5), dtype=torch.uint8),
                skip_detector=True,
                include_predictors=["first"],
            )


@pytest.mark.release_blocker
def test_utilizer_dependency_does_not_depend_on_matching_component_names():
    graph = _component_graph()

    with patch(
        "facetorch.analyzer.core.instantiate", side_effect=graph.instantiate
    ):
        analyzer = FaceAnalyzer(graph.cfg)
        analyzer.run(
            image_source=torch.zeros((3, 4, 5), dtype=torch.uint8),
            skip_detector=True,
            include_predictors=["second"],
        )

    assert graph.utilizers["consumer"].calls == 1
    assert analyzer.loaded_predictors == ("second",)
    assert analyzer.loaded_utilizers == ("consumer",)


@pytest.mark.release_blocker
@pytest.mark.parametrize(
    ("dependencies", "message"),
    [
        ({"missing": ["first"]}, "Unknown utilizer"),
        ({"consumer": ["missing"]}, "unknown predictor"),
        ({"consumer": "second"}, "collection"),
    ],
)
def test_invalid_utilizer_dependency_graph_fails_during_construction(
    dependencies, message
):
    graph = _component_graph()
    graph.cfg.utilizer_dependencies = dependencies

    with patch(
        "facetorch.analyzer.core.instantiate", side_effect=graph.instantiate
    ):
        with pytest.raises(ConfigurationError, match=message):
            FaceAnalyzer(graph.cfg)

    assert graph.constructed == ["reader", "unifier"]


@pytest.mark.release_blocker
def test_empty_include_constructs_no_predictor_and_none_runs_all():
    empty_graph = _component_graph()
    with patch(
        "facetorch.analyzer.core.instantiate", side_effect=empty_graph.instantiate
    ):
        empty_analyzer = FaceAnalyzer(empty_graph.cfg)
        empty_result = empty_analyzer.run(
            image_source=torch.zeros((3, 4, 5), dtype=torch.uint8),
            skip_detector=True,
            include_predictors=[],
        )

    assert empty_result.faces[0].preds == {}
    assert empty_analyzer.loaded_predictors == ()

    default_graph = _component_graph()
    with patch(
        "facetorch.analyzer.core.instantiate", side_effect=default_graph.instantiate
    ):
        default_analyzer = FaceAnalyzer(default_graph.cfg)
        default_result = default_analyzer.run(
            image_source=torch.zeros((3, 4, 5), dtype=torch.uint8),
            skip_detector=True,
        )

    assert tuple(default_result.faces[0].preds) == ("first", "second")
    assert default_analyzer.loaded_predictors == ("first", "second")

    exclude_graph = _component_graph()
    with patch(
        "facetorch.analyzer.core.instantiate", side_effect=exclude_graph.instantiate
    ):
        exclude_analyzer = FaceAnalyzer(exclude_graph.cfg)
        exclude_result = exclude_analyzer.run(
            image_source=torch.zeros((3, 4, 5), dtype=torch.uint8),
            skip_detector=True,
            exclude_predictors=[],
        )

    assert tuple(exclude_result.faces[0].preds) == ("first", "second")


@pytest.mark.release_blocker
def test_predictor_execution_order_follows_configuration_order():
    graph = _component_graph()

    with patch(
        "facetorch.analyzer.core.instantiate", side_effect=graph.instantiate
    ):
        analyzer = FaceAnalyzer(graph.cfg)
        result = analyzer.run(
            image_source=torch.zeros((3, 4, 5), dtype=torch.uint8),
            skip_detector=True,
            include_predictors=["second", "first"],
        )

    assert tuple(result.faces[0].preds) == ("first", "second")


@pytest.mark.release_blocker
@pytest.mark.parametrize(
    ("selection", "message"),
    [
        ({"include_predictors": ["missing"]}, "Unknown predictor"),
        ({"include_predictors": ["first", "first"]}, "duplicate"),
        (
            {"include_predictors": [], "exclude_predictors": []},
            "Cannot specify both",
        ),
        ({"include_predictors": "first"}, "collection"),
    ],
)
def test_invalid_selection_fails_before_input_or_model_work(selection, message):
    graph = _component_graph()

    with patch(
        "facetorch.analyzer.core.instantiate", side_effect=graph.instantiate
    ):
        analyzer = FaceAnalyzer(graph.cfg)
        with pytest.raises(ConfigurationError, match=message):
            analyzer.run(
                image_source=torch.zeros((3, 4, 5), dtype=torch.uint8),
                skip_detector=True,
                **selection,
            )

    assert graph.reader.calls == 0
    assert analyzer.loaded_predictors == ()
    assert analyzer.detector_loaded is False
    assert "detector" not in graph.constructed


@pytest.mark.release_blocker
def test_no_faces_loads_detector_once_but_never_loads_predictors():
    graph = _component_graph()

    with patch(
        "facetorch.analyzer.core.instantiate", side_effect=graph.instantiate
    ):
        analyzer = FaceAnalyzer(graph.cfg)
        for _ in range(2):
            analyzer.run(
                image_source=torch.zeros((3, 4, 5), dtype=torch.uint8),
            )

    assert graph.constructed.count("detector") == 1
    assert graph.detector.calls == 2
    assert analyzer.detector_loaded is True
    assert analyzer.loaded_predictors == ()


@pytest.mark.release_blocker
@pytest.mark.parametrize("component_class", [FacePredictor, FaceDetector])
@pytest.mark.parametrize(
    "compile_kwargs",
    [{}, {"compile_model": False, "compile_options": {"backend": "eager"}}],
)
def test_model_wrapper_compile_is_disabled_by_default(
    component_class, compile_kwargs, tmp_path
):
    path_model = tmp_path / "model.pt"
    scripted = torch.jit.trace(torch.nn.Identity(), torch.zeros((1, 3)))
    torch.jit.save(scripted, str(path_model))

    with patch("torch.compile") as compile_spy:
        component_class(
            downloader=SimpleNamespace(path_local=str(path_model)),
            device=torch.device("cpu"),
            preprocessor=object(),
            postprocessor=object(),
            **compile_kwargs,
        )

    compile_spy.assert_not_called()


@pytest.mark.release_blocker
@pytest.mark.parametrize("component_class", [FacePredictor, FaceDetector])
def test_model_wrappers_reject_undeclared_constructor_options(component_class):
    parameters = inspect.signature(component_class).parameters.values()
    assert all(parameter.kind != inspect.Parameter.VAR_KEYWORD for parameter in parameters)
