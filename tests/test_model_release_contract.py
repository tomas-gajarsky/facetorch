import json
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version
import pytest
import yaml

from scripts.export_model_cohorts_hf import _build_validation_cases, _model_specs

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]


def _project_metadata():
    with (REPO_ROOT / "pyproject.toml").open("rb") as project_file:
        return tomllib.load(project_file)["project"]


def _torch_requirement():
    requirements = [Requirement(item) for item in _project_metadata()["dependencies"]]
    return next(item for item in requirements if item.name == "torch")


def _huggingface_model_configs():
    for path in sorted((REPO_ROOT / "conf" / "analyzer").rglob("*.yaml")):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        downloader = config.get("downloader") if isinstance(config, dict) else None
        if not isinstance(downloader, dict):
            continue
        if downloader.get("_target_") != "facetorch.downloader.DownloaderHuggingFace":
            continue
        yield path, downloader


@pytest.mark.release_blocker
def test_torch_dependency_is_bounded_to_the_validated_matrix():
    requirement = _torch_requirement()
    assert any(spec.operator in {"<", "<="} for spec in requirement.specifier)


@pytest.mark.release_blocker
def test_every_accepted_torch_2_5_runtime_has_a_schema_7_cohort():
    requirement = _torch_requirement()
    if Version("2.5.0") not in requirement.specifier:
        return

    missing = []
    for path, downloader in _huggingface_model_configs():
        cohorts = downloader.get("export_filenames_by_torch_minor", {})
        if "2.5" not in cohorts:
            missing.append(str(path.relative_to(REPO_ROOT)))

    assert missing == []


@pytest.mark.release_blocker
def test_remote_model_descriptors_are_immutable_and_hash_verified():
    manifest = json.loads(
        (REPO_ROOT / "facetorch" / "models" / "manifest.json").read_text()
    )
    for model_id, model in manifest["models"].items():
        assert len(model["revision"]) == 40, model_id
        for artifact in model["artifacts"]:
            assert len(artifact["sha256"]) == 64, artifact["id"]
            assert artifact["size_bytes"] > 0, artifact["id"]


@pytest.mark.release_blocker
def test_state_dict_reconstruction_is_strict_for_every_model():
    violations = [
        spec["id"]
        for spec in _model_specs("2.6")
        if "state_dict" in spec["strategy"] and spec.get("strict") is not True
    ]
    assert violations == []


@pytest.mark.release_blocker
def test_model_specs_declare_an_independent_validation_reference():
    violations = []
    for spec in _model_specs("2.6"):
        reference = spec.get("validation_reference", {})
        if not {"kind", "source", "sha256"}.issubset(reference):
            violations.append(spec["id"])

    assert violations == []


@pytest.mark.release_blocker
def test_detector_validation_exercises_dynamic_spatial_shapes():
    detector = next(
        spec for spec in _model_specs("2.6") if spec["id"] == "detector-retinaface"
    )
    cases = _build_validation_cases(
        detector,
        batch_sizes=[1],
        seeds=[0],
        scales=[1.0],
    )
    spatial_shapes = {tuple(case["x"].shape[-2:]) for case in cases}

    assert len(spatial_shapes) > 1
