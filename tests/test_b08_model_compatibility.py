import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version
import pytest

from facetorch.artifacts import ArtifactManifest, get_model_manifest
from facetorch.exceptions import ConfigurationError, ModelCompatibilityError
from scripts.audit_model_manifest_hf import audit_remote_manifest
from scripts.export_model_cohorts_hf import _environment_metadata, _model_specs
from scripts.verify_model_release_matrix import (
    ReleaseMatrixError,
    verify_release_matrix,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "facetorch" / "models" / "manifest.json"
COMPATIBILITY_PATH = REPO_ROOT / "facetorch" / "models" / "compatibility.json"
GOVERNANCE_PATH = REPO_ROOT / "facetorch" / "models" / "governance.json"
SUPPORTED_TORCH = {"2.3", "2.6", "2.11"}
UNSUPPORTED_TORCH = {"2.4", "2.5", "2.7", "2.8", "2.9", "2.10"}
CUDA_ENVIRONMENT_LOCKS = {
    "2.3": "environments/torch-2.3-cu121/uv.lock",
    "2.6": "environments/torch-2.6-cu124/uv.lock",
    "2.11": "environments/torch-2.11-cu130/uv.lock",
}


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _torch_requirement():
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]
    return next(
        Requirement(item)
        for item in project["dependencies"]
        if Requirement(item).name == "torch"
    )


@pytest.mark.release_blocker
def test_package_metadata_exactly_matches_the_candidate_torch_matrix():
    compatibility = _json(COMPATIBILITY_PATH)
    requirement = _torch_requirement()
    assert str(requirement.specifier) == str(
        SpecifierSet(compatibility["torch"]["specifier"])
    )
    assert set(compatibility["torch"]["supported_minor_lines"]) == SUPPORTED_TORCH
    assert (
        set(compatibility["torch"]["explicitly_unsupported_minor_lines"])
        == UNSUPPORTED_TORCH
    )
    for minor in SUPPORTED_TORCH:
        assert Version(f"{minor}.0") in requirement.specifier
    for minor in UNSUPPORTED_TORCH:
        assert Version(f"{minor}.0") not in requirement.specifier


@pytest.mark.release_blocker
def test_every_model_has_exactly_one_current_metadata_artifact_per_cohort():
    manifest = get_model_manifest()
    assert manifest.supported_torch_minors == ("2.3", "2.6", "2.11")
    for model_id, artifacts in manifest.models.items():
        exports = [item for item in artifacts if item.format == "pt2"]
        assert len(exports) == 3, model_id
        assert {
            item.torch_min for item in exports
        } == SUPPORTED_TORCH, model_id
        assert all(
            item.validation_metadata == f"{item.filename}.meta.json"
            for item in exports
        )


@pytest.mark.release_blocker
def test_unsupported_torch_fails_before_even_explicit_legacy_fallback():
    manifest = get_model_manifest()
    for minor in UNSUPPORTED_TORCH:
        with pytest.raises(ModelCompatibilityError, match="no model download"):
            manifest.candidates(
                "fer-efficientnet-b0",
                torch_version=f"{minor}.0",
                device="cpu",
                allow_legacy_models=True,
            )


@pytest.mark.release_blocker
def test_provenance_records_cover_every_model_but_block_premature_approval():
    manifest_raw = _json(MANIFEST_PATH)
    compatibility = _json(COMPATIBILITY_PATH)
    governance = _json(GOVERNANCE_PATH)
    assert set(governance["models"]) == set(manifest_raw["models"])
    assert governance["status"] == "incomplete"
    for model_id, record in governance["models"].items():
        assert record["release_eligible"] is False, model_id
        assert record["rights"]["weights_license"] == "unverified", model_id
        assert record["limitations"], model_id
        assert record["intended_use"], model_id

    manifest_raw["status"] = "approved"
    with pytest.raises(ConfigurationError, match="complete provenance"):
        ArtifactManifest.from_mapping(
            manifest_raw,
            compatibility=compatibility,
            governance=governance,
        )


@pytest.mark.release_blocker
def test_export_specs_match_manifest_ids_and_pin_every_source():
    manifest = _json(MANIFEST_PATH)
    specs = _model_specs("2.11")
    assert {item["id"] for item in specs} == set(manifest["models"])
    assert all(item["strict"] is True for item in specs)
    assert all(
        item["cross_device_tolerances"]["max_abs"]
        > item["tolerances"]["max_abs"]
        for item in specs
    )
    assert all(
        item["validation_reference"]["device"] == "cpu" for item in specs
    )
    for spec in specs:
        source = spec["source_artifact"]
        model = manifest["models"][spec["id"]]
        assert source["repo_id"] == model["repo_id"]
        assert source["revision"] == model["revision"]
        assert source["sha256"] == model["source_weight_sha256"]
        assert len(source["revision"]) == 40
        assert len(source["sha256"]) == 64

    magface = next(item for item in specs if item["id"] == "verify-magface")
    assert magface["class_path"] == "model_defs.verify_model.MagFaceIResNet100"
    adaface = next(item for item in specs if item["id"] == "verify-adaface")
    assert adaface["validation_reference"]["device"] == "cpu"
    detector = next(item for item in specs if item["id"] == "detector-retinaface")
    assert detector["validation_reference"]["device"] == "cpu"
    assert detector["strategy"] == "native_from_torchscript_complete_state"
    assert detector["dynamic_hw"] is True
    au = next(item for item in specs if item["id"] == "au-opengraph")
    assert au["validation_reference"]["batch_mode"] == "per_sample"
    assert au["strategy"] == "reuse_existing_exported_program"
    assert au["reused_artifact_id"] == "au-opengraph-torch2.11"
    assert au["reuse_reason"]
    exporter = (REPO_ROOT / "scripts" / "export_model_cohorts_hf.py").read_text()
    assert "class _MagFaceIResNet100" not in exporter


def _stage_candidate_matrix(tmp_path):
    manifest = _json(MANIFEST_PATH)
    compatibility = _json(COMPATIBILITY_PATH)
    policy = compatibility["validation_policy"]
    summaries = []
    for cohort_record in compatibility["cohorts"]:
        cohort = cohort_record["torch_minor"]
        cohort_root = tmp_path / f"torch-{cohort}"
        results = []
        for model_id, model in manifest["models"].items():
            model_root = cohort_root / model_id
            model_root.mkdir(parents=True)
            artifact = model_root / f"model-torch{cohort}.pt2"
            artifact.write_bytes(f"{model_id}:{cohort}".encode())
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            is_detector = model_id == "detector-retinaface"
            batches = (
                policy["detector"]["batch_sizes"]
                if is_detector
                else policy["predictor_batch_sizes"]
            )
            shapes = (
                policy["detector"]["spatial_shapes"]
                if is_detector
                else [[8, 8]]
            )
            cases = [
                {
                    "case_id": (
                        f"b{batch}_h{height}_w{width}_seed{seed}_"
                        f"scale{scale}_{variant}"
                    ),
                    "status": "ok",
                    "batch": batch,
                    "seed": seed,
                    "scale": scale,
                    "variant": variant,
                    "input_shape": [batch, 3, height, width],
                }
                for batch in batches
                for height, width in shapes
                for seed in policy["seeds"]
                for scale in policy["scales"]
                for variant in policy["input_variants"]
            ]
            devices = [
                {
                    "device": device,
                    "status": "ok",
                    "num_cases": len(cases),
                    "cases": cases,
                }
                for device in compatibility["platform_policy"]["required_devices"]
            ]
            total_cases = len(cases) * len(devices)
            metadata = {
                "schema_version": 2,
                "model_id": model_id,
                "artifact_sha256": digest,
                "artifact_size_bytes": artifact.stat().st_size,
                "source_artifact": {
                    "revision": model["revision"],
                    "sha256": model["source_weight_sha256"],
                },
                "source": {
                    "validation_reference": {
                        "execution_device": policy["reference_device"],
                        "batch_mode": (
                            "per_sample"
                            if model_id
                            in policy["reference_batching"]["per_sample_models"]
                            else policy["reference_batching"]["default"]
                        ),
                    }
                },
                "validation": {
                    "status": "ok",
                    "num_cases": total_cases,
                    "fixed_reference_device": policy["reference_device"],
                    "max_abs_tolerance": policy["same_device_tolerances"][
                        "max_abs"
                    ],
                    "mean_abs_tolerance": policy["same_device_tolerances"][
                        "mean_abs"
                    ],
                    "cross_device_max_abs_tolerance": policy[
                        "cross_device_tolerances"
                    ]["max_abs"],
                    "cross_device_mean_abs_tolerance": policy[
                        "cross_device_tolerances"
                    ]["mean_abs"],
                    "numeric_policy": {
                        **policy["numeric"],
                        "restores_caller_settings": True,
                    },
                    "devices": devices,
                },
            }
            meta = artifact.with_suffix(".pt2.meta.json")
            meta.write_text(json.dumps(metadata), encoding="utf-8")
            results.append(
                {
                    "model_id": model_id,
                    "status": "ok",
                    "validation_status": "ok",
                    "num_cases": total_cases,
                    "artifact": str(artifact),
                    "meta": str(meta),
                    "sha256": digest,
                }
            )
        summary = {
            "status": "ok",
            "torch_minor": cohort,
            "runtime_torch_minor": cohort,
            "validate_devices": ["cpu", "cuda"],
            "batch_sizes": policy["predictor_batch_sizes"],
            "seeds": policy["seeds"],
            "scales": policy["scales"],
            "environment": {
                "torch_version": cohort_record["validated_patch"],
                "cuda_runtime": cohort_record["cuda"]["runtime"],
                "cuda_devices": [{"name": "test GPU"}],
                "export_schema": cohort_record["export_schema"],
                "platform": {"system": "Linux", "machine": "x86_64"},
                "source_tree": {"commit": "a" * 40, "clean": False},
                "environment_lock": {
                    "path": CUDA_ENVIRONMENT_LOCKS[cohort],
                    "sha256": hashlib.sha256(
                        (REPO_ROOT / CUDA_ENVIRONMENT_LOCKS[cohort]).read_bytes()
                    ).hexdigest(),
                },
            },
            "results": results,
        }
        summary_path = cohort_root / f"summary-torch{cohort}.json"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        summaries.append(summary_path)
    return summaries


@pytest.mark.release_blocker
def test_candidate_matrix_requires_every_cohort_model_and_device(tmp_path):
    summaries = _stage_candidate_matrix(tmp_path)
    report = verify_release_matrix(
        staging_root=tmp_path,
        summary_paths=summaries,
        manifest_path=MANIFEST_PATH,
        allow_dirty_source=True,
        require_approval=False,
    )
    assert report["status"] == "ok"
    assert len(report["lanes"]) == 3
    assert all(len(lane["artifacts"]) == 10 for lane in report["lanes"])

    first_summary = _json(summaries[0])
    first_metadata_path = Path(first_summary["results"][0]["meta"])
    original_metadata = first_metadata_path.read_text(encoding="utf-8")
    first_metadata = json.loads(original_metadata)
    cuda_record = next(
        record
        for record in first_metadata["validation"]["devices"]
        if record["device"] == "cuda"
    )
    cuda_record.update({"status": "skipped", "num_cases": 0, "cases": []})
    first_metadata_path.write_text(json.dumps(first_metadata), encoding="utf-8")
    with pytest.raises(ReleaseMatrixError, match="non-ok devices"):
        verify_release_matrix(
            staging_root=tmp_path,
            summary_paths=summaries,
            manifest_path=MANIFEST_PATH,
            allow_dirty_source=True,
            require_approval=False,
        )
    first_metadata_path.write_text(original_metadata, encoding="utf-8")

    with pytest.raises(ReleaseMatrixError, match="missing=.*2.11"):
        verify_release_matrix(
            staging_root=tmp_path,
            summary_paths=summaries[:-1],
            manifest_path=MANIFEST_PATH,
            allow_dirty_source=True,
            require_approval=False,
        )

    with pytest.raises(ReleaseMatrixError, match="approved manifest"):
        verify_release_matrix(
            staging_root=tmp_path,
            summary_paths=summaries,
            manifest_path=MANIFEST_PATH,
        )


def test_environment_metadata_records_schema_lock_source_and_cuda():
    environment = _environment_metadata(REPO_ROOT)
    assert environment["torch_version"]
    assert set(environment["export_schema"]) == {"major", "minor"}
    assert all(isinstance(value, int) for value in environment["export_schema"].values())
    assert len(environment["source_tree"]["commit"]) == 40
    assert environment["environment_lock"]["path"] == "uv.lock"
    assert len(environment["environment_lock"]["sha256"]) == 64
    assert environment["platform"]["system"]


def test_hub_inventory_uses_lfs_sha_and_immutable_revision(tmp_path):
    manifest = _json(MANIFEST_PATH)
    model_id, model = next(iter(manifest["models"].items()))
    one_model_manifest = dict(manifest)
    one_model_manifest["models"] = {model_id: model}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(one_model_manifest), encoding="utf-8")

    metadata_files = {}
    siblings = [SimpleNamespace(rfilename="README.md", size=1, lfs=None)]
    for artifact in model["artifacts"]:
        siblings.append(
            SimpleNamespace(
                rfilename=artifact["filename"],
                size=artifact["size_bytes"],
                lfs=SimpleNamespace(sha256=artifact["sha256"]),
            )
        )
        if artifact.get("validation_metadata"):
            filename = artifact["validation_metadata"]
            siblings.append(SimpleNamespace(rfilename=filename, size=1, lfs=None))
            path = tmp_path / filename
            path.write_text(json.dumps({"legacy": True}), encoding="utf-8")
            metadata_files[filename] = path

    class FakeApi:
        def model_info(self, repo_id, revision, files_metadata):
            assert repo_id == model["repo_id"]
            assert revision == model["revision"]
            assert files_metadata is True
            return SimpleNamespace(sha=revision, siblings=siblings)

    def fake_download(*, repo_id, filename, revision):
        return str(metadata_files[filename])

    report = audit_remote_manifest(
        manifest_path,
        require_current_metadata=False,
        api=FakeApi(),
        download_fn=fake_download,
    )
    assert report["status"] == "ok"
    assert all(
        artifact["lfs_oid_verified"]
        for artifact in report["results"][0]["artifacts"]
    )
