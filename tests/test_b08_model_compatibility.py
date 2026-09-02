import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version
import pytest
import yaml

from facetorch.artifacts import ArtifactManifest, get_model_manifest
from facetorch.exceptions import ModelCompatibilityError
from scripts import audit_model_manifest_hf as hub_audit
from scripts.audit_model_manifest_hf import audit_remote_manifest
from scripts.export_model_cohorts_hf import _environment_metadata, _model_specs
from scripts.render_model_cards import ModelCardError, render_model_documents
from scripts.verify_model_release_matrix import (
    ReleaseMatrixError,
    _require_approved_governance,
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
SUPPORTED_TORCH = {"2.6", "2.7", "2.8", "2.9", "2.10", "2.11", "2.12", "2.13"}
ARTIFACT_COHORTS = {"2.6", "2.11"}
UNSUPPORTED_TORCH = {"2.3", "2.4", "2.5"}
CUDA_ENVIRONMENT_LOCKS = {
    "2.6": "environments/torch-2.6-cu124/uv.lock",
    "2.11": "environments/torch-2.11-cu130/uv.lock",
}


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _nested_mappings(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _nested_mappings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _nested_mappings(child)


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
    assert set(manifest.supported_torch_minors) == SUPPORTED_TORCH
    for model_id, artifacts in manifest.models.items():
        exports = [item for item in artifacts if item.format == "pt2"]
        assert len(exports) == 2, model_id
        assert {item.artifact_cohort for item in exports} == ARTIFACT_COHORTS, model_id
        assert {
            (item.artifact_cohort, item.torch_min, item.torch_max_exclusive)
            for item in exports
        } == {
            ("2.6", "2.6", "2.9"),
            ("2.11", "2.9", "2.14"),
        }, model_id
        assert all(
            item.validation_metadata == f"{item.filename}.meta.json" for item in exports
        )
        assert all(item.torch_min != "2.3" for item in artifacts), model_id


@pytest.mark.release_blocker
@pytest.mark.parametrize(
    ("runtime", "artifact_cohort"),
    [
        ("2.6.0", "2.6"),
        ("2.7.1", "2.6"),
        ("2.8.0", "2.6"),
        ("2.9.1", "2.11"),
        ("2.10.0", "2.11"),
        ("2.11.0", "2.11"),
        ("2.12.1", "2.11"),
        ("2.13.0", "2.11"),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_every_supported_runtime_routes_to_one_declared_artifact_cohort(
    runtime, artifact_cohort, device
):
    manifest = get_model_manifest()
    for model_id in manifest.models:
        candidates = manifest.candidates(
            model_id,
            torch_version=runtime,
            device=device,
        )
        assert len(candidates) == 1, model_id
        assert candidates[0].artifact_cohort == artifact_cohort, model_id


@pytest.mark.release_blocker
@pytest.mark.parametrize("runtime", ["2.5.1", "2.14.0"])
def test_runtimes_outside_the_official_matrix_fail_before_download(runtime):
    with pytest.raises(ModelCompatibilityError, match="no model download"):
        get_model_manifest().candidates(
            "fer-efficientnet-b0",
            torch_version=runtime,
            device="cpu",
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
def test_provenance_and_matrix_approve_every_published_model():
    manifest_raw = _json(MANIFEST_PATH)
    compatibility = _json(COMPATIBILITY_PATH)
    governance = _json(GOVERNANCE_PATH)
    assert set(governance["models"]) == set(manifest_raw["models"])
    assert governance["status"] == "approved"
    assert governance["license_policy"]["status"] == "approved"
    assert (
        "not treated as interchangeable"
        in governance["license_policy"]["no_license_conversion"]
    )
    for model_id, record in governance["models"].items():
        assert record["status"] == "approved", model_id
        assert record["release_eligible"] is True, model_id
        assert record["source_checkpoint"]["upstream_checkpoint_mapping"] == (
            "verified"
        ), model_id
        assert record["source_checkpoint"]["hosted_sha256_verified"] is True, model_id
        assert record["rights"]["weights_license"] in {
            "MIT",
            "Apache-2.0",
        }, model_id
        assert record["rights"]["redistribution"] == "approved", model_id
        assert record["rights"]["attribution"] == "approved", model_id
        assert record["rights"]["owner_approval"] == "approved", model_id
        assert record["limitations"], model_id
        assert record["intended_use"], model_id

    assert manifest_raw["status"] == "approved"
    assert compatibility["status"] == "approved"
    assert compatibility["candidate_evidence"]["status"] == ("validated_clean_commit")
    manifest = ArtifactManifest.from_mapping(
        manifest_raw,
        compatibility=compatibility,
        governance=governance,
    )
    assert manifest.status == "approved"
    assert all(
        descriptor.export_commit == compatibility["candidate_evidence"]["source_commit"]
        for descriptor in manifest.iter_descriptors()
    )


@pytest.mark.release_blocker
@pytest.mark.parametrize(
    ("mode", "value"),
    [
        ("missing", None),
        ("value", False),
        ("value", 1),
        ("value", "true"),
    ],
)
def test_release_matrix_requires_exact_hosted_digest_proof(mode, value):
    manifest = _json(MANIFEST_PATH)
    compatibility = _json(COMPATIBILITY_PATH)
    governance = _json(GOVERNANCE_PATH)
    checkpoint = next(iter(governance["models"].values()))["source_checkpoint"]
    if mode == "missing":
        checkpoint.pop("hosted_sha256_verified")
    else:
        checkpoint["hosted_sha256_verified"] = value

    with pytest.raises(ReleaseMatrixError, match="rights/provenance"):
        _require_approved_governance(manifest, compatibility, governance)


@pytest.mark.release_blocker
def test_export_specs_match_manifest_ids_and_pin_every_source():
    manifest = _json(MANIFEST_PATH)
    compatibility = _json(COMPATIBILITY_PATH)
    assert compatibility["validation_policy"]["golden_reference_cohort"] == "2.6"
    specs = _model_specs("2.11")
    assert {item["id"] for item in specs} == set(manifest["models"])
    assert all(item["strict"] is True for item in specs)
    assert all(
        item["cross_device_tolerances"]["max_abs"] > item["tolerances"]["max_abs"]
        for item in specs
    )
    assert all(item["validation_reference"]["device"] == "cpu" for item in specs)
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


@pytest.mark.release_blocker
def test_manifest_revisions_bind_governance_export_specs_and_all_configs():
    manifest = _json(MANIFEST_PATH)
    governance = _json(GOVERNANCE_PATH)
    specs = {spec["id"]: spec for spec in _model_specs("2.11")}
    for model_id, model in manifest["models"].items():
        revision = model["revision"]
        repo_id = model["repo_id"]
        assert model["license_ref"] == (
            f"https://huggingface.co/{repo_id}/blob/{revision}/LICENSE"
        )
        assert governance["models"][model_id]["hosted_model_card"] == (
            f"https://huggingface.co/{repo_id}/blob/{revision}/README.md"
        )
        assert specs[model_id]["source_artifact"]["revision"] == revision

    seen = set()
    config_roots = [REPO_ROOT / "conf", REPO_ROOT / "facetorch" / "configs"]
    for root in config_roots:
        paths = set(root.rglob("*.yaml")) | set(root.rglob("*.yml"))
        for path in sorted(paths):
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            for mapping in _nested_mappings(config):
                model_id = mapping.get("manifest_id")
                if not model_id:
                    continue
                assert model_id in manifest["models"], path
                model = manifest["models"][model_id]
                assert mapping.get("repo_id") == model["repo_id"], path
                assert mapping.get("revision") == model["revision"], path
                assert not {"filename", "sha256", "size_bytes"} & set(mapping), path
                seen.add(model_id)
    assert seen == set(manifest["models"])


@pytest.mark.release_blocker
def test_published_model_card_usage_matches_artifact_manifest_api():
    parameters = inspect.signature(ArtifactManifest.candidates).parameters
    for name in ("torch_version", "device", "allow_legacy_models"):
        assert name in parameters
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["allow_legacy_models"].default is False
    for documents in render_model_documents().values():
        card = documents["README.md"].decode("utf-8")
        assert "get_model_manifest().candidates(" in card
        assert "torch_version=torch.__version__" in card
        assert "device=device" in card
        assert "allow_legacy_models=False" in card


def _stage_candidate_matrix(tmp_path):
    manifest = _json(MANIFEST_PATH)
    compatibility = _json(COMPATIBILITY_PATH)
    policy = compatibility["validation_policy"]
    summaries = []
    for cohort_record in compatibility["cohorts"]:
        cohort = cohort_record["torch_minor"]
        cohort_root = tmp_path / f"torch-{cohort}"
        environment = {
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
        }
        exporter_arguments = {
            "mode": "export",
            "artifact_cohort": cohort,
            "batch_sizes": policy["predictor_batch_sizes"],
            "seeds": policy["seeds"],
            "scales": policy["scales"],
            "validate_devices": compatibility["platform_policy"]["required_devices"],
            "golden_reference_mode": (
                "record" if cohort == policy["golden_reference_cohort"] else "reuse"
            ),
            "golden_reference_cohort": policy["golden_reference_cohort"],
            "model_ids": list(manifest["models"]),
        }
        results = []
        for model_id, model in manifest["models"].items():
            golden_reference = (
                tmp_path / "golden-references" / model_id / "golden-reference.pt"
            )
            golden_reference.parent.mkdir(parents=True, exist_ok=True)
            if not golden_reference.exists():
                golden_reference.write_bytes(f"golden:{model_id}".encode())
            golden_sha = hashlib.sha256(golden_reference.read_bytes()).hexdigest()
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
            shapes = policy["detector"]["spatial_shapes"] if is_detector else [[8, 8]]
            cases = [
                {
                    "case_id": (
                        f"b{batch}_h{height}_w{width}_seed{seed}_"
                        f"scale{scale}_{variant}"
                    ),
                    "status": "ok",
                    "input_sha256": hashlib.sha256(
                        (
                            f"b{batch}_h{height}_w{width}_seed{seed}_"
                            f"scale{scale}_{variant}"
                        ).encode()
                    ).hexdigest(),
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
                "mode": "export",
                "model_id": model_id,
                "repo_id": model["repo_id"],
                "torch_version": cohort_record["validated_patch"],
                "torch_minor": cohort,
                "runtime_torch_minor": cohort,
                "environment": environment,
                "exporter_arguments": exporter_arguments,
                "artifact": artifact.name,
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
                    "max_abs_tolerance": policy["same_device_tolerances"]["max_abs"],
                    "mean_abs_tolerance": policy["same_device_tolerances"]["mean_abs"],
                    "cross_device_max_abs_tolerance": policy["cross_device_tolerances"][
                        "max_abs"
                    ],
                    "cross_device_mean_abs_tolerance": policy[
                        "cross_device_tolerances"
                    ]["mean_abs"],
                    "numeric_policy": {
                        **policy["numeric"],
                        "restores_caller_settings": True,
                    },
                    "golden_reference": {
                        "schema_version": 1,
                        "status": (
                            "recorded"
                            if cohort == policy["golden_reference_cohort"]
                            else "reused"
                        ),
                        "source_cohort": policy["golden_reference_cohort"],
                        "sha256": golden_sha,
                        "size_bytes": golden_reference.stat().st_size,
                        "case_count": len(cases),
                    },
                    "devices": devices,
                },
            }
            meta = artifact.with_suffix(".pt2.meta.json")
            meta.write_text(json.dumps(metadata), encoding="utf-8")
            results.append(
                {
                    "model_id": model_id,
                    "repo_id": model["repo_id"],
                    "status": "ok",
                    "validation_status": "ok",
                    "num_cases": total_cases,
                    "artifact": str(artifact),
                    "meta": str(meta),
                    "meta_sha256": hashlib.sha256(meta.read_bytes()).hexdigest(),
                    "sha256": digest,
                    "golden_reference": str(golden_reference),
                    "golden_reference_sha256": golden_sha,
                    "golden_reference_size_bytes": golden_reference.stat().st_size,
                }
            )
        summary = {
            "schema_version": 2,
            "status": "ok",
            "mode": "export",
            "torch_version": cohort_record["validated_patch"],
            "torch_minor": cohort,
            "runtime_torch_minor": cohort,
            "validate_devices": ["cpu", "cuda"],
            "requested_model_ids": list(manifest["models"]),
            "batch_sizes": policy["predictor_batch_sizes"],
            "seeds": policy["seeds"],
            "scales": policy["scales"],
            "environment": environment,
            "exporter_arguments": exporter_arguments,
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
    assert len(report["lanes"]) == len(ARTIFACT_COHORTS)
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
    first_summary["results"][0]["meta_sha256"] = hashlib.sha256(
        first_metadata_path.read_bytes()
    ).hexdigest()
    summaries[0].write_text(json.dumps(first_summary), encoding="utf-8")
    with pytest.raises(ReleaseMatrixError, match="non-ok devices"):
        verify_release_matrix(
            staging_root=tmp_path,
            summary_paths=summaries,
            manifest_path=MANIFEST_PATH,
            allow_dirty_source=True,
            require_approval=False,
        )
    first_metadata_path.write_text(original_metadata, encoding="utf-8")
    first_summary["results"][0]["meta_sha256"] = hashlib.sha256(
        first_metadata_path.read_bytes()
    ).hexdigest()
    summaries[0].write_text(json.dumps(first_summary), encoding="utf-8")

    with pytest.raises(ReleaseMatrixError, match="missing=.*2.11"):
        verify_release_matrix(
            staging_root=tmp_path,
            summary_paths=summaries[:-1],
            manifest_path=MANIFEST_PATH,
            allow_dirty_source=True,
            require_approval=False,
        )

    with pytest.raises(ReleaseMatrixError, match="dirty tree"):
        verify_release_matrix(
            staging_root=tmp_path,
            summary_paths=summaries,
            manifest_path=MANIFEST_PATH,
        )


@pytest.mark.release_blocker
@pytest.mark.parametrize("mutation", ("duplicate", "undeclared"))
def test_candidate_matrix_rejects_non_exact_metadata_device_coverage(
    tmp_path, mutation
):
    summaries = _stage_candidate_matrix(tmp_path)
    summary = _json(summaries[0])
    result = summary["results"][0]
    metadata_path = Path(result["meta"])
    metadata = _json(metadata_path)
    copied = dict(metadata["validation"]["devices"][0])
    if mutation == "undeclared":
        copied["device"] = "tpu"
    metadata["validation"]["devices"].append(copied)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    result["meta_sha256"] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    summaries[0].write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ReleaseMatrixError, match="device coverage"):
        verify_release_matrix(
            staging_root=tmp_path,
            summary_paths=summaries,
            manifest_path=MANIFEST_PATH,
            allow_dirty_source=True,
            require_approval=False,
        )


@pytest.mark.release_blocker
def test_candidate_matrix_rejects_undeclared_pt2_manifest_cohort(tmp_path):
    summaries = _stage_candidate_matrix(tmp_path / "staging")
    source_manifest = _json(MANIFEST_PATH)
    model_id = next(iter(source_manifest["models"]))
    extra_artifact = dict(source_manifest["models"][model_id]["artifacts"][0])
    extra_artifact.update(
        {
            "id": f"{model_id}-torch2.99",
            "filename": "model-torch2.99.pt2",
            "torch_min": "2.99",
            "torch_max_exclusive": "2.100",
        }
    )
    source_manifest["models"][model_id]["artifacts"].append(extra_artifact)

    repo = tmp_path / "candidate-repo"
    model_root = repo / "facetorch" / "models"
    model_root.mkdir(parents=True)
    manifest_path = model_root / "manifest.json"
    manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")
    for source in (COMPATIBILITY_PATH, GOVERNANCE_PATH):
        (model_root / source.name).write_bytes(source.read_bytes())
    for relative in CUDA_ENVIRONMENT_LOCKS.values():
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO_ROOT / relative).read_bytes())

    with pytest.raises(ReleaseMatrixError, match="PT2 cohort coverage"):
        verify_release_matrix(
            staging_root=tmp_path / "staging",
            summary_paths=summaries,
            manifest_path=manifest_path,
            allow_dirty_source=True,
            require_approval=False,
        )


@pytest.mark.release_blocker
def test_candidate_matrix_rejects_uncontracted_metadata_filename(tmp_path):
    summaries = _stage_candidate_matrix(tmp_path)
    summary = _json(summaries[0])
    result = summary["results"][0]
    metadata_path = Path(result["meta"])
    renamed = metadata_path.with_name("uncontracted.meta.json")
    renamed.write_bytes(metadata_path.read_bytes())
    result["meta"] = str(renamed)
    summaries[0].write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ReleaseMatrixError, match="metadata filename"):
        verify_release_matrix(
            staging_root=tmp_path,
            summary_paths=summaries,
            manifest_path=MANIFEST_PATH,
            allow_dirty_source=True,
            require_approval=False,
        )


@pytest.mark.release_blocker
@pytest.mark.parametrize(
    ("field_path", "mutated_value"),
    (
        (("schema_version",), 1),
        (("mode",), "validate"),
        (("model_id",), "wrong-model"),
        (("repo_id",), "owner/wrong-repository"),
        (("artifact",), "wrong.pt2"),
        (("torch_version",), "2.5.0"),
        (("torch_minor",), "2.5"),
        (("runtime_torch_minor",), "2.5"),
        (("environment", "source_tree", "commit"), "b" * 40),
        (("exporter_arguments", "artifact_cohort"), "2.5"),
    ),
)
def test_matrix_rejects_each_metadata_identity_mutation(
    tmp_path, field_path, mutated_value
):
    summaries = _stage_candidate_matrix(tmp_path)
    summary = _json(summaries[0])
    result = summary["results"][0]
    metadata_path = Path(result["meta"])
    metadata = _json(metadata_path)
    target = metadata
    for field in field_path[:-1]:
        target = target[field]
    target[field_path[-1]] = mutated_value
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    result["meta_sha256"] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    summaries[0].write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ReleaseMatrixError, match="identity"):
        verify_release_matrix(
            staging_root=tmp_path,
            summary_paths=summaries,
            manifest_path=MANIFEST_PATH,
            allow_dirty_source=True,
            require_approval=False,
        )


@pytest.mark.release_blocker
def test_matrix_requires_one_source_commit_across_cohorts(tmp_path):
    summaries = _stage_candidate_matrix(tmp_path)
    summary = _json(summaries[1])
    summary["environment"]["source_tree"]["commit"] = "b" * 40
    for result in summary["results"]:
        metadata_path = Path(result["meta"])
        metadata = _json(metadata_path)
        metadata["environment"] = summary["environment"]
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        result["meta_sha256"] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    summaries[1].write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ReleaseMatrixError, match="source commit"):
        verify_release_matrix(
            staging_root=tmp_path,
            summary_paths=summaries,
            manifest_path=MANIFEST_PATH,
            allow_dirty_source=True,
            require_approval=False,
        )


def test_environment_metadata_records_schema_lock_source_and_cuda():
    environment = _environment_metadata(REPO_ROOT)
    assert environment["torch_version"]
    assert set(environment["export_schema"]) == {"major", "minor"}
    assert all(
        isinstance(value, int) for value in environment["export_schema"].values()
    )
    assert len(environment["source_tree"]["commit"]) == 40
    assert environment["environment_lock"]["path"] == "uv.lock"
    assert len(environment["environment_lock"]["sha256"]) == 64
    assert environment["platform"]["system"]


def test_hub_inventory_uses_lfs_sha_immutable_revision_and_exact_legal_files(
    tmp_path,
):
    manifest = _json(MANIFEST_PATH)
    model_id, model = next(iter(manifest["models"].items()))
    one_model_manifest = dict(manifest)
    one_model_manifest["models"] = {model_id: model}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(one_model_manifest), encoding="utf-8")

    download_files = {}
    siblings = []
    legal_documents = render_model_documents()[model_id]
    for filename, contents in legal_documents.items():
        path = tmp_path / f"legal-{filename}"
        path.write_bytes(contents)
        download_files[filename] = path
        siblings.append(
            SimpleNamespace(rfilename=filename, size=len(contents), lfs=None)
        )
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
            download_files[filename] = path

    class FakeApi:
        def model_info(self, repo_id, revision, files_metadata):
            assert repo_id == model["repo_id"]
            assert revision == model["revision"]
            assert files_metadata is True
            return SimpleNamespace(sha=revision, siblings=siblings)

    def fake_download(*, repo_id, filename, revision):
        assert repo_id == model["repo_id"]
        assert revision == model["revision"]
        return str(download_files[filename])

    report = audit_remote_manifest(
        manifest_path,
        require_current_metadata=False,
        api=FakeApi(),
        download_fn=fake_download,
    )
    assert report["status"] == "ok"
    assert all(
        artifact["lfs_oid_verified"] for artifact in report["results"][0]["artifacts"]
    )
    assert all(
        document["bytes_verified"]
        for document in report["results"][0]["legal_documents"]
    )

    mutations = {
        "README.md": b"x" * len(legal_documents["README.md"]),
        "LICENSE": b"",
        "THIRD_PARTY_NOTICES.md": b"?" * len(legal_documents["THIRD_PARTY_NOTICES.md"]),
    }
    for filename, mutation in mutations.items():
        path = download_files[filename]
        original = path.read_bytes()
        path.write_bytes(mutation)
        failed = audit_remote_manifest(
            manifest_path,
            require_current_metadata=False,
            api=FakeApi(),
            download_fn=fake_download,
        )
        assert failed["status"] == "failed"
        assert filename in failed["failures"][0]["error"]
        path.write_bytes(original)

    filename = "README.md"
    missing_sibling = next(item for item in siblings if item.rfilename == filename)
    siblings.remove(missing_sibling)
    failed = audit_remote_manifest(
        manifest_path,
        require_current_metadata=False,
        api=FakeApi(),
        download_fn=fake_download,
    )
    assert failed["status"] == "failed"
    assert filename in failed["failures"][0]["error"]
    siblings.append(missing_sibling)

    filename = "LICENSE"
    wrong_size = next(item for item in siblings if item.rfilename == filename)
    original_size = wrong_size.size
    wrong_size.size += 1
    failed = audit_remote_manifest(
        manifest_path,
        require_current_metadata=False,
        api=FakeApi(),
        download_fn=fake_download,
    )
    assert failed["status"] == "failed"
    assert filename in failed["failures"][0]["error"]
    wrong_size.size = original_size


def _exact_remote_audit_fixture(tmp_path):
    staged_root = tmp_path / "staged"
    summaries = _stage_candidate_matrix(staged_root)
    packaged = _json(MANIFEST_PATH)
    model_id = next(iter(packaged["models"]))
    model = json.loads(json.dumps(packaged["models"][model_id]))
    manifest_revision = "c" * 40
    manifest_repo = "owner/model-manifest"
    manifest_filename = "manifests/approved.json"
    metadata_files = {}
    remote_records = []
    for summary_path in summaries:
        summary = _json(summary_path)
        cohort = summary["torch_minor"]
        result = next(
            item for item in summary["results"] if item["model_id"] == model_id
        )
        metadata = _json(Path(result["meta"]))
        artifact = next(
            item
            for item in model["artifacts"]
            if item.get("format") == "pt2" and item.get("artifact_cohort") == cohort
        )
        metadata["artifact"] = artifact["filename"]
        metadata["artifact_sha256"] = artifact["sha256"]
        metadata["artifact_size_bytes"] = artifact["size_bytes"]
        metadata["environment"]["source_tree"] = {
            "commit": model["export_commit"],
            "clean": True,
        }
        metadata["exporter_arguments"]["model_ids"] = [model_id]
        metadata["validation"]["golden_reference"].update(
            {
                "sha256": artifact["golden_reference_sha256"],
                "size_bytes": artifact["golden_reference_size_bytes"],
                "source_cohort": artifact["golden_reference_source_cohort"],
            }
        )
        metadata_path = tmp_path / "remote" / artifact["validation_metadata"]
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        metadata_digest = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
        artifact["metadata_sha256"] = metadata_digest
        metadata_files[artifact["validation_metadata"]] = metadata_path
        remote_records.append(
            {
                "model_id": model_id,
                "repo_id": model["repo_id"],
                "cohort": cohort,
                "revision": model["revision"],
                "artifact_filename": artifact["filename"],
                "artifact_sha256": artifact["sha256"],
                "artifact_size_bytes": artifact["size_bytes"],
                "metadata_filename": artifact["validation_metadata"],
                "metadata_sha256": metadata_digest,
                "golden_reference_sha256": artifact["golden_reference_sha256"],
                "golden_reference_size_bytes": artifact["golden_reference_size_bytes"],
                "golden_reference_source_cohort": artifact[
                    "golden_reference_source_cohort"
                ],
                "required_devices": artifact["devices"],
            }
        )

    remote_manifest = {
        "schema_version": 1,
        "status": "approved",
        "plan_id": "approved-plan",
        "models": remote_records,
    }
    remote_manifest_path = tmp_path / "remote-manifest.json"
    remote_manifest_path.write_text(
        json.dumps(remote_manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    packaged.update(
        {
            "manifest_revision": manifest_revision,
            "manifest_repo_id": manifest_repo,
            "manifest_filename": manifest_filename,
            "manifest_sha256": hashlib.sha256(
                remote_manifest_path.read_bytes()
            ).hexdigest(),
            "models": {model_id: model},
        }
    )
    model_root = tmp_path / "models"
    model_root.mkdir()
    manifest_path = model_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(packaged, sort_keys=True) + "\n", encoding="utf-8"
    )
    (model_root / packaged["compatibility_ref"]).write_bytes(
        COMPATIBILITY_PATH.read_bytes()
    )
    (model_root / packaged["governance_ref"]).write_bytes(GOVERNANCE_PATH.read_bytes())

    downloads = dict(metadata_files)
    siblings = []
    legal_documents = render_model_documents(
        manifest_path, require_complete_contract=False
    )[model_id]
    for filename, contents in legal_documents.items():
        legal_path = tmp_path / "remote" / filename
        legal_path.write_bytes(contents)
        downloads[filename] = legal_path
        siblings.append(
            SimpleNamespace(rfilename=filename, size=len(contents), lfs=None)
        )
    for artifact in model["artifacts"]:
        siblings.append(
            SimpleNamespace(
                rfilename=artifact["filename"],
                size=artifact["size_bytes"],
                lfs=SimpleNamespace(sha256=artifact["sha256"]),
            )
        )
        metadata_filename = artifact.get("validation_metadata")
        if metadata_filename:
            siblings.append(
                SimpleNamespace(
                    rfilename=metadata_filename,
                    size=metadata_files[metadata_filename].stat().st_size,
                    lfs=None,
                )
            )

    class FakeApi:
        def model_info(self, repo_id, revision, files_metadata):
            assert repo_id == model["repo_id"]
            assert revision == model["revision"]
            assert files_metadata is True
            return SimpleNamespace(sha=revision, siblings=siblings)

    def fake_download(*, repo_id, filename, revision):
        assert repo_id == model["repo_id"]
        assert revision == model["revision"]
        return str(downloads[filename])

    return {
        "manifest_path": manifest_path,
        "remote_manifest_path": remote_manifest_path,
        "api": FakeApi(),
        "download_fn": fake_download,
    }


@pytest.mark.release_blocker
def test_exact_hub_audit_binds_remote_metadata_digest_and_identity(tmp_path):
    fixture = _exact_remote_audit_fixture(tmp_path)
    report = audit_remote_manifest(
        fixture["manifest_path"],
        require_remote_manifest=True,
        remote_manifest_path=fixture["remote_manifest_path"],
        api=fixture["api"],
        download_fn=fixture["download_fn"],
    )
    assert report["status"] == "ok"
    assert all(
        artifact["metadata_sha256_verified"] and artifact["metadata_identity_verified"]
        for artifact in report["results"][0]["artifacts"]
        if artifact["metadata_status"] != "not_applicable"
    )

    remote = _json(fixture["remote_manifest_path"])
    remote["models"][0]["metadata_sha256"] = "f" * 64
    fixture["remote_manifest_path"].write_text(
        json.dumps(remote, sort_keys=True) + "\n", encoding="utf-8"
    )
    packaged = _json(fixture["manifest_path"])
    packaged["manifest_sha256"] = hashlib.sha256(
        fixture["remote_manifest_path"].read_bytes()
    ).hexdigest()
    fixture["manifest_path"].write_text(
        json.dumps(packaged, sort_keys=True) + "\n", encoding="utf-8"
    )
    failed = audit_remote_manifest(
        fixture["manifest_path"],
        require_remote_manifest=True,
        remote_manifest_path=fixture["remote_manifest_path"],
        api=fixture["api"],
        download_fn=fixture["download_fn"],
    )
    assert failed["status"] == "failed"
    assert "metadata_sha256" in failed["failures"][0]["error"]


@pytest.mark.release_blocker
def test_hub_audit_reports_a_schema_stable_local_contract_failure(
    tmp_path,
    monkeypatch,
):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"manifest_revision": "schema-test", "models": {}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(hub_audit, "render_model_documents", lambda *_a, **_k: {})
    ok = hub_audit.audit_remote_manifest(
        manifest_path,
        api=object(),
        download_fn=lambda **_kwargs: "unused",
    )
    assert ok["status"] == "ok"

    def fail_render(*_args, **_kwargs):
        raise ModelCardError("deliberate local contract failure")

    monkeypatch.setattr(hub_audit, "render_model_documents", fail_render)
    report = hub_audit.audit_remote_manifest(
        manifest_path,
        api=object(),
        download_fn=lambda **_kwargs: "unused",
    )
    assert report["status"] == "failed"
    assert set(report) == set(ok)
    assert report["results"] == []
    assert report["failures"] == [
        {
            "model_id": "model-card-contract",
            "repo_id": None,
            "error_type": "ModelCardError",
            "error": "deliberate local contract failure",
        }
    ]


@pytest.mark.release_blocker
@pytest.mark.parametrize("contents", [None, "{", "[]"])
def test_hub_audit_reports_manifest_read_failures_with_the_stable_schema(
    contents,
    tmp_path,
):
    manifest_path = tmp_path / "manifest.json"
    if contents is not None:
        manifest_path.write_text(contents, encoding="utf-8")
    report = hub_audit.audit_remote_manifest(
        manifest_path,
        api=object(),
        download_fn=lambda **_kwargs: "unused",
    )
    assert set(report) == {
        "schema_version",
        "status",
        "manifest_revision",
        "packaged_manifest_sha256",
        "remote_manifest",
        "download_artifacts",
        "require_current_metadata",
        "verify_legal_documents",
        "results",
        "failures",
    }
    assert report["status"] == "failed"
    assert report["manifest_revision"] is None
    assert report["packaged_manifest_sha256"] is None
    assert report["remote_manifest"] is None
    assert report["results"] == []
    assert report["failures"][0]["model_id"] == "manifest-contract"
