import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.verify_runtime_compatibility_matrix import (
    CUDA_ENVIRONMENT_LOCKS,
    RuntimeCompatibilityError,
    verify_runtime_matrix,
)

RUNTIME_LANES = {
    "2.6": ("2.6.0", "0.21.0", "2.6", "12.4"),
    "2.7": ("2.7.1", "0.22.1", "2.6", "12.6"),
    "2.8": ("2.8.0", "0.23.0", "2.6", "12.6"),
    "2.9": ("2.9.1", "0.24.1", "2.11", "13.0"),
    "2.10": ("2.10.0", "0.25.0", "2.11", "13.0"),
    "2.11": ("2.11.0", "0.26.0", "2.11", "13.0"),
    "2.12": ("2.12.1", "0.27.1", "2.11", "13.0"),
    "2.13": ("2.13.0", "0.28.0", "2.11", "13.0"),
}


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _stage_runtime_matrix(tmp_path):
    repo = tmp_path / "repo"
    staging = tmp_path / "staging"
    models_root = repo / "facetorch" / "models"
    source_revision = "a" * 40
    source_sha = "b" * 64
    golden = staging / "golden" / "golden-reference.pt"
    golden.parent.mkdir(parents=True)
    golden.write_bytes(b"golden")
    golden_sha = _sha256(golden)

    contracts = {}
    for cohort in ("2.6", "2.11"):
        artifact = (
            staging
            / "artifacts"
            / f"torch-{cohort}"
            / "model-a"
            / (f"model-torch{cohort}.pt2")
        )
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(f"artifact:{cohort}".encode())
        contracts[cohort] = (artifact, _sha256(artifact), artifact.stat().st_size)

    compatibility = {
        "schema_version": 2,
        "status": "approved",
        "torch": {"supported_minor_lines": list(RUNTIME_LANES)},
        "platform_policy": {"required_devices": ["cpu", "cuda"]},
        "validation_policy": {
            "reference_device": "cpu",
            "golden_reference_cohort": "2.6",
            "numeric": {
                "dtype": "float32",
                "cudnn_allow_tf32": False,
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
                "cuda_matmul_allow_tf32": False,
                "float32_matmul_precision": "highest",
            },
            "predictor_batch_sizes": [1, 2],
            "seeds": [0],
            "scales": [1.0],
            "input_variants": ["randn"],
            "detector": {"batch_sizes": [1], "spatial_shapes": [[32, 32]]},
            "same_device_tolerances": {"max_abs": 0.0001, "mean_abs": 0.00001},
            "cross_device_tolerances": {"max_abs": 0.002, "mean_abs": 0.001},
        },
        "cohorts": [
            {
                "artifact_cohort": "2.6",
                "torch_min": "2.6",
                "torch_max_exclusive": "2.9",
            },
            {
                "artifact_cohort": "2.11",
                "torch_min": "2.9",
                "torch_max_exclusive": "2.14",
            },
        ],
        "runtime_lanes": [
            {
                "torch_minor": runtime,
                "validated_patch": patch,
                "torchvision_patch": torchvision,
                "artifact_cohort": cohort,
                "cuda": {"runtime": cuda},
            }
            for runtime, (patch, torchvision, cohort, cuda) in RUNTIME_LANES.items()
        ],
    }
    artifacts = []
    for cohort, (artifact, digest, size) in contracts.items():
        artifacts.append(
            {
                "format": "pt2",
                "artifact_cohort": cohort,
                "filename": artifact.name,
                "sha256": digest,
                "size_bytes": size,
                "torch_min": "2.6" if cohort == "2.6" else "2.9",
                "torch_max_exclusive": "2.9" if cohort == "2.6" else "2.14",
                "golden_reference_sha256": golden_sha,
                "golden_reference_size_bytes": golden.stat().st_size,
            }
        )
    manifest = {
        "compatibility_ref": "runtime-matrix.json",
        "models": {
            "model-a": {
                "repo_id": "owner/model-a",
                "revision": source_revision,
                "source_weight_sha256": source_sha,
                "artifacts": artifacts,
            }
        },
    }
    manifest_path = models_root / "manifest.json"
    compatibility_path = models_root / "runtime-matrix.json"
    _write_json(manifest_path, manifest)
    _write_json(compatibility_path, compatibility)

    lock_digests = {}
    for runtime, relative in CUDA_ENVIRONMENT_LOCKS.items():
        lock = repo / relative
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text(f"lock:{runtime}\n", encoding="utf-8")
        lock_digests[runtime] = _sha256(lock)

    summaries = []
    for runtime, (patch, torchvision, cohort, cuda) in RUNTIME_LANES.items():
        artifact, artifact_sha, artifact_size = contracts[cohort]
        environment = {
            "torch_version": patch,
            "torchvision_version": torchvision,
            "platform": {
                "system": "Linux",
                "release": "test",
                "machine": "x86_64",
            },
            "cuda_runtime": cuda,
            "cuda_devices": [{"name": "test GPU"}],
            "source_tree": {"commit": "c" * 40, "clean": True},
            "environment_lock": {
                "path": CUDA_ENVIRONMENT_LOCKS[runtime],
                "sha256": lock_digests[runtime],
            },
        }
        exporter_arguments = {
            "mode": "validate",
            "artifact_cohort": cohort,
            "batch_sizes": [1, 2],
            "seeds": [0],
            "scales": [1.0],
            "validate_devices": ["cpu", "cuda"],
            "model_ids": ["model-a"],
        }
        cases = [
            {
                "case_id": f"batch-{batch}",
                "status": "ok",
                "input_sha256": hashlib.sha256(f"input:{batch}".encode()).hexdigest(),
                "batch": batch,
                "seed": 0,
                "scale": 1.0,
                "variant": "randn",
                "input_shape": [batch, 3, 16, 16],
            }
            for batch in (1, 2)
        ]
        validation = {
            "status": "ok",
            "num_cases": 4,
            "fixed_reference_device": "cpu",
            "max_abs_tolerance": 0.0001,
            "mean_abs_tolerance": 0.00001,
            "cross_device_max_abs_tolerance": 0.002,
            "cross_device_mean_abs_tolerance": 0.001,
            "numeric_policy": {
                **compatibility["validation_policy"]["numeric"],
                "restores_caller_settings": True,
            },
            "golden_reference": {
                "status": "reused",
                "source_cohort": "2.6",
                "sha256": golden_sha,
                "size_bytes": golden.stat().st_size,
                "case_count": 2,
            },
            "devices": [
                {
                    "device": device,
                    "status": "ok",
                    "num_cases": 2,
                    "cases": copy.deepcopy(cases),
                }
                for device in ("cpu", "cuda")
            ],
        }
        metadata = {
            "schema_version": 2,
            "mode": "validate",
            "model_id": "model-a",
            "repo_id": "owner/model-a",
            "torch_version": patch,
            "torch_minor": cohort,
            "runtime_torch_minor": runtime,
            "environment": environment,
            "exporter_arguments": exporter_arguments,
            "source_artifact": {"revision": source_revision, "sha256": source_sha},
            "artifact": artifact.name,
            "artifact_sha256": artifact_sha,
            "artifact_size_bytes": artifact_size,
            "validation": validation,
        }
        metadata_path = staging / "runtime" / runtime / "model-a.meta.json"
        _write_json(metadata_path, metadata)
        result = {
            "model_id": "model-a",
            "repo_id": "owner/model-a",
            "status": "ok",
            "validation_status": "ok",
            "artifact": str(artifact),
            "meta": str(metadata_path),
            "meta_sha256": _sha256(metadata_path),
            "sha256": artifact_sha,
            "size_bytes": artifact_size,
            "golden_reference": str(golden),
            "golden_reference_sha256": golden_sha,
            "golden_reference_size_bytes": golden.stat().st_size,
            "num_cases": 4,
        }
        summary = {
            "schema_version": 2,
            "status": "ok",
            "mode": "validate",
            "torch_version": patch,
            "torch_minor": cohort,
            "runtime_torch_minor": runtime,
            "validate_devices": ["cpu", "cuda"],
            "requested_model_ids": ["model-a"],
            "batch_sizes": [1, 2],
            "seeds": [0],
            "scales": [1.0],
            "environment": environment,
            "exporter_arguments": exporter_arguments,
            "results": [result],
        }
        summary_path = staging / "runtime" / runtime / "summary.json"
        _write_json(summary_path, summary)
        summaries.append(summary_path)

    return staging, manifest_path, summaries


@pytest.mark.release_blocker
def test_runtime_evidence_covers_all_eight_routes(tmp_path):
    staging, manifest, summaries = _stage_runtime_matrix(tmp_path)
    report = verify_runtime_matrix(
        staging_root=staging,
        summary_paths=summaries,
        manifest_path=manifest,
    )
    assert report["status"] == "ok"
    assert report["source_commit"] == "c" * 40
    assert [lane["torch_minor"] for lane in report["runtime_lanes"]] == list(
        RUNTIME_LANES
    )
    assert [lane["artifact_cohort"] for lane in report["runtime_lanes"]] == [
        "2.6",
        "2.6",
        "2.6",
        "2.11",
        "2.11",
        "2.11",
        "2.11",
        "2.11",
    ]


@pytest.mark.release_blocker
def test_runtime_evidence_rejects_missing_lane(tmp_path):
    staging, manifest, summaries = _stage_runtime_matrix(tmp_path)
    with pytest.raises(RuntimeCompatibilityError, match="summary coverage"):
        verify_runtime_matrix(
            staging_root=staging,
            summary_paths=summaries[:-1],
            manifest_path=manifest,
        )


@pytest.mark.release_blocker
def test_runtime_evidence_rejects_wrong_torchvision_patch(tmp_path):
    staging, manifest, summaries = _stage_runtime_matrix(tmp_path)
    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    summary["environment"]["torchvision_version"] = "99.0.0"
    _write_json(summaries[0], summary)
    with pytest.raises(RuntimeCompatibilityError, match="torchvision patch"):
        verify_runtime_matrix(
            staging_root=staging,
            summary_paths=summaries,
            manifest_path=manifest,
        )


@pytest.mark.release_blocker
def test_runtime_evidence_rejects_incomplete_case_identity(tmp_path):
    staging, manifest, summaries = _stage_runtime_matrix(tmp_path)
    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    result = summary["results"][0]
    metadata_path = Path(result["meta"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["validation"]["devices"][0]["cases"][0]["batch"] = 99
    _write_json(metadata_path, metadata)
    result["meta_sha256"] = _sha256(metadata_path)
    _write_json(summaries[0], summary)
    with pytest.raises(RuntimeCompatibilityError, match="case identities"):
        verify_runtime_matrix(
            staging_root=staging,
            summary_paths=summaries,
            manifest_path=manifest,
        )
