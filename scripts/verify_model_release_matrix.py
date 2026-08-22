#!/usr/bin/env python3
"""Verify that staged model cohorts satisfy the declared release matrix."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


class ReleaseMatrixError(RuntimeError):
    """Raised when staged cohort evidence is incomplete or inconsistent."""


CUDA_ENVIRONMENT_LOCKS = {
    "2.3": "environments/torch-2.3-cu121/uv.lock",
    "2.6": "environments/torch-2.6-cu124/uv.lock",
    "2.11": "environments/torch-2.11-cu130/uv.lock",
}


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseMatrixError(f"Cannot read JSON evidence {path}.") from exc
    if not isinstance(value, Mapping):
        raise ReleaseMatrixError(f"JSON evidence {path} is not an object.")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _staged_path(staging_root: Path, value: Any, label: str) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = staging_root / path
    path = path.resolve()
    try:
        path.relative_to(staging_root)
    except ValueError as exc:
        raise ReleaseMatrixError(f"{label} escapes the staging root: {path}") from exc
    if not path.is_file():
        raise ReleaseMatrixError(f"{label} is missing: {path}")
    return path


def _referenced_json(manifest_path: Path, manifest: Mapping[str, Any], field: str):
    filename = str(manifest.get(field, ""))
    if Path(filename).name != filename or not filename.endswith(".json"):
        raise ReleaseMatrixError(f"Manifest has invalid {field}: {filename!r}")
    return _read_json(manifest_path.parent / filename)


def _require_approved_governance(
    manifest: Mapping[str, Any],
    compatibility: Mapping[str, Any],
    governance: Mapping[str, Any],
) -> None:
    if manifest.get("status") != "approved":
        raise ReleaseMatrixError("Release verification requires an approved manifest.")
    if compatibility.get("status") != "approved":
        raise ReleaseMatrixError(
            "Release verification requires an approved compatibility matrix."
        )
    if governance.get("status") != "approved":
        raise ReleaseMatrixError("Release verification requires approved governance.")

    incomplete = []
    for model_id, record in governance.get("models", {}).items():
        rights = record.get("rights", {})
        checkpoint = record.get("source_checkpoint", {})
        if not (
            record.get("status") == "approved"
            and record.get("release_eligible") is True
            and checkpoint.get("upstream_checkpoint_mapping") == "verified"
            and rights.get("weights_license") not in {None, "unverified"}
            and rights.get("redistribution") == "approved"
            and rights.get("attribution") == "approved"
            and rights.get("owner_approval") == "approved"
        ):
            incomplete.append(model_id)
    if incomplete:
        raise ReleaseMatrixError(
            "Models lack approved rights/provenance: " + ", ".join(sorted(incomplete))
        )


def verify_release_matrix(
    *,
    staging_root: Path,
    summary_paths: Sequence[Path],
    manifest_path: Path,
    allow_dirty_source: bool = False,
    require_approval: bool = True,
) -> dict[str, Any]:
    """Verify every model, cohort, device, artifact, and metadata record."""
    staging_root = staging_root.resolve()
    manifest_path = manifest_path.resolve()
    manifest = _read_json(manifest_path)
    compatibility = _referenced_json(
        manifest_path, manifest, "compatibility_ref"
    )
    governance = _referenced_json(manifest_path, manifest, "governance_ref")
    if require_approval:
        _require_approved_governance(manifest, compatibility, governance)

    models = manifest.get("models", {})
    if not isinstance(models, Mapping) or not models:
        raise ReleaseMatrixError("Manifest has no models.")
    expected_models = set(models)
    cohort_records = {
        str(item["torch_minor"]): item for item in compatibility.get("cohorts", [])
    }
    expected_cohorts = set(
        compatibility.get("torch", {}).get("supported_minor_lines", [])
    )
    if not expected_cohorts or expected_cohorts != set(cohort_records):
        raise ReleaseMatrixError("Compatibility cohort declarations are inconsistent.")
    required_devices = tuple(
        compatibility.get("platform_policy", {}).get("required_devices", [])
    )
    if not required_devices:
        raise ReleaseMatrixError("Compatibility matrix declares no required devices.")
    validation_policy = compatibility.get("validation_policy", {})
    expected_batches = list(validation_policy.get("predictor_batch_sizes", []))
    expected_seeds = list(validation_policy.get("seeds", []))
    expected_scales = list(validation_policy.get("scales", []))
    expected_variants = list(validation_policy.get("input_variants", []))
    detector_policy = validation_policy.get("detector", {})
    expected_detector_batches = list(detector_policy.get("batch_sizes", []))
    expected_detector_shapes = [
        tuple(int(value) for value in shape)
        for shape in detector_policy.get("spatial_shapes", [])
    ]
    expected_numeric_policy = validation_policy.get("numeric", {})
    reference_batching = validation_policy.get("reference_batching", {})
    per_sample_reference_models = set(
        reference_batching.get("per_sample_models", [])
    )
    if not (
        expected_batches
        and expected_seeds
        and expected_scales
        and expected_variants
        and expected_detector_batches
        and expected_detector_shapes
        and expected_numeric_policy
        and validation_policy.get("reference_device")
    ):
        raise ReleaseMatrixError("Compatibility validation policy is incomplete.")

    summaries: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    for raw_path in summary_paths:
        path = _staged_path(staging_root, raw_path, "Summary")
        summary = _read_json(path)
        cohort = str(summary.get("torch_minor", ""))
        if cohort in summaries:
            raise ReleaseMatrixError(f"Duplicate summary for torch {cohort}.")
        summaries[cohort] = (path, summary)

    missing_cohorts = sorted(expected_cohorts - set(summaries))
    extra_cohorts = sorted(set(summaries) - expected_cohorts)
    if missing_cohorts or extra_cohorts:
        raise ReleaseMatrixError(
            f"Cohort summary mismatch; missing={missing_cohorts}, extra={extra_cohorts}."
        )

    lanes = []
    for cohort in sorted(expected_cohorts, key=lambda item: tuple(map(int, item.split(".")))):
        _summary_path, summary = summaries[cohort]
        record = cohort_records[cohort]
        if summary.get("status") != "ok":
            raise ReleaseMatrixError(f"Torch {cohort} summary is not ok.")
        if summary.get("runtime_torch_minor") != cohort:
            raise ReleaseMatrixError(
                f"Torch {cohort} artifact was validated on runtime "
                f"{summary.get('runtime_torch_minor')!r}."
            )
        if set(summary.get("validate_devices", [])) != set(required_devices):
            raise ReleaseMatrixError(
                f"Torch {cohort} did not request every required device."
            )
        for field, expected in (
            ("batch_sizes", expected_batches),
            ("seeds", expected_seeds),
            ("scales", expected_scales),
        ):
            if summary.get(field) != expected:
                raise ReleaseMatrixError(
                    f"Torch {cohort} used an incomplete {field} validation matrix."
                )

        environment = summary.get("environment", {})
        expected_schema = record.get("export_schema")
        if environment.get("export_schema") != expected_schema:
            raise ReleaseMatrixError(
                f"Torch {cohort} export schema disagrees with the matrix."
            )
        source_tree = environment.get("source_tree", {})
        if not re.fullmatch(r"[0-9a-f]{40}", str(source_tree.get("commit", ""))):
            raise ReleaseMatrixError(f"Torch {cohort} lacks an immutable source commit.")
        if not allow_dirty_source and source_tree.get("clean") is not True:
            raise ReleaseMatrixError(f"Torch {cohort} was produced from a dirty tree.")
        lock = environment.get("environment_lock") or {}
        if not re.fullmatch(r"[0-9a-f]{64}", str(lock.get("sha256", ""))):
            raise ReleaseMatrixError(f"Torch {cohort} lacks an environment-lock digest.")
        expected_lock_relative = CUDA_ENVIRONMENT_LOCKS.get(cohort)
        if lock.get("path") != expected_lock_relative:
            raise ReleaseMatrixError(
                f"Torch {cohort} evidence does not identify its exact CUDA lock."
            )
        expected_lock_path = manifest_path.parents[2] / expected_lock_relative
        if not expected_lock_path.is_file() or _sha256(expected_lock_path) != lock.get(
            "sha256"
        ):
            raise ReleaseMatrixError(
                f"Torch {cohort} environment-lock digest does not match the source tree."
            )
        if environment.get("platform", {}).get("system") != "Linux" or environment.get(
            "platform", {}
        ).get("machine") != "x86_64":
            raise ReleaseMatrixError(f"Torch {cohort} used an undeclared platform.")
        if not environment.get("cuda_devices"):
            raise ReleaseMatrixError(f"Torch {cohort} has no CUDA device attestation.")
        if str(environment.get("cuda_runtime")) != str(record["cuda"]["runtime"]):
            raise ReleaseMatrixError(f"Torch {cohort} used an undeclared CUDA runtime.")

        results = summary.get("results", [])
        result_by_model = {
            str(item.get("model_id")): item
            for item in results
            if isinstance(item, Mapping)
        }
        if set(result_by_model) != expected_models or len(results) != len(expected_models):
            raise ReleaseMatrixError(
                f"Torch {cohort} model coverage differs from the manifest."
            )

        lane_artifacts = []
        for model_id in sorted(expected_models):
            result = result_by_model[model_id]
            if (
                result.get("status") != "ok"
                or result.get("validation_status") != "ok"
                or int(result.get("num_cases", 0)) <= 0
            ):
                raise ReleaseMatrixError(
                    f"Torch {cohort} model {model_id} did not validate completely."
                )
            artifact = _staged_path(staging_root, result.get("artifact"), "Artifact")
            metadata_path = _staged_path(staging_root, result.get("meta"), "Metadata")
            if _sha256(artifact) != result.get("sha256"):
                raise ReleaseMatrixError(
                    f"Torch {cohort} model {model_id} artifact digest changed."
                )
            metadata = _read_json(metadata_path)
            if (
                metadata.get("artifact_sha256") != result.get("sha256")
                or int(metadata.get("artifact_size_bytes", -1)) != artifact.stat().st_size
            ):
                raise ReleaseMatrixError(
                    f"Torch {cohort} model {model_id} metadata does not bind its artifact."
                )
            source_artifact = metadata.get("source_artifact", {})
            model_record = models[model_id]
            if (
                source_artifact.get("revision") != model_record.get("revision")
                or source_artifact.get("sha256")
                != model_record.get("source_weight_sha256")
            ):
                raise ReleaseMatrixError(
                    f"Torch {cohort} model {model_id} source provenance disagrees."
                )
            validation = metadata.get("validation", {})
            if validation.get("fixed_reference_device") != validation_policy.get(
                "reference_device"
            ):
                raise ReleaseMatrixError(
                    f"Torch {cohort} model {model_id} used the wrong golden device."
                )
            numeric_policy = validation.get("numeric_policy", {})
            if any(
                numeric_policy.get(key) != value
                for key, value in expected_numeric_policy.items()
            ):
                raise ReleaseMatrixError(
                    f"Torch {cohort} model {model_id} used the wrong numeric policy."
                )
            if (
                float(validation.get("max_abs_tolerance", -1))
                != float(validation_policy["same_device_tolerances"]["max_abs"])
                or float(validation.get("mean_abs_tolerance", -1))
                != float(validation_policy["same_device_tolerances"]["mean_abs"])
                or float(validation.get("cross_device_max_abs_tolerance", -1))
                != float(validation_policy["cross_device_tolerances"]["max_abs"])
                or float(validation.get("cross_device_mean_abs_tolerance", -1))
                != float(validation_policy["cross_device_tolerances"]["mean_abs"])
            ):
                raise ReleaseMatrixError(
                    f"Torch {cohort} model {model_id} used undeclared tolerances."
                )
            reference = metadata.get("source", {}).get("validation_reference", {})
            expected_batch_mode = (
                "per_sample"
                if model_id in per_sample_reference_models
                else reference_batching.get("default")
            )
            if (
                reference.get("execution_device")
                != validation_policy.get("reference_device")
                or reference.get("batch_mode") != expected_batch_mode
            ):
                raise ReleaseMatrixError(
                    f"Torch {cohort} model {model_id} used the wrong golden policy."
                )
            device_records = {
                item.get("device"): item for item in validation.get("devices", [])
            }
            non_ok = [
                device
                for device in required_devices
                if device_records.get(device, {}).get("status") != "ok"
                or int(device_records.get(device, {}).get("num_cases", 0)) <= 0
            ]
            if non_ok:
                raise ReleaseMatrixError(
                    f"Torch {cohort} model {model_id} has non-ok devices: {non_ok}."
                )

            is_detector = model_id == "detector-retinaface"
            batches = expected_detector_batches if is_detector else expected_batches
            shapes = expected_detector_shapes if is_detector else None
            expected_cases_per_device = (
                len(batches)
                * (len(shapes) if shapes is not None else 1)
                * len(expected_seeds)
                * len(expected_scales)
                * len(expected_variants)
            )
            expected_total_cases = expected_cases_per_device * len(required_devices)
            if (
                int(validation.get("num_cases", -1)) != expected_total_cases
                or int(result.get("num_cases", -1)) != expected_total_cases
            ):
                raise ReleaseMatrixError(
                    f"Torch {cohort} model {model_id} has the wrong case count."
                )

            for device in required_devices:
                device_record = device_records[device]
                cases = device_record.get("cases", [])
                if (
                    int(device_record.get("num_cases", -1))
                    != expected_cases_per_device
                    or len(cases) != expected_cases_per_device
                    or any(case.get("status") != "ok" for case in cases)
                ):
                    raise ReleaseMatrixError(
                        f"Torch {cohort} model {model_id} has incomplete {device} cases."
                    )
                observed_shapes = {
                    tuple(int(value) for value in case.get("input_shape", [])[-2:])
                    for case in cases
                }
                if shapes is None:
                    if len(observed_shapes) != 1:
                        raise ReleaseMatrixError(
                            f"Torch {cohort} model {model_id} changed fixed input shape."
                        )
                    case_shapes = sorted(observed_shapes)
                else:
                    if observed_shapes != set(shapes):
                        raise ReleaseMatrixError(
                            f"Torch {cohort} detector spatial coverage is incomplete."
                        )
                    case_shapes = shapes
                expected_identities = set(
                    itertools.product(
                        batches,
                        case_shapes,
                        expected_seeds,
                        expected_scales,
                        expected_variants,
                    )
                )
                observed_identities = {
                    (
                        int(case.get("batch", -1)),
                        tuple(
                            int(value)
                            for value in case.get("input_shape", [])[-2:]
                        ),
                        int(case.get("seed", -1)),
                        float(case.get("scale", -1)),
                        str(case.get("variant", "")),
                    )
                    for case in cases
                }
                if observed_identities != expected_identities:
                    raise ReleaseMatrixError(
                        f"Torch {cohort} model {model_id} has incomplete {device} "
                        "case identities."
                    )
            lane_artifacts.append(
                {
                    "model_id": model_id,
                    "sha256": result["sha256"],
                    "size_bytes": artifact.stat().st_size,
                    "num_cases": int(result["num_cases"]),
                }
            )

        lanes.append(
            {
                "torch_minor": cohort,
                "torch_version": environment["torch_version"],
                "cuda_runtime": environment["cuda_runtime"],
                "source_commit": source_tree["commit"],
                "source_clean": source_tree.get("clean") is True,
                "artifacts": lane_artifacts,
            }
        )

    return {
        "schema_version": 1,
        "status": "ok",
        "release_approval_required": not require_approval,
        "manifest_revision": manifest.get("manifest_revision"),
        "required_devices": list(required_devices),
        "lanes": lanes,
    }


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging-root", required=True)
    parser.add_argument("--summary", action="append", required=True)
    parser.add_argument(
        "--manifest", default="facetorch/models/manifest.json"
    )
    parser.add_argument(
        "--candidate-evidence",
        action="store_true",
        help="Verify completeness without claiming manifest/governance approval.",
    )
    parser.add_argument(
        "--allow-dirty-source",
        action="store_true",
        help="Non-release diagnostic only; a release must use a clean commit.",
    )
    parser.add_argument("--report")
    args = parser.parse_args()

    try:
        report = verify_release_matrix(
            staging_root=Path(args.staging_root),
            summary_paths=[Path(item) for item in args.summary],
            manifest_path=Path(args.manifest),
            allow_dirty_source=args.allow_dirty_source,
            require_approval=not args.candidate_evidence,
        )
    except ReleaseMatrixError as exc:
        parser.error(str(exc))

    if args.report:
        _write_json_atomic(Path(args.report), report)
    print(
        f"Verified {len(report['lanes'])} cohorts across "
        f"{len(report['lanes'][0]['artifacts'])} models."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
