#!/usr/bin/env python3
"""Verify cross-version execution of the declared artifact routing matrix."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

if __package__:
    from scripts.model_evidence_contract import (
        ModelEvidenceContractError,
        expected_metadata_identity,
        validate_metadata_identity,
        validate_summary_identity,
    )
else:
    from model_evidence_contract import (
        ModelEvidenceContractError,
        expected_metadata_identity,
        validate_metadata_identity,
        validate_summary_identity,
    )


class RuntimeCompatibilityError(RuntimeError):
    """Raised when runtime compatibility evidence is incomplete."""


CUDA_ENVIRONMENT_LOCKS = {
    "2.6": "environments/torch-2.6-cu124/uv.lock",
    "2.7": "environments/torch-2.7-cu126/uv.lock",
    "2.8": "environments/torch-2.8-cu126/uv.lock",
    "2.9": "environments/torch-2.9-cu130/uv.lock",
    "2.10": "environments/torch-2.10-cu130/uv.lock",
    "2.11": "environments/torch-2.11-cu130/uv.lock",
    "2.12": "environments/torch-2.12-cu130/uv.lock",
    "2.13": "environments/torch-2.13-cu130/uv.lock",
}


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeCompatibilityError(f"Cannot read JSON evidence {path}.") from exc
    if not isinstance(value, Mapping):
        raise RuntimeCompatibilityError(f"JSON evidence {path} is not an object.")
    return value


def _referenced_json(
    manifest_path: Path, manifest: Mapping[str, Any], field: str
) -> tuple[Path, Mapping[str, Any]]:
    relative = manifest.get(field)
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise RuntimeCompatibilityError(f"Manifest {field} is invalid.")
    path = (manifest_path.parent / relative).resolve()
    try:
        path.relative_to(manifest_path.parent.resolve())
    except ValueError as exc:
        raise RuntimeCompatibilityError(
            f"Manifest {field} escapes its directory."
        ) from exc
    return path, _read_json(path)


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
        raise RuntimeCompatibilityError(
            f"{label} escapes the staging root: {path}"
        ) from exc
    if not path.is_file():
        raise RuntimeCompatibilityError(f"{label} is missing: {path}")
    return path


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


def verify_runtime_matrix(
    *,
    staging_root: Path,
    summary_paths: Sequence[Path],
    manifest_path: Path,
    allow_dirty_source: bool = False,
) -> dict[str, Any]:
    """Verify every supported runtime against its routed immutable artifact."""

    staging_root = staging_root.resolve()
    manifest_path = manifest_path.resolve()
    manifest = _read_json(manifest_path)
    compatibility_path, compatibility = _referenced_json(
        manifest_path, manifest, "compatibility_ref"
    )
    models = manifest.get("models", {})
    if not isinstance(models, Mapping) or not models:
        raise RuntimeCompatibilityError("Manifest has no models.")

    required_devices = tuple(
        compatibility.get("platform_policy", {}).get("required_devices", [])
    )
    if set(required_devices) != {"cpu", "cuda"}:
        raise RuntimeCompatibilityError("Runtime validation requires CPU and CUDA.")

    cohort_records = {
        str(record.get("artifact_cohort")): record
        for record in compatibility.get("cohorts", [])
        if isinstance(record, Mapping)
    }
    runtime_records = {
        str(record.get("torch_minor")): record
        for record in compatibility.get("runtime_lanes", [])
        if isinstance(record, Mapping)
    }
    supported = set(compatibility.get("torch", {}).get("supported_minor_lines", []))
    if (
        not cohort_records
        or not runtime_records
        or set(runtime_records) != supported
        or set(CUDA_ENVIRONMENT_LOCKS) != supported
    ):
        raise RuntimeCompatibilityError("Compatibility runtime declarations differ.")

    artifact_contracts: dict[tuple[str, str], Mapping[str, Any]] = {}
    for model_id, model in models.items():
        artifacts = [
            artifact
            for artifact in model.get("artifacts", [])
            if isinstance(artifact, Mapping) and artifact.get("format") == "pt2"
        ]
        by_cohort = {
            str(artifact.get("artifact_cohort")): artifact for artifact in artifacts
        }
        if set(by_cohort) != set(cohort_records) or len(by_cohort) != len(artifacts):
            raise RuntimeCompatibilityError(
                f"Artifact cohort coverage differs for {model_id}."
            )
        for cohort, artifact in by_cohort.items():
            route = cohort_records[cohort]
            if artifact.get("torch_min") != route.get("torch_min") or artifact.get(
                "torch_max_exclusive"
            ) != route.get("torch_max_exclusive"):
                raise RuntimeCompatibilityError(
                    f"Artifact route differs for {model_id}/{cohort}."
                )
            artifact_contracts[(str(model_id), cohort)] = artifact

    summaries: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    for raw_path in summary_paths:
        path = _staged_path(staging_root, raw_path, "Runtime summary")
        summary = _read_json(path)
        runtime = str(summary.get("runtime_torch_minor", ""))
        if runtime in summaries:
            raise RuntimeCompatibilityError(f"Duplicate summary for Torch {runtime}.")
        summaries[runtime] = (path, summary)
    if set(summaries) != supported:
        raise RuntimeCompatibilityError(
            "Runtime summary coverage differs; "
            f"missing={sorted(supported - set(summaries))}, "
            f"extra={sorted(set(summaries) - supported)}."
        )

    policy = compatibility.get("validation_policy", {})
    expected_batches = list(policy.get("predictor_batch_sizes", []))
    expected_seeds = list(policy.get("seeds", []))
    expected_scales = list(policy.get("scales", []))
    expected_variants = list(policy.get("input_variants", []))
    detector_policy = policy.get("detector", {})
    detector_batches = list(detector_policy.get("batch_sizes", []))
    detector_shapes = [
        tuple(int(value) for value in shape)
        for shape in detector_policy.get("spatial_shapes", [])
    ]
    numeric_policy = policy.get("numeric", {})
    if not all(
        (
            expected_batches,
            expected_seeds,
            expected_scales,
            expected_variants,
            detector_batches,
            detector_shapes,
            numeric_policy,
        )
    ):
        raise RuntimeCompatibilityError("Runtime validation policy is incomplete.")

    source_commits = set()
    report_lanes = []
    for runtime in sorted(
        supported, key=lambda value: tuple(map(int, value.split(".")))
    ):
        summary_path, summary = summaries[runtime]
        lane = runtime_records[runtime]
        cohort = str(lane.get("artifact_cohort", ""))
        if summary.get("status") != "ok" or summary.get("torch_minor") != cohort:
            raise RuntimeCompatibilityError(
                f"Torch {runtime} summary route is invalid."
            )
        try:
            identity = validate_summary_identity(
                summary,
                expected_model_ids=sorted(models),
                expected_devices=required_devices,
                expected_mode="validate",
                require_native_runtime=False,
            )
        except ModelEvidenceContractError as exc:
            raise RuntimeCompatibilityError(
                f"Torch {runtime} summary identity is invalid: {exc}."
            ) from exc
        if identity["runtime_torch_minor"] != runtime:
            raise RuntimeCompatibilityError(
                f"Torch {runtime} runtime identity differs."
            )
        for field, expected in (
            ("batch_sizes", expected_batches),
            ("seeds", expected_seeds),
            ("scales", expected_scales),
        ):
            if summary.get(field) != expected:
                raise RuntimeCompatibilityError(
                    f"Torch {runtime} used an incomplete {field} matrix."
                )

        environment = identity["environment"]
        if str(environment.get("torch_version", "")).split("+", 1)[0] != str(
            lane.get("validated_patch", "")
        ):
            raise RuntimeCompatibilityError(f"Torch {runtime} patch version differs.")
        source_tree = environment.get("source_tree", {})
        source_commit = str(source_tree.get("commit", ""))
        if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
            raise RuntimeCompatibilityError(f"Torch {runtime} source is not immutable.")
        if not allow_dirty_source and source_tree.get("clean") is not True:
            raise RuntimeCompatibilityError(f"Torch {runtime} source tree is dirty.")
        source_commits.add(source_commit)
        lock = environment.get("environment_lock", {})
        expected_lock = CUDA_ENVIRONMENT_LOCKS[runtime]
        lock_path = manifest_path.parents[2] / expected_lock
        if (
            lock.get("path") != expected_lock
            or not lock_path.is_file()
            or lock.get("sha256") != _sha256(lock_path)
        ):
            raise RuntimeCompatibilityError(f"Torch {runtime} lock evidence differs.")
        platform_record = environment.get("platform", {})
        if (
            platform_record.get("system") != "Linux"
            or platform_record.get("machine") != "x86_64"
            or not environment.get("cuda_devices")
            or str(environment.get("cuda_runtime"))
            != str(lane.get("cuda", {}).get("runtime"))
        ):
            raise RuntimeCompatibilityError(
                f"Torch {runtime} platform or CUDA evidence differs."
            )
        if str(environment.get("torchvision_version", "")).split("+", 1)[0] != str(
            lane.get("torchvision_patch", "")
        ):
            raise RuntimeCompatibilityError(
                f"Torch {runtime} torchvision patch version differs."
            )

        results = summary.get("results", [])
        result_by_model = {
            str(result.get("model_id")): result
            for result in results
            if isinstance(result, Mapping)
        }
        if len(results) != len(models) or set(result_by_model) != set(models):
            raise RuntimeCompatibilityError(f"Torch {runtime} model coverage differs.")

        artifacts = []
        for model_id in sorted(models):
            result = result_by_model[model_id]
            model = models[model_id]
            contract = artifact_contracts[(model_id, cohort)]
            if (
                result.get("status") != "ok"
                or result.get("validation_status") != "ok"
                or result.get("repo_id") != model.get("repo_id")
            ):
                raise RuntimeCompatibilityError(
                    f"Torch {runtime} model {model_id} did not validate."
                )
            artifact = _staged_path(staging_root, result.get("artifact"), "Artifact")
            metadata_path = _staged_path(
                staging_root, result.get("meta"), "Validation metadata"
            )
            if (
                artifact.name != contract.get("filename")
                or artifact.stat().st_size != contract.get("size_bytes")
                or _sha256(artifact) != contract.get("sha256")
                or result.get("sha256") != contract.get("sha256")
                or result.get("size_bytes") != contract.get("size_bytes")
                or _sha256(metadata_path) != result.get("meta_sha256")
            ):
                raise RuntimeCompatibilityError(
                    f"Torch {runtime} model {model_id} artifact binding differs."
                )
            metadata = _read_json(metadata_path)
            try:
                expected_identity = expected_metadata_identity(
                    identity,
                    model_id=model_id,
                    repo_id=str(model["repo_id"]),
                    artifact_filename=artifact.name,
                )
                validate_metadata_identity(metadata, expected_identity)
            except ModelEvidenceContractError as exc:
                raise RuntimeCompatibilityError(
                    f"Torch {runtime} model {model_id} metadata differs: {exc}."
                ) from exc
            if metadata.get("artifact_sha256") != contract.get(
                "sha256"
            ) or metadata.get("artifact_size_bytes") != contract.get("size_bytes"):
                raise RuntimeCompatibilityError(
                    f"Torch {runtime} model {model_id} metadata artifact binding differs."
                )
            source_artifact = metadata.get("source_artifact", {})
            if source_artifact.get("revision") != model.get(
                "revision"
            ) or source_artifact.get("sha256") != model.get("source_weight_sha256"):
                raise RuntimeCompatibilityError(
                    f"Torch {runtime} model {model_id} source provenance differs."
                )
            validation = metadata.get("validation", {})
            observed_numeric = validation.get("numeric_policy", {})
            if (
                any(
                    observed_numeric.get(key) != value
                    for key, value in numeric_policy.items()
                )
                or observed_numeric.get("restores_caller_settings") is not True
            ):
                raise RuntimeCompatibilityError(
                    f"Torch {runtime} model {model_id} numeric policy differs."
                )
            if (
                float(validation.get("max_abs_tolerance", -1))
                != float(policy["same_device_tolerances"]["max_abs"])
                or float(validation.get("mean_abs_tolerance", -1))
                != float(policy["same_device_tolerances"]["mean_abs"])
                or float(validation.get("cross_device_max_abs_tolerance", -1))
                != float(policy["cross_device_tolerances"]["max_abs"])
                or float(validation.get("cross_device_mean_abs_tolerance", -1))
                != float(policy["cross_device_tolerances"]["mean_abs"])
            ):
                raise RuntimeCompatibilityError(
                    f"Torch {runtime} model {model_id} tolerances differ."
                )
            if validation.get("fixed_reference_device") != policy.get(
                "reference_device"
            ):
                raise RuntimeCompatibilityError(
                    f"Torch {runtime} model {model_id} reference device differs."
                )
            golden_path = _staged_path(
                staging_root, result.get("golden_reference"), "Golden reference"
            )
            golden = validation.get("golden_reference", {})
            golden_sha = str(result.get("golden_reference_sha256", ""))
            golden_size = int(result.get("golden_reference_size_bytes", -1))
            if (
                golden.get("status") != "reused"
                or golden.get("source_cohort") != policy.get("golden_reference_cohort")
                or golden.get("sha256") != golden_sha
                or int(golden.get("size_bytes", -1)) != golden_size
                or golden_sha != contract.get("golden_reference_sha256")
                or golden_size != contract.get("golden_reference_size_bytes")
                or _sha256(golden_path) != golden_sha
                or golden_path.stat().st_size != golden_size
            ):
                raise RuntimeCompatibilityError(
                    f"Torch {runtime} model {model_id} golden evidence differs."
                )
            devices = validation.get("devices", [])
            device_records = {
                str(record.get("device")): record
                for record in devices
                if isinstance(record, Mapping)
            }
            if len(devices) != len(required_devices) or set(device_records) != set(
                required_devices
            ):
                raise RuntimeCompatibilityError(
                    f"Torch {runtime} model {model_id} device coverage differs."
                )
            is_detector = model_id == "detector-retinaface"
            batches = detector_batches if is_detector else expected_batches
            shapes = detector_shapes if is_detector else None
            case_shapes = shapes if shapes is not None else [None]
            expected_cases = len(
                list(
                    itertools.product(
                        batches,
                        case_shapes,
                        expected_seeds,
                        expected_scales,
                        expected_variants,
                    )
                )
            )
            for device, record in device_records.items():
                cases = record.get("cases", [])
                if (
                    record.get("status") != "ok"
                    or record.get("num_cases") != expected_cases
                    or len(cases) != expected_cases
                    or any(case.get("status") != "ok" for case in cases)
                ):
                    raise RuntimeCompatibilityError(
                        f"Torch {runtime} model {model_id} has incomplete {device} cases."
                    )
                observed_shapes = {
                    tuple(int(value) for value in case.get("input_shape", [])[-2:])
                    for case in cases
                }
                if shapes is None:
                    if len(observed_shapes) != 1:
                        raise RuntimeCompatibilityError(
                            f"Torch {runtime} model {model_id} changed fixed input shape."
                        )
                    identity_shapes = sorted(observed_shapes)
                else:
                    if observed_shapes != set(shapes):
                        raise RuntimeCompatibilityError(
                            f"Torch {runtime} detector spatial coverage differs."
                        )
                    identity_shapes = shapes
                expected_identities = set(
                    itertools.product(
                        batches,
                        identity_shapes,
                        expected_seeds,
                        expected_scales,
                        expected_variants,
                    )
                )
                observed_identities = {
                    (
                        int(case.get("batch", -1)),
                        tuple(int(value) for value in case.get("input_shape", [])[-2:]),
                        int(case.get("seed", -1)),
                        float(case.get("scale", -1)),
                        str(case.get("variant", "")),
                    )
                    for case in cases
                }
                if observed_identities != expected_identities or any(
                    re.fullmatch(r"[0-9a-f]{64}", str(case.get("input_sha256", "")))
                    is None
                    for case in cases
                ):
                    raise RuntimeCompatibilityError(
                        f"Torch {runtime} model {model_id} has incomplete {device} "
                        "case identities."
                    )
            total_cases = expected_cases * len(required_devices)
            if (
                result.get("num_cases") != total_cases
                or validation.get("num_cases") != total_cases
                or validation.get("status") != "ok"
                or int(golden.get("case_count", -1)) != expected_cases
            ):
                raise RuntimeCompatibilityError(
                    f"Torch {runtime} model {model_id} case count differs."
                )
            artifacts.append(
                {
                    "model_id": model_id,
                    "artifact_cohort": cohort,
                    "sha256": contract["sha256"],
                    "size_bytes": contract["size_bytes"],
                    "num_cases": total_cases,
                }
            )

        report_lanes.append(
            {
                "torch_minor": runtime,
                "artifact_cohort": cohort,
                "torch_version": identity["torch_version"],
                "torchvision_version": lane["torchvision_patch"],
                "cuda_runtime": environment["cuda_runtime"],
                "source_commit": source_commit,
                "source_clean": source_tree.get("clean") is True,
                "summary": summary_path.relative_to(staging_root).as_posix(),
                "summary_sha256": _sha256(summary_path),
                "artifacts": artifacts,
            }
        )

    if len(source_commits) != 1:
        raise RuntimeCompatibilityError("Runtime lanes do not share one source commit.")
    return {
        "schema_version": 1,
        "status": "ok",
        "source_commit": next(iter(source_commits)),
        "required_devices": list(required_devices),
        "contracts": {
            "manifest": {
                "path": str(manifest_path),
                "sha256": _sha256(manifest_path),
            },
            "compatibility": {
                "path": str(compatibility_path),
                "sha256": _sha256(compatibility_path),
            },
        },
        "runtime_lanes": report_lanes,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--summary", action="append", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("facetorch/models/manifest.json"),
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--allow-dirty-source", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = verify_runtime_matrix(
        staging_root=args.staging_root,
        summary_paths=args.summary,
        manifest_path=args.manifest,
        allow_dirty_source=args.allow_dirty_source,
    )
    _write_json_atomic(args.report.resolve(), report)
    print(f"Runtime compatibility report: {args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
