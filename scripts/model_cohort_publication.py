#!/usr/bin/env python3
"""Prepare and publish reviewed facetorch model-cohort candidates.

Export and validation are intentionally handled by ``export_model_cohorts_hf.py``.
This module accepts only complete, successful staging summaries. It binds every
local byte and immutable parent revision into a deterministic plan, requires a
separate approval for that exact plan, commits artifact plus metadata atomically
per model repository, and publishes the manifest commit only after every model
commit succeeds.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

PLAN_SCHEMA_VERSION = 3
APPROVAL_SCHEMA_VERSION = 1
RECEIPT_SCHEMA_VERSION = 1
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


class PublicationError(RuntimeError):
    """Raised when a cohort publication safety condition is not satisfied."""


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as output:
            output.write(_canonical_json_bytes(value))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PublicationError(f"Cannot read JSON document {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PublicationError(f"JSON document must contain an object: {path}")
    return value


def _require_commit(value: Any, label: str) -> str:
    revision = str(value).lower()
    if _COMMIT_PATTERN.fullmatch(revision) is None:
        raise PublicationError(f"{label} must be an immutable 40-character commit")
    return revision


def _require_finite_nonnegative(
    value: Any, label: str
) -> float:
    """Return one JSON number after rejecting omissions and unsafe values."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PublicationError(f"{label} must be a finite nonnegative number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise PublicationError(f"{label} must be a finite nonnegative number")
    return number


def _require_positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PublicationError(f"{label} must be a positive integer")
    return value


def _cohort_key(value: Any):
    text = str(value)
    if re.fullmatch(r"\d+\.\d+", text) is None:
        raise PublicationError(f"Invalid PyTorch cohort: {text!r}")
    return tuple(int(part) for part in text.split("."))


def _safe_staged_file(staging_root: Path, value: Any, label: str) -> Path:
    root = staging_root.resolve()
    candidate = Path(str(value))
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise PublicationError(f"{label} is outside staging root: {candidate}") from exc
    if candidate.is_symlink() or not resolved.is_file():
        raise PublicationError(
            f"{label} must be a regular, non-symlink file: {candidate}"
        )
    return resolved


def _relative_path(staging_root: Path, path: Path) -> str:
    return path.resolve().relative_to(staging_root.resolve()).as_posix()


def _base_revision(
    base_revisions: Mapping[str, Any], model_id: str, repo_id: str
) -> str:
    value = base_revisions.get(model_id, base_revisions.get(repo_id))
    if value is None:
        raise PublicationError(
            f"No immutable parent revision declared for {model_id} ({repo_id})"
        )
    return _require_commit(value, f"Parent revision for {model_id}")


def _validated_model_record(
    staging_root: Path,
    summary: Mapping[str, Any],
    result: Mapping[str, Any],
    base_revisions: Mapping[str, Any],
) -> Dict[str, Any]:
    model_id = str(result.get("model_id", ""))
    repo_id = str(result.get("repo_id", ""))
    cohort = str(summary.get("torch_minor", ""))
    if not model_id or not repo_id or not cohort:
        raise PublicationError(
            "Staging result lacks model, repository, or cohort identity"
        )
    if result.get("status") != "ok" or result.get("validation_status") != "ok":
        raise PublicationError(f"Model {model_id} does not have an ok staging result")
    if int(result.get("num_cases", 0)) < 1:
        raise PublicationError(f"Model {model_id} executed zero validation cases")

    artifact = _safe_staged_file(staging_root, result.get("artifact"), "artifact")
    metadata = _safe_staged_file(staging_root, result.get("meta"), "metadata")
    observed_artifact_sha = _sha256(artifact)
    observed_metadata_sha = _sha256(metadata)
    if observed_artifact_sha != str(result.get("sha256", "")):
        raise PublicationError(f"Staged artifact digest changed for {model_id}")
    if observed_metadata_sha != str(result.get("meta_sha256", "")):
        raise PublicationError(f"Staged metadata digest changed for {model_id}")
    if artifact.stat().st_size != int(result.get("size_bytes", -1)):
        raise PublicationError(f"Staged artifact size changed for {model_id}")

    golden_reference = _safe_staged_file(
        staging_root, result.get("golden_reference"), "golden reference"
    )
    observed_golden_sha = _sha256(golden_reference)
    observed_golden_size = golden_reference.stat().st_size
    if observed_golden_sha != str(result.get("golden_reference_sha256", "")):
        raise PublicationError(f"Golden reference digest changed for {model_id}")
    if observed_golden_size != int(result.get("golden_reference_size_bytes", -1)):
        raise PublicationError(f"Golden reference size changed for {model_id}")

    metadata_value = _read_json(metadata)
    validation = metadata_value.get("validation")
    if not isinstance(validation, dict) or validation.get("status") != "ok":
        raise PublicationError(f"Metadata validation is not ok for {model_id}")
    requested_value = validation.get(
        "requested_devices", summary.get("validate_devices", [])
    )
    if not isinstance(requested_value, list):
        raise PublicationError(f"Validation device matrix is invalid for {model_id}")
    requested_devices = [
        str(device).strip().lower() for device in requested_value
    ]
    if (
        not requested_devices
        or any(not device for device in requested_devices)
        or len(set(requested_devices)) != len(requested_devices)
    ):
        raise PublicationError(f"Validation device matrix is invalid for {model_id}")

    device_results = validation.get("devices")
    if not isinstance(device_results, list):
        raise PublicationError(f"Validation device matrix is invalid for {model_id}")
    devices_by_name: Dict[str, Mapping[str, Any]] = {}
    for device in device_results:
        if not isinstance(device, dict):
            raise PublicationError(f"Validation device matrix is invalid for {model_id}")
        device_name = str(device.get("device", "")).strip().lower()
        if not device_name or device_name in devices_by_name:
            raise PublicationError(f"Validation device matrix is invalid for {model_id}")
        devices_by_name[device_name] = device
    if set(devices_by_name) != set(requested_devices):
        raise PublicationError(f"Validation device matrix is incomplete for {model_id}")
    non_ok = [
        device
        for device in requested_devices
        if devices_by_name[device].get("status") != "ok"
    ]
    if non_ok:
        raise PublicationError(
            f"Required validation devices are not ok for {model_id}: {non_ok}"
        )
    if metadata_value.get("artifact_sha256") != observed_artifact_sha:
        raise PublicationError(f"Metadata artifact digest disagrees for {model_id}")
    if int(metadata_value.get("artifact_size_bytes", -1)) != artifact.stat().st_size:
        raise PublicationError(f"Metadata artifact size disagrees for {model_id}")
    golden_metadata = validation.get("golden_reference")
    if not isinstance(golden_metadata, dict) or golden_metadata.get("status") not in {
        "recorded",
        "reused",
    }:
        raise PublicationError(f"Persistent golden reference is missing for {model_id}")
    if (
        golden_metadata.get("sha256") != observed_golden_sha
        or int(golden_metadata.get("size_bytes", -1)) != observed_golden_size
        or int(golden_metadata.get("case_count", 0)) < 1
        or not str(golden_metadata.get("source_cohort", ""))
    ):
        raise PublicationError(f"Golden reference metadata disagrees for {model_id}")
    golden_source_cohort = str(golden_metadata["source_cohort"])
    expected_golden_status = "recorded" if cohort == golden_source_cohort else "reused"
    if golden_metadata["status"] != expected_golden_status:
        raise PublicationError(
            f"Golden reference status disagrees with its source cohort for {model_id}"
        )
    reference_device = str(validation.get("fixed_reference_device", "")).strip().lower()
    if not reference_device or reference_device not in requested_devices:
        raise PublicationError(f"Golden reference device is invalid for {model_id}")

    max_abs_tolerance = _require_finite_nonnegative(
        validation.get("max_abs_tolerance"),
        f"Same-device maximum tolerance for {model_id}",
    )
    mean_abs_tolerance = _require_finite_nonnegative(
        validation.get("mean_abs_tolerance"),
        f"Same-device mean tolerance for {model_id}",
    )
    cross_device_max_abs_tolerance = _require_finite_nonnegative(
        validation.get("cross_device_max_abs_tolerance"),
        f"Cross-device maximum tolerance for {model_id}",
    )
    cross_device_mean_abs_tolerance = _require_finite_nonnegative(
        validation.get("cross_device_mean_abs_tolerance"),
        f"Cross-device mean tolerance for {model_id}",
    )
    validation_failures = validation.get("failures")
    if not isinstance(validation_failures, list) or validation_failures:
        raise PublicationError(f"Validation failures are inconsistent for {model_id}")

    validation_cases: Dict[str, Dict[str, Any]] = {}
    total_cases = 0
    global_worst: Optional[tuple[str, str, float, float]] = None
    for device_name in requested_devices:
        device = devices_by_name[device_name]
        cases = device.get("cases")
        if not isinstance(cases, list) or not cases:
            raise PublicationError(
                f"Validation cases are missing for {model_id} on {device_name}"
            )
        if device.get("failures") != []:
            raise PublicationError(
                f"Validation failures are inconsistent for {model_id} on {device_name}"
            )
        if _require_positive_integer(
            device.get("num_cases"),
            f"Validation case count for {model_id} on {device_name}",
        ) != len(cases):
            raise PublicationError(
                f"Validation case count disagrees for {model_id} on {device_name}"
            )

        use_cross_device_tolerance = device_name != reference_device
        expected_max_tolerance = (
            cross_device_max_abs_tolerance
            if use_cross_device_tolerance
            else max_abs_tolerance
        )
        expected_mean_tolerance = (
            cross_device_mean_abs_tolerance
            if use_cross_device_tolerance
            else mean_abs_tolerance
        )
        expected_tolerance_kind = (
            "cross_device" if use_cross_device_tolerance else "same_device"
        )
        if (
            str(device.get("reference_execution_device", "")).strip().lower()
            != reference_device
            or device.get("reference_tolerance_kind") != expected_tolerance_kind
        ):
            raise PublicationError(
                f"Reference execution contract disagrees for {model_id} on {device_name}"
            )

        fingerprints: Dict[str, Dict[str, Any]] = {}
        device_worst: Optional[tuple[str, float, float]] = None
        for case in cases:
            if not isinstance(case, dict) or case.get("status") != "ok":
                raise PublicationError(
                    f"Validation case is not ok for {model_id} on {device_name}"
                )
            case_id = str(case.get("case_id", ""))
            input_sha = str(case.get("input_sha256", ""))
            reference_sha = str(case.get("reference_output_sha256", ""))
            exported_sha = str(case.get("exported_output_sha256", ""))
            if (
                not case_id
                or re.fullmatch(r"[0-9a-f]{64}", input_sha) is None
                or re.fullmatch(r"[0-9a-f]{64}", reference_sha) is None
                or re.fullmatch(r"[0-9a-f]{64}", exported_sha) is None
            ):
                raise PublicationError(
                    f"Validation fingerprints are incomplete for {model_id}"
                )
            if case_id in fingerprints:
                raise PublicationError(
                    f"Validation case {case_id} is duplicated for {model_id}"
                )
            max_abs_diff = _require_finite_nonnegative(
                case.get("max_abs_diff_vs_reference"),
                f"Maximum comparison statistic for {model_id}/{device_name}/{case_id}",
            )
            mean_abs_diff = _require_finite_nonnegative(
                case.get("mean_abs_diff_vs_reference"),
                f"Mean comparison statistic for {model_id}/{device_name}/{case_id}",
            )
            declared_max_tolerance = _require_finite_nonnegative(
                case.get("reference_max_abs_tolerance"),
                f"Declared maximum tolerance for {model_id}/{device_name}/{case_id}",
            )
            declared_mean_tolerance = _require_finite_nonnegative(
                case.get("reference_mean_abs_tolerance"),
                f"Declared mean tolerance for {model_id}/{device_name}/{case_id}",
            )
            if (
                declared_max_tolerance != expected_max_tolerance
                or declared_mean_tolerance != expected_mean_tolerance
                or str(case.get("reference_execution_device", "")).strip().lower()
                != reference_device
            ):
                raise PublicationError(
                    f"Validation tolerance contract disagrees for "
                    f"{model_id}/{device_name}/{case_id}"
                )
            _require_positive_integer(
                case.get("numel_compared"),
                f"Compared element count for {model_id}/{device_name}/{case_id}",
            )
            if (
                max_abs_diff > declared_max_tolerance
                or mean_abs_diff > declared_mean_tolerance
            ):
                raise PublicationError(
                    f"Validation drift exceeds tolerance for "
                    f"{model_id}/{device_name}/{case_id}"
                )
            fingerprints[case_id] = {
                "input_sha256": input_sha,
                "reference_output_sha256": reference_sha,
                "exported_output_sha256": exported_sha,
                "max_abs_diff_vs_reference": max_abs_diff,
                "mean_abs_diff_vs_reference": mean_abs_diff,
            }
            if device_worst is None or max_abs_diff > device_worst[1]:
                device_worst = (case_id, max_abs_diff, mean_abs_diff)
            if global_worst is None or max_abs_diff > global_worst[2]:
                global_worst = (device_name, case_id, max_abs_diff, mean_abs_diff)

        assert device_worst is not None
        reported_device_worst = (
            str(device.get("worst_case_id", "")),
            _require_finite_nonnegative(
                device.get("worst_max_abs_diff_vs_reference"),
                f"Device maximum comparison statistic for {model_id}/{device_name}",
            ),
            _require_finite_nonnegative(
                device.get("worst_mean_abs_diff_vs_reference"),
                f"Device mean comparison statistic for {model_id}/{device_name}",
            ),
        )
        if reported_device_worst != device_worst:
            raise PublicationError(
                f"Device worst-case evidence disagrees for {model_id} on {device_name}"
            )
        validation_cases[device_name] = fingerprints
        total_cases += len(fingerprints)

    case_matrix = {frozenset(cases) for cases in validation_cases.values()}
    if len(case_matrix) != 1:
        raise PublicationError(f"Validation case matrix differs by device for {model_id}")
    unique_case_count = len(next(iter(case_matrix)))
    if int(golden_metadata.get("case_count", 0)) != unique_case_count:
        raise PublicationError(f"Golden reference case count disagrees for {model_id}")
    if (
        _require_positive_integer(
            validation.get("num_cases"), f"Validation total case count for {model_id}"
        )
        != total_cases
        or _require_positive_integer(
            result.get("num_cases"), f"Staging total case count for {model_id}"
        )
        != total_cases
    ):
        raise PublicationError(f"Validation total case count disagrees for {model_id}")

    assert global_worst is not None
    reported_global_worst = (
        str(validation.get("worst_device", "")).strip().lower(),
        str(validation.get("worst_case_id", "")),
        _require_finite_nonnegative(
            validation.get("worst_max_abs_diff_vs_reference"),
            f"Global maximum comparison statistic for {model_id}",
        ),
        _require_finite_nonnegative(
            validation.get("worst_mean_abs_diff_vs_reference"),
            f"Global mean comparison statistic for {model_id}",
        ),
    )
    if reported_global_worst != global_worst:
        raise PublicationError(f"Global worst-case evidence disagrees for {model_id}")
    summary_numeric_contract = {
        "max_abs_tolerance": max_abs_tolerance,
        "mean_abs_tolerance": mean_abs_tolerance,
        "worst_max_abs_diff": global_worst[2],
        "worst_mean_abs_diff": global_worst[3],
    }
    for key, expected_value in summary_numeric_contract.items():
        if _require_finite_nonnegative(
            result.get(key), f"Staging {key} for {model_id}"
        ) != expected_value:
            raise PublicationError(f"Staging numerical evidence disagrees for {model_id}")

    baseline_device = requested_devices[0]
    cross_device_results = validation.get("cross_device")
    if not isinstance(cross_device_results, list):
        raise PublicationError(f"Cross-device evidence is invalid for {model_id}")
    expected_cross_device_cases = {
        (baseline_device, device_name, case_id)
        for device_name in requested_devices[1:]
        for case_id in validation_cases[baseline_device]
    }
    observed_cross_device_cases = set()
    for comparison in cross_device_results:
        if not isinstance(comparison, dict) or comparison.get("status") != "ok":
            raise PublicationError(f"Cross-device evidence is invalid for {model_id}")
        identity = (
            str(comparison.get("baseline_device", "")).strip().lower(),
            str(comparison.get("device", "")).strip().lower(),
            str(comparison.get("case_id", "")),
        )
        if identity in observed_cross_device_cases:
            raise PublicationError(f"Cross-device evidence is duplicated for {model_id}")
        max_abs_diff = _require_finite_nonnegative(
            comparison.get("max_abs_diff"),
            f"Cross-device maximum comparison statistic for {model_id}/{identity[2]}",
        )
        mean_abs_diff = _require_finite_nonnegative(
            comparison.get("mean_abs_diff"),
            f"Cross-device mean comparison statistic for {model_id}/{identity[2]}",
        )
        if (
            max_abs_diff > cross_device_max_abs_tolerance
            or mean_abs_diff > cross_device_mean_abs_tolerance
        ):
            raise PublicationError(
                f"Cross-device drift exceeds tolerance for {model_id}/{identity[2]}"
            )
        observed_cross_device_cases.add(identity)
    if observed_cross_device_cases != expected_cross_device_cases:
        raise PublicationError(f"Cross-device evidence is incomplete for {model_id}")

    return {
        "model_id": model_id,
        "repo_id": repo_id,
        "cohort": cohort,
        "parent_revision": _base_revision(base_revisions, model_id, repo_id),
        "artifact_path": _relative_path(staging_root, artifact),
        "artifact_filename": artifact.name,
        "artifact_sha256": observed_artifact_sha,
        "artifact_size_bytes": artifact.stat().st_size,
        "metadata_path": _relative_path(staging_root, metadata),
        "metadata_filename": metadata.name,
        "metadata_sha256": observed_metadata_sha,
        "golden_reference_path": _relative_path(staging_root, golden_reference),
        "golden_reference_sha256": observed_golden_sha,
        "golden_reference_size_bytes": observed_golden_size,
        "golden_reference_source_cohort": golden_source_cohort,
        "golden_reference_status": str(golden_metadata["status"]),
        "reference_device": reference_device,
        "required_devices": requested_devices,
        "num_validation_cases": total_cases,
        "max_abs_tolerance": max_abs_tolerance,
        "mean_abs_tolerance": mean_abs_tolerance,
        "cross_device_max_abs_tolerance": cross_device_max_abs_tolerance,
        "cross_device_mean_abs_tolerance": cross_device_mean_abs_tolerance,
        "validation_cases": validation_cases,
    }


def _cross_cohort_comparisons(records: Sequence[Mapping[str, Any]]):
    by_model: Dict[str, list] = {}
    for record in records:
        by_model.setdefault(str(record["model_id"]), []).append(record)

    comparisons = []
    for model_id, model_records in sorted(by_model.items()):
        for left, right in itertools.combinations(
            sorted(model_records, key=lambda item: _cohort_key(item["cohort"])), 2
        ):
            if (
                left["golden_reference_sha256"] != right["golden_reference_sha256"]
                or left["golden_reference_source_cohort"]
                != right["golden_reference_source_cohort"]
            ):
                raise PublicationError(
                    f"Cohorts do not share one golden reference for {model_id}"
                )
            if set(left["required_devices"]) != set(right["required_devices"]):
                raise PublicationError(
                    f"Cross-cohort device matrices differ for {model_id}: "
                    f"{left['cohort']} vs {right['cohort']}"
                )
            for device in sorted(left["required_devices"]):
                left_cases = left["validation_cases"].get(device, {})
                right_cases = right["validation_cases"].get(device, {})
                if not left_cases or set(left_cases) != set(right_cases):
                    raise PublicationError(
                        f"Cross-cohort case matrices differ for {model_id} on {device}"
                    )
                exact_exports = 0
                worst_guaranteed_max_abs = 0.0
                worst_guaranteed_mean_abs = 0.0
                guaranteed_max_abs_limit = None
                guaranteed_mean_abs_limit = None
                for case_id in sorted(left_cases):
                    left_case = left_cases[case_id]
                    right_case = right_cases[case_id]
                    if left_case["input_sha256"] != right_case["input_sha256"]:
                        raise PublicationError(
                            "Validation input differs across cohorts for "
                            f"{model_id}/{device}/{case_id}"
                        )
                    if (
                        left_case["reference_output_sha256"]
                        != right_case["reference_output_sha256"]
                    ):
                        raise PublicationError(
                            "Independent reference output differs across cohorts for "
                            f"{model_id}/{device}/{case_id}; use an immutable golden "
                            "reference before publication"
                        )
                    if (
                        left_case["exported_output_sha256"]
                        == right_case["exported_output_sha256"]
                    ):
                        exact_exports += 1
                    guaranteed = (
                        left_case["max_abs_diff_vs_reference"]
                        + right_case["max_abs_diff_vs_reference"]
                    )
                    guaranteed_mean = (
                        left_case["mean_abs_diff_vs_reference"]
                        + right_case["mean_abs_diff_vs_reference"]
                    )
                    left_limit = (
                        left["max_abs_tolerance"]
                        if device == left["reference_device"]
                        else left["cross_device_max_abs_tolerance"]
                    )
                    right_limit = (
                        right["max_abs_tolerance"]
                        if device == right["reference_device"]
                        else right["cross_device_max_abs_tolerance"]
                    )
                    left_mean_limit = (
                        left["mean_abs_tolerance"]
                        if device == left["reference_device"]
                        else left["cross_device_mean_abs_tolerance"]
                    )
                    right_mean_limit = (
                        right["mean_abs_tolerance"]
                        if device == right["reference_device"]
                        else right["cross_device_mean_abs_tolerance"]
                    )
                    allowed = left_limit + right_limit
                    allowed_mean = left_mean_limit + right_mean_limit
                    if guaranteed > allowed or guaranteed_mean > allowed_mean:
                        raise PublicationError(
                            "Cross-cohort drift exceeds the combined tolerance for "
                            f"{model_id}/{device}/{case_id}"
                        )
                    if (
                        guaranteed_max_abs_limit is None
                        or guaranteed > worst_guaranteed_max_abs
                    ):
                        worst_guaranteed_max_abs = guaranteed
                        guaranteed_max_abs_limit = allowed
                    if (
                        guaranteed_mean_abs_limit is None
                        or guaranteed_mean > worst_guaranteed_mean_abs
                    ):
                        worst_guaranteed_mean_abs = guaranteed_mean
                        guaranteed_mean_abs_limit = allowed_mean
                comparisons.append(
                    {
                        "model_id": model_id,
                        "left_cohort": left["cohort"],
                        "right_cohort": right["cohort"],
                        "device": device,
                        "num_cases": len(left_cases),
                        "exact_export_cases": exact_exports,
                        "worst_guaranteed_max_abs": worst_guaranteed_max_abs,
                        "guaranteed_max_abs_limit": guaranteed_max_abs_limit,
                        "worst_guaranteed_mean_abs": worst_guaranteed_mean_abs,
                        "guaranteed_mean_abs_limit": guaranteed_mean_abs_limit,
                    }
                )
    return comparisons


def _plan_core(plan: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": plan["schema_version"],
        "staging_root": plan["staging_root"],
        "cohorts": plan["cohorts"],
        "models": plan["models"],
        "cross_cohort_comparisons": plan["cross_cohort_comparisons"],
        "manifest_target": plan["manifest_target"],
    }


def prepare_publication_plan(
    *,
    staging_root: Path,
    summary_paths: Sequence[Path],
    base_revisions: Mapping[str, Any],
    manifest_repo_id: str,
    manifest_base_revision: str,
    output_path: Path,
) -> Dict[str, Any]:
    """Create a deterministic plan only from a complete green staging matrix."""

    root = staging_root.resolve()
    if not root.is_dir():
        raise PublicationError(f"Staging root is not a directory: {root}")
    if not summary_paths:
        raise PublicationError("At least one staging summary is required")
    if not manifest_repo_id or "/" not in manifest_repo_id:
        raise PublicationError("A namespaced manifest repository ID is required")

    records = []
    cohorts = set()
    seen = set()
    for raw_summary_path in summary_paths:
        summary_path = _safe_staged_file(root, raw_summary_path, "staging summary")
        summary = _read_json(summary_path)
        if summary.get("status") != "ok":
            raise PublicationError(f"Staging summary is not ok: {summary_path}")
        requested_ids = [str(value) for value in summary.get("requested_model_ids", [])]
        results = summary.get("results", [])
        if not requested_ids or not isinstance(results, list):
            raise PublicationError(
                f"Staging summary has no requested matrix: {summary_path}"
            )
        result_ids = [
            str(result.get("model_id", ""))
            for result in results
            if isinstance(result, dict)
        ]
        if sorted(result_ids) != sorted(requested_ids) or len(result_ids) != len(
            set(result_ids)
        ):
            raise PublicationError(
                f"Staging summary model matrix is incomplete: {summary_path}"
            )

        cohort = str(summary.get("torch_minor", ""))
        cohorts.add(cohort)
        for result in results:
            if not isinstance(result, dict):
                raise PublicationError(f"Invalid staging result in {summary_path}")
            record = _validated_model_record(root, summary, result, base_revisions)
            identity = (record["model_id"], record["cohort"])
            if identity in seen:
                raise PublicationError(f"Duplicate staged model/cohort: {identity}")
            seen.add(identity)
            records.append(record)

    records.sort(key=lambda item: (item["model_id"], _cohort_key(item["cohort"])))
    core = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "staging_root": str(root),
        "cohorts": sorted(cohorts, key=_cohort_key),
        "models": records,
        "cross_cohort_comparisons": _cross_cohort_comparisons(records),
        "manifest_target": {
            "repo_id": manifest_repo_id,
            "parent_revision": _require_commit(
                manifest_base_revision, "Manifest parent revision"
            ),
        },
    }
    plan_id = _sha256_bytes(_canonical_json_bytes(core))
    plan = {
        **core,
        "plan_id": plan_id,
        "candidate_branch": f"facetorch-candidate-{plan_id[:16]}",
    }
    _write_json_atomic(output_path, plan)
    return plan


def create_approval_template(plan_path: Path, output_path: Path) -> Dict[str, Any]:
    plan = verify_publication_plan(plan_path)
    approval = {
        "schema_version": APPROVAL_SCHEMA_VERSION,
        "status": "pending",
        "scope": "complete-plan",
        "plan_id": plan["plan_id"],
        "plan_sha256": _sha256(plan_path),
        "approved_by": "",
        "approved_at_utc": "",
        "notes": "",
    }
    _write_json_atomic(output_path, approval)
    return approval


def verify_publication_plan(plan_path: Path) -> Dict[str, Any]:
    plan = _read_json(plan_path)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise PublicationError("Unsupported publication plan schema")
    expected_plan_id = _sha256_bytes(_canonical_json_bytes(_plan_core(plan)))
    if plan.get("plan_id") != expected_plan_id:
        raise PublicationError("Publication plan ID does not match its contents")
    expected_branch = f"facetorch-candidate-{expected_plan_id[:16]}"
    if plan.get("candidate_branch") != expected_branch:
        raise PublicationError("Publication candidate branch is not deterministic")

    root = Path(str(plan.get("staging_root", "")))
    if not root.is_absolute() or not root.is_dir():
        raise PublicationError("Publication plan staging root is unavailable")
    models = plan.get("models")
    if not isinstance(models, list) or not models:
        raise PublicationError("Publication plan contains no models")
    seen = set()
    for model in models:
        if not isinstance(model, dict):
            raise PublicationError("Publication plan has an invalid model record")
        identity = (model.get("model_id"), model.get("cohort"))
        if identity in seen:
            raise PublicationError(f"Publication plan repeats {identity}")
        seen.add(identity)
        _require_commit(model.get("parent_revision"), f"Parent revision for {identity}")
        artifact = _safe_staged_file(root, model.get("artifact_path"), "artifact")
        metadata = _safe_staged_file(root, model.get("metadata_path"), "metadata")
        golden_reference = _safe_staged_file(
            root, model.get("golden_reference_path"), "golden reference"
        )
        if _sha256(artifact) != model.get("artifact_sha256"):
            raise PublicationError(f"Artifact changed after planning: {identity}")
        if artifact.stat().st_size != int(model.get("artifact_size_bytes", -1)):
            raise PublicationError(f"Artifact size changed after planning: {identity}")
        if _sha256(metadata) != model.get("metadata_sha256"):
            raise PublicationError(f"Metadata changed after planning: {identity}")
        if _sha256(golden_reference) != model.get("golden_reference_sha256"):
            raise PublicationError(
                f"Golden reference changed after planning: {identity}"
            )
        if golden_reference.stat().st_size != int(
            model.get("golden_reference_size_bytes", -1)
        ):
            raise PublicationError(
                f"Golden reference size changed after planning: {identity}"
            )

    manifest_target = plan.get("manifest_target")
    if not isinstance(manifest_target, dict):
        raise PublicationError("Publication plan lacks a manifest target")
    _require_commit(manifest_target.get("parent_revision"), "Manifest parent revision")
    if manifest_target.get("repo_id") in {model["repo_id"] for model in models}:
        raise PublicationError(
            "Manifest repository must be separate from model repositories"
        )
    return plan


def _validate_approval_for_plan(
    plan_path: Path, plan: Mapping[str, Any], approval_path: Path
) -> Dict[str, Any]:
    approval = _read_json(approval_path)
    if approval.get("schema_version") != APPROVAL_SCHEMA_VERSION:
        raise PublicationError("Unsupported approval schema")
    if approval.get("status") != "approved" or approval.get("scope") != "complete-plan":
        raise PublicationError("Publication approval must approve the complete plan")
    if approval.get("plan_id") != plan["plan_id"]:
        raise PublicationError("Approval plan ID does not match")
    if approval.get("plan_sha256") != _sha256(plan_path):
        raise PublicationError("Approval is not bound to the current plan bytes")
    if not str(approval.get("approved_by", "")).strip():
        raise PublicationError("Approval must identify the reviewer")
    approved_at = str(approval.get("approved_at_utc", ""))
    try:
        timestamp = datetime.fromisoformat(approved_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PublicationError("Approval timestamp must be ISO 8601") from exc
    if timestamp.tzinfo is None:
        raise PublicationError("Approval timestamp must include a timezone")
    return approval


def validate_approval(plan_path: Path, approval_path: Path) -> Dict[str, Any]:
    plan = verify_publication_plan(plan_path)
    return _validate_approval_for_plan(plan_path, plan, approval_path)


def _commit_oid(commit_info: Any) -> str:
    for attribute in ("oid", "sha", "commit_id"):
        value = getattr(commit_info, attribute, None)
        if value is not None and _COMMIT_PATTERN.fullmatch(str(value).lower()):
            return str(value).lower()
    raise PublicationError("Hub commit response did not contain an immutable commit ID")


def _verify_remote_commit(api: Any, repo_id: str, revision: str) -> None:
    info = api.repo_info(repo_id=repo_id, revision=revision)
    observed = _require_commit(
        getattr(info, "sha", None), f"Remote revision for {repo_id}"
    )
    if observed != revision:
        raise PublicationError(
            f"Remote repository {repo_id} did not resolve expected commit {revision}"
        )


def _verify_direct_parent(
    api: Any,
    *,
    repo_id: str,
    revision: str,
    parent_revision: str,
    label: str,
) -> None:
    """Require ``revision`` to be the direct child of the approved parent."""

    list_repo_commits = getattr(api, "list_repo_commits", None)
    if not callable(list_repo_commits):
        raise PublicationError(
            "Installed huggingface_hub cannot inspect repository commit history"
        )
    try:
        commits = list_repo_commits(repo_id=repo_id, revision=revision)
        if not isinstance(commits, list) or len(commits) < 2:
            raise PublicationError(
                f"{label} {repo_id}@{revision} has no verifiable direct parent"
            )
        observed_revision = _commit_oid(commits[0])
        observed_parent = _commit_oid(commits[1])
    except PublicationError:
        raise
    except Exception as exc:
        raise PublicationError(
            f"Cannot inspect commit history for {label} {repo_id}@{revision}"
        ) from exc
    if observed_revision != revision or observed_parent != parent_revision:
        raise PublicationError(
            f"{label} {repo_id}@{revision} is not a direct child of approved "
            f"parent {parent_revision}"
        )


def _content_contract(value: Path | bytes) -> Dict[str, Any]:
    """Describe bytes using both Hub storage digest schemes."""
    if isinstance(value, Path):
        size = value.stat().st_size
        sha256 = hashlib.sha256()
        git_blob = hashlib.sha1()
        git_blob.update(f"blob {size}\0".encode("ascii"))
        with value.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                sha256.update(chunk)
                git_blob.update(chunk)
        sha256_value = sha256.hexdigest()
        git_blob_value = git_blob.hexdigest()
    else:
        size = len(value)
        sha256_value = _sha256_bytes(value)
        git_blob_value = hashlib.sha1(
            f"blob {size}\0".encode("ascii") + value
        ).hexdigest()
    return {
        "size_bytes": size,
        "sha256": sha256_value,
        "git_blob_id": git_blob_value,
    }


def _remote_tree(api: Any, repo_id: str, revision: str) -> Dict[str, Any]:
    """Return immutable file signatures for one Hub tree."""
    try:
        list_repo_tree = getattr(api, "list_repo_tree", None)
        if callable(list_repo_tree):
            entries = list_repo_tree(
                repo_id=repo_id,
                revision=revision,
                recursive=True,
                expand=True,
            )
        else:
            list_files_info = getattr(api, "list_files_info", None)
            if not callable(list_files_info):
                raise PublicationError(
                    "Installed huggingface_hub cannot inspect repository files"
                )
            entries = list_files_info(
                repo_id=repo_id,
                paths=None,
                revision=revision,
                expand=True,
            )
        tree = {}
        for entry in entries:
            blob_id = getattr(entry, "blob_id", None)
            size = getattr(entry, "size", None)
            if blob_id is None or size is None:
                continue
            path = str(getattr(entry, "path", ""))
            if not path or path in tree:
                raise PublicationError(
                    f"Remote repository {repo_id}@{revision} has an invalid tree"
                )
            lfs = getattr(entry, "lfs", None)
            lfs_sha256 = (
                lfs.get("sha256")
                if isinstance(lfs, Mapping)
                else getattr(lfs, "sha256", None)
            )
            tree[path] = {
                "size_bytes": int(size),
                "blob_id": str(blob_id).lower(),
                "lfs_sha256": (
                    str(lfs_sha256).lower() if lfs_sha256 is not None else None
                ),
            }
        return tree
    except PublicationError:
        raise
    except Exception as exc:
        raise PublicationError(
            f"Cannot inspect remote repository tree {repo_id}@{revision}"
        ) from exc


def _remote_file_matches(
    observed: Mapping[str, Any], expected: Mapping[str, Any]
) -> bool:
    if observed.get("size_bytes") != expected.get("size_bytes"):
        return False
    lfs_sha256 = observed.get("lfs_sha256")
    if lfs_sha256 is not None:
        return lfs_sha256 == expected.get("sha256")
    return observed.get("blob_id") == expected.get("git_blob_id")


def _tree_contract_differences(
    parent_tree: Mapping[str, Any],
    candidate_tree: Mapping[str, Any],
    expected_files: Mapping[str, Mapping[str, Any]],
) -> tuple[set[str], set[str]]:
    planned_paths = set(expected_files)
    unplanned_changes = {
        path
        for path in set(parent_tree).union(candidate_tree) - planned_paths
        if parent_tree.get(path) != candidate_tree.get(path)
    }
    mismatched = {
        path
        for path, contract in expected_files.items()
        if path not in candidate_tree
        or not _remote_file_matches(candidate_tree[path], contract)
    }
    return unplanned_changes, mismatched


def _verify_remote_commit_contents(
    api: Any,
    *,
    repo_id: str,
    revision: str,
    parent_revision: str,
    expected_files: Mapping[str, Mapping[str, Any]],
    label: str,
) -> None:
    """Verify that a recorded commit is the exact approved change over its parent."""
    _verify_remote_commit(api, repo_id, revision)
    _verify_direct_parent(
        api,
        repo_id=repo_id,
        revision=revision,
        parent_revision=parent_revision,
        label=label,
    )
    parent_tree = _remote_tree(api, repo_id, parent_revision)
    commit_tree = _remote_tree(api, repo_id, revision)
    unplanned_changes, mismatched = _tree_contract_differences(
        parent_tree, commit_tree, expected_files
    )
    if unplanned_changes or mismatched:
        details = []
        if unplanned_changes:
            details.append(f"unplanned paths {sorted(unplanned_changes)}")
        if mismatched:
            details.append(f"mismatched planned paths {sorted(mismatched)}")
        raise PublicationError(
            f"{label} {repo_id}@{revision} does not match the approved plan: "
            + "; ".join(details)
        )


def _reconcile_candidate_branch(
    api: Any,
    *,
    repo_id: str,
    branch: str,
    parent_revision: str,
    expected_files: Mapping[str, Mapping[str, Any]],
) -> Optional[str]:
    """Recover an exact post-commit state, or require a fresh parent commit."""
    try:
        info = api.repo_info(repo_id=repo_id, revision=branch)
    except Exception as exc:
        raise PublicationError(
            f"Cannot resolve candidate branch {repo_id}@{branch}"
        ) from exc
    head = _require_commit(
        getattr(info, "sha", None), f"Candidate branch head for {repo_id}"
    )
    if head == parent_revision:
        return None

    _verify_direct_parent(
        api,
        repo_id=repo_id,
        revision=head,
        parent_revision=parent_revision,
        label="Candidate branch commit",
    )

    parent_tree = _remote_tree(api, repo_id, parent_revision)
    candidate_tree = _remote_tree(api, repo_id, head)
    unplanned_changes, mismatched = _tree_contract_differences(
        parent_tree, candidate_tree, expected_files
    )
    if unplanned_changes or mismatched:
        details = []
        if unplanned_changes:
            details.append(f"unplanned paths {sorted(unplanned_changes)}")
        if mismatched:
            details.append(f"mismatched planned paths {sorted(mismatched)}")
        raise PublicationError(
            f"Candidate branch {repo_id}@{branch} diverged from the approved plan: "
            + "; ".join(details)
        )
    return head


def _new_receipt(plan: Mapping[str, Any], plan_path: Path) -> Dict[str, Any]:
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "plan_id": plan["plan_id"],
        "plan_sha256": _sha256(plan_path),
        "candidate_branch": plan["candidate_branch"],
        "status": "incomplete",
        "models": {},
        "manifest": None,
    }


def _load_receipt(
    receipt_path: Path, plan: Mapping[str, Any], plan_path: Path
) -> Dict[str, Any]:
    if not receipt_path.exists():
        return _new_receipt(plan, plan_path)
    receipt = _read_json(receipt_path)
    expected = _new_receipt(plan, plan_path)
    for key in ("schema_version", "plan_id", "plan_sha256", "candidate_branch"):
        if receipt.get(key) != expected[key]:
            raise PublicationError(f"Publication receipt has mismatched {key}")
    if not isinstance(receipt.get("models"), dict):
        raise PublicationError("Publication receipt model state is invalid")
    return receipt


def _group_models(plan: Mapping[str, Any]) -> Sequence[Dict[str, Any]]:
    groups: Dict[str, Dict[str, Any]] = {}
    for model in plan["models"]:
        model_id = model["model_id"]
        group = groups.setdefault(
            model_id,
            {
                "model_id": model_id,
                "repo_id": model["repo_id"],
                "parent_revision": model["parent_revision"],
                "artifacts": [],
            },
        )
        if (
            group["repo_id"] != model["repo_id"]
            or group["parent_revision"] != model["parent_revision"]
        ):
            raise PublicationError(
                f"Model {model_id} has inconsistent repository or parent revisions"
            )
        group["artifacts"].append(model)
    for group in groups.values():
        group["artifacts"].sort(key=lambda item: _cohort_key(item["cohort"]))
    return [groups[key] for key in sorted(groups)]


def _completed_model_matches(
    record: Mapping[str, Any], group: Mapping[str, Any]
) -> bool:
    expected_artifacts = {
        model["cohort"]: {
            "artifact_sha256": model["artifact_sha256"],
            "metadata_sha256": model["metadata_sha256"],
        }
        for model in group["artifacts"]
    }
    return (
        record.get("repo_id") == group.get("repo_id")
        and record.get("parent_revision") == group.get("parent_revision")
        and record.get("artifacts") == expected_artifacts
        and _COMMIT_PATTERN.fullmatch(str(record.get("commit_revision", "")))
        is not None
    )


def _manifest_payload(
    plan: Mapping[str, Any], receipt: Mapping[str, Any]
) -> Dict[str, Any]:
    models = []
    for model in plan["models"]:
        completed = receipt["models"][model["model_id"]]
        models.append(
            {
                "model_id": model["model_id"],
                "repo_id": model["repo_id"],
                "cohort": model["cohort"],
                "revision": completed["commit_revision"],
                "artifact_filename": model["artifact_filename"],
                "artifact_sha256": model["artifact_sha256"],
                "artifact_size_bytes": model["artifact_size_bytes"],
                "metadata_filename": model["metadata_filename"],
                "metadata_sha256": model["metadata_sha256"],
                "golden_reference_sha256": model["golden_reference_sha256"],
                "golden_reference_size_bytes": model["golden_reference_size_bytes"],
                "golden_reference_source_cohort": model[
                    "golden_reference_source_cohort"
                ],
                "required_devices": model["required_devices"],
            }
        )
    return {
        "schema_version": 1,
        "status": "candidate",
        "plan_id": plan["plan_id"],
        "cohorts": plan["cohorts"],
        "cross_cohort_comparisons": plan["cross_cohort_comparisons"],
        "models": models,
    }


def publish_publication_plan(
    *,
    plan_path: Path,
    approval_path: Path,
    receipt_path: Path,
    token: Optional[str] = None,
    api: Any = None,
) -> Dict[str, Any]:
    """Publish a reviewed plan to candidate refs and promote its manifest last."""

    plan = verify_publication_plan(plan_path)
    _validate_approval_for_plan(plan_path, plan, approval_path)
    if api is None:
        if not token:
            raise PublicationError("A Hugging Face token is required for publication")
        try:
            from huggingface_hub import HfApi
        except (
            ImportError
        ) as exc:  # pragma: no cover - dependency is required by project
            raise PublicationError(
                "huggingface_hub is required for publication"
            ) from exc
        api = HfApi(token=token)

    try:
        from huggingface_hub import CommitOperationAdd
    except ImportError as exc:  # pragma: no cover - dependency is required by project
        raise PublicationError("Hub commit operations are unavailable") from exc

    receipt = _load_receipt(receipt_path, plan, plan_path)
    root = Path(plan["staging_root"])
    branch = plan["candidate_branch"]

    for group in _group_models(plan):
        key = group["model_id"]
        staged_files = []
        artifact_receipts = {}
        expected_files = {}
        for model in group["artifacts"]:
            artifact = _safe_staged_file(root, model["artifact_path"], "artifact")
            metadata = _safe_staged_file(root, model["metadata_path"], "metadata")
            for filename, path in (
                (model["artifact_filename"], artifact),
                (model["metadata_filename"], metadata),
            ):
                if filename in expected_files:
                    raise PublicationError(
                        f"Duplicate planned Hub path for {key}: {filename}"
                    )
                expected_files[filename] = _content_contract(path)
                staged_files.append((filename, path))
            artifact_receipts[model["cohort"]] = {
                "artifact_sha256": model["artifact_sha256"],
                "metadata_sha256": model["metadata_sha256"],
            }

        completed = receipt["models"].get(key)
        if completed is not None:
            if not _completed_model_matches(completed, group):
                raise PublicationError(f"Receipt bytes do not match plan for {key}")
            _verify_remote_commit_contents(
                api,
                repo_id=group["repo_id"],
                revision=completed["commit_revision"],
                parent_revision=group["parent_revision"],
                expected_files=expected_files,
                label=f"Recorded model commit for {key}",
            )
            continue

        operations = [
            CommitOperationAdd(path_in_repo=filename, path_or_fileobj=str(path))
            for filename, path in staged_files
        ]
        try:
            api.create_branch(
                repo_id=group["repo_id"],
                branch=branch,
                revision=group["parent_revision"],
                exist_ok=True,
            )
            commit_revision = _reconcile_candidate_branch(
                api,
                repo_id=group["repo_id"],
                branch=branch,
                parent_revision=group["parent_revision"],
                expected_files=expected_files,
            )
            if commit_revision is None:
                commit = api.create_commit(
                    repo_id=group["repo_id"],
                    operations=operations,
                    revision=branch,
                    parent_commit=group["parent_revision"],
                    commit_message=(
                        f"Stage {group['model_id']} cohorts for "
                        f"{plan['plan_id'][:16]}"
                    ),
                    commit_description=(
                        "Candidate only; the release manifest is promoted after "
                        f"every model succeeds. Plan: {plan['plan_id']}"
                    ),
                )
                commit_revision = _commit_oid(commit)
        except Exception:
            receipt["status"] = "incomplete"
            _write_json_atomic(receipt_path, receipt)
            raise

        receipt["models"][key] = {
            "repo_id": group["repo_id"],
            "parent_revision": group["parent_revision"],
            "commit_revision": commit_revision,
            "artifacts": artifact_receipts,
        }
        _write_json_atomic(receipt_path, receipt)

    manifest_target = plan["manifest_target"]
    manifest = _manifest_payload(plan, receipt)
    manifest_bytes = _canonical_json_bytes(manifest)
    manifest_filename = f"manifests/{plan['plan_id']}.json"
    expected_files = {manifest_filename: _content_contract(manifest_bytes)}
    existing_manifest = receipt.get("manifest")
    if isinstance(existing_manifest, dict):
        expected_receipt = {
            "repo_id": manifest_target["repo_id"],
            "filename": manifest_filename,
            "sha256": _sha256_bytes(manifest_bytes),
        }
        if any(
            existing_manifest.get(key) != value
            for key, value in expected_receipt.items()
        ):
            raise PublicationError("Manifest receipt bytes do not match plan")
        revision = _require_commit(
            existing_manifest.get("commit_revision"), "Manifest receipt revision"
        )
        _verify_remote_commit_contents(
            api,
            repo_id=manifest_target["repo_id"],
            revision=revision,
            parent_revision=manifest_target["parent_revision"],
            expected_files=expected_files,
            label="Recorded manifest commit",
        )
    else:
        try:
            api.create_branch(
                repo_id=manifest_target["repo_id"],
                branch=branch,
                revision=manifest_target["parent_revision"],
                exist_ok=True,
            )
            revision = _reconcile_candidate_branch(
                api,
                repo_id=manifest_target["repo_id"],
                branch=branch,
                parent_revision=manifest_target["parent_revision"],
                expected_files=expected_files,
            )
            if revision is None:
                commit = api.create_commit(
                    repo_id=manifest_target["repo_id"],
                    operations=[
                        CommitOperationAdd(
                            path_in_repo=manifest_filename,
                            path_or_fileobj=manifest_bytes,
                        )
                    ],
                    revision=branch,
                    parent_commit=manifest_target["parent_revision"],
                    commit_message=f"Pin cohort manifest {plan['plan_id'][:16]}",
                    commit_description=(
                        "Immutable candidate manifest created only after all model "
                        f"commits succeeded. Plan: {plan['plan_id']}"
                    ),
                )
                revision = _commit_oid(commit)
        except Exception:
            receipt["status"] = "incomplete"
            _write_json_atomic(receipt_path, receipt)
            raise

        receipt["manifest"] = {
            "repo_id": manifest_target["repo_id"],
            "filename": manifest_filename,
            "sha256": _sha256_bytes(manifest_bytes),
            "commit_revision": revision,
        }

    receipt["status"] = "complete"
    _write_json_atomic(receipt_path, receipt)
    return receipt


def _load_revision_map(path: Path) -> Dict[str, Any]:
    value = _read_json(path)
    revisions = value.get("revisions", value)
    if not isinstance(revisions, dict):
        raise PublicationError("Base revision file must contain an object mapping")
    return revisions


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Build a deterministic plan")
    prepare.add_argument("--staging-root", required=True)
    prepare.add_argument("--summary", action="append", required=True)
    prepare.add_argument("--base-revisions", required=True)
    prepare.add_argument("--manifest-repo-id", required=True)
    prepare.add_argument("--manifest-base-revision", required=True)
    prepare.add_argument("--plan", required=True)
    prepare.add_argument("--approval-template")

    verify = subparsers.add_parser("verify", help="Verify staged bytes and approval")
    verify.add_argument("--plan", required=True)
    verify.add_argument("--approval", required=True)

    publish = subparsers.add_parser("publish", help="Publish approved candidate refs")
    publish.add_argument("--plan", required=True)
    publish.add_argument("--approval", required=True)
    publish.add_argument("--receipt", required=True)
    publish.add_argument("--hf-token-env", default="HF_TOKEN")

    args = parser.parse_args()
    if args.command == "prepare":
        plan_path = Path(args.plan).resolve()
        prepare_publication_plan(
            staging_root=Path(args.staging_root),
            summary_paths=[Path(value) for value in args.summary],
            base_revisions=_load_revision_map(Path(args.base_revisions)),
            manifest_repo_id=args.manifest_repo_id,
            manifest_base_revision=args.manifest_base_revision,
            output_path=plan_path,
        )
        print(f"Publication plan written to {plan_path}")
        if args.approval_template:
            approval_path = Path(args.approval_template).resolve()
            create_approval_template(plan_path, approval_path)
            print(f"Pending approval template written to {approval_path}")
    elif args.command == "verify":
        validate_approval(Path(args.plan), Path(args.approval))
        print("Publication plan, staged bytes, and approval are valid")
    else:
        token = os.getenv(args.hf_token_env)
        receipt = publish_publication_plan(
            plan_path=Path(args.plan),
            approval_path=Path(args.approval),
            receipt_path=Path(args.receipt),
            token=token,
        )
        print(
            "Publication complete; immutable manifest revision: "
            f"{receipt['manifest']['commit_revision']}"
        )


if __name__ == "__main__":
    main()
