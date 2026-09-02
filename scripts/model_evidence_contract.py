#!/usr/bin/env python3
"""Pure schema and identity checks shared by model release tooling."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

SUMMARY_SCHEMA_VERSION = 2
METADATA_SCHEMA_VERSION = 2
_TORCH_MINOR_RE = re.compile(r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$")
_TORCH_VERSION_RE = re.compile(
    r"^(?P<minor>(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*))"
    r"\.(?:0|[1-9][0-9]*)(?:[+.-].*)?$"
)


class ModelEvidenceContractError(ValueError):
    """Raised when staged or hosted model evidence has conflicting identity."""


def _json_value(value: Any, label: str) -> Any:
    """Return a detached JSON value, rejecting non-JSON evidence objects."""

    try:
        return json.loads(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise ModelEvidenceContractError(f"{label} must be valid JSON") from exc


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ModelEvidenceContractError(f"{label} must be a non-empty string")
    return value


def _unique_texts(value: Any, label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise ModelEvidenceContractError(
            f"{label} must be a non-empty unique string list"
        )
    return list(value)


def validate_summary_identity(
    summary: Mapping[str, Any],
    *,
    expected_model_ids: Sequence[str] | None = None,
    expected_devices: Sequence[str] | None = None,
    expected_mode: str = "export",
    require_native_runtime: bool = True,
) -> dict[str, Any]:
    """Normalize one self-consistent exporter summary identity."""

    if summary.get("schema_version") != SUMMARY_SCHEMA_VERSION:
        raise ModelEvidenceContractError(
            f"summary schema_version must be {SUMMARY_SCHEMA_VERSION}"
        )
    if expected_mode not in {"export", "validate"}:
        raise ModelEvidenceContractError("expected summary mode is invalid")
    if summary.get("mode") != expected_mode:
        raise ModelEvidenceContractError(f"summary mode must be {expected_mode}")
    cohort = _text(summary.get("torch_minor"), "summary torch_minor")
    if _TORCH_MINOR_RE.fullmatch(cohort) is None:
        raise ModelEvidenceContractError("summary torch_minor is not canonical")
    runtime_minor = _text(
        summary.get("runtime_torch_minor"), "summary runtime_torch_minor"
    )
    if require_native_runtime and runtime_minor != cohort:
        raise ModelEvidenceContractError(
            "summary runtime_torch_minor disagrees with torch_minor"
        )
    torch_version = _text(summary.get("torch_version"), "summary torch_version")
    version_match = _TORCH_VERSION_RE.fullmatch(torch_version)
    if version_match is None or version_match.group("minor") != runtime_minor:
        raise ModelEvidenceContractError(
            "summary torch_version disagrees with runtime_torch_minor"
        )

    model_ids = _unique_texts(
        summary.get("requested_model_ids"), "summary requested_model_ids"
    )
    devices = _unique_texts(summary.get("validate_devices"), "summary validate_devices")
    if expected_model_ids is not None and set(model_ids) != set(expected_model_ids):
        raise ModelEvidenceContractError(
            "summary model coverage differs from the authoritative manifest"
        )
    if expected_devices is not None and set(devices) != set(expected_devices):
        raise ModelEvidenceContractError(
            "summary device coverage differs from the authoritative policy"
        )

    environment = summary.get("environment")
    if not isinstance(environment, Mapping) or not environment:
        raise ModelEvidenceContractError("summary environment must be an object")
    normalized_environment = _json_value(environment, "summary environment")
    if normalized_environment.get("torch_version") != torch_version:
        raise ModelEvidenceContractError(
            "summary environment torch_version disagrees with the summary"
        )

    exporter_arguments = summary.get("exporter_arguments")
    if not isinstance(exporter_arguments, Mapping) or not exporter_arguments:
        raise ModelEvidenceContractError("summary exporter_arguments must be an object")
    normalized_arguments = _json_value(exporter_arguments, "summary exporter_arguments")
    expected_arguments = {
        "mode": expected_mode,
        "artifact_cohort": cohort,
        "validate_devices": devices,
        "model_ids": model_ids,
    }
    for field in ("batch_sizes", "seeds", "scales"):
        if field in summary:
            expected_arguments[field] = _json_value(summary[field], f"summary {field}")
    differing = sorted(
        field
        for field, expected in expected_arguments.items()
        if normalized_arguments.get(field) != expected
    )
    if differing:
        raise ModelEvidenceContractError(
            "summary exporter_arguments disagree: " + ", ".join(differing)
        )

    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "mode": expected_mode,
        "torch_version": torch_version,
        "torch_minor": cohort,
        "runtime_torch_minor": runtime_minor,
        "environment": normalized_environment,
        "exporter_arguments": normalized_arguments,
        "requested_model_ids": model_ids,
        "validate_devices": devices,
    }


def expected_metadata_identity(
    summary_identity: Mapping[str, Any],
    *,
    model_id: str,
    repo_id: str,
    artifact_filename: str,
) -> dict[str, Any]:
    """Build the complete metadata identity implied by a summary and result."""

    return {
        "schema_version": METADATA_SCHEMA_VERSION,
        "mode": summary_identity.get("mode", "export"),
        "model_id": _text(model_id, "model_id"),
        "repo_id": _text(repo_id, "repo_id"),
        "torch_version": summary_identity["torch_version"],
        "torch_minor": summary_identity["torch_minor"],
        "runtime_torch_minor": summary_identity["runtime_torch_minor"],
        "environment": summary_identity["environment"],
        "exporter_arguments": summary_identity["exporter_arguments"],
        "artifact": _text(artifact_filename, "artifact filename"),
    }


def validate_metadata_identity(
    metadata: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, Any]:
    """Require exact equality for the complete release metadata identity."""

    required = {
        "schema_version",
        "mode",
        "model_id",
        "repo_id",
        "torch_version",
        "torch_minor",
        "runtime_torch_minor",
        "environment",
        "exporter_arguments",
        "artifact",
    }
    if set(expected) != required:
        raise ModelEvidenceContractError(
            "expected metadata identity is incomplete or contains extra fields"
        )
    normalized = {
        field: _json_value(metadata.get(field), f"metadata {field}")
        for field in sorted(required)
    }
    differing = sorted(
        field for field in required if normalized[field] != expected[field]
    )
    if differing:
        raise ModelEvidenceContractError(
            "metadata identity disagrees: " + ", ".join(differing)
        )
    return normalized
