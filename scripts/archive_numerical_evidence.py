#!/usr/bin/env python3
"""Archive and verify portable numerical records without loading model binaries.

This checks recorded errors against the recorded validation bounds. The existing
matrix gates check authoritative model/case coverage and execute the binaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

if __package__:
    from .model_evidence_contract import (
        expected_metadata_identity,
        validate_metadata_identity,
        validate_summary_identity,
    )
else:
    from model_evidence_contract import (
        expected_metadata_identity,
        validate_metadata_identity,
        validate_summary_identity,
    )

INDEX = "numerical-evidence-index.json"
RUNNER = "local-cuda-runner-report.json"


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _inside(root, name):
    path = Path(name)
    _require(
        not path.is_absolute() and ".." not in path.parts, "Non-portable evidence path"
    )
    path = (root / path).resolve()
    _require(path.is_relative_to(root), "Evidence path escapes archive")
    return path


def _checked(root, entry):
    path = _inside(root, entry["path"])
    _require(
        _sha(path) == entry["sha256"], f"Evidence checksum mismatch: {entry['path']}"
    )
    return _read(path)


def _number(value):
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0,
        "Numerical evidence must be finite and non-negative",
    )
    return value


def _numeric(metadata, result, required_devices):
    validation = metadata["validation"]
    _require(
        validation["status"] == "ok" and not validation["failures"], "Failed validation"
    )
    _require(
        validation["fixed_reference_device"] == "cpu", "Missing fixed CPU reference"
    )
    bounds = {
        "cpu": (
            _number(validation["max_abs_tolerance"]),
            _number(validation["mean_abs_tolerance"]),
        ),
        "cross": (
            _number(validation["cross_device_max_abs_tolerance"]),
            _number(validation["cross_device_mean_abs_tolerance"]),
        ),
    }
    _require(
        bounds["cpu"] == (result["max_abs_tolerance"], result["mean_abs_tolerance"]),
        "Summary tolerances disagree",
    )
    records = validation["devices"]
    _require(
        len(records) == len(required_devices)
        and {record["device"] for record in records} == set(required_devices),
        "Device coverage differs",
    )
    by_device = {}
    total = 0
    for record in records:
        device = record["device"]
        cases = record["cases"]
        _require(
            record["status"] == "ok" and not record["failures"], "Failed device record"
        )
        _require(
            bool(cases) and len(cases) == record["num_cases"],
            "Device case count differs",
        )
        bound = bounds["cpu" if device == "cpu" else "cross"]
        identities = {}
        for case in cases:
            case_id = case["case_id"]
            _require(
                isinstance(case_id, str) and case_id and case_id not in identities,
                "Duplicate or invalid case identity",
            )
            _require(
                case["status"] == "ok" and case["reference_execution_device"] == "cpu",
                "Invalid case reference",
            )
            _require(
                (
                    case["reference_max_abs_tolerance"],
                    case["reference_mean_abs_tolerance"],
                )
                == bound,
                "Case tolerances disagree",
            )
            _require(
                _number(case["max_abs_diff_vs_reference"]) <= bound[0]
                and _number(case["mean_abs_diff_vs_reference"]) <= bound[1],
                "Recorded numerical error exceeds tolerance",
            )
            _require(
                type(case["numel_compared"]) is int and case["numel_compared"] > 0,
                "Empty numerical comparison",
            )
            for field in (
                "input_sha256",
                "reference_output_sha256",
                "exported_output_sha256",
            ):
                _require(
                    isinstance(case[field], str)
                    and re.fullmatch(r"[0-9a-f]{64}", case[field]),
                    "Invalid case digest",
                )
            identities[case_id] = (
                case["input_sha256"],
                case["reference_output_sha256"],
            )
        by_device[device] = identities
        total += len(cases)
    _require("cpu" in by_device, "Missing CPU device")
    _require(
        all(cases == by_device["cpu"] for cases in by_device.values()),
        "Devices do not share the same inputs and fixed references",
    )
    _require(
        total == result["num_cases"] == validation["num_cases"],
        "Total case count differs",
    )
    comparisons = validation["cross_device"]
    expected = {
        (device, case)
        for device in by_device
        if device != "cpu"
        for case in by_device[device]
    }
    observed = set()
    for record in comparisons:
        key = (record["device"], record["case_id"])
        _require(
            key in expected and key not in observed,
            "Unexpected cross-device comparison",
        )
        _require(
            record["baseline_device"] == "cpu" and record["status"] == "ok",
            "Failed cross-device comparison",
        )
        _require(
            _number(record["max_abs_diff"]) <= bounds["cross"][0]
            and _number(record["mean_abs_diff"]) <= bounds["cross"][1],
            "Cross-device numerical error exceeds tolerance",
        )
        observed.add(key)
    _require(observed == expected, "Missing cross-device comparisons")
    return total


def create_index(
    root: Path, *, source_sha: str, runner_path: Path | None = None
) -> dict:
    """Bind every runner-registered summary to its original per-model records."""
    root = root.resolve()
    runner_path = (runner_path or root / RUNNER).resolve()
    runner = _read(runner_path)
    entries = []
    for kind in ("summaries", "artifact_summaries"):
        for binding in runner[kind]:
            summary = _checked(root, binding)
            records = []
            for result in summary["results"]:
                original = Path(result["meta"])
                source = (
                    original.resolve()
                    if original.is_absolute()
                    else (root / original).resolve()
                )
                relative = source.relative_to(root).as_posix()
                _require(
                    _sha(source) == result["meta_sha256"], "Metadata checksum mismatch"
                )
                records.append(
                    {
                        "model_id": result["model_id"],
                        "path": relative,
                        "sha256": result["meta_sha256"],
                        "original_path": result["meta"],
                    }
                )
            entries.append({**binding, "kind": kind, "records": records})
    index = {
        "schema_version": 1,
        "source_sha": source_sha,
        "runner": {
            "path": runner_path.relative_to(root).as_posix(),
            "sha256": _sha(runner_path),
        },
        "summaries": entries,
    }
    # Verify before writing, so a partial run cannot produce a valid-looking index.
    verify_index(root, source_sha=source_sha, index=index)
    (root / INDEX).write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return index


def verify_index(root: Path, *, source_sha: str, index: dict | None = None) -> dict:
    root = root.resolve()
    index = _read(root / INDEX) if index is None else index
    _require(re.fullmatch(r"[0-9a-f]{40}", source_sha), "Expected full source SHA")
    _require(
        index["schema_version"] == 1 and index["source_sha"] == source_sha,
        "Wrong index source",
    )
    runner = _checked(root, index["runner"])
    _require(
        runner["source_sha"] == source_sha
        and runner["source_clean"] is True
        and runner["status"] == "ok",
        "Wrong runner source/status",
    )
    expected = {}
    for kind in ("summaries", "artifact_summaries"):
        _require(bool(runner[kind]), "Missing runner summary group")
        for entry in runner[kind]:
            _require(entry["path"] not in expected, "Duplicate runner summary")
            expected[entry["path"]] = (kind, entry["sha256"])
    observed = set()
    metadata_paths = set()
    case_count = 0
    for entry in index["summaries"]:
        path = entry["path"]
        _require(
            path not in observed
            and expected.get(path) == (entry["kind"], entry["sha256"]),
            "Unexpected or duplicate summary binding",
        )
        observed.add(path)
        summary = _checked(root, entry)
        mode = "validate" if entry["kind"] == "summaries" else "export"
        identity = validate_summary_identity(
            summary, expected_mode=mode, require_native_runtime=(mode == "export")
        )
        source = identity["environment"]["source_tree"]
        _require(
            source["commit"] == source_sha
            and source["clean"] is True
            and summary["status"] == "ok",
            "Wrong summary source/status",
        )
        results = {result["model_id"]: result for result in summary["results"]}
        records = {record["model_id"]: record for record in entry["records"]}
        _require(
            len(results)
            == len(summary["results"])
            == len(records)
            == len(entry["records"])
            and set(results) == set(records) == set(identity["requested_model_ids"]),
            "Missing or duplicate model record",
        )
        for model_id, result in results.items():
            record = records[model_id]
            _require(record["path"] not in metadata_paths, "Duplicate metadata path")
            metadata_paths.add(record["path"])
            _require(
                record["original_path"] == result["meta"]
                and record["sha256"] == result["meta_sha256"],
                "Metadata binding differs",
            )
            metadata = _checked(root, record)
            validate_metadata_identity(
                metadata,
                expected_metadata_identity(
                    identity,
                    model_id=model_id,
                    repo_id=result["repo_id"],
                    artifact_filename=Path(result["artifact"]).name,
                ),
            )
            _require(
                result["status"] == result["validation_status"] == "ok",
                "Failed model record",
            )
            _require(
                metadata["artifact_sha256"] == result["sha256"]
                and metadata["artifact_size_bytes"] == result["size_bytes"],
                "Artifact identity differs",
            )
            case_count += _numeric(metadata, result, identity["validate_devices"])
    _require(observed == set(expected), "Missing summary binding")
    return {
        "status": "ok",
        "source_sha": source_sha,
        "summaries": len(observed),
        "model_records": len(metadata_paths),
        "cases": case_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("create", "verify"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    if args.command == "create":
        create_index(args.root, source_sha=args.source_sha)
    print(json.dumps(verify_index(args.root, source_sha=args.source_sha), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
