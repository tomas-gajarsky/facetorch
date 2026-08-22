#!/usr/bin/env python3
"""Bind production-image smoke reports to exact local Docker image IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as output:
            temporary = Path(output.name)
            json.dump(value, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _image_metadata(image: str) -> dict[str, Any]:
    result = subprocess.run(
        [
            "docker",
            "image",
            "inspect",
            "--format",
            "{{json .Id}}|{{json .Os}}|{{json .Architecture}}|{{json .Config.User}}",
            image,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"Cannot inspect image {image}")
    values = [json.loads(item) for item in result.stdout.strip().split("|")]
    if len(values) != 4:
        raise RuntimeError(f"Docker returned incomplete metadata for {image}")
    image_id, operating_system, architecture, configured_user = values
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(image_id)):
        raise RuntimeError(f"Docker returned an invalid image ID for {image}")
    if operating_system != "linux" or architecture != "amd64":
        raise RuntimeError(f"Docker image {image} has an unsupported platform")
    if configured_user != "facetorch":
        raise RuntimeError(f"Docker image {image} is not configured as facetorch")
    return {
        "local_reference": image,
        "image_id": image_id,
        "os": operating_system,
        "architecture": architecture,
        "configured_user": configured_user,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--runner-report", type=Path, required=True)
    parser.add_argument("--cpu-smoke", type=Path, required=True)
    parser.add_argument("--gpu-smoke", type=Path, required=True)
    parser.add_argument("--cpu-image", required=True)
    parser.add_argument("--gpu-image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_sha):
        raise RuntimeError("--source-sha must be a full lowercase commit SHA")
    repo_root = args.repo_root.resolve()
    staging_root = args.staging_root.resolve()
    resolved = {
        "runner": args.runner_report.resolve(),
        "cpu": args.cpu_smoke.resolve(),
        "gpu": args.gpu_smoke.resolve(),
        "output": args.output.resolve(),
    }
    for label, path in resolved.items():
        try:
            path.relative_to(staging_root)
        except ValueError as exc:
            raise RuntimeError(f"{label} evidence escapes the staging root") from exc
    for device in ("cpu", "gpu"):
        report = _read_json(resolved[device])
        expected = "cuda" if device == "gpu" else "cpu"
        if (
            report.get("status") != "ok"
            or report.get("device") != expected
            or report.get("uid") != 10001
        ):
            raise RuntimeError(f"{device} image smoke report is not successful")
    runner_report = _read_json(resolved["runner"])
    if (
        runner_report.get("status") != "ok"
        or runner_report.get("source_sha") != args.source_sha
    ):
        raise RuntimeError("CUDA runner report does not bind the requested source")

    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if git_head != args.source_sha:
        raise RuntimeError("Checked-out source changed before image evidence binding")
    images = {
        "cpu": {
            **_image_metadata(args.cpu_image),
            "dockerfile_sha256": _sha256(repo_root / "docker" / "Dockerfile"),
            "smoke_report_sha256": _sha256(resolved["cpu"]),
        },
        "gpu": {
            **_image_metadata(args.gpu_image),
            "dockerfile_sha256": _sha256(repo_root / "docker" / "Dockerfile.gpu"),
            "smoke_report_sha256": _sha256(resolved["gpu"]),
        },
    }
    report = {
        "schema_version": 1,
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": args.source_sha,
        "runner_report_sha256": _sha256(resolved["runner"]),
        "images": images,
        "runtime_constraints": {
            "network": "none",
            "root_filesystem": "read-only",
            "container_user": 10001,
        },
        "publication_performed": False,
    }
    _write_json_atomic(resolved["output"], report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
