#!/usr/bin/env python3
"""Produce clean-commit technical CUDA evidence on an ephemeral local runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


UV_VERSION = "0.9.14"
CUDA_PROFILES = {
    "2.6": "environments/torch-2.6-cu124",
    "2.11": "environments/torch-2.11-cu130",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(
    command: list[str],
    *,
    cwd: Path,
    environment: Mapping[str, str] | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(command), flush=True)
    result = subprocess.run(
        command,
        cwd=cwd,
        env=dict(environment) if environment is not None else None,
        text=True,
        capture_output=capture,
        check=False,
    )
    if result.returncode != 0:
        if capture:
            sys.stderr.write(result.stdout)
            sys.stderr.write(result.stderr)
        raise RuntimeError(f"Command failed with exit {result.returncode}: {command[0]}")
    return result


def _git(repo_root: Path, *arguments: str) -> str:
    return _run(
        ["git", *arguments], cwd=repo_root, capture=True
    ).stdout.strip()


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


def _ensure_evidence_root(path: Path) -> None:
    """Create a private-but-traversable root or validate an existing one."""
    created = False
    try:
        path.mkdir(parents=True)
    except FileExistsError:
        if not path.is_dir():
            raise RuntimeError(f"Staging root is not a directory: {path}") from None
    else:
        created = True

    if created:
        path.chmod(0o711)

    mode = path.stat().st_mode & 0o777
    if mode != 0o711:
        raise RuntimeError(
            "Staging root must have mode 0711 so non-root production images can "
            f"traverse it without listing evidence; got {mode:04o}: {path}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--python", default="3.10")
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--candidate-evidence",
        action="store_true",
        help="Allow provisional governance for a non-release technical diagnostic.",
    )
    return parser.parse_args()


def _require_approved_release_metadata(repo_root: Path) -> None:
    model_root = repo_root / "facetorch" / "models"
    values = {
        name: json.loads((model_root / f"{name}.json").read_text(encoding="utf-8"))
        for name in ("manifest", "compatibility", "governance")
    }
    non_approved = [
        name for name, value in values.items() if value.get("status") != "approved"
    ]
    ineligible = [
        model_id
        for model_id, record in values["governance"].get("models", {}).items()
        if record.get("release_eligible") is not True
    ]
    if non_approved or ineligible:
        raise RuntimeError(
            "Release CUDA evidence requires approved manifest, compatibility, and "
            f"governance metadata; non_approved={non_approved}, "
            f"ineligible_models={sorted(ineligible)}"
        )


def main() -> int:
    args = _parse_args()
    repo_root = args.repo_root.resolve()
    staging_root = args.staging_root.resolve()
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_sha):
        raise RuntimeError("--source-sha must be a full lowercase commit SHA")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("Release CUDA evidence requires Linux x86_64")
    try:
        staging_root.relative_to(repo_root)
    except ValueError:
        pass
    else:
        raise RuntimeError("Staging root must be outside the source checkout")
    if _git(repo_root, "rev-parse", "HEAD") != args.source_sha:
        raise RuntimeError("Checked-out commit differs from --source-sha")
    if _git(repo_root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("Exact-candidate CUDA evidence requires a clean checkout")
    if not args.candidate_evidence:
        _require_approved_release_metadata(repo_root)
    uv_version = _run(["uv", "--version"], cwd=repo_root, capture=True).stdout.strip()
    if uv_version != f"uv {UV_VERSION}":
        raise RuntimeError(f"Expected uv {UV_VERSION}, got {uv_version!r}")
    gpu_query = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        cwd=repo_root,
        capture=True,
    ).stdout.strip()
    if not gpu_query:
        raise RuntimeError("No NVIDIA GPU attestation was returned")

    _ensure_evidence_root(staging_root)
    summaries = []
    commands = []
    source_environment = os.environ.copy()
    source_environment["PYTHONPATH"] = str(repo_root)
    for cohort, profile_relative in CUDA_PROFILES.items():
        profile = repo_root / profile_relative
        lock_relative = f"{profile_relative}/uv.lock"
        sync_command = [
            "uv",
            "sync",
            "--project",
            str(profile),
            "--frozen",
            "--python",
            args.python,
        ]
        if cohort == "2.6":
            sync_command.extend(["--extra", "release"])
        _run(sync_command, cwd=repo_root)
        commands.append(sync_command)
        cohort_python = profile / ".venv" / "bin" / "python"
        inventory = staging_root / f"source-inventory-torch{cohort}.json"
        prepare_command = [
            str(cohort_python),
            str(repo_root / "scripts" / "export_model_cohorts_hf.py"),
            "prepare-sources",
            "--repo-root",
            str(repo_root),
            "--cohort",
            cohort,
            "--environment-lock",
            lock_relative,
            "--inventory",
            str(inventory),
        ]
        _run(prepare_command, cwd=repo_root, environment=source_environment)
        commands.append(prepare_command)
        cohort_root = staging_root / f"torch-{cohort}"
        export_command = [
            str(cohort_python),
            str(repo_root / "scripts" / "export_model_cohorts_hf.py"),
            "export",
            "--repo-root",
            str(repo_root),
            "--out-root",
            str(cohort_root),
            "--environment-lock",
            lock_relative,
            "--validate-devices",
            "cpu,cuda",
        ]
        _run(export_command, cwd=repo_root, environment=source_environment)
        commands.append(export_command)
        summaries.append(cohort_root / f"summary-torch{cohort}.json")

    matrix_report = staging_root / "candidate-matrix-report.json"
    verify_command = [
        sys.executable,
        str(repo_root / "scripts" / "verify_model_release_matrix.py"),
        "--staging-root",
        str(staging_root),
    ]
    for summary in summaries:
        verify_command.extend(["--summary", str(summary)])
    if args.candidate_evidence:
        verify_command.append("--candidate-evidence")
    verify_command.extend(["--report", str(matrix_report)])
    _run(verify_command, cwd=repo_root, environment=source_environment)
    commands.append(verify_command)

    packaging_residue = [
        path
        for path in (
            repo_root / "build",
            repo_root / "dist",
            repo_root / "facetorch.egg-info",
        )
        if path.exists()
    ]
    if packaging_residue:
        raise RuntimeError(
            "Refusing to build with ignored packaging residue: "
            + ", ".join(path.name for path in packaging_residue)
        )
    distributions = staging_root / "distributions"
    build_command = [
        "uv",
        "build",
        "--wheel",
        "--out-dir",
        str(distributions),
    ]
    _run(build_command, cwd=repo_root)
    commands.append(build_command)
    wheels = list(distributions.glob("facetorch-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("Expected exactly one candidate wheel")
    production_profile = repo_root / CUDA_PROFILES["2.6"]
    production_python = production_profile / ".venv" / "bin" / "python"
    wheel_check_command = [
        str(production_profile / ".venv" / "bin" / "check-wheel-contents"),
        str(wheels[0]),
    ]
    _run(wheel_check_command, cwd=staging_root)
    commands.append(wheel_check_command)
    install_command = [
        "uv",
        "pip",
        "install",
        "--python",
        str(production_python),
        "--no-deps",
        str(wheels[0]),
    ]
    _run(install_command, cwd=staging_root)
    commands.append(install_command)
    smoke_report = staging_root / "default-analyzer-cuda-smoke.json"
    smoke_environment = os.environ.copy()
    smoke_environment.pop("PYTHONPATH", None)
    smoke_command = [
        str(production_python),
        str(repo_root / "scripts" / "smoke_staged_default_analyzer.py"),
        "--repo-root",
        str(repo_root),
        "--staging-root",
        str(staging_root),
        "--summary",
        str(staging_root / "torch-2.6" / "summary-torch2.6.json"),
        "--device",
        "cuda",
        "--report",
        str(smoke_report),
    ]
    _run(smoke_command, cwd=staging_root, environment=smoke_environment)
    commands.append(smoke_command)

    notebook_path = staging_root / "facetorch-notebook-executed.ipynb"
    notebook_report = staging_root / "facetorch-notebook-report.json"
    notebook_command = [
        str(production_python),
        str(repo_root / "scripts" / "execute_candidate_notebook.py"),
        "--repo-root",
        str(repo_root),
        "--staging-root",
        str(staging_root),
        "--summary",
        str(staging_root / "torch-2.6" / "summary-torch2.6.json"),
        "--wheel",
        str(wheels[0]),
        "--device",
        "cuda",
        "--output-notebook",
        str(notebook_path),
        "--report",
        str(notebook_report),
    ]
    _run(notebook_command, cwd=staging_root, environment=smoke_environment)
    commands.append(notebook_command)

    if _git(repo_root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("Release runner changed the exact source checkout")
    report_path = (args.report or staging_root / "local-cuda-runner-report.json").resolve()
    report = {
        "schema_version": 1,
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": args.source_sha,
        "source_clean": True,
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "gpu_attestation": gpu_query,
        "uv_version": uv_version,
        "manifest_sha256": _sha256(
            repo_root / "facetorch" / "models" / "manifest.json"
        ),
        "compatibility_sha256": _sha256(
            repo_root / "facetorch" / "models" / "compatibility.json"
        ),
        "governance_sha256": _sha256(
            repo_root / "facetorch" / "models" / "governance.json"
        ),
        "environment_locks": {
            cohort: {
                "path": f"{profile}/uv.lock",
                "sha256": _sha256(repo_root / profile / "uv.lock"),
            }
            for cohort, profile in CUDA_PROFILES.items()
        },
        "summaries": [
            {"path": str(path.relative_to(staging_root)), "sha256": _sha256(path)}
            for path in summaries
        ],
        "matrix_report_sha256": _sha256(matrix_report),
        "wheel": {"filename": wheels[0].name, "sha256": _sha256(wheels[0])},
        "default_analyzer_smoke_sha256": _sha256(smoke_report),
        "notebook_report_sha256": _sha256(notebook_report),
        "executed_notebook_sha256": _sha256(notebook_path),
        "commands": commands,
        "publication_performed": False,
        "candidate_evidence_only": args.candidate_evidence,
    }
    _write_json_atomic(report_path, report)
    print(f"Local CUDA release report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
