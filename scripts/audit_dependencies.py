#!/usr/bin/env python3
"""Audit exact uv profiles and emit hashed exports and CycloneDX SBOMs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import date
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, Iterable

PROFILE_PROJECTS = {
    "root": Path("."),
    "torch-2.6-cpu": Path("environments/torch-2.6-cpu"),
    "torch-2.7-cpu": Path("environments/torch-2.7-cpu"),
    "torch-2.8-cpu": Path("environments/torch-2.8-cpu"),
    "torch-2.9-cpu": Path("environments/torch-2.9-cpu"),
    "torch-2.10-cpu": Path("environments/torch-2.10-cpu"),
    "torch-2.11-cpu": Path("environments/torch-2.11-cpu"),
    "torch-2.12-cpu": Path("environments/torch-2.12-cpu"),
    "torch-2.13-cpu": Path("environments/torch-2.13-cpu"),
    "torch-2.6-cu124": Path("environments/torch-2.6-cu124"),
    "torch-2.7-cu126": Path("environments/torch-2.7-cu126"),
    "torch-2.8-cu126": Path("environments/torch-2.8-cu126"),
    "torch-2.9-cu130": Path("environments/torch-2.9-cu130"),
    "torch-2.10-cu130": Path("environments/torch-2.10-cu130"),
    "torch-2.11-cu130": Path("environments/torch-2.11-cu130"),
    "torch-2.12-cu130": Path("environments/torch-2.12-cu130"),
    "torch-2.13-cu130": Path("environments/torch-2.13-cu130"),
}
PIP_AUDIT_VERSION = "2.10.1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sbom_content_sha256(path: Path) -> str:
    """Hash the dependency graph without per-generation SBOM identity fields."""
    sbom = json.loads(path.read_text(encoding="utf-8"))
    sbom.pop("serialNumber", None)
    metadata = sbom.get("metadata")
    if isinstance(metadata, dict):
        metadata.pop("timestamp", None)
    canonical = json.dumps(
        sbom, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def _require_locked_auditor() -> None:
    try:
        installed = version("pip-audit")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "pip-audit must be installed from the locked release dependency group"
        ) from exc
    if installed != PIP_AUDIT_VERSION:
        raise RuntimeError(f"Expected pip-audit {PIP_AUDIT_VERSION}, found {installed}")


def _load_exceptions(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported advisory exception schema")
    maximum_days = data.get("maximum_exception_days")
    if not isinstance(maximum_days, int) or maximum_days <= 0:
        raise ValueError("maximum_exception_days must be a positive integer")
    return data


def _exception_for(
    exceptions: Iterable[Dict[str, Any]],
    *,
    profile: str,
    package: str,
    version: str,
    vulnerability_ids: set[str],
    today: date,
    maximum_days: int,
) -> Dict[str, Any] | None:
    for exception in exceptions:
        exception_ids = {
            str(exception.get("vulnerability_id", "")),
            *(str(item) for item in exception.get("aliases", [])),
        }
        if (
            exception.get("package") != package
            or version not in exception.get("versions", [])
            or profile not in exception.get("profiles", [])
            or not exception_ids.intersection(vulnerability_ids)
        ):
            continue
        if exception.get("status") != "approved":
            return None
        approved_on = date.fromisoformat(str(exception["approved_on"]))
        expires_on = date.fromisoformat(str(exception["expires_on"]))
        duration = (expires_on - approved_on).days
        if (
            approved_on > today
            or expires_on < today
            or duration < 0
            or duration > maximum_days
        ):
            return None
        if not exception.get("rationale") or not exception.get("mitigations"):
            return None
        return exception
    return None


def _audit_profile(
    repo_root: Path,
    profile: str,
    project: Path,
    output_dir: Path,
    exception_policy: Dict[str, Any],
) -> Dict[str, Any]:
    project_root = (repo_root / project).resolve()
    profile_output = output_dir / profile
    profile_output.mkdir(parents=True, exist_ok=True)
    requirements_path = profile_output / "requirements.txt"
    sbom_path = profile_output / "sbom.cdx.json"
    audit_path = profile_output / "pip-audit.json"

    export_base = [
        "uv",
        "export",
        "--project",
        str(project_root),
        "--frozen",
        "--no-dev",
        "--no-emit-project",
    ]
    requirements = _run(
        [*export_base, "--no-header", "--output-file", str(requirements_path)],
        cwd=repo_root,
    )
    if requirements.returncode != 0:
        raise RuntimeError(requirements.stdout + requirements.stderr)
    sbom = _run(
        [
            *export_base,
            "--format",
            "cyclonedx1.5",
            "--output-file",
            str(sbom_path),
        ],
        cwd=repo_root,
    )
    if sbom.returncode != 0:
        raise RuntimeError(sbom.stdout + sbom.stderr)

    audit = _run(
        [
            sys.executable,
            "-m",
            "pip_audit",
            "--progress-spinner=off",
            "--require-hashes",
            "--disable-pip",
            "--aliases=on",
            "--desc=off",
            "--format=json",
            f"--output={audit_path}",
            f"--requirement={requirements_path}",
        ],
        cwd=repo_root,
    )
    if audit.returncode not in {0, 1} or not audit_path.is_file():
        raise RuntimeError(audit.stdout + audit.stderr)

    report = json.loads(audit_path.read_text(encoding="utf-8"))
    unresolved = []
    accepted = []
    seen = set()
    today = date.today()
    for dependency in report.get("dependencies", []):
        package = str(dependency.get("name", "")).lower().replace("_", "-")
        version = str(dependency.get("version", ""))
        for vulnerability in dependency.get("vulns", []):
            primary_id = str(vulnerability.get("id", ""))
            ids = {
                primary_id,
                *(str(item) for item in vulnerability.get("aliases", [])),
            }
            identity = (package, version, primary_id)
            if identity in seen:
                continue
            seen.add(identity)
            finding = {
                "package": package,
                "version": version,
                "vulnerability_id": primary_id,
                "aliases": sorted(ids - {primary_id}),
                "fix_versions": vulnerability.get("fix_versions", []),
            }
            exception = _exception_for(
                exception_policy["exceptions"],
                profile=profile,
                package=package,
                version=version,
                vulnerability_ids=ids,
                today=today,
                maximum_days=exception_policy["maximum_exception_days"],
            )
            if exception is None:
                unresolved.append(finding)
            else:
                accepted.append(
                    {**finding, "exception_expires_on": exception["expires_on"]}
                )

    lock_path = project_root / "uv.lock"
    project_path = project_root / "pyproject.toml"
    return {
        "profile": profile,
        "status": "ok" if not unresolved else "failed",
        "project_sha256": _sha256(project_path),
        "lock_sha256": _sha256(lock_path),
        "requirements_sha256": _sha256(requirements_path),
        "sbom_sha256": _sha256(sbom_path),
        "sbom_content_sha256": _sbom_content_sha256(sbom_path),
        "pip_audit_version": PIP_AUDIT_VERSION,
        "accepted_exceptions": accepted,
        "unresolved_findings": unresolved,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        action="append",
        choices=sorted(PROFILE_PROJECTS),
        help="Profile to audit; repeat as needed. Defaults to every profile.",
    )
    parser.add_argument(
        "--exceptions",
        type=Path,
        default=Path("security/advisory-exceptions.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("build/dependency-audit")
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _require_locked_auditor()
    repo_root = Path(__file__).resolve().parents[1]
    exception_path = (repo_root / args.exceptions).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    exception_policy = _load_exceptions(exception_path)
    profiles = args.profile or list(PROFILE_PROJECTS)

    reports = []
    for profile in profiles:
        reports.append(
            _audit_profile(
                repo_root,
                profile,
                PROFILE_PROJECTS[profile],
                output_dir,
                exception_policy,
            )
        )
    summary = {
        "schema_version": 1,
        "status": "ok" if all(item["status"] == "ok" for item in reports) else "failed",
        "exception_policy_sha256": _sha256(exception_path),
        "profiles": reports,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
