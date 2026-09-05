#!/usr/bin/env python3
"""Audit exact uv profiles and emit hashed exports and CycloneDX SBOMs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import date
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, Iterable
from urllib.parse import urlsplit

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

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


def _audit_inventory(requirements: str, lock: dict, profile: str) -> dict:
    """Bind advisory queries to the exact active, hashed lock inventory.

    PyPI's advisory service does not recognize official Torch local build tags.
    Only the two official Torch distributions may project such tags to their
    upstream release, retaining the actual identity for exception matching.
    """
    inventory = {}
    environment = default_environment()
    for line in requirements.replace("\\\n", " ").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line.split("--hash=", 1)[0].strip())
        if requirement.marker and not requirement.marker.evaluate(environment):
            continue
        name = canonicalize_name(requirement.name)
        pins = list(requirement.specifier)
        if requirement.url or len(pins) != 1 or pins[0].operator != "==":
            raise ValueError(f"Audit dependency must have one exact version: {name}")
        actual = str(Version(pins[0].version))
        hashes = sorted(set(re.findall(r"--hash=sha256:([a-f0-9]{64})(?=\s|$)", line)))
        if not hashes or name in inventory:
            raise ValueError(
                f"Missing hashes or duplicate active audit dependency: {name}"
            )
        packages = [
            item
            for item in lock.get("package", [])
            if canonicalize_name(item["name"]) == name and item["version"] == actual
        ]
        if len(packages) != 1:
            raise ValueError(
                f"Audit dependency is not unique in the lock: {name}=={actual}"
            )
        package = packages[0]
        distributions = [*package.get("wheels", [])]
        if package.get("sdist"):
            distributions.append(package["sdist"])
        locked_hashes = {
            item.get("hash", "").removeprefix("sha256:") for item in distributions
        }
        if not set(hashes) <= locked_hashes:
            raise ValueError(f"Export hashes do not match the lock: {name}=={actual}")
        audited = actual
        parsed = Version(actual)
        source = package.get("source", {})
        record = {"version": actual, "source": source, "sha256": hashes}
        if parsed.local is not None and name in {"torch", "torchvision"}:
            flavor = "cpu" if profile == "root" else profile.rsplit("-", 1)[-1]
            index = f"https://download.pytorch.org/whl/{flavor}"
            if (
                not re.fullmatch(r"cpu|cu[0-9]{3}", flavor)
                or parsed.local != flavor
                or source != {"registry": index}
            ):
                raise ValueError(f"Unapproved Torch build mapping: {name}=={actual}")
            wheels = package.get("wheels", [])
            if not wheels or any(
                urlsplit(item.get("url", "")).scheme != "https"
                or urlsplit(item.get("url", "")).netloc
                not in {"download.pytorch.org", "download-r2.pytorch.org"}
                for item in wheels
            ):
                raise ValueError(f"Unapproved Torch wheel source: {name}=={actual}")
            audited = parsed.public
            record["mapping"] = "official-pytorch-build-to-upstream-advisory-version"
            record["wheels"] = wheels
        record["audited_version"] = audited
        inventory[name] = record
    if not inventory or not {"torch", "torchvision"} <= inventory.keys():
        raise ValueError("Audit inventory is empty or missing Torch/torchvision")
    return {"environment": environment, "dependencies": inventory}


def _evaluate_audit(
    report, inventory: dict, profile: str, policy: dict, *, today=None
) -> dict:
    """Reject incomplete coverage as well as findings without exact exceptions."""
    expected = inventory["dependencies"]
    coverage_errors = []
    dependencies = report.get("dependencies") if isinstance(report, dict) else None
    if not isinstance(dependencies, list) or not dependencies:
        coverage_errors.append("Auditor returned no dependency inventory")
        dependencies = []
    accepted, unresolved = [], []
    seen, audited = set(), set()
    for dependency in dependencies:
        if not isinstance(dependency, dict) or not isinstance(
            dependency.get("name"), str
        ):
            coverage_errors.append("Malformed audited dependency")
            continue
        name = canonicalize_name(dependency["name"])
        if name not in expected or name in seen:
            coverage_errors.append(
                f"Unexpected or duplicate audited dependency: {name}"
            )
            continue
        seen.add(name)
        record = expected[name]
        if dependency.get("skip_reason"):
            coverage_errors.append(f"Unaudited {name}: {dependency['skip_reason']}")
            continue
        if dependency.get("version") != record["audited_version"]:
            coverage_errors.append(f"Missing or mismatched audited version: {name}")
            continue
        vulns = dependency.get("vulns")
        if not isinstance(vulns, list):
            coverage_errors.append(f"Missing vulnerability result: {name}")
            continue
        audited.add(name)
        seen_ids = set()
        for vulnerability in vulns:
            if (
                not isinstance(vulnerability, dict)
                or not isinstance(vulnerability.get("id"), str)
                or not vulnerability["id"]
                or not isinstance(vulnerability.get("aliases", []), list)
            ):
                coverage_errors.append(f"Malformed vulnerability result: {name}")
                continue
            primary_id = vulnerability["id"]
            aliases = vulnerability.get("aliases", [])
            if not all(isinstance(item, str) and item for item in aliases):
                coverage_errors.append(f"Malformed vulnerability aliases: {name}")
                continue
            ids = {primary_id, *aliases}
            if primary_id in seen_ids:
                continue
            seen_ids.add(primary_id)
            finding = {
                "package": name,
                "version": record["version"],
                "audited_version": record["audited_version"],
                "vulnerability_id": primary_id,
                "aliases": sorted(ids - {primary_id}),
                "fix_versions": vulnerability.get("fix_versions", []),
            }
            exception = _exception_for(
                policy["exceptions"],
                profile=profile,
                package=name,
                version=record["version"],
                vulnerability_ids=ids,
                today=today or date.today(),
                maximum_days=policy["maximum_exception_days"],
            )
            if exception is None:
                unresolved.append(finding)
            else:
                accepted.append(
                    {**finding, "exception_expires_on": exception["expires_on"]}
                )
    for name in sorted(expected.keys() - seen):
        coverage_errors.append(f"Missing audited dependency: {name}")
    return {
        "status": "failed" if coverage_errors or unresolved else "ok",
        "coverage": {
            "expected_count": len(expected),
            "audited_count": len(audited),
            "errors": coverage_errors,
        },
        "accepted_exceptions": accepted,
        "unresolved_findings": unresolved,
    }


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
    audit_requirements_path = profile_output / "audit-requirements.txt"
    inventory_path = profile_output / "audit-inventory.json"

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

    lock_path = project_root / "uv.lock"
    inventory = _audit_inventory(
        requirements_path.read_text(encoding="utf-8"),
        tomllib.loads(lock_path.read_text(encoding="utf-8")),
        profile,
    )
    inventory_path.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    audit_requirements_path.write_text(
        "# Advisory-only projection; requirements.txt retains actual build identities.\n"
        + "\n".join(
            f"{name}=={item['audited_version']} "
            + " ".join(f"--hash=sha256:{digest}" for digest in item["sha256"])
            for name, item in sorted(inventory["dependencies"].items())
        )
        + "\n",
        encoding="utf-8",
    )
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
            f"--requirement={audit_requirements_path}",
        ],
        cwd=repo_root,
    )
    if audit.returncode not in {0, 1} or not audit_path.is_file():
        raise RuntimeError(audit.stdout + audit.stderr)

    report = json.loads(audit_path.read_text(encoding="utf-8"))
    project_path = project_root / "pyproject.toml"
    return {
        "profile": profile,
        **_evaluate_audit(report, inventory, profile, exception_policy),
        "project_sha256": _sha256(project_path),
        "lock_sha256": _sha256(lock_path),
        "requirements_sha256": _sha256(requirements_path),
        "audit_requirements_sha256": _sha256(audit_requirements_path),
        "audit_inventory_sha256": _sha256(inventory_path),
        "sbom_sha256": _sha256(sbom_path),
        "sbom_content_sha256": _sbom_content_sha256(sbom_path),
        "pip_audit_version": PIP_AUDIT_VERSION,
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
