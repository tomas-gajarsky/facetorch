#!/usr/bin/env python3
"""Plan, verify, and safely resume a coordinated facetorch release.

The module deliberately separates immutable artifact preparation from external
publication.  Each public channel is reconciled against the digest in one release
plan: an identical remote object is accepted on retry, a missing object may be
published, and a different object for the same version is always fatal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen


PLAN_SCHEMA_VERSION = 1
RECEIPT_SCHEMA_VERSION = 1
PROJECT_NAME = "facetorch"
IMMUTABLE_CHANNELS = (
    "model-manifest",
    "github-release",
    "docker-cpu",
    "docker-gpu",
    "pypi",
)
PUBLICATION_ORDER = (
    "model-manifest",
    "github-release",
    "docker-cpu",
    "docker-gpu",
    "pypi",
)

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_TORCH_MINOR_RE = re.compile(r"^(?P<major>0|[1-9][0-9]*)\.(?P<minor>0|[1-9][0-9]*)$")
_REPO_ID_RE = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,94}[A-Za-z0-9])?/"
    r"[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,94}[A-Za-z0-9])?$"
)
_TAG_RE = re.compile(
    r"^v(?P<major>0|[1-9][0-9]*)\."
    r"(?P<minor>0|[1-9][0-9]*)\."
    r"(?P<patch>0|[1-9][0-9]*)"
    r"(?:-rc\.(?P<rc>[1-9][0-9]*))?$"
)


class ReleaseError(RuntimeError):
    """Raised when a release identity or remote state is unsafe."""


class PublicationBackend(Protocol):
    """Minimal interface used by the retry/failure state-machine tests."""

    def observe(self, channel: str) -> str | None:
        ...

    def publish(self, channel: str, expected_digest: str) -> str:
        ...


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_bytes_atomic(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as output:
            output.write(value)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, value: Any) -> None:
    _write_bytes_atomic(path, _canonical_json_bytes(value))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseError(f"Cannot read JSON document {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReleaseError(f"JSON document must contain an object: {path}")
    return value


def _require_sha(value: Any, label: str) -> str:
    result = str(value).strip().lower()
    if _SHA_RE.fullmatch(result) is None:
        raise ReleaseError(f"{label} must be a full lowercase 40-character SHA")
    return result


def _require_sha256(value: Any, label: str) -> str:
    result = str(value).strip().lower()
    if _SHA256_RE.fullmatch(result) is None:
        raise ReleaseError(f"{label} must be a lowercase SHA-256 digest")
    return result


def _require_image_digest(value: Any, label: str) -> str:
    result = str(value).strip().lower()
    if _IMAGE_DIGEST_RE.fullmatch(result) is None:
        raise ReleaseError(f"{label} must be a sha256: container digest")
    return result


def parse_release_tag(tag: str) -> dict[str, Any]:
    """Validate the public tag grammar and return normalized release identity."""

    match = _TAG_RE.fullmatch(str(tag).strip())
    if match is None:
        raise ReleaseError(
            "Release tag must be vMAJOR.MINOR.PATCH or vMAJOR.MINOR.PATCH-rc.N"
        )
    base = ".".join(match.group(name) for name in ("major", "minor", "patch"))
    rc = match.group("rc")
    return {
        "tag": match.group(0),
        "project_version": base if rc is None else f"{base}rc{rc}",
        "docker_tag": base if rc is None else f"{base}-rc.{rc}",
        "release_kind": "stable" if rc is None else "rc",
        "is_prerelease": rc is not None,
    }


def tag_for_project_version(version: str) -> str:
    """Convert a strict PEP 440 stable/RC project version to the public tag."""

    match = re.fullmatch(
        r"(?P<base>(?:0|[1-9][0-9]*)\."
        r"(?:0|[1-9][0-9]*)\."
        r"(?:0|[1-9][0-9]*))(?:rc(?P<rc>[1-9][0-9]*))?",
        str(version).strip(),
    )
    if match is None:
        raise ReleaseError("Project version must be a strict stable or RC version")
    suffix = "" if match.group("rc") is None else f"-rc.{match.group('rc')}"
    return f"v{match.group('base')}{suffix}"


def _run_git(repo_root: Path, *arguments: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        raise ReleaseError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def _project_version(repo_root: Path) -> str:
    content = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    project = content.split("[project]", 1)
    if len(project) != 2:
        raise ReleaseError("pyproject.toml has no [project] table")
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"\s*$', project[1])
    if match is None:
        raise ReleaseError("pyproject.toml has no static project version")
    return match.group(1)


def validate_candidate_identity(
    repo_root: Path,
    *,
    source_sha: str,
    tag: str,
    allow_missing_tag: bool,
) -> dict[str, Any]:
    """Bind metadata, changelog, checkout, and optional annotated tag."""

    root = repo_root.resolve()
    source = _require_sha(source_sha, "Source SHA")
    identity = parse_release_tag(tag)
    version = _project_version(root)
    if version != identity["project_version"]:
        raise ReleaseError(
            f"Tag {identity['tag']} resolves to version {identity['project_version']}, "
            f"but pyproject.toml declares {version}"
        )
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    heading = re.compile(rf"(?m)^##\s+{re.escape(version)}(?:\s|$)")
    if heading.search(changelog) is None:
        raise ReleaseError(f"CHANGELOG.md has no section for version {version}")

    head = _run_git(root, "rev-parse", "HEAD")
    if head != source:
        raise ReleaseError(f"Checked-out HEAD {head} differs from source SHA {source}")
    dirty = _run_git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise ReleaseError("Release candidate checkout must be clean")

    tag_ref = f"refs/tags/{identity['tag']}"
    tag_exists = (
        subprocess.run(
            ["git", "show-ref", "--verify", "--quiet", tag_ref],
            cwd=root,
            check=False,
        ).returncode
        == 0
    )
    if not tag_exists:
        if not allow_missing_tag:
            raise ReleaseError(f"Required annotated tag does not exist: {identity['tag']}")
    else:
        tagged_commit = _run_git(root, "rev-parse", f"{tag_ref}^{{commit}}")
        if tagged_commit != source:
            raise ReleaseError(
                f"Tag {identity['tag']} resolves to {tagged_commit}, not {source}"
            )
        if not allow_missing_tag and _run_git(root, "cat-file", "-t", tag_ref) != "tag":
            raise ReleaseError("Publication requires an annotated tag")

    return {**identity, "source_sha": source, "tag_exists": tag_exists}


def _safe_relative_file(root: Path, relative: str) -> Path:
    value = PurePosixPath(relative)
    if value.is_absolute() or ".." in value.parts or not value.parts:
        raise ReleaseError(f"Unsafe bundle path: {relative!r}")
    path = root.joinpath(*value.parts)
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root.resolve())
    except (OSError, ValueError) as exc:
        raise ReleaseError(f"Bundle file escapes its root: {relative!r}") from exc
    if path.is_symlink() or not resolved.is_file():
        raise ReleaseError(f"Bundle member must be a regular non-symlink file: {relative}")
    return resolved


def validate_model_manifest(
    path: Path,
    *,
    repo_id: str,
    revision: str,
    expected_sha256: str,
) -> dict[str, Any]:
    """Validate the immutable manifest identity and its complete artifact list."""

    if _REPO_ID_RE.fullmatch(repo_id) is None:
        raise ReleaseError("Model manifest repository must be OWNER/REPOSITORY")
    commit = _require_sha(revision, "Model manifest revision")
    expected = _require_sha256(expected_sha256, "Model manifest digest")
    observed = sha256_file(path)
    if observed != expected:
        raise ReleaseError(
            f"Model manifest digest mismatch: expected {expected}, observed {observed}"
        )
    manifest = _read_json(path)
    if manifest.get("schema_version") != 1:
        raise ReleaseError("Unsupported model manifest schema")
    if manifest.get("status") not in {"candidate", "approved"}:
        raise ReleaseError("Model manifest is not a release candidate")
    models = manifest.get("models")
    if not isinstance(models, list) or not models:
        raise ReleaseError("Model manifest contains no model cohort records")
    seen = set()
    for model in models:
        if not isinstance(model, dict):
            raise ReleaseError("Model manifest contains an invalid record")
        identity = (str(model.get("model_id", "")), str(model.get("cohort", "")))
        if not all(identity) or identity in seen:
            raise ReleaseError(f"Invalid or duplicate model cohort record: {identity}")
        seen.add(identity)
        _require_sha(model.get("revision"), f"Revision for {identity}")
        _require_sha256(model.get("artifact_sha256"), f"Artifact digest for {identity}")
        if int(model.get("artifact_size_bytes", 0)) <= 0:
            raise ReleaseError(f"Artifact size is missing for {identity}")
        devices = model.get("required_devices")
        if not isinstance(devices, list) or set(devices) != {"cpu", "cuda"}:
            raise ReleaseError(f"Required CPU/CUDA evidence is incomplete for {identity}")
    return {
        "repo_id": repo_id,
        "revision": commit,
        "filename": path.name,
        "sha256": observed,
        "plan_id": manifest.get("plan_id"),
        "model_cohort_count": len(models),
    }


def validate_packaged_model_governance(
    repo_root: Path,
    *,
    remote_manifest_path: Path,
    remote_revision: str,
) -> None:
    """Require approved rights/compatibility records bound to the Hub manifest."""

    model_root = repo_root.resolve() / "facetorch" / "models"
    packaged = _read_json(model_root / "manifest.json")
    compatibility = _read_json(model_root / "compatibility.json")
    governance = _read_json(model_root / "governance.json")
    revision = _require_sha(remote_revision, "Model manifest revision")
    if packaged.get("status") != "approved" or packaged.get("manifest_revision") != revision:
        raise ReleaseError(
            "Packaged model manifest must be approved and bind the exact Hub revision"
        )
    if compatibility.get("status") != "approved":
        raise ReleaseError("Packaged model compatibility matrix is not approved")
    if governance.get("status") != "approved":
        raise ReleaseError("Packaged model governance is not approved")

    packaged_models = packaged.get("models")
    governance_models = governance.get("models")
    if (
        not isinstance(packaged_models, dict)
        or not packaged_models
        or not isinstance(governance_models, dict)
        or set(governance_models) != set(packaged_models)
    ):
        raise ReleaseError("Packaged governance must exactly cover every model")
    for model_id, record in governance_models.items():
        rights = record.get("rights", {}) if isinstance(record, dict) else {}
        checkpoint = (
            record.get("source_checkpoint", {}) if isinstance(record, dict) else {}
        )
        if (
            record.get("status") != "approved"
            or record.get("release_eligible") is not True
            or checkpoint.get("upstream_checkpoint_mapping") != "verified"
            or rights.get("weights_license") in {None, "unverified"}
            or rights.get("redistribution") != "approved"
            or rights.get("attribution") != "approved"
            or rights.get("owner_approval") != "approved"
            or not record.get("limitations")
        ):
            raise ReleaseError(f"Model governance is incomplete for {model_id}")

    torch_policy = compatibility.get("torch")
    platform_policy = compatibility.get("platform_policy")
    supported_values = (
        torch_policy.get("supported_minor_lines")
        if isinstance(torch_policy, dict)
        else None
    )
    required_values = (
        platform_policy.get("required_devices")
        if isinstance(platform_policy, dict)
        else None
    )
    if (
        not isinstance(supported_values, list)
        or not supported_values
        or not isinstance(required_values, list)
        or not required_values
    ):
        raise ReleaseError("Packaged compatibility cohorts or devices are incomplete")

    def exact_text(value: Any, label: str) -> str:
        if not isinstance(value, str) or not value:
            raise ReleaseError(f"{label} must be a non-empty string")
        return value

    def cohort_range(value: Any, label: str) -> tuple[str, str]:
        cohort = exact_text(value, label)
        match = _TORCH_MINOR_RE.fullmatch(cohort)
        if match is None:
            raise ReleaseError(f"{label} must be a canonical Torch major.minor line")
        upper = f"{match.group('major')}.{int(match.group('minor')) + 1}"
        return cohort, upper

    supported_cohorts = {
        cohort_range(value, "Supported Torch cohort")[0] for value in supported_values
    }
    if len(supported_cohorts) != len(supported_values):
        raise ReleaseError("Packaged compatibility has duplicate Torch cohorts")
    if any(not isinstance(value, str) or not value for value in required_values) or len(
        set(required_values)
    ) != len(required_values):
        raise ReleaseError("Packaged compatibility has invalid required devices")
    required_devices = tuple(sorted(required_values))

    expected_records: dict[tuple[str, str], dict[str, Any]] = {}
    for model_id, model in packaged_models.items():
        if not isinstance(model, dict):
            raise ReleaseError(f"Packaged model record is invalid for {model_id}")
        repo_id = exact_text(
            model.get("repo_id"), f"Packaged repository for {model_id}"
        )
        if _REPO_ID_RE.fullmatch(repo_id) is None:
            raise ReleaseError(f"Packaged repository is invalid for {model_id}")
        model_revision = _require_sha(
            exact_text(
                model.get("revision"), f"Packaged revision for {model_id}"
            ),
            f"Packaged revision for {model_id}",
        )
        artifacts = model.get("artifacts")
        if not isinstance(artifacts, list):
            raise ReleaseError(f"Packaged artifacts are invalid for {model_id}")
        model_cohorts: set[str] = set()
        for artifact in artifacts:
            if not isinstance(artifact, dict) or artifact.get("format") != "pt2":
                continue
            cohort, expected_upper = cohort_range(
                artifact.get("torch_min"),
                f"Packaged cohort for {model_id}",
            )
            packaged_upper = exact_text(
                artifact.get("torch_max_exclusive"),
                f"Packaged maximum Torch version for {model_id}/{cohort}",
            )
            if packaged_upper != expected_upper:
                raise ReleaseError(
                    f"Packaged cohort range is invalid for {model_id}/{cohort}"
                )
            key = (str(model_id), cohort)
            if key in expected_records:
                raise ReleaseError(
                    f"Duplicate packaged cohort record: {model_id}/{cohort}"
                )
            size = artifact.get("size_bytes")
            devices = artifact.get("devices")
            if isinstance(size, bool) or not isinstance(size, int) or size < 1:
                raise ReleaseError(
                    f"Packaged artifact size is invalid for {model_id}/{cohort}"
                )
            if (
                not isinstance(devices, list)
                or any(not isinstance(value, str) or not value for value in devices)
                or len(set(devices)) != len(devices)
                or tuple(sorted(devices)) != required_devices
            ):
                raise ReleaseError(
                    f"Packaged devices are invalid for {model_id}/{cohort}"
                )
            filename = exact_text(
                artifact.get("filename"),
                f"Packaged artifact filename for {model_id}/{cohort}",
            )
            if PurePosixPath(filename).name != filename:
                raise ReleaseError(
                    f"Packaged artifact filename is invalid for {model_id}/{cohort}"
                )
            expected_records[key] = {
                "model_id": str(model_id),
                "repo_id": repo_id,
                "cohort": cohort,
                "revision": model_revision,
                "artifact_filename": filename,
                "artifact_sha256": _require_sha256(
                    exact_text(
                        artifact.get("sha256"),
                        f"Packaged artifact digest for {model_id}/{cohort}",
                    ),
                    f"Packaged artifact digest for {model_id}/{cohort}",
                ),
                "artifact_size_bytes": size,
                "required_devices": required_devices,
                "torch_min": cohort,
                "torch_max_exclusive": expected_upper,
            }
            model_cohorts.add(cohort)
        if model_cohorts != supported_cohorts:
            raise ReleaseError(
                f"Packaged cohorts do not match compatibility for {model_id}"
            )
        export_commit = model.get("export_commit")
        license_ref = model.get("license_ref")
        if (
            not isinstance(export_commit, str)
            or _SHA_RE.fullmatch(export_commit) is None
            or not isinstance(license_ref, str)
            or not license_ref
        ):
            raise ReleaseError(f"Export provenance is incomplete for model {model_id}")

    remote = _read_json(remote_manifest_path)
    remote_models = remote.get("models")
    if not isinstance(remote_models, list) or not remote_models:
        raise ReleaseError("Remote model manifest contains no records")
    observed_records: dict[tuple[str, str], dict[str, Any]] = {}
    for record in remote_models:
        if not isinstance(record, dict):
            raise ReleaseError("Remote model manifest contains an invalid record")
        model_id = exact_text(record.get("model_id"), "Remote model ID")
        cohort, inferred_upper = cohort_range(
            record.get("cohort"),
            f"Remote cohort for {model_id or 'unknown model'}",
        )
        key = (model_id, cohort)
        if key in observed_records:
            raise ReleaseError(f"Duplicate remote cohort record: {model_id}/{cohort}")
        size = record.get("artifact_size_bytes")
        devices = record.get("required_devices")
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ReleaseError(
                f"Remote artifact size is invalid for {model_id}/{cohort}"
            )
        if (
            not isinstance(devices, list)
            or any(not isinstance(value, str) or not value for value in devices)
            or len(set(devices)) != len(devices)
        ):
            raise ReleaseError(f"Remote devices are invalid for {model_id}/{cohort}")
        filename = exact_text(
            record.get("artifact_filename"),
            f"Remote artifact filename for {model_id}/{cohort}",
        )
        if PurePosixPath(filename).name != filename:
            raise ReleaseError(
                f"Remote artifact filename is invalid for {model_id}/{cohort}"
            )
        observed_records[key] = {
            "model_id": model_id,
            "repo_id": exact_text(
                record.get("repo_id"), f"Remote repository for {model_id}/{cohort}"
            ),
            "cohort": cohort,
            "revision": _require_sha(
                exact_text(
                    record.get("revision"),
                    f"Remote revision for {model_id}/{cohort}",
                ),
                f"Remote revision for {model_id}/{cohort}",
            ),
            "artifact_filename": filename,
            "artifact_sha256": _require_sha256(
                exact_text(
                    record.get("artifact_sha256"),
                    f"Remote artifact digest for {model_id}/{cohort}",
                ),
                f"Remote artifact digest for {model_id}/{cohort}",
            ),
            "artifact_size_bytes": size,
            "required_devices": tuple(sorted(devices)),
            "torch_min": exact_text(
                record.get("torch_min", cohort),
                f"Remote minimum Torch version for {model_id}/{cohort}",
            ),
            "torch_max_exclusive": exact_text(
                record.get("torch_max_exclusive", inferred_upper),
                f"Remote maximum Torch version for {model_id}/{cohort}",
            ),
        }

    if set(observed_records) != set(expected_records):
        raise ReleaseError("Packaged and remote model cohort coverage differs")
    for key, expected in expected_records.items():
        observed = observed_records[key]
        differing = sorted(
            field for field in expected if observed.get(field) != expected[field]
        )
        if differing:
            identity = "/".join(key)
            raise ReleaseError(
                f"Remote cohort record differs for {identity}: {', '.join(differing)}"
            )


def validate_local_release_evidence(
    repo_root: Path,
    evidence_root: Path,
    *,
    source_sha: str,
    cpu_image_digest: str,
    gpu_image_digest: str,
) -> dict[str, Any]:
    """Verify exact-candidate CUDA and production-image evidence fail closed."""

    repo = repo_root.resolve()
    root = evidence_root.resolve()
    source = _require_sha(source_sha, "GPU evidence source SHA")
    cpu_digest = _require_image_digest(cpu_image_digest, "GPU evidence CPU image")
    gpu_digest = _require_image_digest(gpu_image_digest, "GPU evidence GPU image")

    runner_path = _safe_relative_file(root, "local-cuda-runner-report.json")
    matrix_path = _safe_relative_file(root, "candidate-matrix-report.json")
    default_smoke_path = _safe_relative_file(
        root, "default-analyzer-cuda-smoke.json"
    )
    notebook_report_path = _safe_relative_file(root, "facetorch-notebook-report.json")
    notebook_path = _safe_relative_file(root, "facetorch-notebook-executed.ipynb")
    container_path = _safe_relative_file(root, "container-evidence.json")
    container_smokes = {
        "cpu": _safe_relative_file(root, "container-reports/cpu-image-smoke.json"),
        "gpu": _safe_relative_file(root, "container-reports/gpu-image-smoke.json"),
    }

    runner = _read_json(runner_path)
    if (
        runner.get("schema_version") != 1
        or runner.get("status") != "ok"
        or runner.get("source_sha") != source
        or runner.get("source_clean") is not True
        or runner.get("publication_performed") is not False
        or runner.get("candidate_evidence_only") is not False
        or runner.get("uv_version") != "uv 0.9.14"
        or runner.get("platform") != {"system": "Linux", "machine": "x86_64"}
        or not runner.get("gpu_attestation")
    ):
        raise ReleaseError("Local CUDA runner evidence is not release eligible")

    model_root = repo / "facetorch" / "models"
    source_records = {
        "manifest_sha256": model_root / "manifest.json",
        "compatibility_sha256": model_root / "compatibility.json",
        "governance_sha256": model_root / "governance.json",
    }
    for field, path in source_records.items():
        if runner.get(field) != sha256_file(path):
            raise ReleaseError(f"Local CUDA evidence has a different {field}")

    compatibility = _read_json(model_root / "compatibility.json")
    supported_cohorts = {
        str(value) for value in compatibility.get("torch", {}).get("supported_minor_lines", [])
    }
    required_devices = {
        str(value)
        for value in compatibility.get("platform_policy", {}).get(
            "required_devices", []
        )
    }
    locks = runner.get("environment_locks")
    if not isinstance(locks, dict) or set(locks) != supported_cohorts:
        raise ReleaseError("Local CUDA evidence does not cover every supported cohort lock")
    for cohort, record in locks.items():
        if not isinstance(record, dict):
            raise ReleaseError(f"Invalid environment lock evidence for Torch {cohort}")
        lock_path = _safe_relative_file(repo, str(record.get("path", "")))
        if record.get("sha256") != sha256_file(lock_path):
            raise ReleaseError(f"Environment lock changed for Torch {cohort}")

    summaries = runner.get("summaries")
    if not isinstance(summaries, list) or len(summaries) != len(supported_cohorts):
        raise ReleaseError("Local CUDA evidence has incomplete cohort summaries")
    summary_paths = set()
    for record in summaries:
        if not isinstance(record, dict):
            raise ReleaseError("Local CUDA evidence contains an invalid summary")
        relative = str(record.get("path", ""))
        if relative in summary_paths:
            raise ReleaseError("Local CUDA evidence repeats a cohort summary")
        summary_paths.add(relative)
        path = _safe_relative_file(root, relative)
        if record.get("sha256") != sha256_file(path):
            raise ReleaseError(f"CUDA cohort summary changed: {relative}")

    matrix = _read_json(matrix_path)
    lanes = matrix.get("lanes")
    if (
        runner.get("matrix_report_sha256") != sha256_file(matrix_path)
        or matrix.get("schema_version") != 1
        or matrix.get("status") != "ok"
        or matrix.get("release_approval_required") is not False
        or set(matrix.get("required_devices", [])) != required_devices
        or not isinstance(lanes, list)
        or {str(lane.get("torch_minor", "")) for lane in lanes if isinstance(lane, dict)}
        != supported_cohorts
        or any(
            not isinstance(lane, dict)
            or lane.get("source_commit") != source
            or lane.get("source_clean") is not True
            for lane in lanes
        )
    ):
        raise ReleaseError("Candidate CUDA matrix evidence is incomplete or mismatched")

    default_smoke = _read_json(default_smoke_path)
    notebook_report = _read_json(notebook_report_path)
    if (
        runner.get("default_analyzer_smoke_sha256") != sha256_file(default_smoke_path)
        or default_smoke.get("status") != "ok"
        or default_smoke.get("device") != "cuda"
        or default_smoke.get("legacy_fallback") is not False
        or runner.get("notebook_report_sha256") != sha256_file(notebook_report_path)
        or notebook_report.get("status") != "ok"
        or notebook_report.get("device") != "cuda"
        or runner.get("executed_notebook_sha256") != sha256_file(notebook_path)
    ):
        raise ReleaseError("CUDA analyzer or notebook evidence is incomplete")

    container = _read_json(container_path)
    if (
        container.get("schema_version") != 1
        or container.get("status") != "ok"
        or container.get("source_sha") != source
        or container.get("publication_performed") is not False
        or container.get("runner_report_sha256") != sha256_file(runner_path)
        or container.get("runtime_constraints")
        != {
            "network": "none",
            "root_filesystem": "read-only",
            "container_user": 10001,
        }
    ):
        raise ReleaseError("Production-container evidence is incomplete")
    expected_images = {"cpu": cpu_digest, "gpu": gpu_digest}
    images = container.get("images")
    if not isinstance(images, dict) or set(images) != set(expected_images):
        raise ReleaseError("Production-container evidence has the wrong image set")
    for flavor, expected_digest in expected_images.items():
        report = _read_json(container_smokes[flavor])
        expected_device = "cuda" if flavor == "gpu" else "cpu"
        image = images[flavor]
        if (
            not isinstance(image, dict)
            or image.get("image_id") != expected_digest
            or image.get("os") != "linux"
            or image.get("architecture") != "amd64"
            or image.get("configured_user") != "facetorch"
            or image.get("dockerfile_sha256")
            != sha256_file(repo / "docker" / ("Dockerfile.gpu" if flavor == "gpu" else "Dockerfile"))
            or image.get("smoke_report_sha256") != sha256_file(container_smokes[flavor])
            or report.get("status") != "ok"
            or report.get("device") != expected_device
            or report.get("uid") != 10001
            or report.get("legacy_fallback") is not False
        ):
            raise ReleaseError(f"Exact {flavor} production image evidence is invalid")

    return {
        "schema_version": 1,
        "status": "verified",
        "source_sha": source,
        "runner_report_sha256": sha256_file(runner_path),
        "matrix_report_sha256": sha256_file(matrix_path),
        "container_evidence_sha256": sha256_file(container_path),
        "cpu_image_digest": cpu_digest,
        "gpu_image_digest": gpu_digest,
        "supported_cohorts": sorted(supported_cohorts),
        "required_devices": sorted(required_devices),
    }


def fetch_model_manifest(
    *,
    repo_id: str,
    revision: str,
    filename: str,
    expected_sha256: str,
    output_path: Path,
    opener: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    """Fetch one public Hub file by immutable revision and verify it before use."""

    if _REPO_ID_RE.fullmatch(repo_id) is None:
        raise ReleaseError("Model manifest repository must be OWNER/REPOSITORY")
    commit = _require_sha(revision, "Model manifest revision")
    expected = _require_sha256(expected_sha256, "Model manifest digest")
    filename_value = str(filename)
    remote_path = PurePosixPath(filename_value)
    if (
        len(filename_value) > 512
        or remote_path.is_absolute()
        or not remote_path.parts
        or any(
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", part) is None
            for part in filename_value.split("/")
        )
    ):
        raise ReleaseError("Model manifest filename must be a safe relative path")
    encoded = "/".join(quote(part, safe="") for part in remote_path.parts)
    url = f"https://huggingface.co/{repo_id}/resolve/{commit}/{encoded}"
    try:
        with opener(url, timeout=30) as response:
            payload = response.read(16 * 1024 * 1024 + 1)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise ReleaseError(f"Cannot fetch immutable model manifest: {exc}") from exc
    if len(payload) > 16 * 1024 * 1024:
        raise ReleaseError("Model manifest exceeds the 16 MiB release limit")
    if _sha256_bytes(payload) != expected:
        raise ReleaseError("Downloaded model manifest digest does not match approval")
    _write_bytes_atomic(output_path, payload)
    return validate_model_manifest(
        output_path,
        repo_id=repo_id,
        revision=commit,
        expected_sha256=expected,
    )


def _artifact_records(bundle_root: Path, excluded: set[Path]) -> list[dict[str, Any]]:
    root = bundle_root.resolve()
    records = []
    for path in sorted(root.rglob("*")):
        if path in excluded or path.is_dir():
            continue
        if path.is_symlink() or not path.is_file():
            raise ReleaseError(f"Release bundle contains an unsafe member: {path}")
        relative = path.relative_to(root).as_posix()
        records.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return records


def _aggregate_digest(records: Sequence[Mapping[str, Any]]) -> str:
    core = [
        {"path": item["path"], "sha256": item["sha256"], "size_bytes": item["size_bytes"]}
        for item in records
    ]
    return _sha256_bytes(_canonical_json_bytes(core))


def _validate_bundle_layout(records: Sequence[Mapping[str, Any]]) -> None:
    paths = {str(record["path"]) for record in records}
    wheels = [path for path in paths if path.startswith("distributions/") and path.endswith(".whl")]
    sdists = [path for path in paths if path.startswith("distributions/") and path.endswith(".tar.gz")]
    if len(wheels) != 1 or len(sdists) != 1:
        raise ReleaseError("Release bundle needs exactly one wheel and one source distribution")
    required = {
        "images/facetorch-cpu.tar.zst",
        "images/facetorch-gpu.tar.zst",
        "evidence/model-manifest.json",
        "evidence/release-inputs.json",
    }
    missing = sorted(required - paths)
    if missing:
        raise ReleaseError(f"Release bundle is missing required files: {missing}")
    sboms = [path for path in paths if path.startswith("sboms/") and path.endswith(".json")]
    if len(sboms) < 3:
        raise ReleaseError("Release bundle needs distribution, CPU-image, and GPU-image SBOMs")


def prepare_release_plan(
    *,
    repo_root: Path,
    bundle_root: Path,
    source_sha: str,
    tag: str,
    model_manifest_repo: str,
    model_manifest_revision: str,
    model_manifest_sha256: str,
    cpu_image_digest: str,
    gpu_image_digest: str,
    output_path: Path,
    allow_missing_tag: bool = False,
) -> dict[str, Any]:
    """Create the one digest-bound plan consumed by every release channel."""

    identity = validate_candidate_identity(
        repo_root,
        source_sha=source_sha,
        tag=tag,
        allow_missing_tag=allow_missing_tag,
    )
    root = bundle_root.resolve()
    if not root.is_dir():
        raise ReleaseError(f"Release bundle root is unavailable: {root}")
    output = output_path.resolve()
    excluded = {output, (root / "SHA256SUMS").resolve(), (root / "publication-receipt.json").resolve()}
    records = _artifact_records(root, excluded)
    _validate_bundle_layout(records)
    manifest_path = _safe_relative_file(root, "evidence/model-manifest.json")
    manifest = validate_model_manifest(
        manifest_path,
        repo_id=model_manifest_repo,
        revision=model_manifest_revision,
        expected_sha256=model_manifest_sha256,
    )
    validate_packaged_model_governance(
        repo_root,
        remote_manifest_path=manifest_path,
        remote_revision=model_manifest_revision,
    )
    distribution_records = [
        record for record in records if str(record["path"]).startswith("distributions/")
    ]
    cpu_digest = _require_image_digest(cpu_image_digest, "CPU image config digest")
    gpu_digest = _require_image_digest(gpu_image_digest, "GPU image config digest")
    local_gpu_evidence = validate_local_release_evidence(
        repo_root,
        root / "evidence" / "local-gpu",
        source_sha=identity["source_sha"],
        cpu_image_digest=cpu_digest,
        gpu_image_digest=gpu_digest,
    )
    core = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "project": PROJECT_NAME,
        "version": identity["project_version"],
        "tag": identity["tag"],
        "docker_tag": identity["docker_tag"],
        "release_kind": identity["release_kind"],
        "is_prerelease": identity["is_prerelease"],
        "source_sha": identity["source_sha"],
        "tag_exists": identity["tag_exists"],
        "model_manifest": manifest,
        "local_gpu_evidence": local_gpu_evidence,
        "artifacts": records,
        "channel_subjects": {
            "model-manifest": manifest["sha256"],
            "pypi": _aggregate_digest(distribution_records),
            "docker-cpu": cpu_digest,
            "docker-gpu": gpu_digest,
        },
    }
    plan_id = _sha256_bytes(_canonical_json_bytes(core))
    plan = {
        **core,
        "plan_id": plan_id,
        "channel_subjects": {
            **core["channel_subjects"],
            "github-release": plan_id,
        },
    }
    _write_json_atomic(output_path, plan)
    return plan


def verify_release_plan(plan_path: Path, bundle_root: Path) -> dict[str, Any]:
    """Verify plan identity and every local release byte."""

    plan = _read_json(plan_path)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ReleaseError("Unsupported release plan schema")
    if plan.get("project") != PROJECT_NAME:
        raise ReleaseError("Release plan targets the wrong project")
    identity = parse_release_tag(str(plan.get("tag", "")))
    for field in ("project_version", "docker_tag", "release_kind", "is_prerelease"):
        plan_field = "version" if field == "project_version" else field
        if plan.get(plan_field) != identity[field]:
            raise ReleaseError(f"Release plan has inconsistent {plan_field}")
    _require_sha(plan.get("source_sha"), "Release plan source SHA")

    artifacts = plan.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ReleaseError("Release plan has no artifacts")
    seen = set()
    observed_records = []
    for record in artifacts:
        if not isinstance(record, dict):
            raise ReleaseError("Release plan has an invalid artifact record")
        relative = str(record.get("path", ""))
        if relative in seen:
            raise ReleaseError(f"Release plan repeats artifact {relative}")
        seen.add(relative)
        path = _safe_relative_file(bundle_root.resolve(), relative)
        observed = {
            "path": relative,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        if observed != record:
            raise ReleaseError(f"Release artifact changed after planning: {relative}")
        observed_records.append(observed)
    _validate_bundle_layout(observed_records)
    root = bundle_root.resolve()
    excluded = {
        plan_path.resolve(),
        (root / "SHA256SUMS").resolve(),
        (root / "publication-receipt.json").resolve(),
    }
    if _artifact_records(root, excluded) != observed_records:
        raise ReleaseError("Release bundle file set changed after planning")

    subjects = plan.get("channel_subjects")
    if not isinstance(subjects, dict) or set(subjects) != set(IMMUTABLE_CHANNELS):
        raise ReleaseError("Release plan channel set is incomplete")
    _require_sha256(subjects["model-manifest"], "Model manifest subject")
    _require_sha256(subjects["pypi"], "PyPI subject")
    _require_image_digest(subjects["docker-cpu"], "CPU image subject")
    _require_image_digest(subjects["docker-gpu"], "GPU image subject")

    core = {key: value for key, value in plan.items() if key != "plan_id"}
    core["channel_subjects"] = {
        key: value for key, value in subjects.items() if key != "github-release"
    }
    expected_plan_id = _sha256_bytes(_canonical_json_bytes(core))
    if plan.get("plan_id") != expected_plan_id:
        raise ReleaseError("Release plan ID does not match its contents")
    if subjects["github-release"] != expected_plan_id:
        raise ReleaseError("GitHub release subject must equal the release plan ID")

    distribution_records = [
        record
        for record in observed_records
        if str(record["path"]).startswith("distributions/")
    ]
    if subjects["pypi"] != _aggregate_digest(distribution_records):
        raise ReleaseError("PyPI subject does not match distribution bytes")
    manifest = plan.get("model_manifest")
    if not isinstance(manifest, dict) or manifest.get("sha256") != subjects["model-manifest"]:
        raise ReleaseError("Model manifest subject does not match the release plan")
    manifest_path = _safe_relative_file(bundle_root.resolve(), "evidence/model-manifest.json")
    if sha256_file(manifest_path) != subjects["model-manifest"]:
        raise ReleaseError("Model manifest bytes changed after planning")
    local_gpu = plan.get("local_gpu_evidence")
    if (
        not isinstance(local_gpu, dict)
        or local_gpu.get("status") != "verified"
        or local_gpu.get("source_sha") != plan["source_sha"]
        or local_gpu.get("cpu_image_digest") != subjects["docker-cpu"]
        or local_gpu.get("gpu_image_digest") != subjects["docker-gpu"]
    ):
        raise ReleaseError("Release plan has inconsistent local GPU evidence")
    return plan


def write_checksums(bundle_root: Path, output_path: Path) -> None:
    """Write sorted SHA256SUMS without including the checksum file itself."""

    root = bundle_root.resolve()
    output = output_path.resolve()
    lines = []
    for record in _artifact_records(root, {output}):
        lines.append(f"{record['sha256']}  {record['path']}\n")
    _write_bytes_atomic(output_path, "".join(lines).encode("utf-8"))


def verify_checksums(bundle_root: Path, checksums_path: Path) -> None:
    """Verify a strict, duplicate-free SHA256SUMS file."""

    seen = set()
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise ReleaseError("Malformed SHA256SUMS entry")
        expected, relative = match.groups()
        if relative in seen:
            raise ReleaseError(f"Duplicate checksum entry: {relative}")
        seen.add(relative)
        if sha256_file(_safe_relative_file(bundle_root.resolve(), relative)) != expected:
            raise ReleaseError(f"Checksum mismatch: {relative}")
    observed = {
        str(record["path"])
        for record in _artifact_records(bundle_root.resolve(), {checksums_path.resolve()})
    }
    if seen != observed:
        raise ReleaseError("SHA256SUMS does not cover the exact release bundle")


def _new_receipt(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "plan_id": plan["plan_id"],
        "status": "incomplete",
        "channels": {},
    }


def _load_receipt(path: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return _new_receipt(plan)
    receipt = _read_json(path)
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise ReleaseError("Unsupported publication receipt schema")
    if receipt.get("plan_id") != plan.get("plan_id"):
        raise ReleaseError("Publication receipt belongs to a different release plan")
    if not isinstance(receipt.get("channels"), dict):
        raise ReleaseError("Publication receipt channels are invalid")
    return receipt


def record_channel(
    plan: Mapping[str, Any],
    receipt_path: Path,
    channel: str,
    observed_digest: str,
    *,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically record one identical immutable channel, rejecting drift."""

    if channel not in IMMUTABLE_CHANNELS:
        raise ReleaseError(f"Unknown coordinated release channel: {channel}")
    expected = str(plan["channel_subjects"][channel])
    observed = str(observed_digest).strip().lower()
    if observed != expected:
        raise ReleaseError(
            f"{channel} digest differs: expected {expected}, observed {observed}"
        )
    receipt = _load_receipt(receipt_path, plan)
    record = {"subject_digest": expected, "status": "verified"}
    if details:
        record["details"] = dict(details)
    existing = receipt["channels"].get(channel)
    if existing is not None and existing != record:
        raise ReleaseError(f"Existing {channel} receipt differs from this release")
    receipt["channels"][channel] = record
    receipt["status"] = (
        "complete"
        if set(receipt["channels"]) == set(IMMUTABLE_CHANNELS)
        else "incomplete"
    )
    _write_json_atomic(receipt_path, receipt)
    return receipt


def verify_publication_receipt(
    plan: Mapping[str, Any], receipt_path: Path
) -> dict[str, Any]:
    receipt = _load_receipt(receipt_path, plan)
    unexpected = set(receipt["channels"]) - set(IMMUTABLE_CHANNELS)
    if unexpected:
        raise ReleaseError(f"Publication receipt has unexpected channels: {unexpected}")
    for channel, record in receipt["channels"].items():
        if (
            not isinstance(record, dict)
            or record.get("status") != "verified"
            or record.get("subject_digest") != plan["channel_subjects"][channel]
        ):
            raise ReleaseError(f"Publication receipt is invalid for {channel}")
    expected_status = (
        "complete"
        if set(receipt["channels"]) == set(IMMUTABLE_CHANNELS)
        else "incomplete"
    )
    if receipt.get("status") != expected_status:
        raise ReleaseError("Publication receipt completion status is inconsistent")
    return receipt


def assert_stable_alias_promotion(
    plan: Mapping[str, Any],
    receipt_path: Path,
    *,
    current_latest_tag: str | None = None,
) -> dict[str, Any]:
    """Require a monotonic stable release and every immutable channel."""

    if plan.get("is_prerelease") or plan.get("release_kind") != "stable":
        raise ReleaseError("Release candidates must never move stable aliases")
    target = parse_release_tag(str(plan.get("tag", "")))
    if target["is_prerelease"] or target["release_kind"] != "stable":
        raise ReleaseError("Stable alias target must use a stable release tag")
    if target["project_version"] != plan.get("version"):
        raise ReleaseError("Stable alias target tag does not match the release plan")
    if current_latest_tag:
        current = parse_release_tag(current_latest_tag)
        if current["is_prerelease"] or current["release_kind"] != "stable":
            raise ReleaseError("Current latest release must use a stable release tag")
        target_version = tuple(int(part) for part in target["project_version"].split("."))
        current_version = tuple(
            int(part) for part in current["project_version"].split(".")
        )
        if target_version < current_version:
            raise ReleaseError(
                f"Stable alias promotion would move latest backward from "
                f"{current['tag']} to {target['tag']}"
            )
    receipt = verify_publication_receipt(plan, receipt_path)
    missing = sorted(set(IMMUTABLE_CHANNELS) - set(receipt["channels"]))
    if missing:
        raise ReleaseError(f"Stable alias promotion is missing channels: {missing}")
    return receipt


def run_publication_transaction(
    plan: Mapping[str, Any],
    receipt_path: Path,
    backend: PublicationBackend,
) -> dict[str, Any]:
    """Reconcile and resume immutable channels; useful for injected-failure tests."""

    for channel in PUBLICATION_ORDER:
        expected = str(plan["channel_subjects"][channel])
        observed = backend.observe(channel)
        if observed is None:
            observed = backend.publish(channel, expected)
        record_channel(plan, receipt_path, channel, observed)
    return verify_publication_receipt(plan, receipt_path)


def pypi_distribution_state(
    distributions: Sequence[Path], remote_metadata: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Compare local distribution bytes with a PyPI version response."""

    paths = list(distributions)
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ReleaseError("Local PyPI distributions must be regular files")
    local = {path.name: sha256_file(path) for path in paths}
    wheels = [name for name in local if name.endswith(".whl")]
    sdists = [name for name in local if name.endswith(".tar.gz")]
    if len(local) != len(paths) or len(wheels) != 1 or len(sdists) != 1:
        raise ReleaseError("Local PyPI publication needs exactly one wheel and one sdist")
    remote = {}
    if remote_metadata is not None:
        urls = remote_metadata.get("urls")
        if not isinstance(urls, list):
            raise ReleaseError("PyPI version response has no distribution list")
        for item in urls:
            if not isinstance(item, dict):
                raise ReleaseError("PyPI version response contains an invalid file")
            filename = str(item.get("filename", ""))
            digest = str(item.get("digests", {}).get("sha256", "")).lower()
            _require_sha256(digest, f"PyPI digest for {filename}")
            if filename in remote:
                raise ReleaseError(f"PyPI repeats distribution filename {filename}")
            remote[filename] = digest
    unexpected = sorted(set(remote) - set(local))
    if unexpected:
        raise ReleaseError(f"PyPI contains unexpected files for this version: {unexpected}")
    mismatched = sorted(
        filename for filename in set(local) & set(remote) if local[filename] != remote[filename]
    )
    if mismatched:
        raise ReleaseError(f"PyPI already contains different bytes: {mismatched}")
    missing = sorted(set(local) - set(remote))
    return {
        "status": "identical" if not missing else "publish-required",
        "local": local,
        "remote": remote,
        "missing": missing,
    }


def inspect_pypi(
    *,
    project: str,
    version: str,
    distributions: Sequence[Path],
    opener: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    url = f"https://pypi.org/pypi/{quote(project, safe='')}/{quote(version, safe='')}/json"
    try:
        with opener(url, timeout=30) as response:
            payload = response.read(8 * 1024 * 1024 + 1)
    except HTTPError as exc:
        if exc.code == 404:
            return pypi_distribution_state(distributions, None)
        raise ReleaseError(f"Cannot inspect PyPI version: HTTP {exc.code}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise ReleaseError(f"Cannot inspect PyPI version: {exc}") from exc
    if len(payload) > 8 * 1024 * 1024:
        raise ReleaseError("PyPI version response exceeds the 8 MiB limit")
    try:
        metadata = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ReleaseError("PyPI version response is not JSON") from exc
    if not isinstance(metadata, dict):
        raise ReleaseError("PyPI version response must contain an object")
    return pypi_distribution_state(distributions, metadata)


def _container_config_digests(value: Any) -> set[str]:
    entries = value if isinstance(value, list) else [value]
    digests = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        descriptor = entry.get("Descriptor", {})
        platform = descriptor.get("platform", {}) if isinstance(descriptor, dict) else {}
        if platform and (
            platform.get("os") != "linux" or platform.get("architecture") != "amd64"
        ):
            continue
        for key in ("SchemaV2Manifest", "OCIManifest"):
            manifest = entry.get(key)
            if isinstance(manifest, dict):
                digest = manifest.get("config", {}).get("digest")
                if isinstance(digest, str):
                    digests.add(digest.lower())
        digest = entry.get("config", {}).get("digest")
        if isinstance(digest, str):
            digests.add(digest.lower())
    return digests


def docker_distribution_state(
    manifest: Any | None, expected_config_digest: str
) -> dict[str, Any]:
    expected = _require_image_digest(expected_config_digest, "Expected image config")
    if manifest is None:
        return {"status": "publish-required", "config_digest": expected}
    observed = _container_config_digests(manifest)
    if observed != {expected}:
        raise ReleaseError(
            f"Registry image differs: expected config {expected}, observed {sorted(observed)}"
        )
    return {"status": "identical", "config_digest": expected}


def inspect_docker(reference: str, expected_config_digest: str) -> dict[str, Any]:
    if not reference or any(character.isspace() for character in reference):
        raise ReleaseError("Docker reference is empty or contains whitespace")
    result = subprocess.run(
        ["docker", "manifest", "inspect", "--verbose", reference],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        error = (result.stderr + result.stdout).lower()
        if any(value in error for value in ("no such manifest", "manifest unknown", "not found")):
            return docker_distribution_state(None, expected_config_digest)
        raise ReleaseError(f"Cannot inspect Docker reference {reference}: {error.strip()}")
    try:
        manifest = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ReleaseError("Docker manifest inspection returned invalid JSON") from exc
    return docker_distribution_state(manifest, expected_config_digest)


def _copy_missing_distributions(
    state: Mapping[str, Any], distributions: Sequence[Path], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = set(state["missing"])
    for path in distributions:
        if path.name in expected:
            shutil.copyfile(path, output_dir / path.name)


def _write_output(path: Path, value: Mapping[str, Any]) -> None:
    _write_json_atomic(path, value)


def _load_plan_without_bundle(path: Path) -> dict[str, Any]:
    plan = _read_json(path)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION or not plan.get("plan_id"):
        raise ReleaseError("Invalid release plan")
    return plan


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    tag = subparsers.add_parser("tag-for-version")
    tag.add_argument("version")

    fetch = subparsers.add_parser("fetch-model-manifest")
    fetch.add_argument("--repo", required=True)
    fetch.add_argument("--revision", required=True)
    fetch.add_argument("--filename", required=True)
    fetch.add_argument("--sha256", required=True)
    fetch.add_argument("--output", required=True)
    fetch.add_argument("--report", required=True)

    identity = subparsers.add_parser("validate-identity")
    identity.add_argument("--repo-root", default=".")
    identity.add_argument("--source-sha", required=True)
    identity.add_argument("--tag", required=True)
    identity.add_argument("--allow-missing-tag", action="store_true")
    identity.add_argument("--output", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--repo-root", default=".")
    prepare.add_argument("--bundle-root", required=True)
    prepare.add_argument("--source-sha", required=True)
    prepare.add_argument("--tag", required=True)
    prepare.add_argument("--model-manifest-repo", required=True)
    prepare.add_argument("--model-manifest-revision", required=True)
    prepare.add_argument("--model-manifest-sha256", required=True)
    prepare.add_argument("--cpu-image-digest", required=True)
    prepare.add_argument("--gpu-image-digest", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--allow-missing-tag", action="store_true")

    verify = subparsers.add_parser("verify")
    verify.add_argument("--plan", required=True)
    verify.add_argument("--bundle-root", required=True)
    verify.add_argument("--checksums")

    checksums = subparsers.add_parser("checksums")
    checksums.add_argument("--bundle-root", required=True)
    checksums.add_argument("--output", required=True)

    pypi = subparsers.add_parser("pypi-state")
    pypi.add_argument("--plan", required=True)
    pypi.add_argument("--dist-dir", required=True)
    pypi.add_argument("--missing-dir", required=True)
    pypi.add_argument("--output", required=True)

    docker = subparsers.add_parser("docker-state")
    docker.add_argument("--plan", required=True)
    docker.add_argument("--channel", choices=("docker-cpu", "docker-gpu"), required=True)
    docker.add_argument("--reference", required=True)
    docker.add_argument("--output", required=True)

    record = subparsers.add_parser("record-channel")
    record.add_argument("--plan", required=True)
    record.add_argument("--receipt", required=True)
    record.add_argument("--channel", choices=IMMUTABLE_CHANNELS, required=True)
    record.add_argument("--digest", required=True)
    record.add_argument("--details")

    promote = subparsers.add_parser("assert-stable-alias")
    promote.add_argument("--plan", required=True)
    promote.add_argument("--receipt", required=True)
    promote.add_argument("--current-latest-tag", default="")

    verify_receipt = subparsers.add_parser("verify-receipt")
    verify_receipt.add_argument("--plan", required=True)
    verify_receipt.add_argument("--receipt", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "tag-for-version":
        print(tag_for_project_version(args.version))
        return 0
    if args.command == "fetch-model-manifest":
        report = fetch_model_manifest(
            repo_id=args.repo,
            revision=args.revision,
            filename=args.filename,
            expected_sha256=args.sha256,
            output_path=Path(args.output),
        )
        _write_output(Path(args.report), report)
        return 0
    if args.command == "validate-identity":
        identity = validate_candidate_identity(
            Path(args.repo_root),
            source_sha=args.source_sha,
            tag=args.tag,
            allow_missing_tag=args.allow_missing_tag,
        )
        _write_output(Path(args.output), identity)
        return 0
    if args.command == "prepare":
        prepare_release_plan(
            repo_root=Path(args.repo_root),
            bundle_root=Path(args.bundle_root),
            source_sha=args.source_sha,
            tag=args.tag,
            model_manifest_repo=args.model_manifest_repo,
            model_manifest_revision=args.model_manifest_revision,
            model_manifest_sha256=args.model_manifest_sha256,
            cpu_image_digest=args.cpu_image_digest,
            gpu_image_digest=args.gpu_image_digest,
            output_path=Path(args.output),
            allow_missing_tag=args.allow_missing_tag,
        )
        return 0
    if args.command == "verify":
        verify_release_plan(Path(args.plan), Path(args.bundle_root))
        if args.checksums:
            verify_checksums(Path(args.bundle_root), Path(args.checksums))
        return 0
    if args.command == "checksums":
        write_checksums(Path(args.bundle_root), Path(args.output))
        return 0
    if args.command == "pypi-state":
        plan = _load_plan_without_bundle(Path(args.plan))
        distributions = sorted(Path(args.dist_dir).glob("*"))
        state = inspect_pypi(
            project=plan["project"], version=plan["version"], distributions=distributions
        )
        _copy_missing_distributions(state, distributions, Path(args.missing_dir))
        _write_output(Path(args.output), state)
        return 0
    if args.command == "docker-state":
        plan = _load_plan_without_bundle(Path(args.plan))
        state = inspect_docker(args.reference, plan["channel_subjects"][args.channel])
        _write_output(Path(args.output), state)
        return 0
    if args.command == "record-channel":
        plan = _load_plan_without_bundle(Path(args.plan))
        details = _read_json(Path(args.details)) if args.details else None
        record_channel(
            plan,
            Path(args.receipt),
            args.channel,
            args.digest,
            details=details,
        )
        return 0
    if args.command == "verify-receipt":
        plan = _load_plan_without_bundle(Path(args.plan))
        verify_publication_receipt(plan, Path(args.receipt))
        return 0
    plan = _load_plan_without_bundle(Path(args.plan))
    assert_stable_alias_promotion(
        plan,
        Path(args.receipt),
        current_latest_tag=args.current_latest_tag or None,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReleaseError as exc:
        print(f"release transaction rejected: {exc}", file=sys.stderr)
        raise SystemExit(2)
