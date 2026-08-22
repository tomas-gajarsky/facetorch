#!/usr/bin/env python3
"""Validate public requirements, exact release profiles, and lock metadata."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib  # type: ignore


REQ_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)\s*(.*)$")
PROFILE_ROOT = Path("environments")
PROFILE_SPECS = {
    "torch-2.6-cpu": {
        "torch": "2.6.0",
        "torchvision": "0.21.0",
        "index": "https://download.pytorch.org/whl/cpu",
    },
    "torch-2.11-cpu": {
        "torch": "2.11.0",
        "torchvision": "0.26.0",
        "index": "https://download.pytorch.org/whl/cpu",
    },
    "torch-2.6-cu124": {
        "torch": "2.6.0",
        "torchvision": "0.21.0",
        "index": "https://download.pytorch.org/whl/cu124",
    },
    "torch-2.11-cu130": {
        "torch": "2.11.0",
        "torchvision": "0.26.0",
        "index": "https://download.pytorch.org/whl/cu130",
    },
}


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _parse_python_requirement(raw: str) -> Tuple[str, str]:
    requirement = Requirement(raw)
    return _normalize_name(requirement.name), str(requirement.specifier)


def _parse_conda_requirement(raw: str) -> Tuple[str, str]:
    match = REQ_RE.match(raw.strip())
    if not match:
        raise ValueError(f"Cannot parse dependency specification: {raw!r}")
    return _normalize_name(match.group(1)), match.group(2).strip()


def _normalize_spec(spec: str | None) -> str | None:
    if spec is None:
        return None
    return re.sub(r"\s+", "", spec)


def _load_toml(path: Path) -> Dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _dependency_map(raw_dependencies: Iterable[str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for raw in raw_dependencies:
        name, spec = _parse_python_requirement(raw)
        result[name] = spec
    return result


def _load_conda_deps(env_path: Path) -> Dict[str, str]:
    dependencies: Dict[str, str] = {}
    in_dependencies = False
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.strip() == "dependencies:":
            in_dependencies = True
            continue
        if not in_dependencies:
            continue
        if not line.startswith("  - "):
            if line.strip():
                in_dependencies = False
            continue
        raw_dependency = line[len("  - ") :].strip()
        if raw_dependency:
            name, spec = _parse_conda_requirement(raw_dependency)
            dependencies[name] = spec
    if not dependencies:
        raise ValueError(f"No dependencies parsed from {env_path}")
    return dependencies


def _conda_exact_version(spec: str) -> str | None:
    normalized = spec.strip().lstrip("=")
    if not normalized or any(char in normalized for char in "<>,!|"):
        return None
    return normalized.removesuffix(".*")


def _verify_public_matrix(
    project: Dict[str, Any], compatibility: Dict[str, Any]
) -> Iterable[str]:
    errors = []
    metadata = project["project"]
    dependencies = _dependency_map(metadata["dependencies"])
    python_spec = metadata["requires-python"]
    matrix_python = compatibility["python"]["specifier"]
    matrix_torch = compatibility["torch"]["specifier"]
    if _normalize_spec(matrix_python) != _normalize_spec(python_spec):
        errors.append(
            "compatibility.json: Python spec does not match pyproject.toml "
            f"({matrix_python} != {python_spec})"
        )
    if str(SpecifierSet(matrix_torch)) != str(SpecifierSet(dependencies["torch"])):
        errors.append(
            "compatibility.json: Torch spec does not match pyproject.toml "
            f"({matrix_torch} != {dependencies['torch']})"
        )

    torch_specifier = SpecifierSet(matrix_torch)
    supported = compatibility["torch"]["supported_minor_lines"]
    unsupported = compatibility["torch"]["explicitly_unsupported_minor_lines"]
    cohorts = {item["torch_minor"] for item in compatibility["cohorts"]}
    if set(supported) != cohorts:
        errors.append(
            "compatibility.json: supported Torch lines and cohort records differ"
        )
    for minor in supported:
        if Version(f"{minor}.0") not in torch_specifier:
            errors.append(
                f"compatibility.json: supported Torch {minor} is excluded by metadata"
            )
    for minor in unsupported:
        if Version(f"{minor}.0") in torch_specifier:
            errors.append(
                f"compatibility.json: unsupported Torch {minor} is accepted by metadata"
            )
    return errors


def _verify_profile(
    repo_root: Path,
    profile_name: str,
    expected: Dict[str, str],
    public_project: Dict[str, Any],
) -> Iterable[str]:
    errors = []
    profile_root = repo_root / PROFILE_ROOT / profile_name
    project_path = profile_root / "pyproject.toml"
    lock_path = profile_root / "uv.lock"
    if not project_path.is_file() or not lock_path.is_file():
        return [f"{profile_name}: pyproject.toml and uv.lock are required"]

    profile = _load_toml(project_path)
    profile_metadata = profile["project"]
    public_metadata = public_project["project"]
    profile_dependencies = _dependency_map(profile_metadata["dependencies"])
    public_dependencies = _dependency_map(public_metadata["dependencies"])
    profile_test = _dependency_map(
        profile_metadata.get("optional-dependencies", {}).get("test", [])
    )
    profile_release = _dependency_map(
        profile_metadata.get("optional-dependencies", {}).get("release", [])
    )
    public_dev = _dependency_map(
        public_metadata.get("optional-dependencies", {}).get("dev", [])
    )
    public_release = _dependency_map(
        public_metadata.get("optional-dependencies", {}).get("release", [])
    )

    if _normalize_spec(profile_metadata["requires-python"]) != _normalize_spec(
        public_metadata["requires-python"]
    ):
        errors.append(f"{profile_name}: Python support differs from pyproject.toml")
    for dependency, public_spec in public_dependencies.items():
        if dependency in {"torch", "torchvision"}:
            continue
        if _normalize_spec(profile_dependencies.get(dependency)) != _normalize_spec(
            public_spec
        ):
            errors.append(
                f"{profile_name}: {dependency} does not match the public requirement"
            )
    if set(profile_dependencies) != set(public_dependencies):
        errors.append(f"{profile_name}: runtime dependency names differ from pyproject.toml")
    if profile_test != public_dev:
        errors.append(f"{profile_name}: test dependencies differ from the dev extra")
    if profile_release != public_release:
        errors.append(
            f"{profile_name}: release dependencies differ from the release extra"
        )
    for dependency in ("torch", "torchvision"):
        if profile_dependencies.get(dependency) != f"=={expected[dependency]}":
            errors.append(
                f"{profile_name}: expected {dependency}=={expected[dependency]}"
            )

    sources = profile.get("tool", {}).get("uv", {}).get("sources", {})
    indexes = profile.get("tool", {}).get("uv", {}).get("index", [])
    index_by_name = {item["name"]: item for item in indexes}
    for dependency in ("torch", "torchvision"):
        index_name = sources.get(dependency, {}).get("index")
        index = index_by_name.get(index_name, {})
        if index.get("url") != expected["index"] or index.get("explicit") is not True:
            errors.append(f"{profile_name}: {dependency} index is not exact and explicit")

    lock = _load_toml(lock_path)
    if _normalize_spec(lock.get("requires-python")) != _normalize_spec(
        public_metadata["requires-python"]
    ):
        errors.append(f"{profile_name}: uv.lock Python support is stale")
    for dependency in ("torch", "torchvision"):
        locked = [item for item in lock.get("package", []) if item["name"] == dependency]
        base_versions = {item["version"].split("+", 1)[0] for item in locked}
        registries = {item.get("source", {}).get("registry") for item in locked}
        if base_versions != {expected[dependency]}:
            errors.append(f"{profile_name}: uv.lock has stale {dependency} versions")
        if expected["index"] not in registries:
            errors.append(f"{profile_name}: uv.lock has the wrong {dependency} source")
    return errors


def _verify_conda_sources(
    repo_root: Path, public_project: Dict[str, Any]
) -> Iterable[str]:
    errors = []
    public_dependencies = _dependency_map(public_project["project"]["dependencies"])
    cpu = _load_conda_deps(repo_root / "environment.yml")
    gpu = _load_conda_deps(repo_root / "gpu.environment.yml")

    for dependency, public_spec in public_dependencies.items():
        conda_name = "pytorch-cpu" if dependency == "torch" else dependency
        if conda_name not in cpu:
            errors.append(f"environment.yml: missing dependency {conda_name!r}")
            continue
        if dependency not in {"torch", "torchvision"} and _normalize_spec(
            cpu[conda_name]
        ) != _normalize_spec(public_spec):
            errors.append(f"environment.yml: {conda_name} differs from pyproject.toml")

    expected_cpu = {"pytorch-cpu": "2.6.0", "torchvision": "0.21.0"}
    for dependency, expected in expected_cpu.items():
        if _conda_exact_version(cpu.get(dependency, "")) != expected:
            errors.append(f"environment.yml: expected {dependency}={expected}")
    if _normalize_spec(cpu.get("python")) != ">=3.12,<3.13":
        errors.append("environment.yml: production Python must be >=3.12,<3.13")

    expected_gpu = {"python", "cuda-version", "pip"}
    if set(gpu) != expected_gpu:
        errors.append(
            "gpu.environment.yml: conda layer must contain only Python, CUDA, and pip; "
            "Python packages come from environments/torch-2.6-cu124/uv.lock"
        )
    if _normalize_spec(gpu.get("python")) != ">=3.12,<3.13":
        errors.append("gpu.environment.yml: production Python must be >=3.12,<3.13")
    if _conda_exact_version(gpu.get("cuda-version", "")) != "12.4":
        errors.append("gpu.environment.yml: expected cuda-version=12.4")
    return errors


def _verify_conda_lock(
    path: Path, expected: Dict[str, str], forbidden: Iterable[str] = ()
) -> Iterable[str]:
    errors = []
    package_versions: Dict[str, set[str]] = {}
    current_name = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("- name: "):
            current_name = _normalize_name(line[len("- name: ") :].strip())
        elif current_name and line.startswith("  version: "):
            version = line[len("  version: ") :].strip().strip("'\"")
            package_versions.setdefault(current_name, set()).add(version)
            current_name = None
    for dependency, version in expected.items():
        locked_versions = package_versions.get(dependency, set())
        matches = (
            bool(locked_versions)
            and all(item.startswith(version[:-1]) for item in locked_versions)
            if version.endswith("*")
            else locked_versions == {version}
        )
        if not matches:
            errors.append(f"{path.name}: expected locked {dependency} {version}")
    for dependency in forbidden:
        if dependency in package_versions:
            errors.append(f"{path.name}: forbidden package {dependency} is locked")
    return errors


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    project = _load_toml(repo_root / "pyproject.toml")
    compatibility = json.loads(
        (repo_root / "facetorch" / "models" / "compatibility.json").read_text(
            encoding="utf-8"
        )
    )

    errors = []
    errors.extend(_verify_public_matrix(project, compatibility))
    errors.extend(_verify_conda_sources(repo_root, project))
    for profile_name, expected in PROFILE_SPECS.items():
        errors.extend(_verify_profile(repo_root, profile_name, expected, project))
    errors.extend(
        _verify_conda_lock(
            repo_root / "conda-lock.yml",
            {"python": "3.12.*", "pytorch-cpu": "2.6.0", "torchvision": "0.21.0"},
        )
    )
    errors.extend(
        _verify_conda_lock(
            repo_root / "gpu.conda-lock.yml",
            {"python": "3.12.*", "cuda-version": "12.4"},
            forbidden=("pytorch", "pytorch-cpu", "pytorch-gpu", "torchvision"),
        )
    )

    if errors:
        print("Dependency alignment check failed:")
        for error in errors:
            print(f" - {error}")
        return 1
    print("Dependency alignment check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
