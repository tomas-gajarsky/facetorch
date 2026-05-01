#!/usr/bin/env python3
"""Validate dependency baseline alignment between pyproject and conda env files."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


REQ_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)\s*(.*)$")
MIN_RE = re.compile(r">=\s*([A-Za-z0-9_.+-]+)")


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _parse_requirement(raw: str) -> Tuple[str, str]:
    match = REQ_RE.match(raw.strip())
    if not match:
        raise ValueError(f"Cannot parse dependency specification: {raw!r}")
    name = _normalize_name(match.group(1))
    spec = match.group(2).strip()
    return name, spec


def _extract_min_version(spec: str) -> str | None:
    match = MIN_RE.search(spec)
    if not match:
        return None
    return match.group(1)


def _normalize_spec(spec: str | None) -> str | None:
    if spec is None:
        return None
    return re.sub(r"\s+", "", spec)


def _load_pyproject_deps(
    pyproject_path: Path,
) -> Tuple[str, str, Dict[str, str | None]]:
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = data["project"]
    requires_python = project["requires-python"]
    py_deps = project["dependencies"]

    dep_map: Dict[str, str | None] = {}
    for dep in py_deps:
        name, spec = _parse_requirement(dep)
        dep_map[name] = _extract_min_version(spec)

    py_spec = requires_python.strip()
    py_min = _extract_min_version(py_spec)
    if py_min is None:
        raise ValueError(f"requires-python must include a >= lower bound, got {py_spec!r}")

    return py_spec, py_min, dep_map


def _load_conda_deps(
    env_path: Path,
) -> Tuple[Dict[str, str | None], Dict[str, str]]:
    deps: Dict[str, str | None] = {}
    specs: Dict[str, str] = {}
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

        raw_dep = line[len("  - ") :].strip()
        if not raw_dep:
            continue
        name, spec = _parse_requirement(raw_dep)
        deps[name] = _extract_min_version(spec)
        specs[name] = spec

    if not deps:
        raise ValueError(f"No dependencies parsed from {env_path}")
    return deps, specs


def _verify_env(
    env_name: str,
    env_deps: Dict[str, str | None],
    env_specs: Dict[str, str],
    py_spec: str,
    py_min: str,
    py_deps: Dict[str, str | None],
    torch_name: str,
) -> Iterable[str]:
    errors = []

    env_python_min = env_deps.get("python")
    if env_python_min != py_min:
        errors.append(
            f"{env_name}: python minimum mismatch (pyproject>={py_min}, {env_name}>={env_python_min})"
        )

    env_python_spec = _normalize_spec(env_specs.get("python"))
    py_python_spec = _normalize_spec(py_spec)
    if env_python_spec != py_python_spec:
        errors.append(
            f"{env_name}: python spec mismatch "
            f"(pyproject {py_python_spec}, {env_name} {env_python_spec})"
        )

    for dep_name, py_dep_min in py_deps.items():
        env_key = torch_name if dep_name == "torch" else dep_name
        if env_key not in env_deps:
            errors.append(f"{env_name}: missing dependency {env_key!r}")
            continue
        env_min = env_deps[env_key]
        if env_min != py_dep_min:
            errors.append(
                f"{env_name}: minimum mismatch for {env_key} "
                f"(pyproject {dep_name}>={py_dep_min}, {env_name} {env_key}>={env_min})"
            )

    return errors


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    pyproject_path = repo_root / "pyproject.toml"
    cpu_env_path = repo_root / "environment.yml"
    gpu_env_path = repo_root / "gpu.environment.yml"

    py_spec, py_min, py_deps = _load_pyproject_deps(pyproject_path)
    cpu_deps, cpu_specs = _load_conda_deps(cpu_env_path)
    gpu_deps, gpu_specs = _load_conda_deps(gpu_env_path)

    errors = []
    errors.extend(
        _verify_env(
            env_name="environment.yml",
            env_deps=cpu_deps,
            env_specs=cpu_specs,
            py_spec=py_spec,
            py_min=py_min,
            py_deps=py_deps,
            torch_name="pytorch-cpu",
        )
    )
    errors.extend(
        _verify_env(
            env_name="gpu.environment.yml",
            env_deps=gpu_deps,
            env_specs=gpu_specs,
            py_spec=py_spec,
            py_min=py_min,
            py_deps=py_deps,
            torch_name="pytorch-gpu",
        )
    )

    if errors:
        print("Dependency alignment check failed:")
        for err in errors:
            print(f" - {err}")
        return 1

    print("Dependency alignment check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
