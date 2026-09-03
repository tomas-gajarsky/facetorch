#!/usr/bin/env python3
"""Execute the public notebook against an installed, fully staged candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import nbformat
from nbclient import NotebookClient


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _within(root: Path, path: Path, label: str) -> Path:
    root = root.resolve()
    path = path.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"{label} must remain inside {root}") from exc
    return path


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--pinned-artifacts-root",
        type=Path,
        help="Exact digest-bound artifact cohort staged beneath --staging-root",
    )
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-notebook", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=1800)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.timeout <= 0:
        raise RuntimeError("--timeout must be positive")
    repo_root = args.repo_root.resolve()
    staging_root = args.staging_root.resolve()
    summary = _within(staging_root, args.summary, "Summary")
    wheel = _within(staging_root, args.wheel, "Candidate wheel")
    output_notebook = _within(
        staging_root, args.output_notebook, "Executed notebook"
    )
    report_path = _within(staging_root, args.report, "Notebook report")
    if not summary.is_file() or not wheel.is_file():
        raise RuntimeError("The staged summary and candidate wheel must exist")
    if wheel.suffix != ".whl":
        raise RuntimeError("--wheel must identify the exact candidate wheel")
    pinned_artifacts_root = (
        _within(staging_root, args.pinned_artifacts_root, "Pinned artifacts")
        if args.pinned_artifacts_root is not None
        else None
    )
    if pinned_artifacts_root is not None and not pinned_artifacts_root.is_dir():
        raise RuntimeError("--pinned-artifacts-root must identify a directory")

    notebook_path = repo_root / "notebooks" / "facetorch_notebook_demo.ipynb"
    image_path = repo_root / "data" / "input" / "test.jpg"
    notebook = nbformat.read(notebook_path, as_version=4)
    install_index = next(
        (
            index
            for index, cell in enumerate(notebook.cells)
            if cell.cell_type == "code" and "FACETORCH_NOTEBOOK_WHEEL" in cell.source
        ),
        None,
    )
    if install_index is None:
        raise RuntimeError("Notebook has no candidate-wheel installation cell")

    scripts_root = repo_root / "scripts"
    cache_root = staging_root / f"notebook-cache-{args.device}"
    pinned_artifacts_expression = (
        "None"
        if pinned_artifacts_root is None
        else f"_FacetorchPath({json.dumps(str(pinned_artifacts_root))})"
    )
    injected_source = f"""
import sys as _facetorch_sys
import hashlib as _facetorch_hashlib
import zipfile as _facetorch_zipfile
from importlib.metadata import distribution as _facetorch_distribution
from pathlib import Path as _FacetorchPath

_facetorch_scripts = _FacetorchPath({json.dumps(str(scripts_root))})
_facetorch_sys.path.insert(0, str(_facetorch_scripts))
from smoke_staged_default_analyzer import (
    _candidate_manifest as _facetorch_candidate_manifest,
    _prepare_cache as _facetorch_prepare_cache,
    _read_json as _facetorch_read_json,
)
import facetorch as _facetorch_package
import facetorch.downloader as _facetorch_downloader
import torch as _facetorch_torch

_facetorch_device = {json.dumps(args.device)}
assert (_facetorch_device == "cuda") == _facetorch_torch.cuda.is_available()
_facetorch_wheel = _FacetorchPath({json.dumps(str(wheel))})
_facetorch_installed = _facetorch_distribution("facetorch")
with _facetorch_zipfile.ZipFile(_facetorch_wheel) as _facetorch_archive:
    for _facetorch_member in _facetorch_archive.infolist():
        if _facetorch_member.filename.endswith(".dist-info/RECORD"):
            continue
        _facetorch_installed_path = _facetorch_installed.locate_file(
            _facetorch_member.filename
        )
        assert _facetorch_hashlib.sha256(
            _facetorch_installed_path.read_bytes()
        ).digest() == _facetorch_hashlib.sha256(
            _facetorch_archive.read(_facetorch_member)
        ).digest()
_facetorch_summary_path = _FacetorchPath({json.dumps(str(summary))})
_facetorch_staging = _FacetorchPath({json.dumps(str(staging_root))})
_facetorch_repo = _FacetorchPath({json.dumps(str(repo_root))})
_facetorch_summary = _facetorch_read_json(_facetorch_summary_path)
_facetorch_manifest, _facetorch_paths = _facetorch_candidate_manifest(
    _facetorch_repo,
    _facetorch_staging,
    _facetorch_summary,
    pinned_artifacts_root={pinned_artifacts_expression},
)
_facetorch_profile = "gpu" if _facetorch_device == "cuda" else "cpu"
_facetorch_config = _facetorch_package.load_config(_facetorch_profile, offline=True)
_facetorch_prepare_cache(
    _facetorch_staging,
    _facetorch_config,
    _facetorch_manifest,
    _facetorch_paths,
    _facetorch_device,
)
_facetorch_downloader.get_model_manifest = lambda: _facetorch_manifest
_facetorch_sys.path.remove(str(_facetorch_scripts))
""".strip()
    notebook.cells.insert(
        install_index + 1,
        nbformat.v4.new_code_cell(
            injected_source,
            metadata={"facetorch_release_validation": "candidate_manifest"},
        ),
    )
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            """
assert result.faces
assert all(set(face.preds) == set(analyzer.configured_predictors) for face in result.faces)
_facetorch_components = [analyzer.detector] + [
    analyzer.predictors[name] for name in analyzer.configured_predictors
]
assert all(
    component.downloader.active_descriptor is not None
    and component.downloader.active_descriptor.format == "pt2"
    for component in _facetorch_components
)
""".strip(),
            metadata={"facetorch_release_validation": "postconditions"},
        )
    )

    os.environ.update(
        {
            "FACETORCH_NOTEBOOK_WHEEL": str(wheel),
            "FACETORCH_NOTEBOOK_IMAGE": str(image_path),
            "FACETORCH_CACHE_DIR": str(cache_root),
            "FACETORCH_OFFLINE": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INDEX": "1",
        }
    )
    output_notebook.parent.mkdir(parents=True, exist_ok=True)
    client = NotebookClient(
        notebook,
        timeout=args.timeout,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(staging_root)}},
    )
    client.execute()
    nbformat.write(notebook, output_notebook)

    report = {
        "schema_version": 1,
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": args.device,
        "pinned_manifest_artifacts": pinned_artifacts_root is not None,
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "runner_python": sys.version.split()[0],
        "source_notebook_sha256": _sha256(notebook_path),
        "executed_notebook_sha256": _sha256(output_notebook),
        "summary_sha256": _sha256(summary),
        "candidate_wheel": {"filename": wheel.name, "sha256": _sha256(wheel)},
        "image_sha256": _sha256(image_path),
        "pip_index_disabled": True,
        "artifact_offline_mode": True,
        "local_image_override": True,
    }
    _write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
