#!/usr/bin/env python3
"""Stage and attest the verified 3D alignment metadata for release smokes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping

from hydra.utils import instantiate

import facetorch

ARTIFACT_ID = "align-3dmm-metadata-v1"
STAGED_RELATIVE_PATH = Path("runtime-inputs/3dmm/meta.pt")
REPORT_RELATIVE_PATH = Path("alignment-metadata-report.json")
REPORT_KEYS = {
    "schema_version",
    "status",
    "artifact_id",
    "source",
    "downloader",
    "file_id",
    "revision",
    "expected_format",
    "staged_path",
    "size_bytes",
    "sha256",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
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
        temporary.chmod(0o644)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _release_directory(root: Path, relative: Path) -> Path:
    current = root
    for part in relative.parts:
        current = current / part
        if os.path.lexists(current) and current.is_symlink():
            raise RuntimeError(f"Release metadata path contains a symlink: {current}")
        current.mkdir(exist_ok=True)
        if not current.is_dir() or current.resolve(strict=True) != current:
            raise RuntimeError(f"Release metadata directory escapes staging: {current}")
        current.chmod(0o755)
    return current


def _restore_environment(name: str, previous: str | None) -> None:
    if previous is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous


def _alignment_metadata_contract() -> dict[str, Any]:
    inputs_path = Path(__file__).resolve().parents[1] / "security/release-inputs.json"
    inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
    contract = inputs.get("alignment_metadata")
    if not isinstance(contract, dict) or set(contract) != REPORT_KEYS:
        raise RuntimeError("Alignment metadata release contract is invalid")
    return contract


def stage_alignment_metadata(staging_root: Path) -> tuple[Path, Path]:
    """Download one pinned metadata object into a bounded, traversable location."""
    root = staging_root.resolve(strict=True)
    if not root.is_dir():
        raise RuntimeError(f"Staging root is not a directory: {root}")
    metadata_root = _release_directory(root, Path("runtime-inputs"))
    metadata_parent = _release_directory(root, STAGED_RELATIVE_PATH.parent)
    expected_path = root / STAGED_RELATIVE_PATH

    variable = "FACETORCH_METADATA_DIR"
    previous = os.environ.get(variable)
    os.environ[variable] = os.fspath(metadata_root)
    try:
        config = facetorch.load_config("cpu", offline=False)
        descriptor = config.analyzer.utilizer.align.downloader_meta
        configured_path = Path(str(descriptor.path_local)).resolve(strict=False)
        if configured_path != expected_path:
            raise RuntimeError(
                "Alignment metadata downloader does not target the bounded staging path"
            )
        expected_sha256 = str(descriptor.sha256)
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise RuntimeError("Alignment metadata SHA-256 is invalid")
        expected_size = int(descriptor.size_bytes)
        if expected_size <= 0 or str(descriptor.expected_format) != "torch_data":
            raise RuntimeError("Alignment metadata descriptor is incomplete")
        downloader_target = str(descriptor._target_)
        if downloader_target != "facetorch.downloader.DownloaderGDrive":
            raise RuntimeError(
                "Alignment metadata downloader is not the pinned provider"
            )
        report = {
            "schema_version": 1,
            "status": "ok",
            "artifact_id": ARTIFACT_ID,
            "source": "gdrive",
            "downloader": downloader_target,
            "file_id": str(descriptor.file_id),
            "revision": str(descriptor.revision),
            "expected_format": str(descriptor.expected_format),
            "staged_path": STAGED_RELATIVE_PATH.as_posix(),
            "size_bytes": expected_size,
            "sha256": expected_sha256,
        }
        if report != _alignment_metadata_contract():
            raise RuntimeError(
                "Candidate alignment metadata differs from the reviewed release contract"
            )
        downloader = instantiate(descriptor)
        downloaded_path = Path(str(downloader.run()))
        if downloaded_path.is_symlink():
            raise RuntimeError("Alignment metadata downloader returned a symlink")
        downloaded = downloaded_path.resolve(strict=True)
    finally:
        _restore_environment(variable, previous)

    if (
        downloaded != expected_path
        or not downloaded.is_file()
        or downloaded.is_symlink()
    ):
        raise RuntimeError("Alignment metadata downloader returned an unbounded file")
    if (
        downloaded.stat().st_size != expected_size
        or _sha256(downloaded) != expected_sha256
    ):
        raise RuntimeError("Staged alignment metadata failed its byte binding")
    metadata_root.chmod(0o755)
    metadata_parent.chmod(0o755)
    downloaded.chmod(0o644)

    report_path = root / REPORT_RELATIVE_PATH
    _write_json_atomic(report_path, report)
    return downloaded, report_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    metadata_path, report_path = stage_alignment_metadata(args.staging_root)
    print(
        json.dumps(
            {
                "status": "ok",
                "metadata": str(metadata_path),
                "report": str(report_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
