#!/usr/bin/env python3
"""Run the installed default analyzer against one fully staged model cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch

import facetorch
from facetorch.artifacts import ArtifactManifest, verify_artifact

ALIGNMENT_METADATA_ID = "align-3dmm-metadata-v1"
ALIGNMENT_METADATA_RELATIVE_PATH = Path("runtime-inputs/3dmm/meta.pt")
ALIGNMENT_METADATA_REPORT = Path("alignment-metadata-report.json")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bounded_file(root: Path, value: Any, expected_relative: Path) -> Path:
    root = root.resolve(strict=True)
    expected_relative = Path(expected_relative)
    if expected_relative.is_absolute() or ".." in expected_relative.parts:
        raise RuntimeError("Expected staged path must be relative and bounded")

    supplied = Path(str(value))
    if ".." in supplied.parts:
        raise RuntimeError(f"Staged path contains traversal: {supplied}")
    if supplied.is_absolute():
        suffix = supplied.parts[-len(expected_relative.parts) :]
        if suffix != expected_relative.parts:
            raise RuntimeError(
                f"Staged absolute path has the wrong artifact suffix: {supplied}"
            )
    elif supplied != expected_relative:
        raise RuntimeError(f"Staged relative path is not canonical: {supplied}")

    candidate = root / expected_relative
    descendant = root
    for part in expected_relative.parts:
        descendant = descendant / part
        if descendant.is_symlink():
            raise RuntimeError(f"Staged path contains a symlink: {descendant}")
    path = candidate.resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"Staged path escapes its root: {path}") from exc
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Staged path is not a regular file: {path}")
    return path


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
        temporary.chmod(0o644)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _candidate_manifest(
    repo_root: Path, staging_root: Path, summary: Mapping[str, Any]
) -> tuple[ArtifactManifest, dict[str, Path]]:
    manifest_path = repo_root / "facetorch" / "models" / "manifest.json"
    manifest = _read_json(manifest_path)
    compatibility = _read_json(manifest_path.parent / manifest["compatibility_ref"])
    governance = _read_json(manifest_path.parent / manifest["governance_ref"])
    cohort = str(summary["torch_minor"])
    results = {str(result["model_id"]): result for result in summary.get("results", [])}
    if set(results) != set(manifest["models"]):
        raise RuntimeError(
            "Staged summary does not cover the complete default analyzer"
        )

    staged_paths = {}
    for model_id, model in manifest["models"].items():
        result = results[model_id]
        if result.get("status") != "ok" or result.get("validation_status") != "ok":
            raise RuntimeError(f"Staged model is not validated: {model_id}")
        matching = [
            artifact
            for artifact in model["artifacts"]
            if artifact.get("format") == "pt2"
            and str(artifact.get("artifact_cohort", artifact.get("torch_min")))
            == cohort
        ]
        if len(matching) != 1:
            raise RuntimeError(f"Manifest cohort selection is ambiguous: {model_id}")
        descriptor = matching[0]
        cohort_root = Path(f"torch-{cohort}") / model_id
        artifact_path = _bounded_file(
            staging_root,
            result["artifact"],
            cohort_root / str(descriptor["filename"]),
        )
        metadata_path = _bounded_file(
            staging_root,
            result["meta"],
            cohort_root / str(descriptor["validation_metadata"]),
        )
        metadata = _read_json(metadata_path)
        if (
            _sha256(artifact_path) != result.get("sha256")
            or metadata.get("artifact_sha256") != result.get("sha256")
            or metadata.get("artifact_size_bytes") != artifact_path.stat().st_size
        ):
            raise RuntimeError(f"Staged artifact binding failed: {model_id}")
        descriptor["sha256"] = result["sha256"]
        descriptor["size_bytes"] = artifact_path.stat().st_size
        staged_paths[model_id] = artifact_path

    candidate = ArtifactManifest.from_mapping(
        manifest, compatibility=compatibility, governance=governance
    )
    return candidate, staged_paths


def _promote_candidate(source: Path, target: Path, descriptor) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
            temporary = Path(output.name)
            with source.open("rb") as input_file:
                shutil.copyfileobj(input_file, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        verify_artifact(temporary, descriptor)
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _staged_alignment_metadata(staging_root: Path, descriptor) -> Path:
    report_path = _bounded_file(
        staging_root,
        ALIGNMENT_METADATA_REPORT,
        ALIGNMENT_METADATA_REPORT,
    )
    report = _read_json(report_path)
    expected_sha256 = str(descriptor.sha256)
    expected_size = int(descriptor.size_bytes)
    expected = {
        "schema_version": 1,
        "status": "ok",
        "artifact_id": ALIGNMENT_METADATA_ID,
        "source": "gdrive",
        "downloader": str(descriptor._target_),
        "file_id": str(descriptor.file_id),
        "revision": str(descriptor.revision),
        "expected_format": str(descriptor.expected_format),
        "staged_path": ALIGNMENT_METADATA_RELATIVE_PATH.as_posix(),
        "size_bytes": expected_size,
        "sha256": expected_sha256,
    }
    if report != expected:
        raise RuntimeError("Staged alignment metadata report is not canonical")
    source = _bounded_file(
        staging_root,
        report["staged_path"],
        ALIGNMENT_METADATA_RELATIVE_PATH,
    )
    if source.stat().st_size != expected_size or _sha256(source) != expected_sha256:
        raise RuntimeError("Staged alignment metadata failed its byte binding")
    return source


def _promote_metadata(source: Path, target: Path, descriptor) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.expanduser().absolute() == source:
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
            temporary = Path(output.name)
            with source.open("rb") as input_file:
                shutil.copyfileobj(input_file, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        if temporary.stat().st_size != int(descriptor.size_bytes) or _sha256(
            temporary
        ) != str(descriptor.sha256):
            raise RuntimeError("Copied alignment metadata failed its byte binding")
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _prepare_cache(
    staging_root: Path,
    config,
    manifest: ArtifactManifest,
    staged_paths: Mapping[str, Path],
    device: str,
) -> list[str]:
    configured = [config.analyzer.detector.downloader]
    configured.extend(value.downloader for value in config.analyzer.predictor.values())
    artifact_ids = []
    for downloader in configured:
        model_id = str(downloader.manifest_id)
        descriptor = manifest.candidates(
            model_id,
            torch_version=str(torch.__version__),
            device=device,
            allow_legacy_models=False,
        )[0]
        target = descriptor.cache_path(str(downloader.path_local))
        _promote_candidate(staged_paths[model_id], target, descriptor)
        artifact_ids.append(descriptor.artifact_id)

    metadata = config.analyzer.utilizer.align.downloader_meta
    metadata_source = _staged_alignment_metadata(staging_root, metadata)
    metadata_target = Path(str(metadata.path_local))
    _promote_metadata(metadata_source, metadata_target, metadata)
    artifact_ids.append(ALIGNMENT_METADATA_ID)
    return artifact_ids


def _au_logits(result) -> list[torch.Tensor]:
    return [face.preds["au"].logits.detach().cpu().clone() for face in result.faces]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--image", type=Path, default=Path("data/input/test.jpg"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.repeats < 2:
        raise RuntimeError("At least two repeated inference calls are required")
    repo_root = args.repo_root.resolve()
    staging_root = args.staging_root.resolve()
    summary = _read_json(args.summary.resolve())
    runtime_minor = ".".join(torch.__version__.split("+", 1)[0].split(".")[:2])
    if summary.get("status") != "ok" or summary.get("torch_minor") != runtime_minor:
        raise RuntimeError("Summary does not match the installed Torch runtime")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the staged default-analyzer smoke")

    cache_root = (
        args.cache_root.resolve()
        if args.cache_root is not None
        else staging_root / f"runtime-cache-{args.device}"
    )
    os.environ["FACETORCH_CACHE_DIR"] = str(cache_root)
    manifest, staged_paths = _candidate_manifest(repo_root, staging_root, summary)
    profile = "gpu" if args.device == "cuda" else "cpu"
    config = facetorch.load_config(profile, offline=True)
    artifact_ids = _prepare_cache(
        staging_root, config, manifest, staged_paths, args.device
    )

    import facetorch.downloader as downloader_module

    downloader_module.get_model_manifest = lambda: manifest
    analyzer = facetorch.FaceAnalyzer(config.analyzer)
    image_path = args.image if args.image.is_absolute() else repo_root / args.image
    results = [
        analyzer.run(
            image_source=image_path,
            face_batch_size=8,
            include_tensors=True,
        )
        for _ in range(args.repeats)
    ]
    if not results[0].faces:
        raise RuntimeError("Default analyzer smoke detected no faces")
    expected_predictors = set(analyzer.configured_predictors)
    if any(
        set(face.preds) != expected_predictors
        for result in results
        for face in result.faces
    ):
        raise RuntimeError("Default analyzer did not execute every predictor")

    reference = _au_logits(results[0])
    worst_au_repeat_diff = 0.0
    for result in results[1:]:
        current = _au_logits(result)
        if len(current) != len(reference):
            raise RuntimeError("Repeated AU inference changed face count")
        for expected, actual in zip(reference, current):
            difference = float((expected - actual).abs().max().item())
            worst_au_repeat_diff = max(worst_au_repeat_diff, difference)
            if difference > 1e-6:
                raise RuntimeError("Repeated AU inference is not stable")

    components = [analyzer.detector]
    components.extend(
        analyzer.predictors[name] for name in analyzer.configured_predictors
    )
    active = [component.downloader.active_descriptor for component in components]
    if any(descriptor is None or descriptor.format != "pt2" for descriptor in active):
        raise RuntimeError("Legacy fallback occurred during default-analyzer smoke")

    report = {
        "schema_version": 1,
        "status": "ok",
        "uid": os.getuid(),
        "device": args.device,
        "torch_version": str(torch.__version__),
        "cuda_runtime": str(torch.version.cuda),
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "summary_sha256": _sha256(args.summary.resolve()),
        "image_sha256": _sha256(image_path),
        "repeats": args.repeats,
        "face_counts": [len(result.faces) for result in results],
        "predictors": list(analyzer.configured_predictors),
        "active_artifacts": artifact_ids,
        "legacy_fallback": False,
        "worst_au_repeat_max_abs": worst_au_repeat_diff,
    }
    _write_json_atomic(args.report.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
