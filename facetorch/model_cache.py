"""Public planning, prefetch, inspection, and cache-recovery APIs."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable, Optional, Sequence

from hydra.utils import instantiate
import torch

from facetorch.artifacts import (
    ArtifactDescriptor,
    detect_model_format,
    get_model_manifest,
    incompatibility_key,
    read_incompatible_artifact_ids,
    sha256_file,
    verify_artifact,
)
from facetorch.configuration import load_config
from facetorch.exceptions import ArtifactIntegrityError, ConfigurationError
from facetorch.paths import get_metadata_dir, get_model_dir


@dataclass(frozen=True)
class PrefetchItem:
    """One selected artifact and its current verified-cache state."""

    component: str
    artifact_id: str
    path: Path
    format: str
    size_bytes: int
    sha256: str
    cached: bool


@dataclass(frozen=True)
class PrefetchPlan:
    """Download-cost estimate produced before any network request."""

    profile: str
    items: tuple[PrefetchItem, ...]

    @property
    def total_bytes(self) -> int:
        return sum(item.size_bytes for item in self.items)

    @property
    def cached_bytes(self) -> int:
        return sum(item.size_bytes for item in self.items if item.cached)

    @property
    def download_bytes(self) -> int:
        return self.total_bytes - self.cached_bytes


@dataclass(frozen=True)
class PrefetchResult:
    """Completed prefetch result with authenticated local paths."""

    plan: PrefetchPlan
    paths: tuple[Path, ...]


@dataclass(frozen=True)
class CacheEntryInspection:
    """Non-executing inspection result for one possible legacy artifact."""

    path: Path
    size_bytes: int
    sha256: str
    detected_format: str
    mislabeled: bool


@dataclass(frozen=True)
class CacheCleanupReport:
    """Quarantine inventory and optional explicit cleanup result."""

    paths: tuple[Path, ...]
    total_bytes: int
    deleted: bool


def _selected_predictors(cfg, include_predictors: Optional[Iterable[str]]) -> list[str]:
    configured = list(cfg.analyzer.predictor) if "predictor" in cfg.analyzer else []
    if include_predictors is None:
        return configured
    if isinstance(include_predictors, (str, bytes)):
        raise ConfigurationError("include_predictors must be a collection of names.")
    requested = list(include_predictors)
    if any(not isinstance(name, str) or not name for name in requested):
        raise ConfigurationError("Predictor names must be non-empty strings.")
    if len(set(requested)) != len(requested):
        raise ConfigurationError("include_predictors contains a duplicate name.")
    unknown = sorted(set(requested) - set(configured))
    if unknown:
        raise ConfigurationError(f"Unknown predictor(s): {', '.join(unknown)}.")
    requested_set = set(requested)
    return [name for name in configured if name in requested_set]


def _is_verified(path: Path, descriptor: ArtifactDescriptor) -> bool:
    if not path.is_file():
        return False
    try:
        verify_artifact(path, descriptor)
    except ArtifactIntegrityError:
        return False
    return True


def _metadata_prefetch_item(cfg) -> PrefetchItem:
    downloader = cfg.analyzer.utilizer.align.downloader_meta
    path = Path(str(downloader.path_local)).expanduser()
    expected_size = int(downloader.size_bytes)
    expected_hash = str(downloader.sha256)
    cached = (
        path.is_file()
        and path.stat().st_size == expected_size
        and sha256_file(path) == expected_hash
    )
    return PrefetchItem(
        component="align-metadata",
        artifact_id="align-3dmm-metadata-v1",
        path=path,
        format="torch_data",
        size_bytes=expected_size,
        sha256=expected_hash,
        cached=cached,
    )


def plan_model_prefetch(
    profile: str = "cpu",
    *,
    include_predictors: Optional[Iterable[str]] = None,
    skip_detector: bool = False,
    offline: Optional[bool] = None,
    allow_legacy_models: bool = False,
    overrides: Optional[Sequence[str]] = None,
) -> PrefetchPlan:
    """Resolve exact artifacts and costs without creating files or using the network."""
    if not isinstance(skip_detector, bool):
        raise ConfigurationError("skip_detector must be a boolean.")
    cfg = load_config(
        profile,
        overrides=overrides,
        offline=offline,
        allow_legacy_models=allow_legacy_models,
    )
    predictor_names = _selected_predictors(cfg, include_predictors)
    selected_configs: list[tuple[str, object]] = []
    if not skip_detector and "detector" in cfg.analyzer:
        selected_configs.append(("detector", cfg.analyzer.detector.downloader))
    selected_configs.extend(
        (f"predictor.{name}", cfg.analyzer.predictor[name].downloader)
        for name in predictor_names
    )

    manifest = get_model_manifest()
    items: list[PrefetchItem] = []
    for component, downloader in selected_configs:
        sidecar = (
            Path(str(downloader.path_local)).expanduser().parent
            / ".incompatible.json"
        )
        key = incompatibility_key(
            manifest.manifest_revision,
            str(torch.__version__),
            str(downloader.device),
        )
        try:
            incompatible = read_incompatible_artifact_ids(sidecar, key)
        except ArtifactIntegrityError:
            # Planning is deliberately non-mutating. Runtime resolution will
            # quarantine the malformed sidecar and make this same empty choice.
            incompatible = set()
        candidates = manifest.candidates(
            str(downloader.manifest_id),
            torch_version=str(torch.__version__),
            device=str(downloader.device),
            allow_legacy_models=allow_legacy_models,
            incompatible_artifact_ids=incompatible,
        )
        descriptor = candidates[0]
        path = descriptor.cache_path(str(downloader.path_local))
        items.append(
            PrefetchItem(
                component=component,
                artifact_id=descriptor.artifact_id,
                path=path,
                format=descriptor.format,
                size_bytes=descriptor.size_bytes,
                sha256=descriptor.sha256,
                cached=_is_verified(path, descriptor),
            )
        )
    if "align" in predictor_names:
        items.append(_metadata_prefetch_item(cfg))
    return PrefetchPlan(profile=profile, items=tuple(items))


def prefetch_models(
    profile: str = "cpu",
    *,
    include_predictors: Optional[Iterable[str]] = None,
    skip_detector: bool = False,
    offline: Optional[bool] = None,
    allow_legacy_models: bool = False,
    overrides: Optional[Sequence[str]] = None,
    confirm: bool = False,
) -> PrefetchResult:
    """Download exactly a planned selection after explicit bulk-cost confirmation."""
    requested_predictors = (
        tuple(include_predictors)
        if include_predictors is not None
        and not isinstance(include_predictors, (str, bytes))
        else include_predictors
    )
    plan = plan_model_prefetch(
        profile,
        include_predictors=requested_predictors,
        skip_detector=skip_detector,
        offline=offline,
        allow_legacy_models=allow_legacy_models,
        overrides=overrides,
    )
    if plan.download_bytes and len(plan.items) > 1 and not confirm:
        mib = plan.download_bytes / (1024 * 1024)
        raise ConfigurationError(
            f"Prefetch would download approximately {mib:.1f} MiB across "
            f"{len(plan.items)} artifacts. Review plan_model_prefetch() and pass "
            "confirm=True to continue."
        )

    cfg = load_config(
        profile,
        overrides=overrides,
        offline=offline,
        allow_legacy_models=allow_legacy_models,
    )
    predictor_names = _selected_predictors(cfg, requested_predictors)
    downloader_configs = []
    if not skip_detector and "detector" in cfg.analyzer:
        downloader_configs.append(cfg.analyzer.detector.downloader)
    downloader_configs.extend(
        cfg.analyzer.predictor[name].downloader for name in predictor_names
    )
    if "align" in predictor_names:
        downloader_configs.append(
            cfg.analyzer.utilizer.align.downloader_meta
        )

    paths = []
    for downloader_config in downloader_configs:
        downloader = instantiate(downloader_config)
        paths.append(Path(downloader.run()))
    return PrefetchResult(plan=plan, paths=tuple(paths))


def inspect_legacy_cache(path: str | os.PathLike) -> tuple[CacheEntryInspection, ...]:
    """Hash and classify old model files without deserializing or executing them."""
    root = Path(path).expanduser()
    if not root.exists():
        raise ConfigurationError(f"Legacy cache path does not exist: {root}.")
    candidates = [root] if root.is_file() else sorted(root.rglob("*"))
    entries = []
    for candidate in candidates:
        if not candidate.is_file() or candidate.suffix.lower() not in {".pt", ".pt2"}:
            continue
        detected = detect_model_format(candidate)
        entries.append(
            CacheEntryInspection(
                path=candidate,
                size_bytes=candidate.stat().st_size,
                sha256=sha256_file(candidate),
                detected_format=detected,
                mislabeled=candidate.suffix.lower() == ".pt2"
                and detected == "torchscript",
            )
        )
    return tuple(entries)


def migrate_legacy_artifact(
    source: str | os.PathLike,
    artifact_id: str,
    destination: str | os.PathLike,
) -> Path:
    """Copy one exact manifest match into v1 layout without changing the source."""
    source_path = Path(source).expanduser()
    destination_path = Path(destination).expanduser()
    descriptor = get_model_manifest().descriptor(artifact_id)
    if destination_path.name != descriptor.filename:
        raise ConfigurationError(
            f"Migration destination must preserve the authenticated filename "
            f"{descriptor.filename!r}."
        )
    verify_artifact(source_path, descriptor)
    if destination_path.exists():
        try:
            return verify_artifact(destination_path, descriptor)
        except ArtifactIntegrityError as exc:
            raise ArtifactIntegrityError(
                f"Migration destination already exists and is not the requested "
                f"artifact: {destination_path}."
            ) from exc
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination_path.name}.",
            suffix=".tmp",
            dir=destination_path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            with source_path.open("rb") as source_file:
                shutil.copyfileobj(source_file, temporary, length=1024 * 1024)
            temporary.flush()
            os.fsync(temporary.fileno())
        verify_artifact(temporary_path, descriptor)
        os.replace(temporary_path, destination_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return verify_artifact(destination_path, descriptor)


def _allowed_quarantine_roots(
    root: Optional[str | os.PathLike],
) -> tuple[Path, ...]:
    allowed = (get_model_dir().resolve(), get_metadata_dir().resolve())
    if root is None:
        return allowed
    selected = Path(root).expanduser().resolve()
    if not any(selected == base or selected.is_relative_to(base) for base in allowed):
        raise ConfigurationError(
            "Quarantine cleanup is restricted to facetorch's versioned model and "
            "metadata cache directories."
        )
    return (selected,)


def inspect_quarantined_cache(
    root: Optional[str | os.PathLike] = None,
) -> CacheCleanupReport:
    """Report quarantined entries and reclaimable bytes without deleting anything."""
    paths = []
    for cache_root in _allowed_quarantine_roots(root):
        if cache_root.exists():
            paths.extend(
                path
                for path in cache_root.rglob("*.quarantine.*")
                if path.is_file()
            )
    unique_paths = tuple(sorted(set(paths)))
    return CacheCleanupReport(
        paths=unique_paths,
        total_bytes=sum(path.stat().st_size for path in unique_paths),
        deleted=False,
    )


def cleanup_quarantined_cache(
    root: Optional[str | os.PathLike] = None,
    *,
    confirm: bool = False,
) -> CacheCleanupReport:
    """Delete only reported quarantine files and only after explicit confirmation."""
    report = inspect_quarantined_cache(root)
    if not confirm:
        return report
    for path in report.paths:
        path.unlink()
    return CacheCleanupReport(
        paths=report.paths,
        total_bytes=report.total_bytes,
        deleted=True,
    )


def inspect_incompatible_cache(
    root: Optional[str | os.PathLike] = None,
) -> CacheCleanupReport:
    """Report persisted runtime/schema rejections without changing the cache."""
    model_root = get_model_dir().resolve()
    selected = model_root if root is None else Path(root).expanduser().resolve()
    if selected != model_root and not selected.is_relative_to(model_root):
        raise ConfigurationError(
            "Incompatibility reset is restricted to facetorch's versioned model "
            "cache directory."
        )
    paths = (
        tuple(sorted(selected.rglob(".incompatible.json")))
        if selected.exists()
        else ()
    )
    files = tuple(path for path in paths if path.is_file())
    return CacheCleanupReport(
        paths=files,
        total_bytes=sum(path.stat().st_size for path in files),
        deleted=False,
    )


def reset_incompatible_cache(
    root: Optional[str | os.PathLike] = None,
    *,
    confirm: bool = False,
) -> CacheCleanupReport:
    """Explicitly clear persisted runtime/schema rejections after remediation."""
    report = inspect_incompatible_cache(root)
    if not confirm:
        return report
    for path in report.paths:
        path.unlink()
    return CacheCleanupReport(
        paths=report.paths,
        total_bytes=report.total_bytes,
        deleted=True,
    )


__all__ = [
    "CacheCleanupReport",
    "CacheEntryInspection",
    "PrefetchItem",
    "PrefetchPlan",
    "PrefetchResult",
    "cleanup_quarantined_cache",
    "inspect_incompatible_cache",
    "inspect_legacy_cache",
    "inspect_quarantined_cache",
    "migrate_legacy_artifact",
    "plan_model_prefetch",
    "prefetch_models",
    "reset_incompatible_cache",
]
