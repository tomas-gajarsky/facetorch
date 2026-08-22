"""Authenticated, manifest-aware model artifact downloaders."""

from __future__ import annotations

from contextlib import AbstractContextManager
import errno
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Dict, List, Mapping, Optional
from uuid import uuid4
import warnings

import gdown
from huggingface_hub import hf_hub_download

from facetorch import base
from facetorch.artifacts import (
    ArtifactDescriptor,
    ArtifactManifest,
    get_model_manifest,
    normalize_device,
    parse_runtime_version,
    verify_artifact,
)
from facetorch.exceptions import (
    ArtifactIntegrityError,
    CacheLockError,
    ConfigurationError,
    LegacyModelWarning,
    ModelCompatibilityError,
    OfflineCacheError,
)
from facetorch.logger import LoggerJsonFile
from facetorch.paths import get_offline_mode

logger = LoggerJsonFile().logger


class _DirectoryLock(AbstractContextManager):
    """Small cross-platform lock based on atomic directory creation."""

    def __init__(self, path: Path, *, timeout: float = 600.0) -> None:
        self.path = path
        self.timeout = timeout

    @staticmethod
    def _process_exists(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError as exc:
            return exc.errno != errno.ESRCH
        return True

    def _reclaim_stale_owner(self) -> bool:
        """Atomically remove a lock whose recorded process no longer exists."""
        owner_path = self.path / "owner.json"
        try:
            owner = json.loads(owner_path.read_text(encoding="utf-8"))
            pid = int(owner["pid"])
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
            try:
                old_enough = time.time() - self.path.stat().st_mtime > 5.0
            except OSError:
                return False
            if not old_enough:
                return False
        else:
            if self._process_exists(pid):
                return False

        stale_path = self.path.with_name(
            f"{self.path.name}.stale.{uuid4().hex}"
        )
        try:
            os.replace(self.path, stale_path)
        except (FileNotFoundError, FileExistsError, OSError):
            return False
        shutil.rmtree(stale_path, ignore_errors=True)
        return True

    def __enter__(self) -> "_DirectoryLock":
        started = time.monotonic()
        while True:
            try:
                self.path.mkdir()
                break
            except FileExistsError:
                if self._reclaim_stale_owner():
                    continue
                if time.monotonic() - started >= self.timeout:
                    raise CacheLockError(
                        f"Timed out waiting for model cache lock {self.path}. "
                        "Confirm no facetorch process owns it before removing the "
                        "lock directory and retrying."
                    )
                time.sleep(0.05)
        try:
            (self.path / "owner.json").write_text(
                json.dumps({"pid": os.getpid(), "created": time.time()}),
                encoding="utf-8",
            )
        except OSError:
            pass
        return self

    def __exit__(self, *_exc: object) -> None:
        shutil.rmtree(self.path, ignore_errors=True)


def _ensure_directory(path: Path) -> None:
    if path == Path("."):
        return
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as exc:
        raise ConfigurationError(
            f"Cannot create model cache directory {os.fspath(path)!r}. "
            "Set FACETORCH_CACHE_DIR to a writable directory or override path_local."
        ) from exc


def _quarantine(path: Path, reason: str) -> Optional[Path]:
    """Atomically retain an invalid cache entry under a visible quarantine name."""
    if not path.exists():
        return None
    destination = path.with_name(
        f"{path.name}.quarantine.{int(time.time())}.{uuid4().hex[:8]}"
    )
    try:
        os.replace(path, destination)
    except OSError as exc:
        raise ArtifactIntegrityError(
            f"Invalid cache entry {path} could not be quarantined: {reason}."
        ) from exc
    logger.warning(f"Quarantined invalid cache entry {path}: {reason}")
    return destination


def _atomic_promote(
    candidate: Path,
    target: Path,
    descriptor: ArtifactDescriptor,
) -> None:
    """Move into same-filesystem staging, verify it, then atomically replace."""
    staging_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as staging:
            staging_path = Path(staging.name)
        moved = False
        if not candidate.is_symlink():
            try:
                os.replace(candidate, staging_path)
                moved = True
            except OSError as exc:
                if exc.errno != errno.EXDEV:
                    raise
        if not moved:
            with candidate.open("rb") as downloaded, staging_path.open("wb") as staging:
                shutil.copyfileobj(downloaded, staging, length=1024 * 1024)
                staging.flush()
                os.fsync(staging.fileno())
        else:
            with staging_path.open("rb") as staging:
                os.fsync(staging.fileno())
        verify_artifact(staging_path, descriptor)
        os.replace(staging_path, target)
    finally:
        if staging_path is not None:
            staging_path.unlink(missing_ok=True)


def _direct_descriptor(
    *,
    source: str,
    repo_id: str,
    revision: Optional[str],
    filename: str,
    path_local: str,
    sha256: Optional[str],
    size_bytes: Optional[int],
    expected_format: Optional[str],
    device: Any,
) -> ArtifactDescriptor:
    if not revision or not sha256 or size_bytes is None:
        raise ConfigurationError(
            "Remote artifacts outside the facetorch manifest must declare an immutable "
            "revision, SHA-256, and size_bytes before download."
        )
    artifact_format = expected_format
    if artifact_format is None:
        suffix = Path(filename or path_local).suffix.lower()
        artifact_format = "pt2" if suffix == ".pt2" else "torchscript"
    return ArtifactDescriptor.from_mapping(
        "external-artifact",
        {
            "task": "external",
            "source": source,
            "repo_id": repo_id,
            "revision": revision,
        },
        {
            "id": "external-artifact",
            "filename": filename,
            "format": artifact_format,
            "sha256": sha256,
            "size_bytes": size_bytes,
            "torch_min": None,
            "torch_max_exclusive": None,
            "devices": [normalize_device(device)],
            "schema_major": None,
            "schema_minor": None,
            "validation_metadata": None,
        },
    )


class _VerifiedDownloader(base.BaseDownloader):
    """Shared verification, cache locking, and atomic promotion behavior."""

    def __init__(
        self,
        file_id: str,
        path_local: str,
        *,
        offline: Optional[bool],
        allow_legacy_models: bool,
        verify_on_use: bool,
    ) -> None:
        super().__init__(file_id, path_local)
        if not isinstance(allow_legacy_models, bool):
            raise ConfigurationError("allow_legacy_models must be a boolean.")
        if offline is not None and not isinstance(offline, bool):
            raise ConfigurationError("offline must be a boolean or None.")
        if not isinstance(verify_on_use, bool):
            raise ConfigurationError("verify_on_use must be a boolean.")
        self.offline = get_offline_mode() if offline is None else offline
        self.allow_legacy_models = allow_legacy_models
        self.enable_legacy_models = allow_legacy_models
        self.verify_on_use = verify_on_use
        self.active_descriptor: Optional[ArtifactDescriptor] = None
        self.active_format: Optional[str] = None
        self._legacy_warning_emitted = False

    def _warn_if_legacy(self, descriptor: ArtifactDescriptor) -> None:
        if descriptor.format != "torchscript" or self._legacy_warning_emitted:
            return
        warnings.warn(
            f"Using explicitly enabled legacy TorchScript artifact "
            f"{descriptor.artifact_id!r}; prefer a validated .pt2 cohort.",
            LegacyModelWarning,
            stacklevel=3,
        )
        self._legacy_warning_emitted = True

    def _activate(self, descriptor: ArtifactDescriptor, path: Path) -> str:
        self.active_descriptor = descriptor
        self.active_format = descriptor.format
        self.path_local = os.fspath(path)
        self._warn_if_legacy(descriptor)
        return self.path_local

    def _verified_existing(
        self, descriptor: ArtifactDescriptor, target: Path
    ) -> Optional[str]:
        if not target.is_file():
            return None
        try:
            verify_artifact(target, descriptor)
        except ArtifactIntegrityError as exc:
            _quarantine(target, str(exc))
            return None
        return self._activate(descriptor, target)


class DownloaderGDrive(_VerifiedDownloader):
    """Verified Google Drive downloader for explicitly described artifacts."""

    def __init__(
        self,
        file_id: str,
        path_local: str,
        *,
        sha256: Optional[str] = None,
        size_bytes: Optional[int] = None,
        expected_format: Optional[str] = None,
        revision: str = "local-gdrive-object-v1",
        offline: Optional[bool] = None,
        allow_legacy_models: bool = False,
        verify_on_use: bool = True,
        device: Any = "cpu",
    ) -> None:
        super().__init__(
            file_id,
            path_local,
            offline=offline,
            allow_legacy_models=allow_legacy_models,
            verify_on_use=verify_on_use,
        )
        self.sha256 = sha256
        self.size_bytes = size_bytes
        self.expected_format = expected_format
        self.revision = revision
        self.device = device

    def _descriptor(self) -> ArtifactDescriptor:
        filename = Path(self.path_local).name
        return _direct_descriptor(
            source="gdrive",
            repo_id=self.file_id,
            revision=self.revision,
            filename=filename,
            path_local=self.path_local,
            sha256=self.sha256,
            size_bytes=self.size_bytes,
            expected_format=self.expected_format,
            device=self.device,
        )

    def run(self, force_download: bool = False) -> str:
        target = Path(self.path_local).expanduser()
        _ensure_directory(target.parent)
        descriptor = self._descriptor()
        if descriptor.format == "torchscript" and not self.allow_legacy_models:
            raise ModelCompatibilityError(
                "Google Drive TorchScript models require allow_legacy_models=True."
            )
        with _DirectoryLock(target.parent / ".facetorch-download.lock"):
            if not force_download:
                existing = self._verified_existing(descriptor, target)
                if existing is not None:
                    return existing
            if self.offline:
                raise OfflineCacheError(
                    f"Offline mode requires a verified cached artifact at {target}."
                )
            with tempfile.TemporaryDirectory(
                prefix=".facetorch-download-", dir=target.parent
            ) as temporary_dir:
                temporary_path = Path(temporary_dir) / descriptor.filename
                url = (
                    "https://drive.google.com/uc?&id="
                    f"{self.file_id}&confirm=t"
                )
                downloaded = gdown.download(
                    url, output=os.fspath(temporary_path), quiet=False
                )
                candidate = Path(downloaded) if downloaded else temporary_path
                _atomic_promote(candidate, target, descriptor)
        return self._activate(descriptor, target)


class DownloaderHuggingFace(_VerifiedDownloader):
    """Resolve and authenticate one immutable Hugging Face model artifact."""

    def __init__(
        self,
        file_id: str,
        path_local: str,
        repo_id: Optional[str] = None,
        filename: Optional[str] = None,
        export_filenames_by_torch_minor: Optional[Dict[str, str]] = None,
        fallback_filenames: Optional[List[str]] = None,
        enable_default_torch_export_routing: bool = False,
        *,
        manifest_id: Optional[str] = None,
        revision: Optional[str] = None,
        sha256: Optional[str] = None,
        size_bytes: Optional[int] = None,
        expected_format: Optional[str] = None,
        offline: Optional[bool] = None,
        allow_legacy_models: bool = False,
        verify_on_use: bool = True,
        device: Any = "cpu",
        manifest: Optional[ArtifactManifest] = None,
        torch_version: Optional[str] = None,
    ) -> None:
        super().__init__(
            file_id,
            path_local,
            offline=offline,
            allow_legacy_models=allow_legacy_models,
            verify_on_use=verify_on_use,
        )
        self.repo_id = repo_id if repo_id else file_id
        self.filename = filename if filename else Path(path_local).name
        self.manifest_id = manifest_id
        self.revision = revision
        self.sha256 = sha256
        self.size_bytes = size_bytes
        self.expected_format = expected_format
        self.device = device
        self.manifest = manifest or get_model_manifest()
        self.torch_version = torch_version
        # Retained as inert attributes for source-configuration compatibility.
        self.export_filenames_by_torch_minor = export_filenames_by_torch_minor or {}
        self.fallback_filenames = fallback_filenames or []
        self.enable_default_torch_export_routing = enable_default_torch_export_routing
        self._candidate_index = -1
        self._resolved_candidates: tuple[ArtifactDescriptor, ...] = ()
        self._active_filename: Optional[str] = None
        self._last_candidates: List[str] = []

    def _runtime_version(self) -> str:
        if self.torch_version is not None:
            return self.torch_version
        import torch

        return str(torch.__version__)

    @property
    def _incompatibility_path(self) -> Path:
        return Path(self.path_local).expanduser().parent / ".incompatible.json"

    def _incompatibility_key(self) -> str:
        device = normalize_device(self.device)
        runtime = parse_runtime_version(self._runtime_version())
        return (
            f"{self.manifest.manifest_revision}|"
            f"{runtime[0]}.{runtime[1]}|{device}"
        )

    def _read_incompatible(self) -> set[str]:
        path = self._incompatibility_path
        if not path.is_file():
            return set()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            values = raw.get(self._incompatibility_key(), [])
            if not isinstance(values, list) or not all(
                isinstance(value, str) for value in values
            ):
                raise ValueError("invalid incompatibility record")
            return set(values)
        except (OSError, ValueError, json.JSONDecodeError):
            _quarantine(path, "invalid incompatibility sidecar")
            return set()

    def mark_incompatible(self) -> None:
        """Persist a runtime/schema rejection without executing the artifact again."""
        if self.active_descriptor is None or self.manifest_id is None:
            return
        path = self._incompatibility_path
        _ensure_directory(path.parent)
        with _DirectoryLock(path.parent / ".facetorch-sidecar.lock"):
            raw: Mapping[str, Any] = {}
            if path.is_file():
                try:
                    loaded = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        raw = loaded
                except (OSError, json.JSONDecodeError):
                    _quarantine(path, "invalid incompatibility sidecar")
            updated = dict(raw)
            values = set(updated.get(self._incompatibility_key(), []))
            values.add(self.active_descriptor.artifact_id)
            updated[self._incompatibility_key()] = sorted(values)
            temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
            temporary.write_text(
                json.dumps(updated, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, path)

    def _resolve_candidates(self) -> tuple[ArtifactDescriptor, ...]:
        if self.manifest_id is None:
            descriptor = _direct_descriptor(
                source="huggingface",
                repo_id=self.repo_id,
                revision=self.revision,
                filename=self.filename,
                path_local=self.path_local,
                sha256=self.sha256,
                size_bytes=self.size_bytes,
                expected_format=self.expected_format,
                device=self.device,
            )
            candidates = (descriptor,)
        else:
            candidates = self.manifest.candidates(
                self.manifest_id,
                torch_version=self._runtime_version(),
                device=self.device,
                allow_legacy_models=self.allow_legacy_models,
            )
            if any(item.repo_id != self.repo_id for item in candidates):
                raise ConfigurationError(
                    f"Configured repo_id for {self.manifest_id!r} does not match "
                    "the packaged immutable manifest."
                )
            if self.revision is not None and any(
                item.revision != self.revision for item in candidates
            ):
                raise ConfigurationError(
                    f"Configured revision for {self.manifest_id!r} does not match "
                    "the packaged immutable manifest."
                )
        incompatible = self._read_incompatible() if self.manifest_id else set()
        filtered = tuple(
            item for item in candidates if item.artifact_id not in incompatible
        )
        if not filtered:
            raise ModelCompatibilityError(
                f"All eligible artifacts for {self.manifest_id!r} were already "
                f"rejected by torch {self._runtime_version()} on "
                f"{normalize_device(self.device)}."
            )
        self._resolved_candidates = filtered
        self._last_candidates = [item.filename for item in filtered]
        return filtered

    def _target_for(self, descriptor: ArtifactDescriptor) -> Path:
        return descriptor.cache_path(self.path_local)

    def _download_descriptor(
        self, descriptor: ArtifactDescriptor, *, force_download: bool = False
    ) -> str:
        target = self._target_for(descriptor)
        _ensure_directory(target.parent)
        with _DirectoryLock(target.parent / ".facetorch-download.lock"):
            if not force_download:
                existing = self._verified_existing(descriptor, target)
                if existing is not None:
                    self._active_filename = descriptor.filename
                    return existing
            if self.offline:
                raise OfflineCacheError(
                    f"Offline mode requires verified artifact "
                    f"{descriptor.artifact_id!r} at {target}."
                )
            with tempfile.TemporaryDirectory(
                prefix=".facetorch-download-", dir=target.parent
            ) as temporary_dir:
                downloaded_path = hf_hub_download(
                    repo_id=descriptor.repo_id,
                    filename=descriptor.filename,
                    revision=descriptor.revision,
                    local_dir=temporary_dir,
                    force_download=force_download,
                )
                candidate = Path(downloaded_path)
                _atomic_promote(candidate, target, descriptor)
        self._active_filename = descriptor.filename
        return self._activate(descriptor, target)

    def _download_one_candidate(
        self, filename: str, force_download: bool = False
    ) -> str:
        """Download one authenticated candidate; retained for targeted callers."""
        candidates = self._resolved_candidates or self._resolve_candidates()
        try:
            descriptor = next(item for item in candidates if item.filename == filename)
        except StopIteration as exc:
            raise ConfigurationError(
                f"Filename {filename!r} is not an eligible authenticated candidate."
            ) from exc
        return self._download_descriptor(descriptor, force_download=force_download)

    def _build_candidate_filenames(self) -> List[str]:
        """Return only manifest-eligible candidates; never synthesize filenames."""
        return [item.filename for item in self._resolve_candidates()]

    def run(self, force_download: bool = False) -> str:
        candidates = self._resolve_candidates()
        self._candidate_index = 0
        return self._download_descriptor(
            candidates[0], force_download=force_download
        )

    def try_next(self, force_download: bool = False) -> bool:
        """Select one next manifest candidate after a persisted load rejection."""
        candidates = self._resolve_candidates()
        if self.active_descriptor is None:
            next_index = 0
        else:
            try:
                current = next(
                    index
                    for index, item in enumerate(candidates)
                    if item.artifact_id == self.active_descriptor.artifact_id
                )
                next_index = current + 1
            except StopIteration:
                next_index = 0
        if next_index >= len(candidates):
            return False
        self._candidate_index = next_index
        self._download_descriptor(
            candidates[next_index], force_download=force_download
        )
        return True


__all__ = ["DownloaderGDrive", "DownloaderHuggingFace"]
