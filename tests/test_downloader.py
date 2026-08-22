import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import time
from unittest.mock import patch

import pytest
import torch

from facetorch.artifacts import ArtifactManifest, detect_model_format
from facetorch.downloader import (
    DownloaderGDrive,
    DownloaderHuggingFace,
    _DirectoryLock,
    _atomic_promote,
)
from facetorch.exceptions import (
    ArtifactIntegrityError,
    CacheLockError,
    ConfigurationError,
    LegacyModelWarning,
    ModelCompatibilityError,
    OfflineCacheError,
)


REVISION = "a" * 40


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_export(path):
    exported = torch.export.export(torch.nn.Identity(), (torch.ones(1),))
    torch.export.save(exported, str(path))
    assert detect_model_format(path) == "pt2"
    return path


def _make_torchscript(path):
    scripted = torch.jit.trace(torch.nn.Identity(), torch.ones(1))
    torch.jit.save(scripted, str(path))
    assert detect_model_format(path) == "torchscript"
    return path


def _artifact(path, artifact_id, filename, artifact_format, devices, priority):
    return {
        "id": artifact_id,
        "filename": filename,
        "format": artifact_format,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "torch_min": "2.11",
        "torch_max_exclusive": "2.12",
        "devices": devices,
        "schema_major": 8 if artifact_format == "pt2" else None,
        "schema_minor": 17 if artifact_format == "pt2" else None,
        "validation_metadata": None,
        "priority": priority,
    }


def _manifest(export_path, legacy_path=None, *, export_format="pt2"):
    artifacts = [
        _artifact(
            export_path,
            "toy-torch2.11",
            "toy.pt2",
            export_format,
            ["cpu", "cuda"],
            1,
        )
    ]
    if legacy_path is not None:
        artifacts.append(
            _artifact(
                legacy_path,
                "toy-legacy",
                "toy.pt",
                "torchscript",
                ["cpu"],
                100,
            )
        )
    return ArtifactManifest.from_mapping(
        {
            "manifest_version": 1,
            "manifest_revision": "test-v1",
            "status": "provisional",
            "models": {
                "toy": {
                    "task": "test",
                    "source": "huggingface",
                    "repo_id": "owner/toy",
                    "revision": REVISION,
                    "source_weight_sha256": None,
                    "export_commit": None,
                    "license_ref": None,
                    "artifacts": artifacts,
                }
            },
        }
    )


def _downloader(tmp_path, manifest, **kwargs):
    return DownloaderHuggingFace(
        file_id="owner/toy",
        repo_id="owner/toy",
        filename="model.pt2",
        path_local=str(tmp_path / "cache" / "model.pt2"),
        manifest_id="toy",
        manifest=manifest,
        torch_version="2.11.0",
        device=kwargs.pop("device", "cpu"),
        **kwargs,
    )


def _hub_copy(source):
    def copy_to_local_dir(*, filename, local_dir, **_kwargs):
        destination = Path(local_dir) / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return str(destination)

    return copy_to_local_dir


@pytest.mark.unit
@pytest.mark.downloader
def test_cache_verification_is_secure_by_default_with_explicit_opt_out(tmp_path):
    source = _make_export(tmp_path / "source.pt2")
    manifest = _manifest(source)

    assert _downloader(tmp_path, manifest).verify_on_use is True
    assert (
        _downloader(tmp_path, manifest, verify_on_use=False).verify_on_use is False
    )
    with pytest.raises(ConfigurationError, match="verify_on_use"):
        _downloader(tmp_path, manifest, verify_on_use="sometimes")


@pytest.mark.unit
@pytest.mark.downloader
def test_gdrive_download_is_verified_and_cached(tmp_path):
    source = tmp_path / "source.data"
    source.write_bytes(b"authenticated metadata")
    target = tmp_path / "cache" / "meta.pt"
    downloader = DownloaderGDrive(
        file_id="immutable-file-id",
        path_local=str(target),
        revision="gdrive-object-v1",
        sha256=_sha256(source),
        size_bytes=source.stat().st_size,
        expected_format="torch_data",
    )

    def copy_to_output(_url, *, output, quiet):
        assert quiet is False
        shutil.copy2(source, output)
        return output

    with patch("facetorch.downloader.gdown.download", side_effect=copy_to_output) as call:
        assert Path(downloader.run()) == target
        assert Path(downloader.run()) == target

    assert target.read_bytes() == source.read_bytes()
    call.assert_called_once()


@pytest.mark.unit
@pytest.mark.downloader
def test_external_remote_without_integrity_metadata_fails_closed(tmp_path):
    downloader = DownloaderHuggingFace(
        file_id="owner/toy",
        path_local=str(tmp_path / "model.pt2"),
    )
    with patch("facetorch.downloader.hf_hub_download") as download:
        with pytest.raises(ConfigurationError, match="SHA-256"):
            downloader.run()
    download.assert_not_called()


@pytest.mark.unit
@pytest.mark.downloader
def test_hub_request_is_commit_pinned_and_atomic(tmp_path):
    source = _make_export(tmp_path / "source.pt2")
    downloader = _downloader(tmp_path, _manifest(source))

    with patch(
        "facetorch.downloader.hf_hub_download", side_effect=_hub_copy(source)
    ) as download:
        result = Path(downloader.run())

    assert result.name == "toy.pt2"
    assert result.read_bytes() == source.read_bytes()
    assert downloader.active_format == "pt2"
    assert download.call_args.kwargs["revision"] == REVISION
    assert Path(download.call_args.kwargs["local_dir"]).parent == result.parent


@pytest.mark.unit
@pytest.mark.downloader
def test_verified_cache_is_reused_by_a_new_downloader_without_network(tmp_path):
    source = _make_export(tmp_path / "source.pt2")
    manifest = _manifest(source)
    first = _downloader(tmp_path, manifest)
    with patch(
        "facetorch.downloader.hf_hub_download", side_effect=_hub_copy(source)
    ):
        cached_path = first.run()

    restarted = _downloader(tmp_path, manifest, offline=True)
    with patch("facetorch.downloader.hf_hub_download") as download:
        assert restarted.run() == cached_path
    download.assert_not_called()


@pytest.mark.unit
@pytest.mark.downloader
def test_offline_missing_cache_never_accesses_network(tmp_path):
    source = _make_export(tmp_path / "source.pt2")
    downloader = _downloader(tmp_path, _manifest(source), offline=True)
    with patch("facetorch.downloader.hf_hub_download") as download:
        with pytest.raises(OfflineCacheError, match="Offline mode"):
            downloader.run()
    download.assert_not_called()


@pytest.mark.unit
@pytest.mark.downloader
def test_corrupt_cache_is_quarantined_and_not_executed_offline(tmp_path):
    source = _make_export(tmp_path / "source.pt2")
    downloader = _downloader(tmp_path, _manifest(source), offline=True)
    target = tmp_path / "cache" / "toy.pt2"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"truncated")

    with patch("facetorch.downloader.hf_hub_download") as download:
        with pytest.raises(OfflineCacheError):
            downloader.run()

    assert not target.exists()
    assert len(list(target.parent.glob("toy.pt2.quarantine.*"))) == 1
    download.assert_not_called()


@pytest.mark.unit
@pytest.mark.downloader
def test_wrong_hash_and_wrong_format_downloads_are_rejected(tmp_path):
    valid = _make_export(tmp_path / "valid.pt2")
    wrong_hash = _make_export(tmp_path / "wrong-hash.pt2")
    wrong_hash.write_bytes(wrong_hash.read_bytes() + b"changed")
    downloader = _downloader(tmp_path, _manifest(valid))
    with patch(
        "facetorch.downloader.hf_hub_download",
        side_effect=_hub_copy(wrong_hash),
    ):
        with pytest.raises(ArtifactIntegrityError):
            downloader.run()
    assert not (tmp_path / "cache" / "toy.pt2").exists()

    torchscript = _make_torchscript(tmp_path / "wrong-format.pt")
    wrong_format_manifest = _manifest(torchscript, export_format="pt2")
    wrong_format = _downloader(tmp_path, wrong_format_manifest)
    with patch(
        "facetorch.downloader.hf_hub_download",
        side_effect=_hub_copy(torchscript),
    ):
        with pytest.raises(ArtifactIntegrityError, match="format"):
            wrong_format.run()


@pytest.mark.unit
@pytest.mark.downloader
def test_network_failure_does_not_cascade_to_legacy(tmp_path):
    exported = _make_export(tmp_path / "export.pt2")
    legacy = _make_torchscript(tmp_path / "legacy.pt")
    downloader = _downloader(
        tmp_path,
        _manifest(exported, legacy),
        allow_legacy_models=True,
    )
    with patch(
        "facetorch.downloader.hf_hub_download",
        side_effect=ConnectionError("network unavailable"),
    ) as download:
        with pytest.raises(ConnectionError, match="network unavailable"):
            downloader.run()

    assert download.call_count == 1
    assert download.call_args.kwargs["filename"] == "toy.pt2"


@pytest.mark.unit
@pytest.mark.downloader
def test_legacy_requires_opt_in_and_is_never_cuda_eligible(tmp_path):
    exported = _make_export(tmp_path / "export.pt2")
    legacy = _make_torchscript(tmp_path / "legacy.pt")
    manifest = _manifest(exported, legacy)

    disabled = _downloader(tmp_path, manifest)
    assert disabled._build_candidate_filenames() == ["toy.pt2"]

    cpu = _downloader(tmp_path, manifest, allow_legacy_models=True)
    assert cpu._build_candidate_filenames() == ["toy.pt2", "toy.pt"]

    with pytest.raises(ModelCompatibilityError):
        manifest.candidates(
            "toy",
            torch_version="2.10.0",
            device="cuda",
            allow_legacy_models=True,
        )


@pytest.mark.unit
@pytest.mark.downloader
def test_schema_rejection_survives_restart_and_uses_real_legacy_extension(tmp_path):
    exported = _make_export(tmp_path / "export.pt2")
    legacy = _make_torchscript(tmp_path / "legacy.pt")
    manifest = _manifest(exported, legacy)
    cache = tmp_path / "cache"
    cache.mkdir()
    shutil.copy2(exported, cache / "toy.pt2")
    shutil.copy2(legacy, cache / "toy.pt")

    first = _downloader(tmp_path, manifest, allow_legacy_models=True, offline=True)
    assert Path(first.run()).suffix == ".pt2"
    first.mark_incompatible()

    restarted = _downloader(
        tmp_path,
        manifest,
        allow_legacy_models=True,
        offline=True,
    )
    with pytest.warns(LegacyModelWarning, match="legacy TorchScript"):
        result = Path(restarted.run())
    assert result.name == "toy.pt"
    assert restarted.active_format == "torchscript"


@pytest.mark.unit
@pytest.mark.downloader
def test_failed_forced_replacement_preserves_verified_cache(tmp_path):
    source = _make_export(tmp_path / "source.pt2")
    manifest = _manifest(source)
    target = tmp_path / "cache" / "toy.pt2"
    target.parent.mkdir()
    shutil.copy2(source, target)
    before = target.read_bytes()
    downloader = _downloader(tmp_path, manifest)

    with patch(
        "facetorch.downloader.hf_hub_download", side_effect=_hub_copy(source)
    ), patch("facetorch.downloader.os.replace", side_effect=OSError("interrupted")):
        with pytest.raises(OSError, match="interrupted"):
            downloader.run(force_download=True)

    assert target.read_bytes() == before


@pytest.mark.unit
@pytest.mark.downloader
def test_concurrent_first_use_converges_on_one_verified_download(tmp_path):
    source = _make_export(tmp_path / "source.pt2")
    manifest = _manifest(source)
    first = _downloader(tmp_path, manifest)
    second = _downloader(tmp_path, manifest)

    def delayed_copy(**kwargs):
        time.sleep(0.1)
        return _hub_copy(source)(**kwargs)

    with patch(
        "facetorch.downloader.hf_hub_download", side_effect=delayed_copy
    ) as download:
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda item: item.run(), (first, second)))

    assert results[0] == results[1]
    assert Path(results[0]).read_bytes() == source.read_bytes()
    assert download.call_count == 1


@pytest.mark.unit
@pytest.mark.downloader
def test_directory_lock_reclaims_a_dead_recorded_owner(tmp_path, monkeypatch):
    lock_path = tmp_path / ".facetorch-download.lock"
    lock_path.mkdir()
    (lock_path / "owner.json").write_text(
        json.dumps({"pid": 12345, "created": 0}), encoding="utf-8"
    )
    monkeypatch.setattr(_DirectoryLock, "_process_exists", staticmethod(lambda _pid: False))

    with _DirectoryLock(lock_path, timeout=0.1):
        owner = json.loads((lock_path / "owner.json").read_text(encoding="utf-8"))
        assert owner["pid"] > 0

    assert not lock_path.exists()


@pytest.mark.unit
@pytest.mark.downloader
def test_directory_lock_timeout_has_an_actionable_error(tmp_path, monkeypatch):
    lock_path = tmp_path / ".facetorch-download.lock"
    lock_path.mkdir()
    (lock_path / "owner.json").write_text(
        json.dumps({"pid": 12345, "created": time.time()}), encoding="utf-8"
    )
    monkeypatch.setattr(_DirectoryLock, "_process_exists", staticmethod(lambda _pid: True))

    with pytest.raises(CacheLockError, match="Confirm no facetorch process"):
        with _DirectoryLock(lock_path, timeout=0.01):
            pass


@pytest.mark.unit
@pytest.mark.downloader
def test_atomic_promotion_renames_same_filesystem_candidate(tmp_path):
    candidate = _make_export(tmp_path / "candidate.pt2")
    descriptor = _manifest(candidate).descriptor("toy-torch2.11")
    target = tmp_path / descriptor.filename

    with patch("facetorch.downloader.shutil.copyfileobj") as copy:
        _atomic_promote(candidate, target, descriptor)

    copy.assert_not_called()
    assert target.is_file()
    assert not candidate.exists()
