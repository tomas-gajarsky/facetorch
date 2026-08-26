import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
import torch
import yaml
from hydra.utils import instantiate
from omegaconf import OmegaConf

import facetorch
from facetorch.analyzer.core import FaceAnalyzer
from facetorch.analyzer.utilizer.save import ImageSaver
from facetorch.artifacts import get_model_manifest
from facetorch.datastruct import ImageData
from facetorch.downloader import DownloaderGDrive
from facetorch.exceptions import ConfigurationError, OfflineCacheError
from facetorch.logger import LoggerJsonFile
from facetorch.paths import _default_cache_dir


REPO_ROOT = Path(__file__).resolve().parents[1]


def _nested_strings(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from _nested_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _nested_strings(child)
    elif isinstance(value, str):
        yield value


@pytest.mark.release_blocker
def test_packaged_profiles_compose_from_read_only_empty_working_directory(
    tmp_path, monkeypatch
):
    cache_dir = tmp_path / "not-created-cache"
    empty_cwd = tmp_path / "empty-read-only"
    empty_cwd.mkdir()
    empty_cwd.chmod(0o555)
    monkeypatch.delenv("FACETORCH_MODEL_DIR", raising=False)
    monkeypatch.delenv("FACETORCH_METADATA_DIR", raising=False)
    monkeypatch.setenv("FACETORCH_CACHE_DIR", str(cache_dir))

    try:
        monkeypatch.chdir(empty_cwd)
        cpu = facetorch.load_config()
        gpu = facetorch.load_config("gpu")
    finally:
        empty_cwd.chmod(0o755)

    assert cpu.analyzer.device == "cpu"
    assert gpu.analyzer.device == "cuda"
    assert cpu.analyzer.logger.path_file is None
    assert cpu.path_output is None
    assert Path(cpu.analyzer.detector.downloader.path_local).is_relative_to(cache_dir)
    assert Path(
        cpu.analyzer.utilizer.align.downloader_meta.path_local
    ).is_relative_to(cache_dir)
    assert not cache_dir.exists()
    assert not any(
        value.startswith("/opt/")
        for value in _nested_strings(OmegaConf.to_container(cpu, resolve=True))
    )


@pytest.mark.release_blocker
def test_analyzer_initialization_does_not_create_portable_cache(tmp_path, monkeypatch):
    cache_dir = tmp_path / "lazy-cache"
    monkeypatch.delenv("FACETORCH_MODEL_DIR", raising=False)
    monkeypatch.delenv("FACETORCH_METADATA_DIR", raising=False)
    monkeypatch.setenv("FACETORCH_CACHE_DIR", str(cache_dir))

    analyzer = FaceAnalyzer(facetorch.load_config().analyzer)

    assert analyzer.detector_loaded is False
    assert analyzer.loaded_predictors == ()
    assert analyzer.loaded_utilizers == ()
    assert not cache_dir.exists()


@pytest.mark.release_blocker
def test_packaged_loader_applies_group_scalar_and_device_overrides():
    cfg = facetorch.load_config(
        overrides=[
            "analyzer.device=cuda",
            "analyzer.optimize_transforms=false",
            "analyzer/predictor/fer=efficientnet_b0_7",
        ]
    )

    assert cfg.analyzer.device == "cuda"
    assert cfg.analyzer.optimize_transforms is False
    assert "efficientnet-b0" in cfg.analyzer.predictor.fer.downloader.repo_id


@pytest.mark.release_blocker
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"profile": "tpu"}, "Unknown configuration profile"),
        ({"overrides": "analyzer.device=cpu"}, "sequence"),
        ({"overrides": [""]}, "non-empty"),
    ],
)
def test_packaged_loader_rejects_ambiguous_options(kwargs, message):
    with pytest.raises(ConfigurationError, match=message):
        facetorch.load_config(**kwargs)


@pytest.mark.release_blocker
def test_external_hydra_tree_is_composed_from_explicit_file(tmp_path):
    config_root = tmp_path / "deployment-config"
    analyzer_group = config_root / "analyzer"
    analyzer_group.mkdir(parents=True)
    (config_root / "application.yaml").write_text(
        "defaults:\n  - analyzer: base\n  - _self_\nmarker: external\n",
        encoding="utf-8",
    )
    (analyzer_group / "base.yaml").write_text(
        "device: cpu\nmode: external-default\n",
        encoding="utf-8",
    )

    cfg = facetorch.load_config_from_path(
        config_root / "application.yaml",
        overrides=["analyzer.device=cuda"],
    )

    assert cfg.marker == "external"
    assert cfg.analyzer.device == "cuda"
    assert cfg.analyzer.mode == "external-default"


@pytest.mark.release_blocker
@pytest.mark.parametrize("path", ["missing.yaml", "config.json"])
def test_external_config_path_errors_are_actionable(path, tmp_path):
    candidate = tmp_path / path
    if candidate.suffix == ".json":
        candidate.write_text("{}", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="configuration"):
        facetorch.load_config_from_path(candidate)


@pytest.mark.release_blocker
def test_packaged_and_source_runtime_configs_remain_synchronized():
    source_root = REPO_ROOT / "conf"
    packaged_root = REPO_ROOT / "facetorch" / "configs"
    relative_paths = [Path("config.yaml")]
    relative_paths.extend(
        path.relative_to(source_root)
        for path in sorted((source_root / "analyzer").rglob("*.yaml"))
    )

    mismatches = [
        os.fspath(relative)
        for relative in relative_paths
        if (source_root / relative).read_text(encoding="utf-8").splitlines()
        != (packaged_root / relative).read_text(encoding="utf-8").splitlines()
    ]
    assert mismatches == []


@pytest.mark.release_blocker
def test_default_detector_is_a_nonduplicating_hugging_face_alias():
    detector_root = REPO_ROOT / "facetorch" / "configs" / "analyzer" / "detector"
    assert (detector_root / "retinaface.yaml").read_bytes() != (
        detector_root / "retinaface_hf.yaml"
    ).read_bytes()

    detector = facetorch.load_config(offline=True).analyzer.detector
    assert detector._target_ == "facetorch.analyzer.detector.FaceDetector"
    assert detector.downloader.manifest_id == "detector-retinaface"


@pytest.mark.release_blocker
def test_manifest_bound_packaged_config_instantiates_and_checks_offline_cache(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("FACETORCH_MODEL_DIR", str(tmp_path / "models"))
    config = facetorch.load_config(offline=True)
    downloader = instantiate(
        config.analyzer.detector.downloader,
        torch_version="2.11.0",
    )
    expected = get_model_manifest().candidates(
        "detector-retinaface",
        torch_version="2.11.0",
        device="cpu",
        allow_legacy_models=False,
    )[0]
    resolved = downloader._resolve_candidates()[0]

    assert downloader.filename.endswith("model.pt2")
    assert downloader.sha256 is None
    assert downloader.size_bytes is None
    assert resolved == expected
    assert resolved.sha256 == expected.sha256
    assert resolved.size_bytes == expected.size_bytes

    target = resolved.cache_path(downloader.path_local)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"tampered cache entry")
    with pytest.raises(OfflineCacheError, match="verified artifact"):
        downloader.run()
    assert not target.exists()
    assert len(list(target.parent.glob(f"{target.name}.quarantine.*"))) == 1


@pytest.mark.release_blocker
def test_packaged_config_is_composable_from_a_zip_import(tmp_path):
    archive_path = tmp_path / "facetorch-source.zip"
    package_root = REPO_ROOT / "facetorch"
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        directories = {
            path.parent.relative_to(REPO_ROOT).as_posix() + "/"
            for path in package_root.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        }
        for directory in sorted(directories):
            archive.writestr(directory, b"")
        for path in sorted(package_root.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.suffix not in {".py", ".yaml"}:
                continue
            archive.write(path, path.relative_to(REPO_ROOT).as_posix())

    empty_cwd = tmp_path / "zip-cwd"
    empty_cwd.mkdir()
    smoke = """
from pathlib import Path
import facetorch

assert '.zip/' in facetorch.__file__.replace('\\\\', '/')
cfg = facetorch.load_config('gpu')
assert cfg.analyzer.device == 'cuda'
assert cfg.analyzer.predictor.fer is not None
assert Path(cfg.analyzer.detector.downloader.path_local).is_absolute()
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(archive_path)
    result = subprocess.run(
        [sys.executable, "-c", smoke],
        cwd=empty_cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.release_blocker
def test_cache_environment_overrides_are_explicit_and_versioned(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache-root"
    model_root = tmp_path / "models-managed"
    metadata_root = tmp_path / "metadata-managed"
    monkeypatch.delenv("FACETORCH_MODEL_DIR", raising=False)
    monkeypatch.delenv("FACETORCH_METADATA_DIR", raising=False)
    monkeypatch.setenv("FACETORCH_CACHE_DIR", str(cache_root))

    assert facetorch.get_cache_dir() == cache_root
    assert facetorch.get_model_dir() == cache_root / "models" / "v1"
    assert facetorch.get_metadata_dir() == cache_root / "metadata" / "v1"

    monkeypatch.setenv("FACETORCH_MODEL_DIR", str(model_root))
    monkeypatch.setenv("FACETORCH_METADATA_DIR", str(metadata_root))
    assert facetorch.get_model_dir() == model_root
    assert facetorch.get_metadata_dir() == metadata_root


@pytest.mark.release_blocker
def test_default_cache_locations_follow_operating_system_conventions():
    home = Path("/users/facetorch")

    assert _default_cache_dir(
        environ={"XDG_CACHE_HOME": "/xdg/cache"},
        platform="linux",
        home=home,
    ) == Path("/xdg/cache/facetorch")
    assert _default_cache_dir(
        environ={}, platform="darwin", home=home
    ) == home / "Library" / "Caches" / "facetorch"
    assert _default_cache_dir(
        environ={"LOCALAPPDATA": "/local/appdata"},
        platform="win32",
        home=home,
    ) == Path("/local/appdata/facetorch/Cache")


@pytest.mark.release_blocker
def test_cpu_and_gpu_compose_services_mount_explicit_separate_caches():
    compose_config = yaml.safe_load(
        (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    )

    for service_name, volume_name in (
        ("facetorch", "facetorch-cache-cpu"),
        ("facetorch-gpu", "facetorch-cache-gpu"),
        ("facetorch-gpu-no-device", "facetorch-cache-gpu"),
    ):
        service = compose_config["services"][service_name]
        assert service["environment"]["FACETORCH_CACHE_DIR"] == (
            "/var/cache/facetorch"
        )
        assert (
            f"{volume_name}:/var/cache/facetorch" in service["volumes"]
        )
        assert "./data/input:/workspace/data/input:ro" in service["volumes"]
        assert "facetorch-output:/workspace/data/output" in service["volumes"]
        assert "entrypoint" not in service

    assert set(compose_config["volumes"]) == {
        "facetorch-cache-cpu",
        "facetorch-cache-gpu",
        "facetorch-output",
    }


@pytest.mark.release_blocker
def test_unwritable_model_cache_error_names_the_override(tmp_path):
    downloader = DownloaderGDrive(
        file_id="probe",
        path_local=str(tmp_path / "denied" / "model.pt"),
    )

    with patch("facetorch.downloader.os.makedirs", side_effect=PermissionError):
        with pytest.raises(ConfigurationError, match="FACETORCH_CACHE_DIR"):
            downloader.run()


@pytest.mark.release_blocker
def test_unwritable_log_and_output_errors_are_actionable(tmp_path):
    log_path = tmp_path / "denied-log" / "facetorch.log"
    with patch("facetorch.logger.os.makedirs", side_effect=PermissionError):
        with pytest.raises(ConfigurationError, match="log directory"):
            LoggerJsonFile(path_file=str(log_path))

    saver = ImageSaver(None, torch.device("cpu"), False)
    data = ImageData(
        path_output=str(tmp_path / "denied-output" / "face.png"),
        img=torch.zeros((3, 4, 5), dtype=torch.uint8),
    )
    with patch(
        "facetorch.analyzer.utilizer.save.os.makedirs",
        side_effect=PermissionError,
    ):
        with pytest.raises(ConfigurationError, match="image output directory"):
            saver.run(data)
