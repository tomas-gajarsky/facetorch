import sys
import os
from os.path import abspath
from os.path import dirname as d
from pathlib import Path
from unittest.mock import patch

import torch
import pytest
from hydra import compose, initialize
from omegaconf import DictConfig, ListConfig

root_dir = d(d(abspath(__file__)))
# Test the checkout deterministically, even when an older facetorch wheel is
# already installed and pytest was launched through its console script.
if root_dir in sys.path:
    sys.path.remove(root_dir)
sys.path.insert(0, root_dir)

from facetorch import FaceAnalyzer  # noqa: E402
from facetorch.datastruct import AnalysisResult  # noqa: E402
from facetorch.analyzer.reader import (  # noqa: E402
    ImageReader,
    TensorReader,
    UniversalReader,
)

DEFAULT_TEST_ROOT = "/opt/facetorch"
DEFAULT_LOG_ROOT = "/opt/logs"
REPO_ROOT = Path(root_dir).resolve()
TEST_MODEL_ROOT_CONFIGURED = "FACETORCH_TEST_MODEL_ROOT" in os.environ
TEST_MODEL_ROOT = Path(
    os.environ.get(
        "FACETORCH_TEST_MODEL_ROOT",
        str(REPO_ROOT / ".pytest_cache" / "facetorch-models"),
    )
).resolve()


def _rewrite_default_root_paths(node) -> None:
    """Make test configs portable outside the Docker /opt/facetorch layout."""
    if str(REPO_ROOT) == DEFAULT_TEST_ROOT and not TEST_MODEL_ROOT_CONFIGURED:
        return

    if isinstance(node, DictConfig):
        for key in list(node.keys()):
            value = node[key]
            if isinstance(value, (DictConfig, ListConfig)):
                _rewrite_default_root_paths(value)
            elif isinstance(value, str) and _is_default_container_path(value):
                node[key] = _rewrite_default_root_path(value)
            elif key == "path_local" and isinstance(value, str):
                # Freeze portable resolver values while the isolated test cache
                # environment is active; do not leak into a user's normal cache.
                node[key] = value
        return

    if isinstance(node, ListConfig):
        for idx, value in enumerate(node):
            if isinstance(value, (DictConfig, ListConfig)):
                _rewrite_default_root_paths(value)
            elif isinstance(value, str) and _is_default_container_path(value):
                node[idx] = _rewrite_default_root_path(value)


def _is_default_container_path(value: str) -> bool:
    return value.startswith(DEFAULT_TEST_ROOT + "/") or value.startswith(
        DEFAULT_LOG_ROOT + "/"
    )


def _rewrite_default_root_path(value: str) -> str:
    if value.startswith(DEFAULT_LOG_ROOT + "/"):
        suffix = value[len(DEFAULT_LOG_ROOT) + 1 :]
        return str(REPO_ROOT / ".pytest_cache" / "logs" / suffix)
    models_prefix = DEFAULT_TEST_ROOT + "/models"
    if value.startswith(models_prefix + "/"):
        return str(TEST_MODEL_ROOT) + value[len(models_prefix):]
    return str(REPO_ROOT) + value[len(DEFAULT_TEST_ROOT):]


def pytest_configure(config):
    """Performs initial configuration of the session. Official docs:
    https://docs.pytest.org/en/stable/reference.html#pytest.hookspec.pytest_configure

    Args:
        config (Config): Config object.

    """
    config.addinivalue_line(
        "markers", "analyzer: mark tests related to the FaceAnalyzer"
    )
    config.addinivalue_line(
        "markers", "callable: mark tests related to __call__ methods"
    )
    config.addinivalue_line("markers", "reader: mark tests related to the BaseReader")
    config.addinivalue_line(
        "markers", "detector: mark tests related to the FaceDetector"
    )
    config.addinivalue_line("markers", "unifier: mark tests related to the FaceUnifier")
    config.addinivalue_line(
        "markers", "predictor: mark tests related to the FacePredictor"
    )
    config.addinivalue_line(
        "markers", "embed: mark tests related to Face Representation Learning Predictor"
    )
    config.addinivalue_line(
        "markers", "verify: mark tests related to Face Verification Predictor"
    )
    config.addinivalue_line(
        "markers", "fer: mark tests related to Facial Expression Recognition Predictor"
    )
    config.addinivalue_line(
        "markers", "au: mark tests related to Facial Action Unit Detection"
    )
    config.addinivalue_line(
        "markers", "va: mark tests related to Facial Valence Arousal Predictor"
    )
    config.addinivalue_line(
        "markers", "deepfake: mark tests related to Deepfake Detection Predictor"
    )
    config.addinivalue_line(
        "markers", "align: mark tests related to Face Alignment Predictor"
    )
    config.addinivalue_line(
        "markers", "utilizer: mark tests related to the BaseUtilizer"
    )
    config.addinivalue_line(
        "markers", "draw: mark tests related to the BoxDrawer utilizer"
    )
    config.addinivalue_line(
        "markers", "save: mark tests related to the ImageSaver utilizer"
    )
    config.addinivalue_line(
        "markers", "downloader: mark tests related to the BaseDownloader"
    )
    config.addinivalue_line("markers", "model: mark tests related to the BaseModel")
    config.addinivalue_line("markers", "response: mark tests related to the ImageData")
    config.addinivalue_line(
        "markers", "transforms: mark tests related to the facetorch transforms"
    )
    config.addinivalue_line(
        "markers", "endtoend: mark tests related to the end-to-end pipeline"
    )
    config.addinivalue_line(
        "markers", "integration: mark tests related to the integration"
    )
    config.addinivalue_line("markers", "performance: mark tests related to performance")
    config.addinivalue_line("markers", "unit: mark tests related to the unit tests")
    config.addinivalue_line(
        "markers",
        "release_blocker: executable v1 release contract or audited-defect regression",
    )
    config.addinivalue_line(
        "markers",
        "upstream_network: opt-in verification against immutable upstream sources",
    )


@pytest.fixture(
    scope="session",
    params=[
        "tests.config.1",
        "tests.config.2",
        "tests.config.3",
        "tests.config.4",
        "tests.config.5",
    ],
)
def cfg(request) -> None:
    test_cache_env = {
        "FACETORCH_MODEL_DIR": str(TEST_MODEL_ROOT),
        "FACETORCH_METADATA_DIR": str(TEST_MODEL_ROOT / "metadata"),
    }
    with patch.dict(os.environ, test_cache_env):
        with initialize(version_base=None, config_path="../conf"):
            cfg = compose(config_name=request.param)
        _rewrite_default_root_paths(cfg)
    return cfg


@pytest.fixture(scope="session")
def analyzer(cfg) -> FaceAnalyzer:
    analyzer = FaceAnalyzer(cfg.analyzer)
    return analyzer


@pytest.fixture(scope="session")
def response(cfg, analyzer) -> AnalysisResult:
    if isinstance(analyzer.reader, UniversalReader) or isinstance(
        analyzer.reader, ImageReader
    ):
        response = analyzer.run(
            image_source=cfg.path_image,
            face_batch_size=cfg.batch_size,
            fix_img_size=cfg.fix_img_size,
            include_tensors=cfg.include_tensors,
            path_output=cfg.path_output,
        )
    elif isinstance(analyzer.reader, TensorReader):
        pytest.skip("Do not use tensor for this test.")
    else:
        pytest.skip("No reader provided in config.")
    return response


@pytest.fixture(scope="session")
def tensor(cfg) -> torch.Tensor:
    if hasattr(cfg, "path_tensor"):
        tensor = torch.load(
            cfg.path_tensor,
        ).to(cfg.analyzer.device)
    else:
        pytest.skip("No tensor path provided in config.")
    return tensor
