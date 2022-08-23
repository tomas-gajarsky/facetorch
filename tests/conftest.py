import sys
from os.path import abspath
from os.path import dirname as d

import pytest
from facetorch import FaceAnalyzer
from facetorch.datastruct import ImageData
from hydra import compose, initialize

root_dir = d(d(abspath(__file__)))
sys.path.append(root_dir)


def pytest_configure(config):
    """Performs initial configuration of the session. Official docs:
    https://docs.pytest.org/en/stable/reference.html#pytest.hookspec.pytest_configure

    Args:
        config (Config): Config object.

    """
    config.addinivalue_line(
        "markers", "analyzer: mark tests related to the FaceAnalyzer"
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
        "markers", "embed: mark tests related to Face Representation Learning embedder"
    )
    config.addinivalue_line(
        "markers", "fer: mark tests related to Facial Expression Recognition Predictor"
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


@pytest.fixture(
    scope="session",
    params=["tests.config.1", "tests.config.2", "tests.config.3", "tests.config.4"],
)
def cfg(request) -> None:
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name=request.param)
    return cfg


@pytest.fixture()
def analyzer(cfg) -> FaceAnalyzer:
    analyzer = FaceAnalyzer(cfg.analyzer)
    return analyzer


@pytest.fixture()
def response(cfg, analyzer) -> ImageData:
    response = analyzer.run(
        path_image=cfg.path_image,
        batch_size=cfg.batch_size,
        fix_img_size=cfg.fix_img_size,
        return_img_data=cfg.return_img_data,
        include_tensors=cfg.include_tensors,
        path_output=cfg.path_output,
    )
    return response
