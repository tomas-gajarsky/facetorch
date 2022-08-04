import os

import facetorch
import pytest
import torch


@pytest.mark.integration
@pytest.mark.detector
@pytest.mark.downloader
def test_downloader_run(analyzer):
    if os.path.exists(analyzer.detector.downloader.path_local):
        os.remove(analyzer.detector.downloader.path_local)
    analyzer.detector.downloader.run()
    assert os.path.exists(analyzer.detector.downloader.path_local)
    os.remove(analyzer.detector.downloader.path_local)


@pytest.mark.integration
@pytest.mark.detector
@pytest.mark.downloader
def test_downloader_base_type(analyzer):
    assert isinstance(analyzer.detector.downloader, facetorch.base.BaseDownloader)


@pytest.mark.integration
@pytest.mark.detector
def test_base_type(analyzer):
    assert isinstance(analyzer.detector, facetorch.base.BaseModel)


@pytest.mark.integration
@pytest.mark.detector
def test_type(analyzer):
    assert isinstance(analyzer.detector, facetorch.analyzer.detector.FaceDetector)


@pytest.mark.integration
@pytest.mark.detector
def test_model_type(analyzer):
    assert isinstance(analyzer.detector.model, torch.jit.ScriptModule)


@pytest.mark.integration
@pytest.mark.detector
def test_preprocessor_base_type(analyzer):
    assert isinstance(analyzer.detector.preprocessor, facetorch.base.BaseProcessor)


@pytest.mark.integration
@pytest.mark.detector
def test_preprocessor_base_2_type(analyzer):
    assert isinstance(
        analyzer.detector.preprocessor,
        facetorch.analyzer.detector.pre.BaseDetPreProcessor,
    )


@pytest.mark.integration
@pytest.mark.detector
def test_postprocessor_base_type(analyzer):
    assert isinstance(analyzer.detector.postprocessor, facetorch.base.BaseProcessor)


@pytest.mark.integration
@pytest.mark.detector
def test_postprocessor_base_2_type(analyzer):
    assert isinstance(
        analyzer.detector.postprocessor,
        facetorch.analyzer.detector.post.BaseDetPostProcessor,
    )


@pytest.mark.endtoend
@pytest.mark.detector
def test_face_locations_larger_or_equal_zero(response):
    for face in response.faces:
        assert face.loc.x1 >= 0
        assert face.loc.y1 >= 0
        assert face.loc.x2 >= 0
        assert face.loc.y2 >= 0


@pytest.mark.endtoend
@pytest.mark.detector
def test_number_of_faces(response, cfg):
    if "test.jpg" in cfg.path_image:
        assert len(response.faces) == 4
    elif "test2.jpg" in cfg.path_image:
        assert len(response.faces) == 1
    elif "test3.jpg" in cfg.path_image:
        assert len(response.faces) == 25
    elif "test4.jpg" in cfg.path_image:
        assert len(response.faces) == 2
    elif "test5.jpg" in cfg.path_image:
        assert len(response.faces) == 0
