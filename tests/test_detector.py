import os

import facetorch
import pytest
import torch


@pytest.mark.integration
@pytest.mark.detector
@pytest.mark.downloader
def test_downloader_run(analyzer):
    if not os.path.exists(analyzer.detector.downloader.path_local):
        analyzer.detector.downloader.run()
    assert os.path.exists(analyzer.detector.downloader.path_local)


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


@pytest.mark.integration
@pytest.mark.detector
def test_preprocessor_normalization_order(analyzer):
    dummy_image = torch.ones(1, 3, 224, 224) * 255

    preprocessed_image = analyzer.detector.preprocessor.transform(dummy_image)

    if getattr(analyzer.detector.preprocessor, "reverse_colors", False):
        dummy_image = dummy_image[:, [2, 1, 0], :, :]

    mean = torch.tensor([123.0, 117.0, 104.0]).view(1, 3, 1, 1)
    std = torch.tensor([1.0, 1.0, 1.0]).view(1, 3, 1, 1)
    expected_image = (dummy_image - mean) / std

    assert torch.allclose(preprocessed_image, expected_image, atol=1e-5)


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
