import os

import facetorch
import pytest
import torch


@pytest.mark.integration
@pytest.mark.predictor
@pytest.mark.downloader
def test_downloaders_run(analyzer):
    for predictor in analyzer.predictors.values():
        if os.path.exists(predictor.downloader.path_local):
            os.remove(predictor.downloader.path_local)
        predictor.downloader.run()
        assert os.path.exists(predictor.downloader.path_local)


@pytest.mark.integration
@pytest.mark.predictor
@pytest.mark.downloader
def test_downloader_base_types(analyzer):
    for predictor in analyzer.predictors.values():
        assert isinstance(predictor.downloader, facetorch.base.BaseDownloader)


@pytest.mark.integration
@pytest.mark.predictor
def test_base_types(analyzer):
    for predictor in analyzer.predictors.values():
        assert isinstance(predictor, facetorch.base.BaseModel)


@pytest.mark.integration
@pytest.mark.predictor
def test_types(analyzer):
    for predictor in analyzer.predictors.values():
        assert isinstance(predictor, facetorch.analyzer.predictor.FacePredictor)


@pytest.mark.integration
@pytest.mark.predictor
def test_model_types(analyzer):
    for predictor in analyzer.predictors.values():
        assert isinstance(predictor.model, torch.jit.ScriptModule)


@pytest.mark.integration
@pytest.mark.predictor
def test_preprocessor_base_types(analyzer):
    for predictor in analyzer.predictors.values():
        assert isinstance(predictor.preprocessor, facetorch.base.BaseProcessor)


@pytest.mark.integration
@pytest.mark.predictor
def test_preprocessor_base_2_types(analyzer):
    for predictor in analyzer.predictors.values():
        assert isinstance(
            predictor.preprocessor,
            facetorch.analyzer.predictor.pre.BasePredPreProcessor,
        )


@pytest.mark.integration
@pytest.mark.predictor
def test_postprocessor_base_types(analyzer):
    for predictor in analyzer.predictors.values():
        assert isinstance(predictor.postprocessor, facetorch.base.BaseProcessor)


@pytest.mark.integration
@pytest.mark.predictor
def test_postprocessor_base_2_types(analyzer):
    for predictor in analyzer.predictors.values():
        assert isinstance(
            predictor.postprocessor,
            facetorch.analyzer.predictor.post.BasePredPostProcessor,
        )


@pytest.mark.endtoend
@pytest.mark.predictor
def test_response_preds_length_match(analyzer, response, cfg):
    if "test5.jpg" in cfg.path_image:
        pytest.skip("test5.jpg has no faces")

    assert len(analyzer.predictors) == len(response.faces[0].preds)
