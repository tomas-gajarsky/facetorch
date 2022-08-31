import pytest


@pytest.mark.integration
@pytest.mark.predictor
@pytest.mark.verify
def test_embed_in_preds(response):
    for face in response.faces:
        assert "verify" in face.preds.keys()


@pytest.mark.endtoend
@pytest.mark.predictor
@pytest.mark.verify
def test_length(response, cfg):
    if cfg.include_tensors is True:
        pytest.skip("Tensors were not included by the user.")
    if "test.jpg" not in cfg.path_image:
        pytest.skip("Only test.jpg is used for this recognition embedding length test.")
    assert response.faces[0].preds["verify"].logits.shape[0] > 0
