import pytest


@pytest.mark.integration
@pytest.mark.predictor
@pytest.mark.embed
def test_embed_in_preds(response):
    for face in response.faces:
        assert "embed" in face.preds.keys()


@pytest.mark.endtoend
@pytest.mark.predictor
@pytest.mark.embed
def test_length(response, cfg):
    if cfg.include_tensors is True:
        pytest.skip("Tensors were not included by the user.")
    if "test.jpg" not in cfg.path_image:
        pytest.skip("Only test.jpg is used for this embedding length test.")
    assert response.faces[0].preds["embed"].logits.shape[0] > 0
