import pytest


@pytest.mark.integration
@pytest.mark.predictor
@pytest.mark.au
def test_au_in_preds(response):
    for face in response.faces:
        assert "au" in face.preds.keys()


@pytest.mark.endtoend
@pytest.mark.predictor
@pytest.mark.au
def test_lip_pucker(response, cfg):
    if "test.jpg" not in cfg.path_image:
        pytest.skip("Only test.jpg is used for this test.")
    if hasattr(cfg, "path_tensor"):
        if "tensor.pt" in cfg.path_tensor:
            pytest.skip("Only test.jpg is used for this test.")
    assert response.faces[1].preds["au"].label == "lip_pucker"
