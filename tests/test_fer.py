import pytest


@pytest.mark.integration
@pytest.mark.predictor
@pytest.mark.fer
def test_fer_in_preds(response):
    for face in response.faces:
        assert "fer" in face.preds.keys()


@pytest.mark.endtoend
@pytest.mark.predictor
@pytest.mark.fer
def test_happiness(response, cfg):
    if "test2.jpg" not in cfg.path_image:
        pytest.skip("Ony test2.jpg is used for this happy test.")
    assert response.faces[0].preds["fer"].label == "Happiness"
