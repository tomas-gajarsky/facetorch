import pytest


@pytest.mark.integration
@pytest.mark.predictor
@pytest.mark.deepfake
def test_deepfake_in_preds(response):
    for face in response.faces:
        assert "deepfake" in face.preds.keys()


@pytest.mark.endtoend
@pytest.mark.predictor
@pytest.mark.deepfake
def test_real_face(response, cfg):
    if "test2.jpg" not in cfg.path_image:
        pytest.skip("Only test2.jpg is used for this deepfake test.")
    assert response.faces[0].preds["deepfake"].label == "Real"


@pytest.mark.endtoend
@pytest.mark.predictor
@pytest.mark.deepfake
def test_fake_face(response, cfg):
    if "test4.jpg" not in cfg.path_image:
        pytest.skip("Only test4.jpg is used for this deepfake test.")
    assert response.faces[1].preds["deepfake"].label == "Fake"
