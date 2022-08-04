import facetorch
import pytest


@pytest.mark.integration
@pytest.mark.response
def test_type(response):
    assert isinstance(response, facetorch.datastruct.ImageData) or isinstance(
        response, facetorch.datastruct.Response
    )


@pytest.mark.integration
@pytest.mark.response
def test_location_type(response):
    for face in response.faces:
        assert isinstance(face.loc, facetorch.datastruct.Location)


@pytest.mark.integration
@pytest.mark.response
def test_dims_type(response):
    for face in response.faces:
        assert isinstance(face.dims, facetorch.datastruct.Dimensions)


@pytest.mark.integration
@pytest.mark.response
def test_preds_type(response):
    for face in response.faces:
        assert isinstance(face.preds, dict)


@pytest.mark.integration
@pytest.mark.response
def test_preds_value_type(response):
    for face in response.faces:
        for pred in face.preds.values():
            assert isinstance(pred, facetorch.datastruct.Prediction)
