import facetorch
import pytest
from facetorch.datastruct import Location


@pytest.mark.unit
@pytest.mark.response
def test_location_expand():
    loc = Location(x1=100, y1=100, x2=200, y2=200)
    loc.expand(0.5)
    assert loc.x1 < 100
    assert loc.y1 < 100
    assert loc.x2 > 200
    assert loc.y2 > 200


@pytest.mark.unit
@pytest.mark.response
def test_location_form_square_noop():
    loc = Location(x1=0, y1=0, x2=100, y2=100)
    loc.form_square()
    assert loc.x1 == 0
    assert loc.y1 == 0


@pytest.mark.integration
@pytest.mark.response
def test_type(response):
    assert isinstance(response, facetorch.datastruct.AnalysisResult)


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
