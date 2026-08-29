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


@pytest.mark.unit
@pytest.mark.response
@pytest.mark.parametrize(
    ("location", "expected"),
    (
        (Location(x1=0, y1=0, x2=10, y2=11), (0, 0, 11, 11)),
        (Location(x1=0, y1=0, x2=11, y2=10), (0, 0, 11, 11)),
        (Location(x1=-5, y1=-7, x2=5, y2=5), (-6, -7, 6, 5)),
        (Location(x1=-7, y1=-5, x2=5, y2=5), (-7, -6, 5, 6)),
        (Location(x1=1, y1=2, x2=9, y2=14), (-1, 2, 11, 14)),
        (Location(x1=1, y1=2, x2=13, y2=10), (1, 0, 13, 12)),
    ),
)
def test_location_form_square_handles_odd_even_and_negative_coordinates(
    location, expected
):
    identity = id(location)

    result = location.form_square()

    assert result is None
    assert id(location) == identity
    assert (location.x1, location.y1, location.x2, location.y2) == expected
    assert location.x2 - location.x1 == location.y2 - location.y1


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
