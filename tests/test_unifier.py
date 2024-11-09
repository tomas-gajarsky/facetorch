import facetorch
import pytest


@pytest.mark.integration
@pytest.mark.unifier
def test_base_type(analyzer):
    assert isinstance(analyzer.unifier, facetorch.base.BaseProcessor)


@pytest.mark.endtoend
@pytest.mark.unifier
def test_face_crops_in_range(response):
    for face in response.faces:
        face_crop = face.tensor
        if face_crop.nelement() == 0:
            continue
        assert face_crop.dim() == 3, "face_crop should be a 3D tensor [C, H, W]"
        assert face_crop.size(0) == 3, "Channel dimension should be 3 (RGB)"
        assert (
            face_crop.size(1) > 0 and face_crop.size(2) > 0
        ), "Height and width must be positive"
        assert face_crop.min() >= 0.0
        assert face_crop.max() <= 1.0
