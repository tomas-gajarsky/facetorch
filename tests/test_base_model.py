import os
import torch
import pytest
from unittest.mock import MagicMock

from facetorch.base import BaseModel


class ConcreteModel(BaseModel):
    """Minimal concrete subclass for testing BaseModel loading."""

    def run(self, *args, **kwargs):
        return self.inference(*args, **kwargs)


def _make_dummy_downloader(path_local):
    dl = MagicMock()
    dl.path_local = path_local
    return dl


DETECTOR_PT = "/opt/facetorch/models_local/detector.pt"
DETECTOR_PTH = "/opt/facetorch/models_local/state_dicts/detector.pth"
DETECTOR_CLASS = "model_defs.detector_model.RetinaFaceResNet50"


@pytest.mark.unit
@pytest.mark.model
class TestLoadNativeModel:

    @pytest.mark.skipif(
        not os.path.exists(DETECTOR_PT), reason="detector.pt not available"
    )
    def test_native_from_torchscript(self):
        dl = _make_dummy_downloader(DETECTOR_PT)
        m = ConcreteModel(
            downloader=dl,
            device=torch.device("cpu"),
            native_model_class=DETECTOR_CLASS,
        )
        assert m.model is not None
        x = torch.randn(1, 3, 480, 640)
        out = m.run(x)
        assert isinstance(out, tuple)
        assert len(out) == 3

    @pytest.mark.skipif(
        not os.path.exists(DETECTOR_PTH), reason="detector.pth not available"
    )
    def test_native_from_state_dict(self):
        dl = _make_dummy_downloader(DETECTOR_PTH)
        m = ConcreteModel(
            downloader=dl,
            device=torch.device("cpu"),
            native_model_class=DETECTOR_CLASS,
        )
        assert m.model is not None
        x = torch.randn(1, 3, 480, 640)
        out = m.run(x)
        assert isinstance(out, tuple)
        assert len(out) == 3


@pytest.mark.unit
@pytest.mark.model
class TestLoadExportedModel:

    def test_exported_model_bad_file(self, tmp_path):
        bad_pt2 = str(tmp_path / "bad.pt2")
        with open(bad_pt2, "wb") as f:
            f.write(b"not a real model")
        dl = _make_dummy_downloader(bad_pt2)
        with pytest.raises(Exception):
            ConcreteModel(downloader=dl, device=torch.device("cpu"))


@pytest.mark.unit
@pytest.mark.model
class TestBaseModelMisc:

    @pytest.mark.skipif(
        not os.path.exists(DETECTOR_PT), reason="detector.pt not available"
    )
    def test_callable(self):
        dl = _make_dummy_downloader(DETECTOR_PT)
        m = ConcreteModel(downloader=dl, device=torch.device("cpu"))
        x = torch.randn(1, 3, 480, 640)
        out = m(x)
        assert isinstance(out, tuple)

    @pytest.mark.skipif(
        not os.path.exists(DETECTOR_PT), reason="detector.pt not available"
    )
    def test_inference_moves_tensor_to_device(self):
        dl = _make_dummy_downloader(DETECTOR_PT)
        m = ConcreteModel(downloader=dl, device=torch.device("cpu"))
        x = torch.randn(1, 3, 480, 640)
        out = m.inference(x)
        assert isinstance(out, tuple)
