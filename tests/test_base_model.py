import os
import torch
import pytest
from unittest.mock import MagicMock, patch

from facetorch.base import BaseModel


class ConcreteModel(BaseModel):
    """Minimal concrete subclass for testing BaseModel loading."""

    def run(self, *args, **kwargs):
        return self.inference(*args, **kwargs)


class _FakeModule(torch.nn.Module):
    """Minimal nn.Module for mock-based tests."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


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
class TestLoadNativeModelMocked:

    def test_native_from_pth_mocked(self, tmp_path):
        """Cover _load_native_model .pth branch with mocked imports."""
        fake = _FakeModule()
        pth_file = str(tmp_path / "model.pth")
        torch.save(fake.state_dict(), pth_file)

        dl = _make_dummy_downloader(pth_file)
        fake_module = MagicMock()
        fake_module._FakeModule = _FakeModule

        with patch("importlib.import_module", return_value=fake_module):
            m = ConcreteModel(
                downloader=dl,
                device=torch.device("cpu"),
                native_model_class="tests.test_base_model._FakeModule",
            )
        assert m.model is not None
        x = torch.randn(1, 4)
        out = m.run(x)
        assert out.shape == (1, 2)

    def test_native_from_torchscript_mocked(self, tmp_path):
        """Cover _load_native_model .pt branch with mocked TorchScript."""
        fake = _FakeModule()
        scripted = torch.jit.script(fake)
        pt_file = str(tmp_path / "model.pt")
        torch.jit.save(scripted, pt_file)

        dl = _make_dummy_downloader(pt_file)
        fake_module = MagicMock()
        fake_module._FakeModule = _FakeModule

        with patch("importlib.import_module", return_value=fake_module):
            m = ConcreteModel(
                downloader=dl,
                device=torch.device("cpu"),
                native_model_class="tests.test_base_model._FakeModule",
            )
        assert m.model is not None
        x = torch.randn(1, 4)
        out = m.run(x)
        assert out.shape == (1, 2)

    def test_torchscript_fallback(self, tmp_path):
        """Cover .pt without native_model_class (legacy TorchScript path)."""
        fake = _FakeModule()
        scripted = torch.jit.script(fake)
        pt_file = str(tmp_path / "model.pt")
        torch.jit.save(scripted, pt_file)

        dl = _make_dummy_downloader(pt_file)
        m = ConcreteModel(downloader=dl, device=torch.device("cpu"))
        assert m.model is not None
        x = torch.randn(1, 4)
        out = m.run(x)
        assert out.shape == (1, 2)


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

    def test_exported_model_schema_version_error(self, tmp_path):
        """Cover the schema version error wrapping in _load_exported_model."""
        bad_pt2 = str(tmp_path / "model.pt2")
        with open(bad_pt2, "wb") as f:
            f.write(b"not a real model")
        dl = _make_dummy_downloader(bad_pt2)

        with patch("torch.export.load", side_effect=RuntimeError("schema version mismatch")):
            with pytest.raises(RuntimeError, match="different PyTorch version"):
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

    def test_callable_mocked(self, tmp_path):
        """Cover __call__ path without real model files."""
        fake = _FakeModule()
        scripted = torch.jit.script(fake)
        pt_file = str(tmp_path / "model.pt")
        torch.jit.save(scripted, pt_file)

        dl = _make_dummy_downloader(pt_file)
        m = ConcreteModel(downloader=dl, device=torch.device("cpu"))
        x = torch.randn(1, 4)
        out = m(x)
        assert out.shape == (1, 2)

    def test_downloader_called_when_file_missing(self, tmp_path):
        """Cover the download-on-missing path in load_model."""
        missing_file = str(tmp_path / "subdir" / "model.pt")
        fake = _FakeModule()
        scripted = torch.jit.script(fake)

        def fake_download():
            os.makedirs(os.path.dirname(missing_file), exist_ok=True)
            torch.jit.save(scripted, missing_file)

        dl = _make_dummy_downloader(missing_file)
        dl.run = fake_download

        m = ConcreteModel(downloader=dl, device=torch.device("cpu"))
        assert m.model is not None
