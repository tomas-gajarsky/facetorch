import facetorch
import pytest
import torch


@pytest.mark.integration
@pytest.mark.reader
def test_base_type(analyzer):
    assert isinstance(analyzer.reader, facetorch.base.BaseReader)


@pytest.mark.reader
def test_output_shape_length(cfg, analyzer):
    if cfg.path_image is None:
        pytest.skip("No image path provided in config.")
    if hasattr(cfg, "path_tensor"):
        pytest.skip("Only test.jpg is used for this test.")
    data = analyzer.reader.run(cfg.path_image)
    assert len(data.tensor.shape) == 4


@pytest.mark.reader
def test_output_shape_batch_channel(cfg, analyzer):
    if cfg.path_image is None:
        pytest.skip("No image path provided in config.")
    if hasattr(cfg, "path_tensor"):
        pytest.skip("Only test.jpg is used for this test.")
    data = analyzer.reader.run(cfg.path_image)
    assert data.tensor.shape[:2] == torch.Size([1, 3])


@pytest.mark.reader
def test_output_shape_length_with_tensor_input(cfg, analyzer, tensor):
    if not hasattr(cfg, "path_tensor"):
        pytest.skip("No tensor path provided in config.")
    data = analyzer.reader.run(tensor)
    assert len(data.tensor.shape) == 4


@pytest.mark.reader
def test_output_shape_batch_channel_with_tensor_input(cfg, analyzer, tensor):
    if not hasattr(cfg, "path_tensor"):
        pytest.skip("No tensor path provided in config.")
    data = analyzer.reader.run(tensor)
    assert data.tensor.shape[:2] == torch.Size([1, 3])


@pytest.mark.reader
def test_output_type(cfg, analyzer, tensor):
    if not hasattr(cfg, "path_tensor"):
        pytest.skip("No tensor path provided in config.")
    data = analyzer.reader.run(tensor)
    assert data.tensor.dtype == torch.float32
    assert data.img.dtype == torch.uint8
