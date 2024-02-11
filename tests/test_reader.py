import facetorch
import pytest
import torch
from PIL import Image
import numpy as np
import io
from facetorch.analyzer.reader import UniversalReader, TensorReader, ImageReader


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
    assert len(data.img.shape) == 3


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


def test_read_image_from_url(analyzer):
    if not isinstance(analyzer.reader, UniversalReader):
        pytest.skip("Only UniversalReader is used for this test.")
    result = analyzer.reader.run(
        "https://github.com/tomas-gajarsky/facetorch/blob/main/data/input/test.jpg?raw=true"
    )
    assert isinstance(result, facetorch.datastruct.ImageData)
    assert result.img is not None
    assert result.tensor is not None


def test_read_image_from_path(cfg, analyzer):
    if not isinstance(analyzer.reader, UniversalReader) or not isinstance(
        analyzer.reader, ImageReader
    ):
        pytest.skip("Only UniversalReader and ImageReader are used for this test.")
    result = analyzer.reader.run(cfg.path_image)
    assert isinstance(result, facetorch.datastruct.ImageData)
    assert result.img is not None
    assert result.tensor is not None


def test_read_tensor(analyzer):
    if not isinstance(analyzer.reader, UniversalReader) or not isinstance(
        analyzer.reader, TensorReader
    ):
        pytest.skip("Only UniversalReader and TensorReader are used for this test.")
    tensor_input = torch.randn(3, 224, 224)
    result = analyzer.reader.run(tensor_input)
    assert isinstance(result, facetorch.datastruct.ImageData)
    assert result.tensor.size() == torch.Size([1, 3, 224, 224])


def test_read_numpy_array(analyzer):
    if not isinstance(analyzer.reader, UniversalReader):
        pytest.skip("Only UniversalReader is used for this test.")
    array_input = np.random.rand(224, 224, 3).astype(np.float32)
    result = analyzer.reader.run(array_input)
    assert isinstance(result, facetorch.datastruct.ImageData)
    assert result.tensor is not None
    assert result.img is not None
    assert result.tensor.size() == torch.Size([1, 3, 224, 224])


def test_read_image_from_bytes(analyzer):
    if not isinstance(analyzer.reader, UniversalReader):
        pytest.skip("Only UniversalReader is used for this test.")
    img = Image.new("RGB", (60, 30), color="red")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    bytes_input = img_byte_arr.getvalue()
    result = analyzer.reader.run(bytes_input)
    assert isinstance(result, facetorch.datastruct.ImageData)
    assert result.tensor is not None
    assert result.img is not None
    assert result.tensor.size() == torch.Size([1, 3, 30, 60])


def test_read_pil_image(analyzer):
    if not isinstance(analyzer.reader, UniversalReader):
        pytest.skip("Only UniversalReader is used for this test.")
    pil_input = Image.new("RGB", (60, 30), color="red")
    result = analyzer.reader.run(pil_input)
    assert isinstance(result, facetorch.datastruct.ImageData)
    assert result.tensor is not None


def test_unsupported_data_type(analyzer):
    if not isinstance(analyzer.reader, UniversalReader):
        pytest.skip("Only UniversalReader is used for this test.")
    with pytest.raises(ValueError):
        analyzer.reader.run(123)  # Passing an integer to trigger the error
