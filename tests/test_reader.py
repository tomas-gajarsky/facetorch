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
    with pytest.raises(facetorch.InputError, match="URLReader"):
        analyzer.reader.run("https://example.invalid/image.jpg")


def test_read_image_from_path(cfg, analyzer):
    if not isinstance(analyzer.reader, (UniversalReader, ImageReader)):
        pytest.skip("Only UniversalReader and ImageReader are used for this test.")
    result = analyzer.reader.run(cfg.path_image)
    assert isinstance(result, facetorch.datastruct.ImageData)
    assert result.img is not None
    assert result.tensor is not None


def test_read_tensor(analyzer):
    if not isinstance(analyzer.reader, (UniversalReader, TensorReader)):
        pytest.skip("Only UniversalReader and TensorReader are used for this test.")
    tensor_input = torch.rand(3, 224, 224)
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


def test_read_grayscale_pil_image(analyzer):
    if not isinstance(analyzer.reader, UniversalReader):
        pytest.skip("Only UniversalReader is used for this test.")
    pil_image = Image.new("L", (60, 30))
    result = analyzer.reader.run(pil_image)
    assert isinstance(result, facetorch.datastruct.ImageData)
    assert result.tensor is not None
    assert result.tensor.size(1) == 3


def test_read_grayscale_image_from_bytes(analyzer):
    if not isinstance(analyzer.reader, UniversalReader):
        pytest.skip("Only UniversalReader is used for this test.")
    pil_image = Image.new("L", (60, 30))
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format="JPEG")
    bytes_input = img_byte_arr.getvalue()
    result = analyzer.reader.run(bytes_input)
    assert isinstance(result, facetorch.datastruct.ImageData)
    assert result.tensor is not None
    assert result.tensor.size(1) == 3


def test_read_rgba_pil_image(analyzer):
    if not isinstance(analyzer.reader, UniversalReader):
        pytest.skip("Only UniversalReader is used for this test.")
    pil_image = Image.new("RGBA", (60, 30), color=(255, 0, 0, 128))
    result = analyzer.reader.run(pil_image)
    assert isinstance(result, facetorch.datastruct.ImageData)
    assert result.tensor is not None
    assert result.tensor.size(1) == 3


def test_read_rgba_image_from_bytes(analyzer):
    if not isinstance(analyzer.reader, UniversalReader):
        pytest.skip("Only UniversalReader is used for this test.")
    pil_image = Image.new("RGBA", (60, 30), color=(255, 0, 0, 128))
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format="PNG")
    bytes_input = img_byte_arr.getvalue()
    result = analyzer.reader.run(bytes_input)
    assert isinstance(result, facetorch.datastruct.ImageData)
    assert result.tensor is not None
    assert result.tensor.size(1) == 3


def test_read_numpy_array_with_real_image(cfg, analyzer):
    if not isinstance(analyzer.reader, UniversalReader):
        pytest.skip("Only UniversalReader is used for this test.")
    if cfg.path_image is None:
        pytest.skip("No image path provided in config.")
    image = Image.open(cfg.path_image).convert("RGB")
    image_rgb = np.array(image)
    result = analyzer.reader.run(image_rgb)
    assert isinstance(result, facetorch.datastruct.ImageData)
    assert result.tensor is not None
    assert result.img is not None
    assert result.tensor.size(1) == 3


@pytest.mark.reader
def test_read_numpy_array_2d_grayscale(analyzer):
    if not isinstance(analyzer.reader, UniversalReader):
        pytest.skip("Only UniversalReader is used for this test.")
    array_input = np.random.rand(224, 224).astype(np.float32)
    result = analyzer.reader.run(array_input)
    assert isinstance(result, facetorch.datastruct.ImageData)
    assert result.tensor.size() == torch.Size([1, 3, 224, 224])


@pytest.mark.reader
def test_read_numpy_array_hwc1_grayscale(analyzer):
    if not isinstance(analyzer.reader, UniversalReader):
        pytest.skip("Only UniversalReader is used for this test.")
    array_input = np.random.rand(224, 224, 1).astype(np.float32)
    result = analyzer.reader.run(array_input)
    assert isinstance(result, facetorch.datastruct.ImageData)
    assert result.tensor.size() == torch.Size([1, 3, 224, 224])


@pytest.mark.reader
def test_read_numpy_array_rgba(analyzer):
    if not isinstance(analyzer.reader, UniversalReader):
        pytest.skip("Only UniversalReader is used for this test.")
    array_input = np.random.rand(224, 224, 4).astype(np.float32)
    result = analyzer.reader.run(array_input)
    assert isinstance(result, facetorch.datastruct.ImageData)
    assert result.tensor.size() == torch.Size([1, 3, 224, 224])


@pytest.mark.unit
@pytest.mark.reader
def test_process_tensor_grayscale_2d(analyzer):
    tensor_input = torch.rand(224, 224)
    result = analyzer.reader.process_tensor(tensor_input, fix_img_size=False)
    assert result.tensor.size() == torch.Size([1, 3, 224, 224])


@pytest.mark.unit
@pytest.mark.reader
def test_process_tensor_grayscale_1chw(analyzer):
    tensor_input = torch.rand(1, 224, 224)
    result = analyzer.reader.process_tensor(tensor_input, fix_img_size=False)
    assert result.tensor.size() == torch.Size([1, 3, 224, 224])


@pytest.mark.unit
@pytest.mark.reader
def test_process_tensor_rgba(analyzer):
    tensor_input = torch.rand(4, 224, 224)
    result = analyzer.reader.process_tensor(tensor_input, fix_img_size=False)
    assert result.tensor.size() == torch.Size([1, 3, 224, 224])


@pytest.mark.unit
@pytest.mark.reader
def test_process_tensor_hwc_rgb(analyzer):
    tensor_input = torch.rand(224, 224, 3)
    result = analyzer.reader.process_tensor(
        tensor_input,
        fix_img_size=False,
        input_spec=facetorch.InputSpec(layout="HWC"),
    )
    assert result.tensor.size() == torch.Size([1, 3, 224, 224])


@pytest.mark.unit
@pytest.mark.reader
def test_process_tensor_batched_not_supported(analyzer):
    tensor_input = torch.rand(2, 3, 224, 224)
    with pytest.raises(ValueError, match="B=1"):
        analyzer.reader.process_tensor(tensor_input, fix_img_size=False)


@pytest.mark.unit
@pytest.mark.reader
@pytest.mark.release_blocker
def test_process_tensor_chw_width_three_is_supported():
    reader = TensorReader(None, torch.device("cpu"), False)
    tensor_input = torch.rand(3, 224, 3)
    result = reader.process_tensor(tensor_input, fix_img_size=False)
    assert result.tensor.size() == torch.Size([1, 3, 224, 3])
