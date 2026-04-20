import io
import numpy as np
import pytest
import torch
from PIL import Image


@pytest.mark.integration
def test_analyzer_image_source(cfg, analyzer):
    if hasattr(cfg, "path_tensor"):
        pytest.skip("This test is only for path_image.")
    if "test.jpg" not in cfg.path_image:
        pytest.skip("Only test.jpg is used for this test.")
    response = analyzer.run(
        image_source=cfg.path_image,
        batch_size=cfg.batch_size,
        fix_img_size=cfg.fix_img_size,
        return_img_data=cfg.return_img_data,
        include_tensors=cfg.include_tensors,
        path_output=cfg.path_output,
    )

    assert response.tensor.shape[1:] == response.img.shape
    assert response.tensor.dtype == torch.float32
    assert len(response.faces[0].preds.keys()) > 0


@pytest.mark.integration
def test_analyzer_path_image(cfg, analyzer):
    if hasattr(cfg, "path_tensor"):
        pytest.skip("This test is only for path_image.")
    if "test.jpg" not in cfg.path_image:
        pytest.skip("Only test.jpg is used for this test.")
    response = analyzer.run(
        path_image=cfg.path_image,
        batch_size=cfg.batch_size,
        fix_img_size=cfg.fix_img_size,
        return_img_data=cfg.return_img_data,
        include_tensors=cfg.include_tensors,
        path_output=cfg.path_output,
    )

    assert response.tensor.shape[1:] == response.img.shape
    assert response.tensor.dtype == torch.float32
    assert len(response.faces[0].preds.keys()) > 0


@pytest.mark.integration
def test_analyzer_tensor(cfg, analyzer):
    if not hasattr(cfg, "path_tensor"):
        pytest.skip("No tensor path provided in config.")
    tensor = torch.load(
        cfg.path_tensor,
        map_location=torch.device(cfg.analyzer.device)
    )
    response = analyzer.run(
        tensor=tensor,
        batch_size=cfg.batch_size,
        fix_img_size=cfg.fix_img_size,
        return_img_data=cfg.return_img_data,
        include_tensors=cfg.include_tensors,
        path_output=cfg.path_output,
    )

    assert response.tensor.shape[1:] == response.img.shape
    assert response.tensor.shape == (1, 3, 1080, 1080)
    assert response.tensor.dtype == torch.float32
    assert response.tensor.device == torch.device(cfg.analyzer.device)
    assert len(response.faces[0].preds.keys()) > 0


@pytest.mark.integration
@pytest.mark.analyzer
def test_analyzer_tensor_via_image_source(cfg, analyzer):
    if hasattr(cfg, "path_tensor"):
        pytest.skip("This test is only for path_image.")
    if "test.jpg" not in cfg.path_image:
        pytest.skip("Only test.jpg is used for this test.")
    pil_image = Image.open(cfg.path_image).convert("RGB")
    tensor_input = torch.from_numpy(np.array(pil_image)).permute(2, 0, 1).contiguous()
    response = analyzer.run(
        image_source=tensor_input,
        batch_size=cfg.batch_size,
        fix_img_size=cfg.fix_img_size,
        return_img_data=cfg.return_img_data,
        include_tensors=cfg.include_tensors,
    )

    assert response.tensor.dtype == torch.float32
    assert len(response.faces) > 0


@pytest.mark.integration
@pytest.mark.analyzer
def test_analyzer_numpy_via_image_source(cfg, analyzer):
    if hasattr(cfg, "path_tensor"):
        pytest.skip("This test is only for path_image.")
    if "test.jpg" not in cfg.path_image:
        pytest.skip("Only test.jpg is used for this test.")
    pil_image = Image.open(cfg.path_image).convert("RGB")
    array_input = np.array(pil_image)
    response = analyzer.run(
        image_source=array_input,
        batch_size=cfg.batch_size,
        fix_img_size=cfg.fix_img_size,
        return_img_data=cfg.return_img_data,
        include_tensors=cfg.include_tensors,
    )

    assert response.tensor.dtype == torch.float32
    assert len(response.faces) > 0


@pytest.mark.integration
@pytest.mark.analyzer
def test_analyzer_pil_via_image_source(cfg, analyzer):
    if hasattr(cfg, "path_tensor"):
        pytest.skip("This test is only for path_image.")
    if "test.jpg" not in cfg.path_image:
        pytest.skip("Only test.jpg is used for this test.")
    pil_image = Image.open(cfg.path_image).convert("RGB")
    response = analyzer.run(
        image_source=pil_image,
        batch_size=cfg.batch_size,
        fix_img_size=cfg.fix_img_size,
        return_img_data=cfg.return_img_data,
        include_tensors=cfg.include_tensors,
    )

    assert response.tensor.dtype == torch.float32
    assert len(response.faces) > 0


@pytest.mark.unit
@pytest.mark.analyzer
def test_analyzer_no_input_raises(analyzer):
    with pytest.raises(ValueError, match="image_source is required"):
        analyzer.run()


@pytest.mark.unit
@pytest.mark.analyzer
def test_analyzer_unsupported_type_raises(analyzer):
    with pytest.raises(TypeError, match="Unsupported image_source type"):
        analyzer.run(image_source=12345)


@pytest.mark.endtoend
@pytest.mark.analyzer
def test_analyzer_include_predictors(cfg, analyzer):
    if hasattr(cfg, "path_tensor"):
        pytest.skip("This test is only for path_image.")
    if "test.jpg" not in cfg.path_image:
        pytest.skip("Only test.jpg is used for this test.")
    predictor_names = list(analyzer.predictors.keys())
    if len(predictor_names) < 2:
        pytest.skip("Need at least 2 predictors for this test.")
    selected = predictor_names[:1]
    response = analyzer.run(
        image_source=cfg.path_image,
        include_predictors=selected,
        return_img_data=True,
        include_tensors=True,
    )
    if len(response.faces) > 0:
        assert set(response.faces[0].preds.keys()) == set(selected)


@pytest.mark.endtoend
@pytest.mark.analyzer
def test_analyzer_exclude_predictors(cfg, analyzer):
    if hasattr(cfg, "path_tensor"):
        pytest.skip("This test is only for path_image.")
    if "test.jpg" not in cfg.path_image:
        pytest.skip("Only test.jpg is used for this test.")
    predictor_names = list(analyzer.predictors.keys())
    if len(predictor_names) < 2:
        pytest.skip("Need at least 2 predictors for this test.")
    excluded = predictor_names[:1]
    response = analyzer.run(
        image_source=cfg.path_image,
        exclude_predictors=excluded,
        return_img_data=True,
        include_tensors=True,
    )
    if len(response.faces) > 0:
        for name in excluded:
            assert name not in response.faces[0].preds.keys()


@pytest.mark.endtoend
@pytest.mark.analyzer
def test_analyzer_skip_detector(cfg, analyzer):
    if hasattr(cfg, "path_tensor"):
        pytest.skip("This test is only for path_image.")
    if "test.jpg" not in cfg.path_image:
        pytest.skip("Only test.jpg is used for this test.")
    response = analyzer.run(
        image_source=cfg.path_image,
        skip_detector=True,
        return_img_data=True,
        include_tensors=True,
    )
    assert len(response.faces) == 1
    assert response.faces[0].ratio == 1.0


@pytest.mark.unit
@pytest.mark.analyzer
def test_analyzer_include_exclude_mutual_exclusion(cfg, analyzer):
    if hasattr(cfg, "path_tensor"):
        pytest.skip("This test is only for path_image.")
    if "test.jpg" not in cfg.path_image:
        pytest.skip("Only test.jpg is used for this test.")
    with pytest.raises(ValueError, match="Cannot specify both"):
        analyzer.run(
            image_source=cfg.path_image,
            include_predictors=["fer"],
            exclude_predictors=["au"],
        )


@pytest.mark.integration
@pytest.mark.analyzer
def test_analyzer_bytes_via_image_source(cfg, analyzer):
    if hasattr(cfg, "path_tensor"):
        pytest.skip("This test is only for path_image.")
    if "test.jpg" not in cfg.path_image:
        pytest.skip("Only test.jpg is used for this test.")
    pil_image = Image.open(cfg.path_image).convert("RGB")
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG")
    response = analyzer.run(
        image_source=buf.getvalue(),
        batch_size=cfg.batch_size,
        fix_img_size=cfg.fix_img_size,
        return_img_data=cfg.return_img_data,
        include_tensors=cfg.include_tensors,
    )
    assert response.tensor.dtype == torch.float32
    assert len(response.faces) > 0


@pytest.mark.integration
@pytest.mark.analyzer
def test_analyzer_grayscale_pil_via_image_source(cfg, analyzer):
    if hasattr(cfg, "path_tensor"):
        pytest.skip("This test is only for path_image.")
    if "test.jpg" not in cfg.path_image:
        pytest.skip("Only test.jpg is used for this test.")
    pil_image = Image.open(cfg.path_image).convert("L")
    response = analyzer.run(
        image_source=pil_image,
        batch_size=cfg.batch_size,
        fix_img_size=cfg.fix_img_size,
        return_img_data=cfg.return_img_data,
        include_tensors=cfg.include_tensors,
    )
    assert response.tensor.shape[1] == 3


@pytest.mark.endtoend
@pytest.mark.analyzer
def test_analyzer_callable(cfg, analyzer):
    if hasattr(cfg, "path_tensor"):
        pytest.skip("This test is only for path_image.")
    if "test.jpg" not in cfg.path_image:
        pytest.skip("Only test.jpg is used for this test.")
    response = analyzer(
        image_source=cfg.path_image,
        return_img_data=True,
        include_tensors=True,
    )
    assert len(response.faces) > 0
