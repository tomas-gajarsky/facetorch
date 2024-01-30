import pytest
import torch


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
    tensor = torch.load(cfg.path_tensor, map_location=torch.device(cfg.analyzer.device))
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
