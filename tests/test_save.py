import os

import pytest

from facetorch.datastruct import ImageData


@pytest.mark.endtoend
@pytest.mark.utilizer
@pytest.mark.save
def test_draw_boxes(analyzer, cfg, response):
    if cfg.path_output is None:
        pytest.skip("No output path")
    if "save" not in cfg.analyzer.utilizer.keys():
        pytest.skip("Save utilizer not configured")
    if response.image is None:
        pytest.skip("Image data was not retained in the analysis result")

    if os.path.exists(cfg.path_output):
        os.remove(cfg.path_output)
    data = ImageData(path_output=cfg.path_output, img=response.image)
    analyzer.utilizers["save"].run(data)
    assert os.path.exists(cfg.path_output)
    os.remove(cfg.path_output)
