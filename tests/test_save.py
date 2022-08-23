import os

import pytest
from facetorch.datastruct import Response


@pytest.mark.endtoend
@pytest.mark.utilizer
@pytest.mark.save
def test_draw_boxes(analyzer, cfg, response):
    if cfg.path_output is None:
        pytest.skip("No output path")
    if "save" not in cfg.analyzer.utilizer.keys():
        pytest.skip("Save utilizer not configured")
    if isinstance(response, Response):
        pytest.skip("No output path in the data response")

    if os.path.exists(cfg.path_output):
        os.remove(cfg.path_output)
    analyzer.utilizers["save"].run(response)
    assert os.path.exists(cfg.path_output)
    os.remove(cfg.path_output)
