import os

import pytest


@pytest.mark.end2end
@pytest.mark.analyzer
def test_draw_boxes(analyzer, cfg):
    if cfg.path_output is None:
        pytest.skip("No output path")

    if os.path.exists(cfg.path_output):
        os.remove(cfg.path_output)
    analyzer.run(
        path_image=cfg.path_image,
        batch_size=cfg.batch_size,
        path_output=cfg.path_output,
    )
    assert os.path.exists(cfg.path_output)
    os.remove(cfg.path_output)
