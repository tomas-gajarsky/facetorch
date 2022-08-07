import facetorch
import pytest
import torch


@pytest.mark.integration
@pytest.mark.reader
def test_base_type(analyzer):
    assert isinstance(analyzer.reader, facetorch.base.BaseReader)


@pytest.mark.endtoend
@pytest.mark.reader
def test_output_shape_length(cfg, analyzer):
    data = analyzer.reader.run(cfg.path_image)
    assert len(data.tensor.shape) == 4


@pytest.mark.endtoend
@pytest.mark.reader
def test_output_shape_batch_channel(cfg, analyzer):
    data = analyzer.reader.run(cfg.path_image)
    assert data.tensor.shape[:2] == torch.Size([1, 3])
