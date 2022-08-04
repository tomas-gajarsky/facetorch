import facetorch
import pytest


@pytest.mark.integration
@pytest.mark.unifier
def test_base_type(analyzer):
    assert isinstance(analyzer.unifier, facetorch.base.BaseProcessor)
