import facetorch
import pytest


@pytest.mark.integration
@pytest.mark.utilizer
def test_grand_base_types(analyzer):
    for utilizer in analyzer.utilizers.values():
        assert isinstance(utilizer, facetorch.base.BaseProcessor)


@pytest.mark.integration
@pytest.mark.utilizer
def test_base_types(analyzer):
    for utilizer in analyzer.utilizers.values():
        assert isinstance(utilizer, facetorch.base.BaseUtilizer)
