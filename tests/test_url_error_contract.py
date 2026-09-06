"""Malformed URLs remain catchable through the public analyzer error boundary."""

from unittest.mock import Mock

import pytest

from facetorch import FaceAnalyzer, load_config
from facetorch.analyzer.reader import core as reader_core
from facetorch.exceptions import InputError

pytestmark = pytest.mark.release_blocker


def _analyzer():
    config = load_config(offline=True).analyzer
    config.logger = None
    config.reader = {
        "_target_": "facetorch.analyzer.reader.URLReader",
        "transform": None,
        "device": "cpu",
        "optimize_transform": False,
        "allowed_schemes": ["http", "https"],
    }
    return FaceAnalyzer(config)


@pytest.mark.parametrize(
    "url", ["http://[", "https://[not-an-ip]/", "https://example.com\uff0fpath"]
)
def test_malformed_authority_is_input_error_before_network_access(monkeypatch, url):
    network = Mock(side_effect=AssertionError("Malformed URLs must not reach DNS"))
    monkeypatch.setattr(reader_core, "_validate_public_url_target", network)
    with pytest.raises(InputError, match="malformed") as caught:
        _analyzer().run(url, skip_detector=True, include_predictors=[])
    assert isinstance(caught.value.__cause__, ValueError)
    network.assert_not_called()


def test_malformed_redirect_closes_response_and_does_not_follow(monkeypatch):
    response = Mock(status=302, headers={"Location": "https://["})
    connection = Mock()
    open_response = Mock(return_value=(connection, response))
    monkeypatch.setattr(
        reader_core, "_validate_public_url_target", lambda *_: ["8.8.8.8"]
    )
    monkeypatch.setattr(reader_core, "_open_pinned_response", open_response)
    with pytest.raises(InputError) as caught:
        _analyzer().run(
            "https://example.com/image", skip_detector=True, include_predictors=[]
        )
    assert isinstance(caught.value.__cause__, ValueError)
    open_response.assert_called_once()
    response.close.assert_called_once()
    connection.close.assert_called_once()
