"""Timing must not prevent independent workers or retain per-request samples."""

from concurrent.futures import ThreadPoolExecutor
import inspect
import logging
import threading

from codetiming import Timer
import pytest
import torch

from facetorch import FaceAnalyzer, load_config
from facetorch._timing import timed
from facetorch.analyzer.reader import UniversalReader

pytestmark = pytest.mark.release_blocker


def _analyzer():
    config = load_config(offline=True).analyzer
    config.logger = None
    return FaceAnalyzer(config)


@pytest.mark.parametrize("level", [logging.DEBUG, logging.ERROR])
def test_separate_analyzers_and_readers_can_overlap(monkeypatch, caplog, level):
    first, second = _analyzer(), _analyzer()
    assert first.reader is not second.reader
    rendezvous = threading.Barrier(2, timeout=5)
    original = UniversalReader.read_tensor

    def read_tensor(self, *args, **kwargs):
        # Both FaceAnalyzer.run and UniversalReader.run are still on the stack.
        rendezvous.wait()
        return original(self, *args, **kwargs)

    monkeypatch.setattr(UniversalReader, "read_tensor", read_tensor)
    tensor = torch.zeros(3, 32, 32, dtype=torch.uint8)
    with caplog.at_level(level, logger="facetorch"), ThreadPoolExecutor(2) as pool:
        futures = [
            pool.submit(analyzer.run, tensor, skip_detector=True, include_predictors=[])
            for analyzer in (first, second)
        ]
        results = [future.result(timeout=10) for future in futures]
    assert all(len(result.faces) == 1 for result in results)
    if level == logging.DEBUG:
        messages = [record.getMessage() for record in caplog.records]
        assert sum(message.startswith("FaceAnalyzer.run:") for message in messages) == 2
        assert (
            sum(message.startswith("UniversalReader.run:") for message in messages) == 2
        )


def test_nested_calls_and_exceptions_have_independent_timing(caplog, monkeypatch):
    logger = logging.getLogger("facetorch.timing-test")
    ticks = iter([0, 1, 2, 3, 4, 5, 6, 7])
    monkeypatch.setattr("facetorch._timing.perf_counter", lambda: next(ticks))

    @timed("recursive", logger=logger)
    def operation(depth: int, *, fail: bool = False) -> int:
        """An inspectable timed callable."""
        if fail:
            raise ValueError("original failure")
        return operation(depth - 1) + 1 if depth else 0

    with caplog.at_level(logging.DEBUG, logger=logger.name):
        assert operation(1) == 1
        with pytest.raises(ValueError, match="original failure"):
            operation(0, fail=True)
        assert operation(0) == 0
    assert [record.getMessage() for record in caplog.records] == [
        "recursive: 1000.00 ms",
        "recursive: 3000.00 ms",
        "recursive: 1000.00 ms",
        "recursive: 1000.00 ms",
    ]
    assert inspect.signature(operation) == inspect.signature(operation.__wrapped__)
    assert operation.__name__ == "operation"
    assert operation.__doc__ == "An inspectable timed callable."


def test_disabled_debug_logging_does_not_collect_runtime_samples(caplog, monkeypatch):
    analyzer = _analyzer()
    tensor = torch.zeros(3, 8, 8, dtype=torch.uint8)
    before = {name: len(samples) for name, samples in Timer.timers._timings.items()}

    def unexpected_clock():
        pytest.fail("Disabled debug timing should not measure or retain samples")

    monkeypatch.setattr("facetorch._timing.perf_counter", unexpected_clock)
    with caplog.at_level(logging.ERROR, logger="facetorch"):
        for _ in range(1000):
            analyzer.run(tensor, skip_detector=True, include_predictors=[])
    assert {
        name: len(samples) for name, samples in Timer.timers._timings.items()
    } == before
