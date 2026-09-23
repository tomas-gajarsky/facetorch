"""Debug timing without shared invocation state or retained measurements."""

from functools import wraps
import logging
from time import perf_counter
from typing import Callable, ParamSpec, TypeVar

_P = ParamSpec("_P")
_R = TypeVar("_R")


def timed(
    name: str, *, logger: logging.Logger
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Log each synchronous call's elapsed time, including calls that raise.

    The start time lives on the invocation stack so nested and concurrent calls
    are independent. No samples or totals are retained. Disabled debug logging
    bypasses timing entirely.
    """

    def decorate(function: Callable[_P, _R]) -> Callable[_P, _R]:
        @wraps(function)
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            if not logger.isEnabledFor(logging.DEBUG):
                return function(*args, **kwargs)
            started = perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                logger.debug("%s: %.2f ms", name, (perf_counter() - started) * 1000)

        return wrapped

    return decorate
