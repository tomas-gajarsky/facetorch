#!/usr/bin/env python3
"""Compare pdoc3 search indexes without platform-specific Lunr scores."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


PREFIX = "let [INDEX, DOCS] = "
URL_SEPARATOR = "; let URLS="
_SCORE = object()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON numeric constant: {value}")


def _load_index(path: Path) -> tuple[dict[str, Any], list[Any], list[Any]]:
    content = path.read_text(encoding="utf-8").strip()
    if not content.startswith(PREFIX):
        raise ValueError(f"{path} is not a prebuilt pdoc3 search index")
    payload_text, separator, urls_text = content[len(PREFIX) :].partition(
        URL_SEPARATOR
    )
    if not separator:
        raise ValueError(f"{path} does not contain the pdoc3 URL table")
    payload = json.loads(payload_text, parse_constant=_reject_json_constant)
    urls = json.loads(urls_text, parse_constant=_reject_json_constant)
    if (
        not isinstance(payload, list)
        or len(payload) != 2
        or not isinstance(payload[0], dict)
        or not isinstance(payload[1], list)
        or not isinstance(urls, list)
    ):
        raise ValueError(f"{path} has an unexpected pdoc3 search-index schema")
    return payload[0], payload[1], urls


def _without_scores(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("pdoc3 search index contains a non-finite score")
        return _SCORE
    if isinstance(value, list):
        return [_without_scores(item) for item in value]
    if isinstance(value, dict):
        return {key: _without_scores(item) for key, item in value.items()}
    return value


def compare_indexes(generated_path: Path, committed_path: Path) -> None:
    generated_index, generated_docs, generated_urls = _load_index(generated_path)
    committed_index, committed_docs, committed_urls = _load_index(committed_path)
    if generated_docs != committed_docs:
        raise ValueError("generated pdoc3 search document corpus is stale")
    if generated_urls != committed_urls:
        raise ValueError("generated pdoc3 search URL table is stale")
    if _without_scores(generated_index) != _without_scores(committed_index):
        raise ValueError("generated pdoc3 search index structure is stale")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generated", type=Path)
    parser.add_argument("committed", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    compare_indexes(args.generated, args.committed)
    print("pdoc3 search index content is synchronized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
