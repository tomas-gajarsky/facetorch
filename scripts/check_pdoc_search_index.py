#!/usr/bin/env python3
"""Compare pdoc3 search indexes without platform-specific Lunr scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PREFIX = "let [INDEX, DOCS] = "
URL_SEPARATOR = "; let URLS="


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


def _canonical_documents(documents: list[Any], urls: list[Any]) -> list[Any]:
    if not all(isinstance(url, str) for url in urls) or len(set(urls)) != len(urls):
        raise ValueError("pdoc3 search index has an invalid URL table")
    canonical = []
    references = set()
    for position, document in enumerate(documents):
        if not isinstance(document, dict) or document.get("i") != position:
            raise ValueError("pdoc3 search index has invalid document ordinals")
        reference = document.get("ref")
        url_index = document.get("url")
        if (
            not isinstance(reference, str)
            or reference in references
            or not isinstance(url_index, int)
            or isinstance(url_index, bool)
            or not 0 <= url_index < len(urls)
        ):
            raise ValueError("pdoc3 search index has an invalid document reference")
        references.add(reference)
        canonical.append(
            {
                **{
                    key: value
                    for key, value in document.items()
                    if key not in {"i", "url"}
                },
                "url": urls[url_index],
            }
        )
    return sorted(canonical, key=lambda document: document["ref"])


def _index_signature(index: dict[str, Any]) -> dict[str, Any]:
    version = index.get("version")
    fields = index.get("fields")
    pipeline = index.get("pipeline", [])
    inverted_index = index.get("invertedIndex")
    field_vectors = index.get("fieldVectors")
    if (
        not isinstance(version, str)
        or not isinstance(fields, list)
        or not all(isinstance(field, str) for field in fields)
        or not isinstance(pipeline, list)
        or not isinstance(inverted_index, list)
        or not isinstance(field_vectors, list)
    ):
        raise ValueError("pdoc3 search index has an invalid Lunr schema")
    tokens = []
    for entry in inverted_index:
        if not isinstance(entry, list) or not entry or not isinstance(entry[0], str):
            raise ValueError("pdoc3 search index has an invalid token entry")
        tokens.append(entry[0])
    if len(tokens) != len(set(tokens)):
        raise ValueError("pdoc3 search index contains duplicate tokens")
    return {
        "version": version,
        "fields": fields,
        "pipeline": pipeline,
        "tokens": sorted(tokens),
    }


def compare_indexes(generated_path: Path, committed_path: Path) -> None:
    generated_index, generated_docs, generated_urls = _load_index(generated_path)
    committed_index, committed_docs, committed_urls = _load_index(committed_path)
    if _canonical_documents(generated_docs, generated_urls) != _canonical_documents(
        committed_docs, committed_urls
    ):
        raise ValueError("generated pdoc3 search document corpus is stale")
    if set(generated_urls) != set(committed_urls):
        raise ValueError("generated pdoc3 search URL table is stale")
    if _index_signature(generated_index) != _index_signature(committed_index):
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
