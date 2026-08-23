#!/usr/bin/env python3
"""Render the ten release-bound Hugging Face model cards and notices."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = REPO_ROOT / "model_cards" / "catalog.json"
MANIFEST_PATH = REPO_ROOT / "facetorch" / "models" / "manifest.json"
GOVERNANCE_PATH = REPO_ROOT / "facetorch" / "models" / "governance.json"
SPDX_TAGS = {"MIT": "mit", "Apache-2.0": "apache-2.0"}
UPSTREAM_LICENSE_ROOT = (REPO_ROOT / "model_cards" / "upstream_licenses").resolve()


class ModelCardError(RuntimeError):
    """Raised when card source data is incomplete or inconsistent."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ModelCardError(f"JSON document must contain an object: {path}")
    return value


def _markdown(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _source_license_bytes(source: Mapping[str, Any]) -> bytes:
    relative = Path(str(source.get("license_file", "")))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ModelCardError(f"Invalid upstream license file: {relative}")
    path = (REPO_ROOT / relative).resolve()
    if not path.is_relative_to(UPSTREAM_LICENSE_ROOT):
        raise ModelCardError(f"Upstream license file leaves its allowed roots: {path}")
    data = path.read_bytes()
    if source.get("license_strip_final_newline"):
        if not data.endswith(b"\n"):
            raise ModelCardError(f"Expected one normalized final newline in {path}")
        data = data[:-1]
    expected_digest = str(source.get("license_sha256", ""))
    actual_digest = hashlib.sha256(data).hexdigest()
    if actual_digest != expected_digest:
        raise ModelCardError(
            f"Upstream license digest mismatch for {source.get('license_url')}: "
            f"expected {expected_digest}, got {actual_digest}"
        )
    revision = str(source.get("revision", ""))
    if revision not in str(source.get("license_url", "")):
        raise ModelCardError("Upstream license URL is not bound to its revision")
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ModelCardError(f"Upstream license is not UTF-8: {path}") from exc
    return data


def _license_bytes(governance: Mapping[str, Any]) -> bytes:
    license_id = governance["rights"]["weights_license"]
    sources = [
        source
        for source in governance["upstream_sources"]
        if source.get("license_role") == "weights"
    ]
    if not sources:
        raise ModelCardError("Model has no weights-license source")
    texts = []
    for source in sources:
        if source.get("code_license") != license_id:
            raise ModelCardError(
                f"Weights license {license_id} disagrees with {source.get('url')}"
            )
        texts.append(_source_license_bytes(source))
    if any(text != texts[0] for text in texts[1:]):
        raise ModelCardError("Weights-license sources do not contain identical text")
    return texts[0]


def _artifact_table(model: Mapping[str, Any]) -> list[str]:
    lines = [
        "| File | Format | Runtime | Devices | SHA-256 |",
        "|---|---|---|---|---|",
    ]
    for artifact in model["artifacts"]:
        runtime = (
            f">={artifact['torch_min']}, <{artifact['torch_max_exclusive']}"
            if artifact.get("torch_min")
            else "See manifest"
        )
        lines.append(
            "| `{filename}` | {format} | {runtime} | {devices} | `{sha}` |".format(
                filename=_markdown(artifact["filename"]),
                format=_markdown(artifact["format"]),
                runtime=_markdown(runtime),
                devices=", ".join(artifact["devices"]),
                sha=artifact["sha256"],
            )
        )
    return lines


def _source_table(governance: Mapping[str, Any]) -> list[str]:
    lines = [
        "| Upstream | Immutable revision | Role | License |",
        "|---|---|---|---|",
    ]
    for source in governance["upstream_sources"]:
        revision = source["revision"]
        lines.append(
            f"| [{_markdown(source['url'])}]({source['url']}) | "
            f"[`{revision}`]({source['url']}/tree/{revision}) | "
            f"{_markdown(source.get('role', 'source'))} | "
            f"[{source['code_license']}]({source['license_url']}) |"
        )
    return lines


def _checkpoint_table(checkpoint: Mapping[str, Any]) -> list[str]:
    lines = [
        "| Upstream checkpoint | SHA-256 | Source |",
        "|---|---|---|",
    ]
    for artifact in checkpoint["upstream_artifacts"]:
        digest = artifact.get("sha256") or "Unavailable from expired source"
        lines.append(
            f"| `{_markdown(artifact['filename'])}` | `{_markdown(digest)}` | "
            f"[publisher location]({artifact['source_url']}) |"
        )
    return lines


def _render_card(
    model_id: str,
    card: Mapping[str, Any],
    manifest_model: Mapping[str, Any],
    governance: Mapping[str, Any],
    approved_on: str,
) -> str:
    rights = governance["rights"]
    checkpoint = governance["source_checkpoint"]
    license_id = rights["weights_license"]
    spdx_tag = SPDX_TAGS.get(license_id)
    if spdx_tag is None:
        raise ModelCardError(f"No Hugging Face license tag for {license_id}")
    shape = ", ".join(str(value) for value in card["input_shape"])

    lines = [
        "---",
        "library_name: pytorch",
        f"license: {spdx_tag}",
        "tags:",
        "- facetorch",
        "- torch-export",
        "- face-analysis",
        "---",
        "",
        f"# {card['title']}",
        "",
        str(card["summary"]),
        "",
        "This repository contains immutable model artifacts used by "
        "[Facetorch](https://github.com/tomas-gajarsky/facetorch). Use the "
        "packaged Facetorch manifest to select a revision and artifact; do not "
        "treat mutable `main` or older unlisted files as a release contract.",
        "",
        "## Contract",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Model ID | `{model_id}` |",
        f"| Architecture | {_markdown(card['architecture'])} |",
        f"| Input | {_markdown(card['input'])} |",
        f"| Output | {_markdown(card['output'])} |",
        f"| Dynamic shapes | {_markdown(card['dynamic_shapes'])} |",
        f"| Weights license | [{license_id}](LICENSE) |",
        "",
        f"Preprocessing: {card['preprocessing']}",
        "",
        "### Release artifacts",
        "",
        *_artifact_table(manifest_model),
        "",
        "Facetorch v1 supports the Torch 2.6 and 2.11 cohort files listed in "
        "its manifest. The legacy TorchScript object is CPU-only and requires "
        "the explicit legacy opt-in. Files from unsupported cohorts are not part "
        "of the v1 release contract.",
        "",
        "## Loading the manifest-selected artifact",
        "",
        "```python",
        "import torch",
        "from huggingface_hub import hf_hub_download",
        "from facetorch.artifacts import get_model_manifest",
        "",
        f'MODEL_ID = "{model_id}"',
        'device = "cuda" if torch.cuda.is_available() else "cpu"',
        "artifact = get_model_manifest().candidates(",
        "    MODEL_ID,",
        "    torch_version=torch.__version__,",
        "    device=device,",
        "    allow_legacy_models=False,",
        ")[0]",
        "path = hf_hub_download(",
        "    repo_id=artifact.repo_id,",
        "    revision=artifact.revision,",
        "    filename=artifact.filename,",
        ")",
        "model = torch.export.load(path).module().to(device).eval()",
        f"example = torch.randn({shape}, device=device)",
        "with torch.inference_mode():",
        "    output = model(example)",
        "```",
        "",
        "The random tensor above is only a loading smoke test. Use Facetorch's "
        "documented preprocessing for meaningful inference.",
        "",
        "## Provenance",
        "",
        *_source_table(governance),
        "",
        *_checkpoint_table(checkpoint),
        "",
        f"Mapping method: `{checkpoint['verification_method']}`.",
        "",
        f"Result: {checkpoint['verification_result']}",
    ]
    if checkpoint.get("evidence_limit"):
        lines.extend(["", f"Evidence limitation: {checkpoint['evidence_limit']}"])
    lines.extend(
        [
            "",
            "The repository owner approved the mapping and redistribution record "
            f"on {approved_on}. Under the recorded policy, an author-published "
            "checkpoint in a permissively licensed repository with no separate "
            "checkpoint terms uses that repository license. MIT and Apache-2.0 "
            "have not been converted or treated as interchangeable. See "
            "[LICENSE](LICENSE), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), "
            "and Facetorch's `facetorch/models/governance.json`.",
            "",
        ]
    )
    if card.get("papers"):
        lines.extend(["## Papers", ""])
        for paper in card["papers"]:
            lines.append(f"- [{paper['title']}]({paper['url']})")
        lines.append("")
    lines.extend(["## Intended use", ""])
    lines.extend(f"- {value}" for value in governance["intended_use"])
    lines.extend(["", "## Limitations and responsible use", ""])
    lines.extend(f"- {value}" for value in governance["limitations"])
    lines.extend(
        [
            "- The artifact license does not itself license training datasets, "
            "input data, or a deployment's processing of personal data.",
            "- Do not use model output as the sole basis for consequential "
            "decisions about a person.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_notices(
    model_id: str,
    governance: Mapping[str, Any],
) -> str:
    lines = [
        "# Third-party notices",
        "",
        f"This file records the upstream material used by `{model_id}`. The "
        "checkpoint license is in `LICENSE`; source repositories and their exact "
        "license files are listed below.",
        "",
        *_source_table(governance),
    ]
    for source in governance["upstream_sources"]:
        if source.get("license_role") != "notice":
            continue
        name = str(source["url"]).rstrip("/").rsplit("/", 1)[-1]
        license_text = _source_license_bytes(source).decode("utf-8").rstrip("\n")
        lines.extend(
            [
                "",
                f"## {name} — {source['code_license']}",
                "",
                f"Source: {source['url']}",
                "",
                f"Pinned license: {source['license_url']}",
                "",
                license_text,
            ]
        )
    lines.append("")
    return "\n".join(lines)


def render_model_documents(
    manifest_path: Path = MANIFEST_PATH,
    *,
    require_complete_contract: bool = True,
) -> dict[str, dict[str, bytes]]:
    """Return release-bound Hub documents without touching the filesystem."""
    catalog = _read_json(CATALOG_PATH)
    manifest = _read_json(manifest_path)
    governance = _read_json(GOVERNANCE_PATH)
    cards = catalog.get("models", {})
    models = manifest.get("models", {})
    records = governance.get("models", {})
    model_ids = set(models)
    complete = set(cards) == model_ids and set(records) == model_ids
    covered = model_ids <= set(cards) and model_ids <= set(records)
    if not model_ids or (require_complete_contract and not complete) or not covered:
        raise ModelCardError(
            "Catalog and governance do not cover the requested manifest models"
        )
    if governance.get("status") != "approved":
        raise ModelCardError("Model cards may be published only from approved governance")
    approved_on = str(governance.get("approved_on", "")).strip()
    if not approved_on:
        raise ModelCardError("Approved governance must record approved_on")

    rendered: dict[str, dict[str, bytes]] = {}
    for model_id in sorted(models):
        record = records[model_id]
        if record.get("status") != "approved" or not record.get("release_eligible"):
            raise ModelCardError(f"Model governance is not approved: {model_id}")
        values = {
            "README.md": _render_card(
                model_id,
                cards[model_id],
                models[model_id],
                record,
                approved_on,
            ).encode("utf-8"),
            "LICENSE": _license_bytes(record),
            "THIRD_PARTY_NOTICES.md": _render_notices(
                model_id, record
            ).encode("utf-8"),
        }
        rendered[model_id] = values
    return rendered


def render_model_cards(output_root: Path) -> dict[str, list[str]]:
    documents = render_model_documents()
    rendered: dict[str, list[str]] = {}
    for model_id, values in documents.items():
        model_root = output_root / model_id
        model_root.mkdir(parents=True, exist_ok=True)
        for filename, value in values.items():
            (model_root / filename).write_bytes(value)
        rendered[model_id] = sorted(values)
    return rendered


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    rendered = render_model_cards(args.output_root.resolve())
    print(json.dumps(rendered, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
