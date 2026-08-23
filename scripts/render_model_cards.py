#!/usr/bin/env python3
"""Render the ten release-bound Hugging Face model cards and notices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = REPO_ROOT / "model_cards" / "catalog.json"
MANIFEST_PATH = REPO_ROOT / "facetorch" / "models" / "manifest.json"
GOVERNANCE_PATH = REPO_ROOT / "facetorch" / "models" / "governance.json"
SPDX_TAGS = {"MIT": "mit", "Apache-2.0": "apache-2.0"}


class ModelCardError(RuntimeError):
    """Raised when card source data is incomplete or inconsistent."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ModelCardError(f"JSON document must contain an object: {path}")
    return value


def _markdown(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _mit_license(copyright_notice: str) -> str:
    return f"""MIT License

{copyright_notice}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the \"Software\"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def _license_text(model: Mapping[str, Any], rights: Mapping[str, Any]) -> str:
    license_id = rights["weights_license"]
    if license_id == "Apache-2.0":
        return (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")
    if license_id == "MIT":
        notice = str(model.get("license_copyright", "")).strip()
        if not notice:
            raise ModelCardError("MIT model is missing its copyright notice")
        return _mit_license(notice)
    raise ModelCardError(f"Unsupported weights license: {license_id!r}")


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
            "on 2026-08-23. Under the recorded policy, an author-published "
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
    card: Mapping[str, Any],
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
    for notice in card.get("additional_mit_notices", []):
        lines.extend(
            [
                "",
                f"## {notice['name']} — MIT",
                "",
                f"Source: {notice['url']}",
                "",
                _mit_license(str(notice["copyright"])).rstrip(),
            ]
        )
    lines.append("")
    return "\n".join(lines)


def render_model_cards(output_root: Path) -> dict[str, list[str]]:
    catalog = _read_json(CATALOG_PATH)
    manifest = _read_json(MANIFEST_PATH)
    governance = _read_json(GOVERNANCE_PATH)
    cards = catalog.get("models", {})
    models = manifest.get("models", {})
    records = governance.get("models", {})
    if not cards or set(cards) != set(models) or set(records) != set(models):
        raise ModelCardError(
            "Catalog, manifest, and governance must cover exactly the same models"
        )
    if governance.get("status") != "approved":
        raise ModelCardError("Model cards may be published only from approved governance")

    rendered: dict[str, list[str]] = {}
    for model_id in sorted(models):
        record = records[model_id]
        if record.get("status") != "approved" or not record.get("release_eligible"):
            raise ModelCardError(f"Model governance is not approved: {model_id}")
        model_root = output_root / model_id
        model_root.mkdir(parents=True, exist_ok=True)
        values = {
            "README.md": _render_card(
                model_id, cards[model_id], models[model_id], record
            ),
            "LICENSE": _license_text(cards[model_id], record["rights"]),
            "THIRD_PARTY_NOTICES.md": _render_notices(
                model_id, cards[model_id], record
            ),
        }
        for filename, value in values.items():
            (model_root / filename).write_text(value, encoding="utf-8")
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
