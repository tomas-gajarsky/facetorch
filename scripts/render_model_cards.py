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
UPSTREAM_NOTICE_ROOT = (REPO_ROOT / "model_cards" / "upstream_notices").resolve()


class ModelCardError(RuntimeError):
    """Raised when card source data is incomplete or inconsistent."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelCardError(f"Cannot read JSON contract document: {path}") from exc
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
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ModelCardError(f"Cannot read upstream license file: {path}") from exc
    expected_digest = str(source.get("license_sha256", ""))
    actual_digest = hashlib.sha256(data).hexdigest()
    if actual_digest != expected_digest:
        raise ModelCardError(
            f"Upstream license digest mismatch for {source.get('license_url')}: "
            f"expected {expected_digest}, got {actual_digest}"
        )
    revision = str(source.get("revision", ""))
    source_url = str(source.get("url", "")).rstrip("/")
    expected_prefix = f"{source_url}/blob/{revision}/"
    if not str(source.get("license_url", "")).startswith(expected_prefix):
        raise ModelCardError(
            "Upstream license URL is not bound to its source and revision"
        )
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ModelCardError(f"Upstream license is not UTF-8: {path}") from exc
    return data


def _source_notice_bytes(source: Mapping[str, Any]) -> bytes | None:
    """Return a verified Apache NOTICE, or explicit verified absence."""
    if source.get("code_license") != "Apache-2.0":
        return None
    required = {"notice_url", "notice_file", "notice_sha256"}
    missing = sorted(required - set(source))
    if missing:
        raise ModelCardError(
            "Apache-2.0 source does not record NOTICE state: " + ", ".join(missing)
        )
    values = [source.get(name) for name in sorted(required)]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ModelCardError("Apache-2.0 NOTICE metadata must be complete or all null")

    relative = Path(str(source["notice_file"]))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ModelCardError(f"Invalid upstream NOTICE file: {relative}")
    path = (REPO_ROOT / relative).resolve()
    if not path.is_relative_to(UPSTREAM_NOTICE_ROOT):
        raise ModelCardError(f"Upstream NOTICE file leaves its allowed root: {path}")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ModelCardError(f"Cannot read upstream NOTICE file: {path}") from exc
    expected_digest = str(source["notice_sha256"])
    actual_digest = hashlib.sha256(data).hexdigest()
    if actual_digest != expected_digest:
        raise ModelCardError(
            f"Upstream NOTICE digest mismatch for {source['notice_url']}: "
            f"expected {expected_digest}, got {actual_digest}"
        )
    revision = str(source.get("revision", ""))
    source_url = str(source.get("url", "")).rstrip("/")
    if not str(source["notice_url"]).startswith(f"{source_url}/blob/{revision}/"):
        raise ModelCardError(
            "Upstream NOTICE URL is not bound to its source and revision"
        )
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ModelCardError(f"Upstream NOTICE is not UTF-8: {path}") from exc
    return data


def _validate_model_governance(
    model_id: str,
    governance: Any,
) -> None:
    """Validate required model-governance mappings before rendering."""
    if not isinstance(governance, Mapping):
        raise ModelCardError(f"Model {model_id} governance must be an object")
    required = {
        "status",
        "release_eligible",
        "hosted_model_card",
        "upstream_sources",
        "source_checkpoint",
        "rights",
        "intended_use",
        "limitations",
    }
    missing = sorted(required - set(governance))
    if missing:
        raise ModelCardError(
            f"Model {model_id} governance is missing required fields: "
            + ", ".join(missing)
        )
    if governance["status"] != "approved":
        raise ModelCardError(f"Model {model_id} governance status must be approved")
    if governance["release_eligible"] is not True:
        raise ModelCardError(f"Model {model_id} release_eligible must be true")

    rights = governance["rights"]
    required_rights = {
        "weights_license",
        "redistribution",
        "attribution",
        "owner_approval",
    }
    if not isinstance(rights, Mapping):
        raise ModelCardError(f"Model {model_id} rights must be an object")
    missing_rights = sorted(required_rights - set(rights))
    if missing_rights:
        raise ModelCardError(
            f"Model {model_id} rights are missing required fields: "
            + ", ".join(missing_rights)
        )
    for name in ("redistribution", "attribution", "owner_approval"):
        if rights[name] != "approved":
            raise ModelCardError(f"Model {model_id} rights.{name} must be approved")

    sources = governance["upstream_sources"]
    if not isinstance(sources, list) or not sources:
        raise ModelCardError(
            f"Model {model_id} upstream_sources must be a non-empty list"
        )
    required_source = {
        "url",
        "revision",
        "code_license",
        "license_url",
        "license_role",
        "license_file",
        "license_sha256",
    }
    for index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            raise ModelCardError(
                f"Model {model_id} upstream source {index} must be an object"
            )
        missing_source = sorted(required_source - set(source))
        if missing_source:
            raise ModelCardError(
                f"Model {model_id} upstream source {index} is missing required "
                "fields: " + ", ".join(missing_source)
            )

    checkpoint = governance["source_checkpoint"]
    required_checkpoint = {
        "hosted_sha256_verified",
        "upstream_checkpoint_mapping",
        "upstream_artifacts",
        "verification_method",
        "verification_result",
    }
    if not isinstance(checkpoint, Mapping):
        raise ModelCardError(f"Model {model_id} source_checkpoint must be an object")
    missing_checkpoint = sorted(required_checkpoint - set(checkpoint))
    if missing_checkpoint:
        raise ModelCardError(
            f"Model {model_id} source_checkpoint is missing required fields: "
            + ", ".join(missing_checkpoint)
        )
    if checkpoint["upstream_checkpoint_mapping"] != "verified":
        raise ModelCardError(
            f"Model {model_id} upstream checkpoint mapping must be verified"
        )
    if checkpoint["hosted_sha256_verified"] is not True:
        raise ModelCardError(
            f"Model {model_id} hosted SHA-256 verification must be true"
        )

    for name in ("intended_use", "limitations"):
        values = governance[name]
        if not isinstance(values, list) or not values:
            raise ModelCardError(f"Model {model_id} {name} must be a non-empty list")


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


def _checkpoint_table(
    model_id: str,
    checkpoint: Mapping[str, Any],
) -> list[str]:
    lines = [
        "| Upstream checkpoint | SHA-256 | Source |",
        "|---|---|---|",
    ]
    try:
        artifacts = checkpoint["upstream_artifacts"]
        if not isinstance(artifacts, list) or not artifacts:
            raise TypeError("upstream_artifacts must be a non-empty list")
        for artifact in artifacts:
            digest = artifact.get("sha256") or "Unavailable from expired source"
            lines.append(
                f"| `{_markdown(artifact['filename'])}` | `{_markdown(digest)}` | "
                f"[publisher location]({artifact['source_url']}) |"
            )
    except (AttributeError, KeyError, TypeError) as exc:
        raise ModelCardError(
            f"Model {model_id} has invalid upstream checkpoint artifacts"
        ) from exc
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
        "Facetorch v1 routes supported Torch 2.6-2.13 runtimes through the "
        "2.6 and 2.11 artifact cohort files listed in its manifest. The legacy "
        "TorchScript object is CPU-only and requires "
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
        "model = torch.export.load(path).module().to(device)",
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
        *_checkpoint_table(model_id, checkpoint),
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
        notice_bytes = _source_notice_bytes(source)
        if source.get("license_role") != "notice":
            if notice_bytes is None:
                continue
        name = str(source["url"]).rstrip("/").rsplit("/", 1)[-1]
        if source.get("license_role") == "notice":
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
        if notice_bytes is not None:
            lines.extend(
                [
                    "",
                    f"## {name} — upstream NOTICE",
                    "",
                    f"Pinned NOTICE: {source['notice_url']}",
                    "",
                    notice_bytes.decode("utf-8").rstrip("\n"),
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
    if not all(isinstance(value, Mapping) for value in (cards, models, records)):
        raise ModelCardError("Catalog, manifest, and governance models must be objects")
    model_ids = set(models)
    complete = set(cards) == model_ids and set(records) == model_ids
    covered = model_ids <= set(cards) and model_ids <= set(records)
    if not model_ids or (require_complete_contract and not complete) or not covered:
        raise ModelCardError(
            "Catalog and governance do not cover the requested manifest models"
        )
    if governance.get("status") != "approved":
        raise ModelCardError(
            "Model cards may be published only from approved governance"
        )
    license_policy = governance.get("license_policy")
    if (
        not isinstance(license_policy, Mapping)
        or license_policy.get("status") != "approved"
    ):
        raise ModelCardError(
            "Model cards may be published only from an approved license policy"
        )
    approved_on = str(governance.get("approved_on", "")).strip()
    if not approved_on:
        raise ModelCardError("Approved governance must record approved_on")

    rendered: dict[str, dict[str, bytes]] = {}
    for model_id in sorted(models):
        record = records[model_id]
        _validate_model_governance(model_id, record)
        values = {
            "README.md": _render_card(
                model_id,
                cards[model_id],
                models[model_id],
                record,
                approved_on,
            ).encode("utf-8"),
            "LICENSE": _license_bytes(record),
            "THIRD_PARTY_NOTICES.md": _render_notices(model_id, record).encode("utf-8"),
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
