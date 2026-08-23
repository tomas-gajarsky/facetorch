#!/usr/bin/env python3
"""Audit immutable Hugging Face objects referenced by the model manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

if __package__:
    from scripts.render_model_cards import render_model_documents
else:
    from render_model_cards import render_model_documents


class HubManifestError(RuntimeError):
    """Raised when a remote manifest object is missing or disagrees with its pin."""


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise HubManifestError(f"Manifest {path} is not a JSON object.")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audit_report(
    manifest: Mapping[str, Any],
    *,
    download_artifacts: bool,
    require_current_metadata: bool,
    results: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the single stable result schema for every audit outcome."""
    return {
        "schema_version": 1,
        "status": "failed" if failures else "ok",
        "manifest_revision": manifest.get("manifest_revision"),
        "download_artifacts": download_artifacts,
        "require_current_metadata": require_current_metadata,
        "results": results,
        "failures": failures,
    }


def audit_remote_manifest(
    manifest_path: Path,
    *,
    download_artifacts: bool = False,
    require_current_metadata: bool = True,
    api=None,
    download_fn=None,
) -> dict[str, Any]:
    """Verify revisions, legal documents, artifacts, and validation metadata."""
    try:
        manifest = _read_json(manifest_path)
    except Exception as exc:
        return _audit_report(
            {},
            download_artifacts=download_artifacts,
            require_current_metadata=require_current_metadata,
            results=[],
            failures=[
                {
                    "model_id": "manifest-contract",
                    "repo_id": None,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            ],
        )

    if api is None or download_fn is None:
        from huggingface_hub import HfApi, hf_hub_download

        api = api or HfApi()
        download_fn = download_fn or hf_hub_download

    try:
        expected_legal_documents = render_model_documents(
            manifest_path,
            require_complete_contract=False,
        )
    except Exception as exc:
        return _audit_report(
            manifest,
            download_artifacts=download_artifacts,
            require_current_metadata=require_current_metadata,
            results=[],
            failures=[
                {
                    "model_id": "model-card-contract",
                    "repo_id": None,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            ],
        )
    results = []
    failures = []
    for model_id, model in sorted(manifest.get("models", {}).items()):
        repo_id = model["repo_id"]
        revision = model["revision"]
        try:
            info = api.model_info(
                repo_id=repo_id,
                revision=revision,
                files_metadata=True,
            )
            if info.sha != revision:
                raise HubManifestError(
                    f"{repo_id} resolved {revision} to unexpected commit {info.sha}."
                )
            siblings = {item.rfilename: item for item in info.siblings}
            model_legal_documents = expected_legal_documents.get(model_id)
            if model_legal_documents is None:
                raise HubManifestError(f"No generated legal contract for {model_id}.")
            legal_results = []
            for filename, expected_bytes in sorted(model_legal_documents.items()):
                sibling = siblings.get(filename)
                if sibling is None:
                    raise HubManifestError(
                        f"{repo_id}@{revision} has no {filename}."
                    )
                if int(sibling.size) != len(expected_bytes):
                    raise HubManifestError(
                        f"{repo_id}@{revision}/{filename} has an unexpected size."
                    )
                remote_path = Path(
                    download_fn(
                        repo_id=repo_id,
                        filename=filename,
                        revision=revision,
                    )
                )
                remote_bytes = remote_path.read_bytes()
                if remote_bytes != expected_bytes:
                    raise HubManifestError(
                        f"{repo_id}@{revision}/{filename} does not match the "
                        "generated release contract."
                    )
                legal_results.append(
                    {
                        "filename": filename,
                        "sha256": hashlib.sha256(expected_bytes).hexdigest(),
                        "size_bytes": len(expected_bytes),
                        "bytes_verified": True,
                    }
                )
            expected_license_ref = (
                f"https://huggingface.co/{repo_id}/blob/{revision}/LICENSE"
            )
            if model.get("license_ref") != expected_license_ref:
                raise HubManifestError(
                    f"{repo_id}@{revision} has an unbound license reference."
                )

            artifact_results = []
            for artifact in model["artifacts"]:
                filename = artifact["filename"]
                sibling = siblings.get(filename)
                if sibling is None:
                    raise HubManifestError(
                        f"{repo_id}@{revision} is missing {filename}."
                    )
                lfs = getattr(sibling, "lfs", None)
                if (
                    lfs is None
                    or lfs.sha256 != artifact["sha256"]
                    or int(sibling.size) != int(artifact["size_bytes"])
                ):
                    raise HubManifestError(
                        f"{repo_id}@{revision}/{filename} disagrees with its LFS "
                        "SHA-256 or size."
                    )

                bytes_verified = False
                if download_artifacts:
                    path = Path(
                        download_fn(
                            repo_id=repo_id,
                            filename=filename,
                            revision=revision,
                        )
                    )
                    if (
                        path.stat().st_size != int(artifact["size_bytes"])
                        or _sha256(path) != artifact["sha256"]
                    ):
                        raise HubManifestError(
                            f"Downloaded bytes disagree for {repo_id}/{filename}."
                        )
                    bytes_verified = True

                metadata_status = "not_applicable"
                metadata_filename = artifact.get("validation_metadata")
                if metadata_filename:
                    if metadata_filename not in siblings:
                        raise HubManifestError(
                            f"{repo_id}@{revision} is missing {metadata_filename}."
                        )
                    metadata_path = Path(
                        download_fn(
                            repo_id=repo_id,
                            filename=metadata_filename,
                            revision=revision,
                        )
                    )
                    metadata = _read_json(metadata_path)
                    current = (
                        metadata.get("schema_version") == 2
                        and metadata.get("artifact_sha256") == artifact["sha256"]
                        and metadata.get("validation", {}).get("status") == "ok"
                    )
                    metadata_status = "current" if current else "legacy"
                    if require_current_metadata and not current:
                        raise HubManifestError(
                            f"{repo_id}/{metadata_filename} is not current, "
                            "digest-bound validation metadata."
                        )

                artifact_results.append(
                    {
                        "artifact_id": artifact["id"],
                        "filename": filename,
                        "sha256": artifact["sha256"],
                        "size_bytes": int(artifact["size_bytes"]),
                        "lfs_oid_verified": True,
                        "downloaded_bytes_verified": bytes_verified,
                        "metadata_status": metadata_status,
                    }
                )
            results.append(
                {
                    "model_id": model_id,
                    "repo_id": repo_id,
                    "revision": revision,
                    "status": "ok",
                    "legal_documents": legal_results,
                    "artifacts": artifact_results,
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "model_id": model_id,
                    "repo_id": repo_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    return _audit_report(
        manifest,
        download_artifacts=download_artifacts,
        require_current_metadata=require_current_metadata,
        results=results,
        failures=failures,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", default="facetorch/models/manifest.json"
    )
    parser.add_argument("--download-artifacts", action="store_true")
    parser.add_argument(
        "--allow-legacy-metadata",
        action="store_true",
        help="Inventory mode only; legacy metadata remains a release blocker.",
    )
    parser.add_argument("--report")
    args = parser.parse_args()

    report = audit_remote_manifest(
        Path(args.manifest),
        download_artifacts=args.download_artifacts,
        require_current_metadata=not args.allow_legacy_metadata,
    )
    if args.report:
        Path(args.report).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if report["status"] != "ok":
        for failure in report["failures"]:
            print(
                f"{failure['model_id']}: {failure['error_type']}: "
                f"{failure['error']}"
            )
        return 1
    print(f"Verified {len(report['results'])} immutable model repositories.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
