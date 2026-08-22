#!/usr/bin/env python3
"""Audit immutable Hugging Face objects referenced by the model manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


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


def audit_remote_manifest(
    manifest_path: Path,
    *,
    download_artifacts: bool = False,
    require_current_metadata: bool = True,
    api=None,
    download_fn=None,
) -> dict[str, Any]:
    """Verify revision, LFS object ID, size, metadata, and optionally bytes."""
    if api is None or download_fn is None:
        from huggingface_hub import HfApi, hf_hub_download

        api = api or HfApi()
        download_fn = download_fn or hf_hub_download

    manifest = _read_json(manifest_path)
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
            if "README.md" not in siblings:
                raise HubManifestError(f"{repo_id}@{revision} has no model card.")

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

    return {
        "schema_version": 1,
        "status": "failed" if failures else "ok",
        "manifest_revision": manifest.get("manifest_revision"),
        "download_artifacts": download_artifacts,
        "require_current_metadata": require_current_metadata,
        "results": results,
        "failures": failures,
    }


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
