#!/usr/bin/env python3
"""Audit immutable Hugging Face objects referenced by the model manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

if __package__:
    from scripts.model_evidence_contract import (
        ModelEvidenceContractError,
        expected_metadata_identity,
        validate_metadata_identity,
    )
    from scripts.render_model_cards import render_model_documents
else:
    from model_evidence_contract import (
        ModelEvidenceContractError,
        expected_metadata_identity,
        validate_metadata_identity,
    )
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
    verify_legal_documents: bool,
    results: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    packaged_manifest_sha256: str | None = None,
    remote_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the single stable result schema for every audit outcome."""
    return {
        "schema_version": 1,
        "status": "failed" if failures else "ok",
        "manifest_revision": manifest.get("manifest_revision"),
        "packaged_manifest_sha256": packaged_manifest_sha256,
        "remote_manifest": dict(remote_manifest) if remote_manifest else None,
        "download_artifacts": download_artifacts,
        "require_current_metadata": require_current_metadata,
        "verify_legal_documents": verify_legal_documents,
        "results": results,
        "failures": failures,
    }


def audit_remote_manifest(
    manifest_path: Path,
    *,
    download_artifacts: bool = False,
    require_current_metadata: bool = True,
    require_remote_manifest: bool = False,
    remote_manifest_path: Path | None = None,
    verify_legal_documents: bool = True,
    api=None,
    download_fn=None,
) -> dict[str, Any]:
    """Verify revisions, legal documents, artifacts, and validation metadata."""
    try:
        manifest = _read_json(manifest_path)
        packaged_manifest_sha256 = _sha256(manifest_path)
    except Exception as exc:
        return _audit_report(
            {},
            download_artifacts=download_artifacts,
            require_current_metadata=require_current_metadata,
            verify_legal_documents=verify_legal_documents,
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

    remote_manifest_identity = None
    remote_records: dict[tuple[str, str], Mapping[str, Any]] = {}
    compatibility: Mapping[str, Any] = {}
    if require_remote_manifest:
        try:
            repo_id = str(manifest.get("manifest_repo_id", ""))
            revision = str(manifest.get("manifest_revision", ""))
            filename = str(manifest.get("manifest_filename", ""))
            expected_digest = str(manifest.get("manifest_sha256", ""))
            if re.fullmatch(r"[^/\s]+/[^/\s]+", repo_id) is None:
                raise HubManifestError("Packaged remote manifest repository is invalid.")
            if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
                raise HubManifestError("Packaged remote manifest revision is invalid.")
            if (
                not filename
                or Path(filename).is_absolute()
                or ".." in Path(filename).parts
            ):
                raise HubManifestError("Packaged remote manifest filename is invalid.")
            if re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None:
                raise HubManifestError("Packaged remote manifest digest is invalid.")
            fetched_path = (
                Path(remote_manifest_path)
                if remote_manifest_path is not None
                else Path(
                    download_fn(
                        repo_id=repo_id,
                        filename=filename,
                        revision=revision,
                    )
                )
            )
            if _sha256(fetched_path) != expected_digest:
                raise HubManifestError("Fetched remote manifest digest disagrees.")
            remote = _read_json(fetched_path)
            if remote.get("schema_version") != 1 or remote.get("status") != "approved":
                raise HubManifestError("Remote model manifest is not final and approved.")
            for record in remote.get("models", []):
                if not isinstance(record, Mapping):
                    raise HubManifestError("Remote model manifest record is invalid.")
                identity = (str(record.get("model_id", "")), str(record.get("cohort", "")))
                if not all(identity) or identity in remote_records:
                    raise HubManifestError(
                        f"Remote model manifest identity is invalid: {identity}."
                    )
                remote_records[identity] = record
            expected_remote_identities = {
                (str(model_id), str(artifact.get("torch_min", "")))
                for model_id, model in manifest.get("models", {}).items()
                for artifact in model.get("artifacts", [])
                if isinstance(artifact, Mapping) and artifact.get("format") == "pt2"
            }
            if set(remote_records) != expected_remote_identities:
                raise HubManifestError(
                    "Remote model manifest coverage differs from the packaged manifest."
                )
            compatibility_ref = str(manifest.get("compatibility_ref", ""))
            compatibility_path = manifest_path.parent / compatibility_ref
            compatibility = _read_json(compatibility_path)
            remote_manifest_identity = {
                "repo_id": repo_id,
                "revision": revision,
                "filename": filename,
                "sha256": expected_digest,
                "plan_id": remote.get("plan_id"),
                "status": remote.get("status"),
            }
        except Exception as exc:
            return _audit_report(
                manifest,
                download_artifacts=download_artifacts,
                require_current_metadata=require_current_metadata,
                verify_legal_documents=verify_legal_documents,
                results=[],
                failures=[
                    {
                        "model_id": "remote-manifest-contract",
                        "repo_id": manifest.get("manifest_repo_id"),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                ],
                packaged_manifest_sha256=packaged_manifest_sha256,
                remote_manifest=remote_manifest_identity,
            )

    expected_legal_documents = {}
    if verify_legal_documents:
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
                verify_legal_documents=verify_legal_documents,
                results=[],
                failures=[
                    {
                        "model_id": "model-card-contract",
                        "repo_id": None,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                ],
                packaged_manifest_sha256=packaged_manifest_sha256,
                remote_manifest=remote_manifest_identity,
            )
    results = []
    failures = []
    cohort_records = {
        str(record.get("torch_minor", "")): record
        for record in compatibility.get("cohorts", [])
        if isinstance(record, Mapping)
    }
    validation_policy = compatibility.get("validation_policy", {})
    required_devices = list(
        compatibility.get("platform_policy", {}).get("required_devices", [])
    )
    expected_model_ids = list(manifest.get("models", {}))
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
            legal_results = []
            if verify_legal_documents:
                model_legal_documents = expected_legal_documents.get(model_id)
                if model_legal_documents is None:
                    raise HubManifestError(
                        f"No generated legal contract for {model_id}."
                    )
                for filename, expected_bytes in sorted(
                    model_legal_documents.items()
                ):
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
                metadata_sha256_verified = False
                metadata_identity_verified = False
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
                    cohort = str(artifact.get("torch_min", ""))
                    remote_record = remote_records.get((model_id, cohort))
                    if require_remote_manifest:
                        if remote_record is None:
                            raise HubManifestError(
                                f"Remote manifest omits {model_id}/{cohort}."
                            )
                        expected_remote_fields = {
                            "model_id": model_id,
                            "repo_id": repo_id,
                            "cohort": cohort,
                            "revision": revision,
                            "artifact_filename": filename,
                            "artifact_sha256": artifact["sha256"],
                            "artifact_size_bytes": int(artifact["size_bytes"]),
                            "metadata_filename": metadata_filename,
                            "metadata_sha256": artifact.get("metadata_sha256"),
                            "golden_reference_sha256": artifact.get(
                                "golden_reference_sha256"
                            ),
                            "golden_reference_size_bytes": artifact.get(
                                "golden_reference_size_bytes"
                            ),
                            "golden_reference_source_cohort": artifact.get(
                                "golden_reference_source_cohort"
                            ),
                            "required_devices": required_devices,
                        }
                        differing_remote = sorted(
                            field
                            for field, expected in expected_remote_fields.items()
                            if remote_record.get(field) != expected
                        )
                        if differing_remote:
                            raise HubManifestError(
                                f"Remote manifest differs for {model_id}/{cohort}: "
                                + ", ".join(differing_remote)
                                + "."
                            )
                        observed_metadata_sha = _sha256(metadata_path)
                        if observed_metadata_sha != remote_record.get(
                            "metadata_sha256"
                        ):
                            raise HubManifestError(
                                f"{repo_id}/{metadata_filename} digest disagrees "
                                "with the remote manifest."
                            )
                        metadata_sha256_verified = True

                        cohort_record = cohort_records.get(cohort)
                        if not isinstance(cohort_record, Mapping):
                            raise HubManifestError(
                                f"Compatibility omits Torch cohort {cohort}."
                            )
                        environment = metadata.get("environment")
                        arguments = metadata.get("exporter_arguments")
                        if not isinstance(environment, Mapping) or not isinstance(
                            arguments, Mapping
                        ):
                            raise HubManifestError(
                                f"{repo_id}/{metadata_filename} lacks provenance."
                            )
                        expected_patch = str(cohort_record.get("validated_patch", ""))
                        torch_version = str(metadata.get("torch_version", ""))
                        expected_golden_mode = (
                            "record"
                            if cohort
                            == validation_policy.get("golden_reference_cohort")
                            else "reuse"
                        )
                        expected_golden_status = (
                            "recorded" if expected_golden_mode == "record" else "reused"
                        )
                        expected_arguments = {
                            "mode": "export",
                            "artifact_cohort": cohort,
                            "batch_sizes": validation_policy.get(
                                "predictor_batch_sizes"
                            ),
                            "seeds": validation_policy.get("seeds"),
                            "scales": validation_policy.get("scales"),
                            "validate_devices": required_devices,
                            "golden_reference_mode": expected_golden_mode,
                            "golden_reference_cohort": validation_policy.get(
                                "golden_reference_cohort"
                            ),
                        }
                        argument_model_ids = arguments.get("model_ids")
                        model_id_contract_ok = (
                            isinstance(argument_model_ids, list)
                            and all(
                                isinstance(value, str) and value
                                for value in argument_model_ids
                            )
                            and len(set(argument_model_ids))
                            == len(argument_model_ids)
                            and set(argument_model_ids) == set(expected_model_ids)
                        )
                        source_tree = environment.get("source_tree", {})
                        if (
                            torch_version.split("+", 1)[0] != expected_patch
                            or environment.get("torch_version") != torch_version
                            or environment.get("export_schema")
                            != cohort_record.get("export_schema")
                            or source_tree.get("commit") != model.get("export_commit")
                            or source_tree.get("clean") is not True
                            or not model_id_contract_ok
                            or any(
                                arguments.get(field) != expected
                                for field, expected in expected_arguments.items()
                            )
                        ):
                            raise HubManifestError(
                                f"{repo_id}/{metadata_filename} provenance disagrees "
                                "with the packaged release contract."
                            )
                        summary_identity = {
                            "torch_version": torch_version,
                            "torch_minor": cohort,
                            "runtime_torch_minor": cohort,
                            "environment": dict(environment),
                            "exporter_arguments": dict(arguments),
                        }
                        try:
                            expected_identity = expected_metadata_identity(
                                summary_identity,
                                model_id=model_id,
                                repo_id=repo_id,
                                artifact_filename=filename,
                            )
                            validate_metadata_identity(metadata, expected_identity)
                        except ModelEvidenceContractError as exc:
                            raise HubManifestError(
                                f"{repo_id}/{metadata_filename} identity is invalid: "
                                f"{exc}."
                            ) from exc
                        golden = metadata.get("validation", {}).get(
                            "golden_reference", {}
                        )
                        if (
                            golden.get("status") != expected_golden_status
                            or golden.get("source_cohort")
                            != remote_record.get("golden_reference_source_cohort")
                            or golden.get("sha256")
                            != remote_record.get("golden_reference_sha256")
                            or golden.get("size_bytes")
                            != remote_record.get("golden_reference_size_bytes")
                        ):
                            raise HubManifestError(
                                f"{repo_id}/{metadata_filename} golden-reference "
                                "contract disagrees."
                            )
                        metadata_identity_verified = True
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
                        "metadata_sha256_verified": metadata_sha256_verified,
                        "metadata_identity_verified": metadata_identity_verified,
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
        verify_legal_documents=verify_legal_documents,
        results=results,
        failures=failures,
        packaged_manifest_sha256=packaged_manifest_sha256,
        remote_manifest=remote_manifest_identity,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", default="facetorch/models/manifest.json"
    )
    parser.add_argument(
        "--remote-manifest",
        help="Already fetched exact remote manifest; otherwise download its packaged pin.",
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
        require_remote_manifest=True,
        remote_manifest_path=(
            Path(args.remote_manifest) if args.remote_manifest else None
        ),
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
