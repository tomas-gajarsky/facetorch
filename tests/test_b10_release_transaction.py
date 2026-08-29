import hashlib
import io
import json
import subprocess

import pytest

from scripts.release_transaction import (
    IMMUTABLE_CHANNELS,
    ReleaseError,
    assert_stable_alias_promotion,
    docker_distribution_state,
    fetch_model_manifest,
    parse_release_tag,
    prepare_release_plan,
    pypi_distribution_state,
    record_channel,
    run_publication_transaction,
    validate_local_image_id,
    validate_model_audit_report,
    validate_packaged_model_governance,
    validate_local_release_evidence,
    verify_bundle_checksums,
    verify_github_release_assets,
    verify_public_checksums,
    verify_publication_receipt,
    verify_release_plan,
    write_bundle_checksums,
    write_public_checksums,
)


SOURCE_SHA = "1" * 40
MODEL_REVISION = "2" * 40
MODEL_FILENAME = "manifests/approved-plan.json"
CPU_IMAGE_DIGEST = "sha256:" + "3" * 64
GPU_IMAGE_DIGEST = "sha256:" + "4" * 64
METADATA_SHA256 = "9" * 64
GOLDEN_REFERENCE_SHA256 = "a" * 64
GOLDEN_REFERENCE_SIZE = 42


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True) + "\n").encode()


def _model_manifest_data():
    return {
        "schema_version": 1,
        "status": "approved",
        "plan_id": "model-plan",
        "models": [
            {
                "model_id": "detector",
                "repo_id": "owner/detector",
                "cohort": "2.6",
                "revision": "5" * 40,
                "artifact_filename": "model-torch2.6.pt2",
                "artifact_sha256": "6" * 64,
                "artifact_size_bytes": 123,
                "metadata_filename": "model-torch2.6.pt2.meta.json",
                "metadata_sha256": METADATA_SHA256,
                "golden_reference_sha256": GOLDEN_REFERENCE_SHA256,
                "golden_reference_size_bytes": GOLDEN_REFERENCE_SIZE,
                "golden_reference_source_cohort": "2.6",
                "required_devices": ["cpu", "cuda"],
            }
        ],
    }


def _git(repo, *arguments):
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _candidate_repo(tmp_path, version="1.0.0"):
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Release Test")
    _git(repo, "config", "user.email", "release@example.test")
    (repo / "pyproject.toml").write_text(
        f'[project]\nname = "facetorch"\nversion = "{version}"\n',
        encoding="utf-8",
    )
    (repo / "CHANGELOG.md").write_text(
        f"# Change Log\n\n## {version} (Unreleased)\n\nCandidate.\n",
        encoding="utf-8",
    )
    models = repo / "facetorch" / "models"
    models.mkdir(parents=True)
    docker = repo / "docker"
    docker.mkdir()
    (docker / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    (docker / "Dockerfile.gpu").write_text("FROM scratch\n", encoding="utf-8")
    lock = repo / "environments" / "torch-2.6-cu124" / "uv.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("version = 1\n", encoding="utf-8")
    _write_json(
        models / "manifest.json",
        {
            "manifest_version": 1,
            "manifest_revision": MODEL_REVISION,
            "manifest_repo_id": "owner/facetorch-model-manifest",
            "manifest_filename": MODEL_FILENAME,
            "manifest_sha256": hashlib.sha256(
                _json_bytes(_model_manifest_data())
            ).hexdigest(),
            "status": "approved",
            "compatibility_ref": "compatibility.json",
            "governance_ref": "governance.json",
            "models": {
                "detector": {
                    "task": "detector",
                    "source": "huggingface",
                    "repo_id": "owner/detector",
                    "revision": "5" * 40,
                    "source_weight_sha256": "7" * 64,
                    "export_commit": "8" * 40,
                    "license_ref": "https://example.test/license",
                    "artifacts": [
                        {
                            "id": "detector-torch2.6",
                            "filename": "model-torch2.6.pt2",
                            "format": "pt2",
                            "sha256": "6" * 64,
                            "size_bytes": 123,
                            "torch_min": "2.6",
                            "torch_max_exclusive": "2.7",
                            "devices": ["cpu", "cuda"],
                            "schema_major": 8,
                            "schema_minor": 2,
                            "validation_metadata": "model-torch2.6.pt2.meta.json",
                            "metadata_sha256": METADATA_SHA256,
                            "golden_reference_sha256": GOLDEN_REFERENCE_SHA256,
                            "golden_reference_size_bytes": GOLDEN_REFERENCE_SIZE,
                            "golden_reference_source_cohort": "2.6",
                        }
                    ],
                }
            },
        },
    )
    _write_json(
        models / "compatibility.json",
        {
            "schema_version": 1,
            "status": "approved",
            "python": {"specifier": ">=3.10,<3.13"},
            "torch": {"specifier": ">=2.6,<2.7", "supported_minor_lines": ["2.6"]},
            "platform_policy": {"required_devices": ["cpu", "cuda"]},
        },
    )
    _write_json(
        models / "governance.json",
        {
            "schema_version": 1,
            "status": "approved",
            "models": {
                "detector": {
                    "status": "approved",
                    "release_eligible": True,
                    "source_checkpoint": {
                        "upstream_checkpoint_mapping": "verified",
                        "hosted_sha256_verified": True,
                    },
                    "rights": {
                        "weights_license": "MIT",
                        "redistribution": "approved",
                        "attribution": "approved",
                        "owner_approval": "approved",
                    },
                    "limitations": ["Synthetic test model."],
                }
            },
        },
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "candidate")
    return repo, _git(repo, "rev-parse", "HEAD")


def _model_manifest(path):
    _write_json(path, _model_manifest_data())


def _local_release_evidence(bundle, repo, source_sha):
    root = bundle / "evidence" / "local-gpu"
    summary = root / "torch-2.6/summary-torch2.6.json"
    _write_json(summary, {"schema_version": 1, "status": "ok"})
    matrix = root / "candidate-matrix-report.json"
    _write_json(
        matrix,
        {
            "schema_version": 2,
            "status": "ok",
            "release_approval_required": False,
            "required_devices": ["cpu", "cuda"],
            "lanes": [
                {
                    "torch_minor": "2.6",
                    "source_commit": source_sha,
                    "source_clean": True,
                }
            ],
        },
    )
    default_smoke = root / "default-analyzer-cuda-smoke.json"
    notebook_report = root / "facetorch-notebook-report.json"
    notebook = root / "facetorch-notebook-executed.ipynb"
    _write_json(
        default_smoke,
        {"schema_version": 1, "status": "ok", "device": "cuda", "legacy_fallback": False},
    )
    _write_json(notebook_report, {"schema_version": 1, "status": "ok", "device": "cuda"})
    _write_json(notebook, {"nbformat": 4, "nbformat_minor": 5, "cells": [], "metadata": {}})
    container_smokes = {}
    for flavor, device in (("cpu", "cpu"), ("gpu", "cuda")):
        path = root / f"container-reports/{flavor}-image-smoke.json"
        _write_json(
            path,
            {
                "schema_version": 1,
                "status": "ok",
                "uid": 10001,
                "device": device,
                "legacy_fallback": False,
            },
        )
        container_smokes[flavor] = path

    runner = root / "local-cuda-runner-report.json"
    model_root = repo / "facetorch/models"
    lock = repo / "environments/torch-2.6-cu124/uv.lock"
    _write_json(
        runner,
        {
            "schema_version": 1,
            "status": "ok",
            "source_sha": source_sha,
            "source_clean": True,
            "platform": {"system": "Linux", "machine": "x86_64"},
            "gpu_attestation": "synthetic GPU",
            "uv_version": "uv 0.9.14",
            "manifest_sha256": _sha256(model_root / "manifest.json"),
            "compatibility_sha256": _sha256(model_root / "compatibility.json"),
            "governance_sha256": _sha256(model_root / "governance.json"),
            "environment_locks": {
                "2.6": {
                    "path": "environments/torch-2.6-cu124/uv.lock",
                    "sha256": _sha256(lock),
                }
            },
            "summaries": [
                {
                    "path": "torch-2.6/summary-torch2.6.json",
                    "sha256": _sha256(summary),
                }
            ],
            "matrix_report_sha256": _sha256(matrix),
            "default_analyzer_smoke_sha256": _sha256(default_smoke),
            "notebook_report_sha256": _sha256(notebook_report),
            "executed_notebook_sha256": _sha256(notebook),
            "publication_performed": False,
            "candidate_evidence_only": False,
        },
    )
    _write_json(
        root / "container-evidence.json",
        {
            "schema_version": 1,
            "status": "ok",
            "source_sha": source_sha,
            "runner_report_sha256": _sha256(runner),
            "images": {
                "cpu": {
                    "image_id": CPU_IMAGE_DIGEST,
                    "os": "linux",
                    "architecture": "amd64",
                    "configured_user": "facetorch",
                    "dockerfile_sha256": _sha256(repo / "docker/Dockerfile"),
                    "smoke_report_sha256": _sha256(container_smokes["cpu"]),
                },
                "gpu": {
                    "image_id": GPU_IMAGE_DIGEST,
                    "os": "linux",
                    "architecture": "amd64",
                    "configured_user": "facetorch",
                    "dockerfile_sha256": _sha256(repo / "docker/Dockerfile.gpu"),
                    "smoke_report_sha256": _sha256(container_smokes["gpu"]),
                },
            },
            "runtime_constraints": {
                "network": "none",
                "root_filesystem": "read-only",
                "container_user": 10001,
            },
            "publication_performed": False,
        },
    )


def _model_audit_evidence(bundle, repo, remote_manifest):
    packaged_manifest = repo / "facetorch/models/manifest.json"
    _write_json(
        bundle / "evidence/model-manifest-audit.json",
        {
            "schema_version": 1,
            "status": "ok",
            "manifest_revision": MODEL_REVISION,
            "packaged_manifest_sha256": _sha256(packaged_manifest),
            "remote_manifest": {
                "repo_id": "owner/facetorch-model-manifest",
                "revision": MODEL_REVISION,
                "filename": MODEL_FILENAME,
                "sha256": _sha256(remote_manifest),
                "plan_id": "model-plan",
                "status": "approved",
            },
            "download_artifacts": True,
            "require_current_metadata": True,
            "verify_legal_documents": True,
            "results": [
                {
                    "model_id": "detector",
                    "repo_id": "owner/detector",
                    "revision": "5" * 40,
                    "status": "ok",
                    "legal_documents": [
                        {
                            "filename": filename,
                            "sha256": hashlib.sha256(filename.encode()).hexdigest(),
                            "size_bytes": len(filename),
                            "bytes_verified": True,
                        }
                        for filename in (
                            "README.md",
                            "LICENSE",
                            "THIRD_PARTY_NOTICES.md",
                        )
                    ],
                    "artifacts": [
                        {
                            "artifact_id": "detector-torch2.6",
                            "filename": "model-torch2.6.pt2",
                            "sha256": "6" * 64,
                            "size_bytes": 123,
                            "lfs_oid_verified": True,
                            "downloaded_bytes_verified": True,
                            "metadata_status": "current",
                            "metadata_sha256_verified": True,
                            "metadata_identity_verified": True,
                        }
                    ],
                }
            ],
            "failures": [],
        },
    )


def _bundle(tmp_path, repo=None, source_sha=None):
    bundle = tmp_path / "bundle"
    files = {
        "distributions/facetorch-1.0.0-py3-none-any.whl": b"wheel",
        "distributions/facetorch-1.0.0.tar.gz": b"sdist",
        "images/facetorch-cpu.tar.zst": b"cpu image",
        "images/facetorch-gpu.tar.zst": b"gpu image",
        "release-evidence.tar.zst": b"release evidence",
        "sboms/distributions.spdx.json": b"{}\n",
        "sboms/facetorch-cpu.spdx.json": b"{}\n",
        "sboms/facetorch-gpu.spdx.json": b"{}\n",
        "evidence/release-inputs.json": b"{}\n",
    }
    for relative, content in files.items():
        path = bundle / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    manifest = bundle / "evidence/model-manifest.json"
    _model_manifest(manifest)
    _write_json(
        bundle / "evidence/model-manifest-report.json",
        {
            "repo_id": "owner/facetorch-model-manifest",
            "revision": MODEL_REVISION,
            "filename": MODEL_FILENAME,
            "sha256": _sha256(manifest),
            "plan_id": "model-plan",
            "model_cohort_count": 1,
        },
    )
    if repo is not None and source_sha is not None:
        _model_audit_evidence(bundle, repo, manifest)
        _local_release_evidence(bundle, repo, source_sha)
    return bundle, manifest


def _release_plan(tmp_path, version="1.0.0", tag="v1.0.0"):
    repo, source_sha = _candidate_repo(tmp_path, version)
    bundle, manifest = _bundle(tmp_path, repo, source_sha)
    plan_path = bundle / "release-plan.json"
    plan = prepare_release_plan(
        repo_root=repo,
        bundle_root=bundle,
        source_sha=source_sha,
        tag=tag,
        model_manifest_repo="owner/facetorch-model-manifest",
        model_manifest_revision=MODEL_REVISION,
        model_manifest_filename=MODEL_FILENAME,
        model_manifest_sha256=_sha256(manifest),
        cpu_image_digest=CPU_IMAGE_DIGEST,
        gpu_image_digest=GPU_IMAGE_DIGEST,
        output_path=plan_path,
        allow_missing_tag=True,
    )
    return plan, plan_path, bundle


def _public_payloads(bundle):
    return {
        "facetorch-1.0.0-py3-none-any.whl": (
            bundle / "distributions/facetorch-1.0.0-py3-none-any.whl"
        ),
        "facetorch-1.0.0.tar.gz": (
            bundle / "distributions/facetorch-1.0.0.tar.gz"
        ),
        "release-evidence.tar.zst": bundle / "release-evidence.tar.zst",
        "release-plan.json": bundle / "release-plan.json",
    }


def _github_release_asset_fixture(tmp_path):
    plan, plan_path, bundle = _release_plan(tmp_path)
    checksums = bundle / "SHA256SUMS"
    write_public_checksums(bundle, checksums)

    receipts = tmp_path / "receipts"
    receipts.mkdir()
    for channel in IMMUTABLE_CHANNELS:
        record_channel(
            plan,
            receipts / f"receipt-{channel}.json",
            channel,
            plan["channel_subjects"][channel],
        )
    publication_receipt = tmp_path / "publication-receipt.json"
    for channel in IMMUTABLE_CHANNELS:
        record_channel(
            plan,
            publication_receipt,
            channel,
            plan["channel_subjects"][channel],
        )

    expected_paths = [
        *sorted((bundle / "distributions").iterdir()),
        bundle / "release-evidence.tar.zst",
        plan_path,
        checksums,
        *(receipts / f"receipt-{channel}.json" for channel in IMMUTABLE_CHANNELS),
        publication_receipt,
    ]
    downloaded = tmp_path / "downloaded"
    downloaded.mkdir()
    for path in expected_paths:
        (downloaded / path.name).write_bytes(path.read_bytes())
    metadata = tmp_path / "release-assets.json"
    _write_json(
        metadata,
        {
            "assets": [
                {
                    "name": path.name,
                    "size": path.stat().st_size,
                    "digest": f"sha256:{_sha256(path)}",
                }
                for path in expected_paths
            ]
        },
    )
    return {
        "plan": plan,
        "plan_path": plan_path,
        "bundle": bundle,
        "receipts": receipts,
        "publication_receipt": publication_receipt,
        "downloaded": downloaded,
        "metadata": metadata,
    }


@pytest.mark.release_blocker
def test_release_tag_parser_rejects_shell_text_and_normalizes_rc():
    stable = parse_release_tag("v1.2.3")
    candidate = parse_release_tag("v1.2.3-rc.4")

    assert stable["project_version"] == "1.2.3"
    assert stable["is_prerelease"] is False
    assert candidate["project_version"] == "1.2.3rc4"
    assert candidate["docker_tag"] == "1.2.3-rc.4"
    assert candidate["is_prerelease"] is True
    for unsafe in (
        "v1.2.3;echo owned",
        "$(touch owned)",
        "v01.2.3",
        "v1.2",
        "v1.2.3-rc.0",
        "v1.2.3+mutable",
    ):
        with pytest.raises(ReleaseError):
            parse_release_tag(unsafe)


@pytest.mark.release_blocker
@pytest.mark.parametrize(
    "filename", ["../manifest.json", "manifest.json\nforged=true", "/manifest.json", "a//b"]
)
def test_model_manifest_filename_rejects_output_and_path_injection(tmp_path, filename):
    with pytest.raises(ReleaseError, match="safe relative path"):
        fetch_model_manifest(
            repo_id="owner/repository",
            revision=MODEL_REVISION,
            filename=filename,
            expected_sha256="0" * 64,
            output_path=tmp_path / "manifest.json",
        )


@pytest.mark.release_blocker
def test_fetched_model_manifest_preserves_its_remote_path(tmp_path):
    source = tmp_path / "source-manifest.json"
    _model_manifest(source)
    payload = source.read_bytes()

    report = fetch_model_manifest(
        repo_id="owner/repository",
        revision=MODEL_REVISION,
        filename=MODEL_FILENAME,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        output_path=tmp_path / "model-manifest.json",
        opener=lambda *_args, **_kwargs: io.BytesIO(payload),
    )

    assert report["filename"] == MODEL_FILENAME


@pytest.mark.release_blocker
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("filename", "manifests/other-plan.json"),
        ("plan_id", "other-publication-plan"),
        ("model_cohort_count", 2),
    ],
)
def test_release_plan_rejects_mismatched_manifest_report(tmp_path, field, value):
    repo, source_sha = _candidate_repo(tmp_path)
    bundle, manifest = _bundle(tmp_path, repo, source_sha)
    report_path = bundle / "evidence/model-manifest-report.json"
    report = json.loads(report_path.read_text())
    report[field] = value
    _write_json(report_path, report)

    with pytest.raises(ReleaseError, match="report disagrees"):
        prepare_release_plan(
            repo_root=repo,
            bundle_root=bundle,
            source_sha=source_sha,
            tag="v1.0.0",
            model_manifest_repo="owner/facetorch-model-manifest",
            model_manifest_revision=MODEL_REVISION,
            model_manifest_filename=MODEL_FILENAME,
            model_manifest_sha256=_sha256(manifest),
            cpu_image_digest=CPU_IMAGE_DIGEST,
            gpu_image_digest=GPU_IMAGE_DIGEST,
            output_path=bundle / "release-plan.json",
            allow_missing_tag=True,
        )


@pytest.mark.release_blocker
@pytest.mark.parametrize(
    ("field_path", "value"),
    [
        (("download_artifacts",), False),
        (("verify_legal_documents",), False),
        (("packaged_manifest_sha256",), "0" * 64),
        (("remote_manifest", "revision"), "0" * 40),
        (("results", 0, "revision"), "0" * 40),
        (("results", 0, "legal_documents", 0, "bytes_verified"), False),
        (("results", 0, "artifacts", 0, "downloaded_bytes_verified"), False),
        (("results", 0, "artifacts", 0, "metadata_sha256_verified"), False),
        (("results", 0, "artifacts", 0, "metadata_identity_verified"), False),
    ],
)
def test_release_plan_rejects_incomplete_or_cross_release_model_audit(
    tmp_path, field_path, value
):
    repo, source_sha = _candidate_repo(tmp_path)
    bundle, _manifest = _bundle(tmp_path, repo, source_sha)
    audit_path = bundle / "evidence/model-manifest-audit.json"
    audit = json.loads(audit_path.read_text())
    target = audit
    for field in field_path[:-1]:
        target = target[field]
    target[field_path[-1]] = value
    _write_json(audit_path, audit)
    remote_manifest = json.loads(
        (bundle / "evidence/model-manifest-report.json").read_text()
    )

    with pytest.raises(ReleaseError, match="Model audit"):
        validate_model_audit_report(
            repo,
            audit_path,
            remote_manifest=remote_manifest,
        )


@pytest.mark.release_blocker
def test_release_plan_binds_every_artifact_and_detects_changed_bytes(tmp_path):
    plan, plan_path, bundle = _release_plan(tmp_path)
    public_checksums = bundle / "SHA256SUMS"
    bundle_checksums = bundle / "BUNDLE-SHA256SUMS"
    write_public_checksums(bundle, public_checksums)
    write_bundle_checksums(bundle, bundle_checksums)

    assert set(plan["channel_subjects"]) == set(IMMUTABLE_CHANNELS)
    assert plan["model_manifest"]["filename"] == MODEL_FILENAME
    assert plan["model_audit"]["download_artifacts"] is True
    assert plan["model_audit"]["model_count"] == 1
    assert verify_release_plan(plan_path, bundle) == plan
    verify_public_checksums(
        public_checksums,
        payloads=_public_payloads(bundle),
    )
    verify_bundle_checksums(bundle, bundle_checksums)

    rogue = bundle / "distributions/unplanned.txt"
    rogue.write_bytes(b"not in the approved plan")
    with pytest.raises(ReleaseError, match="file set changed"):
        verify_release_plan(plan_path, bundle)
    with pytest.raises(ReleaseError, match="exact release bundle"):
        verify_bundle_checksums(bundle, bundle_checksums)
    rogue.unlink()

    wheel = bundle / "distributions/facetorch-1.0.0-py3-none-any.whl"
    wheel.write_bytes(b"different wheel")
    with pytest.raises(ReleaseError, match="changed after planning"):
        verify_release_plan(plan_path, bundle)
    with pytest.raises(ReleaseError, match="Public checksum mismatch"):
        verify_public_checksums(
            public_checksums,
            payloads=_public_payloads(bundle),
        )
    with pytest.raises(ReleaseError, match="Bundle checksum mismatch"):
        verify_bundle_checksums(bundle, bundle_checksums)


@pytest.mark.release_blocker
def test_internal_and_public_checksums_have_disjoint_explicit_scopes(tmp_path):
    _plan, _plan_path, bundle = _release_plan(tmp_path)
    public_path = bundle / "SHA256SUMS"
    bundle_path = bundle / "BUNDLE-SHA256SUMS"

    write_public_checksums(bundle, public_path)
    write_bundle_checksums(bundle, bundle_path)

    public_names = {
        line.split("  ", 1)[1]
        for line in public_path.read_text(encoding="utf-8").splitlines()
    }
    internal_names = {
        line.split("  ", 1)[1]
        for line in bundle_path.read_text(encoding="utf-8").splitlines()
    }
    assert public_names == set(_public_payloads(bundle))
    assert "SHA256SUMS" in internal_names
    assert "BUNDLE-SHA256SUMS" not in internal_names
    assert "evidence/model-manifest.json" in internal_names


@pytest.mark.release_blocker
@pytest.mark.parametrize(
    "mutation", ("missing", "extra", "duplicate", "renamed", "changed")
)
def test_public_checksum_verifier_rejects_contract_mutations(tmp_path, mutation):
    _plan, _plan_path, bundle = _release_plan(tmp_path)
    checksums = bundle / "SHA256SUMS"
    write_public_checksums(bundle, checksums)
    lines = checksums.read_text(encoding="utf-8").splitlines()

    if mutation == "missing":
        lines.pop(0)
    elif mutation == "extra":
        lines.append(f"{'0' * 64}  rogue.bin")
    elif mutation == "duplicate":
        lines.append(lines[0])
    elif mutation == "renamed":
        lines[0] = lines[0].replace(
            "facetorch-1.0.0-py3-none-any.whl", "renamed.whl"
        )
    else:
        digest, name = lines[0].split("  ", 1)
        lines[0] = f"{'f' if digest[0] != 'f' else 'e'}{digest[1:]}  {name}"
    checksums.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ReleaseError, match="checksum|payload|Duplicate"):
        verify_public_checksums(checksums, payloads=_public_payloads(bundle))


@pytest.mark.release_blocker
def test_github_release_assets_are_revalidated_immediately_before_publish(tmp_path):
    candidate = _github_release_asset_fixture(tmp_path)

    report = verify_github_release_assets(
        plan_path=candidate["plan_path"],
        bundle_root=candidate["bundle"],
        receipt_dir=candidate["receipts"],
        publication_receipt_path=candidate["publication_receipt"],
        asset_metadata_path=candidate["metadata"],
        downloaded_assets_dir=candidate["downloaded"],
    )

    assert report["status"] == "identical"
    assert {asset["name"] for asset in report["assets"]} == {
        "facetorch-1.0.0-py3-none-any.whl",
        "facetorch-1.0.0.tar.gz",
        "publication-receipt.json",
        "receipt-docker-cpu.json",
        "receipt-docker-gpu.json",
        "receipt-github-release.json",
        "receipt-model-manifest.json",
        "receipt-pypi.json",
        "release-evidence.tar.zst",
        "release-plan.json",
        "SHA256SUMS",
    }


@pytest.mark.release_blocker
@pytest.mark.parametrize("mutation", ("missing", "unexpected", "replaced"))
def test_github_release_asset_revalidation_rejects_draft_drift(tmp_path, mutation):
    candidate = _github_release_asset_fixture(tmp_path)
    metadata = json.loads(candidate["metadata"].read_text(encoding="utf-8"))
    wheel = candidate["downloaded"] / "facetorch-1.0.0-py3-none-any.whl"
    if mutation == "missing":
        wheel.unlink()
        metadata["assets"] = [
            asset for asset in metadata["assets"] if asset["name"] != wheel.name
        ]
    elif mutation == "unexpected":
        rogue = candidate["downloaded"] / "unapproved.bin"
        rogue.write_bytes(b"rogue")
        metadata["assets"].append(
            {
                "name": rogue.name,
                "size": rogue.stat().st_size,
                "digest": f"sha256:{_sha256(rogue)}",
            }
        )
    else:
        wheel.write_bytes(b"other")
        remote = next(
            asset for asset in metadata["assets"] if asset["name"] == wheel.name
        )
        remote["size"] = wheel.stat().st_size
        remote["digest"] = f"sha256:{_sha256(wheel)}"
    _write_json(candidate["metadata"], metadata)

    with pytest.raises(ReleaseError, match="asset set differs|asset digest differs"):
        verify_github_release_assets(
            plan_path=candidate["plan_path"],
            bundle_root=candidate["bundle"],
            receipt_dir=candidate["receipts"],
            publication_receipt_path=candidate["publication_receipt"],
            asset_metadata_path=candidate["metadata"],
            downloaded_assets_dir=candidate["downloaded"],
        )


@pytest.mark.release_blocker
def test_github_release_asset_revalidation_rejects_untrusted_receipts(tmp_path):
    candidate = _github_release_asset_fixture(tmp_path)
    receipt = candidate["receipts"] / "receipt-pypi.json"
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["channels"]["pypi"]["subject_digest"] = "0" * 64
    _write_json(receipt, payload)

    with pytest.raises(ReleaseError, match="invalid for pypi"):
        verify_github_release_assets(
            plan_path=candidate["plan_path"],
            bundle_root=candidate["bundle"],
            receipt_dir=candidate["receipts"],
            publication_receipt_path=candidate["publication_receipt"],
            asset_metadata_path=candidate["metadata"],
            downloaded_assets_dir=candidate["downloaded"],
        )


@pytest.mark.release_blocker
@pytest.mark.parametrize("top_level_incomplete", [True, False])
def test_packaged_model_governance_is_a_fail_closed_publication_gate(
    tmp_path, top_level_incomplete
):
    repo, _ = _candidate_repo(tmp_path)
    _, manifest = _bundle(tmp_path)
    governance_path = repo / "facetorch/models/governance.json"
    governance = json.loads(governance_path.read_text(encoding="utf-8"))
    if top_level_incomplete:
        governance["status"] = "incomplete"
    else:
        governance["models"]["detector"]["rights"]["owner_approval"] = "pending"
    _write_json(governance_path, governance)

    with pytest.raises(ReleaseError, match="governance is not approved|incomplete"):
        validate_packaged_model_governance(
            repo,
            remote_manifest_path=manifest,
            remote_revision=MODEL_REVISION,
        )


@pytest.mark.release_blocker
@pytest.mark.parametrize(
    ("mode", "value"),
    [
        ("missing", None),
        ("value", False),
        ("value", 1),
        ("value", "true"),
    ],
)
def test_packaged_governance_requires_exact_hosted_digest_proof(
    tmp_path,
    mode,
    value,
):
    repo, _ = _candidate_repo(tmp_path)
    _, manifest = _bundle(tmp_path)
    governance_path = repo / "facetorch/models/governance.json"
    governance = json.loads(governance_path.read_text(encoding="utf-8"))
    checkpoint = governance["models"]["detector"]["source_checkpoint"]
    if mode == "missing":
        checkpoint.pop("hosted_sha256_verified")
    else:
        checkpoint["hosted_sha256_verified"] = value
    _write_json(governance_path, governance)

    with pytest.raises(ReleaseError, match="governance is incomplete"):
        validate_packaged_model_governance(
            repo,
            remote_manifest_path=manifest,
            remote_revision=MODEL_REVISION,
        )


@pytest.mark.release_blocker
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repo_id", "owner/different"),
        ("cohort", "2.7"),
        ("artifact_size_bytes", 124),
        ("required_devices", ["cpu"]),
        ("artifact_filename", "different.pt2"),
        ("artifact_sha256", "9" * 64),
        ("metadata_filename", "different.meta.json"),
        ("metadata_sha256", "b" * 64),
        ("golden_reference_sha256", "b" * 64),
        ("golden_reference_size_bytes", GOLDEN_REFERENCE_SIZE + 1),
        ("golden_reference_source_cohort", "2.11"),
    ],
)
def test_model_governance_binds_every_remote_cohort_field(tmp_path, field, value):
    repo, _ = _candidate_repo(tmp_path)
    _, manifest = _bundle(tmp_path)
    remote = json.loads(manifest.read_text(encoding="utf-8"))
    remote["models"][0][field] = value
    _write_json(manifest, remote)

    with pytest.raises(ReleaseError, match="Remote|cohort coverage"):
        validate_packaged_model_governance(
            repo,
            remote_manifest_path=manifest,
            remote_revision=MODEL_REVISION,
        )


@pytest.mark.release_blocker
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_id", 1),
        ("cohort", 2.6),
        ("artifact_filename", 123),
        ("revision", int("5" * 40)),
        ("artifact_sha256", int("6" * 64)),
    ],
)
def test_model_governance_rejects_non_string_remote_identity_fields(
    tmp_path, field, value
):
    repo, _ = _candidate_repo(tmp_path)
    _, manifest = _bundle(tmp_path)
    remote = json.loads(manifest.read_text(encoding="utf-8"))
    remote["models"][0][field] = value
    _write_json(manifest, remote)

    with pytest.raises(ReleaseError, match="must be a non-empty string"):
        validate_packaged_model_governance(
            repo,
            remote_manifest_path=manifest,
            remote_revision=MODEL_REVISION,
        )


@pytest.mark.release_blocker
@pytest.mark.parametrize(
    ("maximum", "message"),
    [("2.8", "cohort range is invalid"), (2.7, "must be a non-empty string")],
)
def test_model_governance_rejects_invalid_packaged_cohort_ranges(
    tmp_path, maximum, message
):
    repo, _ = _candidate_repo(tmp_path)
    _, manifest = _bundle(tmp_path)
    packaged_path = repo / "facetorch/models/manifest.json"
    packaged = json.loads(packaged_path.read_text(encoding="utf-8"))
    packaged["models"]["detector"]["artifacts"][0]["torch_max_exclusive"] = maximum
    _write_json(packaged_path, packaged)

    with pytest.raises(ReleaseError, match=message):
        validate_packaged_model_governance(
            repo,
            remote_manifest_path=manifest,
            remote_revision=MODEL_REVISION,
        )


@pytest.mark.release_blocker
def test_model_governance_rejects_swapped_cohort_labels(tmp_path):
    repo, _ = _candidate_repo(tmp_path)
    _, manifest = _bundle(tmp_path)
    packaged_path = repo / "facetorch/models/manifest.json"
    compatibility_path = repo / "facetorch/models/compatibility.json"
    packaged = json.loads(packaged_path.read_text(encoding="utf-8"))
    compatibility = json.loads(compatibility_path.read_text(encoding="utf-8"))
    first_artifact = packaged["models"]["detector"]["artifacts"][0]
    second_artifact = json.loads(json.dumps(first_artifact))
    second_artifact.update(
        {
            "id": "detector-torch2.11",
            "filename": "model-torch2.11.pt2",
            "sha256": "9" * 64,
            "size_bytes": 456,
            "torch_min": "2.11",
            "torch_max_exclusive": "2.12",
            "schema_minor": 17,
            "validation_metadata": "model-torch2.11.pt2.meta.json",
        }
    )
    packaged["models"]["detector"]["artifacts"].append(second_artifact)
    compatibility["torch"]["supported_minor_lines"].append("2.11")
    _write_json(packaged_path, packaged)
    _write_json(compatibility_path, compatibility)

    remote = json.loads(manifest.read_text(encoding="utf-8"))
    second_remote = json.loads(json.dumps(remote["models"][0]))
    second_remote.update(
        {
            "cohort": "2.11",
            "artifact_filename": "model-torch2.11.pt2",
            "artifact_sha256": "9" * 64,
            "artifact_size_bytes": 456,
            "metadata_filename": "model-torch2.11.pt2.meta.json",
        }
    )
    remote["models"].append(second_remote)
    _write_json(manifest, remote)
    validate_packaged_model_governance(
        repo,
        remote_manifest_path=manifest,
        remote_revision=MODEL_REVISION,
    )

    remote["models"][0]["cohort"] = "2.11"
    remote["models"][1]["cohort"] = "2.6"
    _write_json(manifest, remote)
    with pytest.raises(ReleaseError, match="Remote cohort record differs"):
        validate_packaged_model_governance(
            repo,
            remote_manifest_path=manifest,
            remote_revision=MODEL_REVISION,
        )


@pytest.mark.release_blocker
def test_local_gpu_evidence_binds_exact_source_images_and_reports(tmp_path):
    repo, source_sha = _candidate_repo(tmp_path)
    bundle, _ = _bundle(tmp_path, repo, source_sha)
    evidence = bundle / "evidence/local-gpu"

    validated = validate_local_release_evidence(
        repo,
        evidence,
        source_sha=source_sha,
        cpu_image_digest=CPU_IMAGE_DIGEST,
        gpu_image_digest=GPU_IMAGE_DIGEST,
    )
    assert validated["status"] == "verified"

    smoke = evidence / "container-reports/gpu-image-smoke.json"
    value = json.loads(smoke.read_text(encoding="utf-8"))
    value["status"] = "skipped"
    _write_json(smoke, value)
    with pytest.raises(ReleaseError, match="production image evidence is invalid"):
        validate_local_release_evidence(
            repo,
            evidence,
            source_sha=source_sha,
            cpu_image_digest=CPU_IMAGE_DIGEST,
            gpu_image_digest=GPU_IMAGE_DIGEST,
        )


@pytest.mark.release_blocker
def test_publication_retry_accepts_identical_state_and_rejects_drift(tmp_path):
    plan, _, _ = _release_plan(tmp_path)
    receipt = tmp_path / "receipt.json"

    for channel in IMMUTABLE_CHANNELS:
        record_channel(plan, receipt, channel, plan["channel_subjects"][channel])
    assert verify_publication_receipt(plan, receipt)["status"] == "complete"

    record_channel(plan, receipt, "pypi", plan["channel_subjects"]["pypi"])
    with pytest.raises(ReleaseError, match="digest differs"):
        record_channel(plan, receipt, "pypi", "0" * 64)


class _FailureBackend:
    def __init__(self, *, existing=None, fail_once=None):
        self.remote = dict(existing or {})
        self.fail_once = fail_once
        self.failed = False
        self.published = []

    def observe(self, channel):
        return self.remote.get(channel)

    def publish(self, channel, expected_digest):
        if channel == self.fail_once and not self.failed:
            self.failed = True
            raise RuntimeError(f"injected {channel} failure")
        self.remote[channel] = expected_digest
        self.published.append(channel)
        return expected_digest


@pytest.mark.release_blocker
@pytest.mark.parametrize(
    "preexisting,failed_channel",
    [
        (("pypi",), "docker-cpu"),
        (("docker-cpu",), "docker-gpu"),
        (("github-release",), "pypi"),
    ],
)
def test_partial_publication_resumes_without_republishing(
    tmp_path, preexisting, failed_channel
):
    plan, _, _ = _release_plan(tmp_path)
    receipt = tmp_path / "receipt.json"
    remote = {channel: plan["channel_subjects"][channel] for channel in preexisting}
    for channel, digest in remote.items():
        record_channel(plan, receipt, channel, digest)
    backend = _FailureBackend(existing=remote, fail_once=failed_channel)

    with pytest.raises(RuntimeError, match="injected"):
        run_publication_transaction(plan, receipt, backend)

    completed = run_publication_transaction(plan, receipt, backend)
    assert completed["status"] == "complete"
    assert not set(preexisting).intersection(backend.published)


@pytest.mark.release_blocker
def test_rerun_stops_when_remote_version_has_different_bytes(tmp_path):
    plan, _, _ = _release_plan(tmp_path)
    receipt = tmp_path / "receipt.json"
    backend = _FailureBackend(existing={"docker-cpu": "sha256:" + "9" * 64})

    with pytest.raises(ReleaseError, match="digest differs"):
        run_publication_transaction(plan, receipt, backend)


@pytest.mark.release_blocker
def test_rc_never_promotes_latest_and_stable_requires_every_channel(tmp_path):
    rc_plan, _, _ = _release_plan(tmp_path / "rc", "1.0.0rc1", "v1.0.0-rc.1")
    rc_receipt = tmp_path / "rc-receipt.json"
    for channel in IMMUTABLE_CHANNELS:
        record_channel(
            rc_plan,
            rc_receipt,
            channel,
            rc_plan["channel_subjects"][channel],
        )
    with pytest.raises(ReleaseError, match="must never move"):
        assert_stable_alias_promotion(rc_plan, rc_receipt)

    stable_plan, _, _ = _release_plan(tmp_path / "stable")
    stable_receipt = tmp_path / "stable-receipt.json"
    record_channel(
        stable_plan,
        stable_receipt,
        "model-manifest",
        stable_plan["channel_subjects"]["model-manifest"],
    )
    with pytest.raises(ReleaseError, match="missing channels"):
        assert_stable_alias_promotion(stable_plan, stable_receipt)


@pytest.mark.release_blocker
def test_stable_alias_promotion_is_monotonic_and_idempotent(tmp_path):
    plan, _, _ = _release_plan(tmp_path)
    receipt = tmp_path / "receipt.json"
    for channel in IMMUTABLE_CHANNELS:
        record_channel(plan, receipt, channel, plan["channel_subjects"][channel])

    assert_stable_alias_promotion(
        plan, receipt, current_latest_tag="v0.9.9"
    )
    assert_stable_alias_promotion(
        plan, receipt, current_latest_tag="v1.0.0"
    )
    with pytest.raises(ReleaseError, match="move latest backward"):
        assert_stable_alias_promotion(
            plan, receipt, current_latest_tag="v1.0.1"
        )
    with pytest.raises(ReleaseError, match="stable release tag"):
        assert_stable_alias_promotion(
            plan, receipt, current_latest_tag="v1.1.0-rc.1"
        )


@pytest.mark.release_blocker
def test_pypi_and_docker_reconciliation_are_fail_closed(tmp_path):
    sdist = tmp_path / "facetorch-1.0.0.tar.gz"
    wheel = tmp_path / "facetorch-1.0.0-py3-none-any.whl"
    sdist.write_bytes(b"sdist")
    wheel.write_bytes(b"wheel")
    distributions = [sdist, wheel]
    remote = {
        "urls": [
            {
                "filename": distribution.name,
                "digests": {"sha256": _sha256(distribution)},
            }
            for distribution in distributions
        ]
    }

    missing = pypi_distribution_state(distributions, None)
    identical = pypi_distribution_state(distributions, remote)
    assert missing["status"] == "publish-required"
    assert identical["status"] == "identical"
    with pytest.raises(ReleaseError, match="different bytes"):
        pypi_distribution_state(
            distributions,
            {
                "urls": [
                    {
                        "filename": sdist.name,
                        "digests": {"sha256": "8" * 64},
                    },
                    remote["urls"][1],
                ]
            },
        )

    assert docker_distribution_state(None, CPU_IMAGE_DIGEST)["status"] == "publish-required"
    manifest = {"config": {"digest": CPU_IMAGE_DIGEST}}
    assert docker_distribution_state(manifest, CPU_IMAGE_DIGEST)["status"] == "identical"
    with pytest.raises(ReleaseError, match="Registry image differs"):
        docker_distribution_state(manifest, GPU_IMAGE_DIGEST)


@pytest.mark.release_blocker
def test_local_image_binding_rejects_a_mutable_version_tag_race():
    plan = {
        "channel_subjects": {
            "docker-cpu": CPU_IMAGE_DIGEST,
            "docker-gpu": GPU_IMAGE_DIGEST,
        }
    }
    initially_verified_remote = {"config": {"digest": CPU_IMAGE_DIGEST}}

    assert (
        docker_distribution_state(initially_verified_remote, CPU_IMAGE_DIGEST)[
            "status"
        ]
        == "identical"
    )
    with pytest.raises(ReleaseError, match="differs from the release plan"):
        validate_local_image_id(plan, "docker-cpu", GPU_IMAGE_DIGEST)

    assert validate_local_image_id(plan, "docker-cpu", CPU_IMAGE_DIGEST) == (
        CPU_IMAGE_DIGEST
    )
