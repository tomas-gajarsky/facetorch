import hashlib
import json
import os
from types import SimpleNamespace

import pytest

from scripts.run_local_cuda_release_matrix import (
    _ensure_evidence_root,
    _release_subprocess_environment,
)
import scripts.stage_alignment_metadata as alignment_metadata
import scripts.smoke_staged_default_analyzer as analyzer_smoke
from scripts.smoke_staged_default_analyzer import _staged_alignment_metadata


@pytest.mark.release_blocker
def test_new_evidence_root_is_0711_with_restrictive_umask(tmp_path):
    staging_root = tmp_path / "nested" / "evidence"
    previous_umask = os.umask(0o077)
    try:
        _ensure_evidence_root(staging_root)
    finally:
        os.umask(previous_umask)

    assert staging_root.stat().st_mode & 0o777 == 0o711
    _ensure_evidence_root(staging_root)


@pytest.mark.release_blocker
def test_existing_evidence_root_with_wrong_mode_fails_closed(tmp_path):
    staging_root = tmp_path / "evidence"
    staging_root.mkdir()
    staging_root.chmod(0o700)

    with pytest.raises(RuntimeError, match="must have mode 0711"):
        _ensure_evidence_root(staging_root)

    assert staging_root.stat().st_mode & 0o777 == 0o700


@pytest.mark.release_blocker
def test_release_smoke_environment_scrubs_external_metadata_routing(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/external/source")
    monkeypatch.setenv("FACETORCH_METADATA_DIR", "/external/metadata")
    monkeypatch.setenv("FACETORCH_TEST_SENTINEL", "preserved")

    environment = _release_subprocess_environment()

    assert "PYTHONPATH" not in environment
    assert "FACETORCH_METADATA_DIR" not in environment
    assert environment["FACETORCH_TEST_SENTINEL"] == "preserved"
    assert os.environ["FACETORCH_METADATA_DIR"] == "/external/metadata"


@pytest.mark.release_blocker
def test_candidate_smoke_uses_exact_pinned_artifact_root(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    model_root = repo_root / "facetorch/models"
    model_root.mkdir(parents=True)
    staging_root = tmp_path / "evidence"
    pinned_root = staging_root / "pinned-artifacts/torch-2.6"
    pinned_model_root = pinned_root / "test-model"
    pinned_model_root.mkdir(parents=True)
    artifact = pinned_model_root / "model-torch2.6.pt2"
    artifact.write_bytes(b"exact published artifact")
    artifact_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    metadata_value = {
        "model_id": "test-model",
        "artifact_sha256": artifact_sha256,
        "artifact_size_bytes": artifact.stat().st_size,
    }
    metadata = pinned_model_root / "model-torch2.6.pt2.meta.json"
    metadata.write_text(
        json.dumps(metadata_value, sort_keys=True),
        encoding="utf-8",
    )
    metadata_sha256 = hashlib.sha256(metadata.read_bytes()).hexdigest()

    decoy_root = staging_root / "torch-2.6/test-model"
    decoy_root.mkdir(parents=True)
    (decoy_root / artifact.name).write_bytes(b"fresh nondeterministic export")
    (decoy_root / metadata.name).write_text("{}", encoding="utf-8")
    manifest = {
        "compatibility_ref": "compatibility.json",
        "governance_ref": "governance.json",
        "models": {
            "test-model": {
                "artifacts": [
                    {
                        "format": "pt2",
                        "artifact_cohort": "2.6",
                        "filename": artifact.name,
                        "validation_metadata": metadata.name,
                        "sha256": artifact_sha256,
                        "size_bytes": artifact.stat().st_size,
                        "metadata_sha256": metadata_sha256,
                    }
                ]
            }
        },
    }
    for filename, value in (
        ("manifest.json", manifest),
        ("compatibility.json", {}),
        ("governance.json", {}),
    ):
        (model_root / filename).write_text(json.dumps(value), encoding="utf-8")
    summary = {
        "torch_minor": "2.6",
        "results": [
            {
                "model_id": "test-model",
                "status": "ok",
                "validation_status": "ok",
                "artifact": (
                    "/different/runner/evidence/pinned-artifacts/torch-2.6/"
                    "test-model/model-torch2.6.pt2"
                ),
                "meta": (
                    "/different/runner/evidence/runtime-validation/torch-2.6/"
                    "test-model/validation.meta.json"
                ),
                "sha256": artifact_sha256,
                "size_bytes": artifact.stat().st_size,
            }
        ],
    }
    monkeypatch.setattr(
        analyzer_smoke,
        "ArtifactManifest",
        SimpleNamespace(from_mapping=lambda value, **_kwargs: value),
    )

    candidate, staged_paths = analyzer_smoke._candidate_manifest(
        repo_root,
        staging_root,
        summary,
        pinned_artifacts_root=pinned_root,
    )

    assert candidate == manifest
    assert staged_paths == {"test-model": artifact.resolve()}
    metadata_value["substituted"] = True
    metadata.write_text(json.dumps(metadata_value, sort_keys=True), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Pinned manifest artifact binding failed"):
        analyzer_smoke._candidate_manifest(
            repo_root,
            staging_root,
            summary,
            pinned_artifacts_root=pinned_root,
        )


@pytest.mark.release_blocker
def test_alignment_metadata_is_staged_and_attested_for_offline_smokes(
    tmp_path, monkeypatch
):
    staging_root = tmp_path / "evidence"
    staging_root.mkdir()
    staging_root.chmod(0o711)
    payload = b"verified alignment metadata"
    digest = hashlib.sha256(payload).hexdigest()

    def fake_load_config(profile, *, offline):
        assert profile == "cpu"
        assert offline is False
        descriptor = SimpleNamespace(
            path_local=(
                os.path.join(os.environ["FACETORCH_METADATA_DIR"], "3dmm", "meta.pt")
            ),
            sha256=digest,
            size_bytes=len(payload),
            expected_format="torch_data",
            file_id="pinned-file-id",
            revision="pinned-revision",
            _target_="facetorch.downloader.DownloaderGDrive",
        )
        return SimpleNamespace(
            analyzer=SimpleNamespace(
                utilizer=SimpleNamespace(
                    align=SimpleNamespace(downloader_meta=descriptor)
                )
            )
        )

    class FakeDownloader:
        def __init__(self, descriptor):
            self.path = descriptor.path_local

        def run(self):
            with open(self.path, "wb") as output:
                output.write(payload)
            return self.path

    expected_report = {
        "schema_version": 1,
        "status": "ok",
        "artifact_id": "align-3dmm-metadata-v1",
        "source": "gdrive",
        "downloader": "facetorch.downloader.DownloaderGDrive",
        "file_id": "pinned-file-id",
        "revision": "pinned-revision",
        "expected_format": "torch_data",
        "staged_path": "runtime-inputs/3dmm/meta.pt",
        "size_bytes": len(payload),
        "sha256": digest,
    }

    monkeypatch.setenv("FACETORCH_METADATA_DIR", "preserved-value")
    monkeypatch.setattr(alignment_metadata.facetorch, "load_config", fake_load_config)
    monkeypatch.setattr(
        alignment_metadata, "instantiate", lambda value: FakeDownloader(value)
    )
    monkeypatch.setattr(
        alignment_metadata,
        "_alignment_metadata_contract",
        lambda: expected_report,
    )

    staged, report_path = alignment_metadata.stage_alignment_metadata(staging_root)

    assert os.environ["FACETORCH_METADATA_DIR"] == "preserved-value"
    assert staged == staging_root / "runtime-inputs" / "3dmm" / "meta.pt"
    assert staged.read_bytes() == payload
    assert staged.stat().st_mode & 0o777 == 0o644
    assert staged.parent.stat().st_mode & 0o777 == 0o755
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report == expected_report
    descriptor = fake_load_config(
        "cpu", offline=False
    ).analyzer.utilizer.align.downloader_meta
    assert _staged_alignment_metadata(staging_root, descriptor) == staged.resolve()

    report["staged_path"] = "../meta.pt"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(RuntimeError, match="report is not canonical"):
        _staged_alignment_metadata(staging_root, descriptor)


@pytest.mark.release_blocker
def test_alignment_metadata_staging_rejects_symlinked_input_root(tmp_path):
    staging_root = tmp_path / "evidence"
    staging_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (staging_root / "runtime-inputs").symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="contains a symlink"):
        alignment_metadata.stage_alignment_metadata(staging_root)
