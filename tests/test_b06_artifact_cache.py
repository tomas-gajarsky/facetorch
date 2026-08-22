import hashlib
import json
from importlib import resources
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from facetorch.artifacts import (
    ArtifactDescriptor,
    ArtifactManifest,
    get_model_manifest,
)
from facetorch.analyzer.utilizer.align import Lmk3DMeshPose
from facetorch.exceptions import (
    ArtifactIntegrityError,
    ConfigurationError,
    ModelCompatibilityError,
)
from facetorch.model_cache import (
    cleanup_quarantined_cache,
    inspect_incompatible_cache,
    inspect_quarantined_cache,
    migrate_legacy_artifact,
    plan_model_prefetch,
    prefetch_models,
    reset_incompatible_cache,
)


@pytest.mark.release_blocker
def test_packaged_manifest_covers_every_model_and_real_format():
    manifest = get_model_manifest()
    assert manifest.manifest_version == 1
    assert manifest.status == "provisional"
    assert len(manifest.models) == 10
    descriptors_per_model = len(manifest.supported_torch_minors) + 1
    assert len(tuple(manifest.iter_descriptors())) == (
        len(manifest.models) * descriptors_per_model
    )

    for model_id, artifacts in manifest.models.items():
        assert len(artifacts) == descriptors_per_model, model_id
        assert {item.format for item in artifacts} == {"pt2", "torchscript"}
        assert all(len(item.revision) == 40 for item in artifacts)
        assert all(item.size_bytes > 0 and len(item.sha256) == 64 for item in artifacts)
        assert all(
            item.filename.endswith(".pt2")
            for item in artifacts
            if item.format == "pt2"
        )
        assert all(
            item.filename.endswith(".pt")
            for item in artifacts
            if item.format == "torchscript"
        )


@pytest.mark.release_blocker
def test_incomplete_manifest_cannot_be_promoted_to_approved():
    raw = json.loads(
        resources.files("facetorch.models")
        .joinpath("manifest.json")
        .read_text(encoding="utf-8")
    )
    raw["status"] = "approved"
    with pytest.raises(ConfigurationError, match="complete provenance"):
        ArtifactManifest.from_mapping(raw)


@pytest.mark.release_blocker
def test_manifest_selection_is_exact_and_fails_for_undeclared_runtime():
    manifest = get_model_manifest()
    selected = manifest.candidates(
        "detector-retinaface",
        torch_version="2.11.0+cu130",
        device="cuda:0",
    )
    assert [item.artifact_id for item in selected] == [
        "detector-retinaface-torch2.11"
    ]

    with pytest.raises(ModelCompatibilityError, match="outside facetorch"):
        manifest.candidates(
            "detector-retinaface",
            torch_version="2.5.1",
            device="cpu",
        )


@pytest.mark.release_blocker
def test_au_legacy_artifact_is_cpu_only():
    legacy = next(
        item
        for item in get_model_manifest().models["au-opengraph"]
        if item.format == "torchscript"
    )
    assert legacy.devices == ("cpu",)


@pytest.mark.release_blocker
def test_prefetch_plan_matches_requested_runtime_components():
    torch_minor = ".".join(str(torch.__version__).split("+", 1)[0].split(".")[:2])
    fer = plan_model_prefetch(
        "cpu",
        include_predictors=["fer"],
        skip_detector=True,
        offline=True,
    )
    assert [(item.component, item.artifact_id) for item in fer.items] == [
        ("predictor.fer", f"fer-efficientnet-b2-torch{torch_minor}")
    ]

    align = plan_model_prefetch(
        "cpu",
        include_predictors=["align"],
        skip_detector=True,
        offline=True,
    )
    assert [item.component for item in align.items] == [
        "predictor.align",
        "align-metadata",
    ]


@pytest.mark.release_blocker
def test_empty_prefetch_selection_performs_no_network_or_instantiation():
    with patch("facetorch.model_cache.instantiate") as instantiate, patch(
        "facetorch.downloader.hf_hub_download"
    ) as hub, patch("facetorch.downloader.gdown.download") as drive:
        result = prefetch_models(
            "cpu",
            include_predictors=[],
            skip_detector=True,
            offline=False,
        )

    assert result.paths == ()
    instantiate.assert_not_called()
    hub.assert_not_called()
    drive.assert_not_called()


@pytest.mark.release_blocker
def test_prefetch_instantiates_only_requested_downloader(tmp_path):
    selected = []

    class FakeDownloader:
        def __init__(self, config):
            selected.append(str(config.manifest_id))
            self.path_local = str(tmp_path / f"{config.manifest_id}.pt2")

        def run(self):
            return self.path_local

    with patch(
        "facetorch.model_cache.instantiate",
        side_effect=lambda config: FakeDownloader(config),
    ), patch("facetorch.downloader.hf_hub_download") as hub, patch(
        "facetorch.downloader.gdown.download"
    ) as drive:
        result = prefetch_models(
            "cpu",
            include_predictors=["fer"],
            skip_detector=True,
            offline=False,
            confirm=True,
        )

    assert selected == ["fer-efficientnet-b2"]
    assert len(result.paths) == 1
    hub.assert_not_called()
    drive.assert_not_called()


@pytest.mark.release_blocker
def test_bulk_prefetch_requires_cost_confirmation_before_instantiation(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("FACETORCH_CACHE_DIR", str(tmp_path / "empty-cache"))
    with patch("facetorch.model_cache.instantiate") as instantiate:
        with pytest.raises(ConfigurationError, match="confirm=True"):
            prefetch_models(
                "cpu",
                include_predictors=["fer", "va"],
                skip_detector=True,
                offline=False,
            )
    instantiate.assert_not_called()


def _legacy_descriptor(path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return ArtifactDescriptor.from_mapping(
        "toy",
        {
            "task": "test",
            "source": "huggingface",
            "repo_id": "owner/toy",
            "revision": "a" * 40,
        },
        {
            "id": "toy-legacy",
            "filename": "model.pt",
            "format": "torchscript",
            "sha256": digest,
            "size_bytes": path.stat().st_size,
            "torch_min": "2.11",
            "torch_max_exclusive": "2.12",
            "devices": ["cpu"],
            "schema_major": None,
            "schema_minor": None,
            "validation_metadata": None,
        },
    )


@pytest.mark.release_blocker
def test_explicit_legacy_migration_verifies_and_never_mutates_source(tmp_path):
    source = tmp_path / "old-model.pt2"
    scripted = torch.jit.trace(torch.nn.Identity(), torch.ones(1))
    torch.jit.save(scripted, str(source))
    before = source.read_bytes()
    descriptor = _legacy_descriptor(source)

    class Manifest:
        def descriptor(self, artifact_id):
            assert artifact_id == "toy-legacy"
            return descriptor

    destination = tmp_path / "v1" / "model.pt"
    with patch("facetorch.model_cache.get_model_manifest", return_value=Manifest()):
        result = migrate_legacy_artifact(source, "toy-legacy", destination)

    assert result == destination
    assert destination.read_bytes() == before
    assert source.read_bytes() == before


@pytest.mark.release_blocker
def test_quarantine_cleanup_is_report_only_until_explicit_confirmation(
    tmp_path, monkeypatch
):
    model_root = tmp_path / "models" / "v1"
    metadata_root = tmp_path / "metadata" / "v1"
    monkeypatch.setenv("FACETORCH_MODEL_DIR", str(model_root))
    monkeypatch.setenv("FACETORCH_METADATA_DIR", str(metadata_root))
    quarantined = model_root / "detector" / "model.pt2.quarantine.1.deadbeef"
    quarantined.parent.mkdir(parents=True)
    quarantined.write_bytes(b"bad cache")

    report = cleanup_quarantined_cache(confirm=False)
    assert report.deleted is False
    assert report.paths == (quarantined,)
    assert quarantined.exists()

    deleted = cleanup_quarantined_cache(confirm=True)
    assert deleted.deleted is True
    assert not quarantined.exists()

    with pytest.raises(ConfigurationError, match="restricted"):
        inspect_quarantined_cache(tmp_path)


@pytest.mark.release_blocker
def test_incompatibility_reset_is_report_only_until_confirmed(tmp_path, monkeypatch):
    model_root = tmp_path / "models" / "v1"
    monkeypatch.setenv("FACETORCH_MODEL_DIR", str(model_root))
    sidecar = model_root / "predictor" / ".incompatible.json"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text('{"key": ["artifact"]}\n', encoding="utf-8")

    report = inspect_incompatible_cache()
    assert report.paths == (sidecar,)
    assert reset_incompatible_cache(confirm=False).deleted is False
    assert sidecar.exists()
    assert reset_incompatible_cache(confirm=True).deleted is True
    assert not sidecar.exists()

    with pytest.raises(ConfigurationError, match="restricted"):
        inspect_incompatible_cache(tmp_path)


@pytest.mark.release_blocker
def test_verified_alignment_metadata_uses_restricted_cpu_deserialization(tmp_path):
    metadata_path = tmp_path / "meta.pt"
    metadata_path.write_bytes(b"already verified by downloader")
    downloader = SimpleNamespace(
        path_local=str(metadata_path),
        verify_on_use=True,
        run=lambda: str(metadata_path),
    )
    metadata = {
        "keypoints": torch.tensor([0]),
        "param_mean": torch.zeros(62),
        "param_std": torch.ones(62),
        "u_exp": torch.zeros(3),
        "u_shp": torch.zeros(3),
        "w_exp": torch.zeros((3, 10)),
        "w_shp": torch.zeros((3, 40)),
    }

    with patch(
        "facetorch.analyzer.utilizer.align.torch.load", return_value=metadata
    ) as load:
        Lmk3DMeshPose(
            transform=None,
            device=torch.device("cpu"),
            optimize_transform=False,
            downloader_meta=downloader,
        )

    load.assert_called_once_with(
        str(metadata_path),
        map_location="cpu",
        weights_only=True,
    )


@pytest.mark.release_blocker
def test_invalid_verified_alignment_metadata_uses_public_integrity_error(tmp_path):
    metadata_path = tmp_path / "meta.pt"
    metadata_path.write_bytes(b"verified placeholder")
    downloader = SimpleNamespace(
        path_local=str(metadata_path),
        verify_on_use=False,
    )

    with patch("facetorch.analyzer.utilizer.align.torch.load", return_value={}):
        with pytest.raises(ArtifactIntegrityError, match="invalid structure"):
            Lmk3DMeshPose(
                transform=None,
                device=torch.device("cpu"),
                optimize_transform=False,
                downloader_meta=downloader,
            )
