"""Fail-closed edge cases for the public artifact trust contract."""

import hashlib
from zipfile import ZipFile

import pytest

from facetorch.artifacts import (
    ArtifactDescriptor,
    ArtifactManifest,
    ModelGovernance,
    detect_model_format,
    normalize_device,
    parse_runtime_version,
    verify_artifact,
)
from facetorch.exceptions import (
    ArtifactIntegrityError,
    ConfigurationError,
    ModelCompatibilityError,
)

pytestmark = pytest.mark.release_blocker


def _model(**overrides):
    value = {
        "task": "probe",
        "source": "huggingface",
        "repo_id": "owner/model",
        "revision": "a" * 40,
        "source_weight_sha256": "b" * 64,
        "export_commit": "c" * 40,
        "license_ref": "Apache-2.0",
    }
    value.update(overrides)
    return value


def _artifact(**overrides):
    value = {
        "id": "probe-torch2.11",
        "filename": "model-torch2.11.pt2",
        "format": "pt2",
        "sha256": "d" * 64,
        "size_bytes": 10,
        "torch_min": "2.11",
        "torch_max_exclusive": "2.12",
        "devices": ["cpu", "cuda"],
        "schema_major": 8,
        "schema_minor": 17,
        "validation_metadata": "model-torch2.11.pt2.meta.json",
    }
    value.update(overrides)
    return value


def _descriptor(**overrides):
    return ArtifactDescriptor.from_mapping("probe", _model(), _artifact(**overrides))


def _governance(**overrides):
    value = {
        "status": "incomplete",
        "release_eligible": False,
        "hosted_model_card": "https://huggingface.co/owner/model",
        "upstream_sources": [{"url": "https://example.test/source"}],
        "source_checkpoint": {
            "upstream_checkpoint_mapping": "unverified",
            "hosted_sha256_verified": False,
        },
        "rights": {
            "weights_license": "unverified",
            "redistribution": "pending",
            "attribution": "pending",
            "owner_approval": "pending",
        },
        "intended_use": ["testing"],
        "limitations": ["not validated"],
    }
    value.update(overrides)
    return value


def _manifest_raw(**overrides):
    value = {
        "manifest_version": 1,
        "manifest_revision": "probe-v1",
        "status": "provisional",
        "models": {"probe": {**_model(), "artifacts": [_artifact()]}},
    }
    value.update(overrides)
    return value


def _compatibility(**overrides):
    value = {
        "schema_version": 1,
        "status": "candidate",
        "torch": {
            "specifier": ">=2.11,<2.12",
            "supported_minor_lines": ["2.11"],
        },
        "python": {"specifier": ">=3.10,<3.13"},
        "platform_policy": {"required_devices": ["cpu", "cuda"]},
    }
    value.update(overrides)
    return value


def _governance_file(**overrides):
    value = {
        "schema_version": 1,
        "status": "incomplete",
        "models": {"probe": _governance()},
    }
    value.update(overrides)
    return value


def test_runtime_version_and_device_parsing_fail_closed():
    assert parse_runtime_version(" 2.11.0+cpu") == (2, 11)
    assert normalize_device("CUDA:0") == "cuda"
    with pytest.raises(ConfigurationError, match="parse runtime version"):
        parse_runtime_version("nightly")
    with pytest.raises(ConfigurationError, match="only cpu or cuda"):
        normalize_device("mps")


def test_descriptor_structure_errors_are_public_configuration_errors():
    with pytest.raises(ConfigurationError, match="Invalid artifact descriptor"):
        ArtifactDescriptor.from_mapping("probe", _model(), {"id": "missing"})


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"id": ""}, "cannot be empty"),
        ({}, "must pin a 40-character"),
        ({"filename": "../model.pt2"}, "must be a basename"),
        ({"format": "pickle"}, "unsupported format"),
        ({"sha256": "bad"}, "valid SHA-256"),
        ({"size_bytes": 0}, "valid SHA-256"),
        ({"devices": []}, "invalid device"),
        ({"devices": ["mps"]}, "invalid device"),
        (
            {"torch_min": "2.12", "torch_max_exclusive": "2.11"},
            "empty runtime range",
        ),
        ({"filename": "model.pt"}, "must preserve .pt2"),
        (
            {"format": "torchscript", "filename": "model.pt2"},
            "must preserve .pt",
        ),
    ],
)
def test_descriptor_validation_rejects_untrusted_metadata(overrides, message):
    model = _model()
    if overrides == {}:
        model["revision"] = "mutable-main"
    with pytest.raises(ConfigurationError, match=message):
        ArtifactDescriptor.from_mapping("probe", model, _artifact(**overrides))


def test_descriptor_support_and_cache_path_are_exact(tmp_path):
    descriptor = _descriptor()
    assert descriptor.supports((2, 11), "cpu")
    assert not descriptor.supports((2, 10), "cpu")
    assert not descriptor.supports((2, 12), "cpu")
    assert descriptor.cache_path(tmp_path / "configured.pt2") == (
        tmp_path / descriptor.filename
    )


def test_governance_structure_and_required_fields_fail_closed():
    with pytest.raises(ConfigurationError, match="Invalid governance record"):
        ModelGovernance.from_mapping("probe", {})

    invalid_records = [
        ({"status": "draft"}, "Invalid governance status"),
        ({"release_eligible": "yes"}, "must be boolean"),
        ({"hosted_model_card": "http://example.test"}, "HTTPS model-card"),
        ({"upstream_sources": []}, "incomplete provenance"),
        ({"intended_use": []}, "incomplete provenance"),
        ({"limitations": []}, "incomplete provenance"),
        ({"rights": {}}, "incomplete rights"),
    ]
    for overrides, message in invalid_records:
        with pytest.raises(ConfigurationError, match=message):
            ModelGovernance.from_mapping("probe", _governance(**overrides))


def test_governance_approval_requires_every_right_and_provenance_field():
    assert ModelGovernance.from_mapping("probe", _governance()).approved is False
    approved = ModelGovernance.from_mapping(
        "probe",
        _governance(
            status="approved",
            release_eligible=True,
            source_checkpoint={
                "upstream_checkpoint_mapping": "verified",
                "hosted_sha256_verified": True,
            },
            rights={
                "weights_license": "Apache-2.0",
                "redistribution": "approved",
                "attribution": "approved",
                "owner_approval": "approved",
            },
        ),
    )
    assert approved.approved is True


@pytest.mark.parametrize(
    "source_checkpoint",
    [
        {"upstream_checkpoint_mapping": "verified"},
        {
            "upstream_checkpoint_mapping": "verified",
            "hosted_sha256_verified": False,
        },
        {
            "upstream_checkpoint_mapping": "verified",
            "hosted_sha256_verified": 1,
        },
        {
            "upstream_checkpoint_mapping": "verified",
            "hosted_sha256_verified": "true",
        },
    ],
)
def test_governance_approval_requires_exact_hosted_digest_proof(source_checkpoint):
    governance = ModelGovernance.from_mapping(
        "probe",
        _governance(
            status="approved",
            release_eligible=True,
            source_checkpoint=source_checkpoint,
            rights={
                "weights_license": "Apache-2.0",
                "redistribution": "approved",
                "attribution": "approved",
                "owner_approval": "approved",
            },
        ),
    )

    assert governance.approved is False


def test_manifest_constructor_rejects_invalid_metadata_duplicates_and_matrix():
    descriptor = _descriptor()
    with pytest.raises(ConfigurationError, match="Unsupported model manifest version"):
        ArtifactManifest(
            manifest_version=2,
            manifest_revision="probe",
            status="provisional",
            models={"probe": (descriptor,)},
        )
    with pytest.raises(ConfigurationError, match="metadata is incomplete"):
        ArtifactManifest(
            manifest_version=1,
            manifest_revision="",
            status="provisional",
            models={"probe": (descriptor,)},
        )
    with pytest.raises(ConfigurationError, match="globally unique"):
        ArtifactManifest(
            manifest_version=1,
            manifest_revision="probe",
            status="provisional",
            models={"one": (descriptor,), "two": (descriptor,)},
        )
    with pytest.raises(ConfigurationError, match="exactly one artifact"):
        ArtifactManifest(
            manifest_version=1,
            manifest_revision="probe",
            status="provisional",
            models={"probe": (descriptor,)},
            supported_torch_minors=("2.6",),
            required_devices=("cpu", "cuda"),
        )


def test_manifest_mapping_rejects_incomplete_compatibility_and_governance():
    empty_model = {**_model(), "artifacts": []}
    with pytest.raises(ConfigurationError, match="Every manifest model"):
        ArtifactManifest.from_mapping(_manifest_raw(models={"probe": empty_model}))
    with pytest.raises(ConfigurationError, match="Unsupported compatibility schema"):
        ArtifactManifest.from_mapping(
            _manifest_raw(), compatibility=_compatibility(schema_version=2)
        )
    with pytest.raises(ConfigurationError, match="Invalid compatibility status"):
        ArtifactManifest.from_mapping(
            _manifest_raw(), compatibility=_compatibility(status="draft")
        )

    incomplete = _compatibility()
    incomplete["torch"]["supported_minor_lines"] = []
    with pytest.raises(ConfigurationError, match="matrix is incomplete"):
        ArtifactManifest.from_mapping(_manifest_raw(), compatibility=incomplete)

    duplicate = _compatibility()
    duplicate["torch"]["supported_minor_lines"] = ["2.11", "2.11"]
    with pytest.raises(ConfigurationError, match="matrix is incomplete"):
        ArtifactManifest.from_mapping(_manifest_raw(), compatibility=duplicate)

    invalid_device = _compatibility()
    invalid_device["platform_policy"]["required_devices"] = ["mps"]
    with pytest.raises(ConfigurationError, match="matrix is incomplete"):
        ArtifactManifest.from_mapping(_manifest_raw(), compatibility=invalid_device)

    with pytest.raises(ConfigurationError, match="Unsupported governance schema"):
        ArtifactManifest.from_mapping(
            _manifest_raw(), governance=_governance_file(schema_version=2)
        )
    with pytest.raises(ConfigurationError, match="Invalid governance status"):
        ArtifactManifest.from_mapping(
            _manifest_raw(), governance=_governance_file(status="draft")
        )
    with pytest.raises(ConfigurationError, match="exactly cover"):
        ArtifactManifest.from_mapping(
            _manifest_raw(), governance=_governance_file(models={})
        )
    with pytest.raises(ConfigurationError, match="Invalid model manifest structure"):
        ArtifactManifest.from_mapping({"models": []})


def test_manifest_json_lookup_and_candidate_errors_are_actionable():
    with pytest.raises(ConfigurationError, match="not valid JSON"):
        ArtifactManifest.from_json("{")
    with pytest.raises(ConfigurationError, match="root must be an object"):
        ArtifactManifest.from_json("[]")

    manifest = ArtifactManifest.from_mapping(_manifest_raw())
    with pytest.raises(ConfigurationError, match="Unknown artifact ID"):
        manifest.descriptor("missing")
    with pytest.raises(ConfigurationError, match="Unknown manifest model"):
        manifest.candidates("missing", torch_version="2.11", device="cpu")
    with pytest.raises(ModelCompatibilityError, match="publish a validated cohort"):
        manifest.candidates("probe", torch_version="2.6", device="cpu")
    assert [item.artifact_id for item in manifest.iter_descriptors()] == [
        "probe-torch2.11"
    ]


def test_legacy_only_candidate_explains_the_explicit_opt_in():
    legacy = _artifact(
        id="probe-legacy",
        filename="model.pt",
        format="torchscript",
        torch_min=None,
        torch_max_exclusive=None,
        devices=["cpu"],
        schema_major=None,
        schema_minor=None,
        validation_metadata=None,
    )
    raw = _manifest_raw(models={"probe": {**_model(), "artifacts": [legacy]}})
    manifest = ArtifactManifest.from_mapping(raw)
    with pytest.raises(ModelCompatibilityError, match="allow_legacy_models=True"):
        manifest.candidates("probe", torch_version="2.11", device="cpu")
    assert (
        manifest.candidates(
            "probe",
            torch_version="2.11",
            device="cpu",
            allow_legacy_models=True,
        )[0].format
        == "torchscript"
    )


def test_archive_detection_and_integrity_failures_never_deserialize(tmp_path):
    invalid = tmp_path / "not-an-archive.pt2"
    invalid.write_bytes(b"not zip data")
    assert detect_model_format(invalid) == "unknown"

    unknown = tmp_path / "unknown.pt2"
    with ZipFile(unknown, "w") as archive:
        archive.writestr("README", "unknown")
    assert detect_model_format(unknown) == "unknown"

    exported = tmp_path / "model.pt2"
    with ZipFile(exported, "w") as archive:
        archive.writestr("serialized_exported_program.json", "{}")
    digest = hashlib.sha256(exported.read_bytes()).hexdigest()
    descriptor = _descriptor(sha256=digest, size_bytes=exported.stat().st_size)
    assert verify_artifact(exported, descriptor) == exported

    with pytest.raises(ArtifactIntegrityError, match="Cannot inspect"):
        verify_artifact(tmp_path / "missing.pt2", descriptor)
    with pytest.raises(ArtifactIntegrityError, match="has size"):
        verify_artifact(invalid, descriptor)

    wrong_digest = _descriptor(
        sha256="0" * 64,
        size_bytes=exported.stat().st_size,
    )
    with pytest.raises(ArtifactIntegrityError, match="SHA-256"):
        verify_artifact(exported, wrong_digest)

    torchscript = tmp_path / "script.pt2"
    with ZipFile(torchscript, "w") as archive:
        archive.writestr("archive/data.pkl", "data")
        archive.writestr("archive/code/module.py", "pass")
    wrong_format = _descriptor(
        sha256=hashlib.sha256(torchscript.read_bytes()).hexdigest(),
        size_bytes=torchscript.stat().st_size,
    )
    with pytest.raises(ArtifactIntegrityError, match="has format"):
        verify_artifact(torchscript, wrong_format)
