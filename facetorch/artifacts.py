"""Immutable model-manifest resolution and cache-integrity primitives."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from importlib import resources
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Optional
from zipfile import BadZipFile, ZipFile

from facetorch.exceptions import (
    ArtifactIntegrityError,
    ConfigurationError,
    ModelCompatibilityError,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_VERSION_RE = re.compile(r"^(\d+)\.(\d+)")
_FORMATS = {"pt2", "torchscript", "torch_data"}
_MANIFEST_STATUSES = {"provisional", "approved"}
_COMPATIBILITY_STATUSES = {"candidate", "approved"}
_GOVERNANCE_STATUSES = {"incomplete", "approved"}


def parse_runtime_version(value: str) -> tuple[int, int]:
    """Parse a runtime version into a comparable major/minor tuple."""
    match = _VERSION_RE.match(str(value).strip())
    if not match:
        raise ConfigurationError(f"Cannot parse runtime version {value!r}.")
    return int(match.group(1)), int(match.group(2))


def normalize_device(value: Any) -> str:
    """Normalize torch-style device values to their manifest device family."""
    device = str(value).strip().lower().split(":", 1)[0]
    if device not in {"cpu", "cuda"}:
        raise ConfigurationError(
            f"Model artifacts support only cpu or cuda, got {value!r}."
        )
    return device


def _version_bound(value: Optional[str]) -> Optional[tuple[int, int]]:
    return None if value is None else parse_runtime_version(value)


@dataclass(frozen=True)
class ArtifactDescriptor:
    """One immutable, hash-addressed remote artifact candidate."""

    artifact_id: str
    model_id: str
    task: str
    source: str
    repo_id: str
    revision: str
    filename: str
    format: str
    sha256: str
    size_bytes: int
    torch_min: Optional[str]
    torch_max_exclusive: Optional[str]
    devices: tuple[str, ...]
    schema_major: Optional[int]
    schema_minor: Optional[int]
    validation_metadata: Optional[str]
    source_weight_sha256: Optional[str]
    export_commit: Optional[str]
    license_ref: Optional[str]
    priority: int = 0

    @classmethod
    def from_mapping(
        cls,
        model_id: str,
        model: Mapping[str, Any],
        raw: Mapping[str, Any],
    ) -> "ArtifactDescriptor":
        try:
            descriptor = cls(
                artifact_id=str(raw["id"]),
                model_id=model_id,
                task=str(model["task"]),
                source=str(model.get("source", "huggingface")),
                repo_id=str(model["repo_id"]),
                revision=str(model["revision"]),
                filename=str(raw["filename"]),
                format=str(raw["format"]),
                sha256=str(raw["sha256"]).lower(),
                size_bytes=int(raw["size_bytes"]),
                torch_min=raw.get("torch_min"),
                torch_max_exclusive=raw.get("torch_max_exclusive"),
                devices=tuple(str(item).lower() for item in raw["devices"]),
                schema_major=raw.get("schema_major"),
                schema_minor=raw.get("schema_minor"),
                validation_metadata=raw.get("validation_metadata"),
                source_weight_sha256=model.get("source_weight_sha256"),
                export_commit=model.get("export_commit"),
                license_ref=model.get("license_ref"),
                priority=int(raw.get("priority", 0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"Invalid artifact descriptor for model {model_id!r}."
            ) from exc
        descriptor.validate()
        return descriptor

    def validate(self) -> None:
        if not self.artifact_id or not self.model_id or not self.task:
            raise ConfigurationError("Artifact identifiers and task cannot be empty.")
        if self.source == "huggingface" and not _REVISION_RE.fullmatch(self.revision):
            raise ConfigurationError(
                f"Artifact {self.artifact_id!r} must pin a 40-character Hub commit."
            )
        if Path(self.filename).name != self.filename:
            raise ConfigurationError(
                f"Artifact filename must be a basename: {self.filename!r}."
            )
        if self.format not in _FORMATS:
            raise ConfigurationError(
                f"Artifact {self.artifact_id!r} has unsupported format {self.format!r}."
            )
        if not _SHA256_RE.fullmatch(self.sha256) or self.size_bytes <= 0:
            raise ConfigurationError(
                f"Artifact {self.artifact_id!r} needs a valid SHA-256 and size."
            )
        if not self.devices or any(item not in {"cpu", "cuda"} for item in self.devices):
            raise ConfigurationError(
                f"Artifact {self.artifact_id!r} has invalid device eligibility."
            )
        lower = _version_bound(self.torch_min)
        upper = _version_bound(self.torch_max_exclusive)
        if lower is not None and upper is not None and lower >= upper:
            raise ConfigurationError(
                f"Artifact {self.artifact_id!r} has an empty runtime range."
            )
        if self.format == "pt2" and not self.filename.endswith(".pt2"):
            raise ConfigurationError(
                f"Export artifact {self.artifact_id!r} must preserve .pt2."
            )
        if self.format == "torchscript" and not self.filename.endswith(".pt"):
            raise ConfigurationError(
                f"TorchScript artifact {self.artifact_id!r} must preserve .pt."
            )

    def supports(self, runtime: tuple[int, int], device: str) -> bool:
        lower = _version_bound(self.torch_min)
        upper = _version_bound(self.torch_max_exclusive)
        return (
            device in self.devices
            and (lower is None or runtime >= lower)
            and (upper is None or runtime < upper)
        )

    def cache_path(self, configured_path: str | Path) -> Path:
        """Place each candidate under its authenticated real filename."""
        return Path(configured_path).expanduser().parent / self.filename


@dataclass(frozen=True)
class ModelGovernance:
    """Release-eligibility and responsible-use record for one hosted model."""

    model_id: str
    status: str
    release_eligible: bool
    hosted_model_card: str
    upstream_sources: tuple[Mapping[str, Any], ...]
    source_checkpoint: Mapping[str, Any]
    rights: Mapping[str, Any]
    intended_use: tuple[str, ...]
    limitations: tuple[str, ...]

    @classmethod
    def from_mapping(
        cls, model_id: str, raw: Mapping[str, Any]
    ) -> "ModelGovernance":
        try:
            record = cls(
                model_id=model_id,
                status=str(raw["status"]),
                release_eligible=raw["release_eligible"],
                hosted_model_card=str(raw["hosted_model_card"]),
                upstream_sources=tuple(raw["upstream_sources"]),
                source_checkpoint=dict(raw["source_checkpoint"]),
                rights=dict(raw["rights"]),
                intended_use=tuple(str(item) for item in raw["intended_use"]),
                limitations=tuple(str(item) for item in raw["limitations"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"Invalid governance record for model {model_id!r}."
            ) from exc
        record.validate()
        return record

    def validate(self) -> None:
        if self.status not in _GOVERNANCE_STATUSES:
            raise ConfigurationError(
                f"Invalid governance status for model {self.model_id!r}."
            )
        if not isinstance(self.release_eligible, bool):
            raise ConfigurationError(
                f"release_eligible must be boolean for model {self.model_id!r}."
            )
        if not self.hosted_model_card.startswith("https://"):
            raise ConfigurationError(
                f"Model {self.model_id!r} needs an HTTPS model-card reference."
            )
        if not self.upstream_sources or not self.intended_use or not self.limitations:
            raise ConfigurationError(
                f"Model {self.model_id!r} has incomplete provenance or limitations."
            )
        required_rights = {
            "weights_license",
            "redistribution",
            "attribution",
            "owner_approval",
        }
        if not required_rights.issubset(self.rights):
            raise ConfigurationError(
                f"Model {self.model_id!r} has an incomplete rights record."
            )

    @property
    def approved(self) -> bool:
        return (
            self.status == "approved"
            and self.release_eligible
            and self.source_checkpoint.get("upstream_checkpoint_mapping")
            == "verified"
            and self.rights.get("weights_license") not in {None, "unverified"}
            and self.rights.get("redistribution") == "approved"
            and self.rights.get("attribution") == "approved"
            and self.rights.get("owner_approval") == "approved"
        )


class ArtifactManifest:
    """Validated in-memory view of the packaged model manifest."""

    def __init__(
        self,
        *,
        manifest_version: int,
        manifest_revision: str,
        status: str,
        models: Mapping[str, tuple[ArtifactDescriptor, ...]],
        compatibility_status: str = "unspecified",
        torch_specifier: str = "",
        python_specifier: str = "",
        supported_torch_minors: tuple[str, ...] = (),
        required_devices: tuple[str, ...] = (),
        governance_status: str = "unspecified",
        governance: Mapping[str, ModelGovernance] | None = None,
    ) -> None:
        if manifest_version != 1:
            raise ConfigurationError(
                f"Unsupported model manifest version {manifest_version!r}."
            )
        if not manifest_revision or status not in _MANIFEST_STATUSES:
            raise ConfigurationError("Model manifest metadata is incomplete.")
        self.manifest_version = manifest_version
        self.manifest_revision = manifest_revision
        self.status = status
        self.models = dict(models)
        self.compatibility_status = compatibility_status
        self.torch_specifier = torch_specifier
        self.python_specifier = python_specifier
        self.supported_torch_minors = supported_torch_minors
        self.required_devices = required_devices
        self.governance_status = governance_status
        self.governance = dict(governance or {})
        self._by_artifact_id = {
            artifact.artifact_id: artifact
            for artifacts in self.models.values()
            for artifact in artifacts
        }
        expected = sum(len(items) for items in self.models.values())
        if len(self._by_artifact_id) != expected:
            raise ConfigurationError("Artifact IDs must be globally unique.")
        if self.supported_torch_minors:
            for model_id, artifacts in self.models.items():
                for minor in self.supported_torch_minors:
                    runtime = parse_runtime_version(minor)
                    matching = [
                        artifact
                        for artifact in artifacts
                        if artifact.format == "pt2"
                        and all(
                            artifact.supports(runtime, device)
                            for device in self.required_devices
                        )
                    ]
                    if len(matching) != 1:
                        raise ConfigurationError(
                            f"Model {model_id!r} needs exactly one artifact for "
                            f"supported torch {minor}; found {len(matching)}."
                        )
        if status == "approved":
            incomplete = [
                artifact.artifact_id
                for artifact in self._by_artifact_id.values()
                if not artifact.source_weight_sha256
                or not _SHA256_RE.fullmatch(artifact.source_weight_sha256)
                or not artifact.export_commit
                or not _REVISION_RE.fullmatch(artifact.export_commit)
                or (artifact.format == "pt2" and not artifact.validation_metadata)
                or not artifact.license_ref
            ]
            governance_incomplete = sorted(
                model_id
                for model_id in self.models
                if model_id not in self.governance
                or not self.governance[model_id].approved
            )
            if (
                incomplete
                or governance_incomplete
                or self.compatibility_status != "approved"
                or self.governance_status != "approved"
            ):
                raise ConfigurationError(
                    "An approved manifest requires complete provenance, validation, "
                    "rights, and compatibility metadata; incomplete artifacts: "
                    f"{', '.join(incomplete) or 'none'}; incomplete models: "
                    f"{', '.join(governance_incomplete) or 'none'}."
                )

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        compatibility: Mapping[str, Any] | None = None,
        governance: Mapping[str, Any] | None = None,
    ) -> "ArtifactManifest":
        try:
            raw_models = raw["models"]
            models = {
                model_id: tuple(
                    ArtifactDescriptor.from_mapping(model_id, model, artifact)
                    for artifact in model["artifacts"]
                )
                for model_id, model in raw_models.items()
            }
            if not models or any(not artifacts for artifacts in models.values()):
                raise ConfigurationError("Every manifest model needs an artifact.")

            compatibility_status = "unspecified"
            torch_specifier = ""
            python_specifier = ""
            supported_torch_minors: tuple[str, ...] = ()
            required_devices: tuple[str, ...] = ()
            if compatibility is not None:
                if int(compatibility["schema_version"]) != 1:
                    raise ConfigurationError("Unsupported compatibility schema.")
                compatibility_status = str(compatibility["status"])
                if compatibility_status not in _COMPATIBILITY_STATUSES:
                    raise ConfigurationError("Invalid compatibility status.")
                torch_record = compatibility["torch"]
                python_record = compatibility["python"]
                platform_policy = compatibility["platform_policy"]
                torch_specifier = str(torch_record["specifier"])
                python_specifier = str(python_record["specifier"])
                supported_torch_minors = tuple(
                    str(item) for item in torch_record["supported_minor_lines"]
                )
                required_devices = tuple(
                    str(item) for item in platform_policy["required_devices"]
                )
                if (
                    not supported_torch_minors
                    or len(set(supported_torch_minors))
                    != len(supported_torch_minors)
                    or not required_devices
                    or any(item not in {"cpu", "cuda"} for item in required_devices)
                ):
                    raise ConfigurationError("Compatibility matrix is incomplete.")

            governance_status = "unspecified"
            governance_records: dict[str, ModelGovernance] = {}
            if governance is not None:
                if int(governance["schema_version"]) != 1:
                    raise ConfigurationError("Unsupported governance schema.")
                governance_status = str(governance["status"])
                if governance_status not in _GOVERNANCE_STATUSES:
                    raise ConfigurationError("Invalid governance status.")
                governance_records = {
                    model_id: ModelGovernance.from_mapping(model_id, record)
                    for model_id, record in governance["models"].items()
                }
                if set(governance_records) != set(models):
                    raise ConfigurationError(
                        "Governance records must exactly cover manifest models."
                    )
            return cls(
                manifest_version=int(raw["manifest_version"]),
                manifest_revision=str(raw["manifest_revision"]),
                status=str(raw["status"]),
                models=models,
                compatibility_status=compatibility_status,
                torch_specifier=torch_specifier,
                python_specifier=python_specifier,
                supported_torch_minors=supported_torch_minors,
                required_devices=required_devices,
                governance_status=governance_status,
                governance=governance_records,
            )
        except ConfigurationError:
            raise
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError("Invalid model manifest structure.") from exc

    @classmethod
    def from_json(cls, value: str) -> "ArtifactManifest":
        try:
            raw = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ConfigurationError("Model manifest is not valid JSON.") from exc
        if not isinstance(raw, Mapping):
            raise ConfigurationError("Model manifest root must be an object.")
        return cls.from_mapping(raw)

    def descriptor(self, artifact_id: str) -> ArtifactDescriptor:
        try:
            return self._by_artifact_id[artifact_id]
        except KeyError as exc:
            raise ConfigurationError(f"Unknown artifact ID {artifact_id!r}.") from exc

    def candidates(
        self,
        model_id: str,
        *,
        torch_version: str,
        device: Any,
        allow_legacy_models: bool = False,
    ) -> tuple[ArtifactDescriptor, ...]:
        try:
            artifacts = self.models[model_id]
        except KeyError as exc:
            raise ConfigurationError(f"Unknown manifest model {model_id!r}.") from exc
        runtime = parse_runtime_version(torch_version)
        device_family = normalize_device(device)
        runtime_label = f"{runtime[0]}.{runtime[1]}"
        if (
            self.supported_torch_minors
            and runtime_label not in self.supported_torch_minors
        ):
            supported = ", ".join(self.supported_torch_minors)
            raise ModelCompatibilityError(
                f"Torch {runtime_label} is outside facetorch's supported minor "
                f"lines ({supported}); no model download was attempted."
            )
        exports = [
            item
            for item in artifacts
            if item.format == "pt2" and item.supports(runtime, device_family)
        ]
        legacy = [
            item
            for item in artifacts
            if item.format == "torchscript" and item.supports(runtime, device_family)
        ]
        exports.sort(key=lambda item: item.priority)
        legacy.sort(key=lambda item: item.priority)
        selected = exports + (legacy if allow_legacy_models else [])
        if selected:
            return tuple(selected)

        if legacy and not allow_legacy_models:
            remedy = " Set allow_legacy_models=True to use an eligible verified legacy artifact."
        else:
            remedy = " Install a documented supported runtime or publish a validated cohort."
        raise ModelCompatibilityError(
            f"No compatible artifact for {model_id!r} with torch {runtime_label} "
            f"on {device_family}.{remedy}"
        )

    def iter_descriptors(self) -> Iterable[ArtifactDescriptor]:
        for model_id in sorted(self.models):
            yield from self.models[model_id]


@lru_cache(maxsize=1)
def get_model_manifest() -> ArtifactManifest:
    """Load and validate the immutable manifest packaged with facetorch."""
    model_resources = resources.files("facetorch.models")
    manifest_file = model_resources.joinpath("manifest.json")
    raw = json.loads(manifest_file.read_text(encoding="utf-8"))

    def referenced_json(field: str) -> Mapping[str, Any]:
        filename = str(raw[field])
        if Path(filename).name != filename or not filename.endswith(".json"):
            raise ConfigurationError(f"Invalid manifest resource reference: {filename!r}")
        value = json.loads(
            model_resources.joinpath(filename).read_text(encoding="utf-8")
        )
        if not isinstance(value, Mapping):
            raise ConfigurationError(f"Manifest resource {filename!r} is not an object.")
        return value

    return ArtifactManifest.from_mapping(
        raw,
        compatibility=referenced_json("compatibility_ref"),
        governance=referenced_json("governance_ref"),
    )


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Stream a file SHA-256 without loading the artifact into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as artifact_file:
        while chunk := artifact_file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def detect_model_format(path: str | Path) -> str:
    """Identify modern export versus TorchScript by archive members, without execution."""
    try:
        with ZipFile(path) as archive:
            names = tuple(archive.namelist())
    except (BadZipFile, OSError):
        return "unknown"
    if any(name.endswith("serialized_exported_program.json") for name in names) or (
        any(name.endswith("archive_format") for name in names)
        and any(name.endswith("models/model.json") for name in names)
    ):
        return "pt2"
    if any(name.endswith("data.pkl") for name in names) and any(
        "/code/" in f"/{name}" for name in names
    ):
        return "torchscript"
    return "unknown"


def verify_artifact(path: str | Path, descriptor: ArtifactDescriptor) -> Path:
    """Verify size, digest, and non-executing format before a cache entry is trusted."""
    artifact_path = Path(path)
    try:
        actual_size = artifact_path.stat().st_size
    except OSError as exc:
        raise ArtifactIntegrityError(
            f"Cannot inspect artifact {artifact_path}."
        ) from exc
    if actual_size != descriptor.size_bytes:
        raise ArtifactIntegrityError(
            f"Artifact {descriptor.artifact_id!r} has size {actual_size}; expected "
            f"{descriptor.size_bytes}."
        )
    actual_hash = sha256_file(artifact_path)
    if actual_hash != descriptor.sha256:
        raise ArtifactIntegrityError(
            f"Artifact {descriptor.artifact_id!r} failed SHA-256 verification."
        )
    if descriptor.format in {"pt2", "torchscript"}:
        actual_format = detect_model_format(artifact_path)
        if actual_format != descriptor.format:
            raise ArtifactIntegrityError(
                f"Artifact {descriptor.artifact_id!r} has format {actual_format!r}; "
                f"expected {descriptor.format!r}."
            )
    return artifact_path


__all__ = [
    "ArtifactDescriptor",
    "ArtifactManifest",
    "detect_model_format",
    "get_model_manifest",
    "normalize_device",
    "parse_runtime_version",
    "sha256_file",
    "verify_artifact",
]
