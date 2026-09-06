"""Supported resource-backed and external Hydra configuration loaders."""

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Union

from omegaconf import DictConfig

from facetorch._hydra import compose_config
from facetorch.exceptions import ConfigurationError
from facetorch.paths import register_path_resolvers

ConfigPath = Union[str, os.PathLike]
_PROFILE_OVERRIDES = {
    "cpu": "analyzer.device=cpu",
    "gpu": "analyzer.device=cuda",
}


def _normalize_overrides(overrides: Optional[Sequence[str]]) -> list[str]:
    if overrides is None:
        return []
    if isinstance(overrides, (str, bytes)):
        raise ConfigurationError(
            "overrides must be a sequence of Hydra override strings, not one string."
        )
    normalized = list(overrides)
    if any(not isinstance(override, str) or not override for override in normalized):
        raise ConfigurationError(
            "overrides must contain only non-empty Hydra override strings."
        )
    return normalized


def _profile_overrides(
    profile: Optional[str], user_overrides: Sequence[str]
) -> list[str]:
    if profile is None:
        return list(user_overrides)
    if profile not in _PROFILE_OVERRIDES:
        supported = ", ".join(sorted(_PROFILE_OVERRIDES))
        raise ConfigurationError(
            f"Unknown configuration profile {profile!r}; choose one of: {supported}."
        )

    has_explicit_device = any(
        override.lstrip("+~").split("=", 1)[0] == "analyzer.device"
        for override in user_overrides
    )
    result = [] if has_explicit_device else [_PROFILE_OVERRIDES[profile]]
    result.extend(user_overrides)
    return result


def _option_overrides(
    user_overrides: Sequence[str],
    *,
    offline: Optional[bool],
    allow_legacy_models: Optional[bool],
) -> list[str]:
    result = list(user_overrides)
    explicit_keys = {
        override.lstrip("+~").split("=", 1)[0] for override in user_overrides
    }
    options = {
        "offline": offline,
        "allow_legacy_models": allow_legacy_models,
    }
    for key, value in options.items():
        if value is None or key in explicit_keys:
            continue
        if not isinstance(value, bool):
            raise ConfigurationError(f"{key} must be a boolean when supplied.")
        result.append(f"{key}={'true' if value else 'false'}")
    return result


def load_config(
    profile: Optional[str] = "cpu",
    *,
    overrides: Optional[Sequence[str]] = None,
    offline: Optional[bool] = None,
    allow_legacy_models: Optional[bool] = None,
) -> DictConfig:
    """Compose packaged facetorch defaults independently of the current directory.

    Safe to call inside an existing Hydra application: composition uses its own
    loader and does not replace the application's Hydra instance or search path.

    Args:
        profile: ``"cpu"`` (default), ``"gpu"``, or ``None`` to retain the
            packaged device value.
        overrides: Hydra override strings applied after the selected profile.
        offline: Explicitly disable or allow model network access. ``None`` uses
            the packaged environment-backed default.
        allow_legacy_models: Explicit opt-in for eligible verified TorchScript
            fallback artifacts. ``None`` retains the packaged false default.

    Returns:
        A fully composed configuration with ``cfg.analyzer`` available.
    """
    register_path_resolvers()
    option_overrides = _option_overrides(
        _normalize_overrides(overrides),
        offline=offline,
        allow_legacy_models=allow_legacy_models,
    )
    composed_overrides = _profile_overrides(profile, option_overrides)
    try:
        return compose_config(
            search_path="pkg://facetorch.configs",
            config_name="config",
            overrides=composed_overrides,
            job_name="facetorch-load-config",
        )
    except ConfigurationError:
        raise
    except Exception as exc:
        raise ConfigurationError(
            "Could not compose packaged facetorch configuration."
        ) from exc


def load_config_from_path(
    path: ConfigPath,
    *,
    profile: Optional[str] = None,
    overrides: Optional[Sequence[str]] = None,
    offline: Optional[bool] = None,
    allow_legacy_models: Optional[bool] = None,
) -> DictConfig:
    """Compose an advanced external Hydra YAML tree from an explicit file path.

    Relative paths are resolved against the caller's current directory. The file's
    parent becomes Hydra's configuration directory, so its ``defaults`` list and
    sibling configuration groups are composed normally. An existing host Hydra
    application keeps its instance, search path, job state and compatibility mode.
    """
    register_path_resolvers()
    config_file = Path(path).expanduser().resolve()
    if config_file.suffix.lower() not in {".yaml", ".yml"}:
        raise ConfigurationError("External configuration must be a .yaml or .yml file.")
    if not config_file.is_file():
        raise ConfigurationError(
            f"External configuration file does not exist: {config_file}."
        )

    option_overrides = _option_overrides(
        _normalize_overrides(overrides),
        offline=offline,
        allow_legacy_models=allow_legacy_models,
    )
    composed_overrides = _profile_overrides(profile, option_overrides)
    try:
        return compose_config(
            search_path=os.fspath(config_file.parent),
            config_name=config_file.name[: -len(config_file.suffix)],
            overrides=composed_overrides,
            job_name="facetorch-load-external-config",
        )
    except ConfigurationError:
        raise
    except Exception as exc:
        raise ConfigurationError(
            f"Could not compose external configuration: {config_file}."
        ) from exc


__all__ = ["load_config", "load_config_from_path"]
