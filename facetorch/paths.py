"""Portable runtime locations used by packaged facetorch configuration."""

import os
import sys
from pathlib import Path
from typing import Mapping, Optional

from omegaconf import OmegaConf

from facetorch.exceptions import ConfigurationError


CACHE_DIR_ENV = "FACETORCH_CACHE_DIR"
MODEL_DIR_ENV = "FACETORCH_MODEL_DIR"
METADATA_DIR_ENV = "FACETORCH_METADATA_DIR"
OFFLINE_ENV = "FACETORCH_OFFLINE"

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off", ""}


def _normalized_path(value: str) -> Path:
    return Path(os.path.abspath(os.path.expandvars(os.path.expanduser(value))))


def _default_cache_dir(
    *,
    environ: Optional[Mapping[str, str]] = None,
    platform: Optional[str] = None,
    home: Optional[Path] = None,
) -> Path:
    env = os.environ if environ is None else environ
    active_platform = sys.platform if platform is None else platform
    home_dir = Path.home() if home is None else Path(home)

    if active_platform == "win32":
        base = env.get("LOCALAPPDATA")
        if base:
            return _normalized_path(base) / "facetorch" / "Cache"
        return home_dir / "AppData" / "Local" / "facetorch" / "Cache"

    if active_platform == "darwin":
        return home_dir / "Library" / "Caches" / "facetorch"

    xdg_cache = env.get("XDG_CACHE_HOME")
    if xdg_cache:
        return _normalized_path(xdg_cache) / "facetorch"
    return home_dir / ".cache" / "facetorch"


def get_cache_dir() -> Path:
    """Return the configured OS-appropriate facetorch cache root without creating it."""
    configured = os.environ.get(CACHE_DIR_ENV)
    if configured:
        return _normalized_path(configured)
    return _default_cache_dir()


def get_model_dir() -> Path:
    """Return the versioned model cache location without creating it."""
    configured = os.environ.get(MODEL_DIR_ENV)
    if configured:
        return _normalized_path(configured)
    return get_cache_dir() / "models" / "v1"


def get_metadata_dir() -> Path:
    """Return the versioned generated/model metadata location without creating it."""
    configured = os.environ.get(METADATA_DIR_ENV)
    if configured:
        return _normalized_path(configured)
    return get_cache_dir() / "metadata" / "v1"


def get_offline_mode(*, environ: Optional[Mapping[str, str]] = None) -> bool:
    """Return whether model network access is disabled by the public environment flag."""
    env = os.environ if environ is None else environ
    raw = str(env.get(OFFLINE_ENV, "")).strip().lower()
    if raw in _TRUE_VALUES:
        return True
    if raw in _FALSE_VALUES:
        return False
    raise ConfigurationError(
        f"{OFFLINE_ENV} must be one of 1/0, true/false, yes/no, or on/off; "
        f"got {env.get(OFFLINE_ENV)!r}."
    )


def register_path_resolvers() -> None:
    """Register portable path resolvers used by source and packaged Hydra configs."""
    OmegaConf.register_new_resolver(
        "facetorch.cache_dir",
        lambda: os.fspath(get_cache_dir()),
        replace=True,
        use_cache=False,
    )
    OmegaConf.register_new_resolver(
        "facetorch.model_dir",
        lambda: os.fspath(get_model_dir()),
        replace=True,
        use_cache=False,
    )
    OmegaConf.register_new_resolver(
        "facetorch.metadata_dir",
        lambda: os.fspath(get_metadata_dir()),
        replace=True,
        use_cache=False,
    )
    OmegaConf.register_new_resolver(
        "facetorch.offline",
        get_offline_mode,
        replace=True,
        use_cache=False,
    )


register_path_resolvers()


__all__ = [
    "CACHE_DIR_ENV",
    "METADATA_DIR_ENV",
    "MODEL_DIR_ENV",
    "OFFLINE_ENV",
    "get_cache_dir",
    "get_metadata_dir",
    "get_model_dir",
    "get_offline_mode",
    "register_path_resolvers",
]
