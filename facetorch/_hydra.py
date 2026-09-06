"""Isolated composition adapter for the supported Hydra 1.3 series."""

from datetime import datetime
import sys

from hydra._internal.config_loader_impl import ConfigLoaderImpl
from hydra._internal.utils import create_config_search_path
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import DictConfig, OmegaConf, open_dict


def _register_missing_resolvers() -> None:
    # Match Hydra's Compose API resolvers without setup_globals(), which replaces
    # an application's registered callbacks even when Hydra is already active.
    resolvers = {
        "now": (lambda pattern: datetime.now().strftime(pattern), True),
        "hydra": (lambda path: OmegaConf.select(HydraConfig.get(), path), False),
        "python_version": (
            lambda level="minor": {
                "major": str(sys.version_info.major),
                "minor": f"{sys.version_info.major}.{sys.version_info.minor}",
                "micro": ".".join(map(str, sys.version_info[:3])),
            }.get(level),
            False,
        ),
    }
    for name, (resolver, use_cache) in resolvers.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, resolver, use_cache=use_cache)


# Import initialization is serialized by Python; concurrent composition calls do
# not register or replace shared resolvers.
_register_missing_resolvers()


def compose_config(
    search_path: str, config_name: str, overrides: list[str], *, job_name: str
) -> DictConfig:
    """Compose with a fresh loader, leaving the host's Hydra instance untouched.

    Hydra's public initialize/compose pair uses GlobalHydra. Calling the loader
    directly avoids clearing, swapping or reinitializing that singleton, changing
    JobRuntime, or changing the host's compatibility version. This small private
    adapter is covered at both the minimum and locked Hydra/OmegaConf versions.
    """
    loader = ConfigLoaderImpl(create_config_search_path(search_path))
    config = loader.load_configuration(
        config_name=config_name,
        overrides=[f"hydra.job.name={job_name}", *overrides],
        run_mode=RunMode.RUN,
        from_shell=False,
    )
    with open_dict(config):
        del config["hydra"]
    return config
