"""Library composition must coexist with the application's Hydra lifecycle."""

from concurrent.futures import ThreadPoolExecutor
import copy
import subprocess
import sys
import threading

from hydra import compose, initialize_config_dir, version
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import JobRuntime
from omegaconf import OmegaConf
import pytest

from facetorch import load_config, load_config_from_path
from facetorch._hydra import ConfigLoaderImpl
from facetorch.exceptions import ConfigurationError

pytestmark = pytest.mark.release_blocker


def test_import_leaves_host_resolvers_available_for_registration():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
from omegaconf import OmegaConf
names = ('now', 'hydra', 'python_version')
assert not any(OmegaConf.has_resolver(name) for name in names)
import facetorch
assert not any(OmegaConf.has_resolver(name) for name in names)
for name in names:
    OmegaConf.register_new_resolver(name, lambda *args: 'host')
facetorch.load_config()
for name in names:
    assert OmegaConf.create({'value': '${' + name + ':}'}).value == 'host'
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_first_concurrent_compositions_register_resolvers_without_retaining_state():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
from concurrent.futures import ThreadPoolExecutor
import threading
from omegaconf import OmegaConf
from facetorch import load_config
barrier = threading.Barrier(4, timeout=10)
def load(_):
    barrier.wait()
    return load_config().analyzer.device
with ThreadPoolExecutor(4) as pool:
    assert list(pool.map(load, range(4))) == ['cpu'] * 4
assert OmegaConf.create({'value': '${python_version:major}'}).value == '3'
OmegaConf.clear_resolver('python_version')
load_config()
assert OmegaConf.create({'value': '${python_version:major}'}).value == '3'
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def config_trees(tmp_path):
    host = tmp_path / "host"
    external = tmp_path / "external"
    host.mkdir()
    external.mkdir()
    (host / "config.yaml").write_text("marker: host\n")
    (external / "analyzer").mkdir()
    (external / "analyzer" / "base.yaml").write_text("device: cpu\n")
    (external / "config.yaml").write_text(
        "defaults:\n  - analyzer: base\n  - _self_\nmarker: external\n"
    )
    return host, external / "config.yaml"


@pytest.mark.parametrize("compatibility", ["1.1", "1.3"])
@pytest.mark.parametrize("external", [False, True])
def test_host_state_and_composition_survive_success_and_failure(
    config_trees, compatibility, external
):
    host, config_file = config_trees
    with initialize_config_dir(
        config_dir=str(host), job_name="host-job", version_base=compatibility
    ):
        instance = GlobalHydra.instance().hydra
        search_path = [
            (entry.provider, entry.path)
            for entry in instance.config_loader.get_search_path().get_path()
        ]
        job_state = copy.deepcopy(JobRuntime().conf)
        hydra_state = HydraConfig.instance().cfg
        version_base = version.getbase()
        resolver = OmegaConf._get_resolver("now")

        def load(**kwargs):
            return (
                load_config_from_path(config_file, **kwargs)
                if external
                else load_config(**kwargs)
            )

        assert load(overrides=["analyzer.device=cuda"]).analyzer.device == "cuda"
        with pytest.raises(ConfigurationError):
            load(overrides=["analyzer/does-not-exist=missing"])
        assert GlobalHydra.instance().hydra is instance
        assert [
            (entry.provider, entry.path)
            for entry in instance.config_loader.get_search_path().get_path()
        ] == search_path
        assert JobRuntime().conf == job_state
        assert HydraConfig.instance().cfg is hydra_state
        assert version.getbase() == version_base
        assert OmegaConf._get_resolver("now") is resolver
        assert compose(config_name="config").marker == "host"


def test_host_can_compose_while_library_composition_is_in_progress(
    config_trees, monkeypatch
):
    host, config_file = config_trees
    entered = threading.Barrier(3, timeout=10)
    release = threading.Event()
    original = ConfigLoaderImpl.load_configuration
    with initialize_config_dir(config_dir=str(host), version_base=None):
        host_loader = GlobalHydra.instance().hydra.config_loader

        def load(self, *args, **kwargs):
            if self is not host_loader:
                entered.wait()
                assert release.wait(timeout=10)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(ConfigLoaderImpl, "load_configuration", load)
        with ThreadPoolExecutor(2) as pool:
            packaged = pool.submit(load_config)
            external = pool.submit(load_config_from_path, config_file)
            try:
                entered.wait()
                assert GlobalHydra.instance().hydra.config_loader is host_loader
                assert compose(config_name="config").marker == "host"
            finally:
                release.set()
            assert packaged.result(timeout=10).analyzer.device == "cpu"
            assert external.result(timeout=10).marker == "external"


def test_loaders_do_not_initialize_global_hydra(config_trees):
    _, config_file = config_trees
    assert not GlobalHydra.instance().is_initialized()
    assert load_config().analyzer.device == "cpu"
    assert load_config_from_path(config_file).marker == "external"
    assert not GlobalHydra.instance().is_initialized()
