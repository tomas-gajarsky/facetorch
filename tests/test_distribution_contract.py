import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from zipfile import ZipFile

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]

WHEEL_REQUIRED_FILES = {
    "facetorch/__init__.py",
    "facetorch/configuration.py",
    "facetorch/paths.py",
    "facetorch/configs/__init__.py",
    "facetorch/configs/config.yaml",
    "facetorch/models/manifest.json",
}


def test_model_trust_root_is_not_excluded_from_git():
    required = [
        "facetorch/models/__init__.py",
        "facetorch/models/manifest.json",
        "facetorch/models/compatibility.json",
        "facetorch/models/governance.json",
    ]
    ignored = [
        path
        for path in required
        if subprocess.run(
            ["git", "check-ignore", "--quiet", "--", path],
            cwd=REPO_ROOT,
            check=False,
        ).returncode
        == 0
    ]
    assert ignored == [], "facetorch/models must be committed as package data"


SDIST_ALLOWED_TOP_LEVEL = {
    "CHANGELOG.md",
    "LICENSE",
    "MANIFEST.in",
    "MODEL_NOTICE.md",
    "PKG-INFO",
    "README.md",
    "conf",
    "data",
    "docs",
    "environment.yml",
    "environments",
    "facetorch",
    "facetorch.egg-info",
    "gpu.environment.yml",
    "gpu.conda-lock.yml",
    "model_defs",
    "model_cards",
    "notebooks",
    "pyproject.toml",
    "pytest.ini",
    "scripts",
    "security",
    "setup.cfg",
    "tests",
    "uv.lock",
    "conda-lock.yml",
}


def _install_without_dependencies(artifact, install_root, *, sdist=False):
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-deps",
    ]
    if sdist:
        command.append("--no-build-isolation")
    command.extend(["--target", str(install_root), str(artifact)])
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _run_from_read_only_empty_cwd(code, install_root, empty_cwd):
    empty_cwd.mkdir()
    empty_cwd.chmod(0o555)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(install_root)
    try:
        return subprocess.run(
            [sys.executable, "-c", code],
            cwd=empty_cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    finally:
        empty_cwd.chmod(0o755)


def _distribution_copy_ignore(path, names):
    ignored = set(
        shutil.ignore_patterns(
            ".git",
            ".pytest_cache",
            ".venv",
            "__pycache__",
            "*.egg-info",
            "build",
            "dist",
        )(path, names)
    )
    if Path(path).resolve() == REPO_ROOT:
        ignored.update({"models", "models_local", "outputs"})
    return ignored


@pytest.fixture(scope="session")
def built_distributions(tmp_path_factory):
    """Build the current working tree without polluting or importing the checkout."""
    work_root = tmp_path_factory.mktemp("distribution-contract")
    source_copy = work_root / "source"
    shutil.copytree(
        REPO_ROOT,
        source_copy,
        ignore=_distribution_copy_ignore,
    )
    dist_dir = work_root / "dist"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--sdist",
            "--outdir",
            str(dist_dir),
        ],
        cwd=source_copy,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return {
        "root": work_root,
        "wheel": next(dist_dir.glob("*.whl")),
        "sdist": next(dist_dir.glob("*.tar.gz")),
    }


@pytest.mark.release_blocker
def test_wheel_exposes_only_facetorch_top_level_namespace(built_distributions):
    wheel = built_distributions["wheel"]
    with ZipFile(wheel) as archive:
        roots = {
            name.split("/", 1)[0]
            for name in archive.namelist()
            if "/" in name and not name.startswith(".")
        }

    unexpected = {
        root
        for root in roots
        if root != "facetorch" and not root.endswith(".dist-info")
    }
    assert unexpected == set()


@pytest.mark.release_blocker
def test_wheel_content_matches_runtime_allowlist(built_distributions):
    with ZipFile(built_distributions["wheel"]) as archive:
        names = {name.rstrip("/") for name in archive.namelist() if name}

    unexpected = {
        name
        for name in names
        if not name.startswith("facetorch/")
        and ".dist-info/" not in name
        and not name.endswith(".dist-info")
    }
    generated = {
        name
        for name in names
        if "__pycache__" in name or name.endswith((".pyc", ".pyo"))
    }

    assert unexpected == set()
    assert generated == set()
    assert WHEEL_REQUIRED_FILES <= names


@pytest.mark.release_blocker
def test_installed_wheel_loads_composed_defaults_outside_checkout(
    built_distributions,
):
    wheel = built_distributions["wheel"]
    install_root = built_distributions["root"] / "installed"
    empty_cwd = built_distributions["root"] / "empty-cwd"
    _install_without_dependencies(wheel, install_root)

    smoke = """
from pathlib import Path

import facetorch
from omegaconf import OmegaConf

install_root = Path(r'INSTALL_ROOT').resolve()
imported = Path(facetorch.__file__).resolve()
assert imported.is_relative_to(install_root), imported

cfg = facetorch.load_config()
assert cfg.analyzer is not None

def strings(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from strings(child)
    elif isinstance(value, str):
        yield value

values = list(strings(OmegaConf.to_container(cfg, resolve=False)))
assert not any(value.startswith('/opt/') for value in values)
""".replace("INSTALL_ROOT", str(install_root))
    result = _run_from_read_only_empty_cwd(smoke, install_root, empty_cwd)

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.release_blocker
def test_installed_wheel_executes_readme_smoke(built_distributions):
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    marked = readme.split("<!-- facetorch-readme-smoke:start -->", 1)[1]
    marked = marked.split("<!-- facetorch-readme-smoke:end -->", 1)[0]
    code = marked.split("```python", 1)[1].split("```", 1)[0].strip()

    install_root = built_distributions["root"] / "readme-installed"
    _install_without_dependencies(built_distributions["wheel"], install_root)
    result = _run_from_read_only_empty_cwd(
        code,
        install_root,
        built_distributions["root"] / "readme-empty-cwd",
    )

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.release_blocker
def test_sdist_dependency_check_is_self_contained(built_distributions):
    extract_root = built_distributions["root"] / "sdist-extracted"
    with tarfile.open(built_distributions["sdist"], mode="r:gz") as archive:
        archive.extractall(extract_root)

    source_root = next(path for path in extract_root.iterdir() if path.is_dir())
    result = subprocess.run(
        [sys.executable, "scripts/check_dependency_sync.py"],
        cwd=source_root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.release_blocker
def test_sdist_content_matches_source_allowlist(built_distributions):
    with tarfile.open(built_distributions["sdist"], mode="r:gz") as archive:
        members = [member.name for member in archive.getmembers()]

    root = members[0].split("/", 1)[0]
    relative = {
        name.split("/", 1)[1]
        for name in members
        if "/" in name and name.split("/", 1)[1]
    }
    top_level = {name.split("/", 1)[0] for name in relative}
    forbidden = {
        name
        for name in relative
        if name.startswith((".git/", ".env", ".venv/", "dist/", "models/"))
        or "/.venv/" in name
        or "__pycache__" in name
        or name.endswith((".pyc", ".pyo"))
    }

    assert root.startswith("facetorch-")
    assert top_level <= SDIST_ALLOWED_TOP_LEVEL
    assert forbidden == set()
    assert {
        "gpu.environment.yml",
        "gpu.conda-lock.yml",
        "conda-lock.yml",
        "uv.lock",
        "scripts/check_dependency_sync.py",
        "scripts/audit_dependencies.py",
        "scripts/audit_model_manifest_hf.py",
        "scripts/model_cohort_publication.py",
        "scripts/render_model_cards.py",
        "scripts/release_transaction.py",
        "model_cards/catalog.json",
        "model_cards/upstream_licenses/adaface-LICENSE",
        "notebooks/facetorch_notebook_demo.ipynb",
    } <= relative


@pytest.mark.release_blocker
def test_sdist_model_manifest_auditor_has_its_renderer_inputs(built_distributions):
    extract_root = built_distributions["root"] / "auditor-extracted"
    with tarfile.open(built_distributions["sdist"], mode="r:gz") as archive:
        archive.extractall(extract_root)
    source_root = next(path for path in extract_root.iterdir() if path.is_dir())
    smoke = """
from pathlib import Path

from scripts.audit_model_manifest_hf import audit_remote_manifest

class OfflineApi:
    def model_info(self, **kwargs):
        raise RuntimeError("deliberate offline audit probe")

report = audit_remote_manifest(
    Path("facetorch/models/manifest.json"),
    api=OfflineApi(),
    download_fn=lambda **kwargs: None,
)
assert report["status"] == "failed"
assert report["failures"]
assert not any(
    item["model_id"] == "model-card-contract"
    for item in report["failures"]
), report
assert all(
    item["error"] == "deliberate offline audit probe"
    for item in report["failures"]
), report
"""
    result = subprocess.run(
        [sys.executable, "-c", smoke],
        cwd=source_root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.release_blocker
def test_sdist_installs_and_loads_config_outside_checkout(built_distributions):
    install_root = built_distributions["root"] / "sdist-installed"
    _install_without_dependencies(
        built_distributions["sdist"], install_root, sdist=True
    )
    smoke = """
from pathlib import Path

import facetorch

install_root = Path(r'INSTALL_ROOT').resolve()
assert Path(facetorch.__file__).resolve().is_relative_to(install_root)
assert facetorch.load_config().analyzer.device == 'cpu'
""".replace("INSTALL_ROOT", str(install_root))
    result = _run_from_read_only_empty_cwd(
        smoke,
        install_root,
        built_distributions["root"] / "sdist-empty-cwd",
    )

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.release_blocker
def test_sdist_examples_use_installed_api_without_checkout_assumptions(
    built_distributions,
):
    extract_root = built_distributions["root"] / "examples-extracted"
    with tarfile.open(built_distributions["sdist"], mode="r:gz") as archive:
        archive.extractall(extract_root)
    source_root = next(path for path in extract_root.iterdir() if path.is_dir())

    install_root = built_distributions["root"] / "examples-installed"
    _install_without_dependencies(built_distributions["wheel"], install_root)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(install_root)
    empty_cwd = built_distributions["root"] / "examples-empty-cwd"
    empty_cwd.mkdir()

    for script_name in ("example.py", "example_tensor.py", "repeated_inference.py"):
        script = source_root / "scripts" / script_name
        content = script.read_text(encoding="utf-8")
        assert "OmegaConf.load" not in content
        assert "../conf" not in content
        assert "import hydra" not in content
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=empty_cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
