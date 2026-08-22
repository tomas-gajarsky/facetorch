import os
from pathlib import Path
import shutil
import subprocess

import facetorch
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.release_blocker
def test_test_session_imports_facetorch_from_checkout():
    """The source suite must never validate an unrelated installed wheel."""
    imported = Path(facetorch.__file__).resolve()
    assert imported.is_relative_to(REPO_ROOT / "facetorch"), imported


@pytest.mark.release_blocker
def test_bare_pytest_ignores_a_stale_site_package(tmp_path):
    """Exercise the console-script import order that exposed F39."""
    fake_site = tmp_path / "fake-site"
    fake_package = fake_site / "facetorch"
    fake_package.mkdir(parents=True)
    (fake_package / "__init__.py").write_text(
        "raise RuntimeError('stale facetorch package imported')\n",
        encoding="utf-8",
    )

    pytest_executable = shutil.which("pytest")
    assert pytest_executable is not None

    env = os.environ.copy()
    env["PYTHONPATH"] = str(fake_site)
    result = subprocess.run(
        [
            pytest_executable,
            "--collect-only",
            "-q",
            "tests/test_import_isolation.py::test_test_session_imports_facetorch_from_checkout",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
