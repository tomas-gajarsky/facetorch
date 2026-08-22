import json
from pathlib import Path
import re

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version
import pytest
import yaml

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_ROOT = REPO_ROOT / ".github" / "workflows"
DEPENDENCY_PROFILES = {
    "torch-2.3-cpu": ("2.3.1", "0.18.1", "/whl/cpu"),
    "torch-2.3-cu121": ("2.3.1", "0.18.1", "/whl/cu121"),
    "torch-2.6-cpu": ("2.6.0", "0.21.0", "/whl/cpu"),
    "torch-2.6-cu124": ("2.6.0", "0.21.0", "/whl/cu124"),
    "torch-2.11-cpu": ("2.11.0", "0.26.0", "/whl/cpu"),
    "torch-2.11-cu130": ("2.11.0", "0.26.0", "/whl/cu130"),
}


def _workflow_commands(path):
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    return "\n".join(
        str(step.get("run", ""))
        for job in workflow.get("jobs", {}).values()
        for step in job.get("steps", [])
        if isinstance(step, dict)
    )


def _nested_strings(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from _nested_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _nested_strings(child)
    elif isinstance(value, str):
        yield value


def _project_metadata():
    with (REPO_ROOT / "pyproject.toml").open("rb") as project_file:
        return tomllib.load(project_file)["project"]


@pytest.mark.release_blocker
def test_install_ci_smokes_the_branch_wheel_and_packaged_config():
    commands = _workflow_commands(WORKFLOW_ROOT / "python-package.yml")
    assert "import facetorch" in commands
    assert "load_config" in commands
    assert re.search(r"pip install\s+[^\n]*\.whl", commands)


@pytest.mark.release_blocker
def test_conda_ci_validates_the_local_candidate_not_public_facetorch():
    commands = _workflow_commands(WORKFLOW_ROOT / "conda-env.yml")
    assert not re.search(r"conda install[^\n]*\bfacetorch\b", commands)
    assert re.search(r"pip install\s+[^\n]*(?:\.whl|dist/)", commands)
    assert "gpu.conda-lock.yml" in commands
    assert "environments/torch-2.6-cu124" in commands
    assert 'torch.version.cuda == "12.4"' in commands


@pytest.mark.release_blocker
@pytest.mark.parametrize("dockerfile", ["Dockerfile", "Dockerfile.gpu"])
def test_production_images_use_a_frozen_dependency_definition(dockerfile):
    content = (REPO_ROOT / "docker" / dockerfile).read_text(encoding="utf-8")
    assert "uv.lock" in content
    assert "--frozen" in content
    assert "--upgrade torch torchvision" not in content


@pytest.mark.release_blocker
def test_cpu_ci_has_an_explicit_supported_torch_cohort_matrix():
    workflow = (WORKFLOW_ROOT / "cpu-cohorts.yml").read_text(encoding="utf-8")
    for cohort in ("2.3", "2.6", "2.11"):
        assert f'torch-cohort: "{cohort}"' in workflow
        assert f"profile: environments/torch-{cohort}-cpu" in workflow
    assert "python -m pytest -q" in workflow
    assert "Path(facetorch.__file__).resolve().is_relative_to(Path.cwd())" in workflow


@pytest.mark.release_blocker
def test_every_supported_torch_device_pair_has_an_exact_uv_lock():
    profile_root = REPO_ROOT / "environments"
    assert {path.name for path in profile_root.iterdir() if path.is_dir()} == set(
        DEPENDENCY_PROFILES
    )
    public = _project_metadata()
    for name, (torch_version, vision_version, index_suffix) in DEPENDENCY_PROFILES.items():
        root = profile_root / name
        with (root / "pyproject.toml").open("rb") as project_file:
            project = tomllib.load(project_file)["project"]
        with (root / "uv.lock").open("rb") as lock_file:
            lock = tomllib.load(lock_file)
        requirements = {
            Requirement(raw).name: Requirement(raw)
            for raw in project["dependencies"]
        }
        assert str(requirements["torch"].specifier) == f"=={torch_version}"
        assert str(requirements["torchvision"].specifier) == f"=={vision_version}"
        assert project["requires-python"] == public["requires-python"]
        assert project["optional-dependencies"]["release"] == public[
            "optional-dependencies"
        ]["release"]
        packages = lock["package"]
        locked_torch = next(item for item in packages if item["name"] == "torch")
        locked_vision = next(item for item in packages if item["name"] == "torchvision")
        assert locked_torch["version"].split("+", 1)[0] == torch_version
        assert locked_vision["version"].split("+", 1)[0] == vision_version
        assert locked_torch["source"]["registry"].endswith(index_suffix)
        assert locked_vision["source"]["registry"].endswith(index_suffix)


@pytest.mark.release_blocker
def test_python_313_claim_has_a_frozen_resolution_and_smoke_lane():
    python_support = SpecifierSet(_project_metadata()["requires-python"])
    if Version("3.13") not in python_support:
        return

    workflow = (WORKFLOW_ROOT / "python-package.yml").read_text(encoding="utf-8")
    commands = _workflow_commands(WORKFLOW_ROOT / "python-package.yml")
    assert '"3.13"' in workflow or "'3.13'" in workflow
    assert "--frozen" in commands or "--constraint" in commands
    assert "import facetorch" in commands


@pytest.mark.release_blocker
def test_ci_runs_a_dependency_advisory_gate():
    workflows = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(WORKFLOW_ROOT.glob("*.yml"))
    )
    assert re.search(r"pip-audit|osv-scanner|dependency-review-action", workflows)


@pytest.mark.release_blocker
def test_advisory_gate_emits_sboms_and_requires_bounded_approved_exceptions():
    workflow = (WORKFLOW_ROOT / "dependency-audit.yml").read_text(encoding="utf-8")
    auditor = (REPO_ROOT / "scripts" / "audit_dependencies.py").read_text(
        encoding="utf-8"
    )
    exceptions = json.loads(
        (REPO_ROOT / "security" / "advisory-exceptions.json").read_text(
            encoding="utf-8"
        )
    )
    assert "scripts/audit_dependencies.py" in workflow
    assert "uv sync --frozen --extra release" in workflow
    assert "cyclonedx1.5" in auditor
    assert '"--no-header"' in auditor
    assert "sbom_content_sha256" in auditor
    assert '"uvx"' not in auditor
    assert '"pip_audit"' in auditor and "sys.executable" in auditor
    assert all(name in auditor for name in DEPENDENCY_PROFILES)
    assert exceptions["maximum_exception_days"] <= 90
    for exception in exceptions["exceptions"]:
        assert exception["status"] in {"approved", "pending_founder_approval"}
        assert exception["versions"]
        assert exception["rationale"] and exception["mitigations"]
        if exception["status"] == "approved":
            assert exception["approved_on"]


@pytest.mark.release_blocker
def test_advisory_exceptions_reject_invalid_approval_windows():
    from datetime import date

    from scripts.audit_dependencies import _exception_for

    approved = {
        "vulnerability_id": "PYSEC-test",
        "package": "example",
        "versions": ["1.0.0"],
        "profiles": ["root"],
        "status": "approved",
        "approved_on": "2026-08-22",
        "expires_on": "2026-11-20",
        "rationale": "Bounded upstream incompatibility.",
        "mitigations": ["No affected operation is used."],
    }
    arguments = {
        "profile": "root",
        "package": "example",
        "version": "1.0.0",
        "vulnerability_ids": {"PYSEC-test"},
        "today": date(2026, 8, 22),
        "maximum_days": 90,
    }

    assert _exception_for([approved], **arguments) == approved
    assert _exception_for(
        [approved], **{**arguments, "version": "1.0.1"}
    ) is None
    assert _exception_for(
        [{**approved, "approved_on": "2026-08-23"}], **arguments
    ) is None
    assert _exception_for(
        [{**approved, "approved_on": "2026-11-21"}], **arguments
    ) is None
    assert _exception_for(
        [{**approved, "expires_on": "2026-11-21"}], **arguments
    ) is None


@pytest.mark.release_blocker
def test_sbom_dependency_graph_digest_ignores_generation_identity(tmp_path):
    from scripts.audit_dependencies import _sbom_content_sha256

    base = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "metadata": {"tools": [{"name": "uv"}]},
        "components": [{"name": "torch", "version": "2.6.0"}],
    }
    paths = []
    for index in (1, 2):
        path = tmp_path / f"sbom-{index}.json"
        path.write_text(
            json.dumps(
                {
                    **base,
                    "serialNumber": f"urn:uuid:00000000-0000-0000-0000-{index:012d}",
                    "metadata": {
                        **base["metadata"],
                        "timestamp": f"2026-08-22T00:00:0{index}Z",
                    },
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)

    assert paths[0].read_bytes() != paths[1].read_bytes()
    assert _sbom_content_sha256(paths[0]) == _sbom_content_sha256(paths[1])


@pytest.mark.release_blocker
def test_local_cuda_release_runner_is_explicit_and_manually_gated():
    content = (WORKFLOW_ROOT / "local-gpu-release.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch" in content
    assert not re.search(r"^\s+(?:pull_request|push|schedule):", content, re.MULTILINE)
    assert "[self-hosted, linux, x64, facetorch-ephemeral-gpu]" in content
    assert "github.ref_protected" in content
    assert "github.repository_owner" in content
    assert "source != os.environ[\"WORKFLOW_SOURCE_SHA\"]" in content
    assert "FACETORCH_RUNNER_EPHEMERAL" in content
    assert "persist-credentials: false" in content
    assert "Persistent publication credential is forbidden" in content
    assert "run_local_cuda_release_matrix.py" in content
    assert "--candidate-evidence" not in content
    assert "execute_candidate_notebook.py" in (
        REPO_ROOT / "scripts" / "run_local_cuda_release_matrix.py"
    ).read_text(encoding="utf-8")
    assert "--network none --read-only" in content
    assert "--gpus all" in content
    assert "record_container_evidence.py" in content


@pytest.mark.release_blocker
def test_install_matrix_covers_every_supported_python_in_an_empty_environment():
    workflow = (WORKFLOW_ROOT / "python-package.yml").read_text(encoding="utf-8")
    for version in ("3.10", "3.11", "3.12"):
        assert f'"{version}"' in workflow
    assert "python -m venv" in workflow
    assert "uv pip sync" in workflow
    assert "uv pip install" in workflow and "--no-deps dist/*.whl" in workflow
    assert "cd \"$RUNNER_TEMP/empty\"" in workflow


@pytest.mark.release_blocker
@pytest.mark.parametrize(
    ("dockerfile", "profile"),
    [
        ("Dockerfile", "environments/torch-2.6-cpu"),
        ("Dockerfile.gpu", "environments/torch-2.6-cu124"),
    ],
)
def test_production_images_install_the_candidate_wheel_as_non_root(
    dockerfile, profile
):
    content = (REPO_ROOT / "docker" / dockerfile).read_text(encoding="utf-8")
    final_stage = "FROM " + content.rsplit("\nFROM ", 1)[-1]
    assert profile in final_stage
    assert "uv pip install" in final_stage and "--no-deps" in final_stage
    assert "USER facetorch" in final_stage
    assert "COPY facetorch/" not in final_stage


@pytest.mark.release_blocker
def test_docker_context_excludes_nested_profile_virtual_environments():
    patterns = {
        line.strip()
        for line in (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert "**/.venv/" in patterns


@pytest.mark.release_blocker
def test_local_release_runner_rejects_ignored_packaging_residue():
    content = (REPO_ROOT / "scripts" / "run_local_cuda_release_matrix.py").read_text(
        encoding="utf-8"
    )
    for residue in ('repo_root / "build"', 'repo_root / "dist"', 'repo_root / "facetorch.egg-info"'):
        assert residue in content
    assert "Refusing to build with ignored packaging residue" in content


@pytest.mark.release_blocker
def test_container_smoke_securely_rebases_host_absolute_evidence(tmp_path):
    from scripts.smoke_staged_default_analyzer import (
        _bounded_file,
        _write_json_atomic,
    )

    staging_root = tmp_path / "mounted-evidence"
    expected = Path("torch-2.6/model-a/model-torch2.6.pt2")
    artifact = staging_root / expected
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"candidate")

    host_path = Path("/different/host/staging") / expected
    assert _bounded_file(staging_root, host_path, expected) == artifact.resolve()
    assert _bounded_file(staging_root, expected, expected) == artifact.resolve()

    with pytest.raises(RuntimeError, match="wrong artifact suffix"):
        _bounded_file(
            staging_root,
            "/different/host/staging/torch-2.6/model-b/model-torch2.6.pt2",
            expected,
        )

    artifact.unlink()
    artifact.symlink_to(tmp_path / "outside.pt2")
    with pytest.raises(RuntimeError, match="contains a symlink"):
        _bounded_file(staging_root, host_path, expected)

    report = tmp_path / "container-report.json"
    _write_json_atomic(report, {"status": "ok"})
    assert report.stat().st_mode & 0o777 == 0o644


@pytest.mark.release_blocker
def test_container_evidence_requires_linux_amd64_non_root_image(monkeypatch):
    from types import SimpleNamespace

    import scripts.record_container_evidence as recorder

    image_id = "sha256:" + "a" * 64

    def inspect_with(user):
        monkeypatch.setattr(
            recorder.subprocess,
            "run",
            lambda *args, **kwargs: SimpleNamespace(
                returncode=0,
                stdout=f'"{image_id}"|"linux"|"amd64"|"{user}"\n',
                stderr="",
            ),
        )
        return recorder._image_metadata("facetorch:candidate")

    metadata = inspect_with("facetorch")
    assert metadata["configured_user"] == "facetorch"
    with pytest.raises(RuntimeError, match="not configured as facetorch"):
        inspect_with("")

    recorder_source = (REPO_ROOT / "scripts" / "record_container_evidence.py").read_text(
        encoding="utf-8"
    )
    smoke_source = (REPO_ROOT / "scripts" / "smoke_staged_default_analyzer.py").read_text(
        encoding="utf-8"
    )
    assert 'report.get("uid") != 10001' in recorder_source
    assert '"uid": os.getuid()' in smoke_source


@pytest.mark.release_blocker
def test_public_runtime_defaults_do_not_depend_on_docker_paths():
    violations = []
    public_configs = [REPO_ROOT / "conf" / "config.yaml"]
    public_configs.extend(sorted((REPO_ROOT / "conf" / "analyzer").rglob("*.yaml")))
    for path in public_configs:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        for value in _nested_strings(config):
            if value.startswith("/opt/"):
                violations.append(f"{path.relative_to(REPO_ROOT)}: {value}")

    assert violations == []


@pytest.mark.release_blocker
def test_release_metadata_does_not_claim_stable_before_rc_soak():
    classifiers = _project_metadata()["classifiers"]
    assert "Development Status :: 5 - Production/Stable" not in classifiers


@pytest.mark.release_blocker
def test_release_metadata_does_not_claim_unverified_os_support():
    classifiers = _project_metadata()["classifiers"]
    assert "Operating System :: OS Independent" not in classifiers


@pytest.mark.release_blocker
def test_notebook_uses_immutable_release_inputs_and_consistent_result_access():
    notebook_path = REPO_ROOT / "notebooks" / "facetorch_notebook_demo.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    sources = ["".join(cell.get("source", [])) for cell in notebook.get("cells", [])]
    combined = "\n".join(sources)

    assert "facetorch>=1.0.0" not in combined
    assert "/main/" not in combined and "blob/main/" not in combined
    if "response.img" in combined:
        assert "return_img_data=False" not in combined
