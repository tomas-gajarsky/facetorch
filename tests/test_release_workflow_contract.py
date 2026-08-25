from pathlib import Path
import json
import re

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_WORKFLOWS = (
    REPO_ROOT / ".github" / "workflows" / "release.yml",
    REPO_ROOT / ".github" / "workflows" / "auto-release.yml",
)
ALL_WORKFLOWS = tuple(sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml")))
PUBLICATION_COMMANDS = ("twine upload", "docker push", "gh release create")


def _workflow(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _commands(job):
    return "\n".join(
        str(step.get("run", ""))
        for step in job.get("steps", [])
        if isinstance(step, dict)
    )


def _needs(job):
    needs = job.get("needs", [])
    if isinstance(needs, str):
        return {needs}
    return set(needs)


def _publication_jobs(workflow):
    for name, job in workflow.get("jobs", {}).items():
        commands = _commands(job)
        if any(command in commands for command in PUBLICATION_COMMANDS):
            yield name, job, commands


@pytest.mark.release_blocker
def test_dispatch_input_is_never_interpolated_directly_into_shell():
    unsafe = re.compile(r"\$\{\{\s*(?:github\.event\.)?inputs\.tag\s*\}\}")
    violations = []
    for path in RELEASE_WORKFLOWS:
        for job_name, job in _workflow(path).get("jobs", {}).items():
            if unsafe.search(_commands(job)):
                violations.append(f"{path.name}:{job_name}")

    assert violations == []


@pytest.mark.release_blocker
def test_publication_jobs_checkout_the_resolved_immutable_candidate():
    violations = []
    for path in RELEASE_WORKFLOWS:
        for job_name, job, _commands_text in _publication_jobs(_workflow(path)):
            checkout_steps = [
                step
                for step in job.get("steps", [])
                if isinstance(step, dict)
                and str(step.get("uses", "")).startswith("actions/checkout@")
            ]
            for step in checkout_steps:
                checkout_ref = str(step.get("with", {}).get("ref", ""))
                if (
                    not checkout_ref
                    or checkout_ref in {"main", "master"}
                    or "inputs.tag" in checkout_ref
                ):
                    violations.append(f"{path.name}:{job_name}")

    assert violations == []


@pytest.mark.release_blocker
@pytest.mark.parametrize("workflow_name", ("local-gpu-release.yml", "release.yml"))
def test_manual_release_checkouts_use_the_trusted_event_sha(workflow_name):
    workflow = _workflow(REPO_ROOT / ".github" / "workflows" / workflow_name)
    checkout_refs = [
        str(step.get("with", {}).get("ref", ""))
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if isinstance(step, dict)
        and str(step.get("uses", "")).startswith("actions/checkout@")
    ]

    assert checkout_refs
    assert set(checkout_refs) == {"${{ github.sha }}"}


@pytest.mark.release_blocker
def test_github_release_is_draft_until_external_publication_succeeds():
    violations = []
    for path in RELEASE_WORKFLOWS:
        for job_name, job, commands in _publication_jobs(_workflow(path)):
            if "gh release create" not in commands:
                continue
            if "--draft" not in commands and len(_needs(job)) < 3:
                violations.append(f"{path.name}:{job_name}")

    assert violations == []


@pytest.mark.release_blocker
def test_latest_alias_is_promoted_only_after_immutable_artifacts():
    violations = []
    for path in RELEASE_WORKFLOWS:
        for job_name, job in _workflow(path).get("jobs", {}).items():
            commands = _commands(job)
            if not re.search(r"docker push\s+\S+:latest(?:\s|$)", commands):
                continue
            if "docker compose build" in commands or len(_needs(job)) < 3:
                violations.append(f"{path.name}:{job_name}")

    assert violations == []


@pytest.mark.release_blocker
def test_release_pipeline_has_a_no_publish_dry_run():
    manual_workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )
    assert re.search(r"(?m)^\s+dry_run:\s*$", manual_workflow)


@pytest.mark.release_blocker
def test_docker_credentials_are_passed_via_standard_input():
    violations = []
    for path in RELEASE_WORKFLOWS:
        for job_name, job in _workflow(path).get("jobs", {}).items():
            for line in _commands(job).splitlines():
                if "docker login" in line and "--password-stdin" not in line:
                    violations.append(f"{path.name}:{job_name}")

    assert violations == []


@pytest.mark.release_blocker
def test_every_external_action_is_commit_pinned_and_recorded():
    release_inputs = json.loads(
        (REPO_ROOT / "security" / "release-inputs.json").read_text(encoding="utf-8")
    )
    reviewed = release_inputs["github_actions"]
    violations = []
    observed = set()
    uses_pattern = re.compile(r"(?m)^\s*-?\s*uses:\s*([^\s#]+)")
    for path in ALL_WORKFLOWS:
        for action in uses_pattern.findall(path.read_text(encoding="utf-8")):
            if action.startswith("./"):
                continue
            name, separator, revision = action.partition("@")
            owner_repo = "/".join(name.split("/")[:2])
            observed.add(owner_repo)
            if (
                not separator
                or re.fullmatch(r"[0-9a-f]{40}", revision) is None
                or reviewed.get(owner_repo) != revision
            ):
                violations.append(f"{path.name}:{action}")

    assert violations == []
    assert observed == set(reviewed)


@pytest.mark.release_blocker
def test_miniforge_input_matches_the_reviewed_installer():
    release_inputs = json.loads(
        (REPO_ROOT / "security" / "release-inputs.json").read_text(encoding="utf-8")
    )
    reviewed = release_inputs["tools"]["miniforge"]
    version = reviewed["version"]
    workflow = _workflow(REPO_ROOT / ".github" / "workflows" / "conda-env.yml")
    configured_versions = {
        str(step.get("with", {}).get("miniforge-version"))
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if str(step.get("uses", "")).startswith("conda-incubator/setup-miniconda@")
    }

    assert re.fullmatch(r"\d+\.\d+\.\d+-\d+", version)
    assert configured_versions == {version}
    assert reviewed["asset"] == f"Miniforge3-{version}-Linux-x86_64.sh"
    assert re.fullmatch(r"[0-9a-f]{64}", reviewed["sha256"])


@pytest.mark.release_blocker
def test_dependency_review_allowlist_matches_approved_advisory_exceptions():
    policy = json.loads(
        (REPO_ROOT / "security" / "advisory-exceptions.json").read_text(
            encoding="utf-8"
        )
    )
    expected = {
        alias
        for exception in policy["exceptions"]
        if exception["status"] == "approved"
        for alias in exception.get("aliases", [])
        if alias.startswith("GHSA-")
    }
    workflow = _workflow(REPO_ROOT / ".github" / "workflows" / "security.yml")
    review_step = next(
        step
        for step in workflow["jobs"]["dependency-review"]["steps"]
        if str(step.get("uses", "")).startswith("actions/dependency-review-action@")
    )
    configured = {
        item.strip()
        for item in str(review_step.get("with", {}).get("allow-ghsas", "")).split(",")
        if item.strip()
    }

    assert configured == expected


@pytest.mark.release_blocker
def test_release_uses_trusted_publishing_attestations_and_fail_closed_governance():
    workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )
    transaction = (REPO_ROOT / "scripts" / "release_transaction.py").read_text(
        encoding="utf-8"
    )
    assert "pypa/gh-action-pypi-publish@" in workflow
    assert "id-token: write" in workflow
    assert "actions/attest-build-provenance@" in workflow
    assert "anchore/sbom-action@" in workflow
    assert "${{ secrets.TWINE_PASSWORD }}" not in workflow
    assert "${{ secrets.PYPI_API_TOKEN }}" not in workflow
    assert "twine upload" not in workflow
    assert "validate_packaged_model_governance" in transaction
    assert 'governance.get("status") != "approved"' in transaction
    assert 'packaged.get("manifest_revision") != revision' in transaction
    assert "validate_local_release_evidence" in transaction


@pytest.mark.release_blocker
def test_release_binds_full_cuda_validation_to_the_exact_built_images():
    workflow = _workflow(REPO_ROOT / ".github" / "workflows" / "release.yml")
    jobs = workflow["jobs"]
    local = jobs["validate-exact-images-on-local-gpu"]
    commands = _commands(local)

    assert set(local["runs-on"]) == {
        "self-hosted",
        "linux",
        "x64",
        "facetorch-ephemeral-gpu",
    }
    assert "build-images" in _needs(local)
    assert "zstd -d --stdout" in commands
    assert "docker image inspect" in commands
    assert "--gpus all" in commands
    assert "--network none" in commands
    assert "record_container_evidence.py" in commands
    assert "validate-exact-images-on-local-gpu" in _needs(jobs["assemble-candidate"])


@pytest.mark.release_blocker
def test_release_source_is_the_selected_protected_ref_and_tools_are_bounded():
    release_path = REPO_ROOT / ".github" / "workflows" / "release.yml"
    workflow_text = release_path.read_text(encoding="utf-8")
    workflow = _workflow(release_path)
    release_inputs = json.loads(
        (REPO_ROOT / "security" / "release-inputs.json").read_text(encoding="utf-8")
    )

    validate = _commands(workflow["jobs"]["validate-inputs"])
    assert "SOURCE_REF_PROTECTED" in validate
    assert "WORKFLOW_SOURCE_SHA" in workflow_text
    assert "source_sha must equal the selected protected ref commit" in validate
    assert "runs-on: ubuntu-latest" not in workflow_text
    assert 'python-version: "3.12.12"' in workflow_text
    assert release_inputs["tools"]["build_python"] == "3.12.12"
    assert release_inputs["tools"]["github_runner"] == "ubuntu-24.04"


@pytest.mark.release_blocker
def test_external_write_jobs_are_excluded_from_dry_run_and_aliases_are_last():
    workflow = _workflow(REPO_ROOT / ".github" / "workflows" / "release.yml")
    jobs = workflow["jobs"]
    for job_name in (
        "create-github-draft",
        "publish-images",
        "publish-pypi",
        "finalize-github-release",
        "promote-stable-aliases",
    ):
        assert "dry_run == 'false'" in str(jobs[job_name].get("if", ""))
    aliases = jobs["promote-stable-aliases"]
    assert {
        "create-github-draft",
        "publish-images",
        "publish-pypi",
        "finalize-github-release",
    }.issubset(_needs(aliases))
    assert "is_prerelease == 'false'" in str(aliases["if"])


@pytest.mark.release_blocker
def test_production_container_inputs_are_digest_pinned_and_recorded():
    release_inputs = json.loads(
        (REPO_ROOT / "security" / "release-inputs.json").read_text(encoding="utf-8")
    )
    expected_digests = set(release_inputs["container_inputs"].values())
    dockerfiles = (
        REPO_ROOT / "docker" / "Dockerfile",
        REPO_ROOT / "docker" / "Dockerfile.gpu",
    )
    observed = set()
    unpinned = []
    image_pattern = re.compile(
        r"(?m)^(?:FROM|COPY\s+--from=)\s*([^\s]+(?:\s+AS\s+\w+)?)",
        re.IGNORECASE,
    )
    for path in dockerfiles:
        for raw in image_pattern.findall(path.read_text(encoding="utf-8")):
            image = raw.split()[0]
            if image == "builder":
                continue
            match = re.search(r"@(sha256:[0-9a-f]{64})$", image)
            if match is None:
                unpinned.append(f"{path.name}:{image}")
            else:
                observed.add(match.group(1))

    assert unpinned == []
    assert observed == expected_digests


@pytest.mark.release_blocker
def test_public_repository_security_automation_is_present():
    workflow = (REPO_ROOT / ".github" / "workflows" / "security.yml").read_text(
        encoding="utf-8"
    )
    dependabot = (REPO_ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8")
    assert "actions/dependency-review-action@" in workflow
    assert "gitleaks/gitleaks-action@" in workflow
    for ecosystem in ("github-actions", "pip", "docker"):
        assert f"package-ecosystem: {ecosystem}" in dependabot
