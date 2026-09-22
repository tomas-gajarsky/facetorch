"""Advisory coverage and exact local-build exception regressions."""

import copy
import hashlib
import json
import subprocess
from datetime import date
from pathlib import Path

import pytest

from scripts import audit_dependencies as audit

pytestmark = pytest.mark.release_blocker


def test_audited_python_minors_match_the_public_support_contract():
    root = Path(__file__).resolve().parents[1]
    compatibility = json.loads(
        (root / "facetorch/models/compatibility.json").read_text()
    )
    assert list(audit.PYTHON_MINOR_LINES) == compatibility["python"]["minor_lines"]


def test_owner_approval_adds_only_reviewed_profile_version_pairs():
    root = Path(__file__).resolve().parents[1]
    approval_ref = "security/v1-advisory-approval-2026-09-06.json"
    approval = json.loads((root / approval_ref).read_text())
    proposal_path = root / approval["approved_proposal"]["path"]
    assert hashlib.sha256(proposal_path.read_bytes()).hexdigest() == (
        approval["approved_proposal"]["sha256"]
    )
    proposal = json.loads(proposal_path.read_text())
    policy = audit._load_exceptions(root / "security/advisory-exceptions.json")
    activated = [
        entry for entry in policy["exceptions"]
        if entry.get("approval_ref") == approval_ref
    ]
    previous = [entry for entry in policy["exceptions"] if "approval_ref" not in entry]
    assert len(previous) == approval["previous_exception_count"] == 9
    assert len(activated) == approval["new_exact_profile_version_records"] == 76
    assert len(policy["exceptions"]) == len(previous) + len(activated)
    assert approval["status"] == "approved"
    assert approval["approved_on"] == "2026-09-06"
    assert approval["expires_on"] == proposal["proposed_expires_on"] == "2026-11-20"
    reviewed = {item["vulnerability_id"]: item for item in proposal["proposals"]}
    for entry in activated:
        assert len(entry["versions"]) == len(entry["profiles"]) == 1
        assert entry["approved_on"] == approval["approved_on"]
        assert entry["expires_on"] == approval["expires_on"]
        assert entry["review_owner"] == approval["review_owner"]
        assert entry["rationale"] and entry["mitigations"] and entry["residual_risk"]
        assert entry["removal_condition"]
        assert entry["aliases"] == reviewed[entry["vulnerability_id"]]["aliases"]
        assert entry["mitigations"] == reviewed[entry["vulnerability_id"]]["mitigations"]

    observed = {
        (entry["vulnerability_id"], entry["profiles"][0], entry["versions"][0])
        for entry in activated
    }
    expected = {
        (item["vulnerability_id"], scope["profile"], scope["version"])
        for item in proposal["proposals"] for scope in item["scopes"]
    }
    assert observed == expected
    assert len(proposal["proposals"]) == approval["new_advisory_count"] == 18
    assert audit._exception_for(
        policy["exceptions"],
        profile="torch-2.7-cpu",
        package="torch",
        version="2.7.1+cpu",
        vulnerability_ids={"GHSA-unreviewed-finding"},
        today=date(2026, 9, 6),
        maximum_days=policy["maximum_exception_days"],
    ) is None

    profiles = {profile for _, profile, _ in expected} | {"root", "torch-2.14-cpu"}
    for item in proposal["proposals"]:
        # Exercise the real matcher across the profile/version cross product;
        # CPU/CUDA or runtime-line swaps must not inherit another pair's approval.
        versions = {scope["version"] for scope in item["scopes"]} | {"2.14.0+cpu"}
        for profile in profiles:
            for version in versions:
                match = audit._exception_for(
                    activated,
                    profile=profile,
                    package="torch",
                    version=version,
                    vulnerability_ids={item["vulnerability_id"], *item["aliases"]},
                    today=date(2026, 9, 6),
                    maximum_days=policy["maximum_exception_days"],
                )
                assert (match is not None) == (
                    (item["vulnerability_id"], profile, version) in expected
                )
        for scope in item["scopes"]:
            for outside_window in (date(2026, 9, 5), date(2026, 11, 21)):
                assert audit._exception_for(
                    activated,
                    profile=scope["profile"],
                    package="torch",
                    version=scope["version"],
                    vulnerability_ids={item["vulnerability_id"]},
                    today=outside_window,
                    maximum_days=policy["maximum_exception_days"],
                ) is None


@pytest.fixture
def export_fixture():
    packages, lines = [], []
    for name, version, digest in [
        ("torch", "2.6.0+cpu", "a" * 64),
        ("torchvision", "0.21.0+cpu", "b" * 64),
    ]:
        lines.append(f"{name}=={version} --hash=sha256:{digest}")
        packages.append(
            {
                "name": name,
                "version": version,
                "source": {"registry": "https://download.pytorch.org/whl/cpu"},
                "wheels": [
                    {
                        "url": f"https://download.pytorch.org/whl/cpu/{name}.whl",
                        "hash": f"sha256:{digest}",
                    }
                ],
            }
        )
    return "\n".join(lines), {"package": packages}


def _inventory(export_fixture):
    return audit._audit_inventory(*export_fixture, "torch-2.6-cpu")


def _report(inventory):
    return {
        "dependencies": [
            {"name": name, "version": record["audited_version"], "vulns": []}
            for name, record in inventory["dependencies"].items()
        ]
    }


def _policy():
    return {
        "maximum_exception_days": 90,
        "exceptions": [
            {
                "package": "torch",
                "versions": ["2.6.0+cpu"],
                "profiles": ["torch-2.6-cpu"],
                "vulnerability_id": "CVE-test",
                "status": "approved",
                "approved_on": "2026-09-01",
                "expires_on": "2026-09-30",
                "rationale": "Test-only scoped exception",
                "mitigations": ["Test-only mitigation"],
            }
        ],
    }


def test_official_build_projection_preserves_original_identity_and_hashes(
    export_fixture,
):
    inventory = _inventory(export_fixture)
    record = inventory["dependencies"]["torch"]
    assert record["version"] == "2.6.0+cpu"
    assert record["audited_version"] == "2.6.0"
    assert record["sha256"] == ["a" * 64]
    assert record["wheels"] == export_fixture[1]["package"][0]["wheels"]


@pytest.mark.parametrize(
    "changed", ["registry", "wheel-host", "hash", "flavor", "missing-critical"]
)
def test_untrusted_or_incomplete_build_projection_is_rejected(export_fixture, changed):
    requirements, lock = copy.deepcopy(export_fixture)
    package = lock["package"][0]
    if changed == "registry":
        package["source"]["registry"] = "https://example.invalid/whl/cpu"
    elif changed == "wheel-host":
        package["wheels"][0][
            "url"
        ] = "https://download.pytorch.org.evil.invalid/torch.whl"
    elif changed == "hash":
        package["wheels"][0]["hash"] = "sha256:" + "c" * 64
    elif changed == "flavor":
        requirements = requirements.replace("2.6.0+cpu", "2.6.0+private")
        package["version"] = "2.6.0+private"
    else:
        requirements = requirements.splitlines()[0]
    with pytest.raises(ValueError):
        audit._audit_inventory(requirements, lock, "torch-2.6-cpu")


def test_active_inventory_respects_environment_markers(export_fixture):
    requirements, lock = export_fixture
    requirements += (
        '\nnever-installed==1 ; python_version < "1" --hash=sha256:' + "c" * 64
    )
    assert set(
        audit._audit_inventory(requirements, lock, "torch-2.6-cpu")["dependencies"]
    ) == {"torch", "torchvision"}


@pytest.mark.parametrize(
    "failure",
    ["skip", "missing", "empty", "malformed", "version", "duplicate", "vulns", "alias"],
)
def test_auditor_cannot_report_success_without_complete_results(
    export_fixture, failure
):
    inventory = _inventory(export_fixture)
    report = _report(inventory)
    if failure == "skip":
        report["dependencies"][0] = {
            "name": "torch",
            "skip_reason": "Could not be audited",
        }
    elif failure == "missing":
        report["dependencies"].pop()
    elif failure == "empty":
        report["dependencies"] = []
    elif failure == "malformed":
        report = {}
    elif failure == "version":
        report["dependencies"][0]["version"] = "2.13.0"
    elif failure == "duplicate":
        report["dependencies"].append(report["dependencies"][0])
    elif failure == "vulns":
        del report["dependencies"][0]["vulns"]
    else:
        report["dependencies"][0]["vulns"] = [{"id": "CVE-test", "aliases": [{}]}]
    result = audit._evaluate_audit(report, inventory, "torch-2.6-cpu", _policy())
    assert result["status"] == "failed"
    assert result["coverage"]["errors"]


@pytest.mark.parametrize(
    "today,accepted", [(date(2026, 9, 5), True), (date(2026, 10, 1), False)]
)
def test_torch_exception_matches_the_actual_build_and_expires(
    export_fixture, today, accepted
):
    inventory = _inventory(export_fixture)
    report = _report(inventory)
    report["dependencies"][0]["vulns"] = [{"id": "CVE-test", "aliases": []}]
    result = audit._evaluate_audit(
        report, inventory, "torch-2.6-cpu", _policy(), today=today
    )
    assert result["status"] == ("ok" if accepted else "failed")
    assert result["coverage"] == {"expected_count": 2, "audited_count": 2, "errors": []}
    findings = (
        result["accepted_exceptions"] if accepted else result["unresolved_findings"]
    )
    assert findings[0]["version"] == "2.6.0+cpu"
    assert findings[0]["audited_version"] == "2.6.0"


def test_profile_audit_uses_hashed_projection_and_records_coverage(
    tmp_path, monkeypatch, export_fixture
):
    requirements, lock = export_fixture
    (tmp_path / "uv.lock").write_text("lock placeholder")
    (tmp_path / "pyproject.toml").write_text("project placeholder")
    monkeypatch.setattr(audit.tomllib, "loads", lambda _: lock)

    def run(command, *, cwd):
        if command[0] == "uv":
            target = Path(command[command.index("--output-file") + 1])
            target.write_text(
                '{"components": []}' if "cyclonedx1.5" in command else requirements
            )
        else:
            projected = Path(
                next(
                    arg.split("=", 1)[1]
                    for arg in command
                    if arg.startswith("--requirement=")
                )
            )
            assert "torch==2.6.0 " in projected.read_text()
            assert "--hash=sha256:" + "a" * 64 in projected.read_text()
            target = Path(
                next(
                    arg.split("=", 1)[1]
                    for arg in command
                    if arg.startswith("--output=")
                )
            )
            report = _report(_inventory(export_fixture))
            # Auditor exit zero must not hide incomplete coverage.
            report["dependencies"][0] = {"name": "torch", "skip_reason": "unavailable"}
            target.write_text(json.dumps(report))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(audit, "_run", run)
    result = audit._audit_profile(
        tmp_path, "torch-2.6-cpu", Path("."), tmp_path / "output", _policy()
    )
    assert result["status"] == "failed"
    assert result["coverage"]["audited_count"] == 3
    assert {r["python_version"] for r in result["python_inventories"]} == {
        "3.10",
        "3.11",
        "3.12",
    }
    for report in result["python_inventories"]:
        assert report["audit_inventory_sha256"]
        assert report["audit_report_sha256"]
        assert report["audit_requirements_sha256"] != result["requirements_sha256"]


def test_profile_audit_rejects_a_finding_only_active_on_another_python(
    tmp_path, monkeypatch, export_fixture
):
    requirements, lock = copy.deepcopy(export_fixture)
    for name, version, marker in (
        ("numpy", "2.2.6", 'python_version < "3.11"'),
        ("numpy", "2.4.6", 'python_version == "3.11"'),
        ("numpy", "2.5.2", 'python_version >= "3.12"'),
    ):
        requirements += f"\n{name}=={version} ; {marker} --hash=sha256:" + "c" * 64
        lock["package"].append(
            {
                "name": name,
                "version": version,
                "source": {"registry": "https://pypi.org/simple"},
                "wheels": [{"hash": "sha256:" + "c" * 64}],
            }
        )
    (tmp_path / "uv.lock").write_text("lock placeholder")
    (tmp_path / "pyproject.toml").write_text("project placeholder")
    monkeypatch.setattr(audit.tomllib, "loads", lambda _: lock)
    queried = []

    def run(command, *, cwd):
        if command[0] == "uv":
            target = Path(command[command.index("--output-file") + 1])
            target.write_text(
                '{"components": []}' if "cyclonedx1.5" in command else requirements
            )
        else:
            paths = {
                key: Path(
                    next(
                        arg.split("=", 1)[1]
                        for arg in command
                        if arg.startswith(f"--{key}=")
                    )
                )
                for key in ("requirement", "output")
            }
            dependencies = []
            for line in paths["requirement"].read_text().splitlines():
                if not line or line.startswith("#"):
                    continue
                name, version = line.split()[0].split("==")
                queried.append((name, version))
                vulns = (
                    [{"id": "CVE-test-python310-only"}]
                    if (name == "numpy" and version == "2.2.6")
                    else []
                )
                dependencies.append({"name": name, "version": version, "vulns": vulns})
            paths["output"].write_text(json.dumps({"dependencies": dependencies}))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(audit, "_run", run)
    result = audit._audit_profile(
        tmp_path, "torch-2.6-cpu", Path("."), tmp_path / "output", _policy()
    )
    assert result["status"] == "failed"
    assert {("numpy", v) for v in ("2.2.6", "2.4.6", "2.5.2")} <= set(queried)
    assert result["unresolved_findings"][0]["python_version"] == "3.10"
