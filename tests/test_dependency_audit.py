"""Advisory coverage and exact local-build exception regressions."""

import copy
import json
import subprocess
from datetime import date
from pathlib import Path

import pytest

from scripts import audit_dependencies as audit

pytestmark = pytest.mark.release_blocker


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
    assert result["coverage"]["audited_count"] == 1
    assert result["audit_inventory_sha256"]
    assert result["audit_requirements_sha256"] != result["requirements_sha256"]
