"""The final commit needs every source check, including the full CPU matrix."""

import copy
import json
from pathlib import Path

import pytest
import yaml

from scripts.verify_source_checks import verify_source_checks

ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 40
POLICY = json.loads((ROOT / "security/required-source-checks.json").read_text())
pytestmark = pytest.mark.release_blocker


def _checks():
    return [
        {
            "check_runs": [
                {
                    "id": index,
                    "name": name,
                    "head_sha": SHA,
                    "app": {"id": POLICY["github_actions_app_id"]},
                    "status": "completed",
                    "conclusion": "success",
                }
                for index, name in enumerate(POLICY["required_checks"], 1)
            ]
        }
    ]


@pytest.mark.parametrize(
    "failure",
    ["failure", "cancelled", "skipped", "neutral", "pending", "missing", "wrong-app"],
)
def test_any_incomplete_supported_matrix_blocks_release(failure):
    pages = _checks()
    matrix = pages[0]["check_runs"][-1]
    assert matrix["name"] == "cpu-cohorts-complete"
    if failure == "pending":
        matrix["status"] = "in_progress"
    elif failure == "missing":
        pages[0]["check_runs"].pop()
    elif failure == "wrong-app":
        matrix["app"]["id"] = 999
    else:
        matrix["conclusion"] = failure
    assert (
        verify_source_checks(pages, source_sha=SHA, policy=POLICY)["status"] == "failed"
    )


def test_exact_successful_checks_pass_but_old_success_cannot_mask_a_rerun():
    pages = _checks()
    assert verify_source_checks(pages, source_sha=SHA, policy=POLICY)["status"] == "ok"
    rerun = copy.deepcopy(pages[0]["check_runs"][-1])
    rerun.update(id=1000, status="in_progress", conclusion=None)
    pages.append({"check_runs": [rerun]})
    assert (
        verify_source_checks(pages, source_sha=SHA, policy=POLICY)["status"] == "failed"
    )


def test_checks_from_another_commit_are_rejected():
    pages = _checks()
    pages[0]["check_runs"][0]["head_sha"] = "b" * 40
    with pytest.raises(ValueError, match="candidate commit"):
        verify_source_checks(pages, source_sha=SHA, policy=POLICY)


def test_cpu_aggregate_runs_after_failure_and_checks_all_matrix_results():
    workflow = yaml.safe_load((ROOT / ".github/workflows/cpu-cohorts.yml").read_text())
    job = workflow["jobs"]["cpu-cohorts-complete"]
    assert job["if"] == "always()"
    assert job["needs"] == "cpu-cohort"
    assert job["steps"][0]["env"]["MATRIX_RESULT"] == "${{ needs.cpu-cohort.result }}"
    assert job["steps"][0]["run"] == 'test "$MATRIX_RESULT" = success'
    matrix = workflow["jobs"]["cpu-cohort"]
    assert matrix.get("continue-on-error", False) is False
    assert not any(
        row.get("continue-on-error", False)
        for row in matrix["strategy"]["matrix"]["include"]
    )


def test_candidate_resolution_checks_source_before_model_preparation():
    workflow = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())
    steps = workflow["jobs"]["resolve-candidate"]["steps"]
    gate = next(
        i
        for i, step in enumerate(steps)
        if "scripts/verify_source_checks.py" in step.get("run", "")
    )
    manifest = next(
        i
        for i, step in enumerate(steps)
        if "fetch-model-manifest" in step.get("run", "")
    )
    assert gate < manifest
    assert workflow["permissions"]["checks"] == "read"
