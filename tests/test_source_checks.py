"""The final commit needs every source check, including the full CPU matrix."""

import copy
import json
from pathlib import Path
import subprocess

import pytest
import yaml

from scripts.verify_source_checks import verify_source_checks, wait_for_source_checks

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
    assert "--wait-timeout 5400 --poll-interval 30" in steps[gate]["run"]
    assert workflow["jobs"]["resolve-candidate"]["timeout-minutes"] > 90


class FakeClock:
    def __init__(self):
        self.now = 0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def _wait(fetch, clock, **kwargs):
    return wait_for_source_checks(
        fetch, source_sha=SHA, policy=POLICY, clock=clock, sleep=clock.sleep, **kwargs
    )


@pytest.mark.parametrize(
    "initial",
    [
        "missing",
        "queued",
        "in_progress",
        "waiting",
        "pending",
        "requested",
        "wrong-app",
        "rerun",
    ],
)
def test_waits_for_registration_and_completion_without_accepting_stale_checks(initial):
    pages = _checks()
    target = pages[0]["check_runs"][-1]
    if initial == "missing":
        pages[0]["check_runs"].pop()
    elif initial == "wrong-app":
        target["app"]["id"] = 999
    elif initial == "rerun":
        pending = copy.deepcopy(target)
        pending.update(id=1000, status="in_progress", conclusion=None)
        pages.append({"check_runs": [pending]})
    else:
        target.update(status=initial, conclusion=None)
    successful = _checks()
    successful[0]["check_runs"][-1]["id"] = 1000
    responses = iter([pages, successful])
    clock = FakeClock()
    budgets = []

    def fetch(timeout):
        budgets.append(timeout)
        return next(responses)

    report = _wait(fetch, clock, wait_timeout=10, poll_interval=3)
    assert report["status"] == "ok"
    assert report["reason"] == "checks-complete"
    assert report["checks"][-1]["id"] == 1000
    assert report["attempts"] == 2
    assert clock.sleeps == [3]
    assert budgets == [10, 7]


@pytest.mark.parametrize(
    "conclusion",
    [
        "failure",
        "cancelled",
        "skipped",
        "neutral",
        "timed_out",
        "action_required",
        "stale",
        None,
    ],
)
def test_terminal_failure_does_not_wait(conclusion):
    pages = _checks()
    pages[0]["check_runs"][-1]["conclusion"] = conclusion
    clock = FakeClock()
    report = _wait(lambda _: pages, clock, wait_timeout=10)
    assert report["status"] == "failed"
    assert report["reason"] == "terminal-check-result"
    assert report["attempts"] == 1
    assert clock.sleeps == []


@pytest.mark.parametrize("initial", ["missing", "wrong-app", "in_progress"])
def test_unfinished_checks_time_out_with_last_evidence(initial):
    pages = _checks()
    if initial == "missing":
        pages[0]["check_runs"].pop()
    elif initial == "wrong-app":
        pages[0]["check_runs"][-1]["app"]["id"] = 999
    else:
        pages[0]["check_runs"][-1].update(status=initial, conclusion=None)
    clock = FakeClock()
    report = _wait(lambda _: pages, clock, wait_timeout=5, poll_interval=3)
    assert report["status"] == "failed"
    assert report["reason"] == "timeout"
    assert report["wait_seconds"] == 5
    assert report["attempts"] == 2
    assert report["checks"][-1]["status"] == (
        "in_progress" if initial == "in_progress" else "missing"
    )
    assert clock.sleeps == [3, 2]


def test_api_time_is_included_in_the_total_deadline():
    clock = FakeClock()

    def slow_fetch(timeout):
        assert timeout == 5
        clock.now += timeout
        return _checks()

    report = _wait(slow_fetch, clock, wait_timeout=5)
    assert report["status"] == "failed"
    assert report["reason"] == "timeout"
    assert clock.sleeps == []


@pytest.mark.parametrize(
    "error",
    [subprocess.TimeoutExpired("gh", 5), subprocess.CalledProcessError(1, "gh")],
)
def test_api_failure_leaves_a_failed_report(error):
    def fetch(_):
        raise error

    report = _wait(fetch, FakeClock(), wait_timeout=5)
    assert report["status"] == "failed"
    assert report["reason"] in {"api-timeout", "api-error"}


def test_wrong_commit_is_rejected_while_waiting():
    pages = _checks()
    pages[0]["check_runs"][0]["head_sha"] = "b" * 40
    with pytest.raises(ValueError, match="candidate commit"):
        _wait(lambda _: pages, FakeClock(), wait_timeout=5)


def test_default_one_shot_never_waits_for_incomplete_checks():
    clock = FakeClock()
    report = _wait(lambda _: [{"check_runs": []}], clock)
    assert report["status"] == "failed"
    assert report["reason"] == "checks-not-ready"
    assert report["attempts"] == 1
    assert clock.sleeps == []


@pytest.mark.parametrize(
    "options",
    [
        dict(wait_timeout=-1),
        dict(wait_timeout=float("nan")),
        dict(wait_timeout=float("inf")),
        dict(poll_interval=0),
        dict(poll_interval=float("inf")),
    ],
)
def test_wait_budget_must_be_finite_and_valid(options):
    with pytest.raises(ValueError):
        _wait(lambda _: _checks(), FakeClock(), **options)
