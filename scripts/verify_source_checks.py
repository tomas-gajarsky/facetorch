#!/usr/bin/env python3
"""Read GitHub check results for the exact candidate; never mutate repository settings."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import subprocess
import time


def verify_source_checks(pages, *, source_sha: str, policy: dict) -> dict:
    if not re.fullmatch(r"[a-f0-9]{40}", source_sha):
        raise ValueError("Expected a full immutable source SHA")
    required = policy.get("required_checks")
    if (
        policy.get("schema_version") != 1
        or not isinstance(required, list)
        or not required
        or any(not isinstance(name, str) or not name for name in required)
        or len(set(required)) != len(required)
        or not isinstance(policy.get("github_actions_app_id"), int)
    ):
        raise ValueError("Invalid required source-check policy")
    if not isinstance(pages, list) or not pages:
        raise ValueError("Missing GitHub check-run pages")
    latest = {}
    for page in pages:
        if not isinstance(page, dict) or not isinstance(page.get("check_runs"), list):
            raise ValueError("Malformed GitHub check-run page")
        for check in page["check_runs"]:
            if not isinstance(check, dict) or check.get("head_sha") != source_sha:
                raise ValueError("Check result does not belong to the candidate commit")
            name = check.get("name")
            if (
                name not in required
                or check.get("app", {}).get("id") != policy["github_actions_app_id"]
            ):
                continue
            if not isinstance(check.get("id"), int):
                raise ValueError("Check result has no run identity")
            if name not in latest or check["id"] > latest[name]["id"]:
                latest[name] = check
    results = []
    for name in required:
        check = latest.get(name, {})
        results.append(
            {
                "name": name,
                "id": check.get("id"),
                "url": check.get("html_url"),
                "status": check.get("status", "missing"),
                "conclusion": check.get("conclusion"),
            }
        )
    return {
        "schema_version": 1,
        "source_sha": source_sha,
        "status": (
            "ok"
            if all(
                item["status"] == "completed" and item["conclusion"] == "success"
                for item in results
            )
            else "failed"
        ),
        "checks": results,
    }


def wait_for_source_checks(
    fetch_pages,
    *,
    source_sha: str,
    policy: dict,
    wait_timeout: float = 0,
    poll_interval: float = 30,
    clock=time.monotonic,
    sleep=time.sleep,
) -> dict:
    """Wait for registration/completion, never accepting an incomplete check.

    A zero timeout preserves one-shot verification. A positive timeout bounds
    both API calls and sleeps. Failed final results and malformed evidence fail
    immediately; only missing or nonterminal checks can become ready by waiting.
    """
    if not math.isfinite(wait_timeout) or wait_timeout < 0:
        raise ValueError("wait-timeout must be finite and nonnegative")
    if not math.isfinite(poll_interval) or poll_interval <= 0:
        raise ValueError("poll-interval must be finite and positive")
    report = verify_source_checks(
        [{"check_runs": []}], source_sha=source_sha, policy=policy
    )
    started = clock()
    deadline = started + wait_timeout
    attempts = 0
    pending = {"missing", "queued", "in_progress", "waiting", "requested", "pending"}
    while True:
        remaining = deadline - clock()
        if wait_timeout and remaining <= 0:
            reason = "timeout"
            break
        attempts += 1
        try:
            pages = fetch_pages(min(120, remaining) if wait_timeout else 120)
        except subprocess.TimeoutExpired:
            reason = "api-timeout"
            break
        except subprocess.CalledProcessError:
            reason = "api-error"
            break
        report = verify_source_checks(pages, source_sha=source_sha, policy=policy)
        if wait_timeout and clock() >= deadline:
            report["status"] = "failed"
            reason = "timeout"
            break
        if report["status"] == "ok":
            reason = "checks-complete"
            break
        if any(
            item["status"] not in pending
            and not (item["status"] == "completed" and item["conclusion"] == "success")
            for item in report["checks"]
        ):
            reason = "terminal-check-result"
            break
        if not wait_timeout:
            reason = "checks-not-ready"
            break
        sleep(min(poll_interval, max(0, deadline - clock())))
    report.update(
        reason=reason, attempts=attempts, wait_seconds=round(clock() - started, 3)
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument(
        "--policy", type=Path, default=Path("security/required-source-checks.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checks-json", type=Path, help="Verify saved API pages offline"
    )
    parser.add_argument(
        "--wait-timeout",
        type=float,
        default=0,
        help="Seconds to wait for live checks; zero performs one read (default)",
    )
    parser.add_argument("--poll-interval", type=float, default=30)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repo):
        parser.error("repo must be OWNER/REPOSITORY")
    if not re.fullmatch(r"[a-f0-9]{40}", args.source_sha):
        parser.error("source-sha must be a full lowercase commit SHA")
    if not math.isfinite(args.wait_timeout) or args.wait_timeout < 0:
        parser.error("wait-timeout must be finite and nonnegative")
    if not math.isfinite(args.poll_interval) or args.poll_interval <= 0:
        parser.error("poll-interval must be finite and positive")
    if args.checks_json and args.wait_timeout:
        parser.error("Cannot wait for saved offline check results")

    def fetch_pages(timeout):
        if args.checks_json:
            return json.loads(args.checks_json.read_text(encoding="utf-8"))
        result = subprocess.run(
            [
                "gh",
                "api",
                "--paginate",
                "--slurp",
                f"repos/{args.repo}/commits/{args.source_sha}/check-runs?filter=all&per_page=100",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return json.loads(result.stdout)

    report = wait_for_source_checks(
        fetch_pages,
        source_sha=args.source_sha,
        policy=json.loads(args.policy.read_text(encoding="utf-8")),
        wait_timeout=args.wait_timeout,
        poll_interval=args.poll_interval,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
