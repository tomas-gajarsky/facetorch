#!/usr/bin/env python3
"""Read GitHub check results for the exact candidate; never mutate repository settings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess


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
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repo):
        parser.error("repo must be OWNER/REPOSITORY")
    if not re.fullmatch(r"[a-f0-9]{40}", args.source_sha):
        parser.error("source-sha must be a full lowercase commit SHA")
    if args.checks_json:
        pages = json.loads(args.checks_json.read_text(encoding="utf-8"))
    else:
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
            timeout=120,
        )
        pages = json.loads(result.stdout)
    report = verify_source_checks(
        pages,
        source_sha=args.source_sha,
        policy=json.loads(args.policy.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
