#!/usr/bin/env python3
"""Exercise the installed cache lock in separate Docker PID namespaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import time
import uuid


def check(image: str) -> dict:
    """Check exclusion and crash recovery; remove only containers created here."""
    prefix = "facetorch-lock-check-" + uuid.uuid4().hex[:12]
    names = []

    def docker(*args, check=True):
        return subprocess.run(
            ["docker", *args], capture_output=True, text=True, check=check, timeout=60
        )

    identity = docker("image", "inspect", "--format", "{{.Id}}", image).stdout.strip()
    with tempfile.TemporaryDirectory(prefix=prefix) as directory:
        shared = Path(directory)
        # The production image runs as UID 10001. This empty, disposable test
        # directory contains no user cache or credentials.
        shared.chmod(0o777)

        def run(role, code, *, detach=False):
            name = prefix + "-" + role
            names.append(name)
            args = ["run", "--pull", "never", "--name", name]
            if detach:
                args.append("--detach")
            args.extend(
                [
                    "--network",
                    "none",
                    "--read-only",
                    "--cap-drop",
                    "ALL",
                    "--security-opt",
                    "no-new-privileges",
                    "--tmpfs",
                    "/tmp",
                    "--mount",
                    f"type=bind,src={shared},dst=/shared",
                    "--entrypoint",
                    "python",
                    image,
                    "-c",
                    code,
                ]
            )
            return docker(*args)

        prelude = (
            "import json, os, time\nfrom pathlib import Path\n"
            "from facetorch.downloader import _DirectoryLock\n"
            "from facetorch.exceptions import CacheLockError\n"
            "assert os.getpid() == 1\n"
        )
        try:
            run(
                "holder",
                prelude
                + (
                    "with _DirectoryLock(Path('/shared/model.lock'), timeout=1):\n"
                    " Path('/shared/ready').write_text(str(os.getpid()))\n"
                    " time.sleep(45)\n"
                ),
                detach=True,
            )
            deadline = time.monotonic() + 30
            while not (shared / "ready").exists():
                if time.monotonic() > deadline:
                    logs = docker("logs", names[0], check=False)
                    raise RuntimeError(
                        "Container lock holder did not start: " + logs.stderr
                    )
                time.sleep(0.05)
            result = run(
                "contender",
                prelude
                + (
                    "try:\n"
                    " with _DirectoryLock(Path('/shared/model.lock'), timeout=0.3):\n"
                    "  raise AssertionError('two PID-1 containers acquired one lock')\n"
                    "except CacheLockError:\n"
                    " print(json.dumps({'excluded': True, 'pid': os.getpid()}))\n"
                ),
            )
            contender = json.loads(result.stdout)
            inode = (shared / "model.lock").stat().st_ino
            docker("kill", names[0])
            result = run(
                "recovery",
                prelude
                + (
                    "with _DirectoryLock(Path('/shared/model.lock'), timeout=2):\n"
                    " print(json.dumps({'recovered': True, 'pid': os.getpid()}))\n"
                ),
            )
            recovery = json.loads(result.stdout)
            if (shared / "model.lock").stat().st_ino != inode:
                raise RuntimeError("Lock inode changed during crash recovery")
            return {
                "schema_version": 1,
                "status": "ok",
                "image": image,
                "image_id": identity,
                "installed_wheel": True,
                "contender": contender,
                "after_holder_killed": recovery,
                "persistent_inode": True,
            }
        finally:
            for name in names:
                docker("rm", "--force", name, check=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="An already built local image")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = check(args.image)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
