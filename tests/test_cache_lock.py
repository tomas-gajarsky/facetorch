"""Kernel locking across real processes, without relying on PID metadata."""

import multiprocessing
import os
import time
from pathlib import Path

import pytest

from facetorch.downloader import _DirectoryLock
from facetorch.exceptions import CacheLockError

pytestmark = [
    pytest.mark.release_blocker,
    pytest.mark.skipif(os.name != "posix", reason="POSIX cache contract"),
]


def _hold_lock(path, ready, release):
    with _DirectoryLock(Path(path), timeout=2):
        Path(ready).touch()
        deadline = time.monotonic() + 10
        while not Path(release).exists() and time.monotonic() < deadline:
            time.sleep(0.02)


@pytest.mark.parametrize("kill_owner", [False, True])
def test_real_process_owner_excludes_waiters_and_releases_after_exit(
    tmp_path, kill_owner
):
    context = multiprocessing.get_context("spawn")
    ready, release = tmp_path / "ready", tmp_path / "release"
    path = tmp_path / ".download.lock"
    process = context.Process(target=_hold_lock, args=(str(path), ready, release))
    process.start()
    try:
        deadline = time.monotonic() + 15
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists(), "owner failed to acquire the cache lock"
        started = time.monotonic()
        with pytest.raises(CacheLockError, match="do not remove"):
            with _DirectoryLock(path, timeout=0.15):
                pytest.fail("a live owner must exclude a second process")
        assert time.monotonic() - started < 1.0
        if kill_owner:
            process.kill()
        else:
            release.touch()
        process.join(5)
        assert not process.is_alive()
        with _DirectoryLock(path, timeout=1):
            assert path.is_file()
    finally:
        if process.is_alive():
            process.kill()
        process.join(5)


def test_lock_file_inode_is_retained_across_owners(tmp_path):
    path = tmp_path / "lock"
    with _DirectoryLock(path):
        inode = path.stat().st_ino
        with pytest.raises(CacheLockError, match="Timed out"):
            with _DirectoryLock(path, timeout=0):
                pytest.fail("separate opens in one process must also compete")
    assert path.is_file()
    with _DirectoryLock(path):
        assert path.stat().st_ino == inode


def test_legacy_lock_is_never_reclaimed_using_a_foreign_pid(tmp_path):
    path = tmp_path / "lock"
    path.mkdir()
    owner = path / "owner.json"
    owner.write_text('{"pid":1,"process_identity":"foreign-container"}')
    with pytest.raises(CacheLockError, match="Stop all users"):
        with _DirectoryLock(path, timeout=0.01):
            pytest.fail("legacy cache ownership cannot be proved locally")
    assert owner.read_text() == '{"pid":1,"process_identity":"foreign-container"}'


def test_symlinks_and_nonregular_lock_files_are_rejected(tmp_path):
    target = tmp_path / "target"
    target.write_text("untouched")
    symlink = tmp_path / "symlink"
    symlink.symlink_to(target)
    with pytest.raises(CacheLockError, match="Could not open"):
        with _DirectoryLock(symlink):
            pass
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(CacheLockError, match="not a regular file"):
        with _DirectoryLock(fifo):
            pass
    assert target.read_text() == "untouched"


@pytest.mark.parametrize("timeout", [-1, float("nan"), float("inf"), True])
def test_lock_rejects_invalid_deadlines(tmp_path, timeout):
    with pytest.raises(CacheLockError, match="finite and non-negative"):
        _DirectoryLock(tmp_path / "lock", timeout=timeout)


def test_lock_object_can_be_reused_but_not_nested(tmp_path):
    lock = _DirectoryLock(tmp_path / "lock")
    with lock:
        with pytest.raises(CacheLockError, match="already acquired"):
            with lock:
                pass
    with lock:
        pass
