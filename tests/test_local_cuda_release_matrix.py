import os

import pytest

from scripts.run_local_cuda_release_matrix import _ensure_evidence_root


@pytest.mark.release_blocker
def test_new_evidence_root_is_0711_with_restrictive_umask(tmp_path):
    staging_root = tmp_path / "nested" / "evidence"
    previous_umask = os.umask(0o077)
    try:
        _ensure_evidence_root(staging_root)
    finally:
        os.umask(previous_umask)

    assert staging_root.stat().st_mode & 0o777 == 0o711
    _ensure_evidence_root(staging_root)


@pytest.mark.release_blocker
def test_existing_evidence_root_with_wrong_mode_fails_closed(tmp_path):
    staging_root = tmp_path / "evidence"
    staging_root.mkdir()
    staging_root.chmod(0o700)

    with pytest.raises(RuntimeError, match="must have mode 0711"):
        _ensure_evidence_root(staging_root)

    assert staging_root.stat().st_mode & 0o777 == 0o700
