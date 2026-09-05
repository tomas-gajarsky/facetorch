"""Numerical evidence remains verifiable after the runner workspace is gone."""

import hashlib
import json
from pathlib import Path
import shutil

import pytest
import yaml

from scripts.archive_numerical_evidence import create_index, verify_index
from test_b07_model_publication import _stage_summary

pytestmark = pytest.mark.release_blocker
SHA = "a" * 40


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path, value):
    path.write_text(json.dumps(value) + "\n")


@pytest.fixture
def evidence(tmp_path):
    root = tmp_path / "original"
    paths = []
    for mode in ("export", "validate"):
        summary_path = _stage_summary(root / mode, "2.6", devices=("cpu", "cuda"))
        summary = json.loads(summary_path.read_text())
        summary["mode"] = summary["exporter_arguments"]["mode"] = mode
        for result in summary["results"]:
            meta_path = Path(result["meta"])
            meta = json.loads(meta_path.read_text())
            meta["mode"] = meta["exporter_arguments"]["mode"] = mode
            _write(meta_path, meta)
            result["meta_sha256"] = _sha(meta_path)
        _write(summary_path, summary)
        paths.append(
            {
                "path": summary_path.relative_to(root).as_posix(),
                "sha256": _sha(summary_path),
            }
        )
    _write(
        root / "local-cuda-runner-report.json",
        {
            "source_sha": SHA,
            "source_clean": True,
            "status": "ok",
            "summaries": [paths[1]],
            "artifact_summaries": [paths[0]],
        },
    )
    return root


def test_archive_verifies_after_relocation_without_binaries(evidence, tmp_path):
    create_index(evidence, source_sha=SHA)
    relocated = tmp_path / "downloaded"
    shutil.copytree(evidence, relocated)
    shutil.rmtree(evidence)
    for pattern in ("*.pt", "*.pt2"):
        for binary in relocated.rglob(pattern):
            binary.unlink()
    assert verify_index(relocated, source_sha=SHA) == {
        "status": "ok",
        "source_sha": SHA,
        "summaries": 2,
        "model_records": 4,
        "cases": 8,
    }


@pytest.mark.parametrize(
    "problem",
    [
        "missing",
        "corrupt",
        "wrong-source",
        "missing-summary",
        "duplicate-summary",
        "escape",
    ],
)
def test_archive_rejects_broken_identity_or_coverage(evidence, problem):
    index = create_index(evidence, source_sha=SHA)
    record = index["summaries"][0]["records"][0]
    if problem == "missing":
        (evidence / record["path"]).unlink()
    elif problem == "corrupt":
        (evidence / record["path"]).write_text("{}")
    elif problem == "wrong-source":
        index["source_sha"] = "b" * 40
    elif problem == "missing-summary":
        index["summaries"].pop()
    elif problem == "duplicate-summary":
        index["summaries"].append(index["summaries"][0])
    else:
        record["path"] = "../outside.json"
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_index(evidence, source_sha=SHA, index=index)


@pytest.mark.parametrize(
    "problem",
    [
        "nan",
        "infinity",
        "exceeds",
        "negative",
        "duplicate-case",
        "missing-cross",
        "empty-comparison",
        "wrong-artifact",
        "wrong-reference",
    ],
)
def test_rehashed_but_invalid_numerical_records_are_rejected(evidence, problem):
    runner_path = evidence / "local-cuda-runner-report.json"
    runner = json.loads(runner_path.read_text())
    binding = runner["summaries"][0]
    summary_path = evidence / binding["path"]
    summary = json.loads(summary_path.read_text())
    result = summary["results"][0]
    meta_path = Path(result["meta"])
    meta = json.loads(meta_path.read_text())
    validation = meta["validation"]
    case = validation["devices"][0]["cases"][0]
    if problem in {"nan", "infinity", "exceeds", "negative"}:
        case["max_abs_diff_vs_reference"] = {
            "nan": float("nan"),
            "infinity": float("inf"),
            "exceeds": 1,
            "negative": -1,
        }[problem]
    elif problem == "duplicate-case":
        validation["devices"][0]["cases"].append(case)
        validation["devices"][0]["num_cases"] = 2
    elif problem == "missing-cross":
        validation["cross_device"] = []
    elif problem == "empty-comparison":
        case["numel_compared"] = 0
    elif problem == "wrong-artifact":
        meta["artifact_sha256"] = "f" * 64
    else:
        validation["devices"][1]["cases"][0]["reference_output_sha256"] = "f" * 64
    _write(meta_path, meta)
    result["meta_sha256"] = _sha(meta_path)
    _write(summary_path, summary)
    binding["sha256"] = _sha(summary_path)
    _write(runner_path, runner)
    with pytest.raises(ValueError):
        create_index(evidence, source_sha=SHA)
    assert not (evidence / "numerical-evidence-index.json").exists()


def test_release_uploads_records_and_checks_archive_before_preparing_plan():
    root = Path(__file__).resolve().parents[1]
    for workflow_name in ("release.yml", "local-gpu-release.yml"):
        workflow = (root / ".github/workflows" / workflow_name).read_text()
        assert "/numerical-evidence-index.json" in workflow
        assert "/torch-*/*/*.meta.json" in workflow
        assert "/runtime-validation/torch-*/*/*.meta.json" in workflow
    workflow = yaml.safe_load((root / ".github/workflows/release.yml").read_text())
    steps = workflow["jobs"]["assemble-candidate"]["steps"]
    check = next(
        i
        for i, step in enumerate(steps)
        if "archive_numerical_evidence.py verify" in step.get("run", "")
    )
    prepare = next(
        i
        for i, step in enumerate(steps)
        if "release_transaction.py prepare" in step.get("run", "")
    )
    assert check < prepare
