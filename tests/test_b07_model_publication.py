import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.model_cohort_publication import (
    PublicationError,
    create_approval_template,
    prepare_publication_plan,
    publish_publication_plan,
    validate_approval,
    verify_publication_plan,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _stage_summary(root, cohort, model_ids=("model-a", "model-b"), devices=("cpu",)):
    results = []
    for index, model_id in enumerate(model_ids, start=1):
        model_root = root / model_id
        model_root.mkdir(parents=True, exist_ok=True)
        golden_reference = root / "golden-references" / model_id / "golden-reference.pt"
        golden_reference.parent.mkdir(parents=True, exist_ok=True)
        golden_reference.write_bytes(f"golden:{model_id}".encode())
        golden_sha = _sha256(golden_reference)
        artifact = model_root / f"model-torch{cohort}.pt2"
        artifact.write_bytes(f"artifact:{model_id}:{cohort}".encode())
        metadata = artifact.with_suffix(".pt2.meta.json")
        reference_output_sha = hashlib.sha256(
            f"reference:{model_id}:case-1".encode()
        ).hexdigest()
        exported_output_sha = hashlib.sha256(
            f"exported:{model_id}:case-1".encode()
        ).hexdigest()
        metadata_value = {
            "schema_version": 2,
            "model_id": model_id,
            "repo_id": f"owner/{model_id}",
            "artifact": artifact.name,
            "artifact_sha256": _sha256(artifact),
            "artifact_size_bytes": artifact.stat().st_size,
            "validation": {
                "status": "ok",
                "num_cases": len(devices),
                "max_abs_tolerance": 1e-4,
                "cross_device_max_abs_tolerance": 2e-4,
                "cross_device_mean_abs_tolerance": 2e-5,
                "fixed_reference_device": "cpu",
                "requested_devices": list(devices),
                "golden_reference": {
                    "schema_version": 1,
                    "status": "recorded" if cohort == "2.6" else "reused",
                    "source_cohort": "2.6",
                    "sha256": golden_sha,
                    "size_bytes": golden_reference.stat().st_size,
                    "case_count": 1,
                },
                "devices": [
                    {
                        "device": device,
                        "status": "ok",
                        "num_cases": 1,
                        "cases": [
                            {
                                "case_id": "case-1",
                                "status": "ok",
                                "input_sha256": hashlib.sha256(
                                    f"input:{model_id}:case-1".encode()
                                ).hexdigest(),
                                "reference_output_sha256": reference_output_sha,
                                "exported_output_sha256": exported_output_sha,
                                "max_abs_diff_vs_reference": 0.0,
                            }
                        ],
                    }
                    for device in devices
                ],
            },
        }
        _write_json(metadata, metadata_value)
        results.append(
            {
                "model_id": model_id,
                "repo_id": f"owner/{model_id}",
                "status": "ok",
                "artifact": str(artifact),
                "meta": str(metadata),
                "sha256": _sha256(artifact),
                "meta_sha256": _sha256(metadata),
                "size_bytes": artifact.stat().st_size,
                "validation_status": "ok",
                "num_cases": len(devices),
                "golden_reference": str(golden_reference),
                "golden_reference_sha256": golden_sha,
                "golden_reference_size_bytes": golden_reference.stat().st_size,
            }
        )

    summary = root / f"summary-torch{cohort}.json"
    _write_json(
        summary,
        {
            "schema_version": 2,
            "status": "ok",
            "torch_minor": cohort,
            "validate_devices": list(devices),
            "requested_model_ids": list(model_ids),
            "results": results,
        },
    )
    return summary


def _prepare(root, summaries, model_ids=("model-a", "model-b")):
    plan_path = root / "publication-plan.json"
    revisions = {
        model_id: f"{index:x}" * 40 for index, model_id in enumerate(model_ids, start=1)
    }
    prepare_publication_plan(
        staging_root=root,
        summary_paths=summaries,
        base_revisions=revisions,
        manifest_repo_id="owner/facetorch-model-manifest",
        manifest_base_revision="f" * 40,
        output_path=plan_path,
    )
    return plan_path


def _approve(plan_path, approval_path):
    approval = create_approval_template(plan_path, approval_path)
    approval.update(
        {
            "status": "approved",
            "approved_by": "release-reviewer",
            "approved_at_utc": "2026-08-21T12:00:00+00:00",
        }
    )
    _write_json(approval_path, approval)


class _FakeHubApi:
    def __init__(self, fail_once_for=None):
        self.fail_once_for = fail_once_for
        self.branches = []
        self.commits = []
        self.verified = []

    def create_branch(self, **kwargs):
        self.branches.append(kwargs)

    def create_commit(self, **kwargs):
        if kwargs["repo_id"] == self.fail_once_for:
            self.fail_once_for = None
            raise RuntimeError("deliberate candidate upload failure")
        self.commits.append(kwargs)
        digit = format(len(self.commits), "x")
        return SimpleNamespace(oid=digit * 40)

    def repo_info(self, *, repo_id, revision):
        self.verified.append((repo_id, revision))
        return SimpleNamespace(sha=revision)


@pytest.mark.release_blocker
def test_plan_rejects_a_required_device_that_is_not_ok(tmp_path):
    summary = _stage_summary(tmp_path, "2.11", model_ids=("model-a",))
    summary_value = json.loads(summary.read_text())
    metadata = Path(summary_value["results"][0]["meta"])
    metadata_value = json.loads(metadata.read_text())
    metadata_value["validation"]["requested_devices"] = ["cpu", "cuda"]
    metadata_value["validation"]["devices"].append(
        {"device": "cuda", "status": "skipped", "num_cases": 0}
    )
    _write_json(metadata, metadata_value)
    summary_value["results"][0]["meta_sha256"] = _sha256(metadata)
    _write_json(summary, summary_value)

    with pytest.raises(PublicationError, match="not ok"):
        _prepare(tmp_path, [summary], model_ids=("model-a",))


@pytest.mark.release_blocker
def test_plan_is_deterministic_and_detects_staged_byte_changes(tmp_path):
    summary = _stage_summary(tmp_path, "2.11", model_ids=("model-a",))
    first = _prepare(tmp_path, [summary], model_ids=("model-a",))
    second = tmp_path / "second-plan.json"
    prepare_publication_plan(
        staging_root=tmp_path,
        summary_paths=[summary],
        base_revisions={"model-a": "1" * 40},
        manifest_repo_id="owner/facetorch-model-manifest",
        manifest_base_revision="f" * 40,
        output_path=second,
    )
    assert first.read_bytes() == second.read_bytes()

    plan = json.loads(first.read_text())
    (tmp_path / plan["models"][0]["artifact_path"]).write_bytes(b"changed")
    with pytest.raises(PublicationError, match="changed after planning"):
        verify_publication_plan(first)


@pytest.mark.release_blocker
def test_plan_detects_golden_reference_changes(tmp_path):
    summary = _stage_summary(tmp_path, "2.6", model_ids=("model-a",))
    plan = _prepare(tmp_path, [summary], model_ids=("model-a",))
    plan_value = json.loads(plan.read_text())
    golden = tmp_path / plan_value["models"][0]["golden_reference_path"]
    golden.write_bytes(b"changed-golden-reference")

    with pytest.raises(PublicationError, match="Golden reference changed"):
        verify_publication_plan(plan)


@pytest.mark.release_blocker
def test_plan_rejects_golden_status_that_disagrees_with_source_cohort(tmp_path):
    summary = _stage_summary(tmp_path, "2.6", model_ids=("model-a",))
    summary_value = json.loads(summary.read_text())
    metadata = Path(summary_value["results"][0]["meta"])
    metadata_value = json.loads(metadata.read_text())
    metadata_value["validation"]["golden_reference"]["status"] = "reused"
    _write_json(metadata, metadata_value)
    summary_value["results"][0]["meta_sha256"] = _sha256(metadata)
    _write_json(summary, summary_value)

    with pytest.raises(PublicationError, match="status disagrees"):
        _prepare(tmp_path, [summary], model_ids=("model-a",))


@pytest.mark.release_blocker
def test_publish_requires_digest_bound_complete_plan_approval(tmp_path):
    summary = _stage_summary(tmp_path, "2.11", model_ids=("model-a",))
    plan = _prepare(tmp_path, [summary], model_ids=("model-a",))
    approval = tmp_path / "approval.json"
    create_approval_template(plan, approval)

    with pytest.raises(PublicationError, match="approve the complete plan"):
        publish_publication_plan(
            plan_path=plan,
            approval_path=approval,
            receipt_path=tmp_path / "receipt.json",
            api=_FakeHubApi(),
        )

    approval_value = json.loads(approval.read_text())
    approval_value.update(
        {
            "status": "approved",
            "approved_by": "reviewer",
            "approved_at_utc": "2026-08-21T12:00:00+00:00",
            "plan_sha256": "0" * 64,
        }
    )
    _write_json(approval, approval_value)
    with pytest.raises(PublicationError, match="not bound"):
        validate_approval(plan, approval)


@pytest.mark.release_blocker
def test_publish_commits_each_model_atomically_then_manifest_last(tmp_path):
    summary = _stage_summary(tmp_path, "2.11")
    plan = _prepare(tmp_path, [summary])
    approval = tmp_path / "approval.json"
    _approve(plan, approval)
    api = _FakeHubApi()

    receipt = publish_publication_plan(
        plan_path=plan,
        approval_path=approval,
        receipt_path=tmp_path / "receipt.json",
        api=api,
    )

    assert receipt["status"] == "complete"
    assert [call["repo_id"] for call in api.commits] == [
        "owner/model-a",
        "owner/model-b",
        "owner/facetorch-model-manifest",
    ]
    assert [len(call["operations"]) for call in api.commits] == [2, 2, 1]
    assert all(
        call["revision"].startswith("facetorch-candidate-") for call in api.commits
    )
    assert receipt["manifest"]["commit_revision"] == "3" * 40


@pytest.mark.release_blocker
def test_failed_publish_is_resumable_without_early_manifest(tmp_path):
    summary = _stage_summary(tmp_path, "2.11")
    plan = _prepare(tmp_path, [summary])
    approval = tmp_path / "approval.json"
    receipt_path = tmp_path / "receipt.json"
    _approve(plan, approval)
    first_api = _FakeHubApi(fail_once_for="owner/model-b")

    with pytest.raises(RuntimeError, match="deliberate"):
        publish_publication_plan(
            plan_path=plan,
            approval_path=approval,
            receipt_path=receipt_path,
            api=first_api,
        )

    interrupted = json.loads(receipt_path.read_text())
    assert interrupted["status"] == "incomplete"
    assert set(interrupted["models"]) == {"model-a"}
    assert interrupted["manifest"] is None

    resumed_api = _FakeHubApi()
    complete = publish_publication_plan(
        plan_path=plan,
        approval_path=approval,
        receipt_path=receipt_path,
        api=resumed_api,
    )
    assert complete["status"] == "complete"
    assert resumed_api.verified == [("owner/model-a", "1" * 40)]
    assert [call["repo_id"] for call in resumed_api.commits] == [
        "owner/model-b",
        "owner/facetorch-model-manifest",
    ]


@pytest.mark.release_blocker
def test_multiple_cohorts_for_one_model_are_one_repository_commit(tmp_path):
    summary_26 = _stage_summary(tmp_path, "2.6", model_ids=("model-a",))
    summary_211 = _stage_summary(tmp_path, "2.11", model_ids=("model-a",))
    plan = _prepare(tmp_path, [summary_26, summary_211], model_ids=("model-a",))
    approval = tmp_path / "approval.json"
    _approve(plan, approval)
    api = _FakeHubApi()

    plan_value = json.loads(plan.read_text())
    assert plan_value["cross_cohort_comparisons"] == [
        {
            "device": "cpu",
            "exact_export_cases": 1,
            "guaranteed_max_abs_limit": 0.0002,
            "left_cohort": "2.6",
            "model_id": "model-a",
            "num_cases": 1,
            "right_cohort": "2.11",
            "worst_guaranteed_max_abs": 0.0,
        }
    ]

    publish_publication_plan(
        plan_path=plan,
        approval_path=approval,
        receipt_path=tmp_path / "receipt.json",
        api=api,
    )

    assert [call["repo_id"] for call in api.commits] == [
        "owner/model-a",
        "owner/facetorch-model-manifest",
    ]
    assert len(api.commits[0]["operations"]) == 4


@pytest.mark.release_blocker
def test_cross_cohort_cuda_bound_uses_cross_device_tolerance(tmp_path):
    summary_26 = _stage_summary(
        tmp_path, "2.6", model_ids=("model-a",), devices=("cpu", "cuda")
    )
    summary_211 = _stage_summary(
        tmp_path, "2.11", model_ids=("model-a",), devices=("cpu", "cuda")
    )

    plan = json.loads(
        _prepare(
            tmp_path, [summary_26, summary_211], model_ids=("model-a",)
        ).read_text()
    )

    comparisons = {item["device"]: item for item in plan["cross_cohort_comparisons"]}
    assert comparisons["cpu"]["guaranteed_max_abs_limit"] == 2e-4
    assert comparisons["cuda"]["guaranteed_max_abs_limit"] == 4e-4


@pytest.mark.release_blocker
def test_cross_cohort_publish_requires_one_immutable_reference(tmp_path):
    summary_26 = _stage_summary(tmp_path, "2.6", model_ids=("model-a",))
    summary_211 = _stage_summary(tmp_path, "2.11", model_ids=("model-a",))
    summary_value = json.loads(summary_211.read_text())
    metadata = Path(summary_value["results"][0]["meta"])
    metadata_value = json.loads(metadata.read_text())
    metadata_value["validation"]["devices"][0]["cases"][0][
        "reference_output_sha256"
    ] = ("0" * 64)
    _write_json(metadata, metadata_value)
    summary_value["results"][0]["meta_sha256"] = _sha256(metadata)
    _write_json(summary_211, summary_value)

    with pytest.raises(PublicationError, match="immutable golden reference"):
        _prepare(tmp_path, [summary_26, summary_211], model_ids=("model-a",))
