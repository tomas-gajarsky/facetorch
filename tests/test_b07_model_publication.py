import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.model_cohort_publication as publication
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
                "mean_abs_tolerance": 1e-5,
                "cross_device_max_abs_tolerance": 2e-4,
                "cross_device_mean_abs_tolerance": 2e-5,
                "fixed_reference_device": "cpu",
                "worst_case_id": "case-1",
                "worst_device": "cpu",
                "worst_max_abs_diff_vs_reference": 0.0,
                "worst_mean_abs_diff_vs_reference": 0.0,
                "failures": [],
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
                        "worst_case_id": "case-1",
                        "worst_max_abs_diff_vs_reference": 0.0,
                        "worst_mean_abs_diff_vs_reference": 0.0,
                        "reference_execution_device": "cpu",
                        "reference_tolerance_kind": (
                            "same_device" if device == "cpu" else "cross_device"
                        ),
                        "failures": [],
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
                                "mean_abs_diff_vs_reference": 0.0,
                                "numel_compared": 4,
                                "reference_execution_device": "cpu",
                                "reference_max_abs_tolerance": (
                                    1e-4 if device == "cpu" else 2e-4
                                ),
                                "reference_mean_abs_tolerance": (
                                    1e-5 if device == "cpu" else 2e-5
                                ),
                            }
                        ],
                    }
                    for device in devices
                ],
                "cross_device": [
                    {
                        "baseline_device": devices[0],
                        "device": device,
                        "case_id": "case-1",
                        "status": "ok",
                        "max_abs_diff": 0.0,
                        "mean_abs_diff": 0.0,
                    }
                    for device in devices[1:]
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
                "max_abs_tolerance": 1e-4,
                "mean_abs_tolerance": 1e-5,
                "worst_max_abs_diff": 0.0,
                "worst_mean_abs_diff": 0.0,
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
        self.branch_heads = {}
        self.trees = {}
        self.parents = {}

    def create_branch(self, **kwargs):
        self.branches.append(kwargs)
        key = (kwargs["repo_id"], kwargs["branch"])
        self.branch_heads.setdefault(key, kwargs["revision"])

    def create_commit(self, **kwargs):
        if kwargs["repo_id"] == self.fail_once_for:
            self.fail_once_for = None
            raise RuntimeError("deliberate candidate upload failure")
        branch_key = (kwargs["repo_id"], kwargs["revision"])
        if self.branch_heads.get(branch_key) != kwargs["parent_commit"]:
            raise RuntimeError("stale parent commit")
        parent_tree = self.trees.get(
            (kwargs["repo_id"], kwargs["parent_commit"]), {}
        )
        tree = dict(parent_tree)
        for operation in kwargs["operations"]:
            source = operation.path_or_fileobj
            if isinstance(source, bytes):
                value = source
            else:
                value = Path(source).read_bytes()
            tree[operation.path_in_repo] = value
        self.commits.append(kwargs)
        digit = format(len(self.commits) + 9, "x")
        revision = digit * 40
        self.trees[(kwargs["repo_id"], revision)] = tree
        self.parents[(kwargs["repo_id"], revision)] = kwargs["parent_commit"]
        self.branch_heads[branch_key] = revision
        return SimpleNamespace(oid=revision)

    def repo_info(self, *, repo_id, revision):
        observed = self.branch_heads.get((repo_id, revision), revision)
        if revision == observed:
            self.verified.append((repo_id, revision))
        return SimpleNamespace(sha=observed)

    def list_repo_tree(self, *, repo_id, revision, recursive, expand):
        observed = self.branch_heads.get((repo_id, revision), revision)
        tree = self.trees.get((repo_id, observed), {})
        return [
            SimpleNamespace(
                path=path,
                size=len(value),
                blob_id=hashlib.sha1(
                    f"blob {len(value)}\0".encode("ascii") + value
                ).hexdigest(),
                lfs=None,
            )
            for path, value in sorted(tree.items())
        ]

    def list_files_info(self, *, repo_id, paths, revision, expand):
        assert paths is None
        return type(self).list_repo_tree(
            self,
            repo_id=repo_id,
            revision=revision,
            recursive=True,
            expand=expand,
        )

    def list_repo_commits(self, *, repo_id, revision):
        observed = self.branch_heads.get((repo_id, revision), revision)
        commits = [SimpleNamespace(commit_id=observed)]
        parent = self.parents.get((repo_id, observed))
        if parent is not None:
            commits.append(SimpleNamespace(commit_id=parent))
        return commits


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
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_abs_diff_vs_reference", None),
        ("mean_abs_diff_vs_reference", float("nan")),
        ("reference_max_abs_tolerance", float("inf")),
        ("reference_mean_abs_tolerance", -1.0),
    ],
)
def test_plan_rejects_missing_or_unsafe_case_numerical_evidence(
    tmp_path, field, value
):
    summary = _stage_summary(tmp_path, "2.11", model_ids=("model-a",))
    summary_value = json.loads(summary.read_text())
    metadata = Path(summary_value["results"][0]["meta"])
    metadata_value = json.loads(metadata.read_text())
    case = metadata_value["validation"]["devices"][0]["cases"][0]
    if value is None:
        case.pop(field)
    else:
        case[field] = value
    _write_json(metadata, metadata_value)
    summary_value["results"][0]["meta_sha256"] = _sha256(metadata)
    _write_json(summary, summary_value)

    with pytest.raises(PublicationError, match="finite nonnegative number"):
        _prepare(tmp_path, [summary], model_ids=("model-a",))


@pytest.mark.release_blocker
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_abs_tolerance", None),
        ("mean_abs_tolerance", float("nan")),
        ("cross_device_max_abs_tolerance", float("inf")),
        ("cross_device_mean_abs_tolerance", -1.0),
    ],
)
def test_plan_rejects_missing_or_unsafe_validation_tolerances(
    tmp_path, field, value
):
    summary = _stage_summary(tmp_path, "2.11", model_ids=("model-a",))
    summary_value = json.loads(summary.read_text())
    metadata = Path(summary_value["results"][0]["meta"])
    metadata_value = json.loads(metadata.read_text())
    if value is None:
        metadata_value["validation"].pop(field)
    else:
        metadata_value["validation"][field] = value
    _write_json(metadata, metadata_value)
    summary_value["results"][0]["meta_sha256"] = _sha256(metadata)
    _write_json(summary, summary_value)

    with pytest.raises(PublicationError, match="finite nonnegative number"):
        _prepare(tmp_path, [summary], model_ids=("model-a",))


@pytest.mark.release_blocker
def test_plan_checks_each_case_against_its_declared_tolerance(tmp_path):
    summary = _stage_summary(tmp_path, "2.11", model_ids=("model-a",))
    summary_value = json.loads(summary.read_text())
    metadata = Path(summary_value["results"][0]["meta"])
    metadata_value = json.loads(metadata.read_text())
    metadata_value["validation"]["devices"][0]["cases"][0][
        "max_abs_diff_vs_reference"
    ] = 2e-4
    _write_json(metadata, metadata_value)
    summary_value["results"][0]["meta_sha256"] = _sha256(metadata)
    _write_json(summary, summary_value)

    with pytest.raises(PublicationError, match="drift exceeds tolerance"):
        _prepare(tmp_path, [summary], model_ids=("model-a",))


@pytest.mark.release_blocker
def test_plan_rejects_incomplete_cross_device_numerical_evidence(tmp_path):
    summary = _stage_summary(
        tmp_path, "2.11", model_ids=("model-a",), devices=("cpu", "cuda")
    )
    summary_value = json.loads(summary.read_text())
    metadata = Path(summary_value["results"][0]["meta"])
    metadata_value = json.loads(metadata.read_text())
    metadata_value["validation"]["cross_device"][0].pop("mean_abs_diff")
    _write_json(metadata, metadata_value)
    summary_value["results"][0]["meta_sha256"] = _sha256(metadata)
    _write_json(summary, summary_value)

    with pytest.raises(PublicationError, match="finite nonnegative number"):
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
    assert receipt["manifest"]["commit_revision"] == "c" * 40


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
    resumed_api.branch_heads.update(first_api.branch_heads)
    resumed_api.trees.update(first_api.trees)
    resumed_api.parents.update(first_api.parents)
    complete = publish_publication_plan(
        plan_path=plan,
        approval_path=approval,
        receipt_path=receipt_path,
        api=resumed_api,
    )
    assert complete["status"] == "complete"
    assert resumed_api.verified == [("owner/model-a", "a" * 40)]
    assert [call["repo_id"] for call in resumed_api.commits] == [
        "owner/model-b",
        "owner/facetorch-model-manifest",
    ]


@pytest.mark.release_blocker
def test_publish_resume_rejects_model_receipt_for_unrelated_commit(tmp_path):
    summary = _stage_summary(tmp_path, "2.11", model_ids=("model-a",))
    plan = _prepare(tmp_path, [summary], model_ids=("model-a",))
    approval = tmp_path / "approval.json"
    receipt_path = tmp_path / "receipt.json"
    _approve(plan, approval)
    api = _FakeHubApi()

    publish_publication_plan(
        plan_path=plan,
        approval_path=approval,
        receipt_path=receipt_path,
        api=api,
    )
    plan_value = json.loads(plan.read_text())
    receipt = json.loads(receipt_path.read_text())
    receipt["models"]["model-a"]["commit_revision"] = plan_value["models"][0][
        "parent_revision"
    ]
    _write_json(receipt_path, receipt)

    with pytest.raises(PublicationError, match="no verifiable direct parent"):
        publish_publication_plan(
            plan_path=plan,
            approval_path=approval,
            receipt_path=receipt_path,
            api=api,
        )


@pytest.mark.release_blocker
def test_publish_resume_rejects_manifest_receipt_for_unrelated_commit(tmp_path):
    summary = _stage_summary(tmp_path, "2.11", model_ids=("model-a",))
    plan = _prepare(tmp_path, [summary], model_ids=("model-a",))
    approval = tmp_path / "approval.json"
    receipt_path = tmp_path / "receipt.json"
    _approve(plan, approval)
    api = _FakeHubApi()

    publish_publication_plan(
        plan_path=plan,
        approval_path=approval,
        receipt_path=receipt_path,
        api=api,
    )
    plan_value = json.loads(plan.read_text())
    receipt = json.loads(receipt_path.read_text())
    receipt["manifest"]["commit_revision"] = plan_value["manifest_target"][
        "parent_revision"
    ]
    _write_json(receipt_path, receipt)

    with pytest.raises(PublicationError, match="no verifiable direct parent"):
        publish_publication_plan(
            plan_path=plan,
            approval_path=approval,
            receipt_path=receipt_path,
            api=api,
        )


@pytest.mark.release_blocker
def test_publish_resume_rejects_exact_tree_from_unrelated_parent(tmp_path):
    summary = _stage_summary(tmp_path, "2.11", model_ids=("model-a",))
    plan = _prepare(tmp_path, [summary], model_ids=("model-a",))
    approval = tmp_path / "approval.json"
    receipt_path = tmp_path / "receipt.json"
    _approve(plan, approval)
    api = _FakeHubApi()

    publish_publication_plan(
        plan_path=plan,
        approval_path=approval,
        receipt_path=receipt_path,
        api=api,
    )
    receipt = json.loads(receipt_path.read_text())
    model_receipt = receipt["models"]["model-a"]
    recorded = model_receipt["commit_revision"]
    unrelated = "e" * 40
    api.trees[("owner/model-a", unrelated)] = dict(
        api.trees[("owner/model-a", recorded)]
    )
    api.parents[("owner/model-a", unrelated)] = "9" * 40
    model_receipt["commit_revision"] = unrelated
    _write_json(receipt_path, receipt)

    with pytest.raises(PublicationError, match="not a direct child"):
        publish_publication_plan(
            plan_path=plan,
            approval_path=approval,
            receipt_path=receipt_path,
            api=api,
        )


@pytest.mark.release_blocker
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repo_id", "owner/unapproved-manifest"),
        ("filename", "manifests/unapproved.json"),
        ("sha256", "0" * 64),
    ],
)
def test_publish_resume_rejects_tampered_manifest_receipt_contract(
    tmp_path, field, value
):
    summary = _stage_summary(tmp_path, "2.11", model_ids=("model-a",))
    plan = _prepare(tmp_path, [summary], model_ids=("model-a",))
    approval = tmp_path / "approval.json"
    receipt_path = tmp_path / "receipt.json"
    _approve(plan, approval)
    api = _FakeHubApi()

    publish_publication_plan(
        plan_path=plan,
        approval_path=approval,
        receipt_path=receipt_path,
        api=api,
    )
    receipt = json.loads(receipt_path.read_text())
    receipt["manifest"][field] = value
    _write_json(receipt_path, receipt)

    with pytest.raises(PublicationError, match="Manifest receipt bytes"):
        publish_publication_plan(
            plan_path=plan,
            approval_path=approval,
            receipt_path=receipt_path,
            api=api,
        )


@pytest.mark.release_blocker
def test_publish_recovers_exact_model_commit_after_receipt_write_crash(
    tmp_path, monkeypatch
):
    summary = _stage_summary(tmp_path, "2.11")
    plan = _prepare(tmp_path, [summary])
    approval = tmp_path / "approval.json"
    receipt_path = tmp_path / "receipt.json"
    _approve(plan, approval)
    api = _FakeHubApi()
    write_json_atomic = publication._write_json_atomic
    crashed = False

    def crash_after_first_model(path, value):
        nonlocal crashed
        if (
            not crashed
            and path == receipt_path
            and set(value.get("models", {})) == {"model-a"}
        ):
            crashed = True
            raise OSError("deliberate receipt write crash")
        write_json_atomic(path, value)

    monkeypatch.setattr(publication, "_write_json_atomic", crash_after_first_model)
    with pytest.raises(OSError, match="receipt write crash"):
        publish_publication_plan(
            plan_path=plan,
            approval_path=approval,
            receipt_path=receipt_path,
            api=api,
        )
    assert [call["repo_id"] for call in api.commits] == ["owner/model-a"]

    monkeypatch.setattr(publication, "_write_json_atomic", write_json_atomic)
    monkeypatch.setattr(api, "list_repo_tree", None)
    complete = publish_publication_plan(
        plan_path=plan,
        approval_path=approval,
        receipt_path=receipt_path,
        api=api,
    )

    assert complete["status"] == "complete"
    assert [call["repo_id"] for call in api.commits] == [
        "owner/model-a",
        "owner/model-b",
        "owner/facetorch-model-manifest",
    ]


@pytest.mark.release_blocker
def test_publish_recovery_rejects_exact_tree_from_unrelated_parent(
    tmp_path, monkeypatch
):
    summary = _stage_summary(tmp_path, "2.11", model_ids=("model-a",))
    plan = _prepare(tmp_path, [summary], model_ids=("model-a",))
    approval = tmp_path / "approval.json"
    receipt_path = tmp_path / "receipt.json"
    _approve(plan, approval)
    api = _FakeHubApi()
    write_json_atomic = publication._write_json_atomic

    def crash_after_model(path, value):
        if path == receipt_path and value.get("models"):
            raise OSError("deliberate receipt write crash")
        write_json_atomic(path, value)

    monkeypatch.setattr(publication, "_write_json_atomic", crash_after_model)
    with pytest.raises(OSError, match="receipt write crash"):
        publish_publication_plan(
            plan_path=plan,
            approval_path=approval,
            receipt_path=receipt_path,
            api=api,
        )
    committed = api.commits[0]
    head = api.branch_heads[(committed["repo_id"], committed["revision"])]
    api.parents[(committed["repo_id"], head)] = "9" * 40

    monkeypatch.setattr(publication, "_write_json_atomic", write_json_atomic)
    with pytest.raises(PublicationError, match="not a direct child"):
        publish_publication_plan(
            plan_path=plan,
            approval_path=approval,
            receipt_path=receipt_path,
            api=api,
        )


@pytest.mark.release_blocker
def test_publish_recovers_exact_manifest_commit_after_receipt_write_crash(
    tmp_path, monkeypatch
):
    summary = _stage_summary(tmp_path, "2.11", model_ids=("model-a",))
    plan = _prepare(tmp_path, [summary], model_ids=("model-a",))
    approval = tmp_path / "approval.json"
    receipt_path = tmp_path / "receipt.json"
    _approve(plan, approval)
    api = _FakeHubApi()
    write_json_atomic = publication._write_json_atomic
    crashed = False

    def crash_after_manifest(path, value):
        nonlocal crashed
        if (
            not crashed
            and path == receipt_path
            and isinstance(value.get("manifest"), dict)
        ):
            crashed = True
            raise OSError("deliberate manifest receipt write crash")
        write_json_atomic(path, value)

    monkeypatch.setattr(publication, "_write_json_atomic", crash_after_manifest)
    with pytest.raises(OSError, match="manifest receipt write crash"):
        publish_publication_plan(
            plan_path=plan,
            approval_path=approval,
            receipt_path=receipt_path,
            api=api,
        )
    assert len(api.commits) == 2

    monkeypatch.setattr(publication, "_write_json_atomic", write_json_atomic)
    complete = publish_publication_plan(
        plan_path=plan,
        approval_path=approval,
        receipt_path=receipt_path,
        api=api,
    )

    assert complete["status"] == "complete"
    assert len(api.commits) == 2


@pytest.mark.release_blocker
def test_publish_recovery_rejects_a_divergent_candidate_branch(
    tmp_path, monkeypatch
):
    summary = _stage_summary(tmp_path, "2.11", model_ids=("model-a",))
    plan = _prepare(tmp_path, [summary], model_ids=("model-a",))
    approval = tmp_path / "approval.json"
    receipt_path = tmp_path / "receipt.json"
    _approve(plan, approval)
    api = _FakeHubApi()
    write_json_atomic = publication._write_json_atomic

    def crash_after_model(path, value):
        if path == receipt_path and value.get("models"):
            raise OSError("deliberate receipt write crash")
        write_json_atomic(path, value)

    monkeypatch.setattr(publication, "_write_json_atomic", crash_after_model)
    with pytest.raises(OSError, match="receipt write crash"):
        publish_publication_plan(
            plan_path=plan,
            approval_path=approval,
            receipt_path=receipt_path,
            api=api,
        )
    committed = api.commits[0]
    head = api.branch_heads[(committed["repo_id"], committed["revision"])]
    api.trees[(committed["repo_id"], head)]["unexpected.txt"] = b"diverged"

    monkeypatch.setattr(publication, "_write_json_atomic", write_json_atomic)
    with pytest.raises(PublicationError, match="diverged from the approved plan"):
        publish_publication_plan(
            plan_path=plan,
            approval_path=approval,
            receipt_path=receipt_path,
            api=api,
        )


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
            "guaranteed_mean_abs_limit": 0.00002,
            "left_cohort": "2.6",
            "model_id": "model-a",
            "num_cases": 1,
            "right_cohort": "2.11",
            "worst_guaranteed_max_abs": 0.0,
            "worst_guaranteed_mean_abs": 0.0,
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
    assert comparisons["cpu"]["guaranteed_mean_abs_limit"] == 2e-5
    assert comparisons["cuda"]["guaranteed_max_abs_limit"] == 4e-4
    assert comparisons["cuda"]["guaranteed_mean_abs_limit"] == 4e-5


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
