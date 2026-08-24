import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import pytest
import torch

import scripts.export_model_cohorts_hf as exporter
from scripts.export_model_cohorts_hf import (
    _load_validation_reference,
    _load_state_dict_strictly,
    _recover_torchscript_state_attributes,
    _run_for_specs,
    _validate_exported_module,
)


class _Identity(torch.nn.Module):
    def forward(self, x):
        return x


class _Offset(torch.nn.Module):
    def forward(self, x):
        return x + 1e-2


class _NonFinite(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        return torch.full_like(x, self.value)


class _NestedNonFinite(torch.nn.Module):
    def forward(self, x):
        return {"head": [x, torch.full_like(x, float("nan"))]}


class _BatchCoupled(torch.nn.Module):
    def forward(self, x):
        return x - x.mean(dim=0, keepdim=True)


def _reference(x):
    return x


def _unexpected_reference(_x):
    raise AssertionError(
        "reused golden references must not execute the runtime reference"
    )


def _spec():
    return {
        "id": "test-model",
        "input_shape": [1, 3, 8, 8],
        "max_abs_tolerance": 1e-4,
        "mean_abs_tolerance": 1e-5,
    }


def test_verified_source_copy_is_readable_by_non_owner_runtime(tmp_path):
    source = tmp_path / "source.pt2"
    target = tmp_path / "cohort" / "model.pt2"
    source.write_bytes(b"verified-export")
    source.chmod(0o600)

    exporter._copy_verified_source(
        source,
        target,
        expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        expected_size=source.stat().st_size,
    )

    assert target.read_bytes() == source.read_bytes()
    assert target.stat().st_mode & 0o777 == 0o644


def test_export_validation_passes_within_tolerance():
    validation = _validate_exported_module(
        _spec(),
        ref_fn=_reference,
        exported_module=_Identity(),
        batch_sizes=[1],
        seeds=[0],
        scales=[1.0],
        devices=["cpu"],
    )

    assert validation["status"] == "ok"
    assert validation["failures"] == []
    assert validation["numeric_policy"] == {
        "dtype": "float32",
        "cudnn_allow_tf32": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cuda_matmul_allow_tf32": False,
        "float32_matmul_precision": "highest",
        "restores_caller_settings": True,
    }


def test_validation_numeric_policy_restores_caller_settings():
    original = {
        "allow_tf32": torch.backends.cudnn.allow_tf32,
        "benchmark": torch.backends.cudnn.benchmark,
        "deterministic": torch.backends.cudnn.deterministic,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "precision": torch.get_float32_matmul_precision(),
    }
    try:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        _validate_exported_module(
            _spec(),
            ref_fn=_reference,
            exported_module=_Identity(),
            batch_sizes=[1],
            seeds=[0],
            scales=[1.0],
            devices=["cpu"],
        )

        assert torch.backends.cudnn.allow_tf32 is True
        assert torch.backends.cudnn.benchmark is True
        assert torch.backends.cudnn.deterministic is False
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.get_float32_matmul_precision() == "high"
    finally:
        torch.backends.cudnn.allow_tf32 = original["allow_tf32"]
        torch.backends.cudnn.benchmark = original["benchmark"]
        torch.backends.cudnn.deterministic = original["deterministic"]
        torch.backends.cuda.matmul.allow_tf32 = original["matmul_tf32"]
        torch.set_float32_matmul_precision(original["precision"])


@pytest.mark.release_blocker
def test_persistent_golden_reference_is_shared_across_runtime_cohorts(tmp_path):
    spec = {
        **_spec(),
        "validation_reference": {
            "kind": "fixture",
            "source": "immutable-reference.pt",
            "sha256": "a" * 64,
            "device": "cpu",
            "batch_mode": "native",
        },
    }
    golden = tmp_path / "golden-reference.pt"
    recorded = _validate_exported_module(
        spec,
        ref_fn=_reference,
        exported_module=_Identity(),
        batch_sizes=[1],
        seeds=[0],
        scales=[1.0],
        devices=["cpu"],
        golden_reference_path=golden,
        golden_reference_mode="record",
        cohort="2.6",
        golden_reference_cohort="2.6",
    )
    recorded_bytes = golden.read_bytes()

    # Recording is deterministic and retry-safe when the bytes are unchanged.
    repeated = _validate_exported_module(
        spec,
        ref_fn=_reference,
        exported_module=_Identity(),
        batch_sizes=[1],
        seeds=[0],
        scales=[1.0],
        devices=["cpu"],
        golden_reference_path=golden,
        golden_reference_mode="record",
        cohort="2.6",
        golden_reference_cohort="2.6",
    )
    reused = _validate_exported_module(
        spec,
        ref_fn=_unexpected_reference,
        exported_module=_Identity(),
        batch_sizes=[1],
        seeds=[0],
        scales=[1.0],
        devices=["cpu"],
        golden_reference_path=golden,
        golden_reference_mode="reuse",
        cohort="2.11",
        golden_reference_cohort="2.6",
    )

    assert golden.read_bytes() == recorded_bytes
    assert recorded["status"] == repeated["status"] == reused["status"] == "ok"
    assert recorded["golden_reference"]["status"] == "recorded"
    assert reused["golden_reference"]["status"] == "reused"
    assert (
        recorded["golden_reference"]["sha256"] == reused["golden_reference"]["sha256"]
    )
    assert (
        recorded["devices"][0]["cases"][0]["reference_output_sha256"]
        == reused["devices"][0]["cases"][0]["reference_output_sha256"]
    )


@pytest.mark.release_blocker
def test_reused_golden_reference_rejects_changed_matrix(tmp_path):
    spec = {
        **_spec(),
        "validation_reference": {
            "kind": "fixture",
            "source": "immutable-reference.pt",
            "sha256": "a" * 64,
            "device": "cpu",
            "batch_mode": "native",
        },
    }
    golden = tmp_path / "golden-reference.pt"
    _validate_exported_module(
        spec,
        ref_fn=_reference,
        exported_module=_Identity(),
        batch_sizes=[1],
        seeds=[0],
        scales=[1.0],
        devices=["cpu"],
        golden_reference_path=golden,
        golden_reference_mode="record",
        cohort="2.6",
        golden_reference_cohort="2.6",
    )

    with pytest.raises(RuntimeError, match="mismatched matrix"):
        _validate_exported_module(
            spec,
            ref_fn=_unexpected_reference,
            exported_module=_Identity(),
            batch_sizes=[1],
            seeds=[0],
            scales=[0.25],
            devices=["cpu"],
            golden_reference_path=golden,
            golden_reference_mode="reuse",
            cohort="2.11",
            golden_reference_cohort="2.6",
        )


@pytest.mark.release_blocker
def test_reused_golden_reference_rejects_corrupt_bundle(tmp_path):
    spec = {
        **_spec(),
        "validation_reference": {
            "kind": "fixture",
            "source": "immutable-reference.pt",
            "sha256": "a" * 64,
            "device": "cpu",
            "batch_mode": "native",
        },
    }
    golden = tmp_path / "golden-reference.pt"
    _validate_exported_module(
        spec,
        ref_fn=_reference,
        exported_module=_Identity(),
        batch_sizes=[1],
        seeds=[0],
        scales=[1.0],
        devices=["cpu"],
        golden_reference_path=golden,
        golden_reference_mode="record",
        cohort="2.6",
        golden_reference_cohort="2.6",
    )
    golden.write_bytes(b"not-a-safe-tensor-bundle")

    with pytest.raises(RuntimeError, match="Cannot load golden reference bundle"):
        _validate_exported_module(
            spec,
            ref_fn=_unexpected_reference,
            exported_module=_Identity(),
            batch_sizes=[1],
            seeds=[0],
            scales=[1.0],
            devices=["cpu"],
            golden_reference_path=golden,
            golden_reference_mode="reuse",
            cohort="2.11",
            golden_reference_cohort="2.6",
        )


def test_per_sample_reference_mode_defines_batch_as_independent_faces(tmp_path):
    path = tmp_path / "reference.pt"
    traced = torch.jit.trace(_BatchCoupled().eval(), torch.ones(1, 2))
    torch.jit.save(traced, str(path))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    ref_fn, metadata = _load_validation_reference(
        {
            "id": "batch-reference",
            "validation_reference": {
                "kind": "torchscript",
                "source": str(path),
                "sha256": digest,
                "batch_mode": "per_sample",
            },
        }
    )

    output = ref_fn(torch.tensor([[1.0, 2.0], [4.0, 8.0]]))

    assert torch.equal(output, torch.zeros_like(output))
    assert metadata["batch_mode"] == "per_sample"


def test_export_validation_fails_outside_tolerance():
    validation = _validate_exported_module(
        _spec(),
        ref_fn=_reference,
        exported_module=_Offset(),
        batch_sizes=[1],
        seeds=[0],
        scales=[1.0],
        devices=["cpu"],
    )

    assert validation["status"] == "failed"
    assert validation["failures"]
    assert validation["worst_max_abs_diff_vs_reference"] > 1e-4


@pytest.mark.release_blocker
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_export_validation_rejects_nonfinite_outputs_even_when_both_match(value):
    nonfinite = _NonFinite(value)
    validation = _validate_exported_module(
        _spec(),
        ref_fn=nonfinite,
        exported_module=nonfinite,
        batch_sizes=[1],
        seeds=[0],
        scales=[1.0],
        devices=["cpu"],
    )

    assert validation["status"] == "failed"
    assert validation["failures"]


@pytest.mark.release_blocker
def test_export_validation_rejects_recursively_nested_nonfinite_outputs():
    validation = _validate_exported_module(
        _spec(),
        ref_fn=_NestedNonFinite(),
        exported_module=_NestedNonFinite(),
        batch_sizes=[1],
        seeds=[0],
        scales=[1.0],
        devices=["cpu"],
    )

    assert validation["status"] == "failed"
    assert "NaN or Inf" in validation["failures"][0]["error"]


@pytest.mark.release_blocker
def test_requested_cuda_cannot_be_hidden_by_cpu_success(monkeypatch):
    monkeypatch.setattr(exporter.torch.cuda, "is_available", lambda: False)
    validation = _validate_exported_module(
        _spec(),
        ref_fn=_reference,
        exported_module=_Identity(),
        batch_sizes=[1],
        seeds=[0],
        scales=[1.0],
        devices=["cpu", "cuda"],
    )

    assert validation["status"] != "ok"
    assert {item["device"]: item["status"] for item in validation["devices"]} == {
        "cpu": "ok",
        "cuda": "skipped",
    }


@pytest.mark.release_blocker
def test_zero_validation_cases_is_a_failure():
    validation = _validate_exported_module(
        _spec(),
        ref_fn=_reference,
        exported_module=_Identity(),
        batch_sizes=[],
        seeds=[0],
        scales=[1.0],
        devices=["cpu"],
    )

    assert validation["status"] == "failed"


@pytest.mark.release_blocker
def test_missing_required_dynamic_batch_is_an_incomplete_matrix():
    spec = {**_spec(), "required_batch_sizes": [1, 2, 4, 8]}
    validation = _validate_exported_module(
        spec,
        ref_fn=_reference,
        exported_module=_Identity(),
        batch_sizes=[1, 2, 4],
        seeds=[0],
        scales=[1.0],
        devices=["cpu"],
    )

    assert validation["status"] == "failed"
    assert validation["failures"][0]["missing_batch_sizes"] == [8]


@pytest.mark.release_blocker
def test_state_reconstruction_rejects_missing_and_unexpected_keys():
    model = torch.nn.Linear(2, 2)
    spec = {"id": "linear", "strict": True}

    with pytest.raises(RuntimeError, match="missing"):
        _load_state_dict_strictly(model, {"weight": model.weight.detach()}, spec)

    state = dict(model.state_dict())
    state["unreviewed.buffer"] = torch.ones(1)
    with pytest.raises(RuntimeError, match="unexpected"):
        _load_state_dict_strictly(model, state, spec)


@pytest.mark.release_blocker
def test_state_reconstruction_allows_only_declared_generated_buffers():
    model = torch.nn.Linear(2, 2)
    state = dict(model.state_dict())
    state["generated.buffer"] = torch.ones(1)
    result = _load_state_dict_strictly(
        model,
        state,
        {
            "id": "linear",
            "strict": True,
            "state_key_allowlist": {
                "missing": [],
                "unexpected": [r"generated\.buffer"],
            },
        },
    )

    assert result["allowlisted_unexpected_keys"] == ["generated.buffer"]


def test_old_torchscript_tensor_attributes_complete_strict_state():
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(2))
    expected = dict(model.state_dict())
    scripted_state = {
        key: value.clone()
        for key, value in expected.items()
        if key not in {"0.running_mean", "0.running_var"}
    }
    scripted = SimpleNamespace(
        state_dict=lambda: scripted_state,
        named_modules=lambda: [
            (
                "0",
                SimpleNamespace(
                    running_mean=expected["0.running_mean"].clone(),
                    running_var=expected["0.running_var"].clone(),
                ),
            )
        ],
    )

    recovered_state, recovered_keys = _recover_torchscript_state_attributes(
        model, scripted
    )

    assert recovered_keys == ["0.running_mean", "0.running_var"]
    model.load_state_dict(recovered_state, strict=True)


def _validation_result(status):
    return {
        "status": status,
        "num_cases": 1 if status == "ok" else 0,
        "max_abs_tolerance": 1e-4,
        "mean_abs_tolerance": 1e-5,
        "worst_case_id": "case" if status == "ok" else None,
        "worst_device": "cpu" if status == "ok" else None,
        "worst_max_abs_diff_vs_reference": 0.0,
        "worst_mean_abs_diff_vs_reference": 0.0,
        "failures": [],
        "devices": [],
    }


class _LoadedProgram:
    def module(self):
        return _Identity()


def _patch_export_pipeline(monkeypatch, validation_status="ok"):
    monkeypatch.setattr(
        exporter,
        "_build_reference_and_exported_program",
        lambda _spec, _cohort: (_reference, object(), {"source": "test"}),
    )
    monkeypatch.setattr(
        exporter.torch.export,
        "save",
        lambda _program, path: Path(path).write_bytes(b"export"),
    )
    monkeypatch.setattr(
        exporter.torch.export,
        "load",
        lambda _path: _LoadedProgram(),
    )
    monkeypatch.setattr(
        exporter,
        "_validate_exported_module",
        lambda *_args, **_kwargs: _validation_result(validation_status),
    )


@pytest.mark.release_blocker
def test_new_export_directories_are_traversable_with_restrictive_umask(
    tmp_path, monkeypatch
):
    _patch_export_pipeline(monkeypatch, validation_status="ok")
    intermediate = tmp_path / "nested"
    out_root = intermediate / "out"
    previous_umask = os.umask(0o077)
    try:
        exporter._ensure_runtime_directory(out_root)
        _summary, failures = _run_for_specs(
            specs=[{"id": "model-a", "repo_id": "owner/model-a"}],
            mode="export",
            repo_root=Path.cwd(),
            cohort="2.6",
            out_root=out_root,
            artifacts_root=tmp_path / "artifacts",
            upload=False,
            hf_token_env="B01_UNUSED_TOKEN",
            batch_sizes=[1],
            seeds=[0],
            scales=[1.0],
            validate_devices=["cpu", "cuda"],
        )
    finally:
        os.umask(previous_umask)

    assert failures == []
    assert intermediate.stat().st_mode & 0o777 == 0o755
    assert out_root.stat().st_mode & 0o777 == 0o755
    assert (out_root / "model-a").stat().st_mode & 0o777 == 0o755


@pytest.mark.release_blocker
def test_existing_export_directory_permissions_are_preserved(tmp_path):
    out_root = tmp_path / "operator-managed"
    out_root.mkdir()
    out_root.chmod(0o700)

    exporter._ensure_runtime_directory(out_root)

    assert out_root.stat().st_mode & 0o777 == 0o700


@pytest.mark.release_blocker
def test_non_ok_validation_status_blocks_the_export_gate(tmp_path, monkeypatch):
    _patch_export_pipeline(monkeypatch, validation_status="skipped")
    summary, failures = _run_for_specs(
        specs=[{"id": "model-a", "repo_id": "owner/model-a"}],
        mode="export",
        repo_root=Path.cwd(),
        cohort="2.3",
        out_root=tmp_path / "out",
        artifacts_root=tmp_path / "artifacts",
        upload=False,
        hf_token_env="B01_UNUSED_TOKEN",
        batch_sizes=[1],
        seeds=[0],
        scales=[1.0],
        validate_devices=["cpu", "cuda"],
    )

    assert failures
    assert summary["results"][0]["status"] == "error"


@pytest.mark.release_blocker
def test_late_model_failure_prevents_all_model_uploads(tmp_path, monkeypatch):
    upload_calls = []

    class FakeApi:
        def __init__(self, token):
            assert token == "test-token"

        def upload_file(self, **kwargs):
            upload_calls.append(kwargs)

    _patch_export_pipeline(monkeypatch, validation_status="ok")
    original_builder = exporter._build_reference_and_exported_program

    def fail_second_model(spec, cohort):
        if spec["id"] == "model-b":
            raise RuntimeError("deliberate late validation failure")
        return original_builder(spec, cohort)

    monkeypatch.setattr(
        exporter, "_build_reference_and_exported_program", fail_second_model
    )
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setenv("B01_HF_TOKEN", "test-token")

    _summary, failures = _run_for_specs(
        specs=[
            {"id": "model-a", "repo_id": "owner/model-a"},
            {"id": "model-b", "repo_id": "owner/model-b"},
        ],
        mode="export",
        repo_root=Path.cwd(),
        cohort="2.3",
        out_root=tmp_path / "out",
        artifacts_root=tmp_path / "artifacts",
        upload=True,
        hf_token_env="B01_HF_TOKEN",
        batch_sizes=[1],
        seeds=[0],
        scales=[1.0],
        validate_devices=["cpu"],
    )

    assert failures and failures[0][0] == "model-b"
    assert upload_calls == []


@pytest.mark.release_blocker
def test_direct_upload_is_disabled_even_after_green_staging(tmp_path, monkeypatch):
    _patch_export_pipeline(monkeypatch, validation_status="ok")
    summary, failures = _run_for_specs(
        specs=[{"id": "model-a", "repo_id": "owner/model-a"}],
        mode="export",
        repo_root=Path.cwd(),
        cohort="2.3",
        out_root=tmp_path / "out",
        artifacts_root=tmp_path / "artifacts",
        upload=True,
        hf_token_env="UNUSED_TOKEN",
        batch_sizes=[1],
        seeds=[0],
        scales=[1.0],
        validate_devices=["cpu"],
    )

    assert failures == [
        (
            "publication",
            "RuntimeError",
            "Direct upload is disabled. Review the complete staging report, create "
            "a digest-bound approval, and use model_cohort_publication.py publish.",
        )
    ]
    assert summary["publication_status"] == "approval_required"
