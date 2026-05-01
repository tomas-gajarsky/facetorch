import torch

from scripts.export_model_cohorts_hf import _validate_exported_module


class _Identity(torch.nn.Module):
    def forward(self, x):
        return x


class _Offset(torch.nn.Module):
    def forward(self, x):
        return x + 1e-2


def _reference(x):
    return x


def _spec():
    return {
        "id": "test-model",
        "input_shape": [1, 3, 8, 8],
        "max_abs_tolerance": 1e-4,
        "mean_abs_tolerance": 1e-5,
    }


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
