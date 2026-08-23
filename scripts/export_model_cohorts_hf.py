#!/usr/bin/env python3
"""Export, validate, and upload versioned `.pt2` cohorts for facetorch models.

The script supports two workflows:

1) Export + validate into local staging for the *current* torch runtime.
2) Validate previously exported artifacts against reference models.

Publication is deliberately separate. After every requested cohort passes and the
staging report is reviewed, use ``scripts/model_cohort_publication.py`` to build a
digest-bound plan and publish candidate commits. This script never uploads.

Examples:
  # Fetch only pinned, digest-verified source references
  PYTHONPATH=. python scripts/export_model_cohorts_hf.py prepare-sources \
    --repo-root . --cohort 2.11

  # Export current torch cohort to a local staging directory
  PYTHONPATH=. python scripts/export_model_cohorts_hf.py export \
    --repo-root . \
    --out-root /tmp/model-cohort-exports/staging

  # Validate existing artifacts for a specific cohort (no export/upload)
  PYTHONPATH=. python scripts/export_model_cohorts_hf.py validate \
    --repo-root . \
    --artifacts-root /tmp/model-cohort-exports/upload26 \
    --cohort 2.6
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import tempfile
import warnings
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch

warnings.filterwarnings(
    "ignore",
    message=r"Manually populate .*num_batches_tracked.*",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as output:
            output.write(_canonical_json_bytes(value))
            output.flush()
            os.fsync(output.fileno())
        temporary.chmod(0o644)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _ensure_runtime_directory(path: Path) -> None:
    """Make each newly created component readable by non-owner runtimes."""
    missing = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent

    if not cursor.is_dir():
        raise NotADirectoryError(f"Export path parent is not a directory: {cursor}")

    for directory in reversed(missing):
        try:
            directory.mkdir()
        except FileExistsError:
            if not directory.is_dir():
                raise
        else:
            directory.chmod(0o755)


def _module_version(module_name: str) -> str | None:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    value = getattr(module, "__version__", None)
    return None if value is None else str(value)


def _git_source_state(repo_root: Path) -> Dict[str, Any]:
    def run(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            check=False,
        )

    revision = run("rev-parse", "HEAD")
    status = run("status", "--porcelain=v1", "--untracked-files=all")
    commit = revision.stdout.decode("ascii", errors="replace").strip()
    valid_commit = bool(re.fullmatch(r"[0-9a-f]{40}", commit))
    clean = status.returncode == 0 and not status.stdout
    return {
        "commit": commit if revision.returncode == 0 and valid_commit else None,
        "clean": clean,
        "status_sha256": hashlib.sha256(status.stdout).hexdigest(),
    }


def _export_schema_version() -> Dict[str, int] | None:
    try:
        from torch._export.serde.schema import SCHEMA_VERSION

        major, minor = SCHEMA_VERSION
        return {"major": int(major), "minor": int(minor)}
    except Exception:
        return None


def _environment_metadata(
    repo_root: Path, environment_lock: str | Path | None = None
) -> Dict[str, Any]:
    lock_path = Path(environment_lock or "uv.lock")
    if not lock_path.is_absolute():
        lock_path = repo_root / lock_path
    lock_path = lock_path.resolve()
    try:
        relative_lock_path = lock_path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise RuntimeError("Environment lock must be inside the source tree") from exc
    if not lock_path.is_file():
        raise RuntimeError(f"Environment lock does not exist: {relative_lock_path}")
    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "total_memory_bytes": int(properties.total_memory),
                }
            )

    cudnn_version = None
    if getattr(torch.backends, "cudnn", None) is not None:
        cudnn_version = torch.backends.cudnn.version()

    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "torch_version": str(torch.__version__),
        "torchvision_version": _module_version("torchvision"),
        "timm_version": _module_version("timm"),
        "cuda_runtime": getattr(torch.version, "cuda", None),
        "cudnn_version": cudnn_version,
        "cuda_devices": devices,
        "export_schema": _export_schema_version(),
        "source_tree": _git_source_state(repo_root),
        "environment_lock": {
            "path": relative_lock_path.as_posix(),
            "sha256": _sha256(lock_path),
        },
    }


def _import_class(class_path: str):
    module_path, class_name = class_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def _summarize_output(o: Any) -> Any:
    if isinstance(o, torch.Tensor):
        return {
            "shape": list(o.shape),
            "dtype": str(o.dtype),
            "finite": bool(
                not (o.is_floating_point() or o.is_complex())
                or torch.isfinite(o).all().item()
            ),
        }
    if isinstance(o, (tuple, list)):
        return [_summarize_output(x) for x in o]
    if isinstance(o, Mapping):
        return {str(key): _summarize_output(value) for key, value in o.items()}
    return str(type(o))


def _clone_output_cpu(o: Any) -> Any:
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().clone()
    if isinstance(o, tuple):
        return tuple(_clone_output_cpu(x) for x in o)
    if isinstance(o, list):
        return [_clone_output_cpu(x) for x in o]
    if isinstance(o, Mapping):
        return {key: _clone_output_cpu(value) for key, value in o.items()}
    return o


def _concat_outputs(outputs: Sequence[Any]) -> Any:
    if not outputs:
        raise RuntimeError("Cannot concatenate an empty output sequence")
    first = outputs[0]
    if isinstance(first, torch.Tensor):
        if not all(isinstance(output, torch.Tensor) for output in outputs):
            raise RuntimeError("Reference outputs changed type across samples")
        return torch.cat(list(outputs), dim=0)
    if isinstance(first, tuple):
        if not all(isinstance(output, tuple) for output in outputs):
            raise RuntimeError("Reference outputs changed type across samples")
        if not all(len(output) == len(first) for output in outputs):
            raise RuntimeError("Reference tuple outputs changed length across samples")
        return tuple(
            _concat_outputs([output[index] for output in outputs])
            for index in range(len(first))
        )
    if isinstance(first, list):
        if not all(isinstance(output, list) for output in outputs):
            raise RuntimeError("Reference outputs changed type across samples")
        if not all(len(output) == len(first) for output in outputs):
            raise RuntimeError("Reference list outputs changed length across samples")
        return [
            _concat_outputs([output[index] for output in outputs])
            for index in range(len(first))
        ]
    raise RuntimeError(
        "Unsupported reference output type for per-sample batching: "
        f"{type(first).__name__}"
    )


def _tensor_paths(o: Any, path: str = "output"):
    if isinstance(o, torch.Tensor):
        yield path, o
        return
    if isinstance(o, (tuple, list)):
        for index, value in enumerate(o):
            yield from _tensor_paths(value, f"{path}[{index}]")
        return
    if isinstance(o, Mapping):
        for key in sorted(o, key=str):
            yield from _tensor_paths(o[key], f"{path}[{key!r}]")
        return
    raise RuntimeError(f"Unsupported output type at {path}: {type(o).__name__}")


def _ensure_finite_output(o: Any, label: str) -> None:
    found_tensor = False
    for path, tensor in _tensor_paths(o):
        found_tensor = True
        if tensor.numel() == 0:
            raise RuntimeError(f"{label} contains an empty tensor at {path}")
        if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(
            tensor
        ).all():
            raise RuntimeError(f"{label} contains NaN or Inf at {path}")
    if not found_tensor:
        raise RuntimeError(f"{label} contains no tensors")


def _output_sha256(o: Any) -> str:
    digest = hashlib.sha256()

    def update(value: Any, path: str) -> None:
        digest.update(path.encode("utf-8"))
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
            return
        if isinstance(value, tuple):
            digest.update(b"tuple")
            for index, item in enumerate(value):
                update(item, f"{path}[{index}]")
            return
        if isinstance(value, list):
            digest.update(b"list")
            for index, item in enumerate(value):
                update(item, f"{path}[{index}]")
            return
        if isinstance(value, Mapping):
            digest.update(b"mapping")
            for key in sorted(value, key=str):
                update(value[key], f"{path}[{key!r}]")
            return
        raise RuntimeError(
            f"Unsupported output type for fingerprint at {path}: "
            f"{type(value).__name__}"
        )

    update(o, "output")
    return digest.hexdigest()


@contextmanager
def _validation_numeric_policy():
    """Use a reproducible float32 inference policy and restore global settings."""
    cudnn = torch.backends.cudnn
    cuda_matmul = torch.backends.cuda.matmul
    original = {
        "cudnn_allow_tf32": bool(cudnn.allow_tf32),
        "cudnn_benchmark": bool(cudnn.benchmark),
        "cudnn_deterministic": bool(cudnn.deterministic),
        "cuda_matmul_allow_tf32": bool(cuda_matmul.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    applied = {
        "dtype": "float32",
        "cudnn_allow_tf32": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cuda_matmul_allow_tf32": False,
        "float32_matmul_precision": "highest",
        "restores_caller_settings": True,
    }
    try:
        cudnn.allow_tf32 = False
        cudnn.benchmark = False
        cudnn.deterministic = True
        cuda_matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        yield applied
    finally:
        cudnn.allow_tf32 = original["cudnn_allow_tf32"]
        cudnn.benchmark = original["cudnn_benchmark"]
        cudnn.deterministic = original["cudnn_deterministic"]
        cuda_matmul.allow_tf32 = original["cuda_matmul_allow_tf32"]
        torch.set_float32_matmul_precision(original["float32_matmul_precision"])


def _ensure_same_structure(a: Any, b: Any, path: str = "output") -> None:
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if tuple(a.shape) != tuple(b.shape):
            raise RuntimeError(
                f"Shape mismatch at {path}: {tuple(a.shape)} != {tuple(b.shape)}"
            )
        if a.dtype != b.dtype:
            raise RuntimeError(f"Dtype mismatch at {path}: {a.dtype} != {b.dtype}")
        return
    if isinstance(a, tuple) and isinstance(b, tuple):
        if len(a) != len(b):
            raise RuntimeError(f"Tuple length mismatch at {path}: {len(a)} != {len(b)}")
        for i, (ai, bi) in enumerate(zip(a, b)):
            _ensure_same_structure(ai, bi, f"{path}[{i}]")
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            raise RuntimeError(f"List length mismatch at {path}: {len(a)} != {len(b)}")
        for i, (ai, bi) in enumerate(zip(a, b)):
            _ensure_same_structure(ai, bi, f"{path}[{i}]")
        return
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        if set(a) != set(b):
            raise RuntimeError(
                f"Mapping keys mismatch at {path}: "
                f"{sorted(a, key=str)} != {sorted(b, key=str)}"
            )
        for key in sorted(a, key=str):
            _ensure_same_structure(a[key], b[key], f"{path}[{key!r}]")
        return
    raise RuntimeError(f"Output type mismatch at {path}: {type(a)} vs {type(b)}")


def _accumulate_abs_diff(a: Any, b: Any, acc: Dict[str, float]) -> None:
    if isinstance(a, torch.Tensor):
        comparison_dtype = torch.complex128 if a.is_complex() else torch.float64
        da = a.detach().cpu().to(comparison_dtype)
        db = b.detach().cpu().to(comparison_dtype)
        d = (da - db).abs()
        if d.numel() == 0:
            raise RuntimeError("Cannot compare empty output tensors")
        if not torch.isfinite(d).all():
            raise RuntimeError("Comparison produced NaN or Inf")
        acc["max_abs"] = max(acc["max_abs"], float(d.max().item()))
        acc["sum_abs"] += float(d.sum().item())
        acc["numel"] += int(d.numel())
        return

    if isinstance(a, tuple):
        for ai, bi in zip(a, b):
            _accumulate_abs_diff(ai, bi, acc)
        return

    if isinstance(a, list):
        for ai, bi in zip(a, b):
            _accumulate_abs_diff(ai, bi, acc)
        return

    if isinstance(a, Mapping):
        for key in sorted(a, key=str):
            _accumulate_abs_diff(a[key], b[key], acc)
        return

    raise RuntimeError(f"Unsupported output type for diff: {type(a)}")


def _compute_diff_stats(a: Any, b: Any) -> Dict[str, float]:
    _ensure_finite_output(a, "Reference output")
    _ensure_finite_output(b, "Exported output")
    _ensure_same_structure(a, b)
    acc = {"max_abs": 0.0, "sum_abs": 0.0, "numel": 0}
    _accumulate_abs_diff(a, b, acc)
    if not acc["numel"]:
        raise RuntimeError("Validation compared zero output elements")
    mean_abs = acc["sum_abs"] / acc["numel"]
    if not math.isfinite(acc["max_abs"]) or not math.isfinite(mean_abs):
        raise RuntimeError("Comparison statistics contain NaN or Inf")
    return {
        "max_abs": acc["max_abs"],
        "mean_abs": mean_abs,
        "numel": int(acc["numel"]),
    }


def _parse_csv_ints(text: str) -> List[int]:
    out = []
    for raw in text.split(","):
        raw = raw.strip()
        if raw:
            out.append(int(raw))
    return out


def _parse_csv_floats(text: str) -> List[float]:
    out = []
    for raw in text.split(","):
        raw = raw.strip()
        if raw:
            out.append(float(raw))
    return out


def _parse_csv_strings(text: str) -> List[str]:
    out = []
    for raw in text.split(","):
        raw = raw.strip()
        if raw:
            out.append(raw)
    return out


def _strip_runtime_assertions(
    ep: "torch.export.ExportedProgram",
) -> Dict[str, Any]:
    """Best-effort removal of export-inserted runtime assertions.

    Some torch versions inject runtime metadata assertions that can pin device
    expectations to the export-time device. For selected models we strip these
    assertion nodes and run DCE to keep the graph executable across devices.
    """
    meta: Dict[str, Any] = {
        "requested": True,
        "applied": False,
        "modified": False,
        "error_type": None,
        "error": None,
    }
    try:
        from torch._export.passes.remove_runtime_assertions import (
            _RemoveRuntimeAssertionsPass,
        )

        result = _RemoveRuntimeAssertionsPass()(ep.graph_module)
        modified = bool(getattr(result, "modified", False))
        meta["modified"] = modified
        if modified:
            ep.graph_module.graph.eliminate_dead_code()
            ep.graph_module.recompile()
        meta["applied"] = True
    except Exception as exc:
        meta["error_type"] = type(exc).__name__
        meta["error"] = str(exc)
    return meta


def _dynamic_shapes(spec: Dict[str, Any]):
    dynamic_shape_spec: Dict[int, Any] = {}
    if spec.get("dynamic_batch", True):
        batch_dim = torch.export.Dim("batch", min=1, max=spec.get("batch_max", 64))
        dynamic_shape_spec[0] = batch_dim

    if spec.get("dynamic_hw", False):
        h_min = int(spec.get("dynamic_h_min", 64))
        h_max = int(spec.get("dynamic_h_max", 2048))
        w_min = int(spec.get("dynamic_w_min", 64))
        w_max = int(spec.get("dynamic_w_max", 2048))
        hw_multiple = int(spec.get("dynamic_hw_multiple", 1))

        if hw_multiple > 1:
            h_base_min = max(1, (h_min + hw_multiple - 1) // hw_multiple)
            h_base_max = max(h_base_min, h_max // hw_multiple)
            w_base_min = max(1, (w_min + hw_multiple - 1) // hw_multiple)
            w_base_max = max(w_base_min, w_max // hw_multiple)
            h_dim = hw_multiple * torch.export.Dim(
                "height_base",
                min=h_base_min,
                max=h_base_max,
            )
            w_dim = hw_multiple * torch.export.Dim(
                "width_base",
                min=w_base_min,
                max=w_base_max,
            )
        else:
            h_dim = torch.export.Dim("height", min=h_min, max=h_max)
            w_dim = torch.export.Dim("width", min=w_min, max=w_max)
        dynamic_shape_spec[2] = h_dim
        dynamic_shape_spec[3] = w_dim

    if not dynamic_shape_spec:
        return None
    return [dynamic_shape_spec]


def _allowed_state_keys(
    keys: Sequence[str], patterns: Sequence[str]
) -> Tuple[List[str], List[str]]:
    compiled = [re.compile(pattern) for pattern in patterns]
    allowed = []
    rejected = []
    for key in keys:
        target = allowed if any(pattern.fullmatch(key) for pattern in compiled) else rejected
        target.append(key)
    return allowed, rejected


def _load_state_dict_strictly(
    model: torch.nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    spec: Dict[str, Any],
) -> Dict[str, List[str]]:
    if spec.get("strict") is not True:
        raise RuntimeError(
            f"Model {spec['id']} must explicitly enable strict state reconstruction"
        )

    result = model.load_state_dict(state_dict, strict=False)
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    allowlist = spec.get("state_key_allowlist", {})
    allowed_missing, rejected_missing = _allowed_state_keys(
        missing, allowlist.get("missing", [])
    )
    allowed_unexpected, rejected_unexpected = _allowed_state_keys(
        unexpected, allowlist.get("unexpected", [])
    )

    if rejected_missing or rejected_unexpected:
        details = []
        if rejected_missing:
            details.append(f"missing={rejected_missing}")
        if rejected_unexpected:
            details.append(f"unexpected={rejected_unexpected}")
        raise RuntimeError(
            f"State reconstruction failed for {spec['id']}: " + "; ".join(details)
        )

    return {
        "missing_keys": [],
        "unexpected_keys": [],
        "allowlisted_missing_keys": allowed_missing,
        "allowlisted_unexpected_keys": allowed_unexpected,
    }


def _recover_torchscript_state_attributes(
    model: torch.nn.Module,
    scripted: torch.jit.ScriptModule,
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """Recover old TorchScript tensor attributes omitted from ``state_dict``."""
    state_dict = dict(scripted.state_dict())
    expected_state = model.state_dict()
    scripted_modules = dict(scripted.named_modules())
    recovered = []
    unrecoverable = []

    for key in sorted(set(expected_state) - set(state_dict)):
        module_path, attribute = key.rsplit(".", 1)
        scripted_module = scripted_modules.get(module_path)
        value = (
            getattr(scripted_module, attribute, None)
            if scripted_module is not None
            else None
        )
        expected = expected_state[key]
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != tuple(expected.shape)
            or value.dtype != expected.dtype
        ):
            unrecoverable.append(key)
            continue
        state_dict[key] = value.detach().cpu().clone()
        recovered.append(key)

    if unrecoverable:
        raise RuntimeError(
            "TorchScript source is missing required state attributes: "
            + ", ".join(unrecoverable)
        )
    return state_dict, recovered


def _verified_reference_path(reference: Mapping[str, Any]) -> Path:
    source = reference.get("source")
    expected_sha256 = str(reference.get("sha256", "")).lower()
    if not isinstance(source, str) or not source:
        raise RuntimeError("Validation reference must declare a source path")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise RuntimeError("Validation reference must declare an exact SHA-256")

    path = Path(source)
    if not path.is_file():
        raise RuntimeError(f"Validation reference not found: {path}")
    observed_sha256 = _sha256(path)
    if observed_sha256 != expected_sha256:
        raise RuntimeError(
            f"Validation reference digest mismatch for {path}: "
            f"expected {expected_sha256}, observed {observed_sha256}"
        )
    return path


def _load_validation_reference(spec: Dict[str, Any]):
    reference = spec.get("validation_reference")
    if not isinstance(reference, Mapping):
        raise RuntimeError(
            f"Model {spec['id']} has no independent validation reference"
        )
    kind = reference.get("kind")
    path = _verified_reference_path(reference)

    if kind != "torchscript":
        raise RuntimeError(
            f"Unsupported validation reference kind for {spec['id']}: {kind!r}"
        )

    reference_model = torch.jit.load(str(path), map_location="cpu")
    reference_model.eval()
    batch_mode = str(reference.get("batch_mode", "native")).strip().lower()
    if batch_mode not in {"native", "per_sample"}:
        raise RuntimeError(
            f"Validation reference batch mode for {spec['id']} must be native or "
            f"per_sample, got {batch_mode!r}"
        )
    fixed_device_value = reference.get("device")
    fixed_device = None
    if fixed_device_value is not None:
        if not isinstance(fixed_device_value, str) or not fixed_device_value.strip():
            raise RuntimeError(
                f"Validation reference device for {spec['id']} must be a device string"
            )
        fixed_device = torch.device(fixed_device_value.strip().lower())
        reference_model.to(fixed_device)

    def ref_fn(x):
        with torch.no_grad():
            if fixed_device is None:
                reference_model.to(x.device)
                reference_input = x
            else:
                reference_input = x.detach().to(fixed_device)
            if batch_mode == "native" or int(reference_input.shape[0]) == 1:
                return reference_model(reference_input)
            return _concat_outputs(
                [
                    reference_model(reference_input[index : index + 1])
                    for index in range(int(reference_input.shape[0]))
                ]
            )

    return ref_fn, {
        "kind": kind,
        "source": str(path),
        "sha256": str(reference["sha256"]).lower(),
        "execution_device": (
            str(fixed_device) if fixed_device is not None else "match_input"
        ),
        "batch_mode": batch_mode,
    }


def _load_native_model(spec: Dict[str, Any]):
    strategy = spec["strategy"]
    source_path = spec["source_path"]
    if spec.get("strict") is not True:
        raise RuntimeError(
            f"Model {spec['id']} must explicitly enable strict source reconstruction"
        )
    cls = _import_class(spec["class_path"])
    model = cls()

    load_meta: Dict[str, Any] = {
        "strategy": strategy,
        "source_path": source_path,
        "source_sha256": _sha256(Path(source_path)),
        "missing_keys": [],
        "unexpected_keys": [],
        "allowlisted_missing_keys": [],
        "allowlisted_unexpected_keys": [],
    }

    if strategy == "native_state_dict":
        sd = _torch_load(source_path)
        load_meta.update(_load_state_dict_strictly(model, sd, spec))
    elif strategy == "native_from_torchscript_state_dict":
        ts = torch.jit.load(source_path, map_location="cpu")
        sd = dict(ts.state_dict())
        load_meta.update(_load_state_dict_strictly(model, sd, spec))
    elif strategy == "native_from_torchscript_complete_state":
        ts = torch.jit.load(source_path, map_location="cpu")
        sd, recovered = _recover_torchscript_state_attributes(model, ts)
        load_meta.update(_load_state_dict_strictly(model, sd, spec))
        load_meta["recovered_torchscript_attribute_count"] = len(recovered)
        load_meta["recovered_torchscript_attributes"] = recovered
    elif strategy == "native_from_torchscript_constants":
        ts = torch.jit.load(source_path, map_location="cpu")
        if not hasattr(model, "load_from_torchscript"):
            raise RuntimeError(f"Model {spec['class_path']} has no load_from_torchscript")
        model.load_from_torchscript(ts)
    else:
        raise RuntimeError(f"Unsupported native strategy: {strategy}")

    model.eval()
    return model, load_meta


def _verified_reused_artifact(spec: Dict[str, Any], torch_minor: str) -> Dict[str, Any]:
    artifact_id = spec.get("reused_artifact_id")
    expected_id = f"{spec['id']}-torch{torch_minor}"
    if artifact_id != expected_id:
        raise RuntimeError(
            f"Reusable artifact for {spec['id']} must be {expected_id}, "
            f"got {artifact_id!r}"
        )

    manifest_path = (
        Path(__file__).resolve().parents[1] / "facetorch" / "models" / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        model_record = manifest["models"][spec["id"]]
        artifact = next(
            item
            for item in model_record["artifacts"]
            if item.get("id") == artifact_id
        )
    except (KeyError, StopIteration) as exc:
        raise RuntimeError(
            f"Reusable artifact {artifact_id} is not in the packaged manifest"
        ) from exc

    source_path = Path(spec["source_path"])
    observed_sha256 = _sha256(source_path)
    observed_size = source_path.stat().st_size
    if (
        artifact.get("format") != "pt2"
        or artifact.get("sha256") != observed_sha256
        or int(artifact.get("size_bytes", -1)) != observed_size
    ):
        raise RuntimeError(
            f"Reusable artifact integrity mismatch for {spec['id']}: "
            f"observed sha256={observed_sha256}, size={observed_size}"
        )

    return {
        "id": artifact_id,
        "repo_id": model_record["repo_id"],
        "revision": model_record["revision"],
        "filename": artifact["filename"],
        "sha256": artifact["sha256"],
        "size_bytes": int(artifact["size_bytes"]),
    }


def _build_reference_and_exported_program(spec: Dict[str, Any], torch_minor: str):
    strategy = spec["strategy"]
    input_shape = spec["input_shape"]
    export_generator = torch.Generator(device="cpu")
    export_generator.manual_seed(int(spec.get("export_seed", 0)))
    dummy = torch.randn(*input_shape, generator=export_generator)

    if strategy in {
        "native_state_dict",
        "native_from_torchscript_state_dict",
        "native_from_torchscript_complete_state",
        "native_from_torchscript_constants",
    }:
        model, load_meta = _load_native_model(spec)
        ep = torch.export.export(
            model,
            (dummy,),
            dynamic_shapes=_dynamic_shapes(spec),
        )
        if spec.get("strip_runtime_assertions", False):
            load_meta["runtime_assertion_stripping"] = _strip_runtime_assertions(ep)
    elif strategy == "ts2ep_reexport_dynamic":
        from torch._export.converter import TS2EPConverter

        ts = torch.jit.load(spec["source_path"], map_location="cpu")
        ts.eval()
        ep_ts = TS2EPConverter(ts, (dummy,), {}).convert()
        mod = ep_ts.module()
        ep = torch.export.export(
            mod,
            (dummy,),
            dynamic_shapes=_dynamic_shapes(spec),
        )
        load_meta = {
            "strategy": strategy,
            "source_path": spec["source_path"],
            "missing_keys": [],
            "unexpected_keys": [],
        }
        if spec.get("strip_runtime_assertions", False):
            load_meta["runtime_assertion_stripping"] = _strip_runtime_assertions(ep)
    elif strategy == "reuse_existing_exported_program":
        reuse_reason = spec.get("reuse_reason")
        if not isinstance(reuse_reason, str) or not reuse_reason.strip():
            raise RuntimeError(
                f"Model {spec['id']} must document why its existing program is reused"
            )
        reused_artifact = _verified_reused_artifact(spec, torch_minor)
        ep = torch.export.load(spec["source_path"])
        load_meta = {
            "strategy": strategy,
            "source_path": spec["source_path"],
            "source_sha256": _sha256(Path(spec["source_path"])),
            "reused_artifact": reused_artifact,
            "reuse_reason": reuse_reason,
            "missing_keys": [],
            "unexpected_keys": [],
        }
    else:
        raise RuntimeError(f"Unknown strategy: {strategy}")

    ref_fn, reference_meta = _load_validation_reference(spec)
    load_meta["validation_reference"] = reference_meta
    return ref_fn, ep, load_meta


def _tensor_schema(*shape: Any) -> Dict[str, Any]:
    return {"type": "tensor", "shape": list(shape), "floating": True}


def _sequence_schema(*items: Mapping[str, Any]) -> Dict[str, Any]:
    return {"type": "sequence", "items": [dict(item) for item in items]}


def _validation_contract(
    *,
    task: str,
    reference_source: str,
    reference_sha256: str,
    output_schema: Mapping[str, Any],
    output_invariants: Sequence[str] = (),
    reference_device: str | None = "cpu",
    reference_batch_mode: str | None = None,
) -> Dict[str, Any]:
    validation_reference = {
        "kind": "torchscript",
        "source": reference_source,
        "sha256": reference_sha256,
    }
    if reference_device is not None:
        validation_reference["device"] = reference_device
    if reference_batch_mode is not None:
        validation_reference["batch_mode"] = reference_batch_mode

    return {
        "task": task,
        "required_batch_sizes": [1, 2, 4, 8],
        "validation_reference": validation_reference,
        "tolerances": {
            "dtype": "float32",
            "max_abs": 1e-4,
            "mean_abs": 1e-5,
            "justification": (
                f"{task} float32 export-equivalence guard. The maximum and mean "
                "bounds are included in the digest-bound publication plan, so "
                "release approval covers these exact numerical tolerances."
            ),
        },
        "cross_device_tolerances": {
            "dtype": "float32",
            "max_abs": 2e-3,
            "mean_abs": 1e-3,
            "justification": (
                f"{task} CPU/CUDA kernel drift guard, distinct from same-device "
                "export equivalence. The bound is included in digest-bound "
                "publication approval."
            ),
        },
        "output_schema": dict(output_schema),
        "output_invariants": list(output_invariants),
    }


def _model_specs(torch_minor: str) -> List[Dict[str, Any]]:
    specs = [
        {
            "id": "detector-retinaface",
            "repo_id": "tomas-gajarsky/facetorch-detector-retinaface",
            "class_path": "model_defs.detector_model.RetinaFaceResNet50",
            "strategy": "native_from_torchscript_complete_state",
            "source_path": "models/torchscript/detector/1/model.pt",
            "strict": True,
            "input_shape": [1, 3, 480, 640],
            "dynamic_batch": False,
            "dynamic_hw": True,
            "dynamic_hw_multiple": 32,
            "batch_max": 32,
            "validation_batch_sizes": [1],
            "validation_spatial_shapes": [
                [480, 640],
                [512, 512],
                [480, 608],
            ],
            **_validation_contract(
                task="face-detection",
                reference_source="models/torchscript/detector/1/model.pt",
                reference_sha256=(
                    "05e524af9b55bbf92b752b064c298f170ae763eb0ac0a4162a92e39fff007def"
                ),
                output_schema=_sequence_schema(
                    _tensor_schema("batch", "any", 4),
                    _tensor_schema("batch", "any", 2),
                    _tensor_schema("batch", "any", 10),
                ),
                output_invariants=("detector_heads",),
                reference_device="cpu",
            ),
            "required_batch_sizes": [1],
        },
        {
            "id": "fer-efficientnet-b2",
            "repo_id": "tomas-gajarsky/facetorch-fer-efficientnet-b2",
            "class_path": "model_defs.fer_model.EfficientNetB2FER",
            "strategy": "native_state_dict",
            "source_path": "models_local/state_dicts/fer.pth",
            "strict": True,
            "input_shape": [2, 3, 260, 260],
            **_validation_contract(
                task="facial-expression-8",
                reference_source="models/torchscript/predictor/fer/2/model.pt",
                reference_sha256=(
                    "91672ff45602901b1631db910ec2e65db21def35bda550c5f4e30a75361c9bdd"
                ),
                output_schema=_tensor_schema("batch", 8),
            ),
        },
        {
            "id": "fer-efficientnet-b0",
            "repo_id": "tomas-gajarsky/facetorch-fer-efficientnet-b0",
            "class_path": "model_defs.fer_model.EfficientNetB0FER",
            "strategy": "native_from_torchscript_state_dict",
            "source_path": "models_local/fer_b0/model.pt",
            "strict": True,
            "input_shape": [2, 3, 244, 244],
            **_validation_contract(
                task="facial-expression-7",
                reference_source="models/torchscript/predictor/fer/1/model.pt",
                reference_sha256=(
                    "39d8046b1fe3eb06d5edb094307f0cd465a40b7f8886d4b477ca4bbaf7cbb62e"
                ),
                output_schema=_tensor_schema("batch", 7),
            ),
        },
        {
            "id": "au-opengraph",
            "repo_id": "tomas-gajarsky/facetorch-au-opengraph",
            "class_path": "model_defs.au_model.OpenGraphAU",
            "strategy": "reuse_existing_exported_program",
            "source_path": (
                "models_local/reused/au-opengraph/"
                f"model-torch{torch_minor}.pt2"
            ),
            "reused_artifact_id": f"au-opengraph-torch{torch_minor}",
            "reuse_reason": (
                "The pinned legacy TorchScript state does not reproduce its outputs "
                "when reconstructed with the current native timm architecture. The "
                "digest-pinned published program independently matches the legacy "
                "reference, so it is preserved pending checkpoint recovery."
            ),
            "strict": True,
            "input_shape": [2, 3, 224, 224],
            **_validation_contract(
                task="action-unit",
                reference_source="models/torchscript/predictor/au/1/model.pt",
                reference_sha256=(
                    "bb437cd01d069e82a448f83e8f2391718a71ac45068435bf276428e15cc1bad8"
                ),
                output_schema=_tensor_schema("batch", 41),
                output_invariants=("bounded_unit_interval",),
                reference_batch_mode="per_sample",
            ),
        },
        {
            "id": "va-elim",
            "repo_id": "tomas-gajarsky/facetorch-va-elim",
            "class_path": "model_defs.va_model.ELIMALAlexNet",
            "strategy": "native_state_dict",
            "source_path": "models_local/state_dicts/va.pth",
            "strict": True,
            "input_shape": [2, 3, 224, 224],
            **_validation_contract(
                task="valence-arousal",
                reference_source="models/torchscript/predictor/va/1/model.pt",
                reference_sha256=(
                    "8f6a9a127e93343e75edf7311f631fbab7fc37b1e0c9af82103c31daf99ec64a"
                ),
                output_schema=_tensor_schema("batch", 2),
            ),
        },
        {
            "id": "embed-resnet50",
            "repo_id": "tomas-gajarsky/facetorch-embed-resnet-50",
            "class_path": "model_defs.embed_model.EmbedResNet50",
            "strategy": "native_state_dict",
            "source_path": "models_local/state_dicts/embed.pth",
            "strict": True,
            "input_shape": [2, 3, 244, 244],
            **_validation_contract(
                task="face-embedding",
                reference_source="models/torchscript/predictor/embed/1/model.pt",
                reference_sha256=(
                    "3911c73efe902ca0810bf5ced0b8a9bbaa84356860131ec3cb61eb1493e43807"
                ),
                output_schema=_sequence_schema(
                    _tensor_schema("batch", 128),
                    _tensor_schema("batch", 3000),
                ),
                output_invariants=("first_embedding_normalized",),
            ),
        },
        {
            "id": "deepfake-efficientnet-b7",
            "repo_id": "tomas-gajarsky/facetorch-deepfake-efficientnet-b7",
            "class_path": "model_defs.deepfake_model.DeepfakeEfficientNetB7",
            "strategy": "native_state_dict",
            "source_path": "models_local/state_dicts/deepfake.pth",
            "strict": True,
            "input_shape": [2, 3, 380, 380],
            **_validation_contract(
                task="deepfake-binary",
                reference_source="models/torchscript/predictor/deepfake/1/model.pt",
                reference_sha256=(
                    "e36197ae83fa7c050e8ceba267a1f028f2d74d8ad805bf5d5037a26e8ca96c13"
                ),
                output_schema=_tensor_schema("batch", 1),
            ),
        },
        {
            "id": "align-synergynet",
            "repo_id": "tomas-gajarsky/facetorch-align-synergynet",
            "class_path": "model_defs.align_model.SynergyNetMobileNetV2",
            "strategy": "native_state_dict",
            "source_path": "models_local/state_dicts/align.pth",
            "strict": True,
            "input_shape": [2, 3, 120, 120],
            **_validation_contract(
                task="face-alignment-3d",
                reference_source="models/torchscript/predictor/align/1/model.pt",
                reference_sha256=(
                    "09220491e3a0c11eb1c076b9c6551e214a078b54b86acb6f0e94165e00785df4"
                ),
                output_schema=_tensor_schema("batch", 62),
            ),
        },
        {
            "id": "verify-magface",
            "repo_id": "tomas-gajarsky/facetorch-verify-magface",
            "class_path": "model_defs.verify_model.MagFaceIResNet100",
            "strategy": "native_from_torchscript_state_dict",
            "source_path": "models/torchscript/predictor/verify/1/model.pt",
            "strict": True,
            "input_shape": [2, 3, 112, 112],
            **_validation_contract(
                task="face-verification-magface",
                reference_source="models/torchscript/predictor/verify/1/model.pt",
                reference_sha256=(
                    "0c54aab654369235b25c4df96214cf9fda23c8535a48eb5c9a0c8a03c79601dc"
                ),
                output_schema=_tensor_schema("batch", 512),
            ),
        },
        {
            "id": "verify-adaface",
            "repo_id": "tomas-gajarsky/facetorch-verify-adaface",
            "class_path": "model_defs.verify_model.VerifyIResNet100",
            "strategy": "native_from_torchscript_constants",
            "source_path": "models/torchscript/predictor/verify/2/model.pt",
            "strict": True,
            "input_shape": [2, 3, 112, 112],
            **_validation_contract(
                task="face-verification-adaface",
                reference_source="models/torchscript/predictor/verify/2/model.pt",
                reference_sha256=(
                    "edc3639bd4affeeaebe0bf01bacbcdf109c5a29713278080955d362191d1aa0a"
                ),
                reference_device="cpu",
                output_schema=_sequence_schema(
                    _tensor_schema("batch", 512),
                    _tensor_schema("batch", 1),
                ),
                output_invariants=("first_embedding_normalized",),
            ),
        },
    ]

    source_revisions = {
        "detector-retinaface": "2e1015f21f8c743ca6a356a1cba3223232096502",
        "align-synergynet": "f90c6d92f0ac022660b80079210d3cdeb46bb1e6",
        "au-opengraph": "3284478cd26b09de911783df1d8fdbb01e037eb3",
        "deepfake-efficientnet-b7": "3ecdd5193f7e981b7f3dad2407658c5456ae671c",
        "embed-resnet50": "e98f06a0c724e7dfd893b12d94165444978a449e",
        "fer-efficientnet-b0": "3ddf81c0c80964e830ff0b0031b4eddd8d454cf7",
        "fer-efficientnet-b2": "2c9829a28c014e2f9b7a8ed683d1c7fd1631b18c",
        "va-elim": "62590460b05d96fb0aafbf71b0462b8b42901f6e",
        "verify-adaface": "348ecc75d4624edc60a809ef3867f6a6537c7f2d",
        "verify-magface": "f0f8b2f12f61a6ba4471d3823a8fad403c4a75f9",
    }
    for spec in specs:
        reference = spec["validation_reference"]
        spec["source_artifact"] = {
            "repo_id": spec["repo_id"],
            "revision": source_revisions[spec["id"]],
            "filename": "model.pt",
            "sha256": reference["sha256"],
        }
    return specs


def _safe_source_target(repo_root: Path, relative_path: str) -> Path:
    target = (repo_root / relative_path).resolve()
    try:
        target.relative_to(repo_root)
    except ValueError as exc:
        raise RuntimeError(f"Model source escapes repository root: {relative_path}") from exc
    return target


def _copy_verified_source(
    source: Path,
    target: Path,
    *,
    expected_sha256: str,
    expected_size: int,
) -> None:
    if _sha256(source) != expected_sha256 or source.stat().st_size != expected_size:
        raise RuntimeError(f"Downloaded source integrity mismatch for {target.name}")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False
        ) as output:
            temporary = Path(output.name)
            with source.open("rb") as input_file:
                shutil.copyfileobj(input_file, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        if (
            _sha256(temporary) != expected_sha256
            or temporary.stat().st_size != expected_size
        ):
            raise RuntimeError(f"Staged source integrity mismatch for {target.name}")
        temporary.chmod(0o644)
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_state_dict_atomic(target: Path, state_dict: Mapping[str, Any]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False
        ) as output:
            temporary = Path(output.name)
        torch.save(dict(state_dict), temporary)
        with temporary.open("rb") as saved:
            os.fsync(saved.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _prepare_model_sources(
    specs: Sequence[Dict[str, Any]],
    *,
    repo_root: Path,
    cohort: str,
    offline: bool = False,
    force_download: bool = False,
    download_fn=None,
    environment_lock: str | Path | None = None,
) -> Dict[str, Any]:
    """Fetch pinned legacy references and derive local export inputs.

    Every remote byte is checked against the packaged manifest before TorchScript
    deserialization. The generated inventory is evidence, not release approval.
    """
    if download_fn is None:
        from huggingface_hub import hf_hub_download

        download_fn = hf_hub_download

    manifest_path = repo_root / "facetorch" / "models" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_models = manifest.get("models", {})
    prepared = []

    for spec in specs:
        model_id = spec["id"]
        try:
            model = manifest_models[model_id]
        except KeyError as exc:
            raise RuntimeError(f"No packaged manifest record for {model_id}") from exc

        source = spec["source_artifact"]
        if (
            model.get("repo_id") != source["repo_id"]
            or model.get("revision") != source["revision"]
            or model.get("source_weight_sha256") != source["sha256"]
        ):
            raise RuntimeError(f"Source contract disagrees with manifest for {model_id}")

        legacy = next(
            (
                artifact
                for artifact in model.get("artifacts", [])
                if artifact.get("filename") == source["filename"]
                and artifact.get("format") == "torchscript"
            ),
            None,
        )
        if legacy is None or legacy.get("sha256") != source["sha256"]:
            raise RuntimeError(f"No digest-bound legacy source for {model_id}")

        downloaded = Path(
            download_fn(
                repo_id=source["repo_id"],
                filename=source["filename"],
                revision=source["revision"],
                local_files_only=offline,
                force_download=force_download,
            )
        )
        reference_path = _safe_source_target(
            repo_root, spec["validation_reference"]["source"]
        )
        _copy_verified_source(
            downloaded,
            reference_path,
            expected_sha256=source["sha256"],
            expected_size=int(legacy["size_bytes"]),
        )

        source_path = _safe_source_target(repo_root, spec["source_path"])
        strategy = spec["strategy"]
        reused_artifact_record = None
        if strategy == "native_state_dict":
            verified_ts = torch.jit.load(str(reference_path), map_location="cpu")
            _write_state_dict_atomic(source_path, verified_ts.state_dict())
        elif strategy in {
            "native_from_torchscript_state_dict",
            "native_from_torchscript_complete_state",
            "native_from_torchscript_constants",
        }:
            if source_path != reference_path:
                _copy_verified_source(
                    downloaded,
                    source_path,
                    expected_sha256=source["sha256"],
                    expected_size=int(legacy["size_bytes"]),
                )
        elif strategy == "reuse_existing_exported_program":
            artifact = next(
                (
                    item
                    for item in model.get("artifacts", [])
                    if item.get("id") == f"{model_id}-torch{cohort}"
                ),
                None,
            )
            if artifact is None or artifact.get("format") != "pt2":
                raise RuntimeError(
                    f"No pinned reusable export for {model_id} cohort {cohort}"
                )
            downloaded_export = Path(
                download_fn(
                    repo_id=source["repo_id"],
                    filename=artifact["filename"],
                    revision=source["revision"],
                    local_files_only=offline,
                    force_download=force_download,
                )
            )
            _copy_verified_source(
                downloaded_export,
                source_path,
                expected_sha256=artifact["sha256"],
                expected_size=int(artifact["size_bytes"]),
            )
            reused_artifact_record = {
                "id": artifact["id"],
                "filename": artifact["filename"],
                "sha256": artifact["sha256"],
                "size_bytes": int(artifact["size_bytes"]),
            }
        else:
            raise RuntimeError(f"Unsupported source preparation strategy: {strategy}")

        prepared.append(
            {
                "model_id": model_id,
                "repo_id": source["repo_id"],
                "revision": source["revision"],
                "reference_path": str(reference_path.relative_to(repo_root)),
                "reference_sha256": source["sha256"],
                "source_path": str(source_path.relative_to(repo_root)),
                "source_sha256": _sha256(source_path),
                "strategy": strategy,
                "reused_artifact": reused_artifact_record,
            }
        )

    return {
        "schema_version": 1,
        "generated_at_utc": _now_iso(),
        "cohort": cohort,
        "environment": _environment_metadata(repo_root, environment_lock),
        "models": prepared,
    }


def _build_validation_cases(
    spec: Dict[str, Any],
    batch_sizes: Sequence[int],
    seeds: Sequence[int],
    scales: Sequence[float],
) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    c, h, w = spec["input_shape"][1:]
    max_batch = int(spec.get("batch_max", 64))
    effective_batch_sizes = spec.get("validation_batch_sizes", batch_sizes)
    spatial_shapes = spec.get("validation_spatial_shapes", [(h, w)])

    for b in effective_batch_sizes:
        if b < 1 or b > max_batch:
            continue
        for case_h, case_w in spatial_shapes:
            for seed in seeds:
                for scale in scales:
                    case_seed = int(
                        seed * 100_003
                        + b * 313
                        + int(case_h) * 37
                        + int(case_w) * 41
                        + round(scale * 1000.0) * 17
                    )

                    g1 = torch.Generator(device="cpu")
                    g1.manual_seed(case_seed)
                    x_randn = (
                        torch.randn((b, c, int(case_h), int(case_w)), generator=g1)
                        * scale
                    )

                    g2 = torch.Generator(device="cpu")
                    g2.manual_seed(case_seed + 7)
                    x_randu = (
                        torch.rand(
                            (b, c, int(case_h), int(case_w)), generator=g2
                        )
                        * 2.0
                        - 1.0
                    ) * scale

                    case_prefix = (
                        f"b{b}_h{int(case_h)}_w{int(case_w)}_"
                        f"seed{seed}_scale{scale}"
                    )
                    cases.append(
                        {
                            "id": f"{case_prefix}_randn",
                            "batch": b,
                            "seed": seed,
                            "scale": scale,
                            "variant": "randn",
                            "x": x_randn,
                        }
                    )
                    cases.append(
                        {
                            "id": f"{case_prefix}_randu",
                            "batch": b,
                            "seed": seed,
                            "scale": scale,
                            "variant": "randu",
                            "x": x_randu,
                        }
                    )

    return cases


def _validate_output_schema_node(
    schema: Mapping[str, Any], output: Any, batch: int, path: str = "output"
) -> None:
    kind = schema.get("type")
    if kind == "tensor":
        if not isinstance(output, torch.Tensor):
            raise RuntimeError(f"{path} must be a tensor, got {type(output).__name__}")
        expected_shape = schema.get("shape")
        if expected_shape is not None:
            if output.ndim != len(expected_shape):
                raise RuntimeError(
                    f"{path} rank mismatch: {output.ndim} != {len(expected_shape)}"
                )
            for axis, (observed, expected) in enumerate(
                zip(output.shape, expected_shape)
            ):
                if expected in {None, "any"}:
                    continue
                expected_value = batch if expected == "batch" else int(expected)
                if int(observed) != expected_value:
                    raise RuntimeError(
                        f"{path} shape mismatch at axis {axis}: "
                        f"{int(observed)} != {expected_value}"
                    )
        if schema.get("floating", True) and not output.is_floating_point():
            raise RuntimeError(f"{path} must have a floating-point dtype")
        return

    if kind == "sequence":
        if not isinstance(output, (tuple, list)):
            raise RuntimeError(
                f"{path} must be a tensor sequence, got {type(output).__name__}"
            )
        items = schema.get("items", [])
        if len(output) != len(items):
            raise RuntimeError(f"{path} length mismatch: {len(output)} != {len(items)}")
        for index, (item_schema, value) in enumerate(zip(items, output)):
            _validate_output_schema_node(
                item_schema, value, batch, f"{path}[{index}]"
            )
        return

    raise RuntimeError(f"Unsupported output schema type at {path}: {kind!r}")


def _validate_task_invariants(spec: Mapping[str, Any], output: Any) -> None:
    for invariant in spec.get("output_invariants", []):
        if invariant == "detector_heads":
            bbox, probabilities, landmarks = output
            anchors = int(bbox.shape[1])
            if anchors < 1 or any(
                int(value.shape[1]) != anchors for value in (probabilities, landmarks)
            ):
                raise RuntimeError("Detector heads do not share a non-empty anchor axis")
            if (probabilities < -1e-6).any() or (probabilities > 1.0 + 1e-6).any():
                raise RuntimeError("Detector class probabilities are outside [0, 1]")
            sums = probabilities.to(torch.float64).sum(dim=-1)
            if not torch.allclose(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5):
                raise RuntimeError("Detector class probabilities do not sum to one")
        elif invariant == "first_embedding_normalized":
            embedding = output[0] if isinstance(output, (tuple, list)) else output
            norms = torch.linalg.vector_norm(embedding.to(torch.float64), dim=1)
            if not torch.allclose(
                norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4
            ):
                raise RuntimeError("Embedding output is not L2-normalized")
        elif invariant == "bounded_unit_interval":
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if (tensor < -1.0 - 1e-5).any() or (tensor > 1.0 + 1e-5).any():
                raise RuntimeError("Output is outside the expected [-1, 1] interval")
        else:
            raise RuntimeError(f"Unknown task invariant: {invariant!r}")


def _validate_output_contract(
    spec: Mapping[str, Any], output: Any, batch: int, label: str
) -> None:
    _ensure_finite_output(output, label)
    schema = spec.get("output_schema")
    if schema is not None:
        _validate_output_schema_node(schema, output, batch)
    _validate_task_invariants(spec, output)


def _validate_exported_module_impl(
    spec: Dict[str, Any],
    ref_fn,
    exported_module,
    batch_sizes: Sequence[int],
    seeds: Sequence[int],
    scales: Sequence[float],
    devices: Sequence[str],
) -> Dict[str, Any]:
    cases = _build_validation_cases(spec, batch_sizes, seeds, scales)
    tolerance = spec.get("tolerances", {})
    max_abs_tolerance = float(
        tolerance.get("max_abs", spec.get("max_abs_tolerance", 1e-4))
    )
    mean_abs_tolerance = float(
        tolerance.get("mean_abs", spec.get("mean_abs_tolerance", 1e-5))
    )
    cross_device_tolerance = spec.get("cross_device_tolerances", tolerance)
    cross_device_max_abs_tolerance = float(
        cross_device_tolerance.get("max_abs", max_abs_tolerance)
    )
    cross_device_mean_abs_tolerance = float(
        cross_device_tolerance.get("mean_abs", mean_abs_tolerance)
    )

    validated_devices = []
    validation_failures = []
    worst_max_abs = 0.0
    worst_mean_abs = 0.0
    worst_case_id = None
    worst_device = None
    total_cases = 0
    requested_devices = [str(device).strip().lower() for device in devices]
    fixed_reference_device = spec.get("validation_reference", {}).get("device")
    if fixed_reference_device is not None:
        fixed_reference_device = str(
            torch.device(str(fixed_reference_device).strip().lower())
        )
    matrix_failures = []
    baseline_device = None
    baseline_outputs: Dict[str, Any] = {}
    fixed_reference_outputs: Dict[str, Any] = {}
    cross_device_results = []

    if not requested_devices or any(not device for device in requested_devices):
        matrix_failures.append(
            {
                "kind": "incomplete_matrix",
                "reason": "at_least_one_validation_device_is_required",
            }
        )
    if len(set(requested_devices)) != len(requested_devices):
        matrix_failures.append(
            {
                "kind": "incomplete_matrix",
                "reason": "validation_devices_must_be_unique",
            }
        )

    observed_batches = {int(case["batch"]) for case in cases}
    required_batches = {int(value) for value in spec.get("required_batch_sizes", [])}
    missing_batches = sorted(required_batches - observed_batches)
    if missing_batches:
        matrix_failures.append(
            {
                "kind": "incomplete_matrix",
                "reason": "missing_required_batch_sizes",
                "missing_batch_sizes": missing_batches,
            }
        )

    observed_shapes = {tuple(int(value) for value in case["x"].shape[-2:]) for case in cases}
    required_shapes = {
        tuple(int(value) for value in shape)
        for shape in spec.get("validation_spatial_shapes", [])
    }
    missing_shapes = sorted(required_shapes - observed_shapes)
    if missing_shapes:
        matrix_failures.append(
            {
                "kind": "incomplete_matrix",
                "reason": "missing_required_spatial_shapes",
                "missing_spatial_shapes": [list(shape) for shape in missing_shapes],
            }
        )

    if not cases:
        matrix_failures.append(
            {
                "kind": "incomplete_matrix",
                "reason": "zero_validation_cases",
            }
        )

    validation_failures.extend(matrix_failures)

    for device_name in requested_devices:
        if not device_name:
            continue

        if device_name.startswith("cuda") and not torch.cuda.is_available():
            validated_devices.append(
                {
                    "device": device_name,
                    "status": "skipped",
                    "reason": "cuda_unavailable",
                    "num_cases": 0,
                    "failures": [],
                    "cases": [],
                }
            )
            validation_failures.append(
                {
                    "kind": "required_device_not_ok",
                    "device": device_name,
                    "status": "skipped",
                    "reason": "cuda_unavailable",
                }
            )
            continue

        case_results = []
        device_failures = list(matrix_failures)
        device_worst_max_abs = 0.0
        device_worst_mean_abs = 0.0
        device_worst_case_id = None
        use_cross_device_reference_tolerance = (
            fixed_reference_device is not None
            and torch.device(device_name) != torch.device(fixed_reference_device)
        )
        reference_max_abs_tolerance = (
            cross_device_max_abs_tolerance
            if use_cross_device_reference_tolerance
            else max_abs_tolerance
        )
        reference_mean_abs_tolerance = (
            cross_device_mean_abs_tolerance
            if use_cross_device_reference_tolerance
            else mean_abs_tolerance
        )

        try:
            torch_device = torch.device(device_name)
            exported_module.to(torch_device)
        except Exception as exc:
            failure = {
                "kind": "device_setup_error",
                "device": device_name,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            validation_failures.append(failure)
            device_failures.append(failure)
            validated_devices.append(
                {
                    "device": device_name,
                    "status": "failed",
                    "num_cases": 0,
                    "failures": device_failures,
                    "cases": [],
                }
            )
            continue

        for case in cases:
            x = case["x"].to(torch_device)
            ref_out = None
            exp_out = None
            try:
                with torch.no_grad():
                    if (
                        fixed_reference_device is not None
                        and case["id"] in fixed_reference_outputs
                    ):
                        ref_out = fixed_reference_outputs[case["id"]]
                    else:
                        ref_out = _clone_output_cpu(ref_fn(x))
                        if fixed_reference_device is not None:
                            fixed_reference_outputs[case["id"]] = ref_out
                    exp_out = _clone_output_cpu(exported_module(x))

                _validate_output_contract(
                    spec, ref_out, int(case["batch"]), "Reference output"
                )
                _validate_output_contract(
                    spec, exp_out, int(case["batch"]), "Exported output"
                )
                stats = _compute_diff_stats(ref_out, exp_out)
            except Exception as exc:
                failure = {
                    "kind": "case_validation_error",
                    "device": device_name,
                    "case_id": case["id"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                validation_failures.append(failure)
                device_failures.append(failure)
                case_results.append(
                    {
                        "case_id": case["id"],
                        "status": "failed",
                        "batch": case["batch"],
                        "seed": case["seed"],
                        "scale": case["scale"],
                        "variant": case["variant"],
                        "input_shape": list(x.shape),
                        "output_summary": (
                            _summarize_output(exp_out) if exp_out is not None else None
                        ),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                continue

            if (
                device_worst_case_id is None
                or stats["max_abs"] > device_worst_max_abs
            ):
                device_worst_max_abs = stats["max_abs"]
                device_worst_mean_abs = stats["mean_abs"]
                device_worst_case_id = case["id"]

            if worst_case_id is None or stats["max_abs"] > worst_max_abs:
                worst_max_abs = stats["max_abs"]
                worst_mean_abs = stats["mean_abs"]
                worst_case_id = case["id"]
                worst_device = device_name

            failed = (
                stats["max_abs"] > reference_max_abs_tolerance
                or stats["mean_abs"] > reference_mean_abs_tolerance
            )
            case_result = {
                "case_id": case["id"],
                "status": "failed" if failed else "ok",
                "batch": case["batch"],
                "seed": case["seed"],
                "scale": case["scale"],
                "variant": case["variant"],
                "input_shape": list(x.shape),
                "output_summary": _summarize_output(exp_out),
                "reference_output_sha256": _output_sha256(ref_out),
                "exported_output_sha256": _output_sha256(exp_out),
                "max_abs_diff_vs_reference": stats["max_abs"],
                "mean_abs_diff_vs_reference": stats["mean_abs"],
                "numel_compared": stats["numel"],
                "reference_execution_device": (
                    fixed_reference_device or device_name
                ),
                "reference_max_abs_tolerance": reference_max_abs_tolerance,
                "reference_mean_abs_tolerance": reference_mean_abs_tolerance,
            }

            if failed:
                failure = {
                    "kind": "reference_drift",
                    "device": device_name,
                    "case_id": case["id"],
                    "max_abs_diff_vs_reference": stats["max_abs"],
                    "mean_abs_diff_vs_reference": stats["mean_abs"],
                }
                validation_failures.append(failure)
                device_failures.append(failure)

            if baseline_device is None:
                baseline_device = device_name
            if device_name == baseline_device:
                baseline_outputs[case["id"]] = exp_out
            elif case["id"] in baseline_outputs:
                cross_stats = _compute_diff_stats(
                    baseline_outputs[case["id"]], exp_out
                )
                cross_failed = (
                    cross_stats["max_abs"] > cross_device_max_abs_tolerance
                    or cross_stats["mean_abs"] > cross_device_mean_abs_tolerance
                )
                cross_result = {
                    "baseline_device": baseline_device,
                    "device": device_name,
                    "case_id": case["id"],
                    "status": "failed" if cross_failed else "ok",
                    "max_abs_diff": cross_stats["max_abs"],
                    "mean_abs_diff": cross_stats["mean_abs"],
                }
                cross_device_results.append(cross_result)
                if cross_failed:
                    failure = {"kind": "cross_device_drift", **cross_result}
                    validation_failures.append(failure)
                    device_failures.append(failure)

            case_results.append(case_result)

        total_cases += len(case_results)
        validated_devices.append(
            {
                "device": device_name,
                "status": "failed" if device_failures else "ok",
                "num_cases": len(case_results),
                "worst_case_id": device_worst_case_id,
                "worst_max_abs_diff_vs_reference": device_worst_max_abs,
                "worst_mean_abs_diff_vs_reference": device_worst_mean_abs,
                "reference_execution_device": (
                    fixed_reference_device or device_name
                ),
                "reference_tolerance_kind": (
                    "cross_device"
                    if use_cross_device_reference_tolerance
                    else "same_device"
                ),
                "failures": device_failures,
                "cases": case_results,
            }
        )

    every_device_ok = bool(validated_devices) and all(
        item["status"] == "ok" for item in validated_devices
    )
    status = (
        "ok"
        if not validation_failures and total_cases > 0 and every_device_ok
        else "failed"
    )

    return {
        "status": status,
        "num_cases": total_cases,
        "requested_devices": requested_devices,
        "max_abs_tolerance": max_abs_tolerance,
        "mean_abs_tolerance": mean_abs_tolerance,
        "tolerance_justification": tolerance.get("justification"),
        "cross_device_max_abs_tolerance": cross_device_max_abs_tolerance,
        "cross_device_mean_abs_tolerance": cross_device_mean_abs_tolerance,
        "cross_device_tolerance_justification": cross_device_tolerance.get(
            "justification"
        ),
        "fixed_reference_device": fixed_reference_device,
        "output_schema": spec.get("output_schema"),
        "worst_case_id": worst_case_id,
        "worst_device": worst_device,
        "worst_max_abs_diff_vs_reference": worst_max_abs,
        "worst_mean_abs_diff_vs_reference": worst_mean_abs,
        "failures": validation_failures,
        "devices": validated_devices,
        "cross_device": cross_device_results,
    }


def _validate_exported_module(
    spec: Dict[str, Any],
    ref_fn,
    exported_module,
    batch_sizes: Sequence[int],
    seeds: Sequence[int],
    scales: Sequence[float],
    devices: Sequence[str],
) -> Dict[str, Any]:
    with _validation_numeric_policy() as numeric_policy:
        result = _validate_exported_module_impl(
            spec,
            ref_fn=ref_fn,
            exported_module=exported_module,
            batch_sizes=batch_sizes,
            seeds=seeds,
            scales=scales,
            devices=devices,
        )
    result["numeric_policy"] = numeric_policy
    return result


def _resolve_artifact_path(artifacts_root: Path, spec: Dict[str, Any], cohort: str) -> Path:
    return artifacts_root / spec["id"] / f"model-torch{cohort}.pt2"


def _run_for_specs(
    specs: Iterable[Dict[str, Any]],
    mode: str,
    repo_root: Path,
    cohort: str,
    out_root: Path,
    artifacts_root: Path,
    upload: bool,
    hf_token_env: str,
    batch_sizes: Sequence[int],
    seeds: Sequence[int],
    scales: Sequence[float],
    validate_devices: Sequence[str],
    environment_lock: str | Path | None = None,
) -> Tuple[Dict[str, Any], List[Tuple[str, str, str]]]:
    if mode not in {"export", "validate"}:
        raise RuntimeError(f"Unsupported mode: {mode}")

    specs_list = list(specs)
    environment = _environment_metadata(repo_root, environment_lock)
    runtime_torch_minor = ".".join(
        str(torch.__version__).split("+")[0].split(".")[:2]
    )
    exporter_arguments = {
        "mode": mode,
        "artifact_cohort": cohort,
        "batch_sizes": list(batch_sizes),
        "seeds": list(seeds),
        "scales": list(scales),
        "validate_devices": list(validate_devices),
        "model_ids": [spec["id"] for spec in specs_list],
    }

    summary: Dict[str, Any] = {
        "schema_version": 2,
        "generated_at_utc": _now_iso(),
        "mode": mode,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_minor": cohort,
        "runtime_torch_minor": runtime_torch_minor,
        "environment": environment,
        "exporter_arguments": exporter_arguments,
        "batch_sizes": list(batch_sizes),
        "seeds": list(seeds),
        "scales": list(scales),
        "validate_devices": list(validate_devices),
        "requested_model_ids": [spec["id"] for spec in specs_list],
        "results": [],
    }

    failures = []

    for idx, spec in enumerate(specs_list, start=1):
        os.chdir(repo_root)
        print(f"[{idx}/{len(specs_list)}] {mode.title()} {spec['id']} ({spec['repo_id']})")

        try:
            ref_fn, ep, load_meta = _build_reference_and_exported_program(spec, cohort)

            if mode == "export":
                out_dir = out_root / spec["id"]
                _ensure_runtime_directory(out_dir)
                artifact_path = out_dir / f"model-torch{cohort}.pt2"
                if spec.get("strategy") == "reuse_existing_exported_program":
                    source_path = Path(spec["source_path"])
                    _copy_verified_source(
                        source_path,
                        artifact_path,
                        expected_sha256=_sha256(source_path),
                        expected_size=source_path.stat().st_size,
                    )
                else:
                    torch.export.save(ep, str(artifact_path))
                artifact_path.chmod(0o644)
            else:
                artifact_path = _resolve_artifact_path(artifacts_root, spec, cohort)
                if not artifact_path.exists():
                    raise RuntimeError(f"Artifact not found: {artifact_path}")

            ep_loaded = torch.export.load(str(artifact_path))
            exported_module = ep_loaded.module()
            validation = _validate_exported_module(
                spec,
                ref_fn=ref_fn,
                exported_module=exported_module,
                batch_sizes=batch_sizes,
                seeds=seeds,
                scales=scales,
                devices=validate_devices,
            )

            meta = {
                "schema_version": 2,
                "mode": mode,
                "model_id": spec["id"],
                "repo_id": spec["repo_id"],
                "torch_version": torch.__version__,
                "torch_minor": cohort,
                "runtime_torch_minor": runtime_torch_minor,
                "environment": environment,
                "exporter_arguments": exporter_arguments,
                "source_artifact": spec.get("source_artifact"),
                "source": load_meta,
                "artifact": artifact_path.name,
                "artifact_sha256": _sha256(artifact_path),
                "artifact_size_bytes": artifact_path.stat().st_size,
                "validation": validation,
            }

            meta_path = artifact_path.with_suffix(artifact_path.suffix + ".meta.json")
            _write_json_atomic(meta_path, meta)

            if validation["status"] != "ok":
                raise RuntimeError(
                    "Validation did not satisfy the complete release gate "
                    f"(max_abs={validation['worst_max_abs_diff_vs_reference']}, "
                    f"mean_abs={validation['worst_mean_abs_diff_vs_reference']}); "
                    f"see {meta_path}"
                )

            summary["results"].append(
                {
                    "model_id": spec["id"],
                    "repo_id": spec["repo_id"],
                    "status": "ok",
                    "artifact": str(artifact_path),
                    "meta": str(meta_path),
                    "sha256": meta["artifact_sha256"],
                    "size_bytes": meta["artifact_size_bytes"],
                    "meta_sha256": _sha256(meta_path),
                    "validation_status": validation["status"],
                    "max_abs_tolerance": validation["max_abs_tolerance"],
                    "mean_abs_tolerance": validation["mean_abs_tolerance"],
                    "worst_max_abs_diff": validation["worst_max_abs_diff_vs_reference"],
                    "worst_mean_abs_diff": validation["worst_mean_abs_diff_vs_reference"],
                    "worst_case_id": validation["worst_case_id"],
                    "num_cases": validation["num_cases"],
                }
            )
            print("  Done")
        except Exception as exc:
            failures.append((spec["id"], type(exc).__name__, str(exc)))
            summary["results"].append(
                {
                    "model_id": spec["id"],
                    "repo_id": spec["repo_id"],
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            print(f"  ERROR: {type(exc).__name__}: {exc}")

    if upload and not failures:
        message = (
            "Direct upload is disabled. Review the complete staging report, create "
            "a digest-bound approval, and use model_cohort_publication.py publish."
        )
        failures.append(("publication", "RuntimeError", message))
        summary["publication_status"] = "approval_required"

    summary["status"] = "failed" if failures else "ok"

    return summary, failures


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common_args(p):
        p.add_argument("--repo-root", default=".")
        p.add_argument("--batch-sizes", default="1,2,4,8")
        p.add_argument("--seeds", default="0,17")
        p.add_argument("--scales", default="1.0,0.25")
        p.add_argument(
            "--validate-devices",
            default="cpu",
            help="Comma-separated devices to validate on (e.g. cpu,cuda).",
        )
        p.add_argument(
            "--model-ids",
            default="",
            help="Optional comma-separated subset of model IDs to process.",
        )
        p.add_argument(
            "--environment-lock",
            default="uv.lock",
            help="Exact uv.lock used by this runtime, relative to --repo-root.",
        )

    p_prepare = sub.add_parser(
        "prepare-sources",
        help="Fetch pinned source references and derive local export inputs",
    )
    p_prepare.add_argument("--repo-root", default=".")
    p_prepare.add_argument(
        "--cohort",
        default=None,
        help="Artifact cohort to prepare. Default: current torch minor.",
    )
    p_prepare.add_argument("--model-ids", default="")
    p_prepare.add_argument(
        "--environment-lock",
        default="uv.lock",
        help="Exact uv.lock used by this runtime, relative to --repo-root.",
    )
    p_prepare.add_argument("--offline", action="store_true")
    p_prepare.add_argument("--force-download", action="store_true")
    p_prepare.add_argument(
        "--inventory",
        default=None,
        help="Output inventory path. Default: models_local/source-inventory-<cohort>.json",
    )

    p_export = sub.add_parser("export", help="Export and validate into local staging")
    add_common_args(p_export)
    p_export.add_argument("--out-root", default="/tmp/model-cohort-exports")
    p_export.add_argument(
        "--upload",
        action="store_true",
        help="Removed unsafe compatibility flag; use model_cohort_publication.py.",
    )
    p_export.add_argument(
        "--hf-token-env",
        default="HF_TOKEN",
        help="Deprecated compatibility option; export performs no network writes.",
    )

    p_validate = sub.add_parser("validate", help="Validate existing artifacts")
    add_common_args(p_validate)
    p_validate.add_argument("--artifacts-root", required=True)
    p_validate.add_argument(
        "--cohort",
        default=None,
        help="Explicit cohort (e.g. 2.3/2.6/2.11). Default: current torch minor.",
    )

    args = parser.parse_args()

    if args.command == "export" and args.upload:
        parser.error(
            "--upload is disabled: stage and review every model first, then use "
            "scripts/model_cohort_publication.py publish with a bound approval"
        )

    repo_root = Path(args.repo_root).resolve()
    torch_version = torch.__version__
    torch_minor = ".".join(torch_version.split("+")[0].split(".")[:2])

    cohort = args.cohort if getattr(args, "cohort", None) else torch_minor
    specs = _model_specs(cohort)
    if args.model_ids.strip():
        selected = set(x.strip() for x in args.model_ids.split(",") if x.strip())
        specs = [s for s in specs if s["id"] in selected]
        missing = sorted(selected - {s["id"] for s in specs})
        if missing:
            raise RuntimeError(f"Unknown model IDs: {', '.join(missing)}")

    if args.command == "prepare-sources":
        if args.offline and args.force_download:
            parser.error("--offline and --force-download cannot be combined")
        inventory = _prepare_model_sources(
            specs,
            repo_root=repo_root,
            cohort=cohort,
            offline=args.offline,
            force_download=args.force_download,
            environment_lock=args.environment_lock,
        )
        inventory_path = (
            Path(args.inventory).resolve()
            if args.inventory
            else repo_root / "models_local" / f"source-inventory-torch{cohort}.json"
        )
        _write_json_atomic(inventory_path, inventory)
        print(f"Prepared {len(specs)} model sources; inventory: {inventory_path}")
        return

    batch_sizes = _parse_csv_ints(args.batch_sizes)
    seeds = _parse_csv_ints(args.seeds)
    scales = _parse_csv_floats(args.scales)
    validate_devices = _parse_csv_strings(args.validate_devices)
    if not validate_devices:
        validate_devices = ["cpu"]

    if args.command == "export":
        out_root = Path(args.out_root).resolve()
        _ensure_runtime_directory(out_root)
        summary, failures = _run_for_specs(
            specs=specs,
            mode="export",
            repo_root=repo_root,
            cohort=cohort,
            out_root=out_root,
            artifacts_root=out_root,
            upload=args.upload,
            hf_token_env=args.hf_token_env,
            batch_sizes=batch_sizes,
            seeds=seeds,
            scales=scales,
            validate_devices=validate_devices,
            environment_lock=args.environment_lock,
        )
        summary_path = out_root / f"summary-torch{cohort}.json"
    else:
        artifacts_root = Path(args.artifacts_root).resolve()
        summary, failures = _run_for_specs(
            specs=specs,
            mode="validate",
            repo_root=repo_root,
            cohort=cohort,
            out_root=artifacts_root,
            artifacts_root=artifacts_root,
            upload=False,
            hf_token_env="HF_TOKEN",
            batch_sizes=batch_sizes,
            seeds=seeds,
            scales=scales,
            validate_devices=validate_devices,
            environment_lock=args.environment_lock,
        )
        summary_path = artifacts_root / f"validation-summary-torch{cohort}.json"

    _write_json_atomic(summary_path, summary)

    print(f"Summary written to {summary_path}")

    if failures:
        print("Failures:")
        for model_id, err_type, err in failures:
            print(f" - {model_id}: {err_type}: {err}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
