#!/usr/bin/env python3
"""Export, validate, and upload versioned `.pt2` cohorts for facetorch models.

The script supports two workflows:

1) Export + validate + optional HF upload for the *current* torch runtime.
2) Validate previously exported artifacts against reference models.

Examples:
  # Export current torch cohort to /tmp and upload
  PYTHONPATH=. python scripts/export_model_cohorts_hf.py export \
    --repo-root . \
    --out-root /tmp/model-cohort-exports/upload \
    --upload \
    --hf-token-env HF_TOKEN

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
import os
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

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


def _import_class(class_path: str):
    module_path, class_name = class_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def _summarize_output(o: Any) -> Any:
    if isinstance(o, torch.Tensor):
        return {
            "shape": list(o.shape),
            "dtype": str(o.dtype),
        }
    if isinstance(o, (tuple, list)):
        return [_summarize_output(x) for x in o]
    return str(type(o))


def _clone_output_cpu(o: Any) -> Any:
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().clone()
    if isinstance(o, tuple):
        return tuple(_clone_output_cpu(x) for x in o)
    if isinstance(o, list):
        return [_clone_output_cpu(x) for x in o]
    return o


def _ensure_same_structure(a: Any, b: Any, path: str = "output") -> None:
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if tuple(a.shape) != tuple(b.shape):
            raise RuntimeError(
                f"Shape mismatch at {path}: {tuple(a.shape)} != {tuple(b.shape)}"
            )
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
    raise RuntimeError(f"Output type mismatch at {path}: {type(a)} vs {type(b)}")


def _accumulate_abs_diff(a: Any, b: Any, acc: Dict[str, float]) -> None:
    if isinstance(a, torch.Tensor):
        da = a.detach().cpu().float()
        db = b.detach().cpu().float()
        d = (da - db).abs()
        if d.numel() == 0:
            return
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

    raise RuntimeError(f"Unsupported output type for diff: {type(a)}")


def _compute_diff_stats(a: Any, b: Any) -> Dict[str, float]:
    _ensure_same_structure(a, b)
    acc = {"max_abs": 0.0, "sum_abs": 0.0, "numel": 0}
    _accumulate_abs_diff(a, b, acc)
    mean_abs = acc["sum_abs"] / acc["numel"] if acc["numel"] else 0.0
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


class _MagFaceIBasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample=None):
        super().__init__()
        self.bn1 = torch.nn.BatchNorm2d(inplanes)
        self.conv1 = torch.nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.prelu = torch.nn.PReLU(planes)
        self.conv2 = torch.nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn3 = torch.nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return out + identity


class _MagFaceIResNet100(torch.nn.Module):
    """IResNet-100 topology matching the MagFace TorchScript state_dict layout."""

    def __init__(self):
        super().__init__()
        self.inplanes = 64
        self.conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.prelu = torch.nn.PReLU(64)
        self.layer1 = self._make_layer(64, 3, stride=2)
        self.layer2 = self._make_layer(128, 13, stride=2)
        self.layer3 = self._make_layer(256, 30, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.bn2 = torch.nn.BatchNorm2d(512)
        self.fc = torch.nn.Linear(512 * 7 * 7, 512)
        self.features = torch.nn.BatchNorm1d(512)

    def _make_layer(self, planes: int, blocks: int, stride: int):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(
                    self.inplanes, planes, kernel_size=1, stride=stride, bias=False
                ),
                torch.nn.BatchNorm2d(planes),
            )

        layers = [_MagFaceIBasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(_MagFaceIBasicBlock(self.inplanes, planes, 1, None))
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.features(x)
        return x


def _load_native_model(spec: Dict[str, Any]):
    strategy = spec["strategy"]
    source_path = spec["source_path"]
    if strategy == "native_magface_iresnet_from_torchscript_state_dict":
        model = _MagFaceIResNet100()
    else:
        cls = _import_class(spec["class_path"])
        model = cls()

    load_meta: Dict[str, Any] = {
        "strategy": strategy,
        "source_path": source_path,
        "missing_keys": [],
        "unexpected_keys": [],
    }

    if strategy == "native_state_dict":
        sd = _torch_load(source_path)
        strict = spec.get("strict", True)
        result = model.load_state_dict(sd, strict=strict)
        load_meta["missing_keys"] = list(getattr(result, "missing_keys", []))
        load_meta["unexpected_keys"] = list(getattr(result, "unexpected_keys", []))
    elif strategy == "native_from_torchscript_state_dict":
        ts = torch.jit.load(source_path, map_location="cpu")
        sd = dict(ts.state_dict())
        strict = spec.get("strict", True)
        result = model.load_state_dict(sd, strict=strict)
        load_meta["missing_keys"] = list(getattr(result, "missing_keys", []))
        load_meta["unexpected_keys"] = list(getattr(result, "unexpected_keys", []))
    elif strategy == "native_from_torchscript_constants":
        ts = torch.jit.load(source_path, map_location="cpu")
        if not hasattr(model, "load_from_torchscript"):
            raise RuntimeError(f"Model {spec['class_path']} has no load_from_torchscript")
        model.load_from_torchscript(ts)
    elif strategy == "native_magface_iresnet_from_torchscript_state_dict":
        ts = torch.jit.load(source_path, map_location="cpu")
        sd = dict(ts.state_dict())
        strict = spec.get("strict", True)
        result = model.load_state_dict(sd, strict=strict)
        load_meta["missing_keys"] = list(getattr(result, "missing_keys", []))
        load_meta["unexpected_keys"] = list(getattr(result, "unexpected_keys", []))
    else:
        raise RuntimeError(f"Unsupported native strategy: {strategy}")

    model.eval()
    return model, load_meta


def _build_reference_and_exported_program(spec: Dict[str, Any], torch_minor: str):
    strategy = spec["strategy"]
    input_shape = spec["input_shape"]
    dummy = torch.randn(*input_shape)

    if strategy in {
        "native_state_dict",
        "native_from_torchscript_state_dict",
        "native_from_torchscript_constants",
        "native_magface_iresnet_from_torchscript_state_dict",
    }:
        model, load_meta = _load_native_model(spec)

        def ref_fn(x):
            with torch.no_grad():
                return model(x)

        ep = torch.export.export(
            model,
            (dummy,),
            dynamic_shapes=_dynamic_shapes(spec),
        )
        return ref_fn, ep, load_meta

    if strategy == "ts2ep_reexport_dynamic":
        from torch._export.converter import TS2EPConverter

        ts = torch.jit.load(spec["source_path"], map_location="cpu")
        ts.eval()

        def ref_fn(x):
            with torch.no_grad():
                return ts(x)

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
        return ref_fn, ep, load_meta

    if strategy == "reuse_existing_exported_program":
        ep = torch.export.load(spec["source_path"])
        mod = ep.module()

        def ref_fn(x):
            with torch.no_grad():
                return mod(x)

        load_meta = {
            "strategy": strategy,
            "source_path": spec["source_path"],
            "missing_keys": [],
            "unexpected_keys": [],
        }
        return ref_fn, ep, load_meta

    raise RuntimeError(f"Unknown strategy: {strategy}")


def _model_specs(torch_minor: str) -> List[Dict[str, Any]]:
    return [
        {
            "id": "detector-retinaface",
            "repo_id": "tomas-gajarsky/facetorch-detector-retinaface",
            "class_path": "model_defs.detector_model.RetinaFaceResNet50",
            "strategy": (
                "reuse_existing_exported_program"
                if torch_minor == "2.3"
                else "native_state_dict"
            ),
            "source_path": (
                "models/exported/detector/1/model.pt2"
                if torch_minor == "2.3"
                else "models_local/state_dicts/detector.pth"
            ),
            "strict": True,
            "input_shape": [1, 3, 480, 640],
            "dynamic_batch": False,
            "dynamic_hw": False if torch_minor == "2.3" else True,
            "dynamic_hw_multiple": 4,
            "batch_max": 32,
            "validation_batch_sizes": [1],
        },
        {
            "id": "fer-efficientnet-b2",
            "repo_id": "tomas-gajarsky/facetorch-fer-efficientnet-b2",
            "class_path": "model_defs.fer_model.EfficientNetB2FER",
            "strategy": "native_state_dict",
            "source_path": "models_local/state_dicts/fer.pth",
            "strict": True,
            "input_shape": [2, 3, 260, 260],
        },
        {
            "id": "fer-efficientnet-b0",
            "repo_id": "tomas-gajarsky/facetorch-fer-efficientnet-b0",
            "class_path": "model_defs.fer_model.EfficientNetB0FER",
            "strategy": "native_from_torchscript_state_dict",
            "source_path": "models_local/fer_b0/model.pt",
            "strict": True,
            "input_shape": [2, 3, 244, 244],
        },
        {
            "id": "au-opengraph",
            "repo_id": "tomas-gajarsky/facetorch-au-opengraph",
            "class_path": "model_defs.au_model.OpenGraphAU",
            "strategy": "native_state_dict",
            "source_path": "models_local/state_dicts/au.pth",
            "strict": False,
            "input_shape": [2, 3, 224, 224],
        },
        {
            "id": "va-elim",
            "repo_id": "tomas-gajarsky/facetorch-va-elim",
            "class_path": "model_defs.va_model.ELIMALAlexNet",
            "strategy": "native_state_dict",
            "source_path": "models_local/state_dicts/va.pth",
            "strict": True,
            "input_shape": [2, 3, 224, 224],
        },
        {
            "id": "embed-resnet-50",
            "repo_id": "tomas-gajarsky/facetorch-embed-resnet-50",
            "class_path": "model_defs.embed_model.EmbedResNet50",
            "strategy": "native_state_dict",
            "source_path": "models_local/state_dicts/embed.pth",
            "strict": True,
            "input_shape": [2, 3, 244, 244],
        },
        {
            "id": "deepfake-efficientnet-b7",
            "repo_id": "tomas-gajarsky/facetorch-deepfake-efficientnet-b7",
            "class_path": "model_defs.deepfake_model.DeepfakeEfficientNetB7",
            "strategy": "native_state_dict",
            "source_path": "models_local/state_dicts/deepfake.pth",
            "strict": True,
            "input_shape": [2, 3, 380, 380],
        },
        {
            "id": "align-synergynet",
            "repo_id": "tomas-gajarsky/facetorch-align-synergynet",
            "class_path": "model_defs.align_model.SynergyNetMobileNetV2",
            "strategy": "native_state_dict",
            "source_path": "models_local/state_dicts/align.pth",
            "strict": True,
            "input_shape": [2, 3, 120, 120],
        },
        {
            "id": "verify-magface",
            "repo_id": "tomas-gajarsky/facetorch-verify-magface",
            "class_path": "model_defs.verify_model.VerifyIResNet100",
            "strategy": "native_magface_iresnet_from_torchscript_state_dict",
            "source_path": "models/torchscript/predictor/verify/1/model.pt",
            "strict": True,
            "input_shape": [2, 3, 112, 112],
        },
        {
            "id": "verify-adaface",
            "repo_id": "tomas-gajarsky/facetorch-verify-adaface",
            "class_path": "model_defs.verify_model.VerifyIResNet100",
            "strategy": "native_from_torchscript_constants",
            "source_path": "models/torchscript/predictor/verify/2/model.pt",
            "strict": False,
            "input_shape": [2, 3, 112, 112],
        },
    ]


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

    for b in effective_batch_sizes:
        if b < 1 or b > max_batch:
            continue
        for seed in seeds:
            for scale in scales:
                case_seed = int(seed * 100_003 + b * 313 + round(scale * 1000.0) * 17)

                g1 = torch.Generator(device="cpu")
                g1.manual_seed(case_seed)
                x_randn = torch.randn((b, c, h, w), generator=g1) * scale

                g2 = torch.Generator(device="cpu")
                g2.manual_seed(case_seed + 7)
                x_randu = (torch.rand((b, c, h, w), generator=g2) * 2.0 - 1.0) * scale

                cases.append(
                    {
                        "id": f"b{b}_seed{seed}_scale{scale}_randn",
                        "batch": b,
                        "seed": seed,
                        "scale": scale,
                        "variant": "randn",
                        "x": x_randn,
                    }
                )
                cases.append(
                    {
                        "id": f"b{b}_seed{seed}_scale{scale}_randu",
                        "batch": b,
                        "seed": seed,
                        "scale": scale,
                        "variant": "randu",
                        "x": x_randu,
                    }
                )

    return cases


def _validate_exported_module(
    spec: Dict[str, Any],
    ref_fn,
    exported_module,
    batch_sizes: Sequence[int],
    seeds: Sequence[int],
    scales: Sequence[float],
) -> Dict[str, Any]:
    cases = _build_validation_cases(spec, batch_sizes, seeds, scales)

    case_results = []
    worst_max_abs = 0.0
    worst_case_id = None

    for case in cases:
        x = case["x"]
        with torch.no_grad():
            ref_out = _clone_output_cpu(ref_fn(x))
            exp_out = _clone_output_cpu(exported_module(x))

        stats = _compute_diff_stats(ref_out, exp_out)
        if worst_case_id is None or stats["max_abs"] > worst_max_abs:
            worst_max_abs = stats["max_abs"]
            worst_case_id = case["id"]

        case_results.append(
            {
                "case_id": case["id"],
                "batch": case["batch"],
                "seed": case["seed"],
                "scale": case["scale"],
                "variant": case["variant"],
                "input_shape": list(x.shape),
                "output_summary": _summarize_output(exp_out),
                "max_abs_diff_vs_reference": stats["max_abs"],
                "mean_abs_diff_vs_reference": stats["mean_abs"],
                "numel_compared": stats["numel"],
            }
        )

    return {
        "num_cases": len(case_results),
        "worst_case_id": worst_case_id,
        "worst_max_abs_diff_vs_reference": worst_max_abs,
        "cases": case_results,
    }


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
) -> Dict[str, Any]:
    if mode not in {"export", "validate"}:
        raise RuntimeError(f"Unsupported mode: {mode}")

    api = None
    if upload:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise RuntimeError("huggingface_hub is required when --upload is enabled") from exc

        token = os.getenv(hf_token_env)
        if not token:
            raise RuntimeError(f"{hf_token_env} is not set")
        api = HfApi(token=token)

    summary: Dict[str, Any] = {
        "generated_at_utc": _now_iso(),
        "mode": mode,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_minor": cohort,
        "batch_sizes": list(batch_sizes),
        "seeds": list(seeds),
        "scales": list(scales),
        "results": [],
    }

    failures = []
    specs_list = list(specs)

    for idx, spec in enumerate(specs_list, start=1):
        os.chdir(repo_root)
        print(f"[{idx}/{len(specs_list)}] {mode.title()} {spec['id']} ({spec['repo_id']})")

        try:
            ref_fn, ep, load_meta = _build_reference_and_exported_program(spec, cohort)

            if mode == "export":
                out_dir = out_root / spec["id"]
                out_dir.mkdir(parents=True, exist_ok=True)
                artifact_path = out_dir / f"model-torch{cohort}.pt2"
                torch.export.save(ep, str(artifact_path))
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
            )

            meta = {
                "generated_at_utc": _now_iso(),
                "mode": mode,
                "model_id": spec["id"],
                "repo_id": spec["repo_id"],
                "torch_version": torch.__version__,
                "torch_minor": cohort,
                "source": load_meta,
                "artifact": str(artifact_path),
                "artifact_sha256": _sha256(artifact_path),
                "validation": validation,
            }

            meta_path = artifact_path.with_suffix(artifact_path.suffix + ".meta.json")
            with meta_path.open("w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            if api is not None:
                print(f"  Uploading {artifact_path.name} ...")
                api.upload_file(
                    path_or_fileobj=str(artifact_path),
                    path_in_repo=artifact_path.name,
                    repo_id=spec["repo_id"],
                    commit_message=f"Add {artifact_path.name} exported with torch {cohort}",
                )
                print(f"  Uploading {meta_path.name} ...")
                api.upload_file(
                    path_or_fileobj=str(meta_path),
                    path_in_repo=meta_path.name,
                    repo_id=spec["repo_id"],
                    commit_message=f"Add metadata for {artifact_path.name}",
                )

            summary["results"].append(
                {
                    "model_id": spec["id"],
                    "repo_id": spec["repo_id"],
                    "status": "ok",
                    "artifact": str(artifact_path),
                    "meta": str(meta_path),
                    "sha256": meta["artifact_sha256"],
                    "worst_max_abs_diff": validation["worst_max_abs_diff_vs_reference"],
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
            "--model-ids",
            default="",
            help="Optional comma-separated subset of model IDs to process.",
        )

    p_export = sub.add_parser("export", help="Export, validate, and optionally upload")
    add_common_args(p_export)
    p_export.add_argument("--out-root", default="/tmp/model-cohort-exports")
    p_export.add_argument("--upload", action="store_true")
    p_export.add_argument("--hf-token-env", default="HF_TOKEN")

    p_validate = sub.add_parser("validate", help="Validate existing artifacts")
    add_common_args(p_validate)
    p_validate.add_argument("--artifacts-root", required=True)
    p_validate.add_argument(
        "--cohort",
        default=None,
        help="Explicit cohort (e.g. 2.3/2.6/2.11). Default: current torch minor.",
    )

    args = parser.parse_args()

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

    batch_sizes = _parse_csv_ints(args.batch_sizes)
    seeds = _parse_csv_ints(args.seeds)
    scales = _parse_csv_floats(args.scales)

    if args.command == "export":
        out_root = Path(args.out_root).resolve()
        out_root.mkdir(parents=True, exist_ok=True)
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
        )
        summary_path = artifacts_root / f"validation-summary-torch{cohort}.json"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary written to {summary_path}")

    if failures:
        print("Failures:")
        for model_id, err_type, err in failures:
            print(f" - {model_id}: {err_type}: {err}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
