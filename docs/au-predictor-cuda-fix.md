# AU Predictor CUDA Fix — Progress & Plan

**Branch:** `fix/au-predictor-torchscript-compat`
**GitHub Issue:** [#85](https://github.com/tomas-gajarsky/facetorch/issues/85)

## Problem

The AU predictor (`open_graph_swin_base`) uses a TorchScript-exported Swin
Transformer model that **deadlocks on CUDA** with PyTorch >= 2.0 and
CUDA >= 12.0. The hang occurs on the second forward pass (any batch size)
or on the first forward pass with batch > 1.

- **CPU inference**: works perfectly (all batch sizes, all repeated calls)
- **CUDA first call, batch=1**: works
- **CUDA second call or batch > 1**: hangs indefinitely

Environment tested: PyTorch 2.3.1+cu121, CUDA 12.1

## Root Cause

The issue is in PyTorch's TorchScript JIT compiler, not in the model weights
or facetorch code. The Swin Transformer's window attention operations
(`torch.roll`, window partition/merge, relative position bias indexing)
trigger a CUDA deadlock in the JIT runtime on repeated execution.

## Approaches Tried

| Approach | Result |
|----------|--------|
| Original TorchScript `.pt` on CUDA | Hangs on 2nd call |
| Re-saved with current PyTorch (`torch.jit.save`) | Hangs on 2nd call |
| `torch.jit.freeze()` | Hangs on 2nd call |
| `torch.jit.optimize_for_inference()` | Hangs on 2nd call |
| `torch.jit.trace()` from native model | Hangs on 1st call |
| `torch.jit.script()` from native model | Fails — Swin source not scriptable |
| CUDA stream isolation | Hangs on 2nd call |
| `copy.deepcopy()` before each call | Hangs on 2nd call |
| Reload model from disk each call | Works, but 365MB reload per call |
| ONNX export + ONNX Runtime (CPU) | Works, adds dependency |
| **CPU-only inference** | **Works** (~0.12s/face) |
| **Native PyTorch nn.Module on CUDA** | **Works** (~0.007s/face) |

## Chosen Solution: Native PyTorch Model

Load the model as a native `nn.Module` instead of TorchScript. The same
`.pt` file is downloaded from HuggingFace — its `state_dict()` is extracted
and loaded into the native model class.

### Output Verification

- CPU output: **bitwise identical** (max diff = 0.0) vs original TorchScript
- CUDA native vs CPU TorchScript: max diff < 2e-7 (float32 precision)

### Performance

| Mode | Latency per face | Batch=13 |
|------|-----------------|----------|
| TorchScript CPU | ~0.12s | ~1.1s |
| Native PyTorch CUDA | ~0.007s | ~0.04s |
| Speedup | **17x** | **27x** |

## Changes Made

### New Files

- **`facetorch/analyzer/predictor/au_model.py`** — Self-contained native
  PyTorch implementation of the OpenGraphAU model (Swin Transformer backbone
  + GNN head). Architecture reconstructed from:
  - [CVI-SZU/ME-GraphAU](https://github.com/CVI-SZU/ME-GraphAU) (ANFL stage 1 GNN)
  - [lingjivoo/OpenGraphAU](https://github.com/lingjivoo/OpenGraphAU) (Head with main/sub AUs)
  - The Swin Transformer is inlined (adapted from Microsoft's MIT-licensed implementation)
  - Requires `timm` for `DropPath`, `to_2tuple`, `trunc_normal_` utilities

### Modified Files

- **`facetorch/base.py`** — `BaseModel` gains an optional `native_model_class`
  parameter. When set, `load_model()` extracts the state_dict from the
  TorchScript file and loads it into an instance of the specified class.
  Backward-compatible: all existing models continue to use TorchScript.

- **`facetorch/analyzer/predictor/core.py`** — `FacePredictor.__init__` passes
  `native_model_class` through to `BaseModel`.

- **`conf/analyzer/predictor/au/open_graph_swin_base.yaml`** — Adds
  `native_model_class: facetorch.analyzer.predictor.au_model.OpenGraphAU`
  and restores `device: ${analyzer.device}` (CUDA support).

- **`conf/analyzer/predictor/au/open_graph_swin_base_hf.yaml`** — Same changes.
- **`conf/analyzer/predictor/au/open_graph_swin_base_gdrive.yaml`** — Same changes.
- **`conf/merged/merged.config.yaml`** — AU device updated.
- **`conf/merged/gpu.merged.config.yaml`** — AU device updated.

## Completed

- [x] Update the HF and GDrive config variants with `native_model_class`
- [x] Update merged configs with `native_model_class`
- [x] Add `timm` to dependencies in `environment.yml` / `gpu.environment.yml`
- [x] Verify the FaceDetector (RetinaFace) is backward-compatible
      (`native_model_class=None` default, no change needed)
- [x] End-to-end test: `_load_native_model()` flow on CUDA passes all tests

## Things Left To Do

- [ ] Run the full test suite (requires Docker for `/opt/facetorch/` paths)
- [ ] Update README CUDA compatibility note (remove "Does not work with CUDA > 12.0")
- [ ] Consider uploading a standalone `state_dict.pth` to the HuggingFace repo
      (currently extracts state_dict from the TorchScript file at load time)

## Future: TorchScript to torch.compile Migration

PyTorch is deprecating TorchScript in favor of `torch.compile` /
`torch.export`. A broader migration of all facetorch models from TorchScript
to native PyTorch + `torch.compile` would permanently solve this class of
issues and provide better performance. This requires the original model
architectures for all predictors/detectors.
