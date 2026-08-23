# Model Architecture Definitions

Native PyTorch model definitions used to export `.pt2` files via `torch.export`.

These files are **not required at inference time** — the `.pt2` files are self-contained
and can be loaded directly with `torch.export.load()`. They are kept here for:

- **Re-export**: producing new `.pt2` files (e.g., for a new PyTorch version or different dynamic shapes)
- **Reference**: documenting the exact architecture each `.pt2` file contains

## Models

| File | Class | Source | Task |
|------|-------|--------|------|
| `align_model.py` | `SynergyNetMobileNetV2` | [choyingw/SynergyNet](https://github.com/choyingw/SynergyNet) | 3D face alignment |
| `au_model.py` | `OpenGraphAU` | [lingjivoo/OpenGraphAU](https://github.com/lingjivoo/OpenGraphAU) | Action unit detection |
| `deepfake_model.py` | `DeepfakeEfficientNetB7` | [selimsef/dfdc_deepfake_challenge](https://github.com/selimsef/dfdc_deepfake_challenge) | Deepfake detection |
| `detector_model.py` | `RetinaFaceResNet50` | [biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface) | Face detection |
| `embed_model.py` | `EmbedResNet50` | [1adrianb/unsupervised-face-representation](https://github.com/1adrianb/unsupervised-face-representation), [cydonia999/VGGFace2-pytorch](https://github.com/cydonia999/VGGFace2-pytorch) | Face embeddings |
| `fer_model.py` | `EfficientNetB2FER`, `EfficientNetB0FER` | [sb-ai-lab/EmotiEffLib](https://github.com/sb-ai-lab/EmotiEffLib) | Facial expression recognition |
| `va_model.py` | `ELIMALAlexNet` | [kdhht2334/ELIM_FER](https://github.com/kdhht2334/ELIM_FER) | Valence-arousal prediction |
| `verify_model.py` | `MagFaceIResNet100`, `VerifyIResNet100` | [junuke/UNPG](https://github.com/junuke/UNPG), [IrvingMeng/MagFace](https://github.com/IrvingMeng/MagFace), [mk-minchul/AdaFace](https://github.com/mk-minchul/AdaFace) | Face verification |

## Re-exporting

```python
import torch
from model_defs.au_model import OpenGraphAU

model = OpenGraphAU()
model.load_state_dict(torch.load("weights.pth", map_location="cpu", weights_only=True))
model.eval()

batch = torch.export.Dim("batch", min=1, max=64)
ep = torch.export.export(model, (torch.randn(2, 3, 224, 224),), dynamic_shapes={"x": {0: batch}})
torch.export.save(ep, "model.pt2")
```

Release cohorts must use `scripts/export_model_cohorts_hf.py` so the independent
legacy reference, finite-output checks, declared output schema, required dynamic
cases, and requested devices are all enforced. That script never uploads.
Reviewed candidates are published separately with
`scripts/model_cohort_publication.py`; see `docs/model-publication.md`.

The export script's `prepare-sources` command reconstructs these inputs only after
the hosted legacy object matches the immutable revision, size, and SHA-256 in the
packaged manifest. Architecture reproducibility does not establish checkpoint
redistribution rights; those remain a separate governance gate.

The detector's old TorchScript archive omits BatchNorm tensors from
`state_dict()` but retains them as module attributes. The cohort exporter recovers
those exact tensors, checks shape and dtype, and then performs a strict native
load; it never substitutes initialized values. Detector height and width are
dynamic multiples of 32, matching runtime preprocessing.

The original OpenGraphAU Swin-Base checkpoint mapping is verified by exact tensor
equality. Its state still does not reproduce the legacy model's one-face behavior
when loaded into the current timm forward definition, so release tooling preserves
each immutable, digest-verified published AU program and validates batched face
outputs against concatenated one-face golden calls. The generic snippet above does
not apply to this controlled program-reuse path.
