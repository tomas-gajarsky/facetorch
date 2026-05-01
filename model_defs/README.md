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
| `embed_model.py` | `EmbedResNet50` | [cydonia999/VGGFace2-pytorch](https://github.com/cydonia999/VGGFace2-pytorch) | Face embeddings |
| `fer_model.py` | `EfficientNetB2FER`, `EfficientNetB0FER` | [HSE-asavchenko/face-emotion-recognition](https://github.com/HSE-asavchenko/face-emotion-recognition) | Facial expression recognition |
| `va_model.py` | `ELIMALAlexNet` | [kdhht2334/ELIM-AL](https://github.com/kdhht2334/ELIM-AL) | Valence-arousal prediction |
| `verify_model.py` | `VerifyIResNet100` | [IrvingMeng/MagFace](https://github.com/IrvingMeng/MagFace), [mk-minchul/AdaFace](https://github.com/mk-minchul/AdaFace) | Face verification |

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
