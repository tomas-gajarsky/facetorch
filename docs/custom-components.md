# Extending facetorch with custom predictors and detectors

Facetorch deliberately separates application extensions from models shipped as
part of facetorch. A private predictor or detector does **not** have to be added
to facetorch's packaged model manifest. The stricter v1 manifest protects the
built-in model names and release artifacts; it is not a plugin allow-list.

This guide covers three supported extension paths:

| Path | Packaged manifest entry | Best fit |
| --- | --- | --- |
| Install an already constructed component | No | Application code, experiments, and models managed by another system |
| Configure a digest-pinned external artifact | No | A reusable private or third-party Hugging Face model |
| Contribute a model to facetorch's defaults | Yes | A model maintained, documented, and released by facetorch |

Choose the first path for the smallest integration surface. Choose external
Hydra configuration when the component graph must be reproducible as data.
Choose the contribution path only when the model should become part of the
official facetorch release contract.

## What the v1 artifact binding means

`DownloaderHuggingFace` has two distinct modes.

In **packaged-manifest mode**, `manifest_id` is present. The identifier must
exist in `facetorch/models/manifest.json`. The configured repository and
revision must agree with that manifest, and the runtime selects exactly one
artifact declared for the active Torch and device cohort. Size, SHA-256, file
format, compatibility evidence, provenance, and model-rights records are part
of the facetorch release gate. This is the mode used by built-in models.

In **direct external mode**, `manifest_id` is omitted. The configuration itself
must provide the Hugging Face repository, real filename, immutable 40-character
commit, SHA-256, byte size, and expected format. Facetorch verifies those values
before trusting the cache, but it does not require the external model to appear
in facetorch's packaged governance records.

These modes must not be mixed. Do not invent a `manifest_id` for a private model,
and do not duplicate a built-in model's filename or digest in its YAML. An
unknown manifest ID, a mutable branch or tag, a mismatched built-in repository,
or a missing direct-artifact digest fails closed before inference.

Direct external mode describes one artifact for one configured device. It does
not automatically provide facetorch's Torch 2.6/2.11 cohort routing. Test the
artifact on the exact Torch major/minor used by the application. If both
supported Torch cohorts are required, publish and select separately validated
artifacts through separate application configs, or implement an application
manifest with equally explicit routing.

## Fastest path: install components at runtime

`FaceAnalyzer.detector` and the mutable `FaceAnalyzer.predictors` mapping accept
already constructed objects. This path does not involve a model downloader or
the packaged manifest. It is useful when the application already owns model
loading, integrity checks, and lifecycle management.

The following complete example replaces the detector and adds one predictor. It
runs offline and does not construct or download a built-in model.

<!-- facetorch-extension-runtime-example:start -->
```python
import torch

from facetorch import (
    Dimensions,
    Face,
    FaceAnalyzer,
    Location,
    Prediction,
    load_config,
)


class WholeImageDetector:
    """Treat one already-cropped input as the detected face."""

    def run(self, data):
        height, width = map(int, data.tensor.shape[-2:])
        data.faces = [
            Face(
                indx=0,
                loc=Location(x1=0, y1=0, x2=width, y2=height),
                dims=Dimensions(height=height, width=width),
                tensor=data.tensor[0],
                ratio=1.0,
            )
        ]
        return data


class MeanIntensityPredictor:
    max_batch_size = 16

    def run(self, faces):
        scores = faces.float().mean(dim=(1, 2, 3))
        return [
            Prediction(
                label="bright" if float(score) >= 0.5 else "dark",
                logits=score.detach().reshape(1),
            )
            for score in scores
        ]


analyzer = FaceAnalyzer(load_config(offline=True).analyzer)
analyzer.detector = WholeImageDetector()
analyzer.predictors["mean_intensity"] = MeanIntensityPredictor()

result = analyzer.run(
    torch.zeros((3, 32, 32), dtype=torch.uint8),
    include_predictors=["mean_intensity"],
)

assert result.faces[0].preds["mean_intensity"].label == "dark"
assert analyzer.detector_loaded
assert "mean_intensity" in analyzer.loaded_predictors
```
<!-- facetorch-extension-runtime-example:end -->

Assigning an existing detector or predictor name replaces that configured
component with the supplied object. Assigning a new predictor name adds it and
leaves every built-in predictor configured. Installing a custom predictor does
not deselect the defaults: pass `include_predictors=[...]` or
`exclude_predictors=[...]` on `run()` to prevent the unwanted built-ins from
being constructed and downloaded. Runtime installation is not lazy with respect
to construction already performed by the application.

### Custom predictor contract

A predictor installed in `analyzer.predictors[name]` must:

- expose `run(faces)`, where `faces` is one `BCHW` tensor produced by the
  configured face unifier;
- return a sized sequence containing exactly one `facetorch.Prediction` for
  every input face, in the same order;
- keep tensors on compatible devices or move them explicitly; and
- expose `max_batch_size` as a positive integer or `None` when it needs a limit.

The analyzer treats `face_batch_size` as a caller upper bound and splits work at
the smaller of that value and `max_batch_size`. A missing `max_batch_size` is
treated like `None`. A bad limit, a result without a length, or the wrong number
of predictions raises an actionable public error.

Subclasses of `FacePredictor`, `BasePredPreProcessor`, and
`BasePredPostProcessor` remain useful when their standard model pipeline fits,
but inheritance is not required for an object installed directly in the
predictor mapping. Custom components must declare their constructor parameters;
the v1 wrappers no longer accept arbitrary extra YAML attributes.

### Custom detector contract

A detector installed through `analyzer.detector` must expose
`run(data: ImageData) -> ImageData`. A fully custom detector owns the following
responsibilities:

- preserve or deliberately restore the canonical source image;
- place public boxes, landmarks, and face locations in original-image
  coordinates;
- clamp geometry to the source dimensions;
- populate `data.faces` with correctly indexed crops; and
- return an empty face list, rather than a special value, when nothing is found.

For stronger guardrails, build on `FaceDetector` with a custom preprocessor and
postprocessor instead of replacing the entire detector. Its public
`DetectorPostprocessorProtocol` requires `run(data, logits) -> ImageData` and is
available from:

```python
from facetorch.analyzer.detector import DetectorPostprocessorProtocol
```

`FaceDetector` retains the source tensor, clamps public geometry, and recrops
faces produced by a custom postprocessor. With facetorch's
`DetectorPreProcessor`, it also maps resized detector coordinates back to the
source. A custom preprocessor used inside `FaceDetector` should preserve spatial
dimensions; otherwise use a fully custom detector that owns an explicit public
coordinate transform. Facetorch's private resize-scale handoff is not a stable
extension API. A postprocessor may alternatively expose `extract_faces(data)`;
that hook is called after the source tensor and any configured coordinate scale
have been restored.

A fully custom detector may leave `data.det` empty when only face crops are
needed. Analysis still succeeds, but `include_tensors=True` returns empty public
detection tensors and the `draw_boxes` and `draw_landmarks` utilizers have no
geometry to draw.

A custom detector preprocessor is assumed capable of mutating its input, so the
wrapper keeps a defensive source copy by default. Do not set
`preserves_input_tensor = True` unless the implementation itself isolates every
in-place transform from the caller's tensor.

## Reusable configuration with an external Hydra tree

Use `load_config_from_path()` when an application wants its custom graph in
version-controlled YAML:

```python
from facetorch import FaceAnalyzer, load_config_from_path

cfg = load_config_from_path(
    "/srv/my-application/facetorch-config/config.yaml",
)
analyzer = FaceAnalyzer(cfg.analyzer)
```

The parent of `config.yaml` becomes Hydra's configuration root. Its `defaults`
list and sibling groups compose normally. Relative paths are resolved from the
caller's working directory before composition. Use `load_config_from_path()`;
plain `OmegaConf.load()` does not compose Hydra defaults.

The `profile=`, `offline=`, and `allow_legacy_models=` arguments become Hydra
overrides. They therefore require an external root config that already defines
`analyzer.device`, `offline`, and `allow_legacy_models`, respectively. Omit an
argument when its key is absent, or add the key explicitly through an override
such as `overrides=["+analyzer.device=cpu"]`.

Every `_target_` must be importable in the installed environment. Keep custom
Python classes in the application package rather than in the configuration
directory, and test the configuration against an installed facetorch wheel—not
only against a source checkout. The repository `conf/` tree is a structural
template; application extensions do not have to edit or fork that tree.

### Configure a direct immutable Hugging Face predictor

The fragment below belongs in the application's predictor configuration group.
It assumes the root config defines `offline`, `analyzer.device`, and
`analyzer.optimize_transforms`, as facetorch's root config does. Replace every
placeholder with values for the exact uploaded bytes.

<!-- facetorch-direct-artifact-yaml:start -->
```yaml
_target_: facetorch.analyzer.predictor.FacePredictor
max_batch_size: 32

downloader:
  _target_: facetorch.downloader.DownloaderHuggingFace
  file_id: my-organization/my-model
  repo_id: my-organization/my-model
  filename: model-torch2.11.pt2
  revision: "0000000000000000000000000000000000000000"
  sha256: "0000000000000000000000000000000000000000000000000000000000000000"
  size_bytes: 12345678
  expected_format: pt2
  path_local: ${facetorch.model_dir:}/external/my_predictor/model-torch2.11.pt2
  offline: ${offline}
  device: ${analyzer.device}

device:
  _target_: torch.device
  type: ${analyzer.device}

preprocessor:
  _target_: facetorch.analyzer.predictor.pre.PredictorPreProcessor
  transform:
    _target_: torchvision.transforms.Compose
    transforms:
      - _target_: torchvision.transforms.Resize
        size: [224, 224]
        antialias: true
      - _target_: torchvision.transforms.Normalize
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
  device:
    _target_: torch.device
    type: ${analyzer.device}
  optimize_transform: ${analyzer.optimize_transforms}
  reverse_colors: false

postprocessor:
  _target_: facetorch.analyzer.predictor.post.PostArgMax
  transform: null
  device:
    _target_: torch.device
    type: ${analyzer.device}
  optimize_transform: ${analyzer.optimize_transforms}
  labels: [class_a, class_b]
  dim: 1
```
<!-- facetorch-direct-artifact-yaml:end -->

`file_id` is retained for downloader API compatibility and should equal
`repo_id` in a new Hugging Face configuration. Some shipped configurations keep
legacy Google Drive IDs in this inert field for source compatibility; do not
copy those values into a new Hugging Face config. There is intentionally no
`manifest_id` in this direct external configuration. `path_local` selects the
cache directory; its basename is ignored and the artifact is always stored under
the authenticated `filename`.

The `revision` must be the Hub commit produced after upload, not `main`, a branch,
or a tag. `filename`, `sha256`, and `size_bytes` must all describe that commit's
same file. On the officially supported Linux platform, obtain local values with:

```bash
sha256sum model-torch2.11.pt2
stat --format='%s' model-torch2.11.pt2
```

`max_batch_size` must not exceed the largest batch accepted by the exported
artifact. It is an execution limit, not metadata inferred from the model file.

The cache is verified on every use by default. A missing file is downloaded, or
fails clearly in offline mode. An existing cache entry that is truncated,
differently formatted, or digest-mismatched is quarantined and is not executed.
Keep `verify_on_use` enabled for remotely sourced artifacts.

Use the same downloader block under `FaceDetector` for a custom detector model,
then supply preprocessors and postprocessors that implement the detector
contracts above. The model's input and output structure must match those
processors; facetorch cannot infer semantic output schemas from arbitrary model
bytes.

### Local and independently managed models

An application may implement `BaseDownloader.run()` and point `path_local` at a
model it manages itself. `BaseModel` can load an exported `.pt2`, a legacy
TorchScript `.pt`, a raw `.pth` state dictionary when `native_model_class` names
the matching `torch.nn.Module`, or a state dictionary extracted from a
TorchScript `.pt` module. A raw state dictionary saved with a `.pt` suffix is not
supported. This path bypasses the built-in remote integrity policy when
`verify_on_use` is false, so the application must authenticate the source and
verify bytes before constructing the predictor or detector.

Direct Hugging Face mode supports authenticated `.pt2` and verified TorchScript
`.pt` artifacts only. It cannot carry an executable `.pth` state dictionary;
state dictionaries must use the application's own downloader and integrity
checks.

Legacy `.pt` selection is explicit. Direct remote TorchScript requires
`allow_legacy_models=True` and emits `LegacyModelWarning`; `.pt2` is the v1
default for new extensions.

## Export and compatibility guidance

`torch.export` artifacts are coupled to PyTorch's exported-program schema. Export
from the same Torch major/minor that the application will run, and validate on
every claimed device. Dynamic dimensions must match the model's actual forward
signature; in a named `dynamic_shapes` mapping, the key such as `x` must equal
the corresponding forward parameter name.

A minimal single-input export can look like this:

```python
import torch

model.eval()
batch = torch.export.Dim("batch", min=1, max=64)
exported = torch.export.export(
    model,
    (dummy_input,),
    dynamic_shapes=({0: batch},),
)
torch.export.save(exported, "model.pt2")
```

Compare the exported program with the original model over representative data,
edge inputs, and batch sizes `1` and the declared maximum. Check finite outputs,
shape and dtype, task-specific invariants, CPU behavior, and CUDA behavior when
CUDA is claimed. A successful load is not evidence of numerical equivalence.

The repository's `scripts/export_model_cohorts_hf.py` is intentionally specific
to model specifications registered in the facetorch source tree. It is not a
generic exporter for an arbitrary user model. Application developers may borrow
its validation ideas, but `--model-ids` selects only repository-registered IDs.

## Contributing an officially shipped model

Adding a model to facetorch's defaults is intentionally more demanding than
using one in an application. Start with a fork and treat the model as a new
release asset:

1. Add an export-only architecture definition under `model_defs/` and register a
   complete model specification in `scripts/export_model_cohorts_hf.py`.
2. Prepare the pinned source checkpoint and validate `.pt2` artifacts for every
   supported Torch/device cohort claimed by the release. Use the script's
   `validate` subcommand with explicit `--cohort`, `--model-ids`, batch sizes,
   seeds, and scales to revalidate existing artifacts.
3. Add immutable artifact records to `facetorch/models/manifest.json`, including
   real filenames, Hub commits, byte sizes, SHA-256 values, runtime bounds,
   devices, validation metadata, export provenance, and license references.
4. Add the corresponding source-checkpoint, rights, attribution, intended-use,
   and limitations decision to `facetorch/models/governance.json`; update the
   compatibility record when the supported matrix changes.
5. Add matching detector or predictor configuration under both `conf/` and
   `facetorch/configs/`. Implement or select preprocessors and postprocessors
   that match the artifact's real input and output contract.
6. Add numerical, batch, device, configuration, cache, and failure-path tests.
   Wire default-cohort coverage through `tests/conftest.py` and the matching
   `conf/tests.config.N.yaml` files. Update the public model table, model card
   material, and changelog.
7. Follow the [model publication runbook](model-publication.md). The export
   command stages evidence but never uploads; reviewed publication is a separate,
   digest-bound transaction.
8. Run the repository lint contract with
   `uv run --frozen --extra dev flake8 --config=.flake8` and the relevant test
   cohorts before requesting review.

For example, revalidate one registered model without exporting or uploading:

```bash
PYTHONPATH=. python scripts/export_model_cohorts_hf.py validate \
  --repo-root . \
  --artifacts-root /secure/staging/torch-2.11 \
  --cohort 2.11 \
  --model-ids MODEL_ID \
  --batch-sizes 1,2,4,8 \
  --seeds 0,17 \
  --scales 1.0,0.25 \
  --validate-devices cpu,cuda
```

Do not copy direct external metadata into a packaged built-in YAML. Once a model
is shipped by facetorch, the packaged manifest is the sole artifact identity
source and the YAML contains only its `manifest_id`, expected repository, and
optional expected revision.

## Extension test checklist

Before deploying a custom component, cover at least:

- an installed-wheel configuration load from a working directory outside the
  facetorch repository;
- the exact Python, Torch, device, input size, dtype, and value range claimed;
- predictor batches of `1` and the declared `max_batch_size`;
- exactly one `Prediction` per input face, including tuple model outputs;
- detector no-face, border-face, oversized-image, padding, coordinate, and crop
  behavior;
- malformed, truncated, wrong-format, and wrong-digest cache entries;
- offline startup with a complete cache and failure with an incomplete cache;
- clear failure on an unsupported Torch schema rather than fallback to an
  unrelated filename; and
- application-level synchronization or one analyzer per worker when custom
  components are stateful. Concurrent `run()` calls are not promised safe.

For face-analysis models, document provenance, license terms, intended use,
limitations, privacy expectations, and populations or conditions not validated.
The packaged governance gate does not automatically certify an external model;
that responsibility remains with the extension owner.
