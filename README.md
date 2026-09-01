# ![](https://raw.githubusercontent.com/tomas-gajarsky/facetorch/55fa112fce2708fdc1bee318e06dfd0e9758f612/data/facetorch-logo-42.png "facetorch logo") facetorch
![build](https://github.com/tomas-gajarsky/facetorch/actions/workflows/build.yml/badge.svg?branch=main)
![lint](https://github.com/tomas-gajarsky/facetorch/actions/workflows/lint.yml/badge.svg?branch=main)
[![PyPI](https://img.shields.io/pypi/v/facetorch)](https://pypi.org/project/facetorch/)
[![Conda (channel only)](https://img.shields.io/conda/vn/conda-forge/facetorch)](https://anaconda.org/conda-forge/facetorch)
[![PyPI - License](https://img.shields.io/pypi/l/facetorch)](https://raw.githubusercontent.com/tomas-gajarsky/facetorch/55fa112fce2708fdc1bee318e06dfd0e9758f612/LICENSE)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

  <a target="_blank" href="https://colab.research.google.com/github/tomas-gajarsky/facetorch/blob/main/notebooks/facetorch_notebook_demo.ipynb">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

[API documentation](https://tomas-gajarsky.github.io/facetorch/facetorch/index.html), [extension guide](https://github.com/tomas-gajarsky/facetorch/blob/55fa112fce2708fdc1bee318e06dfd0e9758f612/docs/custom-components.md), [v0.6.x migration guide](https://github.com/tomas-gajarsky/facetorch/blob/55fa112fce2708fdc1bee318e06dfd0e9758f612/docs/migration-v1.md), [model compatibility](https://github.com/tomas-gajarsky/facetorch/blob/55fa112fce2708fdc1bee318e06dfd0e9758f612/docs/model-compatibility.md)
 
[Docker Hub](https://hub.docker.com/repository/docker/tomasgajarsky/facetorch) [(GPU)](https://hub.docker.com/repository/docker/tomasgajarsky/facetorch-gpu)


**Facetorch** is a Python library designed for facial detection and analysis, leveraging the power of deep neural networks. Its primary aim is to curate open-source face analysis models from the community, package them as portable [torch.export](https://pytorch.org/docs/stable/export.html) models, and integrate them into a versatile face analysis toolkit. The library offers the following key features:

1. **Customizable Configuration:** Easily configure your setup using [Hydra](https://hydra.cc/docs/intro/) and its powerful [OmegaConf](https://omegaconf.readthedocs.io/) capabilities.

2. **Reproducible Environments:** Ensure reproducibility with [uv](https://github.com/astral-sh/uv) for fast Python package management, [conda-lock](https://github.com/conda-incubator/conda-lock) for conda-forge dependency management, and [Docker](https://docs.docker.com/get-docker/) for containerization.

3. **Portable Models:** Models are serialized with `torch.export` (`.pt2` format) — no model source code needed at inference time, with dynamic batch support and `torch.compile` compatibility.

4. **Governed Extensibility:** Custom readers, processors, and configurations can
   be supplied directly. Adding a hosted model to the built-in defaults also
   requires an immutable manifest entry, validation evidence, provenance, and
   model-rights approval.

5. **Deterministic Input:** Accepts local paths, tensors, NumPy arrays, PIL images,
   and bytes through one canonical pipeline. Remote input requires an explicit,
   bounded `URLReader` configuration.

Facetorch provides an efficient, scalable, and user-friendly solution for facial analysis tasks, catering to developers and researchers looking for flexibility and performance.

### Requirements

* Python >= 3.10 and < 3.13
* PyTorch 2.6.x or 2.11.x; other minors are rejected before model download
* Linux x86-64 is the official v1 candidate platform; Windows, macOS, ARM, and
  Apple MPS are experimental

The exact candidate matrix, named CUDA pairs, experimental platforms, and current
model-rights gates are documented in
[Model compatibility and governance](https://github.com/tomas-gajarsky/facetorch/blob/55fa112fce2708fdc1bee318e06dfd0e9758f612/docs/model-compatibility.md).

Please use this library responsibly and with caution. Adhere to the [European Commission's Ethics Guidelines for Trustworthy AI](https://ec.europa.eu/futurium/en/ai-alliance-consultation.1.html) to ensure ethical and fair usage. Keep in mind that the models may have limitations and potential biases, so it is crucial to evaluate their outputs critically and consider their impact.


## Install

> [!IMPORTANT]
> This documentation targets **`1.0.0rc2` (Beta)**. Install the exact candidate
> only after it appears on [PyPI](https://pypi.org/project/facetorch/). Bare
> `pip install facetorch` and Docker `latest` remain on the stable `0.6.2` line
> during the RC soak. Conda-forge is asynchronous and must be verified separately.
> None of those unpinned routes can be assumed to provide the v1 API shown below.

Use a virtual environment and install the supported CPU PyTorch cohort first.
This avoids pip selecting a multi-gigabyte CUDA dependency graph on a CPU host:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu \
  "torch==2.11.0+cpu" "torchvision==0.26.0+cpu"
python -m pip install "facetorch==1.0.0rc2"
```

For the validated CUDA 13.0 cohort, use a compatible NVIDIA host and replace the
PyTorch install above with:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu130 \
  "torch==2.11.0+cu130" "torchvision==0.26.0+cu130"
```

Torch 2.6 CPU/CUDA 12.4 is also supported; see the exact
[compatibility matrix](https://github.com/tomas-gajarsky/facetorch/blob/55fa112fce2708fdc1bee318e06dfd0e9758f612/docs/model-compatibility.md). The default model selection
then needs approximately 1.2 GB of cache data and at least 2 GB of free cache
space while downloads are staged.

[Conda-forge](https://anaconda.org/conda-forge/facetorch) remains an asynchronous
channel. Do not use its unversioned install command for the RC; wait until the
feedstock displays `1.0.0rc2`, then pin that exact version.

## Usage

### Docker option

Docker is optional for the Python API. With [Docker](https://docs.docker.com/get-docker/)
and [Docker Compose](https://docs.docker.com/compose/install/), use the immutable
RC image tag rather than `latest`:

### Run docker example

The production image contains the example script. Compose mounts `data/input`
read-only and keeps generated images in the `facetorch-output` volume, so the
non-root container never needs write access to the source checkout.

CPU:

```bash
FACETORCH_DOCKER_TAG=1.0.0-rc.2 docker compose run --rm facetorch \
  python /opt/facetorch/example.py /workspace/data/input/test.jpg \
  --output /workspace/data/output/test.png
```

GPU:

```bash
FACETORCH_DOCKER_TAG=1.0.0-rc.2 docker compose run --rm facetorch-gpu \
  python /opt/facetorch/example.py /workspace/data/input/test.jpg \
  --profile gpu --output /workspace/data/output/test-gpu.png
```

Copy a result from the persistent volume into the checkout when needed:

```bash
FACETORCH_DOCKER_TAG=1.0.0-rc.2 docker compose run --rm -T facetorch \
  cat /workspace/data/output/test.png > data/output/test.png
```

Check *data/output* for the copied image with bounding boxes and facial 3D landmarks.

(Apple Mac M1) Use Rosetta 2 emulator in Docker Desktop to run the CPU version.

### Python API

This copy-paste smoke checks an installed wheel, packaged configuration, reader,
and result contract without downloading models or requiring a repository checkout:

<!-- facetorch-readme-smoke:start -->
```python
import torch

from facetorch import FaceAnalyzer, load_config

cfg = load_config()
analyzer = FaceAnalyzer(cfg.analyzer)
result = analyzer.run(
    image_source=torch.zeros((3, 32, 32), dtype=torch.uint8),
    skip_detector=True,
    include_predictors=[],
    include_tensors=True,
)

assert len(result.faces) == 1
assert result.image is not None
assert not analyzer.detector_loaded
assert analyzer.loaded_predictors == ()
```
<!-- facetorch-readme-smoke:end -->

For normal analysis, provide one image. The first call downloads only the selected
compatible model artifacts; subsequent calls reuse the local cache:

```python
from facetorch import FaceAnalyzer, InputSpec, load_config

cfg = load_config()  # packaged CPU profile; independent of the working directory
analyzer = FaceAnalyzer(cfg.analyzer)

# Analyze one local path, tensor, NumPy array, PIL image, or bytes object
result = analyzer.run(image_source="path/to/image.jpg")

# Run only specific predictors
result = analyzer.run(image_source="image.jpg", include_predictors=["fer", "embed"])

# Inspect configured names without loading their models
print(analyzer.configured_predictors)

# Batch faces detected within this image; source-image batches are not supported
result = analyzer.run(image_source="image.jpg", face_batch_size=8)

# Strictly describe a normalized pre-cropped CHW face tensor
result = analyzer.run(
    image_source=face_tensor,
    input_policy="strict",
    input_spec=InputSpec(layout="CHW", value_range="0_1", color_space="RGB"),
    skip_detector=True,
    include_tensors=True,
)

print(result.faces)
print(result.image)  # retained because include_tensors=True

# FaceAnalyzer is also callable
result = analyzer("image.jpg")
```

`input_policy="coerce"` is the default. It applies documented, deterministic
range and channel conversions and emits `InputCoercionWarning` when a conversion
may surprise the caller. `input_policy="strict"` accepts source-specific uint8 RGB
conventions by default and requires `InputSpec` for declared layout, range, color,
or alpha conversions. Torch defaults to CHW/BCHW; NumPy defaults to HWC/BHWC; a
four-dimensional input must have `B=1`.

Every successful call returns `AnalysisResult`. Set `include_tensors=True` to retain
the optional `image`, `tensor`, and `detection` fields. The v0.x `batch_size` name is
a warning alias for `face_batch_size` throughout v1.x; supplying both is an error.
`face_batch_size` is an upper bound: the shipped predictor artifacts automatically
split requests into batches of at most 64, while custom predictors may declare a
different `max_batch_size`.
Code that temporarily needs the old `Response`/`ImageData` union can call the
explicit, warning-emitting `analyzer.run_legacy(...)` adapter.

Detector and predictor models are loaded only when execution requests them, then
cached for the lifetime of that `FaceAnalyzer`. Selection-linked utilizers are lazy
too. Their predictor requirements are declared explicitly by
`analyzer.utilizer_dependencies`, so component names do not have to match and
excluding a required predictor cannot trigger a utilizer metadata download.
`include_predictors=None` runs every configured predictor, while
`include_predictors=[]` runs none. Exclusions follow the same configured ordering.
Unknown or duplicate names and any simultaneous use of `include_predictors` and
`exclude_predictors` are rejected before the image is read or a model is loaded.
`skip_detector=True` never constructs the detector. Selected predictors still
require a configured face unifier; use `include_predictors=[]` when no unifier is
configured and predictor-free processing is intended.

Use `configured_predictors` to inspect names without loading models;
`loaded_predictors`, `loaded_utilizers`, and `detector_loaded` expose current cache
state. Accessing `analyzer.detector` or a value in `analyzer.predictors` or
`analyzer.utilizers` explicitly loads and caches that component. Lazy initialization
is protected against concurrent construction, but concurrent `run()` calls are not
guaranteed safe because configured custom readers and processors may be stateful.
Use one analyzer per worker or synchronize calls externally.

Detector and predictor configs may set `compile_model: true` and a
`compile_options` mapping; the options are passed unchanged to `torch.compile` when
that selected component is first loaded. Model wrapper constructors accept only
declared options now—custom extensions should expose their own constructor
parameters instead of relying on arbitrary attributes.

Network access is never inferred from a string. To accept a remote image, configure
`facetorch.analyzer.reader.URLReader` explicitly with allowed HTTP schemes, a
timeout, maximum redirects, and a maximum response size. The reader resolves and
rejects loopback, private, link-local, and other non-public targets before every
request and redirect; it is not a general-purpose internal-network fetcher.

### Configure

`load_config()` composes configuration resources installed inside the `facetorch`
package. It is the supported default for library users and works without a cloned
repository:

```python
from facetorch import load_config, load_config_from_path

cpu_cfg = load_config()
gpu_cfg = load_config("gpu")
customized = load_config(
    overrides=[
        "analyzer.optimize_transforms=false",
        "analyzer/predictor/fer=efficientnet_b0_7",
    ]
)

# Advanced deployments may compose an explicit external Hydra tree.
external = load_config_from_path(
    "/srv/my-app/conf/config.yaml",
    overrides=["analyzer.device=cpu"],
)
```

The no-argument loader deliberately selects CPU because it is the portable,
device-independent baseline. GPU support has not been removed: use
`load_config("gpu")`, the `--profile gpu` script option, or the GPU container to
select CUDA explicitly. Release validation covers both devices, and predictor
batches remain multiple faces detected within one source image—not multiple
source images.

Overrides use standard Hydra override strings and are applied after the selected
CPU/GPU profile. `load_config_from_path()` resolves a relative path against the
caller's working directory and composes the file's `defaults` list using its parent
directory as the Hydra configuration root.

The old source-checkout pattern `OmegaConf.load("conf/config.yaml")` remains a
low-level OmegaConf operation, but it does not compose Hydra defaults and is not the
installed-library configuration API. Source and managed deployments that maintain
an external Hydra tree should migrate to `load_config_from_path()` during v1.x.

### Runtime paths

Models and generated model metadata use a versioned, OS-appropriate user cache.
Configuration loading does not create it; directories are created only when a
selected artifact is first needed. The following environment variables provide
explicit deployment overrides:

- `FACETORCH_CACHE_DIR`: root for all facetorch caches.
- `FACETORCH_MODEL_DIR`: optional model-only override.
- `FACETORCH_METADATA_DIR`: optional metadata-only override.
- `FACETORCH_OFFLINE`: `1`, `true`, `yes`, or `on` forbids every model and
  metadata network request; invalid values fail configuration loading.

Linux follows `XDG_CACHE_HOME` and otherwise uses `~/.cache/facetorch`; macOS uses
`~/Library/Caches/facetorch`; Windows uses the local application-data cache.
File logging and image output are disabled by default. Enable them only with an
explicit writable `analyzer.logger.path_file` override or `path_output` argument.

For containers, mount a persistent volume and set the same cache root explicitly,
for example `FACETORCH_CACHE_DIR=/var/cache/facetorch` with a volume mounted at
`/var/cache/facetorch`. A cache populated by an online run can be mounted at that
same location for later no-network execution; keep its directory layout intact.

## Components
FaceAnalyzer is the main class of facetorch as it is the orchestrator responsible for initializing and running the following components:

1. Reader - reads the image and returns an ImageData object containing the image tensor.
2. Detector - wrapper around a neural network that detects faces.
3. Unifier - processor that unifies sizes of all faces and normalizes them
    between 0 and 1.
4. Predictor dict - set of wrappers around neural networks trained to analyze facial features.
5. Utilizer dict - set of wrappers around any functionality that requires the output of neural networks e.g. drawing bounding boxes or facial landmarks.

### Structure
```
analyzer
    ├── reader
    ├── detector
    ├── unifier
    └── predictor
            ├── embed
            ├── verify
            ├── fer
            ├── au
            ├── va
            ├── deepfake
            └── align
    └── utilizer
            ├── align
            ├── draw
            └── save
```


## Models

The source links below are the original repositories already used by Facetorch.
The weight-license column reflects the artifact-specific review approved on
2026-08-23 and recorded, with checkpoint hashes and mapping methods, in
[`facetorch/models/governance.json`](https://github.com/tomas-gajarsky/facetorch/blob/55fa112fce2708fdc1bee318e06dfd0e9758f612/facetorch/models/governance.json). MIT and
Apache-2.0 are preserved as received; neither license was converted into the
other. These licenses do not grant rights to upstream training datasets.

### Detector

    |     model     |   source  |   params  | weight license | version |
    | ------------- | --------- | --------- | ----------- | ------- |
    |   RetinaFace  |  biubug6  |   27.3M   | MIT license |    1    |

1. biubug6
    * code: [Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)
    * paper: [Deng et al. - RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild](https://openaccess.thecvf.com/content_CVPR_2020/html/Deng_RetinaFace_Single-Shot_Multi-Level_Face_Localisation_in_the_Wild_CVPR_2020_paper.html)



### Predictor

#### Facial Representation Learning (embed)

    |       model       |   source   |  params | weight license | version |
    | ----------------- | ---------- | ------- | ----------- | ------- |
    |  ResNet-50 VGG 1M |  1adrianb  |  28.4M  | MIT license |    1    |

1. 1adrianb
    * code: [unsupervised-face-representation](https://github.com/1adrianb/unsupervised-face-representation)
    * paper: [Bulat et al. - Pre-training strategies and datasets for facial representation learning](https://arxiv.org/abs/2103.16554)
    * Note: ```include_tensors``` needs to be True in order to include the model prediction in Prediction.logits

#### Face Verification (verify)

    |       model      |   source    |  params  |   weight license   | version |
    | ---------------- | ----------- | -------- | ------------------ | ------- |
    |    MagFace+UNPG  | Jung-Jun-Uk |   65.2M  | Apache License 2.0 |    1    |
    |  AdaFaceR100W12M |  mk-minchul |    -     |     MIT License    |    2    |

1. Jung-Jun-Uk
    * code: [UNPG](https://github.com/junuke/UNPG) and [MagFace](https://github.com/IrvingMeng/MagFace)
    * paper: [Jung et al. - Unified Negative Pair Generation toward Well-discriminative Feature Space for Face Recognition](https://arxiv.org/abs/2203.11593)
    * Note: ```include_tensors``` needs to be True in order to include the model prediction in Prediction.logits
2. mk-minchul
    * code: [AdaFace](https://github.com/mk-minchul/adaface)
    * paper: [Kim et al. - AdaFace: Quality Adaptive Margin for Face Recognition](https://arxiv.org/abs/2204.00964)
    * Note: ```include_tensors``` needs to be True in order to include the model prediction in Prediction.logits


#### Facial Expression Recognition (fer)

    |       model       |      source    |  params  |   weight license   | version |
    | ----------------- | -------------- | -------- | ------------------ | ------- |
    | EfficientNet B0 7 |   sb-ai-lab    |    4M    | Apache License 2.0 |    1    |
    | EfficientNet B2 8 |   sb-ai-lab    |   7.7M   | Apache License 2.0 |    2    |

1. sb-ai-lab
    * code and checkpoints: [EmotiEffLib](https://github.com/sb-ai-lab/EmotiEffLib)
    * paper: [Savchenko et al. - Classifying Emotions and Engagement in Online Learning Based on a Single Facial Expression Recognition Neural Network](https://arxiv.org/abs/2203.13436)

#### Facial Action Unit Detection (au)

    |        model        |   source   |  params |   weight license   | version |
    | ------------------- | --------- | ------- | ------------------ | ------- |
    | OpenGraph Swin Base | lingjivoo  |   94M   | Apache License 2.0 |    1    |

1. lingjivoo
    * checkpoint and primary code: [OpenGraphAU](https://github.com/lingjivoo/OpenGraphAU)
    * related code with preserved MIT attribution: [ME-GraphAU](https://github.com/CVI-SZU/ME-GraphAU)
    * paper: [Luo et al. - Learning Multi-dimensional Edge Feature-based AU Relation Graph for Facial Action Unit Recognition](https://arxiv.org/abs/2205.01782)
    * Note: The v1 candidate uses torch.export artifacts for the explicit Torch 2.6 / 2.11 cohorts; release CPU/CUDA evidence is tracked separately.

#### Facial Valence Arousal (va)

    |       model       |   source   |  params | weight license | version |
    | ----------------- | ---------- | ------- | ----------- | ------- |
    |   ELIM AlexNet    | kdhht2334  |  2.3M   | MIT license |    1    |

1. kdhht2334
    * code: [ELIM](https://github.com/kdhht2334/ELIM_FER)
    * paper: [Kim et al. - Optimal Transport-based Identity Matching
for Identity-invariant Facial Expression Recognition](https://arxiv.org/abs/2209.12172)

#### Deepfake Detection (deepfake)

    |         model        |      source      |  params  | weight license | version |
    | -------------------- | ---------------- | -------- | ----------- | ------- |
    |    EfficientNet B7   |     selimsef     |   66.4M  | MIT license |    1    |

1. selimsef
    * code: [dfdc_deepfake_challenge](https://github.com/selimsef/dfdc_deepfake_challenge)
    * challenge: [Seferbekov - Deepfake Detection Challenge 1st place solution](https://www.kaggle.com/competitions/deepfake-detection-challenge/discussion)

#### Face Alignment (align)

    |       model       |      source      |  params  | weight license | version |
    | ----------------- | ---------------- | -------- | ----------- | ------- |
    |    MobileNet v2   |     choyingw     |   4.1M   | MIT license |    1    |

1. choyingw
    * code: [SynergyNet](https://github.com/choyingw/SynergyNet)
    * challenge: [Wu et al. - Synergy between 3DMM and 3D Landmarks for Accurate 3D Facial Geometry](https://arxiv.org/abs/2110.09772)
    * Note: ```include_tensors``` needs to be True in order to include the model prediction in Prediction.logits



### Model download

Models are downloaded on first selected use into facetorch's versioned user cache
using Hugging Face Hub. Each request is pinned to an immutable repository commit.
The packaged manifest selects one artifact for the active PyTorch/device pair, and
records its real format, byte size, SHA-256 digest, schema cohort, and validation
metadata. Downloads are staged beside the destination, verified without executing
them, and atomically promoted under a process-safe cache lock. Existing files are
verified again before load; corrupt entries are quarantined rather than executed.

Full verification on use is the secure default. A deployment with a separately
protected, read-only cache may add `verify_on_use: false` to selected downloader
configs to skip the repeat digest pass; release validation must keep it enabled.

Set `FACETORCH_CACHE_DIR` before configuration loading to choose an explicit
writable or persistent location. The current default model selection occupies
approximately **1.2 GB**; allow at least **2 GB** of free cache space for download
staging and metadata. Selecting fewer predictors reduces the download.

Plan first, then explicitly confirm a multi-artifact prefetch:

```python
from facetorch import plan_model_prefetch, prefetch_models

plan = plan_model_prefetch(
    "cpu",
    include_predictors=["fer", "embed"],
    skip_detector=False,
)
print(plan.download_bytes, plan.items)

prefetch_models(
    "cpu",
    include_predictors=["fer", "embed"],
    skip_detector=False,
    confirm=True,
)
```

`include_predictors=[]` and `skip_detector=True` produce an empty plan and make no
model-network request. Alignment metadata is included only when the selected
predictor set satisfies the composed `analyzer.utilizer_dependencies` requirements
for the `align` utilizer. Bulk prefetch refuses to start before `confirm=True` and
reports the estimated missing bytes first.

For deployment, populate the same cache online and then compose with
`load_config(offline=True)` or set `FACETORCH_OFFLINE=1`. Offline mode verifies and
uses the cache, or raises `OfflineCacheError` before inference; it never attempts a
network fallback.

Legacy TorchScript is disabled by default. An eligible, manifest-pinned legacy
artifact may be selected only with `load_config(allow_legacy_models=True)`, emits
`LegacyModelWarning`, retains its real `.pt` extension, and is never selected for
CUDA. A missing or incompatible `.pt2` does not trigger a filename/download
cascade. The Google Drive model configs are deprecated and fail closed unless an
operator supplies immutable size and digest metadata; the verified 3D alignment
metadata remains supported.

Inspect and recover old caches without executing them:

```python
from facetorch import (
    cleanup_quarantined_cache,
    inspect_incompatible_cache,
    inspect_legacy_cache,
    inspect_quarantined_cache,
    migrate_legacy_artifact,
    reset_incompatible_cache,
)

entries = inspect_legacy_cache("/path/to/v0-cache")
print(entries)  # includes mislabeled TorchScript stored as .pt2

migrate_legacy_artifact(
    "/path/to/v0-cache/model.pt2",
    "detector-retinaface-legacy",
    "/path/to/v1-cache/model.pt",
)

print(inspect_quarantined_cache())
cleanup_quarantined_cache(confirm=True)  # explicit, versioned-cache files only
print(inspect_incompatible_cache())
reset_incompatible_cache(confirm=True)  # after runtime/artifact remediation
```

Migration succeeds only for an exact manifest hash/format match and copies rather
than changes the old file. Facetorch never automatically rewrites or deletes a v0.x
cache. For rollback, keep v0.6.x and v1 model roots separate (for example with
different `FACETORCH_MODEL_DIR` values); do not point v0.6.x at the v1 manifest
layout. Quarantine inspection is non-destructive, and cleanup is restricted to the
versioned facetorch model/metadata roots.

Models are available on the [Hugging Face Hub](https://huggingface.co/tomas-gajarsky).
The legacy [Google Drive folder](https://drive.google.com/drive/folders/19qlklR18wYfFsCChQ78it10XciuTzbDM?usp=sharing)
is retained for manual backward compatibility only. The packaged manifest is
approved and pins the published, digest-bound validation metadata and legal
documents produced by the clean RC1 model-export commit. Model governance is also
approved: all ten records are release-eligible under their pinned upstream
licenses, verified checkpoint mappings, preserved attribution, and owner-approved
redistribution policy. That approval does not grant rights to upstream training
datasets or remove the deployment-specific consent, privacy, performance, and
legal obligations listed in each model record. The coordinated RC1 release retains
its separate exact-final-commit dry-run and protected-environment gates.

Maintainers export and validate every requested model locally before any remote
write. Inline `--upload` is disabled. Publication requires a deterministic plan,
an approval bound to that plan's digest, immutable parent revisions, and a
resumable receipt. Each model's artifact and metadata are committed together to a
candidate branch; the initial immutable manifest commit is created only after every
model repository succeeds. Deterministically rendered legal documents are then
committed and a final manifest binds those resulting immutable revisions. See
[the model publication runbook](https://github.com/tomas-gajarsky/facetorch/blob/55fa112fce2708fdc1bee318e06dfd0e9758f612/docs/model-publication.md).

#### Why exported models?

Facetorch v1 moved default model artifacts from TorchScript (`.pt`) to `torch.export` (`.pt2`) so inference no longer depends on bundled model source code, custom class definitions, or TorchScript-specific runtime behavior. This makes the hosted models easier to validate, redistribute, and load across normal Python package installations. TorchScript artifacts are still useful as legacy fallbacks, but v1 workflows should prefer Hugging Face `.pt2` artifacts.

`torch.export` serialization is tied to PyTorch's exported-program schema, so one
`.pt2` file is not guaranteed to load across future or older PyTorch minors. The
approved manifest has exact cohorts for Torch 2.6 and 2.11. Package metadata uses
the same bounded, disjoint set. Torch 2.3-2.5 and 2.7-2.10 are unsupported and fail
before download; no schema-major or numeric fallback is attempted. Torch 2.3 was
dropped because its affected `torch.load(weights_only=True)` path has a critical
remote-code-execution advisory. Torch 2.6 is temporarily retained under three
moderate, affected-API-specific exceptions documented in
`security/advisory-exceptions.json`, all expiring on 2026-11-20.
Validation uses immutable CPU golden references for both CPU and CUDA artifacts,
with TensorFloat-32 disabled and the numeric policy recorded. Predictor batch
sizes refer only to faces from one input image; multi-image batching is not
supported in v1. See [model compatibility and governance](https://github.com/tomas-gajarsky/facetorch/blob/55fa112fce2708fdc1bee318e06dfd0e9758f612/docs/model-compatibility.md)
for the exact candidate evidence and remaining blockers.


### Execution time

Reference GPU benchmark (AU included, face_batch_size=8, utilizers disabled, default runtime):
- `test.jpg` (4 faces): pass1 `688 ms`, pass2 `216 ms`, pass3 `311 ms`, warm avg (pass2+pass3) `263 ms`
- `test3.jpg` (25 faces): pass1 `904 ms`, pass2 `745 ms`, pass3 `728 ms`, warm avg (pass2+pass3) `737 ms`

Full FaceAnalyzer artifact comparison (detector + predictors `embed`, `verify`, `fer`, `au`, `va`, `deepfake`, `align`):
- benchmark method: 12 passes/image, report warm median over passes 3-12

| image | faces | all `.pt2` median (default JIT) | all `.pt2` median (TS stability flags) | all TorchScript `.pt` median (TS stability flags) | delta TorchScript - `.pt2` (same flags) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `test.jpg` | 4 | `152.8 ms` | `151.2 ms` | `147.1 ms` | `-4.0 ms` |
| `test3.jpg` | 25 | `525.9 ms` | `520.4 ms` | `518.8 ms` | `-1.6 ms` |

CPU full-stack comparison (8 passes/image, warm median over passes 3-8, `torch_num_threads=16`):

| image | faces | all `.pt2` median | all TorchScript `.pt` median | delta TorchScript - `.pt2` |
| --- | ---: | ---: | ---: | ---: |
| `test.jpg` | 4 | `3080.0 ms` | `2796.5 ms` | `-283.5 ms` |
| `test3.jpg` | 25 | `15945.5 ms` | `14727.6 ms` | `-1217.9 ms` |

Environment used for this benchmark:
- GPU: `NVIDIA GeForce RTX 3090`
- Python: `3.10.12`
- Torch: `2.11.0+cu130`

Method notes:
- Reference GPU timings include pass-level values; artifact comparisons use warm medians to avoid first-pass warm-up effects.
- Timings above include all predictors, including AU.
- Utilizers were disabled for this benchmark run.
- Full TorchScript CUDA comparison required disabling TorchScript profiling/fuser paths for runtime stability (`_jit_set_profiling_executor(False)`, `_jit_set_profiling_mode(False)`, `_jit_override_can_fuse_on_gpu(False)`).
- In this setup, TorchScript is only slightly faster in warm median (about 2-4 ms, <2%); treat this as near-parity rather than a large format-level difference.
- On CPU, TorchScript was faster by about 8-9% for this full-stack run.
- TorchScript shows much slower first pass on some runs (runtime graph specialization), so use warm metrics for fair comparison.
- One can monitor component timings in logs using DEBUG level.


## Development
Run the Docker container:
* CPU: ```docker compose -f docker-compose.dev.yml run facetorch-dev```
* GPU: ```docker compose -f docker-compose.dev.yml run facetorch-dev-gpu```

### Extend facetorch

Custom predictors and detectors remain first-class in v1. The packaged model
manifest secures the models shipped by facetorch; it is not an allow-list for
application extensions.

There are three deliberately separate paths:

1. Install an already constructed predictor or detector directly on a
   `FaceAnalyzer`.
2. Use an external Hydra tree with a custom component and a digest-pinned model.
3. Contribute a model to facetorch's built-in defaults and complete the full
   artifact, compatibility, provenance, rights, and release-evidence process.

Private and third-party Hugging Face models use direct external mode: omit
`manifest_id` and declare the exact filename, immutable Hub commit, SHA-256,
byte size, format, and device in the application configuration. Only an
officially shipped model belongs in `facetorch/models/manifest.json`.

The complete [custom predictor and detector guide](https://github.com/tomas-gajarsky/facetorch/blob/55fa112fce2708fdc1bee318e06dfd0e9758f612/docs/custom-components.md)
contains a runnable no-download example, the predictor and detector contracts,
a direct immutable Hugging Face YAML example, external configuration guidance,
Torch cohort responsibilities, testing guidance, and the separate checklist for
contributing an officially governed built-in model.


### Update environment

#### Dependency ownership and release channels
* `pyproject.toml` is the packaging source of truth for PyPI releases and pip/uv installs (including Docker build paths using uv).
* Conda package publishing (`conda-forge/facetorch`) is maintained outside this repository in conda-forge feedstock workflows.
* `environment.yml` and `gpu.environment.yml` are conda environment baselines for conda users.
* `environments/` contains four exact release profiles: Torch 2.6 and 2.11 on CPU, plus Torch 2.6/CUDA 12.4 and 2.11/CUDA 13.0. Each profile has its own `pyproject.toml`, `uv.lock`, and explicit official PyTorch index.
* The production CPU image uses the exact Torch 2.6 CPU profile. The production GPU image uses the exact Torch 2.6/CUDA 12.4 profile; neither image upgrades Torch after resolution.
* The GPU conda baseline is deliberately only a Python 3.12/CUDA 12.4 system layer. Its Python packages come from `environments/torch-2.6-cu124/uv.lock`, because conda-forge's current Torch 2.6 GPU solve requires a newer CUDA line than the validated v1 pair.
* uv uses PyPI for normal packages and explicit named PyTorch indexes only for `torch` and `torchvision`, avoiding global extra-index resolution drift.
* Overlapping dependencies between pyproject and conda env files are intentionally kept aligned.
* CI enforces alignment with `python scripts/check_dependency_sync.py`, audits every exact profile with `python scripts/audit_dependencies.py`, and emits hashed requirements plus CycloneDX SBOMs. Advisory exceptions must be approved, justified, and no longer than 90 days.

#### uv (used by Docker dev/test images)
* Add packages with corresponding versions to ```pyproject.toml``` dependencies
* Lock the environment: ```uv lock```
* Sync the environment: ```uv sync --extra dev```
* Check every exact profile: ```for profile in environments/*; do uv lock --check --project "$profile"; done```

#### conda (for conda-forge users)
CPU:
* Add packages with corresponding versions to ```environment.yml``` file
* Lock the environment: ```conda-lock -p linux-64 -f environment.yml --lockfile conda-lock.yml```
* (Alternative Docker) Lock the environment: ```docker compose -f docker-compose.dev.yml run facetorch-lock```
* Install the locked environment: ```conda-lock install --name env conda-lock.yml```

GPU:
* Add packages with corresponding versions to ```gpu.environment.yml``` file
* Lock the system layer: ```conda-lock --with-cuda 12.4 -p linux-64 -f gpu.environment.yml --lockfile gpu.conda-lock.yml```
* Sync the Python layer: ```uv sync --project environments/torch-2.6-cu124 --frozen```
* (Alternative Docker) Lock the environment: ```docker compose -f docker-compose.dev.yml run facetorch-lock-gpu```
* Install the locked environment: ```conda-lock install --name env gpu.conda-lock.yml```

### Run tests + coverage
* Run tests and generate coverage: ```pytest tests --verbose --cov-report html:coverage --cov facetorch```

### Generate documentation
* Generate documentation from docstrings using pdoc3:  ```pdoc --html facetorch --output-dir docs --force --template-dir pdoc/templates/```

### Profiling
1. Run profiling of the example script: ```python -m cProfile -o profiling/example.prof scripts/example.py```
2. Open profiling file in the browser: ```snakeviz profiling/example.prof```

## Research Highlights Leveraging facetorch

### [Sharma et al. (2024)](https://aclanthology.org/2024.signlang-1.39.pdf)

Sharma, Paritosh, Camille Challant, and Michael Filhol. "Facial Expressions for Sign Language Synthesis using FACSHuman and AZee." *Proceedings of the LREC-COLING 2024 11th Workshop on the Representation and Processing of Sign Languages*, pp. 354–360, 2024.

### [Liang et al. (2023)](https://dl.acm.org/doi/abs/10.1145/3581783.3612854)

Liang, Cong, Jiahe Wang, Haofan Zhang, Bing Tang, Junshan Huang, Shangfei Wang, and Xiaoping Chen. "Unifarn: Unified transformer for facial reaction generation." *Proceedings of the 31st ACM International Conference on Multimedia*, pp. 9506–9510, 2023.

### [Gue et al. (2023)](https://research.monash.edu/en/publications/facial-expression-recognition-as-markers-of-depression)

Gue, Jia Xuan, Chun Yong Chong, and Mei Kuan Lim. "Facial Expression Recognition as markers of Depression." *2023 Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)*, pp. 674–680, 2023.


## Acknowledgements
I would like to thank the open-source community and the researchers who have shared their work and published models. This project would not have been possible without their contributions.


## Citing

If you use facetorch in your work, please make sure to appropriately credit the original authors of the models it employs. Additionally, you may consider citing the facetorch library itself. Below is an example citation for facetorch:

```
@misc{facetorch,
    author = {Gajarsky, Tomas},
    title = {Facetorch: A Python Library for Analyzing Faces Using PyTorch},
    year = {2024},
    publisher = {GitHub},
    journal = {GitHub Repository},
    howpublished = {\url{https://github.com/tomas-gajarsky/facetorch}}
}
```
