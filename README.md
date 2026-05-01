# ![](https://raw.githubusercontent.com/tomas-gajarsky/facetorch/main/data/facetorch-logo-42.png "facetorch logo") facetorch
![build](https://github.com/tomas-gajarsky/facetorch/actions/workflows/build.yml/badge.svg?branch=main)
![lint](https://github.com/tomas-gajarsky/facetorch/actions/workflows/lint.yml/badge.svg?branch=main)
[![PyPI](https://img.shields.io/pypi/v/facetorch)](https://pypi.org/project/facetorch/)
[![Conda (channel only)](https://img.shields.io/conda/vn/conda-forge/facetorch)](https://anaconda.org/conda-forge/facetorch)
[![PyPI - License](https://img.shields.io/pypi/l/facetorch)](https://raw.githubusercontent.com/tomas-gajarsky/facetorch/main/LICENSE)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

  <a href="https://huggingface.co/spaces/tomas-gajarsky/facetorch-app">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue" alt="Hugging Face Spaces">
  </a> <a target="_blank" href="https://colab.research.google.com/github/tomas-gajarsky/facetorch/blob/main/notebooks/facetorch_notebook_demo.ipynb">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

[User Guide](https://medium.com/@gajarsky.tomas/facetorch-user-guide-a0e9fd2a5552), [Documentation](https://tomas-gajarsky.github.io/facetorch/facetorch/index.html), [ChatGPT facetorch guide](https://chat.openai.com/g/g-q8HWAkG4u-facetorch-guide)
 
[Docker Hub](https://hub.docker.com/repository/docker/tomasgajarsky/facetorch) [(GPU)](https://hub.docker.com/repository/docker/tomasgajarsky/facetorch-gpu)


**Facetorch** is a Python library designed for facial detection and analysis, leveraging the power of deep neural networks. Its primary aim is to curate open-source face analysis models from the community, package them as portable [torch.export](https://pytorch.org/docs/stable/export.html) models, and integrate them into a versatile face analysis toolkit. The library offers the following key features:

1. **Customizable Configuration:** Easily configure your setup using [Hydra](https://hydra.cc/docs/intro/) and its powerful [OmegaConf](https://omegaconf.readthedocs.io/) capabilities.

2. **Reproducible Environments:** Ensure reproducibility with [uv](https://github.com/astral-sh/uv) for fast Python package management, [conda-lock](https://github.com/conda-incubator/conda-lock) for conda-forge dependency management, and [Docker](https://docs.docker.com/get-docker/) for containerization.

3. **Portable Models:** Models are serialized with `torch.export` (`.pt2` format) — no model source code needed at inference time, with dynamic batch support and `torch.compile` compatibility.

4. **Simple Extensibility:** Extend the library by uploading your model file to Hugging Face Hub and adding a corresponding configuration YAML file to the repository.

5. **Flexible Input:** Accepts file paths, URLs, tensors, numpy arrays, PIL Images, and bytes. Grayscale and RGBA inputs are automatically converted to RGB.

Facetorch provides an efficient, scalable, and user-friendly solution for facial analysis tasks, catering to developers and researchers looking for flexibility and performance.

### Requirements

* Python >= 3.10 and < 3.14
* PyTorch >= 2.3 (facetorch routes exported model artifacts by torch minor version)

Please use this library responsibly and with caution. Adhere to the [European Commission's Ethics Guidelines for Trustworthy AI](https://ec.europa.eu/futurium/en/ai-alliance-consultation.1.html) to ensure ethical and fair usage. Keep in mind that the models may have limitations and potential biases, so it is crucial to evaluate their outputs critically and consider their impact.


## Install
[PyPI](https://pypi.org/project/facetorch/)
```bash
pip install facetorch
```
[Conda](https://anaconda.org/conda-forge/facetorch)
```bash
conda install -c conda-forge facetorch
```
## Usage

### Prerequisites
* [Docker](https://docs.docker.com/get-docker/)
* [Docker Compose](https://docs.docker.com/compose/install/)

Docker Compose provides an easy way of building a working facetorch environment with a single command.

### Run docker example
    
* CPU: ```docker compose run facetorch python ./scripts/example.py```
* GPU: ```docker compose run facetorch-gpu python ./scripts/example.py analyzer.device=cuda```

Check *data/output* for resulting images with bounding boxes and facial 3D landmarks.

(Apple Mac M1) Use Rosetta 2 emulator in Docker Desktop to run the CPU version.

### Python API

```python
from facetorch import FaceAnalyzer
from omegaconf import OmegaConf

cfg = OmegaConf.load("conf/config.yaml")
analyzer = FaceAnalyzer(cfg.analyzer)

# Analyze from file path, URL, tensor, numpy array, PIL Image, or bytes
response = analyzer.run(image_source="path/to/image.jpg")

# Run only specific predictors
response = analyzer.run(image_source="image.jpg", include_predictors=["fer", "embed"])

# Skip detector for pre-cropped face inputs
response = analyzer.run(image_source=face_tensor, skip_detector=True)

# FaceAnalyzer is also callable
response = analyzer("image.jpg")
```

### Configure

The project is configured by files located in *conf* with the main file: *conf/config.yaml*. One can easily add or remove modules from the configuration.

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

### Detector

    |     model     |   source  |   params  |   license   | version |
    | ------------- | --------- | --------- | ----------- | ------- |
    |   RetinaFace  |  biubug6  |   27.3M   | MIT license |    1    |

1. biubug6
    * code: [Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)
    * paper: [Deng et al. - RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild](https://openaccess.thecvf.com/content_CVPR_2020/html/Deng_RetinaFace_Single-Shot_Multi-Level_Face_Localisation_in_the_Wild_CVPR_2020_paper.html)



### Predictor

#### Facial Representation Learning (embed)

    |       model       |   source   |  params |   license   | version |  
    | ----------------- | ---------- | ------- | ----------- | ------- |
    |  ResNet-50 VGG 1M |  1adrianb  |  28.4M  | MIT license |    1    |

1. 1adrianb
    * code: [unsupervised-face-representation](https://github.com/1adrianb/unsupervised-face-representation)
    * paper: [Bulat et al. - Pre-training strategies and datasets for facial representation learning](https://arxiv.org/abs/2103.16554)
    * Note: ```include_tensors``` needs to be True in order to include the model prediction in Prediction.logits

#### Face Verification (verify)

    |       model      |   source    |  params  |      license       | version |  
    | ---------------- | ----------- | -------- | ------------------ | ------- |
    |    MagFace+UNPG  | Jung-Jun-Uk |   65.2M  | Apache License 2.0 |    1    |
    |  AdaFaceR100W12M |  mk-minchul |    -     |     MIT License    |    2    |

1. Jung-Jun-Uk
    * code: [UNPG](https://github.com/jung-jun-uk/unpg)
    * paper: [Jung et al. - Unified Negative Pair Generation toward Well-discriminative Feature Space for Face Recognition](https://arxiv.org/abs/2203.11593)
    * Note: ```include_tensors``` needs to be True in order to include the model prediction in Prediction.logits
2. mk-minchul
    * code: [AdaFace](https://github.com/mk-minchul/adaface)
    * paper: [Kim et al. - AdaFace: Quality Adaptive Margin for Face Recognition](https://arxiv.org/abs/2204.00964)
    * Note: ```include_tensors``` needs to be True in order to include the model prediction in Prediction.logits


#### Facial Expression Recognition (fer)

    |       model       |      source    |  params  |       license      | version |  
    | ----------------- | -------------- | -------- | ------------------ | ------- |
    | EfficientNet B0 7 | HSE-asavchenko |    4M    | Apache License 2.0 |    1    |
    | EfficientNet B2 8 | HSE-asavchenko |   7.7M   | Apache License 2.0 |    2    |

1. HSE-asavchenko
    * code: [face-emotion-recognition](https://github.com/HSE-asavchenko/face-emotion-recognition)
    * paper: [Savchenko - Facial expression and attributes recognition based on multi-task learning of lightweight neural networks](https://ieeexplore.ieee.org/abstract/document/9582508)

#### Facial Action Unit Detection (au)

    |        model        |   source  |  params |       license      | version |  
    | ------------------- | --------- | ------- | ------------------ | ------- |
    | OpenGraph Swin Base |  CVI-SZU  |   94M   |     MIT License    |    1    |

1. CVI-SZU
    * code: [ME-GraphAU](https://github.com/CVI-SZU/ME-GraphAU)
    * paper: [Luo et al. - Learning Multi-dimensional Edge Feature-based AU Relation Graph for Facial Action Unit Recognition](https://arxiv.org/abs/2205.01782)
    * Note: As of v1.0.0, the AU model uses torch.export format with torch-versioned cohort artifacts validated on CPU and CUDA (torch 2.3 / 2.6 / 2.11)

#### Facial Valence Arousal (va)

    |       model       |   source   |  params |   license   | version |
    | ----------------- | ---------- | ------- | ----------- | ------- |
    |  ELIM AL AlexNet  | kdhht2334  |  2.3M   | MIT license |    1    |

1. kdhht2334
    * code: [ELIM](https://github.com/kdhht2334/ELIM_FER)
    * paper: [Kim et al. - Optimal Transport-based Identity Matching
for Identity-invariant Facial Expression Recognition](https://arxiv.org/abs/2209.12172)

#### Deepfake Detection (deepfake)

    |         model        |      source      |  params  |   license   | version |
    | -------------------- | ---------------- | -------- | ----------- | ------- |
    |    EfficientNet B7   |     selimsef     |   66.4M  | MIT license |    1    |

1. selimsef
    * code: [dfdc_deepfake_challenge](https://github.com/selimsef/dfdc_deepfake_challenge)
    * challenge: [Seferbekov - Deepfake Detection Challenge 1st place solution](https://www.kaggle.com/competitions/deepfake-detection-challenge/discussion)

#### Face Alignment (align)

    |       model       |      source      |  params  |   license   | version |
    | ----------------- | ---------------- | -------- | ----------- | ------- |
    |    MobileNet v2   |     choyingw     |   4.1M   | MIT license |    1    |

1. choyingw
    * code: [SynergyNet](https://github.com/choyingw/SynergyNet)
    * challenge: [Wu et al. - Synergy between 3DMM and 3D Landmarks for Accurate 3D Facial Geometry](https://arxiv.org/abs/2110.09772)
    * Note: ```include_tensors``` needs to be True in order to include the model prediction in Prediction.logits



### Model download

Models are downloaded during runtime automatically to the *models* directory using Hugging Face Hub.
Models are available on the [Hugging Face Hub](https://huggingface.co/tomas-gajarsky). The legacy [Google Drive folder](https://drive.google.com/drive/folders/19qlklR18wYfFsCChQ78it10XciuTzbDM?usp=sharing) is retained for backward compatibility only and is effectively deprecated for v1+ workflows.
For exported `.pt2` models, facetorch can fall back across versioned artifacts when present (e.g. `model-torch2.3.pt2`, `model-torch2.6.pt2`, `model-torch2.11.pt2`).
By default, the downloader tries `model.pt2` first, then versioned cohort artifacts, and finally `model.pt` as a legacy fallback where available.

#### Why exported models?

Facetorch v1 moved default model artifacts from TorchScript (`.pt`) to `torch.export` (`.pt2`) so inference no longer depends on bundled model source code, custom class definitions, or TorchScript-specific runtime behavior. This makes the hosted models easier to validate, redistribute, and load across normal Python package installations. TorchScript artifacts are still useful as legacy fallbacks, but v1 workflows should prefer Hugging Face `.pt2` artifacts.

`torch.export` serialization is tied to PyTorch's exported-program schema, so one `.pt2` file is not guaranteed to load across every future or older PyTorch minor version. To avoid pinning users to one narrow torch version, facetorch publishes and validates cohort artifacts for representative supported runtimes: `torch 2.3`, `torch 2.6`, and `torch 2.11`. Runtime support starts at `PyTorch >= 2.3`; when versioned cohorts are available, the downloader selects from those artifacts and falls back to the next candidate if the current runtime cannot load the first choice.


### Execution time

Reference GPU benchmark (AU included, batch_size=8, utilizers disabled, default runtime):
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

### Add predictor
#### Prerequisites
1. Exported `.pt2` model file (see below)
2. Repository on Hugging Face Hub for hosting the model
3. facetorch [fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo)

Facetorch uses models exported with [torch.export](https://pytorch.org/docs/stable/export.html) (`.pt2` format). Export your model with dynamic batch support:

```python
import torch

model.eval()
batch = torch.export.Dim("batch", min=1, max=64)
ep = torch.export.export(model, (dummy_input,), dynamic_shapes={"x": {0: batch}})
torch.export.save(ep, "model.pt2")
```

Verify that the exported model produces the same outputs as the original. Models are hosted on [Hugging Face Hub](https://huggingface.co/tomas-gajarsky).

For broader PyTorch compatibility, publish recommended version cohorts in the same repo:

- `model-torch2.3.pt2`
- `model-torch2.6.pt2`
- `model-torch2.11.pt2`
- (optional compatibility fallback) `model.pt2`

From a source checkout, export, validate, and upload all facetorch model cohorts for the current torch runtime with:

```bash
PYTHONPATH=. python scripts/export_model_cohorts_hf.py export \
  --repo-root . \
  --out-root /tmp/model-cohort-exports \
  --validate-devices cpu,cuda \
  --upload \
  --hf-token-env HF_TOKEN
```

To re-validate existing artifacts against reference models on multiple inputs and batch sizes:

```bash
PYTHONPATH=. python scripts/export_model_cohorts_hf.py validate \
  --repo-root . \
  --artifacts-root /tmp/model-cohort-exports/upload26 \
  --validate-devices cpu,cuda \
  --cohort 2.6 \
  --batch-sizes 1,2,4,8 \
  --seeds 0,17 \
  --scales 1.0,0.25
```

Use `--model-ids` (for example `--model-ids verify-magface`) to process only a subset.
The script writes a `.meta.json` file next to each artifact and fails the run if validated outputs exceed the configured numerical tolerances.
Export-only architecture definitions live in `model_defs/`; they are included for reproducible re-exporting, but they are not required for normal `.pt2` inference.

#### Configuration
##### Create yaml file
1. Create new folder with a short name of the task in predictor configuration directory 
```/conf/analyzer/predictor/``` following the FER example in ```/conf/analyzer/predictor/fer/```
2. Copy the yaml file ```/conf/analyzer/predictor/fer/efficientnet_b2_8.yaml``` to the new folder 
```/conf/analyzer/predictor/<predictor_name>/```
3. Change the yaml file name to the model you want to use: 
```/conf/analyzer/predictor/<predictor_name>/<model_name>.yaml```

##### Edit yaml file
1. Set up the downloader configuration:
   - For Hugging Face Hub (recommended): specify the `repo_id` and `filename` parameters
   - For legacy Google Drive (deprecated): specify the Google Drive file ID
2. Select the preprocessor (or implement a new one based on BasePredPreProcessor) and specify its parameters e.g. image size and normalization in the yaml file 
to match the requirements of the new model.
3. Select the postprocessor (or implement a new one based on BasePredPostProcessor) and specify its parameters e.g. labels in the yaml file to match 
the requirements of the new model.
4. (Optional) Add BaseUtilizer derivative that uses output of your model to perform some additional actions.

##### Configure tests
1. Add a new predictor to the main *config.yaml* and all *tests.config.n.yaml* files. Alternatively, create a new config file e.g. 
*tests.config.n.yaml* and add it to the ```/tests/conftest.py``` file.
2. Write a test for the new predictor in ```/tests/test_<predictor_name>.py```

#### Test and submit
1. Run linting: ```black facetorch```
2. Add the new predictor to the README model table.
3. Update CHANGELOG and version
4. Submit a pull request to the repository


### Update environment

#### Dependency ownership and release channels
* `pyproject.toml` is the packaging source of truth for PyPI releases and pip/uv installs (including Docker build paths using uv).
* Conda package publishing (`conda-forge/facetorch`) is maintained outside this repository in conda-forge feedstock workflows.
* `environment.yml` and `gpu.environment.yml` are conda environment baselines for conda users.
* The GPU conda baseline uses conda-forge `cuda-version=12.4` instead of `cudatoolkit`; pass `--with-cuda` when regenerating the GPU lock so conda-lock can resolve CUDA virtual packages without requiring a local GPU.
* Overlapping dependencies between pyproject and conda env files are intentionally kept aligned.
* CI enforces this with: `python scripts/check_dependency_sync.py`.

#### uv (used by Docker dev/test images)
* Add packages with corresponding versions to ```pyproject.toml``` dependencies
* Lock the environment: ```uv lock```
* Sync the environment: ```uv sync --extra dev```

#### conda (for conda-forge users)
CPU:
* Add packages with corresponding versions to ```environment.yml``` file
* Lock the environment: ```conda-lock -p linux-64 -f environment.yml --lockfile conda-lock.yml```
* (Alternative Docker) Lock the environment: ```docker compose -f docker-compose.dev.yml run facetorch-lock```
* Install the locked environment: ```conda-lock install --name env conda-lock.yml```

GPU:
* Add packages with corresponding versions to ```gpu.environment.yml``` file
* Lock the environment: ```conda-lock --with-cuda 12.4 -p linux-64 -f gpu.environment.yml --lockfile gpu.conda-lock.yml```
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
