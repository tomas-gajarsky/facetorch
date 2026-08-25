# Change Log

## 1.0.0rc1 (Unreleased)

This is the v1 release candidate change set. The section becomes released only
when the immutable v1.0.0-rc.1 tag and all publication receipts have been verified.

### Breaking Changes
* Minimum Python version raised from 3.8 to 3.10
* Supported Python range is explicitly capped at `<3.13` for Python 3.10-3.12
* PyTorch support is bounded to the explicit 2.6.x and 2.11.x cohort lines;
  2.3-2.5 and 2.7-2.10 are rejected before model download
* Default models migrated from TorchScript (`.pt`) to `torch.export` (`.pt2`);
  verified TorchScript artifacts now require explicit `allow_legacy_models=True`
* Google Drive model configs are deprecated and fail closed without immutable size
  and digest metadata
* `path_image` and `tensor` parameters in `FaceAnalyzer.run()` are deprecated in favor of `image_source`

### Added
* Selective predictor execution via `include_predictors` and `exclude_predictors` parameters in `FaceAnalyzer.run()`
* Pre-cropped face input support via `skip_detector=True` parameter in `FaceAnalyzer.run()`
* Grayscale image handling: automatic conversion of single-channel and RGBA inputs to RGB across all input paths
* `__call__` methods on `FaceAnalyzer`, `BaseProcessor`, `BaseDownloader`, and `BaseModel` (delegates to `run()`)
* Optional logger configuration: `FaceAnalyzer` falls back to `logging.getLogger("facetorch")` when no logger is configured
* Robust input routing in `FaceAnalyzer.run()` — tensor, numpy array, PIL Image, bytes, and file path inputs work with any reader type
* All .pt2 models uploaded to Hugging Face Hub with model cards
* Versioned model-artifact manifest with immutable Hub revisions, byte sizes,
  SHA-256 digests, real formats, runtime/schema cohorts, and device eligibility
* Verified atomic downloads, concurrent first-use locking, cache quarantine,
  explicit offline mode, and persistent schema-incompatibility records
* Cost-planning and explicit prefetch APIs plus non-executing legacy inspection,
  exact-hash migration, quarantine reporting, and confirmed cleanup APIs
* Cohort export/validation/upload script: `scripts/export_model_cohorts_hf.py`
* Device-aware cohort validation in export script via `--validate-devices` (for example `cpu,cuda`)
* Fail-closed cohort validation for recursive non-finite outputs, strict state
  reconstruction, complete batch/shape/device matrices, output schemas, and
  task-level invariants
* Digest-bound, review-gated model publication plans with per-model atomic Hub
  commits, resumable receipts, and manifest-last promotion
* Machine-readable compatibility and per-model governance/limitations records;
  incomplete weight rights or source-checkpoint mapping block manifest approval
* Immutable Hub object auditor and complete local CPU/CUDA cohort-matrix verifier
* CPU-golden model validation with a recorded highest-precision float32 policy,
  deterministic cuDNN settings, and TensorFloat-32 disabled
* Export-only `model_defs/` architecture definitions for reproducible `.pt2` cohort generation without making them runtime model dependencies
* Dependency alignment check script: `scripts/check_dependency_sync.py`
* `uv.lock` for reproducible PyPI-based dependency resolution
* `[tool.uv]` configuration in `pyproject.toml`
* Exact CPU and CUDA uv lock profiles for every supported Torch cohort, refreshed
  CPU/GPU conda locks, dependency advisory gating, and CycloneDX SBOM generation
* Tiered CI for source tests, Python 3.10-3.12 wheel installs, Torch 2.6/2.11
  CPU cohorts, branch-wheel conda validation, and frozen production images
* Owner-only protected-commit local GPU workflow covering all cohort devices, the
  installed default analyzer, public notebook, and both production images

### Changed
* Migrated from `setup.py` + `version` file to `pyproject.toml` (PEP 621)
* All model files migrated from TorchScript (.pt) to torch.export (.pt2) portable format with dynamic batch support
* Model artifact strategy changed to prioritize portability and install-time simplicity over TorchScript-specific runtime behavior: `.pt2` artifacts do not require bundled model source code, while versioned cohorts handle PyTorch exported-program schema differences across supported torch runtimes (`2.6`, `2.11`)
* Built-in Hugging Face model configs now select exactly one declared compatible
  manifest artifact; they no longer synthesize filenames or cascade downloads
* AU predictor model rewritten with timm Swin Transformer backbone for torch.export compatibility
* Build metadata now uses PEP 639 SPDX license fields with `setuptools>=77.0.3`, removing the setuptools license-table deprecation warning
* uv PyTorch index configuration now uses explicit `torch`/`torchvision` sources instead of a global extra index, so non-PyTorch packages resolve from PyPI by default
* Docker images no longer set a global uv PyTorch extra index; GPU images install CUDA torch wheels explicitly with uv's `--torch-backend cu124` after the base install/sync step
* Docker dev/test images no longer read or write Python bytecode from bind-mounted source trees
* Docker dev/test images use container-local model and coverage caches for tests, avoiding root-owned artifacts in bind-mounted checkouts
* Docker build contexts now exclude local model/export artifacts that are not part of release images
* Docker dev/test images migrated from conda/conda-lock to [uv](https://github.com/astral-sh/uv) for faster builds
* Docker production images now install facetorch from the checked-out release source with uv instead of racing against the PyPI publication job
* Development dependencies consolidated from `requirements.dev.txt` into `pyproject.toml`
* Development release-validation dependencies now include `build` and `packaging>=25.0` for Metadata 2.4 / PEP 639 checks
* Replaced the misleading open-ended Torch dependency with a bounded disjoint
  specifier matching the two declared artifact cohorts
* Removed the Torch 2.3 cohort because GHSA-53q9-r3pm-6pq6 is a critical
  `torch.load(weights_only=True)` remote-code-execution advisory
* Retained the validated Torch 2.6/CUDA 12.4 cohort under three exact moderate
  advisory exceptions, limited to unused APIs and expiring on 2026-11-20
* Docker base images updated to Python 3.12 and CUDA 12.4
* Production images now build and install the exact branch wheel as a non-root
  user from the same locked Torch 2.6 CPU/CUDA 12.4 profiles used in CI
* Candidate Python support narrowed to 3.10, 3.11, and 3.12 pending final CI lanes
* Predictor cohort validation defines batches as independent faces from one image;
  multi-image batching remains outside the v1 API
* GPU environment updated from CUDA 11.2 to CUDA 12.4+
* Development status remains Beta through the approved release-candidate soak
* Google Colab notebook updated to v1.0.0 (uses `image_source`, removes pinned torch versions)

### Fixed
* "File name too long" error when passing tensor/array to `FaceAnalyzer.run()` with `ImageReader`
* AU predictor YAML indentation error in merged config files
* Numpy array reader now handles (H, W) and (H, W, 1) grayscale arrays
* AU `.pt2` CUDA device mismatch by re-exporting validated AU cohorts (`2.6`, `2.11`) and publishing refreshed Hugging Face artifacts with metadata
* Exported-model schema mismatches are recorded per manifest/runtime/device so the
  same incompatible artifact is not downloaded or executed repeatedly
* Export cohort validation now fails on numerical drift beyond configured max/mean absolute-difference tolerances instead of only recording comparison metrics
* Model export no longer uploads inline; a late validation failure cannot publish
  an early model, and requested CUDA cannot be hidden by a successful CPU result
* Detector cohorts recover exact legacy TorchScript BatchNorm attributes before a
  strict native load and support dynamic multiple-of-32 spatial inputs
* AU validation compares batched artifacts with concatenated one-face golden calls,
  avoiding the legacy trace's batch-coupled behavior
* Test configs now resolve `/opt/facetorch` paths to the checked-out repository during local pytest runs, and honor `FACETORCH_TEST_MODEL_ROOT` for isolated model caches
* `python-json-logger` v4 import deprecation warning by using the new module path with fallback for older versions


## 0.6.2

Released on April 17, 2026.

### Fixed
* AU predictor CUDA deadlock with PyTorch >= 2.0 and CUDA >= 12.0 by loading model as native PyTorch nn.Module instead of TorchScript

### Added
* Native PyTorch implementation of OpenGraphAU model (Swin Transformer backbone + GNN head)
* Support for loading native PyTorch models via `native_model_class` parameter in BaseModel
* `timm` dependency for model utilities (DropPath, to_2tuple, trunc_normal_)

### Changed
* AU predictor device restored from forced CPU to configurable device (CUDA support re-enabled)


## 0.6.1

Released on April 14, 2026.

### Fixed
* PostArgMax post-processor to handle tuple inputs (resolves TypeError: argmax(): argument 'input' must be Tensor, not tuple)
* PostSigmoidBinary post-processor to handle tuple inputs for consistency with other post-processors

### Changed
* Replaced pypi-publish and docker-push workflows with unified release workflow triggered by GitHub Release
* Switched conda CI from miniconda (classic solver) to miniforge (libmamba), reducing run time from ~60 min to ~1 min
* Updated all GitHub Actions to latest versions (checkout@v4, setup-python@v5, setup-miniconda@v3)
* Removed non-working paperswithcode badges from README for better readability

### Added
* Unit tests for all post-processor tuple input handling
* Version tag validation in release workflow
* Auto-release workflow that creates GitHub Releases when version changes on main


## 0.6.0

Released on May 24, 2025.

### Added
* DownloaderHuggingFace for downloading models from Hugging Face Hub

### Changed
* default model download source from Google Drive to Hugging Face Hub


## 0.5.1

Released on November 17, 2024.

### Changed

* UnversalReader to read PIL images as RGB
* UniversalReader to read numpy arrays to torch directly
* RetinaFace pre-normalization color space to RGB
* torch.cross torch.linalg.cross in 3D landmark drawer


## 0.5.0

Released on February 11, 2024.

### Added
* UniversalReader for loading data like PIL images, numpy arrays, torch tensors, bytes, urls, and file paths

### Changed
* Enable conda virtual environment by default in Docker images
* FaceAnalyzer run method to accept various input types via image_source parameter


## 0.4.2

Released on January 30, 2024.

### Added
* Tensor input support for FaceAnalyzer run method
* TensorReader for transforming incoming torch tensors


## 0.4.1

Released on December 14, 2023.

### Changed
* postprocessor for label confidence pairs to have no offset by default
* Resize transform configs to enable antialiasing by default
* notebook to version 0.4.0 or higher
* notebook to include Action Unit and Valence Arousal predictors


## 0.4.0

Released on December 13, 2023.

### Added
* predictor for facial valence arousal - ELIM AL from Kim et al.
* predictor postprocessor for creating label confidence pairs

### Changed
* FaceAnalyzer can run without any predictors or utilizers


## 0.3.1

Released on December 10, 2023.

### Added
* link to ChatGPT guide that has knowledge about facetorch

### Fixed
* stuck AU predictor inference in the notebook by specifying torch version to install


## 0.3.0

Released on February 9, 2023.

### Added
* predictor for facial action unit detection - OpenGraphAU from Luo et al.

### Changed
* facetorch version retrieval error handling to not fail the run


## 0.2.4

Released on February 4, 2023.

### Changed
* scope of some test fixtures from function to session


## 0.2.3

Released on February 2, 2023.

### Fixed
* error in detector test

### Removed
* deletion of detector model in tests


## 0.2.2

Released on January 28, 2023.

### Fixed
* error in Google Colab demo notebook


## 0.2.1

Released on January 28, 2023.

### Added
* Google Colab demo notebook
* Google Colab demo notebook link to README
* Merged config yaml files for OmegaConf to /conf/merged/ directory

### Changed
* Do not fail the run if facetorch version cannot be retrieved (likely due to local installation)


## 0.2.0

Released on January 28, 2023.

### Added
* Python 3.8 support
* platform to Docker compose file for Apple Silicon M chips
* Docker compose service for locking dependencies


## 0.1.5

Released on January 22, 2023.

### Added
* number of parameters for each model in README

### Changed
* facetorch installation in dev Dockerfiles to not install dependencies from PyPI

### Removed
* unused port bindings from dev Docker compose file


## 0.1.4

Released on November 18, 2022.

### Added
* predictor for face verification and face recognition - AdaFace by Kim et al.


## 0.1.3

Released on November 13, 2022.

### Added
* Torchvision based landmark drawer as a utilizer

### Changed
* drawing of 3D landmarks is 21x faster: 152 ms -> 7 ms

### Removed
* Matplotlib based landmark drawer utilizer
* Matplotlib dependency


## 0.1.2

Released on November 10, 2022.

### Added
* links to Demo App and User Guide in README.md

### Changed
* Box expansion method changed from static absolute value to dynamic relative value


## 0.1.1

Released on August 31, 2022.

### Added
* predictor for face verification task - MagFace+UNPG by Jung et al.


## 0.1.0

Released on August 22, 2022.

### Added
* badges for models with SOTA comparison using papers with code
* predictor for face alignment task - SynergyNet
* utilizers for face alignment (compute 3d landmarks, mesh and pose) and drawing landmarks

### Changed
* default logging level to INFO
* drawing boxes and saving image abstracted to utilizer objects


## 0.0.8

Released on August 12, 2022.

### Added

* conda-forge documentation
* entrypoints to docker compose services

### Changed

* workflow for testing conda installation
* workflow for pushing facetorch-gpu Docker image to Docker Hub


## 0.0.7

Released on August 11, 2022.

### Added

* facetorch GitHub repository link to docs
* secrets to workflows that push to PyPI and Docker Hub


## 0.0.6

Released on August 10, 2022.

### Added

* GitHub workflows


## 0.0.5

Released on August 7, 2022.

### Added

* GPU specific conda environment


## 0.0.4

Released on August 6, 2022.

### Added

* Manifest file for distribution


## 0.0.3

Released on August 5, 2022.

### Added

* Logo


## 0.0.1

Released on August 5, 2022.

### Added

* First version of facetorch package, containing:
	- Tests,
	- Documentation,
	- Code style checking
	- Contributing guidelines
