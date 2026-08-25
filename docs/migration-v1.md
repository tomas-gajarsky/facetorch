# Migrating from facetorch 0.6.x to v1

This guide describes the v1 release-candidate contract. Keep `0.6.2` available
until the v1 candidate has passed your own workload and rollback rehearsal.

## Compatibility and installation

The v1 candidate supports Linux x86-64 with Python 3.10–3.12 and the declared
PyTorch cohorts documented in [model compatibility](model-compatibility.md).
CPU is the safe default profile; GPU remains supported by selecting
`load_config("gpu")` on a compatible CUDA host. Windows, macOS, ARM, and MPS
are experimental until separately validated. Install the exact candidate
version from the release instructions rather than relying on an unbounded
development checkout.

## Runtime API changes

```python
from facetorch import FaceAnalyzer, InputSpec, load_config

analyzer = FaceAnalyzer(load_config("gpu").analyzer)
result = analyzer.run("image.jpg", face_batch_size=8)
```

`image_source` is the preferred input. `path_image` and `tensor` remain
deprecated compatibility parameters. Successful calls return `AnalysisResult`
with stable fields; `run_legacy()` is the explicit, warning-emitting adapter for
code that still consumes the former `Response`/`ImageData` union.

The default input policy is permissive `coerce`. Use `input_policy="strict"`
with an `InputSpec` when a caller needs declared layout, range, color, or alpha
semantics. Conversion warnings are part of the public contract.

`face_batch_size` batches faces detected in one source image for predictors.
It does not batch multiple images. The old `batch_size` spelling is a
`DeprecationWarning` alias through v1.x; supplying both names is an error.

Predictors, the detector, and selection-linked utilizers are lazy. Empty
selection and `skip_detector=True` are supported, and unknown or duplicate
selection names fail before reading the image or downloading a model.
Utilizer dependencies are explicit in `analyzer.utilizer_dependencies`; a
dependent utilizer runs only after all of its declared predictors succeed. A
selected predictor requires a configured face unifier, including with
`skip_detector=True`; use `include_predictors=[]` for predictor-free processing.

## Models, cache, and offline deployment

v1 selects one digest-pinned artifact for the active PyTorch/device cohort.
Legacy TorchScript artifacts require the explicit `allow_legacy_models=True`
opt-in and are not selected for CUDA. Online preparation should use
`plan_model_prefetch()` followed by an explicit `prefetch_models(..., confirm=True)`.
For air-gapped use, populate the cache first and then use `load_config(offline=True)`;
missing or corrupt entries fail closed.

An export-schema rejection is remembered in the versioned cache. After upgrading
PyTorch or installing a corrected artifact, use `inspect_incompatible_cache()` and
then `reset_incompatible_cache(confirm=True)` to permit a retry. Resetting without
fixing the underlying compatibility problem simply recreates the record.

The v1 cache is versioned and separate from the 0.6.x cache. Migration is a
verified copy only: inspect legacy entries, migrate an exact manifest hash, and
never let the library rewrite or delete a 0.6.x cache automatically. Keep both
model roots available while testing rollback. Quarantined entries can be
inspected and explicitly cleaned with the cache APIs.

## Packaging, Docker, and rollback

The v1 wheel and CPU/GPU images are built from the same immutable candidate.
Do not assume a floating `latest` image is equivalent to a tested tag; record
the release tag and digest. Conda metadata may follow PyPI publication, so pin
the artifact source explicitly in deployment automation.

To roll back, stop promotion of `latest`, restore the last known-good v0.6.x
package/image and its separate model root, and document the incident. A broken
but usable Python release should be corrected with a patch release; yank only
an unusable or dangerous distribution. If a model or image must no longer be
trusted, publish an immutable revocation notice and block its digest in the
manifest rather than overwriting bytes.

## Deprecation and support

The v1.x line keeps the `batch_size` alias and legacy result adapter with
warnings. Removal requires a separately announced major-version policy. At v1
general availability, the project will publish the exact end date for the
approved six-month critical/security-only support window for 0.6.x.

Before upgrading, run the project regression suite against representative
images, review model-rights and privacy limitations, and retain a tested
rollback path. Face-analysis output must not be the sole basis for a
consequential decision.
