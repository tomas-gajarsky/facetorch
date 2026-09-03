# Model compatibility and governance

Facetorch v1 uses a bounded release-candidate matrix. Runtime support and artifact
export cohorts are separate concepts: one digest-pinned exported program may serve
several PyTorch lines only after every line has passed CPU and CUDA validation.
Similar export-schema numbers alone are not treated as compatibility proof.

| Python | PyTorch line | Export schema | Artifact cohort | Candidate CUDA runtime |
| --- | --- | --- | --- | --- |
| 3.10-3.12 | 2.6.x | 8.2 | 2.6 | 12.4 |
| 3.10-3.12 | 2.7.x | 8.2 artifact | 2.6 | 12.6 |
| 3.10-3.12 | 2.8.x | 8.2 artifact | 2.6 | 12.6 |
| 3.10-3.12 | 2.9.x | 8.17 artifact | 2.11 | 13.0 |
| 3.10-3.12 | 2.10.x | 8.17 artifact | 2.11 | 13.0 |
| 3.10-3.12 | 2.11.x | 8.17 | 2.11 | 13.0 |
| 3.10-3.12 | 2.12.x | 8.17 artifact | 2.11 | 13.0 |
| 3.10-3.12 | 2.13.x | 8.17 artifact | 2.11 | 13.0 |

The dependencies are deliberately expressed as `torch>=2.6,<2.14` and
`torchvision>=0.21,<0.29`. Torch 2.3-2.5 and 2.14 or newer fail before model
download even when legacy models are explicitly enabled. In particular, Torch 2.5
uses export schema major 7 and has no approved Facetorch cohort. The upper bound
prevents a future resolver choice from silently expanding the support claim.

PyTorch 2.8 emits an upstream deprecation warning when it reads the older PT2
archive container used by the 2.6 cohort. The archive still loads and its outputs
pass the declared tolerances. PyTorch 2.9 is the routing boundary because it loads
the newer 2.11 cohort cleanly, whereas PyTorch 2.8 cannot read that newer archive.

The candidate's official platform target is Linux x86-64 on CPU and the named
NVIDIA CUDA pairs above. Windows, macOS, Linux ARM, and Apple MPS are experimental
until exercised. The matrix becomes an official release claim only after the exact
clean candidate passes every model on CPU and CUDA for all eight rows. Current lane
status is machine-readable in `facetorch/models/compatibility.json`.

## Security support boundary

Torch 2.3 was removed because GHSA-53q9-r3pm-6pq6 is a critical
remote-code-execution issue in `torch.load(weights_only=True)`, an operation used
when Facetorch reads authenticated state dictionaries and metadata. Digest-pinned
artifacts reduce exposure but do not justify retaining a critically affected
runtime as a supported public cohort.

Torch 2.6 remains available for the validated CUDA 12.4 production lane under
three founder-approved moderate exceptions: GHSA-887c-mr87-cxwp,
GHSA-vgrw-7cvw-pwgx, and GHSA-f4hp-rmr7-r7v8. They affect `ctc_loss`,
`unpack_sequence`, and `pad_packed_sequence`, respectively; Facetorch does not call
those APIs. The exceptions are restricted to the exact Torch 2.6 CPU/CUDA profiles
and expire on 2026-11-20. The machine-readable policy and removal conditions are in
`security/advisory-exceptions.json`.

Torch 2.11.0 and 2.12.1 currently constrain their runtime dependency to
`setuptools<82`. Those exact profiles retain the existing Linux-only exception for
CVE-2026-59890 through 2026-11-20. Facetorch does not build source distributions
at runtime, and release source distributions use the isolated setuptools 84 build
backend. Torch lines without that upstream constraint resolve setuptools 84.

## Candidate evidence

On 2026-08-21, the two exported cohorts were exercised as part of a larger diagnostic
on Linux x86-64 with Python 3.10 and an NVIDIA GeForce RTX 3090. Their 20 cohort
artifacts passed 1,248 cases across every model, CPU and CUDA, face batches
1/2/4/8, two seeds, two input scales, and normal/uniform inputs. Detector input
batch remains one image;
its validated spatial sizes are `480x640`, `512x512`, and `480x608`, matching the
runtime's multiple-of-32 padding contract. Every retained case remained within the
declared numeric bounds.

The original diagnostic was superseded on 2026-08-25 by the complete matrix from
clean commit `4aac25033cbafd836d32351e8fe9bc6c0e088ed5`. Its 20 artifacts and
schema-2 validation records were published through the digest-approved plan, and
the final Hub audit verified their immutable LFS identities, sizes, metadata, and
legal documents. Compatibility and the packaged artifact manifest are therefore
approved. This model-artifact approval is distinct from the coordinated RC1
release, which must still run its protected dry run from the exact final source
commit.

On 2026-09-01, the public RC2 wheel and all ten existing artifacts were then tested
from an independent directory on PyTorch 2.6 through 2.13, on CPU and an RTX 3090.
All eight preferred routes loaded and completed real four-face inference within
the published tolerances, and every downloaded artifact matched its manifest
SHA-256. This established the reusable routing boundary above. It was a focused
compatibility probe, not a substitute for the complete synthetic release matrix;
RC3 publication therefore re-runs every batch, seed, scale, input variant, model,
and device through the protected exact-candidate workflow.

## Validation semantics

Every immutable TorchScript reference is kept on CPU. Torch 2.6 is the declared
golden-reference cohort: it records one digest-bound output bundle for the full
case matrix, and every CPU/CUDA runtime lane must reuse those exact
outputs. Validation disables
TensorFloat-32, selects highest float32 matmul precision, and enables deterministic
cuDNN behavior while restoring the caller's backend settings afterward. This
avoids treating TorchScript's runtime-dependent drift as artifact drift and gives
cross-cohort triangle-inequality bounds one immutable reference.

Predictor batches contain independent faces from one source image; multi-image
batching is not a v1 API. The legacy AU trace has batch-coupled behavior, so AU's
golden output is explicitly the concatenation of one-face reference calls. Its
digest-pinned published programs satisfy that contract and are preserved until
the original native checkpoint mapping is recovered. Detector BatchNorm tensors
omitted from the old `state_dict()` are recovered exactly from verified
TorchScript attributes before strict native reconstruction; no value is invented.

## Cache verification and incompatibility recovery

Authenticated downloaders verify artifact size, SHA-256, and archive format on
first write and, by default, again before every process loads an existing cache
entry. This fail-closed default detects corruption or replacement in shared and
mutable caches. Operators using a trusted, read-only cache may explicitly set a
downloader's `verify_on_use: false` to avoid the additional digest pass; doing so
accepts responsibility for protecting the cache outside facetorch. Release and
validation configurations must retain `verify_on_use: true`.

When PyTorch rejects an export schema, facetorch records the artifact/runtime/device
combination in `.incompatible.json` so later processes do not repeatedly execute
the same incompatible bytes. Inspect and reset those records only after changing
the runtime or correcting the artifact:

```python
from facetorch import inspect_incompatible_cache, reset_incompatible_cache

print(inspect_incompatible_cache())
reset_incompatible_cache(confirm=True)
```

Reset is restricted to the versioned facetorch model cache. It does not delete
model artifacts, and retrying without resolving the incompatibility will recreate
the record.

## Model provenance and limitations

`facetorch/models/governance.json` contains one record for every downloadable
model. It separates an upstream code license from rights to a checkpoint: a code
repository's MIT or Apache license does not by itself prove that converted weights
may be redistributed. Each record includes immutable upstream revision evidence,
source-checkpoint mapping status, weight-license and redistribution status,
attribution status, intended use, and task-specific limitations.

All ten records are currently `release_eligible: true`. Each one binds the hosted
weights to pinned upstream checkpoint evidence, preserves the upstream MIT or
Apache-2.0 license without conversion, records attribution and redistribution
approval, and documents intended use and limitations. Release eligibility covers
only the listed artifacts under the recorded owner-approved policy; it does not
license upstream datasets or waive any deployment-specific obligation. Artifact
release eligibility also does not by itself publish the Facetorch package or
satisfy the coordinated release pipeline.

No face-analysis output should be the sole basis for a consequential decision.
Verification scores are not proof of identity; expression, action-unit,
valence/arousal, and alignment outputs do not establish a person's internal state,
intent, health, truthfulness, or protected characteristics. Deployments must
address consent, privacy, domain shift, demographic performance, and applicable
law independently.

## Evidence commands

Prepare source inputs only from immutable, digest-verified Hub objects:

```bash
PYTHONPATH=. python scripts/export_model_cohorts_hf.py prepare-sources \
  --repo-root . --cohort 2.11
```

Inventory the immutable remote objects and identify legacy validation metadata:

```bash
PYTHONPATH=. python scripts/audit_model_manifest_hf.py \
  --allow-legacy-metadata
```

After all cohort summaries exist beneath one protected staging root, verify the
two exported artifact cohorts without claiming governance approval:

```bash
PYTHONPATH=. python scripts/verify_model_release_matrix.py \
  --staging-root /secure/staging \
  --summary /secure/staging/torch-2.6/summary-torch2.6.json \
  --summary /secure/staging/torch-2.11/summary-torch2.11.json \
  --candidate-evidence --allow-dirty-source
```

The protected local runner additionally validates the two staged cohorts through
all eight runtime profiles and verifies those summaries with
`scripts/verify_runtime_compatibility_matrix.py`:

```bash
python scripts/run_local_cuda_release_matrix.py \
  --repo-root . --source-sha "$(git rev-parse HEAD)" \
  --staging-root /secure/staging
```

The relaxed flags are diagnostic only. Release verification omits both flags and
therefore requires a clean immutable source commit plus approved compatibility,
governance, and artifact manifests.
