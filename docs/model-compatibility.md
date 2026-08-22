# Model compatibility and governance

Facetorch v1 uses a bounded release-candidate matrix. Package metadata and runtime
routing accept only the PyTorch minor lines for which the package has a distinct
artifact cohort. Similar export-schema numbers are not treated as compatibility
proof.

| Python | PyTorch line | Export schema | Artifact cohort | Candidate CUDA runtime |
| --- | --- | --- | --- | --- |
| 3.10-3.12 | 2.3.x | 5.1 | 2.3 | 12.1 |
| 3.10-3.12 | 2.6.x | 8.2 | 2.6 | 12.4 |
| 3.10-3.12 | 2.11.x | 8.17 | 2.11 | 13.0 |

The dependency is deliberately expressed as
`torch>=2.3,<2.12,!=2.4.*,!=2.5.*,!=2.7.*,!=2.8.*,!=2.9.*,!=2.10.*`.
Torch 2.4, 2.5, and 2.7-2.10 fail before model download even when legacy models
are explicitly enabled. In particular, Torch 2.5 uses export schema major 7 and
has no approved facetorch cohort. New lines are added only with a complete
artifact and validation matrix; an upper bound prevents a future resolver choice
from silently expanding the support claim.

The candidate's official platform target is Linux x86-64 on CPU and the named
NVIDIA CUDA pairs above. Windows, macOS, Linux ARM, and Apple MPS are experimental
until exercised. The matrix becomes an official release claim only after the exact
clean candidate passes every model on CPU and CUDA for all three rows. Current lane
status is machine-readable in `facetorch/models/compatibility.json`.

## Candidate evidence

On 2026-08-21, all three rows were exercised on Linux x86-64 with Python 3.10
and an NVIDIA GeForce RTX 3090. All 30 cohort artifacts passed: 1,872 cases in
total across every model, CPU and CUDA, face batches 1/2/4/8, two seeds, two
input scales, and normal/uniform inputs. Detector input batch remains one image;
its validated spatial sizes are `480x640`, `512x512`, and `480x608`, matching the
runtime's multiple-of-32 padding contract. The worst reference difference was
`6.06e-5` maximum absolute and remained within the declared bounds.

This is candidate evidence from an uncommitted tree, not release approval. It
must be repeated from the exact clean release commit. No artifact was uploaded,
the packaged manifest still names the current remote objects, and governance is
still incomplete.

## Validation semantics

Every immutable TorchScript reference is kept on CPU and acts as one golden
implementation for both exported CPU and CUDA execution. Validation disables
TensorFloat-32, selects highest float32 matmul precision, and enables deterministic
cuDNN behavior while restoring the caller's backend settings afterward. This
avoids treating TorchScript's own backend-dependent drift as artifact drift.

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

All ten records currently remain `release_eligible: false`: hosted legacy-object
digests are pinned, but the link to the original upstream checkpoint and the
weight redistribution terms have not yet been approved by the provenance owner.
Several README/model-card source descriptions disagree, and the valence/arousal
model card links an unavailable path while the project README links a different
repository. These are release blockers, not warnings. The runtime artifact
manifest cannot be changed to `approved` while any governance record is incomplete.

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
complete candidate matrix without claiming governance approval:

```bash
PYTHONPATH=. python scripts/verify_model_release_matrix.py \
  --staging-root /secure/staging \
  --summary /secure/staging/torch-2.3/summary-torch2.3.json \
  --summary /secure/staging/torch-2.6/summary-torch2.6.json \
  --summary /secure/staging/torch-2.11/summary-torch2.11.json \
  --candidate-evidence --allow-dirty-source
```

The relaxed flags are diagnostic only. Release verification omits both flags and
therefore requires a clean immutable source commit plus approved compatibility,
governance, and artifact manifests.
