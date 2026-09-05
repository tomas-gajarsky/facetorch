# Model compatibility and governance

Facetorch v1 uses a bounded release-candidate matrix. Runtime support and artifact
export cohorts are separate concepts: one digest-pinned exported program may serve
several PyTorch lines only after every line has passed CPU and CUDA validation.
Similar export-schema numbers alone are not treated as compatibility proof.
The [v1 support policy](torch-support-policy.md) records the owner’s decision
to retain all eight lines, recommend Torch 2.13 for new installations, preserve
a CUDA 12 path, and qualify newer runtimes before widening the bounds.

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

The approved exception policy currently contains nine exact-version records:
eight for Torch and one for setuptools, all expiring on 2026-11-20. They are
limited to the profiles and versions listed in
`security/advisory-exceptions.json`; support for an additional runtime does not
extend an exception to it.

The corrected audit on 2026-09-05 covered every active runtime dependency in all 17 lock
profiles. It found 18 distinct unresolved advisories, repeated as 76 entries
across twelve profiles. Root, Torch 2.11 CPU/CUDA, and Torch 2.13 CPU/CUDA had no
unresolved findings under the existing policy. This is a dated audit result,
not a claim that these runtimes have no vulnerabilities. GA remains blocked on
advisory treatment; see [the decision record](v1-advisory-triage.md).

Torch 2.11.0 and 2.12.1 constrain setuptools to `<82`; the existing scoped
setuptools exception covers the affected locked profiles. Runtime compatibility,
artifact-cohort approval, and dependency risk acceptance are separate gates.

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
approved. The coordinated RC1 release was aborted without publication. Artifact approval
is separate from a package release and its exact-source validation.

On 2026-09-01, the public RC2 wheel and all ten existing artifacts were then tested
from an independent directory on PyTorch 2.6 through 2.13, on CPU and an RTX 3090.
All eight preferred routes loaded and completed real four-face inference within
the published tolerances, and every downloaded artifact matched its manifest
SHA-256. This established the reusable routing boundary above. It was a focused
compatibility probe, not a substitute for the complete synthetic release matrix;
RC3 completed that full matrix on September 3, from clean source commit
`12db551d937ac2fa0cc41324f89d71fd9858fa02`: 4,992 runtime cases in eight
CPU/CUDA lanes, plus 1,248 artifact-cohort cases. The
[published RC3 evidence](https://github.com/tomas-gajarsky/facetorch/releases/tag/v1.0.0-rc.3)
is historical evidence for that commit. Each later candidate must rerun the
complete matrix and the corrected dependency gate.

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

## Cache lock upgrade

Shared writable caches use persistent POSIX `flock` files on a local filesystem.
Independent processes and containers sharing the same host filesystem serialize
on the same inode; the kernel releases the lock when the last owning descriptor
closes. Use the same service UID and suitable directory access for all workers.
Cross-host/network-filesystem locking is outside the validated guarantee.

Stop **all** workers before upgrading a shared RC3 cache. If an old lock directory
remains, remove only that directory after confirming every worker has stopped,
then upgrade every worker before restarting. Mixing RC3 directory-lock clients
with the new file-lock clients is unsupported. An existing directory raises
`CacheLockError` rather than guessing whether its PID is alive in another
container. Persistent lock files are normal; never delete or replace them while
any cache worker may be running. Windows lacks the required POSIX lock and remains
outside the supported platform matrix.
