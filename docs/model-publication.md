# Model cohort publication

Model publication is a two-phase release operation. Export and validation never
write to a remote repository. Publication accepts only an exact, reviewed staging
plan and moves the manifest last.

## 1. Stage and validate

Run the exporter independently in every required PyTorch environment. Release
validation must request every required device; a missing CUDA device is a failure,
not a successful CPU-only result.

First materialize source inputs from the manifest-pinned legacy objects. The
command verifies revision, size, and SHA-256 before any TorchScript load and writes
a source inventory:

```bash
PYTHONPATH=. python scripts/export_model_cohorts_hf.py prepare-sources \
  --repo-root . --cohort 2.11
```

```bash
PYTHONPATH=. python scripts/export_model_cohorts_hf.py export \
  --repo-root . \
  --out-root /secure/staging/torch-2.11 \
  --validate-devices cpu,cuda \
  --batch-sizes 1,2,4,8
```

The validator rejects empty matrices; recursively rejects NaN and Inf in both the
independent reference and exported output; enforces output schemas and task
invariants; and requires every requested device status to be `ok`. Predictor cases
cover face batches 1, 2, 4, and 8. The detector remains single-image and covers its
declared standard and nonstandard multiple-of-32 spatial shapes. All immutable
references execute on CPU so CPU and CUDA artifacts share identical golden output.
The validator disables TensorFloat-32, uses highest float32 precision and
deterministic cuDNN settings, records that policy, and restores the caller's
settings afterward. AU reference batches are concatenated from independent
one-face calls because the legacy trace's native multi-face behavior is coupled.

The default two seeds, two scales, two input variants, every required batch/shape,
and both devices are release requirements. The matrix verifier rejects shortened
smoke summaries even if all executed cases happened to pass.

Do not use `--upload`: it is intentionally disabled. Collect all required cohort
summaries under one protected staging root before continuing.

Before publication planning, verify that both declared cohorts contain all
ten models and successful CPU/CUDA cases. Candidate diagnostics may use
`--candidate-evidence`; a release run must not, and must be produced from a clean
immutable commit:

```bash
PYTHONPATH=. python scripts/verify_model_release_matrix.py \
  --staging-root /secure/staging \
  --summary /secure/staging/torch-2.6/summary-torch2.6.json \
  --summary /secure/staging/torch-2.11/summary-torch2.11.json
```

`scripts/audit_model_manifest_hf.py` separately verifies immutable Hub revisions,
LFS SHA-256 object IDs, sizes, metadata presence, and optionally every downloaded
artifact byte. Legacy metadata may be inventoried, but cannot satisfy publication.

## 2. Prepare a deterministic plan

Create a JSON map from each model ID (or repository ID) to the immutable
40-character commit from which its candidate branch must start. Parent revisions
must be reviewed; mutable branch names are rejected.

```bash
PYTHONPATH=. python scripts/model_cohort_publication.py prepare \
  --staging-root /secure/staging \
  --summary torch-2.6/summary-torch2.6.json \
  --summary torch-2.11/summary-torch2.11.json \
  --base-revisions /secure/review/base-revisions.json \
  --manifest-repo-id OWNER/facetorch-model-manifest \
  --manifest-base-revision 0123456789abcdef0123456789abcdef01234567 \
  --plan /secure/review/publication-plan.json \
  --approval-template /secure/review/publication-approval.json
```

Preparation re-verifies every artifact, metadata file, size, digest, model, cohort,
case count, and device status. The canonical plan ID determines a unique candidate
branch. For repeated model/device/case identities across cohorts, it also requires
the same immutable reference-output fingerprint and records either exact exported
agreement or the mathematically guaranteed drift bound through that reference.
Re-running preparation over unchanged inputs produces identical plan bytes.

## 3. Review and approve

Review the complete plan, validation summaries, provenance/rights evidence, and
artifact costs. Change the generated approval document from `pending` to
`approved`, identify the reviewer, and add a timezone-qualified ISO 8601 approval
time. Do not change its plan ID or plan SHA-256. A changed plan or staged byte
invalidates the approval. The final release still requires the independent-human
approval defined by the project release policy; an AI review is supplementary.

Verify locally before granting network credentials:

```bash
PYTHONPATH=. python scripts/model_cohort_publication.py verify \
  --plan /secure/review/publication-plan.json \
  --approval /secure/review/publication-approval.json
```

## 4. Publish and resume safely

```bash
HF_TOKEN=... PYTHONPATH=. python scripts/model_cohort_publication.py publish \
  --plan /secure/review/publication-plan.json \
  --approval /secure/review/publication-approval.json \
  --receipt /secure/review/publication-receipt.json
```

All cohorts for one model and their metadata are one repository commit on the
plan-specific candidate branch. The receipt is atomically updated after each
successful model. If a later upload fails, no manifest is created; rerun the same
command with the same plan, approval, and receipt. Completed immutable commits are
verified and skipped. A mismatched receipt, remote commit, local byte, or parent
revision stops publication.

Only after every model commit succeeds is a candidate manifest committed to the
separate manifest repository. The receipt records its immutable revision. This tool
does not update a stable alias, default branch, package manifest, or public release;
those later promotion steps must consume the reviewed immutable revision.
