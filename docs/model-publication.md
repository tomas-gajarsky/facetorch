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
  --repo-root . --cohort 2.6
```

```bash
PYTHONPATH=. python scripts/export_model_cohorts_hf.py export \
  --repo-root . \
  --out-root /secure/staging/torch-2.6 \
  --validate-devices cpu,cuda \
  --batch-sizes 1,2,4,8 \
  --golden-reference-root /secure/staging/golden-references \
  --golden-reference-mode record \
  --golden-reference-cohort 2.6
```

Every other cohort must reuse those exact golden bytes:

```bash
PYTHONPATH=. python scripts/export_model_cohorts_hf.py export \
  --repo-root . \
  --out-root /secure/staging/torch-2.11 \
  --validate-devices cpu,cuda \
  --batch-sizes 1,2,4,8 \
  --golden-reference-root /secure/staging/golden-references \
  --golden-reference-mode reuse \
  --golden-reference-cohort 2.6
```

The validator rejects empty matrices; recursively rejects NaN and Inf in both the
independent reference and exported output; enforces output schemas and task
invariants; and requires every requested device status to be `ok`. Predictor cases
cover face batches 1, 2, 4, and 8. The detector remains single-image and covers its
declared standard and nonstandard multiple-of-32 spatial shapes. The declared
golden cohort executes every immutable reference once on CPU and records a
digest-bound tensor bundle. Every CPU/CUDA lane and later Torch cohort must reuse
that same bundle, so runtime-specific reference drift cannot be mistaken for
artifact agreement.
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

Preparation re-verifies every artifact, metadata file, golden-reference bundle,
size, digest, model, cohort, case count, and device status. Every tolerance and
comparison statistic must be present, finite, and nonnegative; each same-device
and cross-device case is checked independently against its declared maximum and
mean limits. The canonical plan ID determines a unique candidate branch. For
repeated model/device/case identities across cohorts, preparation also requires
the same immutable reference-output fingerprint and records either exact exported
agreement or mathematically guaranteed maximum and mean drift bounds through that
reference.
Re-running preparation over unchanged inputs produces identical plan bytes.

## 3. Review and approve

Review the complete plan, validation summaries, provenance/rights evidence, and
artifact costs. Change the generated approval document from `pending` to
`approved`, identify the reviewer, and add a timezone-qualified ISO 8601 approval
time. Do not change its plan ID or plan SHA-256. A changed plan or staged byte
invalidates the approval. The final release still requires the approval defined by
the project release policy. Independent human approval is preferred; for
`1.0.0rc1` through `1.0.0`, D20 permits `tomas-gajarsky` to approve as owner under
the documented bounded exception. Record that basis in the approval notes and do
not describe it as independent review. An AI review is supplementary.

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
verified and skipped only when their exact trees match and commit history proves
they are direct children of the approved parent. A mismatched receipt, remote
commit, local byte, or parent revision stops publication.

Only after every model commit succeeds is a candidate manifest committed to the
separate manifest repository. The receipt records its immutable revision. This tool
does not update a stable alias, default branch, package manifest, or public release;
those later promotion steps must consume the reviewed immutable revision.

## 5. Finalize legal contracts and package pins

Legal finalization is a second, separately approved transaction. First fetch the
currently packaged remote manifest at its exact revision and digest. The prepare
command is read-only: it verifies every model parent tree, all artifact LFS
identities and sizes, every metadata digest, and the generated legal bytes. It
then records exactly which of `README.md`, `LICENSE`, and
`THIRD_PARTY_NOTICES.md` would change.

```bash
PYTHONPATH=. python scripts/model_cohort_publication.py legal-prepare \
  --repo-root . \
  --manifest facetorch/models/manifest.json \
  --remote-manifest /secure/review/current-model-manifest.json \
  --plan /secure/review/legal-finalization-plan.json \
  --approval-template /secure/review/legal-finalization-approval.json
```

A human release authority must review the complete plan and change only the
generated approval status, reviewer identity, timestamp, and notes. Independent
review is preferred. For `1.0.0rc1` through `1.0.0`, the bounded D20 exception
allows owner `tomas-gajarsky` to approve; the notes must identify it as owner
self-approval, not independent review. Verify the digest-bound approval and
recheck all immutable parents before supplying write credentials:

```bash
PYTHONPATH=. python scripts/model_cohort_publication.py legal-verify \
  --plan /secure/review/legal-finalization-plan.json \
  --approval /secure/review/legal-finalization-approval.json
```

```bash
HF_TOKEN=... PYTHONPATH=. python scripts/model_cohort_publication.py legal-publish \
  --plan /secure/review/legal-finalization-plan.json \
  --approval /secure/review/legal-finalization-approval.json \
  --receipt /secure/review/legal-finalization-receipt.json
```

Each model commit is a direct child of its reviewed immutable parent and contains
operations only for the three legal documents; artifact and metadata paths must
remain byte-identical. The atomic receipt makes retries resumable. The final
`approved` manifest is created as a direct child of the reviewed current manifest
revision only after every model succeeds.

Generate one deterministic exact-old-value map from the completed receipt, inspect
it, and apply it to the packaged manifest, governance links, and source/packaged
configuration files. The map also emits the four exact
`FACETORCH_MODEL_MANIFEST_*` repository-variable values; update those variables
only after the local changes are accepted.

```bash
PYTHONPATH=. python scripts/model_cohort_publication.py legal-revision-map \
  --plan /secure/review/legal-finalization-plan.json \
  --receipt /secure/review/legal-finalization-receipt.json \
  --repo-root . --output /secure/review/legal-revision-map.json

PYTHONPATH=. python scripts/model_cohort_publication.py legal-apply-revision-map \
  --repo-root . --revision-map /secure/review/legal-revision-map.json

PYTHONPATH=. python scripts/model_cohort_publication.py legal-receipt-verify \
  --plan /secure/review/legal-finalization-plan.json \
  --approval /secure/review/legal-finalization-approval.json \
  --receipt /secure/review/legal-finalization-receipt.json
```

The pre-publication `legal-verify` command intentionally fails after the approved
revision map changes its source pins. `legal-receipt-verify` is the durable
post-publication proof: it validates the exact plan and approval digests, complete
receipt coverage, every direct-child model commit, unchanged non-document paths,
and the manifest-last commit without trusting workspace files that the transaction
was designed to rewrite.

Retain the clean export commit in each model's `export_commit`. Rendering is
non-circular: cards contain the artifact contract but do not embed Hub revision
fields.

Run the strict audit without `--allow-legacy-metadata`; for the release evidence
run, also use `--download-artifacts` so the downloaded bytes are checked in
addition to LFS object IDs:

```bash
PYTHONPATH=. python scripts/audit_model_manifest_hf.py \
  --remote-manifest /secure/review/final-model-manifest.json \
  --download-artifacts \
  --report /secure/review/model-manifest-audit.json
```

Only a zero-exit audit may supply the manifest repository, final revision,
filename, and SHA-256 to the coordinated release workflow. Candidate branches and
immutable commits remain review objects; this process does not move a default Hub
branch or stable alias.
