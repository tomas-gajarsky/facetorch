# v1 release and incident runbook

This runbook is for the repository owner operating a public release. A release
is not complete because a job is green: the immutable source, model governance,
artifact digests, device evidence, and publication receipts must agree.

## Before a candidate

1. Work from a clean, protected `main` commit. Confirm it resolves to the same
   commit as the workflow SHA; review branches are never publication trust roots.
2. Run the full frozen test suite, wheel and sdist checks, dependency/advisory
   checks, and action/workflow lint. Keep the generated evidence with the
   candidate rather than in a mutable cache.
3. Verify every model record has an immutable revision, SHA-256, format, source
   provenance, weight-license decision, attribution, and limitations. Run the
   full-byte Hub audit against the already fetched exact remote manifest; every
   legal byte, metadata identity/digest, and artifact download must pass. Any
   incomplete governance record blocks the manifest.
4. Run the exact CPU/GPU container smokes and the local GPU release matrix on the
   release runner. Every requested device must be `ok`; a skipped CUDA lane is
   not a successful release. The current project has no server GPU, so the
   owner-controlled local GPU runner is the authoritative CUDA lane.
5. Confirm the GitHub environments, PyPI trusted publisher, Docker credentials,
   and required protected-branch checks are configured before enabling writes.

## Publication order

Use the reusable release workflow in dry-run mode first. It creates one canonical
plan, exact artifact set, internal and public checksum contracts, model-audit
evidence, SBOM/provenance, and a digest-bound receipt.
Bind the resulting protected source SHA, release-plan digest, immutable manifest,
and evidence digests into the owner release-approval record. Change it from
`pending` to `approved` and add the timestamp only after every checklist item is
true; any bound input change invalidates that approval.
After review, publish the GitHub release as a draft, then upload PyPI and Docker
artifacts. Reconcile each remote digest with the receipt before promoting stable
aliases. Promote `latest` only after the immutable version tag is verified.

If a job fails, use **Re-run failed jobs** after correcting the issue. Do not
re-run all jobs and accidentally rebuild artifacts that already have receipts.
Resume only when the plan digest, source SHA, model manifest, and artifact bytes
are unchanged. Never overwrite an existing version or tag.

## Rollback, yank, and revocation

For a normal defect, stop `latest` promotion, publish a patch release, and leave
the original immutable bytes available with a clear incident note. Yank a PyPI
distribution only when it is unusable or dangerous; record the reason and keep
the receipt. If a model, container, or source revision is unsafe, issue an
immutable revocation notice, block its digest in the governance/compatibility
manifest, and publish replacement artifacts under new immutable identifiers.
Restore the last known-good package, image digest, and separate v0.6.x model
cache while the correction is prepared. Do not delete evidence or silently
rewrite a model object.

## Security and support operations

Use private vulnerability reporting, redact image/model inputs from evidence,
and acknowledge reports within five business days with an initial assessment
within fourteen days. At general availability, announce the exact end date of
the six-month v0.6.x critical/security-only support period. A second operator is
preferred. For `1.0.0rc1` through `1.0.0`, the bounded D20 exception permits
`tomas-gajarsky` to self-approve and operate the release without a backup. Record
that this is owner risk acceptance rather than independent review, retain every
automated and exact-candidate gate, and rehearse recovery from the receipts.

The release candidate remains provisional until the clean protected dry run,
model-rights approvals, required remote environments, and exact local-GPU
evidence are all present.

## Required source checks and activation

`security/required-source-checks.json` declares thirteen successful checks required
on the exact final source SHA before release preparation. The CPU aggregate
`cpu-cohorts-complete` depends on every supported CPU lane, including Python 3.11
on Torch 2.6. A failed, skipped, cancelled, missing, or unfinished lane fails the
aggregate. The release verifier accepts only GitHub Actions check runs bound to
the candidate SHA and uses the newest attempt for each required check. The
resulting `source-checks.json` is retained in the release evidence.

On September 5, main protection already required the core checks, but named only
three individual CPU lanes. Activate the aggregate in branch protection after
the workflow is available on main and has produced a successful check run:
add `cpu-cohorts-complete` with GitHub Actions app ID `15368`, preserving the
existing contexts, strict up-to-date requirement, administrator enforcement, and
D20 owner-review policy. This is a separately reviewed settings change. Do not
require a check that cannot yet be emitted by the protected branch.

For an independent read-only check of a candidate's source CI:

```bash
python scripts/verify_source_checks.py --repo tomas-gajarsky/facetorch \
  --source-sha FULL_COMMIT_SHA --output source-checks.json
```

A saved paginated API response can be verified using `--checks-json`. Pull-request
`dependency-review` remains a branch-protection requirement; it is not required
on the final push commit, where the full frozen dependency audit is required.

## Portable numerical evidence

The local runner records `numerical-evidence-index.json`, every export/runtime
summary, the per-model `.meta.json` records referenced by those summaries, and
the actual `golden-references/*/golden-reference.pt` bundles. Schema 2 of the
index binds portable relative paths, metadata digests, and reference sizes/hashes while preserving
original summary bytes and runner identity. Both local GPU upload paths retain
these records. Release assembly verifies the downloaded archive before creating
the release plan; partial evidence cannot proceed through that workflow.

After extracting a candidate's evidence, use the scripts from its source commit:

```bash
python scripts/archive_numerical_evidence.py verify \
  --root /path/to/evidence/local-gpu --source-sha FULL_COMMIT_SHA
```

This uses only the Python standard library and the sibling
`model_evidence_contract.py`; no Torch install, GPU, model download, or original
runner directory is needed. It checks source identity, summary/metadata digests,
model/device/case records, finite errors, recorded same/cross-device bounds, and
fixed-reference identities and the archived reference bytes without deserializing
them. It validates the recorded measurements, not a new
execution or independently reconstructed golden tensors. The existing complete
matrix and model-manifest gates still establish authoritative case coverage,
policy tolerances, and artifact/golden bytes before archiving. The public archive
checksum and release receipt remain the external trust roots for the records.

RC3's published archive lacks these numerical records. Its historical summary
is retained as such; this change does not retroactively upgrade RC3 evidence.

The September 5 compatibility investigation recovered all ten historical golden
bundles exactly using Torch 2.6 with 16 CPU threads. Four-thread regeneration
changed their output bytes while remaining within numerical bounds. Retain the
authenticated bundles and record the generation environment; do not assume that
regeneration under different CPU settings will reproduce their hashes. See the
[investigation](torch-support-investigation.md) for the local recovery evidence.
