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
