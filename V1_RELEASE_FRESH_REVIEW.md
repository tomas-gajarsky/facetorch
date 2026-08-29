# facetorch v1.0.0rc1 fresh release review

> Review date: 2026-08-28
>
> Pull request: [#91 — Release v1.0.0rc1](https://github.com/tomas-gajarsky/facetorch/pull/91)
>
> Reviewed range: `dad7d0e276af7d0e25f586a907f73f57885eed58..b950baf37c4f6c9506f52878b7027d27d41921ac`
>
> Decision: **hold merge and RC publication until the remaining remote operational
> gates below are closed and the exact final candidate passes the required evidence
> gates under the approved, bounded D20 owner-self-approval exception.**

## Closure execution status

The local implementation closes P1.2-P1.5 and the two public runtime
postconditions. P1.1 is also closed in the immutable Hub data: legal-finalization
plan `4809f4f8e036012c7384c0968fecf4e6e446440a522ef9811b57c8ed4c39b20e`
was approved by `tomas-gajarsky` under an explicitly accepted owner-approval
exception, published, and verified. This approval is deliberately recorded as a
policy exception, not an independent review.

The transaction created ten document-only direct-child model commits and then the
aggregate manifest commit last. The final approved manifest revision is
`f95f0743df8d4d3277fb7850a32d22ea755fb785`; its exact filename and SHA-256 are
now pinned locally. The deterministic revision map updated every source/packaged
configuration and governance reference. A strict final Hub audit downloaded and
hashed all 30 artifact payloads, validated all 20 current metadata contracts, all
30 legal-document byte contracts, and the final remote-manifest identity with zero
failures. A separate completed-receipt verifier remains valid after the approved
pin rewrite and rechecks every direct-child remote tree plus the manifest-last
commit; pre-publication verification still fails closed once its source pins move.
Candidate branches remain review objects; no Hub default branch or stable alias
moved.

The exact plan, approval, complete receipt, revision map, and audit reports are
retained under
`/home/toga/repos/facetorch-release-evidence/v1.0.0rc1/legal-finalization-4809f4f8e036012c/`;
each durable copy was byte-compared with the executed transaction and the copied
receipt was reverified live. The four generated GitHub repository variables have
not been changed because the accepted sequence updates them only after this local
candidate is accepted and submitted; the current PR snapshot does not contain
these local changes.

```text
FACETORCH_MODEL_MANIFEST_REPO=tomas-gajarsky/facetorch-model-manifest
FACETORCH_MODEL_MANIFEST_REVISION=f95f0743df8d4d3277fb7850a32d22ea755fb785
FACETORCH_MODEL_MANIFEST_FILENAME=manifests/4809f4f8e036012c7384c0968fecf4e6e446440a522ef9811b57c8ed4c39b20e.json
FACETORCH_MODEL_MANIFEST_SHA256=cfebb4514e39d4bbd64dc8e058b0c918748964dc0d49ab6be51bebdcc8f2f149
```

The stale CPU cohort contexts were replaced on both live protected branches with
the three exact emitted names. On 2026-08-28, `tomas-gajarsky` explicitly revised
D20 for the rest of the v1 release train: the owner may self-approve code, model,
environment, checklist, publication, and recovery gates through `1.0.0`. This is
a risk-acceptance exception, not independent review. The durable decision record
is
`/home/toga/repos/facetorch-release-evidence/v1.0.0rc1/owner-self-approval-exception.json`.
Its SHA-256 is
`7ecef57a11f532ad9b5a8065ba82465cfc8e15fd84f7b9f9d115faeb0160e34c`.
A separate exact-candidate approval template is retained at
`/home/toga/repos/facetorch-release-evidence/v1.0.0rc1/owner-release-approval-template.json`
with SHA-256
`e6527f5ac6b7d90d7a81f78ef8b1ccfd556b6e3802be8a0e6294b9baa0b3c7c7`.
It identifies the approved owner authority but remains `pending`: source, release
plan, evidence, timestamp, and every checklist item are deliberately incomplete.
It is not approval of this dirty local snapshot.
Because GitHub cannot count an author's approval of their own pull request, the
required approving-review count must be zero for this owner-authored v1 PR; strict
required checks, protected `main`, linear history, conversation resolution, and
the exact-candidate evidence gates remain mandatory. The release environments
already allow owner approval. Docker credentials exist only as unreadable
repository secrets, and the PyPI trusted-publisher mapping still needs an
authenticated provider-side confirmation.

This is a fresh delta review, not a replacement for
`V1_RELEASE_REMEDIATION_PLAN_DRAFT.md`. It re-evaluates the current 320-file,
82-commit PR after the prior review rounds, including runtime behavior, model
governance/publication, packaging, containers, CI/release automation, live
repository settings, and the immutable remote model repositories.

## Executive assessment

The release branch is substantially better engineered than the historical
baseline. The canonical input boundary, stable result contract, lazy component
lifecycle, verified artifact cache, URL isolation, bounded Torch cohorts,
installed-wheel matrix, and plan/receipt-based release transaction are coherent
and unusually well tested. The complete local source suite is green, as are the
current hosted checks.

At the originally reviewed PR head, five P1 release-integrity issues remained.
The closure implementation now fixes all five, including the executed immutable
model-card refresh, and fixes the two public runtime postconditions identified
below. The candidate is still not ready to authorize because these code and pin
changes are local and unsubmitted, repository variables still describe the prior
manifest, the live required-review count has not yet been reconciled with the D20
exception, and the exact-clean-candidate release gates remain open.

## P1 findings submitted inline

| Finding | Impact | Review thread |
| --- | --- | --- |
| P1.1 Generated model cards no longer match the ten pinned immutable revisions | The documented live release audit fails for every model, so the model trust root is not currently releasable | [Model-card drift](https://github.com/tomas-gajarsky/facetorch/pull/91#discussion_r3878802273) |
| P1.2 Publication metadata is not identity-bound | A plan can approve and upload metadata naming a different model, repository, cohort, runtime, or artifact | [Metadata identity](https://github.com/tomas-gajarsky/facetorch/pull/91#discussion_r3878802277) |
| P1.3 Publication accepts a non-rectangular model/cohort matrix | A partial cohort can create candidate commits while cross-cohort evidence is silently omitted | [Complete matrix](https://github.com/tomas-gajarsky/facetorch/pull/91#discussion_r3878802285) |
| P1.4 Published `SHA256SUMS` describes a different file tree | A user downloading the GitHub release cannot verify it with the supplied checksum file | [Public checksums](https://github.com/tomas-gajarsky/facetorch/pull/91#discussion_r3878802291) |
| P1.5 `latest` promotion re-resolves a mutable Docker tag | A retag between remote verification and pull can promote an image absent from the release plan | [Docker promotion](https://github.com/tomas-gajarsky/facetorch/pull/91#discussion_r3878802296) |

The overall comment-only review is [here](https://github.com/tomas-gajarsky/facetorch/pull/91#pullrequestreview-5048945067).

### P1.1 — immutable model-card drift

`scripts/render_model_cards.py` correctly removed `.eval()` from the direct
`torch.export` loading example at commit `900e3aea`, but the ten immutable Hub
revisions were finalized before that change. At the reviewed head:

```text
uv run --frozen python scripts/audit_model_manifest_hf.py
... README.md has an unexpected size.  # all ten repositories
```

Each hosted `README.md` is exactly seven bytes longer and still contains
`.to(device).eval()`. A follow-up read-only audit verified the remaining remote
surface independently: all 30 artifact filenames, LFS SHA-256 object IDs, sizes,
and current digest-bound metadata records matched; the generated `LICENSE` and
`THIRD_PARTY_NOTICES.md` bytes also matched. Artifact payloads themselves were not
downloaded during this review. The mismatch is isolated, but it is still fatal to
the declared legal/model-card audit.

Resolution requires new immutable legal-document commits for all ten model
repositories, a new final manifest revision, updated packaged manifest/governance
links and every manifest-bound config revision, and a passing live audit. The
artifact bytes do **not** need to be republished: the mismatch is documentation
only and the other pinned remote contracts matched. The repair should therefore
be a planned, approved, resumable legal-document refresh on top of the currently
pinned revisions, followed by the remote manifest last.

The current runbook describes this finalization transaction but leaves it manual.
That is how renderer drift escaped. Extend the existing publication tool with the
same plan/approval/receipt discipline for legal finalization, then make a full-byte
Hub audit a required pre-assembly job in the coordinated release workflow. This
keeps the remote mutation small while making the proof repeatable.

### P1.2 — metadata identity is not bound

`_validated_model_record()` hashes metadata and deeply verifies its numerical
evidence, but does not compare its identity fields with the staging result and
summary. A direct reproduction changed `model_id`, `repo_id`, and `torch_minor`
inside a staged metadata file, recomputed only `result.meta_sha256`, and produced
an accepted publication plan.

The planner and `verify_model_release_matrix.py` should require at least:

- metadata and summary schema versions;
- `model_id` and `repo_id` equality across metadata, result, and manifest/spec;
- artifact basename equality;
- declared cohort and `runtime_torch_minor` equality;
- equality of source/environment/exporter provenance fields that the plan claims
  to have reviewed;
- one shared clean source commit across all cohorts.

The recheck found a related release-side gap: the final remote manifest contains
`metadata_filename`, `metadata_sha256`, and golden-reference contracts, but
`validate_packaged_model_governance()` currently drops those fields when comparing
the packaged and remote manifests. The Hub auditor downloads metadata but only
checks its schema, artifact digest, and top-level validation status. Treat this as
part of P1.2: the exact-candidate audit must bind the remote-manifest metadata
digest and validate its model/repository/artifact/cohort/runtime identity.

Add one mutation test per identity field. `verify_publication_plan()` must recheck
the exact source-contract and summary digests captured by the plan, not merely
rehash artifact files whose meaning was established only during `prepare`.

### P1.3 — incomplete cross-cohort publication

Each staging summary is internally checked, but cohort model sets are never
compared. A 2.6 summary with models A and B plus a 2.11 summary with only A was
accepted, producing records for `(A, 2.6)`, `(A, 2.11)`, and `(B, 2.6)` and no
comparison for B.

Plan creation should require the package manifest and derive the expected models,
repositories, cohorts, devices, schemas, and validation policy from its referenced
compatibility/governance documents. It should require exactly the Cartesian
product of expected models and cohorts. Do not add a second policy implementation:
make `verify_model_release_matrix.py` the authoritative global-matrix gate, call
it from publication `prepare`, and retain only publication-specific normalization
and cross-cohort proof construction in `model_cohort_publication.py`.

The approved publication plan should record the SHA-256 of the manifest,
compatibility, governance, and each staging summary. Plan verification and publish
must revalidate those bindings. A small pure helper for schema and identity checks
may be shared by the matrix verifier, publication planner, and Hub auditor; a broad
late-stage rewrite of all three scripts is unnecessary.

### P1.4 — internal and public checksums are conflated

The generated checksum file intentionally covers the exact internal release
bundle, including `images/*.tar.zst`, `evidence/**`, `sboms/**`, and paths under
`distributions/`. GitHub publishes only the wheel/sdist basenames,
`release-evidence.tar.zst`, the plan, the checksum, and receipts. Consequently,
`sha256sum -c SHA256SUMS` in a directory of downloaded release assets reports
missing files; even the present wheel and sdist have different paths.

Keep two explicit scopes:

- `BUNDLE-SHA256SUMS` is internal transaction evidence and covers the complete
  bundle (including the public checksum) with bundle-relative paths.
- Public `SHA256SUMS` covers the immutable primary download payloads using only
  their release-root basenames: the wheel, sdist, `release-evidence.tar.zst`, and
  `release-plan.json`.

Receipts are intentionally excluded from public `SHA256SUMS`: they do not exist
when the draft's primary assets are created, and rewriting a checksum asset later
would weaken the transaction's idempotence. The existing final asset verifier
should continue to verify every receipt and the exact complete draft asset set
byte-for-byte. Add separate tests for the full internal bundle and for running the
public checksum against a directory containing only those four downloaded primary
assets.

### P1.5 — stable Docker alias TOCTOU

The stable job verifies `$REPOSITORY:$VERSION_TAG`, then later pulls the same
mutable tag. A concurrent or accidental retag between those operations changes
the bytes that are tagged and pushed as `latest`.

The smallest sufficient fix does not require changing the release-plan schema.
For each image, pull the version tag, capture the resulting local image ID, compare
that ID with the plan's approved config digest, and tag `latest` **from that local
ID**, not from the mutable repository tag. After pushing, query `latest` and require
its remote config digest to equal the plan before marking the GitHub release
latest. Exercise the rejection path with a simulated retag between the first
remote check and the pull. Pulling by a recorded registry manifest digest can wait
until multi-platform images make that identity necessary.

## Revised necessity assessment

The implementation plan should be a release-closure plan, not a general cleanup
backlog. The following classification is the result of challenging each earlier
suggestion against a concrete v1 invariant.

| Item | Decision | Reason |
| --- | --- | --- |
| P1.1-P1.4 | Must close before RC authorization | They break model or release integrity in paths exercised by RC1 |
| P1.5 | Fix in this PR before RC | RC1 does not move `latest`, but shipping a known-broken GA path in the v1 release branch is avoidable |
| Binary threshold semantics | Fix before RC | Public binary classification behavior contradicts its documented threshold |
| `Location.form_square()` odd deltas | Fix before RC | Exported public data-structure method fails its named postcondition |
| Predictor-free unifier skip | Defer | It changes retained-face/utilizer semantics and has no benchmarked release impact |
| RetinaFace prior cache | Defer | It is an unmeasured optimization with cache-lifetime/device risks |
| Dependency pruning | Defer | No current vulnerability or runtime failure justifies reopening every frozen solver profile late in RC |

### Binary threshold semantics

`PostSigmoidBinary.run()` zeros probabilities below `threshold` and then calls
`torch.round()`. This does not implement its documented threshold contract:

- with `threshold=0.2`, probabilities from 0.2 through 0.499... remain negative;
- at the default boundary, `torch.round(torch.tensor(0.5))` is zero because
  PyTorch uses ties-to-even rounding.

The shipped deepfake configuration uses `0.7`, so its common path is unaffected,
but the public postprocessor default and custom thresholds are wrong. Compute
indices directly with `(probabilities >= threshold).to(torch.int64)`, validate a
finite non-boolean threshold in `[0, 1]`, require exactly two labels, and test
probabilities below, exactly at, and above `0.2`, `0.5`, and `0.7`. Preserve the
returned confidence as the sigmoid probability.

### `Location.form_square()` fails for odd differences

The method adds `int(diff / 2)` to both sides. For a 10-by-11 box the difference
is one, both adjustments are zero, and the result remains 10-by-11 despite the
public method name and documentation. Preserve even-difference behavior; for an
odd difference, apply `diff // 2` to the low side and the remainder to the high
side. This guarantees equal integer dimensions while moving the center by at most
half a pixel. Cover both orientations, both odd/even differences, negative/boundary
coordinates, mutation-in-place, and the existing square no-op. Do not add clamping:
`Location` has no image bounds, and clamping belongs to the detector's bounded
geometry path.

### Explicit post-RC follow-up

Open benchmark-backed follow-ups, but do not implement them in this closure batch:

- Define a utilizer capability contract before skipping the unifier on
  predictor-free calls; benchmark detector-only and pre-cropped paths first.
- Profile `PriorBox.forward()` before adding a small bounded anchor cache, including
  device lifecycle and memory measurements.
- Audit the installed dependency graph in a separate change. `timm` is unused by
  the wheel runtime but is required by repository-side model publication; moving
  it changes maintainer UX. Any removal must regenerate all locks/SBOMs and repeat
  clean wheel installs.

## Live repository and operational gates

The settings audit shows major progress since the 2026-08-22 remediation-plan
snapshot:

- all five release environments now exist and have protected-branch policies;
- private vulnerability reporting is enabled;
- secret scanning, push protection, and Dependabot security updates are enabled;
- Actions defaults to read-only and cannot approve pull requests;
- `main` enforces administrators, linear history, conversation resolution, one
  approving review, and strict required checks.

The gate status is:

1. **Required check names — completed live.** On both `main` and
   `release/v1.0.0`, the two stale CPU contexts were replaced with the exact three
   names emitted by the matrix:
   `cpu-cohort (2.6, environments/torch-2.6-cpu, 3.10)`,
   `cpu-cohort (2.6, environments/torch-2.6-cpu, 3.11)`, and
   `cpu-cohort (2.11, environments/torch-2.11-cpu, 3.10)`.
2. **Immutable model legal trust root — completed.** The explicitly approved
   owner-exception transaction published and reverified all ten direct-child
   document commits and the manifest-last commit. The full-byte final audit is
   green. Exact local pins are updated; the four repository variables remain on
   the prior manifest until this candidate is accepted and submitted.
3. **Protected `main` release trust root — implemented locally.** Actual
   publication and exact-candidate local-GPU evidence now require
   `github.ref_name == "main"`, `github.ref_protected`, and the workflow source
   SHA. The umbrella release branch remains a review branch only.
4. **Owner-approval policy — approved; live branch rule still open.** On
   2026-08-28, `tomas-gajarsky` extended the owner-approval exception to the
   remaining code, model, environment, checklist, publication, and recovery gates
   through `1.0.0`, accepting the lack of separation of duties and a backup
   operator. AI review remains supplementary technical evidence. PR #91 still
   reports `REVIEW_REQUIRED` because `main` requires one approving review, and
   GitHub does not count author self-review. Set the required approving-review
   count to zero for this owner-authored v1 PR while retaining every other branch
   protection and required check. The environments already name the owner and
   allow self-review.
5. **Environment credential scope — still open.** `DOCKER_USERNAME` and
   `DOCKER_PASSWORD` currently exist as repository secrets while the documented
   design calls for copies scoped to `dockerhub` and `stable-release`. Move them
   and remove the unused legacy `TWINE_USERNAME`/`TWINE_PASSWORD` secrets after
   PyPI trusted publishing is confirmed.
6. **Exact-candidate remote dry run — still open.** The only recorded
   `release.yml` run is the pre-redesign workflow from 2026-04-14. After the
   code/model fixes, perform a no-publication run from the exact clean `main` SHA,
   including the ephemeral local-GPU lane and full Hub audit. A dry run can prove
   assembly and verification, but it cannot prove PyPI/Docker write permission
   without an external write; verify provider configuration separately and do not
   claim the dry run tested it.

## Implementation plan by reviewable batch

### Batch A — make the model evidence boundary authoritative

1. Add a small pure contract helper for schema/identity normalization; do not move
   network or publication behavior into it.
2. Make `verify_model_release_matrix.py` enforce summary/metadata schema 2, export
   mode, exact model/repository/artifact/cohort/runtime/environment/exporter
   identity, one clean source commit, and the exact manifest-derived model × cohort
   × device policy.
3. Require `model_cohort_publication.py prepare --manifest ...` to call that gate.
   Bind manifest, compatibility, governance, and summary digests into the plan;
   recheck them in `verify` and `publish`.
4. Preserve publication-specific numerical/cross-cohort proof construction, but
   require its normalized record set to equal the authoritative Cartesian matrix.
5. Extend the remote-manifest comparison and Hub auditor to bind metadata filename,
   SHA-256, identity, runtime/cohort, golden-reference contract, and final
   `approved` status. The audit report must identify the packaged-manifest SHA-256
   and the fetched remote manifest's repository, immutable revision, filename, and
   SHA-256 so the release plan can reject a report from another model release.

Regression tests must mutate each identity independently, omit one model/cohort,
duplicate a cohort, change one contract after approval, mix source commits, alter a
remote metadata digest, and prove that no Hub write method is reached. Existing
happy-path plan bytes must remain deterministic.

### Batch B — automate legal finalization and repair the ten pins

1. Extend the existing model publication tool with legal `prepare`, `verify`, and
   `publish` phases using a separate digest-bound plan, human approval, atomic
   receipt updates, exact parent checks, and plan-specific candidate branches.
2. Support the current corrective case by using each packaged immutable revision
   as the reviewed parent. Verify its artifact/metadata tree first, then commit only
   `README.md`, `LICENSE`, and `THIRD_PARTY_NOTICES.md`. Do not re-upload model
   artifacts.
3. Create the final `approved` remote manifest only after all ten legal commits
   succeed, as a direct child of the reviewed current manifest revision.
4. Generate and apply one deterministic revision map to the packaged manifest,
   governance URLs, source/packaged configs, merged configs, compatibility
   evidence, and release repository variables. Exact-old-value replacement plus
   exhaustive tests is safer than reserializing every YAML file.
5. Run the enhanced Hub audit with `--download-artifacts`. It must verify all 30
   artifact bytes, legal bytes, metadata identities/digests, and remote-manifest
   bindings at the final revisions. Pass the already fetched final remote manifest
   into the audit rather than independently resolving a mutable alias.

Add a dedicated pre-assembly model-audit job to `release.yml`; upload its report
with the fetched manifest and require the release plan to bind it. This catches
future renderer drift on both dry runs and real releases. Cache downloads by the
immutable manifest revision only as an efficiency improvement; always rehash them.

### Batch C — close the two public runtime postconditions

Implement direct sigmoid-threshold comparison and the odd-remainder square fix as
described above. Keep this batch deliberately independent of analyzer lifecycle,
unifier, detector fitting, or cache changes. Run focused unit tests under both
supported Torch cohorts in addition to the normal source suite.

### Batch D — separate release identities and close Docker promotion

1. Add distinct internal and public checksum generators/verifiers with explicit
   allowlists and non-colliding filenames. Make final draft verification parse and
   validate public `SHA256SUMS`, not only compare its bytes.
2. Pull each versioned Docker tag, verify the local ID, tag from that ID, push, and
   revalidate the remote `latest` config digest. Keep the existing concurrency and
   monotonic-version gate.
3. Add workflow-contract tests for command ordering plus unit tests for missing,
   extra, duplicate, renamed, and changed checksum entries and a mutable-tag race.

### Batch E — close external policy gates

1. Replace stale branch-protection contexts with the three exact emitted CPU matrix
   names on `main` and the active umbrella branch.
2. Require publication/local release evidence from `main`. For this bounded v1
   exception, keep `tomas-gajarsky` as the environment reviewer and permit owner
   approval; do not describe it as independent review.
3. Move Docker credentials into `dockerhub` and `stable-release`; verify the PyPI
   trusted-publisher mapping; remove unused TWINE secrets only after confirming no
   external workflow consumes them.
4. Apply the approved D20 exception by setting the required approving-review count
   to zero for the owner-authored v1 PR, retaining all other protections, and
   recording owner approval for code plus the model/release checklist. AI review
   cannot be represented as independent approval.

## Exact closure sequence

1. Add Batches A, C, and D to PR #91 as isolated commits with focused regressions;
   then add the Batch B tooling.
2. Freeze renderer/manifest inputs, execute the reviewed legal-finalization plan,
   update local pins, and run the full-byte Hub audit. Any later renderer, model
   contract, artifact, lock, or release-workflow change invalidates affected proof.
3. Run the full source suite, lint/lock/compile checks, wheel/sdist and installed
   artifact tests, all hosted Python/Torch/conda/image lanes, and the exact clean-SHA
   local CUDA workflow.
4. Apply Batch E settings, record the bounded owner approval, resolve every review
   thread, and merge through the protected `main` workflow. Do not release from the
   umbrella branch or weaken any protection other than the required approval count
   needed for the documented owner-authored v1 exception.
5. From that exact `main` merge commit, run `release.yml` with `dry_run=true` and
   retain the plan, internal/public checksum proofs, full model audit, image/SBOM
   evidence, and local-GPU reports. A failed gate requires a new run; a content
   change requires a new exact candidate.
6. Create the annotated RC tag on that unchanged commit and run the coordinated
   publication. Publish `1.0.0rc1` without moving stable aliases. After the agreed
   soak, repeat every affected gate for `1.0.0`; only the stable transaction may
   move `latest`, and it does so last.

## Evidence collected at the original reviewed head

| Check | Result |
| --- | --- |
| Complete local source suite | `857 passed, 185 skipped, 80 warnings` in 112.49 s |
| Flake8 | Passed |
| Python compileall | Passed |
| Dependency/profile synchronization | Passed |
| `git diff --check` | Passed |
| Hosted PR checks at reviewed head | All current source, wheel, Python 3.10-3.12, Torch 2.6/2.11 CPU, conda, image, dependency/security, and CodeQL lanes have a successful run |
| Live immutable-model audit | Failed: generated `README.md` mismatch in all 10 repositories |
| Supplemental remote contract inspection | Other legal documents, all 30 artifact LFS/size contracts, and all declared current-metadata contracts matched; large artifact payload bytes were not downloaded |
| Local actionlint | Not run during the original review; hosted lint and workflow-contract tests were green |

## Closure validation and actions

The local closure implementation has been exercised with the complete source
suite after applying the final immutable pins (`935 passed, 185 skipped, 80
warnings`). A final current-worktree skip audit reproduced that result and showed
that all 185 skips are fixture/config selection, deliberately absent legacy model
formats, or the opt-in upstream-network test; no current PT2, packaging,
transaction, or release-blocker lane was silently skipped. The mandatory
release-blocker suite separately passed `546` tests with `574` deselected, no
skips, and no failures. The focused model/release/workflow suite (`174 passed`)
and both supported CPU Torch cohorts (`51 passed, 5 skipped` in each exact
environment) also passed.

The final distribution-content assertions passed. Flake8, compileall,
dependency/profile synchronization, `git diff --check`, tracked JSON parsing, all
11 workflow YAML parses, and actionlint 1.7.12 are clean. Every root/CPU/CUDA lock
is current, and the five-profile dependency/SBOM audit reports no unresolved
findings. Its one accepted exception is the documented `setuptools` 81.0.0
advisory exception, which expires on 2026-11-20. The release transaction's own
audit validator accepted the final Hub report with exact coverage of 10 models
and 30 artifacts. The completed legal-receipt verifier passed against all 11 live
immutable commits after the local revision-map rewrite; its mutation regression
rejects subsequent remote document drift.

The live PR has 11 unresolved, non-outdated review threads. All 11 are addressed
in the local tree with regression coverage, including the five P1 findings and the
six earlier runtime/governance/publication findings. They remain unresolved on
GitHub because PR #91 still points to `b950baf37c4f6c9506f52878b7027d27d41921ac`,
while the 62-file closure set is dirty and unsubmitted. The eight green hosted
workflows therefore validate the older PR head, not this corrected tree.

The live required-check contexts were corrected on `main` and
`release/v1.0.0`. The legal transaction was approved under the explicit owner
exception and completed: ten model repositories and the aggregate-manifest
repository now contain plan-specific candidate branches, the receipt is complete,
and the final full-byte audit reports 10 repositories, 30 artifact payloads, and
zero failures. No Hub default branch or stable alias moved. No GitHub branch push,
pull-request merge, tag, release workflow, package upload, or image upload was
performed. The D20 owner-self-approval exception is now approved and durably
recorded; it does not create independent review or backup-operator coverage. The
live required-review-count reconciliation, repository-variable update, provider
credential confirmation, and exact-clean-candidate remote dry run remain open.
