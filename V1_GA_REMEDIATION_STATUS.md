# Facetorch v1 GA remediation status

Date: September 5, 2026. Branch: `fix/v1-ga-readiness`, based on RC3 / main
`12db551d937ac2fa0cc41324f89d71fd9858fa02`.

The approved correction work is implemented locally. **GA remains blocked by
newly exposed dependency findings.** The corrected audit now checks all active
runtime dependencies and reports 18 distinct unresolved advisories, repeated as 76 entries
across twelve profiles. The owner has confirmed that all eight Torch lines
must be retained. Their remaining risk treatment is documented in
[the advisory triage](docs/v1-advisory-triage.md) and the
[concrete exception proposal](docs/v1-risk-treatment-proposal.md).

This branch has not been pushed, merged, or published. Repository protection and
release approvals have not been changed. The package still has the RC3 version
and Beta classifier; locally built wheels are development validation artifacts,
not replacements for the published RC3 bytes.

## Corrections against the review

| Review item | Implemented result | Remaining release work |
| --- | --- | --- |
| R01, P1: incomplete dependency audit | All 17 lock profiles retain their exact build identities, hashes, and SBOMs. Official Torch build tags are projected to upstream advisory versions only after source/hash checks. Missing, skipped, duplicate, unexpected, or malformed audit records fail. | Resolve the 18 advisories; no exception was broadened or renewed. |
| R02, P1: cross-container cache locks | Persistent POSIX file locks serialize independent PID namespaces and recover when a holder dies. Unsafe directory-lock reclamation was removed. Real container/process regressions exercise contention and recovery. | Stop every old worker before upgrading a shared RC3 cache; follow the documented migration. |
| R03, P1: unloaded native weights | Strict loading rejects empty or incomplete state for a parameterized model. Explicit frozen-model reconstruction hooks and genuinely parameterless models remain supported. | Include the native path in candidate qualification. |
| R04, P1: ineffective URL deadline | The shrinking deadline applies to each raw receive, request send, TLS handshake, redirect, and header/chunk/body parse. Invalid timeout values are rejected. | Ordinary network/decode limits remain documented; this is not a hard CPU-time limit on image decoding. |
| R05, P1: incomplete CPU gate enforcement | `cpu-cohorts-complete` covers every declared CPU lane. The release workflow requires thirteen successful source checks on the exact commit and archives the result. | After review and merge approval, activate the aggregate in main protection as described in the runbook. |
| R06, P2: hidden public errors | Hydra construction preserves `FacetorchError` subclasses, including lazy offline/model failures; other construction errors become `ConfigurationError` with their cause. | Exact-candidate rerun. |
| R07, P2: incomplete sdist | Every Python release helper is included. Unpacked-sdist validation imports the previously missing runtime and alignment helpers plus the new gates. | Build the final candidate from its clean commit. |
| R08, P2: application logging changed | A missing/`None` analyzer logger preserves existing handlers, level, and propagation. | None beyond candidate qualification. |
| R09, P2: invalid custom predictions | Each entry must be a `Prediction` with string label, tensor logits, and dictionary derivatives. Errors name the predictor and face index. | Applications returning invalid objects must follow the documented extension contract. |
| R10, P2: missing numerical records | The runner indexes per-model metadata by relative path/hash, both GPU workflows retain it, and assembly verifies the downloaded records before preparing a release plan. | Produce a fresh complete archive from the final candidate; RC3's historical missing records are not reconstructed. |
| R11, P2: stale policy/status | RC3 is dated as released; security authority follows D20; exception/support prose reflects all eight runtimes and the current audit. | Record the actual GA date and six-month v0.6.x support end date at GA. |

The migration and custom-component guides also clarify the v1 public API boundary
and coordinate frame: public geometry follows the canonical reader image,
including any requested reader resize/padding. The detector's internal resize is
restored to that frame.

## Validation performed

| Check | Observed result | Local evidence |
| --- | --- | --- |
| Complete source suite, Python 3.10.12 / Torch 2.11.0+cu130 | 1,066 passed; 185 skipped; 80 warnings; 96% line coverage (3,207 statements, 138 missed) | `build/v1-ga-pytest.log`, `build/v1-ga-coverage.json` |
| Complete source suite in the frozen root environment, Python 3.12.12 / Torch 2.11.0+cpu | 1,066 passed; 185 skipped; 80 warnings | `build/v1-ga-frozen-root-tests.log`, `build/v1-ga-frozen-root-tests.xml` |
| Affected runtime/release paths in eight frozen CPU profiles, Torch 2.6–2.13 / Python 3.10 | 281 passed per profile; 2,248 total; no failures or skips | `build/v1-ga-cpu-regressions/summary.json` and XML/log files |
| Real HTTP and HTTPS peers | Success, address retry, and timeout on trickled headers/body/chunk framing/redirects passed | `tests/test_url_deadline.py`, CPU regression reports |
| Container PID namespaces | Installed-wheel CPU image excluded a second PID-1 holder and recovered after the first was killed, retaining the lock inode | `build/v1-ga-container-lock.json` |
| Installed local wheel outside checkout | Offline CPU and GPU inference in an empty read-only working directory: four faces, seven default predictors, finite logits on both | `build/v1-ga-installed-wheel-smoke.json` |
| Wheel and source distribution | Built successfully; Twine and wheel-content checks passed; the final unpacked sdist collects all 1,251 tests, and the source suite passed packaging/import checks | `build/v1-ga-distribution-build.log`, `build/v1-ga-twine.log`, `build/v1-ga-wheel-contents.log`, `build/v1-ga-sdist-collection.log` |
| Static and dependency configuration checks | Flake8, actionlint 1.7.12, all frozen lock checks, dependency synchronization, and diff whitespace checks passed | `build/v1-ga-static-checks.log` |
| Generated documentation in the locked Python 3.12.12 environment | pdoc HTML/search regenerated and checked against the committed outputs | `build/v1-ga-pdoc-locked.log`, `build/v1-ga-static-checks.log` |
| Complete dependency inventory on Linux / Python 3.10.12 | 841 active package/profile entries covered; zero coverage errors; gate correctly fails on unresolved advisories | `build/v1-ga-audit/summary.json`, [durable snapshot](security/v1-audit-findings-2026-09-05.json) |
| Dependency gate repeated in the frozen Python 3.12.12 environment | All 17 profiles covered, 832 active entries, zero coverage errors, same 76 unresolved findings | `build/v1-ga-audit-py312/summary.json` |
| Source-check verifier against historical RC3 | Existing required source checks succeed; missing new CPU aggregate correctly prevents approval | `build/v1-ga-historical-source-checks.json` |
| Portable numerical archive | Relocated archive verifies without model binaries; missing records, hash/source mismatches, traversal, duplicate cases, nonfinite/excessive errors, and incomplete cross-device comparisons are rejected | `tests/test_numerical_archive.py` and CPU regression reports |

The dependency inventories cover each profile's default runtime export, with
development and release extras excluded; they are not a complete audit of every
tool used in CI. Coverage is line coverage, not branch coverage. Source checks cover both the local
Torch/CUDA environment and the frozen Python 3.12 CPU environment; the separate
eight-line CPU regressions also use isolated frozen profiles.
These checks are implementation validation. They do not replace the complete
protected-source release workflow, full eight-line CPU/CUDA numerical matrix,
exact image validation, and agreed 7–14 day candidate soak.

## Broad Torch support follow-up

The owner confirmed that all eight Torch 2.6–2.13 lines must remain supported;
Torch 2.13 remains the recommended pair for new installations. The
[support policy](docs/torch-support-policy.md) also preserves a CUDA 12 path and
sets explicit admission criteria for newer runtimes.

The [local investigation](docs/torch-support-investigation.md) exercised all ten
models in every retained CPU/CUDA lane: **4,992 cases passed**. Exploratory Torch
2.14 CPU, CUDA 12.6, and CUDA 13 profiles passed another **1,560 cases**, and all
three complete runtime dependency inventories had no findings. Torch 2.14 is
still an admission candidate, outside the current package bounds.

Runtime validation now executes the pinned programs without re-exporting them.
Exporter regressions passed on Torch 2.6, 2.11, and 2.14 (27 tests each). All
58 affected validation/archive gate tests passed after extending the archive to
retain and authenticate the golden bundles; 34 CI/distribution contract tests
also passed. All ten historical bundles were
recovered byte-for-byte with 16 CPU threads and preserved locally. The numerical
runs used a separate four-thread reference set; the report states that limit
and the untracked-review-document state. They are diagnostic compatibility
evidence, not an approved exact-candidate release run.

The [individual risk proposal](docs/v1-risk-treatment-proposal.md) includes all
18 findings, 76 exact profile/version scopes with lock hashes, and an inventory
of all 20 pinned model programs. It is inactive and unapproved. The existing
advisory exception policy has not changed.

## Decision and activation sequence

1. Preserve all eight Torch lines under the [support policy](docs/torch-support-policy.md).
   The support-range decision is complete. Review the
   [individual risk treatment proposal](docs/v1-risk-treatment-proposal.md);
   accepting new exceptions still needs explicit approval. Current pins are
   already the latest patches of their lines, so patch-only upgrades cannot
   resolve the exposed findings.
2. Review this local branch and authorize any push/merge separately. After its
   workflow can emit the new aggregate, add `cpu-cohorts-complete` (GitHub Actions
   app ID 15368) to the existing protections without removing their other checks.
   See [the runbook](docs/release-runbook.md#required-source-checks-and-activation).
3. Prepare a new candidate identity and run the full qualification from its clean
   protected commit. Preserve and verify the numerical archive, then complete the
   agreed soak. Request explicit publication approval for the resulting concrete
   candidate and evidence; the current local artifacts are not a GA release.

## Follow-up maintenance, without expanding this correction cycle

| Priority | Follow-up | Completion criterion |
| --- | --- | --- |
| Before exception expiry / next support review | Review runtime pins and retirement policy with the owner before November 20, 2026. Keep runtime support distinct from the two artifact export cohorts. | Every retained lane has a working frozen profile and current risk treatment; removals follow an announced compatibility policy. |
| Next model-maintenance cycle | Recover AU's native checkpoint mapping and a reproducible exporter. Preserve the current verified bytes while doing so. | Strict tensor mapping, one-face reference semantics, complete batch/device/golden parity, provenance records, and a separately approved immutable model publication. |
| v1.x, P3 | Consolidate duplicated release-policy tables; review runtime dependency hygiene, type-marker packaging, and additional provenance attestations. | One authoritative policy source where practical, installed-package validation, and no silent support changes. |
| v1.x | Add representative performance evidence separately from release correctness evidence. | Cold/warm latency, throughput, peak CPU/GPU memory, workload sizes, hardware, and repeated-run distributions reported for pinned versions. |

The accepted v1 result model, extension paths, bounded owner authority, legacy
deprecations, and asynchronous conda-forge delivery remain the basis of the plan.
An architecture rewrite and unrelated feature work are outside this correction.
