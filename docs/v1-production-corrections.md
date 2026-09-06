# Version 1 production corrections and readiness reassessment

**Assessment date: September 6, 2026.** All five findings from the production
stability review are addressed on the local `fix/v1-ga-readiness` branch.
The technical commit is `6415fe16219e81346360ba419e1488d7c848b011`.
The branch is ready for the next candidate-qualification stage. **GA readiness
remains conditional on the release gates below.** Nothing was pushed, merged,
tagged or published during this correction cycle.

## Corrections and their evidence

| Finding | Priority | What changed | Evidence |
| --- | --- | --- | --- |
| F01: shared timers and growing sample history | P1 | Replaced all 38 runtime timing decorators with invocation-local debug timing. No measurements or totals are retained; disabled debug logging bypasses timing. | Two independent analyzers overlap inside their readers with logging enabled and disabled. Recursive calls, exception recovery, diagnostic messages, signatures and repeated requests pass regressions. |
| F02: configuration loaders conflict with an active Hydra host | P2 | Both loaders use a fresh Hydra configuration loader and search path. They do not clear or swap the application's Hydra instance or change its job state and compatibility mode. | Packaged and external configurations compose inside a host application, including during overlapping composition. The host remains usable after success and failure. Installed-wheel tests also pass at Hydra 1.3.2 / OmegaConf 2.3.0. |
| F03: release dry run races required source checks | P2 | Added bounded polling: wait for missing, queued or running checks; stop immediately on unsuccessful completed checks. Preserve exact commit, Actions application identity and newest check-run requirements. | Registration, pending reruns, wrong identity, failure, cancellation, skip, timeout and API-error tests pass. A CLI probe observed a pending check become successful before accepting the 13 required checks. |
| F04: source archive cannot execute its shipped tests | P2 | Included the required workflow, container, security and documentation-template files. Generate the tensor fixture from the bundled JPEG. Run the complete unpacked archive suite in its own frozen environment in CI. | Fresh archive: 1,120 passed, 187 skipped. Two skips specifically require Git metadata. Tensor and workflow regression tests also execute from an extracted archive during ordinary packaging tests. |
| F05: malformed URLs escape the public error contract | P2 | Translate URL parsing failures into `InputError`, preserving the original cause. Redirect cleanup and the existing network policy remain intact. | Malformed brackets, invalid IPv6 and Unicode authority syntax fail before DNS or HTTP access. Malformed redirects close the response and connection. Real HTTP and HTTPS deadline regressions pass. |

### Design choices

Timing state now lives on each call's stack. This removes the shared-state
failure without serializing independent workers, and it removes the per-request
memory growth. Applications should still use one analyzer per worker or provide
their own synchronization when custom components share mutable state. This change
does not establish concurrent-use guarantees for a single stateful analyzer.

Hydra's [public Compose API](https://hydra.cc/docs/1.3/advanced/compose_api/)
uses initialized application state. The library therefore has one small private
adapter for the supported Hydra 1.3 series. Tests cover both the locked versions
and the declared minimum versions; the existing `<1.4` bound remains. Future Hydra
admission must exercise these integration tests before changing that bound.

The release gate waits at most 90 minutes and polls every 30 seconds. Its job has
a 100-minute budget to accommodate checkout, identity validation and evidence
upload. API requests and sleeps share the wait deadline. A timeout remains a
failed gate, and waiting never converts missing evidence into approval. Model
preparation still follows successful source verification.

The old tensor fixture is exactly the decoded bundled JPEG, so generating it
avoids adding another 7,373,480-byte binary to the source archive. The final
archive is 3,561,965 bytes. Git-dependent checks are explicitly marked rather
than being silently skipped whenever a file is absent. The environment-metadata
test also checks that an archive reports an unknown commit honestly.

## Validation completed

| Validation | Result |
| --- | --- |
| Full checkout suite, Python 3.12.12 / Torch 2.11 CPU | **1,122 passed, 185 skipped**; 95.93% line coverage. Both new private helpers have 100% line coverage. |
| Full fresh source-archive suite | **1,120 passed, 187 skipped**, using an independent frozen environment and an explicit verified model-fixture cache. No source files or Git metadata were borrowed from the checkout. |
| Torch 2.6–2.13 CPU regression matrix, Python 3.10 | **222 passed in each of eight lanes: 1,776 executions**, with no skips. These were targeted runtime regressions, not eight repetitions of the entire source suite. |
| Installed wheel outside the checkout | **14 passed on each of Python 3.10, 3.11 and 3.12.** |
| Installed wheel with minimum configuration dependencies | **14 passed** on Python 3.12 with Hydra 1.3.2 and OmegaConf 2.3.0. |
| Final workflow contracts | **102 passed**, plus a simulated CLI check transition. Flake8 and actionlint 1.7.12 passed. |
| Packaging, generated documentation and dependency consistency | Build, Twine and wheel-content checks passed. Generated HTML, search index and search page match. Dependency alignment and root plus all 16 profile lock checks passed. |

The checkout's 185 existing skips comprise configuration-specific exclusions,
optional legacy fixtures and an opt-in upstream network test. The archive adds
two explicit Git-only exclusions. The first archive execution exposed one
remaining test assertion that assumed a Git commit existed; that assertion was
corrected, rechecked in the checkout and followed by a passing complete archive
run. Failed intermediate results are retained separately from final evidence.

The source suite preceded that final test-only adjustment; the changed checkout
case passed separately afterward. Installed-wheel runs used the earlier build;
all 108 members of the rebuilt wheel have identical contents. Both actual archive
digests are recorded, without claiming byte-for-byte archive reproducibility.

The tested changes were transferred from the isolated checkout and matched to
the technical commit's complete Git tree. The receipt is
[`security/v1-production-corrections-2026-09-06.json`](../security/v1-production-corrections-2026-09-06.json).
Detailed logs, JUnit reports, source patch and artifact identities are retained
locally under `build/v1-production-corrections-20260906/`.

## Torch coverage and qualification limits

The approved support policy remains **Torch 2.6–2.13**, with **2.13 recommended**.
The root and all CPU/CUDA profiles, model definitions, manifests and approved
advisory policy are unchanged. No new dependency exception was requested or
granted. The existing exceptions still expire on November 20, 2026.

This cycle did not repeat CUDA numerical qualification, production-container
inference, the full remote model-byte audit, notebook execution or the dependency
advisory service scan. The [earlier local qualification](v1-local-qualification.md)
remains useful baseline evidence, including its 6,240 numerical cases, but it
belongs to the earlier technical source revision. It must not be relabeled as
qualification of this corrected candidate. Live GitHub orchestration also awaits
an approved push and the protected workflows.

The 32 pre-existing deletions of environment-profile definition files in the
main working directory were preserved. Their committed definitions remain
intact. Validation used an isolated checkout, and profile environments were
installed one at a time in temporary storage; no virtual environments or new
lock files were committed.

## Remaining gates, in order

1. **Prepare a new immutable candidate identity.** Use RC4 for the corrected
   candidate and update its version, changelog and distribution examples together.
   These local artifacts still identify themselves as `1.0.0rc3`; they must not
   replace the published RC3 artifacts or tag.
2. **Obtain approval for the remote steps and qualify the protected source.**
   Publish the branch for review only after owner approval, require every source
   lane and activate the already planned CPU aggregate protection. Merge and
   publication continue to require explicit approval.
3. **Run full release qualification against the resulting candidate.** Bind
   CPU/CUDA numerical results, installed distributions, production containers,
   model-byte audits and dependency evidence to that exact candidate. The new
   source-archive gate must pass as part of the protected workflow.
4. **Complete the agreed 7–14 day candidate soak.** Include an older supported
   Torch CPU deployment, the recommended current runtime and representative GPU
   workloads. Sustained service use should exercise repeated requests, worker
   concurrency and real application inputs.
5. **Finalize GA metadata and publication approval.** Set the stable identity,
   release date, maturity classifier and documented maintenance dates, satisfy the
   final protected release transaction, then obtain approval to publish.

No further project-philosophy decision was needed to implement these corrections.
The remaining owner input is who will exercise the next candidate and which
workloads they will use during the soak. Archive reproducibility across different
file-permission defaults remains the previously recorded P3 follow-up.
