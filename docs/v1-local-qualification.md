# Version 1 local qualification

This qualification tests a fixed local revision of `fix/v1-ga-readiness` before
asking the owner to approve any push, merge, or publication. The technical source
is `cb202394bb7775573645f39db3968cbdd60b3732`. The local qualification
passed. The protected-main release workflow and publication approval remain
separate gates; this document does not declare version 1 generally available.

## What qualification means

A passing unit suite is necessary, but a user installs a wheel or container and
loads published model files. Qualification follows that complete path. It checks
both the library's behavior and the assembled distribution in the environments
we promise to support.

The numerical matrix has two stages. First, export both supported artifact
cohorts and compare all ten models with the fixed reference outputs. Second,
validate the exact immutable model files users download on every supported Torch
runtime, on both CPU and CUDA. All versions use the same approved reference
bundles. This prevents a successful re-export from hiding an incompatibility in
an already published model.

The full CPU source suites complement this model-level matrix: they exercise
configuration, image input, public results, errors, cache behavior, packaging,
and release-policy contracts. Installed-wheel checks then run outside the
checkout, and container inference runs without a network or writable root
filesystem. The archived evidence binds these results to source and file hashes.

## Findings addressed in this step

| Priority | Finding | Resolution |
| --- | --- | --- |
| P1: blocks the release checks | The pull-request dependency-review allowlist omitted an advisory already covered by the approved policy. The full suite caught the mismatch even though the dependency audit passed. | Added `GHSA-63cw-57p8-fm3p` to the workflow allowlist. The full dependency audit still enforces the exact package, version, profile, approval date, and expiry. No additional risk was accepted. |
| P2: qualification reliability | The local runner retained every large CUDA environment, making a broad matrix impractical on a machine with limited free storage. | Added `--ephemeral-environments`: install one frozen environment at a time in private temporary storage; clean up only owned environments and caches, including on failure. The full coverage and numerical gates remain the same. |

The runner's optional temporary mode trades additional downloads for bounded
storage. The tests verify that cleanup preserves existing project environments
and shared caches. It also binds the uv installation target and selected Python
interpreter even when the caller has an unrelated environment override.

## Additional finding for follow-up

**P3: normalize file permissions for reproducible independent wheel builds.** The
standalone source-distribution build and container builds contain the same 106
files, with identical contents including package metadata and RECORD. Their
wheel archive hashes differ because ZIP entries preserve different 0644/0664
permission bits. The numerical runner wheel also has the same contained bytes and its own
archive digest. Both containers contain the same wheel archive. This is an
archive-reproducibility difference; the actual distributions were tested
separately. Keep each artifact bound to its own digest. A fixed build umask and
canonical source permissions would remove this variation; do not relabel or
modify an already attested archive.

## Validation results

| Check | Measured result |
| --- | --- |
| Full frozen root suite, Python 3.12 / Torch 2.11 CPU | 1,078 passed; 185 skipped; 96% line coverage. |
| Full CPU source matrix | All nine lanes passed: 9,702 passing test executions. This covers Torch 2.6–2.13 on Python 3.10 plus Torch 2.6 on Python 3.11. |
| Fresh artifact exports | Both cohorts passed 624 CPU/CUDA cases each: 1,248 cases. |
| Published artifact compatibility | All eight supported runtime pairs passed 624 CPU/CUDA cases each: 4,992 cases. The strict aggregate verifier passed, with no numerical skips. |
| Installed wheel | Passed outside the checkout on Python 3.10, 3.11, and 3.12; also passed with the minimum supported Hydra/OmegaConf pair on Python 3.12. |
| Production containers and GPU wheel | Each full analyzer smoke passed three offline runs, four faces per run, and all seven predictors. No legacy fallback; repeated AU outputs were identical. Container smokes used a non-root user and read-only root filesystem. |
| Container cache locking | Separate PID-1 containers excluded a competing lock holder and recovered after the holder was killed. |
| Public notebook | Executed successfully against the staged, installed candidate wheel on the GPU with package-index access disabled and local pinned inputs. |
| Package and documentation checks | Wheel and source distribution built; Twine and wheel-content checks passed; 1,263 tests collected from the unpacked source distribution. Generated API documentation matches the committed documentation. Flake8, actionlint, and dependency alignment passed. |
| Full model-byte audit | Ten immutable model repositories passed, including model bytes, metadata identities, remote manifest binding, and required legal documents. |
| Dependency audit | All 17 profiles passed under the approved exceptions: 832 audited entries, 100 accepted finding/profile pairs, no unresolved findings or coverage errors on Python 3.12. Original audit reports were retained; their inputs were verified unchanged at the qualified source revision. |
| Portable archive | All 281 archived files matched their saved hashes. Verification after relocation, using only the included scripts and Python standard library, accepted 10 summaries, 100 model records, 10 reference bundles, and 6,240 cases. A modified reference bundle was correctly rejected. |

The CPU suite's 185 skips in each lane comprise 180 configuration-specific
exclusions, four optional legacy detector fixtures, and one opt-in upstream
network test. These skips are distinct from the required numerical matrix,
which completed every CPU and CUDA case. The full model-byte audit separately
checked the pinned upstream model repositories.

| Torch / torchvision | CUDA runtime | Full CPU suite | Published-model CPU/CUDA cases |
| --- | --- | ---: | ---: |
| 2.6.0 / 0.21.0 | 12.4 | 1,078 passed | 624 passed |
| 2.7.1 / 0.22.1 | 12.6 | 1,078 passed | 624 passed |
| 2.8.0 / 0.23.0 | 12.6 | 1,078 passed | 624 passed |
| 2.9.1 / 0.24.1 | 13.0 | 1,078 passed | 624 passed |
| 2.10.0 / 0.25.0 | 13.0 | 1,078 passed | 624 passed |
| 2.11.0 / 0.26.0 | 13.0 | 1,078 passed | 624 passed |
| 2.12.1 / 0.27.1 | 13.0 | 1,078 passed | 624 passed |
| 2.13.0 / 0.28.0 | 13.0 | 1,078 passed | 624 passed |

These results preserve the [approved broad Torch support policy](torch-support-policy.md).
The first three runtime lines consume the 2.6 artifact cohort; the remaining
five consume the 2.11 cohort. Loader deprecation/buffer warnings were recorded;
the required numerical and integrity checks still passed.

The [committed qualification receipt](../security/v1-local-qualification-2026-09-06.json)
records source identity, counts, report hashes, archive identity, and limitations.
Local evidence is retained at
`build/v1-local-qualification/local-qualification-evidence.tar.gz`
(26,272,683 bytes; SHA-256
`56f9d9fe32511ecdca70a28278d3c9156e4d3f5add372b7ea66f37b739401f79`).
The archive contains its verification tools and a README. It contains the
reference bundles and measured evidence, not the multi-gigabyte model weights.
The earlier failed/superseded attempts remain separately labeled under
`build/v1-local-qualification/superseded-aa61321/` and are not part of the passing
qualification archive.

## Limits and remaining release gates

This local run does not replace protected-main checks, publication approval, or
the agreed release-candidate soak. The package identity remains `1.0.0rc3` and
Beta; locally built development artifacts are not replacements for the already
published release candidate. A new public candidate needs a new identity.

GPU measurements apply to this Linux x86-64 host and its NVIDIA RTX 3090. They
establish the named Torch/torchvision/CUDA pairs, not every possible CUDA build,
driver, GPU architecture, or custom model. Torch 2.14 remains an expansion
candidate outside this eight-line support declaration.

The dependency gate passes with time-limited exceptions. It does not establish
that upstream vulnerabilities have been fixed. The approved exceptions still
expire on 20 November 2026 and require the documented mitigation and follow-up.

## Shared workspace observation

During qualification, the main checkout lost the 32 tracked project and lock
files under `environments/`, together with preexisting local environments. The
committed profiles remain intact in the tested revision and all isolated
checkouts. These deletions were not staged in the qualification commits. A
notification asking whether this was intentional received no answer before its
deadline. Confirm the intended cleanup before restoring those working files.
Restoring tracked profiles would not recreate the large ignored virtual
environments.

## Recommended next steps

1. Review the local changes and completed evidence. Owner approval is required
   before a push, merge, or publication.
2. After approval, run the remote pull-request checks and preserve the existing
   branch protections. Enable the new CPU aggregate protection only once the
   approved workflow has produced the required successful check.
3. Prepare the new candidate identity before its final qualification. After the
   approved changes reach protected main, run the authoritative release workflow
   and container/numerical gates at that exact revision, then complete the agreed
   candidate soak before the final version 1 release.
4. Continue the approved advisory follow-up and assess Torch 2.14 separately;
   neither changes the support commitment exercised here.
