# Torch compatibility investigation — September 5, 2026

**September 6 status update:** the owner approved the scoped advisory treatment,
and all 17 dependency profiles now pass. See the
[activation record](../security/v1-advisory-activation-2026-09-06.json). The
September 5 investigation below remains a historical record; candidate
qualification and publication approval are still required.

All eight existing Torch lines passed **4,992 model cases** across CPU and their
named CUDA profiles. Exploratory Torch 2.14 CPU, CUDA 12.6, and CUDA 13 profiles
passed another **1,560 cases**. These are local compatibility results; GA still
needs advisory approval and an exact-candidate release run. Torch 2.14 remains
outside the package bounds pending full admission.

The [support policy](torch-support-policy.md) records the owner's decision to
retain the broad range and recommend Torch 2.13 for new integrations. No support
line, exception approval, package bound, model pin, or deployed image changed.

## Observed runtime results

Every GPU row includes CPU and CUDA execution of all ten shipped models. The
CPU-only row checks the actual CPU wheel. The matrix covers face batches 1/2/4/8,
the three declared detector sizes, two seeds, two scales, normal/uniform inputs,
output schemas, task invariants, finite values, per-case numerical bounds, and
cross-device comparisons. All rows used the same fixed CPU reference set.

| Frozen profile | Devices | Models | Cases | Result |
| --- | --- | --- | --- | --- |
| Torch 2.6.0 / CUDA 12.4 | CPU, CUDA | 10 | 624 | Passed |
| Torch 2.7.1 / CUDA 12.6 | CPU, CUDA | 10 | 624 | Passed |
| Torch 2.8.0 / CUDA 12.6 | CPU, CUDA | 10 | 624 | Passed |
| Torch 2.9.1 / CUDA 13.0 | CPU, CUDA | 10 | 624 | Passed |
| Torch 2.10.0 / CUDA 13.0 | CPU, CUDA | 10 | 624 | Passed |
| Torch 2.11.0 / CUDA 13.0 | CPU, CUDA | 10 | 624 | Passed |
| Torch 2.12.1 / CUDA 13.0 | CPU, CUDA | 10 | 624 | Passed |
| Torch 2.13.0 / CUDA 13.0 | CPU, CUDA | 10 | 624 | Passed |
| Torch 2.14.0 CPU wheel | CPU | 10 | 312 | Passed |
| Torch 2.14.0 / CUDA 12.6 | CPU, CUDA | 10 | 624 | Passed |
| Torch 2.14.0 / CUDA 13.0 | CPU, CUDA | 10 | 624 | Passed |

All runs used Linux x86-64, Python 3.10.12, an RTX 3090 (Ampere, capability 8.6),
driver 595.71.05, and four CPU threads. The validator came from commit
`29e5ffb633d0ff9b1cd6dda58ab06b64071f4b83`. Tracked files remained unchanged during
the model runs. Two pre-existing untracked review documents made the recorded
source state dirty; that flag remains in every report. Frozen CUDA environments
were installed independently. Only disposable environments and their own caches
were removed between runs to fit available disk space.

This verifies the named software combinations on one GPU, not every architecture
or driver. Optional `torch.compile` modes, arbitrary custom models, and older
historical patch releases are not established by these runs.

## Corrections made

Runtime validation previously rebuilt an exported program before loading the
existing pinned program. It now loads the independent reference and existing
program directly, removing an unnecessary dependency on historical exporter
APIs. Fresh exports remain in the separate artifact-producing gates. The
regression verifies operation when export APIs are unavailable and still rejects
changed reference bytes and incorrect output. Exporter tests passed on Torch
2.6, 2.11, and 2.14: 27 tests per runtime.

The release evidence archive now retains actual golden-reference bundles. Its
schema-2 index binds their portable paths, hashes, and sizes to both summaries
and model metadata. Both GPU workflow uploads include them, and assembly checks
their bytes without deserialization. All **58 affected validation/archive tests**
passed, including relocation, missing/corrupt reference files, altered size
bindings, and path escape rejection. Another 34 CI/distribution contract tests
passed. Flake8, actionlint, and dependency-alignment checks passed.

## Reference recovery and interpretation

The numerical runs first generated a fixed CPU reference set with Torch 2.6 and
four threads. All 312 input hashes match the historical cases, but the regenerated
outputs and bundle hashes were not identical. CPU thread settings changed the
floating-point results. A separate Torch 2.6 run with **16 threads recovered all
ten historical bundles exactly**: every SHA-256 and byte size matches the
immutable published metadata. The largest output difference between the two
reference sets is approximately `1.91e-6`; both sets and their comparison records
are preserved.

The earlier numerical reports were not rewritten to claim they used the
recovered historical bundles. They remain diagnostic compatibility evidence,
with a separate fixed reference and a dirty-source flag. The final candidate
must use approved reference identities and pass its protected-source release
qualification. Recovering reference bundles does not reconstruct RC3's missing
runtime metadata or retroactively upgrade its published archive.

## Torch 2.14 admission assessment

Upstream released [Torch 2.14 with torchvision 0.29.0 on September 2](https://pytorch.org/blog/pytorch-2-14-release-blog/).
The three exploratory profiles resolved from official CPU/CUDA indexes. Complete
hash-bound dependency audits found no unresolved findings or accepted exceptions:
41 active packages in CPU, 60 in CUDA 12.6, and 60 in CUDA 13, with no inventory
coverage errors. This is a dated audit result, not a vulnerability-free promise.

CUDA 12.6 is especially useful because upstream says 2.14 is its
[last release with CUDA 12.x wheels](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432).
This offers an expansion direction that retains a CUDA 12 path. The RTX 3090
results do not qualify older CUDA 12 GPU architectures.

Before admission, update package bounds, model routing, official frozen profiles,
CPU CI and CUDA gates, dependency inventories, and support documentation together.
Run installed-wheel library qualification, including optimized transforms, and
prepare any required immutable model-card revisions. Model program execution
alone does not establish this complete installation contract. No model card or
other remote artifact was published in this investigation.

## Security treatment and release work

The [risk proposal](v1-risk-treatment-proposal.md) assesses each finding against
all 20 pinned programs and the actual loading/scripting paths. Its
[inactive machine-readable proposal](../security/v1-proposed-advisory-exceptions-2026-09-05.json)
covers exactly 18 findings and 76 profile/version pairs with lock hashes. The
active policy is unchanged, so the previously reported dependency gate remains
red. The owner decision concerns the proposed temporary risks, particularly
trusted-model loading on affected older Torch versions. Breadth is settled.

Before another candidate: approve justified advisory treatments, run the full
clean-source qualification using approved references, and preserve the complete
archive. Any push, merge, model update, or publication still needs separate user
approval.

## Evidence and reproduction

The [durable receipt](../security/v1-torch-compatibility-investigation-2026-09-05.json)
records runtime, lock, and summary identities, reference recovery, candidate
audits, and the hash of `build/torch-support/compatibility-evidence.tar.gz`.
Every archived file was read back and checked against its indexed hash and size.
The archive includes original metadata, summaries, both reference sets, candidate
locks, audits, and local reproduction scripts. Model binaries remain in their
manifest-pinned staging/cache locations; their bytes were never changed.

Reproduce a lane with its frozen profile and the exporter's `validate` command,
its declared artifact cohort, all default cases, `--validate-devices cpu,cuda`,
and a selected fixed reference root in reuse mode. Use only CPU for a CPU wheel.
The archived runner records every exact command and lock. On this host, reproduce
the recovered bundles with the archived recovery script, Torch 2.6, and
`OMP_NUM_THREADS=16 MKL_NUM_THREADS=16`. Verify every resulting hash against the
pinned metadata before reuse; do not assume another host produces identical bytes.
