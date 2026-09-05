# Proposed advisory treatment while retaining broad Torch support

Prepared September 5, 2026. **Proposal only: no new exception is approved or
active.** The owner has decided to retain all eight Torch lines. This document
narrows the remaining decision to their dependency risks; it does not reopen
that support decision. The release gate continues to fail on the 76 unresolved
profile entries in the [dated triage](v1-advisory-triage.md).

## Recommended treatment

Retain the current range and use Torch 2.13 as the recommended new-installation
pair. Prepare exact, temporary exceptions for affected older profiles only with
explicit owner acceptance of the residual risks below. Do not suppress the
whole Torch package, infer that a passing test fixes an upstream vulnerability,
or treat a trusted-model assumption as a sandbox for arbitrary models.

The proposed scope is exactly the finding/profile combinations already recorded
in `security/v1-audit-findings-2026-09-05.json`, at the frozen installed versions in
each corresponding lock. The [machine-readable proposal](../security/v1-proposed-advisory-exceptions-2026-09-05.json)
lists all 76 exact profile/version combinations and their lock hashes.
CPU and CUDA scopes stay separate. No other package,
version, advisory alias, or profile is implicitly included. If approved, use the
existing November 20, 2026 review deadline, without renewing any existing
exception. Each active record must still name its owner, approval date,
mitigations, and removal condition under the current policy.

## Evidence for the shipped models

The local inspection verified the manifest's SHA-256 and size before loading each
of the 20 published PT2 programs, recursively counted FX call-function targets,
and inspected serialized dropout arguments. Its [durable inventory](../security/v1-model-operator-inventory-2026-09-05.json)
records each artifact digest and all 57 distinct target names.

No named ctc_loss, packed-sequence, lstm_cell, fractional-max-pool, bitwise-right-
shift, LU, rot90/randn_like, nan_to_num, cummin, sparse-conversion, Fold/col2im, or
hardshrink target appears. Convolution, vector norms, view operations, and dropout
do appear. All 262 dropout nodes across eight cohort programs explicitly use
`train=False`. A missing target name does not prove an algorithm cannot decompose
into other operators or that a custom/compiled model is unaffected.

`BaseModel` defaults to `compile_model=False`; none of the shipped configurations
enables it. This excludes the reported Inductor triggers from the default model
execution path, but applications can enable compilation. Transform optimization
does use `torch.jit.script` by default. JIT is therefore assessed as an active,
controlled-code boundary, not described as unused.

## Individual findings

The affected versions, CPU/CUDA profiles, aliases, and upstream records remain in
[the complete triage table](v1-advisory-triage.md#unresolved-inventory). The following
assessments explain the proposed treatment of each record.

| Finding | Default-path assessment | Residual risk requiring acceptance |
| --- | --- | --- |
| CVE-2025-2999 | No packed-sequence unpacking target or direct production call was found. | Custom recurrent models or application code can invoke the vulnerable API. |
| CVE-2025-3001 | No lstm_cell target or direct production call was found. | Custom recurrent models are outside the graph inventory. |
| PYSEC-2025-194 | Shipped transforms use controlled `nn.Sequential` scripting; the reported bare list/tuple class-annotation trigger was not found in production source. | Scripting is active; custom transform classes can cross the affected compiler boundary. Existing approval does not cover newly found profiles. |
| PYSEC-2025-198 | No PairwiseDistance call was found, but exported vector norms are present. | Name matching cannot exclude a decomposed numerical trigger; model-specific numerical tests cover only the declared cases. |
| PYSEC-2025-199 | No Fold/col2im target; default model inference does not invoke Inductor. | Optional compilation or custom graphs can reach the assertion failure. |
| PYSEC-2025-200 | No FractionalMaxPool2d target; default model inference does not invoke Inductor. | Custom compiled pooling graphs are not covered. |
| PYSEC-2025-201 | No bitwise-right-shift target was found. | Arbitrary tensors/custom graphs can still invoke the affected operation. |
| PYSEC-2025-202 | Dropout exists, but every serialized dropout node uses inference mode and default execution does not invoke Inductor. | Training-mode/custom compiled graphs can reach the reported random decomposition. |
| PYSEC-2025-203 | No LU target or direct production call was found. | Custom linear-algebra graphs can reach the sliced-tensor failure. |
| PYSEC-2025-204 | No rot90 or randn_like target was found. | Application/custom graph use remains outside the inventory. |
| PYSEC-2025-205 | Default inference executes pinned programs without tracing through proxy_tensor. | Custom export/tracing workflows can invoke the reported failing path. |
| PYSEC-2025-206 | No nan_to_num target was found. | Custom preprocessing/graphs can invoke the reported integer conversion. |
| PYSEC-2025-207 | No cummin target; default model inference does not invoke Inductor. | Custom compiled reductions are not covered. |
| PYSEC-2025-208 | Convolution and view targets exist; hardshrink does not. Default model inference does not invoke Inductor. | This does not establish safety for a custom compiled operator combination. |
| PYSEC-2025-209 | No sparse-conversion target; default model inference does not invoke Inductor. | Custom compiled sparse/dense graphs are not covered. |
| PYSEC-2026-139 | `torch.export.load` is the normal loader. The packaged artifacts have immutable trusted-source pins verified before loading by default. | Loading a malicious model can execute code; the upstream database gives no fixed-version event. Authentication limits substitution, not malicious trusted content or arbitrary custom files. |
| PYSEC-2026-1970 | No ctc_loss target or direct production call was found. | Custom losses/models or application code can invoke the denial-of-service trigger. |
| PYSEC-2026-2286 | `torch.load(weights_only=True)` is used for alignment metadata and native checkpoints; default metadata is pinned and verified. | On affected versions, the safe-loading flag does not establish a safe boundary against a crafted checkpoint. Custom native files and verification opt-outs broaden exposure. |

## Conditions and limits of the proposal

The highest-priority decision is the two loading findings. Accepting compatibility
with affected Torch versions means accepting their use **only with trusted model
programs and metadata**, not promising safe processing of adversarial model
uploads. Hash verification stays enabled in defaults and release configurations;
applications choosing custom files or disabling verification must establish their
own model trust and cache protections. A separate service does not eliminate the
risk if it accepts model uploads or runs with broad privileges.

For the remaining operator/compiler records, the evidence supports bounded
exceptions for the shipped default path. The project cannot extend that reasoning
to every custom model, compiler option, tensor input, or application using Torch
in the same interpreter. Document that boundary and rerun the graph and numerical
checks when changing built-in models or enabling compilation by default.

No exploit test was run, and no proof of immunity is claimed. These are
reachability assessments and compatibility results. Patch-only upgrades within
the retained minor lines cannot currently resolve the findings: their frozen
Torch pins are already the latest available patches as of the assessment date.
Recheck upstream backports and advisory corrections before activation and before
the existing expiry date.

## Decision still needed

Approve or reject the proposed exact, temporary risk acceptance, especially the
trusted-model restriction for the two loading findings. The recommendation is to
retain the broad range under these explicit boundaries and continue to recommend
Torch 2.13 for new integrations. Approval would authorize preparing the specific
exception records and rerunning the dependency gate; it would **not** authorize a
push, merge, model-card update, image publication, or GA release. Those actions
remain subject to the user's separate approval.
