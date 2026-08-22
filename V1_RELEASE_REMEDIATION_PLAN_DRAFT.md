# facetorch v1.0.0 remediation and release-readiness plan

> Status: temporary review draft; not approved for implementation or release.
>
> Scope: GitHub PR #91 (`release/v1.0.0` -> `main`).
>
> Audit baseline: commit `0251afb1701ea03c45e1bb610091aa72c889eaf7`, reviewed on 2026-07-17.
>
> Release decision at baseline: **hold approval, merge, and publication** until the mandatory gates in this document pass.
>
> Decision progress: Round 1 approved by the founder on 2026-08-20 (`D01`, `D02`, `D03`, `D15`, `D18`, and `D20`).
>
> Decision progress: Round 2 approved by the founder on 2026-08-21 (`D04`, `D05`, `D09`, `D10`, `D16`, and `D17`).
>
> Decision progress: Round 3 approved by the founder on 2026-08-21 (`D06`, `D07`, `D08`, `D12`, and `D19`).
>
> Decision progress: Round 4 approved by the founder on 2026-08-21 (`D11`, `D13`, `D14`, `D21`, and `D22`). All `D01`-`D22` policy decisions are now approved; operational evidence questions remain open.
>
> Operational update 2026-08-21: no remote/server GPU is available for release testing; a local NVIDIA GPU is available. No non-author human reviewer or backup maintainer is currently available; only the founder and AI agents can review. This does not satisfy D20's independent-human approval gate.

This document converts the v1 audit into an implementable plan. It is deliberately broader than a bug list: each finding is assigned to a work item, an acceptance test, and a release gate. Nothing discovered in the audit is silently deferred.

## How to review this draft

1. Review the founder decisions in [Decision register](#decision-register-foundercreator-input-required).
2. Replace each `Pending` entry with the selected decision or edit the recommendation.
3. Confirm the proposed delivery shape and supported-runtime policy first; they affect most later work.
4. Mark any intentional P2 deferral explicitly in the traceability matrix, with a public issue, owner, milestone, and user impact. P0/P1 findings cannot be deferred from stable v1 under this plan.
5. After approval, rename this file if it should become a maintained release document; otherwise delete it after the work is transferred into issues or project tracking.

## Outcome and operating rules

The target outcome is a v1 release that:

- works from a normal PyPI wheel outside the repository and outside Docker;
- has deterministic, documented input and output behavior;
- loads only compatible, authenticated model artifacts and remains correct after process restarts;
- supports exactly the Python, PyTorch, CUDA, Docker, and conda combinations that the project advertises;
- cannot publish a partial or mismatched public release;
- provides users of v0.6.2 with a practical migration and rollback path;
- is independently reviewed and produces durable release evidence.

The following rules apply throughout implementation:

- Add a failing regression test before, or in the same commit as, every behavioral fix.
- Do not treat a skipped required device, empty validation set, warning, or resolved review thread as proof of correctness.
- Test built artifacts, not only the source checkout.
- Do not execute a remotely downloaded model until its repository revision and digest are verified.
- Keep public API changes explicit, typed, documented, and represented in the migration guide.
- Build release artifacts once from one immutable commit; promote those exact artifacts through every channel.
- Preserve the current audit evidence. If expected behavior changes, document why rather than weakening a test.

## Decision register: founder/creator input required

Recommendations below are defaults, not hidden assumptions. Decisions D03-D10 and D15-D18 change public behavior or architecture and should be settled before their dependent implementation begins.

| ID | Decision | Recommended default | Main alternative and consequence | Needed before | Founder answer |
| --- | --- | --- | --- | --- | --- |
| D01 | How should the remaining work be reviewed? | Keep PR #91 as the umbrella release PR; use small branches/PRs targeting `release/v1.0.0`, or equivalently isolated commits if only one maintainer is available. | Add everything directly to PR #91; faster mechanically, but materially harder to review and bisect. | Work starts | **Approved 2026-08-20:** retain PR #91 as the umbrella; deliver isolated, reviewable remediation batches. |
| D02 | Should v1 publish directly as stable? | Publish `1.0.0rc1`, run a documented 7-14 day soak, then promote validated code to `1.0.0`; do not use the `Production/Stable` classifier for the RC. | Direct stable publication; less work, but public users become the first broad compatibility test. | Release design | **Approved 2026-08-20:** release `1.0.0rc1`, soak for 7-14 days, and promote only after all gates and feedback pass. |
| D03 | What is the official Python/PyTorch support promise? | Publish a bounded, tested compatibility table. Support only combinations exercised in CI/release validation; add an upper Torch bound until newer minors pass. | Keep an open-ended Torch lower bound; permits resolver-selected versions the project has never validated. | W3.1, W4.1 | **Approved 2026-08-20; amended 2026-08-22:** bounded, evidence-based matrix only. Guarantee Python 3.10-3.12 initially. Support Torch 2.6 and 2.11; reject 2.3-2.5 and 2.7-2.10. |
| D04 | What is the public image input contract? | Provide one canonical preprocessing pipeline with `coerce` and `strict` policies. `coerce` accepts broad documented forms and performs safe deterministic conversions; `strict` requires exact or explicitly declared representation. Both reject genuine ambiguity. One source image is accepted per call; only faces within that image are batched for predictors. | Strict-only behavior increases migration cost; reproducing permissive guesses can silently corrupt inference. Multi-image batching adds a separate result/error/shape contract and is out of v1 scope. | W1.1 | **Approved 2026-08-21:** default `input_policy="coerce"`; opt-in `strict` with `InputSpec`. Both modes are deterministic and share one canonical pipeline. One source image per call; reject image batches with `B>1`. Rename predictor batching to `face_batch_size`, retaining `batch_size` as a v1.x warning alias. |
| D05 | What is the supported configuration API? | Add a resource-backed `facetorch` configuration loader that composes Hydra defaults and returns portable paths. Keep filesystem config loading as an explicit advanced API, not the default README path. | Continue caller-relative `conf/config.yaml`; incompatible with normal installed-wheel usage. | W2.1 | **Approved 2026-08-21:** add the packaged, fully composed loader; retain external YAML as an advanced supported override. |
| D06 | What should happen when no compatible `.pt2` artifact exists? | Fail clearly by default. Allow a verified legacy TorchScript fallback only through explicit opt-in; prohibit the unsafe AU TorchScript fallback on CUDA. | Continue silent automatic fallback; maximizes apparent compatibility but hides multi-gigabyte downloads and known safety/correctness risks. | W3.3 | **Approved 2026-08-21:** fail closed by default; require explicit `allow_legacy_models=True` for verified eligible TorchScript artifacts. Never allow the known AU TorchScript CUDA path. |
| D07 | What network/offline behavior should model loading have? | Preserve first-use automatic download for compatible verified artifacts, add an explicit offline mode and prefetch command/API, and fail before inference when required artifacts are absent. | Require manual model installation; stronger control but a less approachable default. | W3.2 | **Approved 2026-08-21:** lazily download only selected compatible models; add explicit prefetch and offline modes; verify immutable revision, size, and SHA-256 before load. |
| D08 | How should existing v0.x model caches be handled? | Use a new manifest-aware cache layout. Detect old caches without mutating them, reuse only after format/hash validation, and otherwise redownload; provide an explicit cleanup/migration command. | Rewrite old cache entries automatically; more convenient but risky and hard to roll back. | W3.2, W3.7 | **Approved 2026-08-21:** introduce a versioned manifest-aware user cache; inspect but never automatically mutate/delete old caches; reuse only independently verified compatible artifacts; make migration and cleanup explicit. |
| D09 | What should selection and lazy-loading mean? | `None` means all configured predictors, `[]` means none, unknown names are errors, excluded components are not downloaded, and `skip_detector=True` does not construct/download the detector. | Retain eager construction; simpler lifecycle but defeats the public selective-execution feature. | W1.3 | **Approved 2026-08-21:** adopt the recommended semantics and ensure selection occurs before construction, download, compilation, or device allocation. |
| D10 | How much v0.6 compatibility should v1 retain? | Make necessary v1 breaks, document each one, and retain safe, low-cost warning shims throughout v1.x where they do not preserve unsafe behavior. Remove them only in v2. | Remove all old surfaces immediately, or preserve all old behavior; both increase either migration cost or technical risk. | W1/W2 docs and API | **Approved 2026-08-21:** safe aliases/adapters remain with `DeprecationWarning` throughout v1.x and may be removed in v2. Incorrect preprocessing, unsafe model fallback, and integrity failures receive no compatibility exception. |
| D11 | Which artifacts and channels define a release? | GitHub release, PyPI wheel/sdist, versioned CPU/GPU images, model-manifest revision, and conda-feedstock handoff. Move `latest` only after stable completion; never for an RC. | Treat PyPI alone as canonical; leaves Docker/model/conda users without a coordinated compatibility promise. | W4.3-W4.6 | **Approved 2026-08-21:** coordinate GitHub, PyPI wheel/sdist, versioned CPU/GPU images, and the immutable model manifest from one tested candidate. RCs never move `latest`; stable moves mutable aliases last. Conda is a tracked asynchronous handoff under D22. |
| D12 | What GPU validation capacity can be guaranteed? | Require a trusted CUDA release job covering every shipped model and representative supported Torch cohorts; run a smaller scheduled matrix between releases. | CPU-only gating; cannot support the current CUDA claims or safely validate AU behavior. | W3.4, W4.1 | **Policy approved 2026-08-21:** the exact release candidate must pass trusted CUDA validation for every shipped model and representative cohorts; normal PRs may use a smaller CPU matrix. Runner identity/capacity remains open under Q10. |
| D13 | Who owns security reports and model provenance? | Publish `SECURITY.md` with a monitored private contact, supported-version policy, and response target; assign a named owner for third-party model licenses/source hashes. | Use public issues for everything; inappropriate for undisclosed vulnerabilities and unclear for model provenance. | W4.4 | **Policy approved 2026-08-21:** enable private vulnerability reporting and make the founder the initial security/provenance owner. Acknowledge within five business days and provide initial assessment within fourteen; no unsupported fix-time promise. Backup owner/contact configuration remains open under Q11-Q12. |
| D14 | What is the v1 breaking-change support window? | Support v0.6.2 migration documentation through the v1.x line, publish a deprecation/removal table, and provide rollback/cache instructions. | No formal window; lowers maintenance but makes user expectations unpredictable. | W4.5 | **Approved 2026-08-21:** retain safe deprecated surfaces throughout v1.x and remove only in v2; provide v0.6.x critical/security-only support for six months after v1 GA, with exact end date published at GA. |
| D15 | Is facetorch library-first, container-first, or equally both? | Library-first: the wheel is the primary product; Docker is a supported reference deployment using the same API/configuration. | Equal guarantees require a larger permanent test matrix; Docker-first would require correcting PyPI/README expectations. | W2 architecture | **Approved 2026-08-20:** library-first; Docker is a supported reference deployment using the same public API and packaged defaults. |
| D16 | Should `run()` return one stable result type? | Establish one result object with optional retained image/tensor fields; do not change the return type based on a flag. | Preserve the current union return and document/type it precisely; lower migration cost but permanently complicates callers. | W1.6, W2.4 | **Approved 2026-08-21:** `run()` returns one `AnalysisResult` for its single source image. Optional image/tensor data are fields. Preserve old `Response`/`ImageData` behavior only through an explicit warning compatibility adapter, not a flag-dependent primary return type. No `BatchResult` or multi-image API in v1. |
| D17 | Which customization and URL behaviors are public? | Support a small documented reader/postprocessor protocol; require explicit URL-reader opt-in with scheme, timeout, redirect, and size limits. | Treat Hydra-instantiated internals as public, which greatly expands compatibility obligations; or declare customization unsupported. | W1.1, W1.2, W1.4 | **Approved 2026-08-21:** support the small reader/postprocessor protocols; keep other Hydra internals private. Remote input is available only through an explicit restricted URL reader. |
| D18 | Which OS/architecture/accelerator combinations are officially supported? | Guarantee tested Linux x86-64 CPU and named Linux/NVIDIA CUDA pairs; mark Windows, macOS, ARM, and MPS experimental until exercised. Remove or qualify `OS Independent`. | Promise broad OS/device support, which requires corresponding hardware and release gates. | W3.1, W4.1 | **Approved 2026-08-20:** officially support tested Linux x86-64 CPU and named NVIDIA CUDA pairs; mark Windows, macOS, ARM, and MPS experimental pending evidence. Qualify/remove `OS Independent`. |
| D19 | What legal/responsible-use bar applies to model weights? | Verify source, weights license, redistribution permission, attribution, provenance, and limitations per model; exclude any model that cannot pass. Publish face-analysis limitations/responsible-use guidance. | Rely on the code's Apache-2.0 license alone; it does not establish rights or suitability for downloaded weights. | W3.6, W4.4 | **Policy approved 2026-08-21:** require per-model code/weights rights, source, redistribution, attribution, provenance, and limitations review; exclude any model that cannot pass. Evidence and indispensable-model choices remain open under Q06-Q07. |
| D20 | Who supplies independent release authority? | Keep branch protection, add meaningful CODEOWNERS/roles, obtain a non-author code approval, and separately sign off the model manifest/release checklist. | Weaken protection for an owner-only release; faster, but defeats the current governance safety control. | Merge/release | **Policy approved 2026-08-20:** retain protection and require non-author approval plus model-manifest/release-checklist sign-off. Reviewer identity and backup operator remain open under Q11. |
| D21 | What is the post-release incident and revocation policy? | Never replace released bytes. Define when to yank, patch, revoke a model/image, roll back manifest pins, and end support for a release line. | Handle incidents ad hoc or retag in place; unpredictable and breaks reproducibility. | W4.3, W4.6 | **Approved 2026-08-21:** never replace released bytes. Use patch releases for corrections, yank only unusable/dangerous Python releases, and publish explicit immutable model/image revocation notices with rollback instructions. |
| D22 | Is conda-forge a launch gate or an asynchronous supported channel? | Treat it as supported but asynchronous: validate the branch locally, state the published conda version honestly, and announce v1 separately after feedstock publication. | Block GA on feedstock completion, or classify conda as community/best effort. | W4.1, W4.5, W4.6 | **Approved 2026-08-21:** conda-forge is supported but asynchronous and does not gate v1 GA; validate the candidate locally, state channel status/version honestly, and announce v1 separately after feedstock publication. |

### Decision status and remaining evidence

All founder policy decisions `D01`-`D22` are approved. Implementation may use these decisions as the public v1 contract without further policy assumptions.

This does not close the evidence questions below. In particular, v1 cannot be released until the required CUDA infrastructure, model rights/provenance records, independent reviewer and backup operator, security configuration, conda ownership/status, and RC participants are established and verified.

### Open information questions

These are facts to establish, rather than policy choices. Record evidence or a conservative assumption before the associated decision is finalized.

| ID | Open question | Why it matters | Blocks/informs | Answer/evidence |
| --- | --- | --- | --- | --- |
| Q01 | How do current users actually install facetorch: PyPI, conda, Docker, source, or embedded downstream packages? | Determines the primary compatibility surface and rollout communication. | D15, D22 | **Unknown** |
| Q02 | Which Python, PyTorch, OS, architecture, and CUDA combinations are current users running? | Prevents dropping a high-use environment or promising an unused/unmaintainable one. | D03, D12, D18 | **Unknown** |
| Q03 | Do downstream users import old AU/model modules or provide custom Hydra readers/postprocessors/configs? | Establishes the real migration and extension-compatibility burden. | D10, D14, D17 | **Unknown** |
| Q04 | Which current callers depend on the flag-dependent `Response`/`ImageData` return behavior? | Determines whether D16 needs a shim and for how long. | D10, D16 | **Unknown** |
| Q05 | Are direct remote image URLs an intentional supported product feature? If yes, which schemes/auth/private-network behavior? | URL fetching introduces SSRF, timeout, size, privacy, and reproducibility obligations. | D07, D17 | **Unknown** |
| Q06 | Which of the ten default models are indispensable for v1 if one fails rights, provenance, or validation review? | Determines whether a failed model blocks v1 or is removed from the default set. | D19, W3.6 | **Unknown** |
| Q07 | Are original source checkpoints, source licenses, training-data terms, and redistribution permissions available for every hosted weight? | Required to establish a trustworthy, legally reviewable model release. | D13, D19, W3.6 | **Unknown** |
| Q08 | Does the project control all Hugging Face model repositories and can it create immutable revisions/tags without rewriting them? | Required for the versioned manifest and revocation process. | D07, D21, W3.2, W3.5 | **Answered 2026-08-22:** the founder confirmed ownership of all ten Hugging Face repositories. Actual candidate-branch/immutable-commit permission remains to be exercised by a non-publishing dry run. |
| Q09 | What model-output drift is acceptable per task, and who has authority to approve tolerances? | Numerical tolerances are product correctness decisions, not merely test constants. | W3.4, W3.6 | **Approved 2026-08-22:** the founder approved B08's float32 same-device `max_abs=1e-4`/`mean_abs=1e-5` and cross-device `max_abs=2e-3`/`mean_abs=1e-3`. The full local matrix observed at most `6.06e-5`/`4.01e-6`; release evidence must use these exact metadata-bound limits. |
| Q10 | What trusted CUDA runner/hardware and maintenance budget are available? | Determines the supportable CUDA matrix and whether release checks can be mandatory. | D12, D18, W4.1 | **Implemented locally 2026-08-22; release evidence open:** no server GPU exists. The owner-only, protected-ref workflow now targets an ephemeral-label local Linux x86-64 runner and rejects persistent publication credentials. A local RTX 3090 passed the dirty-tree CUDA matrix, installed-wheel/notebook smoke, and hardened production-GPU-container inference. The exact clean-SHA workflow and ongoing maintenance capacity must still be proven before RC. |
| Q11 | Who can provide independent code/model/release approval and serve as backup operator? | Current branch protection cannot be satisfied by founder self-approval alone. | D20, P6 | **Answered 2026-08-21:** currently unassigned. Only the founder and AI agents are available. AI review is useful technical evidence but is not a non-author human approval or backup operator; D20 and the protected merge/RC approval gate remain unsatisfied until a human is recruited or the founder explicitly revises the policy with a documented exception. |
| Q12 | Are PyPI trusted publishing, protected GitHub environments, Docker repository permissions, and private vulnerability reporting available/configured? | Determines the release/security implementation path. | D13, D20, W4.3, W4.4 | **Audited read-only 2026-08-22; configuration incomplete:** `main` is protected with one code-owner approval and strict required checks, but administrators are not bound and several required check names are stale after B09. Only the `github-pages` environment exists; `local-gpu-release`, `github-release`, `dockerhub`, `pypi`, and `stable-release` are absent. Private vulnerability reporting, secret scanning, push protection, Dependabot security updates, and repository-level SHA-pin enforcement are disabled; Actions default token permission is write. PyPI trusted publishing and Docker write permission remain unverified. All are pre-publication gates. |
| Q13 | Who maintains the conda-forge feedstock, and what is its realistic publication lag? | Determines whether conda can be a launch gate or must be asynchronous. | D22, W4.5 | **Unknown** |
| Q14 | Which existing users can test an RC, and what duration/sample constitutes a successful soak? | Turns D02 into a meaningful validation stage rather than a version label only. | D02, W4.6 | **Unknown** |
| Q15 | Should the Torch 2.11 runtime's `setuptools==81.0.0` advisory receive a temporary exception? | Torch 2.11 requires `setuptools<82`, while PYSEC-2026-3447 is fixed in 83; without an approved bounded exception the dependency gate must stay red. | W4.2, W4.4 | **Approved 2026-08-22 through 2026-11-20:** the founder approved an exception for exactly `setuptools==81.0.0` in the Linux `root`, `torch-2.11-cpu`, and `torch-2.11-cu130` profiles. The advisory concerns macOS source-distribution path normalization; official production/release targets are Linux, source distributions use an isolated `setuptools>=83` backend, and runtime never builds sdists. Remove on an upstream-compatible Torch patch or drop the cohort; expiry returns the dependency gate to red automatically. |

## Delivery structure and critical path

The recommended structure is to keep PR #91 as the umbrella comparison with `main`, while making each remediation batch independently reviewable against `release/v1.0.0`. This avoids turning an already large release PR into one unreviewable patch.

| Phase | Purpose | Depends on | Exit condition |
| --- | --- | --- | --- |
| P0 | Approve decisions and freeze the support claim | D01-D22 | Required decisions recorded; no ambiguous public contract remains |
| P1 | Land reproductions and release-test scaffolding | P0 where behavior-dependent | Every P0/P1 runtime and packaging defect has a failing regression or artifact smoke test |
| P2 | Complete W1 runtime correctness and W2 installed distribution | P1, D04, D05, D09, D10, D15-D17 | Source and installed-wheel gates pass on CPU; public API/docs agree |
| P3 | Complete W3 model compatibility and trust | P1, D03, D06-D08, D12, D18, D19 | Every advertised cohort/device passes independent validation; cache/fallback is restart-safe |
| P4 | Complete W4 CI, release, security, and migration work | P2, P3, D11-D14, D20-D22 | A dry-run release from an immutable candidate succeeds without public publication |
| P5 | Release candidate and soak | All prior phases | `1.0.0rc1` evidence reviewed; no open release blocker |
| P6 | Stable promotion | P5 | Independent approval, final gates, coordinated publication, rollback evidence retained |

W1 and W2 can proceed in parallel after decisions. W3 tooling can also proceed in parallel, but artifact publication must wait for the final runtime contract and compatibility table. W4 CI scaffolding can start early; final release automation depends on the exact artifact set from W2 and W3.

## Workstream 1 — runtime and public API correctness

**Goal:** make every newly advertised v1 input, selection, detector, predictor, and output path deterministic and testable.

### W1.1 Define and enforce one image contract

Affected areas include `facetorch/base.py`, `facetorch/utils.py`, readers, detector preprocessing/postprocessing, unifiers, and `FaceAnalyzer.run()`.

Implementation:

- Implement `input_policy="coerce"` as the v1 default and opt-in `input_policy="strict"`; both must be deterministic and feed one canonical preprocessing pipeline.
- In `coerce`, accept the broad documented source types and perform safe, documented layout/range/channel conversions with warnings where inference may surprise a caller.
- In `strict`, require exact source-specific conventions or an explicit `InputSpec` describing layout, numeric range, color space, and alpha behavior; reject undeclared coercion.
- Never guess a genuinely ambiguous representation in either policy. Require an `InputSpec` with an actionable error.
- Accept exactly one source image per `run()` call. Reject image batches with `B>1`; a 4D tensor is accepted only when its batch dimension is exactly one.
- Separate local paths from URL input. If URLs remain supported, require the D17 opt-in reader with scheme allowlist, timeouts, redirect/download-size limits, and clear ownership of temporary resources.
- Validate finiteness, dtype, channel count, dimensionality, and value range at the public boundary.
- Canonicalize grayscale and RGBA intentionally; document alpha handling.
- Establish one internal color/range contract for `ImageData.img` and `Face.img`.
- Refactor detector crops and predictor unification so they no longer depend on an implicit mean-subtracted face representation.
- Make `skip_detector=True` create the same canonical face representation as detector output.
- Reject calls supplying multiple input sources instead of silently prioritizing one.
- Close decoded byte/PIL resources in both analyzer-level and direct reader APIs.

Required tests:

- [ ] A contract matrix covers `coerce` and `strict` for paths, bytes, PIL, NumPy HWC, Torch CHW, and 4D Tensor input with `B=1`; grayscale, RGB, and RGBA are included.
- [ ] The same supported image reaches the same canonical tensor through both policies when the strict `InputSpec` describes the conversion performed by `coerce`.
- [ ] `coerce` warnings and conversions are stable and documented; `strict` rejects every undeclared layout/range/channel conversion.
- [ ] NumPy/Torch image batches and any tensor with `B>1` fail before detector or predictor execution.
- [ ] Spatial dimensions equal to 1, 3, or 4, including RGB images three pixels wide and CHW tensors whose width looks like a channel count.
- [ ] Finite float `[0,1]`, finite float `[0,255]`, integer `[0,255]`, and explicit rejection of unsupported/NaN/Inf values.
- [ ] Equivalent predictor preprocessing for the same pre-cropped face through the normal and skipped-detector paths.
- [ ] Exactly one input source required.
- [ ] Local path and opt-in URL behavior cannot be confused; disallowed schemes, oversized responses, and timeouts fail safely if URL support is retained.
- [ ] Direct byte reader closes its temporary image and preserves expected color order.

Acceptance:

- [ ] A white skipped-detector image reaches predictors with the documented canonical/normalized values, not values above 1 caused by added detector means.
- [ ] No valid documented layout is rejected merely because a spatial dimension equals a common channel count.
- [ ] Broad `coerce` input acceptance never creates a second preprocessing implementation or silently guesses an ambiguous layout/range.
- [ ] Public docstrings, README examples, tests, and implementation state the same contract.

### W1.2 Repair detector padding and geometry restoration

Affected area: `facetorch/analyzer/detector/core.py` and detector postprocessors.

Implementation:

- Replace the private `_extract_faces` dependency with a documented public postprocessor contract.
- Restore coordinates from padded to original-image space consistently.
- Perform expansion/squaring before the final clamp, then clamp final locations.
- Keep exposed detection boxes, face locations, dimensions, landmarks, and crop extents in the same coordinate system.
- Define behavior for partially out-of-bounds detections and empty crops.

Required tests:

- [ ] A custom conforming postprocessor retains faces after spatial padding.
- [ ] Edge and corner detections remain in bounds after expansion and square formation.
- [ ] `loc`, `dims`, extracted crop shape, landmarks, and public detection boxes agree.
- [ ] Padded and unpadded executions produce equivalent original-image coordinates.

Acceptance:

- [ ] No private-method feature detection is needed for a custom postprocessor.
- [ ] The regression described in the resolved PR review is represented by an executable test rather than only a comment.

### W1.3 Correct component lifecycle, selection, and model options

Implementation:

- Instantiate/download the detector and predictors lazily or filter factories before construction.
- Apply D09 semantics for `None`, empty collections, unknown names, duplicates, and include/exclude conflicts.
- Ensure `skip_detector=True` avoids detector construction and download.
- Replace `self.__dict__.update(kwargs)` with explicit supported parameters.
- Forward `compile_model` and compile options through `FacePredictor` and `FaceDetector` to `BaseModel`.
- Define whether already-loaded components remain cached across calls and document thread/concurrency expectations.

Required tests:

- [x] `include_predictors=["fer"]` works when an excluded predictor is deliberately unavailable.
- [x] `include_predictors=[]` runs no predictor; `None` follows the configured default.
- [x] Unknown predictor names and simultaneous include/exclude selections fail before inference.
- [x] `skip_detector=True` causes no detector download/load.
- [x] Subclass `compile_model=True` invokes `torch.compile`; false/default does not.

Acceptance:

- [x] Startup/network/device-memory cost matches the components actually requested.
- [x] Selection semantics are stable enough to document as public v1 behavior.

### W1.4 Restore the reader extension contract

Implementation:

- Route every supported input type through one stable public reader entry point, or expose typed public hooks and call them consistently.
- Preserve custom reader validation, preprocessing, color conversion, and metadata hooks for strings, paths, bytes, PIL, NumPy, and tensors.
- Replace the two reader tests whose `not A or not B` conditions always skip with real parametrization/selection.

Acceptance:

- [ ] A custom test reader observes every documented input type.
- [ ] Reader tests skip only for declared external capability reasons, not contradictory conditions.

### W1.5 Make logging and output paths reliable

Implementation:

- Configure console and file handlers idempotently without an existing stream handler suppressing the requested file handler.
- Apply the JSON formatter to the intended handlers.
- Handle basename-only log and image output paths without calling `makedirs("")`.
- Prevent duplicate handlers across repeated analyzer creation.
- Coordinate default paths with W2.2.

Required tests:

- [ ] Configured log file is created after package import.
- [ ] Basename and nested paths work for both logs and saved images.
- [ ] Repeated construction does not duplicate records.
- [ ] JSON log output parses as JSON when configured.

### W1.6 Stabilize the public result and error contract

Implementation depends on D16:

- Return one documented `AnalysisResult` type for every successful `run()` call, with optional image/tensor retention represented by fields rather than a different return class.
- Remove the flag-dependent primary `Response`/`ImageData` union. Preserve old behavior only through an explicit v1.x compatibility adapter that emits `DeprecationWarning` and delegates to the canonical result pipeline.
- Do not add a multi-image `run_batch()` or `BatchResult` in v1.
- Define which detections, faces, predictions, timings, warnings, and source metadata are stable public fields.
- Define actionable exception categories for invalid input, configuration, compatibility, offline cache miss, artifact-integrity failure, and inference failure.
- Avoid including image tensors, facial predictions, credentials, URLs with secrets, or other biometric payloads in logs/errors by default.
- Confirm that exported dataclasses/types and serialization behavior are intentional public API.

Required tests:

- [ ] Static/runtime return behavior is identical across paths, bytes, PIL, NumPy, and Tensor inputs.
- [ ] Image retention on/off follows D16 without a notebook-only assumption.
- [ ] Public exceptions distinguish user input, offline/compatibility, integrity, and execution failures.
- [ ] Default logs and exception strings contain no image/prediction payload.

Acceptance:

- [ ] The README, notebook, docstrings, type annotations, and migration guide show one consistent result contract.
- [ ] The notebook cannot dereference a field absent under its selected return mode.

### W1.7 Make within-image face batching explicit

Implementation:

- Keep the source-image boundary at one image per `run()` call; detection operates on that image only.
- Rename `batch_size` to `face_batch_size` so it unambiguously controls batches of detected faces sent through each predictor.
- Retain `batch_size` as a warning alias throughout v1.x under D10; reject calls supplying both names.
- After detection/unification, batch faces per predictor while preserving the stable mapping from every prediction back to its source face and index.
- Process a partial final face batch correctly and define behavior when no faces are found.
- Treat `skip_detector=True` as exactly one pre-cropped face, not a batch of images or faces.
- Align export metadata and documentation: predictor dynamic batch refers to faces; detector input batch remains one image while its supported spatial dimensions are validated separately.

Required tests:

- [ ] Images containing 0, 1, 2, 4, 8, and more-than-`face_batch_size` faces preserve face and prediction order for every predictor.
- [ ] Predictor face batches of sizes 1, 2, 4, and 8 load and infer on each advertised cohort/device.
- [ ] The partial final face batch maps predictions to the correct faces.
- [ ] `face_batch_size < 1`, both alias names, and source tensor `B>1` fail with actionable errors.
- [ ] The `batch_size` warning alias produces identical results to `face_batch_size`.

Acceptance:

- [ ] No public documentation describes multi-image batching in v1.
- [ ] Increasing `face_batch_size` changes execution grouping only, not result values, ordering, or ownership beyond documented numerical tolerances.

### W1 release gate

- [ ] All new regressions pass in source, installed-wheel, and Docker test contexts.
- [ ] Existing 525-test collection has no accidental import from a stale site-package installation.
- [ ] Skip counts and reasons are reported; none of the new mandatory contract tests can silently skip.
- [ ] CPU tests pass at the minimum and maximum supported Python versions.
- [ ] CUDA smoke covers the default analyzer plus AU repeated inference and skip/select paths.
- [ ] Public result and exception behavior matches D16 and contains no sensitive payload by default.
- [ ] Within-image face batching passes the W1.7 order, partial-batch, alias, cohort, and device tests; multi-image input remains an explicit error.

## Workstream 2 — installed distribution, configuration, and examples

**Goal:** make the artifact users install behave independently of the repository layout and Docker-only paths.

### W2.1 Add a resource-backed configuration API

Implementation:

- Move runtime defaults beneath the `facetorch` package as package data.
- Provide a supported loader, for example `facetorch.load_config(...)` or an equivalent factory, that composes Hydra defaults rather than merely loading the top-level YAML list.
- Support documented profiles such as CPU/GPU without downloading a mutable config from `main`.
- Resolve user overrides explicitly and preserve an advanced filesystem-config path for custom deployments.
- Decide and document the compatibility behavior for `OmegaConf.load("conf/config.yaml")` under D05/D10.

Required tests:

- [x] Load defaults from a wheel while the current directory is empty and read-only.
- [x] Hydra defaults are fully composed and `cfg.analyzer` is available.
- [x] User override and custom filesystem config behavior is deterministic.
- [x] Resource loading works in regular and zipped/importlib-compatible installation contexts where supported.

### W2.2 Replace Docker-only runtime paths

Implementation:

- Use an OS-appropriate user cache directory for models and generated metadata.
- Default logs and image outputs to disabled or user-writable locations; never `/opt/...` for a normal wheel install.
- Allow configuration/environment overrides for containers and managed deployments.
- Ensure path creation is lazy and produces actionable permission errors.
- Document container volumes and offline cache prepopulation.

Acceptance:

- [ ] A non-root user can configure, download, infer, log, and save without writing outside user-owned directories.
- [x] CPU and GPU containers retain explicit, predictable mount locations.

### W2.3 Narrow and verify package contents

Implementation:

- Replace implicit namespace discovery with explicit `facetorch` package discovery.
- Ship required runtime resources under `facetorch`; do not install generic top-level `conf`, `data`, or `docs` namespaces.
- Decide whether maintenance scripts belong in the sdist. If `check_dependency_sync.py` remains, include every input it needs, including `gpu.environment.yml`.
- Add automated wheel/sdist content allowlists.

Acceptance:

- [x] `check-wheel-contents` has no unexplained W005/W009/W010 packaging warnings.
- [x] Wheel contains runtime code/resources only; sdist contains every source/build/maintenance input it advertises.
- [x] `python -m build` and `twine check dist/*` pass from a clean checkout.
- [x] The dependency-sync script either succeeds from extracted sdist or is intentionally excluded and documented.

### W2.4 Repair and execute public examples

Implementation:

- Update README examples to use the supported installed configuration API and public input contract.
- Fix the notebook so its `return_img_data` setting matches the data it reads; do not access a nonexistent `Response.img` field.
- Pin notebook package/config/model-manifest inputs to the candidate release rather than mutable `main` or `facetorch>=1.0.0`.
- Add minimal offline/first-download guidance and expected disk usage.
- Exercise every copy-paste README snippet and execute the notebook without retained outputs.

Acceptance:

- [x] The README quick start passes from a newly created environment and empty working directory.
- [ ] The notebook executes end-to-end against the release candidate.
- [x] Examples use no private API or repository-relative file unless explicitly labeled as source-development usage.

### W2.5 Add installed-artifact smoke tests

Test at least:

- wheel install with `--no-deps` into a prepared supported environment;
- normal dependency resolution at each supported Python boundary;
- import and config composition from outside the checkout;
- one no-network/cached inference and one verified first-download inference;
- sdist build/install and maintenance-content check;
- CPU production image import/inference and GPU production image import/inference.

### W2 release gate

- [ ] The exact wheel and sdist proposed for publication pass all artifact tests.
- [x] No test succeeds only because the repository root is on `sys.path`.
- [x] No default config refers to `/opt/facetorch`, `/opt/logs`, mutable `main`, or test data.
- [x] Package top-level namespace is `facetorch` only, apart from standard distribution metadata.

## Workstream 3 — model compatibility, validation, trust, and cache safety

**Goal:** make model selection and execution compatible, authenticated, reproducible, restart-safe, and independently validated.

### W3.1 Define the supported cohort matrix

Implementation:

- Record Python, PyTorch, torchvision, CUDA/runtime, device, model format, export schema, and artifact cohort for every supported combination.
- Apply D18 to OS, architecture, and accelerator claims; qualify/remove the current `OS Independent` classifier if the release is not tested that broadly.
- Add a PyTorch 2.5/schema-major-7 cohort for all ten model repositories if 2.5 remains supported.
- Explicitly validate 2.4 because the existing 2.3 artifact previously failed there despite sharing schema major 5.
- Validate intermediate/current schema-major-8 versions rather than assuming numeric `<=` routing guarantees compatibility.
- Bound project dependencies to the tested range; do not allow an unlocked production build to select an untested latest Torch.
- Make routing schema/capability-aware and fail with a concise compatibility message when no artifact applies.

Acceptance:

- [x] Every version accepted by package metadata maps to at least one artifact that is actually loaded and inferred in candidate validation.
- [x] Unsupported versions fail before multi-gigabyte fallback attempts.
- [x] The release-candidate documentation and resolver constraints match the tested matrix exactly.
- [x] CUDA support names tested Torch/CUDA pairs rather than an open-ended `CUDA 12.1+` style claim.

### W3.2 Introduce a signed-off artifact manifest and safe cache format

The package/release manifest should record, per model candidate:

- model identifier and task;
- Hugging Face repository and immutable commit/revision;
- exact filename and real format (`pt2` or TorchScript);
- export schema/cohort and compatible runtime range;
- SHA-256 and expected size;
- source-weight digest, export commit, dependency versions, and validation metadata revision;
- license/provenance reference.

Implementation:

- Use a versioned, manifest-aware, OS-appropriate user cache with an explicit configuration/environment override.
- Resolve and lazily download only the detector/predictors selected for actual use; excluded components must not access the network.
- Provide an explicit prefetch API/command for deployment preparation and an offline configuration/environment switch that prevents all model network access.
- Download to a temporary file, stream-verify size/hash, then atomically move into the cache.
- Preserve the real extension/format or store an authenticated sidecar; never rely on process-local `_active_filename`.
- Verify existing cache entries before load and quarantine or ignore mismatches without executing them.
- Add locking/concurrency behavior for simultaneous first use.
- Pin remote revision on every Hugging Face request.
- Load remote metadata safely; avoid unrestricted `torch.load` for untrusted metadata, or use the safest supported restricted mode after digest verification.

Required tests:

- [ ] Two separate processes reuse `.pt2` and legacy caches successfully.
- [ ] Corrupt, truncated, wrong-format, wrong-hash, and interrupted downloads are rejected and never replace a valid cache.
- [ ] Mutable upstream default-branch changes cannot change a released installation's resolved artifact.
- [ ] Concurrent downloads converge on one valid cache entry.
- [ ] Offline mode uses verified cache or fails without network access.
- [ ] Excluded predictors and `skip_detector=True` generate no model-network requests; prefetch selects exactly the requested compatible artifacts.

### W3.3 Make legacy fallback explicit and CUDA-safe

Implementation:

- Apply D06: no silent legacy fallback when an export is incompatible.
- Require explicit `allow_legacy_models=True` before selecting any eligible TorchScript candidate, and keep the default false in packaged configuration and examples.
- Retain a native AU loader for the legacy path or prohibit AU TorchScript on CUDA with an actionable error.
- Emit one structured compatibility warning when the user explicitly opts into a legacy artifact.
- Prevent repeated candidate downloads after a known format/schema incompatibility.
- State disk/download cost before an explicit bulk prefetch.

Acceptance:

- [ ] Repeated AU CUDA inference cannot enter the legacy path known to hang.
- [ ] Fallback behavior is identical before and after process restart.
- [ ] Default analyzer startup cannot cascade through approximately 6 GB of incompatible candidates silently.
- [ ] With legacy opt-in disabled, no TorchScript candidate is downloaded or executed; with it enabled, only a manifest-pinned, hash-verified, device-eligible artifact can load.

### W3.4 Harden cohort validation

Implementation:

- Reject NaN/Inf recursively in native, exported, and comparison outputs before calculating differences.
- Reject missing/unexpected state keys by default. For AU, whitelist only specifically justified generated buffers.
- Compare against original legacy artifacts or immutable golden inputs/outputs, not only the model reconstructed by the exporter itself.
- Require each requested/required device to report `ok`; an overall CPU success cannot conceal skipped CUDA.
- Reject zero executed cases and incomplete model/device/shape matrices.
- Add task-level invariants and output schema checks, not only elementwise drift.
- Validate predictor dynamic face batches at sizes 1, 2, 4, and 8. Validate the detector with image batch fixed at one and representative dynamic H/W cases, including nonstandard sizes.
- Add cross-device and cross-cohort comparison where deterministic behavior permits it.
- Record tolerances per task/dtype with justification.

Required negative tests:

- [ ] NaN-only and Inf-only outputs fail.
- [ ] Missing and unexpected state keys fail unless explicitly allowlisted.
- [ ] Requested CUDA on a CPU-only host blocks release publication.
- [ ] Zero validation cases fail.
- [ ] Deliberately wrong architecture/weights fail independent golden comparison.
- [ ] A late model failure prevents all publication.

### W3.5 Make model publication two-phase and atomic

Implementation:

1. Export all requested models/cohorts into a local staging area.
2. Validate the complete required matrix.
3. Generate deterministic artifacts, metadata, manifest, and checksums.
4. Review/sign off the staging report.
5. Upload artifact plus metadata in one repository commit per model.
6. Create/pin the immutable manifest revision only after all repositories succeed.

No model upload may occur during the validate phase. A failed publish must be resumable without replacing already released revisions or leaving the release manifest pointing at a partial set.

### W3.6 Reproduce and revalidate every hosted model

Implementation:

- Move MagFace's embedded architecture into `model_defs` or correct the claim that all exact architectures live there.
- Record source weights and hashes, repository commit, Python, Torch, torchvision, timm, CUDA, exporter arguments, seeds, and environment lock.
- Re-export/revalidate all ten models for every supported cohort using the hardened schema.
- Normalize the 30+ metadata records to the current validation format.
- Review third-party source/license/redistribution terms before signing the manifest.
- Apply D19: document per-model intended use, known limitations, provenance gaps, and any restrictions relevant to identity, emotion, demographic, or other consequential use; exclude weights whose redistribution rights cannot be established.

Acceptance:

- [ ] A clean documented environment can reproduce each artifact from declared inputs, or the metadata clearly identifies a controlled non-public source and its verification procedure.
- [ ] Every published metadata record has the same required schema and complete device/case status.
- [ ] Artifact hashes in the final manifest match the immutable remote objects and downloaded bytes.
- [ ] Every shipped/default model has an approved rights/provenance record and user-facing limitations statement.

### W3.7 Provide cache migration and recovery

Implementation:

- Detect the legacy `.pt`-renamed-as-`.pt2` state safely without trying to execute it.
- Provide inspect, prefetch, migrate/revalidate, and cleanup operations or documented equivalents.
- Never delete the old cache automatically; report reclaimable space and require explicit user action.
- Document rollback behavior when moving between v0.6.2, an RC, and v1.0.0.

### W3 release gate

- [ ] The exact immutable release candidate—not merely an earlier branch state—passes the trusted CUDA job required by D12.
- [ ] Every advertised runtime/cohort loads and infers all applicable models on required CPU/CUDA devices.
- [ ] All required validation device statuses are `ok`; none are skipped or empty.
- [ ] Independent golden/task checks pass with finite output.
- [ ] Restart, offline, corrupted-cache, concurrency, and fallback tests pass.
- [ ] The release manifest pins immutable revisions and hashes for the complete model set.
- [ ] Model publication dry run proves all-or-nothing manifest promotion.

## Workstream 4 — CI, release engineering, security, and public migration

**Goal:** make green checks representative of the public artifacts and make release publication transactional, secure, reviewable, and recoverable.

### W4.1 Build a tiered compatibility CI matrix

Required PR checks:

- lint/format and dependency-file synchronization;
- fast unit/regression suite against the source checkout;
- build wheel/sdist and run artifact-content checks;
- install the wheel from an empty directory on Python 3.10-3.13, subject to the final D03 matrix;
- import, compose config, and run a no-network lightweight smoke;
- verify test imports resolve to the checkout or installed candidate as intended;
- CPU tests at representative supported Torch schema cohorts;
- conda solve plus installation of this branch's wheel, not the public v0.6.2 package.

Required trusted/scheduled/release checks:

- run the exact-candidate CUDA workflow on a controlled local GPU runner while no server GPU exists; keep the runner ephemeral or freshly provisioned, restrict it to protected trusted release dispatches, and never execute untrusted fork/PR code on it;
- full default-analyzer CUDA inference;
- repeated AU inference and fallback prohibition;
- every required model cohort/device validation;
- production CPU/GPU image build and smoke using the same locked dependency definitions;
- notebook execution;
- dependency, static-security, secret, license/provenance, and artifact scans.

Acceptance:

- [ ] The Python 3.13 frozen environment resolves if 3.13 remains supported.
- [ ] Install jobs do more than `pip install .`; each performs import/config/runtime smoke.
- [ ] Docker, conda, source, wheel, and model tests consume declared compatible dependency sets.
- [ ] Branch protection requires the release-critical checks and independent approval.
- [ ] CUDA evidence identifies the exact source SHA, model-manifest revision, dependency lock, GPU/runtime, commands, and report hashes; the local runner exposes no persistent publication credentials or reusable untrusted workspace state.

### W4.2 Align and secure dependency environments

Implementation:

- Create maintainable lock/constraint outputs for each supported Python/Torch/CPU/CUDA profile.
- Ensure test and production Dockerfiles use supported pinned versions rather than test Torch 2.3.1 while production resolves current/unbounded Torch.
- Exercise `environment.yml`, CPU lock, and GPU lock against the local candidate.
- Regenerate stale vulnerable locks and add `pip-audit` or an equivalent advisory gate with a documented exception process.
- Review broad minimum-only dependencies and add upper constraints only where compatibility requires them.

Acceptance:

- [ ] The existing advisories in certifi, idna, requests, urllib3, and other locked packages are upgraded or have reviewed time-bounded exceptions.
- [ ] All release artifacts share a traceable dependency manifest.
- [ ] An SBOM identifies packages that normal advisory tools cannot inspect, including custom Torch wheels.

### W4.3 Replace release workflows with one transactional pipeline

Implementation:

- Consolidate automatic/manual behavior around one reusable release workflow.
- Pass dispatch input through environment variables, validate strict SemVer, and require equality with project metadata/changelog.
- Resolve one immutable source SHA/tag and checkout that exact ref in every job.
- Build wheel, sdist, CPU image, and GPU image once; test and checksum those exact artifacts.
- Bind the release to one immutable model-manifest revision and require it to pass before any stable mutable alias is moved.
- Create a draft GitHub release only after validation, or keep it draft until all external publications succeed.
- Publish PyPI through trusted publishing/OIDC rather than long-lived credentials.
- Use `docker login --password-stdin`; push immutable version/digest references before moving stable aliases.
- Publish `latest` only after stable PyPI, both versioned images, model manifest, and release notes succeed.
- Publish RCs only under explicit prerelease versions/tags and never move stable or `latest` aliases for an RC.
- Make reruns idempotent: detect and verify already published identical artifacts, resume missing steps, and reject mismatched bytes.
- Produce a dry-run mode that performs every step except external publication.

Failure scenarios that must be tested:

- [x] PyPI succeeds and Docker fails.
- [x] One Docker image succeeds and the other fails.
- [x] GitHub draft creation succeeds and an external channel fails.
- [x] Rerun encounters an existing tag/release/artifact with identical bytes.
- [x] Rerun encounters the same version with different bytes and stops.
- [x] Invalid/malicious dispatch tag is rejected before shell execution.
- [x] RC publication cannot mutate a stable alias, and stable promotion cannot proceed while any coordinated D11 artifact is missing or mismatched.

### W4.4 Add public-project and supply-chain safeguards

Implementation:

- Pin third-party GitHub Actions to reviewed commit SHAs and configure dependency update automation.
- Pin or record digests for `uv`, base images, Miniconda/Miniforge installers, and other release inputs.
- Add dependency-review and secret-scanning-compatible checks where available.
- Generate checksums, SBOMs, and build provenance/attestations for distributions and images.
- Add `SECURITY.md`; enable GitHub private vulnerability reporting; name the founder as initial security/provenance owner; document supported versions, disclosure handling, five-business-day acknowledgement, and fourteen-day initial-assessment targets without promising an unsupported fix deadline.
- Audit `model_defs`, source checkpoints, pretrained weights, and included examples for license, notice, attribution, privacy, and redistribution requirements; publish a notice/provenance record.
- State the privacy/network contract: no telemetry by default, no image or facial-analysis payload logging, and network access limited to documented model retrieval or explicitly selected remote-input behavior.

Acceptance:

- [x] No unreviewed floating action/image/tool controls a release build.
- [ ] A user can verify package/image/model checksums and provenance.
- [ ] Security and third-party-model ownership from D13 is public and monitored.
- [ ] Private vulnerability reporting is enabled and exercised with a non-sensitive test/reporting drill; a backup owner is named before GA.
- [ ] Privacy-sensitive inputs/results are absent from default logs, telemetry, and release evidence.

### W4.5 Publish an honest v1 migration contract

Documentation must include:

- an `Unreleased` changelog section until stable publication;
- exact release date set only during final promotion;
- supported Python/PyTorch/torchvision/CUDA/OS/device matrix;
- old and new configuration-loading examples;
- `coerce` versus `strict` input-policy behavior, `InputSpec`, and image layout/range/color/alpha rules;
- the single-source-image boundary, `face_batch_size` semantics, the v1.x `batch_size` warning alias, and the explicit absence of multi-image batching in v1;
- the stable `AnalysisResult` contract and explicit compatibility path for the former flag-dependent return types;
- selective/lazy loading, empty selection, and skipped-detector semantics;
- model cohort selection, first-download size, immutable manifest, cache location, offline/prefetch, migration, cleanup, and rollback;
- all public module/class/function removals, including the prior AU model module, with replacements;
- custom reader/postprocessor/config migration;
- Docker tags, `latest` policy, PyPI/conda availability, and expected conda lag;
- known limitations and a rollback path to v0.6.2.
- safe v1.x deprecations/removal-in-v2 policy and the exact six-month v0.6.x critical/security-only support end date set at GA;
- patch-versus-yank criteria, immutable model/image revocation behavior, retained prior manifests, and tested rollback instructions.

Issue/community work:

- Confirm PR coverage and close/link issues #82 (direct image input), #83 (grayscale), and #88 (logger) only after their regression tests pass.
- Review remaining open issues for v1 relevance, including #59, #39, and #21, and explicitly classify each as a v1 blocker, a documented follow-up milestone, or declined with rationale.
- Post RC testing instructions and a focused request for Windows/macOS/Linux, CPU/CUDA, custom config, and offline-cache feedback.

### W4.6 Run release candidate, soak, and stable promotion

Recommended sequence:

1. Freeze candidate source and model manifest.
2. Run the full release matrix and archive evidence.
3. Obtain an independent approving review; the repository owner must not self-approve.
4. Publish `1.0.0rc1` without moving stable/`latest` aliases.
5. Triage RC reports; changes require a new RC and complete affected-gate rerun.
6. Rebuild only if the version/source changes; otherwise promote the exact approved artifacts where a channel permits it.
7. Publish stable channels transactionally, move `latest` last, and initiate the asynchronous supported conda-feedstock handoff without representing conda v1 as available prematurely.
8. Run post-publication installs/pulls/downloads from public endpoints and exercise rollback.

Before step 3, apply D20 by recording who may merge the release, publish/revoke models, publish PyPI/Docker artifacts, approve exceptions, and receive security reports. Add CODEOWNERS only where the named owner can actually review and approve.

Before stable publication, apply D21 by documenting patch versus yank criteria, immutable model/image revocation notices, supported 1.x lines, retention of old manifests, and the exact command/config users use to roll back. Apply D22 by treating the conda status as its own verified channel state rather than implying that the existing public 0.6.2 package is v1.

### W4 release gate

- [ ] All required branch checks are green on the final immutable commit.
- [ ] No required check is older than the final source/model/config change.
- [ ] Independent approval is present and branch protection is satisfied.
- [ ] Dry-run release and all failure/retry scenarios pass.
- [ ] Migration, security, provenance, changelog, and support-policy documents are published.
- [ ] A tested rollback procedure and responsible operator are recorded before stable promotion.
- [ ] Publication/revocation authority and post-release support ownership are explicit.
- [ ] No release byte is replaceable in place; patch, yank, revocation-notice, retained-manifest, and rollback procedures have assigned operators and a rehearsal record.
- [ ] The six-month v0.6.x support end date and current conda channel status are stated accurately in public release documentation.

## Proposed reviewable implementation batches

If D01 is accepted, use the following batches as sub-PRs into `release/v1.0.0`. A batch should not combine unrelated cleanup.

| Batch | Scope | Principal output | Depends on | Size | Owner/reviewer | Status |
| --- | --- | --- | --- | --- | --- | --- |
| B01 | Regression harness and import isolation | Reproductions for runtime, wheel, cache, validator, and release failure paths | Relevant founder decisions for expected behavior | L | Codex/founder (AI-only technical review) | **Implemented locally 2026-08-21; validation complete; uncommitted/unpushed** |
| B02 | Input policies, detector, reader, result, face-batching, and output contracts | W1.1, W1.2, W1.4-W1.7 | D04, D10, D16, D17 | XL | Codex/founder (AI-only technical review) | **Implemented locally 2026-08-21; source validation complete; uncommitted/unpushed** |
| B03 | Lazy lifecycle and compile forwarding | W1.3 | D09 | M | Codex/founder (AI-only technical review) | **Implemented locally 2026-08-21; validation complete; uncommitted/unpushed** |
| B04 | Resource config and portable paths | W2.1, W2.2 | D05, D08, D10, D15 | L | Codex/founder (AI-only technical review) | **Implemented locally 2026-08-21; source/wheel validation complete; uncommitted/unpushed** |
| B05 | Packaging, artifact smoke, and examples | W2.3-W2.5 | B04 | L | Codex/founder (AI-only technical review) | **Implemented locally 2026-08-21; clean-copy wheel/sdist validation complete; uncommitted/unpushed** |
| B06 | Artifact manifest, downloader, cache, fallback | W3.2, W3.3, W3.7 | D06-D08 | XL | Codex/founder (AI-only technical review) | **Implemented locally 2026-08-21; provisional-manifest/source validation complete; uncommitted/unpushed** |
| B07 | Cohort validator and atomic publisher | W3.4, W3.5 | D12 | XL | Codex/founder (AI-only technical review) | **Implemented locally 2026-08-21; CPU/tooling validation complete; uncommitted/unpushed/unpublished** |
| B08 | Compatibility matrix and regenerated cohorts | W3.1, W3.6 | B06, B07, D03, D13, D18, D19 | XL | Codex/founder (AI-only technical review) | **Implemented locally 2026-08-21; full dirty-tree CPU/local-GPU candidate matrix complete; governance/clean-release evidence open; uncommitted/unpushed/unpublished** |
| B09 | CI/dependency/production parity | W4.1, W4.2 | B02-B08 | XL | Codex/founder (AI-only technical review) | **Implemented locally 2026-08-22; CPU/GPU image build, dirty-tree CUDA smoke, and approved-exception dependency audit complete; exact clean-SHA run open; uncommitted/unpushed/unpublished** |
| B10 | Transactional release and supply chain | W4.3, W4.4 | D11-D13, D20, D21, B09 | XL | Codex/founder (AI-only technical review) | **Implemented locally 2026-08-22; source/package/workflow/container validation complete; clean protected remote dry run and repository configuration open; uncommitted/unpushed/unpublished** |
| B11 | Migration docs, issue linkage, RC runbook | W4.5, W4.6 | D02, D10, D11, D14, D20-D22 | L | 2026-08-22 / local | Documentation implemented; issue triage, RC soak, and remote release configuration remain open |
| B12 | Integrated release audit and external-review remediation | B01-B11 | All accepted decisions | L | Codex + external AI review / founder | **Implemented locally 2026-08-22; 576-test suite and candidate-tree distribution simulation pass; uncommitted/unpushed** |

### B01 local implementation evidence — 2026-08-21

B01 is implemented in the current `release/v1.0.0` working tree as regression/test scaffolding only. No product behavior, model artifact, workflow, public metadata, tag, commit, push, or PR state was changed. The implementation:

- forces source tests to import the checkout before any installed `facetorch` package and proves the bare-`pytest` stale-package scenario in a subprocess;
- repairs the two contradictory reader skip conditions and replaces the old small-CHW rejection assertion with the approved Torch-CHW contract;
- adds isolated runtime/API, within-image face batching, lazy lifecycle, reader, detector, logger/output, installed wheel/sdist, cache/restart, model-manifest/cohort, validator/publication, CI/container, notebook/documentation, and release-workflow contracts;
- registers `release_blocker` so the deliberately red v1 contract can be run separately while implementation batches turn failures green;
- links executable regression intent to issues #82, #83, and #88 in the relevant tests.

Validation performed on the server CPU/source environment:

- `python -m pytest -q -m release_blocker --tb=no` → **63 failed, 5 passed, 520 deselected**. The 63 failures are expected audited defects/missing v1 contracts; the five controls prove checkout import isolation, one-image-only rejection, within-image face batching/order, and grayscale conversion.
- `python -m pytest -q -m 'not release_blocker' --disable-warnings` → **336 passed, 184 skipped, 68 deselected**. Existing skips are not release approval and still require W1/W3/W4 skip review.
- `python -m flake8` over every changed test file, `python -m compileall -q` over the same files, and `git diff --check` → **passed**.

Primary executable coverage now spans F01-F27, F30-F34, F37, F39-F43, and F45-F46. F35-F36 have partial security/release scaffolding. F28-F29, F38's external issue state, and F44's independent approval are evidence/governance work for their later batches rather than locally reproducible product behavior.

No CUDA test was executed because the server has no GPU. The CPU negative contract proves that requested CUDA cannot be treated as successful when unavailable, and the CI contract remains red until a controlled, manually gated exact-candidate local GPU runner/bundle exists. No independent human reviewed B01; founder-plus-AI review does not satisfy D20 or the RC/GA approval gate.

### B02 local implementation evidence — 2026-08-21

B02 is implemented in the current working tree without a commit, push, PR edit, model upload, workflow change, or release action. The implementation:

- adds a single canonical RGB float32 `0..255` input boundary with default deterministic `coerce` behavior and opt-in `strict` behavior described by `InputSpec`; validates layout, one-image batch ownership, dtype, range, channels, finiteness, color, and alpha handling;
- routes every source through the configured reader protocol, makes URL access an explicit bounded `URLReader`, closes decoded/network resources, and rejects source batches with `B>1`;
- makes `AnalysisResult` the sole primary return type, retains the former union only through `run_legacy()`, introduces public error categories, and avoids logging tensor/prediction payloads;
- renames predictor grouping to `face_batch_size`, keeps `batch_size` as a v1.x warning alias, rejects both names together, and proves order-preserving partial batches for faces from one image only;
- keeps detector preprocessing on a working tensor while restoring/cropping from the canonical image, replaces the private face-extraction dependency with a public hook, decodes and clamps selected boxes/landmarks, and keeps padded/custom postprocessor geometry in source-image space;
- removes the detector-mean compensation from shipped unifiers so detected and `skip_detector=True` faces reach predictors through the same normalization path;
- makes JSON console/file logging idempotent, allows file logging after package import, and supports basename/nested log and image-output paths;
- updates the README, bundled scripts, existing tests, and source-test path rewriting to match the v1 runtime contract.

Validation performed on the server CPU/source environment:

- `python -m pytest -q tests/test_b02_runtime_contract.py --tb=short` → **50 passed**.
- The B02 portion of `tests/test_release_blocker_runtime.py` → **18 passed, 5 B03 tests deselected**.
- `python -m pytest -q -m 'not release_blocker' --disable-warnings --tb=short` → **336 passed, 184 skipped, 118 deselected, 74 warnings**.
- `python -m pytest -q -m release_blocker --tb=no` → **71 passed, 47 failed, 520 deselected**. Every remaining runtime failure belongs to B03; the other red contracts belong to B04-B11.
- repository-wide `python -m flake8`, `python -m compileall -q facetorch tests scripts`, and `git diff --check` → **passed**.

No CUDA, installed-wheel/sdist, Docker, or remote-URL integration test was executed for B02 on this server. URL behavior is covered with deterministic mocked response/timeout/redirect/size tests. Those external-environment gates remain assigned to B05/B09 and the trusted local-GPU release evidence; B02 does not make the branch or v1 release ready by itself. Founder-plus-AI review still does not satisfy D20.

### B03 local implementation evidence — 2026-08-21

B03 is implemented in the current working tree without a commit, push, PR edit, artifact download/upload, workflow change, or release action. The implementation:

- registers the detector, predictors, and utilizers lazily, keeps name inspection load-free, and caches each constructed component for the analyzer lifetime;
- validates include/exclude selections before image reading or component construction: `None` uses configured defaults, an empty include runs none, an empty exclude omits none, unknown and duplicate names fail, and include/exclude cannot be combined even when either is empty;
- executes selected predictors in configuration order, never constructs a skipped detector, avoids predictor construction when no faces exist, and avoids loading a selection-linked utilizer when its predictor did not run;
- lock-protects first component construction while documenting that full concurrent `run()` safety depends on potentially stateful custom processors;
- replaces arbitrary detector/predictor constructor attribute injection with explicit parameters and forwards `native_model_class`, `compile_model`, and `compile_options` to `BaseModel`.

Validation performed on the server CPU/source environment:

- `python -m pytest -q tests/test_b03_component_lifecycle.py --tb=short` → **16 passed**.
- B02 plus runtime-blocker and B03 focused suites → **89 passed, 5 warnings**.
- An actual composed default configuration initialized with **zero loaded detectors, predictors, or utilizers**; a controlled no-model-load access smoke resolved the real `fer` predictor, detector, and box utilizer and recorded only those components as cached.
- `python -m pytest -q -m 'not release_blocker' --disable-warnings --tb=short` → **336 passed, 184 skipped, 134 deselected, 74 warnings**.
- `python -m pytest -q -m release_blocker --tb=no` → **92 passed, 42 failed, 520 deselected, 6 warnings**. All B03 contracts pass; the remaining failures belong to B04-B11.
- repository-wide `python -m flake8`, `python -m compileall -q facetorch tests scripts`, and `git diff --check` → **passed**.

No real model download, production `torch.compile` backend run, CUDA execution, Docker run, installed artifact test, or independent human review was performed for B03. Compile forwarding is tested with a real temporary TorchScript module and a patched `torch.compile` call boundary. GPU, packaging, model-cohort, publication, and independent release-approval gates remain assigned to later batches.

### B04 local implementation evidence — 2026-08-21

B04 is implemented in the current working tree without a commit, push, PR/gist edit, artifact download/upload, workflow change, or release action. The implementation:

- adds `facetorch.load_config()` backed by importlib-compatible packaged Hydra resources, with explicit CPU/GPU profiles and ordered Hydra overrides;
- adds `facetorch.load_config_from_path()` for advanced external YAML trees, composing their defaults rather than returning an unresolved top-level list;
- mirrors all runtime configuration groups beneath `facetorch.configs`, includes them as wheel/sdist package data, and keeps source/runtime copies under an executable synchronization contract;
- replaces Docker-only artifact paths with versioned OS-appropriate user cache paths and supports `FACETORCH_CACHE_DIR`, `FACETORCH_MODEL_DIR`, and `FACETORCH_METADATA_DIR` deployment overrides without creating directories during configuration or analyzer initialization;
- disables file logging and image output by default, makes permission failures name the unwritable target and remedy, and removes `/opt` and bundled test-image paths from public/default and merged configs;
- gives production Compose CPU/GPU services explicit separate persistent caches at the same container path, avoiding accidental cross-runtime cache sharing;
- documents packaged loading, external-config compatibility, cache locations, container mounts, and the low-level status of direct `OmegaConf.load("conf/config.yaml")` usage.

Validation performed on the server CPU/source environment:

- `python -m pytest -q tests/test_b04_configuration_contract.py --tb=short` → **16 passed**, including read-only CWD, CPU/GPU profiles, scalar/group overrides, external trees, zip imports, lazy paths, OS conventions, Compose mounts, and permission diagnostics.
- The installed-wheel composed-default smoke built the current wheel/sdist, installed the wheel outside the checkout, and passed from an empty read-only directory → **1 passed**.
- B02-B04 focused runtime/configuration suites → **105 passed, 5 warnings**.
- `python -m pytest -q -m 'not release_blocker' --disable-warnings --tb=short` → **336 passed, 184 skipped, 150 deselected, 74 warnings**.
- `python -m pytest -q -m release_blocker --tb=no` → **110 passed, 40 failed, 520 deselected, 6 warnings**. All B04 contracts pass; remaining failures belong to B05-B11.
- repository-wide `python -m flake8`, `python -m compileall -q facetorch tests scripts`, and `git diff --check` → **passed**.

No real first-download or cached-model inference, Docker execution, CUDA execution, model-integrity/offline/cache-migration check, complete wheel/sdist allowlist, or independent human review was performed for B04. The remaining non-root download/inference acceptance item is intentionally left open for B05/B06 evidence; B04 establishes and validates its writable path/configuration layer only. The earlier gist remains stale.

### B05 local implementation evidence — 2026-08-21

B05 is implemented in the current working tree without a commit, push, PR/gist edit, model download/upload, workflow/public-metadata change, or release action. The implementation:

- replaces implicit namespace discovery with an explicit regular-package allowlist rooted at `facetorch`, makes every packaged configuration group an explicit package, and keeps all runtime YAML resources inside that namespace;
- defines executable wheel and sdist content allowlists, rejects generated/sensitive content, and proves that `conf`, `data`, `docs`, tests, scripts, and model definitions do not leak into the wheel;
- retains the dependency-sync maintenance script in the sdist and includes both CPU and GPU environment inputs, the public notebook, examples, source configuration, tests, and documented source material intentionally;
- installs both wheel and sdist without dependencies into prepared isolated targets, imports/composes configuration outside the checkout, and runs the README no-download API smoke from a read-only empty directory;
- replaces repository-relative Hydra entry points in all public scripts with explicit command-line inputs and the packaged loader;
- replaces the notebook's mutable `main` downloads, range package requirement, obsolete response union, and deprecated batching flag with exact `1.0.0rc1`/tag inputs, `load_config`, `face_batch_size`, and `AnalysisResult`; all outputs and execution counts are cleared;
- documents first-use/cache behavior and the measured approximately 1.2 GB default cohort footprint, recommending at least 2 GB free space for staging and metadata.

Validation performed on the server CPU/prepared-dependency environment:

- B05 artifact/example contracts → **11 passed**; both built artifacts installed outside the checkout, the extracted-sdist dependency check passed, public script help ran from an empty directory, every README Python block parsed, and the marked README smoke executed from the installed wheel.
- Fresh copied-source `python -m build --no-isolation --wheel --sdist` → **passed without warnings**; `check-wheel-contents` → **OK**; `twine check` → **PASSED** for wheel and sdist.
- A fresh uv-created environment installed the wheel over prepared system dependencies and ran the exact README smoke from a read-only empty directory without loading models.
- Diagnostic artifact SHA-256 values: wheel `d8a37ff0…b794b7`; sdist `38720116…15ab4f`. These are validation artifacts, not approved release artifacts.
- B02-B05 cumulative focused contracts → **116 passed, 5 warnings**.
- `python -m pytest -q -m 'not release_blocker' --disable-warnings --tb=short` → **336 passed, 184 skipped, 157 deselected, 74 warnings**.
- `python -m pytest -q -m release_blocker --tb=no` → **120 passed, 37 failed, 520 deselected, 6 warnings**. All B05-owned regressions pass; remaining failures belong to B06-B11.

The notebook's real detector/predictor execution, verified first download, no-network cached inference, and full non-root download/infer path were not run: the exact RC does not yet exist and executing current mutable/unverified model inputs would preempt B06's trust boundary. Normal dependency resolution across Python boundaries and production CPU/GPU image inference remain B09 gates. No Docker, CUDA, external network inference, independent human review, or remote synchronization was performed; the earlier gist remains stale.

### B06 local implementation evidence — 2026-08-21

B06 is implemented in the current working tree without a commit, push, PR/gist edit, model upload, workflow/public-release action, or automatic deletion of an old cache. The implementation:

- packages a versioned provisional manifest for ten model families and forty artifacts, with immutable 40-character Hugging Face revisions, authenticated sizes/SHA-256 digests, real `.pt2`/`.pt` formats, runtime/schema ranges, device eligibility, validation-metadata references, source-weight fields, export-commit fields, and license references;
- replaces synthesized filename and network-error cascades with one exact manifest selection, pins every Hub request, stream-copies into a same-filesystem temporary file, verifies size/hash/non-executing format, atomically promotes it, re-verifies existing cache entries, quarantines corruption, and lock-protects concurrent first use;
- adds explicit environment/config/API offline mode, persistent manifest/runtime/device incompatibility records, fail-closed external download descriptors, real-extension legacy caching, one `LegacyModelWarning`, default-disabled legacy selection, and CPU-only legacy eligibility (including an unconditional prohibition of AU legacy CUDA selection);
- authenticates 3D alignment metadata and loads it with `map_location="cpu", weights_only=True` after verification;
- exposes cost-planned/confirmed selection-aware prefetch, non-executing old-cache inspection, exact-manifest copy migration, quarantine reporting, and explicit versioned-cache-only cleanup; old cache files are never automatically rewritten or deleted;
- updates source, packaged, Google Drive compatibility, test, and generated merged configurations; packages the manifest in wheel/sdist; and documents offline deployment, prefetch cost, legacy opt-in, migration, quarantine cleanup, and v0.6.x rollback isolation.

Validation performed on the server CPU/Torch 2.11 environment:

- focused downloader, artifact/cache, restart, and base-model suites → **37 passed, 4 skipped**; coverage includes corrupt/truncated/wrong-hash/wrong-format rejection, interrupted replacement preservation, immutable revision use, offline no-network behavior, no network-error cascade, two independent legacy-loading processes, persistent fallback state, real extensions, concurrent first-use convergence, exact prefetch selection, restricted metadata loading, migration, and confirmed cleanup;
- configuration plus built wheel/sdist distribution contracts → **24 passed**, including inclusion and installed availability of `facetorch/models/manifest.json`;
- `python -m pytest -q -m release_blocker --tb=short` → **135 passed, 33 failed, 519 deselected, 6 warnings**; all B06-owned blockers pass and the 33 remaining failures map to B07-B11;
- complete `python -m pytest -q --tb=short` after migrating the already-present authenticated test cache → **469 passed, 184 skipped, 33 failed, 80 warnings**; every failure is a pre-existing B07-B11 release contract, while historical detector/predictor integrations loaded the real verified Torch 2.11 cohorts;
- an explicit offline real-inference smoke patched both Hub and Google Drive calls to fail if reached, then verified the cached detector plus FER cohort and produced **4 faces with 4 FER predictions** with no network access;
- all ten pre-existing generic test-cache exports independently matched the packaged Torch 2.11 descriptors and were copied, not moved or deleted, through `migrate_legacy_artifact()` to their authenticated filenames; this created approximately **1.50 GB** of ignored test-cache copies for validation;
- targeted `flake8`, `compileall`, JSON parsing, dependency synchronization, and `git diff --check` → **passed**. `black --check` was unavailable on the server.

The manifest deliberately remains `provisional`: final export commits, independent validation references, complete provenance/rights sign-off, regenerated cohorts, bounded runtime support, and CPU/local-GPU matrix approval belong to B07/B08. No real clean-cache Hub first download, Docker/non-root production inference, CUDA execution, external network inference, or independent human review was performed. The server has no GPU; the founder's local GPU remains the later trusted release environment. Google Drive model variants now fail closed because their historical objects lack approved hashes; the authenticated Hugging Face legacy route is the supported opt-in path. The earlier gist is stale.

### B07 local implementation evidence — 2026-08-21

B07 is implemented in the current working tree without a commit, push, PR/gist edit, Hub write, model-manifest promotion, credential use, workflow action, or public release action. The implementation:

- recursively rejects NaN/Inf and empty tensors in native-reference, exported, and comparison outputs before drift statistics; enforces matching nested structure, shapes, and dtypes; evaluates declared output schemas and task invariants; and records deterministic output fingerprints;
- requires every requested device to report `ok`, treats skipped CUDA and zero/incomplete device/batch/shape matrices as failures, validates predictor face batches 1/2/4/8, adds three single-image detector spatial shapes, and directly compares successful device outputs;
- reconstructs every state-dict strategy strictly, with AU's only exception limited to explicitly matched timm-generated `attn_mask` and `relative_position_index` buffers; unexpected learned or undeclared keys remain fatal;
- binds all ten model specs to their original TorchScript validation artifact and exact SHA-256 instead of comparing only with the reconstruction used by the exporter, and records per-task float32 tolerances with an explicit provisional justification pending B08/Q09 approval;
- disables inline export upload and emits deterministic `.pt2` plus canonical metadata for a fixed model/runtime/input; staging-plan preparation re-verifies every file, model, cohort, executed case, fingerprint, and required-device result;
- requires immutable 40-character parent commits plus a complete-plan approval bound to the canonical plan SHA-256; commits all requested cohorts and metadata for one model in a single plan-specific candidate-branch commit, atomically checkpoints a resumable receipt, and creates the immutable candidate-manifest commit only after every model repository succeeds;
- compares repeated model/device/case records across cohorts only when they share the same independently fingerprinted reference output, recording exact agreement or a reference-derived numerical bound; a differing reference or incomplete matrix blocks publication.

Validation performed on the server CPU/Torch 2.11 environment:

- focused validator and publisher regressions → **21 passed**; negative coverage includes recursive NaN/Inf, zero cases, skipped required CUDA, incomplete face batches, undeclared state keys, non-`ok` validation, direct upload, staged-byte tampering, stale approval, late model failure, interrupted publication/resume, manifest ordering, and mismatched cross-cohort references;
- cumulative B06+B07 downloader/cache/base-model/validator/publisher suites → **58 passed, 4 skipped**;
- configuration plus built wheel/sdist distribution contracts → **24 passed**, including the packaged provisional manifest and inclusion of the separate publisher in the source distribution;
- a real authenticated FER-B0 Torch 2.11 cohort was exported and independently compared with its digest-pinned original TorchScript reference across face batches 1/2/4/8 and two deterministic input variants → **8 cases, status `ok`, zero maximum drift**;
- two independent real FER-B0 exports produced byte-identical `.pt2` artifacts and canonical metadata; a real publication plan then re-verified their staged bytes and validation evidence without network access;
- `python -m pytest -q -m release_blocker --tb=no` → **157 passed, 23 failed, 519 deselected, 6 warnings**; every B07-owned contract passes and remaining failures map to B08-B11;
- complete `python -m pytest -q --tb=short` → **492 passed, 184 skipped, 23 failed, 80 warnings**; the same known B08-B11 release contracts are the only failures;
- targeted `flake8`, `compileall`, dependency synchronization, JSON parsing, CLI smoke, and `git diff --check` → **passed**. `black --check` remains unavailable on the server.

The publisher was exercised only against deterministic fake Hub clients: no remote branch, artifact, or manifest was created. An upload-time failure can leave immutable, unpromoted candidate commits in some model repositories, because no cross-repository service can make those commits one transaction; the durable receipt resumes them and the release manifest remains absent until all succeed. No real CUDA case, all-model/all-cohort export, cross-runtime comparison, approved tolerance decision, provenance/rights approval, Hub permission test, or independent-human sign-off was performed. Those are B08 and later release gates. The server has no GPU; the founder's local NVIDIA GPU remains the planned trusted CUDA environment.

### B08 local implementation evidence — 2026-08-21

B08 is implemented in the current working tree without a commit, push, PR/gist
edit, Hub upload, manifest promotion, tag, or release action. The implementation:

- bounds Python to 3.10-3.12 and Torch to only 2.3.x, 2.6.x, and 2.11.x; package,
  conda-environment, runtime-routing, documentation, and machine-readable policy
  use the same disjoint specifier, while every other Torch minor fails before any
  model network request;
- declares Linux x86-64 CPU plus exact Torch/CUDA pairs 2.3.1/12.1, 2.6.0/12.4,
  and 2.11.0/13.0 as the candidate matrix; other systems, architectures, and MPS
  remain experimental;
- pins and inventories all ten immutable legacy sources; records Python, Torch,
  torchvision, timm, CUDA/cuDNN/GPU, export schema, source-tree state, lock digest,
  exporter arguments, source digest, and strict reconstruction evidence;
- keeps every immutable reference on CPU, disables TensorFloat-32, selects highest
  float32 precision and deterministic cuDNN behavior, restores caller settings,
  and makes the complete numeric policy/tolerances part of release verification;
- defines predictor batching as independent faces from one image. AU validation
  concatenates one-face golden calls because the legacy trace is batch-coupled;
  its three published programs are preserved byte-for-byte pending recovery of
  the original native checkpoint mapping;
- recovers every detector BatchNorm tensor omitted from the legacy `state_dict()`
  from verified TorchScript attributes, strictly loads the complete state, and
  regenerates dynamic multiple-of-32 detector programs for all three cohorts;
- adds fail-closed compatibility/governance manifests, Hub LFS/byte audit tooling,
  complete three-lane verification, dependency synchronization, documentation,
  responsible-use limitations, and executable B08 contracts. All ten governance
  records remain deliberately ineligible while checkpoint mapping and weight
  redistribution evidence are incomplete.

Validation performed on Linux x86-64/Python 3.10 with a local NVIDIA GeForce RTX
3090:

- `torch==2.3.1+cu121`/`torchvision==0.18.1+cu121`,
  `torch==2.6.0+cu124`/`torchvision==0.21.0+cu124`, and
  `torch==2.11.0+cu130`/`torchvision==0.26.0` each exported/loaded all ten models
  and reported CPU plus CUDA `ok`;
- the full matrix executed **1,872 cases** across **30 artifacts**: predictor face
  batches 1/2/4/8, detector image batch one at three spatial shapes, two seeds,
  two input scales, and random-normal/random-uniform inputs on both devices;
- `scripts/verify_model_release_matrix.py --candidate-evidence
  --allow-dirty-source` re-verified all artifact/metadata digests, runtime schemas,
  CUDA pairs, sources, numeric policy, exact case identities, and device status →
  **3 cohorts x 10 models, status `ok`**;
- worst observed reference drift was detector CUDA maximum `6.06e-5` and mean
  `4.01e-6`, below the declared bounds; focused B07/B08/model-release contracts →
  **38 passed**;
- all 10 Hub repositories and immutable revisions were reachable; manifest LFS
  SHA-256/size records matched **40/40 objects**, while the 30 remote cohort
  metadata files were correctly classified as legacy rather than accepted as new
  release evidence;
- `uv lock` and `uv lock --check` synchronized the bounded dependencies;
  dependency synchronization, JSON parsing, `compileall`, repository-wide
  `flake8`, and `git diff --check` passed;
- `python -m pytest -q -m release_blocker --tb=short` → **169 passed, 17 failed,
  524 deselected, 6 warnings**; `python -m pytest -q --tb=short` → **509 passed,
  184 skipped, 17 failed, 80 warnings**. Every B08 contract passes; the 17 known
  failures are exclusively B09 CI/container parity or B10-B11 release workflow,
  security, migration, and runbook contracts.

This matrix is strong candidate evidence but not release evidence: it was produced
from an uncommitted tree, 27 staged artifacts differ from the current remote
cohort bytes, and no remote write was authorized. The exact clean commit must be
re-run before publication. Q06-Q07 rights/provenance, the Q08 Hub permission dry
run, independent-human approval under D20, and B09-B11 remain open.
No model may be marked eligible and no candidate may be published until those
gates are resolved.

### B09 local implementation evidence — 2026-08-22

B09 is implemented in the current working tree without a commit, push, PR/gist
edit, remote workflow run, credential use, image push, package publication, tag,
or release action. The implementation:

- defines six exact uv profiles for Torch 2.3.1, 2.6.0, and 2.11.0 across their
  supported CPU and CUDA lanes, plus a synchronized root lock; the CPU conda
  environment is exact, while the GPU conda file intentionally supplies only
  Python/CUDA system packages and consumes the exact CUDA 12.4 uv graph because
  conda-forge cannot solve Torch 2.6 with CUDA 12.4;
- expands CI to build/test the branch wheel and sdist in empty Python 3.10-3.12
  environments, run every supported CPU cohort, verify both conda locks against
  the branch wheel, audit all dependency graphs, and build the production images;
- adds an owner-only manual local-GPU release workflow bound to a protected exact
  SHA and an ephemeral runner label. It rejects persistent Hub, PyPI, and Docker
  publication credentials, requires all requested devices to be `ok`, and emits
  model, wheel, notebook, container, and runner evidence without publishing;
- rebuilds production CPU/GPU images from the branch wheel and exact Torch 2.6
  locks, with no source checkout in the final stage and UID 10001 by default;
- adds reproducible hashed dependency exports, CycloneDX 1.5 SBOMs, and a pinned
  `pip-audit` gate with bounded, expiring, founder-approved exceptions only;
- executes the public notebook against the exact installed wheel, local image,
  staged authenticated artifacts, and disabled package/network indexes;
- prevents ignored build residue and nested virtual environments from contaminating
  wheels or Docker contexts, securely rebases host-absolute evidence paths inside
  read-only containers, and makes staged artifacts/reports readable by the
  non-owner container/host boundary without making the container root.

Validation completed locally:

- all seven uv locks, dependency/profile/conda synchronization, targeted flake8,
  compile checks, `git diff --check`, and actionlint across the eight B09 workflows
  passed;
- focused exporter, compatibility, CI, and release-contract tests passed **55/55**;
  distribution contracts passed **8/8** and now copy about 75 MB rather than
  5.5-7.4 GB of irrelevant local model caches per run;
- the release-blocker suite reported **187 passed and 10 failed**;
- the complete suite reported **528 passed, 184 skipped, 10 failed, 80 warnings**;
  all ten failures are the already-scoped B10 transactional-publication or B11
  documentation/security/runbook contracts, not B09 regressions;
- the CPU and GPU production images built from a **9.8 MB** context and each ran
  the full default analyzer three times with four detected faces, all seven
  predictors, PT2-only artifacts, no network, a read-only filesystem, UID 10001,
  and zero AU repeat drift. GPU evidence used Torch 2.6.0+cu124 on the local
  NVIDIA GeForce RTX 3090; temporary validation image tags were removed afterward;
- the CUDA installed-wheel smoke and current public notebook both passed with all
  configured predictors and offline staged artifacts;
- the dependency audit produced SBOMs for all seven profiles. Torch 2.3 CPU/CUDA
  and 2.6 CPU/CUDA are clean without exceptions; root plus Torch 2.11 CPU/CUDA
  accept only the founder-approved, exact-version `PYSEC-2026-3447` exception
  through 2026-11-20. The approved run is green, and two independent pre-approval
  runs produced identical hashed requirements and normalized SBOM graph digests.

These are still dirty-tree diagnostics, not release evidence. No exact clean-SHA
local-runner workflow, hosted Python/CPU matrix, conda-lock installation, remote
CI, or immutable image/wheel promotion was executed. Governance and model rights
remain fail-closed, the setuptools exception requires expiry/removal monitoring,
and no independent human reviewer is available. B10-B12 have since been
implemented locally; remote release gates remain open.

### Approved Torch support/security amendment — 2026-08-22

This amendment supersedes the three-cohort support promise described in the
historical B08 and B09 implementation evidence above. The founder approved:

- dropping Torch 2.3 from package metadata, runtime routing, model manifests,
  exact environments, CI, release tooling, and current public documentation
  because GHSA-53q9-r3pm-6pq6 is a critical advisory affecting
  `torch.load(weights_only=True)`;
- retaining Torch 2.6 as the validated CUDA 12.4 production cohort, alongside
  Torch 2.11, rather than forcing all users onto CUDA 13.0 immediately;
- accepting only GHSA-887c-mr87-cxwp, GHSA-vgrw-7cvw-pwgx, and
  GHSA-f4hp-rmr7-r7v8 for the exact Torch 2.6 CPU/CUDA profiles through
  2026-11-20. Their affected APIs are not used by Facetorch, and executable
  release contracts enforce that premise.

Existing Torch 2.3 objects in the separately owned Hugging Face repositories are
not part of the v1 manifest and need not be destructively removed from their
immutable remote history.

### B10 local implementation evidence — 2026-08-22

B10 is implemented in the current working tree without a commit, push, tag,
release, package/image/model publication, repository-setting mutation, or other
remote write. The implementation:

- replaces the prior release scripts with one reusable dry-run/publication
  workflow bound to a protected ref's exact SHA, strict stable/RC tag grammar,
  project metadata, changelog, approved immutable Hub manifest, and a canonical
  digest plan;
- builds the wheel, sdist, and versioned CPU/GPU images once, then requires the
  ephemeral local RTX runner to download those exact image archives and run the
  complete approved CPU/CUDA cohort matrix, notebook, and full offline analyzer
  in both exact image IDs before the plan can exist;
- keeps GitHub releases draft while Docker and PyPI converge, uses PyPI OIDC,
  Docker password-stdin, checksums, SBOMs, attestations, immutable receipts, and
  stable-only final alias promotion; identical failed-job retries resume while
  different bytes stop;
- pins every external action, build backend, Python toolchain, Buildx setup,
  base/external container image, uv version/image, Miniforge installer, and SBOM
  tool in a reviewed input registry, and normalizes image build timestamps from
  the source commit;
- adds dependency review, Gitleaks, Dependabot configuration, `SECURITY.md`, the
  third-party `MODEL_NOTICE.md`, private-report fallback contact, privacy/network
  boundaries, and fail-closed per-model rights/provenance validation.

Validation completed locally:

- B10 transaction and workflow contracts passed **29/29**, including malicious
  input/path rejection, exact bundle coverage, governance/device evidence,
  partial-channel failure/resume, digest drift, RC alias refusal, PyPI partial
  reconciliation, and Docker identity reconciliation;
- combined release/CI/distribution contracts passed **63 tests** before B11
  documentation was added;
- after B11 documentation was added, the complete repository suite reported
  **562 passed, 184 skipped, 80 warnings**; no test failures remain;
- actionlint across every workflow, exact action-pin registry checks, flake8,
  compile checks, JSON parsing, dependency synchronization, lock checking, and
  `git diff --check` passed;
- a fresh wheel/sdist build passed Twine and wheel-content checks, included the
  model notice in both formats, and excluded nested environment virtual-state;
- the pinned CPU and GPU Dockerfiles rebuilt successfully through the server's
  legacy builder and passed non-root, no-network, read-only offline smokes. The
  server lacks Buildx, so the SHA-pinned Buildx action and reproducible BuildKit
  export path still require the first hosted dry run.

The remote repository was inspected read-only. The five required release
environments do not exist, private vulnerability reporting and secret-scanning
features are disabled, Actions defaults to a write token, SHA-pin enforcement is
off, and main protection names need B09 reconciliation. The packaged model
governance remains intentionally incomplete, no exact clean-SHA local-GPU job or
remote no-publish dry run has executed, and no independent human reviewer/backup
operator exists. Each condition is fail-closed and remains an RC/publication
blocker rather than an implementation success claim.

### B12 integrated review evidence — 2026-08-22

The external B01-B11 review at
`gist.github.com/tomas-gajarsky/296f9b212d4962c7370c6313b707eda9`
was reproduced finding by finding. All three blockers, five high-severity items,
eight medium items, and nine cleanup items were addressed. Material corrections
include making `facetorch/models` commit-visible, preserving the actionable final
schema error plus an explicit incompatibility-reset API, regenerated and
release-gated pdoc output, non-loading registry membership, bounded URL targets,
stale-lock recovery, same-filesystem artifact promotion, reduced tensor scans,
clear HWC guidance, rotating logs, and complete sdist contents.

Two reviewer recommendations were refined rather than followed literally:

- full cached-artifact hashing remains the secure default; trusted read-only
  deployments now have an explicit `verify_on_use: false` opt-out, while release
  validation must keep it enabled;
- development setuptools is pinned to the approved `81.0.0` exception because
  forcing build-backend `84.0.0` into runtime/test profiles makes both Torch 2.11
  CPU and CUDA dependency graphs unsatisfiable. The build backend remains 84.0.0,
  and the exception still expires on 2026-11-20.

Validation after remediation: **576 passed, 184 skipped, 80 expected warnings**;
focused reviewer regressions passed **143 tests with 4 skips**; flake8, actionlint,
dependency synchronization, all seven lock checks, generated-doc byte comparison,
and `git diff --check` passed. Fresh wheel/sdist checks passed Twine and
check-wheel-contents. An alternate Git index plus archived candidate-tree build
proved that all four `facetorch/models` trust-root files would be committed and
were present in the wheel, without modifying the real index or creating a commit.

Sizes are comparative only: `M` is a focused multi-file change, `L` is a substantial cross-cutting change, and `XL` spans multiple runtime or publication environments. Add calendar estimates only after the decisions and reviewer/GPU capacity are known.

Each batch should include:

- a focused problem statement and threat/user-impact statement;
- before/after behavior and breaking-change note;
- tests and exact environments exercised;
- documentation/config updates in the same review;
- no unrelated formatting churn;
- a link back to the relevant W/D/F identifiers in this plan.

## Finding-to-work traceability matrix

`Gate` is the earliest release gate that must prove the fix. No entry is currently approved for deferral.

| Finding ID | Severity | Audited problem | Planned work | Gate |
| --- | --- | --- | --- | --- |
| F01 | P0 | `skip_detector=True` corrupts predictor preprocessing | W1.1 | W1 |
| F02 | P0 | Normal installed wheel cannot use the documented caller-relative config and Docker-only paths | W2.1, W2.2, W2.5 | W2 |
| F03 | P0 | Legacy `.pt` fallback is cached as `.pt2` and fails after restart | W3.2, W3.7 | W3 |
| F04 | P0 | AU legacy TorchScript fallback can reintroduce the documented CUDA hang | W3.3, W4.1 | W3 |
| F05 | P0 | PyTorch 2.5 has no compatible export cohort despite broad support metadata | W3.1, W3.6 | W3 |
| F06 | P0 | Release automation can create a partial release and cannot safely resume | W4.3 | W4 |
| F07 | P0 | Manual release input can control shell/tag behavior and jobs build a different ref | W4.3 | W4 |
| F08 | P0 | Frozen Python 3.13 environment does not resolve | W4.1, W4.2 | W4 |
| F09 | P1 | Tensor/NumPy layout inference rejects or corrupts valid small/ambiguous shapes | W1.1 | W1 |
| F10 | P1 | Detector padding restoration depends on a private method and exposes inconsistent/out-of-bounds geometry | W1.2 | W1 |
| F11 | P1 | `compile_model` is ignored by predictor/detector subclasses | W1.3 | W1 |
| F12 | P1 | Excluded predictors and skipped detector are still eagerly constructed/downloaded | W1.3 | W1 |
| F13 | P1 | Empty includes run all; invalid/multiple selections are not handled consistently | W1.3 | W1 |
| F14 | P1 | Non-string inputs bypass the configured reader contract | W1.4 | W1 |
| F15 | P1 | File logging is suppressed by the import-time handler and output basename paths fail | W1.5 | W1 |
| F16 | P1 | Input float range is ambiguous and unvalidated | W1.1 | W1 |
| F17 | P1 | Reader tests contain contradictory skip conditions; padding/skip/logger regressions are absent | B01, W1.1-W1.5 | W1 |
| F18 | P1 | Direct bytes reader does not close PIL object | W1.1, W1.4 | W1 |
| F19 | P1 | Wheel installs generic `conf`, `data`, and `docs` namespaces | W2.1, W2.3 | W2 |
| F20 | P1 | Notebook reads nonexistent image data and consumes mutable/unreleased inputs | W2.4 | W2 |
| F21 | P1 | Sdist includes a maintenance script but omits its required GPU environment input | W2.3 | W2 |
| F22 | P0 | Cohort validator accepts NaN/Inf output as `ok` | W3.4 | W3 |
| F23 | P0 | Required CUDA can be skipped while overall validation remains `ok` and upload proceeds | W3.4, W3.5 | W3 |
| F24 | P1 | Validator compares export to the same reconstruction and can miss wrong weights/architecture | W3.4 | W3 |
| F25 | P1 | AU reconstruction tolerates missing state keys | W3.4 | W3 |
| F26 | P1 | Model publication uploads partial sets before later validation completes | W3.5 | W3 |
| F27 | P0 | Mutable unpinned/unverified remote model artifacts are executed | W3.2, W3.5 | W3 |
| F28 | P1 | Hosted metadata validation is heterogeneous and lacks reproducible source/environment details | W3.6 | W3 |
| F29 | P2 | MagFace architecture location contradicts model-definition reproducibility claim | W3.6 | W3 |
| F30 | P1 | Detector validator does not exercise claimed dynamic H/W coverage | W3.4 | W3 |
| F31 | P1 | CI full tests cover only one Python 3.12 CPU/Torch 2.3.1 environment | W4.1 | W4 |
| F32 | P1 | Install CI only installs and conda CI tests public v0.6.2 instead of the branch | W4.1 | W4 |
| F33 | P1 | Test, production Docker, GPU, and conda environments resolve incompatible/unverified Torch versions | W3.1, W4.2 | W4 |
| F34 | P1 | Locked dependency graph contains known advisories | W4.2, W4.4 | W4 |
| F35 | P2 | Floating Actions/tools/images and long-lived release credentials weaken the supply chain | W4.3, W4.4 | W4 |
| F36 | P2 | No `SECURITY.md`, SBOM/provenance, dependency review, or complete model notice audit | W4.4 | W4 |
| F37 | P1 | Changelog claims an unreleased date and no v1 migration/rollback guide exists | W4.5 | W4 |
| F38 | P1 | Open issues #82/#83/#88 are not linked/closed with verified regressions | W4.5 | W4 |
| F39 | P1 | Bare local tests can import a stale installed package before repository path setup | B01, W4.1 | W1/W4 |
| F40 | P1 | Production resolution currently selects a newer untested Torch than the tested lock | W3.1, W4.2 | W4 |
| F41 | P1 | Public return mode and notebook disagree about where retained image data exists | W1.6, W2.4 | W1/W2 |
| F42 | P2 | `OS Independent` and broad CUDA wording exceed the tested platform matrix | W3.1, W4.5 | W3/W4 |
| F43 | P1 | Code license does not establish per-model weight rights, provenance, or responsible-use limitations | W3.6, W4.4 | W3/W4 |
| F44 | P1 | Merge is blocked with no qualifying independent approval/release-role assignment | D20, W4.6 | W4 |
| F45 | P1 | No formal patch/yank/model-revocation and post-release support policy exists | D21, W4.3, W4.6 | W4 |
| F46 | P2 | URL/network/privacy behavior is not a deliberately bounded public contract | D07, D13, D17, W1.1, W4.4 | W1/W4 |

### Deferral rule

The default is to address all F01-F46 before stable v1. Only a P2 item may be proposed for deferral, and it must add all of the following to this document before approval:

- founder approval and rationale;
- public user impact and workaround;
- linked issue with owner and milestone;
- evidence that it cannot compromise correctness, executable-artifact trust, installation, migration, or release recovery;
- documentation in the RC known-limitations section.

All P0/P1 findings are stable-v1 blockers and cannot be deferred under this plan. P0 findings F01-F08, F22, F23, and F27 block even an RC unless that RC exposes only a safe disabled replacement path for the affected feature.

## Verification and evidence plan

Commands may be wrapped by CI, but evidence must identify the exact source SHA, built artifact hashes, dependency profile, model-manifest revision, device, and command.

| Layer | Minimum evidence |
| --- | --- |
| Static/source | `flake8`, dependency synchronization, `compileall`, focused type/API checks if adopted, and `git diff --check` |
| Unit/regression | `python -m pytest tests -q` with skip report and coverage threshold; focused tests for every F01-F46 behavioral finding |
| Existing Docker test | `docker compose -f docker-compose.dev.yml run --build facetorch-tests` against the candidate source |
| Distribution | clean `python -m build`; `twine check dist/*`; `check-wheel-contents`; wheel/sdist allowlist inspection |
| Empty-directory wheel | create a clean environment outside the checkout, install the exact wheel, compose default config, import, and run documented smoke |
| Python matrix | frozen/controlled installs and runtime smoke on every D03 Python version, including a resolvable 3.13 path if retained |
| Torch/cohort matrix | actual load plus inference for every accepted Torch range/cohort, not schema inspection alone |
| CUDA | default analyzer with all configured predictors, repeated AU inference, skip/select paths, and cohort validation on a trusted runner |
| Models | independent golden/task validation, finite output, strict weight load, dynamic-shape cases, restart/offline/cache-corruption tests |
| Containers | version-locked CPU/GPU production image build; non-root startup/config/download/inference; immutable image digests |
| Conda | solve declared files, install the local candidate wheel/package, import/config/inference smoke; feedstock handoff tracked separately |
| Security | dependency audit, static scan, secret/dependency review, license/provenance check, SBOM, checksums, and attestations |
| Notebook/docs | execute notebook; run every README quick-start snippet; validate local links and version references |
| Release | no-publish dry run plus injected failure/retry tests; exact artifact digest comparison across jobs/channels |

The final release evidence bundle should contain:

- source SHA and signed/annotated tag information;
- founder decision register and independent review approval;
- CI run links and machine-readable test reports;
- wheel/sdist/image checksums and SBOM/provenance;
- model manifest revision, all artifact hashes, and complete validation report;
- compatibility matrix and dependency locks;
- release notes, migration guide, security policy, third-party notices, and rollback runbook;
- post-publication verification results for PyPI, Docker, GitHub, Hugging Face/model hosting, and conda status.

## Definition of ready for `1.0.0rc1`

- [ ] D01-D22 are answered and reflected in code/docs.
- [ ] F01-F46 are fixed, or a P2 item has an approved documented deferral.
- [ ] W1-W4 release gates pass on one immutable candidate.
- [ ] Public package/model/container artifacts are reproducible and digest-pinned.
- [ ] Migration, offline/cache, compatibility, security, and rollback documentation is reviewable.
- [ ] No known correctness, installability, remote-code/artifact-trust, or release-recovery blocker remains.
- [ ] An independent reviewer approves the candidate.

## Definition of ready for stable `1.0.0`

- [ ] The RC was installed and exercised through the intended public channels without moving stable aliases.
- [ ] RC feedback is triaged; no unresolved P0/P1 regression remains.
- [ ] Any change after RC caused affected tests and release gates to rerun and, when public behavior/artifacts changed, produced a new RC.
- [ ] Changelog date/version and migration links match the exact stable artifacts.
- [ ] PyPI, both Docker images, GitHub release, and model manifest are ready for coordinated transactional publication.
- [ ] `latest` promotion is last and only occurs after every versioned stable artifact succeeds.
- [ ] Conda-feedstock ownership/status and expected availability lag are communicated.
- [ ] Rollback has been tested and an operator is assigned.

## Founder response template

The founder can respond by editing the `Founder answer` column above or completing this block:

```text
D01 delivery/review shape:
D02 RC vs direct stable:
D03 supported Python/PyTorch matrix:
D04 image input/layout/range contract:
D05 installed configuration API:
D06 legacy fallback policy:
D07 online/offline model behavior:
D08 old cache migration behavior:
D09 lazy-loading and selection semantics:
D10 v0.6 compatibility/shims:
D11 canonical release channels and latest policy:
D12 available GPU validation capacity:
D13 security/provenance owner and contact:
D14 breaking-change support window:
D15 library-first/container-first/equal product surface:
D16 stable result type versus union return:
D17 supported extension points and URL behavior:
D18 supported OS/architecture/accelerator matrix:
D19 model-rights and responsible-use bar:
D20 independent reviewer and release authority:
D21 patch/yank/revocation/support policy:
D22 conda launch-gate/asynchronous/community status:

Q01-Q14 answers/evidence (reference IDs as needed):

Approved deferrals, if any (finding ID + rationale + issue/milestone):
Additional constraints or priorities:
```

## Immediate next action after plan approval

B01-B12 now exist locally: runtime and within-image face batching, lazy lifecycle,
packaged configuration, portable distributions, authenticated artifact/cache
handling, fail-closed model publication, the bounded three-cohort matrix, and exact
CI/dependency/container parity plus transactional supply-chain gates are
implemented, and the migration/release runbooks are present. Q15's bounded
setuptools exception is approved through 2026-11-20 and the dependency gate is
green. The external reviewer findings and final B12 audit are resolved locally.
Next, create the reviewed logical commit series, push it for remote CI, and
explicitly configure the remote B10 gates. Do not publish
cohorts, approve model governance, push images/packages, modify release channels,
or create a release tag until the exact clean candidate and every remaining W3/W4
gate pass.
