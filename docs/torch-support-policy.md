# Torch support policy for v1

Decision recorded September 5, 2026: retain broad Torch compatibility as a
product requirement. GA keeps all eight declared lines, Torch 2.6–2.13. A recent
recommended installation must not become an implicit minimum version or remove
older integrations. Resolving dependency findings must preserve this intent;
any unavoidable support change needs a separate owner decision.

## Recommended installation and compatibility baselines

Use **Torch 2.13.0 with torchvision 0.28.0** for new installations within the
current qualified range: the CPU wheels for CPU hosts, or the named CUDA 13.0
wheels on compatible NVIDIA systems. The README already uses these versions.
This is a recommendation, not a forced upgrade for applications embedding
Facetorch into an existing supported environment.

| Torch | torchvision | Official CUDA profile | Artifact cohort |
| --- | --- | --- | --- |
| 2.6.0 | 0.21.0 | 12.4 | 2.6 |
| 2.7.1 | 0.22.1 | 12.6 | 2.6 |
| 2.8.0 | 0.23.0 | 12.6 | 2.6 |
| 2.9.1 | 0.24.1 | 13.0 | 2.11 |
| 2.10.0 | 0.25.0 | 13.0 | 2.11 |
| 2.11.0 | 0.26.0 | 13.0 | 2.11 |
| 2.12.1 | 0.27.1 | 13.0 | 2.11 |
| 2.13.0 | 0.28.0 | 13.0 | 2.11 |

Each row also has an exact CPU profile. Official platform coverage remains Linux
x86-64 and Python 3.10–3.12. Support for a minor line is demonstrated at the exact
patch pair in its frozen profiles; this does not qualify every historical patch,
CUDA wheel, Python version, GPU architecture, or compiler backend.

Keep the production container baselines (Torch 2.6 CPU and CUDA 12.4) distinct
from the recommended library installation, the Torch 2.11 CPU development lock,
and the two artifact export cohorts. Changing the recommended installation does
not silently change deployed image behavior. A future modern default service
image should have an explicit version, frozen profile, image validation, and
migration notes while retaining an identified legacy CUDA path.

## CUDA and hardware coverage

Test the published Torch/torchvision/CUDA wheel combinations. The host NVIDIA
driver must support the selected wheel's CUDA runtime; an unrelated locally
installed CUDA toolkit does not turn an incompatible wheel into a supported
combination. Use the named official indexes and frozen profiles instead of
mixing CPU and CUDA wheels or allowing an image build to upgrade Torch.

Preserve a CUDA 12 installation path as well as CUDA 13. Upstream released
[Torch 2.14 on September 2, 2026](https://pytorch.org/blog/pytorch-2-14-release-blog/)
with torchvision 0.29.0 and CUDA 12.6, 13.0, and 13.2 wheel options.
[Upstream's CUDA transition notice](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432)
says 2.14 is the last release with CUDA 12.x wheels and plans to drop the CUDA
12.6 wheels, and thus Maxwell/Pascal/Volta support, in 2.15. CUDA 13 targets
Turing or newer hardware. This is a concrete reason to maintain older CUDA
coverage rather than treating the newest toolkit as universally preferable.

An RTX 3090 test proves behavior on Ampere with the recorded driver. It does not
prove behavior on Maxwell, Pascal, Volta, Turing, or every later GPU. Before
making an architecture-specific claim, record actual inference on representative
hardware, including each model, shapes/batches, finite outputs, and numerical
bounds. The upstream wheel's advertised architectures establish availability,
not Facetorch qualification on every device.

## How a runtime earns support

1. Resolve the latest suitable patch with its matching torchvision and explicit
   official CPU/CUDA indexes; freeze and audit each complete environment. Prefer
   patch upgrades within a retained line where they exist. The current eight
   frozen Torch pins were already the latest patch releases of their lines in
   the upstream package index on September 5; patch-only upgrades do not remove
   the presently unresolved findings.
2. Validate all ten shipped model programs on every proposed runtime and named
   device. Reuse the immutable CPU golden outputs from the Torch 2.6 reference
   cohort, including all declared batches, detector sizes, seeds, scales, input
   variants, finite-output and task-specific checks, and CPU/CUDA comparisons.
   Preserve per-model metadata and its hashes.
3. Run the library's installed-wheel and source checks, including package/model
   routing, every CPU CI lane, and the complete dependency audit. Keep the
   aggregate release gates synchronized when adding a row.
4. Update package bounds, artifact routing, frozen profiles, checks, model-card
   support statements, and documentation together. A pinned model card needing
   a new immutable revision goes through the existing publication approval path.
   Publish support only after the exact candidate passes the release workflow.

Runtime validation executes the existing pinned programs without rebuilding
exports. Fresh exports remain a separate gate in the two artifact-producing
cohorts. This avoids making a new inference runtime depend on historical export
APIs that it does not need to load the shipped programs.

The release matrix covers the default exported-program inference path.
`torch.compile` is optional; compatibility of every backend, mode, dynamic shape,
and custom graph is not established by those default-path tests. An application
using compilation or custom models must qualify that exact configuration.

## Expansion and maintenance priorities

1. **Before GA:** retain all eight rows, complete their advisory treatment, and
   rerun the exact candidate's full qualification. Broader compatibility does
   not approve a new exception automatically.
2. **Next expansion candidate:** Torch 2.14.0 / torchvision 0.29.0, including CPU,
   CUDA 13.0, and especially CUDA 12.6. The local assessment is recorded in the
   [compatibility investigation](torch-support-investigation.md). Until all
   admission work is complete, the package keeps its existing `<2.14` bound.
3. **Hardware coverage:** seek a reproducible run on older CUDA 12 hardware
   before promising that architecture. Keep newer CUDA coverage alongside it.
4. **Older than Torch 2.6:** do not remove the lower bound speculatively. Torch
   2.5 uses a different export schema and needs a separately verified artifact
   route and security assessment; Torch 2.3 was excluded for a critical loading
   advisory. Prefer retaining useful existing integrations and qualifying 2.14
   before funding a new legacy artifact generation.
5. **At each Torch release and before exception expiry:** review patch releases,
   upstream security fixes, wheel/hardware availability, CI burden, and concrete
   community requests. There is no project telemetry establishing the most-used
   Torch versions, so do not label a line “popular” without evidence. Use issue
   reports and opt-in surveys rather than adding telemetry to library users.

Retirement requires a written rationale, replacement path, advance migration
notice, and an owner decision. A moving recommended default is not a retirement
policy. Security exceptions remain exact, expiring records; approaching their
expiry triggers review rather than an automatic renewal or silent support drop.
