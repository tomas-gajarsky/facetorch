# Local GPU release runner

The local NVIDIA machine is a trusted release executor, not a general CI runner.
The `exact-candidate-local-gpu` workflow has no pull-request, push, or scheduled
trigger. It accepts only an owner-confirmed full commit SHA equal to the protected
workflow ref, and the GPU job uses the `local-gpu-release` environment.

Provision a fresh GitHub Actions runner with the labels `linux`, `x64`, and
`facetorch-ephemeral-gpu`, configure it with GitHub's ephemeral runner option, and
set `FACETORCH_RUNNER_EPHEMERAL=1` in the runner service environment. Install
Docker with the NVIDIA container runtime and exactly `uv 0.9.14`. Do not configure
Hugging Face, PyPI, or container-registry publication credentials on this runner.
The model inputs used by validation are public immutable objects; B09 produces
evidence only and never publishes them.

The coordinated release workflow additionally downloads the exact CPU and GPU
image archives built earlier in that same run, verifies their image IDs, and
executes the full offline analyzer in both before it can create a release plan.
Evidence from a separately rebuilt image remains useful preflight information,
but it cannot substitute for this exact-artifact publication gate.

The release workflow also requires the packaged artifact manifest, compatibility
matrix, and every model-governance record to be approved before it performs the
expensive export. The script offers `--candidate-evidence` only for an explicitly
non-release technical run; the protected GitHub workflow never uses that bypass.

Use a fresh staging root for every protected or diagnostic run. When
`run_local_cuda_release_matrix.py` creates the root, it sets mode `0711`; an
existing root must already have that exact mode or the script fails before the
expensive matrix begins. The root mode lets the production image's non-root UID
10001 traverse known evidence paths without listing the evidence directory.
Newly created cohort, intermediate, and model directories use `0755`; existing
operator-managed export directories keep their original permissions. Before the
container smokes, create `container-reports` with mode `1777`, whose sticky bit
allows the two non-root images to write reports without allowing another local
user to replace them. Never reuse reports from an earlier staging root.

Before dispatch, place the exact candidate at the tip of a protected ref and
configure the `local-gpu-release` GitHub environment to allow only that protected
ref. Dispatch the workflow from that ref, paste its full 40-character SHA, and
select the trusted-source confirmation. The authorization job requires the actor
to equal the repository owner before the local runner is selected.

The workflow rejects reused or incorrectly provisioned execution by requiring the
ephemeral attestation, an exact clean checkout, the expected uv version, and no
persistent publication credential variables. It validates all ten models on CPU
and CUDA for Torch 2.6/CUDA 12.4 and Torch 2.11/CUDA 13.0;
runs every cohort against the single Torch 2.6 CPU golden-reference bundle;
runs the exact wheel and public notebook; and smokes both production images with
networking disabled and a read-only root filesystem. The retained evidence binds
the source SHA, dependency-lock hashes, manifest/governance hashes, GPU/runtime,
model summaries, golden-reference digests, candidate wheel, notebook, and local
Docker image IDs.

After the one-shot runner exits, remove its working directory and registration if
the runner service did not do so automatically. Never attach this label to a
persistent runner, and never add a pull-request trigger to the workflow.
