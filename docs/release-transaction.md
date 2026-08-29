# Transactional release pipeline

Facetorch prepares every release from one clean, protected, immutable commit. The same workflow supports a no-publication dry run and an explicitly approved publication run. A release candidate uses a tag such as `v1.0.0-rc.1`; a stable release uses `v1.0.0`.

## Required repository configuration

Before the first remote dry run, configure these repository variables from the completed model-publication receipt:

- `FACETORCH_MODEL_MANIFEST_REPO`
- `FACETORCH_MODEL_MANIFEST_REVISION`
- `FACETORCH_MODEL_MANIFEST_FILENAME`
- `FACETORCH_MODEL_MANIFEST_SHA256`

The revision must be a full Hugging Face commit and the digest must identify the exact manifest bytes. The packaged model, compatibility, and governance records must all be approved and agree with that remote manifest. An incomplete rights, provenance, device, or cohort record fails before artifact publication.

Configure protected GitHub environments named `local-gpu-release`, `github-release`, `dockerhub`, `pypi`, and `stable-release`. Require the approved release authority on every publication environment. Independent human approval is preferred; the bounded D20 exception permits owner `tomas-gajarsky` to self-approve `1.0.0rc1` through `1.0.0`, and that approval must not be represented as independent. Configure PyPI trusted publishing for the `pypi` environment; no PyPI password or API token is consumed. Store a scoped Docker Hub username and token as `DOCKER_USERNAME` and `DOCKER_PASSWORD` on the `dockerhub` and `stable-release` environments. The workflow passes the Docker token only through standard input.

The selected workflow ref must be protected and its exact head commit must equal `source_sha`, including for a dry run. The source tag must already exist as an annotated tag and resolve to that commit for publication. The tag, project metadata, and changelog version must agree. Dispatch values are validated before checkout or shell use. This equality also prevents a manual dispatch from sending untrusted code to the self-hosted runner.

## Build and evidence flow

The workflow builds the wheel, source distribution, CPU image, and GPU image once. The two exact saved image archives are then downloaded by the ephemeral local Linux x86-64 GPU runner. That runner must use the `facetorch-ephemeral-gpu` label, expose no persistent publication credential, execute the complete CPU/CUDA cohort matrix, and run the full default analyzer in both exact image IDs with networking disabled and a read-only root filesystem. The release plan cannot be created unless that evidence binds the protected source SHA, current locks/governance, every requested CPU/CUDA device, and both image digests.

Before assembly, a dedicated job audits the exact packaged and fetched model
manifests, generated legal documents, metadata identities, and all 30 downloaded
artifact bytes. Its successful report identifies both manifest digests and is
semantically bound into the release plan; a report from another model release or
an inventory-only audit is rejected.

The workflow saves two non-overlapping checksum contracts. Internal
`BUNDLE-SHA256SUMS` covers the complete retained transaction bundle, including
evidence, images, software bills of materials, distributions, and public
`SHA256SUMS`. Public `SHA256SUMS` contains only release-root basenames for the
wheel, source distribution, evidence archive, and release plan, so it works in a
directory containing the downloaded GitHub release assets. Receipts are verified
separately because they are created after the primary draft assets.

The workflow also saves dependency-audit output, pinned build inputs, local
GPU/container evidence, local provenance, the exact model manifest, and a
canonical release plan. Dry-run mode stops after these checks and cannot enter
any job with external write permissions. Because there is no server GPU, the
local ephemeral runner must be online for this gate; CPU-only validation is not
accepted as release evidence.

For publication, a GitHub release remains a draft while immutable versioned Docker images and PyPI files are reconciled. GitHub artifact attestations cover the package files and pushed image manifests. Each successful channel attaches a digest-bound receipt to the draft. The draft becomes public only when receipts for the model manifest, GitHub release, both images, and PyPI form one complete plan-bound set.

Release candidates are marked prerelease and can never move `latest`. A stable
run pulls each versioned image, compares its local image ID with the plan, and tags
`latest` from that verified ID rather than re-resolving a mutable tag. It then
rechecks each remote `latest` config digest before GitHub's latest-release marker
moves last. Conda remains an asynchronous handoff and is not mutated by this
pipeline.

## Retry and failure rules

After an external-channel failure, use GitHub Actions' **Re-run failed jobs** operation on the same workflow run. Successful preparation jobs and their exact image/package artifacts are retained; a missing channel is resumed. Do not start a fresh dispatch or choose **Re-run all jobs** as a publication retry, because that is a new candidate build and is deliberately rejected if its plan differs from the existing draft. An already published file or image is accepted only when its recorded digest matches the retained release plan. The workflow stops if the same version, tag, release asset, image tag, or PyPI filename contains different bytes.

This rule covers a PyPI success followed by Docker failure, one Docker image succeeding before the other, or draft creation followed by an external-channel failure. The GitHub release stays draft during any such partial state. Do not delete or replace successful immutable objects to retry.

Rollback never means rewriting released bytes. Stop before alias promotion when possible. After publication, issue a corrected patch release, yank a Python release only if it is unusable or dangerous, or publish an immutable model/image revocation notice that points users to a retained prior manifest or versioned image. The complete incident and release-candidate operations runbook is finalized in B11.

## Local verification

The transaction state machine and failure injection tests run with:

```bash
uv run --frozen --extra release python -m pytest -q tests/test_b10_release_transaction.py tests/test_release_workflow_contract.py
```

Workflow syntax and immutable action pins are checked with `actionlint` and the release-contract suite. A local pass is implementation evidence only; the first exact clean-commit remote dry run, ephemeral local-runner execution, protected environment configuration, and repository security settings remain release gates.
