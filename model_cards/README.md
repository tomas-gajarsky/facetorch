# Model-card source

The ten Hugging Face cards are rendered from three reviewed inputs:

- `catalog.json` supplies the human-facing architecture, input, output,
  preprocessing, and paper descriptions;
- `facetorch/models/governance.json` supplies the approved checkpoint mapping,
  upstream revisions, exact license-file paths and SHA-256 digests, rights,
  intended use, and limitations; and
- `facetorch/models/manifest.json` supplies the exact hosted artifact contract.

Render a review copy outside the repository:

```bash
python scripts/render_model_cards.py --output-root /tmp/facetorch-model-cards
```

Each model directory contains `README.md`, `LICENSE`, and
`THIRD_PARTY_NOTICES.md`. The renderer preserves MIT and Apache-2.0 as distinct
licenses. It reads the verbatim license bytes verified at each pinned upstream
revision and refuses to render when their recorded SHA-256 does not match.
Identical upstream Apache-2.0 files reuse a dedicated byte-identical file under
`upstream_licenses/`; MIT files are retained there separately, including upstream
whitespace and notices exactly as received rather than reconstructed or completed
speculatively. Every Apache source also records either a pinned NOTICE file or an
explicit, revision-verified absence.

The offline release blockers validate source/revision URL binding and vendored
digests. Re-fetch every pinned upstream license and recheck Apache NOTICE state
before publication with:

```bash
FACETORCH_RUN_UPSTREAM_NETWORK=1 pytest -q -m upstream_network tests/test_model_cards.py
```

Remote publication is a guarded owner operation. Verify every model repository's
current commit against the parent revision in the packaged manifest. For a cohort
update, publish model artifacts first, update their filenames, sizes, and digests
in `manifest.json`, and only then render the cards. Commit the three rendered files
together, replace every packaged/configured source revision and `license_ref` with
the resulting immutable commit, and run the Hub audit. The audit downloads all
three files at that revision and rejects missing, empty, stale, or byte-different
content.

Model-card publication does not publish or approve cohort validation metadata.
Current metadata is produced by the clean CPU/CUDA release matrix and published
through the separately reviewed model-cohort publication plan.
