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
Identical upstream Apache-2.0 files reuse the repository's byte-identical
license file; MIT files are retained under `upstream_licenses/`, including
upstream whitespace and notices exactly as received rather than reconstructed
or completed speculatively.

Remote publication is a guarded owner operation. Verify every model repository's
current commit against the parent revision in the packaged manifest, commit the
three rendered files together, and then replace every packaged/configured source
revision with the resulting immutable commit. The Hub audit downloads all three
files at that revision and rejects missing, empty, stale, or byte-different
content.

Model-card publication does not publish or approve cohort validation metadata.
Current metadata is produced by the clean CPU/CUDA release matrix and published
through the separately reviewed model-cohort publication plan.
