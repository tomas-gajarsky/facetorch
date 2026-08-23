# Model-card source

The ten Hugging Face cards are rendered from three reviewed inputs:

- `catalog.json` supplies the human-facing architecture, input, output,
  preprocessing, paper, and copyright descriptions;
- `facetorch/models/governance.json` supplies the approved checkpoint mapping,
  upstream revisions, rights, intended use, and limitations; and
- `facetorch/models/manifest.json` supplies the exact hosted artifact contract.

Render a review copy outside the repository:

```bash
python scripts/render_model_cards.py --output-root /tmp/facetorch-model-cards
```

Each model directory contains `README.md`, `LICENSE`, and
`THIRD_PARTY_NOTICES.md`. The renderer preserves MIT and Apache-2.0 as distinct
licenses. Apache cards reuse the repository's standard Apache-2.0 text; MIT
cards carry the copyright notice recorded for the checkpoint publisher.

Remote publication is a guarded owner operation. Verify every model repository's
current commit against the parent revision in the packaged manifest, commit the
three rendered files together, and then replace every packaged/configured source
revision with the resulting immutable commit. Run the Hub inventory and compare
the remote files byte-for-byte with the renderer before accepting the change.

Model-card publication does not publish or approve cohort validation metadata.
Current metadata is produced by the clean CPU/CUDA release matrix and published
through the separately reviewed model-cohort publication plan.
