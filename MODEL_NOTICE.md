# Third-party model notice

Facetorch's Apache-2.0 software license applies to the project source code. It
does not, by itself, grant rights to third-party model weights, checkpoints,
training data, or other artifacts that Facetorch can download. The wheel and
source distribution contain model metadata, not model weights.

The authoritative, release-bound records are:

- `facetorch/models/manifest.json` for immutable repository revisions, artifact
  names, hashes, export provenance, and referenced license material;
- `facetorch/models/governance.json` for per-model checkpoint mapping, weight
  rights, redistribution, attribution, intended use, limitations, and owner
  approval; and
- `facetorch/models/compatibility.json` for the tested runtime and device
  matrix.

Do not infer a weight license from an upstream code-repository license or a
model-card label. A model is eligible for a Facetorch release only when all of
its rights and provenance fields are verified, redistribution and attribution
are approved, and the release manifest binds the exact reviewed bytes. The
current v1 working-tree records are deliberately marked incomplete and therefore
fail the publication gate until that review is finished.

Face-analysis outputs are probabilistic model signals. They are not proof of
identity, authenticity, emotion, intent, health, or protected characteristics,
and must not be the sole basis for consequential decisions about a person. Users
are responsible for consent, privacy, applicable law, security, demographic and
domain-shift evaluation, and compliance with each artifact's recorded terms.
