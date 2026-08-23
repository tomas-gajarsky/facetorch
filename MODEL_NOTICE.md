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

On 2026-08-23, the repository owner approved the following policy for the ten
manifested models: when the original authors published a checkpoint from the
same permissively licensed repository and supplied no separate checkpoint
terms, that checkpoint uses the repository license. This is an artifact-specific
rights determination backed by the recorded source mapping; it is not a general
instruction to infer weight rights from any code license. MIT and Apache-2.0 are
preserved as received and are not converted or treated as interchangeable.

The approved per-model rights and evidence are recorded in governance and in
the generated Hugging Face cards. A model remains eligible only while its exact
checkpoint mapping, redistribution, attribution, owner approval, license text,
and immutable hosted bytes all remain bound. This approval does not license an
upstream training dataset. The separate compatibility matrix and release
manifest remain candidate records until a clean release commit passes the full
CPU/CUDA release matrix and is published through the release transaction.

Face-analysis outputs are probabilistic model signals. They are not proof of
identity, authenticity, emotion, intent, health, or protected characteristics,
and must not be the sole basis for consequential decisions about a person. Users
are responsible for consent, privacy, applicable law, security, demographic and
domain-shift evaluation, and compliance with each artifact's recorded terms.
