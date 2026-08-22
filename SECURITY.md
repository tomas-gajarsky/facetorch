# Security policy

## Reporting a vulnerability

Please use [GitHub private vulnerability reporting](https://github.com/tomas-gajarsky/facetorch/security/advisories/new). The repository feature must be enabled before v1 publication. If the link does not show a private report form during the pre-release period, email [the maintainer](mailto:gajarsky.tomas@gmail.com?subject=Facetorch%20security%20report) with a minimal summary and coordinate a safer channel before sending sensitive details. Do not disclose a suspected vulnerability, private image, model input, access token, or exploit details in a public issue.

The founder is the initial security and model-provenance owner. The project aims to acknowledge a private report within five business days and provide an initial assessment within fourteen days. These are communication targets, not a promise that every fix will be available by a particular date. A backup security owner must be assigned before the v1 general-availability release.

## Supported versions

| Version | Support status |
| --- | --- |
| `release/v1.0.0` and published v1 release candidates | Pre-release security evaluation; not a stable-production claim |
| `0.6.x` | Current public line; critical and security fixes continue until the date announced at v1 general availability |
| `<0.6` | Unsupported |

At v1 general availability, the project will publish the exact end date for the approved six-month critical/security-only v0.6.x support window.

## Disclosure and release handling

Reports are triaged privately. Fixes are prepared against supported versions, tested without including sensitive payloads in evidence, and coordinated with reporters when practical. Released package, image, and model bytes are never overwritten in place. A correction uses a patch release or an explicit immutable revocation notice; a Python release is yanked only when it is unusable or dangerous.

## Privacy and network boundary

Facetorch has no telemetry by default. Image bytes, facial-analysis inputs, predictions, and derived payloads are not included in default logs, dependency reports, build provenance, or release evidence. Network access is limited to documented model retrieval and to remote-image input when the caller explicitly selects the restricted URL reader. Security reports should use synthetic or redacted reproductions whenever possible.

Dependency exceptions are exact-version, profile-scoped, time-bounded records under `security/`. Expired or mismatched exceptions fail the release dependency gate.
