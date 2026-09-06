# Security policy

## Reporting a vulnerability

Please use [GitHub private vulnerability reporting](https://github.com/tomas-gajarsky/facetorch/security/advisories/new). Private reporting is enabled for this repository. If the link does not show a private report form during the pre-release period, email [the maintainer](mailto:gajarsky.tomas@gmail.com?subject=Facetorch%20security%20report) with a minimal summary and coordinate a safer channel before sending sensitive details. Do not disclose a suspected vulnerability, private image, model input, access token, or exploit details in a public issue.

The founder is the initial security and model-provenance owner. The project aims to acknowledge a private report within five business days and provide an initial assessment within fourteen days. These are communication targets, not a promise that every fix will be available by a particular date. A backup owner is preferred. Under the approved, bounded D20 exception, `tomas-gajarsky` may self-approve and operate releases through the final `1.0.0` post-publication checks without a backup. This accepts owner-availability risk and does not waive any automated or exact-candidate gate; see the [release runbook](docs/release-runbook.md).

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

The complete audit of the RC3 lock profiles on 2026-09-05 found 18 unresolved
advisories after correcting upstream Torch build-tag coverage. On September 6,
the owner approved the [scoped risk treatment](docs/v1-risk-treatment-proposal.md).
The active policy adds exactly 76 profile/version records, expiring November 20;
the earlier nine approvals retain their dates and scopes. Fresh audits pass all
17 profiles with no coverage gaps or unresolved findings under that policy.

Affected older runtimes are retained for trusted model programs and metadata.
Digest verification authenticates selected bytes; it does not make adversarial
model uploads safe. Keep built-in verification enabled and qualify custom models
and compiler options separately. These are accepted dependency risks, not
upstream fixes or approval to publish GA.
