# GA dependency advisory triage — September 5, 2026

Status: **retain all eight Torch lines; advisory approval remains pending**.
The owner confirmed broad Torch compatibility on September 5. This resolves the
support-range question. It does not approve new dependency exceptions or establish
that the application is remotely exploitable. The GA dependency gate still fails.

## Result and reproducibility

The corrected auditor completed all 17 frozen profiles with every active
runtime package accounted for and no skipped/missing records. There are **76 unresolved
profile entries representing 18 distinct advisories across twelve profiles**.
Root, Torch 2.11 CPU/CUDA, and Torch 2.13 CPU/CUDA have no unresolved entries under
the existing, time-limited policy. CPU/CUDA repetitions are counted separately
because each is an independently supported environment.

The [machine-readable snapshot](../security/v1-audit-findings-2026-09-05.json)
records all affected profiles, advisory aliases, database fix versions, coverage,
and audit/policy digests. Raw reports, SBOMs, exports, and original build hashes
are retained locally in `build/v1-ga-audit/`. Reproduce using the locked release
extras and `python scripts/audit_dependencies.py --output-dir build/dependency-audit`.
This snapshot ran on Linux x86-64 with Python 3.10.12 (841 active entries).
Repeating the audit in the frozen Python 3.12.12 environment covered 832 active
entries, also with zero coverage errors and the same 76 unresolved findings; see
`build/v1-ga-audit-py312/summary.json`. Environment markers explain the inventory
count difference. The nonzero exit is the intended gate result while findings
remain unresolved.

The old audit could omit official local builds such as `2.6.0+cpu`. The replacement
queries the corresponding upstream release only after checking the official
index, trusted wheel host, build flavor, and lock/export hashes. It matches
exceptions against the original CPU/CUDA build version and fails on missing,
skipped, duplicate, unexpected, or malformed dependency records. The normalized
requirements are an advisory projection; they must never be used for installation.

## Reachability and recommended treatment

1. **Treat model loading first.** `CVE-2026-24747` / `PYSEC-2026-2286` affects
   `torch.load(weights_only=True)` before Torch 2.10, including native `.pth`
   loading and alignment metadata paths used here. The
   [maintainer advisory](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p)
   confirms the crafted-checkpoint risk. `CVE-2026-4538` / `PYSEC-2026-139`
   concerns `torch.export.load`, the default model-loading path; the
   [PyPA record](https://github.com/pypa/advisory-database/blob/main/vulns/torch/PYSEC-2026-139.yaml)
   includes versions through 2.10 and does not provide a fixed-version event.
   Its [linked upstream change](https://github.com/pytorch/pytorch/pull/176791)
   addresses forwarding the safe-loading option. Do not infer a patched version
   merely from its absence in an audit response.
2. **Keep model trust explicit.** The shipped manifest pins artifacts and
   alignment metadata by hash, verified before loading by default. This reduces
   substitution risk for trusted packaged inputs; a hash does not establish that
   an application-selected model is safe. Native/custom models and shared caches
   expand exposure. `verify_on_use=False`, arbitrary custom downloads, and
   adversarial models cannot inherit the built-in trust argument. This is a
   code-path assessment, not an exploit demonstration or proof of immunity.
   [PyTorch's security policy](https://github.com/pytorch/pytorch/security/policy)
   treats model programs as executable code and recommends trusted sources.
3. **Review exact operator/compile cases separately.** The twelve 2025
   operator/compiler records below mostly describe numeric or failure behavior
   under particular operator combinations. A production-source search found no
   direct calls to the named specialized APIs, but `BaseModel` exposes optional
   `torch.compile` and model graphs can contain operators absent from Python
   source. Defaults use the pinned exported programs. A source search alone is
   insufficient to approve exceptions for every custom model or compiler path.
4. **Existing exceptions do not carry across runtime lines.** The ctc_loss,
   unpack_sequence, lstm_cell, and TorchScript annotation entries already have
   bounded approval on some older exact versions. The newly audited intermediate
   profiles fall outside those scopes. `facetorch/transforms.py` does call
   `torch.jit.script`; any proposed extension must assess the specific annotation
   trigger and custom-transform boundary instead of claiming JIT is unused.

Retain the existing Torch 2.6–2.13 range, including CUDA 12.4 / Torch 2.6,
under the [recorded support policy](torch-support-policy.md). The current pins
are already the latest patches in their minor lines as of September 5. The
[individual treatment proposal](v1-risk-treatment-proposal.md) now includes
inspection of all 20 pinned model programs, exact affected scopes, residual
risks, mitigations, and the existing expiry deadline. It remains unapproved.
Compatibility support through Torch 2.13 does not justify silently dropping the
older lines or approving their unresolved risks. Do not publish GA with this gate
red, or change the auditor to ignore all Torch findings.

Some advisory text and upstream classifications differ. For example,
PYSEC-2025-202's prose says “3.7.0” while its structured fix version is 2.7.0;
PYSEC-2025-208's upstream issue describes a compiler failure. The table preserves
structured audit results. These inconsistencies warrant individual review, not
a blanket claim of either exploitability or false positives.
[Dropout issue](https://github.com/pytorch/pytorch/issues/142853),
[compiler issue](https://github.com/pytorch/pytorch/issues/151523).

## Unresolved inventory

“Lines” lists affected frozen profiles grouped by Torch minor; each listed line
has CPU and its declared CUDA build. Fixes are database-reported upstream
versions, not validated replacements for the current Facetorch lock profile.

| Advisory | Trigger or affected area | Lines | Reported fix versions |
| --- | --- | --- | --- |
| [CVE-2025-2999](https://github.com/advisories/GHSA-vgrw-7cvw-pwgx) | unpack_sequence: memory corruption | 2.7, 2.8 | 2.9.1 |
| [CVE-2025-3001](https://github.com/advisories/GHSA-qfhq-4f3w-5fph) | lstm_cell: memory corruption | 2.7, 2.8, 2.9 | 2.10.0 |
| [PYSEC-2025-194](https://github.com/pypa/advisory-database/blob/main/vulns/torch/PYSEC-2025-194.yaml) | TorchScript class annotation: memory corruption | 2.7, 2.8, 2.9, 2.10, 2.12 | 2.13.0 |
| [PYSEC-2025-198](https://github.com/pypa/advisory-database/blob/main/vulns/torch/PYSEC-2025-198.yaml) | PairwiseDistance eager numerical mismatch | 2.6 | 2.7.0 |
| [PYSEC-2025-199](https://github.com/pypa/advisory-database/blob/main/vulns/torch/PYSEC-2025-199.yaml) | Fold under Inductor: assertion failure | 2.6 | 2.7.0 |
| [PYSEC-2025-200](https://github.com/pypa/advisory-database/blob/main/vulns/torch/PYSEC-2025-200.yaml) | Compiled FractionalMaxPool2d numerical mismatch | 2.6 | 2.7.0 |
| [PYSEC-2025-201](https://github.com/pypa/advisory-database/blob/main/vulns/torch/PYSEC-2025-201.yaml) | Out-of-range bitwise right shift output | 2.6 | 2.7.0 |
| [PYSEC-2025-202](https://github.com/pypa/advisory-database/blob/main/vulns/torch/PYSEC-2025-202.yaml) | Dropout random fallback decomposition mismatch | 2.6 | 2.7.0 |
| [PYSEC-2025-203](https://github.com/pypa/advisory-database/blob/main/vulns/torch/PYSEC-2025-203.yaml) | LU with sliced tensors: denial of service | 2.6, 2.7, 2.8 | 2.9.0 |
| [PYSEC-2025-204](https://github.com/pypa/advisory-database/blob/main/vulns/torch/PYSEC-2025-204.yaml) | Combined rot90/randn_like behavior | 2.6, 2.7, 2.8 | 2.9.0 |
| [PYSEC-2025-205](https://github.com/pypa/advisory-database/blob/main/vulns/torch/PYSEC-2025-205.yaml) | proxy_tensor syntax failure | 2.6 | 2.7.1 |
| [PYSEC-2025-206](https://github.com/pypa/advisory-database/blob/main/vulns/torch/PYSEC-2025-206.yaml) | nan_to_num followed by integer conversion: overflow | 2.6, 2.7, 2.8 | 2.9.0 |
| [PYSEC-2025-207](https://github.com/pypa/advisory-database/blob/main/vulns/torch/PYSEC-2025-207.yaml) | Compiled cummin: name error | 2.6 | 2.7.1 |
| [PYSEC-2025-208](https://github.com/pypa/advisory-database/blob/main/vulns/torch/PYSEC-2025-208.yaml) | Compiled Conv2d/hardshrink/view/mv combination | 2.6 | 2.7.1 |
| [PYSEC-2025-209](https://github.com/pypa/advisory-database/blob/main/vulns/torch/PYSEC-2025-209.yaml) | Compiled sparse/dense conversion failure | 2.6 | 2.7.1 |
| [PYSEC-2026-139](https://github.com/pypa/advisory-database/blob/main/vulns/torch/PYSEC-2026-139.yaml) | PT2 deserialization: code execution concern | 2.6, 2.7, 2.8, 2.9, 2.10 | None recorded |
| [PYSEC-2026-1970](https://github.com/pypa/advisory-database/blob/main/vulns/torch/PYSEC-2026-1970.yaml) | ctc_loss: denial of service | 2.7 | 2.8.0 |
| [PYSEC-2026-2286](https://github.com/pypa/advisory-database/blob/main/vulns/torch/PYSEC-2026-2286.yaml) | weights_only checkpoint unpickler: memory corruption/code execution | 2.6, 2.7, 2.8, 2.9 | 2.10.0 |

## Decision to record before another candidate

The support decision is settled: retain all eight lines and recommend a recent
qualified pair for new installations. Review the [risk treatment proposal](v1-risk-treatment-proposal.md),
especially the two loading advisories and the trusted-model boundary, before
approving any exact, expiring exception. The five currently passing profiles
are not a replacement support matrix. The full exact-candidate CPU/CUDA matrix,
archive verification, and agreed RC soak remain required.
