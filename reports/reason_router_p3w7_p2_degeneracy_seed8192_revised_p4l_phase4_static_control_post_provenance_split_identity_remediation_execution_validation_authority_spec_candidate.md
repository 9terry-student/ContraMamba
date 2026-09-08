# Post-Remediation Phase-IV Provenance Split-Identity Execution-Validation Authority Candidate

## 1. Authority, phase, verdict, and bounded scope

This is a REPORT-ONLY NEW EXECUTION-VALIDATION AUTHORITY CANDIDATE for the frozen Phase-IV provenance split-identity remediation implementation commit `dd34cd00336d04d384767fd533c33253d2c9c6ac` (the implementation freeze). Its successor adoption authority is `3ce3ccacfc326bb50ae7f65c157dd3331bd71bc6`; historical failed execution authority `d6356bdb66e06e5569209a89a8c7a25ae439f728` and prior implementation anchor `25569c0234086fb05d1120a7b0b5490aa751c182` are historical context only.

Verdict: `PASS_READY_FOR_FRESH_INDEPENDENT_HIGH_RISK_PHASE_IV_POST_REMEDIATION_EXECUTION_VALIDATION_AUTHORITY_VERIFICATION`.

This candidate does not yet authorize execution. IF AND ONLY IF this exact candidate is freshly independently high-risk verified, byte/blob frozen, activated by a dedicated report-only commit, pushed, and remotely verified, the activated authority authorizes the exact focused pytest and conditional checker commands defined below. That authority authorizes only their stated preflight, execution, and read-only post-execution authentication sequence.

No implementation change, training, evaluation, scientific execution, A0/A1/A2/A3 execution, trainer execution, CUDA/GPU, Kaggle, producer/materialization, model/checkpoint loading, staging, commit, push, fetch, or mutation of refs/config/cache/ACL/ignore state is authorized by authoring this candidate.

## 2. Required opening state and activation precondition

Before candidate authoring, the required opening state is:

| Property | Required value |
| --- | --- |
| Branch | `p3w7-a1-a2-a3-factorial-execution-authority-n3-v2` |
| HEAD | `dd34cd00336d04d384767fd533c33253d2c9c6ac` |
| Configured upstream | `origin/p3w7-a1-a2-a3-factorial-execution-authority-n3-v2` |
| Upstream tip | `dd34cd00336d04d384767fd533c33253d2c9c6ac` |
| Ahead/behind | `0/0` |
| Tracked modifications | `0` |
| Staged | `0` |
| Nonignored untracked | `0` |

No fetch is permitted. This candidate does not alter refs, config, cache, ACL, or ignore state. Before any future execution, activation creates a report-only descendant HEAD; therefore the future execution HEAD must equal its configured upstream tip and need not equal `dd34cd0`.

## 3. Implementation-freeze authentication

The implementation freeze must resolve exactly as commit `dd34cd00336d04d384767fd533c33253d2c9c6ac`, with sole parent `3ce3ccacfc326bb50ae7f65c157dd3331bd71bc6`. Its exact changed-fileset is only:

| Path | Additions | Deletions | Committed blob |
| --- | ---: | ---: | --- |
| `scripts/validate_reason_router_p4x_prelaunch_static_control.py` | 2 | 1 | `c49725202aac50e65b8b3dd7a1e0cbe53484047e` |
| `tests/test_reason_router_p4x_prelaunch_static_control.py` | 35 | 3 | `f9c581d29c4333e55e32dbe8828c1730706be45a` |
| Total | 37 | 4 | no third path |

The frozen raw pre-commit implementation-diff identity, prospectively recorded by governing adoption evidence, is `13543` bytes; SHA256 `f4cb1d5a57a43cc79a3cde1067c87704fb49e15302f6a5f8f4b663aa89335173`; LF / CR `84 / 0`. The old `c4c1a5a539935bbd803474a73572af2904b9b49f5cb70c326c3a071add10ef1b` identity remains non-authoritative.

## 4. Successor adoption lineage and non-retroactivity

Commit `3ce3ccacfc326bb50ae7f65c157dd3331bd71bc6` must resolve exactly. Its adoption report path is `reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_phase4_static_control_provenance_split_identity_remediation_authority_semantics_correction_existing_delta_adoption_spec_candidate.md`, with blob `cacc6d3b7806c60447565a2ae32e3e0822e12440`.

That report prospectively adopted exactly the checker/test bytes later frozen at `dd34cd0`. Authentication must establish `dd34cd0` parent equals `3ce3cca` and both `dd34cd0` committed blobs exactly equal those prospectively adopted blobs. The non-retroactivity correction remains preserved: adoption did not retroactively authorize creation of the prior uncommitted delta; it prospectively bound exact bytes before their report-only implementation-freeze commit.

## 5. Historical failed execution authority

`d6356bdb66e06e5569209a89a8c7a25ae439f728` is historical evidence only and MUST NOT be reused to execute new `dd34cd0` bytes. Its historical result was focused pytest `76 passed, 1 skipped`, exit `0`, followed by standalone checker exit `1` with `P4X_PROVENANCE_SPLIT_IDENTITY_MISMATCH`. That result motivated this frozen remediation; it is not evidence that the remediated implementation executes successfully.

## 6. Frozen implementation identities and semantic freeze

The current-HEAD checker must be `scripts/validate_reason_router_p4x_prelaunch_static_control.py`, blob `c49725202aac50e65b8b3dd7a1e0cbe53484047e`, raw SHA256 `bcbf1818cfed1351077d2f3c2db809a70cc6953ef2d50adaa08d5b9e49a409b3`, `25070` bytes, and LF / CR / CRLF `359 / 0 / 0`.

The current-HEAD focused test must be `tests/test_reason_router_p4x_prelaunch_static_control.py`, blob `f9c581d29c4333e55e32dbe8828c1730706be45a`, raw SHA256 `98fe97715a8b0f67cdb04b4ea8e2a0da5fbf3fe525d8ea5ef532e44ef59e61b5`, `33250` bytes, and LF / CR / CRLF `496 / 0 / 0`.

Static authentication must establish that `SPLIT_IDENTITIES` remains the exact revised Seed8192 11-key audit, does not contain `historical_seed174_dev_pair_sha256`, and that `recompute_split()` still requires exact `audit == SPLIT_IDENTITIES` with `P4X_SPLIT_IDENTITY_MISMATCH`. A distinct `PROVENANCE_SPLIT_IDENTITIES` must contain those exact eleven keys plus only `historical_seed174_dev_pair_sha256 = 259bfce57e85121d6c1adccd20f3ac070108ff6310cfff546a2edd054835899d`. `_validate_provenance()` must require exact equality to that provenance-specific contract, retaining `P4X_PROVENANCE_SPLIT_IDENTITY_MISMATCH`; subset or superset tolerance is prohibited.

## 7. Frozen provenance input

The frozen provenance input is `reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_integrity_sidecar_ff181f565cefa0a28280c084246862286daf1f2d_149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b/p3w7_seed8192_revised_p4l_effective_integrity_sidecar_provenance.json`, blob `6c970033fae82286452f6d635b94f441d0f3d048`, physical SHA256 `170647d71d9c074c8bd7e87923b44d590b4159c693348cb335cd91a50ec777e8`.

Its `split_identities` must contain exactly the eleven current Seed8192 identities plus only the historical binding `historical_seed174_dev_pair_sha256 = 259bfce57e85121d6c1adccd20f3ac070108ff6310cfff546a2edd054835899d`.

## 8. Mandatory pre-execution authentication order

After successful activation and before pytest, fail closed in this exact order:

1. Exact activated authority commit identity.
2. Exact authority parent equals `dd34cd00336d04d384767fd533c33253d2c9c6ac`.
3. Exact sole authority report path and blob.
4. Exact branch `p3w7-a1-a2-a3-factorial-execution-authority-n3-v2`.
5. Exact configured upstream `origin/p3w7-a1-a2-a3-factorial-execution-authority-n3-v2`.
6. Current HEAD equals configured upstream tip.
7. Ahead/behind equals `0/0`.
8. Tracked unstaged modifications equal `0`.
9. Staged modifications equal `0`.
10. Authoritative nonignored untracked enumeration equals `0`.
11. Checker blob at current HEAD equals `c49725202aac50e65b8b3dd7a1e0cbe53484047e`.
12. Focused-test blob at current HEAD equals `f9c581d29c4333e55e32dbe8828c1730706be45a`.
13. `dd34cd0` resolves as an exact commit and is an ancestor of current HEAD.
14. `3ce3cca` resolves exactly and is an ancestor of both `dd34cd0` and current HEAD.
15. Frozen provenance blob and physical SHA256 authenticate.
16. `git diff --check` exits `0`.

Any failure prevents pytest. No preflight step may mutate repository state.

## 9. Cache and untracked observability

The already-authenticated local observability fact is that, before implementation freeze, `git ls-files --others --exclude-standard` completed with exit `0`, empty stdout, and empty stderr; `git status --porcelain=v1 --untracked-files=all` completed with exit `0`, empty stderr, and only then-intended implementation modifications. After implementation freeze, the repository was authenticated clean.

For future execution, Git commands themselves determine pre/post cleanliness. Any Git untracked-enumeration error or stderr that indicates inability to inspect the actual local repository fails closed unless separately authenticated by the activated authority. Do not delete caches or change ACLs to manufacture PASS. Known LF-to-CRLF advisory warnings from diff commands are not dirt when byte/blob identities and command exit codes remain exact.

## 10. Exact bounded execution sequence

Only after all Section 8 preflight conditions pass, authorize exactly:

```text
pytest -q tests/test_reason_router_p4x_prelaunch_static_control.py
```

No other pytest target, trainer test suite, or repository-wide suite is authorized. Capture exit code, stdout, and stderr separately. Exit code must be `0`; no failure, error, or xpass is permitted. A skip is allowed only when it is the frozen source-defined symlink-fixture alternative: either `os.symlink` API is unavailable, with exact reason `os.symlink API is unavailable for the symlink fixture`, or Windows symlink creation is blocked specifically by `WinError 1314`, with exact reason `Windows privilege or policy prevents symlink fixture creation (WinError 1314)`. These are alternatives within the same symlink-fixture test path. Where neither condition occurs, zero skips are valid. At most that source-defined symlink-fixture skip may be accepted; no other skip is silently accepted. Arbitrary platform skips, newly added skip markers, plugin- or environment-induced unexplained skips, and any skip not attributable to one of those exact frozen source-defined conditions are not successful validation.

Focused pytest exit `0` establishes only `CODE_CORRECTNESS_EVIDENCE = ESTABLISHED` for this bounded static-control implementation; it establishes no scientific conclusion. If pytest exits nonzero, do not run the checker; capture the exact failure and perform read-only post-failure attestation where possible.

Only after focused pytest exit `0`, authorize exactly:

```text
python scripts/validate_reason_router_p4x_prelaunch_static_control.py --repo-root . --expected-head dd34cd00336d04d384767fd533c33253d2c9c6ac
```

Capture checker exit code, stdout, and stderr separately. Exit `0` is required for checker success. `--expected-head dd34cd00336d04d384767fd533c33253d2c9c6ac` is the immutable implementation-anchor argument, not a requirement that activated-authority HEAD equal `dd34cd0`. The activated report-only authority is a descendant; current HEAD must instead equal configured upstream tip and retain `dd34cd0` as an ancestor.

Static source inspection fixes the checker success-output schema without execution: stdout is one sorted-key JSON object with `status: "PASS"`, `split` (the authenticated split audit), `cohorts` with `train` and `dev` derived cohort objects, and `execution_record_opened: false`; stderr is empty on success. Its failure schema is a sorted-key JSON object on stderr with `status: "FAIL"` and named `contract`, and exit `1`. No broader criterion is invented.

## 11. Post-execution re-authentication and evidence classification

After a pytest/checker attempt, perform read-only authentication of branch, HEAD, configured upstream, upstream tip, ahead/behind, unstaged state, staged state, nonignored untracked enumeration, authority-report blob, checker blob, test blob, and implementation-freeze ancestry. Require no persistent tracked, staged, or nonignored-untracked mutation attributable to the validation sequence. Ignored/transient cache state is acceptable only if Git successfully establishes the required nonignored-untracked result; do not clean it merely to claim PASS.

Classify evidence separately:

| Classification | Permitted basis |
| --- | --- |
| `CODE_CORRECTNESS_EVIDENCE` | Focused pytest exit `0` only establishes it. |
| `PRELAUNCH_STATIC_CONTROL_EXECUTION_SUCCESS` | Checker exit `0`, after pytest exit `0`, establishes it. |
| `ARTIFACT_PROVENANCE_VALIDITY` | Frozen artifact and provenance authentication contributes to it. |
| `SCIENTIFIC_CONCLUSION` | Not established by this authority or any successful validation command. |

If the checker fails, capture its exact named contract/output and perform separate read-only post-failure attestation. No recovery implementation, `recovery4`, ad hoc remediation, scientific execution, training, or evaluation is automatically authorized. Any defect requires a new authority decision.

## 12. Strict prohibitions and lifecycle

Even after activation, this authority does not authorize trainer execution, training, evaluation, A0, A1, A2, A3, CUDA, GPU, Kaggle, model loading, checkpoint loading, producer/materialization, dataset regeneration, sidecar regeneration, provenance regeneration, split changes, label changes, loss changes, gradient changes, EMA changes, calibration changes, or implementation edits.

Required lifecycle: (1) author this candidate; (2) fresh independent high-risk authority verification; (3) candidate byte/blob freeze; (4) explicit report-only stage; (5) cm ship; (6) dedicated execution-validation authority activation commit; (7) push; (8) remote verification; (9) only then preflight; (10) only then focused pytest; (11) only if pytest exit `0`, standalone checker; (12) post-execution re-authentication; (13) interpret only validated code/static-control evidence. No shortcut is authorized.

## 13. Candidate integrity and final state

This candidate must be UTF-8 without BOM, LF-only, final-LF terminated, with zero trailing-whitespace lines. Its final raw SHA256, byte count, line-ending counts, predicted Git blob, and manual Git blob SHA-1 are deliberately not self-embedded and must be measured externally after authoring.

After authoring, required state is HEAD `dd34cd00336d04d384767fd533c33253d2c9c6ac`; tracked modifications `0`; staged `0`; untracked exactly this one candidate; and no implementation modification. Pytest, checker, and trainer must not run. Staging, commit, and push must not occur.

Exact next authorized action: `FRESH_INDEPENDENT_HIGH_RISK_PHASE_IV_POST_REMEDIATION_EXECUTION_VALIDATION_AUTHORITY_VERIFICATION`.
