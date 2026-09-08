# Phase-IV Post-Hash-Serialization-Correction Execution-Validation Authority Specification Candidate

## Status and authority boundary

**Candidate status:** `PASS_READY_FOR_INDEPENDENT_PHASE_IV_POST_HASH_SERIALIZATION_CORRECTION_EXECUTION_VALIDATION_AUTHORITY_VERIFICATION`

**Exact next action:** `FRESH_INDEPENDENT_HIGH_RISK_PHASE_IV_POST_HASH_SERIALIZATION_CORRECTION_EXECUTION_VALIDATION_AUTHORITY_VERIFICATION`

This is a report-only prospective authority-specification candidate, governed by the current controller instruction. It authorizes no action by itself: no pytest, standalone checker, A-series execution, trainer, training, evaluation, GPU/CUDA, Kaggle, model or checkpoint loading, dataset/sidecar/provenance regeneration, staging, commit, push, or scientific conclusion.

Only a separately independently verified, byte/blob-frozen, report-only staged, shipped, dedicated activated, pushed, and remotely verified execution-validation authority may authorize the bounded sequence in this report. Candidate authoring does not supersede, reinterpret, retry, or remediate any historical evidence.

## 1. Frozen authority lineage and implementation anchor

| Role | Immutable identity |
| --- | --- |
| Activated implementation authority | `2050ac82952dc84dcb8fd2507a51be5946ab11e1` |
| Frozen root-cause interpretation | `182ff454ab44a134a33d6b7a15f16356afb2ed8e` |
| Corrected implementation anchor | `8e637bdd439d62d429e1c019281efe398ee8c368` |
| Historical failed implementation | `dd34cd00336d04d384767fd533c33253d2c9c6ac` |
| Historical failed execution authority | `436d37499fd66a7d3b67756246c60223aa32dc48` |

The corrected implementation anchor has exactly one parent:

```text
8e637bdd439d62d429e1c019281efe398ee8c368^
= 2050ac82952dc84dcb8fd2507a51be5946ab11e1
```

The later activation commit is intentionally unknown at candidate authoring. `IMPLEMENTATION_ANCHOR` remains exactly `8e637bdd439d62d429e1c019281efe398ee8c368`; a later activation HEAD is required to contain this anchor as an ancestor, but the checker argument must never be rebound to that activation HEAD.

## 2. Authenticated corrected implementation scope

The corrected anchor changes exactly these two paths, and no others:

```text
scripts/validate_reason_router_p4x_prelaunch_static_control.py
tests/test_reason_router_p4x_prelaunch_static_control.py
```

Its committed delta is exactly two files changed, 33 insertions, and 2 deletions. The exact committed objects are:

| Role | Path | Git blob | Raw SHA256 | Bytes |
| --- | --- | --- | --- | ---: |
| Checker | `scripts/validate_reason_router_p4x_prelaunch_static_control.py` | `90b96a329cd6e92d3c5c79b76d02dc2d31233574` | `2098b05a76ef38721566bf68b75fd708a182370492813c1700810d2f1c9ea04b` | 25130 |
| Focused test | `tests/test_reason_router_p4x_prelaunch_static_control.py` | `e310dcdd72f9dc39ca281626ea20dd6bd19a7132` | `5876c8b90ce6a89539fabb30690cb8d01fec22bbff7f9aac5047562af251c7cb` | 35305 |

The correction is restricted to the hash-input serialization layer. `_identity_hash` hashes precisely the UTF-8 bytes of:

```python
"".join(f"{value}\n" for value in values).encode("utf-8")
```

For selected train/dev rows in original dataset order, each supplied identity value is `id + TAB + pair_id`, so the resulting serialized record is `id + TAB + pair_id + LF`. The correction makes no split-membership change and does not authorize one.

## 3. Frozen implementation-phase evidence and present classification

The following implementation-phase evidence is established for the corrected anchor:

```text
IMPLEMENTATION_DELTA_CORRECTNESS = ESTABLISHED
FOCUSED_TEST_CORRECTNESS = ESTABLISHED
targeted = 3 passed, 1 warning
recovered focused = 86 passed, 1 skipped in 19.09s; exit 0; stderr empty
```

The recovered focused result used the exact focused pytest command with only process-local `TEMP`, `TMP`, and `TMPDIR` redirected to a fresh writable directory outside the repository after a Windows default-temp-root `PermissionError`. It made no ACL modification, pytest-option change, or test weakening.

This historical implementation-phase result is not a substitute for the fresh execution-validation focused run required below. At activation start:

```text
PRELAUNCH_STATIC_CONTROL_EXECUTION_SUCCESS = NOT_ESTABLISHED
SCIENTIFIC_CONCLUSION = NONE
```

## 4. Frozen dataset, split, and provenance contract

The future authority must preserve this exact `SPLIT_IDENTITIES` object:

| Key | Value |
| --- | --- |
| `pair_count` | `300` |
| `train_pair_count` | `240` |
| `dev_pair_count` | `60` |
| `train_row_count` | `2880` |
| `dev_row_count` | `720` |
| `pair_universe_sha256` | `41f7a2cc533b9026a49d2b2587dd34894fadb908deab9f0a79133345569758f2` |
| `shuffled_pair_sha256` | `ef15a6c3dc0f45ccad0f4e4e203eab9ff5dbfe8d64dde96ae14df3811bbd2d55` |
| `train_pair_sha256` | `f6fffb94b6c33112bcfc8afb6da9f3aa76ae6e1327b8c38e69724fa4c2641049` |
| `dev_pair_sha256` | `30951a7c637b10a5693289be40911ec5bf32de6eca3efd37a81f3fa268cd25a4` |
| `ordered_train_row_sha256` | `478013207699462a9434ce8f44991ce75b33650593b9aa942fff0f2be659c2a8` |
| `ordered_dev_row_sha256` | `7870c83fe1f6e3a65311311ab05122736a007e6a92f4f04c28b2c72584ddfaa4` |

`PROVENANCE_SPLIT_IDENTITIES` remains exactly those eleven keys plus only:

```text
historical_seed174_dev_pair_sha256 = 259bfce57e85121d6c1adccd20f3ac070108ff6310cfff546a2edd054835899d
```

No mutation or regeneration of dataset, sidecar, provenance, constants, split, membership, or authority artifacts is authorized. The future authority must authenticate their existing identities before and after execution.

## 5. Required pre-execution authentication

Before any pytest or checker execution, the activated future authority must use read-only Git-native enumeration and fail closed unless all of the following are exact:

```text
worktree = C:\p3w7-a0-n3-validated-evidence-analysis
branch = p3w7-a1-a2-a3-factorial-execution-authority-n3-v2
configured upstream = origin/p3w7-a1-a2-a3-factorial-execution-authority-n3-v2
current HEAD = resolved upstream tip
ahead / behind = 0 / 0
tracked unstaged = 0
staged = 0
nonignored untracked = 0
```

It must also establish: (1) anchor `8e637bdd439d62d429e1c019281efe398ee8c368` exists and is an ancestor of the activated execution-authority HEAD; (2) the current-HEAD checker/test blobs exactly equal `90b96a329cd6e92d3c5c79b76d02dc2d31233574` and `e310dcdd72f9dc39ca281626ea20dd6bd19a7132`; (3) the frozen dataset/provenance identities and all split identities are unchanged; and (4) the execution-authority report itself is present at current HEAD and equals its independently frozen candidate blob. Authentication may not mutate repository state.

## 6. Mandatory fresh focused pytest

Only after every pre-execution control passes, the future authority must run exactly:

```text
pytest -q tests/test_reason_router_p4x_prelaunch_static_control.py
```

No selector, repository-wide pytest, or pytest-option change is authorized. It must capture the exact process-local temp root, command, exit code, stdout, and stderr. Focused success requires exit code `0`.

Because the host’s default Windows pytest temp root is protected, `TEMP`, `TMP`, and `TMPDIR` may be set only for the pytest process to one fresh writable directory outside the repository. `PYTHONDONTWRITEBYTECODE=1` may also be process-local. This environmental isolation must not modify ACLs, delete or alter an existing pytest temp root, create pytest temp/cache directories inside the repository, change pytest CLI options, or weaken/skip tests.

The sole previously authenticated source-defined symlink-fixture skip is allowed only when attributable to unavailable `os.symlink` or Windows `WinError 1314`. If the fresh focused pytest fails for any reason, the authority must stop without running the checker, set execution success to `NOT_ESTABLISHED`, and seek a separate report-only diagnosis/interpretation authority. It authorizes no retry or remediation.

## 7. Conditional standalone checker

Only if the fresh focused pytest passes, the later activated authority may run exactly this unwrapped command:

```text
python scripts/validate_reason_router_p4x_prelaunch_static_control.py --repo-root . --expected-head 8e637bdd439d62d429e1c019281efe398ee8c368
```

`--expected-head` is the immutable corrected implementation anchor, not the future execution-authority HEAD. No alternate implementation SHA, omitted argument, source change, or wrapper that suppresses or rewrites exit code/stdout/stderr is authorized. Capture exact command, exit code, stdout, and stderr.

Checker success requires all of the following: exit `0`; empty stderr; stdout containing exactly one valid JSON object; `status` exactly `PASS`; the returned `split` object exactly equal to frozen `SPLIT_IDENTITIES`; and required `train`/`dev` cohorts present according to the frozen checker contract. A pytest pass is never checker success.

If the checker exits nonzero, preserve exact stdout and stderr, record its exact contract code, stop, and perform no remediation, rerun, implementation change, or artifact regeneration. Any interpretation or subsequent attempt requires a separate report-only diagnosis/interpretation authority.

## 8. Post-execution re-authentication

After the focused pytest and, if reached, checker command, the activated authority must perform read-only re-authentication. It must require: activated execution-authority HEAD unchanged; upstream tip unchanged; ahead/behind `0/0`; tracked unstaged `0`; staged `0`; nonignored untracked `0`; checker/test blobs unchanged; implementation anchor unchanged; and dataset/provenance identities unchanged. If external temp isolation was used, that external directory is not a repository artifact and may not be imported or staged. No repository cleanup mutation is authorized.

## 9. Required evidence-layer separation

The activated authority must state independent outcomes for:

```text
IMPLEMENTATION_DELTA_CORRECTNESS
FOCUSED_TEST_CORRECTNESS
PRELAUNCH_STATIC_CONTROL_EXECUTION_SUCCESS
ARTIFACT_PROVENANCE_VALIDITY
SCIENTIFIC_CONCLUSION
```

At execution start, `IMPLEMENTATION_DELTA_CORRECTNESS = ESTABLISHED`; the historical implementation-phase `FOCUSED_TEST_CORRECTNESS = ESTABLISHED` remains historical; the fresh focused-test result is reported separately. Only checker exit `0` with valid PASS JSON may set `PRELAUNCH_STATIC_CONTROL_EXECUTION_SUCCESS = ESTABLISHED`. Checker success alone is not a scientific conclusion. `SCIENTIFIC_CONCLUSION = NONE` until a later scientific execution/evaluation authority evaluates validated scientific evidence.

## 10. Non-retroactivity and strict prohibition

Historical failure for implementation `dd34cd00336d04d384767fd533c33253d2c9c6ac` under execution authority `436d37499fd66a7d3b67756246c60223aa32dc48`, with contract `P4X_SPLIT_IDENTITY_MISMATCH`, remains valid historical evidence. It is neither changed nor retroactively made successful. The new sequence validates only `8e637bdd439d62d429e1c019281efe398ee8c368`.

The authority is static-control execution validation only. It authorizes none of: A0, A1, A2, A3, trainer execution, training, evaluation, GPU, CUDA, Kaggle, model/checkpoint loading, dataset regeneration, sidecar regeneration, provenance regeneration, split reselection, label change, loss change, gradient change, EMA change, or calibration change.

## 11. Required lifecycle and independent high-risk verification

The mandatory lifecycle is:

1. Author this report-only candidate.
2. Perform fresh independent high-risk authority verification.
3. Freeze the exact candidate bytes/blob.
4. Perform the explicit report-only stage.
5. `cm ship`.
6. Create a dedicated execution-validation authority activation commit.
7. Push.
8. Verify remotely.
9. Perform pre-execution authentication.
10. Run the fresh exact focused pytest.
11. Only if it passes, run the exact standalone checker.
12. Perform post-execution re-authentication.
13. Classify evidence layers.
14. Let the controller decide the next authorized research action.

There is no shortcut. Before activation, a fresh independent high-risk verifier must establish at minimum: exact implementation freeze; both committed blobs; exact authority lineage; immutable `--expected-head` binding to `8e637bdd439d62d429e1c019281efe398ee8c368`; mandatory fresh focused pytest before checker; checker conditional on pytest PASS; exact checker command; no failure rerun/remediation authority; post-execution cleanliness; evidence-layer separation; and absence of A-series/training authority.

## 12. Candidate disposition

No execution is authorized by this authoring. Any mismatch in opening state, implementation authentication, frozen blob, candidate-report authentication, precondition, exact command binding, expected-head anchor, or prohibition boundary blocks before execution and requires a separate report-only authority decision.
