# Post-Failure Phase-IV Provenance Split-Identity Root-Cause / Remediation Authority Candidate

## 1. Authority, phase, and decision

This is a REPORT-ONLY ROOT-CAUSE AND REMEDIATION-AUTHORITY CANDIDATE. It is governed by the post-remediation execution-validation authority `d6356bdb66e06e5569209a89a8c7a25ae439f728`, implementation anchor `25569c0234086fb05d1120a7b0b5490aa751c182`, and remediation-authority lineage `f45341e60c3fa634ffbd2805ae14b3d540441afc`.

Its sole purpose is to freeze the root cause of `P4X_PROVENANCE_SPLIT_IDENTITY_MISMATCH` and prospectively bound a later checker/test remediation. It does not authorize implementation, execution, pytest, checker reruns, training, evaluation, dataset regeneration, checkpoint mutation, staging, committing, or pushing.

Verdict: `PASS_READY_FOR_FRESH_INDEPENDENT_HIGH_RISK_PHASE_IV_PROVENANCE_SPLIT_IDENTITY_ROOT_CAUSE_REMEDIATION_AUTHORITY_VERIFICATION`.

## 2. Opening repository state

Before this candidate was created, the required opening state was authenticated:

- Branch: `p3w7-a1-a2-a3-factorial-execution-authority-n3-v2`.
- HEAD: `d6356bdb66e06e5569209a89a8c7a25ae439f728`.
- Configured upstream: `origin/p3w7-a1-a2-a3-factorial-execution-authority-n3-v2`.
- Upstream tip: `d6356bdb66e06e5569209a89a8c7a25ae439f728`.
- Ahead/behind: `0/0`.
- Tracked modifications: `0`; staged: `0`; untracked: `0`.

No fetch, Git-ref/config change, cache/ACL/ignore change, or external execution was performed.

## 3. Failed execution authority authentication

`d6356bdb66e06e5569209a89a8c7a25ae439f728` has sole parent `25569c0234086fb05d1120a7b0b5490aa751c182`. Its sole changed path is `reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_phase4_static_control_post_remediation_execution_validation_authority_spec_candidate.md`, whose blob is `efbbc6e69c07e6f71937a29eb254d653121def24`.

The implementation anchor `25569c0234086fb05d1120a7b0b5490aa751c182` is an ancestor of `d6356bdb66e06e5569209a89a8c7a25ae439f728`.

At that HEAD, the frozen executable inputs are:

- `scripts/validate_reason_router_p4x_prelaunch_static_control.py`: blob `8b21843b0f78356262fdc843c0aacee9ab419b75`.
- `tests/test_reason_router_p4x_prelaunch_static_control.py`: blob `8d56db1a00963cb188aac9c0927da5c572dc20b1`.

## 4. Observed execution result and post-failure attestation

The authenticated historical execution record is exactly:

- `git diff --check`: exit `0`.
- Focused command: `pytest -q tests/test_reason_router_p4x_prelaunch_static_control.py`.
- Focused result: `76 passed, 1 skipped`; exit `0`; stderr empty.

The skip is the authorized Windows symlink-fixture portability skip. It is not production-symlink PASS. Thus `CODE_CORRECTNESS_EVIDENCE=ESTABLISHED`.

The exact checker command was:

```text
python scripts/validate_reason_router_p4x_prelaunch_static_control.py --repo-root . --expected-head 25569c0234086fb05d1120a7b0b5490aa751c182
```

It exited `1`, produced empty stdout, and produced exactly this stderr JSON:

```json
{"contract": "P4X_PROVENANCE_SPLIT_IDENTITY_MISMATCH", "status": "FAIL"}
```

Therefore `PRELAUNCH_STATIC_CONTROL_EXECUTION_SUCCESS=NOT_ESTABLISHED`, `Scientific execution=FALSE`, `Training/evaluation=FALSE`, and `A0/A1/A2/A3 execution=FALSE`. The original execution block stopped before ordinary step-12 post-execution re-authentication.

The separately performed read-only post-failure attestation recorded HEAD still `d6356bdb66e06e5569209a89a8c7a25ae439f728`; exact branch/upstream retained; ahead/behind `0/0`; unstaged exit `0`; staged exit `0`; untracked command exit `0`; untracked count `0`; authority-report blob still `efbbc6e69c07e6f71937a29eb254d653121def24`; checker blob still `8b21843b0f78356262fdc843c0aacee9ab419b75`; test blob still `8d56db1a00963cb188aac9c0927da5c572dc20b1`; and implementation-anchor ancestry exit `0`. `POST_FAILURE_ATTESTATION=PASS`. No persistent repository mutation resulted from validation.

## 5. Frozen checker split-identity authentication

The checker was read exactly at `d6356bdb66e06e5569209a89a8c7a25ae439f728` and authenticated as blob `8b21843b0f78356262fdc843c0aacee9ab419b75`. Its current revised `SPLIT_IDENTITIES` exact key/value contract is:

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

`recompute_split()` requires `audit == SPLIT_IDENTITIES` and names failure `P4X_SPLIT_IDENTITY_MISMATCH`. This valid current revised split-audit contract must remain unchanged.

In `_validate_provenance()`, the frozen defective line is:

```python
_require(value.get("split_identities") == SPLIT_IDENTITIES, "P4X_PROVENANCE_SPLIT_IDENTITY_MISMATCH")
```

## 6. Frozen provenance authentication and exact structural comparison

The frozen provenance path is `reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_integrity_sidecar_ff181f565cefa0a28280c084246862286daf1f2d_149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b/p3w7_seed8192_revised_p4l_effective_integrity_sidecar_provenance.json`.

At current HEAD its Git blob is `6c970033fae82286452f6d635b94f441d0f3d048` and its physical SHA256 is `170647d71d9c074c8bd7e87923b44d590b4159c693348cb335cd91a50ec777e8`.

Its `split_identities` inventory contains all eleven keys in Section 5 with exactly their Section-5 values, plus one and only one additional key:

| Provenance-only historical key | Value |
| --- | --- |
| `historical_seed174_dev_pair_sha256` | `259bfce57e85121d6c1adccd20f3ac070108ff6310cfff546a2edd054835899d` |

There are no missing shared keys, altered shared values, or other extra keys. The historical identity is provenance-only.

## 7. Root cause and deterministic-failure proof

The production defect is exactly `PRODUCTION_FROZEN_PROVENANCE_SPLIT_IDENTITY_SUPERSET_COMPATIBILITY_DEFECT`.

The checker incorrectly uses the same exact-dict object contract for two semantically distinct structures: (A) the current revised Seed8192 recomputed split audit, and (B) frozen provenance split-identity metadata. Frozen provenance intentionally preserves an additional historical seed174 dev-pair identity, while the recomputed current Seed8192 audit does not and must not contain it. Consequently `value["split_identities"] == SPLIT_IDENTITIES` is structurally false for the authenticated frozen provenance although every current Seed8192 identity agrees. The standalone failure is deterministic under the frozen inputs.

This is not dataset drift, provenance corruption, Seed8192 recomputation failure, sidecar corruption, execution mutation, or scientific failure.

## 8. Critical non-remedy and preserved contracts

It is prohibited to add `historical_seed174_dev_pair_sha256` to `SPLIT_IDENTITIES`: that object is also the expected `recompute_split()` audit, which correctly contains only current split evidence. Such addition would conflate current evidence with historical provenance metadata and could make the valid recompute contract fail.

Also prohibited are deleting the historical provenance key; rewriting or regenerating provenance merely to satisfy equality; arbitrary subset acceptance; generally ignoring unknown provenance keys; and changing Seed8192 counts or hashes.

No change is authorized to the dataset, Seed8192 split selection, `SPLIT_IDENTITIES` values, sidecar, provenance bytes, Phase-II evidence, execution record, trainer, trainer rebind test, cohorts, aggregate identities, historical seed174 rejection, branch/upstream or anchor semantics, cleanliness semantics, Git-blob authentication, symlink defense, upstream-selector remediation, `P4X_UNTRACKED_INPUT` remediation, or scientific semantics.

## 9. Focused-test root cause

The focused test file was read exactly at `d6356bdb66e06e5569209a89a8c7a25ae439f728` and authenticated as blob `8d56db1a00963cb188aac9c0927da5c572dc20b1`. Its provenance-test inventory includes `test_provenance_schema_lineage_authority_and_flags_are_exact`, `test_provenance_flags_require_literal_booleans`, and `test_frozen_provenance_does_not_require_external_phase2_lineage_fields`.

Each positive/synthetic provenance fixture in those inspected tests supplies `"split_identities": p4x.SPLIT_IDENTITIES`. No positive test passes the actual frozen provenance JSON object through `p4x._validate_provenance(...)`. Therefore there is no real-artifact positive coverage.

The test defect is exactly `TEST_FIXTURE_FROZEN_PROVENANCE_REAL_ARTIFACT_COVERAGE_GAP`: the suite tests an internally self-consistent synthetic object whose split identities equal the checker constant, so it cannot expose incompatibility with the actual frozen provenance schema. This explains `76 passed, 1 skipped` while the standalone checker deterministically fails on the authenticated artifact.

## 10. Bounded prospective remediation authority

Only after independent verification, candidate freeze, activation, and a dedicated remediation authority may a future implementation modify exactly:

- `scripts/validate_reason_router_p4x_prelaunch_static_control.py`
- `tests/test_reason_router_p4x_prelaunch_static_control.py`

No third implementation file is authorized. The production change is confined to provenance split-identity validation. It must keep `SPLIT_IDENTITIES` and `recompute_split(): audit == SPLIT_IDENTITIES` unchanged. It must separately enforce an exact provenance key set of all `SPLIT_IDENTITIES` keys plus only `historical_seed174_dev_pair_sha256`; every shared value must equal `SPLIT_IDENTITIES`; and the historical value must exactly equal `259bfce57e85121d6c1adccd20f3ac070108ff6310cfff546a2edd054835899d`.

Missing, altered, malformed, or unauthorized additional provenance split-identity fields must continue to fail `P4X_PROVENANCE_SPLIT_IDENTITY_MISMATCH`. A separate immutable `PROVENANCE_SPLIT_IDENTITIES` constant or equivalent explicit validation is acceptable; formatting is not prescribed.

Future focused tests must prove: the real frozen object contains the eleven current identities plus the exact historical key; that object is accepted; removal or alteration of any current identity fails the named contract; removal or alteration of the historical key fails it; an unknown extra key fails it; and the recompute current contract is unchanged. Positive real-artifact compatibility coverage is preferred and is required alongside any retained synthetic fixture.

## 11. Authority status, scientific boundary, and lifecycle

`d6356bdb66e06e5569209a89a8c7a25ae439f728` is valid historical provenance for the observed failed validation attempt, but must not be reused to execute changed checker/test bytes. It does not authorize remediation after checker failure. A new execution-validation authority is required after a future remediation implementation freeze.

Phase-II and scientific scope remain untouched: no scientific execution, training, evaluation, A0/A1/A2/A3 execution, model/checkpoint loading, CUDA/GPU, Kaggle, or materialization is authorized.

Required lifecycle:

1. Author this report-only candidate.
2. Conduct fresh independent high-risk verification.
3. Freeze candidate bytes/blob.
4. Explicitly stage one report.
5. `cm ship`.
6. Create a dedicated remediation-authority activation commit.
7. Push and perform remote verification.
8. Implement the bounded checker/test remediation.
9. Conduct fresh independent high-risk implementation verification.
10. Freeze implementation bytes/diff and commit/push the implementation freeze.
11. Author a new execution-validation authority candidate.
12. Independently verify, freeze, and activate that authority.
13. Only then rerun focused pytest/checker.

No reuse of the `d6356bdb66e06e5569209a89a8c7a25ae439f728` execution authority is allowed.

## 12. Candidate integrity and next action

This candidate must be UTF-8 without BOM, LF-only with a final LF, and have zero trailing-whitespace lines. Its final raw SHA256, byte count, line-ending counts, predicted Git blob, and manual Git blob SHA-1 are deliberately not embedded and must be measured only after authoring.

Exact next authorized action: `FRESH_INDEPENDENT_HIGH_RISK_PHASE_IV_PROVENANCE_SPLIT_IDENTITY_ROOT_CAUSE_REMEDIATION_AUTHORITY_VERIFICATION`.
