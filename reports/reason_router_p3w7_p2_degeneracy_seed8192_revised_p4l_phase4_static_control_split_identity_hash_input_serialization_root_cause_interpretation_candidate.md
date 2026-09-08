# Phase-IV Split-Identity Hash-Input Serialization Root-Cause Interpretation Candidate

## Status and authority

**Status:** `PASS_READY_FOR_INDEPENDENT_PHASE_IV_SPLIT_IDENTITY_HASH_SERIALIZATION_ROOT_CAUSE_INTERPRETATION_VERIFICATION`

This is a report-only root-cause interpretation candidate. It is governed by controller decision following validated post-failure diagnosis, with activated execution-validation authority `436d37499fd66a7d3b67756246c60223aa32dc48`, frozen remediation implementation `dd34cd00336d04d384767fd533c33253d2c9c6ac`, successor adoption authority `3ce3ccacfc326bb50ae7f65c157dd3331bd71bc6`, revised split design selection authority `b4fbb5666d796161f95ae23612ce2448c25063ee`, and frozen revised P4-L producer implementation `149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b`.

It authorizes no implementation, test, standalone-checker run, trainer run, training, evaluation, GPU/CUDA, Kaggle, commit, push, or scientific conclusion.

## 1. Authenticated opening state

At authoring start, the repository was on branch `p3w7-a1-a2-a3-factorial-execution-authority-n3-v2` at `436d37499fd66a7d3b67756246c60223aa32dc48`. The configured upstream was `origin/p3w7-a1-a2-a3-factorial-execution-authority-n3-v2` at the same SHA, and ahead/behind was `0/0`.

The required Git-native enumerations `git diff --name-only`, `git diff --cached --name-only`, and `git ls-files --others --exclude-standard` returned no tracked unstaged paths, no staged paths, and no nonignored untracked paths. Git emitted permission warnings while encountering existing ignored pytest-cache directories during untracked enumeration; no ignored cache was recursively inspected for this report.

## 2. Frozen execution evidence

The exact validated sequence was:

| Control | Command | Result |
|---|---|---|
| Focused pytest | `pytest -q tests/test_reason_router_p4x_prelaunch_static_control.py` | `83 passed, 1 skipped`; exit `0`; stderr empty |
| Standalone checker | `python scripts/validate_reason_router_p4x_prelaunch_static_control.py --repo-root . --expected-head dd34cd00336d04d384767fd533c33253d2c9c6ac` | exit `1`; stdout empty; stderr exactly `{"contract": "P4X_SPLIT_IDENTITY_MISMATCH", "status": "FAIL"}` |

The sole focused-test skip was authenticated as the source-defined symlink-fixture alternative. Therefore `CODE_CORRECTNESS_EVIDENCE = ESTABLISHED` for frozen implementation `dd34cd00336d04d384767fd533c33253d2c9c6ac`, while `PRELAUNCH_STATIC_CONTROL_EXECUTION_SUCCESS = NOT_ESTABLISHED`. Post-failure repository re-authentication passed and established that no mutation occurred.

## 3. Frozen input authentication

| Input | Git blob | SHA256 / identity | Result |
|---|---|---|---|
| Checker `scripts/validate_reason_router_p4x_prelaunch_static_control.py` | `c49725202aac50e65b8b3dd7a1e0cbe53484047e` | SHA256 `bcbf1818cfed1351077d2f3c2db809a70cc6953ef2d50adaa08d5b9e49a409b3`; 25070 bytes | authenticated |
| Dataset | `2b6829bf04a1333446aac6f7c603d9178b339f36` | Git-canonical physical SHA256 `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`; semantic SHA256 `3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b` | `DATASET_PHYSICAL_IDENTITY = MATCH`; `DATASET_SEMANTIC_IDENTITY = MATCH` |
| Provenance | `6c970033fae82286452f6d635b94f441d0f3d048` | physical SHA256 `170647d71d9c074c8bd7e87923b44d590b4159c693348cb335cd91a50ec777e8` | authenticated |

The dataset physical identity is the Git-canonical blob-byte identity. The working-tree Windows line-ending representation is not used to classify dataset drift. This failure is not dataset drift.

## 4. Frozen expected split audit

`SPLIT_IDENTITIES` has exactly these eleven current revised Seed8192 keys and values:

| Key | Frozen expected value |
|---|---:|
| `pair_count` | 300 |
| `train_pair_count` | 240 |
| `dev_pair_count` | 60 |
| `train_row_count` | 2880 |
| `dev_row_count` | 720 |
| `pair_universe_sha256` | `41f7a2cc533b9026a49d2b2587dd34894fadb908deab9f0a79133345569758f2` |
| `shuffled_pair_sha256` | `ef15a6c3dc0f45ccad0f4e4e203eab9ff5dbfe8d64dde96ae14df3811bbd2d55` |
| `train_pair_sha256` | `f6fffb94b6c33112bcfc8afb6da9f3aa76ae6e1327b8c38e69724fa4c2641049` |
| `dev_pair_sha256` | `30951a7c637b10a5693289be40911ec5bf32de6eca3efd37a81f3fa268cd25a4` |
| `ordered_train_row_sha256` | `478013207699462a9434ce8f44991ce75b33650593b9aa942fff0f2be659c2a8` |
| `ordered_dev_row_sha256` | `7870c83fe1f6e3a65311311ab05122736a007e6a92f4f04c28b2c72584ddfaa4` |

`PROVENANCE_SPLIT_IDENTITIES` is those exact eleven keys plus only `historical_seed174_dev_pair_sha256 = 259bfce57e85121d6c1adccd20f3ac070108ff6310cfff546a2edd054835899d`. The historical key remains provenance-only and is not a current `SPLIT_IDENTITIES` key.

## 5. Diagnostic actual audit and observed failure

An independent bounded pure calculation applied the frozen checker formulas before their equality assertion to authenticated Git dataset bytes. It produced:

| Key | Diagnostic actual value |
|---|---:|
| `pair_count` | 300 |
| `train_pair_count` | 240 |
| `dev_pair_count` | 60 |
| `train_row_count` | 2880 |
| `dev_row_count` | 720 |
| `pair_universe_sha256` | `f3b900dd8f7b00496221ba729f93599a81060724a12c1f19d8ff95bf260a3d0a` |
| `shuffled_pair_sha256` | `87335001f3c24f909c2d40e66c83a5bc8634d77a90cf439ec9972d004aac7af7` |
| `train_pair_sha256` | `540edeafa59f26932439e605dd8048b501e0efb0f761637e91556f6f89ea31fb` |
| `dev_pair_sha256` | `89c2f7af8080d02273877997346fbe1163fb08b5de35cdd18b249b1cbfd2e21e` |
| `ordered_train_row_sha256` | `64f06a3bfe073edfd03b6529e7ab7628387ed07b1e4a4dda9b43295089796c74` |
| `ordered_dev_row_sha256` | `741e210723c12051df78e941bc900880bded0785056e7753499bae16ed6a0072` |

There are exactly six mismatches: `pair_universe_sha256`, `shuffled_pair_sha256`, `train_pair_sha256`, `dev_pair_sha256`, `ordered_train_row_sha256`, and `ordered_dev_row_sha256`. All five cardinality fields match exactly.

## 6. Current checker hash semantics

The frozen checker helper `_identity_hash(values)` hashes UTF-8 encoding of `"\n".join(values)`. It appends no final LF. `recompute_split()` supplies it with: sorted pair IDs for the pair universe; the exact shuffled sequence; canonical sorted selected train and dev pair IDs; and, for ordered train/dev rows, lists containing only `str(row["id"])`.

Thus its pair-list inputs omit the authority-required final LF, and its ordered-row inputs neither use the required `id + TAB + pair_id + LF` record encoding nor include `pair_id`. The semantic split is re-encoded under a different hash-input serialization.

## 7. Original split authority and producer confirmation

Commit `b4fbb5666d796161f95ae23612ce2448c25063ee` has message `Activate P3-W7 revised split design selection authority`. It establishes the substantive split: sorted unique pair IDs; `random.Random(8192).shuffle`; dev ratio `0.2`; dev count `60`; first 60 shuffled pair IDs assigned to dev; remaining 240 assigned to train; and rows selected by pair membership.

Its authoritative pair-list serialization is UTF-8, exactly one pair ID per line, final LF present: canonical sorted IDs for universe/train/dev and exact shuffled sequence for shuffled identity. Its authoritative ordered-row serialization retains dataset order within each selected split and hashes UTF-8 records exactly `"{id}\t{pair_id}\n"`.

Commit `149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b` has message `Implement seed8192 revised P4-L producer enablement`. The producer implements the equivalent of `SHA256("".join(f"{pair_id}\n" for pair_id in pair_ids).encode("utf-8"))` and `SHA256("".join(f"{row['id']}\t{row['pair_id']}\n" for row in rows).encode("utf-8"))`.

Applying that historical-authority/producer serialization to the authenticated current dataset reproduces all six frozen expected hashes exactly. The frozen six hash constants are valid.

## 8. Root-cause interpretation

**Primary classification:** `SPLIT_IDENTITY_HASH_INPUT_SERIALIZATION_MISMATCH`.

**Secondary classification:** `SPLIT_ALGORITHM_AUTHORITY_MISMATCH`.

The secondary classification is strictly limited to the split-audit hash-input serialization layer. It does not assert that the revised Seed8192 split algorithm, seed, pair selection, or split membership is wrong.

The revised Seed8192 split membership is not shown to be wrong. Dataset physical and semantic identities match; all five cardinalities match; the random seed and selected split membership are not implicated; and current Python runtime is not required to explain the mismatch. The Phase-IV checker re-encodes the same semantic split with a different serialization and therefore falsely rejects it.

Rejected causes are: `DATASET_IDENTITY_DRIFT`, `DATASET_SEMANTIC_DRIFT`, `PAIR_UNIVERSE_MEMBERSHIP_DRIFT`, `SPLIT_CARDINALITY_DRIFT`, `SEED8192_SELECTION_DRIFT`, `TRAIN_DEV_MEMBERSHIP_DRIFT`, `PYTHON_RANDOM_RUNTIME_DRIFT`, `HISTORICAL_SEED174_IDENTITY_PROMOTION`, `FROZEN_SPLIT_IDENTITY_CONSTANT_ERROR`, and `PROVENANCE_SPLIT_IDENTITY_CONTENT_ERROR`.

## 9. Constraints on any future bounded correction

This report does not authorize a correction. A future implementation authority must preserve the frozen Seed8192 seed, dev ratio `0.2`, current pair membership, dataset, sidecar, provenance, all eleven current frozen `SPLIT_IDENTITIES` values, the historical Seed174 provenance-only key, exact split equality/fail-closed behavior, and `P4X_SPLIT_IDENTITY_MISMATCH` for a genuine current split mismatch.

It may correct only checker hash-input serialization to reproduce authority-defined identities: universe sorted IDs with one ID per line and final LF; exact shuffled sequence with one ID per line and final LF; canonical sorted train and dev IDs with one ID per line and final LF; and dataset-order selected row records encoded `{id}\t{pair_id}\n`.

It must not replace frozen constants with the current incorrect hashes, add tolerance/subset logic, bypass equality, or hard-code PASS.

## 10. Required future regression obligations

Any later implementation authority must require tests that establish all of the following:

1. Pair-list helper includes final LF, and an empty/no-final-LF alternative cannot silently produce the authoritative identity.
2. Pair universe, shuffled Seed8192 sequence, canonical sorted train pairs, and canonical sorted dev pairs each reproduce their frozen authoritative hash.
3. Ordered train and dev row identities use `id + TAB + pair_id + LF` and reproduce frozen hashes; row hashing cannot collapse to row ID only.
4. Genuine pair-membership mutation still fails `P4X_SPLIT_IDENTITY_MISMATCH`; genuine row membership/order mutation still fails.
5. Provenance exact-equality contract is unchanged, the historical Seed174 key remains provenance-only and outside `SPLIT_IDENTITIES`, and existing repository identity, lineage, symlink, cleanliness, cohort, aggregate, provenance, schema, and trainer-isolation tests remain passing.

## 11. Evidence-layer and scientific boundary

`CODE_CORRECTNESS_EVIDENCE = ESTABLISHED` for frozen `dd34cd00336d04d384767fd533c33253d2c9c6ac` through the focused pytest result. `PRELAUNCH_STATIC_CONTROL_EXECUTION_SUCCESS = NOT_ESTABLISHED`.

`ARTIFACT_PROVENANCE_VALIDITY =` the authenticated frozen artifacts remain valid as identities, but static control has not passed.

`SCIENTIFIC_CONCLUSION = NONE`. No A0/A1/A2/A3 execution is authorized. No trainer, training, evaluation, GPU, CUDA, or Kaggle activity is authorized. This diagnostic does not characterize the scientific URP hypothesis as failed.

## 12. Lifecycle and next phase

One fresh independent high-risk verifier is required before freeze because this interpretation governs provenance-validator semantics. Only after independent verification, exact byte/blob freeze, report-only stage, cm ship, dedicated root-cause interpretation freeze commit, push, and remote verification may the controller authorize the next report-only phase: `PHASE_IV_SPLIT_IDENTITY_HASH_SERIALIZATION_CORRECTION_IMPLEMENTATION_AUTHORITY_SPEC_AUTHORING`.

No code correction follows automatically.
