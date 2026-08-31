# P3-W7-A0 Seed180 Provenance Recovery Prediction Row Count Reconciliation Authority Specification Candidate

Authority/version:

`P3W7_A0_SEED180_PROVENANCE_RECOVERY_PREDICTION_ROW_COUNT_RECONCILIATION_AUTHORITY_V1_CANDIDATE`

## Status

CANDIDATE ONLY.

This is a narrow authority-reconciliation specification candidate for
ContraMamba P3-W7-A0 seed180 provenance recovery. It reconciles the frozen
historical prediction-export row-count requirement with the actual immutable
stage174a_v1 evidence available in the surviving seed180 artifacts.

This candidate does not authorize implementation by existence alone. It does
not authorize training, evaluation, Kaggle execution, recovery collection,
recovery audit-import, seed180 rerun/retry/resume, checkpoint regeneration,
artifact mutation, A1/A2/A3, seed181/seed182 work, result promotion,
scientific interpretation, commit, push, standard `cm` behavior changes,
cleanup, reset, stash, checkout, rename, deletion, staging, or mutation of
unrelated files.

## Authority Chain

Authority precedence for this candidate:

1. current user instruction and current repository state;
2. frozen seed180 recovery execution authority:
   `233ed0be080e1d30dd47de2e66136475ec2ede76`;
3. frozen recovery tooling implementation authority:
   `cdd71ea4f556392eab594ebb5df8258355610e01`;
4. frozen external-cm reconciliation authority:
   `1b6516d16596d1169ff2fa4fd8d8c8f8adb80450`;
5. frozen recovery implementation:
   `15387eb6fd2af9b1171b8b988a64cfcf4417c1cd`;
6. original A0 execution authority:
   `2737c3c6116ae3766b469801f990e2c45ba9a55e`;
7. current repository descendant:
   `31e6d7882586e312f783cb2fd69718eb1ee7e452`;
8. repository-wide `AGENTS.md` controls.

## Repository Preconditions

This candidate may be created only when:

- `git rev-parse HEAD` is exactly
  `31e6d7882586e312f783cb2fd69718eb1ee7e452`;
- tracked worktree is clean;
- index is clean;
- `15387eb6fd2af9b1171b8b988a64cfcf4417c1cd` is an ancestor of current
  `HEAD`;
- `233ed0be080e1d30dd47de2e66136475ec2ede76` is an ancestor of current
  `HEAD`;
- `cdd71ea4f556392eab594ebb5df8258355610e01` is an ancestor of current
  `HEAD`;
- `1b6516d16596d1169ff2fa4fd8d8c8f8adb80450` is an ancestor of current
  `HEAD`;
- `2737c3c6116ae3766b469801f990e2c45ba9a55e` is an ancestor of current
  `HEAD`;
- the tracked diff
  `15387eb6fd2af9b1171b8b988a64cfcf4417c1cd..31e6d7882586e312f783cb2fd69718eb1ee7e452`
  contains only unrelated O0b work and does not modify any seed180 recovery
  authority, recovery script/test, trainer, A0 authority, or original execution
  paths.

Protected unrelated untracked files are not blockers and must not be touched.

## Descendant Independence

The tracked diff from
`15387eb6fd2af9b1171b8b988a64cfcf4417c1cd` to current `HEAD`
`31e6d7882586e312f783cb2fd69718eb1ee7e452` adds only:

`reports/longterm_o0b_matched_control_dataset_validator_implementation_authority_spec_candidate.md`

This is unrelated O0b authority work. It does not modify:

- seed180 recovery authority paths;
- `scripts/reason_router_p3w7_a0_seed180_provenance_recovery.py`;
- `tests/test_reason_router_p3w7_a0_seed180_provenance_recovery.py`;
- trainer paths;
- original A0 execution authority paths;
- original seed180 execution paths.

This independence assessment does not inspect, rely on, or reinterpret the
scientific content of unrelated O0b work.

## Observed Execution Blocker

The first real recovery collect attempt under frozen implementation
`15387eb6fd2af9b1171b8b988a64cfcf4417c1cd` failed closed before package
creation with:

```text
PROVENANCE_RECOVERY_BLOCKER:
prediction_export_row_count expected 720 got None
```

This failure:

- is a provenance recovery blocker only;
- is not training failure;
- is not model failure;
- is not artifact invalidity;
- does not change seed180 attempt disposition `CONSUMED`;
- does not change execution success `OBSERVED`;
- does not establish a scientific conclusion;
- does not authorize retry under unchanged implementation.

The frozen implementation currently searches:

`prediction_export_jsonl_audit.prediction_export_row_count`

then:

`finalization.prediction_export_row_count`

The immutable historical `run_provenance.json` contains neither path.

## Immutable Historical Evidence

Exact immutable historical provenance identity:

| File | Size | SHA256 |
| --- | ---: | --- |
| `run_provenance.json` | 68429 | `4fa8d362000e0a085368306328c1562b09a2ffea7bc7932cf5edeaa037ea9c3b` |

Read-only live Kaggle inspection of that exact byte object established no
historical field at either:

- `prediction_export_jsonl_audit.prediction_export_row_count`;
- `finalization.prediction_export_row_count`.

Actual provenance evidence includes at least:

- `data_provenance.auxiliary_activity.row_counts.dev_rows = 720`;
- `resolved_runtime_config.active_bridge_auxiliary_modes_and_row_counts.row_counts.dev_rows = 720`;
- `split_seed_contract.clean_main_dev_rows = 720`;
- `compatible_positive_margin.sidecar_contract.actual_dev_rows = 720`;
- `compatible_positive_margin.run_activity.single.sidecar_contract.actual_dev_rows = 720`;
- `resolved_runtime_config.compatible_positive_margin.sidecar_validation.actual_dev_rows = 720`.

These fields must not be reinterpreted individually as historical
prediction-export metadata. They corroborate the clean dev set cardinality.

Exact frozen prediction artifacts:

| File | Size | SHA256 |
| --- | ---: | --- |
| `clean_dev_predictions.json` | 4838225 | `92c5f1da0d7fe8c3b51ade4cf323bf660d509f500143d15f475060065c254aa2` |
| `training_report_predictions.jsonl` | 3934123 | `e4fc95992dcd9dc3ea35da7527d948fdd9e419f8764aea6ef87b8b14fe6ac9ef` |

Observed immutable `clean_dev_predictions.json` structure:

- root object;
- `predictions` list length is exactly `720`.

Observed immutable `training_report_predictions.jsonl` structure:

- nonempty JSONL record count is exactly `720`.

These two prediction artifacts are already frozen immutable artifact anchors.

## Root Cause

The frozen recovery authority required recovery to establish:

`prediction export row count = 720`

but phrased the requirement as if `run_provenance.json` itself must
independently contain or confirm that exact export-row-count field.

The actual immutable stage174a_v1 historical provenance does not contain that
field.

Therefore the frozen tooling implementation encoded a provenance path that
cannot succeed against the truthful historical byte object.

This is an authority/schema reconciliation issue, not permission to fabricate,
backfill, or mutate provenance.

## Narrow Supersession

This candidate supersedes only the requirement that:

> `run_provenance.json` independently confirms prediction export row count
> `720`

when interpreted as requiring an explicit prediction-export-row-count field.

No other requirement from the frozen seed180 recovery execution authority,
frozen tooling implementation authority, external-cm reconciliation authority,
frozen recovery implementation, or original A0 execution authority is
superseded.

This candidate does not supersede the substantive cardinality requirement:

`prediction export row count MUST still be exactly 720`.

## Replacement Validation Rule

Recovery must establish prediction export cardinality `720` by independent,
cross-artifact evidence from the immutable historical artifacts.

Required validation after a later separately authorized implementation repair:

1. `clean_dev_predictions.json`

   - exact frozen size/SHA must already pass;
   - strict JSON parse;
   - root must be the actual historical structure;
   - `predictions` must be a list;
   - `len(predictions) == 720`.

2. `training_report_predictions.jsonl`

   - exact frozen size/SHA must already pass;
   - exactly `720` nonempty JSONL records;
   - each nonempty record must be valid JSON;
   - duplicate JSON keys should fail closed if feasible using existing strict
     JSON semantics.

3. Historical `run_provenance.json`

   - exact frozen size/SHA must pass;
   - require the actual stage174a_v1 dev-set cardinality fields that are
     genuinely present and authoritative;
   - at minimum require
     `data_provenance.auxiliary_activity.row_counts.dev_rows == 720`;
   - at minimum require
     `resolved_runtime_config.active_bridge_auxiliary_modes_and_row_counts.row_counts.dev_rows == 720`;
   - at minimum require
     `split_seed_contract.clean_main_dev_rows == 720`;
   - require consistency among required authoritative copies;
   - additional existing authoritative dev-row copies may also be checked if
     verified from the actual writer/schema.

All three evidence groups must agree on `720`.

No single fallback/default is sufficient.

## Non-Fabrication Requirements

Future work must explicitly prohibit:

- adding `prediction_export_row_count` to historical `run_provenance.json`;
- rewriting `run_provenance.json`;
- generating replacement provenance;
- treating `dev_rows` as if it were literally a historical
  export-row-count field;
- inferring historical wrapper metadata;
- modifying prediction artifacts;
- regenerating predictions;
- rerunning training;
- changing frozen hashes.

The reconciliation changes only how the already-frozen cardinality fact is
validated.

## Preserved Recovery State

Preserve unchanged:

- `seed = 180`;
- `attempt = CONSUMED`;
- `execution success = OBSERVED`;
- `standard cm wrapper provenance = INCOMPLETE`;
- `scientific conclusion = NOT_ESTABLISHED`.

Preserve all five artifact identities:

| File | Size | SHA256 |
| --- | ---: | --- |
| `training_report.json` | 306114 | `71423b654722f055d20876bb9e7d4029c8e06e57d878359b1452232516915508` |
| `clean_dev_predictions.json` | 4838225 | `92c5f1da0d7fe8c3b51ade4cf323bf660d509f500143d15f475060065c254aa2` |
| `training_report_predictions.jsonl` | 3934123 | `e4fc95992dcd9dc3ea35da7527d948fdd9e419f8764aea6ef87b8b14fe6ac9ef` |
| `selected_checkpoint.pt` | 518269815 | `dbf663f32c7780ab6ebe949dbe79cc205576cd3e6b7686591e1623fb039282da` |
| `run_provenance.json` | 68429 | `4fa8d362000e0a085368306328c1562b09a2ffea7bc7932cf5edeaa037ea9c3b` |

Preserve:

- exact A0 semantics;
- dataset/P4-L identities;
- command validation;
- selected checkpoint identity;
- TOCTOU object-identity checks;
- ZIP security;
- no-overwrite;
- separate recovery schema;
- no standard cm collect/import impersonation.

## Implementation Boundary

This candidate does not authorize code changes by existence.

A later bounded remediation may occur only after:

1. exact candidate independent verifier PASS;
2. exact candidate immutable Git freeze;
3. explicit controller transition.

Future remediation scope must be limited to:

- replacing the impossible explicit provenance export-row-count lookup;
- adding fail-closed cross-artifact `720`-row validation described above;
- updating synthetic tests only as necessary.

Expected code delta after future authorization remains exactly:

- `scripts/reason_router_p3w7_a0_seed180_provenance_recovery.py`;
- `tests/test_reason_router_p3w7_a0_seed180_provenance_recovery.py`.

No third implementation file is authorized.

## Execution Boundary

This candidate does not authorize another collect attempt.

The failed collect attempt must not be treated as successful recovery
execution.

No retry is authorized until:

- reconciliation candidate verified/frozen;
- bounded remediation implemented;
- remediation independently verified/frozen;
- post-freeze gates PASS;
- new exact implementation commit and exact collect command/hash are
  materialized;
- explicit recovery-execution transition occurs.

The previous command hashes tied to
`15387eb6fd2af9b1171b8b988a64cfcf4417c1cd` are obsolete for future execution
after any implementation change.

## Explicit Non-Authorizations

This candidate does not authorize:

- editing existing files;
- staging, committing, or pushing;
- training;
- evaluation;
- Kaggle execution;
- recovery collection;
- recovery audit-import;
- modifying `cm.ps1`;
- seed180 rerun, retry, resume, or replacement;
- checkpoint regeneration;
- artifact mutation;
- standard `cm` collect/import impersonation;
- scientific conclusion or result promotion;
- cleanup, reset, stash, checkout, rename, deletion, or mutation of unrelated
  files.

## Candidate Transition Rule

The exact recommended next action after creating this candidate is independent
verification of this exact candidate.

Implementation remediation must wait until this candidate receives independent
verifier PASS, is frozen in an immutable commit, and the controller explicitly
transitions to bounded implementation remediation under the narrow scope stated
above.
