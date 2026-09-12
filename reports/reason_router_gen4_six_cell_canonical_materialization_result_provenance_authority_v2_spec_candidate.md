# ContraMamba Gen4 Six-Cell Canonical Materialization Result / Provenance Authority V2 - Candidate

## 1. Status

- Status: CANDIDATE
- Phase: GEN4_SIX_CELL_CANONICAL_MATERIALIZATION_RESULT_PROVENANCE_AUTHORITY_V2
- V2 execution authority: 8593e1478e9ff366d814bb158429340143fae997
- Corrected implementation commit: fbc780ce12cbfbcf4e20a3cb9d1f099553045fc0
- CLI failure-recovery authority: b728dcf95d47e15d2f2a8b50a3872e5dd0c9e0bc
- Failed V1 execution authority: 3c3e0dfba7455a09e36d4fbeabaa6785b79582f2
- Generator semantic authority: 91c3dcd7abadcd6cf0d6d2f1c299f3d6fa6e28ea
- Contrast specification authority: 0a0da5354782e542520fb5bba146ab1a599d17ef
- Training/Evaluation allowed: NO
- Model execution allowed: NO
- Tokenizer execution allowed: NO
- Kaggle allowed: NO

This authority records and binds the validated V2 canonical structural artifact.

It does not establish any scientific outcome effect.

## 2. Canonical artifact

Canonical artifact path:

reports/reason_router_gen4_six_cell_masked_slot_substitution_materialization_fbc780ce12cbfbcf4e20a3cb9d1f099553045fc0/gen4_six_cell_masked_slot_substitution.jsonl

Canonical SHA256:

b9c54604863ed15c237fa17c7890a20f3f5ec062a7429638b39e8b468be050a7

Canonical byte count:

1465573

Therefore:

V2_CANONICAL_ARTIFACT_SHA256 = b9c54604863ed15c237fa17c7890a20f3f5ec062a7429638b39e8b468be050a7

V2_CANONICAL_ARTIFACT_BYTES = 1465573

## 3. Execution identity

V2 execution started at:

8593e1478e9ff366d814bb158429340143fae997

Corrected implementation SHA256:

b95400fd670280eeecd79bab818ad70a9f1aba9b74116201dd52bfaba1301f59

Corrected test SHA256:

1ac54025cd51062ed8b4242349c6ef19723b2fe5306969d9f215aa96abc08746

Frozen generator blob:

baee23a9f71333125f4a8735c2c92d20cab7eb4f

Therefore:

V2_EXECUTION_START_HEAD = 8593e1478e9ff366d814bb158429340143fae997

PROTECTED_FILE_IMMUTABILITY = PASS

## 4. Canonical execution count

The V2 authority authorized one canonical materialization.

Observed:

V2_CANONICAL_MATERIALIZATION_EXECUTIONS = 1

Execution result:

PASS

The prior failed V1 execution authority was not reused.

FAILED_V1_EXECUTION_AUTHORITY_REUSED = NO

The failed V1 output path remains absent.

FAILED_V1_OUTPUT_EXISTS = NO

## 5. Canonical cardinality

Validated canonical row count:

ROW_COUNT = 1800

Validated unique row IDs:

UNIQUE_ROW_IDS = 1800

Validated unique source pair IDs:

UNIQUE_SOURCE_PAIR_IDS = 300

Validated cell counts:

C0_SHAM_COUNT = 300

C1_TITLE_COUNT = 300

C2_NAME_COUNT = 300

C3_ROLE_COUNT = 300

C4_PREDICATE_COUNT = 300

C5_TITLE_NAME_COUNT = 300

Every source pair forms one complete six-cell block.

COMPLETE_BLOCK_VALIDATION = PASS

## 6. Schema validation

Each row contains exactly thirteen fields in the frozen order:

1. schema_version
2. generator_authority_commit
3. generator_source_blob
4. contrast_specification_commit
5. mechanism_id
6. source_pair_id
7. row_id
8. contrast_cell_id
9. axis_mask
10. intended_changed_axes
11. generator_source_fields
12. claim
13. evidence

SCHEMA_ORDER_VALIDATION = PASS

## 7. Structural provenance

Every row was validated against:

schema_version =
GEN4_SIX_CELL_MASKED_SLOT_SUBSTITUTION_V1

generator_authority_commit =
91c3dcd7abadcd6cf0d6d2f1c299f3d6fa6e28ea

generator_source_blob =
baee23a9f71333125f4a8735c2c92d20cab7eb4f

contrast_specification_commit =
0a0da5354782e542520fb5bba146ab1a599d17ef

mechanism_id =
masked_slot_substitution_v1

PROVENANCE_VALIDATION = PASS

## 8. Outcome-field exclusion

No unauthorized outcome-bearing field was observed.

FORBIDDEN_FIELD_VALIDATION = PASS

The artifact remains structural only.

No label, model prediction, evaluator output, training result, or evaluation
result is part of the canonical artifact.

## 9. Serialization validity

Validated properties:

- UTF-8 without BOM;
- LF line endings;
- trailing LF present;
- no blank JSONL rows;
- deterministic field order.

SERIALIZATION_VALIDATION = PASS

## 10. Deterministic rerun

Exactly one authorized deterministic rerun was executed.

V2_DETERMINISTIC_RERUN_EXECUTIONS = 1

Rerun SHA256:

b9c54604863ed15c237fa17c7890a20f3f5ec062a7429638b39e8b468be050a7

Canonical SHA256:

b9c54604863ed15c237fa17c7890a20f3f5ec062a7429638b39e8b468be050a7

The artifacts were byte-identical.

DETERMINISTIC_BYTE_IDENTITY = PASS

The temporary rerun artifact was deleted.

TEMP_RERUN_DELETED = YES

## 11. Repository mutation boundary

The canonical JSONL is the only persistent delta created by V2 execution.

ONLY_PERSISTENT_EXECUTION_DELTA = V2_CANONICAL_JSONL

The canonical artifact is currently untracked.

ARTIFACT_STAGED = NO

No artifact commit or push occurred.

COMMIT_PUSH = NOT_PERFORMED

No existing tracked source or test file changed.

## 12. Execution exclusions

The V2 materialization performed no:

- model execution;
- tokenizer execution;
- training;
- evaluation;
- loss computation;
- gradient computation;
- representation extraction;
- statistical testing;
- Kaggle execution.

Therefore:

MODEL_EXECUTION = NOT_PERFORMED

TOKENIZER_EXECUTION = NOT_PERFORMED

TRAINING_EVALUATION = NOT_PERFORMED

KAGGLE_EXECUTION = NOT_PERFORMED

## 13. Scientific interpretation boundary

This artifact establishes structural materialization and provenance validity only.

It does not establish:

- a title effect;
- a name effect;
- a role effect;
- a predicate effect;
- a title-name interaction effect;
- causal relevance;
- mechanistic relevance;
- predictive utility;
- performance improvement.

Therefore:

SCIENTIFIC_OUTCOME_CONCLUSION = NOT_ESTABLISHED

## 14. Artifact identity freeze rule

The only artifact eligible for later freeze is exactly:

reports/reason_router_gen4_six_cell_masked_slot_substitution_materialization_fbc780ce12cbfbcf4e20a3cb9d1f099553045fc0/gen4_six_cell_masked_slot_substitution.jsonl

with SHA256 exactly:

b9c54604863ed15c237fa17c7890a20f3f5ec062a7429638b39e8b468be050a7

and byte count exactly:

1465573

Any byte drift invalidates freeze eligibility.

## 15. Current artifact-freeze authority state

While this document remains only a candidate:

ARTIFACT_FREEZE = NOT_AUTHORIZED_UNTIL_THIS_AUTHORITY_IS_FROZEN

Do not stage the canonical JSONL yet.

Do not commit the canonical JSONL yet.

Do not push the canonical JSONL yet.

## 16. Authority after freeze

If this result/provenance authority is manually reviewed, committed, pushed,
and frozen without byte drift:

ARTIFACT_FREEZE = AUTHORIZED_FOR_EXACT_CANONICAL_JSONL

ARTIFACT_COMMIT_SCOPE = EXACT_ONE_FILE

The later artifact freeze may stage only the canonical JSONL whose identity is
specified above.

No other file becomes authorized for that artifact commit.

## 17. Artifact freeze validation requirement

Before staging the canonical JSONL after this authority is frozen, revalidate:

- authority commit identity;
- artifact SHA256;
- artifact byte count;
- artifact untracked status;
- row count 1800;
- unique row IDs 1800;
- unique source pairs 300;
- six cell counts each 300;
- protected source identities unchanged;
- no staged unrelated file.

## 18. Scientific execution remains closed

Freezing the structural artifact does not authorize:

- model inference;
- tokenizer execution;
- training;
- evaluation;
- statistical outcome testing;
- feature promotion.

A separate later scientific execution authority is required for any such work.

## 19. Current decision

V2_CANONICAL_ARTIFACT_SHA256 = b9c54604863ed15c237fa17c7890a20f3f5ec062a7429638b39e8b468be050a7

V2_CANONICAL_ARTIFACT_BYTES = 1465573

ROW_COUNT = 1800

UNIQUE_ROW_IDS = 1800

UNIQUE_SOURCE_PAIR_IDS = 300

C0_SHAM_COUNT = 300

C1_TITLE_COUNT = 300

C2_NAME_COUNT = 300

C3_ROLE_COUNT = 300

C4_PREDICATE_COUNT = 300

C5_TITLE_NAME_COUNT = 300

SCHEMA_ORDER_VALIDATION = PASS

COMPLETE_BLOCK_VALIDATION = PASS

PROVENANCE_VALIDATION = PASS

FORBIDDEN_FIELD_VALIDATION = PASS

SERIALIZATION_VALIDATION = PASS

V2_CANONICAL_MATERIALIZATION_EXECUTIONS = 1

V2_DETERMINISTIC_RERUN_EXECUTIONS = 1

DETERMINISTIC_BYTE_IDENTITY = PASS

TEMP_RERUN_DELETED = YES

PROTECTED_FILE_IMMUTABILITY = PASS

FAILED_V1_EXECUTION_AUTHORITY_REUSED = NO

FAILED_V1_OUTPUT_EXISTS = NO

ONLY_PERSISTENT_EXECUTION_DELTA = V2_CANONICAL_JSONL

ARTIFACT_STAGED = NO

SCIENTIFIC_OUTCOME_CONCLUSION = NOT_ESTABLISHED

ARTIFACT_FREEZE = NOT_AUTHORIZED_UNTIL_THIS_AUTHORITY_IS_FROZEN

## 20. Next boundary

If this authority is frozen:

ARTIFACT_FREEZE = AUTHORIZED_FOR_EXACT_CANONICAL_JSONL

ARTIFACT_COMMIT_SCOPE = EXACT_ONE_FILE

The immediate next phase becomes:

GEN4_SIX_CELL_CANONICAL_ARTIFACT_FREEZE_V2

## 21. Stop condition

Stop after this candidate is created and reviewed.

Do not stage the canonical JSONL.

Do not commit the canonical JSONL.

Do not push the canonical JSONL.

Do not run another materialization.

Do not run another deterministic rerun.

Do not train.

Do not evaluate.

Do not execute a model.

Do not execute a tokenizer.

Do not use Kaggle.
