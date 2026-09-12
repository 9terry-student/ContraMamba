# ContraMamba Gen4 Six-Cell Canonical Materialization Execution Authority V2 - Candidate

## 1. Status

- Status: CANDIDATE
- Phase: GEN4_SIX_CELL_CANONICAL_MATERIALIZATION_EXECUTION_AUTHORITY_V2
- Corrected implementation commit: fbc780ce12cbfbcf4e20a3cb9d1f099553045fc0
- CLI failure-recovery authority: b728dcf95d47e15d2f2a8b50a3872e5dd0c9e0bc
- Failed V1 execution authority: 3c3e0dfba7455a09e36d4fbeabaa6785b79582f2
- Original implementation commit: 1e5ee9794c2b849d3b3ca2fbfa36baf63e55a312
- Frozen generator semantic authority: 91c3dcd7abadcd6cf0d6d2f1c299f3d6fa6e28ea
- Frozen contrast specification: 0a0da5354782e542520fb5bba146ab1a599d17ef
- Training/Evaluation allowed: NO
- Model execution allowed: NO
- Tokenizer execution allowed: NO
- Kaggle allowed: NO
- Commit/Push of generated artifact: NO

This V2 authority replaces only the failed canonical execution authorization.

It does not reuse or revive the failed V1 one-time authorization.

## 2. V1 failure disposition

The V1 authorized canonical command failed before materialization because direct
file execution did not place the repository root on sys.path.

Observed failure class:

CLI_ENTRYPOINT_IMPORT_PATH_DEFECT

V1 execution authority:

3c3e0dfba7455a09e36d4fbeabaa6785b79582f2

V1 canonical materialization attempt count:

1

V1 canonical artifact created:

NO

V1 deterministic rerun executed:

NO

Therefore:

FAILED_V1_EXECUTION_AUTHORITY_REUSE = FORBIDDEN

FAILED_V1_OUTPUT_PROMOTION = FORBIDDEN

V2 does not count as a retry under V1.

It is a newly authorized execution bound to a corrected implementation.

## 3. Recovery and correction identity

Failure-recovery implementation authority:

b728dcf95d47e15d2f2a8b50a3872e5dd0c9e0bc

Corrected implementation commit:

fbc780ce12cbfbcf4e20a3cb9d1f099553045fc0

Corrected implementation file:

scripts/materialize_reason_router_gen4_six_cell_contrast.py

Corrected implementation SHA256:

b95400fd670280eeecd79bab818ad70a9f1aba9b74116201dd52bfaba1301f59

Corrected focused test:

tests/test_materialize_reason_router_gen4_six_cell_contrast.py

Corrected test SHA256:

1ac54025cd51062ed8b4242349c6ef19723b2fe5306969d9f215aa96abc08746

Correction validation established:

PY_COMPILE = PASS

FOCUSED_PYTEST = 38 PASSED

DIRECT_FILE_HELP_EXIT_CODE = 0

BUILD_CONTROLLED_V5_UNCHANGED = YES

The correction changed direct-file importability only.

SCIENTIFIC_SEMANTICS_DELTA = NONE

STRUCTURAL_MATERIALIZATION_SEMANTICS_DELTA = NONE

## 4. Frozen generator identity

Generator semantic authority:

91c3dcd7abadcd6cf0d6d2f1c299f3d6fa6e28ea

Generator source:

scripts/build_controlled_v5.py

Frozen generator Git blob:

baee23a9f71333125f4a8735c2c92d20cab7eb4f

The V2 materializer must fail closed on generator identity drift.

## 5. Canonical source population

The V2 canonical source population remains exactly:

300 structured facts

through:

fact_templates_for_count(300)

Therefore:

CANONICAL_SOURCE_PAIR_COUNT = 300

No other source count is canonical.

## 6. Exact six-cell design

Every source pair must produce exactly:

C0_SHAM
C1_TITLE
C2_NAME
C3_ROLE
C4_PREDICATE
C5_TITLE_NAME

Exact expected total:

300 * 6 = 1800

Therefore:

EXPECTED_CANONICAL_ROW_COUNT = 1800

Expected count of every cell:

300

Every pair must be a complete six-cell block.

## 7. Exact structural identities

schema_version:

GEN4_SIX_CELL_MASKED_SLOT_SUBSTITUTION_V1

mechanism_id:

masked_slot_substitution_v1

Canonical axis order:

title
name
role
predicate

Masks:

C0_SHAM       [0,0,0,0]
C1_TITLE      [1,0,0,0]
C2_NAME       [0,1,0,0]
C3_ROLE       [0,0,1,0]
C4_PREDICATE  [0,0,0,1]
C5_TITLE_NAME [1,1,0,0]

These remain generator-declared structural identities.

Rendered text does not define cell identity.

## 8. V2 canonical output path

The failed V1 path was:

reports/reason_router_gen4_six_cell_masked_slot_substitution_materialization_1e5ee9794c2b849d3b3ca2fbfa36baf63e55a312/gen4_six_cell_masked_slot_substitution.jsonl

That path must remain absent.

The only V2 canonical output path is:

reports/reason_router_gen4_six_cell_masked_slot_substitution_materialization_fbc780ce12cbfbcf4e20a3cb9d1f099553045fc0/gen4_six_cell_masked_slot_substitution.jsonl

Therefore:

V2_CANONICAL_OUTPUT_PATH_FROZEN = YES

V1_OUTPUT_PATH_REUSE = FORBIDDEN

The V2 path records the corrected implementation identity externally.

## 9. Exact V2 execution command

After this V2 authority is manually reviewed, committed, pushed, and frozen,
the canonical command authorized exactly once is:

python scripts/materialize_reason_router_gen4_six_cell_contrast.py --num-pairs 300 --output reports/reason_router_gen4_six_cell_masked_slot_substitution_materialization_fbc780ce12cbfbcf4e20a3cb9d1f099553045fc0/gen4_six_cell_masked_slot_substitution.jsonl

The direct-file entrypoint form is intentional.

It is the invocation form covered by the correction regression test.

No alternative command form is authorized as the canonical V2 command.

## 10. Required cardinality validation

Immediately after V2 canonical materialization, validation must establish:

ROW_COUNT = 1800

UNIQUE_ROW_IDS = 1800

UNIQUE_SOURCE_PAIR_IDS = 300

ROWS_PER_PAIR = 6

C0_SHAM_COUNT = 300

C1_TITLE_COUNT = 300

C2_NAME_COUNT = 300

C3_ROLE_COUNT = 300

C4_PREDICATE_COUNT = 300

C5_TITLE_NAME_COUNT = 300

COMPLETE_SIX_CELL_PAIR_BLOCK = REQUIRED

## 11. Required provenance validation

Every row must preserve exactly:

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

No provenance drift is acceptable.

## 12. Exact row schema

Each row must contain exactly, in order:

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

No extra row field is authorized.

## 13. Outcome prohibition

No structural artifact row may contain outcome-bearing fields, including:

- final_label
- frame_compatible_label
- predicate_covered_label
- sufficiency_label
- polarity_label
- primary_failure_type
- predictions
- logits
- probabilities
- evaluator outputs
- error-cohort labels
- training outcomes
- evaluation outcomes

Therefore:

STRUCTURAL_ARTIFACT_LABEL_FIELDS = FORBIDDEN

## 14. Rendered-text boundary

claim and evidence are renderer outputs.

They may not be parsed or interpreted to derive or repair:

contrast_cell_id

axis_mask

intended_changed_axes

generator_source_fields

Therefore:

RENDERED_TEXT_DEFINES_CELL_IDENTITY = NO

## 15. Pair invariants

For every pair:

- source_pair_id is fixed across six rows;
- claim is fixed across six rows;
- mechanism_id is fixed;
- six row IDs are unique;
- exact canonical cell order is preserved;
- exact canonical masks are preserved.

CLAIM_FIXED_WITHIN_PAIR = REQUIRED

COMPLETE_SIX_CELL_PAIR_BLOCK = REQUIRED

## 16. V2 deterministic rerun

Exactly one additional local CPU execution is authorized after successful
canonical generation solely to verify deterministic serialization.

It must:

- use the same corrected implementation commit;
- use --num-pairs 300;
- write to a temporary non-repository canonical path;
- not alter the V2 canonical artifact;
- be compared byte-for-byte with the canonical artifact;
- be deleted after comparison.

Therefore:

V2_DETERMINISTIC_RERUN_COUNT = 1

DETERMINISTIC_BYTE_IDENTITY = REQUIRED

Temporary rerun output must not be committed.

## 17. Execution environment

V2 canonical materialization is:

LOCAL_CPU_ONLY = YES

GPU_REQUIRED = NO

KAGGLE_EXECUTION = NOT_AUTHORIZED

No model or tokenizer execution is involved.

## 18. Persistent repository mutation boundary

The only persistent delta authorized by V2 execution is:

reports/reason_router_gen4_six_cell_masked_slot_substitution_materialization_fbc780ce12cbfbcf4e20a3cb9d1f099553045fc0/gen4_six_cell_masked_slot_substitution.jsonl

No existing tracked file may change.

No unrelated untracked file may be touched.

Do not clean or reset the worktree.

Do not stage the artifact during execution validation.

Commit/push of the artifact is not authorized by V2 execution authority.

## 19. Protected files

The following identities must remain unchanged through execution:

scripts/build_controlled_v5.py

scripts/materialize_reason_router_gen4_six_cell_contrast.py

tests/test_materialize_reason_router_gen4_six_cell_contrast.py

The corrected implementation SHA must remain:

b95400fd670280eeecd79bab818ad70a9f1aba9b74116201dd52bfaba1301f59

The corrected test SHA must remain:

1ac54025cd51062ed8b4242349c6ef19723b2fe5306969d9f215aa96abc08746

## 20. Explicitly forbidden execution

V2 does not authorize:

- training;
- evaluation;
- model inference;
- tokenizer execution;
- representation extraction;
- activation logging;
- loss computation;
- gradient computation;
- statistical outcome testing;
- feature promotion;
- label assignment;
- manual semantic annotation;
- Kaggle.

## 21. Scientific interpretation boundary

A V2 execution PASS would establish only:

- corrected CLI execution success;
- canonical structural materialization success;
- artifact cardinality validity;
- structural provenance validity;
- deterministic byte reproducibility;
- protected-source immutability.

It would not establish:

- title effect;
- name effect;
- role effect;
- predicate effect;
- title-name interaction effect;
- causal relevance;
- mechanistic relevance;
- predictive value;
- performance improvement.

Therefore:

SCIENTIFIC_OUTCOME_CONCLUSION = NOT_ESTABLISHED_BY_MATERIALIZATION

## 22. Failure conditions

V2 execution is BLOCKED if:

- execution starts from a HEAD other than the frozen V2 authority commit;
- corrected implementation SHA differs;
- corrected test SHA differs;
- generator identity differs;
- failed V1 output unexpectedly exists;
- V2 output already exists before authorized execution;
- direct-file materializer exits nonzero;
- row count differs from 1800;
- unique row count differs from 1800;
- pair count differs from 300;
- any cell count differs from 300;
- any pair block is incomplete;
- row schema differs;
- provenance differs;
- forbidden outcome field appears;
- deterministic rerun is not byte-identical;
- protected files change.

A failed V2 artifact must not be promoted.

## 23. Required returned execution evidence

After authorized V2 execution, return:

- V2 execution start HEAD;
- corrected implementation SHA256;
- corrected test SHA256;
- V2 canonical artifact path;
- V2 artifact SHA256;
- V2 artifact byte count;
- row count;
- unique row-id count;
- unique source-pair count;
- six cell counts;
- complete-block validation;
- schema/order validation;
- provenance validation;
- forbidden-field validation;
- deterministic rerun SHA256;
- deterministic byte-identity result;
- temporary rerun deletion confirmation;
- protected-file immutability result;
- confirmation failed V1 path remains absent;
- confirmation no model/tokenizer/training/evaluation/Kaggle execution occurred.

## 24. Post-execution boundary

After successful V2 materialization and validation:

stop.

Do not stage the canonical artifact.

Do not commit it.

Do not push it.

The next object is:

GEN4_SIX_CELL_CANONICAL_MATERIALIZATION_RESULT_PROVENANCE_AUTHORITY_V2

That later authority determines artifact/result freeze.

## 25. Current decision

CORRECTED_IMPLEMENTATION_COMMIT = fbc780ce12cbfbcf4e20a3cb9d1f099553045fc0

CORRECTED_IMPLEMENTATION_SHA256 = b95400fd670280eeecd79bab818ad70a9f1aba9b74116201dd52bfaba1301f59

CORRECTED_TEST_SHA256 = 1ac54025cd51062ed8b4242349c6ef19723b2fe5306969d9f215aa96abc08746

CLI_FAILURE_RECOVERY_AUTHORITY = b728dcf95d47e15d2f2a8b50a3872e5dd0c9e0bc

FAILED_V1_EXECUTION_AUTHORITY = 3c3e0dfba7455a09e36d4fbeabaa6785b79582f2

FAILED_V1_EXECUTION_AUTHORITY_REUSE = FORBIDDEN

CANONICAL_SOURCE_PAIR_COUNT = 300

EXPECTED_CANONICAL_ROW_COUNT = 1800

V2_CANONICAL_OUTPUT_PATH_FROZEN = YES

V1_OUTPUT_PATH_REUSE = FORBIDDEN

V2_DETERMINISTIC_RERUN_COUNT = 1

DETERMINISTIC_BYTE_IDENTITY = REQUIRED

STRUCTURAL_ARTIFACT_LABEL_FIELDS = FORBIDDEN

RENDERED_TEXT_DEFINES_CELL_IDENTITY = NO

TRAINING_EVALUATION = NOT_AUTHORIZED

MODEL_EXECUTION = NOT_AUTHORIZED

TOKENIZER_EXECUTION = NOT_AUTHORIZED

KAGGLE_EXECUTION = NOT_AUTHORIZED

V2_CANONICAL_MATERIALIZATION = NOT_AUTHORIZED_UNTIL_THIS_AUTHORITY_IS_FROZEN

## 26. Next boundary

If this V2 candidate is manually reviewed, committed, pushed, and frozen:

V2_CANONICAL_MATERIALIZATION = AUTHORIZED_ONCE

V2_DETERMINISTIC_RERUN = AUTHORIZED_ONCE

No other execution becomes authorized.

## 27. Stop condition

Stop after this V2 authority candidate is created and reviewed.

Do not run canonical materialization yet.

Do not run deterministic rerun.

Do not reuse the failed V1 authority.

Do not create an artifact.

Do not train.

Do not evaluate.

Do not execute a model.

Do not execute a tokenizer.

Do not use Kaggle.