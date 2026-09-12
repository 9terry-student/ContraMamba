# ContraMamba Gen4 Six-Cell Canonical Materialization Execution Authority - Candidate

## 1. Status

- Status: CANDIDATE
- Phase: GEN4_SIX_CELL_CANONICAL_MATERIALIZATION_EXECUTION_AUTHORITY
- Frozen implementation commit: 1e5ee9794c2b849d3b3ca2fbfa36baf63e55a312
- Frozen implementation authority: 8e3aa94c69360c870fe6f6f808a611916401fa03
- Frozen implementation specification: 40d735d83018c7fc9b086226e95d1e46b141b0ab
- Frozen source-structure feasibility authority: c90c1eda64560875232a3c5c1c33ef21886b2e60
- Frozen contrast specification: 0a0da5354782e542520fb5bba146ab1a599d17ef
- Frozen generator semantic authority: 91c3dcd7abadcd6cf0d6d2f1c299f3d6fa6e28ea
- Canonical materialization currently authorized: NO until this authority is frozen
- Training/Evaluation allowed: NO
- Model execution allowed: NO
- Tokenizer execution allowed: NO
- Kaggle allowed: NO
- Automatic Commit/Push: NO

This authority candidate defines one bounded local CPU canonical structural
materialization.

It does not authorize outcome-bearing experimentation.

## 2. Frozen implementation identity

Implementation file:

scripts/materialize_reason_router_gen4_six_cell_contrast.py

Frozen SHA256:

d2c59260788972c2268f83b8ea1c320b93c923b403944292755001c37d23644c

Focused test file:

tests/test_materialize_reason_router_gen4_six_cell_contrast.py

Frozen SHA256:

553166c6a0557d5d219c384fdb321f67c38f7bb67dcba5e4d9f510f0adfc842e

Frozen implementation commit:

1e5ee9794c2b849d3b3ca2fbfa36baf63e55a312

Pre-freeze focused validation established:

PY_COMPILE = PASS

FOCUSED_SYNTHETIC_PYTEST = 37 PASSED

BUILD_CONTROLLED_V5_UNCHANGED = YES

Therefore the execution authority is tied to these exact implementation bytes.

## 3. Frozen generator identity

Generator semantic authority:

91c3dcd7abadcd6cf0d6d2f1c299f3d6fa6e28ea

Generator source:

scripts/build_controlled_v5.py

Frozen generator Git blob:

baee23a9f71333125f4a8735c2c92d20cab7eb4f

Canonical execution must fail closed if the materializer's source-identity
verification does not confirm this frozen generator identity.

## 4. Canonical source population

The authorized source population is exactly:

300 structured facts

obtained through the frozen generator-side structured-fact interface:

fact_templates_for_count(300)

The frozen source-structure feasibility audit established:

STRUCTURED_FACT_COUNT = 300

UNIQUE_STRUCTURED_PAIR_IDS = 300

PAIR_UNIVERSE_EXACT_MATCH = YES

COMPLETE_SIX_CELL_SOURCE_STRUCTURE_FEASIBLE = YES

No other source count is canonical under this authority.

Therefore:

CANONICAL_SOURCE_PAIR_COUNT = 300

## 5. Exact structural design

Each source pair must produce exactly six rows:

1. C0_SHAM
2. C1_TITLE
3. C2_NAME
4. C3_ROLE
5. C4_PREDICATE
6. C5_TITLE_NAME

Expected canonical row count:

300 * 6 = 1800

Therefore:

EXPECTED_CANONICAL_ROW_COUNT = 1800

Expected count for every contrast cell:

300

No partial block is valid.

## 6. Exact mechanism and schema

Exact mechanism_id:

masked_slot_substitution_v1

Exact schema_version:

GEN4_SIX_CELL_MASKED_SLOT_SUBSTITUTION_V1

Exact canonical axis order:

title
name
role
predicate

Exact masks:

C0_SHAM       [0,0,0,0]
C1_TITLE      [1,0,0,0]
C2_NAME       [0,1,0,0]
C3_ROLE       [0,0,1,0]
C4_PREDICATE  [0,0,0,1]
C5_TITLE_NAME [1,1,0,0]

These identities are structural declarations.

They must not be reconstructed from rendered text.

## 7. Exact canonical output path

The only canonical output path authorized is:

reports/reason_router_gen4_six_cell_masked_slot_substitution_materialization_1e5ee9794c2b849d3b3ca2fbfa36baf63e55a312/gen4_six_cell_masked_slot_substitution.jsonl

Therefore:

CANONICAL_OUTPUT_PATH_FROZEN = YES

No alternate repository path may be treated as the canonical artifact under this
authority.

## 8. Exact authorized canonical execution command

After this authority is reviewed, committed, and frozen, the authorized
canonical materialization command is exactly:

python scripts/materialize_reason_router_gen4_six_cell_contrast.py --num-pairs 300 --output reports/reason_router_gen4_six_cell_masked_slot_substitution_materialization_1e5ee9794c2b849d3b3ca2fbfa36baf63e55a312/gen4_six_cell_masked_slot_substitution.jsonl

This is a local CPU structural generation operation.

No GPU is needed.

No Kaggle execution is authorized.

## 9. Canonical execution cardinality checks

Immediately after materialization, validation must establish:

ROW_COUNT = 1800

UNIQUE_ROW_IDS = 1800

UNIQUE_SOURCE_PAIR_IDS = 300

ROWS_PER_PAIR = 6 for every source pair

C0_SHAM_COUNT = 300

C1_TITLE_COUNT = 300

C2_NAME_COUNT = 300

C3_ROLE_COUNT = 300

C4_PREDICATE_COUNT = 300

C5_TITLE_NAME_COUNT = 300

Every pair must contain the exact canonical six-cell set in canonical order.

## 10. Structural provenance checks

Every row must contain the exact frozen values:

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

## 11. Exact output schema

Every canonical row must contain exactly these fields in this order:

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

No extra field is allowed.

## 12. Outcome-field prohibition

The canonical structural artifact must not contain:

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

No outcome semantics are authorized during materialization or validation.

## 13. Rendered-text boundary

The generated artifact contains claim and evidence because these are rendered
outputs of the frozen structured generator.

However:

RENDERED_TEXT_DEFINES_CELL_IDENTITY = NO

No claim or evidence text may be parsed, searched, classified, or manually
interpreted to repair or determine:

- contrast_cell_id
- axis_mask
- intended_changed_axes
- generator_source_fields

All structural identity must come from the generator declaration.

## 14. Pair invariants

For every source pair:

- all six rows must have the same source_pair_id;
- all six rows must have the same claim;
- all six rows must have mechanism_id = masked_slot_substitution_v1;
- exactly six unique row IDs must exist;
- the exact six canonical cell IDs must exist;
- canonical cell order must be preserved.

Therefore:

COMPLETE_SIX_CELL_PAIR_BLOCK = REQUIRED

CLAIM_FIXED_WITHIN_PAIR = REQUIRED

## 15. Deterministic rerun authorization

One additional local CPU rerun is authorized solely to test deterministic
serialization.

The rerun must:

- use the same frozen implementation;
- use --num-pairs 300;
- write only to a temporary non-canonical path;
- use the same source and mechanism;
- not inspect outcomes;
- not modify the canonical output.

The temporary rerun output must be byte-identical to the canonical artifact.

Therefore:

DETERMINISTIC_RERUN_COUNT = 1

DETERMINISTIC_BYTE_IDENTITY = REQUIRED

The temporary rerun artifact must be deleted after successful comparison.

It is not a scientific artifact and must not be committed.

## 16. Source immutability

After canonical materialization and deterministic rerun, validation must
establish that:

scripts/build_controlled_v5.py

is unchanged.

It must also establish that the two frozen implementation files are unchanged.

The existing Gen4 operator-cell sidecar must remain unchanged.

Existing canonical controlled datasets must remain unchanged.

## 17. Repository mutation boundary

The only persistent repository delta authorized by execution is the canonical
JSONL artifact at the frozen output path.

No report is created by the materialization command itself.

No existing tracked file may change.

No unrelated untracked file may be touched.

Do not clean or reset the worktree.

Do not stage the canonical artifact during execution validation.

Commit/push is not authorized by this execution authority.

## 18. Local CPU boundary

This canonical structural materialization is CPU-only.

GPU use is unnecessary and unauthorized.

Kaggle is unnecessary and unauthorized.

No external service is required.

## 19. Explicitly forbidden execution

This authority does not authorize:

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
- label derivation;
- manual semantic annotation;
- rendered-text semantic reconstruction;
- Kaggle execution.

## 20. Scientific interpretation boundary

Successful execution establishes only:

- canonical materialization execution success;
- structural artifact cardinality validity;
- structural artifact provenance validity;
- deterministic serialization validity;
- source immutability.

It does not establish:

- semantic-axis outcome effects;
- title effect;
- name effect;
- role effect;
- predicate effect;
- title-name interaction effect;
- predictive utility;
- causal relevance;
- mechanistic relevance;
- performance improvement.

Therefore:

SCIENTIFIC_OUTCOME_CONCLUSION = NOT_ESTABLISHED_BY_MATERIALIZATION

## 21. Failure conditions

Canonical execution must be treated as BLOCKED if any of the following occurs:

- HEAD does not equal the frozen execution-authority start commit;
- implementation SHA differs;
- test SHA differs;
- frozen generator identity differs;
- canonical output already exists before the authorized execution;
- materializer exits nonzero;
- row count is not 1800;
- unique row count is not 1800;
- pair count is not 300;
- any pair lacks a six-cell block;
- any cell count is not 300;
- schema field order differs;
- provenance constant differs;
- label/outcome field appears;
- deterministic rerun differs by even one byte;
- protected source or implementation files change.

No failed canonical artifact may be promoted.

## 22. Required execution evidence

After authorized execution, the returned evidence must include:

- execution start HEAD;
- implementation SHA256;
- test SHA256;
- canonical artifact path;
- canonical artifact SHA256;
- canonical artifact byte count;
- row count;
- unique row-id count;
- unique source-pair count;
- six per-cell counts;
- complete-block validation result;
- provenance validation result;
- forbidden-field validation result;
- deterministic rerun SHA256;
- deterministic byte-identity result;
- protected-file immutability result;
- confirmation that the temporary rerun artifact was deleted;
- confirmation that no training/evaluation/model/tokenizer/Kaggle execution occurred.

## 23. Post-execution boundary

After successful materialization and validation:

stop.

Do not stage the canonical artifact yet.

Do not commit it.

Do not push it.

The next object is:

GEN4_SIX_CELL_CANONICAL_MATERIALIZATION_RESULT_PROVENANCE_AUTHORITY

That later authority will decide how the canonical artifact and execution result
are recorded and frozen.

## 24. Current decision

IMPLEMENTATION_COMMIT = 1e5ee9794c2b849d3b3ca2fbfa36baf63e55a312

IMPLEMENTATION_SHA256 = d2c59260788972c2268f83b8ea1c320b93c923b403944292755001c37d23644c

TEST_SHA256 = 553166c6a0557d5d219c384fdb321f67c38f7bb67dcba5e4d9f510f0adfc842e

CANONICAL_SOURCE_PAIR_COUNT = 300

EXPECTED_CANONICAL_ROW_COUNT = 1800

CANONICAL_OUTPUT_PATH_FROZEN = YES

DETERMINISTIC_RERUN_COUNT = 1

STRUCTURAL_ARTIFACT_LABEL_FIELDS = FORBIDDEN

RENDERED_TEXT_DEFINES_CELL_IDENTITY = NO

TRAINING_EVALUATION = NOT_AUTHORIZED

MODEL_EXECUTION = NOT_AUTHORIZED

TOKENIZER_EXECUTION = NOT_AUTHORIZED

KAGGLE_EXECUTION = NOT_AUTHORIZED

CANONICAL_MATERIALIZATION = NOT_AUTHORIZED_UNTIL_THIS_AUTHORITY_IS_FROZEN

## 25. Next boundary

If this candidate is reviewed, committed, and frozen:

CANONICAL_MATERIALIZATION = AUTHORIZED_ONCE

DETERMINISTIC_RERUN = AUTHORIZED_ONCE

No other execution becomes authorized.

## 26. Stop condition for this authority phase

Stop after this authority candidate is created and reviewed.

Do not execute the canonical materializer yet.

Do not create the 1800-row artifact yet.

Do not run the deterministic rerun yet.

Do not stage or commit an artifact.

Do not train.

Do not evaluate.

Do not execute a model.

Do not execute a tokenizer.

Do not use Kaggle.