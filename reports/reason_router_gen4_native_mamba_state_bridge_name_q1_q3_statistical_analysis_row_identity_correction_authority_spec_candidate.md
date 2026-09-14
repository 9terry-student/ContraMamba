# ContraMamba Gen4 Native Mamba State Bridge
# NAME Q1/Q3 Statistical Analysis Row-Identity Correction Authority
# Candidate

STATUS =
CANDIDATE

AUTHORITY_ID =
GEN4_NATIVE_MAMBA_STATE_BRIDGE_NAME_Q1_Q3_STATISTICAL_ANALYSIS_ROW_IDENTITY_CORRECTION_AUTHORITY

THIS_DOCUMENT_AUTHORIZES =
BOUNDED_IMPLEMENTATION_AND_SYNTHETIC_TEST_CORRECTION_ONLY

CANONICAL_STATISTICAL_EXECUTION =
NOT_AUTHORIZED

SCIENTIFIC_CONCLUSION =
NONE


## 1. Affected lineage

STATISTICAL_SPECIFICATION_COMMIT =
2030de52922c251caeafceb36dd5f2f57b5b39c7

IMPLEMENTATION_AUTHORITY_COMMIT =
66a35cb469a24c881830fbf69169fb6efcd6cc0c

AFFECTED_IMPLEMENTATION_COMMIT =
c83b370ce920c6878bb86f6da75da9b7689cf48b

FROZEN_Q1_Q3_EXTRACTION_IMPLEMENTATION_COMMIT =
d5317b2c5be09464c9196325429479c6bff25efd

EXTRACTION_ARTIFACT_PROVENANCE_VALIDATION_FREEZE =
e500beabf8a7e1dfc0f8260586d8f5ed8e374d5c

EXTRACTION_ARTIFACT_RELEASE_BINDING_FREEZE =
a0ce07ad828c5ee0e6a4c36d8f9a6e972090590a


## 2. Defect

The affected statistical implementation treats row_id as globally unique
across the 1200 Q1/Q3 endpoint rows.

That assumption is incompatible with the frozen extraction schema.

A frozen structural/model-input row has one row_id.

For each such row, the extraction implementation emits one endpoint row at
layer 5 and one endpoint row at layer 17 while preserving the same row_id.

Therefore the valid endpoint identity is not:

row_id

alone.

The frozen extraction validator uses:

(row_id, layer_index)

as the endpoint-row uniqueness identity.

A row_id appearing once at layer 5 and once at layer 17 is expected and valid.


## 3. Exact frozen extraction evidence

The frozen extraction implementation constructs endpoint rows using the
original selected model-input row_id unchanged.

For every selected row it emits endpoints at both layers:

5
17

The extraction validator enforces:

identity =
(row_id, layer_index)

and rejects only duplicate identities under that compound key.

Thus:

GLOBAL_ROW_ID_UNIQUENESS =
INVALID_ASSUMPTION

ROW_ID_REUSE_ACROSS_DISTINCT_FROZEN_LAYERS =
EXPECTED

ROW_LAYER_IDENTITY_UNIQUENESS =
REQUIRED


## 4. Synthetic-test defect

The affected synthetic fixture constructs row_id with layer embedded in the
identifier.

That makes every layer-specific endpoint row have a different row_id and
therefore masks the implementation defect.

The corrected synthetic fixture must model the frozen extraction semantics:

for a fixed source pair and contrast cell,
the same model-input row_id must be reused at layer 5 and layer 17.

For example, synthetic row identity may be based on:

(source_pair_id, contrast_cell_id)

without layer_index embedded in row_id.


## 5. Scientific impact

CANONICAL_STATISTICAL_EXECUTION_PERFORMED =
NO

CANONICAL_PAIR_LEVEL_CONTRASTS_COMPUTED =
NO

CANONICAL_T_STATISTICS_COMPUTED =
NO

CANONICAL_P_VALUES_COMPUTED =
NO

CANONICAL_HOLM_RESULTS_COMPUTED =
NO

SCIENTIFIC_RESULT_ARTIFACT_CREATED =
NO

SCIENTIFIC_CONCLUSION_CHANGE =
NONE

The defect would block a future canonical execution before scientific
statistics are produced.

It does not invalidate the frozen Q1/Q3 extraction artifacts.

It does not alter any endpoint value.

It does not alter the six scientific hypotheses.

It does not alter the statistical procedure.


## 6. Superseded implementation-authority clause

The following affected implementation-authority requirement:

duplicate row_id rejection

is corrected.

It must not be interpreted as global row_id uniqueness.

The corrected requirement is:

DUPLICATE_ROW_LAYER_IDENTITY_REJECTION =
REQUIRED

with identity:

(row_id, layer_index)

A repeated row_id across layer 5 and layer 17 is valid.

A repeated identical (row_id, layer_index) is invalid.


## 7. Preserved structural validation

The corrected implementation must continue to require:

- non-empty source_pair_id;
- non-empty row_id;
- semantic_anchor exactly A_NAME;
- layer_index exactly one of {5,17};
- contrast_cell_id exactly one of {C0_SHAM,C2_NAME};
- exactly 300 source pairs;
- exactly four endpoint rows per source pair;
- exactly C0_SHAM and C2_NAME at each layer;
- unique (source_pair_id, layer_index, contrast_cell_id);
- unique (row_id, layer_index);
- finite numeric values for all three frozen endpoints;
- boolean endpoint rejection;
- no missingness;
- no row dropping;
- no imputation.


## 8. Exact correction scope

Exactly these two existing files may be modified:

scripts/reason_router_gen4_native_mamba_state_name_q1_q3_statistical_analysis.py

tests/test_reason_router_gen4_native_mamba_state_name_q1_q3_statistical_analysis.py

No other tracked file may be modified.

IMPLEMENTATION_CHANGE =
replace global row_id uniqueness with row-layer identity uniqueness

TEST_FIXTURE_CHANGE =
reuse each synthetic model-input row_id across layers 5 and 17

TEST_ADDITION =
explicit acceptance of valid cross-layer row_id reuse

TEST_ADDITION =
explicit rejection of duplicate identical (row_id, layer_index)

No statistical formula change is authorized.


## 9. Preserved statistical procedure

SECONDARY_HYPOTHESIS_COUNT =
6

N_PER_HYPOTHESIS =
300

NUMERIC_ANALYSIS_DTYPE =
FLOAT64

STATISTICAL_TEST =
TWO_SIDED_ONE_SAMPLE_STUDENT_T_TEST_ON_PAIR_LEVEL_CONTRASTS

CONFIDENCE_INTERVAL =
TWO_SIDED_95_PERCENT_STUDENT_T_CONFIDENCE_INTERVAL

EFFECT_SIZE =
D_Z

MULTIPLICITY =
GLOBAL_HOLM_BONFERRONI_ALL_6

FAMILYWISE_ALPHA =
0.05

DELTA_NAME_FORMULA_CHANGE =
NO

HYPOTHESIS_ORDER_CHANGE =
NO

DECISION_RULE_CHANGE =
NO

CROSS_LAYER_TEST =
NOT_AUTHORIZED


## 10. Validation requirements

After correction run:

python -m py_compile scripts/reason_router_gen4_native_mamba_state_name_q1_q3_statistical_analysis.py

python -m pytest -q tests/test_reason_router_gen4_native_mamba_state_name_q1_q3_statistical_analysis.py

git diff --check -- scripts/reason_router_gen4_native_mamba_state_name_q1_q3_statistical_analysis.py tests/test_reason_router_gen4_native_mamba_state_name_q1_q3_statistical_analysis.py

The corrected tests must demonstrate at minimum:

1. canonical-like row_id reuse across layer 5 and layer 17 is accepted;
2. duplicate (row_id, layer_index) is rejected;
3. pair/layer/cell uniqueness remains enforced;
4. 1800 synthetic pair-level rows remain unchanged;
5. six-member global Holm behavior remains unchanged;
6. Student-t reference checks remain unchanged;
7. canonical endpoint artifact remains unopened.


## 11. Execution boundary

CANONICAL_INPUT_OPEN =
FORBIDDEN

CANONICAL_INPUT_OUTCOME_INSPECTION =
FORBIDDEN

CANONICAL_STATISTICAL_EXECUTION =
FORBIDDEN

KAGGLE =
NOT_AUTHORIZED

GPU =
NOT_AUTHORIZED

TRAINING =
NOT_AUTHORIZED

MODEL_FORWARD =
NOT_AUTHORIZED


## 12. Promotion condition

The corrected implementation may be reviewed for freeze only after:

- exactly the two authorized implementation/test files are modified;
- py_compile passes;
- the dedicated synthetic suite passes;
- canonical-like cross-layer row_id reuse is explicitly tested;
- git diff --check passes;
- no canonical statistical result is produced.

The existing c83b370 implementation is not eligible for implementation freeze.


## 13. Next phase

NEXT_PHASE =
NAME_Q1_Q3_STATISTICAL_ANALYSIS_ROW_IDENTITY_CORRECTION_IMPLEMENTATION

STATISTICAL_EXECUTION_AUTHORITY =
NOT_YET_AUTHORIZED

SCIENTIFIC_CONCLUSION =
NONE
