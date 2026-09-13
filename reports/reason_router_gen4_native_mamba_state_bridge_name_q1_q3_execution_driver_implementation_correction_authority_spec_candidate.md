# ContraMamba Gen4 Native Mamba State Bridge
# NAME Q1/Q3 Execution Driver
# Implementation Correction Authority
# Candidate

STATUS =
CANDIDATE

AUTHORITY_ID =
GEN4_NATIVE_MAMBA_STATE_BRIDGE_NAME_Q1_Q3_EXECUTION_DRIVER_IMPLEMENTATION_CORRECTION_AUTHORITY

PHASE =
IMPLEMENTATION_CORRECTION_AUTHORITY

SCIENTIFIC_EXECUTION_AUTHORIZED =
NO

CANONICAL_TOKENIZER_EXECUTION_AUTHORIZED =
NO

CHECKPOINT_LOADING_AUTHORIZED =
NO

MODEL_FORWARD_AUTHORIZED =
NO

NATIVE_STATE_EXTRACTION_AUTHORIZED =
NO

STATISTICAL_TESTING_AUTHORIZED =
NO

TRAINING_AUTHORIZED =
NO

KAGGLE_AUTHORIZED =
NO

GPU_AUTHORIZED =
NO


## 1. Parent authority and frozen implementation

IMPLEMENTATION_VALIDATION_FREEZE_COMMIT =
b8f2f22820f8f383dd47c349e63e0327d38b9f56

Q1_Q3_PRIMITIVE_IMPLEMENTATION_COMMIT =
d5317b2c5be09464c9196325429479c6bff25efd

Q1_Q3_IMPLEMENTATION_AUTHORITY =
c395e634448a3b37c30053d89abef37c5a269afe

Q1_Q3_MEASUREMENT_SHA256 =
2236d19a46416e5042057ec84c565dbe523ef791f7266c30b26b033d260a510b

Q1_Q3_EXTRACTION_SHA256 =
9e9ca05aaee43970c8d0aa101c7f0c0b86ace9166ae798f70c800fcb61651c4f

PRIMARY_PHASE_D_MEASUREMENT_SHA256 =
7729424f03058b86b4f120dc0e6da573d6c996b0877858f2d6d38aa94dac268c

PRIMARY_PHASE_D_EXTRACTION_SHA256 =
653f96713d8bdc776cdf03733ae230784e413f960f58a9c120f00cb8c6a6d3eb


## 2. Correction reason

The frozen Q1/Q3 primitive implementation is valid for:

- exact secondary layer set {5,17};
- simultaneous dual-layer capture;
- full frozen R2 coordinate validation;
- C0_SHAM/C2_NAME selection;
- A_NAME event binding;
- support-state construction;
- endpoint construction;
- transactional output bundle construction;
- manifest provenance validation.

However, the frozen Q1/Q3 extraction module does not contain an executable
end-to-end CLI entry point.

It contains no canonical:

- argparse execution interface;
- main() execution entry point;
- execution-HEAD binding orchestration;
- end-to-end tokenizer/model/checkpoint orchestration.

Therefore:

Q1_Q3_PRIMITIVE_CODE_CORRECTNESS =
PRESERVED

Q1_Q3_EXECUTION_READY =
NO_NOT_UNTIL_DRIVER_IS_FROZEN

PRIOR_FREEZE_SCIENTIFIC_CONTENT_CHANGED =
NO

PRIOR_SCIENTIFIC_SPEC_CHANGED =
NO

PRIOR_Q1_Q3_PRIMITIVE_FILES_CHANGED =
NO


## 3. Exact authorized implementation delta

Exactly two new files are authorized:

scripts/reason_router_gen4_native_mamba_state_q1_q3_execution.py

tests/test_reason_router_gen4_native_mamba_state_q1_q3_execution.py

EXISTING_TRACKED_FILE_MODIFICATION =
PROHIBITED

Q1_Q3_FROZEN_FOUR_FILE_MODIFICATION =
PROHIBITED

PRIMARY_PHASE_D_FILE_MODIFICATION =
PROHIBITED

REPORT_MODIFICATION_DURING_IMPLEMENTATION =
PROHIBITED

EXPECTED_DELTA =
EXACTLY_TWO_NEW_FILES


## 4. Driver role

The new execution driver is orchestration only.

It must reuse the frozen Q1/Q3 primitives rather than duplicate or redefine:

- layer-selection semantics;
- native-state capture semantics;
- coordinate hashing semantics;
- A_NAME binding semantics;
- support-state window semantics;
- kinematic endpoint formulas;
- artifact schemas;
- manifest construction semantics.

The driver may reuse the proven primary Phase D orchestration pattern for:

- execution commit binding;
- runtime fail-closed gating;
- canonical tokenizer loading;
- frozen input reconstruction;
- representative model loading;
- CPU-only inference;
- transactional scientific execution;
- CLI structure.

PRIMARY_PHASE_D_SCIENTIFIC_LAYER_POLICY_REUSE =
PROHIBITED

The primary layer-11 scientific policy must not leak into Q1/Q3 execution.


## 5. Required CLI

The future driver must expose an explicit CLI sufficient for a pinned
scientific execution command.

Required arguments:

--output-dir

--expected-execution-head

--driver-implementation-commit

--expected-driver-sha256

--execution-authority-commit

Optional snapshot-location arguments may be provided only for the same frozen
model/tokenizer identities.

ARBITRARY_LAYER_ARGUMENT =
PROHIBITED

ARBITRARY_CELL_ARGUMENT =
PROHIBITED

ARBITRARY_ANCHOR_ARGUMENT =
PROHIBITED

ARBITRARY_CHECKPOINT_ARM_ARGUMENT =
PROHIBITED


## 6. Execution binding

Before canonical scientific work, the driver must fail closed on:

- exact current Git HEAD;
- exact driver implementation commit;
- exact driver SHA256;
- existence/ancestry of the execution authority commit;
- exact frozen Q1/Q3 primitive identities;
- exact frozen primary measurement identity where reused.

No scientific tokenizer reconstruction, checkpoint deserialization, model
construction, or model forward may occur before required execution binding and
runtime gates pass.


## 7. Frozen input-coordinate policy

STRUCTURAL_ROWS =
1800

R2_ENCODED_COORDINATE_SHA256 =
d3cf61e55bb04e6dbebe44be433597cce958cb15989412dcf6efc15a329c576a

The driver must reconstruct and validate the complete 1800-row frozen
coordinate before selecting the scientific subset.

The coordinate identity is over exactly:

- row_id;
- source_pair_id;
- contrast_cell_id;
- input_ids;
- attention_mask;
- claim_mask;
- evidence_mask.

INPUT_IDS_ONLY_HASH =
PROHIBITED

SUBSET_BEFORE_FULL_COORDINATE_HASH =
PROHIBITED

ALTERNATE_RETOKENIZATION =
PROHIBITED


## 8. Frozen scientific subset

ESTIMAND =
DELTA_NAME_ONLY

CELL_SET =
{C0_SHAM,C2_NAME}

SOURCE_PAIR_COUNT =
300

MODEL_INPUT_ROW_COUNT =
600

SEMANTIC_ANCHOR =
A_NAME

EVENT_COORDINATE_FIELD =
absolute_anchor_token_index

LAYER_SET =
{5,17}


## 9. Model and forward policy

REPRESENTATIVE_SEED =
180

REPRESENTATIVE_ARM =
G3-GROUP-D-HALF

REPRESENTATIVE_CHECKPOINT_SHA256 =
1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f

DEVICE =
CPU_ONLY

MODEL_REPLICATION_COUNT =
1

EXPECTED_BACKBONE_FORWARD_COUNT =
600

LAYERS_CAPTURED_PER_FORWARD =
2

CAPTURE_MODE =
SIMULTANEOUS_SINGLE_FORWARD

SEQUENTIAL_1200_FORWARD_LAYER_SPLIT =
PROHIBITED

TRAINING =
PROHIBITED

BACKWARD =
PROHIBITED

GRADIENT_UPDATES =
PROHIBITED


## 10. Frozen measurements

SUPPORT_WINDOW =
[a-1,a+4]

SUPPORT_STATES_PER_ROW_PER_LAYER =
6

EXPECTED_SUPPORT_STATE_ROWS =
7200

EXPECTED_ENDPOINT_ROWS =
1200

ENDPOINTS =
POST4_SPEED
POST4_TURNING
POST4_PATH_EFFICIENCY

MIXED_LAYER_ENDPOINT =
PROHIBITED


## 11. Future artifact contract

A later separately authorized scientific execution may create exactly:

manifest.json

support_state_rows.jsonl

support_states.npy

kinematic_endpoints.jsonl

SHA256SUMS.txt

The driver must use the already frozen Q1/Q3 bundle-validation and manifest
semantics wherever applicable.

No statistical test belongs in this driver.


## 12. Implementation-phase restrictions

During this implementation correction:

CANONICAL_TOKENIZER_EXECUTION =
NO

SCIENTIFIC_CHECKPOINT_LOAD =
NO

CANONICAL_MODEL_FORWARD =
NO

Q1_Q3_SCIENTIFIC_EXTRACTION =
NO

STATISTICAL_TESTING =
NO

TRAINING =
NO

KAGGLE =
NO

GPU =
NO

SCIENTIFIC_EVIDENCE_CREATED =
NO


## 13. Allowed validation

Allowed validation is synthetic/static only.

Required:

- Python compile of the new driver and test;
- dedicated synthetic driver tests;
- existing frozen Q1/Q3 16-test suite;
- git diff --check;
- exact two-new-file scope check;
- byte-identity proof for all previously frozen Q1/Q3 files;
- no canonical tokenizer/model/checkpoint execution.

Tests must demonstrate at minimum:

- no model constructor before full coordinate gate;
- exactly 600 synthetic future forwards;
- both layers returned by each forward;
- layer set fixed to {5,17};
- execution binding fails closed;
- wrong driver SHA fails closed;
- wrong HEAD fails closed;
- output collision fails closed;
- no training/statistics path;
- CLI exposes required execution-binding arguments.


## 14. Completion criterion

Successful implementation correction may establish only:

PASS_READY_FOR_EXECUTION_DRIVER_VALIDATION_FREEZE

It does not authorize scientific execution.


## 15. Next phase after successful correction

NEXT_PHASE =
NAME_Q1_Q3_EXECUTION_DRIVER_VALIDATION_FREEZE

SCIENTIFIC_EXTRACTION_EXECUTION_AUTHORITY =
NOT_YET_CREATED

Only after the driver implementation and its validation freeze are committed
and pushed may a separate scientific extraction execution authority be created.
