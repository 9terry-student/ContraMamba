# ContraMamba Gen4 Native Mamba State Bridge
# NAME Q1/Q3 Measurement / Extraction
# Implementation Validation and Freeze
# Candidate

STATUS =
CANDIDATE

AUTHORITY_ID =
GEN4_NATIVE_MAMBA_STATE_BRIDGE_NAME_Q1_Q3_IMPLEMENTATION_VALIDATION_FREEZE

PHASE =
NAME_Q1_Q3_IMPLEMENTATION_VALIDATION_AND_FREEZE

SCIENTIFIC_CONCLUSION =
NONE

THIS_DOCUMENT_CREATES_NEW_SCIENTIFIC_EVIDENCE =
NO

SCIENTIFIC_EXECUTION_ALLOWED =
NO

TOKENIZER_EXECUTION_ALLOWED =
NO

CHECKPOINT_LOADING_ALLOWED =
NO

MODEL_FORWARD_ALLOWED =
NO

NATIVE_STATE_EXTRACTION_ALLOWED =
NO

STATISTICAL_TESTING_ALLOWED =
NO

TRAINING_ALLOWED =
NO

KAGGLE_ALLOWED =
NO

GPU_ALLOWED =
NO


## 1. Frozen authority chain

Q1_Q3_IMPLEMENTATION_AUTHORITY =
c395e634448a3b37c30053d89abef37c5a269afe

Q1_Q3_IMPLEMENTATION_COMMIT =
d5317b2c5be09464c9196325429479c6bff25efd

Q1_Q3_INPUT_COORDINATE_RECONSTRUCTION_CORRECTION =
b9d26005b2e475d9e2645c507eedcbd0c79bbac3

Q1_Q3_EXTRACTION_FEASIBILITY_PROVENANCE =
c7d841a920a0a6d075f7f3804da9212ce4706673

Q1_Q3_SECONDARY_LOCALIZATION_SCIENTIFIC_SPECIFICATION =
01801ad1617b2ebc3ffa859ba440636d4755a55c

Q1_Q3_ARCHITECTURE_DEPTH_INDEX_SPECIFICATION =
ecbf6ba720c0e173ac7089a00bb5b783aa16fa6c

PHASE_F_VALIDATED_RESULT_FREEZE =
ab3428e7be08af26fa1fdafd1483a34e48fbcf8c


## 2. Exact implementation freeze

Exactly four new files are frozen by the implementation commit.

MEASUREMENT_IMPLEMENTATION_PATH =
scripts/reason_router_gen4_native_mamba_state_q1_q3_measurement.py

MEASUREMENT_IMPLEMENTATION_SHA256 =
2236d19a46416e5042057ec84c565dbe523ef791f7266c30b26b033d260a510b

EXTRACTION_IMPLEMENTATION_PATH =
scripts/reason_router_gen4_native_mamba_state_q1_q3_extraction.py

EXTRACTION_IMPLEMENTATION_SHA256 =
9e9ca05aaee43970c8d0aa101c7f0c0b86ace9166ae798f70c800fcb61651c4f

MEASUREMENT_TEST_PATH =
tests/test_reason_router_gen4_native_mamba_state_q1_q3_measurement.py

MEASUREMENT_TEST_SHA256 =
89ed02ba89c2795930163d4ea60f6850f933b07916e4a55e738d947655ff41bb

EXTRACTION_TEST_PATH =
tests/test_reason_router_gen4_native_mamba_state_q1_q3_extraction.py

EXTRACTION_TEST_SHA256 =
5258885878a1e230114503ecd3f20e1270ad2eb7ca6a94817e96c45a1fc48cdc

REMOTE_COMMIT_SCOPE =
EXACTLY_FOUR_NEW_FILES

EXISTING_TRACKED_FILE_MODIFICATION =
NONE


## 3. Preserved primary Phase D implementation

PRIMARY_MEASUREMENT_PATH =
scripts/reason_router_gen4_native_mamba_state_measurement.py

PRIMARY_MEASUREMENT_SHA256 =
7729424f03058b86b4f120dc0e6da573d6c996b0877858f2d6d38aa94dac268c

PRIMARY_EXTRACTION_PATH =
scripts/reason_router_gen4_native_mamba_state_extraction.py

PRIMARY_EXTRACTION_SHA256 =
653f96713d8bdc776cdf03733ae230784e413f960f58a9c120f00cb8c6a6d3eb

PRIMARY_PHASE_D_IMPLEMENTATION_CHANGED =
NO

PRIMARY_LAYER_11_POLICY_CHANGED =
NO


## 4. Dedicated Q1/Q3 validation

DEDICATED_Q1_Q3_TEST_RESULT =
PASS_16_OF_16

PY_COMPILE =
PASS

GIT_DIFF_CHECK =
PASS

The dedicated suite covers the corrected implementation after independent
authority-conformance review.

The initial five-test implementation was not frozen.

Before freeze, the implementation was corrected to address substantive defects
identified during independent review.


## 5. Corrected frozen-coordinate identity

R2_ENCODED_COORDINATE_SHA256 =
d3cf61e55bb04e6dbebe44be433597cce958cb15989412dcf6efc15a329c576a

The implementation validates the complete R2 encoded coordinate over:

- row_id;
- source_pair_id;
- contrast_cell_id;
- input_ids;
- attention_mask;
- claim_mask;
- evidence_mask.

INPUT_IDS_ONLY_HASH =
PROHIBITED

FULL_1800_COORDINATE_VALIDATION_BEFORE_SUBSET =
REQUIRED

COORDINATE_HASH_MISMATCH =
BLOCK_BEFORE_MODEL_CONSTRUCTION_OR_FORWARD

ALTERNATE_RETOKENIZATION =
PROHIBITED


## 6. Frozen event-coordinate correction

EVENT_MANIFEST_SHA256 =
70c84c68b36751bb7c7145b33ccb71ab91bc8ee9e6cc5f2c7a0d4e925f36581f

EVENT_MANIFEST_BYTES =
2268260

SEMANTIC_ANCHOR =
A_NAME

CANONICAL_ANCHOR_INDEX_FIELD =
absolute_anchor_token_index

INCORRECT_ANCHOR_INDEX_FIELD =
anchor_token_index

The implementation was corrected before freeze to consume the actual frozen
event-manifest coordinate field.

The event manifest itself was not changed.


## 7. Windows line-ending validation note

The frozen event manifest Git blob is:

BYTES =
2268260

SHA256 =
70c84c68b36751bb7c7145b33ccb71ab91bc8ee9e6cc5f2c7a0d4e925f36581f

On the Windows working tree, core.autocrlf produced a CRLF representation whose
raw byte count and raw SHA differed.

LF normalization reproduced the exact frozen Git blob byte count and SHA.

CLASSIFICATION =
WINDOWS_CRLF_CHECKOUT_ONLY

SCIENTIFIC_ARTIFACT_DRIFT =
NO

The Q1/Q3 implementation therefore uses CRLF-safe canonical-byte validation
where required for frozen source identity.


## 8. Primary Phase D regression validation

The existing primary Phase D suite was revalidated independently.

PRIMARY_PHASE_D_NON_CRLF_TESTS =
PASS_74

PRIMARY_PHASE_D_CRLF_SENSITIVE_TEST =
REPLACED_BY_CANONICAL_GIT_BLOB_EQUIVALENT_VALIDATION

CANONICAL_EVENT_GIT_BLOB_BYTES =
PASS

CANONICAL_EVENT_GIT_BLOB_SHA256 =
PASS

CANONICAL_EVENT_SCHEMA =
PASS

CANONICAL_EVENT_ROWS =
PASS_3600

CANONICAL_EVENT_POST4_ELIGIBILITY =
PASS

PRIMARY_PHASE_D_REGRESSION =
NO_EVIDENCE_OF_REGRESSION


## 9. Frozen Q1/Q3 layer policy

NATIVE_MAMBA_LAYER_COUNT =
24

Q1_LAYER_INDEX =
5

Q3_LAYER_INDEX =
17

SECONDARY_LAYER_SET =
{5,17}

PRIMARY_LAYER_11_IN_Q1_Q3_PATH =
PROHIBITED

ARBITRARY_LAYER_PARAMETERIZATION =
PROHIBITED

BEST_LAYER_SCAN =
PROHIBITED

FALLBACK_LAYER_SELECTION =
PROHIBITED


## 10. Dual-layer future forward policy

DUAL_LAYER_CAPTURE_MODE =
SIMULTANEOUS_SINGLE_FORWARD

FUTURE_MODEL_INPUT_ROWS =
600

FUTURE_EXPECTED_BACKBONE_FORWARD_COUNT =
600

FUTURE_LAYERS_CAPTURED_PER_FORWARD =
2

SEQUENTIAL_1200_FORWARD_LAYER_SPLIT =
PROHIBITED

The implementation captures layers 5 and 17 from the same future backbone
forward for each selected row.


## 11. Frozen scientific population

STRUCTURAL_ESTIMAND =
DELTA_NAME_ONLY

Q1_Q3_CELL_SET =
C0_SHAM
C2_NAME

SOURCE_PAIR_COUNT =
300

MODEL_INPUT_ROW_COUNT =
600

SEMANTIC_ANCHOR =
A_NAME

OUTCOME_DEPENDENT_ROW_SELECTION =
PROHIBITED


## 12. Support-state and endpoint contract

SUPPORT_STATE_WINDOW =
[a-1,a+4]

SUPPORT_STATES_PER_ROW_PER_LAYER =
6

FUTURE_SUPPORT_STATE_ROW_COUNT =
7200

FUTURE_ENDPOINT_ROW_COUNT =
1200

FROZEN_ENDPOINTS =
POST4_SPEED
POST4_TURNING
POST4_PATH_EFFICIENCY

MIXED_LAYER_ENDPOINT =
PROHIBITED

Every endpoint must use support states from one and only one layer.


## 13. Future output contract

A later authorized execution may produce exactly:

manifest.json

support_state_rows.jsonl

support_states.npy

kinematic_endpoints.jsonl

SHA256SUMS.txt

CURRENT_CANONICAL_Q1_Q3_OUTPUT =
NONE

No Q1/Q3 scientific extraction artifact was produced during implementation.


## 14. Manifest provenance contract

The future manifest must fail closed on required provenance and cardinalities.

It binds at minimum:

- implementation authority commit;
- implementation commit;
- measurement implementation SHA256;
- extraction implementation SHA256;
- frozen primary measurement SHA256;
- structural artifact SHA256;
- event manifest SHA256;
- R2 encoded-coordinate SHA256;
- tokenizer revision and tokenizer file identities;
- runtime versions;
- Mamba and cache source identities;
- representative checkpoint SHA256;
- exact layer set {5,17};
- semantic anchor A_NAME;
- source-pair count 300;
- model-input row count 600;
- backbone forward count 600;
- support-state row count 7200;
- endpoint row count 1200;
- peer-artifact SHA256 values.

MANIFEST_SELF_HASH_DEPENDENCY =
PROHIBITED


## 15. Numerical validation correction

One synthetic turning test initially required mathematical zero at default
pytest tolerance.

Observed frozen float32 computation produced:

5.960464477539063e-08

The test was corrected to:

ABSOLUTE_TOLERANCE =
1e-6

KINEMATIC_FORMULA_CHANGED =
NO

SCIENTIFIC_ENDPOINT_DEFINITION_CHANGED =
NO

The corrected dedicated suite then passed:

16_OF_16


## 16. Scientific execution boundary

No canonical tokenizer was executed during implementation validation.

CANONICAL_TOKENIZER_EXECUTION =
NO

SCIENTIFIC_CHECKPOINT_LOAD =
NO

CANONICAL_MODEL_FORWARD =
NO

Q1_Q3_SCIENTIFIC_NATIVE_STATE_EXTRACTION =
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


## 17. Statistical and inferential boundary

SECONDARY_HYPOTHESIS_COUNT =
6

STATISTICAL_TEST_IMPLEMENTED_IN_EXTRACTION_RUNNER =
NO

DEPTH_SELECTIVITY =
NOT_ESTABLISHED

DIRECT_CROSS_LAYER_DIFFERENCE_TEST =
NOT_PERFORMED

OVERALL_ADAPTIVE_PROGRAM_FWER =
NOT_CLAIMED

This implementation freeze authorizes no scientific interpretation.


## 18. Implementation validation conclusion

CODE_CORRECTNESS =
PASS

IMPLEMENTATION_SCOPE =
PASS_EXACTLY_FOUR_NEW_FILES

PRIMARY_PHASE_D_PRESERVATION =
PASS

FROZEN_COORDINATE_PROVENANCE =
PASS

EVENT_COORDINATE_BINDING =
PASS

DUAL_LAYER_POLICY =
PASS

ARTIFACT_CONTRACT =
PASS

SCIENTIFIC_EXECUTION =
NOT_PERFORMED

SCIENTIFIC_CONCLUSION =
NONE

IMPLEMENTATION_STATUS =
FROZEN_READY_FOR_SEPARATE_EXECUTION_AUTHORITY


## 19. Next phase

NEXT_PHASE =
NAME_Q1_Q3_SCIENTIFIC_EXTRACTION_EXECUTION_AUTHORITY

NEXT_EXECUTION =
NOT_YET_AUTHORIZED

A later execution authority must independently bind the exact implementation
commit and exact implementation SHA256 values frozen here before any canonical
tokenizer reconstruction, scientific checkpoint load, model forward, or native
state extraction.
