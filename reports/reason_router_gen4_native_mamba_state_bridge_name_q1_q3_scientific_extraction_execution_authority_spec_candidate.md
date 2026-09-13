# ContraMamba Gen4 Native Mamba State Bridge
# NAME Q1/Q3 Scientific Extraction
# Execution Authority
# Candidate

STATUS =
CANDIDATE

AUTHORITY_ID =
GEN4_NATIVE_MAMBA_STATE_BRIDGE_NAME_Q1_Q3_SCIENTIFIC_EXTRACTION_EXECUTION_AUTHORITY

PHASE =
NAME_Q1_Q3_SCIENTIFIC_EXTRACTION

SCIENTIFIC_EXECUTION_ALLOWED =
YES_BOUNDED

CANONICAL_TOKENIZER_EXECUTION_ALLOWED =
YES_BOUNDED

CHECKPOINT_LOADING_ALLOWED =
YES_BOUNDED

MODEL_FORWARD_ALLOWED =
YES_BOUNDED

NATIVE_STATE_EXTRACTION_ALLOWED =
YES_BOUNDED

STATISTICAL_TESTING_ALLOWED =
NO

TRAINING_ALLOWED =
NO

BACKWARD_ALLOWED =
NO

GRADIENT_UPDATE_ALLOWED =
NO

KAGGLE_ALLOWED =
YES_BOUNDED

GPU_ALLOWED =
NO

DEVICE =
CPU_ONLY


## 1. Frozen implementation chain

EXECUTION_DRIVER_VALIDATION_FREEZE =
c1c13095bab991642be7e9851b483661d1c91e5d

Q1_Q3_EXECUTION_DRIVER_IMPLEMENTATION_COMMIT =
131f304901719547e2dca046909861a65e34ac59

Q1_Q3_EXECUTION_DRIVER_SHA256 =
1d932c22779cf55cbe81a76250d61ea62ff654077aec585cf85b94d2d2516a06

Q1_Q3_EXECUTION_DRIVER_TEST_SHA256 =
5f0de450acde5eb4857fee5d8ad1bb93cb44dc66a6a258c8d1a3d9adb61c29f1

Q1_Q3_PRIMITIVE_IMPLEMENTATION_COMMIT =
d5317b2c5be09464c9196325429479c6bff25efd

Q1_Q3_MEASUREMENT_SHA256 =
2236d19a46416e5042057ec84c565dbe523ef791f7266c30b26b033d260a510b

Q1_Q3_EXTRACTION_SHA256 =
9e9ca05aaee43970c8d0aa101c7f0c0b86ace9166ae798f70c800fcb61651c4f


## 2. Scientific scope

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

LAYER_SET =
{5,17}

CAPTURE_MODE =
SIMULTANEOUS_SINGLE_FORWARD

EXPECTED_BACKBONE_FORWARD_COUNT =
600

LAYERS_CAPTURED_PER_FORWARD =
2

SEQUENTIAL_1200_FORWARD_LAYER_SPLIT =
PROHIBITED


## 3. Frozen input-coordinate gate

FULL_STRUCTURAL_ROW_COUNT =
1800

R2_ENCODED_COORDINATE_SHA256 =
d3cf61e55bb04e6dbebe44be433597cce958cb15989412dcf6efc15a329c576a

The complete frozen coordinate must be authenticated before scientific subset
selection or model construction.

The coordinate identity covers exactly:

- row_id;
- source_pair_id;
- contrast_cell_id;
- input_ids;
- attention_mask;
- claim_mask;
- evidence_mask.

INPUT_IDS_ONLY_HASH =
PROHIBITED

SUBSET_BEFORE_FULL_COORDINATE_GATE =
PROHIBITED

MODEL_LOAD_BEFORE_FULL_COORDINATE_GATE =
PROHIBITED

MODEL_FORWARD_BEFORE_FULL_COORDINATE_GATE =
PROHIBITED

ALTERNATE_RETOKENIZATION =
PROHIBITED


## 4. Frozen tokenizer

TOKENIZER_REVISION =
40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37

TOKENIZER_JSON_SHA256 =
b074ad869d4f45d1265ca5c9814f78604f3d7e187acc063b15dd232b27585fcf

TOKENIZER_CONFIG_SHA256 =
9d7016c33747c6309346e59bd7bf63bfc33c9d9366ecb7e514b3b84dc6b46acb

SPECIAL_TOKENS_MAP_SHA256 =
57491904f8680d4b52ed440f1f7ba48cad1c31ecf3eb453b03484e6ff4723ae8

TOKEN_CONSTRUCTION =
63_CLAIM_1_EOS_64_EVIDENCE_MAX

MAX_SEQUENCE_LENGTH =
128

ADD_SPECIAL_TOKENS =
FALSE


## 5. Frozen event coordinate

EVENT_MANIFEST_SHA256 =
70c84c68b36751bb7c7145b33ccb71ab91bc8ee9e6cc5f2c7a0d4e925f36581f

EVENT_MANIFEST_BYTES =
2268260

ANCHOR_NAME =
A_NAME

ANCHOR_INDEX_FIELD =
absolute_anchor_token_index

SUPPORT_WINDOW =
[a-1,a+4]


## 6. Representative model

REPRESENTATIVE_SEED =
180

REPRESENTATIVE_ARM =
G3-GROUP-D-HALF

REPRESENTATIVE_CHECKPOINT_SHA256 =
1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f

MODEL_REPLICATION_COUNT =
1

DEVICE =
CPU_ONLY

TRAINING =
FALSE

BACKWARD =
FALSE

STATISTICAL_TESTING =
FALSE


## 7. Runtime contract

EXPECTED_PYTHON_VERSION =
3.12.13

EXPECTED_NUMPY_VERSION =
2.0.2

EXPECTED_TORCH_VERSION =
2.10.0+cpu

EXPECTED_TRANSFORMERS_VERSION =
5.0.0

Any runtime mismatch must block before scientific forward execution.


## 8. Measurement contract

EXPECTED_SUPPORT_STATE_ROWS =
7200

EXPECTED_ENDPOINT_ROWS =
1200

SUPPORT_VECTOR_SIZE =
24576

ENDPOINTS =
POST4_SPEED
POST4_TURNING
POST4_PATH_EFFICIENCY

MIXED_LAYER_ENDPOINT =
PROHIBITED

DEPTH_SELECTIVITY =
NOT_ESTABLISHED

DIRECT_CROSS_LAYER_DIFFERENCE_TEST =
NOT_PERFORMED


## 9. Exact output contract

OUTPUT_DIRECTORY =
reports/reason_router_gen4_native_mamba_state_bridge_name_q1_q3_scientific_extraction_v1

The execution may create exactly:

manifest.json

support_state_rows.jsonl

support_states.npy

kinematic_endpoints.jsonl

SHA256SUMS.txt

OUTPUT_COLLISION =
BLOCK

STAGING_COLLISION =
BLOCK

PARTIAL_PUBLISH_ON_FAILURE =
PROHIBITED


## 10. Execution binding

The scientific execution must occur at the exact commit containing this
authority document.

The runtime command must bind:

EXPECTED_EXECUTION_HEAD =
THE_EXECUTION_AUTHORITY_COMMIT

EXECUTION_AUTHORITY_COMMIT =
THE_EXECUTION_AUTHORITY_COMMIT

Q1_Q3_EXECUTION_DRIVER_IMPLEMENTATION_COMMIT =
131f304901719547e2dca046909861a65e34ac59

Q1_Q3_EXECUTION_DRIVER_SHA256 =
1d932c22779cf55cbe81a76250d61ea62ff654077aec585cf85b94d2d2516a06

The execution driver must fail closed if:

- HEAD differs;
- authority commit differs from HEAD;
- driver implementation is not ancestral;
- driver SHA differs;
- frozen dependencies differ;
- repository is dirty.


## 11. Kaggle policy

RUN_NAME =
gen4-name-q1q3-scientific-extraction-v1

KAGGLE =
AUTHORIZED_BOUNDED

GPU =
OFF

The GPU must remain disabled for the complete execution.

No training or evaluation job is authorized.

The only authorized scientific work is the frozen CPU-only NAME Q1/Q3 native
state extraction described by this document.


## 12. Scientific interpretation boundary

This authority permits measurement artifact generation only.

It does not authorize:

- the six-hypothesis statistical family;
- Holm correction;
- cross-layer comparison;
- depth-selectivity inference;
- adaptive-program FWER claims;
- mediation;
- causal interpretation.

SCIENTIFIC_CONCLUSION_FROM_EXECUTION_ALONE =
NONE

A successful extraction run establishes execution success and candidate
measurement artifacts only.

Artifact/provenance validation must occur before statistical interpretation.


## 13. Completion criterion

A successful authorized run must establish:

EXECUTION_SUCCESS =
PASS

BACKBONE_FORWARD_COUNT =
600

SUPPORT_STATE_ROW_COUNT =
7200

ENDPOINT_ROW_COUNT =
1200

OUTPUT_ARTIFACT_SET =
EXACT_FIVE

TRAINING =
FALSE

BACKWARD =
FALSE

STATISTICAL_TESTING =
FALSE

The resulting bundle must then be collected and imported under exact
commit/hash provenance before any statistical authority is created.


## 14. Next phase

NEXT_PHASE_AFTER_VALIDATED_IMPORT =
NAME_Q1_Q3_SIX_HYPOTHESIS_STATISTICAL_EXECUTION_AUTHORITY

That later phase remains unauthorized by this document.
