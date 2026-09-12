# ContraMamba Gen4 Six-Cell Tier-2 Checkpoint Loadability Validation Report - Candidate

## 1. Authority

R4_EXECUTION_AUTHORITY_COMMIT =
83f32cbb8bfbed7b2b88d0cf864499422e650954

R4_EXECUTION_AUTHORITY_SHA256 =
5ab6a068dbd86865de7df2dfafc0870b3d95d6cce0c8d6210ece6bf9732e9c14

R3_RESULT_FREEZE_COMMIT =
64cc172ae1446e8ba9200d2365a08aad6f5d87ee

R3_IMPLEMENTATION_COMMIT =
d62a424c8d13e582ffde7e8d2f8b7e0f43b610b2

HISTORICAL_EVALUATOR_SOURCE_COMMIT =
3e0e9a435068c552abf20f3a74e0c3eccca344a3

VALIDATION_MODE =
POST_EXECUTION_ARTIFACT_AND_PROVENANCE_VALIDATION

## 2. Result classification

CODE_CORRECTNESS =
UNCHANGED_FROM_R3_PASS

EXECUTION_SUCCESS =
PASS_CHECKPOINT_LOADABILITY_ONLY

ARTIFACT_PROVENANCE_VALIDITY =
PASS

CHECKPOINT_LOADABILITY =
PASS_18_OF_18

SCIENTIFIC_MODEL_OUTCOME =
NOT_EVALUATED

GEN4_SCIENTIFIC_OUTCOME =
NOT_EVALUATED

SCIENTIFIC_CONCLUSION =
NOT_EVALUATED

R4 is infrastructure evidence only.

## 3. Exact result artifact

R4_SUMMARY_PATH =
reports/reason_router_gen4_six_cell_tier2_checkpoint_loadability_83f32cbb8bfbed7b2b88d0cf864499422e650954/r4_checkpoint_loadability_summary.json

R4_SUMMARY_SHA256 =
ecd9e939e8103fbb80bd7c77114e35e53ed6352a514405676572bae8408632bd

R4_SUMMARY_BYTES =
14036

R4_SUMMARY_SCHEMA =
gen4_r4_checkpoint_loadability_summary_v1

R4_SUMMARY_ROW_COUNT =
18

DETERMINISTIC_SERIALIZATION =
PASS

## 4. Runtime

PYTHON_VERSION =
3.13.2

TORCH_VERSION =
2.12.1+cpu

TRANSFORMERS_VERSION =
5.12.1

DEVICE =
cpu

MAP_LOCATION =
cpu

WEIGHTS_ONLY =
true

STRICT_STATE_DICT =
true

GPU_USED =
NO

KAGGLE_USED =
NO

## 5. Model/config provenance

HISTORICAL_MODEL_SNAPSHOT_SHA256 =
8c365bfa857157d91f363358d5db3abaab425dec3e0d7c62683b4207a589b6a5

DEDICATED_INFERENCE_ADAPTER_SHA256 =
83177c351f82a781586c63bd8d4ef1b40e759b5a94858dc1837d65502cbff6e5

MODEL_CONFIG_REVISION_REFERENCE =
40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37

MODEL_CONFIG_SHA256 =
784825b6b6cdde47a1602278db0e66d6764169af4a8e662404701bc636a2686a

MODEL_CONFIG_CONTENT_AUTHENTICATION =
PASS

## 6. Evaluator population

EXPECTED_EVALUATOR_COUNT =
18

OBSERVED_EVALUATOR_COUNT =
18

EVALUATOR_POPULATION =
PASS_EXACT_18

CHECKPOINT_SHA_REGISTRY_BINDING =
PASS_18_OF_18

CHECKPOINT_BYTE_COUNT =
518270455

CHECKPOINT_BYTE_COUNT_BINDING =
PASS_18_OF_18

RUN_PROVENANCE_BINDING =
PASS_18_OF_18

SELECTED_EPOCH_BINDING =
PASS_18_OF_18

No evaluator was dropped, replaced, reselected, or reweighted.

## 7. Deserialization

CHECKPOINT_SHA_AUTHENTICATED_BEFORE_DESERIALIZATION =
PASS_18_OF_18

DESERIALIZATION_MODE =
torch.load(map_location="cpu",weights_only=True)

WEIGHTS_ONLY_DESERIALIZATION =
PASS_18_OF_18

PAYLOAD_SCHEMA =
stage176a0_selected_checkpoint_v1

PAYLOAD_SCHEMA_VALIDATION =
PASS_18_OF_18

## 8. Strict state-dict compatibility

CHECKPOINT_KEY_COUNT =
278

MODEL_KEY_COUNT =
278

MISSING_KEY_COUNT =
0_FOR_ALL_18

UNEXPECTED_KEY_COUNT =
0_FOR_ALL_18

SHAPE_MISMATCH_COUNT =
0_FOR_ALL_18

DTYPE_MISMATCH_COUNT =
0_FOR_ALL_18

NON_TENSOR_STATE_VALUE_COUNT =
0_FOR_ALL_18

STRICT_STATE_DICT_LOAD =
PASS_18_OF_18

STRICT_FALSE_USED =
NO

STATE_DICT_KEY_REWRITE_USED =
NO

STATE_DICT_FILTERING_USED =
NO

## 9. Runtime warning interpretation

TRANSFORMERS_FAST_PATH_WARNING_OBSERVED =
YES_ON_INITIAL_MODEL_CONSTRUCTION

WARNING_EFFECT_ON_R4_LOADABILITY =
NONE

The warning concerned availability of accelerated Mamba forward kernels.
R4 performed no model forward, so it does not alter the checkpoint-loadability
result. No fallback forward output was produced or evaluated.

## 10. Execution boundary

MODEL_FORWARD =
NOT_PERFORMED

GEN4_INPUT_ENCODING =
NOT_PERFORMED

TOKENIZER_EXECUTION =
NOT_PERFORMED

PREDICTION =
NOT_PERFORMED

Q_AUTHORIZED_EXTRACTION =
NOT_PERFORMED

LOGIT_EXTRACTION =
NOT_PERFORMED

TRAINING =
NOT_PERFORMED

BACKWARD =
NOT_PERFORMED

STATISTICAL_TESTING =
NOT_PERFORMED

SCIENTIFIC_INFERENCE =
NOT_PERFORMED

## 11. What R4 establishes

R4 establishes that the exact frozen 18-checkpoint evaluator population:

1. is locally present at the frozen paths;
2. matches frozen checkpoint SHA256 identities;
3. matches frozen checkpoint byte counts;
4. deserializes on CPU with weights_only=True;
5. matches the expected stage176a0 payload contract;
6. matches selected-epoch provenance;
7. has exact 278-key compatibility with the recovered historical model shell;
8. has zero missing, unexpected, shape-mismatch, and dtype-mismatch entries;
9. strict-loads successfully for all 18 evaluators.

R4_CHECKPOINT_LOADABILITY =
PASS_18_OF_18

## 12. What R4 does not establish

R4 does not establish:

- successful model forward behavior;
- numerical equivalence of current runtime forward kernels to the historical
  runtime;
- correctness of q_authorized values;
- correctness of entitlement probabilities or logits;
- validity of any Gen4 scientific outcome;
- validity of the 32400-row scientific inference matrix;
- any statistical contrast result;
- any scientific hypothesis conclusion.

## 13. Phase transition

PHASE_R4 =
PASS_PENDING_RESULT_FREEZE

CHECKPOINT_LOADABILITY =
PASS_18_OF_18

MODEL_INFERENCE =
NOT_AUTHORIZED

SCIENTIFIC_INFERENCE =
NOT_AUTHORIZED

STATISTICAL_TESTING =
NOT_AUTHORIZED

TRAINING =
NOT_AUTHORIZED

R4 PASS does not itself authorize R5.

After this summary and validation report are independently reviewed and frozen,
the next object may be a separate R5 scientific-inference execution authority.

NEXT_OBJECT_AFTER_R4_RESULT_FREEZE =
GEN4_SIX_CELL_TIER2_SCIENTIFIC_INFERENCE_EXECUTION_AUTHORITY

## 14. Result

R4_VALIDATION_RESULT =
READY_FOR_FREEZE_REVIEW
