# ContraMamba Gen4 Six-Cell Tier-2 Dedicated Inference Adapter Implementation Validation Report - Candidate

## 1. Authority

R3_IMPLEMENTATION_AUTHORITY_COMMIT =
b5bd5491ee0c77d2b407f07b68ccea398ab65da8

R3_IMPLEMENTATION_AUTHORITY_SPEC_SHA256 =
571b321e84b3c048ce45f82a9a851c4f04626f1a4bee018de054724fd09c26b1

R3_IMPLEMENTATION_COMMIT =
d62a424c8d13e582ffde7e8d2f8b7e0f43b610b2

HISTORICAL_SOURCE_COMMIT =
3e0e9a435068c552abf20f3a74e0c3eccca344a3

VALIDATION_MODE =
POST_IMPLEMENTATION_STATIC_CODE_AND_PROVENANCE_VALIDATION

This report validates the frozen R3 dedicated inference-adapter implementation
only. It does not authorize or report checkpoint deserialization, model
instantiation, model forward execution, scientific inference, training, or
statistical testing.

## 2. Result classification

CODE_CORRECTNESS =
PASS_STATIC_VALIDATION

STATIC_TEST_EXECUTION =
PASS

IMPLEMENTATION_PROVENANCE_VALIDITY =
PASS

CHECKPOINT_LOADABILITY =
NOT_ESTABLISHED

MODEL_EXECUTION_SUCCESS =
NOT_EVALUATED

SCIENTIFIC_OUTCOME =
NOT_EVALUATED

ARTIFACT_OUTCOME_VALIDITY =
NOT_EVALUATED

The R3 result is therefore a code-correctness and implementation-provenance
result, not scientific evidence.

## 3. Exact implementation scope

AUTHORIZED_IMPLEMENTATION_FILE_COUNT =
3

IMPLEMENTATION_FILES =
1. src/contramamba/modeling_v6b_minimal_gen3_grouped_snapshot.py
2. scripts/reason_router_gen4_six_cell_tier2_inference_adapter.py
3. tests/test_reason_router_gen4_six_cell_tier2_inference_adapter.py

EXISTING_TRACKED_FILES_MODIFIED_BY_R3_IMPLEMENTATION =
NO

R3_IMPLEMENTATION_COMMIT_SCOPE =
EXACT_THREE_FILES

No fourth implementation file was introduced.

## 4. Exact frozen file identities

HISTORICAL_MODEL_SNAPSHOT_SHA256 =
8c365bfa857157d91f363358d5db3abaab425dec3e0d7c62683b4207a589b6a5

DEDICATED_INFERENCE_ADAPTER_SHA256 =
83177c351f82a781586c63bd8d4ef1b40e759b5a94858dc1837d65502cbff6e5

STATIC_TEST_SHA256 =
5e00d9204f7e80be4de5957b1568c6a95d96693b5c3c47db1f3b043028cd9d64

HISTORICAL_MODEL_SNAPSHOT_SOURCE =
3e0e9a435068c552abf20f3a74e0c3eccca344a3:src/contramamba/modeling_v6b_minimal.py

HISTORICAL_MODEL_SNAPSHOT_SOURCE_DELTA =
ZERO_BYTES

HISTORICAL_MODEL_SNAPSHOT_BYTE_IDENTITY =
PASS

## 5. Historical dependency binding

HISTORICAL_HEADS_TREE =
68d26855aa511fcd41d6f395ae5f87177a162678

CURRENT_HEADS_REUSED =
YES

CURRENT_HEADS_HISTORICAL_TREE_IDENTITY =
PASS

CURRENT_MODEL_MODULE_SUBSTITUTION =
FORBIDDEN

HISTORICAL_SNAPSHOT_MODULE_BINDING =
PASS

The adapter binds the exact historical model snapshot rather than substituting
the current src/contramamba/modeling_v6b_minimal.py implementation.

## 6. Historical grouped model construction contract

FRAME_SIZE =
128

PREDICATE_SIZE =
128

SUFFICIENCY_SIZE =
128

ENERGY_SIZE =
64

DROPOUT =
0.1

FREEZE_ENCODER =
true

FREEZE_A_LOG =
true

DECISION_MODE =
explicit_product

REASON_ROUTER_EPSILON =
1e-8

GRADIENT_OWNERSHIP_MODE =
edge_specific

GLOBAL_GRADIENT_OWNERSHIP_LAMBDA =
null

USE_TEMPORAL_COMPARATOR =
false

USE_PREDICATE_COMPARATOR =
false

ALPHA_TEMPORAL_INIT =
1.25

ALPHA_PREDICATE_INIT =
1.25

HISTORICAL_CONSTRUCTION_CONTRACT_BINDING =
PASS

The initial under-bound builder candidate was not frozen. Before the R3
implementation freeze, the historical production construction contract was
recovered from frozen source/provenance and bound explicitly.

## 7. Frozen six-arm resolved edge-gradient registry

EDGE_MAP_SEED_INVARIANCE =
PASS_6_OF_6_ARMS

G3-GROUP-D-HALF =
F_TO_P:1.0,F_TO_S:1.0,P_TO_S:1.0,F_TO_Q:1.0,P_TO_Q:1.0,S_TO_Q:1.0,F_TO_D:0.5,P_TO_D:0.5,S_TO_D:0.5,Q_TO_D:0.5

G3-GROUP-Q-D-HALF =
F_TO_P:1.0,F_TO_S:1.0,P_TO_S:1.0,F_TO_Q:0.5,P_TO_Q:0.5,S_TO_Q:0.5,F_TO_D:0.5,P_TO_D:0.5,S_TO_D:0.5,Q_TO_D:0.5

G3-GROUP-Q-HALF =
F_TO_P:1.0,F_TO_S:1.0,P_TO_S:1.0,F_TO_Q:0.5,P_TO_Q:0.5,S_TO_Q:0.5,F_TO_D:1.0,P_TO_D:1.0,S_TO_D:1.0,Q_TO_D:1.0

G3-GROUP-U-D-HALF =
F_TO_P:0.5,F_TO_S:0.5,P_TO_S:0.5,F_TO_Q:1.0,P_TO_Q:1.0,S_TO_Q:1.0,F_TO_D:0.5,P_TO_D:0.5,S_TO_D:0.5,Q_TO_D:0.5

G3-GROUP-U-HALF =
F_TO_P:0.5,F_TO_S:0.5,P_TO_S:0.5,F_TO_Q:1.0,P_TO_Q:1.0,S_TO_Q:1.0,F_TO_D:1.0,P_TO_D:1.0,S_TO_D:1.0,Q_TO_D:1.0

G3-GROUP-U-Q-HALF =
F_TO_P:0.5,F_TO_S:0.5,P_TO_S:0.5,F_TO_Q:0.5,P_TO_Q:0.5,S_TO_Q:0.5,F_TO_D:1.0,P_TO_D:1.0,S_TO_D:1.0,Q_TO_D:1.0

ARBITRARY_RUNTIME_EDGE_MAP_INPUT =
REMOVED

The future forward adapter resolves the edge map from the frozen evaluator arm
rather than accepting an arbitrary caller-supplied map.

## 8. Frozen evaluator registry

EVALUATOR_SEEDS =
180,181,182

EVALUATOR_ARM_COUNT =
6

FROZEN_EVALUATOR_COUNT =
18

CHECKPOINT_REGISTRY_COVERAGE =
PASS_18_OF_18

CHECKPOINT_AUTHENTICATION_POLICY =
SHA256_BEFORE_DESERIALIZATION

UNKNOWN_SEED_OR_ARM_POLICY =
FAIL_CLOSED

STRICT_STATE_DICT_POLICY =
REQUIRED

STRICT_FALSE_LOADING =
FORBIDDEN

STATE_DICT_FILTERING_OR_KEY_REWRITE =
FORBIDDEN

R3 did not deserialize any real checkpoint.

## 9. Label-free Gen4 input contract

CANONICAL_GEN4_ARTIFACT_SHA256 =
b9c54604863ed15c237fa17c7890a20f3f5ec062a7429638b39e8b468be050a7

CANONICAL_GEN4_ROW_COUNT =
1800

CANONICAL_GEN4_SOURCE_PAIR_COUNT =
300

MAX_LENGTH =
128

CLAIM_BUDGET =
63

SEPARATOR_TOKEN_COUNT =
1

EVIDENCE_BUDGET =
64

EOS_TOKEN_ID =
0

EFFECTIVE_PAD_TOKEN_ID =
0

LABEL_DEPENDENCY =
ZERO

REQUIRED_ROW_IDENTITIES =
row_id,source_pair_id,contrast_cell_id

DUPLICATE_ROW_ID_POLICY =
FAIL_CLOSED

The R3 adapter preserves the R1/R2 recovered 63/1/64 label-free input
construction contract.

## 10. Canonical tokenizer binding

CANONICAL_TOKENIZER_FAMILY =
A

CANONICAL_TOKENIZER_REVISION_REFERENCE =
40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37

TOKENIZER_JSON_SHA256 =
b074ad869d4f45d1265ca5c9814f78604f3d7e187acc063b15dd232b27585fcf

TOKENIZER_CONFIG_JSON_SHA256 =
9d7016c33747c6309346e59bd7bf63bfc33c9d9366ecb7e514b3b84dc6b46acb

SPECIAL_TOKENS_MAP_JSON_SHA256 =
57491904f8680d4b52ed440f1f7ba48cad1c31ecf3eb453b03484e6ff4723ae8

HISTORICAL_TRANSFORMERS_WRAPPER_RUNTIME_EQUIVALENCE =
UNRESOLVED

The unresolved historical wrapper-runtime equivalence from R2 remains
unresolved. R3 does not silently upgrade that result.

## 11. Dedicated outcome serializer contract

PRIMARY_OUTCOME =
q_authorized

SECONDARY_OUTCOMES =
entitlement_prob,support_vs_best_nonsupport_logit_margin

FINAL_EXTERNAL_CLASS_ORDER =
REFUTE,NOT_ENTITLED,SUPPORT

REQUIRED_LOGIT_FIELDS =
refute_logit,ne_logit,support_logit

PREDICTION_SOURCE =
ARGMAX_FINAL_LOGITS

SUPPORT_MARGIN_DEFINITION =
support_logit - max(refute_logit,ne_logit)

HISTORICAL_PREDICTION_CONSISTENCY_CHECK =
REQUIRED

## 12. Future output matrix contract

FUTURE_OUTPUT_PRIMARY_KEY =
(seed,arm,row_id)

EXPECTED_SCIENTIFIC_OUTPUT_ROWS =
32400

EXPECTED_EVALUATORS_PER_ROW =
18

EXPECTED_ROWS_PER_SOURCE_PAIR =
108

COMPLETE_MATRIX_POLICY =
FAIL_CLOSED

No 32400-row scientific outcome artifact was generated in R3.

## 13. Static-test validation

R3_STATIC_TEST_COUNT =
23

R3_STATIC_TEST_RESULT =
PASS_23_OF_23

STATIC_TEST_COVERAGE_INCLUDES =
- historical snapshot exact SHA256
- historical/current heads tree identity
- exact 18-evaluator registry
- unknown seed/arm rejection
- checkpoint SHA rejection before loader callback
- label-free input acceptance
- required identity-field rejection
- duplicate row_id rejection
- 63/1/64 feature construction
- separator and padding masks
- fixed external class order
- synthetic historical-output serialization
- prediction/logit consistency
- complete-matrix fail-closed behavior
- deterministic provenance construction
- explicit CLI-operation requirement
- historical snapshot module binding
- no model/checkpoint execution from R3 tests
- exact historical constructor contract
- exact six-arm resolved edge-gradient registry
- mandatory arm binding for future builder/forward

STATIC_SOURCE_VALIDATION =
PASS

STAGED_DIFF_CHECK_BEFORE_IMPLEMENTATION_FREEZE =
PASS

## 14. R3 execution boundary

MODEL_INSTANTIATION =
NOT_PERFORMED

CHECKPOINT_DESERIALIZATION =
NOT_PERFORMED

CHECKPOINT_LOADABILITY_TEST =
NOT_PERFORMED

MODEL_FORWARD =
NOT_PERFORMED

SCIENTIFIC_INFERENCE =
NOT_PERFORMED

TRAINING =
NOT_PERFORMED

STATISTICAL_TESTING =
NOT_PERFORMED

KAGGLE =
NOT_USED

NETWORK_MODEL_DOWNLOAD =
NOT_PERFORMED

This boundary is material: R3 PASS establishes neither checkpoint loadability
nor scientific model behavior.

## 15. What R3 establishes

R3 establishes all of the following:

1. an exact historical model snapshot is frozen;
2. the dedicated adapter is label-free;
3. historical input construction and model construction are explicitly bound;
4. the six frozen grouped arms have a deterministic resolved edge registry;
5. all 18 frozen evaluator checkpoints have exact expected SHA256 identities;
6. future checkpoint authentication fails before deserialization on mismatch;
7. future state-dict loading is required to be strict;
8. required Gen4 scientific outcome fields and complete-matrix validation are
   implemented;
9. static/unit validation passes without loading or executing a model.

## 16. What R3 does not establish

R3 does not establish:

- that any checkpoint can actually be deserialized in the current runtime;
- that strict state-dict loading succeeds;
- that the historical model can be instantiated successfully;
- that Transformers/Mamba runtime compatibility is valid;
- that a model forward pass succeeds;
- that any Gen4 outcome value is scientifically valid;
- that the 18-evaluator matrix can be produced;
- that any contrast is significant;
- that any scientific hypothesis is supported or rejected.

## 17. Phase transition

PHASE_R3_IMPLEMENTATION =
PASS_PENDING_RESULT_FREEZE

R3_CODE_CORRECTNESS =
PASS

R3_IMPLEMENTATION_PROVENANCE =
PASS

CHECKPOINT_LOADABILITY =
NOT_YET_ESTABLISHED

MODEL_INFERENCE =
NOT_AUTHORIZED

STATISTICAL_TESTING =
NOT_AUTHORIZED

TRAINING =
NOT_AUTHORIZED

KAGGLE =
NOT_AUTHORIZED

After this validation report is independently reviewed and frozen, the next
recovery object may be an R4 checkpoint-loadability execution authority.

NEXT_OBJECT_AFTER_R3_RESULT_FREEZE =
GEN4_SIX_CELL_TIER2_CHECKPOINT_LOADABILITY_EXECUTION_AUTHORITY

## 18. Result

R3_IMPLEMENTATION_VALIDATION_RESULT =
READY_FOR_FREEZE_REVIEW
