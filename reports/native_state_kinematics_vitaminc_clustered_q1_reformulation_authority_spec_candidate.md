# Native State Kinematics Q1
# VitaminC Clustered-Cohort Reformulation Authority Specification

STATUS = AUTHORIZED_FOR_STATIC_REFORMULATION_FEASIBILITY_ONLY

BASE_REPOSITORY_HEAD =
150b7afff45c34451318e3424b224a8b581856b3

PARENT_DESIGN_FILE =
reports/native_state_kinematics_first_confident_error_design_authority_spec_candidate.md

REFORMULATION_REASON =
The frozen A0 720-row clean-dev population produced only one
post4-eligible confident-wrong primary candidate, so the original
confidence-matched confirmatory cohort was infeasible.

SCIENTIFIC_TARGET_RETAINED =
prefix-only native-state kinematic difference between
confidence-matched correct and wrong decisive factual commitments

FUTURE_TARGET_MODEL_RETAINED =
state-spaces/mamba-130m-hf

FUTURE_A0_ARCHITECTURE =
v6b_minimal

FUTURE_A0_TRAINING_SEED =
180

FUTURE_A0_SPLIT_SEED =
8192

FUTURE_A0_SELECTED_CHECKPOINT_SHA256 =
4f7ad019bddb988a534c477b58b36bdabe2775d6c9748331e8311653c07c864c

CANDIDATE_DATASET =
tals/vitaminc validation first 5000 rows

CANDIDATE_DATASET_SHA256 =
452a24ec32302b4db6100c0f5897533d726f3d4408662b26bc3089b0b46d3191

CANDIDATE_DATASET_FINGERPRINT =
925a91578795c751

INDEPENDENT_SAMPLING_UNIT =
case_id

ROW_LEVEL_PSEUDOREPLICATION =
PROHIBITED

MAX_PRIMARY_WRONG_PER_CASE_ID =
1

MAX_PRIMARY_CONTROL_PER_CASE_ID =
1

WRONG_CONTROL_CASE_ID_OVERLAP =
PROHIBITED

EPistemicBERT_ROLE =
EXTERNAL_DIFFICULTY_AND_FEASIBILITY_DIAGNOSTIC_ONLY

EPistemicBERT_PREDICTION_OR_CONFIDENCE_AS_MAMBA_COHORT_LABEL =
PROHIBITED

DISTILBERT_TOKEN_COORDINATE_AS_MAMBA_EVENT_TIME =
PROHIBITED

STATIC_CONTROLLER_AUDIT_OBSERVATIONS_TO_REPRODUCE =
5000_rows
1497_unique_case_ids
1010_complete_2claim_x_2evidence_case_ids
681_complete_2x2_cases_with_at_least_4_raw_whitespace_tokens_after_edit_span
128_historical_decisive_error_rows
93_unique_raw_error_rows
78_unique_error_case_ids

EXISTING_NATIVE_Q1_PAIRS_128_STATUS =
DIAGNOSTIC_ONLY_NOT_CONFIRMATORY

EXISTING_PAIR_DEFECTS_TO_REPRODUCE =
same_predicted_class_57_of_128
wrong_control_case_id_overlap_4

CANDIDATE_EVENT_CONSTRUCTION =
For a complete 2-claim x 2-evidence case_id, compare the two frozen raw
evidence variants without using any model prediction, confidence, correctness,
native state, or downstream representation.

Derive the raw evidence-edit span deterministically from the common
prefix/suffix structure of the paired evidence revisions.

This raw span is only a candidate event span.

A later stage must map the frozen raw span into the exact Mamba tokenizer
coordinate and verify that the event is consumed and that tau_e + 4 remains
strictly prefix-only.

PRIMARY_NATIVE_ENDPOINTS_RETAINED =
POST4_SPEED
POST4_TURNING
POST4_PATH_EFFICIENCY

PRIMARY_LAYER_RULE_RETAINED =
architecture midpoint only

MODEL_FORWARD_ALLOWED =
NO

CHECKPOINT_INFERENCE_ALLOWED =
NO

NATIVE_STATE_ACCESS_ALLOWED =
NO

TRAINING_ALLOWED =
NO

CUDA_ALLOWED =
NO

KAGGLE_ALLOWED =
NO

IMPLEMENTATION_ALLOWED =
YES_STATIC_ONLY

AUTHORIZED_IMPLEMENTATION_FILES =
scripts/audit_native_state_kinematics_vitaminc_cluster_feasibility.py
tests/test_native_state_kinematics_vitaminc_cluster_feasibility.py

STATIC_IMPLEMENTATION_MUST_VERIFY =
dataset SHA256 and schema
case_id cluster cardinality
complete 2x2 claim/evidence structure
candidate evidence-edit spans
post-edit lexical suffix capacity
historical 128-row duplicate structure
existing wrong/control case leakage
no model or native-state access

NEXT_STAGE_ON_STATIC_PASS =
MAMBA_PREDICTION_ONLY_COHORT_CENSUS_PREPARATION

The next stage may evaluate only the frozen A0 Mamba model on the frozen
VitaminC sampling frame to establish Mamba-native correctness, predicted
class, confidence, cluster-independent sample capacity, and matching
feasibility before any native-state trajectory is inspected.

END_OF_NATIVE_STATE_KINEMATICS_VITAMINC_CLUSTERED_Q1_REFORMULATION_AUTHORITY