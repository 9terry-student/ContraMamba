# Native State Kinematics Q1
# VitaminC Mamba Prediction-Only Cohort Census Authority

STATUS =
AUTHORIZED_FOR_BOUNDED_IMPLEMENTATION_AND_POST_FREEZE_PREDICTION_ONLY_EXECUTION

BASE_REPOSITORY_HEAD =
3223122cab2add464316c1c80fd77b0a7d6ce990

PARENT_REFORMULATION_AUTHORITY =
reports/native_state_kinematics_vitaminc_clustered_q1_reformulation_authority_spec_candidate.md

FROZEN_STATIC_FEASIBILITY_EVIDENCE =
reports/native_state_kinematics_vitaminc_cluster_feasibility_runs/native-q1-vitaminc-cluster-static-1c501df-r1/native_q1_vitaminc_cluster_feasibility_summary.json

FROZEN_STATIC_FEASIBILITY_EVIDENCE_SHA256 =
05cdec64e08bec5812f37d0d7a1174ec1a4c755da841e979f796626dc3e0296b

SCIENTIFIC_TARGET_RETAINED =
confidence-matched correct versus wrong decisive factual commitments
using the same frozen Mamba model that will later supply native recurrent states

PHASE =
MAMBA_PREDICTION_ONLY_COHORT_CENSUS

PURPOSE =
Determine whether the frozen A0 replacement-R1 Mamba model produces a
case_id-independent confident-correct and confident-wrong decisive population
large enough to justify later event-coordinate and native-state preparation.

THIS_PHASE_IS_NOT =
native-state scientific execution
matched-cohort scientific execution
tau_e execution
training
checkpoint selection
calibration
threshold search

# ---------------------------------------------------------------------------
# Frozen dataset
# ---------------------------------------------------------------------------

DATASET =
tals/vitaminc validation first 5000 rows

DATASET_SHA256 =
452a24ec32302b4db6100c0f5897533d726f3d4408662b26bc3089b0b46d3191

DATASET_FINGERPRINT =
925a91578795c751

EXPECTED_ROWS =
5000

EXPECTED_UNIQUE_CASE_IDS =
1497

INDEPENDENT_SAMPLING_UNIT =
case_id

MAX_PRIMARY_WRONG_PER_CASE_ID =
1

MAX_PRIMARY_CONTROL_PER_CASE_ID =
1

WRONG_CONTROL_CASE_ID_OVERLAP =
PROHIBITED

ROW_LEVEL_PSEUDOREPLICATION =
PROHIBITED

# ---------------------------------------------------------------------------
# Frozen Mamba target
# ---------------------------------------------------------------------------

A0_SOURCE_POPULATION =
A0_REPLACEMENT_R1_SEED180

A0_ARCHITECTURE =
v6b_minimal

A0_BACKBONE =
mamba

A0_MODEL_NAME =
state-spaces/mamba-130m-hf

A0_TRAINING_SEED =
180

A0_SPLIT_SEED =
8192

A0_SELECTED_CHECKPOINT_SHA256 =
4f7ad019bddb988a534c477b58b36bdabe2775d6c9748331e8311653c07c864c

CHECKPOINT_PATH_AUTHORITY =
CONTENT_HASH_AND_METADATA_NOT_UNVERIFIED_PATH

CHECKPOINT_STRICT_LOAD_REQUIRED =
YES

CHECKPOINT_SELECTION_OR_REGENERATION_ALLOWED =
NO

TRAINING_ALLOWED =
NO

OPTIMIZER_CREATION_ALLOWED =
NO

GRADIENT_COMPUTATION_ALLOWED =
NO

# ---------------------------------------------------------------------------
# Frozen tokenizer / input coordinate
# ---------------------------------------------------------------------------

CANONICAL_ANALYSIS_TOKENIZER_REFERENCE =
40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37

CANONICAL_TOKENIZER_JSON_SHA256 =
b074ad869d4f45d1265ca5c9814f78604f3d7e187acc063b15dd232b27585fcf

TOKENIZER_NETWORK_DOWNLOAD_ALLOWED =
NO

A0_MAX_LENGTH =
128

A0_CLAIM_BUDGET =
63

A0_EVIDENCE_BUDGET =
64

A0_ADD_SPECIAL_TOKENS =
FALSE

A0_SEPARATOR =
EOS_TOKEN_ID

A0_PAD_RULE =
IF_MISSING_SET_PAD_TO_EOS

INPUT_ENCODING_MUST_MATCH =
frozen_A0_active_encoding_contract

# ---------------------------------------------------------------------------
# VitaminC external-label normalization
# ---------------------------------------------------------------------------

LABEL_MAPPING =
SUPPORTS -> SUPPORT
REFUTES -> REFUTE
NOT ENOUGH INFO -> NOT_ENTITLED

ROW_IDENTITY =
raw_idx plus unique_id

CLUSTER_IDENTITY =
case_id

EXTERNAL_INTERVENTION_TYPE =
stage43b1_external_factver

A0_FLAG_SOURCE_SEMANTICS =
controlled_heuristic

TEMPORAL_MISMATCH_FLAG =
ZERO_FOR_ALL_ROWS

PREDICATE_MISMATCH_FLAG =
ZERO_FOR_ALL_ROWS

FLAG_JUSTIFICATION =
The frozen controlled_heuristic path produces temporal all-zero flags and
predicate flag one only for intervention_type equal to predicate_swap.
The external VitaminC intervention identity is stage43b1_external_factver.

AUXILIARY_GOLD_LABELS_MAY_AFFECT_MODEL_FORWARD =
NO

IMPLEMENTATION_MUST_ASSERT_AUXILIARY_LABEL_INDEPENDENCE =
YES

# ---------------------------------------------------------------------------
# Confidence and decisive commitment
# ---------------------------------------------------------------------------

CONFIDENCE_STATISTIC =
PREDICTED_CLASS_FINAL_PROBABILITY

CONFIDENCE_THRESHOLD =
0.5

CONFIDENCE_THRESHOLD_SWEEP =
PROHIBITED

DECISIVE_PREDICTIONS =
SUPPORT
REFUTE

NOT_ENTITLED_PRIMARY_COHORT =
EXCLUDED

CORRECT =
predicted_label_equals_gold_label

WRONG =
predicted_label_differs_from_gold_label

EPistemicBERT_PREDICTIONS_USED_FOR_MAMBA_COHORT =
NO

EPistemicBERT_CONFIDENCE_USED_FOR_MAMBA_COHORT =
NO

# ---------------------------------------------------------------------------
# Census outputs
# ---------------------------------------------------------------------------

REQUIRED_ROW_OUTPUT_FIELDS =
raw_idx
unique_id
case_id
gold_label
pred_label
final_probs
confidence
correct
decisive_prediction
confident
confident_decisive_correct
confident_decisive_wrong
input_token_length
claim_token_length
evidence_token_length
claim_truncated
evidence_truncated

REQUIRED_SUMMARY_OUTPUTS =
dataset_row_count
unique_case_id_count
prediction_distribution
correct_wrong_counts
decisive_counts
confident_decisive_counts
confident_decisive_correct_rows
confident_decisive_wrong_rows
confident_decisive_correct_case_ids
confident_decisive_wrong_case_ids
counts_by_predicted_class
row_multiplicity_per_case
wrong_case_id_set
correct_control_case_id_set_excluding_all_wrong_cases
predicted_class_case_level_pair_capacity
total_case_level_pair_capacity
input_length_distribution
truncation_counts

FINAL_MATCHED_COHORT_SELECTED =
NO

FINAL_MATCHING_ALGORITHM_FROZEN =
NO

CENSUS_PAIR_CAPACITY_ROLE =
FEASIBILITY_ONLY

MINIMUM_MATCHED_PAIR_COUNT_REFERENCE =
45

MINIMUM_POTENTIAL_PAIRS_PER_REPRESENTED_PREDICTED_CLASS_REFERENCE =
10

# ---------------------------------------------------------------------------
# Explicit native-state embargo
# ---------------------------------------------------------------------------

MAMBA_TAU_E_MAPPING_ALLOWED =
NO

CANDIDATE_RAW_EDIT_SPAN_MAY_BE_PROMOTED_TO_TAU_E =
NO

NATIVE_STATE_ACCESS_ALLOWED =
NO

HIDDEN_STATE_EXPORT_ALLOWED =
NO

RECURRENT_STATE_EXPORT_ALLOWED =
NO

POST4_SPEED_ALLOWED =
NO

POST4_TURNING_ALLOWED =
NO

POST4_PATH_EFFICIENCY_ALLOWED =
NO

CORRECT_VS_WRONG_STATE_COMPARISON_ALLOWED =
NO

P1_P2_P3_ALLOWED =
NO

# ---------------------------------------------------------------------------
# Authorized implementation
# ---------------------------------------------------------------------------

IMPLEMENTATION_ALLOWED =
YES_BOUNDED

AUTHORIZED_IMPLEMENTATION_FILES =
scripts/run_native_state_kinematics_vitaminc_prediction_census.py
tests/test_native_state_kinematics_vitaminc_prediction_census.py

EXISTING_TRAINER_MODIFICATION_ALLOWED =
NO

EXISTING_MODEL_SOURCE_MODIFICATION_ALLOWED =
NO

IMPLEMENTATION_REQUIREMENTS =
load exact checkpoint only after SHA256 verification
validate checkpoint metadata before model forward
use exact canonical local tokenizer content
encode claim/evidence under frozen A0 contract
restore exact v6b_minimal A0 model configuration
strictly load checkpoint state
run eval mode under torch.no_grad
export only final task logits/probabilities and input-coordinate diagnostics
never request or serialize native recurrent states
never train
never select a checkpoint
never alter the confidence threshold
never use EpistemicBERT outputs for Mamba correctness or confidence
fail closed on any provenance or schema mismatch

UNIT_TESTS_MUST_COVER =
VitaminC label canonicalization
case_id independence rules
controlled_heuristic external flag equivalence
confidence definition
decisive prediction definition
case-level capacity exclusion of wrong/control overlap
forbidden native-state output keys
output collision protection
checkpoint SHA fail-closed behavior

# ---------------------------------------------------------------------------
# Post-freeze execution authority
# ---------------------------------------------------------------------------

CHECKPOINT_INFERENCE_ALLOWED =
YES_ONLY_AFTER_IMPLEMENTATION_IS_COMMITTED_AND_PUSHED

EVALUATION_EXECUTION_ALLOWED =
YES_PREDICTION_ONLY_AFTER_IMPLEMENTATION_FREEZE

KAGGLE_ALLOWED =
YES_AFTER_IMPLEMENTATION_FREEZE

CUDA_ALLOWED =
YES_FOR_THIS_PREDICTION_ONLY_FORWARD

GPU_NATIVE_STATE_EXTRACTION_ALLOWED =
NO

EXECUTION_COUNT =
ONE_R1_CENSUS_ATTEMPT

RUN_NAMING_RULE =
native-q1-vitaminc-mamba-prediction-census-<implementation-shortcommit>-r1

CM_RUN_REQUIRED =
YES

EXECUTION_COMMAND_MUST_BIND =
exact implementation commit
exact dataset SHA256
exact checkpoint SHA256
exact tokenizer content SHA256
no-overwrite output root
prediction-only mode
native-state-disabled invariant

CHECKPOINT_DISCOVERY_RULE =
The execution wrapper may locate candidate checkpoint files only to identify
a unique file whose SHA256 equals A0_SELECTED_CHECKPOINT_SHA256.
Path convenience may never substitute for the exact hash.

EXECUTION_OUTPUTS =
prediction_rows.jsonl
prediction_census_summary.json
run_provenance.json

# ---------------------------------------------------------------------------
# Stop conditions
# ---------------------------------------------------------------------------

STOP_IF =
dataset SHA256 mismatch
dataset row count mismatch
case_id schema mismatch
canonical tokenizer content mismatch
checkpoint absent
checkpoint SHA256 mismatch
checkpoint metadata inconsistent with frozen A0 replacement-R1 identity
strict checkpoint load failure
unexpected model architecture
label mapping mismatch
nonzero temporal or predicate external flags
output collision
native-state or hidden-state key requested or emitted
training or optimizer path activated
confidence threshold differs from 0.5
implementation HEAD differs from cm run registered HEAD

# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------

PASS_MEANS =
The frozen A0 Mamba prediction population on frozen VitaminC has been
validly enumerated under the case_id independence contract.

PASS_DOES_NOT_MEAN =
native-state precursor supported
native-state precursor not supported
tau_e resolved
matching balance passed
prospective power passed
P1-P3 tested

NEXT_STAGE_IF_CENSUS_CAPACITY_PASSES =
MAMBA_TOKEN_EVENT_COORDINATE_AND_MATCHING_FEASIBILITY_PREPARATION

NEXT_STAGE_IF_CENSUS_CAPACITY_FAILS =
REFORMULATE_WITHOUT_NATIVE_STATE_EXTRACTION

END_OF_NATIVE_STATE_KINEMATICS_VITAMINC_MAMBA_PREDICTION_CENSUS_AUTHORITY