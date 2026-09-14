# ContraMamba Gen4 Native Mamba State Bridge
# NAME Q1/Q3 Secondary Six-Family Statistical Analysis
# Scientific Specification Candidate

STATUS =
CANDIDATE

AUTHORITY_ID =
GEN4_NATIVE_MAMBA_STATE_BRIDGE_NAME_Q1_Q3_SECONDARY_SIX_STATISTICAL_ANALYSIS_SPECIFICATION

THIS_DOCUMENT_CREATES_NEW_SCIENTIFIC_EVIDENCE =
NO

IMPLEMENTATION_ALLOWED =
NO

STATISTICAL_EXECUTION_ALLOWED =
NO

MODEL_FORWARD_ALLOWED =
NO

CHECKPOINT_LOADING_ALLOWED =
NO

TOKENIZER_EXECUTION_ALLOWED =
NO

TRAINING_ALLOWED =
NO

KAGGLE_ALLOWED =
NO

GPU_ALLOWED =
NO


## 1. Frozen authorities

Q1_Q3_SCIENTIFIC_SPECIFICATION_COMMIT =
01801ad1617b2ebc3ffa859ba440636d4755a55c

DEPTH_SELECTIVITY_INTERPRETATION_CORRECTION =
2e076cbd8e9633c3ab7abb222a05e409366539a7

PHASE_F_PRIMARY_RESULT_FREEZE =
ab3428e7be08af26fa1fdafd1483a34e48fbcf8c

Q1_Q3_PROVENANCE_VALIDATION_FREEZE =
e500beabf8a7e1dfc0f8260586d8f5ed8e374d5c

Q1_Q3_EXTRACTION_ARTIFACT_STORAGE_FREEZE =
60b430fccec9e51e4d1f12131eabf37d03a8be66

Q1_Q3_EXTRACTION_ARTIFACT_RELEASE_BINDING_FREEZE =
a0ce07ad828c5ee0e6a4c36d8f9a6e972090590a

EXTRACTION_ARTIFACT_STORAGE =
CLOSED


## 2. Frozen canonical statistical input

CANONICAL_INPUT_PATH =
reports/reason_router_gen4_native_mamba_state_bridge_name_q1_q3_scientific_extraction_v1/kinematic_endpoints.jsonl

CANONICAL_INPUT_SHA256 =
b47e32496f73f5493e8737b040b30bc4a23dd8cd22454a85244fa812273b6605

CANONICAL_INPUT_BYTES =
513396

CANONICAL_INPUT_ROWS =
1200

SOURCE_PAIR_COUNT =
300

ROWS_PER_SOURCE_PAIR =
4

RAW_SUPPORT_STATE_TENSOR_REQUIRED_FOR_STATISTICS =
NO

SUPPORT_STATES_SHA256 =
8b68e303f06da687052cd64bb3f8336e4b52d2686404a26d62d77869608cebc3

The inferential analysis consumes the already frozen kinematic endpoint rows.

It must not recompute native-state measurements from the raw tensor.

During this specification and later implementation/synthetic-validation phase:

CANONICAL_INPUT_OUTCOME_INSPECTION =
FORBIDDEN

CANONICAL_STATISTICAL_COMPUTATION =
FORBIDDEN


## 3. Required endpoint matrix

For every source pair require exactly:

LAYER_5:
C0_SHAM
C2_NAME

LAYER_17:
C0_SHAM
C2_NAME

SEMANTIC_ANCHOR =
A_NAME

LAYER_SET =
{5,17}

CELL_SET =
{C0_SHAM,C2_NAME}

ROWS_PER_SOURCE_PAIR =
4

Incomplete, duplicate, or extra pair/layer/cell blocks are fatal.

MISSINGNESS =
FORBIDDEN

IMPUTATION =
FORBIDDEN

ROW_DROPPING =
FORBIDDEN


## 4. Frozen endpoints

ENDPOINT_ORDER =
POST4_SPEED
POST4_TURNING
POST4_PATH_EFFICIENCY

ENDPOINT_COUNT =
3

No additional endpoint is allowed.


## 5. Frozen structural estimand

STRUCTURAL_ESTIMAND =
DELTA_NAME

For source pair p, endpoint K, and layer l in {5,17}:

DELTA_NAME_K_L(p) =
K(p,C2_NAME,A_NAME,layer=l)
-
K(p,C0_SHAM,A_NAME,layer=l)

TITLE =
OUT_OF_SCOPE

ROLE =
OUT_OF_SCOPE

PREDICATE =
OUT_OF_SCOPE

TITLE_NAME_INTERACTION =
OUT_OF_SCOPE

TITLE_MINUS_NAME =
OUT_OF_SCOPE


## 6. Inferential unit

PRIMARY_INFERENTIAL_UNIT =
SOURCE_PAIR

N_PER_HYPOTHESIS =
300

PSEUDOREPLICATION_BY_LAYER =
FORBIDDEN

PSEUDOREPLICATION_BY_ENDPOINT_ROW =
FORBIDDEN

PSEUDOREPLICATION_BY_TOKEN =
FORBIDDEN

PSEUDOREPLICATION_BY_STATE_DIMENSION =
FORBIDDEN


## 7. Exact six-hypothesis family

SECONDARY_HYPOTHESIS_COUNT =
6

LAYER_ORDER =
5
17

H1 =
LAYER_5 / POST4_SPEED / DELTA_NAME

H2 =
LAYER_5 / POST4_TURNING / DELTA_NAME

H3 =
LAYER_5 / POST4_PATH_EFFICIENCY / DELTA_NAME

H4 =
LAYER_17 / POST4_SPEED / DELTA_NAME

H5 =
LAYER_17 / POST4_TURNING / DELTA_NAME

H6 =
LAYER_17 / POST4_PATH_EFFICIENCY / DELTA_NAME

MIDPOINT_LAYER_11_IN_SECONDARY_FAMILY =
NO

PHASE_F_RESULTS_RECOMPUTED =
NO

PHASE_F_DECISIONS_REPLACED =
NO


## 8. Point estimates

For every pair-level contrast distribution report exactly:

n
mean
sample_sd
standard_error
median
minimum
maximum

STANDARD_ERROR =
sample_sd / sqrt(300)

NUMERIC_ANALYSIS_DTYPE =
FLOAT64

Frozen JSON numeric endpoint values are parsed without alteration and all
statistical contrasts and summaries are computed in binary64 arithmetic.


## 9. Statistical test

STATISTICAL_TEST =
TWO_SIDED_ONE_SAMPLE_STUDENT_T_TEST_ON_PAIR_LEVEL_CONTRASTS

NULL_MEAN =
0

SIDEDNESS =
TWO_SIDED

DF =
299

For pair-level contrast vector d:

t =
mean(d) / (sample_sd(d) / sqrt(300))

ZERO_VARIANCE_HANDLING =
BLOCK_SECONDARY_ANALYSIS

No epsilon variance, synthetic p-value, directional rescue, or normal
approximation is authorized.


## 10. Confidence interval

UNCERTAINTY_INTERVAL =
TWO_SIDED_95_PERCENT_STUDENT_T_CONFIDENCE_INTERVAL

DF =
299

The interval is reported for the raw mean pair-level contrast.

The confidence interval itself is not the multiplicity decision rule.


## 11. Effect size

EFFECT_SIZE =
D_Z

D_Z =
mean(d) / sample_sd(d)

ZERO_VARIANCE_EFFECT_SIZE =
UNDEFINED_AND_SECONDARY_ANALYSIS_BLOCKED

No small/medium/large verbal effect-size threshold is authorized.


## 12. Multiplicity

FAMILYWISE_ALPHA =
0.05

MULTIPLICITY_METHOD =
HOLM_BONFERRONI

GLOBAL_SECONDARY_MULTIPLICITY =
ALL_6_HYPOTHESES_TOGETHER

Separate Holm correction by layer is forbidden.

Separate Holm correction by endpoint is forbidden.

Raw p-values and Holm-adjusted p-values must both be serialized.


## 13. Measurement blocker inheritance

The frozen measurement implementation treats zero transitions needed by
turning and zero POST4 path length as hard measurement blockers.

ZERO_TRANSITION_BEHAVIOR =
INHERIT_FROZEN_MEASUREMENT_BLOCKER

ZERO_PATH_BEHAVIOR =
INHERIT_FROZEN_MEASUREMENT_BLOCKER

STATISTICAL_RECOMPUTATION_OF_STATE_KINEMATICS =
FORBIDDEN

NONFINITE_ENDPOINT_HANDLING =
BLOCK_SECONDARY_ANALYSIS

No endpoint row may be repaired, replaced, shortened, imputed, or assigned an
epsilon.

The statistical analysis must require the already validated extraction
provenance and consume only the frozen endpoint artifact.


## 14. Per-hypothesis decision labels

A hypothesis with:

Holm-adjusted p < 0.05

receives:

PRESPECIFIED_SECONDARY_LAYER_LOCAL_NAME_KINEMATIC_CORRELATE_SUPPORTED

Otherwise it receives:

PRESPECIFIED_SECONDARY_LAYER_LOCAL_NAME_KINEMATIC_CORRELATE_NOT_ESTABLISHED

Non-rejection does not establish exact zero.

The sign of the raw mean describes direction only after the two-sided test.

OUTPUT_SIGN_MATCHING =
NOT_REQUIRED


## 15. Family-level decision

If at least one of the six hypotheses is supported:

SECONDARY_LAYER_NAME_LOCALIZATION =
SUPPORTED

and the exact supported layer/endpoint combinations must be reported.

If none is supported:

SECONDARY_LAYER_NAME_LOCALIZATION =
NOT_ESTABLISHED

The latter does not establish:

NAME_NATIVE_STATE_EFFECT_EQUALS_ZERO

NAME_NATIVE_STATE_EFFECT_ABSENT_AT_ALL_LAYERS


## 16. Deterministic ordering

LAYER_ORDER =
5
17

ENDPOINT_ORDER =
POST4_SPEED
POST4_TURNING
POST4_PATH_EFFICIENCY

PAIR_ORDER =
LEXICOGRAPHIC_SOURCE_PAIR_ID

SECONDARY_RESULT_ROW_ORDER =
LAYER_ORDER_THEN_ENDPOINT_ORDER

PAIR_CONTRAST_ROW_ORDER =
LAYER_ORDER_THEN_ENDPOINT_ORDER_THEN_PAIR_ORDER

PAIR_LEVEL_CONTRAST_ROWS =
1800

SECONDARY_RESULT_ROWS =
6


## 17. Deterministic serialization and future outputs

A later implementation must be capable of producing exactly:

name_q1_q3_pair_level_contrasts.csv
name_q1_q3_secondary_confirmatory_results.csv
name_q1_q3_statistical_analysis_manifest.json
name_q1_q3_statistical_analysis_report_candidate.md

The pair-level CSV must contain exactly the frozen implementation-defined
ordered columns including at minimum:

schema_version
source_pair_id
layer_index
endpoint
estimand
contrast_value

The secondary-result CSV must contain at minimum:

schema_version
layer_index
endpoint
estimand
n
mean
sample_sd
standard_error
median
minimum
maximum
df
t_statistic
raw_p_value
holm_adjusted_p_value
reject_holm_alpha_0_05
ci95_low
ci95_high
d_z
decision

CSV_ENCODING =
UTF8

CSV_LINE_ENDINGS =
LF

CSV_INDEX_COLUMN =
FORBIDDEN

JSON_ENCODING =
UTF8

JSON_SORTED_KEYS =
YES

JSON_COMPACT_SEPARATORS =
YES

JSON_ALLOW_NAN =
NO

JSON_TERMINAL_LF =
EXACTLY_ONE

The future manifest must bind at minimum:

this statistical specification commit
Q1/Q3 scientific specification commit
release-binding freeze commit
canonical input path
canonical input SHA256
canonical input byte count
canonical input row count
source-pair count
layer set and deterministic order
endpoint set and deterministic order
hypothesis count
familywise alpha
multiplicity method
numeric analysis dtype
all output SHA256 values


## 18. Cross-layer inference boundary

CROSS_LAYER_DIFFERENCE_ESTIMAND =
NOT_DEFINED

CROSS_LAYER_DIFFERENCE_TEST =
NOT_AUTHORIZED

MIDPOINT_VERSUS_Q1_TEST =
NOT_AUTHORIZED

MIDPOINT_VERSUS_Q3_TEST =
NOT_AUTHORIZED

Q1_VERSUS_Q3_TEST =
NOT_AUTHORIZED

DEPTH_SELECTIVITY =
NOT_ESTABLISHED_BY_THIS_FAMILY

A pattern of support at one secondary layer and non-support at another layer or
at midpoint layer 11 is not a direct between-layer statistical difference.


## 19. Bounded interpretation

A supported member may establish only:

PRESPECIFIED_SECONDARY_LAYER_LOCAL_NAME_KINEMATIC_CORRELATE

Given the already frozen R6 behavioral NAME effect, a supported member may be
described as a prespecified local native-state kinematic correlate at that
exact secondary layer/endpoint.

The analysis may not establish:

causal mediation
necessity
sufficiency
state-to-output causation
significant depth selectivity
arbitrary-model generalization
arbitrary-dataset generalization


## 20. Prohibited rescue analyses

BEST_LAYER_SCAN =
PROHIBITED

ALL_LAYER_SCAN =
PROHIBITED

NEW_ENDPOINT_SCAN =
PROHIBITED

TOKEN_OFFSET_SCAN =
PROHIBITED

WINDOW_REDEFINITION =
PROHIBITED

POOL_LAYERS_AS_REPLICATES =
PROHIBITED

DIRECTIONAL_RETEST =
PROHIBITED

PHASE_F_RECLASSIFICATION =
PROHIBITED


## 21. Execution boundary

IMPLEMENTATION_ALLOWED_BY_THIS_DOCUMENT =
NO

STATISTICAL_EXECUTION_ALLOWED_BY_THIS_DOCUMENT =
NO

MODEL_EXECUTION =
FORBIDDEN

KAGGLE =
NOT_REQUIRED

GPU =
NOT_REQUIRED

No canonical pair-level contrast, t statistic, p-value, confidence interval,
effect size, Holm-adjusted p-value, or decision may be computed under this
specification alone.


## 22. Next phase

NEXT_PHASE =
NAME_Q1_Q3_STATISTICAL_ANALYSIS_IMPLEMENTATION_AUTHORITY

SCIENTIFIC_CONCLUSION =
NONE
