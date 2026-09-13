# ContraMamba Gen4 Native Mamba State Bridge
# Phase F Primary-15 Statistical Analysis Specification
# Candidate

STATUS =
CANDIDATE

AUTHORITY_ID =
GEN4_NATIVE_MAMBA_STATE_BRIDGE_PHASE_F_PRIMARY_15_STATISTICAL_ANALYSIS_SPECIFICATION

THIS_DOCUMENT_CREATES_NEW_SCIENTIFIC_EVIDENCE =
NO

STATISTICAL_EXECUTION_ALLOWED =
NO

MODEL_FORWARD_ALLOWED =
NO

CHECKPOINT_LOADING_ALLOWED =
NO

TRAINING_ALLOWED =
NO

KAGGLE_ALLOWED =
NO

GPU_REQUIRED =
NO

## 1. Authorities

MECHANISTIC_BRIDGE_SPECIFICATION =
a2617aa037d1a9834003535b62ac81770a5b96aa

PHASE_D_EXECUTION_AUTHORITY =
bff0a75a6e2815a0bb3565f1e91ca91193de73e1

PHASE_E_MEASUREMENT_ARTIFACT_FREEZE =
4a2ef821122a36c8e48ec261012519102b63d6c3

PHASE_E_RELEASE_BINDING_FREEZE =
1695430fabf4d3ebce28d2b1622c218e65b052c3

## 2. Frozen statistical input

PRIMARY_INPUT_PATH =
reports/reason_router_gen4_native_mamba_state_extraction_bff0a75_v1/kinematic_endpoints.jsonl

PRIMARY_INPUT_SHA256 =
7ff4d24b7895745efbf5a00e0361a4fe409ad585db17ece7151e4dedb289b07c

PRIMARY_INPUT_ROW_COUNT =
3600

SOURCE_PAIR_COUNT =
300

PRIMARY_LAYER =
11

RAW_SUPPORT_STATE_TENSOR_REQUIRED_FOR_PRIMARY_STATISTICS =
NO

SUPPORT_STATES_SHA256 =
810fdde63504c9e64622e50ccecefa0681ff5f6ee70244257acdf63d62fedc06

The raw support-state tensor remains frozen provenance/reconstruction evidence.
The primary Phase F inferential analysis consumes the already frozen kinematic
endpoint rows and must not recompute or alter native-state measurements.

## 3. Required endpoint matrix

Each source pair must have exactly the frozen anchor/cell endpoint rows needed
by the mechanistic bridge:

A_TITLE:
C0_SHAM
C1_TITLE

A_NAME:
C0_SHAM
C2_NAME

A_ROLE:
C0_SHAM
C3_ROLE

A_PREDICATE:
C0_SHAM
C4_PREDICATE

A_IDENTITY:
C0_SHAM
C1_TITLE
C2_NAME
C5_TITLE_NAME

ROWS_PER_SOURCE_PAIR =
12

EXPECTED_TOTAL_ROWS =
3600

Incomplete or duplicate pair/anchor/cell identity blocks are fatal.

MISSINGNESS =
FORBIDDEN

IMPUTATION =
FORBIDDEN

## 4. Frozen endpoints

ENDPOINT_1 =
POST4_SPEED

ENDPOINT_2 =
POST4_TURNING

ENDPOINT_3 =
POST4_PATH_EFFICIENCY

PRIMARY_ENDPOINT_COUNT =
3

No additional primary endpoint may be introduced after inspection.

## 5. Structural estimands

For endpoint K and source pair p:

DELTA_TITLE_K =
K(p,C1_TITLE,A_TITLE) - K(p,C0_SHAM,A_TITLE)

DELTA_NAME_K =
K(p,C2_NAME,A_NAME) - K(p,C0_SHAM,A_NAME)

DELTA_ROLE_K =
K(p,C3_ROLE,A_ROLE) - K(p,C0_SHAM,A_ROLE)

DELTA_PREDICATE_K =
K(p,C4_PREDICATE,A_PREDICATE) - K(p,C0_SHAM,A_PREDICATE)

INTERACTION_TITLE_NAME_K =
K(p,C5_TITLE_NAME,A_IDENTITY)
- K(p,C1_TITLE,A_IDENTITY)
- K(p,C2_NAME,A_IDENTITY)
+ K(p,C0_SHAM,A_IDENTITY)

PRIMARY_STRUCTURAL_ESTIMAND_COUNT =
5

STATE_LEVEL_TITLE_MINUS_NAME =
NOT_AUTHORIZED

## 6. Confirmatory family

PRIMARY_HYPOTHESIS_COUNT =
15

The family is the Cartesian product of:

5 structural estimands
x
3 frozen endpoints

All 15 belong to one global confirmatory family.

SIDEDNESS =
TWO_SIDED

DIRECTION_INFERRED_FROM_R6_OUTPUT_SIGN =
NO

## 7. Inferential unit

PRIMARY_INFERENTIAL_UNIT =
SOURCE_PAIR

N_PER_HYPOTHESIS =
300

No endpoint row, token, state coordinate, or model dimension is an independent
inferential replicate.

PSEUDOREPLICATION_BY_STATE_DIMENSION =
FORBIDDEN

PSEUDOREPLICATION_BY_TOKEN =
FORBIDDEN

## 8. Point estimates

For every one of the 15 pair-level contrast distributions report:

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

The frozen JSON endpoint values are parsed without alteration and statistical
contrasts/summaries are computed in binary64 arithmetic.

## 9. Primary statistical test

STATISTICAL_TEST =
TWO_SIDED_ONE_SAMPLE_STUDENT_T_TEST_ON_PAIR_LEVEL_CONTRASTS

NULL_MEAN =
0

DF =
299

For pair-level contrast vector d:

t =
mean(d) / (sample_sd(d) / sqrt(300))

No directional alternative is authorized.

If sample_sd(d) equals zero, the Student t statistic is undefined.

ZERO_VARIANCE_HANDLING =
BLOCK_PRIMARY_ANALYSIS

No artificial p-value or epsilon variance may be introduced.

## 10. Confidence interval

UNCERTAINTY_INTERVAL =
TWO_SIDED_95_PERCENT_STUDENT_T_CONFIDENCE_INTERVAL

DF =
299

The interval is reported for each raw mean pair-level contrast.

The confidence interval itself is not the multiplicity decision rule.

## 11. Effect size

EFFECT_SIZE =
D_Z

D_Z =
mean(d) / sample_sd(d)

ZERO_VARIANCE_EFFECT_SIZE =
UNDEFINED_AND_PRIMARY_ANALYSIS_BLOCKED

No small/medium/large verbal threshold is authorized.

## 12. Multiplicity

CONFIRMATORY_FAMILYWISE_ALPHA =
0.05

MULTIPLICITY_METHOD =
HOLM_BONFERRONI

GLOBAL_PRIMARY_MULTIPLICITY =
ALL_15_HYPOTHESES_TOGETHER

Separate Holm correction within each endpoint is forbidden.

Raw p-values and Holm-adjusted p-values must both be serialized.

## 13. Measurement validity / zero path

The frozen Phase D implementation defines zero POST4 path length as a hard
measurement blocker and prohibits an epsilon denominator.

PHASE_F_ZERO_PATH_HANDLING =
BLOCK_IF_PRESENT

NONFINITE_ENDPOINT_HANDLING =
BLOCK_IF_PRESENT

No row may be dropped, repaired, imputed, shortened, or assigned an epsilon.

A successful Phase D bundle already passed its frozen endpoint finiteness and
reconstruction gates; Phase F must revalidate input structure before testing.

## 14. Confirmatory decision rule

For an estimand/endpoint hypothesis:

PRESPECIFIED_LOCAL_NATIVE_STATE_KINEMATIC_RESPONSE_SUPPORTED

requires:

Holm-adjusted p < 0.05

A non-rejected hypothesis receives:

PRESPECIFIED_LOCAL_NATIVE_STATE_KINEMATIC_RESPONSE_NOT_ESTABLISHED

Non-rejection does not establish exact zero.

For the interaction estimand, a rejection may additionally be described as:

PRESPECIFIED_LOCAL_NATIVE_STATE_KINEMATIC_NONADDITIVITY_SUPPORTED

The sign of the raw mean describes direction only after the two-sided test.
No output-sign matching criterion is used.

## 15. Bounded interpretation

If an R6-supported semantic estimand also has a supported Phase F native-state
contrast, the strongest bridge statement is:

OUTPUT_EFFECT_HAS_A_PRESPECIFIED_LOCAL_NATIVE_STATE_KINEMATIC_CORRELATE

The analysis may not claim:

causal mediation
necessity
sufficiency
state-to-output causation
arbitrary-model generalization
arbitrary-dataset generalization

Title remains a primary control estimand even though its R6 main effect was not
established.

## 16. Deterministic ordering

ENDPOINT_ORDER =
POST4_SPEED
POST4_TURNING
POST4_PATH_EFFICIENCY

ESTIMAND_ORDER =
DELTA_TITLE
DELTA_NAME
DELTA_ROLE
DELTA_PREDICATE
INTERACTION_TITLE_NAME

PAIR_ORDER =
LEXICOGRAPHIC_SOURCE_PAIR_ID

PRIMARY_RESULT_ROW_ORDER =
ENDPOINT_ORDER_THEN_ESTIMAND_ORDER

PAIR_CONTRAST_ROW_ORDER =
ENDPOINT_ORDER_THEN_ESTIMAND_ORDER_THEN_PAIR_ORDER

## 17. Future output contract

A later implementation authority must produce exactly:

phase_f_pair_level_contrasts.csv
phase_f_primary_confirmatory_results.csv
phase_f_statistical_analysis_manifest.json
phase_f_statistical_analysis_report_candidate.md

PAIR_LEVEL_CONTRAST_ROWS =
4500

PRIMARY_RESULT_ROWS =
15

The pair-level CSV must contain at minimum:

schema_version
source_pair_id
endpoint
estimand
contrast_value

The primary-result CSV must contain at minimum:

schema_version
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

The manifest must bind:

this statistical specification commit
input path
input SHA256
input byte count
input row count
source-pair count
hypothesis count
familywise alpha
multiplicity method
numeric analysis dtype
all output SHA256 values

JSON serialization must be canonical:
UTF-8
sorted keys
compact separators
allow_nan false
single terminal LF

CSV serialization must be deterministic:
UTF-8
LF line endings
fixed column order
fixed row order
no index column

## 18. Execution boundary

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

A separate implementation authority and validation gate are required before
computing any of the 15 statistical results.

SCIENTIFIC_CONCLUSION =
NONE

NEXT_PHASE =
PHASE_F_STATISTICAL_ANALYSIS_IMPLEMENTATION_AUTHORITY
