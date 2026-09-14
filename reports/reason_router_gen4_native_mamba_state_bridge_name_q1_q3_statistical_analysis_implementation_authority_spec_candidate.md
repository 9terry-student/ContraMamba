# ContraMamba Gen4 Native Mamba State Bridge
# NAME Q1/Q3 Statistical Analysis Implementation Authority
# Candidate

STATUS =
CANDIDATE

AUTHORITY_ID =
GEN4_NATIVE_MAMBA_STATE_BRIDGE_NAME_Q1_Q3_STATISTICAL_ANALYSIS_IMPLEMENTATION_AUTHORITY

THIS_DOCUMENT_AUTHORIZES =
IMPLEMENTATION_AND_SYNTHETIC_VALIDATION_ONLY

CANONICAL_STATISTICAL_EXECUTION =
NOT_AUTHORIZED

SCIENTIFIC_CONCLUSION =
NONE


## 1. Frozen scientific authority

Q1_Q3_SECONDARY_SIX_STATISTICAL_SPECIFICATION_COMMIT =
2030de52922c251caeafceb36dd5f2f57b5b39c7

Q1_Q3_SCIENTIFIC_SPECIFICATION_COMMIT =
01801ad1617b2ebc3ffa859ba440636d4755a55c

Q1_Q3_EXTRACTION_ARTIFACT_RELEASE_BINDING_FREEZE =
a0ce07ad828c5ee0e6a4c36d8f9a6e972090590a

Q1_Q3_EXTRACTION_ARTIFACT_STORAGE_FREEZE =
60b430fccec9e51e4d1f12131eabf37d03a8be66

Q1_Q3_PROVENANCE_VALIDATION_FREEZE =
e500beabf8a7e1dfc0f8260586d8f5ed8e374d5c

PHASE_F_VALIDATED_STATISTICAL_IMPLEMENTATION_COMMIT =
e917e4c4fe0c94aa4ef5be33f2a69e6c64732189

PHASE_F_STATISTICAL_SPECIFICATION_COMMIT =
830f7ea697ce24388dddd181c9aa301ec2b442fc

The Q1/Q3 six-family statistical definitions are immutable.

This authority does not redefine:

- the two frozen secondary layers;
- the three kinematic endpoints;
- DELTA_NAME;
- the six-hypothesis family;
- the source-pair inferential unit;
- sidedness;
- numeric dtype;
- confidence interval;
- effect size;
- multiplicity;
- hypothesis order;
- decision semantics;
- cross-layer inference boundaries;
- scientific interpretation.


## 2. Frozen future canonical input

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

During implementation and synthetic validation:

CANONICAL_INPUT_INSPECTION =
FORBIDDEN

CANONICAL_INPUT_OPEN =
FORBIDDEN

CANONICAL_PAIR_LEVEL_CONTRAST_COMPUTATION =
FORBIDDEN

CANONICAL_T_STATISTIC_COMPUTATION =
FORBIDDEN

CANONICAL_P_VALUE_COMPUTATION =
FORBIDDEN

CANONICAL_HOLM_COMPUTATION =
FORBIDDEN

CANONICAL_SCIENTIFIC_RESULT_OUTPUT =
FORBIDDEN

The implementation may encode the frozen input path, SHA256, byte count,
row count, and cardinality as fail-closed constants for a later separately
authorized execution.


## 3. Exact implementation scope

Exactly one new implementation file is authorized:

scripts/reason_router_gen4_native_mamba_state_name_q1_q3_statistical_analysis.py

Exactly one new dedicated test file is authorized:

tests/test_reason_router_gen4_native_mamba_state_name_q1_q3_statistical_analysis.py

IMPLEMENTATION_FILE_COUNT =
1

TEST_FILE_COUNT =
1

No existing source file may be modified.

No existing test file may be modified.

No report, extraction artifact, tokenizer artifact, checkpoint, workflow,
configuration, release asset, or scientific-result artifact may be modified.

The implementation phase therefore permits exactly two new tracked files.


## 4. Runtime boundary

CPU_ONLY =
YES

GPU =
NOT_REQUIRED

KAGGLE =
NOT_AUTHORIZED

MODEL_FORWARD =
NOT_AUTHORIZED

CHECKPOINT_LOADING =
NOT_AUTHORIZED

TOKENIZER_EXECUTION =
NOT_AUTHORIZED

NATIVE_STATE_EXTRACTION =
NOT_AUTHORIZED

TRAINING =
NOT_AUTHORIZED

BACKWARD =
NOT_AUTHORIZED

OPTIMIZER =
NOT_AUTHORIZED

NEW_THIRD_PARTY_DEPENDENCIES =
FORBIDDEN

Only deterministic CPU statistical logic and synthetic validation are in
scope.


## 5. Canonical input identity contract for future execution

A later execution must fail closed before statistical interpretation unless
the canonical input has exactly:

PATH =
reports/reason_router_gen4_native_mamba_state_bridge_name_q1_q3_scientific_extraction_v1/kinematic_endpoints.jsonl

SHA256 =
b47e32496f73f5493e8737b040b30bc4a23dd8cd22454a85244fa812273b6605

BYTES =
513396

ROWS =
1200

SOURCE_PAIRS =
300

No alternate input path, regenerated endpoint file, filtered subset, or
transformed copy may substitute for the frozen endpoint artifact.


## 6. Required endpoint-row validation

The implementation must fail closed on at least:

- wrong total row count;
- non-object JSONL row;
- missing source_pair_id;
- empty source_pair_id;
- missing row_id;
- duplicate row_id;
- missing contrast_cell_id;
- contrast cell outside {C0_SHAM,C2_NAME};
- semantic_anchor other than A_NAME;
- layer_index outside {5,17};
- duplicate (source_pair_id, layer_index, contrast_cell_id) identity;
- source-pair count other than 300;
- source pair with row count other than 4;
- missing layer 5 C0_SHAM;
- missing layer 5 C2_NAME;
- missing layer 17 C0_SHAM;
- missing layer 17 C2_NAME;
- unexpected extra pair/layer/cell identity;
- missing required endpoint;
- boolean endpoint value;
- non-numeric endpoint;
- NaN endpoint;
- infinite endpoint.

Required endpoint fields are exactly the frozen three inferential quantities:

POST4_SPEED
POST4_TURNING
POST4_PATH_EFFICIENCY

No row dropping, imputation, repair, replacement, epsilon substitution, or
outcome-dependent filtering is permitted.


## 7. Exact pair-level estimand

For each source pair p, each layer l in {5,17}, and each endpoint K:

DELTA_NAME_K_L(p) =
K(p,C2_NAME,A_NAME,layer=l)
-
K(p,C0_SHAM,A_NAME,layer=l)

No other structural contrast is authorized.

ESTIMAND =
DELTA_NAME

PAIR_LEVEL_CONTRAST_ROWS =
1800

N_PER_HYPOTHESIS =
300

The implementation must enforce source pair as the inferential unit.


## 8. Deterministic hypothesis order

LAYER_ORDER =
5
17

ENDPOINT_ORDER =
POST4_SPEED
POST4_TURNING
POST4_PATH_EFFICIENCY

HYPOTHESIS_ORDER =
LAYER_ORDER_THEN_ENDPOINT_ORDER

PAIR_ORDER =
LEXICOGRAPHIC_SOURCE_PAIR_ID

PAIR_LEVEL_ROW_ORDER =
LAYER_ORDER_THEN_ENDPOINT_ORDER_THEN_PAIR_ORDER

SECONDARY_HYPOTHESIS_COUNT =
6

SECONDARY_RESULT_ROWS =
6


## 9. Frozen statistical procedure

NUMERIC_ANALYSIS_DTYPE =
FLOAT64

STATISTICAL_TEST =
TWO_SIDED_ONE_SAMPLE_STUDENT_T_TEST_ON_PAIR_LEVEL_CONTRASTS

NULL_MEAN =
0

N =
300

DF =
299

CONFIDENCE_INTERVAL =
TWO_SIDED_95_PERCENT_STUDENT_T_CONFIDENCE_INTERVAL

EFFECT_SIZE =
D_Z

D_Z =
mean / sample_sd

ZERO_VARIANCE =
BLOCK_SECONDARY_ANALYSIS

FAMILYWISE_ALPHA =
0.05

MULTIPLICITY_METHOD =
HOLM_BONFERRONI

MULTIPLICITY_FAMILY =
ALL_6_SECONDARY_HYPOTHESES_TOGETHER

Separate correction by layer is forbidden.

Separate correction by endpoint is forbidden.

No normal approximation is authorized.

No epsilon variance is authorized.

No synthetic p-value for a zero-variance distribution is authorized.


## 10. Phase F numerical-reference boundary

The already validated Phase F implementation may be used as a mathematical
reference for synthetic validation only:

scripts/reason_router_gen4_native_mamba_state_phase_f_statistical_analysis.py

PHASE_F_VALIDATED_IMPLEMENTATION_COMMIT =
e917e4c4fe0c94aa4ef5be33f2a69e6c64732189

The Q1/Q3 implementation may reuse or reproduce the validated mathematical
approach for:

- finite numeric validation;
- descriptive statistics;
- two-sided Student-t p-values;
- Student-t critical values;
- confidence intervals;
- deterministic CSV serialization;
- canonical JSON serialization.

However:

PHASE_F_15_MEMBER_HOLM_FUNCTION_DIRECT_REUSE =
PROHIBITED

because the Q1/Q3 multiplicity family contains exactly six hypotheses.

The Q1/Q3 implementation must implement and test an exact six-member global
Holm-Bonferroni procedure.

Synthetic tests should cross-check Student-t probability and critical-value
behavior against the frozen Phase F implementation on synthetic numeric
inputs only.

No Phase F scientific result value may be used as an expected Q1/Q3 answer.


## 11. Point-estimate requirements

For every hypothesis report:

n
mean
sample_sd
standard_error
median
minimum
maximum

STANDARD_ERROR =
sample_sd / sqrt(n)

The implementation must compute these quantities in Python binary64
arithmetic.

For N=300:

DF =
299


## 12. Decision semantics

For each hypothesis:

Holm-adjusted p < 0.05

maps to:

PRESPECIFIED_SECONDARY_LAYER_LOCAL_NAME_KINEMATIC_CORRELATE_SUPPORTED

Otherwise:

PRESPECIFIED_SECONDARY_LAYER_LOCAL_NAME_KINEMATIC_CORRELATE_NOT_ESTABLISHED

FAMILY_LEVEL_SUPPORTED_IF =
AT_LEAST_ONE_OF_SIX_IS_SUPPORTED

FAMILY_LEVEL_SUPPORTED_LABEL =
SECONDARY_LAYER_NAME_LOCALIZATION_SUPPORTED

FAMILY_LEVEL_NOT_ESTABLISHED_LABEL =
SECONDARY_LAYER_NAME_LOCALIZATION_NOT_ESTABLISHED

Non-rejection must not be serialized or described as exact zero.

Output sign matching against the R6 behavioral effect is not a decision
criterion.


## 13. Cross-layer boundary

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

The implementation must not compute a difference of layer-specific DELTA_NAME
effects as a confirmatory or exploratory test.


## 14. Deterministic output contract

A future separately authorized canonical execution must be capable of
producing exactly:

name_q1_q3_pair_level_contrasts.csv
name_q1_q3_secondary_confirmatory_results.csv
name_q1_q3_statistical_analysis_manifest.json
name_q1_q3_statistical_analysis_report_candidate.md

PAIR_LEVEL_FIELDS =
schema_version
source_pair_id
layer_index
endpoint
estimand
contrast_value

SECONDARY_RESULT_FIELDS =
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

PAIR_LEVEL_RESULT_ROWS =
1800

SECONDARY_RESULT_ROWS =
6

CSV_ENCODING =
UTF8

CSV_LINE_ENDINGS =
LF

CSV_FIXED_COLUMN_ORDER =
YES

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

Output directory collision must fail closed.

Partial canonical output publication is forbidden.


## 15. Future manifest requirements

The implementation must support a future manifest that binds:

- statistical specification commit
  2030de52922c251caeafceb36dd5f2f57b5b39c7;
- Q1/Q3 scientific specification commit
  01801ad1617b2ebc3ffa859ba440636d4755a55c;
- extraction release-binding freeze
  a0ce07ad828c5ee0e6a4c36d8f9a6e972090590a;
- canonical input path;
- canonical input SHA256;
- canonical input byte count;
- canonical input row count;
- source-pair count;
- layer set and order;
- endpoint set and order;
- structural estimand;
- hypothesis count;
- N per hypothesis;
- familywise alpha;
- Holm-Bonferroni multiplicity;
- numeric dtype;
- implementation commit;
- implementation script SHA256;
- later statistical execution authority commit;
- all generated output SHA256 values.

The manifest must not claim an overall adaptive-program FWER across Phase F
and this post-primary follow-up.


## 16. Synthetic test requirements

Tests must use synthetic fixtures only.

Tests must not open, read, parse, hash through the implementation, or inspect:

reports/reason_router_gen4_native_mamba_state_bridge_name_q1_q3_scientific_extraction_v1/kinematic_endpoints.jsonl

or any canonical Q1/Q3 statistical-result artifact.

Tests must cover at minimum:

1. exact 4-row per-pair matrix acceptance;
2. layer set exactly {5,17};
3. cell set exactly {C0_SHAM,C2_NAME};
4. semantic anchor exactly A_NAME;
5. duplicate row_id rejection;
6. duplicate pair/layer/cell rejection;
7. missing layer/cell identity rejection;
8. extra layer rejection;
9. extra cell rejection;
10. wrong anchor rejection;
11. nonfinite endpoint rejection;
12. boolean endpoint rejection;
13. all three endpoint contrasts;
14. both layer contrasts;
15. DELTA_NAME formula C2_NAME minus C0_SHAM;
16. lexicographic source-pair ordering;
17. layer-first then endpoint result ordering;
18. 1800 pair-level row cardinality for 300 synthetic pairs;
19. six hypothesis result cardinality;
20. N=300 enforcement;
21. sample mean;
22. sample standard deviation;
23. standard error;
24. median/minimum/maximum;
25. two-sided one-sample Student-t statistic;
26. df=299;
27. two-sided Student-t p-value;
28. synthetic agreement with validated Phase F Student-t p-value primitive;
29. 95 percent Student-t critical value;
30. synthetic agreement with validated Phase F Student-t critical-value primitive;
31. 95 percent confidence interval;
32. d_z;
33. zero-variance hard blocker;
34. exact global Holm adjustment over six hypotheses;
35. deterministic tie handling in Holm ordering;
36. Holm step-down adjusted-p monotonicity;
37. Holm decision threshold uses adjusted p < 0.05;
38. proof that separate per-layer Holm is not used;
39. proof that separate per-endpoint Holm is not used;
40. supported decision label;
41. not-established decision label;
42. family-level supported semantics;
43. family-level not-established semantics;
44. deterministic pair-level CSV bytes;
45. deterministic secondary-result CSV bytes;
46. canonical JSON bytes;
47. exactly one terminal LF in canonical JSON;
48. deterministic report generation;
49. output collision rejection;
50. canonical input path remains unopened by synthetic tests.

Synthetic tests may use 300-pair generated fixtures in memory or temporary
directories.


## 17. Required implementation validation

Before implementation-freeze review run exactly:

python -m py_compile scripts/reason_router_gen4_native_mamba_state_name_q1_q3_statistical_analysis.py

python -m pytest -q tests/test_reason_router_gen4_native_mamba_state_name_q1_q3_statistical_analysis.py

git diff --check -- scripts/reason_router_gen4_native_mamba_state_name_q1_q3_statistical_analysis.py tests/test_reason_router_gen4_native_mamba_state_name_q1_q3_statistical_analysis.py

All commands must pass.

The dedicated synthetic tests must not consume canonical Q1/Q3 endpoint
values.

Passing these tests does not authorize canonical statistical execution.


## 18. Promotion condition

The implementation may be frozen only when:

- exactly the two authorized implementation/test files are added;
- no existing tracked file is modified;
- py_compile passes;
- all dedicated synthetic tests pass;
- git diff --check passes;
- canonical endpoint values were not inspected by implementation tests;
- no canonical pair-level contrast was produced;
- no canonical p-value was produced;
- no canonical Holm decision was produced;
- no canonical statistical-result artifact was created;
- no unrelated tracked file changed.

After implementation freeze, a separate:

GEN4_NATIVE_MAMBA_STATE_BRIDGE_NAME_Q1_Q3_STATISTICAL_ANALYSIS_EXECUTION_AUTHORITY

must bind the exact implementation commit and exact implementation-script
SHA256 before any canonical six-hypothesis computation.


## 19. Stop condition

Stop after implementation and synthetic validation have been independently
reviewed and frozen.

Do not compute any canonical Q1/Q3 pair-level contrast.

Do not compute any canonical Q1/Q3 t statistic.

Do not compute any canonical Q1/Q3 p-value.

Do not compute any canonical Q1/Q3 Holm-adjusted p-value.

Do not inspect canonical Q1/Q3 effect directions.

Do not rerun native-state extraction.

Do not use Kaggle.

Do not use GPU.

SCIENTIFIC_CONCLUSION =
NONE

NEXT_PHASE =
NAME_Q1_Q3_STATISTICAL_ANALYSIS_IMPLEMENTATION_AND_SYNTHETIC_VALIDATION
