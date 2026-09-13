# ContraMamba Gen4 Native Mamba State Bridge
# Phase F Statistical Analysis Implementation Authority
# Candidate

STATUS =
CANDIDATE

AUTHORITY_ID =
GEN4_NATIVE_MAMBA_STATE_BRIDGE_PHASE_F_STATISTICAL_ANALYSIS_IMPLEMENTATION_AUTHORITY

THIS_DOCUMENT_AUTHORIZES =
IMPLEMENTATION_AND_SYNTHETIC_VALIDATION_ONLY

CANONICAL_STATISTICAL_EXECUTION =
NOT_AUTHORIZED

## 1. Frozen scientific authority

PHASE_F_STATISTICAL_SPECIFICATION_COMMIT =
830f7ea697ce24388dddd181c9aa301ec2b442fc

PHASE_E_MEASUREMENT_ARTIFACT_FREEZE_COMMIT =
4a2ef821122a36c8e48ec261012519102b63d6c3

PHASE_E_RELEASE_BINDING_FREEZE_COMMIT =
1695430fabf4d3ebce28d2b1622c218e65b052c3

MECHANISTIC_BRIDGE_SPECIFICATION =
a2617aa037d1a9834003535b62ac81770a5b96aa

The frozen Phase F statistical definitions are immutable.

This authority does not redefine:
- the 15-hypothesis family;
- the three kinematic endpoints;
- the five structural estimands;
- the inferential unit;
- sidedness;
- multiplicity;
- confidence interval;
- effect size;
- decision rules;
- scientific interpretation.

## 2. Frozen future canonical input

CANONICAL_INPUT_PATH =
reports/reason_router_gen4_native_mamba_state_extraction_bff0a75_v1/kinematic_endpoints.jsonl

CANONICAL_INPUT_SHA256 =
7ff4d24b7895745efbf5a00e0361a4fe409ad585db17ece7151e4dedb289b07c

CANONICAL_INPUT_BYTES =
1862461

CANONICAL_INPUT_ROWS =
3600

SOURCE_PAIR_COUNT =
300

ROWS_PER_SOURCE_PAIR =
12

During implementation and synthetic validation:

CANONICAL_INPUT_INSPECTION =
FORBIDDEN

CANONICAL_STATISTICAL_COMPUTATION =
FORBIDDEN

The implementation may encode the frozen path, hash, byte count, and
cardinality as fail-closed constants for a later execution authority.

## 3. Exact implementation scope

Exactly one implementation file is authorized:

scripts/reason_router_gen4_native_mamba_state_phase_f_statistical_analysis.py

Exactly one dedicated test file is authorized:

tests/test_reason_router_gen4_native_mamba_state_phase_f_statistical_analysis.py

No other existing source, test, report, artifact, tokenizer, checkpoint,
workflow, or configuration file may be modified during this implementation
phase.

IMPLEMENTATION_FILE_COUNT =
1

TEST_FILE_COUNT =
1

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

TRAINING =
NOT_AUTHORIZED

BACKWARD =
NOT_AUTHORIZED

OPTIMIZER =
NOT_AUTHORIZED

NEW_THIRD_PARTY_DEPENDENCIES =
FORBIDDEN

The implementation must use the existing Python runtime and standard-library
numeric/statistical logic. No package installation is authorized.

## 5. Required input validation

The implementation must fail closed on:

- wrong canonical SHA256;
- wrong canonical byte count;
- wrong row count;
- non-object JSONL rows;
- duplicate (source_pair_id, anchor_name, contrast_cell_id) identities;
- missing source pair;
- source-pair count other than 300;
- source pair with row count other than 12;
- unexpected anchor;
- unexpected cell for an anchor;
- missing required anchor/cell identity;
- unexpected extra anchor/cell identity;
- layer_index other than 11;
- missing endpoint;
- boolean endpoint value;
- non-numeric endpoint;
- NaN endpoint;
- infinite endpoint.

No imputation or row dropping is permitted.

## 6. Exact endpoint matrix

For every source pair, require exactly:

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

TOTAL_ROWS_PER_PAIR =
12

## 7. Frozen endpoints

ENDPOINT_ORDER =
POST4_SPEED
POST4_TURNING
POST4_PATH_EFFICIENCY

PRIMARY_ENDPOINT_COUNT =
3

## 8. Frozen estimands

ESTIMAND_ORDER =
DELTA_TITLE
DELTA_NAME
DELTA_ROLE
DELTA_PREDICATE
INTERACTION_TITLE_NAME

For endpoint K:

DELTA_TITLE =
K(C1_TITLE,A_TITLE) - K(C0_SHAM,A_TITLE)

DELTA_NAME =
K(C2_NAME,A_NAME) - K(C0_SHAM,A_NAME)

DELTA_ROLE =
K(C3_ROLE,A_ROLE) - K(C0_SHAM,A_ROLE)

DELTA_PREDICATE =
K(C4_PREDICATE,A_PREDICATE) - K(C0_SHAM,A_PREDICATE)

INTERACTION_TITLE_NAME =
K(C5_TITLE_NAME,A_IDENTITY)
- K(C1_TITLE,A_IDENTITY)
- K(C2_NAME,A_IDENTITY)
+ K(C0_SHAM,A_IDENTITY)

STATE_LEVEL_TITLE_MINUS_NAME =
FORBIDDEN

PAIR_LEVEL_CONTRAST_ROWS =
4500

## 9. Statistical procedure

PRIMARY_HYPOTHESIS_COUNT =
15

N_PER_HYPOTHESIS =
300

NUMERIC_ANALYSIS_DTYPE =
FLOAT64

STATISTICAL_TEST =
TWO_SIDED_ONE_SAMPLE_STUDENT_T_TEST_ON_PAIR_LEVEL_CONTRASTS

DF =
299

CONFIDENCE_INTERVAL =
TWO_SIDED_95_PERCENT_STUDENT_T

EFFECT_SIZE =
D_Z

D_Z =
mean / sample_sd

ZERO_VARIANCE =
BLOCK_PRIMARY_ANALYSIS

No epsilon variance or synthetic p-value is permitted.

## 10. Multiplicity

FAMILYWISE_ALPHA =
0.05

MULTIPLICITY_METHOD =
HOLM_BONFERRONI

MULTIPLICITY_FAMILY =
ALL_15_PRIMARY_HYPOTHESES_TOGETHER

Holm adjustment must operate globally over all 15 raw two-sided p-values.

Separate endpoint-wise multiplicity correction is forbidden.

## 11. Numerical implementation

The implementation must provide deterministic pure functions for at least:

- endpoint-row validation;
- complete matrix validation;
- pair-level contrast construction;
- arithmetic summaries;
- two-sided Student-t p-value;
- 95 percent Student-t critical value / confidence interval;
- d_z;
- Holm-Bonferroni adjusted p-values;
- Holm step-down decisions;
- deterministic CSV serialization;
- canonical JSON serialization.

The Student-t implementation may follow the already validated mathematical
approach used by:

scripts/reason_router_gen4_six_cell_r6_statistical_analysis.py

but Phase F must not depend on mutable outcome values or R6 result artifacts.

No R6 scientific result may be used as a Phase F expected answer.

## 12. Deterministic output contract

A future canonical execution must be capable of producing exactly:

phase_f_pair_level_contrasts.csv
phase_f_primary_confirmatory_results.csv
phase_f_statistical_analysis_manifest.json
phase_f_statistical_analysis_report_candidate.md

PRIMARY_RESULT_ROWS =
15

PAIR_LEVEL_RESULT_ROWS =
4500

PRIMARY_RESULT_ORDER =
ENDPOINT_ORDER_THEN_ESTIMAND_ORDER

PAIR_LEVEL_RESULT_ORDER =
ENDPOINT_ORDER_THEN_ESTIMAND_ORDER_THEN_LEXICOGRAPHIC_SOURCE_PAIR_ID

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

## 13. Synthetic test requirements

Tests must use synthetic fixtures only.

They must cover at minimum:

1. exact 12-row per-pair endpoint matrix acceptance;
2. duplicate identity rejection;
3. missing anchor/cell rejection;
4. unexpected anchor/cell rejection;
5. wrong layer rejection;
6. non-finite endpoint rejection;
7. boolean endpoint rejection;
8. all five contrast formulas;
9. all three endpoints;
10. 4500 pair-level row cardinality for 300 synthetic pairs;
11. lexicographic pair ordering;
12. N=300 enforcement;
13. sample mean;
14. sample standard deviation;
15. standard error;
16. median/minimum/maximum;
17. two-sided one-sample Student-t statistic;
18. df=299;
19. two-sided Student-t p-value;
20. 95 percent Student-t confidence interval;
21. d_z;
22. zero-variance hard blocker;
23. exact global Holm adjustment across 15 hypotheses;
24. Holm step-down decision semantics;
25. proof that endpoint-wise Holm is not used;
26. deterministic CSV bytes;
27. canonical JSON bytes;
28. deterministic report generation;
29. canonical-input path is not opened by synthetic tests.

Tests must not read:

reports/reason_router_gen4_native_mamba_state_extraction_bff0a75_v1/kinematic_endpoints.jsonl

or any Phase F canonical result artifact.

## 14. Required validation

Before implementation freeze review:

python -m py_compile scripts/reason_router_gen4_native_mamba_state_phase_f_statistical_analysis.py

python -m pytest -q tests/test_reason_router_gen4_native_mamba_state_phase_f_statistical_analysis.py

git diff --check

All must pass.

Passing synthetic tests does not authorize canonical statistical execution.

## 15. Promotion condition

The implementation may be frozen only when:

- exactly the two authorized implementation/test files are changed;
- py_compile passes;
- dedicated synthetic tests pass;
- git diff --check passes;
- canonical endpoint values were not inspected by implementation tests;
- no canonical Phase F statistical result was produced;
- no unrelated tracked file changed.

After implementation freeze, a separate:

GEN4_NATIVE_MAMBA_STATE_BRIDGE_PHASE_F_STATISTICAL_ANALYSIS_EXECUTION_AUTHORITY

must bind the exact implementation commit and script SHA256 before any
canonical 15-hypothesis computation.

## 16. Stop condition

Stop after implementation and synthetic validation are created, validated,
reviewed, and frozen.

Do not compute any canonical pair-level contrast.

Do not compute any canonical p-value.

Do not inspect canonical Phase F effect directions.

Do not use Kaggle.

Do not use GPU.

SCIENTIFIC_CONCLUSION =
NONE

NEXT_PHASE =
PHASE_F_STATISTICAL_ANALYSIS_IMPLEMENTATION_AND_SYNTHETIC_VALIDATION
