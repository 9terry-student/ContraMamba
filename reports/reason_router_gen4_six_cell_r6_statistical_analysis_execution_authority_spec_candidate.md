# ContraMamba Gen4 R6 Statistical Analysis Execution Authority - Candidate

## 1. Status

STATUS =
CANDIDATE

PHASE =
GEN4_R6_STATISTICAL_ANALYSIS_EXECUTION_AUTHORITY

THIS_DOCUMENT_AUTHORIZES =
ONE_CANONICAL_CPU_STATISTICAL_ANALYSIS_AFTER_THIS_AUTHORITY_IS_FROZEN

## 2. Frozen authority chain

GEN4_OUTCOME_STATISTICAL_TESTING_SPECIFICATION =
4dc5bacd10a254b5ecd339ac1fe78bad9def5c47

R5_VALIDATED_SCIENTIFIC_INFERENCE_ARTIFACT_FREEZE =
a3b5bcf2ded8dc0e86e859bbba12b5601a2fdea0

R6_IMPLEMENTATION_AUTHORITY =
5ae6d7cfd11f9d2b64617a145c7c248becd97aad

R6_STATISTICAL_ANALYSIS_IMPLEMENTATION =
32209bead3915e233ae3f4eca0090860cb245a68

No statistical definition may be changed by this execution authority.

## 3. Frozen R6 implementation

Implementation file:

scripts/reason_router_gen4_six_cell_r6_statistical_analysis.py

SHA256:

c58b90936f97d9daf823e320717ff721d98045017a9d8f652410b23ed9744a90

Test file:

tests/test_reason_router_gen4_six_cell_r6_statistical_analysis.py

SHA256:

e0e1d5e7f95c7dce8ab7d88be7f4443741accd2d10cad82e51b33ba5ea78d1ac

Frozen validation before implementation freeze:

PY_COMPILE =
PASS

SYNTHETIC_TESTS =
19_PASSED

Independent pre-execution numerical verification must also pass before this
candidate is created:

STUDENT_T_P_REFERENCE_CROSSCHECK =
PASS

DF299_CRITICAL_REFERENCE_CROSSCHECK =
PASS

ONE_SAMPLE_T_AND_CI_REFERENCE_CROSSCHECK =
PASS

HOLM_REFERENCE_CROSSCHECK =
PASS

ZERO_VARIANCE_SEMANTICS =
PASS

The independent cross-check uses direct numerical integration of the
Student-t density and does not inspect canonical R5 outcomes.

## 4. Canonical statistical input

Evaluator-row artifact:

reports/reason_router_gen4_r5_execution_cf0826174c2ab1b2203f68afbdeed9da3ff64aa2_retry1/r5_scientific_evaluator_rows.jsonl

SHA256:

e3157cb5e4e878e4fbe99914a689e57648568d18b4ae71162e3ba1278093204e

Bytes:

53510707

Required rows:

32400

Required unique composite keys:

32400

Scientific inference summary:

reports/reason_router_gen4_r5_execution_cf0826174c2ab1b2203f68afbdeed9da3ff64aa2_retry1/r5_scientific_inference_summary.json

SHA256:

e6887d86c1ac2242ce5c2d74c379912fd81ac441d71b23a13d0db37412d2905b

Bytes:

4635

Canonical execution must fail closed on any byte-identity mismatch.

## 5. Frozen statistical contract

PRIMARY_Y =
q_authorized

EVALUATOR_AGGREGATION =
EQUAL_WEIGHT_FIXED_POPULATION_MEAN

EVALUATOR_COUNT =
18

PRIMARY_INFERENTIAL_UNIT =
SOURCE_PAIR

PRIMARY_PAIR_COUNT =
300

CONFIRMATORY_HYPOTHESIS_COUNT =
6

SIDEDNESS =
TWO_SIDED

STATISTICAL_TEST =
ONE_SAMPLE_STUDENT_T_TEST_AGAINST_ZERO

DF =
299

CONFIDENCE_INTERVAL =
ORDINARY_TWO_SIDED_95_PERCENT_STUDENT_T

MULTIPLICITY_METHOD =
HOLM_BONFERRONI

FAMILYWISE_ALPHA =
0.05

CONFIRMATORY_REJECTION_RULE =
HOLM_ADJUSTED_P_VALUE_LESS_THAN_0_05

No evaluator pseudoreplication is permitted.

No evaluator dropping, promotion, weighting, or selection is permitted.

## 6. Exact confirmatory family

Exactly these six estimands are authorized:

delta_title

delta_name

delta_role

delta_predicate

interaction_title_name

title_minus_name

No post-hoc confirmatory estimand is authorized.

## 7. Secondary outputs

The following are descriptive only:

entitlement_prob

support_vs_best_nonsupport_logit_margin

prediction summaries where implemented

evaluator heterogeneity where implemented

No secondary confirmatory p-value family is authorized.

## 8. Authorized execution environment

EXECUTION_LOCATION =
LOCAL_CONTRAMAMBA_REPOSITORY

DEVICE =
CPU

GPU =
NOT_REQUIRED_AND_NOT_AUTHORIZED

KAGGLE =
NOT_REQUIRED_AND_NOT_AUTHORIZED

MODEL_INFERENCE =
NOT_AUTHORIZED

TOKENIZER_EXECUTION =
NOT_AUTHORIZED

CHECKPOINT_LOADING =
NOT_AUTHORIZED

TRAINING =
NOT_AUTHORIZED

BACKWARD =
NOT_AUTHORIZED

OPTIMIZER =
NOT_AUTHORIZED

The canonical R6 operation is statistical analysis over already frozen R5
outputs only.

## 9. Exact output location

The single authorized canonical output directory is:

reports/reason_router_gen4_six_cell_r6_statistical_analysis_a3b5bcf/

It must not exist before canonical execution.

Exactly the implementation-defined canonical result files are expected:

r6_primary_confirmatory_results.csv

r6_secondary_descriptive_results.csv

r6_statistical_analysis_summary.json

r6_statistical_analysis_report_candidate.md

## 10. Pre-execution gates

Before canonical execution, all of the following must hold:

1. this execution authority is frozen in Git;
2. local HEAD equals the frozen execution-authority commit;
3. local HEAD equals its upstream remote;
4. no tracked unstaged modification exists;
5. no staged file exists;
6. the implementation SHA256 equals the frozen implementation identity;
7. the canonical R5 JSONL SHA256 and byte count equal the frozen identity;
8. the canonical R5 summary SHA256 and byte count equal the frozen identity;
9. the output directory does not already exist;
10. the dedicated synthetic R6 suite still passes.

Unrelated untracked files outside the exact output directory do not authorize
scope expansion and must not be modified.

## 11. Single authorized canonical command

After all gates pass, exactly one canonical statistical execution is
authorized using the frozen implementation:

python scripts/reason_router_gen4_six_cell_r6_statistical_analysis.py run-canonical \
  --input-jsonl reports/reason_router_gen4_r5_execution_cf0826174c2ab1b2203f68afbdeed9da3ff64aa2_retry1/r5_scientific_evaluator_rows.jsonl \
  --input-summary reports/reason_router_gen4_r5_execution_cf0826174c2ab1b2203f68afbdeed9da3ff64aa2_retry1/r5_scientific_inference_summary.json \
  --output-dir reports/reason_router_gen4_six_cell_r6_statistical_analysis_a3b5bcf

No alternate input, alternate output directory, partial matrix, exploratory
rerun, threshold search, or additional hypothesis family is authorized.

## 12. Required post-execution validation

Execution success alone does not establish the scientific claim.

Before interpretation, verify at minimum:

result =
PASS_STATISTICAL_ANALYSIS_PRODUCED

row_count =
32400

unique_key_count =
32400

pair_count =
300

evaluator_count =
18

confirmatory_hypothesis_count =
6

familywise_alpha =
0.05

multiplicity_method =
HOLM_BONFERRONI

primary_inferential_unit =
SOURCE_PAIR

statistical_testing =
true

model_inference =
false

training =
false

backward =
false

The primary output must contain exactly six estimands and each must report:

N=300

df=299

mean

sample_sd

standard_error

median

minimum

maximum

t_statistic

raw_p_value

Holm-adjusted p-value

Holm rejection decision

95 percent confidence interval

d_z or the frozen zero-variance sentinel

## 13. Scientific interpretation boundary

A Holm-rejected estimand supports only the corresponding systematic
within-mechanism q_authorized effect for the frozen 300 source-pair population
under the fixed prespecified 18-evaluator population.

A non-rejected estimand means:

SYSTEMATIC_EFFECT_NOT_ESTABLISHED

It does not establish exact zero.

The analysis does not establish:

arbitrary-dataset generalization

arbitrary-model generalization

native-Mamba-state causality

training benefit

task-performance improvement

feature usefulness

mechanism-invariant generalization

## 14. Result handling

After canonical execution:

1. validate all four result files and their identities;
2. keep code correctness, execution success, artifact validity, and scientific
   interpretation distinct;
3. create no additional statistical tests before the frozen results are
   reviewed;
4. move to explicit result freeze review before commit/push.

No exploratory follow-up is automatically authorized by a significant or
non-significant result.

## 15. Stop condition

This authority permits one canonical CPU statistical analysis only after the
authority itself is frozen.

Stop after producing and validating the frozen-contract R6 outputs.

Do not change hypotheses.

Do not add tests.

Do not tune thresholds.

Do not use Kaggle.

Do not use GPU.

SCIENTIFIC_STATISTICAL_CONCLUSION =
NOT_YET_ESTABLISHED_UNTIL_VALIDATED_R6_OUTPUT_REVIEW
