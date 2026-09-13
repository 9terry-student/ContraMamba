# ContraMamba Gen4 R6 Statistical Analysis Implementation Authority - Candidate

## 1. Status

STATUS =
CANDIDATE

PHASE =
GEN4_R6_STATISTICAL_ANALYSIS_IMPLEMENTATION_AUTHORITY

THIS_DOCUMENT_AUTHORIZES =
IMPLEMENTATION_AND_SYNTHETIC_VALIDATION_ONLY

CANONICAL_STATISTICAL_EXECUTION =
NOT_AUTHORIZED

## 2. Frozen authorities

GEN4_OUTCOME_STATISTICAL_TESTING_SPECIFICATION =
4dc5bacd10a254b5ecd339ac1fe78bad9def5c47

R5_VALIDATED_SCIENTIFIC_INFERENCE_ARTIFACT_FREEZE =
a3b5bcf2ded8dc0e86e859bbba12b5601a2fdea0

R5_EXECUTION_HEAD =
cf0826174c2ab1b2203f68afbdeed9da3ff64aa2

The statistical definitions in the frozen Gen4 outcome/statistical
specification are immutable.

This authority does not redefine the scientific question, outcome,
estimands, multiplicity family, inferential unit, or decision rules.

## 3. Frozen canonical R5 inputs

Canonical evaluator-row artifact:

reports/reason_router_gen4_r5_execution_cf0826174c2ab1b2203f68afbdeed9da3ff64aa2_retry1/r5_scientific_evaluator_rows.jsonl

SHA256:

e3157cb5e4e878e4fbe99914a689e57648568d18b4ae71162e3ba1278093204e

Bytes:

53510707

Expected rows:

32400

Expected unique key count:

32400

Required composite key:

(evaluator_seed, evaluator_arm, row_id)

Required multiplicity:

each row_id occurs exactly 18 times

each source_pair_id occurs exactly 108 times

Canonical R5 scientific summary:

reports/reason_router_gen4_r5_execution_cf0826174c2ab1b2203f68afbdeed9da3ff64aa2_retry1/r5_scientific_inference_summary.json

SHA256:

e6887d86c1ac2242ce5c2d74c379912fd81ac441d71b23a13d0db37412d2905b

Bytes:

4635

Canonical synthetic preflight summary:

reports/reason_router_gen4_r5_execution_cf0826174c2ab1b2203f68afbdeed9da3ff64aa2_retry1/r5_synthetic_preflight_summary.json

SHA256:

9b29036f54fbf6b78f0a0e987be4a799a9d0757784a3e7240cfe46c5abd1e26f

Bytes:

1043

The implementation may encode these identities as fail-closed constants.

During this implementation phase, tests must not read or statistically
inspect the canonical R5 evaluator-row artifact.

## 4. Implementation scope

Exactly one analysis implementation file is authorized:

scripts/reason_router_gen4_six_cell_r6_statistical_analysis.py

Exactly one dedicated test file is authorized:

tests/test_reason_router_gen4_six_cell_r6_statistical_analysis.py

No other existing source, test, report, model, checkpoint, tokenizer,
artifact, or workflow file may be modified by this implementation phase.

The implementation must be deterministic and CPU-only.

No Kaggle-specific dependency is permitted.

## 5. Primary analysis contract

PRIMARY_Y =
q_authorized

EVALUATOR_COUNT =
18

EVALUATOR_AGGREGATION =
EQUAL_WEIGHT_FIXED_POPULATION_MEAN

For source pair p and cell c:

Ybar_pc =
mean over the 18 prespecified evaluator q_authorized values.

PRIMARY_INFERENTIAL_UNIT =
SOURCE_PAIR

PRIMARY_PAIR_COUNT =
300

Evaluator observations must never be treated as independent inferential
replicates.

PSEUDOREPLICATION_BY_EVALUATOR =
FORBIDDEN

## 6. Exact confirmatory estimands

For every source pair p:

Delta_title(p) =
Ybar(p,C1_TITLE) - Ybar(p,C0_SHAM)

Delta_name(p) =
Ybar(p,C2_NAME) - Ybar(p,C0_SHAM)

Delta_role(p) =
Ybar(p,C3_ROLE) - Ybar(p,C0_SHAM)

Delta_predicate(p) =
Ybar(p,C4_PREDICATE) - Ybar(p,C0_SHAM)

Interaction_title_name(p) =
Ybar(p,C5_TITLE_NAME)
- Ybar(p,C1_TITLE)
- Ybar(p,C2_NAME)
+ Ybar(p,C0_SHAM)

Title_minus_name(p) =
Delta_title(p) - Delta_name(p)

The confirmatory family contains exactly these six estimands.

No additional confirmatory estimand is authorized.

## 7. Exact statistical procedure

For each confirmatory estimand:

N =
300

STATISTICAL_TEST =
TWO_SIDED_ONE_SAMPLE_STUDENT_T_TEST_AGAINST_ZERO

DF =
299

Report:

mean
sample standard deviation
standard error
median
minimum
maximum
t statistic
raw two-sided p-value
ordinary two-sided 95 percent Student-t confidence interval
paired standardized effect d_z

d_z =
mean / sample_sd

when sample_sd > 0.

When sample_sd = 0:

d_z =
UNDEFINED_ZERO_VARIANCE

The implementation must not invent an epsilon denominator.

## 8. Multiplicity

CONFIRMATORY_HYPOTHESIS_COUNT =
6

FAMILYWISE_ALPHA =
0.05

MULTIPLICITY_METHOD =
HOLM_BONFERRONI

The implementation must report both raw and Holm-adjusted p-values.

Confirmatory rejection status must be based only on:

Holm-adjusted p < 0.05

No uncorrected p-value may be used as the confirmatory decision criterion.

## 9. Secondary diagnostics

SECONDARY_Y_1 =
entitlement_prob

SECONDARY_Y_2 =
support_vs_best_nonsupport_logit_margin

These are descriptive only.

No secondary confirmatory p-value family is authorized.

Prediction may be summarized descriptively.

Evaluator-specific heterogeneity summaries may be implemented as descriptive
outputs only.

They may not alter, rank, drop, select, or reweight evaluators.

## 10. Required fail-closed validation

The implementation must reject canonical execution if any of the following
fails:

- input SHA256 identity;
- input byte-count identity where frozen;
- scientific summary result is not PASS_EXECUTION_MATRIX_PRODUCED;
- scientific_row_count is not 32400;
- unique_key_count is not 32400;
- complete_matrix_validation is not PASS;
- statistical_testing in the R5 summary is not false;
- duplicate composite keys exist;
- required evaluator coordinates are missing;
- any row_id does not occur exactly 18 times;
- any source_pair_id does not occur exactly 108 times;
- any source pair lacks one of the six exact contrast cells;
- any required q_authorized value is missing or non-finite;
- evaluator averaging does not use exactly 18 equal-weight observations;
- primary pair count is not exactly 300.

No imputation is permitted.

No partial-evaluator primary analysis is permitted.

No incomplete source-pair block is permitted.

## 11. Future canonical output contract

A later execution authority may permit canonical production of:

reports/reason_router_gen4_six_cell_r6_statistical_analysis_a3b5bcf/

with deterministic outputs including at minimum:

r6_primary_confirmatory_results.csv

r6_secondary_descriptive_results.csv

r6_statistical_analysis_summary.json

r6_statistical_analysis_report_candidate.md

The implementation must be capable of recording:

input artifact path
input SHA256
input byte count
statistical specification commit
R5 artifact freeze commit
row count
unique key count
pair count
evaluator count
all six raw statistics
all six raw p-values
all six Holm-adjusted p-values
all six confidence intervals
all six d_z values or zero-variance sentinel
confirmatory decisions
secondary descriptive summaries
scientific-scope limitations

No canonical R6 output may be produced during this implementation phase.

## 12. Test requirements

Tests must use synthetic fixtures only.

Tests must cover at minimum:

- exact equal-weight evaluator averaging;
- all six pair-level estimand formulas;
- N=300 enforcement;
- two-sided one-sample t calculation;
- df=299;
- 95 percent t confidence interval;
- d_z calculation;
- zero-variance d_z sentinel;
- exact Holm-Bonferroni adjusted p-values;
- Holm step-down rejection semantics;
- duplicate-key rejection;
- missing-evaluator rejection;
- missing-cell rejection;
- wrong row multiplicity rejection;
- wrong source-pair multiplicity rejection;
- non-finite primary outcome rejection;
- forbidden incomplete matrix rejection;
- deterministic serialization.

Tests must not inspect the canonical R5 evaluator outcomes.

## 13. Required implementation validation

Before any implementation freeze review:

python -m py_compile scripts/reason_router_gen4_six_cell_r6_statistical_analysis.py

and the dedicated R6 test suite must pass.

A later controller decision may additionally run relevant existing non-scientific
tests if needed.

Passing tests do not authorize canonical statistical execution.

## 14. Explicitly unauthorized operations

During this implementation phase:

CANONICAL_R5_OUTCOME_INSPECTION =
NOT_AUTHORIZED

CANONICAL_STATISTICAL_TESTING =
NOT_AUTHORIZED

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

KAGGLE_EXECUTION =
NOT_AUTHORIZED

GPU_EXECUTION =
NOT_AUTHORIZED

No model-side computation is needed for R6.

## 15. Promotion condition

The implementation may be frozen only after:

1. the exact two-file implementation scope is confirmed;
2. py_compile passes;
3. the dedicated synthetic R6 tests pass;
4. no canonical R5 outcome values were inspected by tests;
5. no unrelated tracked file was modified.

After implementation freeze, the next object is:

GEN4_R6_STATISTICAL_ANALYSIS_EXECUTION_AUTHORITY

That later authority must bind the exact frozen implementation commit before
the canonical 32400-row R5 artifact may be statistically analyzed.

## 16. Stop condition

Stop after the R6 statistical-analysis implementation and synthetic tests are
created, validated, reviewed, and frozen.

Do not perform the six confirmatory tests.

Do not compute canonical R6 descriptive results.

Do not inspect canonical pair-level effects.

Do not use Kaggle.

Do not use GPU.

SCIENTIFIC_STATISTICAL_CONCLUSION =
NOT_ESTABLISHED
