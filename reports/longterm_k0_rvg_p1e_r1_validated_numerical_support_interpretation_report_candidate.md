# K0-RVG-P1E-R1 Validated Numerical-Support Interpretation Report Candidate

**Status:** validated numerical-support interpretation candidate.

**Frozen R1E execution-authority commit:**

`f7646d7f8103f9d955c7eca3c53d8edf39e8f2fe`

**Validated diagnostic artifact SHA256:**

`0ca27d99bbbf7a8bb6e48261189cebd8ac955d5fad951c2672f593587d9af336`

**Validated diagnostic classification:**

`EXACT_RECURRENCE_INTACT_FLOAT32_REARRANGEMENT_TOLERANCE_FAILURE`

**Validated local support profile:**

`SINGLE_COORDINATE_FAILURE_WITHIN_FIXED_BRANCH`

This report interprets only the already validated R1 diagnostic artifact.

It does not authorize a rerun.

It does not authorize a tolerance change.

It does not authorize P1 endpoint computation.

It does not establish any P1 scientific endpoint result.

## 1. Validation status carried into interpretation

The diagnostic artifact passed:

`R1_ARTIFACT_SET = PASS_EXACT_ONE_FILE`

`R1_ARTIFACT_JSON_PARSE = PASS`

`R1_ARTIFACT_SCHEMA = PASS`

`R1_ARTIFACT_PROVENANCE = PASS`

`R1_ARTIFACT_LOCAL_ITEM_BINDING = PASS`

`R1_ARTIFACT_COORDINATE_COUNT = PASS_9`

`R1_ARTIFACT_NUMERICAL_INTERNAL_CONSISTENCY = PASS`

`R1_ARTIFACT_CLASSIFICATION_RECOMPUTATION = PASS`

`R1_ARTIFACT_SUPPORT_RECOMPUTATION = PASS`

`R1_ARTIFACT_SAFETY_FLAGS = PASS`

Therefore numerical interpretation may proceed on the frozen diagnostic evidence.

## 2. Fixed diagnostic scope

The validated R1 diagnostic covered exactly:

`R1_LOCAL_TEMPLATE_INDEX = 0`

`R1_BRANCH_ROLE = MATCHED_CORR`

`R1_PRIMARY_LAYER = 23`

`R1_TARGET_COORDINATE_COUNT = 9`

The captured token indices were:

`41, 42, 43, 44, 45, 46, 47, 48, 49`

Only this fixed branch/scope is interpreted here.

No population-wide prevalence inference is authorized.

## 3. Primary validated result

Across the nine fixed layer-23 coordinates:

- exact recurrence pass:
  `9 / 9`;
- frozen allclose pass:
  `8 / 9`;
- frozen allclose fail:
  `1 / 9`;
- first and only failing token:
  `49`.

Validated classification:

`EXACT_RECURRENCE_INTACT_FLOAT32_REARRANGEMENT_TOLERANCE_FAILURE`

Validated local support profile:

`SINGLE_COORDINATE_FAILURE_WITHIN_FIXED_BRANCH`

This establishes that the original P1E failure was not caused by failure of the exact native recurrence reconstruction at the first failing record.

## 4. Exact recurrence integrity

At token 49:

`exact_recurrence_pass = true`

The frozen recurrence identity:

`G * S_prev + W == S_post`

passed exact equality before the velocity-rearrangement tolerance failure.

Therefore:

`R1_FAILING_RECORD_NATIVE_RECURRENCE_INTEGRITY = PASS`

Within the validated record, there is no evidence of:

- missing state capture;
- wrong token coordinate;
- wrong layer;
- corrupted `S_prev`;
- corrupted `G`;
- corrupted `W`;
- corrupted `S_post`;
- recurrence-source mismatch.

This does not prove that every possible future population record would pass.

## 5. Failure sparsity

At token 49:

`element_count = 24576`

`failing_element_count = 1`

`failing_element_fraction = 4.0690104166666664e-05`

Thus the hard allclose failure was caused by exactly one element among 24,576 elements.

Therefore:

`R1_FAILURE_SPARSITY = ONE_OF_24576_ELEMENTS`

The other 24,575 elements satisfied the frozen elementwise tolerance at the failing coordinate.

## 6. Failure magnitude

At token 49:

`max_abs_residual = 1.9073486328125e-06`

The maximum-residual element had:

`tolerance_scale = 1.4385375379788456e-06`

and:

`scaled_residual = 1.3258942365646362`

Thus the failing element exceeded the frozen tolerance by approximately:

`1.3258942365646362 x`

on the frozen scaled-residual measure.

The exceedance is real under the original float32 hard guard.

It is not a schema, provenance, or coordinate-selection artifact.

## 7. Global vector-scale context

At token 49:

`residual_frobenius_norm = 2.3294188807773285e-06`

`v_raw32_frobenius_norm = 44.18905166220619`

`v_rearr32_frobenius_norm = 44.1890516110915`

`relative_frobenius_residual = 5.271484209672739e-08`

Therefore the two velocity vectors are globally extremely close in Frobenius geometry.

The hard failure was driven by a single elementwise tolerance exceedance, not by a large vector-level discrepancy.

Result:

`R1_VECTOR_LEVEL_DISCREPANCY = EXTREMELY_SMALL`

This is a numerical-support statement, not a P1 endpoint statement.

## 8. Failing-element arithmetic

At the single maximum-residual element:

`flat_index = 22767`

Captured float32 values:

`G = 0.005250484216958284`

`S_prev = 25.219133377075195`

`W = 25.130573272705078`

`S_post = 25.262985229492188`

Raw velocity:

`V_raw32 = S_post - S_prev`

`V_raw32 = 0.04385185241699219`

Rearranged float32 velocity:

`V_rearr32 = (G - 1) * S_prev + W`

`V_rearr32 = 0.043853759765625`

Difference:

`D32 = -1.9073486328125e-06`

The rearranged expression combines a large negative carry term with a large positive write term to produce a much smaller net velocity.

Using the captured float32 scalar snapshots but evaluating the arithmetic in float64 gives approximately:

`(G - 1) * S_prev = -25.086720715313497`

and:

`V_rearr64_snapshot = 0.043852557391581115`

while:

`V_raw64_snapshot = 0.04385185241699219`

giving:

`D64_snapshot = -7.049745889275982e-07`

This reduction is diagnostic evidence for finite-precision intermediate-rounding / cancellation sensitivity.

## 9. Float64 snapshot result

For the full token-49 tensor, using only captured float32 snapshots converted to float64:

`float64_max_abs_residual = 7.049745889275982e-07`

`float64_residual_frobenius_norm = 1.189814088656077e-06`

`float64_v_raw_frobenius_norm = 44.189051702360686`

`float64_relative_frobenius_residual = 2.6925540214580218e-08`

The float64 snapshot calculation reduces the maximum residual below `1e-6`.

No float64 model rerun occurred.

No scientific geometry was changed.

This result isolates arithmetic evaluation precision as a material contributor to the original hard-guard failure.

## 10. Cancellation / rounding interpretation

The failing element has:

- `S_prev` around `25.22`;
- write term `W` around `25.13`;
- `G` near `0.00525`;
- resulting carry contribution near `-25.09`;
- final velocity near `0.04385`.

Thus the rearranged expression forms the final velocity by subtractive cancellation between terms roughly three orders of magnitude larger than their residual mismatch.

The validated evidence supports:

`R1_NUMERICAL_MECHANISM = FLOAT32_INTERMEDIATE_ROUNDING_UNDER_CANCELLING_REARRANGEMENT`

This mechanism is consistent with:

1. exact native recurrence equality passing;
2. only one element failing;
3. very small vector-level relative residual;
4. lower residual under float64 snapshot arithmetic.

## 11. Root-cause conclusion

For the original first P1E failure record:

`R1_ORIGINAL_FAILURE_ROOT_CAUSE_ESTABLISHED = YES_FOR_FIXED_FAILURE_RECORD`

The validated root cause is:

> A hard elementwise float32 allclose guard on an algebraically rearranged velocity expression rejected one otherwise exact-recurrence record because finite-precision intermediate arithmetic under cancellation produced a local residual above the frozen tolerance.

Therefore the original failure is not supported as:

- native recurrence corruption;
- state-capture corruption;
- wrong layer/token binding;
- P0 provenance failure;
- P1 endpoint support failure.

It is supported as an instrument-level numerical guard failure.

## 12. Scope limitation

The diagnostic does **not** establish:

- how often this occurs across all 336 P1 items;
- whether another item would exhibit a larger residual;
- whether all future records would pass exact recurrence;
- the population distribution of scaled residuals;
- any P1 turning endpoint;
- any P1 coherence endpoint.

Therefore:

`R1_POPULATION_WIDE_NUMERICAL_PREVALENCE = NOT_ESTABLISHED`

No population-wide diagnostic sweep is authorized by this report.

## 13. Consequence for the frozen P1 scientific estimand

The P1 primary velocity is defined from captured native states as:

`V_t = S_t - S_(t-1)`

The failing hard check concerns the auxiliary algebraic rearrangement:

`(G_t - 1) * S_(t-1) + W_t`

The exact native recurrence itself passed.

Therefore the validated failure does not invalidate the mathematical definition of the preregistered raw velocity from `S_post - S_prev`.

Result:

`P1_PRIMARY_RAW_VELOCITY_DEFINITION_INVALIDATED = NO`

This does not authorize resumption of P1 execution.

## 14. Consequence for the hard validation guard

The validated evidence shows that:

- exact native recurrence is a direct integrity check on the captured recurrence tuple;
- the rearrangement identity is algebraically exact but not operation-order invariant in finite-precision float32 arithmetic;
- a hard elementwise tolerance on the rearranged expression can reject an exact-recurrence record due to intermediate rounding.

Therefore:

`R1_HARD_REARRANGEMENT_GUARD_SHOWN_TO_BE_NUMERICALLY_OVERCONSTRAINING = YES_FOR_VALIDATED_RECORD`

This is an instrument-contract conclusion.

It is not a license to select a looser tolerance from the observed value.

## 15. No tolerance retuning

The diagnostic observed:

`max_scaled_tolerance_residual = 1.3258942365646362`

This value must not be used to choose:

- a new `atol`;
- a new `rtol`;
- a safety multiplier;
- a percentile threshold;
- a minimum passing threshold.

Result:

`POST_OBSERVATION_TOLERANCE_RETUNING_AUTHORIZED = NO`

## 16. Preferred correction principle

The next correction stage should preserve the scientific estimand and avoid data-dependent tolerance tuning.

The preferred correction principle is:

1. keep exact native recurrence reconstruction as the hard capture/integrity contract;
2. keep primary velocity as `S_post - S_prev`;
3. retain rearrangement residuals as numerical diagnostics;
4. do not use rearrangement allclose as a hard scientific-execution blocker when exact recurrence has passed;
5. do not change layer, window, metric, population, support threshold, endpoint formulas, or statistical tests.

This principle follows from the validated numerical diagnosis rather than from P1 endpoint outcomes.

## 17. Required authority before correction

No code modification is authorized by this interpretation report.

A separate correction authority/specification is required before changing:

- the frozen observer;
- the P1 runner;
- any validation path;
- execution-authority logic.

The correction stage must independently validate that the scientific estimand is unchanged.

## 18. Scientific-result boundary

The R1 diagnostic and this interpretation report do not compute:

- `X_turn`;
- `X_coh`;
- phase-block scores;
- sign-test p-values;
- Holm-adjusted p-values;
- overall P1 raw-vector verdict.

Therefore:

`P1_SCIENTIFIC_CONCLUSION = NONE`

The original P1 scientific question remains unresolved.

## 19. Branch state

`K_SUCCESSOR_BRANCH_SELECTION = DEFERRED`

`BRANCH_A_STATUS = PARKED_NOT_CLOSED`

`BRANCH_B_STATUS = PARKED_NOT_ACTIVE`

No successor branch is activated by this numerical diagnosis.

## 20. K4 boundary

`K4_EXECUTION_AUTHORIZED = NO`

## 21. Final interpretation markers

`R1_DIAGNOSTIC_ARTIFACT_VALIDATED = YES`

`R1_VALIDATED_CLASSIFICATION = EXACT_RECURRENCE_INTACT_FLOAT32_REARRANGEMENT_TOLERANCE_FAILURE`

`R1_VALIDATED_LOCAL_SUPPORT_PROFILE = SINGLE_COORDINATE_FAILURE_WITHIN_FIXED_BRANCH`

`R1_VALIDATED_FAILING_COORDINATE_COUNT = 1`

`R1_VALIDATED_FIRST_FAILING_TOKEN_INDEX = 49`

`R1_VALIDATED_MAX_SCALED_RESIDUAL = 1.3258942365646362`

`R1_VALIDATED_MEDIAN_SCALED_RESIDUAL = 0.1367410272359848`

`R1_FAILING_RECORD_NATIVE_RECURRENCE_INTEGRITY = PASS`

`R1_FAILURE_SPARSITY = ONE_OF_24576_ELEMENTS`

`R1_VECTOR_LEVEL_DISCREPANCY = EXTREMELY_SMALL`

`R1_NUMERICAL_MECHANISM = FLOAT32_INTERMEDIATE_ROUNDING_UNDER_CANCELLING_REARRANGEMENT`

`R1_ORIGINAL_FAILURE_ROOT_CAUSE_ESTABLISHED = YES_FOR_FIXED_FAILURE_RECORD`

`R1_POPULATION_WIDE_NUMERICAL_PREVALENCE = NOT_ESTABLISHED`

`P1_PRIMARY_RAW_VELOCITY_DEFINITION_INVALIDATED = NO`

`R1_HARD_REARRANGEMENT_GUARD_SHOWN_TO_BE_NUMERICALLY_OVERCONSTRAINING = YES_FOR_VALIDATED_RECORD`

`POST_OBSERVATION_TOLERANCE_RETUNING_AUTHORIZED = NO`

`P1_SCIENTIFIC_CONCLUSION = NONE`

`K_SUCCESSOR_BRANCH_SELECTION = DEFERRED`

`K4_EXECUTION_AUTHORIZED = NO`

`READY_FOR_P1E_R2_NUMERICAL_GUARD_CORRECTION_AUTHORITY_DRAFT = YES`

`NEXT_STAGE = K0-RVG-P1E-R2_NUMERICAL_GUARD_CORRECTION_AUTHORITY_DRAFT`

This interpretation report becomes authoritative only after it is committed and pushed as the immediate one-file child of `f7646d7f8103f9d955c7eca3c53d8edf39e8f2fe`.
