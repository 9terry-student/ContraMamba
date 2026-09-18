# Gen4 PP3-Excluded Residual Aggregate Necessity — Confirmatory Result

## Status

`PP3_EXCLUDED_RESIDUAL_AGGREGATE_NECESSITY_OVER_QUARTER_TURN_MATCHED_CONTROL_SUPPORTED_ON_FRESH_XG1_HOLDOUT`

This report records the completed prospective confirmatory aggregate residual
necessity experiment on the fresh XG1 holdout.

The result is positive under the frozen design and exactly one confirmatory
one-sided Student t-test.

## Scientific question

Does neutralizing the complete frozen PP3-excluded principal-plane residual
subspace

`R = P1 ⊕ P2 ⊕ P4 ⊕ P5`

reduce the frozen layer-17 / target-token susceptibility endpoint more than an
equal-norm, orthogonal, response-blind quarter-turn perturbation in the same
residual subspace?

This experiment tests aggregate local causal necessity contribution of the
PP3-excluded residual subspace.

It does not test individual causal necessity of P1, P2, P4, or P5.

## Frozen design

Design commit:

`c4518d20f4417ca9f057fbd4940c28539e4ffb2c`

Static preparation commit:

`4a7698264488e811370bdf071c3cde73735757e0`

Implementation authority commit:

`7356e81d34b2883e74b8fa24b7751f725d9ca1db`

Original implementation commit:

`f8226d9bc0b075d1ce477389d3a3171e7e7c2470`

Original execution authority:

`02f6f862e2177e0d1e2ee58367bf916be0569094`

Endpoint defect correction:

`d8cd5d63f528fb397cca10acde149b9b96a261cc`

Retry execution freeze:

`5c75b975ab30c05e4d0fdafda1b69b3ab62b0dcb`

## Population

Prospective fresh XG1 holdout:

`xg1_fact_1501..xg1_fact_1800`

Pair count:

`N = 300`

The holdout was frozen before scientific execution.

## Conditions

Exactly three conditions were run in frozen order:

1. `native`
2. `residual_neutralized`
3. `quarter_turn_control`

Treatment neutralized the complete native residual component in:

`R = P1 ⊕ P2 ⊕ P4 ⊕ P5`

using branch-local native residual coefficients.

The matched control independently quarter-turned each native residual
coefficient pair by 90 degrees, producing an equal-norm and orthogonal
correction in the same residual subspace.

The matched control was response-blind and used no outcome-guided weighting or
plane selection.

## Endpoint

For each item:

`Q0 = Q(native)`

`QR = Q(residual_neutralized)`

`QC = Q(quarter_turn_control)`

`A_R = Q0 - QR`

`A_C = Q0 - QC`

Primary confirmatory endpoint:

`D_RES_NEC = A_R - A_C = QC - QR`

The corrected runner records the frozen canonical form:

`D_RES_NEC = QC - QR`

No endpoint definition changed after observing scientific outcomes.

## Execution

Run name:

`g4k-residual-aggregate-necessity-xg1-1501-1800-5c75b97-retry2`

Execution HEAD:

`5c75b975ab30c05e4d0fdafda1b69b3ab62b0dcb`

Run command SHA256:

`829dc6a6a5f17f1888c2827ca86fa4c680d8e4468d04a6ec36f4e246abc5a50c`

Run log SHA256:

`4fa7f9777c2886dd1c0bb4e405e68e67db1528f5a5feef2983fae533d5d43a72`

Run meta SHA256:

`459d7561a522c53a3602c7f9c39af75a9a79ca472d2be1821600314f00b33d84`

Imported handoff ZIP SHA256:

`2c25e967b0f148906cd0d227ef6bb20eb9b3d914027ee80baf196bce61a6a41d`

Execution result:

`PASS_PP3_EXCLUDED_RESIDUAL_AGGREGATE_NECESSITY_RAW_OBSERVATION`

GPU count:

`2`

Shard 0:

`xg1_fact_1501..xg1_fact_1650`

Scientific forwards:

`18000`

Shard 1:

`xg1_fact_1651..xg1_fact_1800`

Scientific forwards:

`18000`

Total scientific model forwards:

`36000`

Baseline model forwards:

`0`

Raw primary inference:

`False`

Raw scientific conclusion:

`None`

## Raw artifact validation

Imported raw files:

1. `SHA256SUMS.txt`
2. `artifact_manifest.json`
3. `pp3_excluded_residual_aggregate_necessity_items.jsonl`
4. `pp3_excluded_residual_aggregate_necessity_summary.json`

Import validation:

`PASS`

Frozen runner artifact validation:

`PASS`

Item count:

`300`

Endpoint revalidation:

`PASS`

All 300 items satisfied the canonical frozen endpoint identity:

`D_RES_NEC = QC - QR`

No raw artifact contained a statistical conclusion or primary inference.

## Confirmatory test

Frozen hypotheses:

`H0: mean(D_RES_NEC) <= 0`

`H1: mean(D_RES_NEC) > 0`

Test:

one-sample Student t-test, one-sided greater

Sample size:

`N = 300`

Degrees of freedom:

`df = 299`

Alpha:

`0.05`

Confirmatory p-value count:

`1`

No additional p-values, subgroup tests, tail analyses, rescue analyses, or
outcome-guided plane tests were executed.

## Confirmatory results

Mean native endpoint:

`mean(Q0) = 1.8699048379014811e-07`

Mean aggregate residual-neutralization attenuation:

`mean(A_R) = 6.6142907048184626e-08`

Mean matched-control-adjusted residual necessity endpoint:

`mean(D_RES_NEC) = 4.9220194461529442e-08`

Sample standard deviation:

`sd(D_RES_NEC) = 3.6836066499708061e-08`

Student t statistic:

`t(299) = 23.143588788576249`

One-sided confirmatory p-value:

`p = 6.4276188677606773e-69`

## Frozen positive gates

Gate 1:

`mean(Q0) > 0`

Result:

`PASS`

Gate 2:

`mean(A_R) > 0`

Result:

`PASS`

Gate 3:

`mean(D_RES_NEC) > 0`

Result:

`PASS`

Gate 4:

one-sided `p < 0.05`

Result:

`PASS`

All four prospectively frozen positive gates passed.

## Scientific conclusion

The fresh XG1 holdout supports the claim that the complete PP3-excluded
residual principal-plane subspace

`R = P1 ⊕ P2 ⊕ P4 ⊕ P5`

makes an aggregate local causal necessity contribution to the frozen
layer-17 / target-token native-Mamba susceptibility endpoint beyond the
pre-specified equal-norm orthogonal quarter-turn matched control.

This result strengthens the previously established picture:

- PP3 is a shared causal core;
- the PP3-excluded residual is structured and prospectively transportable;
- the aggregate PP3-excluded residual subspace is also locally necessary
  relative to the frozen matched within-R perturbation.

## Interpretation boundary

This result does **not** establish:

- individual necessity of P1;
- individual necessity of P2;
- individual necessity of P4;
- individual necessity of P5;
- individual sufficiency of any residual plane;
- aggregate residual sufficiency;
- causal necessity or sufficiency of the XG2-like residual template
  orientation itself;
- that XG1 and XG2 are globally identical;
- behavioral or downstream task effects;
- model-wide or architecture-wide universality.

The experiment identifies an aggregate causal contribution of the complete
PP3-excluded residual subspace under the frozen local intervention and endpoint.

It does not select or promote an individual secondary plane.

## Prior failed retry provenance

The earlier run

`g4k-residual-aggregate-necessity-xg1-1501-1800-02f6f86-retry1`

failed on the internal exact-float assertion:

`D_RES_NEC_INTERNAL`

before any pair item or shard artifact was written.

That failure was diagnosed as an IEEE-754 implementation defect in a redundant
exact equality check between mathematically equivalent subtraction paths.

The correction changed only the canonical computation of the already frozen
endpoint and added a regression test.

The failed run produced no scientific artifact and no scientific conclusion.

## Final status

Raw execution:

`PASS`

Artifact/provenance validation:

`PASS`

Confirmatory inference:

`PASS`

Frozen positive gates:

`4 / 4 PASS`

Scientific conclusion:

`PP3_EXCLUDED_RESIDUAL_AGGREGATE_NECESSITY_OVER_QUARTER_TURN_MATCHED_CONTROL_SUPPORTED_ON_FRESH_XG1_HOLDOUT`
