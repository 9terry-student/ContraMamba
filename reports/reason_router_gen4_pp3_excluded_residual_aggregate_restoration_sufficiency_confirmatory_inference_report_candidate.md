# Gen4 PP3-Excluded Residual Aggregate Restoration Sufficiency — Confirmatory Inference

## Status

`PP3_EXCLUDED_RESIDUAL_AGGREGATE_RESTORATION_SUFFICIENCY_OVER_QUARTER_TURN_REPLACEMENT_SUPPORTED_ON_FRESH_XG1_HOLDOUT`

## Frozen provenance

Execution HEAD:

`92efdd06974f3db96937d07f8df00b8ca0fea6ff`

Accepted raw run:

`g4k-residual-aggregate-restoration-sufficiency-xg1-2401-2700-92efdd0-retry3`

Population:

`xg1_fact_2401..xg1_fact_2700`

N:

`300`

df:

`299`

Scientific model forwards in accepted raw run:

`36000`

Baseline model forwards:

`0`

Confirmatory inference executed during raw run:

`False`

This report records the single post-import confirmatory inference authorized by the execution freeze.

## Raw artifact validation

Frozen raw validator:

`PASS`

Imported artifact file count:

`4`

Raw artifact SHA256:

- `SHA256SUMS.txt`: `4bedd0b89f2e383a7cfbea11ffa1c7e55bf4ef454ac8717e7cfc07b9938e5e9f`
- `artifact_manifest.json`: `24efaa5b982ab8b6ce2b5c0796b1b6b46e9f1cdfe047a07aab5c8742eaadb064`
- `pp3_excluded_residual_aggregate_restoration_sufficiency_items.jsonl`: `9d5dabbef82a8fcfaccc4e610bbb4d91f1e7ee9ea2f2627a92f999c88034ac49`
- `pp3_excluded_residual_aggregate_restoration_sufficiency_summary.json`: `e38d8c3943594e934535ecf1682788868f815e28630e4ac44cc92864f44bd7d9`

## Frozen confirmatory test

Primary endpoint:

`D_RES_SUF = Q0 - Q_C`

Hypotheses:

`H0: mean(D_RES_SUF) <= 0`

`H1: mean(D_RES_SUF) > 0`

Test:

`one-sample one-sided Student t-test`

Alternative:

`greater`

N:

`300`

df:

`299`

Alpha:

`0.05`

Multiplicity correction:

`none`

Confirmatory p-value count:

`1`

## Results

`mean(Q0) = 1.8756264438819782e-07`

`mean(S_R) = 6.7896230673924773e-08`

`mean(D_RES_SUF) = 6.2820798711523715e-08`

`SD(D_RES_SUF) = 4.7373154473562475e-08`

`t(299) = 22.968454676401052`

`p_one_sided = 2.7574959205716982e-68`

## Positive-label gates

`mean(Q0) > 0`: `True`

`mean(S_R) > 0`: `True`

`mean(D_RES_SUF) > 0`: `True`

`p_one_sided < 0.05`: `True`

All positive-label gates:

`True`

## Confirmatory conclusion

`PP3_EXCLUDED_RESIDUAL_AGGREGATE_RESTORATION_SUFFICIENCY_OVER_QUARTER_TURN_REPLACEMENT_SUPPORTED_ON_FRESH_XG1_HOLDOUT`

If supported, the result supports only the bounded claim that the complete pre-specified PP3-excluded residual principal-plane subspace has an aggregate local restoration-sufficiency contribution to the frozen layer-17 / target-token native-Mamba susceptibility endpoint relative to the pre-specified equal-norm orthogonal within-R quarter-turn replacement from the same complete residual-neutralized background.

It does not establish additive equality between aggregate and individual effects, independence or absence of cross-plane interactions, plane ranking or dominance, causal status of the XG2-like template direction, behavioral or downstream sufficiency, benchmark improvement, or universality.

No individual-plane test, interaction test, additive-decomposition test, subgroup analysis, alternative tail, rescue analysis, or additional confirmatory p-value was executed.

## Inference runtime

Python:

`3.13.2`

NumPy:

`2.5.3`

SciPy:

`1.18.1`

Scientific model forwards during confirmatory inference:

`0`

Checkpoint loads during confirmatory inference:

`0`

GPU required for confirmatory inference:

`False`
