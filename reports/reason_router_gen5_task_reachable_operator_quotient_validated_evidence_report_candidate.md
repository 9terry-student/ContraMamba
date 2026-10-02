# Gen5 Task-Reachable Operator Quotient Validated Evidence Report Candidate

## Status

VALIDATED_TASK_REACHABLE_OPERATOR_QUOTIENT_EVIDENCE_CANDIDATE

## Evidence identity

Task-reachable quotient audit authority commit:

`c731270221c4e0e131fb68173f40bf0ad8a2bfdd`

Source functional-equivalence evidence freeze:

`5f079a66f7b0eb0caea30a8d5bc9a0fe757cc449`

Source functional-fingerprint execution:

`17a783c61e310d6298d4105233ea3f1b71ec8add`

Source causal-geometry evidence freeze:

`22f465803f592e87c7ffc3a0fe0e0dac47a7ebc4`

Source causal intervention execution:

`87f82551c721f953f710cd5dc102aca23161c4e7`

Run:

`gen5-task-reachable-operator-quotient-c731270-r1`

Pinned command SHA256:

`a4a8cb9411a8be257026891cb60ed532dc291fbc274d025a1252f1d575a4b9cb`

Imported ZIP SHA256:

`33d8387f3ed488a02776f23daa1dacd126792db9faae20fa17ee242a0c0fe8f3`

Collector status:

`PASS`

Import status:

`PASS`

Validated imported files:

`3`

## Execution boundary

The audit used only the frozen Phase3A P0 dev set:

- arm: `G5-C0`
- pressure: `P0`
- dev rows: `840`
- split seed: `16384`

Frozen dev encoding SHA256:

`e3162804bfd184907ee1b22b3f4b4cf3ecee1069fe55661a1a4dbaeecb2cca51`

Frozen dev row-order SHA256:

`b42f64ec4961907fb59eb5fdf9e2e1714649b7952e551c4e9abf7e99c1456e25`

Valid task tokens captured at the layer-22 wrapper input:

`60094`

Model forward count:

`1`

No training, backward pass, optimizer construction, checkpoint mutation, or
confirmatory 9601-9900 access occurred.

## Task-state spectrum

The layer-22 task-state Gram matrix showed strong anisotropy.

Participation-ratio effective dimension:

`3.58532279714`

Cumulative covariance-energy fractions:

- top 1: `0.514807209755`
- top 2: `0.616320258854`
- top 4: `0.672563012120`
- top 8: `0.736281281097`
- top 16: `0.801582397795`
- top 32: `0.862357183017`
- top 64: `0.923497942254`
- top 128: `0.970657505734`

These are descriptive properties of the frozen dev-state distribution only.
They do not establish a universal intrinsic dimension.

## Same A-init, different training RNG

Mean ambient row(A) affinity:

`0.999963824427`

Mean task read-signal mean-squared canonical correlation:

`0.999982674791`

Mean ambient operator cosine:

`0.999868757516`

Mean ambient operator normalized residual:

`0.0161880520363`

Mean task-restricted correction-action cosine:

`0.999908167884`

Mean task-restricted correction-action normalized residual:

`0.0136460447223`

Mean quotient suppression ratio:

`0.842068112411`

Thus the small residual differences remaining under fixed A-init are also small
after task restriction.

## Same training RNG, different A-init

Mean ambient row(A) affinity:

`0.144047932169`

Mean task read-signal mean-squared canonical correlation:

`0.557718413442`

Mean ambient operator cosine:

`0.450626226215`

Mean ambient operator normalized residual:

`1.04816095366`

Mean task-restricted correction-action cosine:

`0.904811006791`

Mean task-restricted correction-action normalized residual:

`0.459227092986`

Mean quotient suppression ratio:

`0.438154063771`

Ranges for the different-A / same-RNG task-restricted action:

- cosine: `[0.893701267317, 0.920933925814]`
- normalized residual: `[0.433699274866, 0.483131968378]`
- quotient ratio: `[0.418736990800, 0.464102979189]`

## Important metric distinction

Two nearby values answer different questions:

- `0.450626...` is the mean **ambient operator cosine** for same-RNG,
  different-A pairs.
- `0.459227...` is the mean **task-restricted correction-action normalized
  residual** for those pairs.

The latter is the unresolved residual that must be propagated through the
correction recurrence and downstream readout path.

It must not be interpreted as a fraction of semantic information.

## Primary result

Task restriction strongly suppresses the A-init-dependent ambient operator
difference but does not eliminate it.

For same-RNG, different-A pairs:

- ambient normalized residual: `1.04816095366`
- task-action normalized residual: `0.459227092986`
- ratio: `0.438154063771`

Equivalently, the task-reachable state restriction reduces the normalized
operator discrepancy to about 43.8% of its ambient value.

At the same time, operator cosine rises from about `0.4506` in ambient space to
about `0.9048` on task-reachable states.

Therefore there is a strong but incomplete task-manifold quotient effect.

## Relationship to the functional-equivalence result

The frozen functional-fingerprint evaluation established that the nine cells
were almost indistinguishable at the final dev-output level:

- same-A centered delta-logit cosine: `0.999999957562`
- same-RNG different-A centered delta-logit cosine: `0.999917998280`
- all pairwise prediction agreements in the tested groups: `1.0`

Therefore the remaining task-restricted layer-22 correction-action residual of
approximately `0.4592` cannot by itself imply a comparably large final
functional difference.

Additional collapse must occur after the raw layer-22 correction write.

## Supported bounded conclusions

`GEN5_TASK_REACHABLE_LAYER22_STATES_STRONGLY_SUPPRESS_A_INIT_DEPENDENT_AMBIENT_OPERATOR_DIFFERENCES_UNDER_FIXED_PHASE3A_P0`

`GEN5_TASK_REACHABLE_RESTRICTION_DOES_NOT_FULLY_COLLAPSE_DIFFERENT_A_INIT_CORRECTION_ACTIONS_AT_THE_RAW_LAYER22_WRITE_LEVEL`

`GEN5_SUBSTANTIAL_DIFFERENT_A_INIT_CORRECTION_ACTION_RESIDUAL_REMAINS_AT_LAYER22_DESPITE_NEAR_IDENTICAL_FINAL_DEV_FUNCTION`

`GEN5_ADDITIONAL_FUNCTIONAL_COLLAPSE_MUST_OCCUR_DOWNSTREAM_OF_THE_RAW_LAYER22_CORRECTION_WRITE_UNDER_THE_TESTED_CONTRACT`

## What is not established

The present evidence does not establish:

- that the remaining `0.4592` residual is decision-relevant;
- that it is decision-irrelevant;
- where the remaining residual is suppressed;
- exact global function equivalence;
- a formal gauge symmetry;
- a universal quotient space;
- a universal task-state intrinsic dimension;
- behavior outside the frozen Phase3A P0 dev distribution;
- behavior on confirmatory seeds 9601-9900;
- a population-level statistical claim.

## Next scientific action

Perform a no-training correction residual propagation localization over the
same frozen 3x3 checkpoint grid and the same frozen Phase3A P0 dev set.

For each checkpoint, measure the correction-specific signal at the exact
successive stages already defined by the frozen implementation:

1. `raw_write = B A x`
2. recurrent `correction_states`
3. `C` readout before gate
4. gated correction scan
5. `out_proj` layer-22 correction contribution
6. final correction-induced logits relative to the parent

For each pair, especially same-RNG / different-A pairs, compute stagewise:

- cosine;
- normalized residual;
- pairwise residual survival ratio relative to the preceding stage;
- cumulative survival ratio relative to the raw write.

The primary localization question is:

> At which frozen operation does the large different-A task-action residual
> first undergo the dominant collapse required to reconcile the approximately
> `0.4592` raw-write residual with the approximately `0.999918` final
> correction-induced logit cosine?

No new training, seed, rank sweep, optimizer change, architecture change, or
confirmatory population access is warranted before this localization.
