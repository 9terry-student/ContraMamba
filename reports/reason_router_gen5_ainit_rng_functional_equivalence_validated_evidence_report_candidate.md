# Gen5 A-init RNG Functional Equivalence Validated Evidence Report Candidate

## Status

VALIDATED_FUNCTIONAL_EQUIVALENCE_EVIDENCE_CANDIDATE

## Evidence identity

Functional fingerprint evaluation authority commit:

`17a783c61e310d6298d4105233ea3f1b71ec8add`

Source causal-geometry evidence freeze:

`22f465803f592e87c7ffc3a0fe0e0dac47a7ebc4`

Source causal intervention execution:

`87f82551c721f953f710cd5dc102aca23161c4e7`

Source corrected implementation freeze:

`06a245f24e4474e2b039b954abd7b6cbf436e822`

Run:

`gen5-ainit-rng-functional-fingerprint-17a783c-r1`

Run command SHA256:

`811502f5b7168be4fcf66be290c2902a954cbc9ad3e70b265a52f44fd3494be0`

Imported ZIP SHA256:

`c3814a866fdc7eed3e41da508b7bf77c59af6c477e292975ac90a2c7a50e21c8`

Imported artifacts:

- `cell_fingerprints.pt`
- `functional_fingerprint_summary.json`
- `run_provenance.json`

Collector status:

`PASS`

Import status:

`PASS`

Validated imported files:

`3`

## Evaluation boundary

The evaluation used only the frozen Phase3A P0 dev set:

- arm: `G5-C0`
- pressure: `P0`
- dev rows: `840`
- split seed: `16384`

Frozen dev encoding SHA256:

`e3162804bfd184907ee1b22b3f4b4cf3ecee1069fe55661a1a4dbaeecb2cca51`

Frozen dev order SHA256:

`b42f64ec4961907fb59eb5fdf9e2e1714649b7952e551c4e9abf7e99c1456e25`

The exact frozen 3x3 checkpoint grid was evaluated:

- A6201-R6201
- A6201-R6202
- A6201-R6203
- A6202-R6201
- A6202-R6202
- A6202-R6203
- A6203-R6201
- A6203-R6202
- A6203-R6203

The parent baseline used the same frozen parent checkpoint with the layer-22 correction contribution disabled by exact-zero B.

No training, optimizer construction, backward pass, checkpoint mutation, or confirmatory 9601-9900 population access occurred.

## Parent baseline

Parent dev cross entropy:

`1.34096550941`

Parent dev accuracy:

`0.269047619048`

Each trained correction cell changed the prediction relative to parent on:

`835 / 840`

dev examples, corresponding to:

`0.994047619048`

of the dev set.

## Primary functional endpoints

For every cell pair, the correction-induced functional fingerprint was defined from:

`delta_logits(cell) = logits(cell) - logits(parent)`

Primary endpoint 1 used per-example class-centered delta logits before flattening.

Primary endpoint 2 used the three unique pairwise class-margin deltas per example.

### Same A-init, different training RNG

Mean centered-delta-logit cosine:

`0.999999957562`

Range:

`[0.999999815314, 0.999999997144]`

Mean margin-delta cosine:

`0.999999957562`

Range:

`[0.999999815314, 0.999999997144]`

Mean prediction agreement:

`1.0`

Mean parent-relative prediction-change-mask agreement:

`1.0`

### Same training RNG, different A-init

Mean centered-delta-logit cosine:

`0.999917998280`

Range:

`[0.999841628474, 0.999985224603]`

Mean margin-delta cosine:

`0.999917998280`

Range:

`[0.999841628486, 0.999985224604]`

Mean prediction agreement:

`1.0`

Mean parent-relative prediction-change-mask agreement:

`1.0`

### Descriptive factor contrasts

Same-A-init minus same-training-RNG centered-delta-logit cosine:

`0.0000819592813645`

Same-A-init minus same-training-RNG margin-delta cosine:

`0.0000819592818746`

Prediction-agreement contrast:

`0.0`

Prediction-change-mask-agreement contrast:

`0.0`

Thus A-init identity leaves a detectable but extremely small signature in the correction-induced dev-logit fingerprint, while all tested cells produce identical class predictions and identical parent-relative prediction-change masks on the frozen dev set.

## Cellwise task behavior

All nine cells reached dev accuracy:

`0.714285714286`

Dev CE remained seed-group dependent but tightly bounded:

- A6201 group: approximately `0.84248` to `0.84258`
- A6202 group: approximately `0.83892` to `0.83896`
- A6203 group: approximately `0.84150` to `0.84156`

The correction-induced centered-logit L2 norms also varied modestly by A-init group, but the full directional functional fingerprints remained nearly collinear across the entire 3x3 grid.

## Relationship to the causal geometry result

The preceding frozen causal-geometry experiment established:

Mean final row(A) affinity for same A-init / different training RNG:

`0.999963824427`

Mean final row(A) affinity for same training RNG / different A-init:

`0.144047932169`

Difference:

`0.855915892257`

Mean dominant-right cosine for same A-init / different training RNG:

`0.999984480062`

Mean dominant-right cosine for same training RNG / different A-init:

`0.535662952798`

Difference:

`0.464321527264`

Therefore the same intervention produces two sharply different observations:

1. A initialization strongly controls the final ambient read-side geometry.
2. The resulting solutions are nevertheless almost functionally indistinguishable on the frozen Phase3A dev distribution.

This is not consistent with a simple mechanism in which the A-init-selected ambient geometry uniquely determines a correspondingly distinct dev-level decision function.

## Combined interpretation

The strongest bounded interpretation is:

`GEN5_A_INITIALIZATION_SELECTS_DISTINCT_AMBIENT_READ_SIDE_GEOMETRIC_REPRESENTATIVES_THAT_ARE_NEARLY_FUNCTIONALLY_EQUIVALENT_ON_THE_FROZEN_PHASE3A_DEV_DISTRIBUTION`

The evidence supports a many-to-one relationship from learned ambient geometry to observed task behavior under the tested contract.

A initialization is therefore causally important for which read-side geometric representative is reached, but the frozen dev task admits substantial functional equivalence across those distinct representatives.

The residual functional differences are nonzero at the logit level and should not be described as exact global function identity.

## Supported bounded conclusions

`GEN5_A_INIT_CAUSALLY_SELECTS_FINAL_READ_SIDE_GEOMETRIC_IDENTITY_UNDER_FIXED_PHASE3A_P0_TRAINING`

`GEN5_DISTINCT_A_INIT_SELECTED_READ_SIDE_GEOMETRIES_ARE_NEARLY_FUNCTIONALLY_EQUIVALENT_ON_THE_FROZEN_PHASE3A_P0_DEV_SET`

`GEN5_FUNCTIONAL_EQUIVALENCE_IS_MUCH_STRONGER_THAN_AMBIENT_GEOMETRIC_EQUIVALENCE_ACROSS_A_INIT_LEVELS`

`GEN5_A_INIT_DEPENDENCE_OF_AMBIENT_GEOMETRY_DOES_NOT_IMPLY_COMPARABLY_LARGE_A_INIT_DEPENDENCE_OF_DEV_DECISION_BEHAVIOR`

## What is not established

The present evidence does not establish:

- exact global function equivalence;
- equivalence outside the frozen 840-example Phase3A dev set;
- equivalence on confirmatory 9601-9900 data;
- that all geometric differences lie outside the task-reachable state manifold;
- that A-init-specific geometric degrees of freedom are a formal gauge symmetry;
- that the same phenomenon holds at other layers, pressures, models, tasks, optimizers, learning rates, or horizons;
- a universal intrinsic dimension;
- a general optimization theorem;
- a confirmatory population-level statistical claim.

The current result is a validated dev-distribution functional-equivalence observation under the frozen Gen5 Phase3A P0 contract.

## Next scientific action

The next discriminating question is:

> Are the large ambient operator differences between A-init-selected solutions suppressed when restricted to the task-reachable layer-22 state distribution?

Perform a no-training task-reachable operator quotient audit over the already frozen 3x3 checkpoints.

For the exact frozen Phase3A dev rows, capture the actual hidden-state vectors read by the layer-22 correction before the correction is applied.

For each cell, evaluate the correction operator on exactly those states.

Primary objects:

`z_i(x) = A_i x`

`c_i(x) = B_i A_i x`

For each cell pair, compare:

1. ambient read-plane and operator distance;
2. task-state projected read-coordinate similarity;
3. task-state projected correction-output similarity;
4. residual action norm:
   `||(B_i A_i - B_j A_j) X_task||`
5. the same residual normalized by each cell's correction action norm.

Primary grouped contrasts remain:

- same A-init / different training RNG;
- same training RNG / different A-init.

No new training, new seed, rank sweep, hyperparameter sweep, architecture change, or confirmatory population access should occur.

If ambient operator differences collapse strongly on the task-reachable state set, the bounded mechanism becomes:

`A initialization -> ambient geometric representative selection -> task-reachable quotient equivalence -> near-identical dev function`

If they do not collapse at the layer-22 correction output despite near-identical final logits, then the equivalence must emerge downstream and the next localization target is the post-layer-22 propagation path.
