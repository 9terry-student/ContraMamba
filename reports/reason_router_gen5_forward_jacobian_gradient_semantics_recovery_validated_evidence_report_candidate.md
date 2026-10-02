# Gen5 Forward-Jacobian Gradient-Semantics Recovery Validated Evidence Report

## Evidence identity

- execution authority commit:
  `1c9d650c4cc06169c221c1525e93610ba495bdcb`
- run:
  `gen5-forward-jacobian-recovery-1c9d650-r4`
- run command SHA256:
  `3c7220f2684c8395473bdb8a6894d39c619698d97f06a93059b4474f3fef5eca`
- imported handoff ZIP SHA256:
  `111abd4779cba8c90989dad2a9a3dc8a9549ff40a11a44f7b528981b2182f878`
- source task-sensitivity evidence freeze:
  `ed196d5003f279dbfe1dc9a631e84e22dc049ac7`
- source task-sensitivity execution:
  `b8b1a5e95c7932df2c0319e766d10b10f19b3081`
- source residual-localization evidence freeze:
  `3c0a3d8a67e9910f91de2354ba29a5c4b3b28942`
- source functional-equivalence evidence:
  `5f079a66f7b0eb0caea30a8d5bc9a0fe757cc449`

## Execution and provenance validity

The recovery run completed successfully on the exact authorized commit and frozen
Phase3A P0 dev contract.

Observed execution guardrails:

- grid cells: 9
- dev rows: 840
- training executed: false
- optimizer constructed: false
- parameter gradients accumulated: false
- `.backward()` called: false
- checkpoint mutation: false
- confirmatory 9601–9900 loaded: false

Artifact collection and import succeeded with three validated files:

- `forward_jacobian_recovery_summary.json`
- `recovered_task_sensitive_subspace.pt`
- `run_provenance.json`

## Gradient-semantics defect resolution

The historical evaluator used edge-specific gradient ownership with the parent
arm `G3-GROUP-D-HALF`.

The recovery compared that historical analysis gradient against a true
forward-Jacobian analysis path using `gradient_ownership_mode="joint"` while
preserving the exact forward computation.

The two forward functions authenticated as identical:

- joint-vs-edge forward max absolute difference: `0`
- frozen functional-authentication max absolute error:
  `1.43051147461e-06`

The historical gradient field was an essentially exact positive scalar multiple
of the true forward-Jacobian gradient field:

- edge/joint gradient cosine mean: `1.0`
- edge/joint gradient norm ratio mean: `0.5`
- best scalar mapping joint -> edge mean: `0.5`
- relative residual after best scalar mean: approximately `1.93e-08`

Therefore the historical gradient-ownership path preserved gradient direction
and subspace geometry but halved the absolute gradient scale.

## Recovered task-sensitive geometry

The recovered true forward-Jacobian task-sensitive spectrum remains strongly
low-dimensional.

The recovered participation-ratio effective dimension is approximately the
same as the prior frozen audit (`~3.05`), and the normalized eigenspectrum and
leading subspaces are numerically unchanged within recovery precision.

Comparison to the prior frozen audit:

- recovered/prior directional gain: approximately `2.0`
- prior/recovered local task-row-space energy difference: approximately `0`
- prior/recovered normalized-spectrum max absolute difference:
  approximately `4.76e-08`
- prior/recovered top-k subspace squared-cosine alignment:
  approximately `1.0` for the reported leading dimensions

Thus the absolute gain values in the prior audit were scaled by the historical
gradient-ownership contract, but the task-sensitive directions, normalized
spectrum, row-space fractions, and actual/control geometric contrasts survive.

## Same-training-RNG / different-A-init residual

The recovered audit preserves the central geometric result from the prior
frozen analysis.

For same-training-RNG / different-A-init residuals:

- actual/control directional-gain ratio remains approximately `9.82x`
- actual/control exact local task-row-energy ratio remains approximately
  `57.19x`
- mean exact local task-row-space residual-energy fraction remains approximately
  `0.00416`, i.e. `0.416%`

The `0.416%` quantity is a squared-energy fraction of the inter-solution
layer-22 residual projected into the exact local two-margin Jacobian row space.

It does NOT mean that only `0.416%` of the model is task-relevant.

Expressed as a residual-norm fraction, `sqrt(0.00416)` is approximately `6.4%`
under that exact local two-margin projection.

The complementary residual energy is therefore overwhelmingly outside the
instantaneous two-margin Jacobian row space, but the small visible component is
far more structured and task-aligned than norm-preserving random-orientation
controls.

## Scientific interpretation

The recovery validates the following bounded interpretation:

`GEN5_A_INIT_LAYER22_INTER_SOLUTION_VARIATION_IS_DOMINATED_BY_TASK_NULL_OR_LOW_GAIN_DIRECTIONS_WHILE_RETAINING_A_SMALL_STRUCTURED_TASK_VISIBLE_COMPONENT`

This supports a decomposition of the form:

`d_A = d_visible + d_null_or_low_gain`

where the ambient residual norm is dominated by the second component, while
the first component is small in energy but non-randomly aligned with task
sensitivity.

The evidence does not establish that the null/low-gain component corresponds
to a distinct optimization trajectory during training. Only trained endpoints
have been causally compared.

Therefore statements such as “different A-init follows different paths” remain
a plausible interpretation of endpoint geometry, not a measured training-flow
result.

## Consequence for the interpolation hypothesis

The discarded diagnostic interpolation run did not support a strong
excursion-and-return geometry. Its forward-only behavior was close to monotonic
and approximately linear between endpoints.

After the forward-Jacobian recovery, the main scientific question is no longer
whether a highly curved downstream equivalence path is required.

The sharper next question is:

> How much of the different-A endpoint separation remains after projection into
> the recovered task-sensitive subspace, and how much lies in its orthogonal
> complement?

## Next authorized scientific target

The next stage should be a static task-visible / null decomposition and
projection-clustering audit using only the recovered frozen subspace and frozen
3x3 endpoint grid.

It should compare:

- full 768-D endpoint distances;
- distances in recovered top-k task-sensitive coordinates for
  `k = {1,2,4,8,16,32,64}`;
- orthogonal-complement distances;
- same-training-RNG / different-A-init versus same-A-init / different-RNG;
- within-group versus between-group dispersion in the task-sensitive
  coordinates;
- fraction of residual energy retained by each task-sensitive projection.

No new training, optimizer step, confirmatory population, or new scientific
seed is required.
