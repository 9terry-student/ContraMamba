# ContraMamba Gen4 Factor-2 Gradient-Ownership Root-Cause Closure

## 0. Status

- Status: STATIC ROOT-CAUSE / CLAIM-IMPACT CLOSURE
- Evidence base HEAD: `3f24dd4c49e40d9c343d2baa64e698ce9bdc2b88`
- Branch: `gen4-mamba370m-core-replication`
- New training: NO
- New model forward: NO
- New backward: NO
- New tokenizer execution: NO
- New Kaggle / GPU execution: NO
- New p-values: 0
- Row filtering / rescue / adaptive rerun: NO

This report closes the Gen4 factor-2 calibration investigation. It does not
modify or replace any frozen raw artifact. It records the source-defined
gradient semantics that explain the previously observed approximately two-fold
difference between finite forward response and the stored autograd directional
readout.

## 1. Frozen empirical evidence

The factor-2 arc is supported by three already-frozen stages.

### 1.1 Positive small-alpha calibration

Frozen static calibration commit:

`57644369580c6f560e8eeb1ddbfc09d714a4c789`

For `K(alpha) = origin-slope[D_BEH(alpha) on alpha * Delta_L]`:

| alpha | Mamba-370M | Mamba-1.4B |
|---:|---:|---:|
| 0.25 | 2.0229766493 | 1.9892794255 |
| 0.125 | 2.0113698669 | 1.9949103227 |
| 0.0625 | 2.0056364651 | 1.9982356895 |
| 0.03125 | 2.0035676915 | 2.0001894009 |

Thus the approximately two-fold calibration persisted as positive alpha was
reduced through `0.03125`.

### 1.2 Symmetric gradient-consistency diagnostic

Frozen static analysis commit:

`9dbace6a97225668a626abc0632d0a7c1c6cff69`

The central finite-difference directional derivative was stably approximately
two times the frozen autograd `Delta_L` for both scales and all tested epsilons.

At epsilon `0.03125`:

- Mamba-370M central-vs-frozen origin slope: `1.999144658977673`
- Mamba-1.4B central-vs-frozen origin slope: `2.00099321465723`

Wrong-class switching was absent in the symmetric diagnostic.

### 1.3 Independent slow/reference backend

Raw freeze commit:

`0d4b4a06059609f4c1538df923fc9e7ffab0f5e9`

Static evidence freeze commit:

`3f24dd4c49e40d9c343d2baa64e698ce9bdc2b88`

At epsilon `0.03125`, on the same fast-derived direction:

Mamba-370M:

- fast autograd replay vs frozen: `1.000000660265663`
- fast central vs frozen autograd: `2.0000336403551464`
- slow autograd vs frozen fast autograd: `1.0000018693614587`
- slow central vs frozen fast autograd: `2.0016940275584951`
- slow central vs slow autograd: `2.0016902880817065`
- fast central vs slow central: `1.0008268279806969`
- max native-margin fast/slow absolute difference:
  `3.0994415283203125e-06`

Mamba-1.4B:

- fast autograd replay vs frozen: `0.99999985195764496`
- fast central vs frozen autograd: `1.9996185943479485`
- slow autograd vs frozen fast autograd: `0.99999965242966016`
- slow central vs frozen fast autograd: `2.0008085667687823`
- slow central vs slow autograd: `2.0008092625970257`
- fast central vs slow central: `1.0005935872325216`
- max native-margin fast/slow absolute difference:
  `4.5299530029296875e-06`

Therefore the mismatch is not specific to the fused CUDA Mamba backend, is not
specific to the sequential slow Mamba backend, and is not explained by a
meaningful fast/slow forward discrepancy.

## 2. The frozen model uses edge-specific gradient ownership

The historical inference adapter freezes:

`GRADIENT_OWNERSHIP_MODE = "edge_specific"`.

All three readout programs relevant to the current Gen4 synthesis use:

`ARM = "G3-GROUP-D-HALF"`.

This is explicit for:

- Mamba-130M through
  `reason_router_gen4_seed181_behavioral_restoration_bridge_fast_cuda.py`;
- Mamba-370M through
  `reason_router_gen4_mamba370m_geometry_prepare_fast_cuda.py`;
- Mamba-1.4B through
  `reason_router_gen4_mamba14b_geometry_prepare_fast_cuda.py`.

For `G3-GROUP-D-HALF`, the frozen edge map has:

- `F_TO_D = 0.5`
- `P_TO_D = 0.5`
- `S_TO_D = 0.5`
- `Q_TO_D = 0.5`

while the earlier internal F/P/S/Q propagation edges are `1.0`.

## 3. Exact source semantics

The frozen grouped model defines:

```python
def _partial_grad(tensor, gradient_ownership_lambda):
    detached = tensor.detach()
    return detached + gradient_ownership_lambda * (tensor - detached)
```

The source comment states that this preserves the forward value while scaling
only downstream gradients.

For any tensor `x` and lambda `lambda_D`:

`partial_grad(x, lambda_D) = x`

in forward value, because `x - detach(x)` is numerically zero.

But PyTorch autograd sees:

`d partial_grad / dx = lambda_D * I`.

Therefore, for `lambda_D = 0.5`, the forward decision input is unchanged while
its backward sensitivity is halved.

In `edge_specific` mode the final decision head receives:

- `frame_prob` through `F_TO_D`;
- `predicate_coverage_prob` through `P_TO_D`;
- `sufficiency_prob` through `S_TO_D`;
- positive and negative polarity energy through `Q_TO_D`.

For the frozen `G3-GROUP-D-HALF` arm, each of these final-decision recipient
aliases applies lambda `0.5`.

## 4. No active final-logit bypass in the frozen inference configuration

The historical constructor freezes:

- `use_temporal_comparator = False`
- `use_predicate_comparator = False`

The model forward defaults:

- `temporal_adapter_final_penalty_scale = 0.0`
- `temporal_channel_gated_penalty_scale = 0.0`

The historical adapter calls the model without supplying nonzero values for
these optional modulation inputs.

Thus, in the frozen readout executions considered here, the task logits are
obtained from the ordinary final decision path whose upstream inputs receive the
`G3-GROUP-D-HALF` D-edge aliases.

There is no active final-logit path in this frozen configuration that restores
an unscaled upstream derivative around the D-edge.

## 5. Root-cause derivation

Let `h` denote the intervention-boundary activation and let `M_forward(h)` be
the numerical task margin obtained from ordinary forward evaluation.

Let `g_owned` be the gradient returned by autograd through the frozen
gradient-ownership graph.

Because every upstream route into the final decision is passed through a
D-edge with lambda `0.5`, while the forward value is unchanged:

`g_owned = 0.5 * grad_h M_forward(h)`.

The frozen readout stores:

`Delta_L_owned = g_owned^T (C_sel - C_ctrl)`.

The forward-equivalent directional derivative is therefore:

`Delta_L_forward`
`= grad_h M_forward(h)^T (C_sel - C_ctrl)`
`= 2 * Delta_L_owned`.

For the behavioral correction:

`delta = C_ctrl - C_sel`.

A true first-order expansion of the numerical forward function gives:

`D_BEH(alpha)`
`= M_forward(h) - M_forward(h + alpha*delta)`
`= alpha * Delta_L_forward + O(alpha^2)`
`= 2 * alpha * Delta_L_owned + O(alpha^2)`.

This is exactly the calibration repeatedly observed in the frozen evidence.

## 6. Root-cause disposition

The following explanations are NOT supported as the cause of the systematic
factor 2:

- duplicated intervention application;
- selected/control sign reversal;
- two-cell sum-versus-mean error;
- wrong-class switching;
- logit serialization scale;
- finite-positive-alpha curvature as the primary explanation;
- fused CUDA Mamba backward defect;
- sequential slow Mamba backward defect;
- meaningful fast/slow forward-backend divergence.

The source-defined explanation is:

`FACTOR_2_ROOT_CAUSE = G3_GROUP_D_HALF_GRADIENT_OWNERSHIP`

More precisely:

`STORED_AUTOGRAD_DELTA_L = 0.5 * FORWARD_EQUIVALENT_DELTA_L`

for the frozen `G3-GROUP-D-HALF` readout configuration.

The factor-2 observation is therefore not a Mamba numerical anomaly. It is the
expected consequence of the intentionally surrogate gradient graph used by the
frozen historical model.

## 7. Correction to the earlier static-audit interpretation

The earlier calibration audit stated that, for a smooth differentiable
function under the same gradient definition:

`lim_(alpha->0) D_BEH(alpha)/(alpha*Delta_L) = 1`.

That statement is not applicable to the stored `Delta_L` in this frozen arm.

The reason is that the numerical forward mapping and the autograd graph have
deliberately different local derivatives because of `_partial_grad`.

The correct local identities are:

`lim_(alpha->0) D_BEH(alpha)/(alpha*Delta_L_forward) = 1`

and, for `G3-GROUP-D-HALF`:

`lim_(alpha->0) D_BEH(alpha)/(alpha*Delta_L_owned) = 2`.

Accordingly, the observed small-alpha plateau near 2 is not evidence of a
failure to reach a local regime.

## 8. Impact on prior scientific claims

### 8.1 Claims preserved

The following results are invariant to a common positive factor `0.5` applied
to every `Delta_L` within these frozen D-half programs:

- sign of `Delta_L`;
- fraction of positive `Delta_L`;
- rank ordering;
- Pearson correlation under positive scalar rescaling;
- Spearman correlation;
- sign agreement with behavioral displacement;
- one-sample t statistic for testing a zero mean;
- paired t statistic on `Delta_L_370M - Delta_L_1.4B` when both scales share
  the same positive factor;
- the 370M-versus-1.4B sign-reversal conclusion;
- native-state geometry and intervention-state evidence;
- all raw behavioral margins and behavioral intervention effects.

Therefore the previously frozen conclusions:

- positive 130M readout alignment;
- positive 370M mean readout alignment;
- negative 1.4B mean readout alignment;
- 370M-versus-1.4B readout sign reversal

remain supported as sign/order claims.

Their originally executed p-values are unchanged by the uniform positive
rescaling and do not need to be recomputed.

### 8.2 Quantities requiring semantic relabeling

The stored quantity should no longer be described without qualification as the
ordinary derivative of the numerical forward task margin.

For existing immutable artifacts:

`Delta_L` should be interpreted as:

`Delta_L_owned`

or:

`gradient-ownership-weighted directional readout`.

For a forward-equivalent directional derivative in the current D-half arm use:

`Delta_L_forward = 2 * Delta_L_owned`.

Do not rewrite historical raw artifacts. The conversion is a semantic
interpretation attached to the frozen configuration.

### 8.3 Absolute means under forward-equivalent calibration

Previously reported stored means:

- Mamba-130M: `+0.0011203660365321265`
- Mamba-370M: `+0.0005089563518854174`
- Mamba-1.4B: `-0.0011954475058862238`

Forward-equivalent means for the same frozen directions are:

- Mamba-130M: `+0.002240732073064253`
- Mamba-370M: `+0.0010179127037708348`
- Mamba-1.4B: `-0.0023908950117724476`

These are deterministic rescalings, not new measurements.

### 8.4 Behavioral calibration claim

The empirical statement:

`D_BEH(alpha) ~= 2 * alpha * Delta_L`

remains numerically true when `Delta_L` denotes the frozen stored readout.

Its interpretation changes.

It is equivalently:

`D_BEH(alpha) ~= alpha * Delta_L_forward`.

Thus the factor near 2 should not be presented as an unexplained gain or
nonlinear amplification.

## 9. Three-scale synthesis impact

The existing three-scale synthesis can retain its sign pattern:

- 130M: positive;
- 370M: positive;
- 1.4B: negative.

Because all three programs use `G3-GROUP-D-HALF`, multiplying all three
`Delta_L` values by two preserves:

- sign;
- positive fraction;
- relative ordering within a scale;
- the qualitative three-scale sign pattern.

However, the phrase `local first-order task-margin readout` should be qualified
when referring to stored values. Preferred wording is:

`gradient-ownership-weighted local task-margin readout`.

If the intended object is the derivative of the numerical forward function,
use the forward-equivalent rescaling defined above.

## 10. Scope boundary

This closure applies to the frozen Gen4 readout programs using
`G3-GROUP-D-HALF`.

It must not be generalized blindly to a different arm.

For example, an arm whose D-edge lambdas are `1.0` would not receive this
factor-2 conversion.

The edge map, not the model scale itself, determines the conversion.

## 11. Stop rule and next research state

No additional small-alpha sweep, smaller epsilon, negative-alpha expansion,
backend comparison, or larger factor-2 cohort is scientifically necessary to
resolve this calibration question.

The factor-2 investigation is CLOSED as a root-cause problem.

Future Gen4 work should:

1. preserve all frozen artifacts unchanged;
2. label historical stored `Delta_L` as ownership-weighted when discussing
   derivative magnitude;
3. use `2 * Delta_L` only when a forward-equivalent derivative is required for
   the frozen D-half arm;
4. preserve sign/rank conclusions that are invariant to the positive scaling;
5. avoid describing the historical factor 2 as a Mamba kernel defect,
   finite-displacement anomaly, or unexplained nonlinear gain.

`FACTOR_2_ROOT_CAUSE_CLOSED = YES`
