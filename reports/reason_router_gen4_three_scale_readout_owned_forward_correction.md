# ContraMamba Gen4 Three-Scale Readout Ownership Correction

## Status

`STATIC_FINAL_READOUT_CORRECTION`

Evidence base HEAD:

`b5b5436a7df4c01385262ac0ce327d264aa33b38`

This artifact introduces:

- no model execution;
- no training;
- no forward or backward pass;
- no new p-value;
- no new inferential family;
- no row filtering, rescue, or re-selection.

It freezes the final semantic correction for the Mamba-130M, Mamba-370M, and
Mamba-1.4B readout evidence after the factor-2 root cause was localized to the
frozen `G3-GROUP-D-HALF` gradient-ownership graph.

## 1. Frozen gradient semantics

All three readout programs use:

`ARM = G3-GROUP-D-HALF`

with:

- `F_TO_D = 0.5`
- `P_TO_D = 0.5`
- `S_TO_D = 0.5`
- `Q_TO_D = 0.5`

The grouped model implements recipient-only gradient scaling as:

`partial_grad(x, lambda) = stopgrad(x) + lambda * (x - stopgrad(x))`.

Therefore the numerical forward value is unchanged while the local backward
Jacobian is multiplied by `lambda`.

For the frozen D-half arm:

`Delta_L_owned = 0.5 * Delta_L_forward`

and hence:

`Delta_L_forward = 2 * Delta_L_owned`.

`Delta_L_owned` denotes the values already stored in the immutable historical
readout artifacts.

`Delta_L_forward` is a deterministic semantic conversion for the derivative of
the numerical forward task margin along the same frozen selected-minus-control
direction. It is not a new measurement.

## 2. Final three-scale correction table

| Scale | Frozen population | Selected / control | Mean `Delta_L_owned` | Mean `Delta_L_forward = 2x owned` | Positive fraction | Inferential status |
|---|---|---|---:|---:|---:|---|
| Mamba-130M | `xg1_fact_2701..3000` | P3 / P5 | `+0.001120366036532127` | `+0.002240732073064253` | `0.5366666666666666` | positive mean supported; one-sided t=`3.125242461870215`, p=`0.0009755452781221727` |
| Mamba-370M | `xg1_fact_4801..5100` | P3 / P5 | `+0.0005089563518854174` | `+0.001017912703770835` | `0.5766666666666667` | positive sign gate in frozen paired 370M-vs-1.4B test |
| Mamba-1.4B | `xg1_fact_4801..5100` | P5 / P4 | `-0.001195447505886224` | `-0.002390895011772448` | `0.35333333333333333` | negative sign gate in frozen paired 370M-vs-1.4B test |

Cross-scale paired endpoint:

| Quantity | Owned-gradient scale | Forward-equivalent scale |
|---|---:|---:|
| `mean(Delta_L_370M - Delta_L_1.4B)` | `0.001704403857771641` | `0.003408807715543282` |
| paired t statistic | `10.771333954019786` | `10.771333954019786` |
| one-sided p-value | `2.2238330789610916e-23` | `2.2238330789610916e-23` |

The t statistic and p-value are identical because the forward-equivalent
conversion multiplies every paired difference by the same positive constant.

## 3. What is invariant under the x2 conversion

The following frozen results are unchanged:

- sign of each `Delta_L`;
- positive / negative fractions;
- pair ranking;
- Pearson correlation;
- Spearman correlation;
- sign agreement;
- cosine-based quantities where the common positive gradient scale cancels;
- one-sample t statistic against zero for 130M;
- paired 370M-vs-1.4B t statistic;
- their corresponding p-values;
- the 130M positive-alignment conclusion;
- the 370M positive / 1.4B negative sign pattern;
- the 370M-vs-1.4B sign-reversal conclusion.

Therefore no N=300 rerun or new inferential analysis is required.

## 4. What must be relabeled

Historical stored `Delta_L` must not be described without qualification as the
ordinary derivative of the numerical forward task margin.

Preferred terminology:

- stored artifact value: `gradient-ownership-weighted directional readout`
  or `Delta_L_owned`;
- forward numerical derivative on the same frozen direction:
  `Delta_L_forward = 2 * Delta_L_owned`.

Historical raw artifacts remain immutable and are not rewritten.

## 5. Behavioral calibration correction

The previously observed relation:

`D_BEH(alpha) ~= 2 * alpha * Delta_L_owned`

is retained numerically but its interpretation is corrected.

The forward-equivalent form is:

`D_BEH(alpha) ~= alpha * Delta_L_forward`.

Thus the stable factor near two is not evidence of unexplained nonlinear gain,
finite-displacement amplification, or a Mamba backend defect. It is the expected
consequence of the frozen D-half gradient-ownership graph.

## 6. Scope boundary

This correction is valid only for readouts whose frozen execution uses the
`G3-GROUP-D-HALF` edge map above.

It must not be transferred automatically to other arms.

In particular, the Precursor-v4 analytic causal-LM VJP is excluded: that
measurement differentiates a `MambaForCausalLM` block-35 local leaf directly to
the next-token SUPPORT-vs-REFUTE logit margin and does not traverse the grouped
ContraMamba edge-specific ownership graph.

## 7. Final corrected readout statement

> Across the frozen three-scale readout studies, the gradient-ownership-weighted
> local directional readout is positive at Mamba-130M and Mamba-370M and negative
> at Mamba-1.4B. Because all three use the same `G3-GROUP-D-HALF` D-edge scaling,
> the corresponding numerical-forward directional derivatives are exactly twice
> the stored magnitudes, with all sign, rank, correlation, t-statistic, p-value,
> and sign-reversal conclusions unchanged.

`THREE_SCALE_READOUT_CORRECTION_FROZEN = YES`
