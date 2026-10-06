# Gen5 A-init First-Update Origin — Static Evidence Report

## Status

`PASS_FIRST_UPDATE_ORIGIN_STATIC_DECOMPOSITION`

This report performs no training, backward pass, optimizer step, task evaluation,
or GPU execution. It statically reconstructs the already executed and frozen
Phase A `t=0 -> t=1` update and joins it to the already frozen behavioral onset
evidence.

No new execution authority is created by this report.

## Frozen inputs

- analysis HEAD: `76dbf99883cd6dd99d50270b2ddbc1540d8c4c5e`
- Phase A trajectory SHA256: `0f7cd4248faa92223597e0816597b59e426f08829dadd366f9603f56a8de809e`
- Phase A summary SHA256: `37cab4866c729ad03262df562762ac8b6b91e4e24e2498223483848871ad1df6`
- behavioral source execution HEAD: `1633d67146a7f049c6ecf28b333f53fdf79ad00e`
- behavioral source ZIP SHA256: `dffbdf75b1d9d485cee30309a443553aea6995d5f109e7c4720c8c994bf62afd`

## First-update algebra

All 9 cells satisfy:

- `B0 == 0` exactly.
- `grad_A0 == 0` exactly.
- maximum total step-0 gradient norm = `0.39248752593994141`.
- clipping threshold = `5`.
- therefore gradient clipping is inactive in every cell.
- for fixed A seed, `A0` is exactly identical across all three training-RNG seeds.
- for fixed A seed, `A1` is also exactly identical across all three training-RNG seeds.

The first update is reconstructed as:

`A1 = A0 * (1 - lr * weight_decay)`

and, because `B0 == 0`, the first AdamW update reduces to:

`B1 = -lr * grad_B0 / (abs(grad_B0) + eps)`.

Observed-vs-reconstructed residuals across all 9 cells:

- max `A1` absolute residual: `0`
- max `B1` absolute residual: `4.6566128730773926e-10`
- max `W1 = B1 A1` relative Frobenius residual: `1.7024441400390503e-07`

Thus the frozen `t=1` operator is numerically accounted for by the first
optimizer update to numerical precision.

## A-init vs training-RNG factor path

Grouped same-R/different-A versus same-A/different-R pair-distance ratios:

- `grad_B0`: `45816.814286749301`
- `B1`: `120.34559389542568`
- `W1 = B1 A1`: `126.53445271722663`

The first AdamW coordinate normalization strongly compresses the raw magnitude
contrast from `grad_B0` to `B1`, but A-init remains dominant by roughly two
orders of magnitude at `B1` and `W1`.

`A1` has exactly zero same-A/different-R distance, so its grouped A/R ratio is
undefined rather than finite: the R denominator is exactly zero.

## Behavioral join at t=1

Frozen behavioral evidence gives:

- t0 A-axis disagreement sum: `0`
- t0 R-axis disagreement sum: `0`
- t1 A-axis disagreement sum: `72`
- t1 R-axis disagreement sum: `0`

Therefore the first step at which the operator becomes nonzero is also the first
step at which prediction divergence appears, and the prediction divergence is
initially confined to the A-init axis in this 3x3 factorial.

## Bounded conclusion

The executed evidence supports the following mechanism-level statement:

> Under the frozen Gen5 replay, A-init enters the first-update path through
> `grad_B0`; with clipping inactive and `grad_A0=0`, the first AdamW step
> deterministically creates `B1` and hence `W1=B1A1`. A-init dominance survives
> AdamW normalization, and A-axis behavioral divergence appears at the same
> `t=1` boundary while the training-RNG axis remains behaviorally identical.

This is stronger than a temporal correlation because the optimizer update is
reconstructed from frozen pre-update gradients and parameters and matches the
observed post-update parameters/operator to numerical precision.

It is still **not** a finite factor-swap intervention. That stronger causal
counterfactual remains reserved for the later factor-swap stage. This report
also does not explain why the common 120 rows become transiently vulnerable or
why all cells behaviorally reconverge by t=17.
