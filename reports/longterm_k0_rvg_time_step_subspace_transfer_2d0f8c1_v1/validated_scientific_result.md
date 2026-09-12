# K0-RVG Time-Step Subspace Transfer Audit

## Status

VALIDATED SCIENTIFIC EVIDENCE

Execution commit:

`2d0f8c1b32e62e18b0f9449c3ee6dda1a8c7387e`

Parent dt-projection geometry freeze:

`a9f3f8023ec168c5f75ce8b3df4bdbee32eb7109`

No tokenizer, logits, task heads, training, intervention, PCA, probe, or learned
geometry was executed.

Raw vectors were not persisted.

## Exact source identity

At layer 23, the authenticated Transformers 5.12.1 Mamba source gives:

`r_t = W_r U_t`

where:

- `U_t` is the post-convolution / activation Mamba hidden vector, width 1536.
- `W_r` is the first 48 rows of the bias-free `x_proj`.
- `W_r` has shape `(48, 1536)`.
- `r_t` is the 48-dimensional low-rank time-step vector.

Because the time-step path has no bias and no intervening nonlinearity:

`delta_r = W_r delta_U`

exactly up to floating-point reconstruction tolerance.

The magnitude factorization is:

`||delta_r|| = ||delta_U|| * transfer`

with:

`transfer = ||W_r delta_U|| / ||delta_U||`

This audit separates upstream post-convolution difference magnitude from total
direction-dependent transfer through the fixed time-step projection map.

## Artifact authentication

execution_manifest.json SHA256:

`73da298e6eda7f5753f8c923c5010634bbf68af7b3c4946fabe2fd654987ae7e`

summary.json SHA256:

`dfe618d961ba8fd7e0dcd16a75bf53cfb3e45878b2715e55fa656c1f6775d4cb`

Full local metrics artifact:

`time_step_subspace_transfer_metrics.jsonl`

SHA256:

`4e5f37f335726255784054d456a24d0c063c06c554a25d0d6aadfca37744cc45`

## Validation

Trajectory rows:

`5376`

Common paired DDSSSSS cohort:

`330`

Model forwards:

`1344`

At k=-1:

- delta-U exactly zero: 672/672
- delta-r exactly zero: 672/672
- projected delta-r exactly zero: 672/672

Maximum relative reconstruction residual for:

`W_r delta_U = delta_r`

was:

`1.597111301653152e-05`

Maximum scalar factorization absolute error:

`4.440892098500626e-16`

Maximum scalar factorization relative error:

`2.1507197882492626e-16`

All frozen parent delta-r role/k medians were reproduced.

All common-cohort summary medians were independently recomputed.

## Common-330 trajectory

corr k=1:

- ||delta-U||: 1.888356
- ||delta-r||: 1.375355
- transfer: 0.739195

corr k=2:

- ||delta-U||: 1.321293
- ||delta-r||: 1.393869
- transfer: 1.089006

corr k=3:

- ||delta-U||: 0.478579
- ||delta-r||: 0.400450
- transfer: 0.818153

ctrl k=1:

- ||delta-U||: 1.398208
- ||delta-r||: 1.050854
- transfer: 0.767164

ctrl k=2:

- ||delta-U||: 0.486091
- ||delta-r||: 0.440044
- transfer: 0.888864

ctrl k=3:

- ||delta-U||: 0.334294
- ||delta-r||: 0.304529
- transfer: 0.886616

## Corr k=1 -> k=2

Within the common cohort:

- delta-U increased: 45/330
- delta-U decreased: 285/330
- delta-r increased: 160/330
- delta-r decreased: 170/330
- transfer increased: 330/330
- transfer decreased: 0/330

Median paired changes:

- delta-U: -0.624992
- delta-r: -0.018071
- transfer: +0.344623

Median log ratios:

- delta-U: -0.397616
- delta-r: -0.012829
- transfer: +0.380920

Thus the low-rank delta-r magnitude is nearly preserved from corr k=1 to k=2
despite a predominantly contracting upstream delta-U magnitude.

The compensation is supplied by a universal increase in total transfer through
the fixed x_proj time-step map.

Among the 285 items with both delta-U contraction and transfer increase:

- transfer overcompensated and delta-r increased: 115
- transfer undercompensated and delta-r decreased: 170

Median transfer compensation ratio:

`0.876767`

Therefore the result should not be described as universal amplification or
universal rebound of delta-r.

The validated temporal statement is:

**time-step-subspace transfer universally increases at corr k=2 and largely
counteracts a predominantly contracting upstream delta-U magnitude, producing
near preservation of delta-r at the cohort level.**

## Corr versus ctrl at k=2

Within the common cohort:

- corr delta-U > ctrl: 329/330
- corr delta-r > ctrl: 330/330
- corr transfer > ctrl: 270/330
- both delta-U and transfer greater for corr: 269/330

Median corr-minus-ctrl differences:

- delta-U: +0.765502
- delta-r: +0.916364
- transfer: +0.206446

Median log ratios:

- delta-U corr/ctrl: +0.922051
- delta-r corr/ctrl: +1.127454
- transfer corr/ctrl: +0.208166

Item-level log-contribution dominance:

- delta-U log contribution > transfer log contribution: 302/330
- transfer log contribution > delta-U log contribution: 28/330

Median signed contribution to log delta-r separation:

- delta-U contribution: 0.818802
- transfer contribution: 0.181198

Thus the corr-vs-ctrl k2 delta-r separation is primarily already present in
the upstream post-convolution delta-U magnitude.

A secondary transfer advantage through the fixed time-step map strengthens
that separation, but it is not the dominant contribution.

This is distinct from the temporal corr k1-to-k2 phenomenon.

## Corr k=2 -> k=3 collapse

Within the common cohort:

- delta-U decreased: 329/330
- delta-r decreased: 330/330
- transfer decreased: 288/330
- both delta-U and transfer decreased: 287/330

Median log ratios:

- delta-U: -0.965262
- delta-r: -1.242350
- transfer: -0.283767

Contribution dominance:

- |delta-U log change| > |transfer log change|: 298/330
- |transfer log change| > |delta-U log change|: 32/330

Therefore the corr k2-to-k3 collapse is primarily driven by collapse of the
upstream delta-U magnitude, with a smaller additional reduction in time-step
subspace transfer.

## Validated scientific conclusion

The layer-23 low-rank time-step difference obeys the authenticated exact
linear relation:

`delta_r = W_r delta_U`

The corr-specific temporal k2 structure and corr-vs-ctrl role separation have
different factor compositions.

For corr k=1 -> k=2, upstream delta-U magnitude contracts in most items, while
time-step-subspace transfer increases in every item. The transfer increase
largely offsets the upstream contraction, leaving delta-r nearly preserved at
the cohort level. It does not universally overcompensate the contraction.

For corr versus ctrl at k=2, the much larger corr delta-r is primarily
explained by a much larger upstream post-convolution delta-U magnitude.
A smaller but systematic transfer advantage through the fixed time-step map
provides an additional contribution.

For corr k=2 -> k=3, the collapse is again dominated by upstream delta-U
magnitude collapse, with a smaller transfer decrease.

These results are observational and algebraic. They do not establish causal
downstream relevance.

## Next scientific boundary

The dominant unresolved object is now the upstream post-convolution
`delta_U`.

The next stage should statically authenticate the exact Mamba source path that
produces U from the pre-convolution projected hidden stream:

1. `in_proj`
2. split into hidden-state and gate branches
3. depthwise causal convolution
4. activation
5. optional attention-mask application

Before any new forward execution, that exact source transformation should be
authenticated and the simplest decomposition boundary selected.

The next scientific question should then distinguish whether the corr-k2
delta-U magnitude pattern is already present before the depthwise convolution
or is created/reshaped by the convolution-plus-activation transformation.

No training, intervention, tokenizer execution, learned probe, or PCA is
authorized by this result.