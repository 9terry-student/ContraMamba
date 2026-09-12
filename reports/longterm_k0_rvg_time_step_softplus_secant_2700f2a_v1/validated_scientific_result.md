# K0-RVG Time-Step Softplus Secant Audit

## Status

VALIDATED SCIENTIFIC EVIDENCE

Execution commit:

`2700f2aa150d3af78a14a2625d4938b016e01727`

Parent nested discrete-B freeze:

`ab798531b6b6349d961927a9255da244c5fa2dc0`

The experiment reused the frozen-token, divergence-aligned,
equal-length-prefix CPU sequential Mamba protocol.

No tokenizer, logits, task heads, training, causal intervention, PCA,
probe, or learned geometry was executed.

## Parent finding

The previous validated stage established that the layer-23 corr-specific
k=2 discrete-B transient is dominated by the selective time-step term Q_T.

The parent term was preserved exactly:

`Q_T = mean(U_m,U_s) * mean(B_m,B_s) * delta_T`

## Softplus factorization

The frozen slow path computes:

`Z = dt_proj(time_step)`

`T = softplus(Z)`

For a matched/swapped pair:

`delta_T = G_sec * delta_Z`

where G_sec is the elementwise softplus secant gain.

The weighted pre-softplus drive was defined as:

`P_Z = mean(U_m,U_s) * mean(B_m,B_s) * delta_Z`

Therefore:

`Q_T = G_sec * P_Z`

elementwise.

The effective norm-level attenuation factor was:

`G_eff = ||Q_T||_2 / ||P_Z||_2`

Softplus secant gains are bounded in [0,1], so this stage does not interpret
softplus as an amplifier.

## Artifact authentication

execution_manifest.json SHA256:

`558b20f410e3165b21a6710ca99f56dd3a38f79df1a51d0430d8fe97cc6d9330`

summary.json SHA256:

`af0a8fd2c03a2d1632cc167019d289f9f3331b7cbe348d69f740d5baa919924b`

The full time_step_softplus_secant_metrics.jsonl remains in the validated
local run artifact.

SHA256:

`d1967335e469f4bd0e373998047c65d80eb7d94d828561b4f46a8a5c9ce534e9`

## Validation

Rows:

`5376`

Common paired DDSSSSS cohort:

`330`

At k=-1 all 672 pair-role cases showed exact identity for:

- hidden_states
- pre-softplus Z
- discrete_time_step
- B
- write

Maximum vector reconstruction relative residual:

`1.92992018582192e-16`

Global elementwise secant-gain range:

`[0.0, 1.0]`

Maximum effective softplus gain:

`0.9620253263419742`

All common-cohort summary medians were independently recomputed.

All frozen parent Q_T role/k medians were exactly reproduced.

## Common-330 trajectory

Median quantities normalized by pair-specific k=0 write divergence.

corr:

- k=1: P_Z 0.869001, Q_T 0.180223, G_eff 0.209679
- k=2: P_Z 0.582655, Q_T 0.539309, G_eff 0.928779
- k=3: P_Z 0.087533, Q_T 0.040379, G_eff 0.597759

ctrl:

- k=1: P_Z 0.600979, Q_T 0.155310, G_eff 0.252369
- k=2: P_Z 0.230014, Q_T 0.149992, G_eff 0.650138
- k=3: P_Z 0.077952, Q_T 0.037067, G_eff 0.523247

## Corr k=1 -> k=2 rebound

Within the common 330-item cohort:

- Q_T increased: 326/330
- P_Z increased: 36/330
- P_Z decreased: 294/330
- G_eff increased: 330/330
- G_eff decreased: 0/330
- Q_T increased while P_Z decreased: 290/330
- Q_T increased while G_eff increased: 326/330
- Q_T increased while P_Z decreased and G_eff increased: 290/330

Median ratios:

- Q_T(k2)/Q_T(k1): 2.9357034640
- P_Z(k2)/P_Z(k1): 0.6499403284
- G_eff(k2)/G_eff(k1): 4.3646338359

Thus the corr k=2 time-step rebound is not a rebound of the weighted
pre-softplus drive magnitude.

In most items, P_Z decreases while Q_T increases.

The rebound is instead associated with a cohort-wide rise in effective
softplus transmission: the pre-softplus difference is much less attenuated
at k=2 than at k=1.

## Ctrl k=1 -> k=2

Within the same cohort:

- Q_T increased: 156/330
- P_Z increased: 1/330
- P_Z decreased: 329/330
- G_eff increased: 327/330
- Q_T increased while P_Z decreased and G_eff increased: 155/330

Median ratios:

- Q_T(k2)/Q_T(k1): 0.9628051049
- P_Z(k2)/P_Z(k1): 0.3725637862
- G_eff(k2)/G_eff(k1): 2.4768147380

Thus ctrl also undergoes reduced softplus attenuation at k=2, but its
pre-softplus drive falls much more strongly, leaving Q_T approximately flat
rather than producing the corr rebound.

## Corr versus ctrl at k=2

Within the common paired cohort:

- corr Q_T > ctrl Q_T: 329/330
- corr P_Z > ctrl P_Z: 328/330
- corr G_eff > ctrl G_eff: 330/330

Median corr/ctrl ratios:

- Q_T: 3.5609017237
- P_Z: 2.4899918144
- G_eff: 1.4155785462

Therefore the absolute corr-versus-ctrl k=2 separation is jointly associated
with:

1. a larger weighted pre-softplus difference drive in corr; and
2. systematically weaker softplus attenuation in corr.

The temporal rebound and the role separation should not be conflated.

For the temporal k1->k2 rebound, attenuation relief is dominant because P_Z
usually decreases.

For the corr-versus-ctrl k2 separation, both drive magnitude and softplus
operating-point sensitivity contribute.

## Corr k=2 -> k=3 collapse

Within the common cohort:

- Q_T decreased: 330/330
- P_Z decreased: 330/330
- G_eff decreased: 327/330
- Q_T and P_Z both decreased: 330/330
- Q_T, P_Z, and G_eff all decreased: 327/330

Median ratios:

- Q_T(k3)/Q_T(k2): 0.0691592417
- P_Z(k3)/P_Z(k2): 0.1459042782
- G_eff(k3)/G_eff(k2): 0.6459200501

Thus the immediate collapse combines a sharp loss of pre-softplus drive with
renewed softplus attenuation.

## Scientific interpretation

The validated evidence supports the following narrow conclusions.

First, the layer-23 corr-specific k=2 selective-time-step rebound is not
produced by growth of the weighted pre-softplus projected difference.

Instead, in most items the weighted pre-softplus drive decreases from k=1 to
k=2 while the transmitted Q_T divergence increases.

The defining change is a cohort-wide shift in softplus operating-point
sensitivity that greatly reduces attenuation at k=2.

Second, the absolute corr-versus-ctrl separation at k=2 cannot be attributed
to softplus sensitivity alone. Corr has both a larger pre-softplus drive and
a larger effective softplus transmission factor.

Third, the k=3 collapse combines both reduced pre-softplus drive and stronger
attenuation.

These are observational factorization results and do not establish causal
downstream relevance.

## Next scientific boundary

The next minimal native-Mamba question is no longer whether softplus matters;
that is now established observationally.

The remaining question is what produces the corr-specific softplus
operating-point shift.

Because:

`Z = dt_proj(time_step)`

and dt_proj is affine with a shared learned bias, matched/swapped differences
remove the bias from delta_Z, but the bias and mean projected location can
still determine the absolute operating point and therefore the secant gain.

The next stage should therefore characterize the matched/swapped Z operating
point itself, especially whether corr k=2 moves into the high-slope softplus
regime in a systematic channel-wise manner.

No intervention or downstream claim is yet authorized by this result.