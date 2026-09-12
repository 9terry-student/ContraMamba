# K0-RVG Softplus Operating-Point Audit

## Status

VALIDATED SCIENTIFIC EVIDENCE

Execution commit:

`0a291efee3f6cb0be84ae316e87b5eb848bc5b11`

Parent softplus-secant freeze:

`4ea1233c0dc45bddb6e48b9b7a3d7ff7d97b77b9`

No tokenizer, logits, task heads, training, causal intervention, PCA,
probe, or learned geometry was executed.

Raw channel vectors were not persisted.

## Scientific question

The validated parent stage established:

`Z = dt_proj(time_step)`

`T = softplus(Z)`

`delta_T = G_sec * delta_Z`

and:

`P_Z = mean(U) * mean(B) * delta_Z`

`Q_T = G_sec * P_Z`

It further established that the corr k=1 -> k=2 Q_T rebound usually occurs
while ||P_Z|| decreases and effective softplus transmission rises.

This audit asked whether that transmission increase reflects:

1. a broad layer-wide movement of Z into a higher-slope softplus regime;
2. redistribution of P_Z divergence energy toward already-high-transmission
   channels;
3. or both.

## Artifact authentication

execution_manifest.json SHA256:

`630fca91e74b44eb2a94519fb3f848a0e6a890e413897d6890a274c97412433c`

summary.json SHA256:

`21ff7ef3a27d4635c2a15a8df24bfbdf63cbd1133699503982a606513a3ce718`

The full local metrics artifact:

`softplus_operating_point_metrics.jsonl`

SHA256:

`936396e224d1aa7216244b24bbde75e63f9687c45f21fb5455e2e4deee8cb36b`

## Validation

Trajectory rows:

`5376`

Common paired DDSSSSS cohort:

`330`

At k=-1 all 672 pair-role cases showed exact identity for:

- hidden_states
- pre-softplus Z
- discrete_time_step
- B
- write

Maximum softplus composition relative residual:

`1.92992018582192e-16`

Maximum discrepancy between exact effective gain and the P_Z-energy-weighted
secant-gain RMS identity:

`4.440892098500626e-15`

All frozen parent P_Z, Q_T, and effective-gain role/k medians were reproduced.

All common-cohort summary medians were independently recomputed.

## Energy weighting

Per-channel divergence-energy weight was:

`w_j = sum_state(P_Z[j,:]^2)`

with:

`P_Z[j,s] = mean(U_j) * mean(B_s) * delta_Z_j`

Therefore:

`w_j = (mean(U_j) * delta_Z_j)^2 * sum_s mean(B_s)^2`

Within a fixed pair/token, the B norm is common to every intermediate channel.

Consequently, normalized channel-energy redistribution is independent of B
and is determined by the channel pattern of:

`|mean(U_j) * delta_Z_j|^2`

This is an exact algebraic property of the weighting definition and is not a
causal claim.

## Common-330 trajectory

Median values:

corr k=1:

- unweighted Z midpoint mean: -1.318897
- P_Z-energy-weighted Z midpoint mean: -8.282662
- unweighted secant-gain mean: 0.238278
- P_Z-energy-weighted gain mean: 0.059894
- effective gain: 0.209679
- P_Z energy at gain >= 0.75: 0.036228
- P_Z energy at gain >= 0.90: 0.031880

corr k=2:

- unweighted Z midpoint mean: -1.389004
- P_Z-energy-weighted Z midpoint mean: 3.429218
- unweighted secant-gain mean: 0.225837
- P_Z-energy-weighted gain mean: 0.900366
- effective gain: 0.928779
- P_Z energy at gain >= 0.75: 0.895106
- P_Z energy at gain >= 0.90: 0.883267

corr k=3:

- unweighted Z midpoint mean: -1.192148
- P_Z-energy-weighted Z midpoint mean: -0.188631
- unweighted secant-gain mean: 0.252328
- P_Z-energy-weighted gain mean: 0.501481
- effective gain: 0.597759
- P_Z energy at gain >= 0.75: 0.204971
- P_Z energy at gain >= 0.90: 0.100350

ctrl k=2:

- unweighted Z midpoint mean: -1.254381
- P_Z-energy-weighted Z midpoint mean: 0.102360
- unweighted secant-gain mean: 0.245397
- P_Z-energy-weighted gain mean: 0.600130
- effective gain: 0.650138
- P_Z energy at gain >= 0.75: 0.097665
- P_Z energy at gain >= 0.90: 0.076849

## Corr k=1 -> k=2

Within the common 330-item cohort:

- Q_T increased: 326/330
- unweighted Z midpoint increased: 1/330
- unweighted Z midpoint decreased: 329/330
- unweighted gain increased: 0/330
- unweighted gain decreased: 330/330
- P_Z-energy-weighted Z midpoint increased: 330/330
- P_Z-energy-weighted gain increased: 330/330
- effective gain increased: 330/330
- energy fraction at gain >= 0.75 increased: 330/330
- energy fraction at gain >= 0.90 increased: 330/330

Median changes:

- unweighted Z midpoint: -0.070899
- P_Z-energy-weighted Z midpoint: +11.615846
- unweighted gain: -0.012176
- P_Z-energy-weighted gain: +0.824705
- effective gain: +0.706018
- energy fraction gain >= 0.75: +0.841874
- energy fraction gain >= 0.90: +0.824884

Thus the corr k=2 transmission rise is not a broad layer-wide shift toward
higher softplus sensitivity.

The broad channel population moves slightly in the opposite direction.

Instead, pre-softplus divergence energy is selectively redistributed onto
channels whose absolute Z operating points have high softplus transmission.

## Corr versus ctrl at k=2

Within the common paired cohort:

- corr unweighted Z midpoint > ctrl: 0/330
- corr unweighted gain > ctrl: 0/330
- corr energy-weighted Z midpoint > ctrl: 330/330
- corr energy-weighted gain > ctrl: 330/330
- corr effective gain > ctrl: 330/330
- corr energy fraction gain >= 0.75 > ctrl: 330/330
- corr energy fraction gain >= 0.90 > ctrl: 330/330

Median corr-minus-ctrl differences:

- unweighted Z midpoint: -0.134780
- unweighted gain: -0.019542
- energy-weighted Z midpoint: +3.326087
- energy-weighted gain: +0.291922
- effective gain: +0.272240
- energy fraction gain >= 0.75: +0.767383
- energy fraction gain >= 0.90: +0.761319

Thus the corr-specific k=2 role separation is also energy-selective rather
than layer-wide.

Corr has a slightly lower broad operating point than ctrl, while its
divergence energy is concentrated much more strongly on high-transmission
channels.

## Corr k=2 -> k=3 collapse

Within the common cohort:

- Q_T decreased: 330/330
- unweighted Z midpoint decreased: 0/330
- unweighted gain decreased: 0/330
- energy-weighted Z midpoint decreased: 328/330
- energy-weighted gain decreased: 327/330
- effective gain decreased: 327/330
- energy fraction gain >= 0.75 decreased: 329/330
- energy fraction gain >= 0.90 decreased: 321/330

Thus the immediate collapse is again not explained by a broad decline in the
layer operating point.

It is associated with loss of the selective concentration of divergence
energy on high-transmission channels.

## Midpoint explanation

Across corr and ctrl k=1..k=3, the P_Z-energy-weighted RMS of sigmoid(Z_mid)
closely reproduces the exact effective secant gain.

For corr k=2:

- median |G_mid_RMS - G_eff|: 0.0011245
- maximum |G_mid_RMS - G_eff|: 0.0049108
- median secant-versus-midpoint RMS gap: 0.0012448
- maximum secant-versus-midpoint RMS gap: 0.0053234

The corresponding discrepancies are even smaller at the other inspected
coordinates.

Therefore the matched/swapped interval width contributes little to the
observed transmission pattern.

The absolute midpoint Z operating point of the energy-carrying channels
explains nearly all of it.

## Validated scientific conclusion

The layer-23 corr-specific k=2 softplus transient is an energy-selective
operating-point phenomenon, not a layer-wide operating-point shift.

From k=1 to k=2, the broad intermediate-channel population moves slightly
toward lower softplus transmission, while P_Z divergence energy moves
universally toward channels with high absolute-Z softplus transmission.

At k=2, corr shows the same separation from ctrl: its broad operating point
is slightly lower, but its divergence energy is universally more concentrated
on high-transmission channels.

The k=3 collapse reverses this selective energy concentration without a broad
decline in the layer operating point.

The transmission is almost entirely explained by the midpoint Z locations of
the channels carrying P_Z energy rather than by large matched/swapped secant
interval effects.

These are observational and algebraic results. They do not establish causal
downstream relevance.

## Next scientific boundary

Because normalized P_Z channel-energy weights are exactly proportional to:

`|mean(U_j) * delta_Z_j|^2`

the next minimal native-Mamba question is upstream of softplus:

Does the corr k=2 high-transmission concentration arise primarily because
delta_Z itself becomes selectively aligned with high-Z channels, because
mean(U) reweights those channels, or because their channel-wise alignment
changes?

B does not need a separate channel-selection audit for this question because
its norm is common across intermediate channels and cancels from normalized
P_Z channel-energy weights.

No causal intervention is implied by this next boundary.