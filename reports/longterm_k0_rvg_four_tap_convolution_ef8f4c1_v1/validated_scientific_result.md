# K0-RVG Exact Four-Tap Depthwise-Convolution Decomposition

## Status

VALIDATED SCIENTIFIC EVIDENCE

Runner commit:

`ef8f4c12b1ebc00b178b5f1ad4ab45a19a85e801`

Runner SHA256:

`49cd2708bf7ba3ef9f6beb5987f845d585cb9f991e459939c86cb303e0ce5070`

Parent evidence freeze:

`77276e1007e797d00dcf4f355c9a7c99d5f19af2`

No tokenizer execution, logits, task heads, training, intervention, PCA, or
learned probe was executed. Raw vectors were not persisted.

## Exact convolution identity

At layer 23:

`Q_l(t) = K_(3-l) elementwise delta_H_(t-l)`

and:

`delta_C_t = Q_0 + Q_1 + Q_2 + Q_3`

with causal lag to kernel-index mapping:

- lag0 -> kernel3
- lag1 -> kernel2
- lag2 -> kernel1
- lag3 -> kernel0

Conv1d bias cancels exactly in matched-minus-swapped differences.

Squared-norm decomposition:

`||sum_l Q_l||^2 = sum_l ||Q_l||^2 + 2 sum_(l<m) <Q_l,Q_m>`

## Frozen kernel structure

Layer-23 kernel RMS by causal lag:

- lag0: 0.2332235421401241
- lag1: 0.029755534435177415
- lag2: 0.005695514971505627
- lag3: 0.0

The lag3 kernel is exactly zero.

Thus the four causal taps have strongly nonuniform fixed magnitude.

## Artifact authentication

Full local metrics:

`four_tap_convolution_metrics.jsonl`

SHA256:

`9af296dc8f9033680fff9cd3d21ca2b299b58755ace4dbca0fb5779384f12e4d`

summary.json SHA256:

`0ad1368107c07b7f0bc2b2a23bf4694da3ff68efaeac77690c19d39f3a038520`

execution_manifest.json SHA256:

`e4864e357b2a7299b53e95bb6ce1da42c5cae21c71a1019d87cc5eb1fc01a411`

Trajectory rows:

`5376`

Common DDSSSSS cohort:

`330`

Scientific forwards:

`1344`

Parent delta-C reproduction:

`PASS`

Maximum four-tap reconstruction relative residual:

`2.0657642883862333e-06`

Maximum squared-norm closure error:

`3.552713678800501e-14`

Maximum exact per-tap factorization absolute error:

`8.881784197001252e-16`

Maximum exact per-tap factorization relative error:

`2.1651479788325842e-16`

Q3 was exactly zero in:

`5376/5376`

rows.

## Q0 dominance

For every common-cohort item, Q0 was the largest tap contribution at k1, k2,
and k3 in both roles:

- corr k1: 330/330
- corr k2: 330/330
- corr k3: 330/330
- ctrl k1: 330/330
- ctrl k2: 330/330
- ctrl k3: 330/330

Median Q0 share of tap RSS energy:

- corr k1: 0.979372
- corr k2: 0.982553
- corr k3: 0.957953
- ctrl k1: 0.970833
- ctrl k2: 0.896217
- ctrl k3: 0.979776

Therefore the convolution output difference is overwhelmingly associated with
the current-token lag0 tap, despite the total pre-convolution receptive-field
difference being dominated by progressively older lags.

## Historical divergence-packet transport

The previously validated receptive-field audit showed that the original
divergence packet becomes the strict dominant pre-convolution RF energy
component at:

- k1 -> lag1
- k2 -> lag2
- k3 -> lag3

for 330/330 items in both roles.

After fixed convolution weighting, however, the transported packet contributes:

corr:

- k1 lag1 Q median: 0.451055
- k2 lag2 Q median: 0.094596
- k3 lag3 Q median: 0

ctrl:

- k1 lag1 Q median: 0.433978
- k2 lag2 Q median: 0.085855
- k3 lag3 Q median: 0

Median tap-RSS energy fractions of that transported packet:

corr:

- k1: 0.020628
- k2: 0.001702
- k3: 0

ctrl:

- k1: 0.029167
- k2: 0.010717
- k3: 0

Thus large historical RF energy is not equivalent to large contribution to the
convolution output. As the divergence packet ages into weaker taps, its direct
contribution becomes small and finally exactly zero at lag3.

## Tap interaction

Median vector-addition ratios are close to one:

corr:

- k1: 1.000064
- k2: 1.001946
- k3: 0.993107

ctrl:

- k1: 0.998857
- k2: 1.004087
- k3: 0.995082

Median cross fractions of tap RSS squared norm are correspondingly small:

corr:

- k1: +0.000127
- k2: +0.003896
- k3: -0.013738

ctrl:

- k1: -0.002285
- k2: +0.008190
- k3: -0.009812

Therefore multi-tap constructive/destructive interaction is not the principal
source of the observed delta-C magnitude pattern.

## Corr k1 -> k2: Q0 factorization

Exact per-item magnitude identity:

`||Q0|| = ||delta_H_current|| * q0_tap_transfer`

Counts:

- current delta-H up: 42/330
- current delta-H down: 288/330
- q0 tap transfer up: 238/330
- q0 tap transfer down: 92/330
- Q0 up: 54/330
- Q0 down: 276/330

Absolute log contribution dominance:

- current delta-H magnitude: 306/330
- q0 tap transfer: 24/330

Median log ratios:

- current delta-H: -0.340143
- q0 tap transfer: +0.032294
- Q0: -0.312085

Thus the corr k1-to-k2 Q0 contraction is primarily associated with reduction
of current-token pre-convolution delta-H magnitude, while lag0 channel-
conditioned tap transfer slightly increases in the median.

## Corr k2 -> k3: Q0 factorization

Counts:

- current delta-H up: 0/330
- current delta-H down: 330/330
- q0 tap transfer up: 12/330
- q0 tap transfer down: 318/330
- Q0 up: 0/330
- Q0 down: 330/330

Absolute log contribution dominance:

- current delta-H magnitude: 330/330
- q0 tap transfer: 0/330

Median log ratios:

- current delta-H: -0.836166
- q0 tap transfer: -0.143125
- Q0: -1.003904

The corr k2-to-k3 Q0 collapse is therefore dominated universally by
current-token pre-convolution delta-H contraction, with a smaller same-
direction reduction in lag0 channel-conditioned transfer.

## Corr versus ctrl at k2: Q0 factorization

Counts:

- corr current delta-H > ctrl: 330/330
- corr q0 tap transfer > ctrl: 330/330
- corr normalized q0 transfer > ctrl: 330/330
- corr Q0 > ctrl: 330/330
- corr delta-C > ctrl: 330/330

Absolute log contribution dominance:

- current delta-H magnitude: 330/330
- q0 tap transfer: 0/330

Median paired log corr/ctrl ratios:

- current delta-H: +0.844772
- q0 tap transfer: +0.217875
- normalized q0 transfer: +0.217875
- Q0: +1.057501
- delta-C: +1.009722

Thus corr-vs-ctrl k2 convolution separation is primarily already present in
the current-token pre-convolution delta-H magnitude. A smaller but perfectly
systematic same-direction channel-conditioned lag0 tap-transfer advantage is
also present.

Because the lag0 kernel itself is fixed and identical between roles, that
secondary transfer difference reflects the channel distribution/direction of
delta-H relative to the fixed lag0 kernel, not a different kernel.

## Corr versus ctrl at k2: tap addition

Counts:

- corr tap RSS > ctrl: 330/330
- corr addition ratio > ctrl: 127/330

Median paired log ratios:

- tap RSS corr/ctrl: +1.010926
- addition ratio corr/ctrl: -0.002044
- delta-C corr/ctrl: +1.009722

Therefore multi-tap vector addition contributes essentially no positive
role-separation effect at k2.

## Validated scientific conclusion

The large pre-convolution receptive-field difference identified in the parent
audit is dominated by an aging historical divergence packet, but that packet
does not dominate the layer-23 depthwise-convolution output.

The fixed convolution kernel strongly favors the current-token lag0 tap:

- lag0 kernel RMS is about an order of magnitude larger than lag1;
- lag1 is itself substantially larger than lag2;
- lag3 is exactly zero.

Consequently, historical difference energy can remain large inside the
four-token receptive field while contributing little or nothing to delta-C.

Across corr and ctrl k1-k3, Q0 is the dominant tap contribution for every
common-cohort item.

The temporal Q0 contraction is primarily associated with contraction of
current-token pre-convolution delta-H magnitude:

- corr k1 -> k2: current delta-H is the larger absolute log factor in 306/330;
- corr k2 -> k3: current delta-H is the larger absolute log factor in 330/330.

At corr versus ctrl k2, current delta-H magnitude is likewise the larger
absolute log factor in 330/330. The fixed lag0 kernel also exhibits a smaller
systematic channel-conditioned transfer advantage for corr.

Tap interaction is small and does not explain the role separation.

Therefore the parent audit's apparent reduction in whole-receptive-field
convolution transfer is now resolved more specifically:

the receptive field retains large historical difference energy in weak or
zero causal taps, while the actual convolution output is dominated by the
strong current-token tap acting on a shrinking and role-dependent current
delta-H vector.

This result is observational and algebraic. It does not establish downstream
causal importance.

## Next scientific boundary

The next narrow boundary is upstream of current-token `H_preconv`.

At layer 23 the hidden branch is produced by the bias-free in-projection.
Let X_t denote the layer-23 input state and W_H the hidden-branch half of the
in-projection matrix.

Then:

`delta_H_t = W_H delta_X_t`

and the next audit can use the exact scalar magnitude factorization:

`||delta_H_t|| = ||delta_X_t|| * (||delta_H_t|| / ||delta_X_t||)`

to distinguish:

1. upstream layer-input delta-X magnitude;
2. direction-conditioned transfer through the fixed hidden-branch in-projection.

This is the appropriate next boundary for both temporal current-H contraction
and corr-vs-ctrl k2 separation.

No new scientific execution should precede freezing the present evidence.