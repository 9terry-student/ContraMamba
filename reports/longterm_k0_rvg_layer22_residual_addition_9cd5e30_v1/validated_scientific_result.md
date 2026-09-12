# K0-RVG Layer-22 Residual Addition Audit

## Status

VALIDATED SCIENTIFIC EVIDENCE

Runner commit:

`9cd5e3036d2528079913afcd8c3d258bc1fbd10e`

Runner SHA256:

`ffeed8c4ce8b7edd7d62100d7f94ad8c558e2604c0a1a68cfc8ffc896ee906ce`

Parent evidence freeze:

`b29e05dd384cddddf8754a9e9925ff9753f49bdb`

No tokenizer execution, logits, task heads, training, causal intervention,
PCA, or learned probe was executed. Raw vectors were not persisted.

## Authenticated boundary

The frozen runtime source establishes:

`R22 -> RMSNorm22 -> Mixer22 -> Y22`

and:

`R23 = R22 + Y22`

Therefore:

`delta_R23 = delta_R22 + delta_Y22`

with exact real-valued squared-norm identity:

`||delta_R23||^2 =
 ||delta_R22||^2 +
 ||delta_Y22||^2 +
 2<delta_R22,delta_Y22>`

Mamba source SHA256:

`23c7b410e204b5da01732566de10c94b70a8418ecb608e409754b00332eb2a41`

Block forward SHA256:

`0f808b4d539a496e1681c81d39799d7072fa5c6da381a07cfa19f35fe825b3e4`

Backbone forward SHA256:

`3f332bd50e6ea4ff64468c8d748672f90ffc43912d22c9d25d02b3e87d1661a6`

## Artifact authentication

Full local metrics:

`layer22_residual_addition_metrics.jsonl`

SHA256:

`62cd93c581140bcc114f8e9e5d363cbde71efc3ce4bbe8ff6e19e72d2abac84e`

summary.json SHA256:

`8641a3a52168bd69f3c5625d11efae80bcf5cf5f13205d801959df50033831d4`

execution_manifest.json SHA256:

`d6b020a47958141285edda3d3e97efd961893940bb905a2b0716e05adb052d17`

Trajectory rows:

`5376`

Common DDSSSSS cohort:

`330`

Scientific forwards:

`1344`

Parent R23 reproduction:

`PASS`

Maximum observed decomposition relative residual:

`1.3268015394738166e-06`

Maximum direct branch R23 reconstruction relative residual:

`0.0`

Maximum float64 squared-norm closure error:

`1.0186340659856796e-10`

At k=-1, delta-R22, delta-Y22, delta-R23, and algebraic Q-sum were exactly
zero for 672/672 pair-role trajectories.

## Common-cohort medians

corr k1:

- delta-R22: 157.182192
- delta-Y22: 115.364063
- delta-R23: 229.351555
- R22 energy fraction: 0.653297
- Y22 energy fraction: 0.346703
- addition ratio: 1.165169
- cross fraction: +0.357620
- residual-add transfer: 1.446540

corr k2:

- delta-R22: 39.869333
- delta-Y22: 90.540620
- delta-R23: 110.178553
- R22 energy fraction: 0.179020
- Y22 energy fraction: 0.820980
- addition ratio: 1.100779
- cross fraction: +0.211714
- residual-add transfer: 2.587487

corr k3:

- delta-R22: 10.474935
- delta-Y22: 10.058254
- delta-R23: 16.288422
- R22 energy fraction: 0.516178
- Y22 energy fraction: 0.483822
- addition ratio: 1.087812
- cross fraction: +0.183335
- residual-add transfer: 1.506140

ctrl k1:

- delta-R22: 117.536795
- delta-Y22: 90.390588
- delta-R23: 172.007021
- R22 energy fraction: 0.632059
- Y22 energy fraction: 0.367941
- addition ratio: 1.158277
- cross fraction: +0.341606
- residual-add transfer: 1.459071

ctrl k2:

- delta-R22: 32.987825
- delta-Y22: 27.960542
- delta-R23: 46.796017
- R22 energy fraction: 0.601748
- Y22 energy fraction: 0.398252
- addition ratio: 1.084241
- cross fraction: +0.175578
- residual-add transfer: 1.405072

ctrl k3:

- delta-R22: 9.555207
- delta-Y22: 7.396159
- delta-R23: 13.992832
- R22 energy fraction: 0.598390
- Y22 energy fraction: 0.401610
- addition ratio: 1.076783
- cross fraction: +0.159462
- residual-add transfer: 1.370776

## Term structure

At corr k1:

- R22 > Y22: 316/330
- positive interaction: 330/330

At corr k2:

- Y22 > R22: 330/330
- positive interaction: 330/330

At corr k3:

- R22 > Y22: 175/330
- Y22 > R22: 155/330
- positive interaction: 326/330

At ctrl k1:

- R22 > Y22: 327/330
- positive interaction: 330/330

At ctrl k2:

- R22 > Y22: 252/330
- positive interaction: 329/330

At ctrl k3:

- R22 > Y22: 248/330
- positive interaction: 306/330

Thus corr k2 is qualitatively distinctive: the mixer-update difference Y22
is larger than the incoming residual difference R22 in every common-cohort
item.

## Corr k1 -> k2

Using:

`||delta_R23|| =
 ||delta_R22|| * residual_add_transfer`

counts are:

- R22 up: 0/330
- R22 down: 330/330
- residual-add transfer up: 330/330
- residual-add transfer down: 0/330
- R23 up: 0/330
- R23 down: 330/330

Absolute-log dominance:

- R22: 330/330
- residual-add transfer: 0/330

Median paired log ratios:

- R22: -1.390288
- residual-add transfer: +0.576698
- R23: -0.757654

Therefore the corr k1-to-k2 residual-stream contraction is already strongly
present at the layer-22 input.

The layer-22 block acts in the opposite direction: its residual-add transfer
increases universally and partially offsets that upstream contraction.

It does not reverse the contraction at R23.

## Corr k2 -> k3

Counts:

- R22 down: 330/330
- residual-add transfer down: 329/330
- R23 down: 330/330

Absolute-log dominance:

- R22: 324/330
- residual-add transfer: 6/330

Median paired log ratios:

- R22: -1.413906
- residual-add transfer: -0.504280
- R23: -1.872523

Therefore the majority of the R23 k2-to-k3 contraction is already present in
the incoming R22 residual stream.

Unlike k1-to-k2, however, layer-22 residual-add transfer also decreases and
reinforces the contraction.

## Corr versus ctrl at k2

Counts:

- corr R22 > ctrl: 250/330
- corr residual-add transfer > ctrl: 329/330
- corr R23 > ctrl: 327/330
- observed corr R23 > ctrl: 327/330

Absolute-log dominance:

- R22 role difference: 16/330
- residual-add transfer role difference: 314/330

Direction:

- both R22 and transfer favor corr: 249/330
- R22 favors corr while transfer favors ctrl: 1/330
- R22 favors ctrl while transfer favors corr: 80/330

Median paired log corr/ctrl ratios:

- R22: +0.148129
- residual-add transfer: +0.590938
- R23: +0.749433
- observed R23: +0.749433

Thus the layer-23 raw residual role separation is not primarily inherited from
the layer-22 input residual.

At this boundary, the layer-22 block's residual-add transfer supplies the
larger role-dependent factor in 314/330 items.

## Corr versus ctrl k2: term magnitude and interaction

Counts:

- corr R22 > ctrl: 250/330
- corr Y22 > ctrl: 327/330
- corr term RSS > ctrl: 327/330
- corr addition ratio > ctrl: 255/330

Role-log comparison:

- Y22 role contrast stronger than R22: 326/330
- R22 role contrast stronger than Y22: 4/330

RSS versus interaction:

- RSS magnitude role effect dominant: 328/330
- addition/interference role effect dominant: 2/330

Median paired log corr/ctrl ratios:

- R22: +0.148129
- Y22: +1.090169
- term RSS: +0.731828
- addition ratio: +0.015721
- Q-sum / R23: +0.749433

Therefore the strong layer-22-local corr-vs-ctrl role separation is associated
primarily with the magnitude of the mixer update delta-Y22, not with
residual/update vector-addition geometry.

The interaction between delta-R22 and delta-Y22 is predominantly constructive
in both roles and contributes only a small differential role effect.

## Corr k2 -> k3 term contraction

Counts:

- R22 down: 330/330
- Y22 down: 330/330
- Y22 absolute-log contraction stronger: 329/330
- R22 absolute-log contraction stronger: 1/330

Median paired log ratios:

- R22 k3/k2: -1.413906
- Y22 k3/k2: -2.099191

Thus the mixer-update difference contracts even more sharply than the incoming
residual difference from k2 to k3.

This does not mean Y22 dominates the total R23 contraction: the exact scalar
R23 factorization still assigns larger absolute-log contribution to R22 in
324/330 items.

Instead, Y22's stronger contraction manifests as an additional decrease in
residual-add transfer, reinforcing the upstream R22 collapse.

## Validated scientific conclusion

The layer-22 residual-addition boundary has different compositions for role
separation and temporal contraction.

For corr versus ctrl at k2, the incoming residual R22 contains only a modest
role separation. The mixer-update term Y22 carries a substantially larger
corr-vs-ctrl difference, with Y22 role contrast exceeding R22 role contrast in
326/330 common-cohort items.

Residual/update interaction is predominantly constructive but its differential
role effect is small. The role separation is therefore associated primarily
with the magnitude of Y22 rather than with addition geometry.

For corr k1 -> k2, R22 contracts strongly while layer-22 transfer increases and
partially offsets the contraction.

For corr k2 -> k3, both R22 and Y22 contract. R22 remains the larger component
of the total R23 log contraction, while the even sharper Y22 contraction causes
layer-22 transfer itself to fall and reinforces the collapse.

All conclusions are observational/algebraic and do not establish causal
importance for downstream behavior.

## Relation to downstream results

The validated local sequence is now:

`layer22 incoming residual R22`
`-> layer22 mixer update Y22 supplies strong k2 role-selective magnitude`
`-> constructive residual addition produces R23`
`-> layer23 RMSNorm further amplifies corr-vs-ctrl separation`
`-> layer23 hidden in-projection partially attenuates it`

This is a factor-associated algebraic description, not a causal chain claim.

## Next scientific boundary

The next primary boundary is inside the layer-22 mixer update Y22.

This choice is justified by two independent observations:

1. at corr versus ctrl k2, Y22 role contrast is stronger than R22 role contrast
   in 326/330 common-cohort items;
2. from corr k2 to k3, Y22 contracts more strongly than R22 in 329/330 items.

The next step should first authenticate the exact runtime source path producing
Y22 and identify the narrowest exact algebraic boundary immediately upstream
of the layer-22 mixer output.

No new scientific execution should precede freezing the present evidence.