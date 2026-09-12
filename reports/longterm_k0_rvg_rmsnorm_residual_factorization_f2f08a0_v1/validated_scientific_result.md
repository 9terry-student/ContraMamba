# K0-RVG Layer-23 RMSNorm Residual Factorization

## Status

VALIDATED SCIENTIFIC EVIDENCE

Runner commit:

`f2f08a0c0d9be04c2d9056aaa29aa22548b97984`

Runner SHA256:

`a4c5a1aea466df2e70b933050714c564ccf2d7db85f8bb9d36dc8c5e06937c4c`

Parent evidence freeze:

`55f27f7444e4cb8d9ae898719ad68b58c3cdf44d`

No tokenizer execution, logits, task heads, training, intervention, PCA, or
learned probe was executed. Raw vectors were not persisted.

## Authenticated source boundary

Transformers Mamba source SHA256:

`23c7b410e204b5da01732566de10c94b70a8418ecb608e409754b00332eb2a41`

Layer-23 block path:

`R -> RMSNorm -> X -> mixer`

Block output:

`R + mixer(X)`

RMSNorm forward SHA256:

`cdbbe12603e777d7097a001fcc5fede6684a0345fcc1c781ad5b0e25414d381f`

RMS epsilon:

`1e-5`

The exact runtime RMSNorm is:

`X = gamma * R * rsqrt(mean(R^2) + epsilon)`

For matched/swapped branches, define:

`delta_R = R_m - R_s`

`delta_s = s_m - s_s`

`R_bar = (R_m + R_s)/2`

`s_bar = (s_m + s_s)/2`

Then the exact symmetric product identity is:

`delta_X = Q_R + Q_s`

where:

`Q_R = gamma * s_bar * delta_R`

`Q_s = gamma * R_bar * delta_s`

and:

`||Q_R + Q_s||^2 =
 ||Q_R||^2 + ||Q_s||^2 + 2<Q_R,Q_s>`

## Artifact authentication

Full local metrics:

`rmsnorm_residual_factorization_metrics.jsonl`

SHA256:

`6ddf2301478abe3cec6533a8b97f2385080a1257373ac1ae54f5c72e4ab559fb`

summary.json SHA256:

`3f8bd3ed5fd7cf98f0f118a6ef8d685938270ed4780150eda69cd75de439aca8`

execution_manifest.json SHA256:

`0d3b4dce0cb2fd539053855ba57f313bd8b4a55817288f27d3c3bc1e22c78081`

Trajectory rows:

`5376`

Common DDSSSSS cohort:

`330`

Scientific forwards:

`1344`

Parent delta-X reproduction:

`PASS`

Maximum observed float32 symmetric-decomposition relative residual:

`1.7650609763139862e-06`

Maximum direct RMSNorm reconstruction relative residual:

`0.0`

Maximum float64 squared-norm closure error:

`9.947598300641403e-13`

At k=-1, delta-R, delta-X, Q-sum, Q-R, and Q-s were exactly zero for
672/672 pair-role trajectories.

## Common-cohort medians

corr k1:

- delta-R: 229.351555
- delta-X: 3.862948
- Q-R: 5.312835
- Q-s: 4.598351
- Q-R energy fraction: 0.569003
- vector addition ratio: 0.566265
- cross fraction: -0.679329

corr k2:

- delta-R: 110.178553
- delta-X: 4.737126
- Q-R: 4.961527
- Q-s: 0.928861
- Q-R energy fraction: 0.960844
- vector addition ratio: 0.941583
- cross fraction: -0.113420

corr k3:

- delta-R: 16.288422
- delta-X: 1.129992
- Q-R: 1.206653
- Q-s: 0.197785
- Q-R energy fraction: 0.978338
- vector addition ratio: 0.992276
- cross fraction: -0.015389

ctrl k1:

- delta-R: 172.007021
- delta-X: 2.415712
- Q-R: 4.142686
- Q-s: 3.165732
- Q-R energy fraction: 0.623896
- vector addition ratio: 0.484046
- cross fraction: -0.765699

ctrl k2:

- delta-R: 46.796017
- delta-X: 0.953187
- Q-R: 1.292350
- Q-s: 0.495760
- Q-R energy fraction: 0.862476
- vector addition ratio: 0.726343
- cross fraction: -0.472426

ctrl k3:

- delta-R: 13.992832
- delta-X: 0.891142
- Q-R: 0.898271
- Q-s: 0.182537
- Q-R energy fraction: 0.951541
- vector addition ratio: 0.983485
- cross fraction: -0.032756

## Q-R / Q-s structure

At corr k1:

- Q-R > Q-s: 290/330
- negative interaction: 330/330

At corr k2:

- Q-R > Q-s: 330/330
- negative interaction: 304/330

At corr k3:

- Q-R > Q-s: 330/330
- negative interaction: 207/330

At ctrl k1:

- Q-R > Q-s: 330/330
- negative interaction: 330/330

At ctrl k2:

- Q-R > Q-s: 330/330
- negative interaction: 306/330

At ctrl k3:

- Q-R > Q-s: 330/330
- negative interaction: 235/330

Thus the residual-vector term Q-R is the dominant RMSNorm term at k2 and k3
for every common-cohort item in both roles.

However, Q-s is not negligible at earlier positions, and interaction between
Q-R and Q-s is strongly destructive at k1 and remains materially destructive
at ctrl k2.

## Corr k1 -> k2

Using the exact scalar magnitude identity:

`||Q_sum|| = ||delta_R|| * RMSNorm_transfer`

where:

`RMSNorm_transfer = ||Q_sum|| / ||delta_R||`

counts are:

- delta-R up: 0/330
- delta-R down: 330/330
- RMSNorm transfer up: 330/330
- RMSNorm transfer down: 0/330
- Q-sum up: 193/330
- Q-sum down: 137/330

Absolute-log contribution dominance:

- delta-R: 137/330
- RMSNorm transfer: 193/330

Median paired log ratios:

- delta-R: -0.757654
- RMSNorm transfer: +0.910118
- Q-sum: +0.157391

Therefore raw layer-23 residual-stream difference contracts universally from
corr k1 to k2, while RMSNorm transfer increases universally.

The normalization effect opposes and often exceeds the residual contraction,
such that normalized delta-X increases in 193/330 items.

Thus the previously observed corr k1-to-k2 increase in delta-X is not inherited
from raw residual magnitude. It is produced at the normalization boundary in
an observational/algebraic sense.

## Corr k2 -> k3

Counts:

- delta-R up: 0/330
- delta-R down: 330/330
- RMSNorm transfer up: 317/330
- RMSNorm transfer down: 13/330
- Q-sum up: 2/330
- Q-sum down: 328/330

Absolute-log contribution dominance:

- delta-R: 328/330
- RMSNorm transfer: 2/330

Median paired log ratios:

- delta-R: -1.872523
- RMSNorm transfer: +0.703125
- Q-sum: -1.275437

Thus the corr k2-to-k3 normalized delta-X collapse is overwhelmingly inherited
from raw residual-stream delta-R collapse.

RMSNorm usually acts in the opposite direction and partially offsets that
collapse.

## Corr versus ctrl at k2

Counts:

- corr delta-R > ctrl: 327/330
- corr RMSNorm transfer > ctrl: 330/330
- corr Q-sum > ctrl: 330/330
- corr observed delta-X > ctrl: 330/330

Absolute-log contribution dominance:

- delta-R: 176/330
- RMSNorm transfer: 154/330

Direction:

- both delta-R and RMSNorm transfer favor corr: 327/330
- delta-R favors corr while RMSNorm transfer favors ctrl: 0/330

Median paired log corr/ctrl ratios:

- delta-R: +0.749433
- RMSNorm transfer: +0.718097
- Q-sum: +1.484877
- observed delta-X: +1.484877

Therefore corr-vs-ctrl k2 role separation is already strongly present before
RMSNorm, in the raw layer-23 residual stream.

RMSNorm does not merely preserve that separation. Its overall transfer also
systematically favors corr and adds a comparably sized positive role-separation
effect.

The two components have comparable median log magnitudes, so neither should be
treated as negligible.

## Corr versus ctrl k2: term magnitudes and interaction

Counts:

- corr Q-R > ctrl: 328/330
- corr Q-s > ctrl: 259/330
- corr Q-term RSS > ctrl: 329/330
- corr vector-addition ratio > ctrl: 286/330

Absolute-log dominance for Q-sum role separation:

- Q-term RSS: 309/330
- addition/interference ratio: 21/330

Median paired log corr/ctrl ratios:

- Q-R: +1.214646
- Q-s: +0.510425
- Q-term RSS: +1.137480
- addition ratio: +0.238186
- Q-sum: +1.484877

Because medians are taken independently, these median values are descriptive
and are not themselves an exact additive decomposition. Exact additivity holds
per item.

The Q-R residual-vector term supplies the dominant energy in both roles.

Interaction is predominantly destructive, but substantially more destructive
for ctrl at k2:

- corr median addition ratio: 0.941583
- ctrl median addition ratio: 0.726343
- corr median cross fraction: -0.113420
- ctrl median cross fraction: -0.472426

Thus stronger destructive Q-R/Q-s interaction on the ctrl side further
increases the normalized corr-vs-ctrl separation.

However, per-item absolute-log dominance shows that term-magnitude separation
is the larger component in 309/330 items, while differential vector addition is
larger in only 21/330.

## Validated scientific conclusion

Layer-23 RMSNorm materially reshapes the native residual-stream difference.

For corr k1 -> k2, raw residual difference contracts in every common-cohort
item, while RMSNorm transfer increases in every item. The normalization effect
often exceeds the raw contraction and reverses its sign at the normalized
delta-X boundary.

For corr k2 -> k3, the opposite composition holds. Raw residual delta-R
collapse dominates in 328/330 items, while RMSNorm usually partially
compensates.

For corr versus ctrl at k2, role separation has two substantial components:

1. a strong upstream residual-stream delta-R separation already present before
   RMSNorm;
2. an additional RMSNorm transfer advantage that systematically favors corr.

Within RMSNorm, Q-R is the dominant-energy term. Q-s interacts destructively
with Q-R, especially for ctrl, and this asymmetric destructive interaction
provides an additional but usually secondary contribution to role separation.

Therefore the normalized mixer-input separation cannot be characterized as
either purely upstream or purely normalization-generated. At k2 it is an
upstream residual separation that is materially amplified by RMSNorm.

All conclusions are observational/algebraic and do not establish downstream
causal importance.

## Relation to the hidden in-projection result

The subsequent hidden in-projection was previously shown to attenuate the
corr-vs-ctrl k2 normalized-input separation:

- delta-X median paired log corr/ctrl ratio: +1.484877
- hidden in-projection transfer median paired log corr/ctrl ratio: -0.623545
- resulting delta-H median paired log corr/ctrl ratio: +0.844772

The current result shows that the large delta-X separation entering that
projection consists of both pre-existing residual-stream separation and
RMSNorm amplification.

Thus the local layer-23 sequence is:

`residual role separation`
`-> additional RMSNorm amplification`
`-> partial hidden in-projection attenuation`

This is a factor-associated algebraic description, not a causal chain claim.

## Next scientific boundary

The next primary boundary is the producer of layer-23 raw residual R.

The authenticated backbone source establishes that layer-23 raw input is the
output of layer 22.

For layer 22:

`R_23 = R_22 + Y_22`

where:

`Y_22 = mixer_22(RMSNorm_22(R_22))`

Therefore the next exact audit should decompose:

`delta_R_23 = delta_R_22 + delta_Y_22`

with squared-norm interaction:

`||delta_R_23||^2 =
 ||delta_R_22||^2 +
 ||delta_Y_22||^2 +
 2<delta_R_22, delta_Y_22>`

The primary questions are:

1. whether corr-vs-ctrl k2 residual separation is already present at the
   layer-22 input residual;
2. whether the layer-22 mixer update enlarges or suppresses it;
3. whether corr k2-to-k3 residual collapse is inherited from R_22 or produced
   by the layer-22 update/addition geometry.

No new scientific execution should precede freezing the present evidence.