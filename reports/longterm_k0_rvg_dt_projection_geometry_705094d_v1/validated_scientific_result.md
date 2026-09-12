# K0-RVG dt-Projection Geometry Audit

## Status

VALIDATED SCIENTIFIC EVIDENCE

Execution commit:

`705094d2962403794c3ee5d511c843b6625e36cf`

Parent Ubar × delta-Z channel-selection freeze:

`634677160f18dfeed0c5de54a35542d649cd5a0a`

No tokenizer, logits, task heads, training, causal intervention, PCA,
probe, or learned geometry was executed.

Raw vectors were not persisted.

## Scientific question

The frozen parent established that delta-Z squared has a meaningful but
incomplete high-softplus-transmission preference at corr k=2.

This audit asked where that projected channel selection comes from.

For the fixed dt projection:

`delta_Z = W_dt * delta_r`

where delta_r is the matched-minus-swapped low-rank pre-dt-projection
time-step difference.

For output row w_j:

`delta_Z_j^2 = ||w_j||^2 * ||delta_r||^2 * cos(theta_j)^2`

Therefore, for nonzero delta_r, the normalized projected delta-Z squared
distribution is exactly proportional to:

`row_norm2_j * direction_cos2_j`

with:

`row_norm2_j = ||w_j||^2`

and:

`direction_cos2_j = cos(w_j, delta_r)^2`

The two marginal reference distributions were compared with their exact
channel-wise product.

## Runtime geometry

time-step rank:

`48`

dt_proj weight shape:

`(1536, 48)`

dt_proj bias shape:

`(1536,)`

dt_proj row-norm-squared range:

`0.07784044718029925` to `16.328619990945764`

## Artifact authentication

execution_manifest.json SHA256:

`880de78a3d787c120b6936737c2540a2f8c06a5c96ef0e20ac806017eb2f36cc`

summary.json SHA256:

`eb407ed465fb149186ea4d70253dc47a4105553818a970ad4343aa6824310b22`

Full local metrics artifact:

`dt_projection_geometry_metrics.jsonl`

SHA256:

`5e7fefdcc60fc7e0ffae7f98b9e73b89725c949a4e2caa3c79362545e359ebad`

## Validation

Trajectory rows:

`5376`

Common paired DDSSSSS cohort:

`330`

At k=-1:

- delta-r exactly zero: 672/672
- observed delta-Z exactly zero: 672/672
- projected delta-Z exactly zero: 672/672

Maximum relative reconstruction residual for:

`W_dt * delta_r = delta_Z`

was:

`8.773251472508623e-06`

Maximum product-versus-observed delta-Z-squared gain-RMS discrepancy:

`7.79037235276725e-07`

Maximum product-versus-observed gain>=0.90 fraction discrepancy:

`7.46595541778472e-07`

All frozen parent delta-Z role/k medians were reproduced.

All common-cohort summary medians were independently recomputed.

## Common-330 geometry

Median softplus gain RMS under each channel weighting:

corr k=1:

- fixed row-norm squared: 0.315476
- directional cos squared: 0.317668
- exact product: 0.346857
- observed delta-Z squared: 0.346857
- joint excess: +0.020651

corr k=2:

- fixed row-norm squared: 0.307218
- directional cos squared: 0.305432
- exact product: 0.493172
- observed delta-Z squared: 0.493172
- joint excess: +0.171761

corr k=3:

- fixed row-norm squared: 0.305585
- directional cos squared: 0.295435
- exact product: 0.342743
- observed delta-Z squared: 0.342743
- joint excess: +0.035863

ctrl k=2:

- fixed row-norm squared: 0.317610
- directional cos squared: 0.266565
- exact product: 0.314620
- observed delta-Z squared: 0.314620
- joint excess: -0.002762

Thus the corr-k2 projected high-gain selection is not present in either
marginal weighting alone.

## Corr k=1 -> k=2 geometry

Within the common cohort:

- row-reference gain increased: 0/330
- row-reference gain decreased: 330/330

- direction-reference gain increased: 151/330
- direction-reference gain decreased: 179/330

- product gain increased: 299/330
- joint-excess gain increased: 314/330

At corr k=2:

- product gain > row reference: 325/330
- product gain > direction reference: 330/330
- product gain > both references: 325/330
- joint-excess gain positive: 325/330
- joint-excess gain>=0.90 fraction positive: 330/330

Median k1 -> k2 changes:

- row gain: -0.008045
- direction gain: -0.015661
- product gain: +0.129081
- joint-excess gain: +0.130412

- row gain>=0.90 fraction: +0.002416
- direction gain>=0.90 fraction: +0.017222
- product gain>=0.90 fraction: +0.129793
- joint-excess gain>=0.90 fraction: +0.116122

Thus the temporal corr-k2 projected selection does not arise from either
marginal weighting becoming broadly more high-gain-selective.

It appears when the fixed row-norm and sample-specific directional weights
are multiplied channel by channel.

## Global row / direction overlap

This effect must not be described as increased global alignment between the
fixed row geometry and delta-r direction.

From corr k=1 to k=2:

- row-direction cosine increased: 13/330
- row-direction cosine decreased: 317/330

- multiplicative enrichment increased: 70/330
- multiplicative enrichment decreased: 260/330

Median changes:

- cosine: -0.060241
- enrichment: -0.182205

Therefore the validated effect is location-specific and gain-conditioned,
not a broad increase in row-direction alignment.

## Corr versus ctrl at k=2

Within the common cohort:

- corr row-reference gain > ctrl: 0/330
- corr direction-reference gain > ctrl: 259/330
- corr product gain > ctrl: 328/330
- corr joint-excess gain > ctrl: 329/330

For gain>=0.90 energy:

- corr row reference > ctrl: 31/330
- corr direction reference > ctrl: 326/330
- corr product > ctrl: 330/330
- corr joint excess > ctrl: 330/330

Median corr-minus-ctrl differences:

- row gain: -0.009969
- direction gain: +0.036629
- product gain: +0.176044
- joint-excess gain: +0.162472

- row gain>=0.90 fraction: -0.000880
- direction gain>=0.90 fraction: +0.028494
- product gain>=0.90 fraction: +0.165073
- joint-excess gain>=0.90 fraction: +0.158639

Thus fixed row-norm geometry alone cannot explain the corr-vs-ctrl k2
difference.

A directional component is present, but the exact projected channel
distribution shows substantially stronger separation than the direction-only
reference.

## Corr k=2 -> k=3 collapse

Within the common cohort:

- row gain decreased: 192/330
- direction gain decreased: 159/330
- product gain decreased: 256/330
- joint-excess gain decreased: 290/330

- row gain>=0.90 fraction decreased: 330/330
- direction gain>=0.90 fraction decreased: 286/330
- product gain>=0.90 fraction decreased: 297/330
- joint-excess gain>=0.90 fraction decreased: 296/330

The collapse is again strongest at the product / joint-localization level.

## Effective channel count

Median effective channel counts:

corr k=1:

- row: 483.769
- direction: 560.719
- product: 40.865

corr k=2:

- row: 483.769
- direction: 610.748
- product: 77.798

corr k=3:

- row: 483.769
- direction: 561.886
- product: 278.830

ctrl k=2:

- row: 483.769
- direction: 698.035
- product: 460.283

The fixed row distribution is invariant, as expected.

The corr-k2 product distribution is much more concentrated than either
marginal distribution, but it is less sparse than corr k=1.

Therefore the k2 effect is not simply increasing sparsity. It changes which
rows carry projected difference energy.

## Static magnitude / projection-transfer audit

A post-execution static audit used only the authenticated persisted scalar
metrics. No new model forward was executed.

Define:

`transfer = ||delta_Z||_2 / ||delta_r||_2`

which measures total norm transfer through the fixed dt projection for the
observed low-rank direction.

Common-cohort medians:

corr k=1:

- ||delta-r||: 1.375355
- ||delta-Z||: 6.552775
- transfer: 4.672461

corr k=2:

- ||delta-r||: 1.393869
- ||delta-Z||: 5.824327
- transfer: 4.138453

corr k=3:

- ||delta-r||: 0.400450
- ||delta-Z||: 1.623581
- transfer: 4.112868

ctrl k=2:

- ||delta-r||: 0.440044
- ||delta-Z||: 2.098744
- transfer: 4.736684

### Corr k=1 -> k=2

- delta-r norm increased: 160/330
- delta-r norm decreased: 170/330

- delta-Z norm increased: 121/330
- delta-Z norm decreased: 209/330

- transfer increased: 75/330
- transfer decreased: 255/330

Median paired changes:

- delta-r norm: -0.018071
- delta-Z norm: -0.788801
- transfer: -0.437684

Therefore the temporal corr-k2 high-gain selection is not explained by growth
of the low-rank difference magnitude or by stronger overall norm transfer
through dt_proj.

### Corr versus ctrl at k=2

- corr delta-r norm > ctrl: 330/330
- corr delta-Z norm > ctrl: 330/330
- corr transfer > ctrl: 49/330

Median corr-minus-ctrl differences:

- delta-r norm: +0.916364
- delta-Z norm: +3.581119
- transfer: -0.565236

Thus the absolute corr-vs-ctrl projected-difference magnitude separation is
associated with a much larger low-rank input difference, despite lower
overall projection transfer for corr.

This is distinct from the temporal k1-to-k2 high-gain localization question.

### Corr k=2 -> k=3

- delta-r norm decreased: 330/330
- delta-Z norm decreased: 330/330
- transfer decreased: 183/330

Median changes:

- delta-r norm: -0.972831
- delta-Z norm: -4.167981
- transfer: -0.119463

The k3 magnitude collapse is therefore dominated by collapse of the low-rank
difference magnitude, not by a large universal change in projection transfer.

## Validated scientific conclusion

The layer-23 corr-specific k2 delta-Z high-softplus-transmission preselection
is a gain-conditioned projected-channel localization phenomenon.

It is not explained by fixed dt_proj row-norm geometry alone.

It is not explained by low-rank delta-r directional weighting alone.

It is not explained by increased global row-direction alignment.

It is not explained temporally by growth of ||delta-r||.

It is not explained temporally by stronger total dt_proj norm transfer.

Instead, the fixed row-norm pattern and the sample-specific directional
projection pattern multiply channel by channel so that projected delta-Z
energy is selectively localized onto rows whose current absolute-Z operating
points have high softplus transmission.

For corr versus ctrl at k=2, a separate fact also holds: corr has a much
larger low-rank delta-r magnitude and consequently a much larger projected
delta-Z magnitude, despite lower overall projection transfer.

The temporal high-gain localization and the absolute role-level difference
magnitude must therefore be kept conceptually separate.

These are observational and algebraic geometry results. They do not establish
causal downstream relevance.

## Next scientific boundary

The remaining upstream object is the origin of the 48-dimensional delta-r
difference itself.

Before defining another audit, the exact frozen Mamba source expression that
produces the low-rank `time_step` vector should be statically authenticated.

The next stage should then ask which upstream native-Mamba component creates
the corr-specific low-rank delta-r magnitude/direction pattern, without
training, intervention, tokenizer execution, or learned probes.