# K0-RVG Ubar × Delta-Z Channel-Selection Audit

## Status

VALIDATED SCIENTIFIC EVIDENCE

Execution commit:

`d1982762474ac8fdefa32f08646debb7474115f8`

Parent operating-point freeze:

`e33e0edfdaad847a11fc4239f6252828f278030b`

No tokenizer, logits, task heads, training, causal intervention, PCA,
probe, or learned geometry was executed.

Raw channel vectors were not persisted.

## Scientific question

The frozen parent established that normalized per-channel P_Z energy is:

`p_product(j) ∝ mean(U_j)^2 * delta_Z_j^2`

because:

`P_Z[j,s] = mean(U_j) * mean(B_s) * delta_Z_j`

and the B-state norm is common across intermediate channels within one
pair/token.

This audit asked whether the corr-k2 high-softplus-transmission channel
selection is:

1. already present in delta-Z squared channel weighting;
2. attributable to mean-U squared weighting;
3. or substantially sharpened only when the two observed channel patterns
   are multiplied.

The compared observed distributions were:

`p_dz2(j) ∝ delta_Z_j^2`

`p_u2(j) ∝ mean(U_j)^2`

`p_product(j) ∝ mean(U_j)^2 * delta_Z_j^2`

The product distribution is exactly the frozen parent normalized P_Z channel
energy distribution.

## Artifact authentication

execution_manifest.json SHA256:

`5b174ecc7d73cbda67e36d25c1875aaa0387693c9afe438a863f725de9f53f76`

summary.json SHA256:

`743944cc424f8f08140e21bced8464e3f8b78f895a1e1528cd419e66d9623ee0`

Full local metrics artifact:

`uz_channel_selection_metrics.jsonl`

SHA256:

`864c8077da217a643985cf09afc640336e34d170bf9495b632ca71cabc770037`

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

Maximum discrepancy between product-weighted gain RMS and the frozen parent
effective softplus gain:

`4.6629367034256575e-15`

All frozen parent operating-point role/k medians were reproduced.

All common-cohort summary medians were independently recomputed.

## Common-330 channel-selection trajectory

Median gain RMS values:

corr k=1:

- delta-Z squared: 0.346857
- mean-U squared: 0.409458
- product: 0.209679
- joint excess over stronger single-factor reference: -0.199306

corr k=2:

- delta-Z squared: 0.493172
- mean-U squared: 0.433192
- product: 0.928779
- joint excess: +0.434683

corr k=3:

- delta-Z squared: 0.342743
- mean-U squared: 0.408193
- product: 0.597759
- joint excess: +0.172360

ctrl k=2:

- delta-Z squared: 0.314620
- mean-U squared: 0.423068
- product: 0.650138
- joint excess: +0.226778

Thus neither delta-Z squared nor mean-U squared alone reproduces the corr-k2
parent transmission pattern.

## High-transmission energy

Median fraction of each distribution on channels with softplus gain >= 0.90:

corr k=1:

- delta-Z squared: 0.056574
- mean-U squared: 0.067602
- product: 0.031880

corr k=2:

- delta-Z squared: 0.198707
- mean-U squared: 0.086153
- product: 0.883267

corr k=3:

- delta-Z squared: 0.033252
- mean-U squared: 0.037804
- product: 0.100350

ctrl k=2:

- delta-Z squared: 0.031280
- mean-U squared: 0.063592
- product: 0.076849

The corr-k2 product distribution therefore places approximately 88% of its
energy on gain >= 0.90 channels, while neither single-factor distribution does.

## Corr k=1 -> k=2

Within the common 330-item cohort:

- delta-Z gain RMS increased: 299/330
- mean-U gain RMS increased: 327/330
- product gain RMS increased: 330/330
- joint-excess gain increased: 330/330

- delta-Z gain>=0.90 fraction increased: 313/330
- mean-U gain>=0.90 fraction increased: 307/330
- product gain>=0.90 fraction increased: 330/330
- joint-excess gain>=0.90 fraction increased: 330/330

At corr k=2:

- product gain > delta-Z gain: 330/330
- product gain > mean-U gain: 330/330
- product gain > both single-factor references: 330/330
- joint-excess gain positive: 330/330
- joint-excess gain>=0.90 fraction positive: 330/330

Median k1 -> k2 changes:

- delta-Z gain RMS: +0.129081
- mean-U gain RMS: +0.024022
- product gain RMS: +0.706018
- joint-excess gain: +0.608154

- delta-Z gain>=0.90 fraction: +0.129793
- mean-U gain>=0.90 fraction: +0.022350
- product gain>=0.90 fraction: +0.824884
- joint-excess gain>=0.90 fraction: +0.688796

Thus delta-Z develops a meaningful but incomplete preference for
high-transmission channels.

Mean-U alone changes only modestly.

The observed product distribution sharply amplifies the high-transmission
selection beyond either single-factor channel distribution.

## Global U-squared / delta-Z-squared overlap

The product effect must not be described as a broad increase in global
U-squared / delta-Z-squared alignment.

From corr k=1 to k=2:

- cosine increased: 59/330
- cosine decreased: 271/330
- multiplicative enrichment increased: 78/330
- multiplicative enrichment decreased: 252/330

Median changes:

- cosine: -0.043838
- multiplicative enrichment: -2.840437

Therefore the corr-k2 product sharpening occurs despite weaker global overlap
by these aggregate measures.

The validated effect is specifically that the observed product channel weight
is selectively located on high-softplus-transmission channels.

It is not a general statement that U-squared and delta-Z-squared become more
globally aligned.

## Corr versus ctrl at k=2

Within the common paired cohort:

- corr delta-Z gain > ctrl: 328/330
- corr mean-U gain > ctrl: 319/330
- corr product gain > ctrl: 330/330
- corr joint-excess gain > ctrl: 317/330

- corr delta-Z gain>=0.90 fraction > ctrl: 330/330
- corr mean-U gain>=0.90 fraction > ctrl: 330/330
- corr product gain>=0.90 fraction > ctrl: 330/330
- corr joint-excess gain>=0.90 fraction > ctrl: 330/330

Median corr-minus-ctrl differences:

- delta-Z gain RMS: +0.176044
- mean-U gain RMS: +0.009605
- product gain RMS: +0.272240
- joint-excess gain: +0.212508

- delta-Z gain>=0.90 fraction: +0.165073
- mean-U gain>=0.90 fraction: +0.023133
- product gain>=0.90 fraction: +0.761319
- joint-excess gain>=0.90 fraction: +0.627130

Thus the corr-vs-ctrl k2 separation contains a substantial delta-Z component,
only a small mean-U-only component, and a much larger product-level
high-transmission localization.

## Corr k=2 -> k=3 collapse

Within the common cohort:

- delta-Z gain decreased: 256/330
- mean-U gain decreased: 305/330
- product gain decreased: 327/330
- joint-excess gain decreased: 328/330

- delta-Z gain>=0.90 fraction decreased: 297/330
- mean-U gain>=0.90 fraction decreased: 314/330
- product gain>=0.90 fraction decreased: 321/330
- joint-excess gain>=0.90 fraction decreased: 327/330

Thus the immediate collapse is again strongest at the product level.

## Effective channel count

Median effective channel counts:

corr k=1:

- delta-Z squared: 40.865
- mean-U squared: 39.760
- product: 1.428

corr k=2:

- delta-Z squared: 77.798
- mean-U squared: 35.733
- product: 2.564

corr k=3:

- delta-Z squared: 278.830
- mean-U squared: 51.826
- product: 14.814

ctrl k=2:

- delta-Z squared: 460.283
- mean-U squared: 27.399
- product: 2.439

The corr-k2 effect is therefore not simply increased global sparsity.

The product distribution is already extremely concentrated at k=1 and becomes
slightly less sparse at k=2.

What changes is primarily the location of that concentrated product energy:
at k=1 it lies mostly on low-transmission channels, while at k=2 it is
relocated onto high-transmission channels.

The similar corr/ctrl product effective channel counts at k=2 further show
that role separation is about channel identity / operating-point location,
not merely the number of active channels.

## Validated scientific conclusion

The layer-23 corr-specific k2 high-softplus-transmission concentration is not
explained by delta-Z channel selection alone and is not explained by mean-U
channel weighting alone.

Delta-Z squared develops a substantial high-transmission preference at k=2,
while mean-U squared changes only modestly.

However, the exact parent channel-energy distribution
`mean(U)^2 * delta_Z^2` is universally more high-transmission-selective than
either single-factor reference at corr k=2.

This product-level sharpening is cohort-wide and also strongly separates corr
from ctrl.

The effect is not a broad increase in global U-squared / delta-Z-squared
alignment: cosine and multiplicative-enrichment measures usually decrease from
corr k=1 to k=2.

The validated statement is therefore gain-conditioned and location-specific:
the observed mean-U and delta-Z channel patterns multiply so that product
energy is selectively placed on channels whose absolute Z operating points
have very high softplus transmission.

The k=3 collapse reverses this product-level high-transmission localization.

These are observational and algebraic reweighting results. They do not
establish causal downstream relevance.

## Next scientific boundary

Because:

`delta_Z = W_dt * delta_time_step`

with the dt-projection bias cancelling from matched-minus-swapped delta-Z,
the next minimal upstream question is whether the corr-k2 delta-Z
high-transmission preselection is already present in the low-dimensional
delta-time-step direction, or is primarily imposed by the fixed dt_proj row
geometry when that direction is projected into intermediate channels.

That question is upstream of softplus and upstream of the product weighting
studied here.

No training, intervention, probe, or causal claim is implied.