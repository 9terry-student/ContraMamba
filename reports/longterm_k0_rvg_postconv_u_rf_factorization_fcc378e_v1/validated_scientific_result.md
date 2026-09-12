# K0-RVG Post-Convolution Receptive-Field Factorization Audit

## Status

VALIDATED SCIENTIFIC EVIDENCE

Execution runner commit:

`fcc378e2830aa860ddef66ac8503b2f40ccdb4ba`

Runner SHA256:

`0f3e9527ff3d5a3c91e646547b1ab17e134e799130833100446bd976e4a4d052`

Parent time-step-subspace-transfer evidence freeze:

`2c54d9dbe62b3eb8581d6d582c497407212fe2f0`

No tokenizer, logits, task heads, training, causal intervention, PCA, or learned
probe was executed.

Raw vectors were not persisted.

## Scientific boundary

For layer 23, the authenticated slow path is:

`H_preconv_4token_receptive_field -> depthwise Conv1d -> C_preact -> SiLU -> U_postact`

The convolution has:

- intermediate width: 1536
- groups: 1536
- kernel size: 4
- padding: 3
- bias: present
- activation: SiLU

The scientific forward used:

- `use_cache=False`
- no attention mask

Therefore the exact pre-convolution object for output position t is the
four-token causal receptive field, not only the current-token hidden vector.

Define:

`delta_H_RF(t) = [delta_H_t, delta_H_(t-1), delta_H_(t-2), delta_H_(t-3)]`

and:

`conv_rf_transfer = ||delta_C_t|| / ||delta_H_RF(t)||`

`activation_transfer = ||delta_U_t|| / ||delta_C_t||`

Then the exact scalar factorization is:

`||delta_U_t|| = ||delta_H_RF(t)|| * conv_rf_transfer * activation_transfer`

The current-token delta-H norm is retained only as a diagnostic.

## Artifact authentication

Full local metrics:

`postconv_u_factorization_metrics.jsonl`

SHA256:

`2ac15f17ef059307cabeecbc218182951c05e02acb88d4fadd266c36aee7be86`

summary.json SHA256:

`76a6efde74276b3e26fdbc7e026c2443b3067bc85bb21e40bb0ce1436d0e88d6`

execution_manifest.json SHA256:

`143f76a6e86099d6fe196e1f5af0594b830a02eac3c6a626f602f5daa09ca8c1`

## Validation

Trajectory rows:

`5376`

Common DDSSSSS cohort:

`330`

Scientific forwards:

`1344`

Parent delta-U medians reproduced:

`PASS`

At k=-1:

- delta-H-current zero: 672/672
- delta-H-RF zero: 672/672
- delta-C zero: 672/672
- delta-U zero: 672/672

Maximum convolution reconstruction relative residual:

`7.672914023384412e-08`

Maximum activation reconstruction relative residual:

`5.518844553449212e-08`

Maximum scalar factorization absolute error:

`8.881784197001252e-16`

Maximum scalar factorization relative error:

`3.2340654961822686e-16`

All common-cohort summary medians were independently recomputed.

## Common-330 trajectory

### corr k=1

- delta-H-current: 9.502844
- delta-H-RF: 23.550700
- delta-C: 3.108996
- delta-U: 1.888356
- conv-RF transfer: 0.139212
- activation transfer: 0.601217

Median RF lag-energy fractions:

- lag0: 0.183229
- lag1: 0.816771
- lag2: 0
- lag3: 0

### corr k=2

- delta-H-current: 6.951394
- delta-H-RF: 24.454390
- delta-C: 2.320440
- delta-U: 1.321293
- conv-RF transfer: 0.096510
- activation transfer: 0.551823

Median RF lag-energy fractions:

- lag0: 0.082305
- lag1: 0.165681
- lag2: 0.743100
- lag3: 0

### corr k=3

- delta-H-current: 2.837561
- delta-H-RF: 24.658415
- delta-C: 0.829732
- delta-U: 0.478579
- conv-RF transfer: 0.036138
- activation transfer: 0.532104

Median RF lag-energy fractions:

- lag0: 0.014543
- lag1: 0.081108
- lag2: 0.162411
- lag3: 0.730554

### ctrl k=1

- delta-H-current: 8.073882
- delta-H-RF: 22.409006
- delta-C: 2.501107
- delta-U: 1.398208
- conv-RF transfer: 0.122722
- activation transfer: 0.559174

Median RF lag-energy fractions:

- lag0: 0.161006
- lag1: 0.838994
- lag2: 0
- lag3: 0

### ctrl k=2

- delta-H-current: 2.827695
- delta-H-RF: 22.511613
- delta-C: 0.799419
- delta-U: 0.486091
- conv-RF transfer: 0.039735
- activation transfer: 0.604498

Median RF lag-energy fractions:

- lag0: 0.019821
- lag1: 0.156917
- lag2: 0.827389
- lag3: 0

### ctrl k=3

- delta-H-current: 2.399005
- delta-H-RF: 22.605205
- delta-C: 0.668077
- delta-U: 0.334294
- conv-RF transfer: 0.031969
- activation transfer: 0.490767

Median RF lag-energy fractions:

- lag0: 0.012853
- lag1: 0.019533
- lag2: 0.154839
- lag3: 0.818337

## Exact receptive-field lag transport

Across the common cohort, lag-norm transport identity was checked for:

`18480`

comparisons.

Maximum absolute error:

`0.0`

At each post-divergence coordinate, the expected historical lag was the strict
dominant receptive-field energy component for every item in both roles:

corr:

- k=1: lag1 strict dominant 330/330
- k=2: lag2 strict dominant 330/330
- k=3: lag3 strict dominant 330/330

ctrl:

- k=1: lag1 strict dominant 330/330
- k=2: lag2 strict dominant 330/330
- k=3: lag3 strict dominant 330/330

Thus the large pre-convolution receptive-field difference does not simply
disappear over k=1..3. Its norm contribution is transported through the finite
causal receptive field as the original divergence moves into progressively
older lags.

This is a receptive-field transport statement, not a downstream causal claim.

## Corr k=1 -> k=2

Item-level paired counts:

- delta-H-RF increased: 330/330
- delta-H-RF decreased: 0/330
- convolution transfer increased: 28/330
- convolution transfer decreased: 302/330
- activation transfer increased: 116/330
- activation transfer decreased: 214/330
- delta-U increased: 45/330
- delta-U decreased: 285/330

The convolution term had the largest absolute log change among the three
factorization terms in:

`269/330`

items.

Median log ratios:

- RF magnitude: +0.042945
- convolution transfer: -0.356780
- activation transfer: -0.079460
- delta-U: -0.397616

Therefore the corr k1-to-k2 delta-U contraction is not attributable to loss of
pre-convolution RF difference magnitude. RF magnitude universally increases.
The dominant changing factor is instead reduced norm transfer through the
fixed depthwise convolution, with a smaller additional activation effect.

## Corr versus ctrl at k=2

Item-level paired counts:

- corr RF magnitude > ctrl: 318/330
- corr convolution transfer > ctrl: 330/330
- corr activation transfer > ctrl: 97/330
- corr delta-U > ctrl: 329/330

The convolution term had the largest absolute log contribution among the
three factorization terms in:

`326/330`

items.

Convolution favored corr while activation opposed the corr separation in:

`233/330`

items.

Median paired log ratios, reported separately:

- RF magnitude corr/ctrl: +0.134389
- convolution transfer corr/ctrl: +0.863160
- activation transfer corr/ctrl: -0.108909
- delta-U corr/ctrl: +0.922051

Because these are separate medians, they are not required to sum. The exact
three-factor log identity was validated item by item.

The corr-specific k2 delta-U separation is therefore localized primarily to
the depthwise convolution transformation stage in this algebraic
factorization. Pre-convolution RF magnitude provides a smaller same-direction
difference. SiLU is not the source of the separation and frequently acts in
the opposite direction.

## Corr k=2 -> k=3

Item-level paired counts:

- RF magnitude decreased: 0/330
- convolution transfer decreased: 330/330
- activation transfer decreased: 150/330
- delta-U decreased: 329/330

The convolution term had the largest absolute log change in:

`328/330`

items.

Median log ratios:

- RF magnitude: +0.007325
- convolution transfer: -1.011299
- activation transfer: +0.022677
- delta-U: -0.965262

Thus the large corr k2-to-k3 delta-U collapse occurs while pre-convolution RF
magnitude does not decrease in any common-cohort item. The collapse is
overwhelmingly associated with reduced depthwise-convolution norm transfer.

## k=0 current-hidden diagnostic

At k=0:

- corr median current delta-H: 20.715351
- ctrl median current delta-H: 20.752216
- paired corr > ctrl: 254/330
- median paired log corr/ctrl ratio: +0.071871

The apparent difference between marginal medians and the paired median log
ratio is not contradictory; these are different summary statistics.

This diagnostic does not support treating current-token pre-convolution
magnitude alone as the primary explanation of the much larger corr-vs-ctrl k2
delta-U separation.

## Validated scientific conclusion

The layer-23 post-convolution delta-U dynamics are not explained by creation or
loss of total pre-convolution difference energy over the four-token causal
receptive field.

Instead, pre-convolution difference energy is retained and transported toward
older receptive-field lags as the sequence advances.

Against that retained receptive-field difference, the fixed depthwise
convolution exhibits strongly coordinate- and role-dependent norm transfer:

- corr k1 -> k2: convolution transfer decreases in 302/330 items and is the
  dominant absolute log-changing factor in 269/330;
- corr versus ctrl k2: convolution transfer favors corr in 330/330 and is the
  dominant absolute log factor in 326/330;
- corr k2 -> k3: convolution transfer decreases in 330/330 and is the dominant
  absolute log-changing factor in 328/330.

SiLU is secondary in this decomposition and frequently counteracts, rather
than creates, the corr-vs-ctrl k2 separation.

Therefore the current evidence localizes the major transformation of the
layer-23 delta-U signal to receptive-field-conditioned depthwise convolution
transfer over an already-present multi-token pre-convolution difference.

This conclusion is observational and algebraic. It does not establish causal
importance for model behavior or downstream predictions.

## Next scientific boundary

The next narrow question is inside the authenticated depthwise convolution.

For each causal tap l:

`Q_l(t) = K_l elementwise-multiplied by delta_H_(t-l)`

with the exact bias-cancelled difference identity:

`delta_C_t = sum_l Q_l(t)`

for four taps.

A future audit should distinguish:

1. fixed tap-weight magnitude effects;
2. lag-specific incoming delta-H magnitude;
3. channelwise alignment/cancellation among the four tap contributions.

The current evidence should be frozen before any such new scientific
execution.

No tokenizer execution, training, intervention, learned probe, or PCA is
authorized by this result.