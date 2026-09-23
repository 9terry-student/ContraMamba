# ContraMamba Gen4 Mamba-1 Native Geometry / Functional Coupling Static Bridge

## Status

`STATIC_NATIVE_MAMBA_TO_FUNCTIONAL_COUPLING_BRIDGE`

This report combines already frozen artifacts only.

No model execution, backward pass, training, new p-value, model-size regression
fit, threshold estimation, rescue, re-selection, or row filtering is introduced.

The bridge covers the four sampled Mamba-1 scales for which homologous
response-blind geometry-preparation artifacts are frozen:

`370M, 790M, 1.4B, 2.8B`.

All cross-scale associations are descriptive. N=4 is insufficient for a
population-level scaling-law claim.

## 1. Evidence classes

Three evidence classes must remain distinct.

### A. Kernel-only pretrained-Mamba evidence

`mu_k2` and the strong/weak channel partition are computed directly from the
pretrained Mamba convolution kernel:

`k = conv_weight[:, 0, lag0]`

`mu_k2 = mean(k^2)`

`strong = {j : k_j^2 > mu_k2}`.

These quantities do not use the ContraMamba task gradient, downstream response,
or backward pass.

### B. Response-blind Mamba-side activation geometry

The frozen cross-backbone geometry-preparation runs capture internal Mamba
activations through forward hooks.

Their frozen contracts record:

- `response_observed = false`;
- `backward_executed = false`;
- `training_executed = false`;
- `plane_selection_performed = false`;
- `xg1_accessed = false`.

Thus these geometry artifacts are not selected or constructed using the XG1
functional readout response.

They are nevertheless measured on the ContraMamba research input families and
therefore are not claimed to characterize all possible natural-language inputs.

### C. ContraMamba functional coupling

For the frozen Delta-L readout:

`Delta_L_i = g_i * n_i * h_i`

where:

- `g_i = ||ContraMamba task-margin gradient||`;
- `n_i = matched Mamba native-component norm`;
- `h_i = cosine(selected, gradient) - cosine(control, gradient)`.

The functional quantities considered here are:

`NORM_GAP = mu_g * Cov(n,h)`

and:

`corr(n,h)`.

These explicitly depend on the ContraMamba task-functional gradient.

## 2. Frozen native Mamba-side scale values

| scale | mu_k2 | strong-channel fraction | lambda1 | XG2 top-1 share | XG4 top-1 share | XG4 top-5 entropy |
|---|---:|---:|---:|---:|---:|---:|
| 370M | 0.024623525099917637 | 0.3173828125 | 0.9671797709152048 | 0.3280979378716348 | 0.3733923621494175 | 1.5166556267802997 |
| 790M | 0.02384271108497352 | 0.3173828125 | 0.8929611457472746 | 0.2981884192530853 | 0.3436979950281655 | 1.5396433286993940 |
| 1.4B | 0.014838786182259655 | 0.202392578125 | 0.8890412240357709 | 0.2919641183309584 | 0.3198001522649058 | 1.5525681960248519 |
| 2.8B | 0.00985710670421181 | 0.1958984375 | 0.8932258269422545 | 0.2851018283239676 | 0.2736762349028396 | 1.5768045710558116 |

The kernel-only `mu_k2` decreases strictly across these four sampled scales.

The strong-channel fraction also decreases from approximately 0.317 at the
smaller two sampled scales to approximately 0.20 at 1.4B and 2.8B.

The response-blind leading spectral concentration also generally decreases,
while XG4 top-5 spectral entropy increases.

These observations establish native Mamba-side scale reorganization independent
of the XG1 ContraMamba functional response.

## 3. Frozen functional coupling values

| scale | NORM_GAP | corr(n,h) | mean Delta-L |
|---|---:|---:|---:|
| 370M | +0.000042452455047035967 | +0.029825055 | +0.0005089563518854173 |
| 790M | -0.00006870662010633534 | -0.169983083 | +0.0022767114498564943 |
| 1.4B | -0.00017656357856181494 | -0.405771234 | -0.0011954475058862238 |
| 2.8B | -0.00017154392604376662 | -0.230747303 | -0.00007417495350234895 |

The task-functional state-magnitude/alignment coupling therefore shifts from
approximately neutral or favorable at 370M toward adverse coupling at the
larger sampled scales.

## 4. Cross-evidence descriptive bridge

No inferential p-values are attached.

### Native quantity versus NORM_GAP

| native Mamba-side quantity | Pearson | Spearman |
|---|---:|---:|
| mu_k2 | +0.8743814771 | +0.8 |
| strong-channel fraction | +0.8969250172 | +0.6 |
| leading-plane lambda1 | +0.8876699825 | +0.8 |
| XG2 top-1 spectral share | +0.9537643063 | +0.8 |
| XG4 top-1 spectral share | +0.8783371506 | +0.8 |
| XG4 top-5 entropy | -0.9105776357 | -0.8 |

### Native quantity versus corr(n,h)

| native Mamba-side quantity | Pearson | Spearman |
|---|---:|---:|
| mu_k2 | +0.6710726583 | +0.8 |
| strong-channel fraction | +0.7813537794 | +0.6 |
| leading-plane lambda1 | +0.8568054343 | +0.8 |
| XG2 top-1 spectral share | +0.8319064386 | +0.8 |
| XG4 top-1 spectral share | +0.6282926732 | +0.8 |
| XG4 top-5 entropy | -0.6826061603 | -0.8 |

## 5. Interpretation

The frozen evidence does not support the claim that the observed functional
Delta-L reversal is merely an arbitrary artifact created entirely inside the
ContraMamba downstream head.

Independent Mamba-side quantities already reorganize across scale:

1. pretrained convolution-kernel statistics change;
2. the strong-channel partition changes;
3. response-blind activation geometry changes;
4. leading spectral concentration generally decreases.

Separately, the ContraMamba functional probe shows increasingly adverse
native-component-magnitude / task-alignment coupling.

Across the four common sampled scales, several independently measured native
Mamba-side quantities co-vary strongly with the functional NORM_GAP endpoint.

This motivates a two-stage mechanistic hypothesis:

`Mamba scale`
`-> native kernel/state-geometry reorganization`
`-> changed coupling to a fixed task-functional readout`.

The arrows denote a research hypothesis, not demonstrated causality.

## 6. What is and is not currently a vanilla-Mamba claim

Supported:

- scale-dependent changes exist in pretrained Mamba kernel statistics;
- scale-dependent changes exist in response-blind Mamba-side activation
  geometry on the frozen research inputs;
- these native changes descriptively track the independently measured
  ContraMamba state-magnitude/alignment coupling.

Not yet supported:

- that vanilla Mamba under an architecture-independent functional objective
  exhibits the same Delta-L sign reversal;
- that the observed functional reversal is universal across heads, tasks, or
  datasets;
- that model size causally produces NORM_GAP through any one native statistic;
- a population-level Mamba scaling law.

Therefore the present bridge supports a Mamba-side mechanistic origin candidate
but does not replace the need for a task-head-independent functional control if
a stronger vanilla-Mamba functional claim is desired.

## 7. Publication implication

The existing evidence is sufficient for a paper framed around:

`native Mamba scaling reorganization revealed through a controlled functional
probe`.

A stronger paper claim of:

`vanilla Mamba itself undergoes the same functional sign reversal`

requires an additional control in which the functional gradient source is
independent of the trained ContraMamba downstream/router.

## Provenance

Five-scale Delta-L mechanistic autopsy:

`ceecb2bc77070aed0aab4004e96fe0fad5f52208`

Model-size / norm-alignment coupling diagnostic:

`64fcc403e38de5273699be99f58a687317f10f1f`

`NATIVE_MAMBA_FUNCTIONAL_COUPLING_STATIC_BRIDGE = FROZEN_CANDIDATE`
