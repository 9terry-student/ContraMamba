# Gen5 M7b — Static Factorial Interaction / Nonseparability Decomposition

## Status

`PASS_STATIC_M7B_INTERACTION_DECOMPOSITION`

Source evidence commit: `3bc4c72529f8c961111e8d0512e0df6bf04a86b0`  
Source run: `gen5-m7-t1-factor-swap-c95194f-r1`

This is a static decomposition of the already-frozen 27-state M7 logits.
No model forward, CUDA work, training, backward, optimizer, parameter update,
new seed, or threshold selection is performed.

## 1. Why M7b is required

M7 established the finite factor-effect ordering

`recipient-A > donor-A/B1-history >> donor-R`,

but did not establish universal recipient ownership. The cross-A hybrids were
recipient-closer on average while a nontrivial subset was donor-closer, and a
large fraction lay outside the recipient-to-donor line segment.

Therefore the remaining question is whether the observed mixture is mainly
additive competition between recipient and donor factors or a genuine
nonseparable recipient-A × donor-B1 compatibility effect.

## 2. Exact balanced finite-grid decomposition

For every row and task coordinate, the complete 3×3×3 grid is decomposed
orthogonally under the uniform finite-grid measure:

`Y = μ + A_rec + A_don + R + A_rec×A_don + A_rec×R + A_don×R + A_rec×A_don×R`.

This is an exact descriptive decomposition of the frozen grid. It is not a
p-value ANOVA and does not assume a population sampling model.

### All 840 rows — two-margin coordinates

| component | squared energy | fraction of centered grid energy |
|---|---:|---:|
| recipient A0 main | 19.0483883032 | 78.386696% |
| donor A/B1-history main | 1.90903266526 | 7.855928% |
| donor RNG main | 8.64932907839e-05 | 0.000356% |
| recipient×donor-A compatibility | 3.34258562926 | 13.755192% |
| recipient×R | 0.000129971540566 | 0.000535% |
| donor-A×R | 0.000125227152892 | 0.000515% |
| three-way | 0.000189052863322 | 0.000778% |

- main-effect fraction: `86.242980%`
- all interaction fraction: `13.757020%`
- recipient×donor-A interaction fraction: `13.755192%`
- orthogonal energy-closure relative error: `1.46198976126e-16`
- classification: `NONSEPARABLE_PRIMARILY_AREC_X_ADON`

### Frozen vulnerable-120 — two-margin coordinates

| component | squared energy | fraction |
|---|---:|---:|
| recipient A0 main | 1.38936095077 | 70.533921% |
| donor A/B1-history main | 0.284866466616 | 14.461864% |
| donor RNG main | 1.82363833848e-05 | 0.000926% |
| recipient×donor-A compatibility | 0.295484442143 | 15.000908% |
| recipient×R | 1.44380442409e-05 | 0.000733% |
| donor-A×R | 1.76300182735e-05 | 0.000895% |
| three-way | 1.48438422835e-05 | 0.000754% |

- main-effect fraction: `84.996710%`
- all interaction fraction: `15.003290%`
- recipient×donor-A interaction fraction: `15.000908%`
- classification: `NONSEPARABLE_PRIMARILY_AREC_X_ADON`

## 3. Hierarchical reconstruction test

This asks how much of the *actual frozen 27-state output* can be reconstructed
without the interaction terms.

### All 840 rows — centered logits

| model | explained centered-grid energy | RMSE | argmax agreement with full grid |
|---|---:|---:|---:|
| main effects only | 86.023564% | 0.00613448994631 | 22595/22680 (99.625220%) |
| main + recipient×donor-A | 99.997927% | 7.47100753818e-05 | 22679/22680 (99.995591%) |
| all main + all pairwise interactions | 99.999140% | 4.81250534119e-05 | 22678/22680 (99.991182%) |
| full factorial reconstruction | 100.000000% | 1.19141387503e-17 | 22680/22680 (100.000000%) |

### Frozen vulnerable-120 — centered logits

| model | explained centered-grid energy | RMSE | argmax agreement |
|---|---:|---:|---:|
| main effects only | 84.927550% | 0.00527356182948 | 3240/3240 (100.000000%) |
| main + recipient×donor-A | 99.997528% | 6.75415319203e-05 | 3240/3240 (100.000000%) |
| all main + all pairwise interactions | 99.999217% | 3.80014164899e-05 | 3240/3240 (100.000000%) |
| full factorial reconstruction | 100.000000% | 1.53682790647e-17 | 3240/3240 (100.000000%) |

If adding only `A_rec×A_don` closes most of the main-only residual, the mixed
M7 pattern is specifically a recipient/donor compatibility interaction rather
than generic higher-order noise. If main effects already reconstruct nearly
all energy, the apparent mixture is primarily additive.

## 4. Ordered cross-A asymmetry

Mean two-margin affinity is averaged over the three donor-R cells for each
ordered `(recipient A0, donor A/B1-history)` pair.

| recipient A | donor A | all-840 mean affinity | vulnerable-120 mean affinity | all-840 outside segment | vuln outside segment | closer on average |
|---:|---:|---:|---:|---:|---:|---|
| 6201 | 6202 | 0.792161004592 | 0.818608105067 | 1968/2520 | 213/360 | recipient |
| 6201 | 6203 | 0.183181261765 | 0.256814898276 | 2157/2520 | 255/360 | recipient |
| 6202 | 6201 | -0.38019496872 | -0.473720251075 | 23/2520 | 0/360 | donor |
| 6202 | 6203 | 0.474653775107 | 0.51907383074 | 0/2520 | 0/360 | recipient |
| 6203 | 6201 | -0.161007633421 | -0.230346264381 | 2492/2520 | 358/360 | donor |
| 6203 | 6202 | 0.894099439007 | 0.902814671562 | 2516/2520 | 360/360 | recipient |

This table tests whether the six ordered swaps share one ownership law or
whether direction-specific pair compatibility is present.

## 5. Recipient×donor-A interaction-strength matrix

Mean rowwise L2 norm of the exact `A_rec×A_don` two-margin interaction term:

| recipient \ donor-A | 6201 | 6202 | 6203 |
|---|---:|---:|---:|
| 6201 | 0.0173653202304 | 0.0111260391363 | 0.0062510061608 |
| 6202 | 0.0179948207605 | 0.0159391733551 | 0.0031552161941 |
| 6203 | 0.00279716885519 | 0.00483333339468 | 0.00429633091234 |

The diagonal entries are retained because this is the full balanced factorial
interaction, not only the 18 cross-A cells.

## 6. Decision boundary

M7b may establish one of:

- `NEAR_ADDITIVE_MAIN_EFFECTS`;
- `NONSEPARABLE_PRIMARILY_AREC_X_ADON`;
- `NONSEPARABLE_MULTI_INTERACTION`.

It does not establish gauge equivalence, latent-basis uniqueness, unseen-seed
generalization, or global training-dynamics causality.

Only after this decomposition is frozen should M8 decide how the residual
representation freedom should be expressed as a task-relevant quotient /
non-identifiability statement.
