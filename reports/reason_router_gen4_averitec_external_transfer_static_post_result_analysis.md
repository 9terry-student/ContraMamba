# AVeriTeC External Transfer — Static Post-Result Analysis

## Status

`PASS_AVERITEC_EXTERNAL_TRANSFER_STATIC_POST_RESULT_ANALYSIS`

This analysis is descriptive only. It adds zero model forwards and zero new p-values.

## Frozen primary context

| Scale | mean D_EXT | Holm p | Cohen dz | supported |
|---|---:|---:|---:|:---:|
| mamba130m | 0.00023104869 | 0.010430705 | 0.11964887 | True |
| mamba370m | 4.5550022e-05 | 0.0113815 | 0.10631035 | True |

## Descriptive decomposition

| Scale | mean A_sel | mean B_ctrl | mean D_EXT | native→control flips |
|---|---:|---:|---:|---:|
| mamba130m | 0.00029725037 | 6.6201684e-05 | 0.00023104869 | 0 |
| mamba370m | 7.4439619e-06 | -3.810606e-05 | 4.5550022e-05 | 0 |

## Source-label heterogeneity

| Scale | Source label | n | mean D_EXT | fraction positive |
|---|---|---:|---:|---:|
| mamba130m | Not Enough Evidence | 35 | -1.3664554e-05 | 0.2 |
| mamba130m | Refuted | 305 | 0.00048267237 | 0.70163934 |
| mamba130m | Supported | 122 | -0.0003278059 | 0.68032787 |
| mamba370m | Not Enough Evidence | 35 | 0.00012954952 | 0.71428571 |
| mamba370m | Refuted | 305 | 5.5046606e-05 | 0.64262295 |
| mamba370m | Supported | 122 | -2.2896524e-06 | 0.48360656 |

## Descriptive correlations

| Scale | native margin vs D | q-delta vs D | A_sel vs D | B_ctrl vs D |
|---|---:|---:|---:|---:|
| mamba130m | 0.12352337 | 0.06467522 | 0.77930318 | -0.59358986 |
| mamba370m | 0.11913352 | 0.10556619 | 0.90822096 | -0.59806561 |

## Cross-scale item alignment

- D_EXT Pearson: `0.069846724`
- sign agreement fraction: `0.57142857`
- both-positive fraction: `0.41774892`
- opposite-sign fraction: `0.42857143`

## Interpretation boundary

- The preregistered primary endpoint is supported at both 130M and 370M.
- This static analysis does not add or alter the primary family.
- Source-label breakdowns are descriptive and are not promoted to new inferential claims.
- The result supports external transfer in correct-class margin, not benchmark accuracy improvement.
- It does not establish uniform transfer across labels or a unique causal mediator.
