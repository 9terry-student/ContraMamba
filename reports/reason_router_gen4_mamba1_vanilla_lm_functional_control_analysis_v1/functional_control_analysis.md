# ContraMamba Gen4 Vanilla-Mamba LM Functional Control

## Status

`STATIC_PROSPECTIVE_FUNCTIONAL_CONTROL_ANALYSIS`

This analysis opens the four prospectively frozen vanilla-LM raw bundles only after all four were independently validated and frozen.

No training, model execution, intervention forward, row filtering, rescue, re-selection, or p-value is introduced here.

## Primary result

The frozen ContraMamba four-scale TASK_MATCHED sign vector is `+,+,-,-`.

The pretrained vanilla-Mamba LM TASK_MATCHED sign vector is `-,-,+,+`.

Therefore `FULL_TASK_MATCHED_SIGN_CONCORDANCE = False`.

| Scale | Vanilla LM mean | Contra forward-equivalent mean | Pearson | Spearman | Pair sign agreement |
|---|---:|---:|---:|---:|---:|
| 370M | -0.00197531911007 | +0.00101791270377 | -0.4593 | +0.0346 | 0.6033 |
| 790M | -0.0159643642888 | +0.00455342289971 | +0.1862 | -0.0379 | 0.4433 |
| 1.4B | +0.00783318280382 | -0.00239089501177 | -0.3664 | -0.1827 | 0.4567 |
| 2.8B | +0.0164484022524 | -0.000148349907005 | +0.1330 | +0.0966 | 0.5733 |

## Cell localization

| Scale | C0 mean | C2 mean |
|---|---:|---:|
| 370M | -0.00154000893813 | -0.00241062928202 |
| 790M | -0.0103930540441 | -0.0215356745336 |
| 1.4B | +0.00834069879684 | +0.0073256668108 |
| 2.8B | +0.0158561595372 | +0.0170406449676 |

Both cells share the aggregate TASK_MATCHED sign at every sampled scale; the four-scale pattern is therefore not localized to only C0_SHAM or C2_NAME.

## Geometry / functional diagnostics

| Scale | mean cosine gap | corr(component norm, cosine gap) | NORM_GAP |
|---|---:|---:|---:|
| 370M | +0.00299865162563 | -0.1670 | -0.00183669176027 |
| 790M | -0.0114677792527 | +0.1972 | +0.00121932483506 |
| 1.4B | +0.0288985160849 | -0.0848 | -0.000507463966294 |
| 2.8B | +0.0344709877053 | +0.0120 | +0.000293461691766 |

At 370M, the vanilla-LM mean cosine gap is positive while the mean endpoint is negative, with adverse component-norm/gap dependence. At 790M both the mean cosine gap and endpoint are negative. At 1.4B and 2.8B the TASK_MATCHED mean cosine gap and endpoint are positive.

## COMMON_P3_P5 sensitivity

| Scale | COMMON_P3_P5 mean | Sign |
|---|---:|:---:|
| 370M | -0.00197531911007 | - |
| 790M | -0.00443946889518 | - |
| 1.4B | -0.00990855313214 | - |
| 2.8B | +0.0164484022524 | + |

The common contrast gives `-,-,-,+`. The 1.4B result is therefore contrast-specific: TASK_MATCHED P5-P4 is positive while common P3-P5 is negative. This sensitivity does not alter the primary comparison, whose TASK_MATCHED contrasts were frozen prospectively.

## Scientific interpretation

The vanilla-LM control does not reproduce the ContraMamba `+,+,-,-` functional sign structure. The native Mamba-side geometry evidence remains unchanged, but the ContraMamba Delta-L reversal should not be described as the same functional reversal already intrinsic to the pretrained LM objective.

Instead, the combined evidence supports an objective-conditioned functional coupling account: scale reorganizes the pretrained native-state substrate, and the downstream task gradient reads that substrate differently from the pretrained next-token LM gradient.

The exact opposite aggregate signs at all four sampled scales must not be promoted to a simple inversion law. Pair-level Pearson/Spearman associations are not uniformly negative, so there is no rowwise one-to-one sign inversion.

## Paper-facing boundary

Appropriate claim: objective-dependent functional readout / repurposing of scale-reorganized native Mamba geometry.

Do not claim: universal vanilla-Mamba sign reversal, continuous scaling law, parameter-count threshold, semantic homology of same-numbered planes, or simple Contra-vs-LM inversion.

`RAW_FREEZE_HEAD = f78c56418902dc208adaef6c9a188378f9a7fa45`
`PLAN_FREEZE_COMMIT = 1fe9a198a15c9cea0e5451d918cd949bc21bf7e0`
