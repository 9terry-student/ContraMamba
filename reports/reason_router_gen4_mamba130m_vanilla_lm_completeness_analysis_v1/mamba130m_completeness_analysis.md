# Mamba-130M Vanilla-LM Functional-Control Completeness Extension

## Status

`POST_PRIMARY_DESCRIPTIVE_COMPLETENESS`

The original four-scale vanilla-LM control remains immutable. This analysis adds the subsequently frozen 130M completeness extension only.

No new p-value, row filtering, rescue, re-selection, monotonicity test, threshold estimate, or scaling-law fit is introduced.

## 130M extension result

| Quantity | Value |
|---|---:|
| Vanilla-LM TASK_MATCHED mean | -0.0564168463864 |
| Vanilla-LM median | -0.0770918814889 |
| Vanilla-LM positive fraction | 0.373333 |
| C0_SHAM mean | -0.0374067622064 |
| C2_NAME mean | -0.0754269305665 |
| ContraMamba forward-equivalent mean | +0.00224073207306 |
| Pair Pearson | +0.320798 |
| Pair Spearman | +0.204381 |
| Pair sign agreement | 0.556667 |

The 130M vanilla-LM mean is negative while the frozen ContraMamba forward-equivalent mean is positive, so `M130_SIGN_CONCORDANCE=False`.

Both C0_SHAM and C2_NAME have negative aggregate means, so the 130M result is not localized to only one of the two frozen cells.

## Geometry / functional diagnostics

| Quantity | Value |
|---|---:|
| Mean cosine gap | -0.0278183711641 |
| corr(component norm, cosine gap) | +0.448660 |
| NORM_GAP | +0.00821313946339 |
| Mean strong gradient norm | +0.210060312114 |
| Max identity reconstruction error | 5.551e-17 |

The mean cosine gap is negative. Component-norm/cosine-gap dependence is positive and NORM_GAP is positive, so the dependence correction partially opposes rather than creates the negative aggregate endpoint.

## Five-scale descriptive completeness

Scale order: `130M, 370M, 790M, 1.4B, 2.8B`.

Vanilla-LM TASK_MATCHED sign vector: `-,-,-,+,+`.

ContraMamba forward-equivalent sign vector: `+,+,+,-,-`.

`aggregate_sign_opposite_at_all_five_sampled_scales = True`.

This five-scale vector is descriptive completeness, not a retroactive replacement of the prospectively frozen four-scale control.

## Interpretation boundary

The additional 130M scale strengthens the descriptive observation that the pretrained LM objective does not reproduce the ContraMamba aggregate sign structure across the five sampled model sizes.

It still does not support a simple rowwise inversion account: the 130M pair-level Pearson and Spearman associations are positive, and sign agreement is above one half.

Do not claim a universal scaling law, continuous size threshold, semantic homology of plane labels, or a deterministic objective-to-objective inversion.

`RAW_FREEZE_HEAD = 9bbea9cb03327d4dca86cdb148eec44a4dd27b34`
`EXTENSION_PLAN_FREEZE = e567338f1dcd99d61ed7465ec39d441566f71fa3`
`PRIOR_FOUR_SCALE_ANALYSIS_FREEZE = 88a6d6c469d44071a070b494485efe56db4faa58`
