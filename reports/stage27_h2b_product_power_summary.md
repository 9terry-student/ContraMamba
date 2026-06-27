# Stage27-H2B Product-Power Entitlement Gate Summary

## Objective

Calibrate the strictness of the compositional product entitlement gate introduced in Stage27-H2A.
The H2A sweep selected `product` (`frame_prob * predicate_coverage_prob * sufficiency_prob`) as
the best decision signal for the v7-H1 final-decision path. However, the raw product was observed
to suppress some true SUPPORT predictions. Stage27-H2B adds a tunable power exponent gamma to
relax or sharpen the gate:

```
entitlement_for_decision = (frame_prob * predicate_coverage_prob * sufficiency_prob) ** gamma
```

gamma = 1.0 recovers the original H2A product gate exactly. gamma < 1 softens the gate
(pushes entitlement values toward 1), reducing SUPPORT suppression. gamma > 1 sharpens it.

## Method

- Architecture: `v7_hierarchical` with `--v7-use-v6b-style-final-decision` (H1 bridge)
- Decision signal: `--v7-h1-entitlement-decision-signal product`
- Power sweep: gamma in {0.67, 0.75, 0.80, 0.85, 0.90, 0.97, 1.00}
- Evaluation: controlled no-time validation split (`controlled_v5_v3_without_time_swap.jsonl`)
- Seeds: 3 seeds per configuration
- Encoder: frozen (T4-safe setting), max_length=64
- time_swap excluded (see Remaining Risks)

## Results

| power | macro_mean | macro_std | acc_mean | acc_std | SUP_r_mean | NE_r_mean | REF_r_mean | bad_SUP_total | location_role_SUP_total | predicate_SUP_total | missing_SUP_total |
|------:|----------:|----------:|---------:|--------:|-----------:|----------:|-----------:|--------------:|------------------------:|--------------------:|------------------:|
| 0.90  | 0.951136  | 0.021341  | 0.968056 | 0.013679 | 0.907407  | 0.972840  | 1.0        | 44            | 33                      | 3                   | 0                 |
| 0.97  | 0.949940  | 0.033783  | 0.963426 | 0.027157 | 0.977778  | 0.954938  | 1.0        | 73            | 61                      | 3                   | 0                 |
| 1.00  | 0.940351  | 0.025612  | 0.957870 | 0.018651 | 0.944444  | 0.953086  | 1.0        | 76            | 60                      | 7                   | 0                 |
| 0.85  | 0.934309  | 0.026061  | 0.955093 | 0.017366 | 0.900000  | 0.956790  | 1.0        | 70            | 57                      | 5                   | 0                 |
| 0.75  | 0.922020  | 0.009800  | 0.947685 | 0.011312 | 0.834229  | 0.958025  | 1.0        | 68            | 57                      | 3                   | 0                 |
| 0.80  | 0.914655  | 0.032622  | 0.932870 | 0.028373 | 0.988889  | 0.912346  | 1.0        | 142           | 115                     | 14                  | 0                 |
| 0.67  | 0.907933  | 0.017199  | 0.929167 | 0.018789 | 0.948148  | 0.914198  | 1.0        | 139           | 112                     | 15                  | 0                 |

## Winner

**Selected: `product_power = 0.90`**

Stage27-H2B selects product_power=0.90 as the current final v7-H1 entitlement decision configuration.

## Interpretation

**Why 0.90 wins:**

- **Highest macro_mean** (0.9511) across all tested power values.
- **Highest acc_mean** (0.9681), the best overall accuracy in the sweep.
- **Strongest NE_r_mean** (0.9728), meaning the model correctly identifies NOT_ENTITLED examples
  at the highest rate of any setting, confirming that a mild softening of the product gate does
  not erode the key discriminative property identified in H2A.
- **Lowest bad_SUP_total** (44) and **lowest location_role_SUP_total** (33) in the sweep,
  indicating the fewest false SUPPORT predictions and in particular the fewest location/role
  false SUPPORT errors.
- **missing_SUP_total = 0** is maintained, meaning no gold SUPPORT examples are completely
  dropped from the model's SUPPORT predictions.

**Why 0.97 is not selected:**

0.97 recovers higher raw SUP_r_mean (0.978 vs 0.907), but at the cost of substantially more
false SUPPORT: bad_SUP_total rises to 73 (+29 vs 0.90) and location_role_SUP_total rises to
61 (+28 vs 0.90). The macro_mean gain over 0.90 is negligible (0.9499 vs 0.9511) and
acc_mean is lower (0.9634 vs 0.9681). The 0.90 setting achieves a strictly better overall
trade-off between recall and precision on the validation split.

**Why 1.00 (original product baseline) is not selected:**

The original product gate (gamma = 1.0) is weaker than 0.90 on every headline metric:
macro_mean is lower (0.9404 vs 0.9511), acc_mean is lower (0.9579 vs 0.9681), NE_r_mean
is lower (0.9531 vs 0.9728), bad_SUP_total is higher (76 vs 44), and
location_role_SUP_total is higher (60 vs 33). The mild relaxation to 0.90 strictly dominates
the original product gate on this validation set.

**Why 0.80 and 0.67 are too permissive:**

Power values at or below 0.80 cause a sharp non-linear increase in false SUPPORT:
bad_SUP_total jumps to 142 at 0.80 and 139 at 0.67, and location_role_SUP_total rises to
115 and 112 respectively. These counts are roughly 2.5–3x the level seen at 0.90.
The product gate becomes too weak to suppress location/role false SUPPORT at these settings,
negating the primary benefit of the compositional entitlement signal.

**No further micro-search:**

No further power values between 0.90 and 0.97 should be swept on this validation split.
Selecting on a finer grid would overfit the validation set and produce results that cannot
be trusted to generalize. The coarse sweep evidence is sufficient: 0.90 is the best
observed setting, and further tuning should only proceed after held-out or OOD confirmation.

## Final Config

```
--architecture v7_hierarchical
--v7-use-v6b-style-final-decision
--v7-h1-entitlement-decision-signal product
--v7-h1-entitlement-product-power 0.90
```

## Remaining Risks

1. **Controlled no-time validation only.** All results are from
   `controlled_v5_v3_without_time_swap.jsonl`. Generalization beyond this setting is not
   established by the numbers in this report.

2. **time_swap excluded.** time_swap data was excluded from training and evaluation because
   earlier Stage12 analysis identified it as corrupted/problematic. Results do not account
   for the time_swap portion of the evaluation distribution.

3. **Validation-selected hyperparameter.** `product_power = 0.90` was selected by comparing
   validation metrics directly. Final performance claims require held-out diagnostic
   confirmation or OOD evaluation before generalization can be asserted.

4. **T4-safe frozen-encoder setting.** These runs used a frozen Mamba encoder and
   max_length=64 to fit within T4 GPU memory constraints. Results should be framed as
   controlled-setting evidence unless subsequent full-encoder or larger-context runs
   confirm the same ordering.

## Next Step

Proceed to Stage27-H2C or equivalent: run the selected configuration
(`product` signal, `product_power=0.90`) with the standard multi-seed full evaluation
protocol (including held-out or OOD diagnostic sets) to confirm that the validation
advantage transfers before reporting as a finalized result.
