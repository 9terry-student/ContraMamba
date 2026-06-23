# Stage20-C Predicate Bias Calibration Notes

## Purpose

Stage20-C replaces the manually selected Stage20-B penalty (1.25) with a
data-driven calibration: the smallest alpha satisfying the guarded-accuracy
objective on a held-out calibration split is chosen automatically per seed.

## Relationship to Stage20-A/B

| Stage | Mechanism | Alpha selection |
| --- | --- | --- |
| Stage20-A | hard NOT_ENTITLED override | none (always override) |
| Stage20-B | soft pseudo-logit penalty sweep | manual (Kaggle multi-seed: 1.25) |
| Stage20-C | soft pseudo-logit penalty, calibration split | automatic per seed |

## Detector

Reuses the Stage20-A lexical predicate detector (`lexical_predicate`).
Detector recall on the Kaggle multi-seed run: 288/300 predicate_mismatch
examples flagged; 0 non-predicate false positives.

## Calibration procedure

1. Split example IDs by `stage15_source_id` (or `id` if unavailable) with
   `--calibration-frac 0.5` and `--split-seed 1`.
2. For each alpha in the grid (default: 0, 0.25, …, 6.0), apply pseudo-logit
   penalty to flagged rows on the calibration split.
3. Select the **smallest alpha** where:
   - calibration predicate_mismatch false-entitled count falls by ≥ 90%; and
   - `changed_count = 0` for all six control groups on the calibration split.
4. If no alpha satisfies both conditions, choose the alpha with best
   calibration accuracy and mark `valid_guarded_alpha_found = false`.
5. Apply the selected alpha to all rows (calibration + heldout) and write
   `stage20_predicate_bias_calibrated_seed{seed}.json`.

## Penalty rule

```text
SUPPORT      -= alpha    (index 0)
NOT_ENTITLED += alpha    (index 1)
REFUTE       -= alpha    (index 2)
```

Probabilities are converted to pseudo-logits with `log(max(p, 1e-8))` when
true logits are unavailable, then re-softmaxed.

## Stage20-B reference result (Kaggle multi-seed, seeds 1/2/3)

- Smallest valid penalty: **1.25**
- predicate_mismatch false-entitled mean: 60.0/100 → 4.333/100 at alpha=1.25
- Plateau at 3.667/100 for alpha ≥ 1.5
- Non-predicate changed_count: 0 at all alphas

The local seed1-only result (0.0) is degenerate and must not be used as the
scientific conclusion; seed1 had 0 base false-entitled after Stage19 temporal
calibration, making the threshold trivially satisfied.

## Caveat

Stage20-C is prediction-level calibration, not an end-to-end trained model.
A calibration-split alpha that satisfies the guarded objective does not
guarantee the same behaviour on an independent test distribution.
