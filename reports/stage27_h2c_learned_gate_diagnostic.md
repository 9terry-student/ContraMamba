# Stage27-H2C Learned Gate Diagnostic

## Objective

Investigate whether the high seed variance in the `learned` v7-H1 entitlement decision
signal (H2A: macro_std=0.0466, acc_std=0.0454) reflects fundamental gate instability or
a recoverable training issue. Compare per-seed learned runs against the H2A product
baseline and the H2B selected configuration (product_power=0.90).

## Inputs

| Parameter | Value |
|---|---|
| h2a_dir | results\h2a |
| h2b_dir | results\h2b |
| output_md | reports\stage27_h2c_learned_gate_diagnostic.md |
| output_json | reports\stage27_h2c_learned_gate_diagnostic.json |

### H2A Reference Aggregates (3-seed, no-time)

| mode | macro_mean | macro_std | acc_mean | acc_std | SUP_r_mean | NE_r_mean | REF_r_mean | bad_SUP_total | location_role_SUP_total | predicate_SUP_total | missing_SUP_total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| product | 0.940351 | 0.025612 | 0.957870 | 0.018651 | 0.944444 | 0.953086 | 1.0 | 76 | 60 | 7 | 0 |
| min | 0.928270 | 0.020529 | 0.945833 | 0.017067 | 0.988889 | 0.929630 | 1.0 | 114 | 78 | 19 | 0 |
| learned | 0.900660 | 0.046553 | 0.922685 | 0.045354 | 0.966952 | 0.902469 | 1.0 | 136 | 104 | 8 | 0 |

### H2B Selected Result (product_power=0.90)

| power | macro_mean | macro_std | acc_mean | acc_std | SUP_r_mean | NE_r_mean | REF_r_mean | bad_SUP_total | location_role_SUP_total | predicate_SUP_total | missing_SUP_total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.90 | 0.951136 | 0.021341 | 0.968056 | 0.013679 | 0.907407 | 0.972840 | 1.0 | 44 | 33 | 3 | 0 |

## Per-Run Table

_No runs found in the provided directories._


## Learned Seed Ranking

Ranked by macro_f1 descending, then bad_SUP ascending, then location_role_SUP ascending.

_No learned runs found._


## Learned Best-vs-Worst Analysis

_No learned runs available._

### Diagnostic Probability Means (Best Seed)

_No learned runs available._


## Comparison to Product Gates

See Per-Run Table for individual comparisons.

- H2A product mean macro: **0.9404** (std 0.0256), bad_SUP_total=76, location_role_SUP_total=60
- H2B product_power=0.90 mean macro: **0.9511** (std 0.0213), bad_SUP_total=44, location_role_SUP_total=33

## Interpretation

- No learned runs found in the provided directories. Interpretation is based on H2A aggregate references only.
- H2A aggregate: learned macro_mean=0.9007 (std=0.0466) vs product macro_mean=0.9404 (std=0.0256). High std in learned suggests instability rather than uniform weakness.
- product_power=0.90 remains the current final configuration. Learned gate is treated as an unstable diagnostic branch pending per-seed data.

## Recommendation

Insufficient per-seed learned data. Keep product_power=0.90 as the current final configuration. Re-run this script after collecting individual seed summaries.

## Remaining Risks

- Results are based on the controlled no-time validation setting
  (`controlled_v5_v3_without_time_swap.jsonl`). Generalization beyond this setting is not
  established.
- time_swap was excluded because earlier Stage12 analysis identified it as
  corrupted/problematic. Results do not cover the time_swap evaluation distribution.
- Per-seed learned analysis carries high uncertainty with only 3 seeds. A single best seed
  cannot establish that learned is reliably competitive.
- Diagnostic probability means (entitlement_prob, frame_prob, etc.) may be absent from
  older summary JSONs that predate Stage26-F extended diagnostics. Missing fields are
  reported as N/A and do not invalidate macro/false-SUPPORT comparisons.
- T4-safe frozen-encoder setting used max_length=64. Claims should be framed as
  controlled-setting evidence unless confirmed by full-encoder runs.

## Conclusion

Stage27-H2C treats learned as an unstable but potentially informative diagnostic branch, while keeping product_power=0.90 as the current final v7-H1 entitlement decision configuration.
