# Stage27-H2D Learned Gate Stabilization Summary

## Objective

Determine whether auxiliary entitlement BCE loss (and related supervision signals) can
stabilize the collapsed learned v7-H1 entitlement gate, using the H2C worst seed (seed1)
as the test case. If a rescue configuration meets the pre-defined success thresholds,
expand to 3 seeds; otherwise, close the H2D branch and keep product_power=0.90 as final.

## Background

Stage27-H2C showed that the learned v7-H1 entitlement decision signal has high seed
variance (H2A macro_std=0.0466). Seed3 was competitive with product_power=0.90, but seed1
collapsed with high false SUPPORT:

| metric | learned seed1 baseline |
|---|---:|
| macro_f1 | 0.858586 |
| acc | 0.875000 |
| SUP_r | 1.000000 |
| NE_r | 0.833333 |
| REF_r | 1.0 |
| bad_SUP | 90 |
| location_role_SUP | 71 |
| missing_SUP | 0 |

The primary failure mode is excessive false SUPPORT on location/role swap interventions
(71 of 90 false SUPPORT predictions), indicating the learned gate does not suppress
entitlement for spatially or role-mismatched claims.

## Method

- Architecture: `v7_hierarchical` with `--v7-use-v6b-style-final-decision` (H1 bridge)
- Decision signal: `--v7-h1-entitlement-decision-signal learned`
- Base seed: seed1 (H2C worst seed, collapsed baseline)
- Evaluation: controlled no-time validation (`controlled_v5_v3_without_time_swap.jsonl`)
- Encoder: frozen (T4-safe), max_length=64
- Configurations screened:
  - Entitlement BCE loss at weights: 0.1, 0.3, 1.0, 1.5, 2.0, 3.0
  - BCE w1.0 with pos_weight=2.0
  - Class-balanced CE (CBCE) at weight 0.3
  - BCE w0.3 + CBCE w0.3 combined

**Pre-defined success thresholds (all must be met to expand to 3 seeds):**
- macro_f1 >= 0.92
- bad_SUP <= 40
- location_role_SUP <= 30
- missing_SUP = 0

## Results

| config | acc | macro | SUP_r | NE_r | REF_r | entitlement_bce_weight | entitled_cbce_weight | loc_SUP | role_SUP | predicate_SUP | entity_SUP | event_SUP | title_SUP | deletion_SUP | truncation_SUP | irrelevant_SUP | location_role_SUP | missing_SUP | bad_SUP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bce_w1p0 | 0.920833 | 0.899954 | 0.977778 | 0.898148 | 1.0 | 1.0 | 0.0 | 24 | 16 | 2 | 5 | 6 | 2 | 0 | 0 | 0 | 40 | 0 | 55 |
| baseline | 0.875000 | 0.858586 | 1.000000 | 0.833333 | 1.0 | 0.0 | 0.0 | 41 | 30 | 3 | 6 | 6 | 4 | 0 | 0 | 0 | 71 | 0 | 90 |
| bce_w0p1 | 0.852778 | 0.840180 | 1.000000 | 0.803704 | 1.0 | 0.1 | 0.0 | 47 | 39 | 4 | 5 | 7 | 4 | 0 | 0 | 0 | 86 | 0 | 106 |
| bce_w2p0 | 0.851389 | 0.829073 | 1.000000 | 0.801852 | 1.0 | 2.0 | 0.0 | 44 | 32 | 4 | 6 | 7 | 4 | 0 | 0 | 0 | 76 | 0 | 97 |
| bce_w0p3 | 0.843056 | 0.828081 | 1.000000 | 0.790741 | 1.0 | 0.3 | 0.0 | 48 | 41 | 4 | 5 | 7 | 4 | 0 | 0 | 0 | 89 | 0 | 109 |
| bce_w1p0_pos2p0 | 0.812500 | 0.807078 | 1.000000 | 0.750000 | 1.0 | 1.0 | 0.0 | 56 | 52 | 8 | 6 | 7 | 4 | 0 | 0 | 0 | 108 | 0 | 133 |
| bce_w3p0 | 0.800000 | 0.799302 | 1.000000 | 0.733333 | 1.0 | 3.0 | 0.0 | 57 | 53 | 8 | 8 | 11 | 6 | 0 | 0 | 0 | 110 | 0 | 143 |
| cbce_w0p3 | 0.787500 | 0.785568 | 1.000000 | 0.716667 | 1.0 | 0.0 | 0.3 | 57 | 54 | 17 | 6 | 9 | 5 | 0 | 0 | 0 | 111 | 0 | 148 |
| bce0p3_cbce0p3 | 0.751389 | 0.767575 | 1.000000 | 0.668519 | 1.0 | 0.3 | 0.3 | 56 | 55 | 19 | 18 | 25 | 6 | 0 | 0 | 0 | 111 | 0 | 179 |
| bce_w1p5 | 0.676389 | 0.720249 | 1.000000 | 0.568519 | 1.0 | 1.5 | 0.0 | 58 | 56 | 18 | 27 | 55 | 19 | 0 | 0 | 0 | 114 | 0 | 233 |

## Best Configuration

**`bce_w1p0`** (entitlement_bce_weight=1.0, entitled_cbce_weight=0.0) is the best H2D
seed1 rescue configuration. It partially improves the collapsed learned seed1 baseline:

| metric | learned seed1 baseline | bce_w1p0 | delta |
|---|---:|---:|---:|
| macro_f1 | 0.858586 | 0.899954 | +0.041368 |
| acc | 0.875000 | 0.920833 | +0.045833 |
| SUP_r | 1.000000 | 0.977778 | -0.022222 |
| NE_r | 0.833333 | 0.898148 | +0.064815 |
| bad_SUP | 90 | 55 | -35 |
| location_role_SUP | 71 | 40 | -31 |
| missing_SUP | 0 | 0 | 0 |

Entitlement BCE at weight=1.0 pushes the learned gate toward correct entitlement signals,
reducing location/role false SUPPORT from 71 to 40 and NE recall from 0.833 to 0.898.
However, bad_SUP remains at 55 (threshold: <=40) and location_role_SUP is exactly at 40
(threshold: <=30).

## Failure Analysis

**bce_w1p0 does not meet the pre-defined success thresholds:**

| threshold | required | bce_w1p0 result | pass? |
|---|---|---|---|
| macro_f1 >= 0.92 | 0.92 | 0.899954 | NO |
| bad_SUP <= 40 | 40 | 55 | NO |
| location_role_SUP <= 30 | 30 | 40 | NO |
| missing_SUP = 0 | 0 | 0 | YES |

**Why higher BCE weights fail:**

- `bce_w1p5` collapses severely: bad_SUP=233, entity_SUP=27, event_SUP=55, title_SUP=19.
  A weight gap from 1.0 to 1.5 produces a non-linear degradation, indicating the BCE
  loss at high weight disrupts the polarity signal.
- `bce_w2p0` and `bce_w3p0` are better than `bce_w1p5` but still worse than `bce_w1p0`
  on every headline metric.

**Why lower BCE weights fail:**

- `bce_w0p1` and `bce_w0p3` are worse than the no-BCE baseline on both macro and
  false SUPPORT counts. BCE signal at low weight is insufficient to overcome the
  seed-level collapse but still disturbs other components.

**Why pos_weight=2.0 fails:**

- `bce_w1p0_pos2p0` worsens macro (0.807 vs 0.900) and nearly doubles location_role_SUP
  (108 vs 40). Over-weighting the positive entitlement class amplifies false SUPPORT
  rather than correcting the gate.

**Why CBCE and BCE+CBCE fail:**

- `cbce_w0p3` has macro=0.786 and bad_SUP=148, far below baseline.
- `bce0p3_cbce0p3` has macro=0.768 and bad_SUP=179. The combination is additive in
  its harm. CBCE pushes the polarity head toward SUPPORT/REFUTE balance independently
  of the entitlement gate, producing more false SUPPORT for hard negatives.

**Summary:** there is no BCE weight regime that simultaneously meets all four success
thresholds for learned seed1. The optimal point (w=1.0) is bounded away from the
target by a margin of 0.020 on macro and 10 counts on bad_SUP.

## Decision

- **Do not expand H2D to 3 seeds.** `bce_w1p0` fails 3 of 4 success thresholds on
  seed1 alone. A 3-seed expansion would consume compute without a credible path to
  matching product_power=0.90.
- **Do not select learned+BCE as the final v7-H1 configuration.** The best rescue
  result (macro=0.900, bad_SUP=55) is below the product_power=0.90 standard
  (macro=0.951, bad_SUP=44).
- **Keep product_power=0.90 as the current final v7-H1 configuration.**

Stage27-H2D finds that entitlement BCE partially rescues learned seed1 but does not stabilize the learned gate enough to replace product_power=0.90.

## Next Step

Proceed to **Stage27-H2E: Hybrid/Residual Gate**. The product gate remains the stable
base entitlement signal; the learned gate contributes only as a bounded residual
correction. This approach avoids the seed instability of learned-only while potentially
capturing the discriminative signal visible in learned seed3.

## Remaining Risks

- Results are based on the controlled no-time validation setting
  (`controlled_v5_v3_without_time_swap.jsonl`). Generalization beyond this setting is
  not established.
- time_swap was excluded because earlier Stage12 analysis identified it as
  corrupted/problematic.
- H2D screening used only seed1 (the collapsed seed). The behavior of `bce_w1p0` on
  seed3 (the good seed) is unknown and untested.
- T4-safe frozen-encoder setting used max_length=64. Results should be framed as
  controlled-setting evidence.
- The success thresholds (macro>=0.92, bad_SUP<=40, location_role_SUP<=30) were set
  before H2D ran. Changing these thresholds post-hoc to accommodate H2D results would
  constitute validation-set overfitting and is not warranted.
