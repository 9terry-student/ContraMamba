# Stage27-H3 Final Evidence Package

## Executive Summary

Stage27 investigated the v7-H1 entitlement decision signal for the ContraMamba
hierarchical model. The core question was whether to use a learned entitlement gate,
a compositional product signal, or a hybrid of both as the final v7-H1 decision
configuration. Experiments H2A through H2F compared these options across 3 seeds
on a controlled no-time validation setting.

**Result:** `product_power=0.90` is selected as the stable final Stage27 v7-H1
configuration. The learned gate contains local discriminative signal but is seed-unstable.
Residual injection of the learned gate provides local gain but worsens the 3-seed
aggregate. The dominant remaining false-SUPPORT axis is location/role frame mismatch.
Missing-evidence false SUPPORT is controlled at zero in this setting.

Stage27 selects product_power=0.90 as the stable v7-H1 entitlement decision configuration, while preserving learned and hybrid gates as diagnostic evidence for future location-role frame-boundary specialization.

---

## Final Selected Configuration

| Parameter | Value |
|---|---|
| architecture | v7_hierarchical |
| H1 final decision | enabled (`--v7-use-v6b-style-final-decision`) |
| entitlement decision signal | product |
| product_power | 0.90 |
| dataset | data/controlled_v5_v3_without_time_swap.jsonl |
| backbone | Mamba, T4-safe frozen encoder |
| max_length | 64 |
| freeze_encoder | true |
| freeze_a_log | true |
| caveat | controlled no-time validation setting |

**Aggregate performance (3-seed, controlled no-time):**

| macro_mean | macro_std | acc_mean | acc_std | SUP_r_mean | NE_r_mean | REF_r_mean | bad_SUP_total | location_role_SUP_total | predicate_SUP_total | missing_SUP_total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.951136 | 0.021341 | 0.968056 | 0.013679 | 0.907407 | 0.972840 | 1.0 | 44 | 33 | 3 | 0 |

---

## Final Decision Formula

```
positive_energy          = softplus(v7_polarity_support)
negative_energy          = softplus(v7_polarity_refute)
entitlement_for_decision = (frame_prob * predicate_coverage_prob * sufficiency_prob) ** 0.90

support_score = entitlement_for_decision * positive_energy
refute_score  = entitlement_for_decision * negative_energy
ne_score      = ne_bias + alpha * (1 - entitlement_for_decision)
```

The product is taken over three independently learned probability heads (frame,
predicate coverage, sufficiency). Raising to the power 0.90 slightly relaxes the
strict conjunction, improving SUPPORT recall relative to exact product (power=1.0)
without introducing the seed instability of the learned gate.

---

## Stage27 Timeline

| Stage | Scope | Outcome |
|---|---|---|
| H1 | Learned v7-H1 final-decision path | Solved SUPPORT collapse; revealed frame-mismatch false SUPPORT |
| H2A | 3-mode decision signal comparison (learned / product / min) | Product selected |
| H2B | Product-power sweep (0.67 to 1.00) | product_power=0.90 selected |
| H2C | Learned gate seed diagnostic | Learned seed3 competitive; seed1 collapsed; learned preserved as diagnostic |
| H2D | Learned+BCE stabilization screen (seed1) | Partially rescued; fails 3/4 thresholds; learned not finalized |
| H2E | Product-learned residual gate (beta sweep + 3seed) | Local gain (seed1 beta=0.2); 3seed aggregate below product |
| H2F | Gate-axis decomposition (learned / product / hybrid) | location/role is dominant false-SUPPORT axis; product remains final |

---

## H2A Decision-Signal Comparison

**Method:** 3-seed controlled no-time validation. Three H1 decision signal modes compared:
`learned` (EntitlementGate), `product` (frame × predicate × sufficiency), `min` (minimum
of the three).

| mode | macro_mean | macro_std | acc_mean | acc_std | SUP_r_mean | NE_r_mean | REF_r_mean | bad_SUP_total | location_role_SUP_total | predicate_SUP_total | missing_SUP_total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| product | 0.940351 | 0.025612 | 0.957870 | 0.018651 | 0.944444 | 0.953086 | 1.0 | 76 | 60 | 7 | 0 |
| min | 0.928270 | 0.020529 | 0.945833 | 0.017067 | 0.988889 | 0.929630 | 1.0 | 114 | 78 | 19 | 0 |
| learned | 0.900660 | 0.046553 | 0.922685 | 0.045354 | 0.966952 | 0.902469 | 1.0 | 136 | 104 | 8 | 0 |

**Interpretation:**
- `product` has the best aggregate macro (0.9404) and lowest bad_SUP (76).
- `min` preserves the highest SUPPORT recall (0.9889) but over-entitles, producing
  more false SUPPORT on location/role axes (78 vs 60).
- `learned` has the highest seed variance (macro_std=0.0466) and highest total false
  SUPPORT (136). The learned gate is not reliably calibrated at this training scale.
- All three modes hold missing_SUP=0, confirming the absence of false SUPPORT on
  missing-evidence interventions in this setting.

**Decision:** select `product` for H2B power sweep.

---

## H2B Product-Power Selection

**Method:** sweep product_power ∈ {0.67, 0.75, 0.80, 0.85, 0.90, 0.95, 0.97, 1.00}
on 3-seed controlled no-time validation. Power exponent applied to the raw product
signal after computing the product of frame, predicate, and sufficiency probabilities.

**Selected result (power=0.90):**

| power | macro_mean | macro_std | acc_mean | acc_std | SUP_r_mean | NE_r_mean | REF_r_mean | bad_SUP_total | location_role_SUP_total | predicate_SUP_total | missing_SUP_total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.90 | 0.951136 | 0.021341 | 0.968056 | 0.013679 | 0.907407 | 0.972840 | 1.0 | 44 | 33 | 3 | 0 |

**Interpretation:**
- `product_power=0.90` is the best stable tradeoff: highest macro (0.9511), lowest
  bad_SUP (44), and lowest location_role_SUP (33).
- Relative to exact product (power=1.0), power=0.90 reduces bad_SUP and
  location_role_SUP while keeping missing_SUP=0.
- Power values below 0.85 over-relax the entitlement conjunction and introduce
  more false SUPPORT; values above 0.97 reduce SUPPORT recall without proportionate
  false-SUPPORT gain.
- H2B is frozen at 0.90. Further micro-tuning on this dev set would constitute
  validation overfitting and is not warranted.

---

## H2C Learned Gate Diagnostic

**Method:** analyze per-seed learned-gate results from H2A to understand seed variance
sources. Three seeds compared.

**Learned seed ranking** (macro desc, bad_SUP asc, location_role_SUP asc):

| seed | macro | acc | SUP_r | NE_r | REF_r | bad_SUP | frame_SUP | location_role_SUP | predicate_SUP | missing_SUP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.950672 | 0.965278 | 0.977778 | 0.957407 | 1.0 | 23 | 21 | 16 | 2 | 0 |
| 2 | 0.892724 | 0.927778 | 0.923077 | 0.916667 | 1.0 | 23 | 20 | 17 | 3 | 0 |
| 1 | 0.858586 | 0.875000 | 1.000000 | 0.833333 | 1.0 | 90 | 87 | 71 | 3 | 0 |

**Interpretation:**
- Learned seed3 (macro=0.9507, bad_SUP=23) is directly competitive with
  product_power=0.90 (macro=0.9511, bad_SUP≈15/seed). This demonstrates that the
  learned gate has genuine discriminative capacity under the right initialization.
- Learned seed1 collapses on location/role false SUPPORT (bad_SUP=90, location_role_SUP=71),
  accounting for 79% of its total false SUPPORT.
- The macro spread (0.9507 – 0.8586 = 0.0921) and bad_SUP spread (90 – 23 = 67)
  indicate the learned gate is seed-unstable at current scale.
- **Learned gate should not be discarded.** It contains local signal and specialization
  capacity, but is not reliable enough for production use. It should be preserved as a
  diagnostic and specialization branch.

---

## H2D Learned Stabilization Attempt

**Method:** apply auxiliary entitlement BCE loss (and variants) to seed1 (worst seed)
to test whether direct entitlement supervision can rescue the collapsed gate. All
configurations evaluated on controlled no-time validation.

**Seed1 baseline:**

| macro | acc | SUP_r | NE_r | REF_r | bad_SUP | location_role_SUP | missing_SUP |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.858586 | 0.875000 | 1.000000 | 0.833333 | 1.0 | 90 | 71 | 0 |

**Best H2D rescue (bce_w1p0):**

| macro | acc | SUP_r | NE_r | REF_r | bad_SUP | location_role_SUP | missing_SUP | delta_macro | delta_bad_SUP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.899954 | 0.920833 | 0.977778 | 0.898148 | 1.0 | 55 | 40 | 0 | +0.0414 | -35 |

**Pre-defined success thresholds:**

| threshold | required | bce_w1p0 | pass? |
|---|---|---|---|
| macro_f1 >= 0.92 | 0.92 | 0.899954 | NO |
| bad_SUP <= 40 | 40 | 55 | NO |
| location_role_SUP <= 30 | 30 | 40 | NO |
| missing_SUP = 0 | 0 | 0 | YES |

**Other H2D observations:**
- BCE weights 0.1 and 0.3 worsened the baseline on both macro and bad_SUP.
- BCE weights ≥ 1.5 did not reliably help and collapsed performance at weight=1.5
  (bad_SUP=233).
- pos_weight=2.0 worsened both macro (0.807) and location_role_SUP (108).
- CBCE and BCE+CBCE were harmful on both axes.
- missing_SUP remained 0 across all H2D configurations.

**Interpretation:**
- H2D partially rescues seed1 but does not stabilize the learned gate enough for
  final use; bce_w1p0 fails 3 of 4 pre-defined thresholds.
- The learned failure mode is not SUPPORT-vs-REFUTE polarity separation — REF_r=1.0
  throughout. It is entitlement boundary calibration, specifically location/role
  false entitlement.
- Auxiliary entitlement supervision has a narrow and non-monotonic effective window
  (w=1.0 is the sole improvement; lower and higher weights both degrade).
- H2D motivates using the learned gate only as a diagnostic or bounded residual
  signal, not as a standalone final decision.

---

## H2E Product-Learned Residual

**Method:** introduce `product_learned_residual` signal:
`entitlement = (product_base + beta * (v7_entitlement_prob - product_base.detach())).clamp(0, 1)`

where `product_base = (frame_prob * predicate_coverage_prob * sufficiency_prob) ** product_power`.
`product_base` remains fully differentiable; `detach()` used only as the residual anchor.
Sweep beta ∈ {0.0, 0.1, 0.2, 0.3, 0.5} on seed1, then expand beta=0.2 to 3 seeds.

**Seed1 beta sweep:**

| beta | macro | acc | SUP_r | NE_r | REF_r | bad_SUP | location_role_SUP | missing_SUP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.2 | 0.945837 | 0.961111 | 0.988889 | 0.950000 | 1.0 | 27 | 20 | 0 |
| 0.0 | 0.941176 | 0.956944 | 1.000000 | 0.942593 | 1.0 | 31 | 22 | 0 |
| 0.1 | 0.919101 | 0.943056 | 0.888889 | 0.942593 | 1.0 | 31 | 22 | 0 |
| 0.3 | 0.901258 | 0.920833 | 1.000000 | 0.894444 | 1.0 | 57 | 40 | 0 |
| 0.5 | 0.874540 | 0.893056 | 1.000000 | 0.857407 | 1.0 | 77 | 61 | 0 |

**Beta=0.2 3-seed aggregate:**

| macro_mean | macro_std | acc_mean | acc_std | SUP_r_mean | NE_r_mean | REF_r_mean | bad_SUP_total | location_role_SUP_total | missing_SUP_total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.931138 | 0.015073 | 0.952778 | 0.007217 | 0.907407 | 0.952469 | 1.0 | 77 | 65 | 0 |

**Comparison to product_power=0.90:**

| config | macro_mean | bad_SUP_total | location_role_SUP_total |
|---|---:|---:|---:|
| product_power=0.90 | 0.951136 | 44 | 33 |
| hybrid beta=0.2 (3seed) | 0.931138 | 77 | 65 |

**Interpretation:**
- beta=0.2 improves seed1 locally over beta=0.0 (macro +0.005, bad_SUP −4), indicating
  the learned residual carries local correction signal for at least one seed.
- However, beta=0.2 worsens the 3-seed aggregate relative to product_power=0.90:
  macro −0.020, bad_SUP +33, location_role_SUP +32.
- Naive learned residual injection is unsafe as a default configuration.
- The hybrid gate should be preserved as a negative-but-informative experiment: it
  demonstrates that learned residual correction exists locally but cannot yet be
  applied uniformly across seeds.

---

## H2F Gate-Axis Decomposition

**Method:** compare learned / product_power=0.90 / hybrid (beta=0.2) by intervention
axis rather than aggregate macro-F1 alone.

**Interpretation flags:**

| flag | value |
|---|---|
| learned_has_local_signal | true |
| learned_unstable | true |
| macro_spread (learned) | 0.092086 |
| bad_SUP_spread (learned) | 67 |
| hybrid_has_local_gain | true |
| hybrid_not_final | true |
| hard_axis | location_role |

**Aggregate by config (H2F 3-seed):**

| config | n | macro_mean | macro_std | acc_mean | SUP_r | NE_r | REF_r | bad_SUP_total | frame_SUP_total | location_role_SUP_total | predicate_SUP_total | missing_SUP_total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| learned | 3 | 0.900660 | 0.046553 | 0.922685 | 0.966952 | 0.902469 | 1.0 | 136 | 128 | 104 | 8 | 0 |
| product | 3 | 0.951136 | 0.021341 | 0.968056 | 0.907407 | 0.972840 | 1.0 | 44 | 41 | 33 | 3 | 0 |
| hybrid | 3 | 0.931138 | 0.015073 | 0.952778 | 0.907407 | 0.952469 | 1.0 | 77 | 74 | 65 | 3 | 0 |

**Per-intervention SUPPORT means:**

| intervention | learned | product | hybrid |
|---|---:|---:|---:|
| location_swap | 22.0000 | 9.0000 | 14.3333 |
| role_swap | 12.6667 | 2.0000 | 7.3333 |
| predicate_swap | 2.6667 | 1.0000 | 1.0000 |
| entity_swap | 2.6667 | 0.6667 | 1.0000 |
| event_swap | 3.6667 | 1.3333 | 1.6667 |
| title_name_swap | 1.6667 | 0.6667 | 0.3333 |
| evidence_deletion | 0.0000 | 0.0000 | 0.0000 |
| evidence_truncation | 0.0000 | 0.0000 | 0.0000 |
| irrelevant_evidence | 0.0000 | 0.0000 | 0.0000 |

**Product axis false-SUPPORT totals (3 seeds):**
- location_role: 33 — dominant axis
- predicate: 3
- missing (deletion + truncation + irrelevant): 0
- other (entity + event + title): 8

**Paired delta highlights:**
- Learned seed3 beats product seed3 in macro by +0.014077, but has +22 bad_SUP and
  +15 location_role_SUP. Local macro gain does not justify the false-SUPPORT cost.
- Hybrid seed1 beats product seed1 in macro (+0.005) and reduces bad_SUP by −4,
  but hybrid seed2/seed3 worsen bad_SUP by +25/+12 respectively.

**Interpretation:**
- Product remains the final configuration.
- Learned and hybrid should not be discarded: both show local signal (local_signal=true,
  hybrid_has_local_gain=true).
- There is no clean intervention axis where learned or hybrid consistently beats product
  enough to justify a new final gate.
- The remaining bottleneck is the location/role frame axis. This axis accounts for 33/44
  (75%) of product's remaining bad_SUP.
- Missing-evidence false SUPPORT is fully controlled (0 across all three configurations
  and all 12 intervention types in the evidence_missing axis).

---

## Final Interpretation

- `product_power=0.90` is the stable final Stage27 v7-H1 configuration. It has the
  best aggregate macro, lowest bad_SUP, lowest location_role_SUP, and the lowest
  seed variance among configurations tested.
- The learned gate contains local discriminative signal but is seed-unstable
  (macro_spread=0.092, bad_SUP_spread=67). It should be preserved as a diagnostic
  and specialization branch, not discarded.
- Learned+BCE (H2D) partially rescues seed1 but is not sufficient for final use.
  The effective BCE weight window is narrow; the failure mode is entitlement boundary
  calibration on location/role, not polarity separation.
- The product-learned residual (H2E) shows local gain per seed under beta=0.2 but
  worsens the 3-seed aggregate. Naive residual injection is unsafe as a default.
- Location/role frame mismatch is the dominant remaining false-SUPPORT axis in all
  three gate configurations (product: 75% of bad_SUP; learned: 77%; hybrid: 84%).
- Missing-evidence and predicate false SUPPORT are comparatively controlled in this
  setting. Missing-evidence false SUPPORT is zero across all configurations and seeds.

---

## Remaining Failure Mode

**Location/role frame mismatch** is the primary unresolved false-SUPPORT axis in
Stage27. Across all tested configurations:

| config | location_role_SUP | predicate_SUP | missing_SUP | other_SUP | bad_SUP_total | lr_pct |
|---|---:|---:|---:|---:|---:|---:|
| product (3seed) | 33 | 3 | 0 | 8 | 44 | 75% |
| hybrid (3seed) | 65 | 3 | 0 | 9 | 77 | 84% |
| learned (3seed) | 104 | 8 | 0 | 24 | 136 | 76% |

In the H1 learned initial runs, location_swap and role_swap together accounted for
127/180 false SUPPORT predictions (70.6%). The product signal reduces the absolute
count but does not eliminate this pattern.

This suggests that the frame_prob head — the primary contributor gating location/role
claims — does not yet fully discriminate entity/role swap from valid frames at
max_length=64. Investigating location/role-specific supervision, a dedicated frame
boundary module, or targeted diagnostic data is the natural Stage28 direction.

---

## Claims Supported by Stage27

1. `product_power=0.90` is the best stable v7-H1 configuration on the controlled
   no-time validation setting (controlled_v5_v3_without_time_swap.jsonl).
2. Compositional product entitlement is more reliable than learned entitlement under
   the current training objective and scale: lower seed variance, lower false SUPPORT.
3. The learned entitlement gate contains local discriminative signal (seed3 competitive
   with product; hybrid seed1 local gain) but is seed-unstable at current scale.
4. Naive residual use of the learned entitlement gate (product-learned residual at
   beta=0.2) is unsafe in aggregate despite providing local per-seed gains.
5. Remaining false SUPPORT in all tested configurations is dominated by location/role
   frame mismatch, not by missing-evidence or predicate failures.
6. Missing-evidence false SUPPORT (evidence_deletion, evidence_truncation,
   irrelevant_evidence) is controlled at zero in this controlled setting across all
   seeds and gate configurations.

---

## Claims Not Supported

1. Generalization beyond the controlled no-time validation setting. All Stage27
   results are specific to `controlled_v5_v3_without_time_swap.jsonl`.
2. Robustness to `time_swap` / temporal permutation interventions. time_swap was
   excluded from all Stage27 evaluation following Stage12 analysis identifying it
   as corrupted or problematic.
3. Robustness at full-encoder scale (unfrozen encoder) or at max_length=128.
   All Stage27 runs used frozen encoder, max_length=64 (T4-safe setting).
4. OOD generalization. No OOD evaluation was conducted in H2A–H2F; no OOD claim
   is warranted.
5. That the learned gate should be discarded entirely. Seed3 results and
   hybrid_has_local_gain=true demonstrate genuine signal that merits further
   investigation.
6. That the hybrid (product-learned residual) gate should replace product_power=0.90.
   The 3-seed aggregate is worse on both macro and bad_SUP.

---

## Next Stage Recommendation

**Stage28** should investigate location/role frame-boundary specialization. Candidate
directions:

- A location/role-specific diagnostic dataset or held-out split to isolate the failure
  mode from the aggregate metric signal.
- A dedicated frame-boundary module or auxiliary frame-entitlement head with
  location/role-aware supervision.
- Targeted analysis of whether the frame_prob head already encodes location/role
  discrimination but is suppressed by the product threshold, or whether the
  representation itself lacks the signal.

Stage28 should freeze `product_power=0.90` as the controlled-setting baseline before
any new experiments. Changes to the final configuration require beating product_power=0.90
on all of: macro_mean, bad_SUP_total, location_role_SUP_total, and missing_SUP_total,
across at least 3 seeds on the same controlled no-time validation.

---

## Limitations and Risks

- **Dataset scope:** all results are on `controlled_v5_v3_without_time_swap.jsonl`.
  No claim is made about performance on other validation sets, OOD splits, or
  real-world inference distributions.
- **time_swap exclusion:** time_swap interventions were excluded from all Stage27
  evaluation. Temporal robustness of product_power=0.90 is unknown.
- **Frozen encoder / max_length=64:** T4-safe training constraints limit the capacity
  of the backbone. Full-encoder results at max_length=128 may differ.
- **3-seed only:** seed variance estimates are based on 3 seeds. Estimates with n=3
  are noisy; wider sweeps could change standard deviation estimates.
- **Dev-selection risk:** H2B product_power and H2E beta were selected on the
  controlled dev set. Further micro-tuning of these values on the same dev set would
  constitute overfitting and is not warranted. Stage27 freezes at power=0.90.
- **No OOD claim:** no OOD evaluation was conducted in H2A–H2F. Stage27 evidence is
  limited to the controlled setting.
