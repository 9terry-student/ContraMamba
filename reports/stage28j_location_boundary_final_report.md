# Stage28-J: Location-Boundary Cap — Final Frozen Report

**Date:** 2026-06-27
**Stage:** Stage28-J (freeze of Stage28-I final candidate)
**Dataset:** `data/controlled_v5_v3_without_time_swap.jsonl`
**Architecture:** `v7_hierarchical`
**Backbone:** `state-spaces/mamba-130m-hf` (real Mamba, frozen encoder, frozen A-log)
**Base decision:** H1 product entitlement (`product_p0p90`)
**Base product power:** 0.90
**Max length:** 64
**Selection metric:** `final_macro_f1`
**Prediction export schema:** `stage28e_v1`
**Time swap:** excluded from controlled data

---

## 1. Background

Stage27 and Stage28 established that the primary remaining false SUPPORT errors under the `product_p0p90_default_off` baseline are location errors. Out of 44 total false SUPPORT events across three controlled no-time seeds, 27 are attributable to location false SUPPORT. The Stage28 program was designed to reduce this class of error without disrupting the product H1 decision path or introducing new error categories.

Stage28-D designed the location-boundary specialist head. Stage28-H characterized the overlap structure between location errors and other frame errors. Stage28-I systematically tested cap modes (hard, soft), gamma values, and loss weights to identify a final candidate. Stage28-J freezes that candidate.

---

## 2. Why Learned/Hybrid Residual Mixing Was Rejected

An earlier direction considered learned or hybrid residual mixing as a way to blend the product entitlement signal with a location-boundary correction. This approach was rejected because:

- The number of product error rescues (cases where mixing would have corrected a product false SUPPORT) was too small to justify the added complexity.
- The mixing introduced newly created false SUPPORT errors in categories that were previously clean, producing a net negative result.
- The cap approach (multiplicative detached soft boundary) achieved the desired location suppression without touching the product scoring pathway directly.

---

## 3. Why Hard Cap Was Rejected

Stage28-I-C included a hard cap condition. The hard cap mode increased `false_SUPPORT_total` to 58, compared to the baseline of 44. The hard cap over-suppressed: it blocked entitlement in cases where the product signal was correct, converting true SUPPORT correct instances into false negatives. The soft cap with `gamma=1.0` and `detach=true` achieves location suppression without the over-suppression penalty.

---

## 4. Stage27/Stage28 Baseline

Config: `product_p0p90_default_off`, n_runs: 3

| Metric | Value |
|---|---|
| false SUPPORT total | 44 |
| location false SUPPORT | 27 |
| role false SUPPORT | 6 |
| predicate false SUPPORT | 3 |
| other-frame false SUPPORT | 8 |
| missing-evidence false SUPPORT | 0 |
| pred SUPPORT mean | 96.6667 |
| pred SUPPORT min | 66 |
| pred SUPPORT max | 121 |
| true SUPPORT correct total | 246 |
| true SUPPORT correct min | 65 |

---

## 5. Stage28-I-C Initial Soft Cap Result

Config: soft cap, `loss_weight=0.30`, `gamma=1.0`, `detach=true`

| Metric | Value |
|---|---|
| false SUPPORT total | 28 |
| location false SUPPORT | 8 |
| role false SUPPORT | 7 |
| predicate false SUPPORT | 6 |
| other-frame false SUPPORT | 7 |
| missing-evidence false SUPPORT | 0 |
| true SUPPORT correct total | 265 |

This confirmed that the soft cap at these settings substantially reduced location false SUPPORT (27 → 8) with no increase in missing-evidence false SUPPORT.

---

## 6. Stage28-I-D Gamma Sweep

Sweep: soft cap, `loss_weight=0.30`, `detach=true`, gamma ∈ {0.50, 0.75, 1.00, 1.25}

| gamma | false SUP total | loc false SUP | role false SUP | pred false SUP | other-frame false SUP | missing false SUP | true SUP correct |
|---|---|---|---|---|---|---|---|
| 0.50 | 45 | 19 | 10 | 7 | 9 | 0 | 267 |
| 0.75 | 43 | 18 | 9 | 7 | 9 | 0 | 265 |
| **1.00** | **28** | **8** | **7** | **6** | **7** | **0** | **265** |
| 1.25 | 43 | 11 | 8 | 18 | 6 | 0 | 266 |

**Winner: gamma=1.0**

- `gamma=0.50` and `gamma=0.75` are too weak: total false SUPPORT remains near or above baseline (45, 43) and location false SUPPORT is only modestly reduced (19, 18 vs. baseline 27).
- `gamma=1.25` over-sharpens the sigmoid: location false SUPPORT drops further (11) but predicate false SUPPORT spikes to 18, indicating that aggressive sharpening pushes errors into other categories.
- `gamma=1.0` achieves the largest net reduction in total false SUPPORT (44 → 28) and location false SUPPORT (27 → 8) without introducing category spikes.

---

## 7. Stage28-I-E Loss-Weight Sweep

Sweep: soft cap, `gamma=1.0`, `detach=true`, loss_weight ∈ {0.10, 0.20, 0.30, 0.50}

| loss_weight | false SUP total | loc false SUP | role false SUP | pred false SUP | other-frame false SUP | missing false SUP | pred SUP mean | pred SUP min | true SUP correct |
|---|---|---|---|---|---|---|---|---|---|
| 0.10 | 35 | 20 | 5 | 5 | 5 | 0 | 100.333 | 99 | 266 |
| 0.20 | 22 | 8 | 6 | 2 | 6 | 0 | 95.333 | 87 | 264 |
| 0.30 | 28 | 8 | 7 | 6 | 7 | 0 | 97.667 | 92 | 265 |
| **0.50** | **18** | **4** | **6** | **2** | **6** | **0** | **94.667** | **91** | **266** |

**Winner: loss_weight=0.50**

- `loss_weight=0.10` is insufficiently strong: location false SUPPORT remains at 20 and total false SUPPORT is 35.
- `loss_weight=0.20` achieves lower total false SUPPORT (22) than 0.30 (28) but has a lower pred SUPPORT min (87 vs. 92) and slightly lower true SUPPORT correct (264 vs. 265). The 0.50 result dominates on total false SUPPORT and true SUPPORT correct simultaneously.
- `loss_weight=0.30` is the Stage28-I-C reference; while competitive, 0.50 achieves a further reduction in total false SUPPORT (28 → 18) and location false SUPPORT (8 → 4) while maintaining true SUPPORT correct at 266.
- `loss_weight=0.50` achieves the best total false SUPPORT (18), best location false SUPPORT (4), and ties for highest true SUPPORT correct (266), with a pred SUPPORT min of 91 (vs. baseline 66). It is selected as the final candidate.

---

## 8. Final Stage28-I Candidate vs. Baseline

Config: `lb_loss0p50_soft_g1p00_detach`
- `v7_use_location_boundary_head=true`
- `v7_use_location_boundary_loss=true`
- `v7_location_boundary_loss_weight=0.50`
- `v7_location_boundary_cap_mode=soft`
- `v7_location_boundary_cap_gamma=1.0`
- `v7_location_boundary_cap_detach=true`
- Base decision: `product_p0p90` (unchanged)

| Metric | Baseline | Final Candidate | Delta |
|---|---|---|---|
| false SUPPORT total | 44 | 18 | **-26** |
| location false SUPPORT | 27 | 4 | **-23** |
| role false SUPPORT | 6 | 6 | 0 |
| predicate false SUPPORT | 3 | 2 | -1 |
| other-frame false SUPPORT | 8 | 6 | -2 |
| missing-evidence false SUPPORT | 0 | 0 | 0 |
| true SUPPORT correct total | 246 | 266 | **+20** |
| pred SUPPORT min | 66 | 91 | **+25** |

---

## 9. Interpretation and Scope

**Controlled no-time evidence only.** All results reported here are on `data/controlled_v5_v3_without_time_swap.jsonl` across three controlled seeds. These are not OOD claims, open-world claims, or external generalization claims. No OOD evaluation was performed in Stage28.

**Time swap excluded.** Time-swap examples remain excluded from the controlled data used in all Stage28 experiments. The effect of this candidate on time-swap examples is unknown and is not claimed.

**Not merely conservative suppression.** The improvement is not achieved by broad evidence rejection. True SUPPORT correct total increases from 246 to 266 (+20), and pred SUPPORT min improves from 66 to 91 (+25). The model is not simply suppressing more entitlement globally; it is more precisely targeting location-inconsistent false SUPPORTs while improving correct SUPPORT coverage.

**Bounded and independent cap.** The location-boundary cap does not replace or modify the product H1 decision path. It operates as a multiplicative soft boundary compatibility probability computed by a separate head, applied with gradient detachment (`detach=true`) so that the product path gradient is not affected. The product scoring pathway and the ranking loss remain unchanged. The cap is bounded in [0, 1] by the soft mode formulation.

---

## 10. Remaining Risks

- **Controlled synthetic setting only.** Results are on controlled no-time data. The candidate has not been evaluated on any OOD, time-swap, or naturalistic distribution. No generalization is claimed or implied.
- **Frozen encoder and max_length=64.** The encoder (Mamba backbone) and A-log parameters are frozen. Results are not directly transferable to fine-tuned or longer-context variants without re-evaluation.
- **Ranking loss active.** The ranking loss remains active by default. This is not a CE-only setup. Any future ablation of the ranking loss would require fresh evaluation to confirm that location-boundary improvements hold.
- **Limited boundary supervision mapping.** The location-boundary target supervision uses a limited positive/negative mapping over the controlled data. Edge cases in boundary annotation or sparse boundary labels in new data could affect head quality.
- **Future evaluation needed.** The candidate should be evaluated outside the controlled no-time setting before broader deployment or additional architectural commitment. Stage29 is the appropriate vehicle for this.

---

## 11. Next Recommended Stage

**Stage29** should test the frozen Stage28-I final candidate (`lb_loss0p50_soft_g1p00_detach`) against additional diagnostic data or OOD-style probes to characterize how the location-boundary cap behaves outside the controlled no-time distribution. Stage29 should not add another router or cap layer unless a specific remaining failure mode demands it. The Stage28-I candidate should be treated as frozen and evaluated as-is; architectural modifications should be deferred until Stage29 diagnostics reveal a concrete gap that existing components cannot address.

---

## 12. Conclusion

Stage28-I finalizes an independent detached soft location-boundary cap over the product-p0.90 H1 decision path. With loss_weight=0.50 and gamma=1.0, the cap reduces total false SUPPORT from 44 to 18 across three controlled no-time seeds, reduces location false SUPPORT from 27 to 4, preserves missing-evidence rejection at zero false SUPPORT, and improves true SUPPORT correctness from 246 to 266.
