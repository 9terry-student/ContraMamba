# Stage29-D External Probe Evaluation Report

**Stage:** 29-D  
**Candidate:** `stage28i_lb_loss0p50_soft_g1p00_detach`  
**Date:** 2026-06-27  
**Protocol:** External probe evaluation only. Probe files are never used for training,
dev split construction, checkpoint selection, calibration, or loss computation.

---

## 1. Eval Protocol

- Train data: `data/controlled_v5_v3_without_time_swap.jsonl`
- `time_swap` excluded from clean controlled training data
- Checkpoint selected on controlled dev macro-F1 only
- External probes evaluated on the selected best checkpoint (eval-only)
- Stage28-I location-boundary cap config:
  - `v7_use_location_boundary_head = true`
  - `v7_location_boundary_loss_weight = 0.50`
  - `v7_location_boundary_cap_mode = soft`
  - `v7_location_boundary_cap_gamma = 1.0`
  - `v7_location_boundary_cap_detach = true`
- `stage15_used_for_training = false`
- `stage15_used_for_checkpoint_selection = false`
- `external_probe_used_for_checkpoint_selection = false`

---

## 2. Controlled Dev Results (n=3 runs)

| Metric | baseline `product_p0p90_default_off` | Stage28-I `lb_loss0p50_soft_g1p00_detach` | Delta |
|---|---|---|---|
| dev_acc_mean | 0.9681 | 0.9894 | +0.0213 |
| dev_macro_f1_mean | 0.9511 | 0.9838 | +0.0327 |
| pred_SUPPORT_mean | 96.7 | 94.7 | -2.0 |
| pred_SUPPORT_min | 66 | 91 | +25 |

Stage28-I substantially improves controlled dev macro-F1 (+0.033) and eliminates low
SUPPORT-prediction outliers (min improves from 66 to 91), indicating the location-boundary
cap stabilises prediction diversity without sacrificing accuracy.

---

## 3. External Probe: Stage10a Number-Swap (n=3 runs, 120 records/run)

| Metric | baseline | Stage28-I | Delta |
|---|---|---|---|
| final_accuracy_mean | 0.5000 | 0.5000 | 0.0000 |
| final_macro_f1_mean | 0.2222 | 0.2222 | 0.0000 |
| pred_SUPPORT_mean | 0 | 0 | 0 |
| pred_NOT_ENTITLED_mean | 120 | 120 | 0 |
| false_SUPPORT_total | 0 | 0 | 0 |
| true_SUPPORT_correct_total | 0 | 0 | 0 |

**Interpretation:** Both configurations collapse all 120 stage10a number-swap examples to
NOT_ENTITLED. Accuracy of 0.50 reflects the 50/50 SUPPORT/NOT_ENTITLED gold label split
being entirely predicted as NOT_ENTITLED. Stage28-I does not worsen number-swap, but also
does not improve it. This probe is degenerate for comparing location-boundary configs: with
zero SUPPORT predictions in both conditions it serves only as a no-false-SUPPORT regression
check, which both configs pass. The number-swap failure mode requires separate targeting in
a future stage.

---

## 4. External Probe: Stage15 Slot Sensitivity (n=3 runs, 540 records/run)

| Metric | baseline | Stage28-I | Delta |
|---|---|---|---|
| final_accuracy_mean | 0.7531 | 0.7450 | -0.0081 |
| final_macro_f1_mean | 0.4887 | 0.4911 | +0.0024 |
| pred_SUPPORT_mean | 238.0 | 248.3 | +10.3 |
| pred_SUPPORT_min | 132 | 226 | +94 |
| false_SUPPORT_total | 257 | 279 | +22 |
| true_SUPPORT_correct_total | 457 | 466 | +9 |

### 4a. Stage15 Axis False SUPPORT Breakdown

| Diagnostic Axis | baseline | Stage28-I | Delta |
|---|---|---|---|
| location | 6 | 4 | -2 |
| predicate | 2 | 2 | 0 |
| role | 3 | 3 | 0 |
| temporal | 246 | 270 | +24 |
| **total** | **257** | **279** | **+22** |

**Axis interpretation:**

- **Location:** Improves from 6 to 4 false SUPPORTs (-2). The Stage28-I location-boundary
  cap provides weak but directionally correct generalisation to the Stage15 location-swap
  axis. This is the intended effect of the cap.
- **Predicate:** Unchanged (2 → 2). The cap has no effect on predicate mismatch, as expected.
- **Role:** Unchanged (3 → 3). No role-axis effect.
- **Temporal:** Worsens from 246 to 270 (+24). The increase in total false SUPPORT is driven
  almost entirely by the temporal axis. The location-boundary cap creates a mild SUPPORT bias
  that the temporal mismatch pathway exploits.

**Overall Stage15 interpretation:**

- Accuracy declines slightly (−0.008); macro-F1 is nearly unchanged (+0.002).
- True SUPPORT correct improves by +9, consistent with the location cap reducing unnecessary
  NOT_ENTITLED predictions on entitled examples.
- The dominant failure mode is **temporal false SUPPORT**, which accounts for 246/257 (96%)
  of baseline false SUPPORTs and rises to 270/279 (97%) under Stage28-I.
- Stage28-I is not a broad OOD or generalization solution. The location benefit is narrow
  and the temporal pathway remains unaddressed.

---

## 5. Decision and Conclusions

> **Required conclusion (Stage29-C carry-forward):**
>
> Stage29-C shows that the Stage28-I detached soft location-boundary cap remains beneficial
> on controlled no-time dev and weakly reduces external location-frame false SUPPORT, but it
> does not provide broad external probe generalization. On Stage15 slot sensitivity, total
> false SUPPORT increases from 257 to 279 because temporal false SUPPORT rises from 246 to
> 270. Therefore Stage28-I should be frozen as a controlled no-time/location-boundary
> improvement, while Stage30 should target the newly exposed temporal false-SUPPORT failure
> mode.

### Decisions

| Decision | Value |
|---|---|
| Freeze Stage28-I as controlled location-boundary improvement | **YES** |
| Claim broad external OOD generalization | **NO** |
| Stage30 primary target | **temporal_false_support** |

### What Stage30 should do first

The next stage should **not** add a broad temporal router immediately. It should first
analyse temporal false-SUPPORT records in Stage15 and compare:

- `temporal_mismatch` probe types (direct time-swap failures)
- `temporal_erased` / `surface` / `sufficiency` controls

Understanding which temporal sub-type drives the 246–270 false-SUPPORT range will determine
whether the correct intervention is a temporal diagnostic head, a separate temporal channel
penalty, or a data-augmentation strategy. Premature routing risks Stage23-style gradient
coupling.

---

## 6. Remaining Risks

1. **Temporal generalisation gap:** 270 temporal false SUPPORTs on Stage15 remain unaddressed.
   No temporal fix is included in Stage28-I; Stage30 must treat this independently.
2. **Number-swap degenerate:** The stage10a probe is currently non-informative. Both configs
   collapse to NOT_ENTITLED. A fix for number-swap coverage requires a separate stage after
   Stage30.
3. **External probe scope:** Stage15 and Stage10a are the only external probes evaluated.
   Results should not be extrapolated to unseen intervention distributions.
4. **n=3 runs:** Aggregate metrics are based on three seeds per config. Variance estimates
   are not reported here; individual-run prediction JSONs should be consulted for stability.
5. **`time_swap` not in training:** The controlled clean training set excludes time_swap.
   Temporal mismatch generalisation cannot be expected from this training regime without
   explicit temporal supervision.
