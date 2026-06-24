# Stage22 G2/G3 Controlled-Only Calibration — Summary

**Date:** 2026-06-24
**Backbone:** mamba (`state-spaces/mamba-130m-hf`)
**Device:** CUDA
**Seeds:** 1, 2, 3
**Stage15 used for shift selection:** false (both G2 and G3)

---

## Conclusion

Controlled-only calibration of a selective NOT_ENTITLED logit shift (Stage22 G2 and G3) selected
zero shift across all three seeds and both calibration pool sizes. The OOD metrics for G2 and G3
are therefore identical to the unshifted baseline. This is a negative but informative result:
the calibration objective on controlled data did not identify a shift that would improve the
preservation/rejection trade-off observed under OOD-tuned shift selection (Stage22 G1). The
model's controlled-data calibration pool correctly distinguishes frame-mismatch records from
preservation-like records without any shift, so no shift is optimal under the controlled objective.
The challenge of recovering OOD preservation behavior without OOD feedback remains unsolved.

---

## Experimental setup

### Common configuration

| Parameter | Value |
|---|---|
| Backbone | `mamba` (state-spaces/mamba-130m-hf) |
| Device | CUDA |
| freeze_encoder | true |
| freeze_a_log | true |
| use_frame_violation_loss | true |
| frame_violation_loss_weight | 0.05 |
| allow_dummy_backbone | false |
| stage15_used_for_shift_selection | false |
| Shift candidates | 0.0, 0.25, 0.5, 0.75, 1.0 |
| Gate | high_sufficiency (sufficiency_prob >= threshold) |
| Threshold | 0.6 |

### Calibration objectives

**Objective (both G2 and G3):**
```
score = pres_accept_rate - frame_penalty × frame_false_entitled_rate
```

- `pres_accept_rate`: fraction of preservation-like records (`intervention_type ∈ {none, paraphrase}`)
  predicted non-NOT_ENTITLED
- `frame_false_entitled_rate`: fraction of frame-mismatch records (`entity_swap`, `event_swap`,
  `location_swap`, `role_swap`, `title_name_swap`) predicted non-NOT_ENTITLED
- Stage15 OOD labels and records are never used in this computation

### G2 vs G3 calibration pool

| Method | Calibration pool | Pres records | Frame records |
|---|---|---|---|
| G2 | Controlled dev only | 4 (mean across seeds) | 10 (mean across seeds) |
| G3 | Controlled train + dev | 20 (mean across seeds) | 50 (mean across seeds) |

G3 adds `--dev-calibrated-ne-frame-penalty-candidates 0.5,1.0,1.5,2.0` to jointly select
the best (penalty, shift) pair.

---

## G2 results — controlled dev calibration (3-seed)

Calibration source: `controlled_dev_only`. Single fixed penalty = 2.0.

### Per-seed calibration

| Seed | Selected shift | Dev objective score | Pres accept rate | Frame FE rate | Dev pres total | Dev frame total |
|---|---|---|---|---|---|---|
| 1 | 0.0 | 0.750 | 0.75 | 0.00 | 4 | 10 |
| 2 | 0.0 | 0.750 | 0.75 | 0.00 | 4 | 10 |
| 3 | 0.0 | 0.750 | 0.75 | 0.00 | 4 | 10 |
| **Mean** | **0.0** | **0.750** | **0.75** | **0.00** | — | — |

Zero shift was optimal in all three seeds because at shift=0.0 the frame false-entitlement rate
on controlled dev was already 0.0, making any positive shift strictly worse under the objective
(it increased `frame_false_entitled_rate` while not improving `pres_accept_rate`).

### Per-seed OOD evaluation (shift = 0.0, no shift applied)

| Seed | OOD acc | OOD macro-F1 | surface FNE | temp_erased FNE | frame_loc FE | frame_role FE |
|---|---|---|---|---|---|---|
| 1 | 0.7037 | 0.3921 | 0.85 | 0.68 | 0.15 | 0.20 |
| 2 | 0.7648 | 0.4714 | 0.62 | 0.53 | 0.30 | 0.30 |
| 3 | 0.6611 | 0.3219 | 0.97 | 0.83 | 0.00 | 0.15 |
| **Mean** | **0.7099** | **0.3951** | **0.8133** | **0.6800** | **0.1500** | **0.2167** |
| **SD** | ±0.0521 | ±0.0748 | ±0.178 | ±0.150 | ±0.150 | ±0.076 |

Comparator-guarded groups (no false entitlement by construction):
- temporal_mismatch acc: 1.000 (all seeds)
- predicate_mismatch acc: 1.000 (all seeds)
- sufficiency_control acc: 1.000 (all seeds)

---

## G3 results — controlled train+dev calibration (3-seed)

Calibration source: `controlled_train_dev_only`. Penalty jointly selected from {0.5, 1.0, 1.5, 2.0}.
Calibration pool: 20 preservation-like records, 50 frame-mismatch records, 120 unflagged total.

### Per-seed calibration

| Seed | Selected shift | Selected penalty | Objective score | Cal pres total | Cal frame total |
|---|---|---|---|---|---|
| 1 | 0.0 | 2.0 | 0.950 | 20 | 50 |
| 2 | 0.0 | 0.5 | 0.920 | 20 | 50 |
| 3 | 0.0 | 2.0 | 0.950 | 20 | 50 |
| **Mean** | **0.0** | **1.5** | **0.940** | — | — |
| **SD** | ±0.0 | ±0.866 | ±0.017 | — | — |

The selected penalty varies across seeds (0.5, 2.0, 2.0) but the selected shift is zero in all
three. The larger calibration pool did not change the qualitative result: zero shift was still
optimal because controlled-data frame-mismatch records were correctly rejected without any shift.

### Per-seed OOD evaluation (shift = 0.0)

| Seed | OOD acc | OOD macro-F1 | surface FNE | temp_erased FNE | frame_loc FE | frame_role FE |
|---|---|---|---|---|---|---|
| 1 | 0.7037 | 0.3921 | 0.85 | 0.68 | 0.15 | 0.20 |
| 2 | 0.7648 | 0.4714 | 0.62 | 0.53 | 0.30 | 0.30 |
| 3 | 0.6611 | 0.3219 | 0.97 | 0.83 | 0.00 | 0.15 |
| **Mean** | **0.7099** | **0.3951** | **0.8133** | **0.6800** | **0.1500** | **0.2167** |
| **SD** | ±0.0521 | ±0.0748 | ±0.178 | ±0.150 | ±0.150 | ±0.076 |

OOD metrics are identical to G2 because both methods selected shift=0.0.

---

## G2 / G3 comparison

| Metric | G2 (dev only) | G3 (train+dev) | Difference |
|---|---|---|---|
| Selected shift (mean) | 0.0 | 0.0 | 0.0 |
| Calibration pres records | 4 | 20 | +16 |
| Calibration frame records | 10 | 50 | +40 |
| OOD acc (mean) | 0.7099 | 0.7099 | 0.0000 |
| OOD macro-F1 (mean) | 0.3951 | 0.3951 | 0.0000 |
| surface FNE (mean) | 0.8133 | 0.8133 | 0.0000 |
| temporal_erased FNE (mean) | 0.6800 | 0.6800 | 0.0000 |
| frame_loc FE (mean) | 0.1500 | 0.1500 | 0.0000 |
| frame_role FE (mean) | 0.2167 | 0.2167 | 0.0000 |

Expanding the calibration pool from 4 pres / 10 frame records (G2) to 20 pres / 50 frame
records (G3) did not change the selected shift or any OOD metric. The negative result is robust
to calibration pool size within the available controlled data.

---

## Interpretation

### Why zero shift was selected on controlled data

The model correctly assigned NOT_ENTITLED to all frame-mismatch records in the controlled
calibration pool at shift=0.0 (frame FE rate = 0.0 for G2). With the objective
`score = pres_accept_rate - penalty × frame_FE_rate`, a shift of zero already achieves
the maximum possible score: frame FE rate cannot go below zero, and applying a positive
shift did not consistently improve pres_accept_rate (it was 0.75 at shift=0 and only
reached 1.0 at shift=1.0 for seed 1, at the cost of increased frame FE rate). Zero
was therefore the only locally and globally optimal shift under this objective on
controlled data.

### Why this does not recover the G1 OOD-tuned improvement

Stage22 G1 (not documented in this file) used the Stage15 OOD probe directly to select
the shift value. The G1 procedure is properly described as a diagnostic upper bound or
oracle measurement — it reveals how much headroom exists for a selective NE shift if
an external OOD signal were available, but it does not provide a deployable mechanism
because the OOD probe cannot be used at deployment time.

G2 and G3 attempt to reproduce the same improvement using only controlled data, but the
controlled calibration pool does not reproduce the distributional conditions under which
a non-zero shift helps on Stage15 OOD. Specifically:

- On Stage15 OOD, many preservation-like records (`surface_control`, `temporal_erased`)
  are incorrectly predicted NOT_ENTITLED (high FNE rates: 0.81 surface, 0.68 temporal_erased).
- A positive shift would reduce this misclassification.
- But on controlled data, these same preservation-like records are correctly classified —
  the model already predicts SUPPORT for them.
- Therefore the controlled objective sees no benefit from a positive shift and correctly
  selects shift=0.

This reveals a genuine distribution shift: the model is calibrated on controlled preservation
records but fails on OOD preservation variants (surface paraphrases and temporal-phrase-erased
variants from the Stage15 probe). Controlled-data calibration cannot close this gap because
it does not expose the failure mode.

### Stage22 G1 as an upper bound, not a claim

Stage22 G1 should be documented as:
- A measurement of the maximum possible improvement from selective NE shifting with an
  oracle OOD signal
- Not a deployable or non-leaky improvement
- Not claim-worthy as a model improvement over the baseline

G2 and G3 are the appropriate non-leaky calibration checks. Their negative result is the
honest characterization of what the model achieves without OOD feedback.

---

## Claim status

**Accurate description of Stage22 calibration results:**

> "Selective NOT_ENTITLED logit shifting exposes a latent preservation/rejection trade-off
> on the Stage15 OOD probe. However, controlled-only calibration (on held-out dev and on
> train+dev, with 4–20 preservation-like and 10–50 frame-mismatch records) consistently
> selects a zero shift across three seeds. The controlled calibration objective correctly
> reaches its maximum at shift=0 because the model already correctly rejects frame-mismatch
> records in the controlled distribution. The OOD-tuned improvement (G1) should be treated
> as a diagnostic upper bound, not a deployable result. No non-leaky mechanism for recovering
> the OOD preservation behavior has been identified in Stage22."

**Do not claim:**
- That G2 or G3 improved OOD preservation accuracy
- That the selective NE shift is a validated deployable improvement
- That Stage22 resolved the preservation/rejection trade-off on the Stage15 probe
- That G1 (OOD-tuned) shift selection is a non-leaky calibration procedure

---

## Remaining risks and next possible directions

### Why the current approach did not generalize

The core problem is a covariate shift between controlled preservation records (clean paraphrases
and `none` interventions from the controlled dataset) and OOD preservation variants (`surface_control`
paraphrases and `temporal_erased` records from the Stage15 probe). The model's behavior on these
two distributions is different: correct on controlled, incorrect on OOD.

### Possible next directions

1. **OOD-informed calibration with held-out OOD dev:** If a portion of Stage15-like OOD data
   were available as a held-out calibration split (distinct from the test probe), it could
   be used as a non-leaky signal. This requires constructing a separate calibration OOD set
   that is not the Stage15 test probe.

2. **Distribution-aligned synthetic augmentation:** Generate synthetic preservation variants
   that more closely match the surface_control / temporal_erased structure of the Stage15 probe
   (without copying Stage15 records). These could be used to extend the controlled calibration
   pool with OOD-like preservation examples.

3. **Representation-level analysis:** The G1/G2/G3 discrepancy indicates the model's
   `sufficiency_prob` gate selects different records on OOD vs. controlled data. Understanding
   which OOD records the gate selects (and why the shift helps them) could guide a
   principled architecture change.

4. **Accept the current result and frame Stage22 as a negative diagnostic:** If no feasible
   non-leaky improvement path is identified, the Stage22 result should be reported as:
   the model exhibits a preservation/rejection trade-off under distribution shift that
   is not correctable through post-hoc logit adjustment calibrated on controlled data alone.

### Provenance notes

All G2 and G3 runs used real Mamba backbone (`state-spaces/mamba-130m-hf`) on CUDA.
`allow_dummy_backbone = false` was enforced by the fail-fast guard added in the G2
implementation. `stage15_used_for_shift_selection = false` is written into every result
JSON and is verifiable from the calibration code path.

Result files:
- [`results/stage22_G2_devcal_summary.json`](../results/stage22_G2_devcal_summary.json)
- [`results/stage22_G3_train_dev_calib_summary.json`](../results/stage22_G3_train_dev_calib_summary.json)
- [`results/stage22_G1_G2_G3_comparison_summary.json`](../results/stage22_G1_G2_G3_comparison_summary.json)
