# Stage22-A4 Final Synthesis Notes

**Date:** 2026-06-24
**Status:** Negative diagnostic result — Stage22-B gate remains rejected
**Scope:** Stage22-A4a through A4e pair-group contrastive frame supervision
**Backbone:** dummy-backbone diagnostic setup only

---

## 1. Executive conclusion

Stage22-A4 investigated whether controlled synthetic pair-contrastive supervision on
`frame_violation_logit` can produce a safe Stage15 OOD frame-mismatch ranking signal
without training on Stage15 records.

The investigation produced functioning data generators, a working pair-contrastive
loss implementation, and evidence that internal pair ranking is learnable from both
generic (A4b/A4c) and OOD-matched (A4d/A4e) controlled pairs. However, none of the
tested configurations produced a transferable Stage15 OOD ranking signal. In all tested
configs the OOD ranking criterion failed:

> `frame_location/frame_role` frame_violation_prob did not exceed `surface_control /
> temporal_erased` frame_violation_prob by >= 0.10.

The failure is interpretated as a representational/generalization mismatch under the
dummy-backbone diagnostic setup, not as an implementation or optimization failure.
Stage22-B positive recovery gate remains rejected. Whether a real Mamba/full-backbone
run changes this conclusion is an open question not answered by this stage.

---

## 2. Timeline of A4a to A4e

| Stage | Output | Status |
|---|---|---|
| A4a | Audit script + alignment plan; confirmed pair_id-based generation feasible | Complete |
| A4b | Generic pair-group contrastive dataset generator (100 pairs, all 5 frame types) | Complete |
| A4b2 | Suitability refinement: support-safe vs non-safe anchors, frame_is_frame_violation filter | Complete |
| A4c | Pair-contrastive ranking loss in training script; 5 new CLI args; internal ranking measurable | Complete |
| A4d | OOD-matched pair dataset builder (20 pairs: surface_like + temporal_erased_like vs location_like + role_like) | Complete |
| A4e | Extended A4c loader to support A4d schema; 3 new use-case filters; schema normalization | Complete |
| Synthesis | This document | Complete |

All implementations passed `py_compile` and `--help` static validation. No training code
or model code was modified beyond the explicitly scoped auxiliary loss path.

---

## 3. Data generation summary

### A4b/A4b2 — generic pair-group contrastive dataset

| Property | Value |
|---|---|
| Source | `controlled_v5_seed.jsonl` (controlled data only) |
| Output pair count | 100 |
| Preservation types | `none` = 50, `paraphrase` = 50 |
| Frame types | `entity_swap` = 20, `event_swap` = 20, `location_swap` = 20, `role_swap` = 20, `title_name_swap` = 20 |
| preservation_is_support_safe_anchor | 50 |
| frame_is_frame_violation | 100 |
| contrastive_use_case breakdown | `frame_violation_contrastive` = 50, `support_safe_frame_contrastive` = 50 |
| Leakage | None — `leakage_note = "constructed_from_controlled_data_only"` |

A4b2 key insight: `intervention_type in {none, paraphrase}` does not guarantee
`final_label == SUPPORT`. Some none/paraphrase records have `polarity_label == REFUTE`.
The `contrastive_use_case` field safely routes downstream filtering.

### A4d — OOD-matched pair dataset

| Property | Value |
|---|---|
| Source | `controlled_v5_seed.jsonl` (controlled data only) |
| Output pair count | 20 |
| pair_ids used | 5 (of 10 available; 5 skipped due to no support-safe anchor or no location/role frame candidate) |
| preservation_construction_type | `surface_like_preservation` = 10, `temporal_erased_like_preservation` = 10 |
| frame_construction_type | `frame_location_like` = 10, `frame_role_like` = 10 |
| All anchors | Support-safe (SUPPORT label + all aux labels consistent) |
| Frame candidates | Filtered to NOT_ENTITLED + primary_failure_type == frame + sufficiency_label == 1 |
| Leakage | None — `leakage_note = "constructed_from_controlled_data_only_no_stage15_records"` |

A4d construction types directly mirror Stage15 OOD group structure:
`surface_like_preservation` vs `surface_control`, `temporal_erased_like_preservation`
vs `temporal_erased`, `frame_location_like` vs `frame_location_mismatch`,
`frame_role_like` vs `frame_role_mismatch`.

---

## 4. Pair loss summary

Loss definition (unchanged across A4c through A4e):

```
pc_loss = relu(margin - (frame_fvl - pres_fvl)).mean()
total_loss += pc_loss_weight * pc_loss
```

Where:
- `frame_fvl` = `frame_violation_logit` from a forward pass on the frame-side evidence
- `pres_fvl` = `frame_violation_logit` from a forward pass on the preservation-side evidence
- Both forward passes use `temporal_mismatch_flags = 0` and `predicate_mismatch_flags = 0`
- `margin` = `--pair-contrastive-frame-margin` (default 0.2)

Hard constraints satisfied throughout:
- `output["logits"]` is never modified — final logits unchanged
- CE, pairwise, and intervention losses continue to use `output["logits"]`
- `output["base_logits"]` remains diagnostic only
- No Stage15 OOD records are loaded at any training step
- No OOD group names or OOD labels appear in the loss or model forward path

---

## 5. A4c results

### Compact OOD table (3-seed mean, dummy backbone)

| config | fv_frame_loc | fv_frame_role | fv_surface | fv_temporal | FE_frame_loc | FE_frame_role | FNE_surface | FNE_temporal | acc_temp_mm | acc_pred_mm |
|---|---|---|---|---|---|---|---|---|---|---|
| no_pair_frame_w0p05 | 0.394 | 0.395 | 0.421 | 0.457 | 0.517 | 0.417 | 0.527 | 0.810 | 1.0 | 1.0 |
| pair_all_w0p05 | 0.331 | 0.330 | 0.365 | 0.413 | 0.583 | 0.450 | 0.483 | 0.817 | 1.0 | 1.0 |
| pair_frame_w0p05 | 0.334 | 0.331 | 0.368 | 0.421 | 0.550 | 0.467 | 0.487 | 0.827 | 1.0 | 1.0 |
| pair_supportsafe_w0p05 | 0.389 | 0.394 | 0.419 | 0.450 | 0.467 | 0.383 | 0.620 | 0.830 | 1.0 | 1.0 |
| pair_supportsafe_w0p2 | 0.343 | 0.348 | 0.372 | 0.407 | 0.533 | 0.417 | 0.533 | 0.830 | 1.0 | 1.0 |

### A4c findings

- Internal pair ranking was achievable (`pair_contrastive_frame_accuracy` reached high values).
- No config fixed OOD ranking: `fv_frame_location` and `fv_frame_role` remained below
  `fv_surface` and `fv_temporal_erased` in all tested configs.
- `pair_supportsafe_w0p05` reduced frame FE slightly but increased surface/temporal FNE
  (0.620 vs 0.527 for surface, baseline recovery attempt trades off against false-not-entitled).
- Adding pair loss (`pair_all_w0p05`) worsened frame FE vs no-pair baseline (0.583 vs 0.517).
- Temporal/predicate comparator guards held at 1.0 across all configs.

---

## 6. A4d/A4e results

### Compact OOD table (3-seed mean, dummy backbone)

| config | fv_frame_loc | fv_frame_role | fv_surface | fv_temporal | FE_frame_loc | FE_frame_role | FNE_surface | FNE_temporal | acc_temp_mm | acc_pred_mm |
|---|---|---|---|---|---|---|---|---|---|---|
| no_pair_frame_w0p05 | 0.394 | 0.395 | 0.421 | 0.457 | 0.517 | 0.417 | 0.527 | 0.810 | 1.0 | 1.0 |
| a4d_oodmatched_w0p05 | 0.381 | 0.383 | 0.411 | 0.444 | 0.450 | 0.400 | 0.623 | 0.827 | 1.0 | 1.0 |
| a4d_oodmatched_w0p2 | 0.310 | 0.308 | 0.347 | 0.388 | 0.533 | 0.450 | 0.527 | 0.840 | 1.0 | 1.0 |
| a4d_surface_w0p05 | 0.328 | 0.329 | 0.363 | 0.407 | 0.617 | 0.450 | 0.460 | 0.827 | 1.0 | 1.0 |
| a4d_temporal_w0p05 | 0.428 | 0.420 | 0.448 | 0.466 | 0.550 | 0.400 | 0.553 | 0.823 | 1.0 | 1.0 |

### A4e findings

- Internal pair ranking succeeded: `a4d_oodmatched_w0p05` achieved
  `pair_contrastive_frame_accuracy = 1.0` with `valid_count = 20`.
- Despite perfect internal ranking, OOD transfer failed entirely.
- In all A4d/A4e configs, `fv_frame_location` and `fv_frame_role` remain lower than
  `fv_surface` and `fv_temporal_erased`. The OOD ranking criterion is not met.
- `a4d_oodmatched_w0p05` slightly reduced frame FE (0.450, 0.400) but raised surface FNE (0.623).
- `a4d_surface_w0p05` (surface_like pairs only) worsened frame FE (0.617) despite surface
  preservation focus — evidence that pair source alignment does not guarantee target alignment.
- `a4d_temporal_w0p05` raised all fv_* values uniformly, suggesting it adds calibration noise.
- The gap between `fv_frame_location` and `fv_surface` narrowed only marginally
  (0.030 at best vs the required 0.10 margin). Direction remained wrong: surface still above frame.

---

## 7. Why this is a transfer failure rather than an implementation failure

Evidence that implementation succeeded:
- Data generators produce correctly structured JSONL with all required label fields.
- `py_compile` and `--help` validation pass on all scripts.
- Internal pair ranking reached accuracy = 1.0 in A4e (a4d_oodmatched_w0p05).
- The `frame_violation_logit` head learns to distinguish controlled pairs correctly.
- Temporal/predicate comparator guards held at 1.0 in all configs — no catastrophic
  interference from the auxiliary loss.

Evidence of generalization failure:
- Despite achieving perfect within-batch pair ranking on controlled A4d pairs, the head
  does not assign higher frame_violation_prob to Stage15 frame_location/frame_role
  records than to Stage15 surface_control/temporal_erased records.
- The controlled within-pair contrast (surface_like vs frame_location_like) is learned,
  but the feature learned does not transfer to the Stage15 OOD within-pair contrast.
- Increasing loss weight (`w0p2`) reduces all fv_* values uniformly, suggesting the head
  is compressing frame_violation_prob toward a lower mean rather than learning a
  discriminative signal.

Root cause interpretation:
The dummy-backbone representation does not separate the surface_like and frame_location_like
records at the feature level in a way that generalizes to Stage15 siblings. The controlled
pair contrast (A4d) mirrors the Stage15 construction at the dataset structure level but
not at the representation level — the backbone sees different token sequences and cannot
learn a slot-level feature from the controlled text differences that would transfer to
Stage15's tighter, slot-specific within-pair differences.

This is a representational capacity / inductive bias problem that cannot be solved by
changing the pair construction logic alone, given the current dummy backbone.

---

## 8. Why Stage22-B remains rejected

Stage22-B was defined as a positive recovery gate: predict SUPPORT when
`preservation_boundary_prob` is high AND `frame_violation_prob` is low.

The gate requires that `frame_violation_prob` safely separates:
- Records that should predict SUPPORT (low frame_violation_prob)
- Records that are frame mismatches but incorrectly predicting SUPPORT (high frame_violation_prob)

Neither condition is met:
1. OOD ranking criterion not met: frame_location/frame_role records consistently receive
   lower or equal `frame_violation_prob` compared to surface_control/temporal_erased records.
   A gate on low frame_violation_prob would admit frame-mismatched records (wrong outcome).
2. Trade-off structure: all configs that reduce frame FE slightly also increase surface/temporal
   FNE, indicating the signal conflates frame-violation and non-frame differences rather than
   separating them.
3. No tested config achieves simultaneously: frame_location FE <= 0.40, frame_role FE <= 0.40,
   surface FNE maintained, and correct OOD ranking. The required combination does not exist
   in the A4c/A4e sweep.

Until a config achieves correct OOD ranking (frame mean > surface/temporal mean by >= 0.10)
without sacrificing surface/temporal preservation, Stage22-B gate implementation is blocked.

---

## 9. Limitations

1. **Dummy backbone only.** All A4 results used the dummy-backbone diagnostic setup. The dummy
   backbone lacks the representation capacity of a real Mamba encoder. It is plausible that a
   full-backbone run produces a different result — the dummy backbone cannot learn slot-level
   features from text differences. This stage does not prove real-backbone failure.

2. **Small pair count.** A4d produced only 20 pairs from 5 pair_ids. This is a small training
   signal. A larger controlled dataset or a real-backbone run with more pair_ids could change
   the outcome.

3. **Temporal-erased-like construction is approximate.** The `_erase_temporal_phrase` function
   removes the first weekday/month temporal phrase from evidence. This is a conservative pattern
   and may miss other temporal phrase structures, potentially under-representing the
   temporal_erased construction type.

4. **No slot-grounded supervision.** Controlled data lacks explicit slot fields (`slot_original`,
   `slot_perturbed`). Without slot-level grounding, the pair-contrastive loss cannot directly
   supervise the slot-level feature that Stage15 OOD tests. Route 1 (slot-based generation) from
   the A4a audit remains the highest-quality path but was blocked by missing slot fields.

5. **Margin hyperparameter not swept exhaustively.** Only `w0p05` (loss_weight=0.05) and `w0p2`
   (loss_weight=0.2) were tested. An exhaustive margin/weight grid could reveal better configs,
   though the uniform-compression behavior at `w0p2` suggests increasing weight is not the
   primary lever.

---

## 10. Recommended next step

Two options, in order of decreasing effort:

### Option A: Real-backbone sanity check

Run Stage22-A4e config `a4d_oodmatched_w0p05` on the real Mamba backbone (not dummy) with
the existing A4d pair data. If the real backbone achieves OOD ranking criterion with 20 A4d pairs,
the dummy-backbone negative result was a capacity artifact and Stage22-B design can continue.
If the real backbone also fails, the supervised design is exhausted under current data structure.

This is the minimum check before closing Stage22 as a negative result.

### Option B: Close Stage22 and pivot

Accept Stage22 as a negative diagnostic result:
- In-domain frame violation supervision (Stage22-A3): insufficient for OOD gating.
- Pair-contrastive supervision on controlled pairs (A4c, A4e): insufficient for OOD transfer
  under dummy backbone.
- Stage22-B positive recovery gate: rejected.

Pivot to a different mechanism for SUPPORT false-not-entitled recovery, such as:
- Evidence sufficiency head directly supervised on sufficiency_label field.
- Claim-claim consistency head (temporal comparator extension).
- Direct retrieval augmentation to surface slot-compatible evidence at inference time.
- Moving to a different intervention taxonomy that explicitly separates surface from frame
  at training time with real-backbone representations.

---

## References

| Artifact | Path |
|---|---|
| Stage22-A4a audit script | `scripts/audit_stage22a4_frame_ood_alignment.py` |
| Stage22-A4a alignment plan | `results/stage22a4_frame_ood_alignment_plan.md` |
| Stage22-A4b generator | `scripts/build_stage22a4_pair_contrastive_frame_data.py` |
| Stage22-A4b/A4c design plan | `results/stage22a4_pair_contrastive_plan.md` |
| Stage22-A4d OOD-matched builder | `scripts/build_stage22a4d_ood_matched_frame_pairs.py` |
| Stage22-A4d design plan | `results/stage22a4d_ood_matched_pair_plan.md` |
| Stage22-A3 boundary/frame head notes | `results/stage22a_boundary_head_notes.md` |
| Synthesis table | `results/stage22a4_final_synthesis_table.csv` |
