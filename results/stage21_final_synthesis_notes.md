# Stage21 Final Synthesis Notes

## Executive Summary

Stage21 evaluated v6B minimal as a targeted OOD guard mechanism across five
experimental sub-stages (E3, F0, F1, G0, G1). The main outcome is a **partial
success**: targeted temporal and predicate comparator guards are verified and
mechanistically explained, but the model's broader entitlement calibration problem
— specifically the entanglement between positive SUPPORT preservation and
frame-sensitive non-entitlement -could not be resolved by any tested post-hoc
calibration approach.

Key numbers (mean across 3 seeds):

- temporal_mismatch FER (v6B): 0.000 (from ~0.230 at v5)
- predicate_mismatch FER (v6B): 0.000 (from ~0.203 at v5)
- surface_control FNE (v6B): 0.697 -severe over-rejection persists
- temporal_erased FNE (v6B): 0.787 -severe over-rejection persists
- frame_location FER (v6B): 0.333 -regression vs v5 (was ~0.250)
- frame_role FER (v6B): 0.350 -regression vs v5 (was ~0.200)

---

## Stage-by-Stage Evidence

### E3 -v6B OOD Evaluation vs v5 Baseline

**Targeted mismatch guard (accepted):**
v6B with temporal and predicate comparator flags reduces temporal_mismatch and
predicate_mismatch false-entitled rates to 0.000 and 0.000 respectively,
from ~0.230 and ~0.203 in v5. This holds consistently across all 3 seeds.

**SUPPORT preservation failure (partial / unresolved):**
surface_control FNE = 0.697 and temporal_erased FNE = 0.787.
These values improve marginally vs v5 but remain severe. v6B does not solve
positive SUPPORT preservation.

**Frame mismatch regression:**
frame_location FER = 0.333 and frame_role FER = 0.350, both worse than v5.
v6B comparator signalling is entangled with frame-sensitive detection.

**Evidence:** stage21_e3_bestdev_v5_vs_v6b_ood_3seed_mean.csv,
stage21_f0_ood_tradeoff_table.csv

---

### F1 -Comparator Ablation

**Mechanistic attribution (confirmed):**
no_flags breaks both guards: temporal_mismatch FER=0.283,
predicate_mismatch FER=0.257.
temporal_only restores temporal guard only (FER=0.000).
predicate_only restores predicate guard only (FER=0.000).
The guards are flag-specific, not caused by a global NOT_ENTITLED bias.

**Preservation flag-independence (confirmed):**
surface_control and temporal_erased FNE are identical across all four ablation modes
(current, no_flags, temporal_only, predicate_only). The preservation failure is
orthogonal to comparator flags and resides in the base entitlement boundary.

**Evidence:** stage21_f1_v6b_ood_ablation_3seed_summary.csv

---

### G0 -Global Unflagged NOT_ENTITLED Shift

**Partial preservation rescue:**
shift=0.25 reduces surface_control FNE to 0.133 (from 0.697).
temporal/predicate guards preserved at FER=0.000 across all shifts.

**Frame blow-up (rejected):**
shift=0.25: frame_location FER=0.900, frame_role FER=0.867.
Any shift that materially reduces SUPPORT over-rejection also causes severe
false-entitled regressions on frame-mismatch groups. Scalar calibration is
too blunt.

**Evidence:** stage21_g0_v6b_ne_shift_3seed_summary.csv

---

### G1 -Auxiliary-Score-Gated Selective NOT_ENTITLED Shift

**Structural guard preservation:**
Gate applies only to unflagged records; temporal/predicate guards are
structurally unaffected.

**Gate failure (rejected):**
No (gate, threshold, shift) triple passed the full safety criterion.
Preservation-improving configurations reduced surface/temporal-erased FNE but
caused large frame_location/frame_role FER regressions.
frame_prob, sufficiency_prob, and predicate_coverage_prob cannot cleanly separate
SUPPORT controls from frame-mismatch records at any tested threshold.

**Evidence:** stage21_g1_selective_gate_sweep_3seed_summary.csv,
stage21_g1_selective_gate_sweep_notes.md

---

## Accepted Claims

1. **v6B temporal/predicate comparators are effective targeted guards.**
   The temporal comparator eliminates temporal_mismatch false-entitled errors and
   the predicate comparator eliminates predicate_mismatch false-entitled errors,
   both verified across 3 seeds. The gains are zero-to-one in the targeted groups.

2. **F1 ablations support a mechanistic interpretation of comparator-specific gains.**
   Each comparator flag independently and selectively guards its target probe type.
   Removing both flags reverts both gains; removing one flag reverts only the
   corresponding guard. This rules out global NOT_ENTITLED bias as an explanation.

3. **Preservation failure is not fixed by simple scalar calibration.**
   Neither a global unflagged NE shift (G0) nor an auxiliary-score-gated selective
   shift (G1) can reduce SUPPORT over-rejection without simultaneously causing
   unacceptable false-entitled regressions on frame-mismatch groups. The failure
   is a structural entanglement, not a threshold artifact.

---

## Rejected Hypotheses

1. **Global unflagged NE shift is a safe solution (G0).**
   Rejected. At any shift value that meaningfully reduces surface_control or
   temporal_erased FNE, frame_location and/or frame_role FER exceeds 0.40.

2. **Auxiliary-score-only selective NE shift is a safe solution (G1).**
   Rejected. No (gate, threshold, shift) triple passed all five safety conditions.
   Auxiliary scores do not provide a discriminating boundary between SUPPORT
   controls and frame-mismatch records.

3. **Preservation failure is merely a thresholding artifact.**
   Rejected by F1. SUPPORT over-rejection on surface_control and temporal_erased
   is identical across all four comparator-flag ablation modes (current, no_flags,
   temporal_only, predicate_only), ruling out any comparator-level threshold as
   the cause. The failure is in the base entitlement decision boundary.

---

## Final Paper-Facing Interpretation

Stage21 separates **targeted comparator success** from **broader entitlement
calibration failure**. The v6B model can learn specific mismatch guards (temporal,
predicate) that selectively suppress false entitlement on their target probe types,
and those gains are mechanistically attributable to the corresponding comparator
flags. This is a positive and novel result.

However, **positive preservation** (correctly labelling SUPPORT records as SUPPORT)
and **frame-sensitive non-entitlement** (correctly labelling frame-mismatched records
as NOT_ENTITLED) remain entangled. The model assigns high NOT_ENTITLED logits to both
surface-control SUPPORT records and frame-mismatch NOT_ENTITLED records; no post-hoc
logit adjustment can safely separate them using only the auxiliary scores available
at the current architecture level.

This is a **useful negative result**: it demonstrates that post-hoc scalar
calibration on model-internal auxiliary probabilities is insufficient to solve the
preservation-vs-frame-mismatch boundary problem. The problem requires a mechanism
that directly models the distinction between these two record types during training
— motivating a **Stage22** approach that adds an explicit preservation-vs-frame
discriminative signal rather than relying on post-hoc shifting.

---

## Remaining Risks

1. **G1 seed JSON files are pending.** The G1 summary CSV is currently header-only
   (Kaggle outputs not yet committed). The G1 conclusion is based on single-seed
   analysis and static notes. If the 3-seed mean differs materially, the "zero
   passing rows" claim should be revisited.

2. **Frame regression cause not isolated.** It is not established whether the
   frame_location and frame_role FER regression in E3 is caused by the temporal
   comparator, the predicate comparator, or the interaction. F1 ablation only
   covered the four canonical flag modes; frame-group metrics were not the primary
   F1 focus.

3. **Shift values in G0 sweep are coarse.** Only shifts 0, 0.25, 0.5, 0.75, 1.0
   were tested. A finer grid near 0.1–0.2 might find a narrow operating point
   that passes safety criteria. G1 used finer shifts on a subset.

4. **Auxiliary probability calibration not verified.** If frame_prob or
   sufficiency_prob are themselves miscalibrated (biased or noisy), the G1 gate
   results may not generalise to other seeds or data distributions.

5. **v5 baseline comparison is 3-seed mean.** Seed-level variance in v5 was not
   tested in Stage21; the 3-seed mean is taken from Stage21-E3 best-dev evaluations
   which were run separately per model.

---

## Summary Table

9 rows in results/stage21_final_synthesis_table.csv covering stages:
E3 (4 rows), F1 (2 rows), G0 (1 row), G1 (1 row), stage21_synthesis (1 row).
