# Stage22 Design: Preservation Boundary Auxiliary Head

**Version:** 2026-06-24
**Status:** Design only -- no model changes in this note
**Motivated by:** Stage21 final synthesis (results/stage21_final_synthesis_notes.md)

---

## 1. Executive Summary

Stage21 established that v6B temporal and predicate comparators are effective, verified,
and mechanistically explained targeted guards (Stage21-E3/F1 accepted claims). However,
no post-hoc logit calibration approach -- global unflagged NE shift (G0) or
auxiliary-score-gated selective shift (G1) -- could safely reduce SUPPORT
over-rejection on `surface_control` and `temporal_erased` without causing
false-entitled regressions on `frame_location_mismatch` and `frame_role_mismatch`.

The Stage21-G1 diagnostic identified the root cause: the existing model-internal
auxiliary scores (`frame_prob`, `sufficiency_prob`, `predicate_coverage_prob`) do not
discriminate preservation-like records from frame-mismatch records because they were
not explicitly trained to do so. They are trained on auxiliary label alignment (frame
compatibility, predicate coverage, sufficiency), not on the preservation-vs-frame-mismatch
boundary.

**Stage22 goal:** Add a minimal supervised auxiliary head -- the `preservation_boundary_head`
-- trained with explicit supervision from training-data `intervention_type` labels to
distinguish preservation-like records (surface change, temporal erasure) from structural
frame-mismatch records (location swap, role swap). Use this head at inference time to
gate a conservative NOT_ENTITLED logit depression on unflagged records, replacing the
failed unsupervised gate from G1.

---

## 2. Stage21 Diagnosis

### What worked

| Claim | Evidence |
|---|---|
| v6B temporal comparator eliminates temporal_mismatch FER | E3: FER 0.230 -> 0.000 (3-seed mean) |
| v6B predicate comparator eliminates predicate_mismatch FER | E3: FER 0.203 -> 0.000 (3-seed mean) |
| Gains are flag-specific, not global NE bias | F1: no_flags reverts both guards; temporal_only / predicate_only are selective |

### What failed

| Problem | Evidence |
|---|---|
| surface_control FNE = 0.697 (v6B) | Severe SUPPORT over-rejection persists |
| temporal_erased FNE = 0.787 (v6B) | Severe SUPPORT over-rejection persists |
| Preservation failure is flag-independent | F1: FNE identical across all 4 ablation modes |
| G0 global NE shift: frame blow-up at any useful shift | shift=0.25: surface_control FNE -> 0.133 but frame_location FER -> 0.900 |
| G1 auxiliary-score gate: zero configurations pass safety | Auxiliary scores cannot separate preservation from frame-mismatch |

### Root cause

The preservation-vs-frame-mismatch boundary is **not encoded** in any existing auxiliary
signal. All of `frame_prob`, `sufficiency_prob`, and `predicate_coverage_prob` carry
useful intermediate information but were not trained to distinguish:

- **Preservation-like records**: surface/temporal-erased controls that change the form
  of a claim without invalidating its logical entitlement relationship with the evidence
- **Frame-mismatch records**: location/role swaps that change which frame the claim
  refers to, making NOT_ENTITLED the correct label

Both groups currently land in the model's high-NOT_ENTITLED logit region because the
base entitlement decision boundary penalises any form of structural uncertainty.

---

## 3. Why Post-Hoc NE Shifting Is Rejected

G0 and G1 applied a scalar shift to the NOT_ENTITLED final logit at inference time,
without any training-time supervision on the preservation boundary. This approach fails
for a structural reason:

**The model has no learned representation that separates the two groups at any tested
threshold.** The failure is not a hyperparameter problem (G0 was coarse, G1 swept a
fine grid); it is an absence of a discriminating learned feature. Adding a scalar shift
at inference cannot create a feature that was never trained.

Furthermore:

1. **G0** applied the shift to all unflagged records globally -- too broad, no
   discrimination possible.
2. **G1** conditioned on existing auxiliary probabilities -- the existing probabilities
   were not trained for this discrimination task, so no threshold combination passed
   the safety criterion.
3. **Any further scalar shifting** on top of unsupervised auxiliary scores will continue
   to fail for the same reason: the unsupervised scores do not carry the required signal.

The solution must be **training-time**, adding a supervised signal that forces the model
to encode the preservation-vs-frame-mismatch boundary in a dedicated head.

---

## 4. Candidate Stage22 Mechanisms

### 4A. Boundary Auxiliary Head (Recommended)

**Description:**
Add a new binary classification head `preservation_boundary_head` that takes the
concatenated slot representations already computed by the forward pass
(`frame_pair_repr`, `predicate_pair_repr`, `sufficiency_repr`) and outputs
`preservation_prob` in [0, 1].

Train it with Binary Cross-Entropy using supervision derived from `intervention_type`
labels in the training data:
- `preservation_label = 1`: surface-preserving interventions (surface paraphrase,
  temporal erasure, no-change baseline where gold = SUPPORT/REFUTE)
- `preservation_label = 0`: structural frame/role/predicate mismatches
  (frame_location_mismatch, frame_role_mismatch, predicate_mismatch, temporal_mismatch)
- Records with ambiguous or missing `intervention_type`: masked out of the BCE loss

At inference: use `preservation_prob` as the gate discriminator in a conservative
positive override -- apply a NOT_ENTITLED logit depression only when:
- `preservation_prob >= beta_high` (e.g. 0.80)
- `temporal_flag == 0` AND `predicate_flag == 0` (same unflagged condition as G0/G1)

**Why recommended:** Directly supervised on the target discrimination. Minimal
architecture change. Preserves final-logits contract. Does not require dual logit
paths or product-protected routing.

---

### 4B. Pairwise Contrastive Preservation Loss

**Description:**
For pairs where one is a preservation-like record and the other is a frame-mismatch
record derived from the same base claim-evidence pair, push apart their
representation-space encodings with a contrastive margin loss.

**Pros:** Could create a richer representation separation than a binary head.

**Cons:**
- Requires matched pairs in the training data (preservation-control vs frame-mismatch
  of the same base pair). The current controlled dataset may not have guaranteed pairs.
- Significantly more complex to implement (pair matching, batching).
- Adds loss routing complexity without clear benefit over option 4A.

**Verdict:** Deferred. Revisit if 4A's head fails to discriminate the boundary.

---

### 4C. Conservative Positive Override Gate (inference-time only)

**Description:**
Apply a conditional SUPPORT/NOT_ENTITLED logit flip at inference when the model's
existing outputs suggest a preservation-like record, using a conjunction of multiple
existing auxiliary thresholds: e.g. `frame_prob >= t1 AND sufficiency_prob >= t2 AND
predicate_coverage_prob >= t3` AND unflagged.

**Pros:** No training changes; immediate testable variant.

**Cons:** This is precisely what Stage21-G1 tested. G1 showed that no conjunction of
thresholds over the existing auxiliary probabilities passes the safety criterion. This
mechanism is already exhausted.

**Verdict:** Rejected -- already falsified by Stage21-G1.

---

### 4D. Two-Stage Calibration (Composite)

**Description:**
Stage A: existing v6B comparator guard (unchanged). Stage B: preservation boundary gate
applied to unflagged records, using a learned head from option 4A.

**Relation to 4A:** This is the inference-time structure of 4A. Stage22 is effectively
a two-stage model: (temporal/predicate guard) + (preservation gate for unflagged records).
The two stages do not interact -- temporal/predicate flags remain exclusive from the
preservation gate's domain.

**Verdict:** This is the recommended Stage22 design structure. It is the combination
of keeping v6B Stage21 and adding the 4A training-time mechanism.

---

## 5. Recommended Stage22 Mechanism

**Name:** Supervised Preservation Boundary Head + Conservative Unflagged Gate

**Core idea:**
1. During training: learn `preservation_prob` from `intervention_type` supervision.
   Add `preservation_boundary_loss` (BCE) to the combined training loss.
2. At evaluation/inference: use `preservation_prob` as the gate for a conservative
   NOT_ENTITLED logit depression on unflagged records, replacing the failed
   unsupervised auxiliary gates from G1.

### How it differs from G0/G1

| Aspect | G0 | G1 | Stage22 |
|---|---|---|---|
| Gate type | None (global) | Unsupervised auxiliary threshold | Supervised learned head |
| Training change | None | None | Yes: new auxiliary loss |
| Gate signal source | Always | frame_prob / sufficiency_prob | New: preservation_prob |
| Gate supervision | None | None | intervention_type labels |
| Preservation-vs-frame discrimination | No | Cannot (unsupervised) | Explicitly trained |
| Final-logits contract | Preserved | Preserved | Preserved |

### Why this is not product-protected loss routing

- Product-protected routing (v6A approach) multiplied logits from two separate prediction
  pathways, adding a second logit head and combining outputs.
- Stage22 has **one logit path** (`output["logits"]`).
- The preservation head produces a scalar in [0, 1], not a logit vector. It is used
  to **condition** a post-forward-pass gate adjustment, not to replace or blend logits.
- The CE/pairwise/intervention losses continue to use `output["logits"]` unchanged.
- `base_logits` remains diagnostic only.
- There are no `loss_logits` or `pairwise_logits`.

### How the final-logits contract is maintained

**During training:**
- `preservation_boundary_loss` is added to the training loss.
- CE/frame/predicate/sufficiency/polarity losses all use existing tensors unchanged.
- `output["logits"]` = current final calibrated logits (temporal/predicate modulated).
- The preservation head outputs are NOT injected back into `output["logits"]` during
  training, to avoid interfering with CE loss gradients.

**At inference (OOD evaluation):**
- After the forward pass, if `--ood-preservation-gate` is set:
  - Identify unflagged records: `temporal_flag == 0 AND predicate_flag == 0`
  - For unflagged records where `preservation_prob >= beta_high`:
    - Subtract `delta_ne` from NOT_ENTITLED final logit (post-hoc, copy-on-write)
  - Recompute predictions from adjusted logits
- This is the same eval-only gate as G0/G1, but conditioned on a supervised head

This separation maintains the training-time final-logits contract fully: losses always
see `output["logits"]` without gate adjustment.

### How temporal/predicate comparator behavior is preserved

- The comparator alphas (`alpha_temporal`, `alpha_predicate`) and their modulation of
  flagged records are entirely unchanged.
- The preservation gate applies only to **unflagged** records (temporal_flag == 0 AND
  predicate_flag == 0). Flagged records are not touched.
- The `preservation_boundary_head` is trained on all records (including flagged ones)
  with appropriate labels, but its gate only activates for unflagged records at inference.
- There is no interaction between the comparator modulation path and the preservation
  gate path.

### OOD group names -- eval-only constraint

The preservation gate at inference uses ONLY model outputs (`preservation_prob`,
`temporal_flag`, `predicate_flag`) to decide whether to apply the shift. It does not
use `stage15_probe_type`, intervention_type, gold labels, or any other group
identifier to select records for the gate.

OOD group names (`temporal_mismatch`, `surface_control`, etc.) are used ONLY for
post-hoc reporting of metrics after predictions are made.

---

## 6. Implementation Plan

The implementation touches exactly two files:

### File 1: `src/contramamba/modeling_v6b_minimal.py`

**Changes:**

1. Add `PreservationBoundaryHead` to the constructor:
   ```python
   self.preservation_boundary_head = nn.Linear(
       frame_size + predicate_size + sufficiency_size,  # concatenated slot repr
       1,  # scalar logit
   )
   ```

2. In `forward()`, after computing slot representations:
   ```python
   preservation_logit = self.preservation_boundary_head(
       torch.cat([
           frame["frame_pair_repr"],
           predicate["predicate_pair_repr"],
           sufficiency["sufficiency_repr"],
       ], dim=-1)
   )
   preservation_prob = torch.sigmoid(preservation_logit).squeeze(-1)
   ```

3. Compute preservation loss when `preservation_labels` is provided:
   ```python
   if preservation_labels is not None and preservation_mask is not None:
       active = preservation_mask.bool()
       if torch.any(active):
           losses["preservation_loss"] = F.binary_cross_entropy_with_logits(
               preservation_logit.squeeze(-1)[active],
               preservation_labels[active].float(),
           )
   ```

4. Add `preservation_logit`, `preservation_prob` to the return dict.
   Do not inject into `output["logits"]` or `output["base_logits"]`.

**What must NOT change:**
- `output["logits"]` = `final_logits` (temporal/predicate modulated, unchanged)
- `output["base_logits"]` = diagnostic only, unchanged
- CE loss uses `output["logits"]`, unchanged
- All comparator alpha logic unchanged

### File 2: `scripts/train_controlled_v6b_minimal.py` (training side)

**Changes:**

1. Derive `preservation_labels` and `preservation_mask` from `intervention_type`
   in the record encoding step (new helper function `encode_preservation_labels`).

2. Pass `preservation_labels` and `preservation_mask` to the forward call during
   training.

3. Include `preservation_loss` in the weighted training loss sum.

4. Add `--preservation-loss-weight` CLI arg (float, default `1.0`) to control
   the contribution of the preservation head to the total loss.

### File 3: `scripts/train_controlled_v6b_minimal.py` (inference side)

**Changes:**

5. Add `--ood-preservation-gate` CLI flag (boolean) that activates the
   supervised gate at OOD evaluation time.

6. Add `--ood-preservation-gate-threshold` CLI arg (float, default `0.8`).

7. Add `--ood-preservation-gate-shift` CLI arg (float, default `0.5`).

8. In OOD evaluation: after forward pass, if gate is active, apply conservative
   preservation-gated NE logit depression using `preservation_prob` from the output.

9. Report `preservation_prob` distribution per OOD group in the output JSON.

### Preservation label derivation

```
intervention_type -> preservation_label:
  "no_change", "baseline", "original"     -> 1 (preservation-positive)
  "surface_paraphrase", "surface_control" -> 1 (preservation-positive)
  "temporal_erased"                       -> 1 (preservation-positive)
  "temporal_mismatch"                     -> 0 (mismatch-negative)
  "predicate_mismatch"                    -> 0 (mismatch-negative)
  "frame_location_mismatch"               -> 0 (mismatch-negative)
  "frame_role_mismatch"                   -> 0 (mismatch-negative)
  all others / missing                    -> masked (excluded from BCE loss)
```

The exact mapping must be confirmed against the training data's `intervention_type`
vocabulary before implementation.

---

## 7. Validation Plan

Reuse Stage15 OOD probe (`data/stage15_slot_sensitivity_probe.jsonl`).
Compare three conditions:
- **Baseline (v6B Stage21 E3)**: existing v6B, no preservation gate
- **Stage22 head-only**: v6B + preservation head trained, no gate at inference
- **Stage22 head + gate**: v6B + preservation head + conservative unflagged gate

OOD flag source: `stage15_probe_type` (same as E3).

### Required metric conditions (all must hold simultaneously)

| Metric | Condition | Rationale |
|---|---|---|
| temporal_mismatch FER | == 0.000 (mean) | Comparator guard must be preserved |
| predicate_mismatch FER | == 0.000 (mean) | Comparator guard must be preserved |
| surface_control FNE | Decreases meaningfully vs Stage21 E3 (< 0.55) | Primary preservation target |
| temporal_erased FNE | Decreases meaningfully vs Stage21 E3 (< 0.65) | Primary preservation target |
| frame_location_mismatch FER | <= 0.40 (mean) | Safety: no frame regression |
| frame_role_mismatch FER | <= 0.40 (mean) | Safety: no frame regression |
| sufficiency_control FER | <= 0.15 (mean) | Safety: no sufficiency regression |

Seeds: 3 seeds minimum; conditions evaluated on mean across seeds.

### Diagnostic metrics (not pass/fail; informational)

- `preservation_prob` distribution per OOD group (does the head discriminate?)
- `selected_rate_among_unflagged` at the chosen threshold
- Distribution of `preservation_prob` for `surface_control` vs `frame_location_mismatch`
  -- a good head should have clearly separated distributions for these two groups

---

## 8. Expected Success Criteria

**Strong success** (all required metric conditions pass + clear discrimination):
- `preservation_prob` is high for `surface_control`/`temporal_erased` (mean > 0.7)
- `preservation_prob` is low for `frame_location_mismatch`/`frame_role_mismatch` (mean < 0.4)
- OOD metric conditions all pass

**Partial success** (discrimination exists but gate threshold must be tuned):
- `preservation_prob` distributions are separated but threshold region overlaps
- Some required metric conditions pass but frame safety (0.40) narrowly violated
- Next step: tighter gate threshold or additional loss weighting

**Failure** (Stage22 does not resolve the boundary):
- `preservation_prob` fails to discriminate the two groups even after training
- This would imply the slot representations (`frame_pair_repr`, `predicate_pair_repr`,
  `sufficiency_repr`) do not contain information sufficient to classify the boundary
- Next step: add a dedicated span-level feature (e.g. explicitly comparing frame
  slot filler tokens between claim and evidence)

---

## 9. Failure Modes

### 9A. Slot representations carry insufficient discriminating information

If `frame_pair_repr`, `predicate_pair_repr`, and `sufficiency_repr` do not encode the
distinction between surface-form interventions and frame-slot swaps, the
`preservation_boundary_head` will not converge to a useful discriminator regardless
of loss weight or training duration.

**Diagnostic:** Check BCE validation loss convergence. If it plateaus near the random
baseline (~0.693), the representations are uninformative for this task.

**Mitigation:** Explore adding the raw token state mean (claim vs evidence) as an
additional input to the preservation head, giving it access to more local lexical
information.

### 9B. Training data distribution mismatch

If the training data (`controlled_v5_seed_no_time_swap.jsonl`) does not contain
balanced examples of both preservation-positive and frame-mismatch intervention types,
the head will be biased toward the majority class.

**Mitigation:** Apply class-weighted BCE loss or oversample the minority class.
Report class balance in the Stage22 experiment log.

### 9C. Gate threshold collapses to same failure mode as G1

Even with a supervised head, if the decision boundary is not sharp, the same
tradeoff will appear: at any threshold that catches enough `surface_control` records,
some `frame_location_mismatch` records are also included.

**Diagnostic:** Plot preservation_prob histogram for all OOD groups. Measure AUC of
preservation_prob as a discriminator between `surface_control` and
`frame_location_mismatch`. If AUC < 0.70, the head is insufficient.

**Mitigation:** Add a second head output (`frame_mismatch_prob`) trained specifically
to recognise frame-slot swaps; use the conjunction `preservation_prob >= beta AND
frame_mismatch_prob < beta_fm` as the gate condition.

### 9D. Preservation loss interferes with comparator guard

If the preservation BCE loss weight is too high, it may distort the slot
representations used by the comparator guard, degrading temporal/predicate guard
performance.

**Mitigation:** Monitor `temporal_mismatch FER` and `predicate_mismatch FER` on dev
set during training. If they degrade, reduce `--preservation-loss-weight` or freeze
the comparator alpha parameters during the preservation head warm-up phase.

### 9E. Preservation gate does not activate sufficiently

If the training distribution has few preservation-positive records in the unflagged
region, `preservation_prob` may be consistently low for all unflagged OOD records
(because the model learns that unflagged records are generally not preservation-positive).

**Mitigation:** Confirm that the training data contains unflagged preservation-positive
records. If not, the training flag source may need adjustment to leave some
preservation-positive records in the unflagged region.

---

## 10. Minimal File-Touch Plan

Stage22 implementation should touch exactly these files:

| File | Change type | Change description |
|---|---|---|
| `src/contramamba/modeling_v6b_minimal.py` | Architecture | Add `preservation_boundary_head`, `preservation_logit`, `preservation_prob` output; add `preservation_loss` |
| `scripts/train_controlled_v6b_minimal.py` | Training | Add `encode_preservation_labels()` helper; pass labels to forward; add `preservation_loss` to combined loss; add `--preservation-loss-weight` CLI arg |
| `scripts/train_controlled_v6b_minimal.py` | Inference | Add `--ood-preservation-gate`, `--ood-preservation-gate-threshold`, `--ood-preservation-gate-shift` CLI args; apply gate in OOD eval |
| `scripts/summarize_stage22_*.py` | New script | Summary and reporting scripts for Stage22 OOD results |
| `results/stage22_*.json` | New result | Per-seed OOD result JSONs (Kaggle outputs) |

**Do not touch:**
- `src/contramamba/heads.py` (existing head implementations)
- `scripts/train_controlled_v5.py` (v5 base script)
- `data/` (no data changes)
- Stage21 result CSVs or notes
- Any Stage21 training or evaluation logic

### Gate implementation note

The gate at inference time follows the same pattern as the G0/G1 sweep code already
in `train_controlled_v6b_minimal.py` (`_apply_ne_shift_and_eval`). The new code
replaces the hardcoded `unflagged_mask` with:

```python
unflagged_mask = (t_cpu == 0) & (p_cpu == 0)
gate_mask = unflagged_mask & (preservation_prob_cpu >= args.ood_preservation_gate_threshold)
```

and calls `_apply_ne_shift_and_eval` with `gate_mask` and `args.ood_preservation_gate_shift`.
This reuses the existing metric computation infrastructure with no new metric logic.

---

## Appendix: Stage21 Numbers Used in This Design

All numbers from `results/stage21_final_synthesis_notes.md` (3-seed means):

| Metric | v5 baseline | v6B Stage21 | Target for Stage22 |
|---|---|---|---|
| temporal_mismatch FER | 0.230 | 0.000 | 0.000 (maintain) |
| predicate_mismatch FER | 0.203 | 0.000 | 0.000 (maintain) |
| surface_control FNE | 0.797 | 0.697 | < 0.55 (meaningful reduction) |
| temporal_erased FNE | 0.830 | 0.787 | < 0.65 (meaningful reduction) |
| frame_location FER | 0.250 | 0.333 | <= 0.40 (safety) |
| frame_role FER | 0.200 | 0.350 | <= 0.40 (safety) |
| sufficiency_control FER | 0.087 | 0.110 | <= 0.15 (safety) |

The Stage21 G0 shift=0.25 numbers illustrate the tradeoff this design must solve:
- surface_control FNE: 0.697 -> 0.133 (strong improvement)
- frame_location FER: 0.333 -> 0.900 (catastrophic regression)

Stage22 succeeds if and only if both rows move in the desired direction simultaneously.
