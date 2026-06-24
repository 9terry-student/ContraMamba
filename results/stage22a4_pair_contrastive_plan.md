# Stage22-A4b/A4b2: pair-group contrastive frame dataset — design plan

**Date:** 2026-06-24
**Status:** A4b generator implemented; A4b2 suitability refinement applied
**Depends on:** Stage22-A4a audit (results/stage22a4_frame_ood_alignment_plan.md)

---

## 1. Motivation

Stage22-A/A3 established that:

- The `preservation_boundary_head` (A2) and `frame_violation_head` (A3) both learn
  in-domain intervention targets but fail to generalize to Stage15 OOD frame mismatch.
- OOD ranking is inverted: `surface_control` and `temporal_erased` receive
  `frame_violation_prob` as high as or higher than `frame_location_mismatch` /
  `frame_role_mismatch`.
- The root cause is that the in-domain training pairs frame positives and preservation
  negatives across different base examples, while Stage15 OOD derives frame and
  preservation siblings from the **same** base claim-evidence pair.
- A design-level intervention is required: the head must see within-pair contrast
  during training to learn the discrimination the OOD probe tests.

Stage22-A4a confirmed that the controlled data contains sufficient structure:
`pair_id` presence is 1.0 and all 10 pair_ids contain both frame and preservation
sibling records. No explicit slot fields are available, so pair_group_based
generation is the primary route.

---

## 2. Why pair-group contrastive supervision is the next step

The key structural difference between in-domain training and Stage15 OOD is:

| Setting | Frame example origin | Preservation example origin |
|---|---|---|
| Controlled training | Different base pairs for different intervention types | Different base pairs |
| Stage15 OOD | Same base pair (e.g. `stage15_frame_location__orion_approval__id`) | Same base pair |

In controlled training the head can distinguish frame from preservation by base-pair
lexical features alone, without learning the within-pair slot-change contrast.
At OOD evaluation it sees same-pair siblings and the base-pair lexical shortcut no
longer works.

Pair-group contrastive training forces the head to solve the harder within-pair
discrimination during training, which mirrors the OOD evaluation structure.

---

## 3. Construction logic

**Input:** Controlled JSONL (`controlled_v5_v3_without_time_swap.jsonl`)

**Grouping:** Group records by `pair_id`.

**Sibling classification:**

| Class | `intervention_type` values |
|---|---|
| Preservation candidates | `none`, `paraphrase` |
| Frame candidates | `entity_swap`, `event_swap`, `location_swap`, `role_swap`, `title_name_swap` |
| Ignored (not used) | `time_swap`, `predicate_swap`, `evidence_deletion`, `evidence_truncation`, `irrelevant_evidence`, `polarity_flip` |

**Pair construction:** For each pair_id with at least one preservation member and
at least one frame member, construct the Cartesian product of
(preservation sibling, frame sibling) pairs. Each pair becomes one output record.

Optional `--max-pairs-per-pair-id N` caps the Cartesian product per pair_id to
avoid over-representation of large groups.

**Canonical claim:** The preservation sibling's `claim` is used as the shared
claim text (`claim` field). The frame sibling's `claim` text is identical for
same-pair_id records in the controlled dataset (same base claim).

---

## 4. Leakage constraints

| Constraint | Implementation |
|---|---|
| Stage15 OOD not read | Script does not import, read, or reference Stage15 data |
| Stage15 not used in template design | Pair construction uses only `pair_id` / `intervention_type` — no OOD schema consulted |
| No OOD group labels added | Output records do not contain `stage15_probe_type`, `probe_type`, or `ood_group` |
| Explicit provenance | Every output record carries `leakage_note = "constructed_from_controlled_data_only"` and `source = "controlled_pair_group"` |
| Not called a training set yet | Output is labelled a diagnostic contrastive dataset pending OOD ranking validation |

---

## 5. Output schema

Each output JSONL record contains:

| Field | Description |
|---|---|
| `contrastive_id` | Unique ID derived from pair_id and both intervention types |
| `pair_id` | Shared pair_id of the two siblings |
| `claim` | Shared claim text (from preservation sibling) |
| `preservation_evidence` | Evidence text from the preservation sibling |
| `frame_evidence` | Evidence text from the frame sibling |
| `preservation_intervention_type` | `none` or `paraphrase` |
| `frame_intervention_type` | `entity_swap`, `event_swap`, `location_swap`, `role_swap`, or `title_name_swap` |
| `preservation_final_label` | Gold label of the preservation sibling |
| `frame_final_label` | Gold label of the frame sibling |
| `target` | `"frame_more_violating_than_preservation"` |
| `preservation_should_be_safe` | `true` — preservation sibling should have low frame_violation_prob |
| `frame_should_be_blocked` | `true` — frame sibling should have high frame_violation_prob |
| `preservation_is_non_frame_anchor` | bool — pres side is a valid non-frame anchor (frame_compatible_label check; does NOT require SUPPORT label) |
| `preservation_is_support_safe_anchor` | bool — pres side passes ALL conservative SUPPORT-safe constraints |
| `frame_is_frame_violation` | bool — frame side has final_label=NOT_ENTITLED and primary_failure_type=frame (where present) |
| `contrastive_use_case` | `"support_safe_frame_contrastive"` \| `"frame_violation_contrastive"` \| `"audit_only"` |
| `source` | `"controlled_pair_group"` |
| `leakage_note` | `"constructed_from_controlled_data_only"` |
| `preservation_source_id` | Source record `id` for the preservation sibling |
| `frame_source_id` | Source record `id` for the frame sibling |
| `frame_compatible_label` | Copied from preservation sibling (optional) |
| `sufficiency_label` | Copied from preservation sibling (optional) |
| `predicate_covered_label` | Copied from preservation sibling (optional) |
| `polarity_label` | Copied from preservation sibling (optional) |
| `primary_failure_type` | Copied from preservation sibling (optional) |
| `frame_frame_compatible_label` | Frame sibling value (optional) |
| `frame_sufficiency_label` | Frame sibling value (optional) |
| `frame_predicate_covered_label` | Frame sibling value (optional) |
| `frame_polarity_label` | Frame sibling value (optional) |
| `frame_primary_failure_type` | Frame sibling value (optional) |

### A4b2 refinement note

`preservation_intervention_type ∈ {none, paraphrase}` does NOT guarantee
`preservation_final_label = SUPPORT`. Controlled records with `intervention_type = none`
may have `polarity_label = REFUTE` or `final_label = REFUTE` (e.g. records where the
original claim-evidence pair is refuting). These are valid non-frame anchors for
frame-vs-non-frame contrastive diagnostics but must NOT be used as SUPPORT-positive
anchors for SUPPORT recovery calibration.

The `contrastive_use_case` field provides a safe downstream filter:
- Use `frame_violation_contrastive` for frame-vs-non-frame contrastive training.
- Use `support_safe_frame_contrastive` only for any future SUPPORT recovery gate work.

---

## 6. How this differs from Stage22-A3 frame_violation_head

| Aspect | Stage22-A3 frame_violation_head | Stage22-A4b pair-group contrastive |
|---|---|---|
| Supervision structure | Single-record BCE label from intervention_type | Within-pair (preservation, frame) sibling contrast |
| Positive/negative pairing | Cross-pair (different base pairs) | Same-pair_id (same base claim) |
| OOD alignment | Trained on different base pairs; OOD tests same-pair siblings | Trained on same-pair siblings; matches OOD structure |
| Training signal | Binary: is this record a frame violation? | Pairwise: does frame sibling score higher than preservation sibling? |
| Stage22-B gate status | Rejected — inverted OOD ranking | Not yet gated — pending OOD ranking validation |
| Data leakage | None (in-domain only) | None (in-domain only, no Stage15 records) |

The pair-group contrastive dataset provides the training signal that A3 lacked:
explicit within-pair contrast that mirrors the evaluation structure.

---

## 7. Validation criteria

The contrastive dataset is accepted as a training source for a new head if, after
training a head on it (A4b training step, not yet implemented), ALL of the following
hold on the Stage15 OOD probe (3-seed mean):

| Criterion | Threshold |
|---|---|
| frame_violation_prob: frame_location mean > surface_control mean | required |
| frame_violation_prob: frame_role mean > temporal_erased mean | required |
| Margin: frame mean − surface/temporal mean | ≥ 0.10 |
| frame_location FE | ≤ 0.40 |
| frame_role FE | ≤ 0.40 |
| surface_control FNE | maintained or improved vs Stage22-A3 baseline |
| temporal_erased FNE | maintained or improved vs Stage22-A3 baseline |
| temporal_mismatch FER | 0.000 (comparator guard preserved) |
| predicate_mismatch FER | 0.000 (comparator guard preserved) |

Until all criteria pass, Stage22-B positive recovery gate remains rejected.

---

## 8. Failure modes

### F1: All pairs skipped — no mixed pair_ids

If every pair_id contains only frame records or only preservation records,
the generator produces zero output pairs. The Stage22-A4a audit confirmed
this is not the case for the current controlled data (all 10 pair_ids have
both sides), but the generator reports skipped pair_ids and reasons explicitly.

**Mitigation:** If future controlled data extensions drop preservation records
for some pair_ids, add targeted generation to restore the mixed structure.

### F2: Cartesian product is too large

If pair_ids contain many frame and preservation records each, the Cartesian
product can be very large (N_frame × N_preservation per pair_id). Use
`--max-pairs-per-pair-id` to cap the expansion.

**Default:** No limit (0). Recommended: inspect summary counts before training.

### F3: Preservation claim ≠ frame claim (claim drift)

Controlled data may have slightly different `claim` text for same-pair_id records
if claims were paraphrased per intervention. The generator uses the preservation
sibling's claim as canonical and provides both `preservation_evidence` and
`frame_evidence`. If the head receives only one claim and two evidence fields,
any claim-text drift is isolated to the evidence comparison — which is the
correct structure for this contrastive task.

**Mitigation:** If claim drift is detected in audit, add a `frame_claim` field
to the output schema for explicit tracking.

### F4: Within-pair contrast is too easy in controlled data

Controlled frame-sibling evidence texts may differ from preservation-sibling
evidence by more than Stage15 OOD siblings differ (which are generated by more
targeted slot substitution). If the head learns surface-form shortcuts from
controlled pairs that do not generalize to Stage15's tighter differences,
OOD ranking will still fail.

**Diagnostic:** After A4b training, compute AUC of frame_violation_prob for
(frame_sibling, preservation_sibling) pairs within Stage15. If AUC < 0.70,
controlled-pair shortcut learning is likely.

**Mitigation:** Add a contrastive data augmentation step that reduces non-slot
text differences between frame and preservation siblings (e.g. replace only the
swapped slot token in the frame evidence, keeping all other tokens identical to
the preservation evidence). This would require explicit slot fields — see Route 1
in `stage22a4_frame_ood_alignment_plan.md`.

### F5: Pairwise loss weight interferes with main task losses

The pairwise ranking or margin loss on the new head may share parameters with
the existing `frame_violation_head` BCE loss. If both losses are active, they
may conflict or produce gradient cancellation for frame-type records that appear
as positives in BCE but as the "more violating" side of contrastive pairs.

**Mitigation:** Treat the pairwise contrastive loss and the BCE frame violation
loss as separate auxiliary losses with independent weights; monitor both BCE
validation accuracy and contrastive pair accuracy during training.

---

## 9. Recommended next step after generation

1. Run `scripts/build_stage22a4_pair_contrastive_frame_data.py` and inspect the
   summary — verify pair counts by frame_intervention_type and that no pair_ids
   are skipped.
2. ~~Design the A4b pairwise training head~~ — **Done (Stage22-A4c)**: see below.
3. Run a 3-seed mini diagnostic with the pairwise head and measure:
   - DEV pair ranking accuracy (`pair_contrastive_frame_accuracy` in training log)
   - DEV pair mean margin (`pair_contrastive_frame_margin_mean`)
   - OOD ranking: frame_location / frame_role `frame_violation_prob` mean vs
     surface_control / temporal_erased mean
4. Accept as Stage22-B gate precondition only if OOD ranking criterion passes
   (frame mean > surface mean by ≥ 0.10).

---

## Stage22-A4c update (pair contrastive loss implemented)

The pair contrastive frame ranking loss is now implemented in
`scripts/train_controlled_v6b_minimal.py`.

### New CLI args

| Arg | Default | Description |
|---|---|---|
| `--pair-contrastive-frame-data` | None | Path to pair contrastive JSONL (controlled data only) |
| `--use-pair-contrastive-frame-loss` | False | Enable the pair contrastive ranking loss |
| `--pair-contrastive-frame-loss-weight` | 0.0 | Weight added to total training loss |
| `--pair-contrastive-frame-margin` | 0.2 | Margin δ in relu(δ − (frame_fvl − pres_fvl)).mean() |
| `--pair-contrastive-use-case` | `frame_violation_contrastive` | Filter by contrastive_use_case |

### Loss definition

For each pair record, two separate forward passes (no flags) produce
`frame_violation_logit` for the preservation-evidence input and the frame-evidence input.

```
pc_loss = relu(margin - (frame_fvl - pres_fvl)).mean()
total_loss += pc_loss_weight * pc_loss
```

`output["logits"]` and predictions are never touched. The ranking loss supervises
only `frame_violation_logit` via the `frame_violation_head` parameters.

### Eval metrics (logged every epoch, captured at best dev epoch)

| Metric | Description |
|---|---|
| `pair_contrastive_frame_accuracy` | Fraction where frame_fvl > pres_fvl |
| `pair_contrastive_frame_margin_mean` | Mean(frame_fvl − pres_fvl) |
| `pair_contrastive_frame_loss` | relu(margin − margin_mean).mean() at eval |
| `pair_contrastive_frame_mean_pres_fv_prob` | Mean sigmoid(pres_fvl) |
| `pair_contrastive_frame_mean_frame_fv_prob` | Mean sigmoid(frame_fvl) |
| `pair_contrastive_frame_valid_count` | Number of pair records used |

### Leakage contract

- Pair contrastive data must have `leakage_note = "constructed_from_controlled_data_only"`.
- Stage15 OOD records are never loaded in this path.
- No OOD group names or labels are used in the loss or forward pass.
- All behavior is disabled by default (requires `--use-pair-contrastive-frame-loss`).
