# Stage22-A4a: OOD-aligned frame supervision — design plan

**Date:** 2026-06-24
**Status:** Design / audit phase — no model or training code changes
**Depends on:** Stage22-A/A2/A3 diagnostic results, Stage15 OOD probe schema

---

## 1. Motivation from Stage22-A/A3

Stage22-A2 showed that the single `preservation_boundary_head` cannot separate
preservation-like records (surface_control, temporal_erased) from frame-mismatch
records (frame_location_mismatch, frame_role_mismatch) at OOD evaluation time.
The boundary_prob signal rises for both groups under increasing loss weight, so
no threshold separates them safely.

Stage22-A3 added a separate `frame_violation_head` and tested whether explicit
in-domain frame supervision resolves the OOD ranking problem. Key findings:

| Finding | Evidence |
|---|---|
| Head learns in-domain frame targets | DEV gap rises to 0.221–0.376 under supervision |
| OOD ranking is inverted | surface_control / temporal_erased frame_violation_prob ≥ frame_location / frame_role |
| Both-head configs worsen frame FE | frame_location FE rises to 0.633–0.717 vs 0.517 for frame_w0p05 |
| Stage22-B gate rejected | No safe (frame_violation_prob, boundary_prob) threshold combination found |

The in-domain frame violation supervision is insufficient for safe OOD discrimination.
A design-level intervention is required before any logit gate.

---

## 2. Why the existing in-domain frame_violation_head is insufficient

The `frame_violation_head` is trained on controlled records labelled with
`intervention_type` ∈ {entity_swap, event_swap, location_swap, role_swap,
title_name_swap} as positives and {none, paraphrase, evidence_deletion,
evidence_truncation, irrelevant_evidence, polarity_flip} as negatives.

This mapping covers the controlled training distribution but does not cover the
Stage15 OOD frame mismatch structure:

| Stage15 OOD group | Structural source | Controlled analog |
|---|---|---|
| frame_location_mismatch | location/entity slot swapped in original claim | location_swap, entity_swap |
| frame_role_mismatch | role/event slot swapped in original claim | role_swap, event_swap |
| surface_control | surface paraphrase, no slot change | none, paraphrase |
| temporal_erased | temporal phrase erased from claim | not present in training |

The failure occurs because Stage15 OOD `surface_control` and `temporal_erased`
records are generated from the same source claims as `frame_location_mismatch` /
`frame_role_mismatch`, so they share lexical and structural overlap. The model
cannot distinguish them via a head trained only on the controlled label space.

Specifically:
- In controlled data, `none` / `paraphrase` records come from a different base
  distribution than `location_swap` / `role_swap`.
- In Stage15 OOD, `surface_control` and `frame_location_mismatch` are derived
  from the **same** base claim-evidence pair, making their representations far
  more similar than the controlled positive/negative pairs are to each other.
- The frame_violation_head learns a feature that fires on controlled frame types
  but that feature does not separate within-pair OOD siblings.

In-domain frame intervention supervision can be kept as a diagnostic or
regularization term. It is not sufficient as an OOD safety gate.

---

## 3. What OOD-aligned frame supervision should capture

An OOD-aligned frame supervision signal must encode the following contrast:

> Given a base claim-evidence pair P, a frame-mismatched variant P' (where a
> frame slot — location, role, entity, event — is swapped to an incompatible
> value) should score higher on `frame_violation_prob` than a surface-preserved
> variant P'' (where the claim is paraphrased without changing the frame slot
> or where the temporal phrase is removed).

This requires the head to discriminate within-pair siblings, not across-pair
negatives. The current training does not provide within-pair contrast.

Required properties of an OOD-aligned frame signal:

1. **Within-pair contrastive structure**: frame-mismatch and preservation variants
   of the same base pair must both appear as training examples.
2. **Slot-level grounding**: the head must see which frame slot changed (location,
   role, entity, event) and whether the change crosses a compatibility boundary.
3. **Separation from surface form**: surface paraphrase should consistently score
   low on frame_violation_prob, even when the base pair's lexical overlap with
   a frame-mismatch variant is high.

---

## 4. Data leakage constraints

The following constraints are non-negotiable:

| Constraint | Reason |
|---|---|
| Do not train on `data/stage15_slot_sensitivity_probe.jsonl` | Stage15 is the evaluation set; training on it inflates OOD metrics |
| Do not use Stage15 record content to derive templates | Template rules must come from the controlled intervention_type taxonomy |
| Do not copy Stage15 record structure into controlled data | Any new synthetic records must be independently generated |
| Stage15 may be used as schema reference only | Auditing field names and group vocabulary is permitted |

All candidate generation routes must draw exclusively from the controlled
training data or from a separate synthetic generator that does not inspect
Stage15 record content.

---

## 5. Candidate train-side generation strategies

The A4a audit script (`scripts/audit_stage22a4_frame_ood_alignment.py`) will
determine which of the following routes is feasible after running on the
controlled data. The routes are listed in decreasing order of alignment quality.

### Route 1: Direct slot-based generation (highest quality)

**Precondition:** Controlled records contain explicit slot fields such as
`slot_type`, `slot_original`, `slot_perturbed`, or separate `original_location` /
`perturbed_location`, etc.

**Method:** For each controlled record with a frame intervention type, reconstruct
a preservation sibling by re-inserting the `slot_original` value into the claim,
creating a surface-match control. Train a contrastive loss on (frame-mismatch,
surface-match-control) pairs derived from the same base record.

**Alignment quality:** High — the contrast is exact within-pair and mirrors the
Stage15 OOD construction.

### Route 2: Pair_id group-based contrastive supervision (medium quality)

**Precondition:** Controlled records share `pair_id` values across intervention
types. Records with the same `pair_id` and different `intervention_type` form
implicit siblings.

**Method:** Group controlled records by `pair_id`. For each group, identify
frame-intervention members (location_swap, role_swap, etc.) and preservation
members (none, paraphrase). Treat these as (positive, negative) contrastive
pairs and train a pairwise margin loss or binary head on same-pair siblings.

**Alignment quality:** Medium — the within-pair structure is preserved but the
slot-level change is implicit (must be inferred from claim text difference).

### Route 3: Text-template fallback (lowest quality, supplemental only)

**Precondition:** Controlled records contain only `claim`, `evidence`, and
`intervention_type` (no explicit slot fields or pair_id grouping).

**Method:** Apply rule-based or NER-guided slot identification to controlled
frame-intervention records. Generate a surface-match sibling by reinserting
the identified slot value. This route has higher error rate and should only
supplement Routes 1 or 2.

**Alignment quality:** Low — template errors may introduce spurious frame signals.

---

## 6. Recommended first implementation path after audit

Run `scripts/audit_stage22a4_frame_ood_alignment.py` to determine which route
is feasible. Based on expected controlled data structure:

**If Route 2 (pair_id-based) is confirmed feasible:**

1. Implement a `FrameMismatchContrastiveHead` that takes a (frame-member,
   preservation-member) pair from the same pair_id and applies a margin loss
   pushing their `frame_violation_prob` outputs apart.
2. Keep the existing `frame_violation_head` BCE loss as an auxiliary regularizer.
3. The contrastive head is trained only on within-pair siblings — no OOD records.
4. Do not gate until the OOD ranking criterion (§7) is met.

**If Route 1 (slot-based) is confirmed feasible:**

Prefer Route 1 — construct explicit (slot_original, slot_perturbed) pairs from
the slot fields and train a slot-compatibility head that compares claim-side and
evidence-side slot filler representations directly.

**In either case:**
- In-domain BCE `frame_violation_head` supervision is kept as a warmup/regularizer.
- No logit gate is added in A4b until the ranking criterion is met.

---

## 7. Validation criteria for A4b

A4b implementation is accepted only if ALL of the following hold on the Stage15
OOD probe (3-seed mean):

| Criterion | Threshold | Rationale |
|---|---|---|
| frame_violation_prob: frame_location mean | > surface_control mean | OOD ranking must be correct |
| frame_violation_prob: frame_role mean | > temporal_erased mean | OOD ranking must be correct |
| frame_location FE | ≤ 0.40 | Safety: no frame regression |
| frame_role FE | ≤ 0.40 | Safety: no frame regression |
| surface_control FNE | Maintained or improved vs Stage22-A3 baseline | No preservation regression |
| temporal_erased FNE | Maintained or improved vs Stage22-A3 baseline | No preservation regression |
| temporal_mismatch FER | 0.000 | Comparator guard must be preserved |
| predicate_mismatch FER | 0.000 | Comparator guard must be preserved |

**Stage22-B positive recovery gate remains rejected** until all OOD ranking
criteria pass with a safe margin (frame mean > surface mean by ≥ 0.10).

---

## 8. Failure modes

### F1: Controlled data lacks within-pair siblings

If no pair_id groups contain both frame-intervention and preservation members,
Route 2 is blocked. Fallback: extend the controlled dataset with a synthetic
generator that explicitly creates matched pairs (does not require Stage15 data).

### F2: Slot fields are absent

If no explicit slot fields exist in controlled records, Route 1 is blocked and
Route 2 must rely on claim-text diffing to recover the slot change. Text diffing
is error-prone for paraphrase-type preservation records.

Mitigation: add a slot extraction preprocessing step using NER on claim text
for location, role, entity, event slots. Validate extraction recall on records
where intervention_type is known.

### F3: Contrastive loss weight interferes with main task

If the contrastive head's loss weight is too high, it may distort the shared
slot representations and degrade the main classification or comparator guards.

Mitigation: monitor temporal_mismatch FER and predicate_mismatch FER during
training; reduce contrastive weight or freeze comparator alpha parameters if
either degrades.

### F4: Within-pair contrast is not sufficient (Stage15 siblings are harder)

Even if Route 2 is implemented correctly, Stage15 OOD pairs may exhibit a
harder within-pair contrast than the controlled pairs (e.g. more lexical overlap
between frame_location and surface_control siblings in Stage15 than in controlled).

Diagnostic: after A4b training, compute AUC of frame_violation_prob for
(frame_mismatch vs surface_control) pairs within Stage15. If AUC < 0.70,
the learned representation is insufficient and Route 1 (slot-based) must be
added even if Route 2 was the primary path.

### F5: OOD alignment requires explicit location / role slot supervision

The Stage15 frame_location and frame_role groups are defined by a specific
slot type (location vs role). If the frame_violation_head conflates these two
types, it may fail to rank location-group records above preservation records
in one group while passing the other.

Mitigation: if F5 is diagnosed, split the frame_violation_head into two
separate heads — `frame_location_violation_head` and `frame_role_violation_head`
— each supervised on the corresponding intervention type subset.

---

## Core conclusion

1. **In-domain frame intervention supervision** (current `frame_violation_head`
   BCE on `location_swap`, `role_swap`, etc.) can be kept as a diagnostic
   regularizer. It is not sufficient as an OOD gating signal.

2. **OOD frame mismatch requires separate aligned supervision** — specifically,
   within-pair contrastive training that mirrors the Stage15 OOD construction
   of (frame-mismatch, preservation) siblings from the same base pair.

3. **Stage22-B positive recovery gate remains rejected** until a frame mismatch
   safety signal generalizes to Stage15 frame_location_mismatch and
   frame_role_mismatch groups with correct OOD ranking
   (frame groups score higher than surface_control / temporal_erased).
