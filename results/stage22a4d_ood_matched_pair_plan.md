# Stage22-A4d: OOD-matched frame-vs-preservation pair dataset — design plan

**Date:** 2026-06-24
**Status:** Builder implemented; not yet run
**Depends on:** Stage22-A4c pair contrastive loss (train_controlled_v6b_minimal.py)
**Builder:** `scripts/build_stage22a4d_ood_matched_frame_pairs.py`

---

## 1. Motivation from A4c failure

Stage22-A4c added a pair contrastive ranking loss (`relu(margin − (frame_fvl − pres_fvl)).mean()`)
on pairs built by A4b/A4b2: preservation side = `none`/`paraphrase` siblings, frame side =
`location_swap`/`role_swap`/`entity_swap`/`event_swap`/`title_name_swap` siblings sharing the same
`pair_id` in controlled data.

A4c improved in-domain pair ranking accuracy (frame FVL > preservation FVL) but **failed the
Stage15 OOD frame ranking criterion**:

| OOD group | frame_violation_prob mean (A4c runs) |
|---|---|
| surface_control | ≥ frame_location / frame_role mean |
| temporal_erased | ≥ frame_location / frame_role mean |
| frame_location_mismatch | not ranked above surface/temporal |
| frame_role_mismatch | not ranked above surface/temporal |

The correct ranking (`frame_location/frame_role > surface_control/temporal_erased`) was not
achieved across any tested A4c config. The Stage22-B positive recovery gate remains rejected.

---

## 2. Why A4c pair loss failed

A4c pairs are structurally misaligned with the OOD failure the probe tests:

| Property | A4c pairs (A4b/A4b2 source) | Stage15 OOD structure |
|---|---|---|
| Preservation side | `none`/`paraphrase` from controlled data (different base pairs per pair_id) | `surface_control` = paraphrase of the **same** base pair as the frame sibling |
| Temporal-erased side | **not present** in A4b/A4c | `temporal_erased` = temporal phrase removed from the same base pair |
| Frame side | `location_swap`, `role_swap`, etc. (general frame swap siblings) | `frame_location_mismatch`, `frame_role_mismatch` — specifically **location** or **role** slot swapped |
| Within-pair overlap | Moderate — same pair_id but none/paraphrase vs swap | High — nearly identical text with one slot changed |

A4c's `none`/`paraphrase` anchors do not mirror the specific structure of `surface_control` or
`temporal_erased` that the probe uses. The model learns to rank general frame swaps over general
preservation, but the OOD probe's preservation variants are more similar to the frame variants than
the controlled data suggests.

Two specific gaps:
1. **No temporal-erased-like pairs** in A4c training. `temporal_erased` is one of the two OOD
   preservation groups and was never represented on the preservation side of A4c pairs.
2. **Frame side too broad.** A4c pairs include `entity_swap`/`event_swap`/`title_name_swap` on the
   frame side, which do not directly correspond to the Stage15 `frame_location`/`frame_role`
   groups. The model may learn frame features that do not transfer to the specific location/role
   slot change the probe targets.

---

## 3. A4d construction logic

A4d narrows the pair construction to better match the two OOD discrimination axes the probe tests.

### Preservation side

Two preservation-side construction types, each derived from a support-safe anchor:

**`surface_like_preservation`**
- Source: the best `paraphrase` support-safe sibling of a pair_id (fallback: `none`).
- No text transformation applied. Evidence is used as-is.
- Mirrors Stage15 `surface_control`: paraphrase of the base claim-evidence pair.

**`temporal_erased_like_preservation`**
- Source: any support-safe `none` or `paraphrase` sibling whose evidence contains a detectable
  temporal phrase (weekday or month name preceded by `during`/`in`/`on`).
- Transformation: remove the first such temporal phrase from evidence text only; do not touch claim.
- Text cleanup applied (collapsed whitespace, trailing ` .` repair).
- Mirrors Stage15 `temporal_erased`: temporal phrase removed from base pair evidence.
- Skipped for pair_ids where no temporal phrase is present in any support-safe evidence.

### Frame side

Two frame-side construction types, narrowed from A4b's five:

| Construction type | Source `intervention_type` | Stage15 analog |
|---|---|---|
| `frame_location_like` | `location_swap` | `frame_location_mismatch` |
| `frame_role_like` | `role_swap` | `frame_role_mismatch` |

Frame candidates must additionally satisfy:
- `final_label == "NOT_ENTITLED"` if present
- `primary_failure_type == "frame"` if present
- `sufficiency_label == 1` if present (evidence is still topically sufficient)

The `entity_swap`, `event_swap`, and `title_name_swap` frame types used in A4b/A4c are **excluded**
from A4d. A4c already trained on those — the gap is alignment to location/role specifically.

### Support-safe preservation anchor criteria

A preservation sibling is support-safe if:
- `intervention_type ∈ {none, paraphrase}`
- `final_label == "SUPPORT"`
- `frame_compatible_label == 1` (if present)
- `sufficiency_label == 1` (if present)
- `predicate_covered_label == 1` (if present)
- `polarity_label == "SUPPORT"` (if present)

If no support-safe sibling is available for a pair_id, that pair_id is skipped.

### Cartesian product

For each qualifying pair_id:
- Surface-like: 1 surface_pres anchor × all qualifying frame candidates (location_swap + role_swap)
- Temporal-erased-like: 1 temporal_erased evidence × all qualifying frame candidates (if temporal phrase found)

Optional `--max-pairs-per-pair-id N` caps expansion per construction type per pair_id.

---

## 4. Leakage constraints

| Constraint | Implementation |
|---|---|
| Stage15 OOD records not read | `data/stage15_slot_sensitivity_probe.jsonl` is never opened by this script |
| No Stage15 content in templates | Temporal erasure pattern derives from weekday/month lists in controlled script; not from Stage15 record text |
| No Stage15 group labels | Output fields `surface_like_preservation`, `temporal_erased_like_preservation`, `frame_location_like`, `frame_role_like` describe construction method, not OOD group membership |
| Explicit provenance | Every output record carries `source = "controlled_ood_matched_pair_builder"` and `leakage_note = "constructed_from_controlled_data_only_no_stage15_records"` |
| Audit trail | Each output record carries `preservation_source_id` and `frame_source_id` linking back to controlled data |
| Not a training set yet | Output is a diagnostic pair dataset pending OOD ranking validation |

The construction-type labels in A4d are **generic descriptions** of how a record was built:
`surface_like_preservation` means "a paraphrase evidence from controlled data, structurally similar
to how surface_control is built" — it does not assert that the record is or was a Stage15 eval
record, nor does it use any Stage15 identifier.

---

## 5. Output schema

Each output JSONL record:

| Field | Description |
|---|---|
| `contrastive_id` | Unique ID derived from pair_id, both construction types, and index |
| `pair_id` | Shared pair_id from controlled data |
| `claim` | Claim text from the preservation anchor |
| `preservation_evidence` | Evidence text (original or temporal-phrase-erased) |
| `frame_evidence` | Evidence text from the frame sibling |
| `preservation_construction_type` | `surface_like_preservation` or `temporal_erased_like_preservation` |
| `frame_construction_type` | `frame_location_like` or `frame_role_like` |
| `preservation_source_intervention_type` | `none` or `paraphrase` |
| `frame_source_intervention_type` | `location_swap` or `role_swap` |
| `preservation_final_label` | Gold final_label of preservation anchor |
| `frame_final_label` | Gold final_label of frame sibling |
| `preservation_should_score_low_frame_violation` | `true` |
| `frame_should_score_high_frame_violation` | `true` |
| `target` | `frame_more_violating_than_ood_matched_preservation` |
| `source` | `controlled_ood_matched_pair_builder` |
| `leakage_note` | `constructed_from_controlled_data_only_no_stage15_records` |
| `preservation_source_id` | Source record `id` from controlled data |
| `frame_source_id` | Source record `id` from controlled data |
| `frame_compatible_label` | Copied from preservation anchor (if present) |
| `sufficiency_label` | Copied from preservation anchor (if present) |
| `predicate_covered_label` | Copied from preservation anchor (if present) |
| `polarity_label` | Copied from preservation anchor (if present) |
| `primary_failure_type` | Copied from preservation anchor (if present) |
| `frame_frame_compatible_label` | Copied from frame sibling (if present) |
| `frame_sufficiency_label` | Copied from frame sibling (if present) |
| `frame_predicate_covered_label` | Copied from frame sibling (if present) |
| `frame_polarity_label` | Copied from frame sibling (if present) |
| `frame_primary_failure_type` | Copied from frame sibling (if present) |

The auxiliary label fields (`frame_compatible_label` etc.) are copied for compatibility with
`v5.encode_records()` and `_pair_record_to_virtual_records()` in the training script. They are used
only to satisfy the tensor constructor, not by the pair contrastive loss.

---

## 6. How A4d aligns better with Stage15 OOD ranking

The Stage15 probe tests whether the model assigns higher `frame_violation_prob` to
`frame_location_mismatch` / `frame_role_mismatch` than to `surface_control` / `temporal_erased`,
when these groups are derived from the same base claim.

A4d pairs map directly onto these four groups:

| A4d construction | Stage15 OOD analog | A4c had this? |
|---|---|---|
| `surface_like_preservation` | `surface_control` | Partially (none/paraphrase were present but not selected as surface-like specifically) |
| `temporal_erased_like_preservation` | `temporal_erased` | **No** — A4c had no temporal-erased-like preservation side |
| `frame_location_like` | `frame_location_mismatch` | Partially (location_swap was in A4c but mixed with entity/event/title) |
| `frame_role_like` | `frame_role_mismatch` | Partially (role_swap was in A4c but mixed with entity/event/title) |

Key improvements over A4c:
1. **Temporal-erased-like pairs added.** The model will see training signal where temporal-phrase-
   erased evidence should rank low on frame_violation_prob against location/role frame siblings.
2. **Frame side narrowed to location+role only.** Removes entity/event/title noise that may dilute
   the location/role-specific signal.
3. **Construction-type labels.** Downstream filtering by `preservation_construction_type` lets the
   trainer use surface-like and temporal-erased-like pairs as separate training subsets or
   combined.

These A4d pairs can be used with the existing A4c `--pair-contrastive-frame-data` argument
(the training script already handles the virtual record construction via
`_pair_record_to_virtual_records`). No training script modification is required for A4d.

---

## 7. Validation criteria

A4d output is accepted as a contributing training source if, after training with A4d pairs
(using the A4c pair contrastive loss), ALL of the following hold on the Stage15 OOD probe
(3-seed mean):

| Criterion | Threshold |
|---|---|
| frame_violation_prob: frame_location mean > surface_control mean | required |
| frame_violation_prob: frame_role mean > temporal_erased mean | required |
| Margin: frame mean − surface/temporal mean | ≥ 0.10 |
| frame_location FE | ≤ 0.40 |
| frame_role FE | ≤ 0.40 |
| surface_control FNE | Maintained or improved vs A4c baseline |
| temporal_erased FNE | Maintained or improved vs A4c baseline |
| temporal_mismatch FER | 0.000 (comparator guard preserved) |
| predicate_mismatch FER | 0.000 (comparator guard preserved) |

**Stage22-B positive recovery gate remains rejected** until all criteria pass.

---

## 8. Failure modes

### F1: No temporal phrase found in any support-safe evidence

If none of the support-safe anchors across all pair_ids have a temporal phrase matching
`(during|in|on) (weekday|month)`, no `temporal_erased_like_preservation` pairs are generated.
The output will contain only `surface_like_preservation` pairs — same as A4c in effect.

**Mitigation:** If temporal-erased-like count is zero, extend the erasure pattern with a broader
`during <N tokens>` regex. Report skipped pair_id count in summary.

### F2: No location_swap or role_swap records for a pair_id

If a pair_id's frame siblings are only `entity_swap` / `event_swap` / `title_name_swap`, the
pair_id is excluded from A4d output (those types are excluded in A4d by design).

**Mitigation:** For pair_ids that have only excluded frame types, A4c pairs (using the A4b
generator) are still available and can be combined with A4d output.

### F3: Surface-like pair is too lexically similar to the temporal-erased-like pair

If the same support-safe anchor's evidence both serves as the surface-like variant and is used to
create the temporal-erased-like variant (by removing one temporal phrase), the two preservation-
side evidence texts may differ by only a few tokens. This creates near-duplicate contrastive pairs
for a given pair_id, which may reduce training diversity without harm.

**Mitigation:** acceptable — the two variants supervise different margins in the probe structure.

### F4: Erasure changes semantics

If removing a temporal phrase makes the evidence logically insufficient (e.g. the temporal scope
was load-bearing for the SUPPORT label), the `temporal_erased_like_preservation` record is
technically not a SUPPORT-safe pair. The builder does not verify post-erasure semantics.

**Mitigation:** The pair contrastive loss does not depend on the preservation-side gold label for
the loss signal — it only requires that `frame_fvl > pres_fvl` in the learned head. A semantically
degraded preservation side still provides a valid training signal for frame_violation ranking
as long as the evidence is not itself a frame mismatch. The `primary_failure_type` field from the
source anchor (if present) provides a partial post-hoc check.

### F5: A4d pairs do not reach the required margin threshold

Even with better-aligned pairs, the pair contrastive loss may not close the gap to the ≥ 0.10
OOD ranking margin within the existing model architecture. If the margin remains < 0.10 after A4d
training, a structural change (e.g. slot-grounded head, separate location/role heads, or Route 1
slot-based generation) is required.

**Diagnostic:** compare `pair_contrastive_frame_margin_mean` at best dev epoch for A4c vs A4d runs.
If A4d margin is higher than A4c margin but OOD ranking still fails, the architecture is the
bottleneck. If A4d margin is the same or lower than A4c, the pair construction alignment was not
the bottleneck.

---

## 9. Next step after generation

1. Run `scripts/build_stage22a4d_ood_matched_frame_pairs.py` on controlled data and inspect the
   summary:
   - Check temporal-erased-like success count — if zero, extend erasure pattern.
   - Check frame construction type counts — confirm location_like and role_like are both present.
   - Inspect a few `temporal_erased_like_preservation` records to verify erasure quality.

2. Use the A4d output JSONL with the existing A4c training loss:
   - Pass A4d JSONL to `--pair-contrastive-frame-data`.
   - The A4c training script already handles A4d output via `_pair_record_to_virtual_records`
     (no `contrastive_use_case` field required — A4d records do not carry it; the training
     script's `load_pair_contrastive_jsonl` uses `use_case_filter="frame_violation_contrastive"`
     which filters by `contrastive_use_case` field; A4d records will be excluded by this filter).
   - **Action required:** either add `contrastive_use_case = "frame_violation_contrastive"` to A4d
     output records, or pass `--pair-contrastive-use-case all` to bypass the filter.
   - Preferred: add the field in the builder (minor update) so A4d records are picked up without
     changing the training CLI default.

3. Run a 3-seed mini diagnostic with A4d pairs and measure:
   - DEV `pair_contrastive_frame_accuracy` and `pair_contrastive_frame_margin_mean`
   - OOD ranking on Stage15 probe: frame_location/frame_role mean vs surface_control/temporal_erased mean
   - Compare to A4c baseline

4. Accept as precondition for Stage22-B gate only if OOD ranking criterion passes.
