# Stage28-D Location-Boundary Specialist Design

## 1. Executive Summary

Stage28-D freezes the design rationale for a future location-boundary specialist module. It does not implement the module. Evidence from Stage27-H2F, Stage28-A, Stage28-B v2, and Stage28-C-lite is consolidated here to justify retaining `location_swap` as a valid diagnostic axis and to specify how a bounded location-boundary specialist should be structured when it is eventually implemented.

The Stage27 frozen baseline uses `product_power=0.90` with architecture `v7_hierarchical`. This baseline is preserved unchanged. Any specialist work is explicitly optional and must be ablatable by flag.

Stage28-D retains location_swap as a valid hard frame-boundary diagnostic axis and recommends a bounded location-boundary decision cap as the safest next specialist design, while preserving product_power=0.90 as the frozen Stage27 baseline.

---

## 2. Why Location Swap Is Retained

`time_swap` was removed from the training dataset because it introduced corrupted examples that could not be cleanly labeled. `location_swap` is not removed for the same reason.

Stage28-C-lite manually inspected 20 `location_swap` examples from the Stage28-B v2 dataset. All 20 were classified as `clean_invalid_entailment`:

- Subject, predicate, object, and time are preserved across the pair.
- Only the event location changes.
- `NOT_ENTITLED` is the correct label for all inspected examples.
- No ambiguous, invalid artifact, or unknown cases were found in the 20-example sample.

This stands in direct contrast to `time_swap`, which was removed due to systematic label corruption. A 20-example sanity sample is not a full audit of all 300 `location_swap` records, and a larger audit may be warranted in future stages. However, immediate removal is not justified by current evidence.

---

## 3. Evidence Chain

The following stages contribute to Stage28-D:

| Stage | Key Finding |
|---|---|
| Stage27 final | `product_power=0.90`, `v7_hierarchical`, controlled no-time validation |
| Stage27-H2F | `location_role` dominates product false SUPPORT (33 cases); learned/hybrid are locally informative but unstable |
| Stage28-A | Location is the dominant sub-axis within `location_role` (27 location vs. 6 role false SUPPORT cases) |
| Stage28-B v2 | 1994-record diagnostic dataset constructed; parser issue fixed; `location_swap` has 300 records with 68.3% source-ID control coverage |
| Stage28-C-lite | 20/20 inspected `location_swap` examples are `clean_invalid_entailment`; clean rate = 1.00 |

---

## 4. Stage27 Final Baseline

| Parameter | Value |
|---|---|
| config | product_power=0.90 |
| architecture | v7_hierarchical |
| decision signal | product |
| entitlement formula | (frame_prob × predicate_coverage_prob × sufficiency_prob)^0.90 |
| dataset | data/controlled_v5_v3_without_time_swap.jsonl |
| validation scope | controlled no-time only |
| frozen encoder | yes (T4-safe) |
| max_length | 64 |
| H1 final decision | enabled with --v7-use-v6b-style-final-decision |

This baseline is the reference point for all Stage28-D candidate designs. No candidate design in this report changes the default behavior of this baseline.

### Stage27-H2F Gate-Axis Decomposition Summary

- `learned_has_local_signal`: true
- `learned_unstable`: true
- `hybrid_has_local_gain`: true
- `hybrid_not_final`: true
- `hard_axis`: location_role

Product axis false SUPPORT totals:

| Axis | Count |
|---|---|
| location_role | 33 |
| predicate | 3 |
| missing | 0 |
| other | 8 |

Interpretation: the remaining bottleneck is location/role frame mismatch. Missing evidence rejection is fully controlled. Predicate false SUPPORT is comparatively small.

### Stage28-A Location/Role Error Anatomy

| Metric | Value |
|---|---|
| location_SUPPORT_total | 27 |
| role_SUPPORT_total | 6 |
| location_role_SUPPORT_total | 33 |
| location_role_balance | 4.5 |
| location_harder_than_role | true |
| role_harder_than_location | false |

Decision at Stage28-A: do not build specialist gate yet; first construct the diagnostic dataset (Stage28-B).

---

## 5. Stage28-B/C Validity Evidence

### Stage28-B v2 Dataset

| Field | Value |
|---|---|
| total_input_records | 3600 |
| total_output_records | 1994 |
| location_swap_audit bucket | 300 |
| role_swap_reference bucket | 300 |
| location_controls bucket | 300 |
| role_controls bucket | 194 |
| predicate_contrast bucket | 300 |
| missing_evidence_contrast bucket | 300 |
| other_frame_contrast bucket | 300 |

Normalized label counts (output):

| Label | Count |
|---|---|
| NOT_ENTITLED | 1500 |
| REFUTE | 248 |
| SUPPORT | 246 |

Source-ID overlap stats:

| Metric | Value |
|---|---|
| n_location_sources | 300 |
| n_role_sources | 300 |
| n_location_control_matches | 205 |
| n_role_control_matches | 157 |
| location_control_coverage | 0.6833 |
| role_control_coverage | 0.5233 |

The Stage28-B v2 parser issue (where `derived_source_id_from_suffix` and `derived_intervention_from_suffix` were being used incorrectly) has been resolved. All 3600 records carry explicit `pair_id` and `final_label`.

### Stage28-C-lite Sanity Audit

| Metric | Value |
|---|---|
| inspected examples | 20 |
| clean_invalid_entailment | 20 |
| ambiguous | 0 |
| invalid_artifact | 0 |
| unknown | 0 |
| clean_rate | 1.00 |

All inspected cases preserve subject, predicate, object, and time. Only location changes. `NOT_ENTITLED` is valid for all 20 inspected examples.

---

## 6. Failure Mode Definition

The target failure mode for the location-boundary specialist is:

> The model incorrectly produces a SUPPORT decision for a claim where the event location described in the claim does not match the source document, while all other semantic content (subject, predicate, object, time) is preserved.

This is distinct from:

- Missing evidence failure: source does not mention the relevant fact at all.
- Predicate failure: what happened is described differently.
- Role failure: who performed or received the action is described differently.
- Time failure: when the event occurred is described differently.

Location false SUPPORT (27 cases in Stage28-A) is the primary unresolved failure mode at the product axis after Stage27.

---

## 7. Design Goals

The Stage28-D specialist design, when eventually implemented, must achieve the following:

1. Reduce `location_swap` false SUPPORT cases at the product axis.
2. Preserve missing evidence rejection (missing SUPPORT must remain 0).
3. Preserve predicate rejection (predicate SUPPORT must not increase materially).
4. Avoid increasing false SUPPORT on non-location axes (role, predicate, missing, other).
5. Keep `product_power=0.90` behavior unchanged by default (specialist is disabled by default).
6. Make the specialist optional and ablatable via an explicit flag.

---

## 8. Non-Goals

The following are explicitly out of scope for Stage28-D and the specialist module it describes:

- No out-of-distribution (OOD) generalization claim.
- No `time_swap` claim (time_swap has already been removed from training).
- No default replacement of the product entitlement gate.
- No learned-only entitlement gate as the primary decision mechanism.
- No broad geography or world-knowledge module.
- No claim that all 300 `location_swap` examples in Stage28-B v2 are clean.
- No claim that the specialist has been validated in any model training run.

---

## 9. Candidate Specialist Designs

Four candidate designs are evaluated below.

### Design 1: Location-Boundary Decision Cap (Preferred)

The base product entitlement is preserved. An optional `location_boundary_prob` signal is used only to cap entitlement downward for location-sensitive examples.

Hard cap:
```
capped_entitlement = min(base_entitlement, location_boundary_prob)
```

Soft cap:
```
capped_entitlement = base_entitlement * (location_boundary_prob ** gamma)
```

| | |
|---|---|
| Pro | Directly targets location false SUPPORT |
| Pro | Preserves product baseline; cannot increase entitlement |
| Pro | Optional; disabled by default |
| Con | Requires a reliable `location_boundary_prob` signal to be constructed |
| Status | **Preferred** |

### Design 2: Factorized Frame Subheads

`frame_prob` would be factorized into subheads: `entity_frame`, `role_frame`, `location_frame`, `event_frame`, `title_frame`. The product/min of these becomes the new `frame_prob`.

| | |
|---|---|
| Pro | Architecture-level interpretability |
| Con | Larger model patch; requires additional labeled data |
| Status | Future Stage29 candidate |

### Design 3: Residual Learned Location Correction

A learned residual correction is applied only for location cases, using the local learned signal observed in H2F.

| | |
|---|---|
| Pro | Uses observed local learned signal |
| Con | Stage27-H2E showed naive residual can worsen aggregate performance |
| Status | Not preferred unless bounded by a cap |

### Design 4: Dataset-Only Split

Partition `location_swap` into `clean_location_swap`, `ambiguous_location_swap`, and `invalid_location_swap` buckets. No model change.

| | |
|---|---|
| Pro | Safest option; no model change required |
| Con | Does not reduce false SUPPORT on its own |
| Status | Required supporting step for any specialist design; not the final solution |

---

## 10. Preferred Design: Location-Boundary Decision Cap

The preferred design is the **location-boundary decision cap**.

Base entitlement (unchanged from Stage27):
```
base_entitlement = (frame_prob * predicate_coverage_prob * sufficiency_prob) ** 0.90
```

Hard cap (when location_boundary_prob is available):
```
capped_entitlement = min(base_entitlement, location_boundary_prob)
```

Soft cap alternative:
```
capped_entitlement = base_entitlement * (location_boundary_prob ** gamma)
```

Final decision remains product-style; learned-only entitlement is not used as the final gate.

**Why preferred**: Stage27-H2F showed that learned and hybrid branches have local signal but are unstable. A bounded cap uses the signal directionally (only to reduce over-entitlement) without allowing learned instability to increase false SUPPORT on other axes. The hard cap has a provable safety property: it cannot raise entitlement above the product baseline.

**Activation policy**: optional flag only; disabled by default. The default `product_power=0.90` behavior is unchanged.

**Safety property**: the cap cannot increase entitlement above the base product entitlement.

---

## 11. Required Prediction Export Before Implementation

Before any specialist can be trained or evaluated, the following prediction fields must be stably exportable from the Stage27 model:

| Field | Purpose |
|---|---|
| stable example IDs | Enable overlap analysis across runs |
| source_id / pair_id | Link test examples to audit buckets |
| normalized gold label | NOT_ENTITLED / REFUTE / SUPPORT |
| normalized pred label | NOT_ENTITLED / REFUTE / SUPPORT |
| intervention type | location_swap / role_swap / control / etc. |
| frame_prob | Component of product entitlement |
| predicate_coverage_prob | Component of product entitlement |
| sufficiency_prob | Component of product entitlement |
| entitlement_for_decision | Final numeric entitlement value |
| final logits / scores | Raw model output |
| per-axis prediction records | Learned and hybrid branch outputs |

This export capability is the deliverable for Stage28-E. Without it, per-axis overlap analysis and false-SUPPORT attribution cannot be performed reliably.

---

## 12. Evaluation Plan

When the specialist is implemented (Stage28-F), evaluation must compare:

- Stage27 `product_power=0.90` baseline (frozen)
- Stage28-F location cap model (specialist enabled by flag)

**Primary metric**: `location_swap` SUPPORT count reduction.

**Secondary metrics**:

| Metric | Notes |
|---|---|
| macro-F1 | Must not drop materially |
| NE recall | Must remain stable |
| SUPPORT recall | Must not collapse |
| role_swap SUPPORT | Must not increase |
| predicate_swap SUPPORT | Must not increase materially |
| missing_evidence SUPPORT | Must remain 0 |
| controls SUPPORT/REFUTE | Preservation check |

**Success criteria**:

1. `location_swap` SUPPORT count decreases relative to Stage27 baseline.
2. `missing_evidence` SUPPORT remains 0.
3. `predicate_swap` SUPPORT does not increase materially.
4. Macro-F1 does not drop materially.
5. SUPPORT recall does not collapse.

---

## 13. Decision

| Decision | Value |
|---|---|
| Retain location_swap | Yes |
| Remove location_swap (like time_swap) | No |
| Implement specialist gate now (Stage28-D) | No |
| Design specialist gate | Yes (this document) |
| Preferred specialist type | Location-boundary decision cap |
| Default product_power=0.90 changed | No |
| Stage28-E deliverable | Prediction export infrastructure |
| Stage28-F deliverable | Minimal location-boundary specialist candidate (disabled by default) |

---

## 14. Remaining Risks

| Risk | Mitigation |
|---|---|
| 20-example sanity audit is not a full audit of all 300 location_swap records | Stage28-E or Stage28-F should complete a larger manual audit before training the specialist |
| location_boundary_prob signal does not yet exist | Must be constructed; design depends on feature availability |
| Soft cap gamma hyperparameter requires tuning | Ablation on controlled diagnostic set; do not tune on held-out |
| Learned local signal is unstable (H2F) | Cap design prevents learned instability from raising entitlement |
| Controlled no-time setting may not generalize | Specialist evaluation is scoped to controlled setting; no OOD claim is made |
| max_length=64 may truncate location evidence | Known constraint; not addressed in Stage28-D |
| Role_swap still has 6 false SUPPORT cases unaddressed | Out of scope for location-boundary specialist; flagged as future work |

---

## 15. Next Stage Recommendation

**Stage28-E**: Implement diagnostic prediction export infrastructure.

Deliverables:
- Stable example IDs in all exported prediction records.
- `source_id` and `pair_id` linkable to Stage28-B v2 audit buckets.
- Normalized gold and pred labels in export.
- Intervention type recorded per example.
- All product component probabilities exported (`frame_prob`, `predicate_coverage_prob`, `sufficiency_prob`, `entitlement_for_decision`).
- Per-axis prediction records for learned and hybrid branches.

**Stage28-F**: Implement minimal location-boundary specialist candidate.

Deliverables:
- Optional `location_boundary_head` or `location_frame_head` (disabled by default flag).
- Trained and evaluated only on controlled diagnostic setting.
- Compared against frozen Stage27 `product_power=0.90` baseline.
- No change to default behavior.

---

*Stage28-D retains location_swap as a valid hard frame-boundary diagnostic axis and recommends a bounded location-boundary decision cap as the safest next specialist design, while preserving product_power=0.90 as the frozen Stage27 baseline.*
