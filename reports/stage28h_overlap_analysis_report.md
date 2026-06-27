# Stage28-H Overlap Analysis Report

## 1. Executive Summary

Stage28-H freezes the conclusions from Stage28-F (full enriched prediction export) and Stage28-G (example-level overlap analysis). Across nine runs — product_p0p90 seeds 1–3, learned seeds 1–3, and hybrid_beta0p20 seeds 1–3 — the overlap analysis demonstrates that learned-only and product_learned_residual hybrid signals are not reliable correction mechanisms for product false SUPPORT errors.

product_p0p90 remains the frozen Stage27/Stage28 baseline with the lowest aggregate false SUPPORT count (44). The learned signal introduces 94 new false SUPPORT errors while rescuing only 2 product false SUPPORT errors. The hybrid_beta0p20 signal introduces 39 new false SUPPORT errors while rescuing only 6. Neither mechanism consistently reduces location-axis false SUPPORT errors, which account for 27 of 44 product false SUPPORT cases and represent the dominant unresolved axis.

Stage28-G rejects learned and hybrid residuals as reliable correction mechanisms: they rescue too few product false SUPPORT errors while introducing substantially more new false SUPPORT errors. The next specialist should use an independent bounded location-boundary cap, not learned/hybrid residual mixing.

---

## 2. Inputs and Scope

**Architecture:** v7_hierarchical  
**Backbone:** mamba  
**Dataset:** controlled_v5_v3_without_time_swap.jsonl  
**Schema:** stage28e_v1  
**max_length:** 64  
**freeze_encoder:** true  
**freeze_a_log:** true  
**H1 final decision:** enabled  
**output_source:** best_dev  
**Validation type:** controlled no-time validation only  

**Runs included:**

| Run | Decision Signal | Product Power | Hybrid Beta | Seeds |
|---|---|---|---|---|
| product_p0p90 | product | 0.90 | 0.25 | 1, 2, 3 |
| learned | learned | 1.0 | 0.25 | 1, 2, 3 |
| hybrid_beta0p20 | product_learned_residual | 0.90 | 0.20 | 1, 2, 3 |

**Stage27 frozen baseline:** product_p0p90 (product signal, product_power=0.90). This baseline is unchanged by Stage28-H.

**Scope of this report:** Analysis and reporting only. No model code, training code, dataset files, or scripts are modified.

---

## 3. Stage28-F Full Export Summary

Stage28-F generated enriched prediction JSONs for all nine runs (3 signals × 3 seeds), each with 720 records.

### Per-run results

| Run | Seed | Signal | pred_SUPPORT | pred_NE | pred_REFUTE | false_SUPPORT | loc_FS | role_FS | pred_FS | missing_FS |
|---|---|---|---|---|---|---|---|---|---|---|
| learned_seed1 | 1 | learned | 180 | 450 | 90 | 90 | 41 | 30 | 3 | 0 |
| learned_seed2 | 2 | learned | 107 | 499 | 114 | 23 | 12 | 5 | 3 | 0 |
| learned_seed3 | 3 | learned | 111 | 519 | 90 | 23 | 13 | 3 | 2 | 0 |
| product_p0p90_seed1 | 1 | product | 121 | 509 | 90 | 31 | 19 | 3 | 2 | 0 |
| product_p0p90_seed2 | 2 | product | 103 | 528 | 89 | 12 | 7 | 3 | 1 | 0 |
| product_p0p90_seed3 | 3 | product | 66 | 564 | 90 | 1 | 1 | 0 | 0 | 0 |
| hybrid_beta0p20_seed1 | 1 | product_learned_residual | 116 | 514 | 90 | 27 | 18 | 2 | 2 | 0 |
| hybrid_beta0p20_seed2 | 2 | product_learned_residual | 128 | 503 | 89 | 37 | 16 | 17 | 1 | 0 |
| hybrid_beta0p20_seed3 | 3 | product_learned_residual | 79 | 551 | 90 | 13 | 9 | 3 | 0 | 0 |

*FS = false_SUPPORT; loc = location; pred = predicate; missing_FS = 0 for all runs.*

### Stage28-F aggregate by signal

| Signal | false_SUPPORT total | location_FS | role_FS | predicate_FS | missing_FS |
|---|---|---|---|---|---|
| product_p0p90 | 44 | 27 | 6 | 3 | 0 |
| learned | 136 | 66 | 38 | 8 | 0 |
| hybrid_beta0p20 | 77 | 43 | 22 | 3 | 0 |

---

## 4. Stage28-G Overlap Method

Stage28-G performed example-level alignment across all nine runs using the stable_id/source_id overlap key from Stage28-E enriched exports. Each seed partition contains 720 common records, and per-axis overlap was computed for:

- **control** (control examples — no label flip expected)
- **missing_evidence** (sufficiency rejection axis)
- **location** (location boundary axis — primary focus)
- **role** (argument role axis)
- **predicate** (predicate identity axis)
- **other_frame** (other frame type axis)

For each axis and seed, the analysis tracks:

- `product_false_SUPPORT`: how many examples product_p0p90 gets wrong
- `learned_false_SUPPORT`: how many examples learned gets wrong
- `hybrid_false_SUPPORT`: how many examples hybrid_beta0p20 gets wrong
- `learned_rescues_product`: product wrong, learned right (a true rescue)
- `hybrid_rescues_product`: product wrong, hybrid right (a true rescue)
- `learned_introduces_vs_product`: product right, learned wrong (a new error)
- `hybrid_introduces_vs_product`: product right, hybrid wrong (a new error)
- Pairwise and three-way overlaps of false SUPPORT sets

---

## 5. Aggregate Results

### Stage28-G aggregate across all axes and seeds

| Metric | Value |
|---|---|
| product false_SUPPORT total | 44 |
| learned false_SUPPORT total | 136 |
| hybrid false_SUPPORT total | 77 |
| learned rescues of product errors | 2 |
| hybrid rescues of product errors | 6 |
| learned introduces vs product | 94 |
| hybrid introduces vs product | 39 |

**Rescue-to-introduction ratio:**

- learned: 2 rescues / 94 introductions = 0.021
- hybrid: 6 rescues / 39 introductions = 0.154

Neither mechanism achieves a favorable tradeoff. Both configurations introduce many more new false SUPPORT errors than they correct.

---

## 6. Axis-Level Results

### 6.1 Control axis

All counts are zero across all seeds for all three signals. The control axis is stable and unaffected.

### 6.2 Missing evidence axis

All counts are zero across all seeds for all three signals. Sufficiency/missing-evidence rejection is fully preserved.

### 6.3 Location axis

| Seed | n_common | product_FS | learned_FS | hybrid_FS | learned_rescues | hybrid_rescues | learned_introduces | hybrid_introduces |
|---|---|---|---|---|---|---|---|---|
| 1 | 60 | 19 | 41 | 18 | 0 | 3 | 22 | 2 |
| 2 | 60 | 7 | 12 | 16 | 0 | 0 | 5 | 9 |
| 3 | 60 | 1 | 13 | 9 | 0 | 0 | 12 | 8 |
| **Total** | **180** | **27** | **66** | **43** | **0** | **3** | **39** | **19** |

**Location axis aggregate:**

- learned rescues product location errors: **0**
- hybrid rescues product location errors: **3**
- learned introduces new location errors: **39**
- hybrid introduces new location errors: **19**

### 6.4 Role axis

| Seed | n_common | product_FS | learned_FS | hybrid_FS | learned_rescues | hybrid_rescues | learned_introduces | hybrid_introduces |
|---|---|---|---|---|---|---|---|---|
| 1 | 60 | 3 | 30 | 2 | 0 | 1 | 27 | 0 |
| 2 | 60 | 3 | 5 | 17 | 2 | 0 | 4 | 14 |
| 3 | 60 | 0 | 3 | 3 | 0 | 0 | 3 | 3 |
| **Total** | **180** | **6** | **38** | **22** | **2** | **1** | **34** | **17** |

### 6.5 Predicate axis

| Seed | n_common | product_FS | learned_FS | hybrid_FS | learned_rescues | hybrid_rescues | learned_introduces | hybrid_introduces |
|---|---|---|---|---|---|---|---|---|
| 1 | 60 | 2 | 3 | 2 | 0 | 0 | 1 | 0 |
| 2 | 60 | 1 | 3 | 1 | 0 | 0 | 2 | 0 |
| 3 | 60 | 0 | 2 | 0 | 0 | 0 | 2 | 0 |
| **Total** | **180** | **3** | **8** | **3** | **0** | **0** | **5** | **0** |

### 6.6 Other frame axis

| Seed | n_common | product_FS | learned_FS | hybrid_FS | learned_rescues | hybrid_rescues | learned_introduces | hybrid_introduces |
|---|---|---|---|---|---|---|---|---|
| 1 | 180 | 7 | 16 | 5 | 0 | 2 | 9 | 0 |
| 2 | 180 | 1 | 3 | 3 | 0 | 0 | 2 | 2 |
| 3 | 180 | 0 | 5 | 1 | 0 | 0 | 5 | 1 |
| **Total** | **540** | **8** | **24** | **9** | **0** | **2** | **16** | **3** |

---

## 7. Learned-Only Failure

The learned signal fails as a correction mechanism for the following reasons:

1. **Severe seed instability.** learned_seed1 produces 90 false SUPPORT errors (versus 31 for product_seed1), a 190% increase. Seeds 2 and 3 are closer to product but still worse (23 vs. 12 and 23 vs. 1 respectively).

2. **Zero location rescues.** The learned signal rescues 0 product location false SUPPORT errors across all three seeds. The dominant error axis is completely unaddressed.

3. **Massive error introduction.** learned introduces 94 new false SUPPORT errors overall and 39 new location false SUPPORT errors. This far exceeds any benefit.

4. **Aggregate degradation.** learned total false SUPPORT = 136 vs. product total = 44. The learned signal triples the error count.

**Conclusion:** learned-only must not be used as the decision signal or as a correction mechanism.

---

## 8. Hybrid Residual Failure

The product_learned_residual hybrid signal (beta=0.20) also fails as a reliable correction mechanism:

1. **Unfavorable rescue-to-introduction ratio.** 6 rescues vs. 39 introductions (ratio 0.154). The hybrid introduces more than six new errors for every error it corrects.

2. **Seed-specific degradation.** hybrid_seed2 produces 37 false SUPPORT errors versus 12 for product_seed2 (208% increase). hybrid_seed3 produces 13 false SUPPORT errors versus 1 for product_seed3 (1200% increase).

3. **Location axis does not improve reliably.** hybrid rescues 3 product location errors but introduces 19 new location errors. Net location impact is strongly negative.

4. **Role axis regression.** hybrid_seed2 produces 17 role false SUPPORT errors versus 3 for product_seed2, driven by hybrid_introduces=14.

5. **Aggregate remains worse than product.** hybrid total false SUPPORT = 77 vs. product total = 44. Even at beta=0.20 the hybrid is 75% worse in aggregate.

**Conclusion:** product_learned_residual hybrid must not be used as a specialist mechanism for reducing product false SUPPORT errors.

---

## 9. Location Axis Interpretation

Location false SUPPORT errors are the dominant unresolved axis for the product_p0p90 baseline:

- product location false SUPPORT: **27 / 44 total** (61.4%)
- learned does not rescue any product location false SUPPORT errors (0 rescues, 39 introductions)
- hybrid rescues only 3 product location false SUPPORT errors at the cost of 19 introductions

The location axis requires a structurally different intervention. The overlap results show that softly mixing a learned residual into the product score does not produce reliable location correction. The error pattern — where both learned and hybrid introduce many new location false SUPPORT errors in the same examples — suggests that learned features do not encode reliable location boundary discrimination at this training scale.

An independent bounded location-boundary cap/head is the appropriate next mechanism because:

1. It acts only on location-axis candidates without disturbing other axes.
2. It can be bounded so that its maximum intervention is limited (unlike learned/hybrid mixing, which can dominate the product signal as shown by seed1 collapse).
3. The role, predicate, control, and missing-evidence axes are already at acceptable levels and must not be disturbed.

---

## 10. Missing-Evidence Preservation

Missing evidence false SUPPORT counts are **zero** for all configurations and all seeds:

- product_p0p90: missing_false_SUPPORT = 0 (all seeds)
- learned: missing_false_SUPPORT = 0 (all seeds)
- hybrid_beta0p20: missing_false_SUPPORT = 0 (all seeds)

This means the current sufficiency/missing-evidence rejection mechanism is fully functional and must not be disturbed by any future specialist. Stage28-G confirms that this behavior is stable across all three decision signals.

Any Stage28-I location-boundary cap/head design must preserve the existing sufficiency rejection path.

---

## 11. Design Consequence

The Stage28-G overlap results lead directly to the following design consequence:

The only reliable path to reducing product location false SUPPORT errors is an independent bounded location-boundary specialist that:

- Operates on location-axis examples only
- Applies a hard or soft cap rather than learned signal mixing
- Does not modify the product signal for non-location axes
- Does not reintroduce learned or hybrid score components
- Does not disturb the sufficiency/missing-evidence rejection behavior already in place

Learned/hybrid residual approaches are excluded by the overlap evidence. The rescue counts (0 and 3 for location) cannot justify the introduction counts (39 and 19 for location).

---

## 12. Decision

Based on Stage28-F and Stage28-G analysis:

1. **Retain product_p0p90 as the frozen Stage27/Stage28 baseline.** It has the lowest aggregate false SUPPORT count (44) and the lowest location false SUPPORT count (27) among the three configurations.

2. **Reject learned-only as a final decision signal.** It produces 136 total false SUPPORT errors (3.1× product), is severely unstable across seeds, and rescues 0 location errors.

3. **Reject product_learned_residual hybrid as a specialist mechanism.** It produces 77 total false SUPPORT errors (1.75× product), introduces 39 new errors versus 6 rescues, and degrades seed2 and seed3.

4. **Preserve missing-evidence/sufficiency behavior.** This is already solved (0 missing_false_SUPPORT) and must not be disrupted by any future stage.

5. **Move to Stage28-I.** The next stage should focus on independent bounded location-boundary cap/head design or implementation planning.

6. **Stage28-H is analysis and reporting only.** No model improvement is claimed from this stage. No OOD generalization is claimed.

---

## 13. Next Stage Recommendation

**Stage28-I: Independent Bounded Location-Boundary Cap/Head**

The Stage28-I design should:

- Define a location-boundary specialist that is architecturally independent from the product score computation
- Bound the specialist's correction magnitude to prevent the kind of signal collapse seen in learned_seed1
- Target the 27 product location false SUPPORT errors across seeds 1–3 as the reduction objective
- Evaluate on the same controlled no-time validation set (controlled_v5_v3_without_time_swap.jsonl)
- Keep the same frozen encoder, frozen_a_log, and max_length=64 constraints
- Leave the sufficiency/missing-evidence path unchanged
- Not use learned signal components or hybrid mixing as part of the specialist mechanism

Stage28-I should begin with design and planning only, consistent with the analysis-only scope established here.

---

## 14. Remaining Risks

1. **Seed instability is a persistent concern.** product_p0p90 itself shows high variance across seeds (false_SUPPORT: 31, 12, 1). A location specialist that works on one seed may not transfer to others.

2. **Controlled no-time validation only.** All results are on controlled_v5_v3_without_time_swap.jsonl. No OOD claim can be made. Behavior on unseen validation distributions is unknown.

3. **max_length=64 constraint.** All results are under this truncation limit. Behavior at longer sequence lengths is not measured.

4. **Frozen encoder.** The encoder is frozen throughout Stage28. Fine-grained location discrimination may require encoder fine-tuning not tested here.

5. **Stage28-C-lite audit coverage is limited.** Only 20 location examples were manually inspected in Stage28-C-lite. The full distribution of location false SUPPORT error types is not characterized.

6. **Location cap/head is not yet designed or implemented.** Stage28-H establishes the requirement; Stage28-I will carry the design. No performance claim can be made about the future specialist.

7. **Synthetic controlled examples only.** All examples are synthetic controlled examples. Real-world distribution behavior is not measured.

8. **300 location_swap examples are not guaranteed to be fully clean.** Manual audit covered 20 examples; the remainder have not been individually inspected.
