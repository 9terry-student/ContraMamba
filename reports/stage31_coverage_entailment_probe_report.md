# Stage31-A Coverage/Entailment Diagnostic Probe Report

## Purpose

Stage31-A introduces the **Coverage/Entailment** diagnostic track for the
ContraMamba architecture. The target architecture hierarchy is:

```
Mamba Encoder
→ Hard Core Validity
→ Coverage / Entailment      ← Stage31 focus
→ Residual Adjudication
→ ANI-style Epistemic Diagnosis
→ Polarity
→ Final Label
```

This probe dataset is **diagnostic only**. It must not be used for
training, calibration, threshold selection, checkpoint selection,
or train/dev split construction.

---

## Why Coverage/Entailment is Distinct from Frame

Frame mismatch asks: *Do claim and evidence involve the same
event/entity/predicate frame?*

Coverage/Entailment asks: *Does the evidence cover or entail the claim's
required scope, strength, or specificity?*

A pair can pass Hard Core Validity (same frame, entity, predicate) while
failing Coverage/Entailment because the evidence is weaker in scope
(e.g. `some` vs `all`), weaker in specificity (general vs specific), or
weaker in exclusivity (`also` vs `only`).

---

## Directional Entailment Rules

| Direction | Label |
|-----------|-------|
| All → Some | SUPPORT |
| Some → All | NOT_ENTITLED |
| Specific → General | SUPPORT |
| General → Specific | NOT_ENTITLED |
| Only → Base membership | SUPPORT |
| Also → Only | NOT_ENTITLED |
| Whole → Included part | SUPPORT |
| Part → Whole | NOT_ENTITLED |
| None → Some | REFUTE |
| Some → None | REFUTE |

---

## Group Definitions and Counts

| Group | Description | Label | Count |
|-------|-------------|-------|-------|
| all_to_some_support | Evidence: ALL X; Claim: SOME X | SUPPORT | 20 |
| some_to_all_not_entitled | Evidence: SOME X; Claim: ALL X | NOT_ENTITLED | 20 |
| specific_to_general_support | Evidence: specific; Claim: general | SUPPORT | 20 |
| general_to_specific_not_entitled | Evidence: general; Claim: specific | NOT_ENTITLED | 20 |
| only_to_base_support | Evidence: only member; Claim: a member | SUPPORT | 20 |
| also_to_only_not_entitled | Evidence: also a member; Claim: only member | NOT_ENTITLED | 20 |
| whole_to_part_support | Evidence: whole set; Claim: included subset | SUPPORT | 20 |
| part_to_whole_not_entitled | Evidence: subset; Claim: whole set | NOT_ENTITLED | 20 |
| none_to_some_refute | Evidence: NO X; Claim: SOME X | REFUTE | 20 |
| some_to_none_refute | Evidence: SOME X; Claim: NO X | REFUTE | 20 |

---

## Label Distribution

| Label | Gold | Count |
|-------|------|-------|
| SUPPORT | 2 | 80 |
| NOT_ENTITLED | 1 | 80 |
| REFUTE | 0 | 40 |
| **Total** | — | **200** |

---

## Stage31-A2 Schema Compatibility

Each row includes both Stage31 diagnostic fields and controlled-style
compatibility fields for v5/v6 external-eval encoders.

Required identity and label fields:

- `id` and stable unique `pair_id` (same value in this probe)
- `claim` and `evidence`
- `label` and `final_label` as string labels
- `gold` and `label_id` as numeric labels
- `group`, `coverage_relation`, `expected_owner`,
  `hard_core_should_pass`, `polarity_should_be`, and `notes`

Controlled-style auxiliary fields:

- `frame_compatible_label = 1`
- `predicate_covered_label = 1`
- `sufficiency_label` and `evidence_sufficient_label` are `1` for
  SUPPORT/REFUTE and `0` for NOT_ENTITLED
- `polarity_label` is SUPPORT, REFUTE, or NONE
- `intervention_type = group` and `probe_type = group`

Numeric mapping remains `REFUTE=0`, `NOT_ENTITLED=1`, `SUPPORT=2`.

These compatibility fields exist only so external prediction export can
read the probe without ad hoc Kaggle-only schema rewrites. They do not
change Stage31 probe semantics.

---

## Owner Rule

All rows in this dataset have `expected_owner = coverage_entailment`.

This means the intended decision axis for each pair is the
Coverage/Entailment component of the target architecture, not Frame,
Residual Adjudication, Polarity, or the Final Composer.

---

## Leakage Policy

> **This dataset is diagnostic-only.**
>
> It must NOT be used for:
> - Main classification training or fine-tuning
> - Calibration
> - Threshold selection
> - Checkpoint, model, or hyperparameter selection
> - Train/dev split construction
> - OOD evaluation benchmarks

Its sole purpose is to probe whether the Coverage/Entailment component
of the target architecture correctly handles directional entailment.

---

## Known Limitations

1. **Synthetic templates** — sentences use controlled vocabulary and
   do not reflect natural language diversity.
2. **Narrow coverage phenomena** — only all/some/no, only/also,
   specific/general, and whole/part patterns are covered.
   Compositional nesting (e.g. `all of some`) is not yet included.
3. **No world-knowledge verification** — all entities are synthetic.
4. **Short and controlled length** — OOD robustness on longer,
   noisier text is untested.
5. **No cross-axis interactions** — each pair keeps frame/entity/
   relation stable to isolate coverage failures, so interactions
   between Frame and Coverage axes are not exercised.

---

*Generated by `scripts/build_stage31_coverage_entailment_probe.py` — Stage31-A, 2026-06-27.*
