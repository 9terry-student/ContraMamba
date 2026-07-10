# Stage157-B External Scalar Availability Synthesis

## 1. Summary decision

Decision: `STAGE157B_EXTERNAL_SCALAR_EXPORT_REQUIRED_CONFIRMED`

Stage157-B confirms the Stage157-A finding: scalar exports are required before recovered-vs-regressed external behavior can be interpreted. This is a report-only synthesis step and does not modify model behavior, training behavior, export behavior, analyzer scripts, source predictions, logits, thresholds, or checkpoint selection.

## 2. Why Stage157-A was needed

Stage156-A established the relevant pairwise comparison between Stage53 frozen external predictions and Stage63 bridge-enabled external predictions. That comparison identified the groups needed for the next audit:

- safe recovery candidates: `recovered_by_stage63`
- unsafe over-release candidates: `regressed_by_stage63`
- persistent failures: `both_wrong`

Stage157-A was needed to determine whether those prediction files already contained the scalar evidence required to explain why Stage63 recovered some examples while regressing others.

## 3. Stage156-A transition context

Stage156-A found:

- `both_wrong`: 626
- `recovered_by_stage63`: 218
- `both_correct`: 104
- `regressed_by_stage63`: 52

Top prediction transitions:

| Transition | Count |
| --- | ---: |
| `NOT_ENTITLED->NOT_ENTITLED` | 489 |
| `NOT_ENTITLED->SUPPORT` | 264 |
| `NOT_ENTITLED->REFUTE` | 191 |
| `REFUTE->REFUTE` | 24 |
| `SUPPORT->SUPPORT` | 18 |
| `REFUTE->SUPPORT` | 9 |
| `SUPPORT->NOT_ENTITLED` | 3 |
| `SUPPORT->REFUTE` | 2 |

This context shows why a scalar-level audit matters: Stage63 improved external accuracy and macro F1, but the recovered and regressed subsets must be separated before drawing architecture conclusions.

## 4. Scalar availability result

Stage157-A confirmed scalar exports are missing from both Stage53 and Stage63 external JSONLs.

Scalar availability:

- `stage53_has_scalars`: false
- `stage63_has_scalars`: false
- `has_both_scalars`: false
- `stage53_n_scalar_fields`: 0
- `stage63_n_scalar_fields`: 0
- `stage53_scalar_fields`: `{}`
- `stage63_scalar_fields`: `{}`

Recovered-vs-regressed scalar comparison cannot be performed yet.

## 5. Why scalar export is now required

External entitlement and routing calibration require scalar-level evidence. Without frame, predicate, sufficiency, entitlement, compositional entitlement, learned entitlement, polarity, and energy exports, the Stage63 recovered-vs-regressed behavior cannot be explained.

The current blocker is not the absence of pairwise groups. Stage156-A and Stage157-A already established those groups. The blocker is that the external prediction exports lack the internal scalar fields needed to inspect the mechanism behind recovery, regression, and persistent failure.

## 6. Why this is not an architecture patch yet

This is not ready for an architecture patch because the available evidence stops at final predictions and transition counts. Final predictions can show that Stage63 changed behavior, but they cannot explain whether the cause is frame recognition, predicate coverage, sufficiency, entitlement calibration, compositional entitlement, learned entitlement, polarity margin, or energy separation.

This is also not a shadow-rule problem. Shadow diagnostics remain frozen and must not be integrated into final predictions. The next step is evidence export, not architectural modification.

## 7. Recommended Stage158-A

Recommended next stage: `Stage158-A`

Goal: `run_current_best_external_scalar_export`

Purpose: produce external prediction JSONL with internal scalar exports for recovered, regressed, and both-wrong analysis.

Required scalar fields:

- `frame_prob`
- `predicate_coverage_prob`
- `sufficiency_prob`
- `entitlement_prob`
- `compositional_entitlement_prob`
- `learned_entitlement_prob`
- `learned_entitlement_logit`
- `polarity_margin`
- `positive_energy`
- `negative_energy`

Preferred inputs:

- same VitaminC sample1000 external data used by Stage53 and Stage63
- current best checkpoint or Stage63-equivalent bridge-enabled checkpoint
- same row ordering or stable IDs to support pairwise join against Stage156-A

Stage158-A should produce current-best or Stage63-equivalent external scalar exports.

## 8. Safety constraints

External labels must not be used for training, threshold tuning, or checkpoint selection.

Safety policy:

- analysis-only synthesis
- do not train on external labels
- do not tune thresholds on external labels
- do not use external labels for checkpoint selection
- do not integrate shadow diagnostics into final predictions
- do not mutate source predictions
- do not modify final logits
- do not modify final predictions
- do not modify training behavior
- do not modify checkpoint selection
- do not modify model code, training code, export behavior, or analyzer scripts
