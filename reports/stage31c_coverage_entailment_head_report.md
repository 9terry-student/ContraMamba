# Stage31-C Coverage/Entailment Diagnostic Head Report

## Purpose
Stage31-C adds a diagnostic directional Coverage/Entailment owner for the v7 hierarchical path. The head is a readout-only signal intended to distinguish valid entailment-preserving SUPPORT, overclaim NOT_ENTITLED, contradiction REFUTE, and OTHER_RESIDUAL cases.

## Stage31-B Bottleneck
Observed Stage31-B result: total_accuracy=0.445, macro_f1=0.3607, coverage_failure_predicted_support=6, support_entailment_predicted_ne=61, refute_case_predicted_support=5, refute_case_predicted_ne=27.

The proxy stack is conservative but under-structured. It can often suppress over-claims, but it cannot reliably preserve valid SUPPORT under weakening, generalization, and part-inclusion entailment.

## Auxiliary Data
Stage31-C uses `data/stage31c_coverage_entailment_aux.jsonl`, separate from the Stage31 evaluation probe. It provides balanced directional supervision:

| Class | ID | Intended final label |
|---|---:|---|
| ENTAILS_SUPPORT | 0 | SUPPORT |
| OVERCLAIM_NOT_ENTITLED | 1 | NOT_ENTITLED |
| CONTRADICTS_REFUTE | 2 | REFUTE |
| OTHER_RESIDUAL | 3 | mixed SUPPORT/NOT_ENTITLED |

The auxiliary file is diagnostic supervision, not final evaluation.

## Head Architecture
The optional head is registered from `scripts/train_controlled_v6b_minimal.py` when `--v7-use-coverage-entailment-head` is enabled.

Input:
`concat([frame_pair_repr, predicate_pair_repr, sufficiency_repr])`

Output:
4-way logits/probabilities over coverage direction classes.

Loss:
optional cross entropy over `coverage_direction_id`, enabled only with `--v7-use-coverage-entailment-loss` and `--v7-coverage-entailment-data`.

## Diagnostic-Only Constraints
Stage31-C does not modify `output["logits"]`, H1 final composer, entitlement, caps, NE boosting, calibration, threshold selection, or checkpoint selection. The Stage31-A/B evaluation probe must not be used for this loss.

## Exported Columns
- `coverage_entails_support_prob`
- `coverage_overclaim_ne_prob`
- `coverage_contradicts_refute_prob`
- `coverage_other_residual_prob`
- `coverage_entailment_pred_id`
- `coverage_entailment_pred_label`
- `coverage_entailment_confidence`

## Expected Interpretation
If final predictions fail but the diagnostic head aligns, the directional signal exists but is not wired into the final composer. If the head also fails, the current representations or auxiliary data do not expose stable directional coverage information.

## Observed Stage31-C Evaluation
Run name: `stage31c_coverage_entailment_head_eval`

Prediction file evaluation on the 200-row Stage31 probe produced:

| Metric | Value |
|---|---:|
| total_accuracy | 0.4650 |
| macro_f1 | 0.3878 |
| owner_accuracy | 0.4650 |
| coverage_direction_alignment_accuracy | 0.4450 |
| support_entailment_recovered_by_head | 39 / 80 |
| overclaim_detected_by_head | 44 / 80 |
| refute_detected_by_head | 6 / 40 |

Final-label failure modes:

| Failure mode | Count |
|---|---:|
| coverage_failure_predicted_support | 5 |
| support_entailment_predicted_ne | 56 |
| refute_case_predicted_support | 10 |
| refute_case_predicted_ne | 22 |
| unexpected_refute_on_support_or_ne | 14 |

Coverage-direction alignment by group:

| Group | Correct | Total | Accuracy |
|---|---:|---:|---:|
| all_to_some_support | 13 | 20 | 0.6500 |
| also_to_only_not_entitled | 6 | 20 | 0.3000 |
| general_to_specific_not_entitled | 9 | 20 | 0.4500 |
| none_to_some_refute | 4 | 20 | 0.2000 |
| only_to_base_support | 13 | 20 | 0.6500 |
| part_to_whole_not_entitled | 13 | 20 | 0.6500 |
| some_to_all_not_entitled | 16 | 20 | 0.8000 |
| some_to_none_refute | 2 | 20 | 0.1000 |
| specific_to_general_support | 6 | 20 | 0.3000 |
| whole_to_part_support | 7 | 20 | 0.3500 |

Interpretation: the diagnostic head does not yet expose a stable enough directional Coverage/Entailment signal. SUPPORT entailment recovery remains weak, especially for specific-to-general and whole-to-part SUPPORT groups, while refute direction is largely not captured. Composer integration should not proceed from this result.

## Leakage Policy
Do not use `data/stage31_coverage_entailment_probe.jsonl` for training, calibration, threshold selection, checkpoint selection, or auxiliary loss construction.

## Next-Step Decision Rule
If the diagnostic head aligns on the Stage31 probe, proceed to Stage31-D composer integration. If it does not align, revise the auxiliary coverage data or representation access before integration.

Current decision: revise Stage31-C auxiliary coverage data or representation access before composer integration.
