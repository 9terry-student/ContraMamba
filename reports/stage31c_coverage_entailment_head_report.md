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

## Leakage Policy
Do not use `data/stage31_coverage_entailment_probe.jsonl` for training, calibration, threshold selection, checkpoint selection, or auxiliary loss construction.

## Next-Step Decision Rule
If the diagnostic head aligns on the Stage31 probe, proceed to Stage31-D composer integration. If it does not align, revise the auxiliary coverage data or representation access before integration.
