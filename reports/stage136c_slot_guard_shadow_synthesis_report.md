# Stage136-C Slot Guard Shadow Synthesis

## Summary decision

Decision: `STAGE136C_SLOT_GUARD_SHADOW_INFRA_READY_BUT_UNINFORMATIVE_ON_COLLAPSED_SANITY_RUN`

Stage136-C concludes that the slot guard shadow-analysis infrastructure is ready, but the collapsed sanity run is not an informative guard-efficacy evaluation.

## What Stage136 successfully established

Stage136-A added the standalone export-only analyzer at `scripts/analyze_stage136_slot_guard_shadow.py`. Its shadow policy is `support_to_ne`: if the exported prediction is SUPPORT and `slot_mismatch_prob >= threshold`, the shadow prediction becomes NOT_ENTITLED. REFUTE and NOT_ENTITLED rows are not changed.

Stage136-B confirmed that per-row prediction JSONL export works and that row-level slot mismatch fields are available. The confirmed fields are:

- `slot_mismatch_logit`
- `slot_mismatch_prob`
- `slot_mismatch_target`
- `slot_mismatch_target_valid`
- `vnext_slot_mismatch_input_mode`
- `vnext_slot_mismatch_head_type`
- `stage135_use_best_slot_aux`

Together, these establish that the analyzer can consume row-level `slot_mismatch_prob` exports and produce threshold/group metrics.

## Stage136-B shadow result

Input JSONL:
`reports/stage136b0_slot_prob_export_sanity_20260710_014612/stage136b0_export_sanity_report_predictions.jsonl`

Analyzer decision:
`STAGE136A_SHADOW_GUARD_UNINFORMATIVE_NO_SUPPORT_PREDICTIONS`

Thresholds evaluated: `0.50`, `0.60`, `0.70`, `0.80`, `0.90`

Key results:

- `n_total = 24`
- `n_with_slot_prob = 24`
- `n_missing_slot_prob = 0`
- `support_pred_before = 0`
- `support_pred_after = 0`
- `n_changed_total = 0`
- `false_support_before = 0`
- `false_support_after = 0`
- `false_ne_before = 6`
- `false_ne_after = 6`
- `macro_f1_before = 0.285714`
- `macro_f1_after = 0.285714`
- `slot_target_auc = 0.466667`
- `slot_target_auprc = 0.700501`
- `slot_target_positive_count = 10`
- `slot_target_negative_count = 6`

## Why the guard utility remains unevaluated

The tested sanity run had zero SUPPORT predictions. Because the shadow policy only changes SUPPORT predictions to NOT_ENTITLED when the threshold condition is met, there were no eligible rows for the guard to affect.

This means the result is not evidence that the guard is useful, and it is not evidence that the guard is harmful. It only shows that the current collapsed sanity export is unsuitable for evaluating guard utility.

## Safety and leakage policy

Stage136-C is a report-only synthesis. No final logits or final predictions were modified. No model forward behavior, training behavior, checkpoint selection, or source prediction JSONL files were modified.

Stage128 remains off. Stage15 was not used. No external data was used for training, and no threshold was used for model selection.

## Recommendation for Stage137

Stage137 should focus on obtaining a non-collapsed prediction export that contains both row-level `slot_mismatch_prob` and SUPPORT predictions, then rerun the Stage136 shadow analysis.

Stage137 should avoid routing `slot_mismatch_prob` into final logits now, avoid claiming guard usefulness from the collapsed sanity run, and avoid spending more effort on tiny sanity exports that predict only NOT_ENTITLED.
