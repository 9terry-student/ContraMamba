# Stage137-C Non-Collapsed Slot-Prob Source Recovery

## Summary decision

Decision: `STAGE137C_NONCOLLAPSED_SOURCE_FOUND_BUT_FRESH_SLOT_EXPORT_COLLAPSED`

Stage137-C records that Stage137-A found a strong historical non-collapsed `core_only` diagnostic source, while Stage137-B confirmed that the `slot_mismatch_prob` export path works. However, Stage137-B's fresh rerun collapsed to all `NOT_ENTITLED`, so the conjunction required for Stage136 shadow guard evaluation was not achieved.

This result must not be interpreted as evidence for or against the slot guard. It only shows that blind short fresh reruns are insufficient for recovering a usable non-collapsed slot-probability source.

## Stage137-A historical non-collapsed source

Stage137-A found a non-collapsed historical candidate:

`reports/stage123b_singleprocess_evidence_interface_sweep_20260708_003428/stage118_diagnostic_core_only_predictions.jsonl`

The sampled candidate had:

- `pred_SUPPORT`: 676 sampled rows
- `support_pred_rate`: 0.1352
- `accuracy`: 0.9972
- `macro_f1`: 0.995890
- `support_recall`: 1.0

The historical Stage123-B `core_only` diagnostic summary was strong:

- `input_jsonl`: `data/stage122_prefix_core_localization_diagnostic_20260708_003422.jsonl`
- `accuracy`: 0.9977
- `macro_f1`: 0.9964
- `pred_counts`: `NOT_ENTITLED = 4636`, `REFUTE = 806`, `SUPPORT = 758`
- `gold_counts`: `NOT_ENTITLED = 4650`, `REFUTE = 806`, `SUPPORT = 744`
- `false_NE_total`: 0
- `false_entitlement_total`: 14
- `polarity_error_total`: 0

Its manifest used `architecture = vnext_minimal`, `router_mode = learned_x_product`, and `train_dev_evidence_interface = full_evidence`. The historical `core_only` output was strong but lacked `slot_mismatch_prob`, so it could not directly feed the Stage136 slot guard shadow analyzer.

## Stage137-B fresh slot export attempt

Stage137-B attempted a fresh `core_only` diagnostic export with slot mismatch probabilities enabled. The run completed successfully with return code 0, and the export audit showed:

- `slot_mismatch_prob_exported`: true
- `prediction_export_jsonl_exists`: true
- `prediction_export_row_count`: 720

However, the fresh attempt collapsed to all `NOT_ENTITLED`.

Clean dev prediction distribution:

- `NOT_ENTITLED`: 720
- `REFUTE`: 0
- `SUPPORT`: 0

Stage118 `core_only` diagnostic prediction distribution:

- `NOT_ENTITLED`: 6200
- `REFUTE`: 0
- `SUPPORT`: 0

Stage118 `core_only` diagnostic metrics:

- `accuracy`: 0.75
- `macro_f1`: 0.2857
- `false_NE_total`: 1550

Fresh Stage137-B had `slot_mismatch_prob`, but it collapsed to all `NOT_ENTITLED`.

## Why Stage137-B is unusable for guard evaluation

Stage136 guard evaluation requires a prediction JSONL that contains both `slot_mismatch_prob` and non-collapsed entitlement behavior, including `SUPPORT` predictions.

Stage137-A provided the non-collapsed historical source, but that source did not include `slot_mismatch_prob`. Stage137-B provided `slot_mismatch_prob`, but its predictions collapsed to all `NOT_ENTITLED`. Therefore, no prediction JSONL had both `slot_mismatch_prob` and `SUPPORT` predictions.

The conjunction needed for Stage136 guard evaluation was not achieved. The Stage137-B output is not usable for guard utility analysis and should not be used to claim slot guard success, failure, usefulness, or harm.

## Safety and leakage policy

No code behavior changed in Stage137-C.

Safety and leakage status:

- No final logits were modified by the slot head.
- Stage128 remained off.
- Stage15 was not used.
- No external data was used for training.
- No threshold was used for model selection.

## Stage138 recommendation

Stage138 should recover or create a non-collapsed model/export path with `slot_mismatch_prob` without blind short reruns. It should recover a non-collapsed slot-prob source rather than evaluate a guard now.

Recommended options:

- Recover the exact historical Stage123-B training config/checkpoint if available.
- Use an anti-collapse/internal-prior-aware selection run before diagnostic export.
- Avoid using the collapsed Stage137-B output for guard evaluation.

Avoid:

- claiming slot guard failure from all-NE outputs
- routing `slot_mismatch_prob` into final logits now
- repeating short 8-epoch blind reruns
