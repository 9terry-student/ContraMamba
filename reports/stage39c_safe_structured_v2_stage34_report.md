# Stage39 Final Composer Opt-In Validation

## 1. Overall Decision
- Run: `stage39c_safe_structured_v2_stage34`
- Safety profile: `stage34`
- Decision: `STAGE39C_SAFE_STRUCTURED_V2_PASS`
- Decision reason: Stage34 composed/exported macro-F1 and accuracy both meet the >=0.90 target, Stage39 introduced no unsafe SUPPORT or SUPPORT<->REFUTE transitions, and at least one row was changed by composition.

## Input File
- Predictions file: `outputs/stage39c_safe_structured_v2_stage34_1epoch/external_stage34/external_probe_stage34_heldout_coverage_predictions.json`
- Row count: 400
- Baseline predictions file: `n/a`
- Probe name: `stage34_heldout_coverage`
- Expected split: `stage34`

## 2. Original vs Composed vs Exported Final Metrics
| Metric | Original | Composed | Exported |
|---|---:|---:|---:|
| accuracy | 0.4500 | 0.9600 | 0.9600 |
| macro_f1 | 0.2069 | 0.9703 | 0.9703 |
| n | 400 | 400 | 400 |

- composed - original accuracy: 0.5100
- composed - original macro_f1: 0.7634
- exported - original accuracy: 0.5100
- exported - original macro_f1: 0.7634
- Original prediction counts: {'NOT_ENTITLED': 400}
- Composed prediction counts: {'NOT_ENTITLED': 196, 'REFUTE': 40, 'SUPPORT': 164}
- Exported prediction counts: {'NOT_ENTITLED': 196, 'REFUTE': 40, 'SUPPORT': 164}

## 3. Change Counts
| Counter | Value |
|---|---:|
| changed_row_count | 204 |
| changed_row_rate | 0.51 |
| changed_to_SUPPORT_count | 164 |
| changed_to_REFUTE_count | 40 |
| changed_to_NOT_ENTITLED_count | 0 |
| REFUTE_to_SUPPORT_count | 0 |
| NOT_ENTITLED_to_SUPPORT_count | 164 |
| SUPPORT_to_NOT_ENTITLED_count | 0 |
| SUPPORT_to_REFUTE_count | 0 |

## 4. Total Safety Counters (may include residual pre-Stage39 errors)
| Counter | Value |
|---|---:|
| overclaim_to_SUPPORT | 0 |
| exception_to_SUPPORT_error | 0 |
| location_scope_to_SUPPORT_error | 0 |
| temporal_scope_to_SUPPORT_error | 0 |
| refute_to_SUPPORT | 0 |
| support_to_refute | 0 |

## 5. Introduced Safety Counters (errors caused by Stage39 composition only)
| Counter | Value |
|---|---:|
| introduced_overclaim_to_SUPPORT | 0 |
| introduced_exception_to_SUPPORT_error | 0 |
| introduced_location_scope_to_SUPPORT_error | 0 |
| introduced_temporal_scope_to_SUPPORT_error | 0 |
| introduced_refute_to_SUPPORT | 0 |
| introduced_support_to_REFUTE | 0 |
| introduced_unsafe_SUPPORT_total | 0 |

## 6. Stage39 Behavior Counters
| Counter | Value |
|---|---:|
| stage39_enabled_count | 400 |
| stage39_changed_count | 204 |
| stage39_support_composed_count | 164 |
| stage39_refute_composed_count | 40 |
| stage39_not_entitled_composed_count | 0 |
| stage39_blocked_by_stage36_count | 0 |
| stage39_blocked_by_refute_to_support_guard_count | 0 |
| stage39_blocked_by_stage37_from_refute_guard_count | 0 |
| stage39_missing_source_count | 0 |

- stage39_composer_action_counts: {'composed_to_support': 164, 'no_change': 196, 'composed_to_refute': 40}
- stage39_composer_reason_counts: {'support_only_from_stage37': 164, 'safe_structured_v2_no_qualifying_transition': 196, 'safe_structured_v2_high_precision_contradiction': 40}

## 7. First 30 Changed Rows
| Row ID | Gold | Original | Composed | Action | Reason |
|---|---|---|---|---|---|
| stage34a_heldout_all_to_some_support_00 | SUPPORT | NOT_ENTITLED | SUPPORT | composed_to_support | support_only_from_stage37 |
| stage34a_heldout_all_to_some_support_01 | SUPPORT | NOT_ENTITLED | SUPPORT | composed_to_support | support_only_from_stage37 |
| stage34a_heldout_all_to_some_support_02 | SUPPORT | NOT_ENTITLED | SUPPORT | composed_to_support | support_only_from_stage37 |
| stage34a_heldout_all_to_some_support_03 | SUPPORT | NOT_ENTITLED | SUPPORT | composed_to_support | support_only_from_stage37 |
| stage34a_heldout_all_to_some_support_04 | SUPPORT | NOT_ENTITLED | SUPPORT | composed_to_support | support_only_from_stage37 |
| stage34a_heldout_all_to_some_support_05 | SUPPORT | NOT_ENTITLED | SUPPORT | composed_to_support | support_only_from_stage37 |
| stage34a_heldout_all_to_some_support_06 | SUPPORT | NOT_ENTITLED | SUPPORT | composed_to_support | support_only_from_stage37 |
| stage34a_heldout_all_to_some_support_07 | SUPPORT | NOT_ENTITLED | SUPPORT | composed_to_support | support_only_from_stage37 |
| stage34a_heldout_all_to_some_support_08 | SUPPORT | NOT_ENTITLED | SUPPORT | composed_to_support | support_only_from_stage37 |
| stage34a_heldout_all_to_some_support_09 | SUPPORT | NOT_ENTITLED | SUPPORT | composed_to_support | support_only_from_stage37 |
| stage34a_heldout_all_to_some_support_10 | SUPPORT | NOT_ENTITLED | SUPPORT | composed_to_support | support_only_from_stage37 |
| stage34a_heldout_all_to_some_support_11 | SUPPORT | NOT_ENTITLED | SUPPORT | composed_to_support | support_only_from_stage37 |
| stage34a_heldout_all_to_some_support_12 | SUPPORT | NOT_ENTITLED | SUPPORT | composed_to_support | support_only_from_stage37 |
| stage34a_heldout_all_to_some_support_13 | SUPPORT | NOT_ENTITLED | SUPPORT | composed_to_support | support_only_from_stage37 |
| stage34a_heldout_all_to_some_support_14 | SUPPORT | NOT_ENTITLED | SUPPORT | composed_to_support | support_only_from_stage37 |
| stage34a_heldout_all_to_some_support_15 | SUPPORT | NOT_ENTITLED | SUPPORT | composed_to_support | support_only_from_stage37 |
| stage34a_heldout_all_to_some_support_16 | SUPPORT | NOT_ENTITLED | SUPPORT | composed_to_support | support_only_from_stage37 |
| stage34a_heldout_all_to_some_support_17 | SUPPORT | NOT_ENTITLED | SUPPORT | composed_to_support | support_only_from_stage37 |
| stage34a_heldout_all_to_some_support_18 | SUPPORT | NOT_ENTITLED | SUPPORT | composed_to_support | support_only_from_stage37 |
| stage34a_heldout_all_to_some_support_19 | SUPPORT | NOT_ENTITLED | SUPPORT | composed_to_support | support_only_from_stage37 |
| stage34a_heldout_none_to_some_refute_00 | REFUTE | NOT_ENTITLED | REFUTE | composed_to_refute | safe_structured_v2_high_precision_contradiction |
| stage34a_heldout_none_to_some_refute_01 | REFUTE | NOT_ENTITLED | REFUTE | composed_to_refute | safe_structured_v2_high_precision_contradiction |
| stage34a_heldout_none_to_some_refute_02 | REFUTE | NOT_ENTITLED | REFUTE | composed_to_refute | safe_structured_v2_high_precision_contradiction |
| stage34a_heldout_none_to_some_refute_03 | REFUTE | NOT_ENTITLED | REFUTE | composed_to_refute | safe_structured_v2_high_precision_contradiction |
| stage34a_heldout_none_to_some_refute_04 | REFUTE | NOT_ENTITLED | REFUTE | composed_to_refute | safe_structured_v2_high_precision_contradiction |
| stage34a_heldout_none_to_some_refute_05 | REFUTE | NOT_ENTITLED | REFUTE | composed_to_refute | safe_structured_v2_high_precision_contradiction |
| stage34a_heldout_none_to_some_refute_06 | REFUTE | NOT_ENTITLED | REFUTE | composed_to_refute | safe_structured_v2_high_precision_contradiction |
| stage34a_heldout_none_to_some_refute_07 | REFUTE | NOT_ENTITLED | REFUTE | composed_to_refute | safe_structured_v2_high_precision_contradiction |
| stage34a_heldout_none_to_some_refute_08 | REFUTE | NOT_ENTITLED | REFUTE | composed_to_refute | safe_structured_v2_high_precision_contradiction |
| stage34a_heldout_none_to_some_refute_09 | REFUTE | NOT_ENTITLED | REFUTE | composed_to_refute | safe_structured_v2_high_precision_contradiction |

## 8. First 30 Unsafe Rows (total, including residual)
None.

## 9. First 30 Introduced-Unsafe Rows (Stage39-caused only)
None.

## 10. Recommendation
The Stage39 opt-in final composer is safe to use as an explicit final-prediction replacement under the evaluated policy/profile. It remains off by default; enabling it in production requires an explicit --stage39-use-final-composer-opt-in (and, to replace predictions, --stage39-final-composer-output-mode replace_pred_final_label) decision outside this script.

## Leakage Policy
Diagnostic-only. This evaluator must not be used for training, calibration, threshold selection, loss, checkpoint selection, or Kaggle selection.
