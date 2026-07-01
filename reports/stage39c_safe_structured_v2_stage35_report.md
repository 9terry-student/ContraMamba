# Stage39 Final Composer Opt-In Validation

## 1. Overall Decision
- Run: `stage39c_safe_structured_v2_stage35`
- Safety profile: `stage35`
- Decision: `STAGE39C_SAFE_STRUCTURED_V2_PASS`
- Decision reason: Stage35 composed/exported macro-F1 >=0.70, accuracy >=0.75, >=140 rows recovered to SUPPORT, and Stage39 introduced no unsafe SUPPORT or SUPPORT<->REFUTE transitions.

## Input File
- Predictions file: `outputs/stage39c_safe_structured_v2_stage35_1epoch/external_stage35/external_probe_stage35_adversarial_coverage_predictions.json`
- Row count: 600
- Baseline predictions file: `n/a`
- Probe name: `stage35_adversarial_coverage`
- Expected split: `stage35`

## 2. Original vs Composed vs Exported Final Metrics
| Metric | Original | Composed | Exported |
|---|---:|---:|---:|
| accuracy | 0.5033 | 0.7850 | 0.7850 |
| macro_f1 | 0.2832 | 0.7202 | 0.7202 |
| n | 600 | 600 | 600 |

- composed - original accuracy: 0.2817
- composed - original macro_f1: 0.4370
- exported - original accuracy: 0.2817
- exported - original macro_f1: 0.4370
- Original prediction counts: {'NOT_ENTITLED': 583, 'REFUTE': 17}
- Composed prediction counts: {'NOT_ENTITLED': 414, 'REFUTE': 36, 'SUPPORT': 150}
- Exported prediction counts: {'NOT_ENTITLED': 414, 'REFUTE': 36, 'SUPPORT': 150}

## 3. Change Counts
| Counter | Value |
|---|---:|
| changed_row_count | 169 |
| changed_row_rate | 0.2817 |
| changed_to_SUPPORT_count | 150 |
| changed_to_REFUTE_count | 19 |
| changed_to_NOT_ENTITLED_count | 0 |
| REFUTE_to_SUPPORT_count | 0 |
| NOT_ENTITLED_to_SUPPORT_count | 150 |
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
| support_to_refute | 7 |

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
| stage39_enabled_count | 600 |
| stage39_changed_count | 169 |
| stage39_support_composed_count | 150 |
| stage39_refute_composed_count | 19 |
| stage39_not_entitled_composed_count | 0 |
| stage39_blocked_by_stage36_count | 0 |
| stage39_blocked_by_refute_to_support_guard_count | 3 |
| stage39_blocked_by_stage37_from_refute_guard_count | 0 |
| stage39_missing_source_count | 0 |

- stage39_composer_action_counts: {'blocked_by_refute_to_support_guard': 3, 'no_change': 428, 'composed_to_support': 150, 'composed_to_refute': 19}
- stage39_composer_reason_counts: {'refute_to_support_guard_active': 3, 'safe_structured_v2_no_qualifying_transition': 428, 'support_only_from_stage37': 150, 'safe_structured_v2_high_precision_contradiction': 19}

## 7. First 30 Changed Rows
| Row ID | Gold | Original | Composed | Action | Reason |
|---|---|---|---|---|---|
| stage35a_adv_whole_to_part_support_verb_diverse_02 | SUPPORT | NOT_ENTITLED | SUPPORT | composed_to_support | support_only_from_stage37 |
| stage35a_adv_whole_to_part_support_verb_diverse_03 | SUPPORT | NOT_ENTITLED | SUPPORT | composed_to_support | support_only_from_stage37 |
| stage35a_adv_whole_to_part_support_verb_diverse_04 | SUPPORT | NOT_ENTITLED | SUPPORT | composed_to_support | support_only_from_stage37 |
| stage35a_adv_whole_to_part_support_verb_diverse_06 | SUPPORT | NOT_ENTITLED | SUPPORT | composed_to_support | support_only_from_stage37 |
| stage35a_adv_whole_to_part_support_verb_diverse_08 | SUPPORT | NOT_ENTITLED | SUPPORT | composed_to_support | support_only_from_stage37 |
| stage35a_adv_whole_to_part_support_verb_diverse_09 | SUPPORT | NOT_ENTITLED | SUPPORT | composed_to_support | support_only_from_stage37 |
| stage35a_adv_whole_to_part_support_verb_diverse_10 | SUPPORT | NOT_ENTITLED | SUPPORT | composed_to_support | support_only_from_stage37 |
| stage35a_adv_whole_to_part_support_verb_diverse_12 | SUPPORT | NOT_ENTITLED | SUPPORT | composed_to_support | support_only_from_stage37 |
| stage35a_adv_whole_to_part_support_verb_diverse_14 | SUPPORT | NOT_ENTITLED | SUPPORT | composed_to_support | support_only_from_stage37 |
| stage35a_adv_whole_to_part_support_verb_diverse_15 | SUPPORT | NOT_ENTITLED | SUPPORT | composed_to_support | support_only_from_stage37 |
| stage35a_adv_whole_to_part_support_verb_diverse_16 | SUPPORT | NOT_ENTITLED | SUPPORT | composed_to_support | support_only_from_stage37 |
| stage35a_adv_whole_to_part_support_verb_diverse_18 | SUPPORT | NOT_ENTITLED | SUPPORT | composed_to_support | support_only_from_stage37 |
| stage35a_adv_whole_to_part_support_verb_diverse_20 | SUPPORT | NOT_ENTITLED | SUPPORT | composed_to_support | support_only_from_stage37 |
| stage35a_adv_whole_to_part_support_verb_diverse_21 | SUPPORT | NOT_ENTITLED | SUPPORT | composed_to_support | support_only_from_stage37 |
| stage35a_adv_whole_to_part_support_verb_diverse_22 | SUPPORT | NOT_ENTITLED | SUPPORT | composed_to_support | support_only_from_stage37 |
| stage35a_adv_whole_to_part_support_verb_diverse_24 | SUPPORT | NOT_ENTITLED | SUPPORT | composed_to_support | support_only_from_stage37 |
| stage35a_adv_whole_to_part_support_fronted_modifier_00 | SUPPORT | NOT_ENTITLED | SUPPORT | composed_to_support | support_only_from_stage37 |
| stage35a_adv_whole_to_part_support_fronted_modifier_04 | SUPPORT | NOT_ENTITLED | SUPPORT | composed_to_support | support_only_from_stage37 |
| stage35a_adv_whole_to_part_support_fronted_modifier_06 | SUPPORT | NOT_ENTITLED | SUPPORT | composed_to_support | support_only_from_stage37 |
| stage35a_adv_whole_to_part_support_fronted_modifier_12 | SUPPORT | NOT_ENTITLED | SUPPORT | composed_to_support | support_only_from_stage37 |
| stage35a_adv_whole_to_part_support_fronted_modifier_16 | SUPPORT | NOT_ENTITLED | SUPPORT | composed_to_support | support_only_from_stage37 |
| stage35a_adv_whole_to_part_support_fronted_modifier_18 | SUPPORT | NOT_ENTITLED | SUPPORT | composed_to_support | support_only_from_stage37 |
| stage35a_adv_whole_to_part_support_fronted_modifier_22 | SUPPORT | NOT_ENTITLED | SUPPORT | composed_to_support | support_only_from_stage37 |
| stage35a_adv_whole_to_part_support_fronted_modifier_24 | SUPPORT | NOT_ENTITLED | SUPPORT | composed_to_support | support_only_from_stage37 |
| stage35a_adv_whole_to_part_support_postnominal_modifier_00 | SUPPORT | NOT_ENTITLED | SUPPORT | composed_to_support | support_only_from_stage37 |
| stage35a_adv_whole_to_part_support_postnominal_modifier_02 | SUPPORT | NOT_ENTITLED | SUPPORT | composed_to_support | support_only_from_stage37 |
| stage35a_adv_whole_to_part_support_postnominal_modifier_03 | SUPPORT | NOT_ENTITLED | SUPPORT | composed_to_support | support_only_from_stage37 |
| stage35a_adv_whole_to_part_support_postnominal_modifier_04 | SUPPORT | NOT_ENTITLED | SUPPORT | composed_to_support | support_only_from_stage37 |
| stage35a_adv_whole_to_part_support_postnominal_modifier_06 | SUPPORT | NOT_ENTITLED | SUPPORT | composed_to_support | support_only_from_stage37 |
| stage35a_adv_whole_to_part_support_postnominal_modifier_08 | SUPPORT | NOT_ENTITLED | SUPPORT | composed_to_support | support_only_from_stage37 |

## 8. First 30 Unsafe Rows (total, including residual)
| Row ID | Group | Gold | Original | Composed | Action | Reason |
|---|---|---|---|---|---|---|
| stage35a_adv_whole_to_part_support_verb_diverse_00 | adv_whole_to_part_support_verb_diverse | SUPPORT | REFUTE | REFUTE | blocked_by_refute_to_support_guard | refute_to_support_guard_active |
| stage35a_adv_whole_to_part_support_fronted_modifier_10 | adv_whole_to_part_support_fronted_modifier | SUPPORT | REFUTE | REFUTE | blocked_by_refute_to_support_guard | refute_to_support_guard_active |
| stage35a_adv_whole_to_part_support_sentence_order_flip_00 | adv_whole_to_part_support_sentence_order_flip | SUPPORT | REFUTE | REFUTE | blocked_by_refute_to_support_guard | refute_to_support_guard_active |
| stage35a_adv_no_except_subset_support_20 | adv_no_except_subset_support | SUPPORT | REFUTE | REFUTE | no_change | safe_structured_v2_no_qualifying_transition |
| stage35a_adv_coordination_support_00 | adv_coordination_support | SUPPORT | REFUTE | REFUTE | no_change | safe_structured_v2_no_qualifying_transition |
| stage35a_adv_coordination_support_02 | adv_coordination_support | SUPPORT | REFUTE | REFUTE | no_change | safe_structured_v2_no_qualifying_transition |
| stage35a_adv_numeric_subset_support_10 | adv_numeric_subset_support | SUPPORT | REFUTE | REFUTE | no_change | safe_structured_v2_no_qualifying_transition |

## 9. First 30 Introduced-Unsafe Rows (Stage39-caused only)
None.

## 10. Recommendation
The Stage39 opt-in final composer is safe to use as an explicit final-prediction replacement under the evaluated policy/profile. It remains off by default; enabling it in production requires an explicit --stage39-use-final-composer-opt-in (and, to replace predictions, --stage39-final-composer-output-mode replace_pred_final_label) decision outside this script.

## Leakage Policy
Diagnostic-only. This evaluator must not be used for training, calibration, threshold selection, loss, checkpoint selection, or Kaggle selection.
