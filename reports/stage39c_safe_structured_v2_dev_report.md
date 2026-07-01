# Stage39 Final Composer Opt-In Validation

## 1. Overall Decision
- Run: `stage39c_safe_structured_v2_dev`
- Safety profile: `dev`
- Decision: `STAGE39A_FINAL_COMPOSER_TOO_WEAK`
- Decision reason: safety_profile=dev: no row's final label was changed by composition, so there is nothing to validate for a production replace_pred_final_label decision yet.

## Input File
- Predictions file: `outputs/stage39c_safe_structured_v2_stage35_1epoch/dev_predictions.json`
- Row count: 720
- Baseline predictions file: `n/a`
- Probe name: `clean_dev`
- Expected split: `dev`

## 2. Original vs Composed vs Exported Final Metrics
| Metric | Original | Composed | Exported |
|---|---:|---:|---:|
| accuracy | 0.5611 | 0.5611 | 0.5611 |
| macro_f1 | 0.3617 | 0.3617 | 0.3617 |
| n | 720 | 720 | 720 |

- composed - original accuracy: 0.0000
- composed - original macro_f1: 0.0000
- exported - original accuracy: 0.0000
- exported - original macro_f1: 0.0000
- Original prediction counts: {'NOT_ENTITLED': 370, 'REFUTE': 350}
- Composed prediction counts: {'NOT_ENTITLED': 370, 'REFUTE': 350}
- Exported prediction counts: {'NOT_ENTITLED': 370, 'REFUTE': 350}

## 3. Change Counts
| Counter | Value |
|---|---:|
| changed_row_count | 0 |
| changed_row_rate | 0.0 |
| changed_to_SUPPORT_count | 0 |
| changed_to_REFUTE_count | 0 |
| changed_to_NOT_ENTITLED_count | 0 |
| REFUTE_to_SUPPORT_count | 0 |
| NOT_ENTITLED_to_SUPPORT_count | 0 |
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
| support_to_refute | 46 |

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
| stage39_enabled_count | 720 |
| stage39_changed_count | 0 |
| stage39_support_composed_count | 0 |
| stage39_refute_composed_count | 0 |
| stage39_not_entitled_composed_count | 0 |
| stage39_blocked_by_stage36_count | 0 |
| stage39_blocked_by_refute_to_support_guard_count | 0 |
| stage39_blocked_by_stage37_from_refute_guard_count | 0 |
| stage39_missing_source_count | 0 |

- stage39_composer_action_counts: {'no_change': 720}
- stage39_composer_reason_counts: {'safe_structured_v2_no_qualifying_transition': 720}

## 7. First 30 Changed Rows
None.

## 8. First 30 Unsafe Rows (total, including residual)
| Row ID | Group | Gold | Original | Composed | Action | Reason |
|---|---|---|---|---|---|---|
| library_digitization__none | none | SUPPORT | REFUTE | REFUTE | no_change | safe_structured_v2_no_qualifying_transition |
| library_digitization__paraphrase | paraphrase | SUPPORT | REFUTE | REFUTE | no_change | safe_structured_v2_no_qualifying_transition |
| glacier_study__none | none | SUPPORT | REFUTE | REFUTE | no_change | safe_structured_v2_no_qualifying_transition |
| language_program__none | none | SUPPORT | REFUTE | REFUTE | no_change | safe_structured_v2_no_qualifying_transition |
| language_program__paraphrase | paraphrase | SUPPORT | REFUTE | REFUTE | no_change | safe_structured_v2_no_qualifying_transition |
| volcano_observatory__paraphrase | paraphrase | SUPPORT | REFUTE | REFUTE | no_change | safe_structured_v2_no_qualifying_transition |
| generated_fact_034__none | none | SUPPORT | REFUTE | REFUTE | no_change | safe_structured_v2_no_qualifying_transition |
| generated_fact_034__paraphrase | paraphrase | SUPPORT | REFUTE | REFUTE | no_change | safe_structured_v2_no_qualifying_transition |
| generated_fact_051__paraphrase | paraphrase | SUPPORT | REFUTE | REFUTE | no_change | safe_structured_v2_no_qualifying_transition |
| generated_fact_052__none | none | SUPPORT | REFUTE | REFUTE | no_change | safe_structured_v2_no_qualifying_transition |
| generated_fact_052__paraphrase | paraphrase | SUPPORT | REFUTE | REFUTE | no_change | safe_structured_v2_no_qualifying_transition |
| generated_fact_057__paraphrase | paraphrase | SUPPORT | REFUTE | REFUTE | no_change | safe_structured_v2_no_qualifying_transition |
| generated_fact_058__none | none | SUPPORT | REFUTE | REFUTE | no_change | safe_structured_v2_no_qualifying_transition |
| generated_fact_058__paraphrase | paraphrase | SUPPORT | REFUTE | REFUTE | no_change | safe_structured_v2_no_qualifying_transition |
| generated_fact_064__none | none | SUPPORT | REFUTE | REFUTE | no_change | safe_structured_v2_no_qualifying_transition |
| generated_fact_064__paraphrase | paraphrase | SUPPORT | REFUTE | REFUTE | no_change | safe_structured_v2_no_qualifying_transition |
| generated_fact_078__none | none | SUPPORT | REFUTE | REFUTE | no_change | safe_structured_v2_no_qualifying_transition |
| generated_fact_078__paraphrase | paraphrase | SUPPORT | REFUTE | REFUTE | no_change | safe_structured_v2_no_qualifying_transition |
| generated_fact_097__paraphrase | paraphrase | SUPPORT | REFUTE | REFUTE | no_change | safe_structured_v2_no_qualifying_transition |
| generated_fact_098__paraphrase | paraphrase | SUPPORT | REFUTE | REFUTE | no_change | safe_structured_v2_no_qualifying_transition |
| generated_fact_108__none | none | SUPPORT | REFUTE | REFUTE | no_change | safe_structured_v2_no_qualifying_transition |
| generated_fact_108__paraphrase | paraphrase | SUPPORT | REFUTE | REFUTE | no_change | safe_structured_v2_no_qualifying_transition |
| generated_fact_114__none | none | SUPPORT | REFUTE | REFUTE | no_change | safe_structured_v2_no_qualifying_transition |
| generated_fact_114__paraphrase | paraphrase | SUPPORT | REFUTE | REFUTE | no_change | safe_structured_v2_no_qualifying_transition |
| generated_fact_118__none | none | SUPPORT | REFUTE | REFUTE | no_change | safe_structured_v2_no_qualifying_transition |
| generated_fact_118__paraphrase | paraphrase | SUPPORT | REFUTE | REFUTE | no_change | safe_structured_v2_no_qualifying_transition |
| generated_fact_124__none | none | SUPPORT | REFUTE | REFUTE | no_change | safe_structured_v2_no_qualifying_transition |
| generated_fact_124__paraphrase | paraphrase | SUPPORT | REFUTE | REFUTE | no_change | safe_structured_v2_no_qualifying_transition |
| generated_fact_126__none | none | SUPPORT | REFUTE | REFUTE | no_change | safe_structured_v2_no_qualifying_transition |
| generated_fact_126__paraphrase | paraphrase | SUPPORT | REFUTE | REFUTE | no_change | safe_structured_v2_no_qualifying_transition |

## 9. First 30 Introduced-Unsafe Rows (Stage39-caused only)
None.

## 10. Recommendation
Composition is safe but did not change any row; there is no benefit yet to justify enabling replace_pred_final_label.

## Leakage Policy
Diagnostic-only. This evaluator must not be used for training, calibration, threshold selection, loss, checkpoint selection, or Kaggle selection.
