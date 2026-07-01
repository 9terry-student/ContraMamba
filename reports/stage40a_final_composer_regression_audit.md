# Stage40-A Integrated Final Composer Regression Audit

## 1. Overall Decision
- Run: `stage40a_final_composer_regression_audit`
- Decision: `STAGE40A_FINAL_COMPOSER_REGRESSION_PASS`
- Decision reason: Dev no-op check passes (or is explicitly allowed), Stage34 and Stage35 final composer checks both pass, and aggregate introduced unsafe SUPPORT / REFUTE<->SUPPORT counters are all zero.

## 2. Input Files
| Input | Path | Present |
|---|---|---|
| dev_report | `reports/stage39c_safe_structured_v2_dev_report.json` | yes |
| stage34_report | `reports/stage39c_safe_structured_v2_stage34_report.json` | yes |
| stage35_report | `reports/stage39c_safe_structured_v2_stage35_report.json` | yes |
| dev_predictions | `None` | no |
| stage34_predictions | `None` | no |
| stage35_predictions | `None` | no |

Missing report keys by split:

## 3. Integrated Pass/Fail Table
| Check | Result |
|---|---:|
| dev_noop_pass | True |
| stage34_final_composer_pass | True |
| stage35_final_composer_pass | True |
| introduced_safety_pass | True |
| default_off_policy_ok | True |
| integrated_pass | True |

## 4. Dev No-Op Check
- Result: `pass`
- Changed row count: 0
- Accuracy equal (within tolerance): True
- Macro-F1 equal (within tolerance): True
- introduced_unsafe_SUPPORT_total: 0
- introduced_refute_to_SUPPORT: 0
- introduced_support_to_REFUTE: 0

## 5. Stage34 Final Composer Check
- Result: `pass`
- Decision label (source report): `STAGE39C_SAFE_STRUCTURED_V2_PASS`
- Composed accuracy: 0.9600
- Composed macro-F1: 0.9703
- Changed rows: 204
- Changed to SUPPORT: 164
- Changed to REFUTE: 40
- introduced_unsafe_SUPPORT_total: 0
- introduced_refute_to_SUPPORT: 0
- introduced_support_to_REFUTE: 0

## 6. Stage35 Adversarial Final Composer Check
- Result: `pass`
- Decision label (source report): `STAGE39C_SAFE_STRUCTURED_V2_PASS`
- Composed accuracy: 0.7850
- Composed macro-F1: 0.7202
- Changed rows: 169
- Changed to SUPPORT: 150
- Changed to REFUTE: 19
- Residual support_to_refute (informational, not a Stage39 failure if introduced_support_to_REFUTE==0): 7
- introduced_unsafe_SUPPORT_total: 0
- introduced_refute_to_SUPPORT: 0
- introduced_support_to_REFUTE: 0

## 7. Introduced Safety Summary (aggregate across dev/Stage34/Stage35)
| Counter | Total |
|---|---:|
| introduced_unsafe_SUPPORT_total | 0 |
| introduced_refute_to_SUPPORT | 0 |
| introduced_support_to_REFUTE | 0 |
| introduced_overclaim_to_SUPPORT | 0 |
| introduced_exception_to_SUPPORT_error | 0 |
| introduced_location_scope_to_SUPPORT_error | 0 |
| introduced_temporal_scope_to_SUPPORT_error | 0 |
| all_zero | True |

## 8. Residual Errors Summary
- `dev`: {'support_to_refute': 46, 'refute_to_SUPPORT': 0, 'overclaim_to_SUPPORT': 0, 'exception_to_SUPPORT_error': 0, 'location_scope_to_SUPPORT_error': 0, 'temporal_scope_to_SUPPORT_error': 0}
- `stage34`: {'support_to_refute': 0, 'refute_to_SUPPORT': 0, 'overclaim_to_SUPPORT': 0, 'exception_to_SUPPORT_error': 0, 'location_scope_to_SUPPORT_error': 0, 'temporal_scope_to_SUPPORT_error': 0}
- `stage35`: {'support_to_refute': 7, 'refute_to_SUPPORT': 0, 'overclaim_to_SUPPORT': 0, 'exception_to_SUPPORT_error': 0, 'location_scope_to_SUPPORT_error': 0, 'temporal_scope_to_SUPPORT_error': 0}
- Residual safety_counters reflect total (pre-existing + Stage39) errors inherited from the original model; they are informational only and are not treated as Stage39-introduced regressions when the corresponding introduced_* counters are zero.

## 9. Default-Off Policy Note
- Status: `policy_confirmed`
- policy_confirmed: True
- This audit only reads Stage39-C evaluator reports; it does not invoke or compare against a default-off run. Default-off behavior must be confirmed at the training-script flag level: Stage39 requires explicit --stage39-use-final-composer-opt-in, and replacing predictions requires --stage39-final-composer-output-mode replace_pred_final_label. The default final composer remains off by design and final logits/loss/training are unchanged by composition.

## 10. Metric Summary Table
| Split | Decision | Orig Acc | Comp Acc | Orig F1 | Comp F1 | d Acc | d F1 | Changed | ->SUPPORT | ->REFUTE | IntroUnsafe | IntroR2S | IntroS2R |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dev | STAGE39A_FINAL_COMPOSER_TOO_WEAK | 0.5611 | 0.5611 | 0.3617 | 0.3617 | 0.0000 | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |
| stage34 | STAGE39C_SAFE_STRUCTURED_V2_PASS | 0.4500 | 0.9600 | 0.2069 | 0.9703 | 0.5100 | 0.7634 | 204 | 164 | 40 | 0 | 0 | 0 |
| stage35 | STAGE39C_SAFE_STRUCTURED_V2_PASS | 0.5033 | 0.7850 | 0.2832 | 0.7202 | 0.2817 | 0.4370 | 169 | 150 | 19 | 0 | 0 | 0 |

## 11. Recommendation
Stage39-C safe_structured_v2 is safe as an explicit opt-in final-prediction replacement under the evaluated profiles. It must remain off by default. The probes/reports used here must not be used for training, calibration, threshold selection, checkpoint selection, or loss design. Further external/naturalistic validation is still required before claiming broad production robustness.

## 12. Leakage Policy
Diagnostic/aggregation-only. This audit reads existing Stage39-C reports; it does not run training, evaluation, or Kaggle, and must not be used for training, calibration, threshold selection, checkpoint selection, or loss design.
