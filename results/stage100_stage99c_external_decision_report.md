# Stage100 - Stage99C External Decision Report

## Decision

`STAGE100_REJECT_STAGE99C_AS_PRIMARY_KEEP_STAGE71`

## Candidate status

`CLEAN_OK_EXTERNAL_FAILED_SUPPORT_SUPPRESSION_WORSENED`

## Summary

| stage    | decision                                         | candidate_status                                      | current_primary_after_stage100                                | rejected_candidate                                    | best_topline_branch   | best_clean_near_miss_branch   | clean_result                 | external_result                | main_interpretation                                                                                                                                                                                                                                                                                      |   stage99c_external_acc |   stage73_external_acc |   stage99c_external_macro_f1_all3 |   stage73_external_macro_f1_all3 |   stage99c_support_recall |   stage97c_support_recall |   stage92c_support_recall |   stage73_support_recall | stage99c_prediction_counts                           | stage97c_prediction_counts                           | stage92c_prediction_counts                           | stage73_prediction_counts                            | recommendations                                                                                                                                                                                                                                                                                                                                                                                                                     |
|:---------|:-------------------------------------------------|:------------------------------------------------------|:--------------------------------------------------------------|:------------------------------------------------------|:----------------------|:------------------------------|:-----------------------------|:-------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------:|-----------------------:|----------------------------------:|---------------------------------:|--------------------------:|--------------------------:|--------------------------:|-------------------------:|:-----------------------------------------------------|:-----------------------------------------------------|:-----------------------------------------------------|:-----------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Stage100 | STAGE100_REJECT_STAGE99C_AS_PRIMARY_KEEP_STAGE71 | CLEAN_OK_EXTERNAL_FAILED_SUPPORT_SUPPRESSION_WORSENED | Stage71 retry2 Stage57+Stage66 bridge-enabled frozen recovery | Stage99C / Stage57+Stage66+Stage92A+Stage97A+Stage99A | Stage97C              | Stage92C                      | Stage99B/99C clean preserved | Stage99C external mixed/failed | Stage99A SUPPORT floor micro bridge did not restore SUPPORT recall. Instead, it pushed the external distribution further away from SUPPORT and toward REFUTE/NOT_ENTITLED. Stage99C keeps macro above Stage73 but loses accuracy and worsens SUPPORT recall relative to Stage97C, Stage92C, and Stage73. |                   0.344 |                  0.353 |                            0.3352 |                         0.326179 |                     0.316 |                     0.348 |                     0.402 |                    0.432 | {"NOT_ENTITLED": 408, "REFUTE": 314, "SUPPORT": 278} | {"NOT_ENTITLED": 394, "REFUTE": 296, "SUPPORT": 310} | {"REFUTE": 223, "NOT_ENTITLED": 412, "SUPPORT": 365} | {"REFUTE": 219, "NOT_ENTITLED": 388, "SUPPORT": 393} | ["Keep Stage71 as primary.", "Reject Stage99C as primary.", "Do not add another large synthetic bridge.", "Preserve Stage97C as diagnostic: first acc/macro top-line win but support-suppressed.", "Preserve Stage99C as negative evidence: support-floor micro failed and increased SUPPORT suppression.", "Only continue with threshold/logit-export diagnostics, not more bridge training, unless a new mechanism is designed."] |

## Comparison

| run                                  |   external_acc |   external_macro_f1_all3 |   support_recall |   refute_recall | prediction_counts                                    | status                            |
|:-------------------------------------|---------------:|-------------------------:|-----------------:|----------------:|:-----------------------------------------------------|:----------------------------------|
| Stage73 / Stage71 current primary    |          0.353 |                 0.326179 |            0.432 |        0.202817 | {"REFUTE": 219, "NOT_ENTITLED": 388, "SUPPORT": 393} | CURRENT_PRIMARY                   |
| Stage92C / Stage92A near-miss        |          0.351 |                 0.330837 |            0.402 |        0.233803 | {"REFUTE": 223, "NOT_ENTITLED": 412, "SUPPORT": 365} | NEAR_MISS                         |
| Stage97C / half anti-NE top-line win |          0.355 |                 0.343    |            0.348 |        0.3183   | {"NOT_ENTITLED": 394, "REFUTE": 296, "SUPPORT": 310} | TOPLINE_WIN_SUPPORT_SUPPRESSED    |
| Stage99C / support-floor micro       |          0.344 |                 0.3352   |            0.316 |        0.3296   | {"NOT_ENTITLED": 408, "REFUTE": 314, "SUPPORT": 278} | CLEAN_OK_EXTERNAL_MIXED_OR_FAILED |

## Delta

| comparison          |   external_acc_delta |   external_macro_delta |   support_recall_delta |   refute_recall_delta |   support_pred_delta |   refute_pred_delta |   ne_pred_delta |
|:--------------------|---------------------:|-----------------------:|-----------------------:|----------------------:|---------------------:|--------------------:|----------------:|
| Stage99C - Stage73  |               -0.009 |             0.00902129 |                 -0.116 |              0.126783 |                 -115 |                  95 |              20 |
| Stage99C - Stage97C |               -0.011 |            -0.0078     |                 -0.032 |              0.0113   |                  -32 |                  18 |              14 |
| Stage99C - Stage92C |               -0.007 |             0.004363   |                 -0.086 |              0.095797 |                  -87 |                  91 |              -4 |

## Checks

| check                               | pass   |
|:------------------------------------|:-------|
| stage99b_clean_preserved            | True   |
| stage99c_clean_preserved            | True   |
| stage99c_external_acc_ge_stage73    | False  |
| stage99c_external_macro_ge_stage73  | True   |
| stage99c_external_acc_ge_stage97c   | False  |
| stage99c_external_macro_ge_stage97c | False  |
| stage99c_support_recall_ge_stage97c | False  |
| stage99c_support_recall_ge_stage92c | False  |
| stage99c_support_pred_ge_stage97c   | False  |
| stage99c_support_pred_ge_stage92c   | False  |
| stage99c_refute_recall_ge_stage97c  | True   |
| stage99c_refute_recall_ge_stage92c  | True   |

## Diagnosis rules

| rule                                                   |
|:-------------------------------------------------------|
| stage99b_clean_preserved                               |
| stage99c_external_macro_still_above_stage73            |
| stage99c_external_accuracy_below_stage73               |
| stage99c_worse_than_stage97c_on_acc_and_macro          |
| stage99c_support_recall_worse_than_stage97c            |
| stage99c_support_prediction_mass_worse_than_stage97c   |
| stage99a_support_floor_micro_failed_to_restore_support |
| stage99a_micro_floor_increased_refute_bias_instead     |
| reject_stage99c_as_primary                             |
| keep_stage71_primary                                   |
| stop_large_bridge_append_search                        |

## Next options

| option                                  |   priority | decision                | description                                                                                                                                                             |
|:----------------------------------------|-----------:|:------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| STOP_BRIDGE_SEARCH                      |          1 | Recommended             | Stop new synthetic bridge appends. Stage95A, Stage97A, and Stage99A show the same external tradeoff: REFUTE/macro can improve, but SUPPORT recall collapses.            |
| THRESHOLD_ONLY_OR_LOGIT_EXPORT_ANALYSIS |          2 | Allowed diagnostic only | No retraining. Inspect whether logits/probs can support a clean-preserving SUPPORT floor. Requires prediction export; do not use external metrics for threshold tuning. |
| PAPER_REPORT_BRANCH                     |          3 | Useful                  | Report Stage92C/97C/99C as diagnostic evidence of an entitlement/polarity tradeoff: external macro improves but SUPPORT mass/reliability is unstable.                   |
