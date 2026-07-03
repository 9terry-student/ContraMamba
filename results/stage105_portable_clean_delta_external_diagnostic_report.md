# Stage105 - Portable Clean-delta External Diagnostic

## Decision

`STAGE105_PORTABLE_CLEAN_DELTA_EXTERNAL_DIAGNOSTIC_WEAK_OR_NEGATIVE`

## Summary

| stage    | decision                                                           | analysis_type                                                    | current_primary_after_stage105                                | promotion_status                              | source_policy                                                                                                                                       | clean_selected_policy_name                         |   clean_selected_changed_from_stage92c | portable_external_rule                                                                          |   stage73_external_acc |   stage73_external_macro |   stage73_support_recall |   stage92c_external_acc |   stage92c_external_macro |   stage92c_support_recall |   stage105_external_acc |   stage105_external_macro |   stage105_support_recall |   stage105_refute_recall | stage105_prediction_counts                           |   stage105_false_NE_total |   stage105_false_entitlement_total | main_interpretation                                                                                                                                                                                                                           | recommended_next_stage                        |
|:---------|:-------------------------------------------------------------------|:-----------------------------------------------------------------|:--------------------------------------------------------------|:----------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------|---------------------------------------:|:------------------------------------------------------------------------------------------------|-----------------------:|-------------------------:|-------------------------:|------------------------:|--------------------------:|--------------------------:|------------------------:|--------------------------:|--------------------------:|-------------------------:|:-----------------------------------------------------|--------------------------:|-----------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------|
| Stage105 | STAGE105_PORTABLE_CLEAN_DELTA_EXTERNAL_DIAGNOSTIC_WEAK_OR_NEGATIVE | optional_external_diagnostic_using_clean_selected_portable_delta | Stage71 retry2 Stage57+Stage66 bridge-enabled frozen recovery | NOT_PROMOTABLE_WITHOUT_INDEPENDENT_VALIDATION | Uses Stage104B clean-only selected nontrivial policy, translated into portable delta-k. No training, no new external run, no external label tuning. | Stage92C clean_rank_floor target=108 scope=ne_only |                                      2 | Among Stage92C external NOT_ENTITLED predictions, flip top 2 by SUPPORT probability to SUPPORT. |                  0.353 |                 0.326179 |                    0.432 |                   0.351 |                  0.330837 |                     0.402 |                   0.353 |                  0.332307 |                     0.406 |                 0.233803 | {"REFUTE": 223, "NOT_ENTITLED": 410, "SUPPORT": 367} |                       343 |                                 78 | Stage105 tests whether the very small nontrivial clean-only delta generalizes directionally to external predictions. Because Stage104B found no strict nontrivial candidate and only a Stage71-preserving candidate, this is diagnostic only. | Stage106 final synthesis and commit selection |

## Comparison

| run                                                      |   accuracy |   macro_f1_all3 |   support_recall |   refute_recall |   ne_recall |   support_pred_count |   refute_pred_count |   ne_pred_count |   false_NE_total |   false_entitlement_total |   polarity_error_total | status                                     |
|:---------------------------------------------------------|-----------:|----------------:|-----------------:|----------------:|------------:|---------------------:|--------------------:|----------------:|-----------------:|--------------------------:|-----------------------:|:-------------------------------------------|
| Stage73 external baseline                                |      0.353 |        0.326179 |            0.432 |        0.202817 |    0.448276 |                  393 |                 219 |             388 |              323 |                        80 |                    244 | CURRENT_PRIMARY_EXTERNAL                   |
| Stage92C external baseline                               |      0.351 |        0.330837 |            0.402 |        0.233803 |    0.462069 |                  365 |                 223 |             412 |              345 |                        78 |                    226 | STAGE92C_BASELINE_EXTERNAL                 |
| Stage105 apply clean-only portable delta NE->SUPPORT k=2 |      0.353 |        0.332307 |            0.406 |        0.233803 |    0.462069 |                  367 |                 223 |             410 |              343 |                        78 |                    226 | CLEAN_ONLY_DELTA_DIAGNOSTIC_NOT_PROMOTABLE |

## Delta

| comparison                 |   acc_delta |   macro_delta |   support_recall_delta |   refute_recall_delta |   false_NE_total_delta |   false_entitlement_total_delta |
|:---------------------------|------------:|--------------:|-----------------------:|----------------------:|-----------------------:|--------------------------------:|
| Stage105 policy - Stage73  |       0     |    0.006128   |                 -0.026 |             0.0309859 |                     20 |                              -2 |
| Stage105 policy - Stage92C |       0.002 |    0.00146949 |                  0.004 |             0         |                     -2 |                               0 |

## Checks

| check                                                  | pass   |
|:-------------------------------------------------------|:-------|
| stage104b_has_nontrivial_stage71_only_candidate        | True   |
| portable_delta_k_positive                              | True   |
| external_changed_count_matches_delta                   | True   |
| stage105_acc_ge_stage73                                | True   |
| stage105_macro_ge_stage73                              | True   |
| stage105_support_recall_ge_stage73                     | False  |
| stage105_refute_recall_ge_stage92c_minus_tiny          | True   |
| stage105_not_promotable_without_independent_validation | True   |

## Changed examples

|   row_index | id                                                   | gold    | baseline_pred   | policy_pred   | baseline_correct   | policy_correct   |   support_prob |   refute_prob |   ne_prob |   support_margin |   support_ne_gap | claim                                                                 | evidence                                                                                                                                  |
|------------:|:-----------------------------------------------------|:--------|:----------------|:--------------|:-------------------|:-----------------|---------------:|--------------:|----------:|-----------------:|-----------------:|:----------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------|
|         549 | tals_vitaminc_validation_sample1000_validation_21977 | SUPPORT | NOT_ENTITLED    | SUPPORT       | False              | True             |       0.401245 |      0.176856 |  0.4219   |      -0.0206549  |       0.0206549  | Mario Götze is Christian .                                            | Götze is a Christian.                                                                                                                     |
|         209 | tals_vitaminc_validation_sample1000_validation_35442 | SUPPORT | NOT_ENTITLED    | SUPPORT       | False              | True             |       0.398793 |      0.196257 |  0.404951 |      -0.00615811 |       0.00615811 | Breaking Benjamin released three promotional songs before the album . | Three subsequent songs , `` Blood '' , `` Psycho '' , and `` Save Yourself '' , were released ahead of the album as promotional releases. |
