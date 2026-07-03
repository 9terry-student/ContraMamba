# Stage98 - Stage97C External Decision Report

## Decision

`STAGE98_REJECT_STAGE97C_AS_PRIMARY_KEEP_STAGE71`

## Candidate status

`TOPLINE_WIN_BUT_SUPPORT_SUPPRESSION_NOT_PRIMARY`

## Summary

| stage   | decision                                        | candidate_status                                | current_primary_after_stage98                                 | candidate                                    | clean_preserved   | beats_stage73_required   | topline_beats_stage73   | support_suppression_blocks_promotion   |   stage97c_external_acc |   stage73_external_acc |   stage97c_external_macro_f1_all3 |   stage73_external_macro_f1_all3 |   stage97c_support_recall |   stage73_support_recall |   stage92c_support_recall | stage97c_prediction_counts                           | stage73_prediction_counts                            | stage92c_prediction_counts                           | main_interpretation                                                                                                                                                                                                                    | recommended_next_stage                                        | recommendations                                                                                                                                                                                                                                                                                                                                                                              |
|:--------|:------------------------------------------------|:------------------------------------------------|:--------------------------------------------------------------|:---------------------------------------------|:------------------|:-------------------------|:------------------------|:---------------------------------------|------------------------:|-----------------------:|----------------------------------:|---------------------------------:|--------------------------:|-------------------------:|--------------------------:|:-----------------------------------------------------|:-----------------------------------------------------|:-----------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Stage98 | STAGE98_REJECT_STAGE97C_AS_PRIMARY_KEEP_STAGE71 | TOPLINE_WIN_BUT_SUPPORT_SUPPRESSION_NOT_PRIMARY | Stage71 retry2 Stage57+Stage66 bridge-enabled frozen recovery | Stage97C / Stage57+Stage66+Stage92A+Stage97A | True              | False                    | True                    | True                                   |                   0.355 |                  0.353 |                             0.343 |                         0.326179 |                     0.348 |                    0.432 |                     0.402 | {"NOT_ENTITLED": 394, "REFUTE": 296, "SUPPORT": 310} | {"REFUTE": 219, "NOT_ENTITLED": 388, "SUPPORT": 393} | {"REFUTE": 223, "NOT_ENTITLED": 412, "SUPPORT": 365} | Stage97C is the first branch to beat Stage73 on both external accuracy and macro-F1, but it does so by shifting mass away from SUPPORT. SUPPORT recall drops below both Stage73 and Stage92C, so it is not safe to promote as primary. | Stage99 support-floor micro design or threshold-only analysis | ["Keep Stage71 as current primary.", "Do not promote Stage97C despite external acc/macro top-line win.", "Retain Stage97C as an important diagnostic branch: half-size numeric/date anti-NE improves macro and REFUTE recall.", "Next intervention must restore SUPPORT recall/mass without adding broad SUPPORT pressure that breaks clean NE.", "Do not run another large bridge append."] |

## Comparison

| run                                                |   external_acc |   external_macro_f1_all3 |   support_recall |   refute_recall | prediction_counts                                    | status                                          |
|:---------------------------------------------------|---------------:|-------------------------:|-----------------:|----------------:|:-----------------------------------------------------|:------------------------------------------------|
| Stage73 / Stage71 primary external                 |          0.353 |                 0.326179 |            0.432 |        0.202817 | {"REFUTE": 219, "NOT_ENTITLED": 388, "SUPPORT": 393} | CURRENT_PRIMARY                                 |
| Stage92C / Stage92A near-miss external             |          0.351 |                 0.330837 |            0.402 |        0.233803 | {"REFUTE": 223, "NOT_ENTITLED": 412, "SUPPORT": 365} | BEST_NEAR_MISS_BEFORE_STAGE97                   |
| Stage97C / Stage92A+Stage97A half anti-NE external |          0.355 |                 0.343    |            0.348 |        0.3183   | {"NOT_ENTITLED": 394, "REFUTE": 296, "SUPPORT": 310} | TOPLINE_WIN_BUT_SUPPORT_SUPPRESSION_NOT_PRIMARY |

## Delta

| comparison          |   external_acc_delta |   external_macro_delta |   support_recall_delta |   refute_recall_delta |   support_pred_count_delta |   refute_pred_count_delta |   ne_pred_count_delta |
|:--------------------|---------------------:|-----------------------:|-----------------------:|----------------------:|---------------------------:|--------------------------:|----------------------:|
| Stage97C - Stage73  |                0.002 |              0.0168213 |                 -0.084 |              0.115483 |                        -83 |                        77 |                     6 |
| Stage97C - Stage92C |                0.004 |              0.012163  |                 -0.054 |              0.084497 |                        -55 |                        73 |                   -18 |

## Promotion checks

| check                                                 | pass   |
|:------------------------------------------------------|:-------|
| clean_preserved                                       | True   |
| external_acc_ge_stage73                               | True   |
| external_macro_ge_stage73                             | True   |
| support_recall_ge_stage73_minus_0p02                  | False  |
| support_recall_ge_stage92c                            | False  |
| support_pred_count_ge_stage92c_minus_20               | False  |
| external_prediction_file_available_for_exact_false_NE | False  |

## Diagnosis rules

| rule                                                |
|:----------------------------------------------------|
| stage97c_external_accuracy_beats_stage73            |
| stage97c_external_macro_beats_stage73               |
| stage97c_fails_support_recall_guardrail             |
| stage97c_worse_than_stage92c_on_support_recall      |
| stage97c_support_prediction_mass_too_low            |
| do_not_promote_without_exact_false_NE_breakdown     |
| keep_stage71_primary                                |
| retain_stage92c_and_stage97c_as_diagnostic_branches |

## Next options

| option                              |   priority | description                                                                                                                                                                       |
|:------------------------------------|-----------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Stage99A_support_floor_micro_bridge |          1 | Add a very small SUPPORT-preservation floor to Stage97A: e.g. SUPPORT 16, NE guardrail 16, no extra REFUTE. Goal is to restore SUPPORT recall without losing Stage97C macro gain. |
| Stage99B_threshold_only_analysis    |          2 | No retraining. Inspect whether the runner can export logits/probs and apply a clean-preserving SUPPORT floor or NE margin rule. Safer if prediction probabilities are available.  |
| Stage99C_stop_branch_and_report     |          3 | Stop recovery-bridge search and report Stage92C/97C as diagnostic evidence: macro can be improved, but SUPPORT recall tradeoff blocks primary promotion.                          |
