# Stage89 - Stage88C External Decision Report

## Decision

`STAGE89_REJECT_STAGE88C_AS_PRIMARY_KEEP_STAGE71`

## Candidate status

`USEFUL_PARTIAL_RECOVERY_NOT_PRIMARY`

## Summary

| stage   | decision                                        | candidate_status                    | current_primary_after_stage89                                 | candidate                           | clean_preserved   | improves_stage84f   | beats_stage73_external   | macro_only_win_vs_stage73   |   stage88c_external_acc |   stage88c_external_macro_f1_all3 |   stage73_external_acc |   stage73_external_macro_f1_all3 |   stage84f_external_acc |   stage84f_external_macro_f1_all3 |   stage88c_false_NE_total |   stage73_false_NE_total |   stage84f_false_NE_total | interpretation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | recommended_next_stage                                                        | recommendations                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
|:--------|:------------------------------------------------|:------------------------------------|:--------------------------------------------------------------|:------------------------------------|:------------------|:--------------------|:-------------------------|:----------------------------|------------------------:|----------------------------------:|-----------------------:|---------------------------------:|------------------------:|----------------------------------:|--------------------------:|-------------------------:|--------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Stage89 | STAGE89_REJECT_STAGE88C_AS_PRIMARY_KEEP_STAGE71 | USEFUL_PARTIAL_RECOVERY_NOT_PRIMARY | Stage71 retry2 Stage57+Stage66 bridge-enabled frozen recovery | Stage88C / Stage57+Stage66+Stage88A | True              | True                | False                    | True                        |                   0.338 |                          0.329458 |                  0.353 |                         0.326179 |                   0.326 |                          0.306672 |                       349 |                      323 |                       367 | Stage88A balanced entitlement recovery preserved clean-dev and partially repaired the Stage83C/Stage84F failure mode, improving external macro-F1 and reducing false_NE_total relative to Stage84F. However, it does not beat Stage71 on external accuracy and still has higher false_NE_total than Stage71. The prediction distribution shifts too far away from SUPPORT, improving REFUTE recall but damaging SUPPORT recall. Therefore Stage71 remains primary, and Stage88C is retained as a useful negative/mixed branch. | Stage90 support-preserving entitlement recovery / REFUTE-overcorrection audit | ["Keep Stage71 retry2 Stage57+Stage66 as current primary.", "Do not promote Stage88C despite clean-dev preservation and macro-F1 improvement.", "Retain Stage88A/88B/88C as a useful mixed branch: it improves over Stage84F but not over Stage71.", "Do not add more generic entitlement bridge rows at the same balance.", "Next stage should diagnose SUPPORT suppression caused by Stage88A, especially SUPPORT→REFUTE and SUPPORT→NOT_ENTITLED errors.", "Future bridge should preserve SUPPORT recall while keeping REFUTE recovery gains and false-entitlement guardrails."] |

## Comparison

| run                                    |   external_acc |   external_macro_f1_all3 | prediction_counts                                    |   false_NE_total |   false_entitlement_total |   polarity_error_total | status            |
|:---------------------------------------|---------------:|-------------------------:|:-----------------------------------------------------|-----------------:|--------------------------:|-----------------------:|:------------------|
| Stage73 / Stage71 primary external     |          0.353 |                 0.326179 | {"REFUTE": 219, "NOT_ENTITLED": 388, "SUPPORT": 393} |              323 |                        80 |                    244 | CURRENT_PRIMARY   |
| Stage84F / Stage83C rejected external  |          0.326 |                 0.306672 | {"REFUTE": 202, "NOT_ENTITLED": 439, "SUPPORT": 359} |              367 |                        73 |                    234 | REJECTED_BRANCH   |
| Stage88C / Stage88A candidate external |          0.338 |                 0.329458 | {"REFUTE": 283, "NOT_ENTITLED": 423, "SUPPORT": 294} |              349 |                        71 |                    242 | MIXED_NOT_PRIMARY |

## Delta

| comparison          |   acc_delta |   macro_f1_delta |   false_NE_total_delta |   false_entitlement_total_delta |   polarity_error_total_delta |
|:--------------------|------------:|-----------------:|-----------------------:|--------------------------------:|-----------------------------:|
| Stage88C - Stage73  |      -0.015 |       0.00327898 |                     26 |                              -9 |                           -2 |
| Stage88C - Stage84F |       0.012 |       0.022786   |                    -18 |                              -2 |                            8 |

## Checks

| check                               | pass   |
|:------------------------------------|:-------|
| clean_preserved_vs_stage71          | True   |
| improves_stage84f_external          | True   |
| beats_stage73_external_all_required | False  |
| macro_only_win_vs_stage73           | True   |
| external_acc_below_stage73          | True   |
| false_NE_above_stage73              | True   |

## Recommendations

|   idx | recommendation                                                                                                                |
|------:|:------------------------------------------------------------------------------------------------------------------------------|
|     1 | Keep Stage71 retry2 Stage57+Stage66 as current primary.                                                                       |
|     2 | Do not promote Stage88C despite clean-dev preservation and macro-F1 improvement.                                              |
|     3 | Retain Stage88A/88B/88C as a useful mixed branch: it improves over Stage84F but not over Stage71.                             |
|     4 | Do not add more generic entitlement bridge rows at the same balance.                                                          |
|     5 | Next stage should diagnose SUPPORT suppression caused by Stage88A, especially SUPPORT→REFUTE and SUPPORT→NOT_ENTITLED errors. |
|     6 | Future bridge should preserve SUPPORT recall while keeping REFUTE recovery gains and false-entitlement guardrails.            |
