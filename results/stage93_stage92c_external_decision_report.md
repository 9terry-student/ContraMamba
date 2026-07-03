# Stage93 - Stage92C External Decision Report

## Decision

`STAGE93_REJECT_STAGE92C_AS_PRIMARY_KEEP_STAGE71`

## Candidate status

`NEAR_MISS_SUPPORT_RECOVERY_NOT_PRIMARY`

## Summary

| stage   | decision                                        | candidate_status                       | current_primary_after_stage93                                 | candidate                           | clean_preserved   | beats_stage73_required   | near_miss   | support_refute_overcorrection_repaired   | false_ne_still_bad   |   stage92c_external_acc |   stage73_external_acc |   stage92c_external_macro_f1_all3 |   stage73_external_macro_f1_all3 |   stage92c_support_recall |   stage73_support_recall |   stage88c_support_recall |   stage92c_false_NE_total |   stage73_false_NE_total |   stage88c_false_NE_total | interpretation                                                                                                                                                                                                                                                                                                                                                                                                                                    | recommended_next_stage                                | recommendations                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
|:--------|:------------------------------------------------|:---------------------------------------|:--------------------------------------------------------------|:------------------------------------|:------------------|:-------------------------|:------------|:-----------------------------------------|:---------------------|------------------------:|-----------------------:|----------------------------------:|---------------------------------:|--------------------------:|-------------------------:|--------------------------:|--------------------------:|-------------------------:|--------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Stage93 | STAGE93_REJECT_STAGE92C_AS_PRIMARY_KEEP_STAGE71 | NEAR_MISS_SUPPORT_RECOVERY_NOT_PRIMARY | Stage71 retry2 Stage57+Stage66 bridge-enabled frozen recovery | Stage92C / Stage57+Stage66+Stage92A | True              | False                    | True        | True                                     | True                 |                   0.351 |                  0.353 |                          0.330837 |                         0.326179 |                     0.402 |                    0.432 |                     0.324 |                       345 |                      323 |                       349 | Stage92A/92C is a strong near-miss branch. It preserves clean-dev, restores SUPPORT prediction mass relative to Stage88C, substantially repairs SUPPORT-to-REFUTE overcorrection, and improves external macro-F1 over Stage71. However, it still fails Stage71 on external accuracy and false_NE_total. The remaining failure is not primarily REFUTE overcorrection; it is residual NOT_ENTITLED absorption of entitled SUPPORT/REFUTE examples. | Stage94 anti-NE entitlement preservation audit/design | ["Keep Stage71 retry2 Stage57+Stage66 as current primary.", "Do not promote Stage92C because external accuracy remains below Stage73/Stage71 and false_NE_total remains above Stage73.", "Retain Stage92A/92B/92C as the best near-miss branch so far: it repaired Stage88C's SUPPORT-to-REFUTE overcorrection.", "Next stage should not add generic SUPPORT pressure or REFUTE pressure.", "Next stage should target residual false_NE absorption under sufficient evidence, especially SUPPORT/REFUTE examples predicted as NOT_ENTITLED.", "Use Stage94 as an anti-NE entitlement preservation design/audit before generating any new bridge."] |

## Comparison

| run                                                       |   external_acc |   external_macro_f1_all3 | prediction_counts                                    |   support_recall |   refute_recall |   false_NE_total |   false_entitlement_total |   false_REFUTE_on_SUPPORT |   false_SUPPORT_on_REFUTE | status                |
|:----------------------------------------------------------|---------------:|-------------------------:|:-----------------------------------------------------|-----------------:|----------------:|-----------------:|--------------------------:|--------------------------:|--------------------------:|:----------------------|
| Stage73 / Stage71 primary external                        |          0.353 |                 0.326179 | {"REFUTE": 219, "NOT_ENTITLED": 388, "SUPPORT": 393} |            0.432 |        0.202817 |              323 |                        80 |                       106 |                       138 | CURRENT_PRIMARY       |
| Stage88C / Stage88A mixed external                        |          0.338 |                 0.329458 | {"REFUTE": 283, "NOT_ENTITLED": 423, "SUPPORT": 294} |            0.324 |        0.287324 |              349 |                        71 |                       141 |                       101 | MIXED_BRANCH          |
| Stage92C / Stage92A support-preserving candidate external |          0.351 |                 0.330837 | {"REFUTE": 223, "NOT_ENTITLED": 412, "SUPPORT": 365} |            0.402 |        0.233803 |              345 |                        78 |                       107 |                       119 | NEAR_MISS_NOT_PRIMARY |

## Delta

| comparison          |   external_acc_delta |   external_macro_delta |   support_recall_delta |   refute_recall_delta |   false_NE_total_delta |   false_entitlement_total_delta |   false_REFUTE_on_SUPPORT_delta |   false_SUPPORT_on_REFUTE_delta |
|:--------------------|---------------------:|-----------------------:|-----------------------:|----------------------:|-----------------------:|--------------------------------:|--------------------------------:|--------------------------------:|
| Stage92C - Stage73  |               -0.002 |             0.0046585  |                 -0.03  |             0.0309858 |                     22 |                              -2 |                               1 |                             -19 |
| Stage92C - Stage88C |                0.013 |             0.00137952 |                  0.078 |            -0.0535212 |                     -4 |                               7 |                             -34 |                              18 |

## Checks

| check                                        | pass   |
|:---------------------------------------------|:-------|
| clean_preserved_vs_stage71                   | True   |
| external_acc_ge_stage73                      | False  |
| external_macro_ge_stage73                    | True   |
| false_NE_total_lte_stage73                   | False  |
| support_recall_ge_stage73_minus_0p02         | False  |
| support_recall_gt_stage88c                   | True   |
| false_REFUTE_on_SUPPORT_repaired_vs_stage88c | True   |
| false_REFUTE_on_SUPPORT_near_stage73         | True   |
| near_miss                                    | True   |
| beats_stage73_required                       | False  |

## Recommendations

|   idx | recommendation                                                                                                                                 |
|------:|:-----------------------------------------------------------------------------------------------------------------------------------------------|
|     1 | Keep Stage71 retry2 Stage57+Stage66 as current primary.                                                                                        |
|     2 | Do not promote Stage92C because external accuracy remains below Stage73/Stage71 and false_NE_total remains above Stage73.                      |
|     3 | Retain Stage92A/92B/92C as the best near-miss branch so far: it repaired Stage88C's SUPPORT-to-REFUTE overcorrection.                          |
|     4 | Next stage should not add generic SUPPORT pressure or REFUTE pressure.                                                                         |
|     5 | Next stage should target residual false_NE absorption under sufficient evidence, especially SUPPORT/REFUTE examples predicted as NOT_ENTITLED. |
|     6 | Use Stage94 as an anti-NE entitlement preservation design/audit before generating any new bridge.                                              |
