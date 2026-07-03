# Stage85 - Stage83C External Decision Report

## Decision

`STAGE85_REJECT_STAGE83C_KEEP_STAGE71_PRIMARY`

## Candidate status

`CLEAN_DEV_GAIN_NOT_EXTERNAL_ROBUST`

## Summary

| stage   | decision                                     | candidate_status                   | current_primary_after_stage85                                 | rejected_candidate                                          | clean_reproduced   | external_regression   |   stage84f_clean_dev_acc |   stage84f_clean_dev_macro_f1 |   stage83c_clean_dev_acc |   stage83c_clean_dev_macro_f1 |   stage84f_external_accuracy |   stage84f_external_macro_f1_all3 |   stage73_external_accuracy |   stage73_external_macro_f1_all3 |   stage84f_minus_stage73_external_acc |   stage84f_minus_stage73_external_macro_f1_all3 | stage84f_external_prediction_counts                  | stage73_external_prediction_counts                   | stage84e2_preflight_decision                | stage84e2_encode_mamba_records_full_ok   | interpretation                                                                                                                                                                                                                                                  | recommendations                                                                                                                                                                                                                                                                                                                                                                                                                  |
|:--------|:---------------------------------------------|:-----------------------------------|:--------------------------------------------------------------|:------------------------------------------------------------|:-------------------|:----------------------|-------------------------:|------------------------------:|-------------------------:|------------------------------:|-----------------------------:|----------------------------------:|----------------------------:|---------------------------------:|--------------------------------------:|------------------------------------------------:|:-----------------------------------------------------|:-----------------------------------------------------|:--------------------------------------------|:-----------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Stage85 | STAGE85_REJECT_STAGE83C_KEEP_STAGE71_PRIMARY | CLEAN_DEV_GAIN_NOT_EXTERNAL_ROBUST | Stage71 retry2 Stage57+Stage66 bridge-enabled frozen recovery | Stage83C / Stage84F Stage57+Stage66+Stage83A NE-safety-only | True               | True                  |                 0.979167 |                      0.969664 |                 0.979167 |                      0.969664 |                        0.326 |                          0.306672 |                       0.353 |                           0.3262 |                                -0.027 |                                      -0.0195283 | {"NOT_ENTITLED": 439, "REFUTE": 202, "SUPPORT": 359} | {"NOT_ENTITLED": 388, "REFUTE": 219, "SUPPORT": 393} | STAGE84E2_EXACT_OOD_ENCODER_PREFLIGHT_READY | True                                     | Stage83C/A3 NE-safety-only clean-dev improvement does not transfer to VitaminC external. The external prediction distribution shifts further toward NOT_ENTITLED relative to Stage73, and both external accuracy and macro-F1 regress. Keep Stage71 as primary. | ["Keep Stage71 retry2 Stage57+Stage66 bridge-enabled frozen recovery as current primary.", "Do not promote Stage83C despite clean-dev gain.", "Treat Stage83C and Stage84F as a valid negative branch result.", "Do not add more NE-safety bridge rows by default.", "Do not revive full Stage80A/Stage75 full bridge.", "Next useful direction should target external entitlement/polarity balance, not additional NE safety."] |

## Comparison

| run                                    |   clean_dev_acc |   clean_dev_macro_f1 |   external_acc |   external_macro_f1_all3 | prediction_counts                                    | status                   |
|:---------------------------------------|----------------:|---------------------:|---------------:|-------------------------:|:-----------------------------------------------------|:-------------------------|
| Stage73 / Stage71 primary external     |      nan        |           nan        |          0.353 |                 0.3262   | {"NOT_ENTITLED": 388, "REFUTE": 219, "SUPPORT": 393} | CURRENT_PRIMARY_BASELINE |
| Stage84F / Stage83C candidate external |        0.979167 |             0.969664 |          0.326 |                 0.306672 | {"NOT_ENTITLED": 439, "REFUTE": 202, "SUPPORT": 359} | REJECT_PRIMARY_PROMOTION |

## Delta

| comparison         |   external_acc_delta |   external_macro_f1_all3_delta | decision   |
|:-------------------|---------------------:|-------------------------------:|:-----------|
| Stage84F - Stage73 |               -0.027 |                     -0.0195283 | REGRESSION |

## Recommendations

|   idx | recommendation                                                                                       |
|------:|:-----------------------------------------------------------------------------------------------------|
|     1 | Keep Stage71 retry2 Stage57+Stage66 bridge-enabled frozen recovery as current primary.               |
|     2 | Do not promote Stage83C despite clean-dev gain.                                                      |
|     3 | Treat Stage83C and Stage84F as a valid negative branch result.                                       |
|     4 | Do not add more NE-safety bridge rows by default.                                                    |
|     5 | Do not revive full Stage80A/Stage75 full bridge.                                                     |
|     6 | Next useful direction should target external entitlement/polarity balance, not additional NE safety. |
