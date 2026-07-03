# Stage79 - Stage75 Bridge Decision Report

## Decision

`STAGE79_STAGE75_BRIDGE_DECISION_REPORT_READY`

## Stage75 bridge decision

`REVISE_NOT_DEFAULT`

## Summary

| stage   | decision                                     | stage75_bridge_decision   | decision_reason                                                                                                                                                                                                                                                                  |   stage73_external_accuracy |   stage73_external_macro_f1 |   stage77_external_accuracy |   stage77_external_macro_f1 |   stage77_minus_stage73_accuracy |   stage77_minus_stage73_macro_f1 |   stage78_minus_stage74_false_ne_total |   stage78_minus_stage74_polarity_error_total |   stage78_minus_stage74_false_entitlement_total |   stage78_minus_stage74_false_support_total |   stage78_minus_stage74_false_refute_total | clean_dev_preserved   | training_executed   | external_eval_executed   | recommended_next_stage                                         |
|:--------|:---------------------------------------------|:--------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------:|----------------------------:|----------------------------:|----------------------------:|---------------------------------:|---------------------------------:|---------------------------------------:|---------------------------------------------:|------------------------------------------------:|--------------------------------------------:|-------------------------------------------:|:----------------------|:--------------------|:-------------------------|:---------------------------------------------------------------|
| Stage79 | STAGE79_STAGE75_BRIDGE_DECISION_REPORT_READY | REVISE_NOT_DEFAULT        | Stage75 preserves clean-dev and slightly improves external accuracy, but external macro-F1 drops versus Stage73 and false SUPPORT / false entitlement increase. Keep Stage71_retry2 as primary; use Stage75 only as diagnostic evidence for a smaller Stage80 targeted revision. |                       0.353 |                      0.3262 |                       0.357 |                      0.3182 |                            0.004 |                           -0.008 |                                      1 |                                          -11 |                                               6 |                                          19 |                                        -24 | True                  | False               | False                    | Stage80A design conservative Stage75v2 bridge or ablation plan |

## Checks

| check                            | pass   |
|:---------------------------------|:-------|
| stage73b2_exists                 | True   |
| stage74_exists                   | True   |
| stage76_exists                   | True   |
| stage77_exists                   | True   |
| stage78_exists                   | True   |
| stage76_clean_dev_preserved      | True   |
| stage77_external_metrics_present | True   |
| stage78_error_audit_ready        | True   |
| stage79_decision_nonempty        | True   |

## Signals

| signal                                 | value   |
|:---------------------------------------|:--------|
| clean_dev_preserved                    | True    |
| external_accuracy_improved_vs_stage73  | True    |
| external_macro_f1_improved_vs_stage73  | False   |
| polarity_error_reduced_vs_stage74      | True    |
| false_refute_reduced_vs_stage74        | True    |
| false_support_increased_vs_stage74     | True    |
| false_entitlement_increased_vs_stage74 | True    |
| false_ne_not_reduced_vs_stage74        | True    |

## External metrics

| system                      |   external_accuracy |   external_macro_f1 | note                                                       |
|:----------------------------|--------------------:|--------------------:|:-----------------------------------------------------------|
| Stage73_retry_Stage57_66    |               0.353 |              0.3262 | previous best external diagnostic before Stage75           |
| Stage77_Stage57_66_75       |               0.357 |              0.3182 | Stage75F configuration external diagnostic                 |
| Delta_Stage77_minus_Stage73 |               0.004 |             -0.008  | positive accuracy but negative macro-F1 means mixed result |

## Clean-dev preservation

| metric                          |       value | pass   |
|:--------------------------------|------------:|:-------|
| Stage75F_minus_Stage71_acc      | -0.00138891 | True   |
| Stage75F_minus_Stage71_macro_f1 | -0.00184226 | True   |
| metadata_checks_pass            |  1          | True   |
| preservation_checks_pass        |  1          | True   |

## Error delta: Stage78 minus Stage74

| metric                  |   stage74_stage73_retry |   stage78_stage77_stage75f |   delta_stage78_minus_stage74 |
|:------------------------|------------------------:|---------------------------:|------------------------------:|
| accuracy                |                   0.353 |                      0.357 |                         0.004 |
| correct_total           |                 353     |                    357     |                         4     |
| error_total             |                 647     |                    643     |                        -4     |
| false_ne_total          |                 323     |                    324     |                         1     |
| polarity_error_total    |                 244     |                    233     |                       -11     |
| false_entitlement_total |                  80     |                     86     |                         6     |
| false_support_total     |                 177     |                    196     |                        19     |
| false_refute_total      |                 147     |                    123     |                       -24     |

## Recommendations

| recommendation                                                       | rationale                                                                                         |
|:---------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------|
| Do not make Stage75 full bridge the default.                         | Stage77 macro-F1 is lower than Stage73 despite slightly higher accuracy.                          |
| Keep Stage71_retry2 as the current primary checkpoint/config.        | Stage71 has better external macro-F1 and slightly better clean-dev macro-F1.                      |
| Use Stage75 error movement as diagnostic signal.                     | Stage75 reduced polarity_error_total and false_refute_total but increased false_support_total.    |
| Next revision should reduce false SUPPORT pressure.                  | Stage78 shows false_support_total rose from 177 to 196 and false_entitlement_total from 80 to 86. |
| Prioritize conservative polarity repair over broad SUPPORT recovery. | Stage75's best signal was polarity_error_total -11 and false_refute_total -24.                    |

## Decision reason

Stage75 preserves clean-dev and slightly improves external accuracy, but external macro-F1 drops versus Stage73 and false SUPPORT / false entitlement increase. Keep Stage71_retry2 as primary; use Stage75 only as diagnostic evidence for a smaller Stage80 targeted revision.

## Recommended next stage

Stage80A design conservative Stage75v2 bridge or ablation plan
