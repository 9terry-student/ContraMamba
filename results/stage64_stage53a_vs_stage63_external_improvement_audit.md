# Stage64 — Stage53A vs Stage63 External Improvement Audit

## Decision

`STAGE64_EXTERNAL_IMPROVEMENT_AUDIT_READY_INCOMPLETE`

## Summary

| stage   | decision                                            | stage53a_dataset_decision         | stage63_dataset_decision              | stage53a_aggregate_decision                 | stage63_aggregate_decision                      |   base_accuracy_delta |   base_macro_f1_delta |   composed_accuracy_delta |   composed_macro_f1_delta |   base_NOT_ENTITLED_delta |   base_SUPPORT_delta |   base_REFUTE_delta | external_improvement_detected   | external_still_incomplete   | clean_dev_preservation_verdict   | allowed_claim                                                                                                                                                                                  | forbidden_claim                                                                                                                                      | recommended_next_stage                        |
|:--------|:----------------------------------------------------|:----------------------------------|:--------------------------------------|:--------------------------------------------|:------------------------------------------------|----------------------:|----------------------:|--------------------------:|--------------------------:|--------------------------:|---------------------:|--------------------:|:--------------------------------|:----------------------------|:---------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------|
| Stage64 | STAGE64_EXTERNAL_IMPROVEMENT_AUDIT_READY_INCOMPLETE | STAGE43C0_EXTERNAL_FACTVER_UNSAFE | STAGE43C0_EXTERNAL_FACTVER_INCOMPLETE | STAGE43C0_EXTERNAL_FACTVER_AGGREGATE_UNSAFE | STAGE43C0_EXTERNAL_FACTVER_AGGREGATE_INCOMPLETE |                 0.166 |              0.199333 |                     0.166 |                  0.198602 |                      -454 |                  270 |                 184 | True                            | True                        | PRESERVED_WITH_SMALL_DROP        | Stage57 bridge-enabled training substantially reduces VitaminC NOT_ENTITLED collapse and improves external diagnostic accuracy/macro-F1 over Stage53A, while preserving clean-dev performance. | Do not claim VitaminC robustness or solved external generalization; Stage63 remains INCOMPLETE and still overpredicts NOT_ENTITLED relative to gold. | Stage65 residual Stage63 external error audit |

## Metric comparison

| metric                             |   stage53a |   stage63 |     delta |
|:-----------------------------------|-----------:|----------:|----------:|
| base_accuracy                      |   0.156    |  0.322    |  0.166    |
| base_macro_f1                      |   0.115986 |  0.315319 |  0.199333 |
| composed_accuracy                  |   0.156    |  0.322    |  0.166    |
| composed_macro_f1                  |   0.116717 |  0.315319 |  0.198602 |
| delta_accuracy                     |   0        |  0        |  0        |
| delta_macro_f1                     |   0.000731 |  0        | -0.000731 |
| introduced_unsafe_SUPPORT_count    |   1        |  0        | -1        |
| introduced_REFUTE_to_SUPPORT_count |   0        |  0        |  0        |
| introduced_SUPPORT_to_REFUTE_count |   0        |  0        |  0        |
| changed_row_count                  |   2        |  0        | -2        |
| recovery_fired_count               |   0        |  0        |  0        |
| blocker_fired_count                |   0        |  0        |  0        |

## Prediction distribution comparison

| kind                         |   NOT_ENTITLED |   REFUTE |   SUPPORT |
|:-----------------------------|---------------:|---------:|----------:|
| gold                         |            145 |      355 |       500 |
| stage53a_base_prediction     |            946 |       33 |        21 |
| stage63_base_prediction      |            492 |      217 |       291 |
| stage53a_composed_prediction |            944 |       33 |        23 |
| stage63_composed_prediction  |            492 |      217 |       291 |

## Label movement audit

| label        |   gold_count |   stage53a_base_pred |   stage63_base_pred |   delta_base_pred |   stage53a_composed_pred |   stage63_composed_pred |   delta_composed_pred |   stage63_base_minus_gold |   stage63_composed_minus_gold |
|:-------------|-------------:|---------------------:|--------------------:|------------------:|-------------------------:|------------------------:|----------------------:|--------------------------:|------------------------------:|
| NOT_ENTITLED |          145 |                  946 |                 492 |              -454 |                      944 |                     492 |                  -452 |                       347 |                           347 |
| REFUTE       |          355 |                   33 |                 217 |               184 |                       33 |                     217 |                   184 |                      -138 |                          -138 |
| SUPPORT      |          500 |                   21 |                 291 |               270 |                       23 |                     291 |                   268 |                      -209 |                          -209 |

## Per-label metric comparison

| mode     | label        |   stage53a_precision |   stage63_precision |   delta_precision |   stage53a_recall |   stage63_recall |   delta_recall |   stage53a_f1 |   stage63_f1 |   delta_f1 |   support |
|:---------|:-------------|---------------------:|--------------------:|------------------:|------------------:|-----------------:|---------------:|--------------:|-------------:|-----------:|----------:|
| base     | NOT_ENTITLED |             0.138478 |            0.164634 |          0.026156 |          0.903448 |         0.558621 |      -0.344827 |      0.240147 |     0.254317 |   0.01417  |       145 |
| base     | REFUTE       |             0.272727 |            0.391705 |          0.118978 |          0.025352 |         0.239437 |       0.214085 |      0.046392 |     0.297203 |   0.250811 |       355 |
| base     | SUPPORT      |             0.761905 |            0.536082 |         -0.225823 |          0.032    |         0.312    |       0.28     |      0.06142  |     0.394437 |   0.333017 |       500 |
| composed | NOT_ENTITLED |             0.137712 |            0.164634 |          0.026922 |          0.896552 |         0.558621 |      -0.337931 |      0.238751 |     0.254317 |   0.015566 |       145 |
| composed | REFUTE       |             0.272727 |            0.391705 |          0.118978 |          0.025352 |         0.239437 |       0.214085 |      0.046392 |     0.297203 |   0.250811 |       355 |
| composed | SUPPORT      |             0.73913  |            0.536082 |         -0.203048 |          0.034    |         0.312    |       0.278    |      0.06501  |     0.394437 |   0.329427 |       500 |

## Clean-dev preservation

|   stage51_best_dev_acc |   stage61_best_dev_acc |   delta_dev_acc |   stage51_best_dev_macro_f1 |   stage61_best_dev_macro_f1 |   delta_dev_macro_f1 | clean_dev_preservation_verdict   |
|-----------------------:|-----------------------:|----------------:|----------------------------:|----------------------------:|---------------------:|:---------------------------------|
|               0.973611 |               0.970833 |     -0.00277776 |                    0.962855 |                    0.958808 |          -0.00404618 | PRESERVED_WITH_SMALL_DROP        |

## Allowed claim

Stage57 bridge-enabled training substantially reduces VitaminC NOT_ENTITLED collapse and improves external diagnostic accuracy/macro-F1 over Stage53A, while preserving clean-dev performance.

## Forbidden claim

Do not claim VitaminC robustness or solved external generalization; Stage63 remains INCOMPLETE and still overpredicts NOT_ENTITLED relative to gold.

## Recommended next stage

Stage65 residual Stage63 external error audit
