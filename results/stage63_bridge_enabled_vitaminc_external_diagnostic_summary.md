# Stage63 — Bridge-enabled VitaminC External Diagnostic Summary

## Compact summary

| decision                                                          | run_dir                                                               | aggregate_decision                              | dataset_decision                      |   row_count |   stage63_base_accuracy |   stage63_base_macro_f1 |   stage63_composed_accuracy |   stage63_composed_macro_f1 |   bridge_rows | bridge_train_only   | bridge_used_for_dev   | external_data_used_for_training   |
|:------------------------------------------------------------------|:----------------------------------------------------------------------|:------------------------------------------------|:--------------------------------------|------------:|------------------------:|------------------------:|----------------------------:|----------------------------:|--------------:|:--------------------|:----------------------|:----------------------------------|
| STAGE63_BRIDGE_ENABLED_VITAMINC_EXTERNAL_DIAGNOSTIC_SUMMARY_READY | results/stage63_bridge_enabled_vitaminc_external_eval_20260702_060044 | STAGE43C0_EXTERNAL_FACTVER_AGGREGATE_INCOMPLETE | STAGE43C0_EXTERNAL_FACTVER_INCOMPLETE |        1000 |                   0.322 |                0.315319 |                       0.322 |                    0.315319 |           520 | True                | False                 | False                             |

## Stage63 vs Stage53A metric comparison

|   stage53a_base_accuracy |   stage63_base_accuracy |   delta_base_accuracy |   stage53a_base_macro_f1 |   stage63_base_macro_f1 |   delta_base_macro_f1 |   stage53a_composed_accuracy |   stage63_composed_accuracy |   delta_composed_accuracy |   stage53a_composed_macro_f1 |   stage63_composed_macro_f1 |   delta_composed_macro_f1 |
|-------------------------:|------------------------:|----------------------:|-------------------------:|------------------------:|----------------------:|-----------------------------:|----------------------------:|--------------------------:|-----------------------------:|----------------------------:|--------------------------:|
|                    0.156 |                   0.322 |                 0.166 |                 0.115986 |                0.315319 |              0.199333 |                        0.156 |                       0.322 |                     0.166 |                     0.116717 |                    0.315319 |                  0.198602 |

## Stage63 vs Stage53A NE collapse comparison

|   stage53a_base_NOT_ENTITLED |   stage63_base_NOT_ENTITLED |   delta_base_NOT_ENTITLED |   stage53a_composed_NOT_ENTITLED |   stage63_composed_NOT_ENTITLED |   delta_composed_NOT_ENTITLED |   stage53a_base_SUPPORT |   stage63_base_SUPPORT |   delta_base_SUPPORT |   stage53a_base_REFUTE |   stage63_base_REFUTE |   delta_base_REFUTE |
|-----------------------------:|----------------------------:|--------------------------:|---------------------------------:|--------------------------------:|------------------------------:|------------------------:|-----------------------:|---------------------:|-----------------------:|----------------------:|--------------------:|
|                          946 |                         492 |                      -454 |                              944 |                             492 |                          -452 |                      21 |                    291 |                  270 |                     33 |                   217 |                 184 |

## Distributions

| kind                         |   NOT_ENTITLED |   REFUTE |   SUPPORT |
|:-----------------------------|---------------:|---------:|----------:|
| stage63_gold                 |            145 |      355 |       500 |
| stage63_base_prediction      |            492 |      217 |       291 |
| stage63_composed_prediction  |            492 |      217 |       291 |
| stage53a_base_prediction     |            946 |       33 |        21 |
| stage53a_composed_prediction |            944 |       33 |        23 |

## Leakage policy

- Stage57 bridge train only: `True`
- Stage57 bridge used for dev: `False`
- Stage57 bridge used for checkpoint selection: `False`
- Stage57 external data used for training: `False`
- Stage57 external metrics used for threshold tuning: `False`
- time_swap used in main clean data: `False`
