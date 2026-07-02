# Stage66 — Residual Bridge Expansion Design

## Decision

`STAGE66_RESIDUAL_BRIDGE_EXPANSION_DESIGN_READY`

## Summary

| stage   | decision                                       | basis                                         |   row_count_stage65 |   error_count_stage65 |   false_ne_total |   polarity_error_total |   false_entitlement_total | primary_design_goal                                                                                            |   stage66_incremental_rows |   stage57_existing_rows |   combined_bridge_rows_if_stage57_plus_stage66 | stage66_label_plan                                  | combined_stage57_stage66_label_plan                  | recommended_generation_mode          | recommended_integration_mode        | recommended_next_stage                                        | allowed_claim                                                                                                      | forbidden_claim                                                                                                                     |
|:--------|:-----------------------------------------------|:----------------------------------------------|--------------------:|----------------------:|-----------------:|-----------------------:|--------------------------:|:---------------------------------------------------------------------------------------------------------------|---------------------------:|------------------------:|-----------------------------------------------:|:----------------------------------------------------|:-----------------------------------------------------|:-------------------------------------|:------------------------------------|:--------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------|
| Stage66 | STAGE66_RESIDUAL_BRIDGE_EXPANSION_DESIGN_READY | Stage65 residual Stage63 external error audit |                1000 |                   678 |              411 |                    203 |                        64 | Reduce remaining false NOT_ENTITLED on gold SUPPORT/REFUTE while improving REFUTE/SUPPORT polarity separation. |                        720 |                     520 |                                           1240 | {'SUPPORT': 360, 'REFUTE': 320, 'NOT_ENTITLED': 40} | {'SUPPORT': 520, 'REFUTE': 480, 'NOT_ENTITLED': 240} | synthetic_nonleaking_residual_bridge | append_train_only_after_clean_split | Stage67 implement synthetic Stage66 residual bridge generator | Stage66 defines a non-leaking synthetic residual bridge expansion plan based on Stage65 error taxonomy and counts. | Do not claim Stage66 uses VitaminC residual text for training; it only uses residual error taxonomy/counts to set synthetic quotas. |

## Proposed bridge families

| stage66_family                      | target_error                           |   planned_rows |   SUPPORT |   REFUTE |   NOT_ENTITLED | purpose                                                                                                                   |
|:------------------------------------|:---------------------------------------|---------------:|----------:|---------:|---------------:|:--------------------------------------------------------------------------------------------------------------------------|
| support_entitlement_recovery_bridge | false_NE_on_SUPPORT                    |            200 |       200 |        0 |              0 | Teach the model that semantically sufficient evidence should license SUPPORT instead of defaulting to NOT_ENTITLED.       |
| refute_entitlement_recovery_bridge  | false_NE_on_REFUTE                     |            160 |         0 |      160 |              0 | Teach the model that same-frame predicate/value conflict should license REFUTE instead of NOT_ENTITLED.                   |
| polarity_disambiguation_bridge      | REFUTE_SUPPORT_polarity_confusion      |            200 |       100 |      100 |              0 | Separate SUPPORT vs REFUTE when frame and predicate are covered but polarity/value relation differs.                      |
| numeric_temporal_comparison_bridge  | numeric_temporal_comparative_residuals |            120 |        60 |       60 |              0 | Reinforce entitlement and polarity in digit/year/comparison-heavy cases identified by Stage65 feature audit.              |
| strict_ne_frame_safety_bridge       | false_SUPPORT_or_REFUTE_on_NE          |             40 |         0 |        0 |             40 | Keep a small safety anchor for related-but-insufficient evidence so positive-skewed residual bridges do not over-entitle. |

## Stage66 incremental label plan

| label        |   stage66_incremental_count |
|:-------------|----------------------------:|
| SUPPORT      |                         360 |
| REFUTE       |                         320 |
| NOT_ENTITLED |                          40 |

## Stage57 + Stage66 combined bridge label plan

| label        |   stage57_existing |   stage66_incremental |   combined_bridge |
|:-------------|-------------------:|----------------------:|------------------:|
| SUPPORT      |                160 |                   360 |               520 |
| REFUTE       |                160 |                   320 |               480 |
| NOT_ENTITLED |                200 |                    40 |               240 |

## Bridge ratio plan

|   stage57_existing_rows |   stage66_incremental_rows |   combined_bridge_rows |   main_rows |   stage66_ratio_vs_main |   combined_bridge_ratio_vs_main |
|------------------------:|---------------------------:|-----------------------:|------------:|------------------------:|--------------------------------:|
|                     520 |                        720 |                   1240 |        3600 |                     0.2 |                        0.344444 |

## Leakage constraints

| constraint                   | requirement                                                                                                                       | reason                                                                                |
|:-----------------------------|:----------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|
| synthetic_only               | All Stage66 rows must be generated from synthetic templates only.                                                                 | Avoid external leakage from Stage63/Stage65 VitaminC residuals.                       |
| no_vitaminc_text_or_labels   | Stage65 residual samples may inform taxonomy/quota only; their claim/evidence text must not be copied, paraphrased, or templated. | Stage66 must remain non-leaking.                                                      |
| no_time_swap                 | Do not use data/controlled_v5_v3.jsonl time_swap rows or any corrupted temporal examples.                                         | time_swap is excluded from main classification training.                              |
| train_only_append            | Stage66 bridge may be appended to train only after clean main split; never to dev/checkpoint selection.                           | Preserve clean-dev evaluation validity.                                               |
| no_external_threshold_tuning | Do not use VitaminC metrics to tune composer/recovery thresholds.                                                                 | Stage63 is diagnostic, not a tuning target.                                           |
| positive_skew_with_ne_safety | Stage66 is SUPPORT/REFUTE-heavy but keeps small NE safety rows.                                                                   | Residual failure is dominated by false NE, but false entitlement must remain bounded. |

## Implementation plan

| next_stage   | task                                                                    | allowed_execution                                                |
|:-------------|:------------------------------------------------------------------------|:-----------------------------------------------------------------|
| Stage67      | Implement synthetic Stage66 residual bridge generator.                  | local/static generation only; no model training.                 |
| Stage68      | Static audit of generated Stage66 residual bridge.                      | dataset audit only; no model training.                           |
| Stage69      | Runner integration plan for Stage57 + Stage66 bridge train-only append. | report-only or runner static patch planning; no external tuning. |

## Allowed claim

Stage66 defines a non-leaking synthetic residual bridge expansion plan based on Stage65 error taxonomy and counts.

## Forbidden claim

Do not claim Stage66 uses VitaminC residual text for training; it only uses residual error taxonomy/counts to set synthetic quotas.
