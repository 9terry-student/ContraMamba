# Stage80C - Conservative Stage75v2 Runner Integration Plan

## Decision

`STAGE80C_RUNNER_INTEGRATION_PLAN_READY`

## Summary

| stage    | decision                               | purpose                                                                               | integration_name                                                  | primary_candidate   | stage75_full_bridge_default   | stage80a_bridge_default_candidate   |   main_clean_total |   clean_train_expected |   clean_dev_expected |   stage57_bridge_rows |   stage66_bridge_rows |   stage80a_bridge_rows |   combined_bridge_rows |   final_train_expected | stage80a_label_counts                                | stage80a_family_counts                                                                                                                                   | pairwise_loss_source   |   pairwise_bridge_rows_excluded | training_executed   | external_eval_executed   | recommended_next_stage                                                        |
|:---------|:---------------------------------------|:--------------------------------------------------------------------------------------|:------------------------------------------------------------------|:--------------------|:------------------------------|:------------------------------------|-------------------:|-----------------------:|---------------------:|----------------------:|----------------------:|-----------------------:|-----------------------:|-----------------------:|:-----------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------|--------------------------------:|:--------------------|:-------------------------|:------------------------------------------------------------------------------|
| Stage80C | STAGE80C_RUNNER_INTEGRATION_PLAN_READY | Plan runner integration for conservative Stage75v2 bridge before patching the runner. | stage57_stage66_stage80a_conservative_stage75v2_append_train_only | True                | False                         | True                                |               3600 |                   2880 |                  720 |                   520 |                   720 |                    500 |                   1740 |                   4620 | {"SUPPORT": 170, "REFUTE": 170, "NOT_ENTITLED": 160} | {"numeric_temporal_polarity_repair_v2_conservative": 180, "lexical_type_polarity_repair_v2_conservative": 160, "strict_ne_false_support_safety_v2": 160} | clean_main_train_only  |                            1740 | False               | False                    | Stage80D patch runner to support Stage80A conservative Stage75v2 bridge flags |

## Checks

| check                         | pass   |
|:------------------------------|:-------|
| stage80a_data_exists          | True   |
| stage80b_report_exists        | True   |
| stage80b_static_exists        | True   |
| stage80a_row_count_500        | True   |
| stage80a_label_counts_exact   | True   |
| stage80a_family_counts_exact  | True   |
| stage79_revise_not_default    | True   |
| stage75_full_not_primary      | True   |
| stage80a_primary_candidate    | True   |
| combined_bridge_1740          | True   |
| final_train_expected_4620     | True   |
| pairwise_bridge_excluded_1740 | True   |
| clean_dev_unchanged_720       | True   |

## Counts

| component            |   rows | train_included   | dev_included   | note                      |
|:---------------------|-------:|:-----------------|:---------------|:--------------------------|
| clean_main_total     |   3600 | False            | False          | source before split       |
| clean_train_expected |   2880 | True             | False          | 80% clean main            |
| clean_dev_expected   |    720 | False            | True           | checkpoint selection only |
| stage57_bridge       |    520 | True             | False          | append_train_only         |
| stage66_bridge       |    720 | True             | False          | append_train_only         |
| stage80a_bridge      |    500 | True             | False          | append_train_only         |
| combined_bridge      |   1740 | True             | False          | excluded from pairwise    |
| final_train_expected |   4620 | True             | False          | clean_train + bridges     |

## Stage80A family x label counts

| family                                           |   SUPPORT |   REFUTE |   NOT_ENTITLED |   total |
|:-------------------------------------------------|----------:|---------:|---------------:|--------:|
| lexical_type_polarity_repair_v2_conservative     |        80 |       80 |              0 |     160 |
| numeric_temporal_polarity_repair_v2_conservative |        90 |       90 |              0 |     180 |
| strict_ne_false_support_safety_v2                |         0 |        0 |            160 |     160 |

## New runner flags

| flag                          | type   | default   | purpose                                                                      | choices                       |
|:------------------------------|:-------|:----------|:-----------------------------------------------------------------------------|:------------------------------|
| --stage80a-bridge-train-jsonl | path   |           | Path to conservative Stage75v2 bridge JSONL.                                 | nan                           |
| --stage80a-bridge-train-mode  | nan    | none      | Append Stage80A bridge only to train split after clean main train/dev split. | ['none', 'append_train_only'] |

## Runner patch requirements

| requirement                                            | detail                                                                                                          |
|:-------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------|
| Add Stage80A CLI flags.                                | --stage80a-bridge-train-jsonl and --stage80a-bridge-train-mode {none,append_train_only}.                        |
| Append only to train.                                  | Stage80A rows must be appended after clean main split and never enter dev/checkpoint data.                      |
| Preserve Stage57/Stage66 behavior.                     | Existing Stage57 and Stage66 flags/metadata must remain unchanged.                                              |
| Do not use Stage75 full bridge in Stage80 primary.     | Stage75 flags remain supported but omitted from the Stage80F canonical command.                                 |
| Exclude bridge rows from pairwise intervention loss.   | Pairwise intervention loss source remains clean_main_train_only; Stage57/66/80A excluded.                       |
| Include Stage80A rows in CE/final/aux training losses. | Like Stage57/66/75 bridge rows, Stage80A should influence supervised heads but not pairwise intervention pairs. |
| Metadata must expose Stage80A counts.                  | Report stage80a row count, label counts, family counts, combined bridge count, final train expected count.      |
| Preserve external diagnostic isolation.                | External eval data must not affect training/checkpoint/calibration.                                             |

## Ablation plan

| stage          | candidate                                 | stage57   | stage66   |   stage80a_rows | description                                                                       |
|:---------------|:------------------------------------------|:----------|:----------|----------------:|:----------------------------------------------------------------------------------|
| Stage80F       | A2_main_stage57_stage66_stage80a_full_500 | True      | True      |             500 | Main conservative candidate: polarity repair plus NE safety.                      |
| Stage81        | VitaminC external diagnostic for Stage80F | True      | True      |             500 | Compare against Stage73 and Stage77.                                              |
| Stage82        | Residual audit for Stage81                | True      | True      |             500 | Check false SUPPORT, false entitlement, false NE, and polarity movement.          |
| Optional later | A1 polarity-only subset                   | True      | True      |             340 | Requires subset JSONL or family filter; not part of Stage80C primary integration. |
| Optional later | A3 NE-safety-only subset                  | True      | True      |             160 | Requires subset JSONL or family filter; diagnostic only.                          |

## Recommended next stage

Stage80D patch runner to support Stage80A conservative Stage75v2 bridge flags
