# Stage52 — Stage51 Frozen Recovery Run Summary

## Decision

`STAGE52_STAGE51_FROZEN_RECOVERY_RUN_SUMMARY_READY`

## Source

- Run directory: `results/stage51_frozen_recovery_run_20260702_044005`
- Train report: `results/stage51_frozen_recovery_run_20260702_044005/stage51_frozen_recovery_train_report.json`
- Stage45C report: `results/stage51_frozen_recovery_run_20260702_044005/stage51_stage45c_recovery_report.json`

## Summary

| stage   | decision                                          | source_run_dir                                      | source_train_report                                                                           | source_stage45c_report                                                                    |   best_epoch |   selected_epoch |   best_dev_accuracy |   best_dev_macro_f1 | prediction_distribution                             | stage45c_enabled   |   stage45c_support_recovery_weight |   stage45c_entitled_ne_penalty_weight | stage45c_loss_terms_active                  |   stage45c_support_recovery_loss_mean |   stage45c_entitled_ne_penalty_loss_mean | stage15_used_for_training   | stage15_used_for_checkpoint_selection   | time_swap_used_in_main_clean_data   | stage44b_decision                                  | stage44b2_decision                       | stage45b_decision                         | audit_warnings                                                                                                                                                                                                                                            |
|:--------|:--------------------------------------------------|:----------------------------------------------------|:----------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------|-------------:|-----------------:|--------------------:|--------------------:|:----------------------------------------------------|:-------------------|-----------------------------------:|--------------------------------------:|:--------------------------------------------|--------------------------------------:|-----------------------------------------:|:----------------------------|:----------------------------------------|:------------------------------------|:---------------------------------------------------|:-----------------------------------------|:------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Stage52 | STAGE52_STAGE51_FROZEN_RECOVERY_RUN_SUMMARY_READY | results/stage51_frozen_recovery_run_20260702_044005 | results/stage51_frozen_recovery_run_20260702_044005/stage51_frozen_recovery_train_report.json | results/stage51_frozen_recovery_run_20260702_044005/stage51_stage45c_recovery_report.json |           80 |               80 |            0.973611 |            0.962855 | {'NOT_ENTITLED': 525, 'REFUTE': 86, 'SUPPORT': 109} | True               |                                0.1 |                                   0.1 | ['support_recovery', 'entitled_ne_penalty'] |                            0.00331168 |                               0.00180811 | False                       | False                                   | False                               | STAGE44B_INTERNAL_ANTI_COLLAPSE_SELECTION_DISABLED | STAGE44B2_PRIOR_AWARE_SELECTION_DISABLED | STAGE45B_INTERNAL_FAMILY_HOLDOUT_DISABLED | ['aux_to_ce_loss_ratio_weighted=0.792 > 0.5: weighted auxiliary losses exceed 50% of CE ??total_loss is not CE-dominated', "ranking_loss is active (ranking_weight=2.0): baseline is not CE-only; see loss_component_epoch_avg_weighted['ranking_loss']"] |

## Per-label metrics

| label        |   precision |   recall |       f1 |
|:-------------|------------:|---------:|---------:|
| NOT_ENTITLED |    0.99619  | 0.968519 | 0.98216  |
| REFUTE       |    1        | 1        | 1        |
| SUPPORT      |    0.844037 | 0.978723 | 0.906404 |

## Pairwise checks

| check                                |        mean |   passed |   pass_rate |
|:-------------------------------------|------------:|---------:|------------:|
| deletion_sufficiency_drop            |   0.998077  |      nan |  nan        |
| deletion_sufficiency_lower           | nan         |        1 |    1        |
| entity_frame_drop                    |   0.927617  |      nan |  nan        |
| entity_frame_lower                   | nan         |        0 |    0.983333 |
| event_frame_drop                     |   0.871232  |      nan |  nan        |
| event_frame_lower                    | nan         |        0 |    0.966667 |
| flip_entitlement_delta               |   0.0557324 |      nan |  nan        |
| paraphrase_gate_delta                |   0.0566948 |      nan |  nan        |
| paraphrase_preserved                 | nan         |        0 |    0.883333 |
| polarity_flip_preserved_and_reversed | nan         |        0 |    0.883333 |
| predicate_coverage_drop              |   0.945649  |      nan |  nan        |
| predicate_disentangled               | nan         |        0 |    0.95     |
| predicate_frame_delta                |   0.0278951 |      nan |  nan        |
| truncation_sufficiency_drop          |   0.998029  |      nan |  nan        |
| truncation_sufficiency_lower         | nan         |        1 |    1        |

## Intervention distributions

| intervention        |   NOT_ENTITLED |   REFUTE |   SUPPORT |   entitlement_prob |   frame_prob |   predicate_coverage_prob |   sufficiency_prob |   polarity_margin |
|:--------------------|---------------:|---------:|----------:|-------------------:|-------------:|--------------------------:|-------------------:|------------------:|
| entity_swap         |             58 |        0 |         2 |        0.0270096   |  0.0320038   |                 0.0584976 |        0.998794    |          3.07598  |
| event_swap          |             54 |        0 |         6 |        0.0724657   |  0.0883886   |                 0.135235  |        0.998881    |          3.69385  |
| evidence_deletion   |             60 |        0 |         0 |        0.000578883 |  0.997322    |                 0.997425  |        0.000581927 |         -2.40015  |
| evidence_truncation |             60 |        0 |         0 |        0.000624727 |  0.995823    |                 0.99788   |        0.00062939  |         -2.29083  |
| irrelevant_evidence |             60 |        0 |         0 |        6.98966e-10 |  0.000846228 |                 0.0010358 |        0.000804691 |         -0.269935 |
| location_swap       |             55 |        0 |         5 |        0.0622426   |  0.0683241   |                 0.943783  |        0.998317    |          4.55855  |
| none                |              1 |       26 |        33 |        0.937831    |  0.959621    |                 0.975447  |        0.998659    |          0.560276 |
| paraphrase          |              1 |       26 |        33 |        0.956466    |  0.978193    |                 0.979341  |        0.997011    |          0.574137 |
| polarity_flip       |              0 |       34 |        26 |        0.989881    |  0.994506    |                 0.996762  |        0.998585    |         -0.850673 |
| predicate_swap      |             59 |        0 |         1 |        0.0286633   |  0.970452    |                 0.0297976 |        0.99653     |          0.292575 |
| role_swap           |             57 |        0 |         3 |        0.0397152   |  0.0832806   |                 0.938048  |        0.98382     |          4.47502  |
| title_name_swap     |             60 |        0 |         0 |        0.00515094  |  0.0183051   |                 0.0570724 |        0.982009    |          2.74883  |

## Leakage / scope

- External data used for training: `false`
- External data used for checkpoint selection: `false`
- Stage15 used for training: `False`
- Stage15 used for checkpoint selection: `False`
- time_swap used in main clean data: `False`
- Scope: `internal_clean_controlled_dev_only`

## Interpretation

Allowed claim:

> Internal controlled clean-dev recovery run completed successfully with Stage47 frozen recovery weights.

Not allowed claim:

> Do not claim external generalization, VitaminC transfer, Climate-FEVER transfer, or naturalistic robustness from this run alone.

Next recommended stage:

> Run a separate frozen external/OOD evaluation or create a formal Stage52 audit report before any external claim.
