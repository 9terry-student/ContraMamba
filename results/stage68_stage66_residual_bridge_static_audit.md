# Stage68 — Stage66 Residual Bridge Static Audit

## Decision

`STAGE68_STAGE66_RESIDUAL_BRIDGE_STATIC_AUDIT_READY`

## Summary

| stage   | decision                                           |   row_count | computed_ready   | polarity_mapping                       | polarity_label_counts                       | encode_label_tensors_ok   | encode_error   | stage67_audit_decision                     | stage71a_decision                          | recommended_next_stage                            |
|:--------|:---------------------------------------------------|------------:|:-----------------|:---------------------------------------|:--------------------------------------------|:--------------------------|:---------------|:-------------------------------------------|:-------------------------------------------|:--------------------------------------------------|
| Stage68 | STAGE68_STAGE66_RESIDUAL_BRIDGE_STATIC_AUDIT_READY |         720 | True             | {'NONE': 0, 'REFUTE': 1, 'SUPPORT': 2} | {'SUPPORT': 360, 'REFUTE': 320, 'NONE': 40} | True                      |                | STAGE67_STAGE66_RESIDUAL_BRIDGE_DATA_READY | STAGE71A_STAGE66_POLARITY_SCHEMA_FIX_READY | Stage69 Stage57 + Stage66 bridge integration plan |

## Checks

| check                                  | pass   |
|:---------------------------------------|:-------|
| row_count_ok                           | True   |
| label_counts_ok                        | True   |
| family_counts_ok                       | True   |
| family_label_counts_ok                 | True   |
| stage67_audit_ready                    | True   |
| stage66_design_ready                   | True   |
| stage71a_polarity_fix_ready            | True   |
| polarity_labels_encoder_compatible     | True   |
| encode_label_tensors_ok                | True   |
| no_id_overlap_with_stage57             | True   |
| no_claim_evidence_overlap_with_stage57 | True   |

## Label counts

| final_label   |   count |   expected |
|:--------------|--------:|-----------:|
| NOT_ENTITLED  |      40 |         40 |
| REFUTE        |     320 |        320 |
| SUPPORT       |     360 |        360 |

## Family counts

| stage66_family                      |   count |   expected |
|:------------------------------------|--------:|-----------:|
| numeric_temporal_comparison_bridge  |     120 |        120 |
| polarity_disambiguation_bridge      |     200 |        200 |
| refute_entitlement_recovery_bridge  |     160 |        160 |
| strict_ne_frame_safety_bridge       |      40 |         40 |
| support_entitlement_recovery_bridge |     200 |        200 |

## Family x label counts

| stage66_family                      | final_label   |   count |   expected |
|:------------------------------------|:--------------|--------:|-----------:|
| numeric_temporal_comparison_bridge  | REFUTE        |      60 |         60 |
| numeric_temporal_comparison_bridge  | SUPPORT       |      60 |         60 |
| polarity_disambiguation_bridge      | REFUTE        |     100 |        100 |
| polarity_disambiguation_bridge      | SUPPORT       |     100 |        100 |
| refute_entitlement_recovery_bridge  | REFUTE        |     160 |        160 |
| strict_ne_frame_safety_bridge       | NOT_ENTITLED  |      40 |         40 |
| support_entitlement_recovery_bridge | SUPPORT       |     200 |        200 |

## Polarity label counts

| polarity_label   |   count |   encoder_id | encoder_compatible   |
|:-----------------|--------:|-------------:|:---------------------|
| NONE             |      40 |            0 | True                 |
| REFUTE           |     320 |            1 | True                 |
| SUPPORT          |     360 |            2 | True                 |
