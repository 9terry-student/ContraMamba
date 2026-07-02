# Stage71A — Stage66 Polarity Schema Fix Audit

## Decision

`STAGE71A_STAGE66_POLARITY_SCHEMA_FIX_READY`

## Summary

| stage    | decision                                   | stage67_audit_decision                     |   row_count | polarity_mapping                       | polarity_label_counts                       | encode_label_tensors_ok   | encode_error   | recommended_next_stage                                             |
|:---------|:-------------------------------------------|:-------------------------------------------|------------:|:---------------------------------------|:--------------------------------------------|:--------------------------|:---------------|:-------------------------------------------------------------------|
| Stage71A | STAGE71A_STAGE66_POLARITY_SCHEMA_FIX_READY | STAGE67_STAGE66_RESIDUAL_BRIDGE_DATA_READY |         720 | {'NONE': 0, 'REFUTE': 1, 'SUPPORT': 2} | {'SUPPORT': 360, 'REFUTE': 320, 'NONE': 40} | True                      |                | Regenerate Stage68/69/70 reports, then rerun Stage71 full training |

## Checks

| check                              | pass   |
|:-----------------------------------|:-------|
| stage67_audit_ready                | True   |
| row_count_ok                       | True   |
| label_counts_ok                    | True   |
| family_counts_ok                   | True   |
| family_label_counts_ok             | True   |
| duplicate_ids_ok                   | True   |
| duplicate_claim_evidence_pairs_ok  | True   |
| bridge_family_equals_family_ok     | True   |
| target_error_mapping_ok            | True   |
| forbidden_text_markers_ok          | True   |
| polarity_labels_encoder_compatible | True   |
| encode_label_tensors_ok            | True   |

## Polarity label counts

| polarity_label   |   count |   encoder_id | encoder_compatible   |
|:-----------------|--------:|-------------:|:---------------------|
| NONE             |      40 |            0 | True                 |
| REFUTE           |     320 |            1 | True                 |
| SUPPORT          |     360 |            2 | True                 |

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

## Recommended next stage

Regenerate Stage68/69/70 reports, then rerun Stage71 full training
