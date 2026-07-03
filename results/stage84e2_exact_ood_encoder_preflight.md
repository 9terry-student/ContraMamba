# Stage84E2 - Exact OOD Encoder Preflight

## Decision

`STAGE84E2_EXACT_OOD_ENCODER_PREFLIGHT_READY`

## Modal aux values

| final_label   | aux_key                 | modal_value   |   modal_count | all_counts         |
|:--------------|:------------------------|:--------------|--------------:|:-------------------|
| NOT_ENTITLED  | frame_compatible_label  | 0             |          1800 | {0: 1800, 1: 900}  |
| NOT_ENTITLED  | polarity_label          | NONE          |          2700 | {'NONE': 2700}     |
| NOT_ENTITLED  | predicate_covered_label | 0             |          1500 | {0: 1500, 1: 1200} |
| NOT_ENTITLED  | sufficiency_label       | 1             |          1800 | {1: 1800, 0: 900}  |
| REFUTE        | frame_compatible_label  | 1             |           450 | {1: 450}           |
| REFUTE        | polarity_label          | REFUTE        |           450 | {'REFUTE': 450}    |
| REFUTE        | predicate_covered_label | 1             |           450 | {1: 450}           |
| REFUTE        | sufficiency_label       | 1             |           450 | {1: 450}           |
| SUPPORT       | frame_compatible_label  | 1             |           450 | {1: 450}           |
| SUPPORT       | polarity_label          | SUPPORT       |           450 | {'SUPPORT': 450}   |
| SUPPORT       | predicate_covered_label | 1             |           450 | {1: 450}           |
| SUPPORT       | sufficiency_label       | 1             |           450 | {1: 450}           |

## Summary

| stage     | decision                                    | src                                                 | main_aux_source                               | out                                                                  |    n | label_counts                                         | intervention_type_counts               | label_required_keys                                                                                         | mamba_required_keys                                         | required_keys_union                                                                                                                                                    | modal_by_label                                                                                                                                                                                                                                                                                                                                                                          | missing_after_fill   | pair_id_unique   | bad_polarity_labels   | encode_label_tensors_full_ok   | encode_mamba_records_sample_ok   | encode_mamba_records_full_ok   |
|:----------|:--------------------------------------------|:----------------------------------------------------|:----------------------------------------------|:---------------------------------------------------------------------|-----:|:-----------------------------------------------------|:---------------------------------------|:------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------|:-----------------|:----------------------|:-------------------------------|:---------------------------------|:-------------------------------|
| Stage84E2 | STAGE84E2_EXACT_OOD_ENCODER_PREFLIGHT_READY | data/stage43b1_vitaminc_validation_sample1000.jsonl | data/controlled_v5_v3_without_time_swap.jsonl | data/stage84e2_vitaminc_validation_sample1000_exact_ood_schema.jsonl | 1000 | {"NOT_ENTITLED": 145, "REFUTE": 355, "SUPPORT": 500} | {"vitaminc_external_validation": 1000} | ["final_label", "frame_compatible_label", "polarity_label", "predicate_covered_label", "sufficiency_label"] | ["claim", "evidence", "id", "intervention_type", "pair_id"] | ["claim", "evidence", "final_label", "frame_compatible_label", "id", "intervention_type", "pair_id", "polarity_label", "predicate_covered_label", "sufficiency_label"] | {"NOT_ENTITLED": {"frame_compatible_label": 0, "polarity_label": "NONE", "predicate_covered_label": 0, "sufficiency_label": 1}, "REFUTE": {"frame_compatible_label": 1, "polarity_label": "REFUTE", "predicate_covered_label": 1, "sufficiency_label": 1}, "SUPPORT": {"frame_compatible_label": 1, "polarity_label": "SUPPORT", "predicate_covered_label": 1, "sufficiency_label": 1}} | {}                   | True             | []                    | True                           | True                             | True                           |
