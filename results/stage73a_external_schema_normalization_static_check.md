# Stage73A — External Schema Normalization Static Check

## Decision

`STAGE73A_EXTERNAL_SCHEMA_NORMALIZATION_STATIC_CHECK_READY`

## Summary

| stage    | decision                                                  | external_eval_jsonl                                 | external_eval_name   |   row_count |   records_normalized |   normalization_error_count |   records_missing_final_label |   records_with_added_aux_labels | encode_label_tensors_ok   | encode_label_tensors_error   | training_executed   | external_eval_executed   | recommended_next_stage                     | canonical_label_counts                               | label_field_counts   | claim_field_counts   | evidence_field_counts   |
|:---------|:----------------------------------------------------------|:----------------------------------------------------|:---------------------|------------:|---------------------:|----------------------------:|------------------------------:|--------------------------------:|:--------------------------|:-----------------------------|:--------------------|:-------------------------|:-------------------------------------------|:-----------------------------------------------------|:---------------------|:---------------------|:------------------------|
| Stage73A | STAGE73A_EXTERNAL_SCHEMA_NORMALIZATION_STATIC_CHECK_READY | data/stage43b1_vitaminc_validation_sample1000.jsonl | stage73_vitaminc     |        1000 |                 1000 |                           0 |                          1000 |                            1000 | True                      |                              | False               | False                    | Stage73 retry external diagnostic full run | {"NOT_ENTITLED": 145, "REFUTE": 355, "SUPPORT": 500} | {"label": 1000}      | {"claim": 1000}      | {"evidence": 1000}      |

## Checks

| check                            | pass   |
|:---------------------------------|:-------|
| row_count_1000                   | True   |
| all_records_normalized           | True   |
| no_normalization_errors          | True   |
| expected_vitaminc_label_counts   | True   |
| claim_field_found_for_all        | True   |
| evidence_field_found_for_all     | True   |
| label_field_found_for_all        | True   |
| encode_label_tensors_ok          | True   |
| help_external_eval_jsonl_present | True   |
| help_external_output_dir_present | True   |
| source_checks_pass               | True   |
| stage71c_ready                   | True   |
| stage72_ready                    | True   |

## Help checks

| flag                         | present_in_help   |
|:-----------------------------|:------------------|
| --external-eval-jsonl        | True              |
| --external-eval-name         | True              |
| --external-output-dir        | True              |
| --stage57-bridge-train-jsonl | True              |
| --stage66-bridge-train-jsonl | True              |

## Source checks

| check                                                         | pass   |
|:--------------------------------------------------------------|:-------|
| source_contains_external_schema_normalized                    | True   |
| source_contains_external_schema_normalization_source          | True   |
| source_contains_external_schema_missing_final_label_fixed     | True   |
| source_contains_external_schema_label_field_used              | True   |
| source_contains_external_schema_records_normalized            | True   |
| source_contains_external_schema_records_with_added_aux_labels | True   |
| source_contains_external_eval_jsonl                           | True   |
| source_contains_external_eval_name                            | True   |
| source_contains_final_label                                   | True   |
| source_contains_polarity_label                                | True   |
| source_contains_frame_compatible_label                        | True   |
| source_contains_predicate_covered_label                       | True   |
| source_contains_sufficiency_label                             | True   |
| label_candidate_gold_label_present                            | True   |
| label_candidate_fact_label_present                            | True   |
| label_candidate_vitaminc_label_present                        | True   |
| label_candidate_stage43_label_present                         | True   |
| label_candidate_original_label_present                        | True   |
| text_candidate_hypothesis_present                             | True   |
| text_candidate_statement_present                              | True   |
| text_candidate_query_present                                  | True   |
| text_candidate_premise_present                                | True   |
| text_candidate_context_present                                | True   |
| text_candidate_passage_present                                | True   |
| text_candidate_document_present                               | True   |
| text_candidate_evidence_text_present                          | True   |
| clear_uncanonicalizable_label_error_present                   | True   |
| clear_missing_claim_evidence_error_present                    | True   |

## Canonical label counts

| canonical_final_label   |   count |
|:------------------------|--------:|
| NOT_ENTITLED            |     145 |
| REFUTE                  |     355 |
| SUPPORT                 |     500 |

## Label field counts

| label_field   |   count |
|:--------------|--------:|
| label         |    1000 |

## Claim field counts

| claim_field   |   count |
|:--------------|--------:|
| claim         |    1000 |

## Evidence field counts

| evidence_field   |   count |
|:-----------------|--------:|
| evidence         |    1000 |

## Added auxiliary label fields

| aux_field_added         |   count |
|:------------------------|--------:|
| frame_compatible_label  |    1000 |
| polarity_label          |    1000 |
| predicate_covered_label |    1000 |
| sufficiency_label       |    1000 |

## Sample normalized records

| record_id                                            |   line_no | label_field   | claim_field   | evidence_field   | original_label   | final_label   | claim_preview                                                                                    | evidence_preview                                                                                                                                                 |
|:-----------------------------------------------------|----------:|:--------------|:--------------|:-----------------|:-----------------|:--------------|:-------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| tals_vitaminc_validation_sample1000_validation_9543  |         1 | label         | claim         | evidence         | NOT_ENTITLED     | NOT_ENTITLED  | Airbourne 's song Live It Up was selected as the official theme song for Extreme Rules in 2013 . | In the movie The Lost Boys : The Tribe , a sequel to the original Lost Boys , the song `` Too Much , Too Young , Too Fast '' is played in the car as Chris and N |
| tals_vitaminc_validation_sample1000_validation_62847 |         2 | label         | claim         | evidence         | REFUTE           | REFUTE        | The Cincinnati Kid is a boy .                                                                    | He considers it the film that allowed him to transition from the lighter comedic films he had previously been making and take on more serious films and subjects |
| tals_vitaminc_validation_sample1000_validation_61972 |         3 | label         | claim         | evidence         | SUPPORT          | SUPPORT       | Efraim Diveroli had a four-year sentence .                                                       | Diveroli was sentenced to four years in federal prison .                                                                                                         |
| tals_vitaminc_validation_sample1000_validation_47744 |         4 | label         | claim         | evidence         | SUPPORT          | SUPPORT       | Emma Watson was born before 1995 .                                                               | Emma Charlotte Duerre Watson -LRB- born 15 April 1990 -RRB- is a French-British actress , model , and activist .                                                 |
| tals_vitaminc_validation_sample1000_validation_59780 |         5 | label         | claim         | evidence         | SUPPORT          | SUPPORT       | Tilda Swinton is a singer .                                                                      | After participating in Celine Dion 's tour , Swinton released her album `` True '' .                                                                             |

## Normalization errors preview

(none)

## Execution policy

No training, no external evaluation, no smoke run, no full run.
Only py_compile, --help, source checks, JSONL normalization checks, and encode_label_tensors static validation were executed.

## Recommended next stage

Stage73 retry external diagnostic full run
