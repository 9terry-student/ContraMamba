# Stage58 — Bridge Dataset Static Audit / Schema Check

## Decision

`STAGE58_BRIDGE_STATIC_AUDIT_READY`

## Sources

- Bridge JSONL: `data/stage57_nonleaking_external_bridge.jsonl`
- Stage57 audit: `results/stage57_nonleaking_external_bridge_audit.json`
- Controlled schema reference: `data/controlled_v5_v3_without_time_swap.jsonl`

## Summary

| decision                          |   row_count | required_fields_present   | label_mapping_ok   | balance_ok   | duplicate_ids_ok   | empty_required_fields_ok   | forbidden_external_markers_ok   | time_swap_absent   | axis_consistency_ok   | axis_rule                     | next_stage                      |
|:----------------------------------|------------:|:--------------------------|:-------------------|:-------------|:-------------------|:---------------------------|:--------------------------------|:-------------------|:----------------------|:------------------------------|:--------------------------------|
| STAGE58_BRIDGE_STATIC_AUDIT_READY |         520 | True                      | True               | True         | True               | True                       | True                            | True               | True                  | primary_failure_type_aware_v2 | Stage59 bridge integration plan |

## Axis rule

`primary_failure_type_aware_v2`

Frame-failure NOT_ENTITLED rows are allowed to have `sufficiency_label=1` when the evidence contains a complete predicate/value for the wrong subject/frame. FrameGate blocks entitlement first, so these are not generator errors.

## Corrected axis consistency checks

| check                                                             |   bad_count | pass   | rationale                                                                |
|:------------------------------------------------------------------|------------:|:-------|:-------------------------------------------------------------------------|
| SUPPORT/REFUTE rows require frame_compatible_label=1              |           0 | True   | Entitled labels require a valid frame.                                   |
| SUPPORT/REFUTE rows require predicate_covered_label=1             |           0 | True   | Entitled labels require predicate coverage.                              |
| SUPPORT/REFUTE rows require sufficiency_label=1                   |           0 | True   | Entitled labels require sufficient evidence.                             |
| SUPPORT rows require polarity_label=SUPPORT                       |           0 | True   | SUPPORT rows must have SUPPORT polarity.                                 |
| REFUTE rows require polarity_label=REFUTE                         |           0 | True   | REFUTE rows must have REFUTE polarity.                                   |
| NOT_ENTITLED frame failures require frame_compatible_label=0      |           0 | True   | Frame-failure NE is caused by incompatible frame.                        |
| NOT_ENTITLED sufficiency failures require sufficiency_label=0     |           0 | True   | Sufficiency-failure NE must have insufficient evidence.                  |
| NOT_ENTITLED predicate failures require predicate_covered_label=0 |           0 | True   | Predicate-failure NE must lack predicate coverage.                       |
| NOT_ENTITLED rows require polarity_label=NONE                     |           0 | True   | NE rows should not assert SUPPORT/REFUTE polarity.                       |
| NOT_ENTITLED rows require an allowed primary_failure_type         |           0 | True   | NE rows should be explained by frame, predicate, or sufficiency failure. |

## Allowed frame-failure NE with sufficiency_label=1

|   allowed_frame_failure_ne_with_sufficiency_1_count | interpretation                                                                   |
|----------------------------------------------------:|:---------------------------------------------------------------------------------|
|                                                  10 | Allowed: frame mismatch blocks entitlement before sufficiency/polarity decision. |

## Label mapping check

| valid_label_ids   | expected_mapping                               | label_mapping_all_rows_ok   |   bad_label_row_count |
|:------------------|:-----------------------------------------------|:----------------------------|----------------------:|
| [0, 1, 2]         | {'REFUTE': 0, 'NOT_ENTITLED': 1, 'SUPPORT': 2} | True                        |                     0 |

## Label counts

| final_label   |   label |   count |
|:--------------|--------:|--------:|
| REFUTE        |       0 |     160 |
| NOT_ENTITLED  |       1 |     200 |
| SUPPORT       |       2 |     160 |

## Family counts

| stage57_bridge_family      |   count |
|:---------------------------|--------:|
| distractor_evidence_bridge |      40 |
| entity_attribute_bridge    |     120 |
| lexical_paraphrase_bridge  |     120 |
| numeric_comparison_bridge  |     120 |
| temporal_comparison_bridge |     120 |

## Family x label counts

| stage57_bridge_family      | final_label   |   label |   count |
|:---------------------------|:--------------|--------:|--------:|
| distractor_evidence_bridge | NOT_ENTITLED  |       1 |      40 |
| entity_attribute_bridge    | REFUTE        |       0 |      40 |
| entity_attribute_bridge    | NOT_ENTITLED  |       1 |      40 |
| entity_attribute_bridge    | SUPPORT       |       2 |      40 |
| lexical_paraphrase_bridge  | REFUTE        |       0 |      40 |
| lexical_paraphrase_bridge  | NOT_ENTITLED  |       1 |      40 |
| lexical_paraphrase_bridge  | SUPPORT       |       2 |      40 |
| numeric_comparison_bridge  | REFUTE        |       0 |      40 |
| numeric_comparison_bridge  | NOT_ENTITLED  |       1 |      40 |
| numeric_comparison_bridge  | SUPPORT       |       2 |      40 |
| temporal_comparison_bridge | REFUTE        |       0 |      40 |
| temporal_comparison_bridge | NOT_ENTITLED  |       1 |      40 |
| temporal_comparison_bridge | SUPPORT       |       2 |      40 |

## Balance checks

| check              | key                                      |   expected |   actual | pass   |
|:-------------------|:-----------------------------------------|-----------:|---------:|:-------|
| family_count       | entity_attribute_bridge                  |        120 |      120 | True   |
| family_count       | numeric_comparison_bridge                |        120 |      120 | True   |
| family_count       | temporal_comparison_bridge               |        120 |      120 | True   |
| family_count       | lexical_paraphrase_bridge                |        120 |      120 | True   |
| family_count       | distractor_evidence_bridge               |         40 |       40 | True   |
| family_label_count | entity_attribute_bridge::REFUTE          |         40 |       40 | True   |
| family_label_count | entity_attribute_bridge::NOT_ENTITLED    |         40 |       40 | True   |
| family_label_count | entity_attribute_bridge::SUPPORT         |         40 |       40 | True   |
| family_label_count | numeric_comparison_bridge::REFUTE        |         40 |       40 | True   |
| family_label_count | numeric_comparison_bridge::NOT_ENTITLED  |         40 |       40 | True   |
| family_label_count | numeric_comparison_bridge::SUPPORT       |         40 |       40 | True   |
| family_label_count | temporal_comparison_bridge::REFUTE       |         40 |       40 | True   |
| family_label_count | temporal_comparison_bridge::NOT_ENTITLED |         40 |       40 | True   |
| family_label_count | temporal_comparison_bridge::SUPPORT      |         40 |       40 | True   |
| family_label_count | lexical_paraphrase_bridge::REFUTE        |         40 |       40 | True   |
| family_label_count | lexical_paraphrase_bridge::NOT_ENTITLED  |         40 |       40 | True   |
| family_label_count | lexical_paraphrase_bridge::SUPPORT       |         40 |       40 | True   |
| family_label_count | distractor_evidence_bridge::NOT_ENTITLED |         40 |       40 | True   |

## Duplicate check

|   duplicate_id_count |   duplicate_claim_evidence_label_count |
|---------------------:|---------------------------------------:|
|                    0 |                                      0 |

## Empty required fields

| field                 |   empty_count |
|:----------------------|--------------:|
| id                    |             0 |
| claim                 |             0 |
| evidence              |             0 |
| final_label           |             0 |
| stage57_bridge_family |             0 |

## Forbidden external marker hits

_empty_

## time_swap hits

_empty_

## Leakage policy

- VitaminC text used for generation: `False`
- VitaminC labels used for generation: `False`
- External metrics used for threshold tuning: `False`
- Synthetic only: `True`
- time_swap used: `False`

## Recommended next stage

Stage59 bridge integration plan: define how to combine clean main data with Stage57 bridge data without using external data for threshold tuning or checkpoint selection.
