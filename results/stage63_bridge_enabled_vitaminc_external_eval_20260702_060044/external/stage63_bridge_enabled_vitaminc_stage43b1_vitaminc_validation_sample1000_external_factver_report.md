# Stage43-C0 External Fact-Verification Evaluation

## 1. Decision

- Run name: `stage63_bridge_enabled_vitaminc_stage43b1_vitaminc_validation_sample1000`
- Decision: `STAGE43C0_EXTERNAL_FACTVER_INCOMPLETE`
- Recommendation: safe_structured_v2 could not be applied for the evaluated rows.

## 2. Dataset/input summary

- Input JSONL: `data/stage43b1_vitaminc_validation_sample1000.jsonl`
- Row count: 1000
- Prediction source: `post_training_in_memory_best_state`

## 3. Evaluation timing

`post_training_after_best_state_restore`

## 4. Base vs composed metrics

| Metric | Base | Composed | Delta |
|---|---:|---:|---:|
| accuracy | 0.322 | 0.322 | 0.0 |
| macro-F1 | 0.315319 | 0.315319 | 0.0 |

## 5. Label distributions

```json
{
  "gold": {
    "NOT_ENTITLED": 145,
    "REFUTE": 355,
    "SUPPORT": 500
  },
  "base_predictions": {
    "REFUTE": 217,
    "NOT_ENTITLED": 492,
    "SUPPORT": 291
  },
  "composed_predictions": {
    "REFUTE": 217,
    "NOT_ENTITLED": 492,
    "SUPPORT": 291
  }
}
```

## 6. Confusion matrices

```json
{
  "base": {
    "SUPPORT": {
      "SUPPORT": 156,
      "REFUTE": 99,
      "NOT_ENTITLED": 245
    },
    "REFUTE": {
      "SUPPORT": 104,
      "REFUTE": 85,
      "NOT_ENTITLED": 166
    },
    "NOT_ENTITLED": {
      "SUPPORT": 31,
      "REFUTE": 33,
      "NOT_ENTITLED": 81
    }
  },
  "composed": {
    "SUPPORT": {
      "SUPPORT": 156,
      "REFUTE": 99,
      "NOT_ENTITLED": 245
    },
    "REFUTE": {
      "SUPPORT": 104,
      "REFUTE": 85,
      "NOT_ENTITLED": 166
    },
    "NOT_ENTITLED": {
      "SUPPORT": 31,
      "REFUTE": 33,
      "NOT_ENTITLED": 81
    }
  }
}
```

## 7. Safety counters

```json
{
  "changed_row_count": 0,
  "changed_to_SUPPORT_count": 0,
  "changed_to_REFUTE_count": 0,
  "changed_to_NOT_ENTITLED_count": 0,
  "introduced_unsafe_SUPPORT_count": 0,
  "introduced_REFUTE_to_SUPPORT_count": 0,
  "introduced_SUPPORT_to_REFUTE_count": 0,
  "total_composed_wrong_SUPPORT_count": 135,
  "total_composed_SUPPORT_to_REFUTE_count": 99,
  "blocker_fired_count": 0,
  "recovery_fired_count": 0
}
```

## 8. Changed-row analysis

None.

## 9. Error analysis

None.

## 10. Risks

- Stage43 external data is evaluation-only and must not guide training or selection.
- Climate-FEVER failures are cross-domain limitations, not training signals.
- safe_structured_v2 composer output was unavailable for every evaluated row.

## Stage43-C1 Diagnostic Audit

### Prediction collapse summary

```json
{
  "base_prediction_counts": {
    "REFUTE": 217,
    "NOT_ENTITLED": 492,
    "SUPPORT": 291
  },
  "per_gold_prediction_distribution": {
    "SUPPORT": {
      "SUPPORT": 156,
      "REFUTE": 99,
      "NOT_ENTITLED": 245
    },
    "REFUTE": {
      "SUPPORT": 104,
      "REFUTE": 85,
      "NOT_ENTITLED": 166
    },
    "NOT_ENTITLED": {
      "SUPPORT": 31,
      "REFUTE": 33,
      "NOT_ENTITLED": 81
    }
  },
  "max_probability_summary": {
    "min": 0.381337,
    "p25": 0.65122,
    "median": 0.654423,
    "p75": 0.970962,
    "p90": 0.994969,
    "p95": 0.996336,
    "max": 0.997952
  },
  "prediction_entropy_summary": {
    "min": 0.016139,
    "p25": 0.149557,
    "median": 0.8842,
    "p75": 0.885451,
    "p90": 0.930851,
    "p95": 0.984034,
    "max": 1.091303
  }
}
```

### Token length/truncation summary

```json
{
  "external_token_length_summary": {
    "min": 16.0,
    "p25": 37.0,
    "median": 48.0,
    "p75": 60.0,
    "p90": 78.0,
    "p95": 91.05,
    "max": 278.0
  },
  "external_truncation_count": 68,
  "external_truncation_rate": 0.068
}
```

### Input template/path audit

```json
{
  "external_input_template": "v5.encode_mamba_records: tokenizer claim span truncated to floor((max_length - 1) / 2), separator token, tokenizer evidence span truncated to remaining budget; exact claim/evidence masks",
  "controlled_dev_input_template": "v5.encode_mamba_records: tokenizer claim span truncated to floor((max_length - 1) / 2), separator token, tokenizer evidence span truncated to remaining budget; exact claim/evidence masks",
  "external_uses_same_prediction_path_as_dev": true,
  "external_tokenizer_source": "state-spaces/mamba-130m-hf",
  "controlled_dev_tokenizer_source": "state-spaces/mamba-130m-hf",
  "external_max_length": 128,
  "controlled_dev_max_length": 128
}
```

### Label mapping audit

```json
{
  "external_uses_same_label_mapping_as_dev": true,
  "label_id_to_name": {
    "0": "REFUTE",
    "1": "NOT_ENTITLED",
    "2": "SUPPORT"
  },
  "name_to_label_id": {
    "NOT_ENTITLED": 1,
    "REFUTE": 0,
    "SUPPORT": 2
  }
}
```

### Composer availability audit

```json
{
  "composer_availability_summary": {
    "requested_composer_mode": "safe_structured_v2",
    "composer_available_row_count": 0,
    "composer_unavailable_row_count": 1000,
    "composer_unavailable_reasons": {
      "missing_source_shadow_label": 1000
    },
    "required_composer_fields_present_counts": {
      "stage37_final_shadow_label": 0,
      "stage36_final_shadow_label": 0,
      "stage32_shadow_label": 0,
      "stage36_support_blocker_fired": 0,
      "stage37_recovered_from_label": 0,
      "stage33_structured_coverage_reason": 0,
      "stage33_structured_coverage_route": 0,
      "stage33_structured_coverage_label": 0,
      "stage33_conditional_override_type": 0,
      "stage36_conditional_override_type": 0,
      "stage37_conditional_override_type": 0
    }
  },
  "stage36_stage37_stage39_field_presence": {
    "stage32_shadow_label": 0,
    "stage32_shadow_reason": 0,
    "stage33_structured_coverage_reason": 0,
    "stage33_structured_coverage_route": 0,
    "stage33_structured_coverage_label": 0,
    "stage33_conditional_override_type": 0,
    "stage36_final_shadow_label": 0,
    "stage36_support_blocker_fired": 0,
    "stage36_support_blocker_reasons": 0,
    "stage36_conditional_override_type": 0,
    "stage37_final_shadow_label": 0,
    "stage37_safe_recovery_fired": 0,
    "stage37_safe_recovery_reasons": 0,
    "stage37_recovered_from_label": 0,
    "stage37_conditional_override_type": 0,
    "stage39_source_shadow_label": 0,
    "stage39_composed_final_label": 1000,
    "stage39_composer_action": 1000,
    "stage39_composer_reason": 1000,
    "stage39_blocked_by_missing_source": 1000
  }
}
```

### Sample collapsed rows

```json
[
  {
    "row_index": 2,
    "id": "tals_vitaminc_validation_sample1000_validation_61972",
    "gold_label": "SUPPORT",
    "base_prediction": "NOT_ENTITLED",
    "claim": "Efraim Diveroli had a four-year sentence .",
    "evidence": "Diveroli was sentenced to four years in federal prison .",
    "source_dataset": "tals_vitaminc_validation_sample1000",
    "max_probability": 0.654421,
    "prediction_entropy": 0.884204,
    "composer_reason": "missing_source_shadow_label"
  },
  {
    "row_index": 3,
    "id": "tals_vitaminc_validation_sample1000_validation_47744",
    "gold_label": "SUPPORT",
    "base_prediction": "NOT_ENTITLED",
    "claim": "Emma Watson was born before 1995 .",
    "evidence": "Emma Charlotte Duerre Watson -LRB- born 15 April 1990 -RRB- is a French-British actress , model , and activist .",
    "source_dataset": "tals_vitaminc_validation_sample1000",
    "max_probability": 0.654014,
    "prediction_entropy": 0.884745,
    "composer_reason": "missing_source_shadow_label"
  },
  {
    "row_index": 4,
    "id": "tals_vitaminc_validation_sample1000_validation_59780",
    "gold_label": "SUPPORT",
    "base_prediction": "NOT_ENTITLED",
    "claim": "Tilda Swinton is a singer .",
    "evidence": "After participating in Celine Dion 's tour , Swinton released her album `` True '' .",
    "source_dataset": "tals_vitaminc_validation_sample1000",
    "max_probability": 0.654423,
    "prediction_entropy": 0.884201,
    "composer_reason": "missing_source_shadow_label"
  },
  {
    "row_index": 6,
    "id": "tals_vitaminc_validation_sample1000_validation_55992",
    "gold_label": "SUPPORT",
    "base_prediction": "NOT_ENTITLED",
    "claim": "Recovery is an Eminem album .",
    "evidence": "Recovery is the seventh studio album by American rapper Eminem .",
    "source_dataset": "tals_vitaminc_validation_sample1000",
    "max_probability": 0.649475,
    "prediction_entropy": 0.890712,
    "composer_reason": "missing_source_shadow_label"
  },
  {
    "row_index": 8,
    "id": "tals_vitaminc_validation_sample1000_validation_17884",
    "gold_label": "SUPPORT",
    "base_prediction": "NOT_ENTITLED",
    "claim": "Atl\u00e9tico Madrid 's players had virtually given up 10 minutes before the end of the 2014 UEFA Champions League Final .",
    "evidence": "With only 10 minutes left , it seemed Atl\u00e9tico 's comeback was mission impossible , with its ' players virtually giving up .",
    "source_dataset": "tals_vitaminc_validation_sample1000",
    "max_probability": 0.64883,
    "prediction_entropy": 0.891578,
    "composer_reason": "missing_source_shadow_label"
  },
  {
    "row_index": 9,
    "id": "tals_vitaminc_validation_sample1000_validation_28252",
    "gold_label": "REFUTE",
    "base_prediction": "NOT_ENTITLED",
    "claim": "Fewer than 300 laboratory-confirmed cases of COVID-19 have been confirmed in China .",
    "evidence": "China confirms sharp rise in cases of SARS-like virus across the country As of 21 January , the number of laboratory-confirmed cases stands at 303 , including 298 in China , two in Thailand , one in Japan , one in South Korea and one in Taiwan .",
    "source_dataset": "tals_vitaminc_validation_sample1000",
    "max_probability": 0.645535,
    "prediction_entropy": 0.895864,
    "composer_reason": "missing_source_shadow_label"
  },
  {
    "row_index": 11,
    "id": "tals_vitaminc_validation_sample1000_validation_33552",
    "gold_label": "REFUTE",
    "base_prediction": "NOT_ENTITLED",
    "claim": "South Korea had more positive cases and a higher number of coronavirus-related deaths than Italy as of February 2020 .",
    "evidence": "As of late February , Italy was hit harder than anywhere else in the EU by the outbreak , and is the country with the third-highest number of positive cases , as well as deaths , in the world , with China and Iran surpassing Italy .",
    "source_dataset": "tals_vitaminc_validation_sample1000",
    "max_probability": 0.594863,
    "prediction_entropy": 0.955363,
    "composer_reason": "missing_source_shadow_label"
  },
  {
    "row_index": 14,
    "id": "tals_vitaminc_validation_sample1000_validation_23116",
    "gold_label": "SUPPORT",
    "base_prediction": "NOT_ENTITLED",
    "claim": "As reported by the Daily Mail , Phil Tufnell married Dawn .",
    "evidence": "Daily Mail Retrieved 10 June 2008 He is now married to Dawn .",
    "source_dataset": "tals_vitaminc_validation_sample1000",
    "max_probability": 0.654421,
    "prediction_entropy": 0.884203,
    "composer_reason": "missing_source_shadow_label"
  },
  {
    "row_index": 16,
    "id": "tals_vitaminc_validation_sample1000_validation_55315",
    "gold_label": "REFUTE",
    "base_prediction": "NOT_ENTITLED",
    "claim": "Philomena is only a book .",
    "evidence": "Philomena is a 2013 British drama film directed by Stephen Frears , based on the book The Lost Child of Philomena Lee by journalist Martin Sixsmith .",
    "source_dataset": "tals_vitaminc_validation_sample1000",
    "max_probability": 0.65398,
    "prediction_entropy": 0.884791,
    "composer_reason": "missing_source_shadow_label"
  },
  {
    "row_index": 19,
    "id": "tals_vitaminc_validation_sample1000_validation_49256",
    "gold_label": "SUPPORT",
    "base_prediction": "NOT_ENTITLED",
    "claim": "Harold Macmillan worked in sports .",
    "evidence": "After his resignation , Macmillan lived out a long retirement as an elder sports coach .",
    "source_dataset": "tals_vitaminc_validation_sample1000",
    "max_probability": 0.654424,
    "prediction_entropy": 0.8842,
    "composer_reason": "missing_source_shadow_label"
  },
  {
    "row_index": 20,
    "id": "tals_vitaminc_validation_sample1000_validation_41341",
    "gold_label": "REFUTE",
    "base_prediction": "NOT_ENTITLED",
    "claim": "`` Roar '' was nominated for the song of the Year .",
    "evidence": "Scared '' was nominated for Song of the Year and Best Pop Solo Performance at the 56th Annual Grammy Awards .",
    "source_dataset": "tals_vitaminc_validation_sample1000",
    "max_probability": 0.652704,
    "prediction_entropy": 0.88648,
    "composer_reason": "missing_source_shadow_label"
  },
  {
    "row_index": 21,
    "id": "tals_vitaminc_validation_sample1000_validation_56563",
    "gold_label": "REFUTE",
    "base_prediction": "NOT_ENTITLED",
    "claim": "Roswell is exclusively a Turkish TV series .",
    "evidence": "Roswell is an American science fiction television series developed , produced , and co-written by Jason Katims .",
    "source_dataset": "tals_vitaminc_validation_sample1000",
    "max_probability": 0.654187,
    "prediction_entropy": 0.884515,
    "composer_reason": "missing_source_shadow_label"
  },
  {
    "row_index": 22,
    "id": "tals_vitaminc_validation_sample1000_validation_46869",
    "gold_label": "REFUTE",
    "base_prediction": "NOT_ENTITLED",
    "claim": "Illumination Mac Guff animated the film Despicable Me 2 .",
    "evidence": "Produced by Illumination Entertainment for Universal Pictures , and animated by Pixar , the film is directed by Pierre Coffin and Chris Renaud , and written by Cinco Paul and Ken Daurio .",
    "source_dataset": "tals_vitaminc_validation_sample1000",
    "max_probability": 0.62908,
    "prediction_entropy": 0.916522,
    "composer_reason": "missing_source_shadow_label"
  },
  {
    "row_index": 23,
    "id": "tals_vitaminc_validation_sample1000_validation_2441",
    "gold_label": "REFUTE",
    "base_prediction": "NOT_ENTITLED",
    "claim": "Based on 19 critics , the film scored has 72 out of 100 .",
    "evidence": "`` On Metacritic , the film has a weighted average score of 72 out of 100 , based on 16 critics , indicating `` '' generally favorable reviews '' '' . ''",
    "source_dataset": "tals_vitaminc_validation_sample1000",
    "max_probability": 0.621911,
    "prediction_entropy": 0.925187,
    "composer_reason": "missing_source_shadow_label"
  },
  {
    "row_index": 26,
    "id": "tals_vitaminc_validation_sample1000_validation_50619",
    "gold_label": "REFUTE",
    "base_prediction": "NOT_ENTITLED",
    "claim": "James Earl Jones was a voice actor in a movie .",
    "evidence": "James Earl Jones is also known for his leading roles as Darth Vader in the Star Wars film series and Mufasa in Disney 's The Lion King Live-Action as well as many other films , stage , and television roles .",
    "source_dataset": "tals_vitaminc_validation_sample1000",
    "max_probability": 0.562214,
    "prediction_entropy": 0.975113,
    "composer_reason": "missing_source_shadow_label"
  },
  {
    "row_index": 31,
    "id": "tals_vitaminc_validation_sample1000_validation_40742",
    "gold_label": "SUPPORT",
    "base_prediction": "NOT_ENTITLED",
    "claim": "Part of the plot of `` Miracle at St. Anna '' takes place in the winter of 1984 .",
    "evidence": "The film was released on September 26 , 2008 , and is set during World War II , in fall of 1944 in Tuscany and in the winter of 1984 in New York City and Rome.",
    "source_dataset": "tals_vitaminc_validation_sample1000",
    "max_probability": 0.654423,
    "prediction_entropy": 0.884201,
    "composer_reason": "missing_source_shadow_label"
  },
  {
    "row_index": 33,
    "id": "tals_vitaminc_validation_sample1000_validation_44414",
    "gold_label": "SUPPORT",
    "base_prediction": "NOT_ENTITLED",
    "claim": "Bones is solely a television pitch .",
    "evidence": "Bones was pitched to the network , but was rejected and the idea has remained shelved ever since .",
    "source_dataset": "tals_vitaminc_validation_sample1000",
    "max_probability": 0.654422,
    "prediction_entropy": 0.884203,
    "composer_reason": "missing_source_shadow_label"
  },
  {
    "row_index": 34,
    "id": "tals_vitaminc_validation_sample1000_validation_683",
    "gold_label": "SUPPORT",
    "base_prediction": "NOT_ENTITLED",
    "claim": "Erwin Schrodinger 's mother was a Lutheran with English roots .",
    "evidence": "His mother was of half Austrian and half English descent ; his father was Catholic and his mother was Lutheran .",
    "source_dataset": "tals_vitaminc_validation_sample1000",
    "max_probability": 0.443481,
    "prediction_entropy": 1.01676,
    "composer_reason": "missing_source_shadow_label"
  },
  {
    "row_index": 35,
    "id": "tals_vitaminc_validation_sample1000_validation_6974",
    "gold_label": "SUPPORT",
    "base_prediction": "NOT_ENTITLED",
    "claim": "In the movie Dumb and Dumberer : When Harry Met Lloyd , the characters Harry and Lloyd head off to enjoy the rest of the day when they are approached by two girls .",
    "evidence": "As Harry and Lloyd head off to enjoy the rest of the day , two girls in a sports car come up to them and offer to bring them to a huge party .",
    "source_dataset": "tals_vitaminc_validation_sample1000",
    "max_probability": 0.65421,
    "prediction_entropy": 0.884484,
    "composer_reason": "missing_source_shadow_label"
  },
  {
    "row_index": 38,
    "id": "tals_vitaminc_validation_sample1000_validation_43137",
    "gold_label": "REFUTE",
    "base_prediction": "NOT_ENTITLED",
    "claim": "Ann Richards was the Governor of New York from 1991 to 1995 .",
    "evidence": "Dorothy Ann Willis Richards -LRB- September 1 , 1933 -- September 13 , 2006 -RRB- was an American politician and the 45th Governor of Texas from 1991 to 1995 .",
    "source_dataset": "tals_vitaminc_validation_sample1000",
    "max_probability": 0.654006,
    "prediction_entropy": 0.884756,
    "composer_reason": "missing_source_shadow_label"
  }
]
```

### Diagnostic conclusion

Diagnostic-only audit: excessive token truncation is not indicated by the token audit; external prediction path matches the controlled dev encode/forward/export path; label mapping matches the controlled dev mapping; missing claim/evidence fields are not indicated; model confidence is available for collapse inspection. safe_structured_v2 composer output is unavailable because the external export rows do not carry the required Stage32/36/37/39 intermediate structures for every row: stage37_final_shadow_label, stage36_final_shadow_label, stage32_shadow_label, stage36_support_blocker_fired, stage37_recovered_from_label, stage33_structured_coverage_reason, stage33_structured_coverage_route, stage33_structured_coverage_label, stage33_conditional_override_type, stage36_conditional_override_type, stage37_conditional_override_type Decisions remain Stage43-C0 decisions and diagnostics do not convert INCOMPLETE into PASS.

### Non-leakage statement

Stage43-B1 external fact-verification data is evaluation-only. It is not used for training, calibration, threshold selection, checkpoint selection, loss design, model selection, or composer behavior changes.

## Stage43-C2 Shadow Export Audit

```json
{
  "stage43c2_shadow_export_enabled": false,
  "stage43c2_shadow_export_available": false,
  "stage43c2_shadow_export_missing_dependencies": [
    "stage43_external_enable_shadow_export_disabled"
  ],
  "stage43c2_required_shadow_fields_present_counts": {
    "stage32_shadow_label": 0,
    "stage32_shadow_reason": 0,
    "stage33_structured_coverage_label": 0,
    "stage33_structured_coverage_reason": 0,
    "stage33_structured_coverage_route": 0,
    "stage36_final_shadow_label": 0,
    "stage36_support_blocker_fired": 0,
    "stage36_support_blocker_reasons": 0,
    "stage37_final_shadow_label": 0,
    "stage37_safe_recovery_fired": 0,
    "stage37_safe_recovery_reasons": 0,
    "stage37_recovered_from_label": 0,
    "stage39_source_shadow_label": 1000,
    "stage39_composed_final_label": 1000,
    "stage39_composer_action": 1000,
    "stage39_composer_reason": 1000
  },
  "stage43c2_composer_application_count": 0,
  "stage43c2_composer_blocked_count": 1000,
  "stage43c2_composer_blocked_reasons": {
    "missing_source_shadow_label": 1000
  },
  "stage43c2_reused_internal_export_path": "prediction_records_v6b -> build_stage32_owner_state -> compute_stage36_support_safety_blocker -> compute_stage37_safe_support_recovery -> compute_stage39_final_composer",
  "stage43c2_forced_eval_only_exports": []
}
```

### Stage43-C2 conclusion

Stage43-C2 shadow export was not requested; safe_structured_v2 may remain unavailable if source shadow labels are absent.

### Stage43-C2 next action

Run with --stage43-external-enable-shadow-export for diagnostic-only external shadow/composer export.

## 11. Recommendation

safe_structured_v2 could not be applied for the evaluated rows.

## 12. Leakage policy

Stage43-B1 external fact-verification data is evaluation-only. It is not used for training, calibration, threshold selection, checkpoint selection, loss design, model selection, or composer behavior changes.
