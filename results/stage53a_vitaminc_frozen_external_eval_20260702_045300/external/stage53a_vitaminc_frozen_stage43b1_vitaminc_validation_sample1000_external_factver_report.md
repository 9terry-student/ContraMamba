# Stage43-C0 External Fact-Verification Evaluation

## 1. Decision

- Run name: `stage53a_vitaminc_frozen_stage43b1_vitaminc_validation_sample1000`
- Decision: `STAGE43C0_EXTERNAL_FACTVER_UNSAFE`
- Recommendation: Composition introduced unsafe SUPPORT or SUPPORT/REFUTE transition(s).

## 2. Dataset/input summary

- Input JSONL: `data/stage43b1_vitaminc_validation_sample1000.jsonl`
- Row count: 1000
- Prediction source: `post_training_in_memory_best_state`

## 3. Evaluation timing

`post_training_after_best_state_restore`

## 4. Base vs composed metrics

| Metric | Base | Composed | Delta |
|---|---:|---:|---:|
| accuracy | 0.156 | 0.156 | 0.0 |
| macro-F1 | 0.115986 | 0.116717 | 0.000731 |

## 5. Label distributions

```json
{
  "gold": {
    "NOT_ENTITLED": 145,
    "REFUTE": 355,
    "SUPPORT": 500
  },
  "base_predictions": {
    "REFUTE": 33,
    "NOT_ENTITLED": 946,
    "SUPPORT": 21
  },
  "composed_predictions": {
    "REFUTE": 33,
    "NOT_ENTITLED": 944,
    "SUPPORT": 23
  }
}
```

## 6. Confusion matrices

```json
{
  "base": {
    "SUPPORT": {
      "SUPPORT": 16,
      "REFUTE": 12,
      "NOT_ENTITLED": 472
    },
    "REFUTE": {
      "SUPPORT": 3,
      "REFUTE": 9,
      "NOT_ENTITLED": 343
    },
    "NOT_ENTITLED": {
      "SUPPORT": 2,
      "REFUTE": 12,
      "NOT_ENTITLED": 131
    }
  },
  "composed": {
    "SUPPORT": {
      "SUPPORT": 17,
      "REFUTE": 12,
      "NOT_ENTITLED": 471
    },
    "REFUTE": {
      "SUPPORT": 3,
      "REFUTE": 9,
      "NOT_ENTITLED": 343
    },
    "NOT_ENTITLED": {
      "SUPPORT": 3,
      "REFUTE": 12,
      "NOT_ENTITLED": 130
    }
  }
}
```

## 7. Safety counters

```json
{
  "changed_row_count": 2,
  "changed_to_SUPPORT_count": 2,
  "changed_to_REFUTE_count": 0,
  "changed_to_NOT_ENTITLED_count": 0,
  "introduced_unsafe_SUPPORT_count": 1,
  "introduced_REFUTE_to_SUPPORT_count": 0,
  "introduced_SUPPORT_to_REFUTE_count": 0,
  "total_composed_wrong_SUPPORT_count": 6,
  "total_composed_SUPPORT_to_REFUTE_count": 12,
  "blocker_fired_count": 0,
  "recovery_fired_count": 0
}
```

## 8. Changed-row analysis

```json
[
  {
    "id": "tals_vitaminc_validation_sample1000_validation_20055",
    "gold_label": "NOT_ENTITLED",
    "base_prediction": "NOT_ENTITLED",
    "composed_prediction": "SUPPORT",
    "composer_reason": "support_only_from_stage37"
  },
  {
    "id": "tals_vitaminc_validation_sample1000_validation_23498",
    "gold_label": "SUPPORT",
    "base_prediction": "NOT_ENTITLED",
    "composed_prediction": "SUPPORT",
    "composer_reason": "support_only_from_stage37"
  }
]
```

## 9. Error analysis

None.

## 10. Risks

- Stage43 external data is evaluation-only and must not guide training or selection.
- Climate-FEVER failures are cross-domain limitations, not training signals.

## Stage43-C1 Diagnostic Audit

### Prediction collapse summary

```json
{
  "base_prediction_counts": {
    "REFUTE": 33,
    "NOT_ENTITLED": 946,
    "SUPPORT": 21
  },
  "per_gold_prediction_distribution": {
    "SUPPORT": {
      "SUPPORT": 16,
      "REFUTE": 12,
      "NOT_ENTITLED": 472
    },
    "REFUTE": {
      "SUPPORT": 3,
      "REFUTE": 9,
      "NOT_ENTITLED": 343
    },
    "NOT_ENTITLED": {
      "SUPPORT": 2,
      "REFUTE": 12,
      "NOT_ENTITLED": 131
    }
  },
  "max_probability_summary": {
    "min": 0.387999,
    "p25": 0.608436,
    "median": 0.60966,
    "p75": 0.609695,
    "p90": 0.609697,
    "p95": 0.609697,
    "max": 0.98911
  },
  "prediction_entropy_summary": {
    "min": 0.0676,
    "p25": 0.939423,
    "median": 0.939463,
    "p75": 0.940835,
    "p90": 0.963018,
    "p95": 1.010943,
    "max": 1.073927
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
    "composer_available_row_count": 1000,
    "composer_unavailable_row_count": 0,
    "composer_unavailable_reasons": {},
    "required_composer_fields_present_counts": {
      "stage37_final_shadow_label": 1000,
      "stage36_final_shadow_label": 1000,
      "stage32_shadow_label": 1000,
      "stage36_support_blocker_fired": 1000,
      "stage37_recovered_from_label": 0,
      "stage33_structured_coverage_reason": 1000,
      "stage33_structured_coverage_route": 1000,
      "stage33_structured_coverage_label": 1000,
      "stage33_conditional_override_type": 1000,
      "stage36_conditional_override_type": 0,
      "stage37_conditional_override_type": 0
    }
  },
  "stage36_stage37_stage39_field_presence": {
    "stage32_shadow_label": 1000,
    "stage32_shadow_reason": 1000,
    "stage33_structured_coverage_reason": 1000,
    "stage33_structured_coverage_route": 1000,
    "stage33_structured_coverage_label": 1000,
    "stage33_conditional_override_type": 1000,
    "stage36_final_shadow_label": 1000,
    "stage36_support_blocker_fired": 1000,
    "stage36_support_blocker_reasons": 1000,
    "stage36_conditional_override_type": 0,
    "stage37_final_shadow_label": 1000,
    "stage37_safe_recovery_fired": 1000,
    "stage37_safe_recovery_reasons": 1000,
    "stage37_recovered_from_label": 0,
    "stage37_conditional_override_type": 0,
    "stage39_source_shadow_label": 1000,
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
    "row_index": 1,
    "id": "tals_vitaminc_validation_sample1000_validation_62847",
    "gold_label": "REFUTE",
    "base_prediction": "NOT_ENTITLED",
    "claim": "The Cincinnati Kid is a boy .",
    "evidence": "He considers it the film that allowed him to transition from the lighter comedic films he had previously been making and take on more serious films and subjects .",
    "source_dataset": "tals_vitaminc_validation_sample1000",
    "max_probability": 0.60926,
    "prediction_entropy": 0.939918,
    "composer_reason": "safe_structured_v2_no_qualifying_transition"
  },
  {
    "row_index": 2,
    "id": "tals_vitaminc_validation_sample1000_validation_61972",
    "gold_label": "SUPPORT",
    "base_prediction": "NOT_ENTITLED",
    "claim": "Efraim Diveroli had a four-year sentence .",
    "evidence": "Diveroli was sentenced to four years in federal prison .",
    "source_dataset": "tals_vitaminc_validation_sample1000",
    "max_probability": 0.609697,
    "prediction_entropy": 0.939421,
    "composer_reason": "safe_structured_v2_no_qualifying_transition"
  },
  {
    "row_index": 3,
    "id": "tals_vitaminc_validation_sample1000_validation_47744",
    "gold_label": "SUPPORT",
    "base_prediction": "NOT_ENTITLED",
    "claim": "Emma Watson was born before 1995 .",
    "evidence": "Emma Charlotte Duerre Watson -LRB- born 15 April 1990 -RRB- is a French-British actress , model , and activist .",
    "source_dataset": "tals_vitaminc_validation_sample1000",
    "max_probability": 0.609687,
    "prediction_entropy": 0.939433,
    "composer_reason": "safe_structured_v2_no_qualifying_transition"
  },
  {
    "row_index": 4,
    "id": "tals_vitaminc_validation_sample1000_validation_59780",
    "gold_label": "SUPPORT",
    "base_prediction": "NOT_ENTITLED",
    "claim": "Tilda Swinton is a singer .",
    "evidence": "After participating in Celine Dion 's tour , Swinton released her album `` True '' .",
    "source_dataset": "tals_vitaminc_validation_sample1000",
    "max_probability": 0.544943,
    "prediction_entropy": 0.999821,
    "composer_reason": "safe_structured_v2_no_qualifying_transition"
  },
  {
    "row_index": 5,
    "id": "tals_vitaminc_validation_sample1000_validation_62168",
    "gold_label": "SUPPORT",
    "base_prediction": "NOT_ENTITLED",
    "claim": "I Kissed a Girl is a work of literature .",
    "evidence": "I Kissed a Girl is a novel written by Katy Perry , first published on April 28 , 2008 , by Capitol Records .",
    "source_dataset": "tals_vitaminc_validation_sample1000",
    "max_probability": 0.572103,
    "prediction_entropy": 0.979278,
    "composer_reason": "safe_structured_v2_no_qualifying_transition"
  },
  {
    "row_index": 6,
    "id": "tals_vitaminc_validation_sample1000_validation_55992",
    "gold_label": "SUPPORT",
    "base_prediction": "NOT_ENTITLED",
    "claim": "Recovery is an Eminem album .",
    "evidence": "Recovery is the seventh studio album by American rapper Eminem .",
    "source_dataset": "tals_vitaminc_validation_sample1000",
    "max_probability": 0.609697,
    "prediction_entropy": 0.939421,
    "composer_reason": "safe_structured_v2_no_qualifying_transition"
  },
  {
    "row_index": 8,
    "id": "tals_vitaminc_validation_sample1000_validation_17884",
    "gold_label": "SUPPORT",
    "base_prediction": "NOT_ENTITLED",
    "claim": "Atl\u00e9tico Madrid 's players had virtually given up 10 minutes before the end of the 2014 UEFA Champions League Final .",
    "evidence": "With only 10 minutes left , it seemed Atl\u00e9tico 's comeback was mission impossible , with its ' players virtually giving up .",
    "source_dataset": "tals_vitaminc_validation_sample1000",
    "max_probability": 0.607999,
    "prediction_entropy": 0.941345,
    "composer_reason": "safe_structured_v2_no_qualifying_transition"
  },
  {
    "row_index": 9,
    "id": "tals_vitaminc_validation_sample1000_validation_28252",
    "gold_label": "REFUTE",
    "base_prediction": "NOT_ENTITLED",
    "claim": "Fewer than 300 laboratory-confirmed cases of COVID-19 have been confirmed in China .",
    "evidence": "China confirms sharp rise in cases of SARS-like virus across the country As of 21 January , the number of laboratory-confirmed cases stands at 303 , including 298 in China , two in Thailand , one in Japan , one in South Korea and one in Taiwan .",
    "source_dataset": "tals_vitaminc_validation_sample1000",
    "max_probability": 0.609692,
    "prediction_entropy": 0.939426,
    "composer_reason": "safe_structured_v2_no_qualifying_transition"
  },
  {
    "row_index": 10,
    "id": "tals_vitaminc_validation_sample1000_validation_42054",
    "gold_label": "REFUTE",
    "base_prediction": "NOT_ENTITLED",
    "claim": "Critics consider that To Pimp a Butterfly is among the best albums of all time .",
    "evidence": "It is considered as the greatest album of all time by many .",
    "source_dataset": "tals_vitaminc_validation_sample1000",
    "max_probability": 0.609286,
    "prediction_entropy": 0.939889,
    "composer_reason": "safe_structured_v2_no_qualifying_transition"
  },
  {
    "row_index": 11,
    "id": "tals_vitaminc_validation_sample1000_validation_33552",
    "gold_label": "REFUTE",
    "base_prediction": "NOT_ENTITLED",
    "claim": "South Korea had more positive cases and a higher number of coronavirus-related deaths than Italy as of February 2020 .",
    "evidence": "As of late February , Italy was hit harder than anywhere else in the EU by the outbreak , and is the country with the third-highest number of positive cases , as well as deaths , in the world , with China and Iran surpassing Italy .",
    "source_dataset": "tals_vitaminc_validation_sample1000",
    "max_probability": 0.609695,
    "prediction_entropy": 0.939423,
    "composer_reason": "safe_structured_v2_no_qualifying_transition"
  },
  {
    "row_index": 12,
    "id": "tals_vitaminc_validation_sample1000_validation_7531",
    "gold_label": "SUPPORT",
    "base_prediction": "NOT_ENTITLED",
    "claim": "Raven Software was founded before 1990 .",
    "evidence": "Raven Software was founded in 1988 by brothers Brian and Steve Raffel .",
    "source_dataset": "tals_vitaminc_validation_sample1000",
    "max_probability": 0.609529,
    "prediction_entropy": 0.939612,
    "composer_reason": "safe_structured_v2_no_qualifying_transition"
  },
  {
    "row_index": 14,
    "id": "tals_vitaminc_validation_sample1000_validation_23116",
    "gold_label": "SUPPORT",
    "base_prediction": "NOT_ENTITLED",
    "claim": "As reported by the Daily Mail , Phil Tufnell married Dawn .",
    "evidence": "Daily Mail Retrieved 10 June 2008 He is now married to Dawn .",
    "source_dataset": "tals_vitaminc_validation_sample1000",
    "max_probability": 0.60969,
    "prediction_entropy": 0.939429,
    "composer_reason": "safe_structured_v2_no_qualifying_transition"
  },
  {
    "row_index": 15,
    "id": "tals_vitaminc_validation_sample1000_validation_5323",
    "gold_label": "SUPPORT",
    "base_prediction": "NOT_ENTITLED",
    "claim": "The MLS decided that the 30th team will pay a $ 325 million entrance fee .",
    "evidence": "MLS has also announced that the ownership groups of the 28th and 29th teams will each pay a $ 200\ufffdmillion entrance fee and that the ownership group of the 30th team will pay a $ 325\ufffdmillion entrance fee .",
    "source_dataset": "tals_vitaminc_validation_sample1000",
    "max_probability": 0.587008,
    "prediction_entropy": 0.964153,
    "composer_reason": "safe_structured_v2_no_qualifying_transition"
  },
  {
    "row_index": 16,
    "id": "tals_vitaminc_validation_sample1000_validation_55315",
    "gold_label": "REFUTE",
    "base_prediction": "NOT_ENTITLED",
    "claim": "Philomena is only a book .",
    "evidence": "Philomena is a 2013 British drama film directed by Stephen Frears , based on the book The Lost Child of Philomena Lee by journalist Martin Sixsmith .",
    "source_dataset": "tals_vitaminc_validation_sample1000",
    "max_probability": 0.609696,
    "prediction_entropy": 0.939422,
    "composer_reason": "safe_structured_v2_no_qualifying_transition"
  },
  {
    "row_index": 17,
    "id": "tals_vitaminc_validation_sample1000_validation_57634",
    "gold_label": "SUPPORT",
    "base_prediction": "NOT_ENTITLED",
    "claim": "So You Think You Can Dance is a competition .",
    "evidence": "So You Think You Can Dance is an American televised dance competition show that airs on Fox in the United States and is the flagship series of the international So You Think You Can Dance television franchise .",
    "source_dataset": "tals_vitaminc_validation_sample1000",
    "max_probability": 0.609687,
    "prediction_entropy": 0.939432,
    "composer_reason": "safe_structured_v2_no_qualifying_transition"
  },
  {
    "row_index": 18,
    "id": "tals_vitaminc_validation_sample1000_validation_23531",
    "gold_label": "REFUTE",
    "base_prediction": "NOT_ENTITLED",
    "claim": "After 2015 , Richard Keogh had won 18 caps .",
    "evidence": "As of September 2013 , he has won 18 caps and scored once.",
    "source_dataset": "tals_vitaminc_validation_sample1000",
    "max_probability": 0.60645,
    "prediction_entropy": 0.943088,
    "composer_reason": "safe_structured_v2_no_qualifying_transition"
  },
  {
    "row_index": 19,
    "id": "tals_vitaminc_validation_sample1000_validation_49256",
    "gold_label": "SUPPORT",
    "base_prediction": "NOT_ENTITLED",
    "claim": "Harold Macmillan worked in sports .",
    "evidence": "After his resignation , Macmillan lived out a long retirement as an elder sports coach .",
    "source_dataset": "tals_vitaminc_validation_sample1000",
    "max_probability": 0.609697,
    "prediction_entropy": 0.939421,
    "composer_reason": "safe_structured_v2_no_qualifying_transition"
  },
  {
    "row_index": 20,
    "id": "tals_vitaminc_validation_sample1000_validation_41341",
    "gold_label": "REFUTE",
    "base_prediction": "NOT_ENTITLED",
    "claim": "`` Roar '' was nominated for the song of the Year .",
    "evidence": "Scared '' was nominated for Song of the Year and Best Pop Solo Performance at the 56th Annual Grammy Awards .",
    "source_dataset": "tals_vitaminc_validation_sample1000",
    "max_probability": 0.600834,
    "prediction_entropy": 0.949267,
    "composer_reason": "safe_structured_v2_no_qualifying_transition"
  },
  {
    "row_index": 21,
    "id": "tals_vitaminc_validation_sample1000_validation_56563",
    "gold_label": "REFUTE",
    "base_prediction": "NOT_ENTITLED",
    "claim": "Roswell is exclusively a Turkish TV series .",
    "evidence": "Roswell is an American science fiction television series developed , produced , and co-written by Jason Katims .",
    "source_dataset": "tals_vitaminc_validation_sample1000",
    "max_probability": 0.609694,
    "prediction_entropy": 0.939424,
    "composer_reason": "safe_structured_v2_no_qualifying_transition"
  },
  {
    "row_index": 22,
    "id": "tals_vitaminc_validation_sample1000_validation_46869",
    "gold_label": "REFUTE",
    "base_prediction": "NOT_ENTITLED",
    "claim": "Illumination Mac Guff animated the film Despicable Me 2 .",
    "evidence": "Produced by Illumination Entertainment for Universal Pictures , and animated by Pixar , the film is directed by Pierre Coffin and Chris Renaud , and written by Cinco Paul and Ken Daurio .",
    "source_dataset": "tals_vitaminc_validation_sample1000",
    "max_probability": 0.609607,
    "prediction_entropy": 0.939524,
    "composer_reason": "safe_structured_v2_no_qualifying_transition"
  }
]
```

### Diagnostic conclusion

Diagnostic-only audit: excessive token truncation is not indicated by the token audit; external prediction path matches the controlled dev encode/forward/export path; label mapping matches the controlled dev mapping; missing claim/evidence fields are not indicated; model confidence is available for collapse inspection; observed predictions are dominated by NOT_ENTITLED, consistent with learned controlled-model behavior under this external distribution. safe_structured_v2 composer diagnostics are available for all exported rows. Decisions remain Stage43-C0 decisions and diagnostics do not convert INCOMPLETE into PASS.

### Non-leakage statement

Stage43-B1 external fact-verification data is evaluation-only. It is not used for training, calibration, threshold selection, checkpoint selection, loss design, model selection, or composer behavior changes.

## Stage43-C2 Shadow Export Audit

```json
{
  "stage43c2_shadow_export_enabled": true,
  "stage43c2_shadow_export_available": true,
  "stage43c2_shadow_export_missing_dependencies": [],
  "stage43c2_required_shadow_fields_present_counts": {
    "stage32_shadow_label": 1000,
    "stage32_shadow_reason": 1000,
    "stage33_structured_coverage_label": 1000,
    "stage33_structured_coverage_reason": 1000,
    "stage33_structured_coverage_route": 1000,
    "stage36_final_shadow_label": 1000,
    "stage36_support_blocker_fired": 1000,
    "stage36_support_blocker_reasons": 1000,
    "stage37_final_shadow_label": 1000,
    "stage37_safe_recovery_fired": 1000,
    "stage37_safe_recovery_reasons": 1000,
    "stage37_recovered_from_label": 1000,
    "stage39_source_shadow_label": 1000,
    "stage39_composed_final_label": 1000,
    "stage39_composer_action": 1000,
    "stage39_composer_reason": 1000
  },
  "stage43c2_composer_application_count": 2,
  "stage43c2_composer_blocked_count": 0,
  "stage43c2_composer_blocked_reasons": {},
  "stage43c2_reused_internal_export_path": "prediction_records_v6b -> build_stage32_owner_state -> compute_stage36_support_safety_blocker -> compute_stage37_safe_support_recovery -> compute_stage39_final_composer",
  "stage43c2_forced_eval_only_exports": [
    "stage32_owner_state_export",
    "stage32_owner_state_shadow_mode",
    "stage33_use_structured_coverage_owner",
    "stage33_structured_coverage_owner_export",
    "stage33_structured_coverage_owner_shadow_mode",
    "stage33_structured_coverage_conditional_fallback",
    "stage36_use_support_safety_blockers",
    "stage36_support_safety_export",
    "stage36_support_safety_shadow_mode",
    "stage37_use_safe_support_recovery",
    "stage37_safe_support_export",
    "stage37_safe_support_shadow_mode",
    "stage39_use_final_composer_opt_in",
    "stage39_final_composer_export",
    "stage39_final_composer_policy=safe_structured_v2",
    "stage39_final_composer_output_mode=export_only"
  ]
}
```

### Stage43-C2 conclusion

Stage43-C2 shadow export produced Stage37 source shadow labels for every prediction row using the internal prediction_records_v6b export path.

### Stage43-C2 next action

Inspect base-vs-composed metrics and safety counters; do not use Stage43-B1 labels for tuning, calibration, or selection.

## 11. Recommendation

Composition introduced unsafe SUPPORT or SUPPORT/REFUTE transition(s).

## 12. Leakage policy

Stage43-B1 external fact-verification data is evaluation-only. It is not used for training, calibration, threshold selection, checkpoint selection, loss design, model selection, or composer behavior changes.
