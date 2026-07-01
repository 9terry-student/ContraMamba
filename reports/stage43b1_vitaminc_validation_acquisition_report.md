# Stage43-B1 Targeted Fact-Verification Acquisition Report

Preparation/acquisition only. This report is produced by `scripts/acquire_stage43_factver_external_validation.py` and does not include model training, model evaluation, external probe evaluation, Kaggle commands, smoke tests, or local model execution.

## 1. Overall decision

**Decision:** `STAGE43B1_FACTVER_ACQUISITION_READY`

## 2. Dataset source/provenance

- Preset: `vitaminc`
- HuggingFace dataset: `tals/vitaminc`
- Split: `validation`
- Source dataset: `tals_vitaminc_validation_sample1000`
- Output JSONL: `data/stage43b1_vitaminc_validation_sample1000.jsonl`
- Rejected JSONL: `reports/stage43b1_vitaminc_validation_rejected.jsonl`

## 3. Preset

`vitaminc`

## 4. Field mapping

```json
{
  "claim_field": "claim",
  "evidence_field": "evidence",
  "label_field": "label",
  "id_field": null,
  "available_fields": [
    "unique_id",
    "case_id",
    "wiki_revision_id",
    "label",
    "claim",
    "evidence",
    "page",
    "revision_type",
    "FEVER_id",
    "big_bench_canary"
  ]
}
```

## 5. Label mapping

```json
{
  "SUPPORT": [
    "SUPPORTS"
  ],
  "REFUTE": [
    "REFUTES"
  ],
  "NOT_ENTITLED": [
    "NOT ENOUGH INFO",
    "NOT_ENOUGH_INFO",
    "NEI"
  ]
}
```

## 6. Evidence handling

```json
{
  "preset": "vitaminc",
  "mode": "first",
  "join_separator": " ",
  "max_evidence_chars": null,
  "dict_key_preference": [
    "evidence",
    "text",
    "sentence",
    "content",
    "passage",
    "context"
  ],
  "empty_after_flattening": "reject"
}
```

## 7. Accepted/rejected summary

- Total rows seen: 1000
- Accepted rows: 1000
- Rejected rows: 0

## 8. Label distribution

```json
{
  "NOT_ENTITLED": 145,
  "REFUTE": 355,
  "SUPPORT": 500
}
```

## 9. Sample accepted rows

```json
[
  {
    "id": "tals_vitaminc_validation_sample1000_validation_9543",
    "claim": "Airbourne 's song Live It Up was selected as the official theme song for Extreme Rules in 2013 .",
    "evidence": "In the movie The Lost Boys : The Tribe , a sequel to the original Lost Boys , the song `` Too Much , Too Young , Too Fast '' is played in the car as Chris and Nicole drive to their new house .",
    "label": "NOT_ENTITLED",
    "source_dataset": "tals_vitaminc_validation_sample1000",
    "source_label": "NOT ENOUGH INFO",
    "stage43_split": "external_validation",
    "metadata": {
      "row_index": 9543,
      "hf_dataset": "tals/vitaminc",
      "hf_config": null,
      "source_split": "validation",
      "preset": "vitaminc",
      "source_label_raw": "NOT ENOUGH INFO",
      "source_label_name": "NOT ENOUGH INFO",
      "dropped_disputed": false,
      "evidence_was_structured": false,
      "evidence_flatten_strategy": "string"
    }
  },
  {
    "id": "tals_vitaminc_validation_sample1000_validation_62847",
    "claim": "The Cincinnati Kid is a boy .",
    "evidence": "He considers it the film that allowed him to transition from the lighter comedic films he had previously been making and take on more serious films and subjects .",
    "label": "REFUTE",
    "source_dataset": "tals_vitaminc_validation_sample1000",
    "source_label": "REFUTES",
    "stage43_split": "external_validation",
    "metadata": {
      "row_index": 62847,
      "hf_dataset": "tals/vitaminc",
      "hf_config": null,
      "source_split": "validation",
      "preset": "vitaminc",
      "source_label_raw": "REFUTES",
      "source_label_name": "REFUTES",
      "dropped_disputed": false,
      "evidence_was_structured": false,
      "evidence_flatten_strategy": "string"
    }
  },
  {
    "id": "tals_vitaminc_validation_sample1000_validation_61972",
    "claim": "Efraim Diveroli had a four-year sentence .",
    "evidence": "Diveroli was sentenced to four years in federal prison .",
    "label": "SUPPORT",
    "source_dataset": "tals_vitaminc_validation_sample1000",
    "source_label": "SUPPORTS",
    "stage43_split": "external_validation",
    "metadata": {
      "row_index": 61972,
      "hf_dataset": "tals/vitaminc",
      "hf_config": null,
      "source_split": "validation",
      "preset": "vitaminc",
      "source_label_raw": "SUPPORTS",
      "source_label_name": "SUPPORTS",
      "dropped_disputed": false,
      "evidence_was_structured": false,
      "evidence_flatten_strategy": "string"
    }
  },
  {
    "id": "tals_vitaminc_validation_sample1000_validation_47744",
    "claim": "Emma Watson was born before 1995 .",
    "evidence": "Emma Charlotte Duerre Watson -LRB- born 15 April 1990 -RRB- is a French-British actress , model , and activist .",
    "label": "SUPPORT",
    "source_dataset": "tals_vitaminc_validation_sample1000",
    "source_label": "SUPPORTS",
    "stage43_split": "external_validation",
    "metadata": {
      "row_index": 47744,
      "hf_dataset": "tals/vitaminc",
      "hf_config": null,
      "source_split": "validation",
      "preset": "vitaminc",
      "source_label_raw": "SUPPORTS",
      "source_label_name": "SUPPORTS",
      "dropped_disputed": false,
      "evidence_was_structured": false,
      "evidence_flatten_strategy": "string"
    }
  },
  {
    "id": "tals_vitaminc_validation_sample1000_validation_59780",
    "claim": "Tilda Swinton is a singer .",
    "evidence": "After participating in Celine Dion 's tour , Swinton released her album `` True '' .",
    "label": "SUPPORT",
    "source_dataset": "tals_vitaminc_validation_sample1000",
    "source_label": "SUPPORTS",
    "stage43_split": "external_validation",
    "metadata": {
      "row_index": 59780,
      "hf_dataset": "tals/vitaminc",
      "hf_config": null,
      "source_split": "validation",
      "preset": "vitaminc",
      "source_label_raw": "SUPPORTS",
      "source_label_name": "SUPPORTS",
      "dropped_disputed": false,
      "evidence_was_structured": false,
      "evidence_flatten_strategy": "string"
    }
  }
]
```

## 10. Sample rejected rows

None.

## 11. Risks

- HuggingFace source schema or label metadata may change over time.
- Acquired rows must remain external-evaluation-only under the leakage policy.

## 12. Recommendation

Accepted rows cover SUPPORT, REFUTE, and NOT_ENTITLED. The file is ready for future Stage43-B2 eval-only use.

## 13. Leakage policy

Stage43-B1 acquired data is external-evaluation-only. It must not be used for training, calibration, threshold selection, checkpoint selection, loss design, or any other model-selection feedback loop.
