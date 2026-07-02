# Stage57: Non-leaking External Bridge Dataset Generator Audit

**Decision:** `STAGE57_NONLEAKING_EXTERNAL_BRIDGE_DATA_READY`

- Output JSONL: `data/stage57_nonleaking_external_bridge.jsonl`
- Total rows: 520
- Seed: 57
- Examples per family/label: 40
- Stage56 design file found: True

## Counts by label

| Label | Count |
|---|---|
| NOT_ENTITLED | 200 |
| REFUTE | 160 |
| SUPPORT | 160 |

## Counts by bridge family

| Family | Count |
|---|---|
| distractor_evidence_bridge | 40 |
| entity_attribute_bridge | 120 |
| lexical_paraphrase_bridge | 120 |
| numeric_comparison_bridge | 120 |
| temporal_comparison_bridge | 120 |

## Counts by bridge family x label

| Family | REFUTE | NOT_ENTITLED | SUPPORT |
|---|---|---|---|
| entity_attribute_bridge | 40 | 40 | 40 |
| numeric_comparison_bridge | 40 | 40 | 40 |
| temporal_comparison_bridge | 40 | 40 | 40 |
| lexical_paraphrase_bridge | 40 | 40 | 40 |
| distractor_evidence_bridge | 0 | 40 | 0 |

## Leakage policy

- `vitaminc_text_used_for_generation`: False
- `vitaminc_labels_used_for_generation`: False
- `external_metrics_used_for_threshold_tuning`: False
- `synthetic_only`: True

## Notes

- This dataset is synthetic training/diagnostic data only. It is NOT an external evaluation result and must not be reported as VitaminC or any other external-benchmark metric.
- This dataset must not be mixed with corrupted time_swap rows from data/controlled_v5_v3.jsonl; temporal bridge rows here are freshly generated and independent of that corruption.
- No VitaminC/Climate-FEVER text, labels, ids, or examples were read or used to produce any row in this file.

## Recommended next stage

- **Stage58**: bridge dataset static audit / schema check — Validate schema, label balance, and non-leakage of this bridge dataset before any training uses it.
