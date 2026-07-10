# Stage161-B Checkpoint-Backed Collapse Synthesis

## 1. Summary decision

Decision: `STAGE161B_CHECKPOINT_BACKED_EXPORT_WORKS_BUT_CURRENT_CONFIG_COLLAPSES`

Stage161-A confirms that checkpoint-backed external scalar export infrastructure works, but the current Stage161 configuration does not produce a usable analysis model. The scalar-bearing predictions are collapsed and should not be used for recovered-vs-regressed scalar conclusions.

## 2. Runtime interruption note

The Kaggle runtime interruption does not erase the copied Stage161-A conclusion. The Stage161-A summary recovered from chat is sufficient for this static synthesis report.

Raw artifacts and the checkpoint may be lost and should not be assumed available. This report therefore treats the copied Stage161-A summary as the source of record and does not rely on reading `/kaggle/working` outputs.

## 3. Stage161-A recovered result

Stage161-A decision: `STAGE161A_CHECKPOINT_BACKED_EXTERNAL_SCALAR_EXPORT_COLLAPSED_NEEDS_LONGER_OR_CONFIG_FIX`

Recovered configuration:

| Field | Value |
| --- | --- |
| Epochs | 30 |
| Seed | 1 |
| Architecture | `vnext_minimal` |
| Router mode | `learned_x_product` |
| Backbone | `mamba` |
| Data | `data/controlled_v5_v3_without_time_swap.jsonl` |
| Save checkpoint mode | `best_clean_dev` |

Recovered quality:

| Metric | Value |
| --- | ---: |
| Rows | 1000 |
| Malformed rows | 0 |
| Accuracy | 0.14 |
| Macro F1 | 0.08626074131076054 |
| Required scalar pass | true |
| Noncollapsed pass | false |
| NOT_ENTITLED rate | 0.985 |
| SUPPORT rate | 0.002 |
| REFUTE rate | 0.013 |

Prediction counts:

| Label | Count |
| --- | ---: |
| NOT_ENTITLED | 985 |
| REFUTE | 13 |
| SUPPORT | 2 |

Gold counts:

| Label | Count |
| --- | ---: |
| NOT_ENTITLED | 145 |
| REFUTE | 355 |
| SUPPORT | 500 |

Confusion summary:

| Gold -> Predicted | Count |
| --- | ---: |
| SUPPORT -> NOT_ENTITLED | 496 |
| REFUTE -> NOT_ENTITLED | 352 |
| NOT_ENTITLED -> NOT_ENTITLED | 137 |
| NOT_ENTITLED -> REFUTE | 6 |
| SUPPORT -> REFUTE | 4 |
| REFUTE -> REFUTE | 3 |
| NOT_ENTITLED -> SUPPORT | 2 |

Stage161-A confirms infrastructure but not a usable analysis model.

## 4. Scalar coverage

Stage161-A scalar coverage passed: all 10 expected scalar fields were present for all 1000 rows.

| Scalar field | Count |
| --- | ---: |
| `compositional_entitlement_prob` | 1000 |
| `entitlement_prob` | 1000 |
| `frame_prob` | 1000 |
| `learned_entitlement_logit` | 1000 |
| `learned_entitlement_prob` | 1000 |
| `negative_energy` | 1000 |
| `polarity_margin` | 1000 |
| `positive_energy` | 1000 |
| `predicate_coverage_prob` | 1000 |
| `sufficiency_prob` | 1000 |

The issue is no longer scalar export availability. The issue is recovering a non-collapsed Stage63-equivalent/current-best configuration.

## 5. Collapse comparison: Stage158 vs Stage161 vs Stage63

| Stage | Status | Accuracy | NOT_ENTITLED | REFUTE | SUPPORT |
| --- | --- | ---: | ---: | ---: | ---: |
| Stage158-A 1epoch | Scalar path confirmed but collapsed | 0.174 | 959 | 0 | 41 |
| Stage161-A 30epoch | Checkpoint-backed export confirmed but more collapsed | 0.14 | 985 | 13 | 2 |
| Stage63 external | Non-collapsed distribution but no scalars | 0.322 | 492 | 217 | 291 |

Stage161-A collapsed to NOT_ENTITLED at 98.5%, worse than the Stage158-A 1epoch diagnostic collapse. Stage63 had a substantially more distributed external prediction profile, even though it lacked scalar fields.

## 6. Interpretation

Checkpoint-backed external scalar export works. Stage161-A saved a checkpoint, ran checkpoint-backed external export, and produced complete scalar coverage.

The model/configuration did not recover Stage63-equivalent behavior. The current Stage161-A setup is not analysis-valid because its scalar-bearing predictions are collapsed. As a result, recovered-vs-regressed scalar analysis should not proceed from Stage161-A outputs.

More epochs are not the first recommended response because 30 epochs made the collapse worse than the 1epoch diagnostic. Do not rerun longer blindly.

## 7. Recommended Stage162-A

Stage162-A should recover Stage63-equivalent/current-best configuration before scalar analysis.

Recommended actions:

- Compare the Stage63 run command, config, and report against the Stage161-A training config.
- Identify missing bridge, composer, or training flags.
- Identify whether Stage63 used bridge-enabled data, recovery composer, or different architecture settings.
- Rerun only after the configuration difference is known.
- Do not blindly increase epochs.

Stage162-A should not proceed to recovered-vs-regressed scalar analysis until a non-collapsed scalar-bearing prediction set exists.

## 8. Safety constraints

This is a report-only synthesis.

Safety constraints:

- Do not train on external labels.
- Do not tune thresholds on external labels.
- Do not use external metrics for checkpoint selection.
- Do not integrate shadow diagnostics.
- Do not use collapsed scalar predictions for recovered-vs-regressed conclusions.
- Do not modify training code.
- Do not modify model code.
- Do not modify export script.
- Do not run Kaggle.
