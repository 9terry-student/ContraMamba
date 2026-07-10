# Stage158-B External Scalar Export Synthesis

## 1. Summary decision

Decision: `STAGE158B_EXTERNAL_SCALAR_EXPORT_PATH_CONFIRMED_CURRENT_RUN_COLLAPSED`

Stage158-A succeeded in confirming scalar export availability for the current external export path. The Stage157-A blocker is therefore resolved at the export-path level: internal scalar fields can be emitted into external prediction JSONL output.

However, Stage158-A should be treated only as a diagnostic/export validation run. It is not a meaningful current-best external model run because the short 1-epoch run collapses mostly to `NOT_ENTITLED` and predicts no `REFUTE`.

## 2. Stage157-A blocker

Stage157-A decision: `STAGE157A_EXTERNAL_SCALAR_EXPORT_REQUIRED`

Stage157-A found that the existing Stage53 and Stage63 external prediction JSONLs did not contain internal scalar fields:

| Check | Value |
| --- | --- |
| `stage53_has_scalars` | `false` |
| `stage63_has_scalars` | `false` |
| `has_both_scalars` | `false` |
| `stage53_n_scalar_fields` | `0` |
| `stage63_n_scalar_fields` | `0` |

This meant recovered-vs-regressed scalar analysis could not proceed from the existing external prediction artifacts.

## 3. Stage158-A scalar export result

Stage158-A decision: `STAGE158A_EXTERNAL_SCALAR_EXPORT_READY`

Stage158-A ran a current `vnext_minimal` external scalar export attempt and produced scalar-bearing external predictions.

| Field | Value |
| --- | --- |
| Output directory | `reports/stage158a_external_scalar_export_20260710_061132` |
| Report JSON | `reports/stage158a_external_scalar_export_20260710_061132/stage158a_external_scalar_export_report.json` |
| Best external JSONL | `reports/stage158a_external_scalar_export_20260710_061132/runs/attempt_1_maxlen_default/stage43_external_scalar_export/stage158a_vitaminc_scalar_stage84e2_vitaminc_validation_sample1000_exact_ood_schema_external_factver_predictions.jsonl` |
| Rows | `1000` |
| Malformed rows | `0` |
| Scalar fields | `10` |
| Expected scalar fields | `10` |

Prediction distribution:

| Label | Count |
| --- | ---: |
| `NOT_ENTITLED` | 959 |
| `SUPPORT` | 41 |
| `REFUTE` | 0 |

Gold distribution:

| Label | Count |
| --- | ---: |
| `NOT_ENTITLED` | 145 |
| `REFUTE` | 355 |
| `SUPPORT` | 500 |

## 4. Exported scalar fields

Each expected scalar field was present for all 1000 rows:

| Scalar field | Count |
| --- | ---: |
| `frame_prob` | 1000 |
| `predicate_coverage_prob` | 1000 |
| `sufficiency_prob` | 1000 |
| `entitlement_prob` | 1000 |
| `compositional_entitlement_prob` | 1000 |
| `learned_entitlement_prob` | 1000 |
| `learned_entitlement_logit` | 1000 |
| `polarity_margin` | 1000 |
| `positive_energy` | 1000 |
| `negative_energy` | 1000 |

## 5. Why the current run is diagnostic-only

Stage158-A confirms that the external scalar export path works, but it should not be used for final external performance claims, checkpoint selection, threshold tuning, architecture comparison, or recovered-vs-regressed scalar conclusions.

The run is diagnostic-only because it produced a severe prediction collapse:

* `NOT_ENTITLED`: 959 / 1000
* `SUPPORT`: 41 / 1000
* `REFUTE`: 0 / 1000

This distribution is not compatible with treating the run as a meaningful current-best external model result. Stage159-A should not draw recovered-vs-regressed scalar conclusions from this 1-epoch run.

Stage158-A should instead be used for:

* confirming scalar export fields;
* validating the Stage43 external scalar export path;
* informing how to run future current-best scalar exports.

## 6. Optional Cell 5 failure interpretation

The final Cell 5 failure was non-blocking and only affected optional Stage159-prep.

Failure status: `non_blocking`

Error: `No Stage156-A output directory found`

Meaning: the optional Stage159-prep join failed because the runtime could not locate prior Stage156-A artifacts. This does not invalidate Stage158-A, because the scalar export itself completed successfully and produced the expected scalar-bearing JSONL.

## 7. Recommended Stage159-A

Recommended Stage159-A goal: prepare a robust recovered-vs-regressed scalar analysis design using Stage158-A export-path knowledge.

The next task is to obtain a Stage63-equivalent/current-best scalar export or design a robust export/join procedure. Stage159-A should not analyze the collapsed 1-epoch run as current-best.

Recommended directions:

* locate a real current-best checkpoint/run configuration and rerun external scalar export;
* patch or configure Stage63-equivalent external export to include scalar fields;
* create robust join code that accepts explicit Stage156-A paired CSV path and Stage158/current-best scalar JSONL path.

Stage159-A must not:

* train on external labels;
* tune thresholds on external labels;
* use external labels for checkpoint selection;
* integrate shadow diagnostics into final predictions.

## 8. Safety constraints

The Stage158-B synthesis is report-only and preserves the evaluation boundary:

| Constraint | Value |
| --- | --- |
| `analysis_export_only` | `true` |
| `external_eval_only` | `true` |
| `external_data_used_for_training` | `false` |
| `external_labels_used_for_training` | `false` |
| `threshold_used_for_model_selection` | `false` |
| `checkpoint_selection_modified` | `false` |
| `shadow_diagnostics_integrated` | `false` |
| `final_logits_modified_by_shadow` | `false` |
| `final_predictions_modified_by_shadow` | `false` |

No training code, model code, export behavior, or analyzer scripts are changed by this report.
