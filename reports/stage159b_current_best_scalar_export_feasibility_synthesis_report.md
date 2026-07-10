# Stage159-B Current-Best Scalar Export Feasibility Synthesis

## 1. Summary decision

Decision: `STAGE159B_STAGE63_EQUIVALENT_SCALAR_EXPORT_BLOCKED_BY_MISSING_LOADABLE_CHECKPOINT_AND_LOAD_SUPPORT`

Stage159-B concludes that a Stage63-equivalent or current-best external scalar export is still required for recovered-vs-regressed scalar analysis, but that export is currently blocked. Stage158-A confirmed scalar export capability, yet its diagnostic run is not analysis-valid because the 1epoch model collapsed. Stage63 remains distributionally more useful for transition analysis, but it lacks scalar fields.

## 2. Stage159-A finding

Stage159-A reported:

- Out dir: `reports/stage159a_current_best_scalar_export_feasibility_20260710_062943`
- Report JSON: `reports/stage159a_current_best_scalar_export_feasibility_20260710_062943/stage159a_current_best_scalar_export_feasibility_report.json`
- Decision: `STAGE159A_CHECKPOINT_EXISTS_BUT_RUNNER_LOAD_SUPPORT_MISSING`

The practical Stage159-A finding is that the scalar export path exists, but the current available artifacts and runner support do not provide a reliable way to produce Stage63-equivalent/current-best scalar outputs.

## 3. Corrected checkpoint interpretation

Stage159-A set `has_checkpoint_candidate = true`, but the listed candidates are mostly `.jsonl`, `.csv`, `.md`, `.py`, or `.pyc` artifacts. These are not verified model weight checkpoints.

Corrected interpretation:

- The listed checkpoint candidates are not verified model weight checkpoints.
- No usable loadable model checkpoint has been found.
- Runner load support is not apparent.
- Runner eval-only support is not apparent.

Therefore, the blocker is stronger than the original checkpoint-candidate flag implied: there is no verified loadable current-best checkpoint and no observed runner path for loading one into an external scalar export pass.

## 4. Stage158-A scalar export status

Stage158-A output path:

`reports/stage158a_external_scalar_export_20260710_061132/runs/attempt_1_maxlen_default/stage43_external_scalar_export/stage158a_vitaminc_scalar_stage84e2_vitaminc_validation_sample1000_exact_ood_schema_external_factver_predictions.jsonl`

Observed quality:

- Rows: 1000
- Accuracy: 0.174
- Prediction counts: `NOT_ENTITLED = 959`, `SUPPORT = 41`, `REFUTE = 0`
- Scalar fields: 10
- Required scalars present: yes

Stage158-A confirmed scalar export capability. The required scalar fields were present:

- `frame_prob`
- `predicate_coverage_prob`
- `sufficiency_prob`
- `entitlement_prob`
- `compositional_entitlement_prob`
- `learned_entitlement_prob`
- `learned_entitlement_logit`
- `polarity_margin`
- `positive_energy`
- `negative_energy`

However, Stage158-A is not analysis-valid because the 1epoch run collapsed toward `NOT_ENTITLED` and produced zero `REFUTE` predictions. It can validate the export mechanism, but it cannot support recovered-vs-regressed conclusions.

## 5. Stage63 external status

Stage63 external prediction path:

`results/stage63_bridge_enabled_vitaminc_external_eval_20260702_060044/external/stage63_bridge_enabled_vitaminc_stage43b1_vitaminc_validation_sample1000_external_factver_predictions.jsonl`

Observed quality:

- Rows: 1000
- Accuracy: 0.322
- Prediction counts: `REFUTE = 217`, `NOT_ENTITLED = 492`, `SUPPORT = 291`
- Scalar fields: 0

Stage63 is distributionally more useful and was central to the Stage156-A pairwise recovery/regression analysis. It supports transition-group analysis better than the collapsed Stage158-A diagnostic run, but it lacks scalar fields and therefore cannot directly support scalar explanation of recovered-vs-regressed routing.

## 6. Core blocker

The core blocker is the absence of a Stage63-equivalent or current-best external scalar export from a non-collapsed model.

Recovered-vs-regressed external routing cannot be explained without internal scalar fields from an analysis-valid model distribution. The current state is blocked by:

- No scalar fields in the Stage63 prediction JSONL.
- Stage158-A scalar run collapsed.
- No apparent runner load/eval-only flags.
- No verified loadable checkpoint artifact.

## 7. Recommended Stage160-A

Stage160-A should create a reliable current-best/Stage63-equivalent scalar export path.

Recommended direction:

Patch the runner or add a dedicated eval-only scalar export script with checkpoint save/load support.

Minimum requirements:

- Save model checkpoint from a trained current-best run.
- Load model checkpoint for external eval-only scalar export.
- Export the same 10 scalar fields already confirmed by Stage158-A.
- Preserve stable row ordering or IDs for Stage156-A transition joins.
- Do not train on external labels.
- Do not tune thresholds on external labels.
- Do not use external labels for checkpoint selection.
- Do not integrate shadow diagnostics.

Fallback if checkpoint recovery is impossible:

- Rerun a sufficiently trained current-best configuration with scalar export enabled.
- Treat the run as diagnostic and report clean-dev selection basis only.
- Use external data only for post-hoc evaluation/reporting.

## 8. Safety constraints

This synthesis is analysis-only and report-only.

Safety policy:

- External data is for evaluation/reporting only.
- No external label may be used for training, threshold tuning, or checkpoint selection.
- Checkpoint selection is not modified.
- Shadow diagnostics are not integrated.
- Final predictions are not modified by shadow diagnostics.
- No training code, model code, export behavior, or analyzer scripts are changed by Stage159-B.
