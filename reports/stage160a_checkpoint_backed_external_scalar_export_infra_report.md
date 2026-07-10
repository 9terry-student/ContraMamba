# Stage160-A Checkpoint-Backed External Scalar Export Infrastructure Report

Decision: `STAGE160A_CHECKPOINT_BACKED_EXTERNAL_SCALAR_EXPORT_INFRA_READY`

Stage160-A is an infrastructure-only step. It does not train, evaluate, smoke test, or claim model performance.

## Why this was needed

Stage159-A established that Stage63 external predictions had a useful distribution but no scalar fields, while Stage158-A scalar external predictions had scalar fields but were collapsed. It also found no obvious eval-only load path in the current runner and no verified loadable model-weight checkpoint artifact. That blocked recovered-vs-regressed scalar analysis.

## Added checkpoint save support

`scripts/train_controlled_v6b_minimal.py` now accepts optional Stage160 checkpoint flags:

- `--save-checkpoint-path`
- `--save-checkpoint-mode final|best_clean_dev`

If `--save-checkpoint-path` is omitted, default training behavior is unchanged. The `best_clean_dev` mode uses only internal clean/dev metrics already computed by the training run. External evaluation, external labels, threshold tuning, and shadow diagnostics do not select or modify the checkpoint.

The saved `torch.save` payload includes model weights, architecture/backbone/router metadata, prediction export schema, label mapping, safe training config metadata, and Stage160 safety flags.

## Added eval-only external scalar export

`scripts/export_external_scalars_from_checkpoint.py` loads a saved checkpoint, reconstructs the model from checkpoint metadata plus CLI overrides, runs external fact-verification JSONL prediction export, and writes:

- `<output-dir>/<run-prefix>_external_scalar_predictions.jsonl`
- `<output-dir>/<run-prefix>_external_scalar_export_report.json`
- `<output-dir>/<run-prefix>_external_scalar_export_report.md`

The prediction rows preserve stable ids and row order for valid rows, optionally copy gold labels only when `--include-gold-label` is used, and emit available internal scalar fields with Stage160 provenance/safety flags.

## Safety constraints

External data is not used for training. External labels are not used for training or threshold tuning. External metrics are not used for checkpoint selection. Shadow diagnostics are not integrated into final predictions, and Stage142/145/147/150 shadow diagnostic scripts were not modified.

## Remaining requirement

Future Stage161-A must run a sufficiently trained current-best configuration with `--save-checkpoint-path`, then use `scripts/export_external_scalars_from_checkpoint.py` to produce analysis-valid external scalar predictions.