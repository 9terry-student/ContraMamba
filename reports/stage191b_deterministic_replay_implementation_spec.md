# Stage191-B deterministic diagnostic replay implementation specification

## Purpose and scope

Stage191-B adds default-off, report-only observability to the frozen Stage189 trainer and builds exactly six deterministic replay commands. Replay is diagnostic only: it may distinguish checkpoint-selection effects, late optimization drift, class redistribution without direct SUPPORT conflict, or failure to reproduce the Stage189 trajectory. It does not authorize model advancement, hyperparameter selection, sweeps, extra seeds, external evaluation, OOD evaluation, or bridge use.

The six runs are baseline and intervention arms for seeds 174, 175, and 176, all with split seed 174, 20 epochs, the clean controlled dataset, Mamba CUDA runtime, and the original Stage189 arguments. Baseline retains zero Stage185 sidecar training access. Intervention requires weight `0.05`, margin logit `0.0`, the exact authoritative Stage185-A JSONL, and semantic SHA256 `5bc03caa2a29f9b9176ab4eb0201db57ebad516352797546db1a18e6ec3373fc`.

## Observability contract

The two flags are `--stage191-trajectory-replay-observability` and `--stage191-save-trajectory-state-capsules`. Each has its own exact `store_true`, `default=False` parser block. Capsules require observability. Neither flag changes model construction, initialization, random-number consumption, data order, optimizer/scheduler behavior, losses, gradients, backward calls, optimizer steps, selection, stopping, existing reports, or existing checkpoints. No additional model forward is performed.

Before the nested trainer is defined, Stage191 builds one runtime context from already-assigned main-scope values: the exact output directory, args, seeds, authoritative Stage187 audit, trainer path/SHA, source identity from the existing provenance helper, and model-construction configuration. The Stage191 contract and export call blocks do not reference later-created _stage174a_provenance_record, _stage174a_parsed_args, or _stage174a_resolved_runtime_config values.

The runtime mapping must contain the canonical labels exactly in the order `REFUTE`, `NOT_ENTITLED`, `SUPPORT`; its values must be exact non-bool integers with set `{0, 1, 2}`; and the integer-keyed reverse mapping must be its exact inverse. This verified mapping controls logit-column order, predictions, labels, the 3-by-3 confusion matrix, and every error metric without string, bool, or integer canonicalization.

Each epoch reuses `dev_output`. Authoritative logits are `output["logits"]`; `loss_logits` is never used. Before metrics, logits must have topology `(720, 3)`, labels `(720,)`, every gold ID must be canonical, records must total 720, and `frame_logit` must be absent or contain exactly 720 values. Gold counts, dense prediction counts, and the confusion matrix must each sum to 720; gold SUPPORT must equal 89. Epoch must be an exact non-bool integer in epochs 1 through 20, the selection score must be finite numeric and non-bool, logits and per-row CE must be finite, frame logits must be finite when present, and exactly 720 row records must be constructed before writing.

Clean CE is mean final-label cross entropy over all 720 rows. SUPPORT recall is gold SUPPORT predicted SUPPORT divided by 89. False entitlement is gold NOT_ENTITLED predicted REFUTE or SUPPORT. False not-entitled is gold REFUTE or SUPPORT predicted NOT_ENTITLED. Polarity error is REFUTE predicted as SUPPORT plus SUPPORT predicted as REFUTE.

Every trajectory row includes dense prediction counts, gold counts, the exact gold-by-prediction 3-by-3 confusion matrix, accuracy, macro-F1, the selection metric/value, and best epoch before/replacement/best epoch after. Every epoch exports exactly 720 fixed-position rows. Source identity precedence is `row_id`, `id`, `stable_id`, then `raw_record.id` only when `raw_record` is already a dictionary; only non-bool strings or integers are accepted. Otherwise the source ID is null, while `dev_position` remains the fixed positional identity. Text is excluded and each export SHA256 is recorded.

With both flags, a capsule is saved after the clean-dev-producing state and before the next epoch. It contains only the epoch, trainable named parameter tensors, named buffers, names/dtype/shape metadata, scalar-safe raw-byte state hashes, seeds, arm, and exact model-construction provenance. It excludes optimizer/scheduler state, gradients, frozen parameter tensors, data rows, and external data.

The trajectory contract records actual validated runtime sidecar configuration from the existing Stage187 activation/access audit. It does not infer access from the arm name. Baseline requires disabled and unaccessed state; intervention requires the exact activated and accessed authoritative path and matching expected and observed semantic hashes.

## Manifest and identity gates

The builder reads exactly `stage191a_trajectory_feasibility_report.json`. Its Stage190-C gate requires the exact authoritative runtime fields, including diagnostic-only status, no blockers, no advancement, no significance or causality claim, the three fixed conflict booleans and shared-gradient fractions, no qualified shared-conflict seeds, and the authorized checkpoint-selection/optimization-trajectory design class.

Each Stage190-A run must be runnable with no blockers and exact seed, arm, split, commits, fixed split identity, checkpoint path/SHA, passed external-use and arm-runtime dictionaries, and current hashes for `training_report.json`, `run_provenance.json`, and `selected_checkpoint.pt`. Each run also freezes the exact regular file `<historical_run_dir>/clean_dev_predictions.json` by path and SHA256; this file is the sole row-by-row selected-checkpoint prediction reference.

For historical argv, the builder requires exactly one separate-token occurrence of `--output-json`, `--output-predictions-json`, and `--stage115-clean-dev-scalar-output-jsonl`, with values resolving to the three exact historical run files. Only those values are changed to their replay-run counterparts. Every other token remains byte-for-byte equal and in order, after which exactly the two Stage191 flags are appended. The emitted argv-difference audit is blocking.

`stage191b_equivalence_gate.csv` retains the replay-equivalence contract. `stage191b_precommitted_gate.csv` contains the actual builder identity/runtime gates with required, observed, passed, and blocking-reason fields.

## Precommitted replay equivalence

For every replay, must cover epochs 1 through 20 exactly; historical macro-F1 and normalized three-label prediction counts must match exactly at every epoch; compatible-positive-margin enabled state, eligible count, and active count must match exactly; selected and final epochs must match; and original selected-checkpoint clean predictions must match row by row. No external/OOD/bridge use is allowed, including Stage57, Stage66, Stage75, and Stage80A bridge paths, and baseline sidecar access remains false. Margin floating values use absolute and relative tolerance `1e-12` only where exact writer equality is not guaranteed. Any mismatch blocks interpretation and is never silently accepted.

READY authorizes replay execution solely for the six exact diagnostic commands. Training for model advancement and model advancement remain false.
