# Stage195-A tail-three trainable-parameter SWA manifest specification

## Scope and frozen hypothesis

Stage194-A closed with `STAGE194A_MIXED_TEMPORAL_AND_BOUNDARY_MECHANISMS`: temporal consensus and persistent entitlement-boundary bias coexist. Stage195 tests only the temporal-consensus mechanism. Its intervention is the post-training mean of the unique trainable named parameters captured at epochs 18, 19, and 20, exactly as implemented by Stage195-P0.

Stage195-A is manifest-generation infrastructure only. It neither executes Stage195-B nor implements or decides Stage195-C. It adds no entitlement calibration; NOT_ENTITLED logit, class, or margin shift; median; trimmed mean; robust-logit aggregation; loss; head; auxiliary data; checkpoint-selection change; or optimizer/scheduler averaging. It does not reinterpret Stage193 or Stage194 results, approve model advancement, select a production SWA model, or authorize later training.

## Frozen provenance identities

- `TRAINER_BLOB_COMMIT` is `bd27e46daf218a57da9a3142c9e4bc5cc44ad53a`. It identifies only the frozen Stage195-P0 bytes of `scripts/train_controlled_v6b_minimal.py`.
- `TRAINER_BLOB_SHA256` is `4fe903c9f3aa21ee6365a0297c27e4a333d295dbb851384efc7bc8d3f7607954`.
- The frozen Stage195-P0 specification is `reports/stage195p0_tail3_parameter_swa_spec.md` at `TRAINER_BLOB_COMMIT`, with SHA256 `a65eab7877c3768e545fed070932b432bd0459374386522051d75af1d5254a60`.
- The authoritative Stage193-A runtime repository commit is `89a9805d0e9c877774f9ce4b356297d31645b74b`.
- The Stage193-A frozen trainer blob commit is `e83d8af756fa84b7a91c14e0910ae388b07b5f02`, with SHA256 `25d42bdcd204219a2b2e5e7bf2a8b14459eafb4945c05c61ab3611bc9e7365bc`.
- The authoritative Stage185 sidecar directory basename is `stage185a_controlled_train_integrity_sidecar_20260715_141914`; its semantic SHA256 is `5bc03caa2a29f9b9176ab4eb0201db57ebad516352797546db1a18e6ec3373fc`.

`stage195_runtime_repository_commit` has a different meaning: it is the future lowercase 40-character commit containing this Stage195-A specification and builder, and it is the exact repository HEAD at which the manifest is built and Stage195-B may run. No `trainer_source_commit` field exists in any Stage195-A output.

The builder verifies with read-only Git argument arrays and `shell=False` that the supplied runtime commit is lowercase 40-hex, equals HEAD, contains byte-identical blobs for both new Stage195-A files, and has no staged or unstaged differences for either file. Independently, it proves that current trainer bytes equal the trainer blob at `TRAINER_BLOB_COMMIT`, that both SHA256 values equal `TRAINER_BLOB_SHA256`, and that the current Stage195-P0 specification equals its blob at the same commit and its frozen SHA256. Global worktree cleanliness is not required.

## Explicit inputs and path safety

The builder requires exactly these explicit path/provenance arguments:

1. `--repo-root`
2. `--stage193a-dir`
3. `--stage185-sidecar-dir`
4. `--stage195b-run-root`
5. `--current-diagnostic-git-commit`
6. `--output-dir`

There is no fuzzy, latest, glob, or timestamp discovery. Every timestamped directory is supplied explicitly.

The Stage193-A input must be an existing immediate child of `<repo-root>/reports` whose basename begins `stage193a_tail3_fresh_seed_manifest_`. The Stage185 input must resolve to the exact authoritative directory named above. The Stage195-A output must be absent or an entirely empty directory, an immediate reports child, and have basename prefix `stage195a_tail3_parameter_swa_manifest_`. The planned Stage195-B root has the same absent-or-empty and immediate-child requirements with prefix `stage195b_tail3_parameter_swa_runs_`. Output, planned run root, Stage193-A input, and Stage185 input must all differ.

The builder creates only the Stage195-A output directory. It never creates the Stage195-B root or a run directory; all Stage195-B paths are plans.

## Frozen source closure

Stage193-B, Stage193-C, and Stage194-A run/result artifacts are not inputs. The complete source closure is:

1. the exact supplied Stage193-A READY manifest directory;
2. the exact authoritative Stage185 sidecar directory;
3. the frozen Stage195-P0 trainer and specification bytes.

The Stage193-A directory must contain exactly these six regular files and no other entry:

1. `stage193a_tail3_fresh_seed_manifest_report.json`
2. `stage193a_tail3_fresh_seed_manifest_report.md`
3. `stage193a_run_manifest.jsonl`
4. `stage193a_run_command_matrix.csv`
5. `stage193a_source_and_template_gate.csv`
6. `stage193a_precommitted_gate.csv`

The report must be `STAGE193A_TAIL3_FRESH_SEED_MANIFEST_READY`, runnable, diagnostic-only, have no blockers, record the exact ordered six Stage193 runs, exact Stage193 runtime and trainer identities, prohibit model advancement and subsequent training, and authorize only its exact six diagnostic runs. Its source/template and precommitted gate arrays must have the exact expected gate names, fixed gate schema, all `passed is True`, and empty blocking reasons. Its six JSONL rows must have the exact Stage193-A schema, exact order, exact argv arrays and command arrays, strict integer identities/cardinalities, frozen arm contracts, all source/template gates passing, all precommitted gates passing, no model advancement, and no subsequent-training authorization.

The Stage185 semantic SHA is recomputed from the authoritative JSONL. `created_at` is removed from every row; remaining row keys and top-level JSON serialization use sorted keys, compact separators, UTF-8, and `ensure_ascii=False`.

## Six-run matrix and argv derivation

The order is exactly:

1. `seed180_baseline`
2. `seed180_intervention`
3. `seed181_baseline`
4. `seed181_intervention`
5. `seed182_baseline`
6. `seed182_intervention`

Training seeds are exact non-bool integers 180, 181, and 182; split seed is the exact non-bool integer 174. Baseline precedes intervention for each seed.

The three Stage193-A argv arrays in each arm are the authoritative template sources. Within an arm, normalization may change only the source training seed and source output paths and may remove only the Stage193/Stage191 observability flags. All three normalized arrays must be byte-for-byte token-equal. A second cross-arm normalization may additionally abstract the compatible-positive weight and remove the two integrity-sidecar arguments; it must prove the two arms differ in no other training semantic.

Generated argv mutation is limited to:

1. replacing `--seed` with 180, 181, or 182;
2. replacing `--output-json`, `--output-predictions-json`, and `--stage115-clean-dev-scalar-output-jsonl` with paths inside the exact run directory;
3. removing `--stage193-tail3-fresh-seed-observability`;
4. removing `--stage191-trajectory-replay-observability` if present;
5. removing `--stage191-save-trajectory-state-capsules` if present;
6. adding `--stage195-tail3-parameter-swa-causal-test`;
7. adding `--stage195-tail3-parameter-swa-output-dir <exact run directory>`.

Every other token and token order remains identical to the arm template. The resolved parent of `--output-json` equals the resolved Stage195 SWA output directory. Generated argv preserve architecture `v6b_minimal`, backbone `mamba`, model `state-spaces/mamba-130m-hf`, CUDA, 20 epochs, `final_macro_f1`, flag source `controlled_heuristic`, split seed 174, and `data/controlled_v5_v3_without_time_swap.jsonl`. They preserve the source optimizer, scheduler, batch sizes, eval batch size, maximum length, learning rates, epoch count, temporal comparator, loss/reporting configuration, and selected-checkpoint semantics.

Smoke, max-train-record truncation, loss sweep, external/OOD data or evaluation, bridge rows/training, `time_swap` main data, temporal auxiliary paths/losses/adapters/channels/heads, coverage-entailment auxiliary path/head/loss, pair-contrastive auxiliary path/loss, constrained selectors, OOD/dev-calibrated NOT_ENTITLED shifts, and state capsules are forbidden. Final CE and logits remain sourced only from `output["logits"]`.

Baseline has margin weight 0.0, margin logit 0.0, and neither sidecar argument. Intervention has weight 0.05, margin logit 0.0, the exact authoritative Stage185 sidecar path, and the frozen semantic SHA256. These margin/sidecar fields are the only permitted arm-level training-semantic differences.

## Per-run planned artifacts and cardinalities

Every READY JSONL row freezes the resolved run directory and these paths:

- `training_report.json`;
- the normal trainer-selected checkpoint using the unchanged exact `selected_checkpoint_filename`;
- `stage191_trajectory_contract.json`;
- `stage191_trajectory_epoch_metrics.jsonl`;
- twenty paths `stage191_dev_predictions_epoch_NNN.jsonl` for NNN 001 through 020;
- `stage195_tail3_parameter_swa_predictions.jsonl`;
- `stage195_tail3_parameter_swa_metrics.json`;
- `stage195_tail3_parameter_swa_contract.json`.

Per run, expected trajectory ledger rows are 20, prediction exports are 20, rows per prediction export are 720, Stage195 SWA prediction rows are 720, state capsules are 0, and SWA checkpoints are 0. No SWA checkpoint or temporary-parameter checkpoint path is recorded.

## Exact output schemas

The builder writes exactly:

1. `stage195a_tail3_parameter_swa_manifest.json`
2. `stage195a_tail3_parameter_swa_manifest.md`
3. `stage195a_run_manifest.jsonl`
4. `stage195a_run_command_matrix.csv`
5. `stage195a_source_and_template_gate.csv`
6. `stage195a_precommitted_gate.csv`

The JSON report has exactly these keys:

`stage`, `decision`, `runnable`, `blocking_reasons`, `diagnostic_only`, `exact_six_run_diagnostic_execution_authorized`, `model_advancement_authorized`, `production_swa_selected`, `entitlement_correction_implemented`, `stage195c_decision_made`, `subsequent_training_authorized`, `statistical_significance_claimed`, `stage195b_training_performed`, `model_loaded`, `checkpoint_loaded`, `external_data_used`, `trainer_blob_commit`, `trainer_blob_sha256`, `stage195_runtime_repository_commit`, `source_identity`, `frozen_source_identities`, `stage195b_run_root`, `ordered_runs`, `run_manifest_count`, `expected_trajectory_rows_per_run`, `expected_prediction_exports_per_run`, `expected_prediction_rows_per_export`, `expected_stage195_swa_prediction_rows_per_run`, `expected_state_capsules_per_run`, `expected_swa_checkpoints_per_run`, `canonical_labels`, `logits_source`, `source_and_template_gates`, `precommitted_gates`, `exception`.

Each READY JSONL row has exactly these keys:

`stage`, `run`, `training_seed`, `split_seed`, `arm`, `canonical_labels`, `trainer_source_path`, `trainer_blob_commit`, `trainer_blob_sha256`, `stage195_runtime_repository_commit`, `argv`, `command_argv`, `command`, `planned_run_directory`, `planned_output_json_path`, `planned_selected_checkpoint_path`, `expected_trajectory_contract_path`, `expected_trajectory_ledger_path`, `expected_prediction_export_paths`, `expected_stage195_swa_predictions_path`, `expected_stage195_swa_metrics_path`, `expected_stage195_swa_contract_path`, `expected_trajectory_rows`, `expected_prediction_exports`, `expected_prediction_rows_per_export`, `expected_stage195_swa_prediction_rows`, `expected_state_capsules`, `expected_swa_checkpoints`, `logits_source`, `arm_contract`, `argv_mutation_audit`, `frozen_training_envelope`, `runnable`, `diagnostic_only`, `exact_six_run_diagnostic_execution_authorized`, `model_advancement_authorized`, `production_swa_selected`, `entitlement_correction_implemented`, `stage195c_decision_made`, `subsequent_training_authorized`, `statistical_significance_claimed`, `external_data_used`.

When READY, JSONL has exactly six rows in frozen order. When BLOCKED it has zero rows but retains valid JSONL syntax as the empty file. Every JSON/JSONL integer identity and cardinality is accepted only when `type(value) is int`; booleans never satisfy an integer contract.

`stage195a_run_command_matrix.csv` has this exact header, in order:

`run,training_seed,split_seed,arm,planned_run_directory,planned_output_json_path,planned_selected_checkpoint_path,expected_trajectory_contract_path,expected_trajectory_ledger_path,expected_prediction_export_paths,expected_stage195_swa_predictions_path,expected_stage195_swa_metrics_path,expected_stage195_swa_contract_path,trainer_source_path,trainer_blob_commit,trainer_blob_sha256,stage195_runtime_repository_commit,command,expected_trajectory_rows,expected_prediction_exports,expected_prediction_rows_per_export,expected_stage195_swa_prediction_rows,expected_state_capsules,expected_swa_checkpoints`

Both gate CSVs have this exact header:

`gate,required,observed,passed,blocking_reason`

The code verifies the exact JSON report key set, every JSONL row key set, CSV row key sets, all fixed headers, output-name closure, row cardinality, and frozen order before publication.

## Decisions, authorization, and failure behavior

READY is `STAGE195A_TAIL3_PARAMETER_SWA_MANIFEST_READY`. It authorizes only the exact six Stage195-B diagnostic runs recorded in the manifest. BLOCKED is `STAGE195A_TAIL3_PARAMETER_SWA_MANIFEST_BLOCKED`.

For READY and BLOCKED alike, model advancement authorization, production SWA selection, entitlement correction implementation, Stage195-C decision, subsequent-training authorization, statistical-significance claim, Stage195-B training performed, model loading, checkpoint loading, and external-data use are false. Stage195-A itself performs no training.

Before safe output establishment, any failure writes nothing. Once the output directory is safely established, every analysis or publication exception is converted to BLOCKED fixed-schema/header outputs, a nonzero exit code, an exception type/message/traceback, and a blocking reason. READY content is serialized and schema-checked before publication; the report is published last. Publication cleanup is limited to the six exact Stage195-A filenames and their private temporary names so a failed READY publication cannot leave a valid-looking READY report.
