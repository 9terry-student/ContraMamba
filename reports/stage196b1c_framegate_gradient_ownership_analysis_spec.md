# Stage196-B1-C paired-seed FrameGate gradient-ownership analysis

## Scope

Stage196-B1-C answers one question under the frozen-Mamba envelope: did blocking direct non-frame gradients through FrameGate outputs improve FrameGate-owned trainable parameters and reduce persistent SUPPORT-to-NOT_ENTITLED failure?

The only authorized mechanism is direct downstream gradient ownership through FrameGate outputs while Mamba remains frozen. The analysis does not test unfrozen shared-encoder interference, representation quality outside this envelope, external/OOD behavior, calibration, or production readiness.

## Invocation and execution policy

The analyzer is `scripts/analyze_stage196b1_framegate_gradient_ownership.py`. All six arguments are required:

```text
python scripts/analyze_stage196b1_framegate_gradient_ownership.py \
  --repo-root <repository> \
  --run-root <repository>/reports/stage196b1_framegate_gradient_ownership_runs \
  --stage196a-report-json <authoritative-stage196a-report.json> \
  --current-git-commit <40-character-analysis-commit> \
  --stage196b1-runtime-git-commit <40-character-training-runtime-commit> \
  --output-dir <empty-output-directory>
```

This document specifies argv-style execution. The analyzer uses argument-vector subprocess execution with `shell=False` only for read-only Git identity inspection. It does not train, load a model or checkpoint, mutate a checkpoint, evaluate external data, search thresholds, calibrate, average parameters, or write outside the requested output directory.

## Exact source closure

The run root is required to resolve exactly to:

```text
reports/stage196b1_framegate_gradient_ownership_runs
```

Its directory entries must be exactly, in the analyzer's canonical order:

1. `seed183_joint`
2. `seed183_frame_local_only`
3. `seed184_joint`
4. `seed184_frame_local_only`
5. `seed185_joint`
6. `seed185_frame_local_only`

No substituted seed, missing arm, additional directory, or line-order-only pairing is accepted. Runtime provenance must close to the caller-supplied `--stage196b1-runtime-git-commit`, Mamba, `state-spaces/mamba-130m-hf`, `v6b_minimal`, CUDA, 20 epochs, split seed 174, frozen encoder and A_log, non-trainable fully isolated shared encoder, and isolation source `frozen_runtime_configuration`. For the frozen experiment the execution value is `9835cbbf86d83aca0964821669e63f7f6deb1c59`; it is supplied rather than inferred or hard-coded by the analyzer. The intervention must not change encoder freeze state.

Every run must declare direct frame loss active at weight 1.0, unchanged forward values, and the exact mode-specific blocking contract:

| Mode | `framegate_nonframe_output_gradient_blocked` |
|---|---|
| `joint` | `false` |
| `frame_local_only` | `true` |

### Provenance-source ownership

Runtime provenance is validated against explicit source ownership; the training report and trajectory contract are not treated as identical schemas.

Commit provenance has three independent roles. `--current-git-commit` is the analysis runtime/source commit, must be lowercase full 40-hex, and must equal repository HEAD. `--stage196b1-runtime-git-commit` is the historical six-run training runtime commit, must be lowercase full 40-hex, and is validated against every run without comparison to analyzer HEAD. Equal or different values are both valid because role ownership, not inequality, is the invariant. The FrameGate implementation-origin commit remains `5a39538ef3ca8f36cc2cc5d3290eae60d6a5f5c8`; it is neither the training runtime commit nor the analysis runtime commit.

Each run resolves the training runtime commit only from `run_provenance.json` at `source_provenance.git_commit` and `stage191_trajectory_contract.json` at `trainer_source_commit`. Both fields are required, lowercase full 40-hex, equal to one another, and equal to the supplied training runtime commit. No recursive search over commit-looking keys is permitted. After all runs load, the analyzer requires the six resolved values to form exactly the singleton set containing the supplied commit.

The authoritative training report owns `training_seed`, `configured_split_seed`, `resolved_split_seed`, `split_seed_explicit`, `split_policy`, `backbone`, `model_name`, `architecture`, `device`, `epochs`, `freeze_encoder`, `freeze_a_log`, `shared_encoder_trainable`, `shared_encoder_gradient_fully_isolated`, `shared_encoder_isolation_source`, `framegate_gradient_ownership_intervention_changed_encoder_freeze_state`, `frame_downstream_gradient_mode`, `frame_direct_loss_active`, `frame_direct_loss_weight`, `frame_downstream_forward_value_changed`, and `framegate_nonframe_output_gradient_blocked`. Deliberately duplicated values within the training report must be uniform. Split provenance is authoritative in `split_seed_contract`; every split field is required there, and any duplicate below `configuration` is cross-checked.
The epoch count is read from the report's exact single-run closure at `runs.single.final_epoch`; the remaining non-split runtime fields are resolved within `configuration`.

The frozen trajectory contract owns `observability_mode`, `stage191_trajectory_observability_implementation_reused`, `authorized_training_seeds`, `training_seed_authorized`, `arm`, `frame_downstream_gradient_mode`, `framegate_nonframe_output_gradient_blocked`, `freeze_encoder`, `freeze_a_log`, `shared_encoder_trainable`, `shared_encoder_gradient_fully_isolated`, `shared_encoder_isolation_source`, `framegate_gradient_ownership_intervention_changed_encoder_freeze_state`, `state_capsule_saving_enabled`, `expected_state_capsules`, `compatible_positive_margin_enabled`, `sidecar_accessed`, `parameter_swa_enabled`, `training_semantics_changed_by_observability`, and `extra_forward_pass_performed_by_observability`. Its existing run identity, cardinality, source-commit, enabled-flag, split-seed, and external-data gates remain required.

Fields intentionally present in both sources are required to equal each other and the frozen expected value: `frame_downstream_gradient_mode`, `framegate_nonframe_output_gradient_blocked`, `freeze_encoder`, `freeze_a_log`, `shared_encoder_trainable`, `shared_encoder_gradient_fully_isolated`, `shared_encoder_isolation_source`, and `framegate_gradient_ownership_intervention_changed_encoder_freeze_state`. Missing `configured_split_seed`, `resolved_split_seed`, `split_seed_explicit`, `split_policy`, `backbone`, `model_name`, `architecture`, `device`, `epochs`, `frame_direct_loss_active`, `frame_direct_loss_weight`, or `frame_downstream_forward_value_changed` in the trajectory contract is valid because those fields remain fail-closed against the training report.

### Prohibited-feature provenance ownership

Executed command provenance is `run_provenance.json:raw_sys_argv`, an ordered array excluding the Python executable and trainer path. The analyzer recovers every occurrence of every option, supports both `--option value` and `--option=value`, rejects positional or ambiguous empty-value forms, and requires the entire ordered argv to equal the frozen Stage196-B1 manifest command. Every explicitly frozen argument is cross-checked against its same-named value in `run_provenance.json:parsed_args`; a missing snapshot or disagreement fails. The frozen comparison includes seed/mode-specific values and exact `clean_dev_scalars.jsonl`, `training_report.json`, and `clean_dev_predictions.json` output paths.

External/OOD denial rejects any executed occurrence of `--ood-data`, `--ood-flag-source`, `--output-ood-json`, `--output-ood-predictions-json`, `--external-eval-jsonl`, `--external-eval-name`, `--external-output-dir`, `--stage43-external-factver-jsonl`, `--stage43-external-output-dir`, `--stage43-external-run-prefix`, `--stage43-external-max-rows`, `--stage43-external-batch-size`, `--stage43-external-enable-shadow-export`, or `--enable-stage43-external-eval`. Repeatable arguments are preserved as multiple occurrences and any occurrence is active. The command denial is cross-checked against `data_provenance.auxiliary_activity.external_evaluation_active=false` and empty `data_provenance.auxiliary_datasets.external_eval_jsonl` and `.stage43_external_factver` arrays. The run directory must also contain no file whose output name is external/OOD/Stage43-specific.

Bridge denial rejects every bridge dataset occurrence. Each of `--stage57-bridge-train-mode`, `--stage66-bridge-train-mode`, `--stage75-bridge-train-mode`, and `--stage80a-bridge-train-mode` may only be absent or have every recorded occurrence equal to `none`; `append_train_only` is rejected. The corresponding four `*-bridge-train-jsonl` options and bridge-specific run-directory artifacts must be absent. The exact `data_provenance.auxiliary_activity.stage57_active`, `.stage66_active`, `.stage75_active`, and `.stage80a_active` fields must all be false; each same-stage `auxiliary_datasets` record must say `configured=false`, `path=null`, and `mode=none`.

The exact supported post-hoc calibration/threshold deny list is `--ood-ablation-modes`, `--ood-unflagged-ne-shift-sweep`, `--ood-selective-ne-shift-sweep`, `--ood-selective-ne-gates`, `--ood-selective-ne-thresholds`, `--dev-calibrated-ne-shift-candidates`, `--dev-calibrated-ne-gate`, `--dev-calibrated-ne-threshold`, `--dev-calibrated-ne-frame-penalty`, `--dev-calibrated-ne-calibration-source`, and `--dev-calibrated-ne-frame-penalty-candidates`. These are post-hoc OOD/dev-calibration controls; ordinary fixed architectural thresholds are not denied by name.

Stage195 SWA is denied jointly by absence of `--stage195-tail3-parameter-swa-causal-test` and `--stage195-tail3-parameter-swa-output-dir`, trajectory-contract `parameter_swa_enabled=false`, zero state capsules, and absence of `stage195*` artifacts. Compatible-positive margin requires exactly one `--compatible-positive-margin-weight 0.0`, exactly one `--compatible-positive-margin-logit 0.0`, no integrity-sidecar path/SHA options, and trajectory-contract `compatible_positive_margin_enabled=false` plus `sidecar_accessed=false`.

New experimental losses are denied by exact ordered comparison with the frozen Stage196-B1 command, while `run_provenance.json:resolved_runtime_config` must retain direct FrameGate BCE active at weight `1.0` with unchanged downstream forward values. Time-swap exclusion requires `--data data/controlled_v5_v3_without_time_swap.jsonl`, matching `data_provenance.expected_main_clean_dataset`, matching `data_provenance.main_data.path`, `configured=true`, `expected=true`, mode `main_clean_classification`, SHA-256 `f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640`, and `data_provenance.auxiliary_activity.time_swap_active=false`.

The frozen `run_provenance.json` schema does not emit the manifest-only `forbidden_feature_assertions` object, so the analyzer does not fabricate it or default missing keys. It does emit `training_selection_policy`, whose exact clean-dev selection, external-evaluation denial, time-swap denial, final-CE source, and shadow-diagnostic fields are required. The analyzer derives the manifest's nine exact false assertions from argv, data provenance, resolved runtime configuration, trajectory contract, selection policy, and run-directory closure. If a root `forbidden_feature_assertions` object is present, it must equal that exact frozen object.

### Complete pre-metric provenance audit

| Semantic requirement | Authoritative artifact | Exact field or argv rule | Cross-check source | Failure condition |
|---|---|---|---|---|
| Analysis source identity | repository | `--current-git-commit`, lowercase 40-hex and equal to `HEAD` | contract CSV | Invalid SHA or HEAD mismatch |
| Stage196-B1 training identity | run provenance + trajectory contract | `source_provenance.git_commit`; `trainer_source_commit` | supplied `--stage196b1-runtime-git-commit`, all six runs | Missing, contradictory, mismatched, or nonuniform commits |
| FrameGate implementation origin | frozen source closure | `5a39538ef3ca8f36cc2cc5d3290eae60d6a5f5c8` | separate analysis/training role row | Origin role/value changed |
| Exact run population | run-root filesystem | canonical six directories only | `RUNS` | Missing, extra, or non-directory entry |
| Stage196-A localization | Stage196-A report and sibling CSVs | decision/counts/source closure/recovered recurrence rows | known cardinalities and universal invariant | Any closure or recovery mismatch |
| Required run artifacts | run-directory filesystem | named report, provenance, prediction, scalar, checkpoint, contract, ledger, and 20 epoch exports | required-file list | Missing required file |
| Trajectory/capsule/SWA files | run-directory filesystem | 20 trajectory exports, zero capsules, zero `stage195*` | trajectory contract and command | Count or absence mismatch |
| Training runtime configuration | training report | `configuration`; epoch at `runs.single.final_epoch` | frozen expected values | Missing or nonuniform runtime value |
| Split provenance | training report | `split_seed_contract`; all report occurrences | configuration duplicates | Missing, non-174, implicit, or contradictory split |
| Trajectory observability | trajectory contract | explicit observability/authorization/isolation/capsule/SWA fields | training report shared fields | Missing or mismatched contract field |
| Shared gradient ownership | report + trajectory contract | eight `CROSS_SOURCE_FIELDS` | frozen expected values | Any disagreement |
| Executed command recovery | run provenance | `raw_sys_argv` ordered string array | frozen manifest argv | Missing, malformed, positional, or ambiguous argv |
| Frozen command and outputs | run provenance | exact ordered `raw_sys_argv` plus explicit `parsed_args`, including three output paths | run directory, seed, mode | Missing parsed snapshot or any token/order/value/path disagreement |
| External/OOD disabled | argv + run provenance | no `EXTERNAL_OOD_OPTIONS`; auxiliary activity false; external dataset arrays empty | run-directory external/OOD artifact absence | Active option, contradictory provenance, or output artifact |
| Bridge training disabled | argv + run provenance | no bridge path; mode absent/all `none`; four activity fields false and dataset records disabled | run-directory bridge artifact absence | Dataset, active/ambiguous mode, contradictory provenance, or artifact |
| Calibration/threshold tuning disabled | executed argv | no listed post-hoc calibration option | frozen exact argv | Any listed option occurrence |
| Stage195 SWA disabled | argv + trajectory contract + filesystem | no Stage195 flags; false/zero contract; no artifacts | existing trajectory/capsule/SWA closure | Any active source or artifact |
| Compatible-positive margin disabled | argv + trajectory contract | weight/logit `0.0`; no sidecar; false/not accessed | training-report margin object | Missing, duplicate, active, or contradictory value |
| New experimental losses absent | frozen manifest argv | exact command closure | `resolved_runtime_config` direct FrameGate BCE fields | Command drift or direct BCE changed |
| Time-swap excluded | argv + run provenance | exact data path/SHA/configured/expected/mode and auxiliary activity false | expected-main and main-data records | Path/SHA/state disagreement or time-swap data |
| Forbidden assertions | run provenance + derived sources | exact `training_selection_policy` and nine false manifest assertions | optional root assertion object if present | Selection policy, derived assertion, or recorded assertion differs |
| CUDA and selected epoch | training report | `cuda_seed`, device fields, `runs.single` epochs | frozen seed/best-epoch map | Device, seed, final, or selected epoch mismatch |
| Trajectory identity/cardinality | trajectory contract | commit, seed, split, arm, 20 epochs, 720 rows, exact enabled flags | run identity | Any identity/cardinality/flag mismatch |
| Ledger closure | trajectory ledger | 20 unique epochs and required metric/hash/capsule fields | recomputed epoch values before scientific aggregation | Missing epoch/field, capsule value, or mismatch |

Compatible-positive margin, Stage195 parameter SWA, state capsules, external evaluation, external training, bridge training, time swap, and external threshold metrics must be absent or false as appropriate. Each run requires exactly 720 selected predictions, 720 scalar rows, 20 epoch prediction exports of 720 rows each, and 20 trajectory-ledger rows. The selected best epochs are provenance invariants (20, 20, 18, 13, 20, 20 in canonical run order), but best-epoch movement does not enter the causal rule.

## Stage196-A source closure

`--stage196a-report-json` supplies the authoritative prior localization; no Stage196-A report path is hard-coded. The analyzer requires its runnable local-channel decision, empty blockers, 127 persistent rows, and native channel threshold 0.5. It verifies the reported 72 `MULTI_LOCAL_CHANNEL_FAILURE` plus 55 `FRAME_ONLY_FAILURE` rows and the fact that those buckets close to all 127 persistent SUPPORT negatives.

The analyzer resolves the report's sibling source-closure and recurrence companions. Every source-closure gate must pass. Recurrent sets are reconstructed from the authoritative recurrence rows, never replaced with embedded position lists. Required recovered set cardinalities are:

| Set | Positions |
|---|---:|
| baseline recurrent | 22 |
| intervention recurrent | 19 |
| common recurrent | 19 |
| universal all-six | 10 |

The known universal positions `24, 25, 72, 108, 276, 324, 359, 479, 503, 635` are used only as a verification invariant after report-based recovery. Failure to recover any set unambiguously produces `STAGE196B1C_ANALYSIS_INCOMPLETE`.

## Row alignment and resolved schema

The primary alignment key is the explicit original row identifier:

```text
clean_dev_predictions.json: predictions[].id
clean_dev_scalars.jsonl:    [].id
trajectory JSONL:           [].source_row_id
```

`trajectory[].dev_position` is a certified stable position and is used for Stage196-A position membership only after the identifier mapping is validated. Incidental JSON/JSONL order is never a join key.

Every selected export and every epoch must contain 720 unique identifiers. Every epoch must contain every certified position 0 through 719 exactly once. The ID-to-position map may not drift across epochs or runs. All six runs must have identical ID populations, claims, evidence, gold labels, pair IDs, and intervention metadata. Prediction/scalar duplicates are compared within normal serialization tolerance.

The statically resolved semantic fields are:

| Meaning | Selected export | Epoch export |
|---|---|---|
| gold final label | `gold_final_label` | `gold_final_label` |
| predicted final label | `pred_final_label` | `predicted_final_label` |
| frame probability | `frame_prob` | `sigmoid(frame_logit)` |
| predicate coverage probability | `predicate_coverage_prob` | unavailable |
| sufficiency probability | `sufficiency_prob` | unavailable |
| SUPPORT score | `final_probs[2]` | `final_logits[2]` |
| NOT_ENTITLED score | `final_probs[1]` | `final_logits[1]` |
| intervention type | `intervention_type` | joined by stable ID |

Selected `final_probs` and epoch `final_logits` use canonical order `REFUTE, NOT_ENTITLED, SUPPORT`. Canonical argmax labels are recomputed. The selected export must agree with its trainer-selected trajectory epoch; it is not silently replaced by epoch 20.

## Selected checkpoint and tail-three separation

Selected-checkpoint metrics use `clean_dev_predictions.json`, whose metadata and row values must match the trainer-selected epoch. They include accuracy, macro-F1, per-label recall, SUPPORT precision, false-NOT_ENTITLED, false entitlement, polarity errors, prediction counts, and mean frame probability by gold label, intervention type, and each Stage196-A recurrent set.

Tail-three behavior always uses exact epochs 18, 19, and 20. A persistent stable SUPPORT negative is:

```text
gold == SUPPORT
and prediction(epoch 18) == NOT_ENTITLED
and prediction(epoch 19) == NOT_ENTITLED
and prediction(epoch 20) == NOT_ENTITLED
```

Stable SUPPORT-correct rows predict SUPPORT at all three epochs. Remaining gold SUPPORT rows are unstable. Persistent REFUTE polarity errors are gold REFUTE predicted SUPPORT at all three epochs. Persistent false entitlement is definable as gold NOT_ENTITLED predicted an entitled label (`REFUTE` or `SUPPORT`) at every tail epoch.

## Baseline-defined stable-correct controls

For each seed, controls are defined only from the joint arm: gold SUPPORT and SUPPORT predictions at epochs 18, 19, and 20. The paired intervention is evaluated on those exact identifiers. Outcomes are mutually exclusive: preserved SUPPORT at all three epochs, changed to NOT_ENTITLED at all three, changed to REFUTE at all three, or unstable/mixed. Preservation rate is preserved divided by the baseline-defined population. Intervention outcomes never redefine the population.

## Frozen local-channel localization

Persistent SUPPORT negatives are localized using the selected-checkpoint scalar snapshot and Stage196-A's native threshold 0.5 for frame, predicate, sufficiency, and entitlement aggregation. Polarity uses the native exported SUPPORT-facing margin boundary (`polarity_margin >= 0`), not a searched threshold. Final composition passes only when the selected final prediction equals gold.

Bucket precedence preserves Stage196-A terminology while adding the required polarity/unresolved distinctions:

1. two or more failed local channels: `MULTI_LOCAL_CHANNEL_FAILURE`
2. exactly one failed local channel: `FRAME_ONLY_FAILURE`, `PREDICATE_ONLY_FAILURE`, or `SUFFICIENCY_ONLY_FAILURE`
3. local channels pass but entitlement aggregation fails: `AGGREGATION_FAILURE`
4. preceding channels pass but polarity fails: `POLARITY_ONLY_FAILURE`
5. preceding channels pass but final label fails: `FINAL_COMPOSITION_FAILURE`
6. otherwise: `UNRESOLVED`

FrameGate failure count is the number of persistent SUPPORT negatives with failed `frame_prob`, independent of the mutually exclusive bucket label.

## Recurrent-position effects and trajectories

The baseline recurrent, intervention recurrent, common recurrent, and universal all-six sets remain separate. For every set, seed, and position the analyzer emits selected joint/intervention frame probabilities and delta; tail-three mean joint/intervention frame probabilities and delta; both tail prediction patterns; rescue; and previously-correct harm.

The epoch trajectory contains exactly 120 rows (`3 × 2 × 20`). Each row reports accuracy, macro-F1, SUPPORT recall, three error counts, mean frame probability on gold SUPPORT, mean frame probability on the union of persistent joint-arm positions, mean frame probability on Stage196-A common recurrent positions, and a tail-membership precursor. The precursor is not meaningful before epoch 18; at epochs 18–20 it means gold SUPPORT and NOT_ENTITLED at every available member of the fixed tail prefix.

## Paired estimands

Every delta is `frame_local_only - joint` and remains visible for seeds 183, 184, and 185. The CSV also includes mean, median, and positive/zero/negative direction counts. Error-count rows state that a negative delta is improvement. Metrics are selected accuracy, macro-F1, SUPPORT recall, false-NOT_ENTITLED, false entitlement, polarity error, persistent stable SUPPORT negatives, FrameGate failures among those negatives, stable-correct SUPPORT preservation, and selected mean frame probability on Stage196-A common and universal sets.

## Precommitted decision logic

The analyzer can select exactly one of:

```text
STAGE196B1C_SUPPORTS_DIRECT_FRAMEGATE_GRADIENT_INTERFERENCE
STAGE196B1C_DOES_NOT_SUPPORT_DIRECT_FRAMEGATE_GRADIENT_INTERFERENCE
STAGE196B1C_MIXED_GRADIENT_OWNERSHIP_EFFECT
STAGE196B1C_ANALYSIS_INCOMPLETE
```

Support requires all six gates: persistent SUPPORT negatives lower in at least two seeds and never higher; FrameGate failures lower in at least two and never higher; common-recurrent frame probability higher in at least two and never lower; full stable-correct preservation in at least two seeds; false entitlement not increased in at least two; and polarity error not increased in at least two.

If support is not met, mixed is selected for material paired-direction conflict, persistent rescue without coherent recurrent frame-probability improvement, multi-seed FrameGate reduction with multi-seed false-entitlement or polarity degradation, selected/tail direction disagreement, or one-seed dominance of the three-part primary signature. Integrity-passing results with neither the complete support signature nor a mixed condition select does-not-support. Missing artifacts, provenance failure, alignment ambiguity, unresolved schema, Stage196-A recovery failure, or contract failure select incomplete and authorize no scientific inference.

## Output closure

The output directory must be absent or empty. The analyzer creates exactly:

1. `stage196b1c_analysis.json`
2. `stage196b1c_report.md`
3. `stage196b1c_run_summary.csv`
4. `stage196b1c_paired_seed_deltas.csv`
5. `stage196b1c_tail3_persistent_rows.csv`
6. `stage196b1c_recurrent_position_effects.csv`
7. `stage196b1c_epoch_trajectory.csv`
8. `stage196b1c_contract.csv`

An analysis failure still writes this exact closure: an incomplete JSON/Markdown report, empty data CSVs with headers, and the accumulated failing contract ledger. Runtime or analysis failure is never converted into scientific evidence.

The contract CSV exposes source ownership with separate per-run gates for `training_report_runtime_provenance`, `training_report_split_provenance`, `trajectory_observability_provenance`, and `cross_source_gradient_ownership_provenance`. Commit-role closure is reported separately by `analysis_runtime_commit_format`, `analysis_runtime_commit_equals_head`, `stage196b1_runtime_commit_format`, `stage196b1_runtime_commit_matches_all_runs`, `stage196b1_runtime_commit_uniform_across_six_runs`, `analysis_and_training_commit_roles_separated`, and `framegate_implementation_origin_commit_preserved`. Per-run prohibited-feature rows are `executed_argv_recovered`, `external_ood_evaluation_disabled_by_command`, `external_ood_output_artifacts_absent`, `bridge_training_disabled_by_command`, `calibration_threshold_tuning_disabled_by_command`, `stage195_swa_disabled_by_command_contract_and_artifacts`, `compatible_positive_margin_disabled`, `frozen_command_configuration_and_output_paths`, `new_experimental_losses_absent`, `time_swap_excluded_from_main_training`, and `forbidden_feature_assertions_cross_validated`. The role-separation row reports both commit values explicitly and passes whether they are equal or different. A gate reports only the source or sources it actually inspected; it never attributes training-only split or configuration validation to the trajectory contract.

Both completed and incomplete analysis JSON include `analysis_runtime_git_commit`, `stage196b1_runtime_git_commit`, and `analysis_runtime_and_training_runtime_commits_are_distinct_roles: true`. The implementation-origin commit is recorded separately as `framegate_implementation_origin_git_commit`.

The historical incomplete directories `reports/stage196b1c_framegate_gradient_ownership_analysis_20260719_185153`, `reports/stage196b1c_framegate_gradient_ownership_analysis_20260719_190903`, and `reports/stage196b1c_framegate_gradient_ownership_analysis_20260719_192056` remain immutable non-scientific history. A corrected rerun must use a new timestamped output directory.

## Interpretation and next stage

A support decision authorizes only: under frozen Mamba, direct non-frame gradients through FrameGate outputs interfered with FrameGate-owned trainable parameters. It recommends `STAGE196B2_PROMOTE_FRAME_LOCAL_GRADIENT_OWNERSHIP` for architectural/training-contract promotion analysis only.

A does-not-support decision authorizes only: under frozen Mamba, direct FrameGate-output gradient interference was not supported as the primary cause. It recommends `STAGE196B2_RETURN_TO_FRAME_REPRESENTATION_HYPOTHESIS`; this permits design of a separately testable representation intervention but does not implement one.

A mixed decision recommends `STAGE196B2_NO_PROMOTION_TARGETED_CAUSAL_FOLLOWUP` and records the smallest unresolved causal question. An incomplete decision recommends `STAGE196B1C_REPAIR_ANALYSIS_INPUTS`.

No decision automatically authorizes retraining, a new loss, a model or trainer change, threshold search, calibration, SWA, data mutation, external/OOD evaluation, or production advancement.
