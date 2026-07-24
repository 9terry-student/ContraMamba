# Stage196-B2-B6P9-P3-P0 Separate Observational Manifest Spec

## Authority

This stage creates an execution-ready manifest for exactly one observer-off
control run and six mutually separate teacher-observer runs. It is backed by
the Git-preserved P9-P2 authority at:

`reports/stage196b2b6p9p2_separate_observability_instrumentations_20260724_143152`

The builder requires:

- `decision = STAGE196B2B6P9P2_SEPARATE_IMPLEMENTATIONS_READY`
- `recommended_next_stage = STAGE196B2B6P9P3_SEPARATE_OBSERVATIONAL_RUNS`
- `blocking_reasons = []`
- `failure = null`
- `failed_contract_count = 0`

It uses the P9-P2 analysis, contract, CLI contract, lifecycle hook audit,
baseline invariance audit, and checkpoint schema audit. It also statically
inspects `scripts/train_controlled_v6b_minimal.py` and
`src/contramamba/teacher_state_observer.py`.

## Base Configuration

The builder uses a deterministic authority hierarchy and does not accept
filenames alone as configuration evidence:

1. exact tracked resolved command and parsed arguments for the seed-183 primary
   Stage196 lineage:
   `reports/stage196b2b3p0_epoch_composer_input_observability_runs_retry_20260722_104834/seed183_joint/trajectory/run_provenance.json`;
2. tracked P8/P7/P6 manifest, checkpoint, and design metadata for the same
   mechanistic lineage;
3. field-level consensus across tracked seed183/184/185 Stage196 primary-arm
   sources after canonical normalization, only when every required field is
   present and the primary arm is independently resolved;
4. otherwise fail closed with
   `STAGE196B2B6P9P3P0_BLOCKED_BASE_CONFIG_AUTHORITY`.

The primary arm is resolved mechanistically, not by performance. `joint` is
accepted only when tracked P7/P8 authority establishes it as the unrestricted
native trainable recipient path and establishes `frame_local_only` as the
separately trained donor/restricted contrast. The authority label is:

`PRIMARY_UNRESTRICTED_JOINT_LINEAGE`

The accepted base must source or explicitly normalize data path, backbone,
model name, device, seed, epochs, train/eval batch sizing, learning rates,
weight decay or its tracked absence, architecture mode, router mode, evidence
interface, training scope, checkpoint-selection behavior, mixed-precision
behavior, active Stage195/Stage196 lineage flags, external-evaluation state,
and bridge-training state. Required closures include `backbone = mamba`,
`model_name = state-spaces/mamba-130m-hf`, `device = cuda`, `seed = 183`, and
external evaluation disabled.

The P9-P3 frozen run seed is a manifest overlay and is not used by itself to
prove base-configuration authority. In the tracked Stage196 lineage, seed 183 is
present in the exact primary command and is corroborated by joint seed183/184/185
consensus showing seed as the only primary-arm seed-varying field after run paths
are removed.

The base authority JSON records the stage, decision, recommended next stage,
blocking reasons, failure, selected base source, selected primary arm, primary-arm
authority, base configuration fingerprint, resolved base configuration, field
provenance, considered and rejected candidate sources, normalizations, unresolved
fields, and conflicts in both ready and blocked states. Manifest contracts require
strict boolean `passed` values; any internal non-boolean contract value is a schema
failure that closes the builder with a controlled blocked decision.

## Frozen Run Set

All seven runs use `seed = 183`. This is a controlled within-seed runtime
comparison and does not claim cross-seed portability.

| run_id | observer_mode | target_family | ema_decay |
| --- | --- | --- | --- |
| control_off_none | off | none | null |
| previous_step_direction | previous_step | direction | null |
| previous_step_candidate_order | previous_step | candidate_order | null |
| previous_epoch_direction | previous_epoch | direction | null |
| previous_epoch_candidate_order | previous_epoch | candidate_order | null |
| ema_direction | ema | direction | 0.99 |
| ema_candidate_order | ema | candidate_order | 0.99 |

No combined target-family run, multiple-teacher run, teacher selection, teacher
ranking, teacher loss, loss coefficient, training intervention, direction/order
combination, or EMA decay selection is authorized.

## EMA Authority

EMA decay is frozen ex ante:

- `ema_decay = 0.99`
- authority label:
  `EX_ANTE_EFFECTIVE_HORIZON_100_SUCCESSFUL_STEPS`

Rationale: the decay is frozen before observing any run result; it corresponds
to an approximate `1 / (1 - decay) = 100` successful-step averaging horizon; it
is used only for observability; it is not selected from clean-dev,
recovery/harm, target density, or performance; and it is not compared against
alternative decay values.

## Manifest Fingerprints

Each row emits `run_id`, `seed`, `observer_mode`, `target_family`, `ema_decay`,
`ema_decay_authority`, `trainer_script`, `trainer_args`,
`normalized_base_args`, `observer_specific_args`, `output_dir`,
`observer_output_dir`, `checkpoint_path`, `stdout_log`, `stderr_log`,
`config_fingerprint`, `base_config_fingerprint`,
`observer_config_fingerprint`, `expected_runtime_sidecars`, and
`expected_checkpoint_observer_state`.

Fingerprints use canonical JSON serialization with sorted keys and SHA256.
`base_config_fingerprint` must be identical across all seven rows after
excluding only observer, run path, and checkpoint path fields.
`config_fingerprint` must be unique per run.

## Runtime Semantics

The control run must pass `--teacher-observer-mode off` and
`--teacher-observer-target-family none`, with no EMA decay. Expected control
behavior is no teacher allocation, no observer sidecars, no
`teacher_observer_state` checkpoint subtree, and no extra observational
teacher/student forward.

Every enabled run must produce exactly five sidecars:

- `teacher_observer_manifest.json`
- `teacher_observer_batch_metrics.jsonl`
- `teacher_observer_epoch_metrics.csv`
- `teacher_observer_run_summary.json`
- `teacher_observer_state_audit.json`

Every enabled checkpoint must contain exactly one namespaced
`teacher_observer_state`. No run may contain more than one teacher state.

## Orchestrator

`scripts/run_stage196b2b6p9p3_separate_observational_runs.py` requires
`--manifest-json`, `--repo-root`, `--output-root`, and `--python-executable`.
Optional `--run-id` executes exactly one matching run; otherwise all seven rows
run sequentially in manifest order. The orchestrator does not run GPU jobs
concurrently.

For each run it validates the manifest row, creates an isolated directory,
writes `resolved_command.json`, launches the trainer through subprocess,
captures complete stdout and stderr, writes `execution_status.json`, validates
sidecar presence or absence, validates checkpoint presence, preserves the
trainer return code, and does not mark failed runs complete.

Restart skipping is allowed only when `execution_status.json` says success, the
resolved command fingerprint matches, all required outputs exist, and all
output hashes match the stored completion record. Incompatible completed or
partial runs are not silently overwritten.

## Analyzer

`scripts/analyze_stage196b2b6p9p3_separate_observational_runs.py` requires
`--manifest-json`, `--runs-root`, `--stage196b2b6p9p2-analysis-json`,
`--repo-root`, and `--output-dir`.

It may read trainer summaries, selected/final checkpoint metadata, observer
sidecars, execution status files, resolved commands, and prediction or scalar
exports already produced by the frozen base configuration. It must not load a
model, run a forward pass, run training, run evaluation, or fabricate unavailable
loss or gradient fields.

Floating summaries are compared with `abs_tol = 1e-8` and `rel_tol = 1e-6`.
Discrete and hash fields require exact equality. Unavailable fields are
classified `UNAVAILABLE`, never as matches.

## Runtime Closure

The analyzer requires exact seven manifest rows, exact seven execution
directories, zero trainer return codes, command fingerprints matching the
manifest, one seed, one base configuration fingerprint, observer-only
configuration differences, zero control sidecars, five sidecars for each
enabled run, no control checkpoint observer state, exactly one enabled
checkpoint observer state, student trajectory exact/tolerance matches,
lifecycle closure, target metric closure, unavailable loss and gradient targets,
no teacher selection, no performance ranking, and exactly ten analysis outputs.

## Trajectory Invariance

Every enabled run is compared to `control_off_none` using every source-backed
invariant available, including selected epoch, final epoch, best clean-dev
epoch, model-state hashes, checkpoint fingerprints, prediction fingerprints,
clean-dev metric vectors, training and dev loss histories, and optimizer
successful/skipped step counts. A mismatch is an observer-safety failure, not a
teacher-quality signal.

## Lifecycle Checks

Previous-step requires initialized teacher state, positive read count, update
count equal to successful optimizer steps, skipped steps not advancing the
observer, first-batch teacher equal to initialized student, and internally
consistent one-successful-step lag metadata.

Previous-epoch requires initialized teacher state, positive read count,
`boundary = start_of_current_epoch`, epoch-1 snapshot equal to initialized
student, update count equal to transitions into later epochs, and no best or
selected checkpoint authority. For `E` completed epochs with no resume, expected
transition count is `max(E - 1, 0)`.

EMA requires initialized teacher state, decay `0.99`, matching decay authority,
positive read count, update count equal to successful optimizer steps, skipped
steps not advancing EMA, complete floating tensor coverage, exact copying of
integer/non-floating buffers, and exported `effective_teacher_age`.

## Target Closure

Direction runs require the eight `direction_*` metric fields. Closure requires
`total = ties + positive + negative` and agreement plus disagreement equal to
the comparable active student/teacher targets. Candidate-order runs require the
eight `order_*` metric fields and `total = ties + positive + negative`.
Exact ties are excluded from active targets. Order must not be derived lexically.

Degeneracy is flagged when active target count is zero, all active targets have
only one sign, or exact tie rate is one. Degeneracy is an observational finding,
not automatically a runtime failure.

Loss and gradient target fields remain unavailable with
`available = false` and
`reason = NO_LOSS_OR_BACKWARD_IN_OBSERVATIONAL_STAGE`.

## Decisions

P9-P3-P0 decision hierarchy:

- `STAGE196B2B6P9P3P0_BLOCKED_UPSTREAM_AUTHORITY`
- `STAGE196B2B6P9P3P0_BLOCKED_BASE_CONFIG_AUTHORITY`
- `STAGE196B2B6P9P3P0_BLOCKED_OBSERVER_CLI_AUTHORITY`
- `STAGE196B2B6P9P3P0_MANIFEST_READY`

P9-P3 runtime decision hierarchy:

- `STAGE196B2B6P9P3_BLOCKED_MANIFEST_AUTHORITY`
- `STAGE196B2B6P9P3_INCOMPLETE_RUN_SET`
- `STAGE196B2B6P9P3_RUNTIME_CONTRACT_FAILURE`
- `STAGE196B2B6P9P3_STUDENT_TRAJECTORY_UNSAFE`
- `STAGE196B2B6P9P3_ALL_TARGETS_DEGENERATE`
- `STAGE196B2B6P9P3_PARTIAL_OBSERVABILITY`
- `STAGE196B2B6P9P3_SEPARATE_OBSERVABILITY_COMPLETE`

`SEPARATE_OBSERVABILITY_COMPLETE` requires all seven runs complete, all runtime
contracts pass, all enabled trajectories match control, at least one
nondegenerate direction candidate, and at least one nondegenerate
candidate-order candidate. It does not require all three teachers to be
nondegenerate and does not authorize a teacher.

## Output Closure

The manifest builder writes exactly:

- `stage196b2b6p9p3p0_manifest.json`
- `stage196b2b6p9p3p0_run_table.csv`
- `stage196b2b6p9p3p0_base_config_authority.json`
- `stage196b2b6p9p3p0_decision_gate.csv`
- `stage196b2b6p9p3p0_contract.csv`

The analyzer writes exactly:

- `stage196b2b6p9p3_analysis.json`
- `stage196b2b6p9p3_report.md`
- `stage196b2b6p9p3_run_completion.csv`
- `stage196b2b6p9p3_config_fingerprint_audit.csv`
- `stage196b2b6p9p3_student_trajectory_invariance.csv`
- `stage196b2b6p9p3_lifecycle_runtime_audit.csv`
- `stage196b2b6p9p3_target_observability_summary.csv`
- `stage196b2b6p9p3_degeneracy_audit.csv`
- `stage196b2b6p9p3_decision_gate.csv`
- `stage196b2b6p9p3_contract.csv`




