# Stage196-B1-A FrameGate gradient-ownership manifest specification

## Scope and causal question

Stage196-B1-A freezes, but does not execute, the six-run Stage196-B1 causal
experiment. It asks only:

> Does blocking direct non-frame gradient paths through FrameGate outputs reduce
> recurrent persistent SUPPORT false-NOT_ENTITLED failures?

The sole intervention is the trainer option
`--frame-downstream-gradient-mode {joint,frame_local_only}` implemented at
commit `5a39538ef3ca8f36cc2cc5d3290eae60d6a5f5c8`. The baseline arm is recorded as
`arm=baseline` and runtime mode `joint`; it is never internally renamed to a
runtime mode called `baseline`. The intervention arm is recorded as
`arm=intervention` and runtime mode `frame_local_only`.

This stage performs no training, model loading, checkpoint loading, smoke run,
mini run, full run, external evaluation, threshold search, or calibration.

## Explicit generator inputs and deterministic paths

The generator requires exactly these explicit arguments:

1. `--repo-root`
2. `--current-git-commit`
3. `--output-dir`

The supplied commit is a lowercase full 40-character Git SHA and must equal the
repository HEAD. The trainer and data files must exist. The runtime trainer
bytes must equal `scripts/train_controlled_v6b_minimal.py` at that supplied
commit, while commit `5a39538ef3ca8f36cc2cc5d3290eae60d6a5f5c8` remains the
frozen FrameGate ownership implementation provenance and must be an ancestor.
The data SHA-256 is recomputed and must equal the frozen clean-data identity.

The future run root is the deterministic sibling
`<output-dir-parent>/stage196b1_framegate_gradient_ownership_runs`. Run
directories are immediate children named by the exact run identifiers below.
The generator creates only `output-dir`; it never creates the future run root or
any run directory. An existing nonempty output directory is rejected.

## Exact paired run matrix

The exact order is:

1. `seed183_joint`: seed 183, baseline, `joint`
2. `seed183_frame_local_only`: seed 183, intervention, `frame_local_only`
3. `seed184_joint`: seed 184, baseline, `joint`
4. `seed184_frame_local_only`: seed 184, intervention, `frame_local_only`
5. `seed185_joint`: seed 185, baseline, `joint`
6. `seed185_frame_local_only`: seed 185, intervention, `frame_local_only`

There are exactly three fresh training seeds, exactly two runs per seed, and
exactly one run of each runtime mode per seed. Baseline precedes intervention
within every seed. All command arguments are stored as arrays. No shell command
string is constructed or recorded.

## Frozen common training configuration

Every run uses:

~~~text
--data data/controlled_v5_v3_without_time_swap.jsonl
--backbone mamba
--model-name state-spaces/mamba-130m-hf
--architecture v6b_minimal
--device cuda
--epochs 20
--seed <183|184|185>
--split-seed 174
--frame-downstream-gradient-mode <joint|frame_local_only>
--stage196b1-framegate-gradient-ownership-observability
--stage115-clean-dev-scalar-output-jsonl <run_dir>/clean_dev_scalars.jsonl
--output-json <run_dir>/training_report.json
--output-predictions-json <run_dir>/clean_dev_predictions.json
--compatible-positive-margin-weight 0.0
--compatible-positive-margin-logit 0.0
~~~

Trajectory provenance is frozen as:

~~~json
{
  "trajectory_observability_mode": "stage196b1_framegate_gradient_ownership",
  "stage191_trajectory_observability_implementation_reused": true,
  "stage191_frozen_seed_contract_modified": false,
  "stage196b1_authorized_training_seeds": [183, 184, 185]
}
~~~


The manifest also materializes the unchanged Stage195 training semantics where
the CLI has an explicit representation: learning rate 0.001, frozen encoder,
frozen `a_log`, maximum length 128, dev ratio 0.2, gradient accumulation 1,
class weighting `none`, selection metric `final_macro_f1`, flag source
`controlled_heuristic`, and selected-checkpoint filename
`selected_checkpoint.pt` with selected-checkpoint saving enabled.

The authoritative Stage195 argv omitted train/eval batch-size arguments and
`--fp16`; therefore Stage196-B1 preserves full-split train/eval forwards and
fp16 disabled. It also preserves `head_lr=None`, `encoder_lr=None`, AdamW via
`v5.build_optimizer`, no scheduler, and no weight decay configured. Those
resolved values are explicit in the manifest even where the CLI has no safe
negative/`None` spelling. Unrelated historical defaults are not consulted.

The old Stage174-C, Stage175-B, and Stage177-C explicit off-options are not
copied into commands: their losses remain disabled by their unchanged defaults,
and this manifest prohibits those interventions rather than treating them as
arms.

After normalizing only run identity, seed, gradient mode, and output paths, all
six argv arrays must be token-for-token equal. Within each seed pair, seed is
already identical, so mode and output paths are the only argv differences.

## Explicit prohibitions

Every manifest row asserts:

~~~text
time_swap_used_in_main_training = false
external_training_data_used = false
external_eval_enabled = false
external_metrics_used_for_selection = false
stage195_parameter_swa_enabled = false
compatible_positive_margin_enabled = false
new_loss_enabled = false
threshold_tuning_enabled = false
calibration_enabled = false
~~~

Commands contain no Stage195 SWA option, Stage193 fresh-seed convenience option,
integrity sidecar, external/OOD or bridge path/flag, selector/composer override,
post-hoc calibration, threshold tuning, compatible-positive intervention, clean
pairwise loss, support-anchor loss, frame pairwise loss, contrastive loss,
boundary loss, frame-violation loss, or temporal auxiliary loss.

## Per-run artifact closure

Each deterministic run directory reserves these ordinary trainer outputs:

- `training_report.json`
- `clean_dev_predictions.json`
- `clean_dev_scalars.jsonl`
- `selected_checkpoint.pt`
- `stage191_trajectory_contract.json`
- `stage191_trajectory_epoch_metrics.jsonl`
- `stage191_dev_predictions_epoch_001.jsonl` through
  `stage191_dev_predictions_epoch_020.jsonl`
- runner-captured `stdout.log` and `stderr.log`

The twenty trajectory prediction exports include, in particular, exact epochs
18, 19, and 20. Expected ledger rows and prediction exports are both 20, with
720 prediction rows per epoch. No state capsule, Stage195 parameter-SWA
artifact, SWA checkpoint, or temporary averaged-parameter checkpoint is
requested.

## Required runtime provenance checks

The future runner must parse every completed `training_report.json` and
`stage191_trajectory_contract.json` and require:

~~~text
training_seed == expected seed
configured_split_seed == 174
resolved_split_seed == 174
split_seed_explicit == true
split_policy == fixed_explicit_split_seed
observability_mode == stage196b1_framegate_gradient_ownership
authorized_training_seeds == [183, 184, 185]
stage196b1_framegate_gradient_ownership_observability == true
stage191_trajectory_replay_observability == false
stage193_tail3_fresh_seed_observability == false
stage195_tail3_parameter_swa_causal_test == false
device == cuda
backbone == mamba
model_name == state-spaces/mamba-130m-hf
architecture == v6b_minimal
frame_downstream_gradient_mode == expected mode
frame_direct_loss_active == true
frame_direct_loss_weight == 1.0
frame_downstream_forward_value_changed == false
shared_encoder_gradient_fully_isolated == false
~~~

For `joint`, `framegate_nonframe_output_gradient_blocked` must be false. For
`frame_local_only`, it must be true. The mode-specific value is frozen in each
run row rather than inferred later from the arm label.

## First-run runtime risk and fail-fast execution policy

`seed183_joint`, run 1, validates ordinary joint execution under the dedicated
Stage196-B1 observability mode. `seed183_frame_local_only`, run 2, is the first
actual runtime test of the hook-based ownership path. Static CLI/source
validation does not eliminate mode resolution, trajectory export, hook ordering,
autograd, device, or full-training runtime risk. The runner must fail immediately
on either run's runtime failure.

For every run the future runner must open and capture `stdout.log` and
`stderr.log`, execute the recorded argv array without a shell, preserve the run
directory on success or failure, stop immediately on any nonzero return, and
never continue to later runs after a hook or other runtime failure. It must not
delete or rewrite partial artifacts. A runtime failure is an execution failure,
not scientific evidence for or against the causal hypothesis.

## Precommitted Stage196-B1-C comparison

The later analyzer compares `joint` with `frame_local_only` within training
seed. Its primary causal quantities are:

- persistent stable SUPPORT-negative count
- recurrent persistent SUPPORT positions
- universal persistent SUPPORT positions
- FrameGate failure count within persistent SUPPORT negatives
- frame probability on Stage196-A recurrent positions
- stable-correct SUPPORT preservation
- false-entitlement count
- false-NOT_ENTITLED count
- polarity error count
- SUPPORT recall
- macro-F1

No numeric success threshold, aggregate decision boundary, or significance rule
is selected by this manifest generator.

## Interpretation restrictions

Even after successful execution, the only authorized causal claim concerns
direct non-frame gradient paths through FrameGate outputs. This manifest does
not authorize claims about external/OOD performance, production readiness, full
shared-representation isolation, intrinsic representation failure,
contrastive-loss necessity, or final architecture superiority.

## Generator outputs and decisions

The generator publishes exactly these seven files inside `output-dir`, with the
JSON report published last:

1. `stage196b1_run_manifest.json`
2. `stage196b1_run_manifest.csv`
3. `stage196b1_run_commands.jsonl`
4. `stage196b1_manifest_report.json`
5. `stage196b1_manifest_report.md`
6. `stage196b1_source_closure.csv`
7. `stage196b1_precommitted_contract.csv`

READY is
`STAGE196B1_FRAMEGATE_GRADIENT_OWNERSHIP_MANIFEST_READY`. BLOCKED is
`STAGE196B1_FRAMEGATE_GRADIENT_OWNERSHIP_MANIFEST_BLOCKED`. READY means only
that the frozen six-run manifest is internally and source consistent. It does
not mean execution occurred or that the runtime hook was validated by a run.

The generator validates exact run order, seeds, pairing, modes, split seed,
data identity, Mamba/CUDA/v6b identity, epoch count, observability and scalar
exports, margins off, forbidden feature absence, unique paths, argv-array use,
commit format, trainer/data existence, mode-specific provenance, output-name
closure, and paired-argument equality. Publication failures are converted to a
fixed BLOCKED closure where safe; no output is overwritten.
