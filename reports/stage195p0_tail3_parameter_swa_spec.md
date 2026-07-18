# Stage195-P0 tail-three trainable-parameter SWA causal-test specification

## Status and scientific scope

Stage195-P0 is default-off diagnostic infrastructure. It does not select a production model and does not authorize a six-run matrix, model advancement, or subsequent training.

Stage194-A emitted `STAGE194A_MIXED_TEMPORAL_AND_BOUNDARY_MECHANISMS`. Its frozen pooled evidence was:

- selected-to-tail3 SUPPORT losses to NOT_ENTITLED: 90;
- selected-consensus outliers: 85, a consensus-outlier share of 0.944444;
- mean-magnitude overrides: 5, a magnitude-override share of 0.055556;
- tail-three mean NOT_ENTITLED false negatives: 258;
- persistent stable negatives: 161, a persistent-bias share of 0.624031.

Both temporal consensus and persistent entitlement bias are present. Temporal consensus has the larger evidence share, so Stage195-P0 isolates only that mechanism by averaging late-epoch trainable parameters. It adds no entitlement calibration, class or margin shift, median, trimmed mean, robust-logit aggregation, loss, head, or data source.

## Frozen modes and seeds

The internal trajectory-observability modes are exactly:

1. `none`;
2. `stage191_deterministic_replay`;
3. `stage193_tail3_fresh_seed_replication`;
4. `stage195_tail3_parameter_swa_causal_test`.

The Stage191, Stage193, and Stage195 enable flags are mutually exclusive. Seed validation is mode-local, never a union:

- `STAGE191_TRAINING_SEEDS = (174, 175, 176)`;
- `STAGE193_TAIL3_REPLICATION_SEEDS = (177, 178, 179)`;
- `STAGE195_TAIL3_PARAMETER_SWA_SEEDS = (180, 181, 182)`.

Stage195 requires an explicit split seed of 174.

The new CLI is:

- `--stage195-tail3-parameter-swa-causal-test`, `action="store_true"`, `default=False`;
- `--stage195-tail3-parameter-swa-output-dir`, `type=Path`, `default=None`.

When the enable flag is absent, the output-dir option must be absent. When enabled, the output directory is required and must resolve exactly to the parent of `--output-json`. The three Stage195 target files must not exist. Stage195 never overwrites them and does not permit an independent output location.

## Runtime envelope

An enabled Stage195 run fails closed unless all of the following resolve exactly:

- architecture `v6b_minimal`;
- backbone `mamba`;
- model `state-spaces/mamba-130m-hf`;
- device `cuda`;
- 20 epochs;
- selection metric `final_macro_f1`;
- flag source `controlled_heuristic`;
- explicit split seed 174;
- main data `data/controlled_v5_v3_without_time_swap.jsonl` with the frozen identity;
- no truncation, smoke mode, or loss sweep;
- no OOD or external evaluation/data;
- no bridge rows or bridge training;
- no `time_swap` main data;
- no temporal auxiliary data, diagnostic loss, residual adapter, adapter loss/final penalty, temporal channel/loss/gated penalty, temporal safety data/head/loss, temporal mismatch multihead data/head/loss, or temporal preservation data/head/loss;
- no coverage-entailment auxiliary data/head/loss;
- no pair-contrastive auxiliary data/loss;
- no TD, preservation, or Stage44 constrained checkpoint selector;
- no OOD or dev-calibrated NOT_ENTITLED shift;
- no Stage191 state capsules.

Stage195 reuses the Stage193 forbidden external, OOD, bridge, auxiliary-path, auxiliary-enable, and auxiliary-mode contracts without weakening Stage193.

The only arms are the frozen Stage193 arms:

- Baseline: compatible-positive margin weight 0.0, margin logit 0.0, no integrity sidecar path, and no expected sidecar semantic SHA256.
- Intervention: compatible-positive margin weight 0.05, margin logit 0.0, the authoritative Stage185 sidecar, and semantic SHA256 `5bc03caa2a29f9b9176ab4eb0201db57ebad516352797546db1a18e6ec3373fc`.

Stage195-P0 authorizes one invocation at a time only. A later Stage195-A manifest must freeze any six-run baseline/intervention matrix independently.

## Default-off and historical invariance

With Stage195 disabled, the trainer does not allocate a parameter accumulator, copy parameters to CPU, add a forward pass, write Stage195 files, or touch model state. The Stage195-only parsed fields are omitted from generic serialized argument snapshots when disabled, preserving legacy report, checkpoint-metadata, and provenance schemas.

Stage191 behavior remains governed only by seeds 174–176. Stage193 remains governed only by seeds 177–179, still rejects state capsules, and retains its existing trajectory contract, epoch-ledger, and prediction schemas. Existing selection, best-epoch, final-epoch, checkpoint, and training-report semantics are unchanged.

## Shared trajectory observability

Stage195 reuses the existing Stage191 implementation exactly as Stage193 does. It emits:

- `stage191_trajectory_contract.json`;
- `stage191_trajectory_epoch_metrics.jsonl` with exactly 20 rows;
- `stage191_dev_predictions_epoch_NNN.jsonl` for epochs 1 through 20, with exactly 720 rows per file and `dev_position` 0 through 719;
- zero state capsules.

Canonical labels and logit columns are `REFUTE`, `NOT_ENTITLED`, `SUPPORT`. Final CE, predictions, and metrics use only `output["logits"]`; `loss_logits` is never used.

The Stage195 trajectory contract records the Stage195 observability mode, authorized seeds `[180, 181, 182]`, exact run seed, split seed 174, 20 epochs, 720 expected dev rows, zero expected state capsules, source epochs `[18, 19, 20]`, reused Stage191 implementation, disabled capsules, unchanged training semantics and gradients, zero extra training forwards, one post-training clean-dev sweep, no external data, no SWA checkpoint, no calibration, and no entitlement-boundary shift.

## Exact parameter scope and optimizer ownership

The averaged mapping is `unique_trainable_named_parameters_only`: the unique sequence returned by `model.named_parameters()` filtered by `parameter.requires_grad is True`.

Stage195 does not average optimizer state, scheduler state, gradients, frozen parameters, buffers, RNG state, or checkpoint-selection metadata. Frozen parameters and buffers remain sourced from epoch 20.

Before training capture, Stage195 compares the object-identity set of the trainable named mapping with the parameters owned by all optimizer groups. It fails on:

- an optimizer parameter without a trainable model name;
- a trainable named parameter absent from the optimizer;
- duplicate optimizer ownership;
- duplicate trainable identities or names;
- a non-floating trainable parameter.

The contract records the ownership validation, trainable parameter count, optimizer parameter count, and total averaged numel. Counts and epoch identities are exact non-bool integers.

## Capture, accumulation, application, and restoration

Source epochs are exactly 18, 19, and 20 in that order. For each source epoch, capture occurs after the normal optimizer step, after the existing clean-dev result, and immediately after the corresponding Stage191 trajectory export. Therefore the captured parameter state is the state that produced the epoch-N exported logits. Capture performs no forward pass, changes no parameter, and uses no random operation.

For every trainable named parameter, capture:

1. detaches the tensor;
2. copies it to CPU in its original dtype;
3. checks finiteness;
4. verifies stable name, object identity, shape, and dtype;
5. adds it to an online CPU `torch.float64` sum.

After epoch 20, each sum is divided by exactly 3 in CPU float64. Each mean is cast to that parameter's original dtype and copied under `torch.no_grad()` into the existing parameter object. The implementation creates no second CUDA model, performs no CUDA `deepcopy(model)`, and retains only the online float64 accumulator plus the exact epoch-20 restoration mapping; the cast mean is materialized one parameter at a time.

Before temporary application, the current trainable fingerprint must equal the captured epoch-20 fingerprint. Exact original-dtype CPU values for all affected epoch-20 parameters are retained. Application, evaluation, and subordinate artifact writing are enclosed by restoration logic. A `finally` block copies every retained epoch-20 value back into its original parameter object, restores the prior train/eval mode, and verifies the restored fingerprint equals the epoch-20 source fingerprint.

No optimizer or scheduler state is read for averaging or changed. No optimizer step follows the diagnostic. The diagnostic runs before existing post-loop checkpoint-selection override logic and before any outer control flow can load the trainer-selected checkpoint.

## Parameter fingerprints

One SHA256 mapping fingerprint is used for source, averaged, and restored trainable states. Parameters are sorted lexicographically by name. For each tensor the hash consumes, in order:

1. UTF-8 parameter name and a NUL byte;
2. canonical dtype text such as `torch.float32` and a NUL byte;
3. compact canonical JSON shape and a NUL byte;
4. contiguous CPU tensor bytes in the parameter's original dtype.

The Stage195 contract records lowercase 64-character hashes for:

- `epoch18_trainable_parameter_sha256`;
- `epoch19_trainable_parameter_sha256`;
- `epoch20_trainable_parameter_sha256`;
- `averaged_trainable_parameter_sha256`;
- `restored_epoch20_trainable_parameter_sha256`.

The restored hash must equal the epoch-20 hash. The contract also records source capture count 3, source epochs, parameter count and numel, accumulator dtype/device, original-dtype casting, and verified restoration.

## One clean-dev diagnostic sweep

The temporarily averaged model is evaluated exactly once over the existing ordered 720-row clean-dev inputs. The model is in eval mode and the sweep uses inference mode, creates no gradients, and invokes no optimizer, scheduler, checkpoint-selection, best-score, or best-epoch mutation.

The single sweep's `output["logits"]` tensor is reused to construct both prediction rows and metrics. Canonical argmax tie order is column order `REFUTE`, `NOT_ENTITLED`, `SUPPORT`. The formulas are the frozen Stage191/Stage193 formulas: mean row CE, accuracy, three-class macro F1, SUPPORT recall, false-entitlement total, false-NOT_ENTITLED total, polarity-error total, dense label counts, and gold-row/prediction-column confusion matrix.

## Runtime artifacts and schemas

Stage195 writes exactly three additional final files in the explicit Stage195 output directory:

1. `stage195_tail3_parameter_swa_predictions.jsonl`;
2. `stage195_tail3_parameter_swa_metrics.json`;
3. `stage195_tail3_parameter_swa_contract.json`.

Predictions and metrics are published atomically first. Their file SHA256 values are computed only after both subordinate files are complete. The final contract is published atomically last, so no valid-looking contract is left when a subordinate artifact is incomplete.

Each of exactly 720 prediction rows contains exactly:

`stage`, `source`, `run`, `training_seed`, `split_seed`, `arm`, `source_epochs`, `dev_position`, `gold_final_label`, `predicted_final_label`, `final_logits`, and `final_ce`.

Fixed values are `stage = "Stage195-P0"`, `source = "tail3_trainable_parameter_swa"`, and `source_epochs = [18, 19, 20]`. Positions are exact integers 0–719. Labels are canonical, logits are finite length-three arrays, predictions are canonical argmax, and CE is finite and nonnegative.

The metrics JSON contains exactly the defined fields:

`stage`, `source`, `run`, `training_seed`, `split_seed`, `arm`, `source_epochs`, `row_count`, `clean_ce`, `accuracy`, `macro_f1`, `support_recall`, `false_entitlement_total`, `false_not_entitled_total`, `polarity_error_total`, `pred_counts`, `gold_counts`, `confusion_matrix`, `canonical_labels`, `logits_source`, `checkpoint_selection_used`, and `external_data_used`.

It requires row count 720, source `tail3_trainable_parameter_swa`, logits source `output["logits"]`, and false values for checkpoint selection and external data.

The final contract records the runtime envelope and run identity, enabled flags, source epochs, parameter and optimizer scope, five fingerprints, counts, accumulator properties, exact restoration, one extra clean-dev sweep, unchanged gradients and selection, no optimizer/scheduler averaging, no checkpoint save, artifact paths and hashes, expected 720 rows, the trainer blob commit from existing provenance, trainer SHA256, and all authorization denials.

Stage195 writes no SWA `.pt`, state capsule, selected checkpoint, temporary checkpoint, pickle, or NumPy archive.

## Provenance separation and authorization restrictions

The Stage195-P0 contract's trainer source commit is the commit identifying the trainer blob used for the run. It is not a future runtime repository commit and must not be conflated with one.

A later Stage195-A must independently freeze:

- the Stage195-P0 trainer blob commit;
- the Stage195-P0 trainer SHA256;
- the later Stage195 runtime repository commit;
- seeds 180, 181, and 182;
- the exact six-run baseline/intervention matrix;
- explicit output directories.

Stage195-P0 explicitly creates no Stage195-A manifest, authorizes no Stage195-B run, makes no Stage195-C decision, selects no production SWA model, implements no entitlement correction, authorizes no model advancement, authorizes no subsequent training, and makes no statistical-significance claim.
