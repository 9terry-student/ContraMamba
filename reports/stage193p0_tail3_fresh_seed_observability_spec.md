# Stage193-P0 tail3 fresh-seed trajectory-observability extension

## Scope

Stage193-P0 adds a narrow, report-only trajectory-observability mode to `scripts/train_controlled_v6b_minimal.py`. It does not build the Stage193-A manifest, implement the Stage193-C analyzer, create run commands, execute a run, authorize Stage193-B, authorize later training, or make a model-advancement decision.

The patch is default-off. With neither observability flag supplied, trainer behavior is unchanged.

## Why Stage191 remains frozen

Stage191-B is a deterministic replay contract for the historical training seeds 174, 175, and 176. Its fixed source constant remains exactly:

```text
STAGE191_TRAINING_SEEDS = (174, 175, 176)
```

Those seeds are provenance-bearing identities, not a general trajectory-export allowlist. Adding fresh seeds to the Stage191 tuple would weaken and reinterpret the frozen Stage191 replay contract. Stage191 therefore retains its existing flag, seed rejection, state-capsule option, output filenames, metric formulas, logits source, and contract values. A Stage191 invocation continues to accept only seeds 174, 175, and 176.

## Separate Stage193 mode

Fresh-seed replication requires the same report-only observations over new training trajectories while keeping Stage191 historically exact. The trainer therefore defines the separate fixed constant:

```text
STAGE193_TAIL3_REPLICATION_SEEDS = (177, 178, 179)
```

It also adds exactly one dedicated CLI flag:

```text
--stage193-tail3-fresh-seed-observability
```

The flag uses `action="store_true"` and `default=False`. There is no seed override, generic allowlist, or dynamic extension mechanism.

The trainer resolves one internal observability mode:

1. `none`
2. `stage191_deterministic_replay`
3. `stage193_tail3_fresh_seed_replication`

The Stage191 and Stage193 flags are mutually exclusive. Each enabled mode validates its own exact seed tuple; the code does not validate against a union of historical and fresh seeds.

## Frozen Stage193 runtime envelope

Stage193 mode accepts only training seeds 177, 178, and 179 with an explicitly resolved split seed of 174. It requires `v6b_minimal`, the Mamba backbone, CUDA, model `state-spaces/mamba-130m-hf`, 20 epochs, and the existing `final_macro_f1` selection semantics. The main dataset must resolve exactly to `data/controlled_v5_v3_without_time_swap.jsonl` and retain its existing frozen dataset identity.

The mode reuses the trainer's resolved option values and the established Stage191 external/OOD/bridge absence sets. The frozen `v6b_minimal` temporal comparator setting is preserved exactly from the Stage189/Stage191 training template; it is existing model behavior and is not a new Stage193 treatment. No temporal auxiliary data, diagnostic loss, residual adapter, temporal channel, cap, or constrained checkpoint selection is permitted, and Stage193-P0 does not introduce or activate any new temporal mechanism. Coverage-entailment, pair-contrastive, and other synthetic auxiliary paths or enabled modes remain rejected. Loss sweeps, smoke execution, and row truncation are rejected.

The baseline arm retains zero compatible-positive margin weight and must omit both Stage185 sidecar options. The intervention arm retains weight 0.05, margin target 0, the exact authoritative Stage185 sidecar path, and its exact semantic SHA. These checks only preserve the frozen arm semantics; this specification does not authorize either arm to run.

## Shared trajectory implementation

Stage193 routes through the existing `_stage191_write_contract` and `_stage191_export_epoch` implementation. It does not duplicate or redefine trajectory formulas. The shared epoch exporter consumes the clean-dev `dev_output` already computed by the normal epoch evaluation. It performs no additional model forward and reads final classifier logits only from `output["logits"]`; `loss_logits` is never used.

For each epoch 1 through 20, the shared implementation preserves:

- exactly 720 clean-dev rows;
- `dev_position` values 0 through 719 in existing dev order;
- canonical label order `REFUTE`, `NOT_ENTITLED`, `SUPPORT`;
- per-row float32 final logits and final CE;
- clean CE, accuracy, macro-F1, SUPPORT recall, false entitlement, false not-entitled, polarity error, prediction counts, gold counts, and canonical confusion-matrix definitions;
- `stage191_trajectory_epoch_metrics.jsonl`;
- `stage191_dev_predictions_epoch_NNN.jsonl`.

The export is diagnostic serialization after the existing dev evaluation and selection-score computation. It does not modify forward computation, initialization, RNG consumption, data loading or splitting, data order, optimizer or scheduler configuration, loss terms or weights, backward calls, optimizer steps, checkpoint selection, stopping, reports, or checkpoints.

## State-capsule separation

Stage193 rejects `--stage191-save-trajectory-state-capsules` during mode validation and again at the shared epoch-export boundary. No Stage193 state capsule is loaded or saved, and no capsule output is requested. The Stage193 contract records `expected_state_capsules == 0` and `state_capsule_saving_enabled == false`.

Stage191 capsule behavior is unchanged: the original Stage191 flag still requires Stage191 observability and remains available only under the historical Stage191 seed contract.

## Stage193 contract provenance

A Stage193 trajectory contract preserves the existing trajectory fields and records at minimum:

- `observability_mode == "stage193_tail3_fresh_seed_replication"`;
- `authorized_training_seeds == [177, 178, 179]`;
- the exact training seed;
- `split_seed == 174`;
- `epoch_count == 20`;
- `expected_dev_rows == 720`;
- `expected_state_capsules == 0`;
- `logits_source == 'output["logits"]'`;
- `stage191_trajectory_observability_implementation_reused == true`;
- `state_capsule_saving_enabled == false`;
- `training_semantics_changed == false`;
- `extra_forward_pass_performed == false`;
- `loss_logits_used == false`;
- `external_data_used == false`.

A Stage191 contract continues to use authorized seeds `[174, 175, 176]` and its existing enabled-flag and capsule values. It is not relabeled as Stage193.

## Authorization and interpretation restrictions

This patch is diagnostic infrastructure only. It uses no external data and authorizes no execution, training matrix, subsequent training, checkpoint selection change, deployment claim, or model advancement. A later Stage193-A manifest must independently freeze provenance, derive exact commands, validate output-directory safety, and decide whether the six-run diagnostic matrix is runnable. Stage193-C remains a separate future analyzer over frozen exports.
