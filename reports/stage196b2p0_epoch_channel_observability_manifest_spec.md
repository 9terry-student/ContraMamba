# Stage196-B2-P0 epoch-channel observability manifest specification

## Generator

`scripts/make_stage196b2p0_epoch_channel_observability_manifest.py` requires:

```text
--repo-root
--current-git-commit
--stage196b1-runtime-git-commit
--output-dir
```

The output argument is an existing parent. The generator creates one new
`stage196b2p0_epoch_channel_observability_manifest_YYYYMMDD_HHMMSS` directory.
It performs no training or model/checkpoint loading.

## Exact seven-file closure

The timestamped directory contains exactly:

1. `stage196b2p0_manifest.json`
2. `stage196b2p0_run_commands.jsonl`
3. `stage196b2p0_expected_outputs.json`
4. `stage196b2p0_pairing_contract.json`
5. `stage196b2p0_source_closure.json`
6. `stage196b2p0_execution_order.txt`
7. `stage196b2p0_manifest_report.md`

Both ready and fail-closed blocked manifests retain this seven-file closure.

## Run order and argv contract

The order is exactly:

```text
seed183_joint
seed183_frame_local_only
seed184_joint
seed184_frame_local_only
seed185_joint
seed185_frame_local_only
```

Every command is represented as an argv array. No shell command string is
constructed. Commands preserve the original Stage196-B1 configuration and add
only the new output root and
`--stage196b2p0-epoch-channel-observability`.

The frozen common values are data
`data/controlled_v5_v3_without_time_swap.jsonl` with SHA-256
`f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640`,
Mamba-130m, `v6b_minimal`, CUDA, 20 epochs, split seed 174, learning rate
0.001, both freeze flags true, direct FrameGate BCE 1.0, margin settings 0.0,
class weighting `none`, final macro-F1 selection, max length 128, dev ratio
0.2, gradient accumulation 1, controlled heuristic flags, and selected
checkpoint saving.

Each seed has joint then frame-local-only. Normalizing seed, mode, and the three
ordinary output paths must make all argv arrays identical.

## Prohibited configuration

The validator rejects external/OOD/Stage43 options, bridge paths or modes,
calibration or threshold options, Stage195, state capsules, integrity sidecars,
smoke, loss sweeps, row truncation, and any nonzero compatible margin. Run output
directories must be six unique, nonexisting children of
`reports/stage196b2p0_epoch_channel_observability_runs`.

## Expected artifacts

Each run reserves ordinary reports, predictions, scalars, checkpoint, trajectory
contract/ledger, 20 existing trajectory prediction exports, and exactly 20 P0
channel sidecars. Each P0 file expects 720 rows and the exact 18-field schema in
the P0 observability specification. State-capsule count is zero.

## Equivalence contract

Historical/new equality covers seed, split seed, data path/SHA, architecture,
backbone, model, device, epochs, learning rate, freezing, gradient mode, direct
FrameGate BCE, margin settings, class weighting, selection metric, max length,
dev ratio, gradient accumulation, flag source, and checkpoint saving.

Authorized differences are only output root, the P0 flag, and P0 sidecars.
Prediction equivalence is explicitly not claimed before execution.

## Source closure

The current commit must be lowercase full SHA and equal HEAD. Current trainer
bytes must equal the trainer at that commit. The supplied historical runtime must
equal `9835cbbf86d83aca0964821669e63f7f6deb1c59`. The FrameGate origin commit
`5a39538ef3ca8f36cc2cc5d3290eae60d6a5f5c8` must be an ancestor of it.
Data SHA is recomputed. The manifest records the historical B1 and new P0
runtime commits as separate roles.

## Execution order

Future execution is fail-fast in the recorded order. The manifest itself does
not execute commands and records `training_performed=false`.

