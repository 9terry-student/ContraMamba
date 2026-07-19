# Stage196-B2-P0 per-epoch downstream-channel observability

## Objective

Stage196-B2-P0 adds serialization only. It supplies the native per-epoch downstream
channels needed by Stage196-B2-A and does not answer the propagation question.

## Dedicated flag and authorization

The trainer flag is:

```text
--stage196b2p0-epoch-channel-observability
```

It defaults to false, is never implied by another flag, and is valid only together
with `--stage196b1-framegate-gradient-ownership-observability`, seeds 183?185,
split seed 174, `v6b_minimal`, Mamba `state-spaces/mamba-130m-hf`, CUDA,
20 epochs, frozen encoder and A_log, and gradient mode `joint` or
`frame_local_only`.

The existing Stage196-B1 validation continues to require the frozen clean data,
direct FrameGate BCE at weight 1.0, compatible-positive margin weight/logit zero,
and absence of external/OOD, bridge, calibration, threshold search, Stage195 SWA,
state capsules, new losses, smoke/truncation, and time-swap main training.

## No-extra-forward implementation

`_stage191_export_epoch` already receives `dev_output` from the clean-dev epoch
evaluation used for predictions, metrics, and trajectory observability. The P0
branch reads only detached tensors from that object. It never calls `model` and
never changes a tensor used by gradients, losses, optimization, model output, or
checkpoint selection.

The trajectory contract records:

```text
stage196b2p0_epoch_channel_observability_enabled = true
stage196b2p0_epoch_channel_file_count = 20
stage196b2p0_epoch_channel_rows_per_file = 720
stage196b2p0_extra_forward_pass_performed = false
stage196b2p0_training_semantics_changed = false
stage196b2p0_gradient_semantics_changed = false
stage196b2p0_checkpoint_selection_changed = false
```

## Stable semantic row schema

Every sidecar row has exactly:

```text
id
source_row_id
dev_position
gold_label
prediction
intervention_type
frame_probability
predicate_coverage_probability
sufficiency_probability
polarity_support_margin
entitlement_probability
support_probability
not_entitled_probability
support_logit
not_entitled_logit
epoch
training_seed
frame_downstream_gradient_mode
```

Native internal sources are `frame_prob`, `predicate_coverage_prob`,
`sufficiency_prob`, `polarity_margin`, `entitlement_prob`, the softmax of
`output["logits"]`, and canonical final-logit columns NOT_ENTITLED=1 and
SUPPORT=2. No semantic alias search or fallback is allowed.

`polarity_support_margin >= 0` is the Stage196-A/B1-C SUPPORT-facing pass.
Both final logits are mandatory, supporting one uniform
`support_logit - not_entitled_logit` margin.

## Runtime assertions and cross-export equality

Before a sidecar is accepted, the trainer requires 720 finite native values per
channel, probability ranges for probability fields, exact schema, and 720 rows.
It asserts `id == source_row_id`, certified position, gold label, prediction,
and both canonical logits against the existing in-memory trajectory rows within
serialization tolerance. Each target must not already exist. The number of P0
sidecars must progress exactly with the epoch number.

## Output closure

The new root is:

```text
reports/stage196b2p0_epoch_channel_observability_runs
```

It contains the exact six-run order:

1. `seed183_joint`
2. `seed183_frame_local_only`
3. `seed184_joint`
4. `seed184_frame_local_only`
5. `seed185_joint`
6. `seed185_frame_local_only`

Each run creates exactly `stage196b2p0_epoch_channels_001.jsonl` through
`stage196b2p0_epoch_channels_020.jsonl`, with 720 rows per file. Historical
Stage196-B1 and B1-C directories are never targets.

## Provenance roles

- `9835cbbf86d83aca0964821669e63f7f6deb1c59`: original Stage196-B1 runtime and source of the completed B1-C mixed result.
- future Stage196-B2-P0 runtime commit: source of observability-rich reruns.
- future Stage196-B2-A analyzer commit: source of artifact-only propagation analysis.
- `5a39538ef3ca8f36cc2cc5d3290eae60d6a5f5c8`: FrameGate implementation origin.

These roles are recorded separately and are never inferred to be equivalent.

## Static execution policy

This implementation stage performs no Python execution, compile, help invocation,
smoke run, training, checkpoint/model loading, external evaluation, Kaggle
command, commit, or push.

