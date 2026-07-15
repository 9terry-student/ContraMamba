# Stage186-A fixed compatible-positive margin spec closure

## Decision

`STAGE186A_FIXED_NO_SWEEP_COMPATIBLE_POSITIVE_MARGIN_SPEC_READY`

Authorized next: `STAGE187_COMPATIBLE_POSITIVE_MARGIN_DEFAULT_OFF_IMPLEMENTATION`.

Stage186-A fixed the trainer-local specification; it did not authorize training or a checkpoint-selection change.

## Authoritative identity

- Dataset: `data/controlled_v5_v3_without_time_swap.jsonl`
- Dataset SHA-256: `f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640`
- Stage185 sidecar: `reports/stage185a_controlled_train_integrity_sidecar_20260715_141914/stage185a_controlled_train_integrity_sidecar.jsonl`
- Semantic sidecar SHA-256: `5bc03caa2a29f9b9176ab4eb0201db57ebad516352797546db1a18e6ec3373fc`

The frozen eligible cohort is 605 train-compatible rows: 121 pairs, exactly five rows per pair, across five families with 121 rows per family.

## Fixed objective

The score is the native pre-sigmoid `output["frame_logit"]`. The loss is the eligible-row mean of `relu(0.0 - frame_logit)`. The only enabled weight is `0.05`; the default is `0.0`. Margin and weight sweeps, schedules, adaptive rescaling, and custom boundary derivatives are outside the authorization.

Native frame BCE is delegated through model-forward auxiliary-loss wiring and was cross-checked against the Stage183-A static audit. Final classification CE remains sourced from `output["logits"]`, and checkpoint selection remains clean-dev `final_macro_f1`.

## Safety boundary

Only a default-off Stage187 trainer implementation is authorized. Runtime validation, loss-enabled training, checkpoint-selection changes, dataset/sidecar rewriting, and scientific claims remain unapproved.