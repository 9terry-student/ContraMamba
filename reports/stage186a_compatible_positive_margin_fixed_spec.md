# Stage186-A compatible-positive margin fixed specification

## Scope and authorization

This stage is a static specification audit only. Stage185-A materialized 605 integrity-filtered train-compatible positives and authorized `STAGE186_COMPATIBLE_POSITIVE_MARGIN_FIXED_SPEC_AUDIT`. No trainer, model, loss, dataset, sidecar, checkpoint, or prior report is modified. Loss implementation and training remain unauthorized.

## Authoritative identity and cohort

The only valid Stage185 input is `reports/stage185a_controlled_train_integrity_sidecar_20260715_141914`. The source dataset SHA-256 is `f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640`; the semantic sidecar SHA-256 is `5bc03caa2a29f9b9176ab4eb0201db57ebad516352797546db1a18e6ec3373fc`. A consumer must exact-join by unique `row_id`, reject missing or extra IDs, and fail closed before training on either SHA mismatch.

The eligible cohort is exactly 605 rows from 121 train pairs and five families. Each eligible pair contributes five rows and each eligible family contributes 121 rows. Eligible rows are train-only, frame-compatible, integrity `ELIGIBLE`, and pass time/source gates. The five observed family names are reported, but family membership never creates eligibility. Only `eligible_for_positive_margin == true`, after all identity gates, is authoritative.

Only 121 of 240 train pairs contribute, so pair coverage is about 50.4%. Family balance is exact within the cohort, but this must not be interpreted as population-wide clean coverage. If later topology differs, row-mean and equal-pair-mean equivalence must be re-audited.

## Fixed objective

The target is the native pre-sigmoid scalar `output["frame_logit"]`, never frame probability or final-classifier logits.

For the authoritative eligible mask `M`:

```text
margin_logit = 0.0
L_margin = mean_{i in M} max(0, margin_logit - frame_logit_i)
         = mean_{i in M} relu(-frame_logit_i)

compatible_positive_margin_weight = 0.05
L_total = L_existing + 0.05 * L_margin
```

The zero logit is the native sigmoid decision boundary. It corrects only eligible compatible positives on the incompatible side and does not introduce an arbitrary confidence target. Neither the Stage182-B median gap nor a dev metric is fitted into the margin.

The weight is fixed at 0.05 with native frame BCE retaining implicit weight 1.0. At `z=0`, positive BCE has gradient magnitude 0.5; the active weighted hinge contributes 0.05, or 10% of that local BCE magnitude. This is a bounded auxiliary correction, not a replacement for BCE. There is no sweep, schedule, warmup, family/pair weighting, inverse-frequency scaling, learned weight, or eligible-rate rescaling.

## Normalization and empty batches

The hinge is averaged over eligible rows only:

```text
eligible_losses = relu(-frame_logit[eligible_mask])
L_margin = eligible_losses.sum() / eligible_count
```

It is not divided by batch size, 1,440 train-compatible rows, or 3,600 source rows, and it is not re-averaged by family or pair. When `eligible_count == 0`, the required semantic result is graph-compatible scalar zero, `frame_logit.sum() * 0.0`. Division by zero, NaN, batch skipping, and changed optimizer-step semantics are forbidden.

## Existing objective integration

Static trainer inspection must identify native `frame_logit`, `binary_cross_entropy_with_logits` frame supervision, existing total-loss assembly, final CE from `output["logits"]`, and clean-dev `final_macro_f1` checkpoint selection. The default-off future term is appended to the existing total objective; it does not change existing auxiliary weights, label construction, optimizer, scheduler, model architecture, final CE, or checkpoint selection.

Stage175 is nonredundant because it targets a final-classifier SUPPORT anchor relative to a detached reference. Stage177 is nonredundant because it targets within-pair compatible-over-incompatible ordering. Stage186 specifies an absolute native frame-head boundary for individually eligible positive rows.

## Default-off Stage187 contract

Recommended unambiguous flags are `--compatible-positive-margin-weight` and `--compatible-positive-margin-logit`. Implementation defaults must be weight `0.0` and logit `0.0`; the fixed intervention setting is weight `0.05`, logit `0.0`. Sidecar path and expected semantic SHA are separate required activation inputs, with semantic fields `controlled_integrity_sidecar_path` and `expected_integrity_sidecar_semantic_sha256`.

Activation with nonzero weight must fail closed unless the dataset SHA, semantic sidecar SHA, exact row-ID join, split, compatible label, integrity status, time/source gates, and authoritative eligibility field all pass. Generic margin aliases and reuse of Stage175/177 flags are forbidden.

## Stage186 gate

Success decision: `STAGE186A_FIXED_NO_SWEEP_COMPATIBLE_POSITIVE_MARGIN_SPEC_READY`.

Authorized next: `STAGE187_COMPATIBLE_POSITIVE_MARGIN_DEFAULT_OFF_IMPLEMENTATION`, limited to default-off trainer implementation, exact sidecar/SHA gates, the fixed masked hinge, and logging/export fields. Stage187 still may not train or evaluate checkpoints.

Blocked decision: `STAGE186A_COMPATIBLE_POSITIVE_MARGIN_SPEC_BLOCKED` when any authoritative identity, cohort topology, directional evidence, trainer integration, nonredundancy, or checkpoint-invariance requirement fails.

## Safety

No Torch/model import, checkpoint load, forward, loss implementation, training, smoke run, external evaluation, calibration, threshold fitting, margin/weight sweep, multi-seed analysis, annotation, LLM labeling, git command, or Kaggle command is authorized.
