# Stage190-A/B/C selected-checkpoint gradient-conflict diagnostic

Stage190 asks whether the frozen Stage189 margin objective has a local gradient conflict with classification CE, gold-SUPPORT CE, SUPPORT decision margins, shared representation/backbone parameters, and the Stage182-B compatible-FN and matched-control frame paths.

This is gradient-enabled evaluation at six selected checkpoints. It is not training-trajectory proof, causal proof of the 20-epoch outcome, generalization evidence, external evaluation, or permission to advance an architecture. Training and diagnostic Git identities are separate and need not match.

## Stage190-A contract

The builder accepts explicit repo-root, Stage189-A, Stage189-D analysis, Stage182-B, Stage185-A, current diagnostic commit, and output paths. It validates the frozen Stage189 identities, all six completed runs and distinct selected-checkpoint hashes, exact `1440`-row Stage189-C exports with `605/716/119` topology, fixed split seed `174`, seeds `174/175/176`, baseline sidecar isolation, intervention weight `0.05` and exact sidecar identity, absence of pre-split artifacts and external paths, and current diagnostic/helper hashes. It emits six fail-closed manifests plus the prescribed report and CSV ledgers.

## Exact cohorts and objectives

All joins are one-to-one row-ID joins in deterministic source order; no cohort is sampled or truncated. Cohorts are train-compatible ELIGIBLE `605`, INELIGIBLE `716`, UNRESOLVED `119`, clean dev `720`, its frozen gold-SUPPORT subset, Stage182-B compatible FN `13`, incompatible FP `1`, matched controls `14`, and clean model failures `14`.

Every objective is an exact dataset mean produced by accumulating batch summed gradients and dividing once by the exact row count. The source is `mean(relu(0-output["frame_logit"]))` on 605 ELIGIBLE rows. Targets are CE on eligible, CE on all clean dev, CE on gold-SUPPORT clean dev, negative SUPPORT-minus-NOT_ENTITLED margin, negative SUPPORT-minus-max(REFUTE, NOT_ENTITLED) margin, and negative mean direct frame logit on compatible FN, matched controls, INELIGIBLE, and UNRESOLVED. Classification uses only `output["logits"]`; margin/frame objectives use only `output["frame_logit"]`; `loss_logits` is forbidden.

## Parameter and gradient contract

Module-owned groups use parameter object IDs: `frame_head` is the exact `frame_gate.frame_classifier` module, `decision_head` is the exact `decision_head` module, `backbone` is `mamba`, `router_and_epistemic_heads` contains the remaining task heads/gates/fusion/router modules including the remainder of `frame_gate`, and `other_trainable` is the exact residual set. Groups must be disjoint and exhaustive, with every name/shape/numel and a parameter-order SHA256 recorded.

For each checkpoint, the source gradient is copied to CPU float32 in deterministic parameter order. Targets are evaluated sequentially, gradients cleared between objectives, and only float64 norm/dot sufficient statistics persist. Trainable parameter state and preserved buffers are hashed before and after and must be unchanged. No optimizer, scheduler, update, checkpoint selection, tuning, AMP by default, or raw-gradient artifact is allowed.

The hypothetical margin-descent direction is `delta_theta = -g_margin`. Therefore `projected_target_change = -dot(g_target, g_margin)`: positive locally worsens the target loss and negative locally improves it. For negative decision-margin objectives, positive reduces the SUPPORT margin and negative increases it.

## Stage190-C precommitment

Valid six-report analysis uses exactly one of the precommitted decisions: regression-aligned coupling conflict, shared conflict not seed-aligned, head-local/nonconflicting, or inconclusive/seed-unstable. Any identity, topology, state, grouping, objective, or finite-gradient failure yields `STAGE190C_GRADIENT_DIAGNOSTIC_BLOCKED`. No Stage190 result is a positive model-advancement decision.

Conditional next design class only: aligned shared conflict permits consideration of stop-gradient, frame-head-local margin, or a detached auxiliary path; head-local/nonconflicting evidence redirects investigation to checkpoint selection and optimization trajectory; inconclusive evidence requires trajectory checkpoints or per-epoch gradient snapshots before architecture change. Stage191 is not implemented.
