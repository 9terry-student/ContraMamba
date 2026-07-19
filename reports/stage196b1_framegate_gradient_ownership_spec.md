# Stage196-B1-P0 FrameGate downstream-gradient ownership specification

## Decision context

Stage196-A selected STAGE196A_RECURRENT_LOCAL_CHANNEL_FAILURE: all 127
persistent stable SUPPORT negatives contained FrameGate failure, comprising 72
MULTI_LOCAL_CHANNEL_FAILURE and 55 FRAME_ONLY_FAILURE cases. Stage196-B0 found
independent raw frame_compatible_label supervision, direct unmasked BCE at
native weight 1.0, optimizer inclusion, and no clear target contamination or
channel-separation defect. It selected
STAGE196B0_FRAME_REPRESENTATION_FAILURE, while leaving intrinsic representation
insufficiency unresolved from destructive downstream-gradient interference.

Stage196-B1-P0 asks one question under the frozen-Mamba training envelope that
produced the Stage196-A failure population:

> Under the frozen-Mamba training envelope, does allowing non-frame losses to
> backpropagate through FrameGate outputs cause the recurrent FrameGate
> false-negative failure?

This stage introduces no contrastive loss, new loss, target change, threshold
change, calibration, architecture replacement, or optimizer change.

## CLI and runtime contract

The trainer exposes exactly one new option:

~~~text
--frame-downstream-gradient-mode {joint,frame_local_only}
~~~

The default is joint.

- joint: no ownership hooks are installed and the original output dictionary
  is passed unchanged to all existing losses. Existing tensor flow and numerical
  training behavior are preserved.
- frame_local_only: FrameGate's original scalar tensors remain authoritative
  for direct frame BCE, metrics, scalar export, trajectory export, diagnostic
  reporting, frame_logit, and frame_prob. Detached aliases are supplied to every
  non-frame consumer.

frame_local_only is defined for --architecture v6b_minimal; selecting it with
another architecture is rejected.

At startup the trainer prints one contract line with the selected mode, whether
direct frame loss is active, whether non-frame FrameGate-output gradients are
blocked, the encoder freeze state, whether the shared encoder is trainable,
whether it is fully gradient-isolated, and the isolation source.

## Implementation and ownership boundary

In frame_local_only, a trainer-owned FrameGate forward hook captures the
authoritative output dictionary and returns tensor-by-tensor detach() aliases
to the remainder of the v6B forward. This includes FrameGate scalar outputs and
FrameGate-owned representations, ensuring that non-frame diagnostic heads do
not regain a path into FrameGate parameters through a representation tensor.

After the model forward, a model hook restores the captured authoritative
frame_logit and frame_prob, creates frame_logit_downstream and
frame_prob_downstream using detach(), and recomputes the native direct frame BCE
from the original frame_logit. The returned authoritative scalar tensors
therefore continue to drive direct frame BCE, metrics, and exports.

Trainer-side non-frame objectives receive a shallow output view in which only
frame_logit and frame_prob point to their downstream aliases. Other predicate,
sufficiency, polarity, backbone, classifier, router, and selector tensors are
not detached by this helper.

The authorized claim is:

> Direct non-frame gradient paths through FrameGate outputs into FrameGate-owned
> parameters are blocked, while direct frame BCE remains active.

FrameGate is not frozen. The shared Mamba backbone is frozen in both arms by the
inherited runtime configuration (`--freeze-encoder true`), and frozen encoder
states retain the same cached role across arms. The intervention neither freezes
nor unfreezes any parameter. No optimizer or optimizer group changes are made.

## Complete FrameGate scalar consumer audit

| Category | Discovered consumer | Ownership |
| --- | --- | --- |
| 1. Frame-local direct supervision | Native BCE-with-logits against frame_compatible_labels; v5.controlled_losses frame BCE | Authoritative non-detached frame_logit |
| 2. Frame-local metric/export | compute_metrics; Stage115/Stage188 scalar rows; Stage191 trajectory rows; prediction and diagnostic reports | Authoritative frame_logit and frame_prob; reporting code later detaches for CPU serialization only |
| 3. PredicateCoverage computation | PredicateCoverageHead frame_prob input | Detached downstream FrameGate output in frame_local_only |
| 4. SufficiencyGate computation | SufficiencyGate frame_prob input | Detached downstream FrameGate output in frame_local_only |
| 5. Entitlement product | FinalEntitlementDecisionHead explicit product, including the frame factor | Detached downstream FrameGate output in frame_local_only |
| 6. Support/refute energy or final logits | Final decision logits consume frame through entitlement and downstream channel representations | Detached downstream FrameGate outputs in frame_local_only |
| 7. Intervention loss | v5.intervention_objective frame ranking and preservation terms; intervention_pairwise_losses frame fields and final logits | Trainer-side detached scalar view; final logits were already composed from detached FrameGate outputs |
| 8. Auxiliary pairwise or margin loss | Stage174-C clean pairwise frame_prob; Stage177-C clean frame pairwise frame_logit; Stage187 compatible-positive margin frame_logit | Trainer-side detached scalar view |
| 9. Selector/router computation | Gate-mask and decision-signal diagnostics read frame_prob only in no-grad/evaluation or detached-CPU reporting paths; v6B has no learned router consuming the scalar in this contract | No training gradient; authoritative value retained for reporting |
| 10. Other non-frame consumer | Boundary and frame-violation diagnostic heads, temporal diagnostic paths when their configured input is not already detached, polarity composition through FrameGate representations | Detached FrameGate-owned representations in frame_local_only |

The Stage22 pair-contrastive frame objective supervises
frame_violation_logit, not native frame_logit; because its diagnostic head
receives detached FrameGate representations in frame_local_only, it cannot send
gradient into FrameGate-owned parameters. Temporal adapter/channel paths that
already detach retain their existing detach behavior.

## Loss ownership

In frame_local_only, direct frame BCE is the only loss allowed to update
FrameGate-owned parameters. Final-label, predicate, sufficiency, polarity,
intervention, compatible-positive margin, clean pairwise, support-anchor,
temporal, diagnostic-head, selector/router, and downstream-composition
objectives receive detached FrameGate outputs wherever they could otherwise
reach FrameGate-owned parameters.

This does not detach or freeze unrelated predicate, sufficiency, polarity, or
classifier tensors. Those head-level components remain trainable under their
existing objectives. The Mamba backbone is already frozen independently by the
common runtime configuration and receives no gradients in either arm.

## Forward-value invariance

detach() returns a tensor sharing the same stored values and does not clone,
round, normalize, recalibrate, or recompute them. The intervention does not
change multiplication order, epsilon, sigmoid placement, thresholds, logits,
router/selector rules, or architecture dimensions. The original
frame_prob?already computed by the FrameGate?is passed downstream as a detached
alias rather than applying sigmoid again.

Consequently, before optimization updates and for identical input and
parameters:

~~~text
frame_logit_joint == frame_logit_frame_local_only
frame_prob_joint == frame_prob_frame_local_only
entitlement_joint == entitlement_frame_local_only
final_logits_joint == final_logits_frame_local_only
~~~

The difference is autograd ownership only.

## Training-report provenance

Configuration, resolved-runtime provenance, and the per-run training report
record runtime values for:

~~~json
{
  "frame_downstream_gradient_mode": "joint|frame_local_only",
  "frame_direct_loss_active": true,
  "frame_direct_loss_weight": 1.0,
  "frame_downstream_forward_value_changed": false,
  "framegate_nonframe_output_gradient_blocked": false,
  "freeze_encoder": true,
  "freeze_a_log": true,
  "shared_encoder_trainable": false,
  "shared_encoder_gradient_fully_isolated": true,
  "shared_encoder_isolation_source": "frozen_runtime_configuration",
  "framegate_gradient_ownership_intervention_changed_encoder_freeze_state": false
}
~~~

framegate_nonframe_output_gradient_blocked is true only for
frame_local_only. It is false for joint.

## Frozen-encoder scope and limitation

The shared Mamba claim/evidence encoder is non-trainable and fully
gradient-isolated in both arms because `freeze_encoder=true`, not because of the
FrameGate ownership intervention. The cached encoder states have the same role
in both arms. `shared_encoder_isolation_source` is therefore
`frozen_runtime_configuration`, while
`framegate_gradient_ownership_intervention_changed_encoder_freeze_state` is
false. Stage196-B1 does not evaluate shared-backbone gradient interference.

A negative Stage196-B1 result does not rule out shared-representation
interference in a separately designed unfrozen-backbone regime. This
specification does not automatically authorize that experiment.

## Stage196-B1 trajectory observability contract

Stage196-B1 uses the explicit default-off flag
`--stage196b1-framegate-gradient-ownership-observability`. It reuses the
Stage191 per-epoch clean-dev trajectory implementation and filenames, but has a
distinct mode, `stage196b1_framegate_gradient_ownership`, and a distinct frozen
seed tuple `(183, 184, 185)`. The Stage191, Stage193, and Stage195 seed tuples and
mode behavior remain unchanged.

The Stage196-B1 mode saves zero state capsules, performs no extra forward pass,
and does not enable Stage195 parameter SWA. Its observability contract records
the actual FrameGate downstream-gradient mode and blocking status. Its causal
arm is derived only from `joint` versus `frame_local_only`; compatible-positive
margin remains off and both Stage185 sidecar options remain absent in both arms.
The shared encoder remains frozen and gradient-isolated by the runtime
configuration in both arms.

The Stage196-B1 trajectory contract includes actual runtime values for:

~~~json
{
  "observability_mode": "stage196b1_framegate_gradient_ownership",
  "authorized_training_seeds": [183, 184, 185],
  "training_seed_authorized": true,
  "arm": "baseline|intervention",
  "frame_downstream_gradient_mode": "joint|frame_local_only",
  "framegate_nonframe_output_gradient_blocked": false,
  "freeze_encoder": true,
  "freeze_a_log": true,
  "shared_encoder_trainable": false,
  "shared_encoder_gradient_fully_isolated": true,
  "shared_encoder_isolation_source": "frozen_runtime_configuration",
  "framegate_gradient_ownership_intervention_changed_encoder_freeze_state": false,
  "state_capsule_saving_enabled": false,
  "expected_state_capsules": 0,
  "compatible_positive_margin_enabled": false,
  "sidecar_accessed": false,
  "parameter_swa_enabled": false,
  "training_semantics_changed_by_observability": false,
  "extra_forward_pass_performed_by_observability": false,
  "enabled_flags": {
    "stage191_trajectory_replay_observability": false,
    "stage191_save_trajectory_state_capsules": false,
    "stage193_tail3_fresh_seed_observability": false,
    "stage195_tail3_parameter_swa_causal_test": false,
    "stage196b1_framegate_gradient_ownership_observability": true
  }
}
~~~

For `frame_local_only`, the mode-specific fields instead include:

~~~json
{
  "frame_downstream_gradient_mode": "frame_local_only",
  "framegate_nonframe_output_gradient_blocked": true
}
~~~

## Future Stage196-B1 controlled experiment

Freeze the following across arms:

- Main data: data/controlled_v5_v3_without_time_swap.jsonl
- Exclude time_swap from main training
- Backbone: mamba
- Model: state-spaces/mamba-130m-hf
- Architecture: v6b_minimal
- Device: cuda
- Freeze encoder: true
- Freeze A_log: true
- Split seed: 174
- Epochs: 20
- Trajectory epochs: 18,19,20
- No external training data
- No external threshold tuning
- No SWA
- No contrastive loss
- No new margin loss

For each fresh seed 183, 184, and 185, run two arms with all arguments
otherwise identical and enable the dedicated Stage196-B1 observability flag:

~~~text
--stage196b1-framegate-gradient-ownership-observability
baseline:     --frame-downstream-gradient-mode joint
intervention: --frame-downstream-gradient-mode frame_local_only
~~~

The Stage196-B1 manifest freezes the exact paired run order.

## Preregistered causal interpretation

Evidence for direct FrameGate-output interference requires the
frame-local-only intervention
to reproducibly reduce persistent stable SUPPORT negatives, increase frame
probability on recurrent persistent SUPPORT cases, preserve stable-correct
SUPPORT controls, avoid disproportionate false-entitlement growth, and preserve
polarity safety. If it does so reproducibly, the authorized claim is: under a
frozen Mamba encoder, direct non-frame gradients through FrameGate outputs
interfered with FrameGate-owned trainable parameters.

Evidence against direct FrameGate-output gradient interference is no
reproducible rescue or comparable harm. The authorized claim is: under a frozen
Mamba encoder, direct FrameGate-output gradient interference was not supported
as the main cause. That outcome does not automatically authorize contrastive
loss or an unfrozen-backbone experiment.

Mixed evidence means seed directions conflict or SUPPORT rescue is offset by
broad entitlement harm. Mixed evidence does not authorize an automatic
contrastive intervention.

Stage196-B1 does not authorize claims that shared Mamba representation
interference or end-to-end gradient isolation was tested, that
unfrozen-backbone behavior is known, or that any broader architecture is
superior.

No final numeric decision thresholds are encoded in the trainer.
