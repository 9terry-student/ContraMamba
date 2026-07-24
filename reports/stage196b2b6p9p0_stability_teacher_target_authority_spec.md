# Stage196-B2-B6P9-P0 Separate Stability Teacher/Target Authority Design

## Scope

Stage196-B2-B6P9-P0 is a design and static/source-backed analysis stage.  It is
not training and not a score-producing intervention run.

This stage does not implement or activate a stability loss.  It audits teacher
and target authority required before either independent intervention family can
be implemented.

The only future selectable variants remain:

```text
baseline
direction_consistency_only
candidate_order_consistency_only
```

There is no combined first-stage variant.  Direction and candidate-order are
evaluated independently and are not assumed to share the same teacher.

## Required CLI Authority

The analyzer is:

```text
scripts/analyze_stage196b2b6p9p0_stability_teacher_target_authority.py
```

It requires explicit paths for:

```text
--stage196b2b6p5-analysis-json
--stage196b2b6p7-analysis-json
--stage196b2b6p6-contract-csv
--stage196b2b6p7-contract-csv
--stage196b2b6p8-analysis-json
--stage196b2b6p8-gradient-connectivity-csv
--repo-root
--output-dir
```

The original P4 analysis path is optional:

```text
--stage196b2b6p4-analysis-json
```

P8 authority must come from exactly:

```text
reports/stage196b2b6p8_full_trainable_path_replay_20260723_203414
```

The analyzer requires:

```text
decision = STAGE196B2B6P8_FULL_TRAINABLE_PATH_REPLAY_COMPLETE
blocking_reasons = []
all P8 contracts PASS
direction_connectivity_passed = true
candidate_order_connectivity_passed = true
```

No earlier blocked P8 output is valid authority.

## P4 Authority Modes

The analyzer supports exactly two P4 authority modes:

```text
ORIGINAL_P4_ANALYSIS
DOWNSTREAM_ATTESTED_P4_MINIMAL_CLOSURE
```

`ORIGINAL_P4_ANALYSIS` is used only when an explicitly supplied P4 analysis file
exists and passes the existing exact authority checks.  The original-file
semantics are unchanged: the decision must be
`STAGE196B2B6P4_ACTION_RESPONSE_TOPOLOGY_UNSTABLE` and `blocking_reasons` must
be exactly `[]`.

`DOWNSTREAM_ATTESTED_P4_MINIMAL_CLOSURE` is used only when the original P4
analysis path is omitted or unavailable.  This mode does not fabricate,
reconstruct, or recover an original-looking P4 analysis JSON.

### Downstream-Attested P4 Checks

The P7 analysis must close the supplied P6 contract hash through:

```text
upstream.p6_artifacts["stage196b2b6p6_contract.csv"]
```

The analyzer computes the actual SHA256 of `--stage196b2b6p6-contract-csv` and
requires exact equality.  A P6 contract whose hash is not closed by P7 is not
trusted.

The P6 contract must contain exactly one `p4_decision_closure` row.  Its
`passed` value must parse as true, and both its `required` and `observed` JSON
payloads must equal exactly:

```json
{
  "blocking_reasons": [],
  "decision": "STAGE196B2B6P4_ACTION_RESPONSE_TOPOLOGY_UNSTABLE",
  "recommended_next_stage": "STAGE196B2B6P5_TRAINING_SIDE_RESPONSE_STABILITY_INTERVENTION_DESIGN"
}
```

The P6 contract must contain exactly one `p4_zero_failed_contracts` row with
strict booleans:

```text
passed = true
required = true
observed = true
```

The P7 analysis must record:

```text
upstream.p4_decision = STAGE196B2B6P4_ACTION_RESPONSE_TOPOLOGY_UNSTABLE
```

The P7 contract must contain exactly one `p4_decision_and_zero_blockers` row.
Its `passed` value must parse as true, and both its `required` and `observed`
JSON payloads must equal exactly:

```json
{
  "blocking_reasons": [],
  "decision": "STAGE196B2B6P4_ACTION_RESPONSE_TOPOLOGY_UNSTABLE"
}
```

### Downstream-Attested Scope

The downstream-attested mode establishes only:

```text
P4 decision identity
zero P4 blockers
P4 recommended next-stage identity
zero failed P4 contracts
```

It does not reconstruct or authorize:

```text
original P4 numerical tables
original P4 row-level data
original P4 source-file hashes
original P4 output-directory identity
original P4 creation timestamp
byte-identical original P4 content
```

A valid downstream-attested minimal closure satisfies the existing
`upstream_p4_authority` gate.  It must not create a blocker merely because the
original P4 artifact is absent.  If downstream attestation fails, the decision is
`STAGE196B2B6P9P0_BLOCKED_UPSTREAM_AUTHORITY`.

## Teacher Candidates

The exact audited candidate set is:

```text
CURRENT_NATIVE_STOP_GRAD
FRAME_LOCAL_ONLY_DONOR_STOP_GRAD
PREVIOUS_STEP_FROZEN_SNAPSHOT
PREVIOUS_EPOCH_FROZEN_SNAPSHOT
EMA_STUDENT_TEACHER
FIXED_RECONSTRUCTED_CHECKPOINT
```

No candidate is automatically approved.  Every candidate is recorded with
availability, required state, second-copy/checkpoint/historical dependencies,
seed and run portability, causal interpretability, self-confirmation risk,
target-drift risk, cross-seed coordinate-locking risk, exact-tie policy,
stop-gradient policy, direction status, order status, and blocking reason.

## Eligibility Criteria

A teacher is eligible only if all applicable criteria pass:

```text
target available during training without future epochs
target stop-gradient
exact sign/order ties ignored
candidate mask lexical order never used as semantic order
no clean-dev label target
no recovery/harm membership target
no seed-specific absolute score-coordinate target
mechanism independently disableable
no unfrozen Mamba requirement
no permanent reconstructed seed183 checkpoint requirement
target reproducible for every intended training seed
teacher does not silently become a second opaque selector
direction and candidate-order approval evaluated independently
teacher state lifecycle exactly explainable
baseline byte/path-equivalent when disabled
```

## Direction Target Design

This stage defines the future target abstractly only:

```text
student_delta[c,k] =
student_counterfactual[c,k] - student_native[k]

teacher_sign[c,k] in {-1,+1}
```

Exact teacher zero is ignored.  The teacher sign is stop-gradient.  P9-P0 does
not prescribe a coefficient or margin.

## Candidate-Order Target Design

This stage defines the future target abstractly only:

```text
student_pair_gap[a,b,k] =
student_counterfactual[a,k] - student_counterfactual[b,k]

teacher_pair_sign[a,b,k] in {-1,+1}
```

Exact teacher ties are ignored.  Candidate-mask lexical order is never used as
the desired order.  P9-P0 does not prescribe a coefficient or margin.

## P8 Gradient Evidence

P8 is used only to establish graph availability.  The analyzer preserves the
distinction among:

```text
CONNECTED_NONZERO
CONNECTED_ZERO_AT_OBSERVED_BATCH
DISCONNECTED
```

Candidate-order coordinates were graph-connected but zero on the observed P8
batch.  This does not invalidate the family, but any future implementation must
expose:

```text
active non-tie teacher target count
violating student target count
nonzero loss term count
nonzero gradient target count
```

Graph connectivity is not evidence that a useful loss signal will occur
automatically.

## Candidate Restrictions

`CURRENT_NATIVE_STOP_GRAD` is invalid if its sign/order is algebraically
identical to the live student quantity after stop-gradient.  It is not an
independent stability teacher.

`FRAME_LOCAL_ONLY_DONOR_STOP_GRAD` is meaningful for FrameGate isolation but not
globally superior by default.  It is blocked if it would force all branches to
imitate a frame-local-only coordinate system or reintroduce harms observed in
earlier stages.

`PREVIOUS_STEP_FROZEN_SNAPSHOT` updates at optimizer-step cadence, has high
temporal locality, requires additional state/copy cost, and risks near-identity
targets.

`PREVIOUS_EPOCH_FROZEN_SNAPSHOT` updates only at epoch boundaries, has stronger
temporal separation, requires checkpoint or state-copy support, and has coarser
target drift.

`EMA_STUDENT_TEACHER` is not approved merely because P7 named it as a conceptual
preference.  Missing source authority must be recorded for initialization,
update timing, decay ownership, buffer handling, dropout/eval policy,
checkpoint serialization, resume behavior, seed determinism, teacher warm-up,
exact tie handling, teacher drift observability, and baseline default-off
closure.

`FIXED_RECONSTRUCTED_CHECKPOINT` remains diagnostic-only unless portability is
proven.  The reconstructed seed183 joint and frame-local-only checkpoints were
valid for P8 replay verification, but they are seed183-specific, reconstructed
after runtime loss, not available for every future seed, and risk locking the
intervention to one historical coordinate system.

## Outputs

The analyzer writes exactly these nine outputs:

```text
stage196b2b6p9p0_analysis.json
stage196b2b6p9p0_report.md
stage196b2b6p9p0_teacher_candidate_audit.csv
stage196b2b6p9p0_direction_target_design.csv
stage196b2b6p9p0_order_target_design.csv
stage196b2b6p9p0_state_lifecycle_audit.csv
stage196b2b6p9p0_portability_audit.csv
stage196b2b6p9p0_decision_gate.csv
stage196b2b6p9p0_contract.csv
```

## Decision Hierarchy

The analyzer emits exactly one decision from:

```text
STAGE196B2B6P9P0_BLOCKED_UPSTREAM_AUTHORITY
STAGE196B2B6P9P0_NO_JUSTIFIED_TEACHER
STAGE196B2B6P9P0_DIRECTION_TEACHER_ONLY_READY
STAGE196B2B6P9P0_ORDER_TEACHER_ONLY_READY
STAGE196B2B6P9P0_SEPARATE_TEACHERS_READY
STAGE196B2B6P9P0_SHARED_TEACHER_READY
```

`STAGE196B2B6P9P0_SHARED_TEACHER_READY` requires an explicit argument that
sharing the teacher does not couple the two intervention mechanisms.

Recommended next-stage values are restricted to:

```text
STAGE196B2B6P9P0_REPAIR_UPSTREAM_AUTHORITY
STAGE196B2B6P9P1_TEACHER_STATE_OBSERVABILITY_DESIGN
STAGE196B2B6P9D_DIRECTION_ONLY_INTERVENTION_IMPLEMENTATION
STAGE196B2B6P9O_ORDER_ONLY_INTERVENTION_IMPLEMENTATION
STAGE196B2B6P9DO_SEPARATE_INTERVENTION_IMPLEMENTATIONS
```

The analyzer must not recommend a combined-loss run.

## Contracts

The contract output includes at minimum:

```text
upstream_p4_authority
p4_authority_mode_valid
p4_original_or_attested_authority_available
p4_downstream_p6_contract_hash_closure
p4_downstream_p6_decision_closure
p4_downstream_p6_zero_failed_contracts
p4_downstream_p7_analysis_concurrence
p4_downstream_p7_contract_concurrence
p4_downstream_attestation_scope_restricted
p4_original_artifact_not_fabricated
upstream_p5_authority
upstream_p7_authority
upstream_p8_final_authority
p8_zero_failed_contracts
p8_direction_graph_available
p8_order_graph_available
exact_teacher_candidate_set
direction_order_independent_evaluation
no_combined_first_variant
no_clean_dev_label_targeting
no_recovery_harm_label_targeting
no_lexical_candidate_order
exact_ties_ignored
teacher_stop_gradient_required
fixed_seed183_checkpoint_not_assumed_portable
ema_not_assumed_preexisting
baseline_default_off_requirement
zero_model_or_trainer_modification
exact_nine_file_closure
```

## Prohibited Actions

This stage may inspect source and upstream reports.  It must not load a model,
load a checkpoint, run a forward pass, train, evaluate predictions, choose
coefficients, add a loss, change optimizer behavior, or change checkpoint
selection.
