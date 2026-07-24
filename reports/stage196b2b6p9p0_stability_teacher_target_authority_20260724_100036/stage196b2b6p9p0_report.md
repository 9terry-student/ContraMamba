# Stage196-B2-B6P9-P0 Separate Stability Teacher/Target Authority Design

## Decision

decision = `STAGE196B2B6P9P0_NO_JUSTIFIED_TEACHER`

recommended_next_stage = `STAGE196B2B6P9P1_TEACHER_STATE_OBSERVABILITY_DESIGN`

This stage does not authorize or activate a stability loss.  P8 is used only
for graph availability: direction has graph availability, and candidate-order
is graph-connected while zero on the observed P8 batch.

## P4 Authority

P4 authority mode = `DOWNSTREAM_ATTESTED_P4_MINIMAL_CLOSURE`

Original P4 artifact available = `False`

The downstream-attested mode establishes only P4 decision identity, zero P4
blockers, P4 recommended next-stage identity, and zero failed P4 contracts. It
does not reconstruct or authorize the original P4 numerical tables, row-level
data, source-file hashes, output-directory identity, creation timestamp, or
byte-identical original content.

## P5 Authority

P5 authority mode = `DOWNSTREAM_ATTESTED_P5_MINIMAL_CLOSURE`

Original P5 artifact available = `False`

The downstream-attested P5 mode establishes only P5 decision identity, zero P5
blockers, P5 recommended-next-stage identity, P5 source-feasibility facts
required by P6, teacher unavailability without additional instrumentation, and
P7 concurrence that no teacher was authorized. It does not reconstruct the
original P5 analysis JSON, original numerical tables, output directory,
timestamps, source hashes, complete prose, or byte-identical original content.

## Teacher Authority

No audited teacher candidate is justified for direction or candidate-order in
the current training source.  `CURRENT_NATIVE_STOP_GRAD` is portable but
algebraically identical to the live student quantity after stop-gradient.
`FRAME_LOCAL_ONLY_DONOR_STOP_GRAD` is mechanistically meaningful for FrameGate
isolation but risks forcing branches to imitate a frame-local-only coordinate
system.  `PREVIOUS_STEP_FROZEN_SNAPSHOT`, `PREVIOUS_EPOCH_FROZEN_SNAPSHOT`, and
`EMA_STUDENT_TEACHER` require new state lifecycle authority.  The fixed
reconstructed checkpoints remain diagnostic-only because seed183-specific,
post-reconstruction artifacts are not portable training infrastructure.

## Target Designs

Direction target, future only:

```text
student_delta[c,k] = student_counterfactual[c,k] - student_native[k]
teacher_sign[c,k] in {-1,+1}
```

Exact teacher zero is ignored.

Candidate-order target, future only:

```text
student_pair_gap[a,b,k] = student_counterfactual[a,k] - student_counterfactual[b,k]
teacher_pair_sign[a,b,k] in {-1,+1}
```

Exact teacher pair ties are ignored.  Lexical candidate-mask order is never
used as semantic order.

## Future Variants

The only independently selectable future variants are:

```text
baseline
direction_consistency_only
candidate_order_consistency_only
```

There is no combined first-stage variant.

## Required Future Observability

Any future implementation must expose active non-tie teacher target count,
violating student target count, nonzero loss term count, and nonzero gradient
target count.  Graph connectivity alone is not evidence that a useful loss
signal will occur automatically.
