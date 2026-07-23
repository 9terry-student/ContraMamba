# Stage196-B2-B6P6 Minimal Gradient-Path Instrumentation Specification

## Scope and upstream authority

P5 closed successfully with `STAGE196B2B6P5_GRADIENT_PATH_INSTRUMENTATION_REQUIRED`; it is a negative feasibility result, not a blocked result. P5 requires gradient-path instrumentation before any stability loss. P4 remains authoritative as `STAGE196B2B6P4_ACTION_RESPONSE_TOPOLOGY_UNSTABLE`, and P2 remains the exact candidate-action composer authority.

P6 is instrumentation only. It adds no intervention objective, training coefficient, combined intervention, optimizer objective, selection rule, prediction behavior, checkpoint field, or candidate-action semantics. Default trainer execution does not call the P6 geometry function, so it incurs no recomposition, backward call, optimizer step, logging requirement, or schema change.

## Source-feasibility evidence consumed

P5 locates native scores at `ContraMambaV6BMinimal.forward -> output["logits"]`. They are live tensors computed from the epistemic heads, `FinalEntitlementDecisionHead`, and optional final-logit modulation.

The prior observability boundary begins inside `return_composer_input_observability`: tensors are detached, then the trainer moves them to CPU and converts them through `item`/Python values before JSON. P4 and P2 post-hoc recomposition therefore operate on serialized values, not a training autograd graph.

P5 also proves that the exact counterfactual needs a separately trained `frame_local_only` donor. That donor state cannot be recovered from a joint native forward. Once joint and donor live primitive tensors exist, the three candidate compositions require composer arithmetic only; no third backbone forward is permitted.

## Canonical differentiable geometry

`stage196b2b6p6_differentiable_composer_geometry` is the only canonical training-side geometry structure. Invocation requires `diagnostic_enabled=True`. It accepts one live joint output, one live frame-local-only donor output, and the P2-authoritative row-wise primitive actions.

The three 14-bit values are opaque selector identities and are never interpreted as primitive bits:

```text
00100000000000
01000000000000
10000000000000
```

Each identity maps, through the P2 candidate CSV, to a row-specific five-bit action in the frozen primitive order `FRAME, PREDICATE, SUFFICIENCY, POSITIVE_ENERGY, NEGATIVE_ENERGY`. Candidate composition uses `torch.where` to select live recipient or donor primitives, calls the existing `model.decision_head`, and adds the live recipient residual `native_logits - native_base_logits`. This preserves P2 `apply_mask` semantics: only the five primitives are substituted; recipient comparator and optional final-modulation branches remain recipient-native.

The structure exposes native and counterfactual SUPPORT, NOT_ENTITLED, and REFUTE scores; SUPPORT-minus-NOT_ENTITLED, SUPPORT-minus-REFUTE, REFUTE-minus-NOT_ENTITLED, and piecewise top1-runner-up margins; and all required counterfactual-minus-native responses. Every coordinate preserves its batch dimension and is retained before detach, CPU transfer, scalar extraction, NumPy, list conversion, or serialization.

## Tensor contract

Every tensor row records shape, dtype, device, requires-grad, grad-fn class, leaf state, finiteness, and batch dimension. Numeric tensors must be finite CUDA tensors. Geometry tensors must retain a grad function and must not be implicitly reduced. Scalar sums exist only inside an individual `torch.autograd.grad` call.

## Probe boundary

The probe requires explicit joint and frame-local-only checkpoints. It also requires P5, P4, and P2 analysis paths; the exact Mamba identity; CUDA; clean main data; batch size; seed; commit; and a nonexistent output directory. It performs one deterministic P2-aligned clean controlled batch.

The supplied seed183 checkpoints may be reconstructed from the original
training commands. Role, seed, configuration, source, and checkpoint
provenance checks remain mandatory; reconstruction is not a claim of byte
identity with lost historical checkpoint files.

The probe creates no optimizer or scheduler, calls neither `optimizer.step` nor `scheduler.step`, writes no checkpoint, runs no epoch, performs no external or OOD evaluation, and never invokes a global training backward. It performs exactly one forward for each authoritative arm because P5 established that a final-composer-only native simulation cannot recover donor primitives.

## Parameter grouping and intended authority

Groups are derived from module references, not name substrings:

- `model.mamba` is split into `frozen_backbone` and `trainable_backbone` by `requires_grad`.
- FrameGate, PredicateCoverageHead, SufficiencyGate, and PolarityEnergyHead form `epistemic_heads`.
- FinalEntitlementDecisionHead and comparator alpha parameters form `final_composer`.
- Existing router or selector module references, if present, form `router_or_selector`; v6b-minimal normally has none.
- Remaining trainable parameters form `other_trainable`.

Before observing gradients, P6 declares `epistemic_heads` as the required recipient for both intervention families and `final_composer` as coordinate-conditional. P5?s graph shows that every response originates in live recipient/donor epistemic primitives, while NOT_ENTITLED-involving coordinates can additionally reach composer bias/alpha; SUPPORT-minus-REFUTE cancels those trainable composer scalars algebraically. There is no trainable selector authority in v6b-minimal.

The authoritative configuration freezes Mamba. Frozen backbone gradients are not required and their absence is not a failure or an unfrozen-Mamba claim.

## Independent gradient probes

The probe independently reduces and probes:

1. each native class score;
2. each class score for each candidate identity;
3. each of the four direction coordinates for each candidate;
4. each signed response difference for the three unordered candidate pairs.

Before each target it clears parameter `.grad` fields and calls `torch.autograd.grad(..., allow_unused=True, create_graph=False)`. The graph is retained only while later independent targets reuse it. No target is mixed with classification loss or another target.

For each target and parameter group the output distinguishes tensor count, trainable parameter count, connected tensors, unused tensors, finite-gradient tensors, nonzero-gradient tensors, L1 norm, L2 norm, and maximum absolute gradient.

Classifications are exactly:

```text
CONNECTED_NONZERO
CONNECTED_ZERO_AT_OBSERVED_BATCH
DISCONNECTED
NONDIFFERENTIABLE
NONFINITE
```

A finite connected zero is never reported as disconnected `None`.

## Exact equivalence

On the observed joint state, the existing native logits and predictions must equal the instrumented native logits and predictions exactly. For each candidate, an independent live-tensor implementation of the P2 formula must exactly equal the canonical instrumented logits. Candidate action identities must match their P2 rows. No tolerance is available to conceal divergence.

Required disagreement counts are all zero: native scores, native predictions, counterfactual composer scores, and candidate-action identities.

## No-mutation and loss nonexistence

Parameter fingerprints are recorded before probing and after native, counterfactual, direction, and candidate-order probes. Persistent-buffer fingerprints and model training/evaluation states are recorded and restored exactly. Scalar parameters and scalar buffers are included and are never skipped.

The canonical tensor-byte boundary first detaches, moves to CPU, makes the tensor contiguous, and then reshapes it to one dimension before reinterpreting it as `uint8`. The digest metadata is written before payload bytes and retains the state name, dtype, and original exact shape; therefore scalar shape `[]` remains distinct from vector shape `[1]` even though both payloads are flat.

Empty tensors produce an empty byte payload while retaining their name, dtype, and original shape metadata. Floating and integer strided dtypes use the same byte reinterpretation. In particular, `bfloat16` is viewed as `uint8` before NumPy conversion, so NumPy is never asked to represent `bfloat16` values. No scalar uses `item()`, a Python float, or textual value serialization. Fingerprint equality remains exact.

Serialization failures include structured fingerprint scope, tensor name, shape, dtype, layout, pre-CPU device, `requires_grad`, and exception type. This repair changes byte serialization only; it does not change model forwards, geometry, gradient targets, classifications, equivalence checks, or mutation checkpoints.

Evaluation mode prevents BatchNorm-like running-state mutation. Optimizer, scheduler, and checkpoint-write counts remain zero.

P6 proves all five values false:

```text
stability loss added to training objective
classification loss changed
optimizer objective changed
training coefficient added
combined intervention implemented
```

## Decisions

The probe evaluates the exact six-way hierarchy without hardcoding observed connectivity: complete, direction-only, candidate-order-only, detached, full-counterfactual-forward-required, or blocked contract failure.

Because P5?s current authoritative semantics use a separately trained donor, exact P2 geometry is classified `FULL_COUNTERFACTUAL_FORWARD_REQUIRED` unless future instrumentation co-emits an equivalent donor state. This scientific result is distinct from contract failure. Direction and candidate-order readiness are still measured independently and reported.

The next stage follows from observed connectivity and computation-path classification, never from a forced intervention choice.

## Outputs and return code

The output directory must not already exist. Exactly nine files are atomically published:

```text
stage196b2b6p6_analysis.json
stage196b2b6p6_report.md
stage196b2b6p6_tensor_schema.csv
stage196b2b6p6_gradient_connectivity.csv
stage196b2b6p6_parameter_group_audit.csv
stage196b2b6p6_forward_equivalence.csv
stage196b2b6p6_no_mutation_audit.csv
stage196b2b6p6_decision_gate.csv
stage196b2b6p6_contract.csv
```

The probe returns 0 when `blocking_reasons == []` and 2 otherwise. Scientifically negative connectivity or a required full donor forward is not by itself a contract blocker. Input identity, schema, finiteness, CUDA, upstream closure, exact equivalence, mutation, loss, output, or decision-reachability failures are blockers.

## Static implementation status

This stage was implemented under a static-inspection-only policy. Python, compilation, help, probe execution, trainer execution, smoke tests, model/checkpoint loading, training, evaluation, commit, and push were not performed during implementation.
