# K0-RVG-S Static Native Recurrence Observation Contract Candidate

**Status:** read-only static source audit / observation-contract candidate.

**Parent authority:** `K0-RVG — Native Raw Vector Geometry / Recurrence-Exact Kinematics`

**Parent commit:** `57addcabfcc2c1cbb8b6396db201b8b4c1ddc1c3`

**Parent specification SHA256:**

`a2360442996b8560d884ab1ac811eb8e5720fb82f4cb7e555ce9d1e03c64d2c7`

This document performs no model forward, no recurrent-state read, no training, no evaluation, no intervention, and no probe/geometry fitting.

Its purpose is to bind the exact frozen source semantics needed to observe the natural Mamba recurrence tuple:

`(S_(t-1), G_t, W_t, S_t)`

without changing the model computation.

## 1. Governing branch state

K0-RVG intentionally deferred the scientific successor fork.

The current state remains:

`K_SUCCESSOR_BRANCH_SELECTION = DEFERRED`

`BRANCH_A_STATUS = PARKED_NOT_CLOSED`

`BRANCH_B_STATUS = PARKED_NOT_ACTIVE`

K0-RVG-S does not activate either branch.

## 2. Frozen runtime family

The static contract binds the already-validated runtime family:

Model:

`state-spaces/mamba-130m-hf`

HF revision:

`5708daa364c50b880e7bd92eab456e0d34492ee9`

Transformers:

`5.12.1`

Native recurrence function:

`MambaMixer.slow_forward`

Previously validated installed source SHA256:

`23c7b410e204b5da01732566de10c94b70a8418ecb608e409754b00332eb2a41`

Upstream Transformers tag:

`v5.12.1`

Upstream source path:

`src/transformers/models/mamba/modeling_mamba.py`

Upstream Git blob SHA:

`87987e3e6646d8d0f9f0048bdd8a155d99c845db`

The installed runtime source SHA and the upstream Git blob SHA are different hash schemes and are recorded separately.

A future implementation must fail closed unless the installed runtime source retains the frozen SHA256 and the expected structural source bindings.

## 3. Existing validated state instrumentation

Frozen helper:

`scripts/longterm_k2s_pair_specific_event_dynamics.py`

Frozen helper SHA256:

`f741780e7199452e64b7c4a3d70f50f3e55ebc84296f69f288aa317582de84b8`

Frozen helper Git blob:

`3a651fb508669bdcf72441a4869b863d6eee6c1f`

The helper already statically analyzes `MambaMixer.slow_forward` and proves:

1. there is a unique sequential loop over token index `i`;
2. the loop contains an `ssm_state` recurrence assignment;
3. that assignment uses `discrete_A`, previous `ssm_state`, `deltaB_u`, and `i`;
4. the immediately following statement computes `scan_output` from the updated `ssm_state` and `C`;
5. ordinary `MambaMixer.forward` contains a slow-forward fallback.

Its existing CPython line collector captures `ssm_state` at the readout line.

Because CPython emits the line event before executing the readout statement, that captured tensor is the state after the immediately preceding recurrence update and before C readout.

This is the validated timing:

`post_consumption_s_t`

K0-RVG-S preserves this timing exactly.

## 4. Exact upstream source-line binding

For Transformers `v5.12.1`, the relevant source bindings are:

`MambaMixer.slow_forward` definition:

line `270`

`discrete_A` definition:

line `322`

`deltaB_u` definition:

line `324`

sequential token loop:

line `349`

recurrence update:

line `350`

post-update C readout:

line `351`

`MambaMixer.forward` definition:

line `366`

The frozen statements are structurally:

```text
discrete_A =
    exp(A * discrete_time_step)

deltaB_u =
    discrete_B * hidden_states

for i in range(seq_len):
    ssm_state =
        discrete_A[:, :, i, :] * ssm_state
        + deltaB_u[:, :, i, :]

    scan_output =
        matmul(ssm_state, C[:, i, :])
```

Numeric line numbers are provenance checks, not the sole binding method.

A future observer must also re-run AST/source-structure validation and fail closed if the expected recurrence structure is not unique.

## 5. Exact recurrence notation

At token step t, define:

`S_(t-1)`

as the value of local `ssm_state` immediately before line 350 executes.

Define:

`G_t = discrete_A[:, :, t, :]`

and:

`W_t = deltaB_u[:, :, t, :]`

from the same `slow_forward` frame and same token index.

Define:

`S_t`

as the value of local `ssm_state` immediately after line 350 has executed and before line 351 executes.

The source recurrence is therefore:

`S_t = G_t ⊙ S_(t-1) + W_t`

with elementwise multiplication.

For a batch of size B:

`S_(t-1), G_t, W_t, S_t`

all have shape:

`[B, intermediate_size, ssm_state_size]`

For the frozen one-example scientific branch convention, B is normally 1.

## 6. Capture timing contract

A future CPython trace observer may use exactly two trace moments inside the same `MambaMixer.slow_forward` frame.

### Pre-update capture

At the line event for recurrence line 350, before line 350 executes, capture:

- layer identity from local `self`;
- token index from local `i`;
- `S_prev = ssm_state`;
- `G = discrete_A[:, :, i, :]`;
- `W = deltaB_u[:, :, i, :]`.

At this moment `ssm_state` is still `S_(t-1)`.

### Post-update capture

At the line event for readout line 351, before line 351 executes, capture:

- the same layer identity;
- the same token index `i`;
- `S_post = ssm_state`.

At this moment line 350 has completed, so `ssm_state` is `S_t`.

The observer must pair pre-update and post-update captures by:

`(layer_index, token_index)`

and must reject:

- missing halves;
- duplicate halves;
- layer ambiguity;
- token-index ambiguity;
- shape mismatch;
- dtype mismatch;
- nonfinite tensors.

## 7. Natural-execution tensor objects

The observer may derive, without learned geometry:

Raw velocity:

`V_t = S_t - S_(t-1)`

Carry-change velocity component:

`V_t^(carry-change) = (G_t - 1) ⊙ S_(t-1)`

Write velocity component:

`V_t^(write) = W_t`

Full retained contribution:

`H_t = G_t ⊙ S_(t-1)`

and therefore:

`V_t^(carry-change) = H_t - S_(t-1)`

The exact conceptual identity is:

`V_t = V_t^(carry-change) + V_t^(write)`

These are natural-execution observational quantities.

They are not interventions.

## 8. Relationship to historical K3/K3C

Historical K3/K3C work must remain separate.

K3 experimentally manipulated/equalized recurrence components and tested causal specialization hypotheses.

K0-RVG-S does not replay or reinterpret those interventions.

In particular:

- no `W_EQ`;
- no `G_EQ`;
- no `H_EQ`;
- no midpoint substitution;
- no counterfactual recurrence;
- no causal claim.

K0-RVG-S only observes the tensors already computed by the unmodified natural forward recurrence.

## 9. Dtype and device contract

Existing validated K2S instrumentation established, for the frozen CPU sequential runtime:

state dtype:

`torch.float32`

captured state device after snapshot:

`cpu`

A future raw recurrence observer must record the original local tensor dtype and device before snapshotting.

For the frozen CPU synthetic validation target, the expected local recurrence tensors are:

- `S_prev`: float32;
- `G`: float32;
- `W`: float32;
- `S_post`: float32.

The observer must snapshot using:

`detach -> cpu -> contiguous -> clone`

and prove no captured tensor aliases the live model tensor.

Any future scientific execution that changes device/backend requires separate authority because a fast kernel path does not expose this exact Python sequential recurrence frame.

## 10. Slow-path requirement

The observation contract is bound to:

`MambaMixer.slow_forward`

and specifically its sequential recurrence loop.

A future observer must fail closed unless ordinary model execution actually traverses this bound slow path.

The current frozen CPU environment is appropriate because ordinary `MambaMixer.forward` falls back to `slow_forward` when the fast CUDA kernel path is unavailable/not selected.

K0-RVG-S does not authorize GPU or fused-kernel scientific observation.

## 11. Noninterference requirement

Observation must be read-only.

The observer may:

- inspect frame locals;
- slice the current token tensor;
- detach;
- clone;
- move the clone to CPU;
- serialize/hash clones outside the model's live state.

The observer may not:

- assign to frame locals;
- mutate `ssm_state`;
- mutate `discrete_A`;
- mutate `deltaB_u`;
- mutate cache state;
- alter model parameters;
- change hidden states;
- change task masks;
- change logits;
- insert a learned module.

Future synthetic validation must compare ordinary outputs with tracing disabled versus tracing enabled and require exact output/logit equality under the frozen runtime.

## 12. Recurrence identity validation

For every captured `(layer, token)` tuple, future synthetic validation must independently reconstruct:

`S_reconstructed = G_t ⊙ S_(t-1) + W_t`

Primary recurrence check:

`torch.equal(S_reconstructed, S_t)`

under the frozen CPU float32 sequential path.

If exact equality fails, the implementation must fail closed during synthetic validation.

It must not silently weaken the check on scientific data.

For the rearranged velocity identity:

`S_t - S_(t-1)`

versus:

`(G_t - 1) ⊙ S_(t-1) + W_t`

the observer must record a numerical residual because algebraic rearrangement changes floating-point operation order.

A tolerance for that rearranged identity must be frozen in a later implementation/validation specification before any scientific population is observed.

The source recurrence equality itself remains the primary exact check.

## 13. Tensor provenance contract

Every serialized natural-recurrence capture must bind:

- runtime Git HEAD;
- model ID;
- HF revision;
- Transformers version;
- installed `modeling_mamba.py` SHA256;
- upstream tag/path/blob identity;
- function qualname;
- source-line bindings;
- layer index;
- token index;
- original tensor shape;
- original dtype;
- original device;
- snapshot dtype;
- snapshot device;
- state timing;
- tensor role: `S_prev`, `G`, `W`, or `S_post`;
- canonical byte order;
- SHA256 of serialized tensor bytes.

Canonical tensor serialization must use a frozen contiguous row-major representation.

Tensor shape metadata must remain separate so flattening cannot erase native-axis provenance.

## 14. Native-axis semantics

For the frozen Mamba implementation:

`ssm_state`

shape is:

`[batch, intermediate_size, ssm_state_size]`

The second axis corresponds to Mamba intermediate channels.

The third axis corresponds to the SSM state dimension.

Any later per-channel or per-state-mode descriptive analysis must preserve this mapping.

An arbitrary flattened index is not scientifically interpretable unless mapped back to these native axes.

## 15. Observer pairing and lifecycle

A future observer instance must be single-use per forward.

It must:

1. begin with no captured tuples;
2. record a pre-update tuple at recurrence line 350;
3. record exactly one post-update tuple at readout line 351;
4. pair them at the same layer/token coordinate;
5. reject duplicates;
6. restore any pre-existing Python trace function after the forward;
7. never carry snapshots from one independent model forward into another.

Fresh-state isolation remains mandatory.

## 16. Required synthetic-only validation before scientific use

K0-RVG-S itself does not execute this validation.

A later separately authorized implementation/validation stage must use synthetic non-study text only and prove:

1. exact source SHA and source-structure binding;
2. exact slow-path traversal;
3. complete pre/post capture pairing;
4. expected shapes;
5. expected float32 recurrence tensors;
6. finite values;
7. no snapshot aliasing;
8. exact recurrence reconstruction;
9. exact logit/output noninterference;
10. fresh-forward byte identity for identical synthetic input;
11. causal common-prefix identity across synthetic continuations;
12. ordinary existing post-consumption `S_t` capture equals the new observer's `S_post` byte-for-byte.

No scientific-population recurrent state may be read during this validation.

## 17. Existing instrumentation must be cross-checked, not replaced silently

The new observer must be validated against the existing K2S/O0c-compatible state capture.

For every synthetic `(layer, token)` coordinate:

`new_S_post`

must be byte-identical to:

`existing_K2S_post_consumption_S_t`

under the same forward.

If not, the new observer is invalid.

This is the bridge that preserves historical state-timing continuity.

## 18. Static audit conclusion

The frozen source is sufficient in principle for raw recurrence observation.

The required natural tensors are simultaneously available in the same Python frame and token step:

- previous state `S_(t-1)`;
- selective retention factor `G_t`;
- direct write term `W_t`;
- updated state `S_t`.

The capture can be defined without:

- learned geometry;
- probe fitting;
- state intervention;
- output modification;
- task-head projection.

Therefore:

`STATIC_NATIVE_RECURRENCE_OBSERVABILITY = SUPPORTED_BY_FROZEN_SOURCE`

This is a source/design conclusion only.

It is not scientific evidence about epistemic failure.

## 19. Next-stage boundary

After K0-RVG-S is frozen, the next allowed stage is:

`K0-RVG-I — Raw Recurrence Observer Implementation Specification`

That stage may specify the bounded implementation delta and synthetic-only validation suite.

It must not itself authorize:

- scientific-population model execution;
- confident-wrong/correct comparison;
- branch-A confirmation;
- branch-B confirmation;
- causal intervention;
- K4.

Implementation must remain a separate authority boundary.

## 20. Authority state

`K0_RVG_PARENT_FROZEN = YES`

`K0_RVG_S_STATIC_AUDIT_COMPLETE = YES`

`STATIC_NATIVE_RECURRENCE_OBSERVABILITY = SUPPORTED_BY_FROZEN_SOURCE`

`K_SUCCESSOR_BRANCH_SELECTION = DEFERRED`

`BRANCH_A_STATUS = PARKED_NOT_CLOSED`

`BRANCH_B_STATUS = PARKED_NOT_ACTIVE`

`RAW_RECURRENCE_OBSERVER_IMPLEMENTATION_AUTHORIZED = NO`

`SYNTHETIC_MODEL_FORWARD_AUTHORIZED = NO`

`SCIENTIFIC_MODEL_FORWARD_AUTHORIZED = NO`

`SCIENTIFIC_RECURRENT_STATE_READ_AUTHORIZED = NO`

`CAUSAL_INTERVENTION_AUTHORIZED = NO`

`K4_EXECUTION_AUTHORIZED = NO`

This document authorizes only drafting the K0-RVG-I implementation specification after K0-RVG-S is frozen.
