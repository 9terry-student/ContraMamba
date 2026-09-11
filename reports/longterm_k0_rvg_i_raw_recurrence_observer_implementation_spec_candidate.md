# K0-RVG-I Raw Recurrence Observer Implementation Specification Candidate

**Status:** implementation-specification candidate only.

**Parent authority:** `K0-RVG-S — Static Native Recurrence Observation Contract`

**Parent commit:** `e048e83083da105ee44ef53e9e32193591c79a90`

**Parent specification SHA256:**

`0de9594c6d271cf220eb713f0c7d17da9c1afb339b0aa04a7d7dde5d5fe72f6c`

This document does not itself implement code and does not execute any model forward.

Once this document is frozen as the immediate one-file child of the parent commit above, it becomes the complete authority for one bounded implementation phase and its synthetic-only validation.

It does not authorize scientific-population execution or scientific recurrent-state reads.

## 1. Objective

Implement a read-only observer for the exact natural Mamba recurrence tuple:

`(S_(t-1), G_t, W_t, S_t)`

with:

`S_t = G_t ⊙ S_(t-1) + W_t`

and no learned projection, no recurrence intervention, and no task-population selection.

The implementation exists only to prove that raw recurrence geometry can be observed faithfully under the already-frozen CPU slow-path runtime.

## 2. Exact implementation scope

After this specification is frozen, implementation may create exactly two new files:

`scripts/longterm_k0_rvg_raw_recurrence_observer.py`

`tests/test_longterm_k0_rvg_raw_recurrence_observer.py`

No existing repository file may be modified.

In particular, implementation must not modify:

`scripts/longterm_k2s_pair_specific_event_dynamics.py`

or any K2S/K2R/K3/K3C/K3T historical runner/report.

Historical untracked K1 files remain untouched:

`scripts/longterm_k1_native_state_kinematics.py`

`tests/test_longterm_k1_native_state_kinematics.py`

## 3. Frozen source/runtime identity

The implementation must bind the same runtime family frozen by K0-RVG-S:

Model ID:

`state-spaces/mamba-130m-hf`

HF revision:

`5708daa364c50b880e7bd92eab456e0d34492ee9`

Transformers:

`5.12.1`

Native recurrence function:

`MambaMixer.slow_forward`

Expected installed source SHA256:

`23c7b410e204b5da01732566de10c94b70a8418ecb608e409754b00332eb2a41`

Expected upstream source blob:

`87987e3e6646d8d0f9f0048bdd8a155d99c845db`

Expected source-line semantics:

- `slow_forward` definition line 270;
- `discrete_A` definition line 322;
- `deltaB_u` definition line 324;
- sequential loop line 349;
- recurrence update line 350;
- post-update C readout line 351;
- ordinary `forward` definition line 366.

Numeric lines are provenance checks only.

The implementation must also structurally prove the recurrence through AST/source inspection and fail closed if the unique structure is absent.

## 4. Frozen historical bridge

The implementation must preserve continuity with the existing validated helper:

`scripts/longterm_k2s_pair_specific_event_dynamics.py`

Expected helper SHA256:

`f741780e7199452e64b7c4a3d70f50f3e55ebc84296f69f288aa317582de84b8`

Expected helper Git blob:

`3a651fb508669bdcf72441a4869b863d6eee6c1f`

The new observer must not silently redefine `post_consumption_s_t`.

During synthetic validation, its `S_post` snapshots must be byte-identical to the existing K2S/O0c-compatible post-consumption snapshots at every common `(layer, token)` coordinate used by the cross-check.

## 5. Required observer semantics

At the CPython line event for recurrence-update line 350, before line 350 executes, capture:

`S_prev = ssm_state`

`G = discrete_A[:, :, i, :]`

`W = deltaB_u[:, :, i, :]`

At the CPython line event for C-readout line 351, before line 351 executes, capture:

`S_post = ssm_state`

The observer must pair the two halves by:

`(layer_index, token_index)`

and reject:

- missing pre-update capture;
- missing post-update capture;
- duplicate pre-update capture;
- duplicate post-update capture;
- unknown layer;
- non-integer token index;
- token index outside requested targets;
- shape disagreement;
- unsupported dtype;
- nonfinite values;
- snapshot aliasing;
- cross-forward collector reuse.

## 6. Required data model

Implementation must expose a compact immutable record or equivalent structure for one recurrence coordinate containing at least:

- `layer_index`;
- `token_index`;
- `S_prev`;
- `G`;
- `W`;
- `S_post`.

The live tensors must never be retained by alias.

Every captured tensor must be snapshotted with semantics equivalent to:

`detach -> cpu -> contiguous -> clone`

The implementation must record or validate the original shape, dtype, and device before snapshotting.

No flattening is permitted inside the capture layer.

The native tensor shape remains:

`[batch, intermediate_size, ssm_state_size]`

## 7. Required exact recurrence validation

For every synthetic captured coordinate, independently reconstruct:

`S_reconstructed = G ⊙ S_prev + W`

and require exact equality:

`torch.equal(S_reconstructed, S_post)`

under the frozen CPU float32 sequential path.

Failure is a contract failure.

No tolerance fallback is permitted for this source-recurrence equality.

## 8. Rearranged velocity validation

The observer may derive:

`V_raw = S_post - S_prev`

`V_carry_change = (G - 1) ⊙ S_prev`

`V_write = W`

and compare:

`V_raw`

against:

`V_carry_change + V_write`.

Because this rearrangement changes floating-point operation order, the implementation must not use `torch.equal` as its only acceptance rule for this secondary identity.

The implementation specification freezes the synthetic-only tolerance as:

absolute tolerance `1e-6`

relative tolerance `1e-5`

using float32 tensors.

This tolerance applies only to the rearranged velocity identity.

It does not weaken the exact primary recurrence reconstruction in Section 7.

The implementation must record maximum absolute residual and maximum relative residual during synthetic validation.

## 9. Source analyzer requirements

The observer must include a source analyzer that proves, from the installed `modeling_mamba.py` bytes:

1. exactly one `MambaMixer` class;
2. exactly one `slow_forward`;
3. exactly one ordinary `forward`;
4. one sequential loop over token index `i`;
5. an `ssm_state` assignment in that loop;
6. recurrence assignment structurally equivalent to:
   `discrete_A[..., i, :] * ssm_state + deltaB_u[..., i, :]`;
7. the immediately following statement reads updated `ssm_state` for the C readout;
8. ordinary `forward` retains a `self.slow_forward(...)` fallback.

The analyzer must return both recurrence-update line and readout line.

The observer must bind its trace only to the exact `slow_forward.__code__` object and exact verified line numbers.

## 10. Trace lifecycle

A collector instance is single-use per forward.

It must:

1. begin with no active snapshot map;
2. preserve any pre-existing Python trace function;
3. install its trace only for the bounded forward;
4. restore the prior trace function in a `finally` path;
5. reject reuse after a completed capture;
6. reject any incomplete pre/post pair at finalization.

The collector must ignore all frames except the exact verified `MambaMixer.slow_forward` code object.

## 11. Layer binding

Layer registration must be by actual mixer object identity.

For the frozen Mamba family:

`N_LAYERS = 24`

The implementation must verify exactly 24 distinct mixer identities and map them to canonical layer indices `0..23`.

No name-based or order-guess fallback is allowed.

## 12. Synthetic-only runtime validation authority

Once this implementation specification is frozen and the two implementation files exist, the implementation phase is authorized to run only non-study synthetic validation.

The synthetic validation may load the already-validated seed180 A0 checkpoint solely to exercise the unchanged model forward and compare outputs.

Frozen handoff identity:

ZIP SHA256:

`96859bad3e400613b4c981990e56aaf35b1d92baaaa930eb11448cf38a63b861`

Checkpoint SHA256:

`4f7ad019bddb988a534c477b58b36bdabe2775d6c9748331e8311653c07c864c`

Common encoder canonical SHA256:

`48a7e9ac9dfa6c8c292090ee0fcb606bd4c85d13706bfc8a3e371af77c440597`

Common encoder raw-concat SHA256:

`968c12c095a6aab883db5984f4c02ad5e893a5ff140781ffbda41b97970401ae`

Synthetic validation must use fabricated non-study text only.

It must not load or enumerate any K2W/K2R/K3/K3C/K3T scientific population.

## 13. Required synthetic validation suite

Before implementation may be considered complete, tests and/or one synthetic-preflight CLI must demonstrate all of the following:

1. source SHA binding PASS;
2. source AST/structure binding PASS;
3. slow-path traversal PASS;
4. 24-layer identity registration PASS;
5. exact pre/post pair completeness PASS;
6. expected tensor shapes PASS;
7. expected float32 recurrence tensors PASS;
8. finite values PASS;
9. no snapshot aliasing PASS;
10. exact `S_post = G ⊙ S_prev + W` reconstruction PASS;
11. velocity rearrangement within frozen tolerance PASS;
12. exact model-output/logit noninterference PASS;
13. identical-input fresh-forward snapshot identity PASS;
14. causal common-prefix state identity across synthetic continuations PASS;
15. byte-identical bridge to existing K2S post-consumption `S_t` PASS;
16. trace restoration PASS;
17. collector-reuse rejection PASS;
18. malformed/ambiguous source analyzer fail-closed tests PASS.

No scientific claim may be made from these validations.

## 14. Output and CLI boundary

The new script may expose:

`--synthetic-preflight`

and ordinary help/versionless inspection behavior.

It must not expose a scientific execution flag in this implementation phase.

Forbidden CLI concepts include:

`--scientific`

`--population`

`--evaluate`

`--compare-correct-wrong`

`--intervene`

or any equivalent scientific-data path.

The script may print a compact synthetic validation manifest to stdout.

Synthetic artifacts, if any, must be clearly labeled synthetic and must not use scientific-run naming.

## 15. Fail-closed repository contract

The implementation must bind branch:

`longterm-k-series-native-state-kinematics`

During implementation validation, the only allowed worktree changes beyond the two new implementation files are the two historical K1 untracked files.

The observer implementation must not require committing, staging, or modifying the K1 files.

The implementation must reject unexpected tracked modifications when its provenance checks are invoked.

## 16. No hidden scientific execution

The implementation must contain no import-time model forward.

Importing the module must not:

- load a checkpoint;
- download a model;
- run a tokenizer;
- execute Mamba;
- inspect a scientific population;
- read recurrent states.

All runtime action must occur only behind an explicit synthetic-preflight or directly invoked test/helper function.

## 17. Noninterference definition

Tracing is valid only if model computation is unchanged.

Synthetic validation must compare the same fabricated input under:

A. ordinary forward with observer disabled;

B. forward with observer enabled.

The output/logit tensors must be exactly equal.

Any difference is:

`RAW_RECURRENCE_OBSERVER_NONINTERFERENCE_FAILURE`

and blocks implementation completion.

## 18. Existing K2S bridge definition

For the same synthetic input and target coordinates, the implementation must obtain existing K2S-style post-consumption snapshots under the already-validated timing.

The new observer's:

`S_post(layer, token)`

must satisfy byte identity with:

`K2S_post_consumption_S_t(layer, token)`.

The implementation may reuse validated helper code by safe import or may independently invoke equivalent existing helper functionality, but it must not modify the helper file.

If the bridge cannot be demonstrated exactly, implementation completion is blocked.

## 19. Scientific boundary

This specification does not authorize:

- reading recurrent state on any historical or prospective scientific population;
- selecting examples by correctness;
- comparing correct versus confident-wrong examples;
- fitting a probe;
- estimating a geometry from study data;
- promoting Branch A;
- promoting Branch B;
- causal intervention;
- K4;
- scientific conclusion.

Synthetic validation PASS establishes only observer correctness.

It does not establish epistemic geometry.

## 20. Completion artifact

After bounded implementation and synthetic validation pass, the next artifact must be a read-only implementation validation report.

That report must record at least:

- implementation commit;
- exact script SHA256;
- exact test SHA256;
- source/runtime identities;
- synthetic-preflight result;
- exact recurrence reconstruction result;
- noninterference result;
- K2S bridge result;
- dirty-state contract;
- explicit statement that no scientific-population recurrent state was read.

No scientific execution authority may be embedded into the implementation commit.

## 21. Next-stage boundary

If and only if implementation plus synthetic validation is frozen and independently verified, the next allowed stage is:

`K0-RVG-V — Raw Recurrence Observer Validation / Scientific-Use Readiness Review`

That stage may decide whether the observer is sufficiently validated to justify drafting a later scientific observation preregistration.

It must not retroactively convert synthetic validation into scientific evidence.

## 22. Frozen implementation authority markers

Before this specification is committed, all implementation remains unauthorized.

After this exact specification is frozen as the immediate one-file child of:

`e048e83083da105ee44ef53e9e32193591c79a90`

the following bounded authority becomes active:

`K0_RVG_I_SPEC_FROZEN = YES`

`RAW_RECURRENCE_OBSERVER_IMPLEMENTATION_AUTHORIZED = YES`

`RAW_RECURRENCE_OBSERVER_IMPLEMENTATION_SCOPE = TWO_NEW_FILES_ONLY`

`SYNTHETIC_MODEL_FORWARD_AUTHORIZED = YES`

`SYNTHETIC_RECURRENT_STATE_READ_AUTHORIZED = YES`

`SCIENTIFIC_MODEL_FORWARD_AUTHORIZED = NO`

`SCIENTIFIC_RECURRENT_STATE_READ_AUTHORIZED = NO`

`SCIENTIFIC_POPULATION_ACCESS_AUTHORIZED = NO`

`BRANCH_A_STATUS = PARKED_NOT_CLOSED`

`BRANCH_B_STATUS = PARKED_NOT_ACTIVE`

`K_SUCCESSOR_BRANCH_SELECTION = DEFERRED`

`CAUSAL_INTERVENTION_AUTHORIZED = NO`

`K4_EXECUTION_AUTHORIZED = NO`

Implementation must stop after the bounded code/test delta and synthetic-only validation. A separate validation/readiness artifact is required before any scientific-use authority can be considered.
