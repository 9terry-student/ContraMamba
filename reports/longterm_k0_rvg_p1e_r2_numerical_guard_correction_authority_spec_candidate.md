# K0-RVG-P1E-R2 Numerical-Guard Correction Implementation Authority Candidate

**Status:** bounded numerical-guard correction implementation authority candidate.

**Immediate parent / frozen R1 validated interpretation commit:**

`37d7044b7111d7b02c7a44fdf7d8782c823afa5b`

**Frozen R1 interpretation report SHA256:**

`bfaa9163d2b0d4f405f5cd806e5f3a9ece5e7503eb94cf2fb03193d9c6db2fea`

**Original failed P1E execution authority:**

`21c298b97c7f33a117ff2be667f640e054cc8125`

**Original frozen P1 implementation:**

`2e6bb106d5d3081b7ae69ec4cde652e79d36070c`

This document authorizes only a bounded correction to the numerical integrity guard that blocked the original P1 execution.

It does not authorize scientific-population execution.

It does not authorize P1 endpoint computation on the scientific population.

It does not authorize a tolerance change.

It does not authorize any change to the preregistered P1 scientific estimand.

## 1. Validated defect

The validated R1 evidence established, for the fixed original failure record:

`R1_ORIGINAL_FAILURE_ROOT_CAUSE_ESTABLISHED = YES_FOR_FIXED_FAILURE_RECORD`

`R1_NUMERICAL_MECHANISM = FLOAT32_INTERMEDIATE_ROUNDING_UNDER_CANCELLING_REARRANGEMENT`

`R1_FAILING_RECORD_NATIVE_RECURRENCE_INTEGRITY = PASS`

`R1_HARD_REARRANGEMENT_GUARD_SHOWN_TO_BE_NUMERICALLY_OVERCONSTRAINING = YES_FOR_VALIDATED_RECORD`

`P1_PRIMARY_RAW_VELOCITY_DEFINITION_INVALIDATED = NO`

`POST_OBSERVATION_TOLERANCE_RETUNING_AUTHORIZED = NO`

The failing record had:

- exact native recurrence:
  `PASS`;
- one frozen-allclose failing element out of:
  `24576`;
- maximum scaled residual:
  `1.3258942365646362`;
- vector-level relative Frobenius residual:
  `5.271484209672739e-08`.

The defect is therefore an instrument-level hard blocking guard, not a validated failure of the native recurrence or primary raw velocity definition.

## 2. Correction principle

The correction must implement exactly this principle:

1. keep exact native recurrence reconstruction as the hard recurrence-integrity contract;
2. keep primary velocity exactly:
   `V_raw = S_post - S_prev`;
3. keep the algebraic rearrangement:
   `(G - 1) * S_prev + W`
   as a numerical diagnostic;
4. retain the original frozen:
   `atol = 1e-6`,
   `rtol = 1e-5`
   for descriptive comparability;
5. do not let a rearrangement-allclose exceedance block scientific execution when exact recurrence has passed;
6. do not drop, replace, or mask a record because of rearrangement-allclose status;
7. do not change any P1 scientific endpoint, geometry, layer, window, population, support threshold, or statistical rule.

This is a guard-semantics correction, not tolerance retuning.

## 3. Exact implementation scope

After this document is frozen, implementation may modify exactly these four existing tracked files:

`scripts/longterm_k0_rvg_raw_recurrence_observer.py`

`tests/test_longterm_k0_rvg_raw_recurrence_observer.py`

`scripts/longterm_k0_rvg_p1_raw_vector_execution.py`

`tests/test_longterm_k0_rvg_p1_raw_vector_execution.py`

No other tracked file may be modified or created by the correction implementation commit.

Historical K1 untracked files must remain untouched:

`scripts/longterm_k1_native_state_kinematics.py`

`tests/test_longterm_k1_native_state_kinematics.py`

## 4. Observer hard-integrity contract

The corrected observer must preserve the following as hard failures:

- metadata shape mismatch;
- dtype mismatch;
- device mismatch;
- nonfinite captured tensor rejection;
- exact native recurrence mismatch:
  `G * S_prev + W != S_post`.

The exact recurrence check must remain:

`torch.equal(G * S_prev + W, S_post)`

and must still raise:

`RECURRENCE_EXACT_RECONSTRUCTION_FAILURE`

on failure.

`R2_EXACT_RECURRENCE_HARD_GATE = YES`

## 5. Observer rearrangement diagnostic contract

The corrected observer must continue to compute, in float32:

`V_raw = S_post - S_prev`

`V_carry = (G - 1) * S_prev`

`V_write = W`

`V_rearranged = V_carry + V_write`

and the unchanged original comparison:

`torch.allclose(V_raw, V_rearranged, atol=1e-6, rtol=1e-5)`.

The original constants must remain:

`VELOCITY_ATOL = 1e-6`

`VELOCITY_RTOL = 1e-5`

However, this allclose result becomes nonblocking after exact recurrence has passed.

The observer must not raise:

`VELOCITY_REARRANGEMENT_TOLERANCE_FAILURE`

for an exact-recurrence record.

Instead, the observer return object must encode one of exactly two rearrangement statuses:

`PASS_TOLERANCE`

or:

`DIAGNOSTIC_TOLERANCE_EXCEEDED`

and must expose an unambiguous boolean equivalent of the frozen allclose result.

## 6. Observer residual diagnostics

The corrected observer must retain the existing diagnostics:

- `max_abs_residual`;
- relative Frobenius residual;
- maximum scaled tolerance residual;
- frozen `velocity_atol`;
- frozen `velocity_rtol`.

No value may be used as a scientific selection rule.

No new data-derived pass threshold may be introduced.

## 7. P1 layer-23 validation semantics

The corrected P1 runner must continue to hard-require:

- primary layer:
  `23`;
- state shape:
  `[1, 1536, 16]`;
- dtype:
  `torch.float32`;
- device:
  `cpu`;
- observer exact recurrence status:
  `PASS_EXACT`.

The runner must accept both observer rearrangement statuses:

`PASS_TOLERANCE`

and:

`DIAGNOSTIC_TOLERANCE_EXCEEDED`.

The runner must not raise or drop the record solely because of:

`DIAGNOSTIC_TOLERANCE_EXCEEDED`.

Unknown rearrangement status must fail closed.

## 8. P1 audit semantics

The corrected `AuditAccumulator` must distinguish, at minimum:

- total exact-recurrence records accepted;
- rearrangement diagnostic pass count;
- rearrangement diagnostic exceedance count;
- maximum absolute rearrangement residual;
- maximum relative Frobenius residual;
- maximum scaled tolerance residual;
- incoming common-state comparison count;
- incoming common-state comparison failure count.

For every accepted layer-23 recurrence record:

`diagnostic_pass_count + diagnostic_exceedance_count = exact_recurrence_accepted_count`

must hold.

No rearrangement exceedance may decrement scientific item count or block aggregation.

## 9. Recurrence audit artifact revision

`recurrence_audit.json` remains an exact required P1 result artifact.

Its schema must be revised to an explicit corrected version, for example:

`k0-rvg-p1-recurrence-audit-v2`

The corrected recurrence audit must make the new semantics explicit and include, at minimum:

- hard integrity gate:
  `EXACT_NATIVE_RECURRENCE`;
- rearrangement blocking:
  `false`;
- exact-recurrence accepted count;
- rearrangement diagnostic pass count;
- rearrangement diagnostic exceedance count;
- frozen `atol`;
- frozen `rtol`;
- maximum residual diagnostics;
- common incoming-state diagnostics.

The schema must not imply that every accepted record passed rearrangement allclose.

## 10. Result artifact names

The exact six P1 scientific result artifact filenames remain unchanged:

`item_metrics.jsonl`

`block_metrics.jsonl`

`endpoint_summary.json`

`recurrence_audit.json`

`state_hash_audit.jsonl`

`execution_manifest.json`

No seventh scientific result artifact is authorized.

## 11. Scientific estimand protection

The R2 correction must not change the semantics of:

- token-contract revalidation;
- target index construction;
- primary layer:
  `23`;
- window:
  `8`;
- four-branch order;
- common incoming-state equality;
- raw Frobenius cosine;
- pair metric construction;
- matched/swapped contrasts;
- `X_turn`;
- `X_coh`;
- reciprocal phase-block aggregation;
- support minimum:
  `160`;
- exact two-sided sign test;
- Holm `m=2`;
- alpha:
  `0.05`;
- zero-vector policy;
- item population:
  `336`;
- phase blocks:
  `168`.

## 12. Protected-function equivalence validation

The implementation test suite must independently compare the corrected P1 runner against parent commit:

`37d7044b7111d7b02c7a44fdf7d8782c823afa5b`

and establish AST-equivalence, ignoring location metadata, for all scientific-estimand functions that are not explicitly part of the numerical-guard correction.

At minimum the protected set must include the functions responsible for:

- target indices;
- raw Frobenius cosine;
- common incoming-state equality;
- pair metric computation;
- item metric computation;
- block aggregation;
- sign-test computation;
- Holm correction;
- endpoint summarization;
- P0 branch-text reconstruction;
- token-contract revalidation.

Any protected-function semantic drift is a blocker.

`R2_SCIENTIFIC_ESTIMAND_AST_EQUIVALENCE_REQUIRED = YES`

## 13. Corrected implementation-authority binding

The corrected observer and corrected P1 runner must identify this R2 correction authority commit as their applicable implementation authority, rather than silently treating the original pre-defect implementation authority as sufficient for the correction.

The exact R2 authority commit will be the commit that freezes this document.

The corrected implementation commit must be its immediate child.

## 14. Corrected implementation commit scope gate

Future corrected P1 execution-authority authentication must validate that the corrected implementation commit:

1. has this frozen R2 authority commit as its direct parent;
2. changes exactly the four authorized files in Section 3;
3. contains the exact corrected:
   - observer;
   - observer test;
   - P1 runner;
   - P1 runner test;
4. has no subsequent blob drift for any of those four files.

The old two-file implementation-scope assumption must not remain authoritative for the corrected implementation.

## 15. Future execution-authority provenance markers

A future corrected scientific execution authority must bind, at minimum:

- corrected implementation commit;
- corrected observer SHA256;
- corrected observer-test SHA256;
- corrected P1 runner SHA256;
- corrected P1 runner-test SHA256;
- R2 correction-authority SHA256;
- six frozen P0 archive SHA256 values;
- K2S identity;
- A0 identity;
- seed180 handoff/checkpoint/encoder identities;
- HF/runtime identities.

No scientific execution authority is created by this document.

## 16. Execution manifest provenance revision

The corrected P1 execution manifest must identify the corrected observer as part of the corrected implementation provenance.

It must not falsely identify the old frozen observer commit as the runtime observer after that file has been corrected.

The corrected manifest must bind the observer through the corrected implementation commit and exact observer file identity.

A schema-version revision is permitted and preferred if needed to avoid ambiguous old/new provenance semantics.

## 17. Synthetic correction fixture

The corrected test suite must contain a deterministic fabricated recurrence fixture with all of the following:

- exact recurrence:
  `PASS_EXACT`;
- frozen rearrangement allclose:
  `false`;
- observer rearrangement status:
  `DIAGNOSTIC_TOLERANCE_EXCEEDED`;
- observer does not raise;
- P1 layer-23 validation does not raise;
- P1 audit exact-recurrence count increments;
- P1 audit diagnostic exceedance count increments;
- no item is dropped.

The fixture must not derive its values from the validated scientific token-49 values.

## 18. Exact-recurrence failure fixture

A separate fabricated fixture must verify:

- exact recurrence mismatch still raises;
- the P1 runner cannot proceed with that record.

This prevents the correction from weakening the native recurrence integrity gate.

## 19. Passing-record compatibility

A fabricated exact-recurrence record that also passes the original rearrangement tolerance must still return:

`PASS_TOLERANCE`

and must preserve prior residual diagnostics.

The corrected logic must therefore distinguish a true diagnostic pass from a nonblocking exceedance.

## 20. Full fabricated synthetic integration

After implementation, a full fabricated P1 synthetic preflight must pass using the corrected observer/runner.

It must confirm:

- authentic model/checkpoint integration;
- exact branch capture completeness;
- exact recurrence hard checks;
- corrected rearrangement diagnostic accounting;
- state hash determinism;
- no logits;
- no causal intervention;
- no scientific P0 population forward;
- no scientific P0 recurrent-state read;
- no scientific endpoint computation on the frozen population.

## 21. Synthetic endpoint compatibility

On fabricated inputs where every recurrence record passes the old rearrangement allclose guard, the corrected runner must produce the same item/block/endpoint scientific quantities as the pre-correction runner.

The test may compare canonicalized fabricated scientific metric structures between:

- parent commit `37d7044...` implementation behavior;
- corrected implementation behavior.

This is a regression check only and does not authorize scientific population execution.

## 22. No post-observation tolerance tuning

The implementation must not change:

`VELOCITY_ATOL = 1e-6`

or:

`VELOCITY_RTOL = 1e-5`.

It must not add:

- a safety factor;
- an observed maximum multiplier;
- an adaptive threshold;
- a percentile threshold;
- a dtype-dependent threshold selected from R1 scientific residuals;
- a special-case token/item exemption.

`R2_POST_OBSERVATION_TOLERANCE_RETUNING = FORBIDDEN`

## 23. No record exclusion

A record with:

- valid metadata;
- exact native recurrence pass;
- finite tensors;

must not be excluded from P1 solely because the rearrangement diagnostic exceeds the frozen allclose tolerance.

No replacement item is authorized.

No failing coordinate may be masked.

## 24. Implementation-stage execution boundary

During R2 implementation and validation:

`R2_SYNTHETIC_MODEL_FORWARD_AUTHORIZED = YES`

`R2_SYNTHETIC_RECURRENT_STATE_READ_AUTHORIZED = YES`

`R2_FABRICATED_ENDPOINT_COMPUTATION_AUTHORIZED = YES`

`R2_P0_ARTIFACT_STATIC_READ_AUTHORIZED = YES`

`R2_SCIENTIFIC_MODEL_FORWARD_AUTHORIZED = NO`

`R2_SCIENTIFIC_RECURRENT_STATE_READ_AUTHORIZED = NO`

`R2_SCIENTIFIC_ENDPOINT_COMPUTATION_AUTHORIZED = NO`

The frozen P0 scientific population may be parsed/authenticated statically but may not be forwarded.

## 25. Training / intervention boundary

`TRAINING_AUTHORIZED = NO`

`HYPERPARAMETER_TUNING_AUTHORIZED = NO`

`LOGITS_READ_AUTHORIZED = NO`

`CAUSAL_INTERVENTION_AUTHORIZED = NO`

`K4_EXECUTION_AUTHORIZED = NO`

## 26. Independent implementation review

Because R2 changes:

- scientific execution gating;
- provenance validation;
- recurrence validation semantics;

the corrected four-file implementation must undergo an independent implementation review before freeze.

The verifier must specifically inspect:

- exact recurrence remains hard;
- rearrangement exceedance is nonblocking only after exact recurrence;
- scientific estimand protected functions are unchanged;
- future execution-authority gate binds all four corrected files;
- no tolerance retuning was introduced;
- no scientific population access occurred.

## 27. Future corrected execution boundary

After:

1. this R2 authority is frozen;
2. the exact four-file correction implementation is frozen;
3. corrected observer/P1 focused tests pass;
4. fabricated synthetic correction validation passes;
5. independent review passes;
6. an R2 implementation validation/readiness report is frozen;

a separate corrected scientific execution authority may be drafted.

That future authority must determine whether the original P1 scientific execution may be restarted from item 0 as a new, provenance-distinct execution.

No automatic rerun is authorized here.

## 28. Scientific-result boundary

This correction stage does not establish:

- `X_turn`;
- `X_coh`;
- any phase-block result;
- sign-test result;
- Holm result;
- overall P1 raw-vector verdict;
- Branch A;
- Branch B;
- K4.

`P1_SCIENTIFIC_CONCLUSION = NONE`

## 29. Branch state

`K_SUCCESSOR_BRANCH_SELECTION = DEFERRED`

`BRANCH_A_STATUS = PARKED_NOT_CLOSED`

`BRANCH_B_STATUS = PARKED_NOT_ACTIVE`

## 30. Final authority markers

`P1E_R2_NUMERICAL_GUARD_CORRECTION_AUTHORITY_FROZEN = YES`

`R2_IMPLEMENTATION_AUTHORIZED = YES`

`R2_IMPLEMENTATION_SCOPE = FOUR_EXISTING_FILES_ONLY`

`R2_EXACT_RECURRENCE_HARD_GATE = YES`

`R2_REARRANGEMENT_DIAGNOSTIC_NONBLOCKING = YES`

`R2_PRIMARY_RAW_VELOCITY = S_POST_MINUS_S_PREV_UNCHANGED`

`R2_VELOCITY_ATOL = 1e-6_UNCHANGED`

`R2_VELOCITY_RTOL = 1e-5_UNCHANGED`

`R2_SCIENTIFIC_ESTIMAND_AST_EQUIVALENCE_REQUIRED = YES`

`R2_POST_OBSERVATION_TOLERANCE_RETUNING = FORBIDDEN`

`R2_SYNTHETIC_MODEL_FORWARD_AUTHORIZED = YES`

`R2_SYNTHETIC_RECURRENT_STATE_READ_AUTHORIZED = YES`

`R2_FABRICATED_ENDPOINT_COMPUTATION_AUTHORIZED = YES`

`R2_P0_ARTIFACT_STATIC_READ_AUTHORIZED = YES`

`R2_SCIENTIFIC_MODEL_FORWARD_AUTHORIZED = NO`

`R2_SCIENTIFIC_RECURRENT_STATE_READ_AUTHORIZED = NO`

`R2_SCIENTIFIC_ENDPOINT_COMPUTATION_AUTHORIZED = NO`

`LOGITS_READ_AUTHORIZED = NO`

`CAUSAL_INTERVENTION_AUTHORIZED = NO`

`TRAINING_AUTHORIZED = NO`

`K4_EXECUTION_AUTHORIZED = NO`

`P1_SCIENTIFIC_CONCLUSION = NONE`

`READY_FOR_P1E_R2_CORRECTION_IMPLEMENTATION = YES`

`NEXT_STAGE = K0-RVG-P1E-R2_NUMERICAL_GUARD_CORRECTION_IMPLEMENTATION`

This authority becomes active only after this exact document is committed and pushed as the immediate one-file child of `37d7044b7111d7b02c7a44fdf7d8782c823afa5b`.
