# K0-RVG-P1E Failed Scientific Attempt Incident Report Candidate

**Status:** failed-attempt incident record and recovery-boundary candidate.

**Frozen P1E execution-authority commit:**

`21c298b97c7f33a117ff2be667f640e054cc8125`

**P1E authority SHA256:**

`6f16c8a309da4e8e248f2fd4a6037646476b92e71d6f798f2d4bb71e381cb883`

**Validated P1 implementation commit:**

`2e6bb106d5d3081b7ae69ec4cde652e79d36070c`

**P1 runner SHA256:**

`2bd59a25a6303dc86c36c9438296b42197cc67019e9d4e9ee2fbbe1d832ec6eb`

**P1 test SHA256:**

`06c30c974029126fdc853a235071376738cc2769409369ccdb60f0878bac7706`

This document records the first authorized K0-RVG-P1 scientific execution attempt and its failure boundary.

It does not authorize a rerun.

It does not change any preregistered endpoint, layer, window, metric, tolerance, support rule, or multiplicity rule.

It does not establish a scientific conclusion.

## 1. Attempt status

The one authorized P1E scientific attempt was launched after the frozen launch gate passed.

The process exited nonzero with:

`k0_rvg_p1_observer.ContractError: VELOCITY_REARRANGEMENT_TOLERANCE_FAILURE`

Therefore:

`P1E_FIRST_SCIENTIFIC_ATTEMPT_STATUS = FAILED`

`P1E_AUTHORIZED_EXECUTION_ATTEMPTS_REMAINING = 0`

`AUTOMATIC_RERUN_AUTHORIZED = NO`

## 2. Launch-gate status before failure

Before the scientific attempt:

- repository state:
  `PASS`;
- frozen authority HEAD:
  `21c298b97c7f33a117ff2be667f640e054cc8125`;
- authority SHA:
  exact match;
- runner SHA:
  exact match;
- test SHA:
  exact match;
- seed180 handoff SHA:
  exact match;
- runner-native execution-authority gate:
  `PASS`;
- frozen P0 static archive authentication:
  `PASS`;
- A0 provenance:
  `PASS`;
- observer/K2S provenance:
  `PASS`;
- focused tests:
  `42 passed`;
- fabricated synthetic preflight:
  `PASS_SYNTHETIC_P1_RAW_VECTOR_RUNNER`;
- scientific population forward during launch gate:
  `NO`;
- scientific population state read during launch gate:
  `NO`;
- scientific endpoint computation during launch gate:
  `NO`.

Thus the failure occurred after a valid launch gate.

## 3. Exact failure boundary

The exception stack was:

`execute_scientific`

→ `_item_scientific_execution`

→ `run_pair`

→ `capture_branch`

→ `validate_layer23_record`

→ frozen observer `validate_recurrence_record`

→ `VELOCITY_REARRANGEMENT_TOLERANCE_FAILURE`.

The runner had entered scientific population execution.

The failure occurred while validating a captured layer-23 recurrence record before completion of the full 336-item population.

No primary block aggregation or final endpoint summary was reached.

## 4. Frozen observer validation order

The frozen observer validates one recurrence record in this order:

1. metadata;
2. exact native recurrence reconstruction:

   `G_t * S_(t-1) + W_t == S_t`

   using exact `torch.equal`;
3. raw velocity:

   `V_raw = S_t - S_(t-1)`;
4. rearranged velocity:

   `V_rearranged = (G_t - 1) * S_(t-1) + W_t`;
5. float32 closeness:

   `torch.allclose(V_raw, V_rearranged, atol=1e-6, rtol=1e-5)`.

The observed exception is step 5.

Because the exact recurrence reconstruction check precedes step 5 and did not raise, the failing record had already passed:

`RECURRENCE_EXACT_RECONSTRUCTION = PASS`

for that record.

This does not imply that every record in the intended 336-item run passed exact recurrence; the run stopped at the first raised failure.

## 5. Failure classification

Current evidence supports the following bounded classification:

`FAILURE_CLASS = VELOCITY_REARRANGEMENT_NUMERICAL_SUPPORT_FAILURE`

This is an execution-instrument validation failure.

It is not currently classified as:

- P0 population provenance failure;
- token-contract failure;
- source-binding failure;
- exact recurrence reconstruction failure;
- endpoint support failure;
- primary scientific endpoint result;
- branch-level scientific result.

The exact failing coordinate, exact residual, tensor norms, and numerical mechanism were not emitted before the exception.

Therefore:

`ROOT_CAUSE_FULLY_ESTABLISHED = NO`

## 6. Synthetic numerical context

The final fabricated synthetic launch-gate preflight passed the same frozen observer contract.

Its maximum observed diagnostics were:

- maximum velocity absolute residual:
  `3.814697265625e-06`;
- maximum velocity relative Frobenius residual:
  `3.4151345205699163e-07`;
- maximum scaled-tolerance residual:
  `0.8391106128692627`.

The scientific failure demonstrates that at least one scientific-population layer-23 record exceeded the frozen elementwise allclose tolerance contract.

No tolerance sweep is authorized.

No new tolerance may be selected from the observed failure.

## 7. Artifact-state result

Read-only post-failure incident inspection recorded:

`OUTPUT_DIR_EXISTS = False`

No matching partial directory was present.

Repository status contained only the historical K1 untracked files:

`scripts/longterm_k1_native_state_kinematics.py`

`tests/test_longterm_k1_native_state_kinematics.py`

Therefore:

`P1E_COMPLETED_SCIENTIFIC_ARTIFACT_SET_EXISTS = NO`

`P1E_PARTIAL_SCIENTIFIC_ARTIFACT_SET_EXISTS = NO`

`P1E_REPOSITORY_CONTAMINATION_DETECTED = NO`

No scientific result artifact may be inferred from this failed attempt.

## 8. Scientific-result boundary

Because execution did not complete and no six-artifact result set exists:

`SCIENTIFIC_EXECUTION_SUCCESS = NO`

`SCIENTIFIC_ARTIFACT_PROVENANCE_VALID = NOT_APPLICABLE_NO_COMPLETED_ARTIFACT_SET`

`SCIENTIFIC_CONCLUSION = NONE`

The following are not established:

- turning signal;
- response-coherence signal;
- raw native vector organization;
- support adequacy of either primary endpoint;
- Branch A;
- Branch B;
- causal carry/write specialization;
- K4.

## 9. One-attempt authority exhaustion

The frozen P1E authority permitted:

`P1E_AUTHORIZED_EXECUTION_ATTEMPTS = 1`

That attempt has occurred.

Therefore:

`P1E_AUTHORITY_EXHAUSTED = YES`

`AUTOMATIC_RERUN_AUTHORIZED = NO`

The original scientific command must not be rerun under P1E.

## 10. No post-observation tuning

The failure does not authorize:

- increasing `atol`;
- increasing `rtol`;
- changing velocity formula;
- changing dtype;
- changing layer;
- changing W;
- dropping failing coordinates;
- dropping failing items;
- replacing items;
- changing primary geometry;
- changing endpoint formulas;
- changing support threshold;
- changing sign-test or Holm rules.

Any scientific-contract change would require a separately justified protocol revision, not a recovery rerun.

## 11. Recovery objective

The next stage is a bounded recovery diagnostic whose sole objective is:

> Determine the exact numerical mechanism and support profile of the frozen velocity-rearrangement validation failure without computing P1 scientific endpoints and without tuning a replacement tolerance from observed outcomes.

The recovery diagnostic should be designed to identify, at minimum:

- first failing scientific item identity;
- branch role;
- layer;
- token coordinate;
- exact recurrence pass/fail;
- `max_abs_residual`;
- relative Frobenius residual;
- maximum scaled-tolerance residual;
- norms and value scales sufficient to distinguish cancellation/rounding from capture corruption;
- whether the failure is isolated or structurally recurrent under a prospectively fixed diagnostic scope.

## 12. Recovery-access boundary

This incident report itself authorizes no new scientific state access.

After this exact report is frozen, the next stage may draft:

`K0-RVG-P1E-R1 — Numerical-Support Recovery Diagnostic Authority`

That authority must be separately frozen before any second scientific-population forward/state read.

The R1 diagnostic must not compute:

- `X_turn`;
- `X_coh`;
- phase-block scores;
- sign-test p-values;
- Holm-adjusted p-values;
- overall raw-vector verdict.

Its purpose is instrument/root-cause diagnosis only.

## 13. Recovery-design principle

The recovery diagnostic must preserve the original scientific estimand.

It may diagnose the observer validation contract.

It may not use scientific endpoint direction or significance to select:

- a new tolerance;
- a new metric;
- a new subset;
- a new layer;
- a new window.

If the existing tolerance is later shown to be an instrument-only overconstraint, any proposed correction must be justified from numerical analysis and synthetic/prospective validation independent of P1 endpoint outcomes.

## 14. Branch boundary

`K_SUCCESSOR_BRANCH_SELECTION = DEFERRED`

`BRANCH_A_STATUS = PARKED_NOT_CLOSED`

`BRANCH_B_STATUS = PARKED_NOT_ACTIVE`

No branch-selection decision is permitted from this failed attempt.

## 15. K4 boundary

`K4_EXECUTION_AUTHORIZED = NO`

## 16. Final incident markers

`P1E_FIRST_SCIENTIFIC_ATTEMPT_STATUS = FAILED`

`P1E_FAILURE_EXCEPTION = VELOCITY_REARRANGEMENT_TOLERANCE_FAILURE`

`FAILURE_CLASS = VELOCITY_REARRANGEMENT_NUMERICAL_SUPPORT_FAILURE`

`RECURRENCE_EXACT_RECONSTRUCTION_AT_FAILING_RECORD = PASSED_BEFORE_FAILURE`

`ROOT_CAUSE_FULLY_ESTABLISHED = NO`

`P1E_COMPLETED_SCIENTIFIC_ARTIFACT_SET_EXISTS = NO`

`P1E_PARTIAL_SCIENTIFIC_ARTIFACT_SET_EXISTS = NO`

`P1E_REPOSITORY_CONTAMINATION_DETECTED = NO`

`P1E_AUTHORITY_EXHAUSTED = YES`

`AUTOMATIC_RERUN_AUTHORIZED = NO`

`SCIENTIFIC_EXECUTION_SUCCESS = NO`

`SCIENTIFIC_CONCLUSION = NONE`

`SCIENTIFIC_MODEL_FORWARD_AUTHORIZED = NO`

`SCIENTIFIC_RECURRENT_STATE_READ_AUTHORIZED = NO`

`SCIENTIFIC_ENDPOINT_COMPUTATION_AUTHORIZED = NO`

`LOGITS_READ_AUTHORIZED = NO`

`CAUSAL_INTERVENTION_AUTHORIZED = NO`

`K4_EXECUTION_AUTHORIZED = NO`

`READY_FOR_P1E_R1_NUMERICAL_SUPPORT_RECOVERY_AUTHORITY_DRAFT = YES`

`NEXT_STAGE = K0-RVG-P1E-R1_NUMERICAL_SUPPORT_RECOVERY_DIAGNOSTIC_AUTHORITY_DRAFT`

This incident report becomes authoritative only after it is committed and pushed as the immediate one-file child of `21c298b97c7f33a117ff2be667f640e054cc8125`.
