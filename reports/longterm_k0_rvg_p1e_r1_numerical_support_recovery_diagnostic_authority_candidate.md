# K0-RVG-P1E-R1 Numerical-Support Recovery Diagnostic Authority Candidate

**Status:** bounded recovery diagnostic protocol / implementation authority candidate.

**Immediate parent / frozen failed-attempt incident commit:**

`6fe6a80f5fa0125971ac83d436e6e400cb7ab6f1`

**Frozen incident report SHA256:**

`0a0cfbf2c390af5537948b31146b844f22b64033958f87fe8649ddfb4b62000e`

**Exhausted P1E execution-authority commit:**

`21c298b97c7f33a117ff2be667f640e054cc8125`

**Validated P1 implementation commit:**

`2e6bb106d5d3081b7ae69ec4cde652e79d36070c`

This document freezes the bounded numerical-support recovery diagnostic protocol and authorizes implementation plus fabricated synthetic validation.

It does **not** authorize a second scientific-population forward or recurrent-state read.

A separately frozen R1 diagnostic execution authority is required before any second scientific-population access.

It does not modify the P1 scientific estimand.

## 1. Recovery objective

The sole objective is:

> Determine the numerical mechanism of the frozen `VELOCITY_REARRANGEMENT_TOLERANCE_FAILURE` at the smallest scientifically sufficient scope, without computing any P1 endpoint and without selecting a replacement tolerance from observed scientific data.

This is an instrument/root-cause diagnostic.

It is not a P1 scientific rerun.

## 2. Frozen incident facts

The first P1E scientific attempt:

`P1E_FIRST_SCIENTIFIC_ATTEMPT_STATUS = FAILED`

Failure exception:

`VELOCITY_REARRANGEMENT_TOLERANCE_FAILURE`

Failure class:

`VELOCITY_REARRANGEMENT_NUMERICAL_SUPPORT_FAILURE`

At the failing record, the observer had already passed the exact recurrence reconstruction check before entering the velocity-rearrangement tolerance check.

Therefore:

`RECURRENCE_EXACT_RECONSTRUCTION_AT_FAILING_RECORD = PASSED_BEFORE_FAILURE`

The original P1E authority is exhausted.

`P1E_AUTHORITY_EXHAUSTED = YES`

`AUTOMATIC_RERUN_AUTHORIZED = NO`

## 3. Smallest sufficient recovery scope

The frozen P1 runner processes scientific items in ascending local-template order beginning at `0`.

Within one item it processes the matched pair before the swapped pair.

Within the matched pair it captures the correction branch before the control branch.

The observed traceback terminated inside the first `capture_branch` call reached through the matched `run_pair`.

Therefore the first failing execution is localized to:

`R1_LOCAL_TEMPLATE_INDEX = 0`

`R1_PAIR_ROLE = MATCHED`

`R1_BRANCH_ROLE = MATCHED_CORR`

`R1_PRIMARY_LAYER = 23`

The exact failing token coordinate among the 9 frozen target coordinates was not emitted.

The R1 diagnostic scope is therefore exactly:

- local template index:
  `0`;
- stable scientific item:
  the frozen P0 item at local index `0`;
- branch:
  `matched_corr`;
- layer used for diagnostic interpretation:
  `23`;
- target coordinates:
  `t_e-1, t_e, ..., t_e+7`;
- scientific model forwards:
  exactly `1`;
- endpoint computations:
  exactly `0`.

No other item or branch may be forwarded under the first R1 diagnostic execution authority.

## 4. Exact P0 binding

Archive directory:

`reports/longterm_k0_rvg_p0_state_blind_provisioning_421d798_v1`

`P0_CANDIDATE_POOL_SHA256 = 743657411af4e143931e4d2c79f17043134bff3a370d30910588504c7f19f246`

`P0_GENERATED_SOURCE_SHA256 = 8137c0020a040faaf0c6be833b123e143dc5a0a09ae092e8e99530bb8671c1bf`

`P0_PHASE_PAIR_MAPPING_SHA256 = c5e1fa2ac946d153821896d5153354fae6e17038a55be87b3d31ab43d2921eda`

`P0_TOKEN_CONTRACTS_SHA256 = 6eb006f7deca28affa73318887421879e1279fd6897fab857b460873add9a998`

`P0_PROVISIONING_MANIFEST_SHA256 = feab9c60e3546ace3258f068e38d5bb577fc63b803b95349379fa8b4db425e52`

`P0_VALIDATION_REPORT_SHA256 = ae2fe4d1db13e1765415eab9c95263aec618fccf9299fe93fad0c8e31141b73a`

R1 must reconstruct item 0 and its matched correction branch from these exact archived bytes.

No regeneration is permitted.

## 5. Frozen P1 / observer binding

P1 runner:

`scripts/longterm_k0_rvg_p1_raw_vector_execution.py`

`P1_RUNNER_SHA256 = 2bd59a25a6303dc86c36c9438296b42197cc67019e9d4e9ee2fbbe1d832ec6eb`

P1 test:

`tests/test_longterm_k0_rvg_p1_raw_vector_execution.py`

`P1_TEST_SHA256 = 06c30c974029126fdc853a235071376738cc2769409369ccdb60f0878bac7706`

Frozen observer:

`scripts/longterm_k0_rvg_raw_recurrence_observer.py`

`OBSERVER_SHA256 = 12542e32d49b368e727de782b0cb833991464503b1fab56d375f15ec58649c25`

Observer Git blob:

`f2dbdfe52661eca384897578ab272e602e36deac`

R1 must not modify the frozen P1 runner or observer.

## 6. Exact implementation scope after R1 freeze

After this exact document is frozen, implementation may create exactly two new files:

`scripts/longterm_k0_rvg_p1e_r1_numerical_support_diagnostic.py`

`tests/test_longterm_k0_rvg_p1e_r1_numerical_support_diagnostic.py`

No existing repository file may be modified.

Historical untracked K1 files must remain untouched:

`scripts/longterm_k1_native_state_kinematics.py`

`tests/test_longterm_k1_native_state_kinematics.py`

## 7. Implementation-stage access boundary

During R1 implementation and validation:

`R1_SYNTHETIC_MODEL_FORWARD_AUTHORIZED = YES`

`R1_SYNTHETIC_RECURRENT_STATE_READ_AUTHORIZED = YES`

`R1_P0_ARTIFACT_STATIC_READ_AUTHORIZED = YES`

`R1_SCIENTIFIC_MODEL_FORWARD_AUTHORIZED = NO`

`R1_SCIENTIFIC_RECURRENT_STATE_READ_AUTHORIZED = NO`

`R1_SCIENTIFIC_ENDPOINT_COMPUTATION_AUTHORIZED = NO`

Fabricated synthetic text may be used to validate the diagnostic implementation.

The frozen scientific item 0 may be parsed/tokenized statically but may not be forwarded through the model during implementation validation.

## 8. Future R1 diagnostic execution scope

A later separately frozen R1 execution authority may authorize exactly one model forward for:

`local_template_index = 0`

`branch_role = matched_corr`

No correction-control pair is required.

No swapped branch is required.

No reciprocal phase mate is required.

The future R1 diagnostic execution must not forward:

- item 0 matched control;
- item 0 swapped correction;
- item 0 swapped control;
- any item `1..335`.

## 9. Target-coordinate contract

The diagnostic must authenticate the frozen token contract for local item `0`.

Let the frozen matched divergence anchor be:

`t_e`.

The diagnostic target coordinates are exactly:

`t_e-1`

and:

`t_e, t_e+1, ..., t_e+7`.

Exactly 9 target token coordinates are permitted.

The diagnostic may use the frozen observer collector to capture all registered Mamba layers required by the existing instrumentation contract, but all non-layer-23 tensors must be discarded immediately after capture-completeness validation.

No non-layer-23 numerical interpretation is authorized.

## 10. Frozen recurrence quantities

For each layer-23 target record, let:

`S_prev = S_(t-1)`

`G = discrete_A`

`W = deltaB_u`

`S_post = S_t`

The diagnostic must first evaluate the exact native recurrence:

`R_exact = G * S_prev + W`

and record:

`torch.equal(R_exact, S_post)`.

No record may be silently dropped.

## 11. Frozen velocity quantities

For each target record:

`V_raw32 = S_post - S_prev`

`V_rearr32 = (G - 1) * S_prev + W`

`D32 = V_raw32 - V_rearr32`

The original frozen tolerance remains:

`VELOCITY_ATOL = 1e-6`

`VELOCITY_RTOL = 1e-5`

The diagnostic records whether:

`torch.allclose(V_raw32, V_rearr32, atol=1e-6, rtol=1e-5)`

passes or fails.

The diagnostic must not raise on this condition.

It must record the result for all 9 coordinates.

## 12. Required per-coordinate numerical diagnostics

For each of the 9 layer-23 records, the diagnostic must record at least:

- token index;
- exact recurrence pass/fail;
- frozen allclose pass/fail;
- `max_abs_residual = max(abs(D32))`;
- Frobenius norm of `D32`;
- Frobenius norm of `V_raw32`;
- Frobenius norm of `V_rearr32`;
- relative Frobenius residual:
  `||D32||_F / max(||V_raw32||_F, 1e-12)`;
- frozen elementwise tolerance scale:
  `1e-6 + 1e-5 * abs(V_rearr32)`;
- maximum scaled-tolerance residual;
- count of elements whose absolute residual exceeds the frozen elementwise tolerance;
- fraction of elements whose residual exceeds tolerance;
- `||S_prev||_F`;
- `||S_post||_F`;
- `||W||_F`;
- `||(G-1)*S_prev||_F`;
- state-to-raw-velocity norm ratio:
  `||S_prev||_F / max(||V_raw32||_F, 1e-12)`.

These are diagnostic quantities only.

## 13. Maximum-residual element audit

For each target coordinate, identify the deterministic flattened index of the maximum absolute residual.

At that element, record the float32 scalar values:

- `S_prev`;
- `S_post`;
- `G`;
- `W`;
- `V_raw32`;
- `V_rearr32`;
- `D32`;
- frozen tolerance scale;
- scaled residual.

This is required to diagnose subtractive cancellation / rounding scale.

No semantic interpretation of tensor channels is authorized.

## 14. Float64 snapshot diagnostic

Using only the captured float32 snapshots converted to float64, compute:

`V_raw64 = float64(S_post) - float64(S_prev)`

`V_rearr64 = (float64(G) - 1) * float64(S_prev) + float64(W)`

`D64 = V_raw64 - V_rearr64`.

Record:

- `max_abs(D64)`;
- `||D64||_F`;
- `||V_raw64||_F`;
- relative Frobenius residual.

This is a numerical diagnostic only.

It does not redefine the P1 primary velocity or authorize float64 state geometry.

## 15. Diagnostic classification rules

The diagnostic implementation may emit descriptive machine labels using only these prospectively frozen rules.

### Exact recurrence corruption

If any target record fails:

`torch.equal(G*S_prev + W, S_post)`

then:

`R1_CLASSIFICATION = EXACT_RECURRENCE_OR_CAPTURE_FAILURE`

### Frozen tolerance failure with exact recurrence intact

If all 9 exact recurrence checks pass and at least one frozen allclose check fails:

`R1_CLASSIFICATION = EXACT_RECURRENCE_INTACT_FLOAT32_REARRANGEMENT_TOLERANCE_FAILURE`

### No reproduced failure

If all 9 exact recurrence checks pass and all 9 frozen allclose checks pass:

`R1_CLASSIFICATION = ORIGINAL_FAILURE_NOT_REPRODUCED_IN_FIXED_SINGLE_BRANCH_DIAGNOSTIC`

No other classification may be promoted without a later authority.

## 16. Local support-profile rule

Within the fixed 9-coordinate scope, record:

- failing coordinate count;
- passing coordinate count;
- first failing token index;
- maximum scaled residual across all 9;
- median scaled residual across all 9.

Descriptive local labels:

If failing count equals `1`:

`R1_LOCAL_SUPPORT_PROFILE = SINGLE_COORDINATE_FAILURE_WITHIN_FIXED_BRANCH`

If failing count is greater than `1`:

`R1_LOCAL_SUPPORT_PROFILE = MULTI_COORDINATE_FAILURE_WITHIN_FIXED_BRANCH`

If failing count equals `0`:

`R1_LOCAL_SUPPORT_PROFILE = NO_FAILURE_REPRODUCED_WITHIN_FIXED_BRANCH`

These labels do not estimate population-wide failure prevalence.

## 17. No tolerance selection

R1 must not:

- sweep `atol`;
- sweep `rtol`;
- calculate a recommended new tolerance from scientific residuals;
- calculate a percentile-derived tolerance;
- calculate the minimum passing tolerance;
- modify the frozen observer constants;
- report an endpoint-conditioned tolerance.

`R1_REPLACEMENT_TOLERANCE_SELECTION_AUTHORIZED = NO`

## 18. No scientific endpoints

The R1 implementation must contain no code path that computes or emits:

- `A_corr`;
- `A_ctrl`;
- `T_M`;
- `T_S`;
- `X_turn`;
- `C_M`;
- `C_S`;
- `X_coh`;
- phase-block scores;
- sign-test p-values;
- Holm p-values;
- overall raw-vector verdict.

`R1_P1_ENDPOINT_COMPUTATION_AUTHORIZED = NO`

## 19. No scientific interpretation

R1 cannot establish:

- turning signal;
- response-coherence signal;
- raw-vector organization;
- endpoint support adequacy;
- Branch A;
- Branch B;
- causal carry/write specialization;
- K4.

`SCIENTIFIC_CONCLUSION_FROM_R1 = NONE`

## 20. Diagnostic output artifact

A future authorized R1 diagnostic execution must write exactly one diagnostic file to a fresh external directory:

`numerical_support_diagnostic.json`

No P1 scientific result artifact names may be reused.

The diagnostic JSON must contain:

- schema version;
- authority/provenance identities;
- local item identity;
- branch role;
- matched divergence anchor;
- 9-coordinate diagnostics;
- fixed classification;
- local support profile;
- explicit flags:
  - P1 endpoints computed:
    `false`;
  - logits read:
    `false`;
  - causal intervention:
    `false`;
  - tolerance sweep:
    `false`;
  - scientific result interpretation:
    `false`.

## 21. Raw tensor persistence boundary

Full recurrent tensors must not be persisted.

The diagnostic output may persist SHA256 hashes of the four recurrence roles for each of the 9 layer-23 coordinates:

- `S_prev`;
- `G`;
- `W`;
- `S_post`.

No raw tensor dump is authorized.

## 22. Required focused tests

The R1 implementation test suite must cover at least:

1. exact frozen authority/provenance constants;
2. fixed local item index `0`;
3. fixed branch role `matched_corr`;
4. exact 9-coordinate target construction;
5. exact recurrence pass/fail diagnostic without raising;
6. frozen allclose pass/fail diagnostic without raising;
7. elementwise tolerance-scale computation;
8. max scaled-residual computation;
9. failing-element count/fraction;
10. Frobenius residuals;
11. state-to-velocity norm ratio;
12. deterministic maximum-residual flattened index;
13. float64 snapshot diagnostic;
14. fixed R1 classification rules;
15. fixed local support-profile rules;
16. no tolerance sweep interface;
17. no layer/window/item/branch CLI override;
18. no P1 endpoint implementation/import;
19. diagnostic JSON schema;
20. JSON null / NaN rejection;
21. raw tensor non-persistence;
22. synthetic fabricated model integration;
23. deterministic repeated synthetic diagnostic;
24. scientific execution gate rejection when authority absent;
25. no import-time model execution.

## 23. Fabricated synthetic validation

Implementation validation may use fabricated non-study text only.

Synthetic validation must exercise:

- one branch forward;
- exact 9-coordinate capture;
- layer-23 recurrence extraction;
- exact recurrence diagnostic;
- frozen allclose diagnostic;
- a fabricated tensor fixture that passes frozen tolerance;
- a fabricated tensor fixture that fails frozen tolerance while exact recurrence remains intact;
- float64 snapshot diagnostics;
- classification;
- deterministic repeated state hashes / diagnostics.

Synthetic validation must not forward any frozen P0 scientific branch.

## 24. Future R1 execution-authority boundary

After:

1. this R1 authority/specification is frozen;
2. the exact two-file R1 implementation is frozen;
3. focused tests pass;
4. fabricated synthetic validation passes;
5. an R1 implementation readiness report is frozen;

a later document may be drafted:

`K0-RVG-P1E-R1E — Fixed Single-Branch Numerical-Support Diagnostic Execution Authority`

Only R1E may authorize the second scientific-population model forward / recurrent-state read.

R1E must authorize exactly one forward for local item 0 matched_corr and no other scientific access.

## 25. Branch state

`K_SUCCESSOR_BRANCH_SELECTION = DEFERRED`

`BRANCH_A_STATUS = PARKED_NOT_CLOSED`

`BRANCH_B_STATUS = PARKED_NOT_ACTIVE`

## 26. K4 boundary

`K4_EXECUTION_AUTHORIZED = NO`

## 27. Final authority markers

`P1E_R1_PROTOCOL_FROZEN = YES`

`P1E_R1_IMPLEMENTATION_AUTHORIZED = YES`

`P1E_R1_IMPLEMENTATION_SCOPE = TWO_NEW_FILES_ONLY`

`R1_LOCAL_TEMPLATE_INDEX = 0`

`R1_BRANCH_ROLE = MATCHED_CORR`

`R1_PRIMARY_LAYER = 23`

`R1_TARGET_COORDINATE_COUNT = 9`

`R1_SYNTHETIC_MODEL_FORWARD_AUTHORIZED = YES`

`R1_SYNTHETIC_RECURRENT_STATE_READ_AUTHORIZED = YES`

`R1_P0_ARTIFACT_STATIC_READ_AUTHORIZED = YES`

`R1_SCIENTIFIC_MODEL_FORWARD_AUTHORIZED = NO`

`R1_SCIENTIFIC_RECURRENT_STATE_READ_AUTHORIZED = NO`

`R1_SCIENTIFIC_ENDPOINT_COMPUTATION_AUTHORIZED = NO`

`R1_P1_ENDPOINT_COMPUTATION_AUTHORIZED = NO`

`R1_REPLACEMENT_TOLERANCE_SELECTION_AUTHORIZED = NO`

`SCIENTIFIC_CONCLUSION_FROM_R1 = NONE`

`LOGITS_READ_AUTHORIZED = NO`

`CAUSAL_INTERVENTION_AUTHORIZED = NO`

`K4_EXECUTION_AUTHORIZED = NO`

`READY_FOR_P1E_R1_IMPLEMENTATION = YES`

`NEXT_STAGE = K0-RVG-P1E-R1_NUMERICAL_SUPPORT_DIAGNOSTIC_IMPLEMENTATION`

This authority becomes active only after this exact document is committed and pushed as the immediate one-file child of `6fe6a80f5fa0125971ac83d436e6e400cb7ab6f1`.
