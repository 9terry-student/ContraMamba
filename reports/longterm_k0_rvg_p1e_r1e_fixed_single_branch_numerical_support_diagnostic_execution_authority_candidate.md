# K0-RVG-P1E-R1E Fixed Single-Branch Numerical-Support Diagnostic Execution Authority Candidate

**Status:** one-time bounded numerical-support diagnostic execution authority candidate.

**Immediate parent / frozen R1 implementation validation report commit:**

`c00e95de1fc07290ec97a5c5f99e71f85c1dbc93`

**R1 implementation validation report SHA256:**

`eb64d5339b568a5ceff1dd3f6962aac037b5a0c7d488cf4a2dc200c3b467194b`

**Validated R1 implementation commit:**

`01ca5f5b11a4a5738ad8f77f97d33bdbf6c6bceb`

**Frozen R1 protocol commit:**

`02f184a186aa6b90fc98ece717723de0adafc89c`

This document is the sole authority that may open exactly one frozen scientific branch to the validated R1 numerical-support diagnostic.

It authorizes exactly one diagnostic model forward / recurrent-state read for local item 0, matched correction only.

It does not authorize a P1 scientific rerun.

It does not authorize P1 endpoint computation, tolerance selection, logits, causal intervention, branch selection, or K4.

## 1. Diagnostic execution decision

The R1 implementation has passed:

- exact two-file implementation scope;
- 38 focused tests before and after commit;
- fabricated synthetic integration;
- explicit exact-recurrence / frozen-tolerance failure fixture;
- non-layer-23 immediate discard validation;
- R1 execution-authority fail-closed validation;
- P0 / P1 / observer / K2S / A0 / handoff provenance checks.

Therefore:

`P1E_R1E_EXECUTION_AUTHORITY_FROZEN = YES`

`R1_SCIENTIFIC_MODEL_FORWARD_AUTHORIZED = YES`

`R1_SCIENTIFIC_RECURRENT_STATE_READ_AUTHORIZED = YES`

The authorization applies only to the fixed branch and exact implementation below.

## 2. Explicit prohibitions

`R1_SCIENTIFIC_ENDPOINT_COMPUTATION_AUTHORIZED = NO`

`R1_P1_ENDPOINT_COMPUTATION_AUTHORIZED = NO`

`R1_REPLACEMENT_TOLERANCE_SELECTION_AUTHORIZED = NO`

`R1_TOLERANCE_SWEEP_AUTHORIZED = NO`

`LOGITS_READ_AUTHORIZED = NO`

`CAUSAL_INTERVENTION_AUTHORIZED = NO`

`TRAINING_AUTHORIZED = NO`

`HYPERPARAMETER_TUNING_AUTHORIZED = NO`

`K4_EXECUTION_AUTHORIZED = NO`

No code modification is authorized by this execution authority.

No scientific endpoint or scientific parameter override is authorized.

## 3. Exact R1 implementation binding

`R1_IMPLEMENTATION_COMMIT = 01ca5f5b11a4a5738ad8f77f97d33bdbf6c6bceb`

`R1_RUNNER_SHA256 = d5a8e8c5e3c7ef5855d12a0284e14b6eaef9f5991a21823025429b03757d5b26`

`R1_TEST_SHA256 = cc7b1c364fb050642fbe9f0b7e8c4f1f825322682ae4b430bfb09b9c93ee4979`

Runner:

`scripts/longterm_k0_rvg_p1e_r1_numerical_support_diagnostic.py`

Test:

`tests/test_longterm_k0_rvg_p1e_r1_numerical_support_diagnostic.py`

The implementation commit must remain the exact direct two-file child of:

`02f184a186aa6b90fc98ece717723de0adafc89c`

No runner/test blob drift is permitted.

## 4. R1 protocol binding

`R1_PROTOCOL_COMMIT = 02f184a186aa6b90fc98ece717723de0adafc89c`

`R1_PROTOCOL_SHA256 = d9a10812c96d9453460356658d09f282a6f4da022b299ce9ceab99cb098f2fb0`

The R1 diagnostic estimand and diagnostic classification rules remain frozen.

## 5. Exact fixed scientific access scope

`R1_LOCAL_TEMPLATE_INDEX = 0`

`R1_BRANCH_ROLE = MATCHED_CORR`

`R1_PRIMARY_LAYER = 23`

`R1_TARGET_COORDINATE_COUNT = 9`

Authorized scientific model forwards:

`1`

Authorized scientific branch:

`local item 0 / matched_corr only`

Authorized target coordinates:

`t_e-1, t_e, ..., t_e+7`

No other branch may be model-forwarded.

No other scientific item may be model-forwarded.

Specifically prohibited:

- item 0 matched control;
- item 0 swapped correction;
- item 0 swapped control;
- items 1 through 335.

## 6. Exact P0 archive binding

Archive:

`reports/longterm_k0_rvg_p0_state_blind_provisioning_421d798_v1`

`P0_CANDIDATE_POOL_SHA256 = 743657411af4e143931e4d2c79f17043134bff3a370d30910588504c7f19f246`

`P0_GENERATED_SOURCE_SHA256 = 8137c0020a040faaf0c6be833b123e143dc5a0a09ae092e8e99530bb8671c1bf`

`P0_PHASE_PAIR_MAPPING_SHA256 = c5e1fa2ac946d153821896d5153354fae6e17038a55be87b3d31ab43d2921eda`

`P0_TOKEN_CONTRACTS_SHA256 = 6eb006f7deca28affa73318887421879e1279fd6897fab857b460873add9a998`

`P0_PROVISIONING_MANIFEST_SHA256 = feab9c60e3546ace3258f068e38d5bb577fc63b803b95349379fa8b4db425e52`

`P0_VALIDATION_REPORT_SHA256 = ae2fe4d1db13e1765415eab9c95263aec618fccf9299fe93fad0c8e31141b73a`

Item 0 must be reconstructed from these exact archived bytes.

No regeneration or replacement is authorized.

## 7. Frozen P1 provenance

P1 implementation commit:

`2e6bb106d5d3081b7ae69ec4cde652e79d36070c`

P1 runner SHA256:

`2bd59a25a6303dc86c36c9438296b42197cc67019e9d4e9ee2fbbe1d832ec6eb`

P1 test SHA256:

`06c30c974029126fdc853a235071376738cc2769409369ccdb60f0878bac7706`

R1E does not reopen the P1 full-population execution path.

## 8. Observer/runtime binding

`OBSERVER_SHA256 = 12542e32d49b368e727de782b0cb833991464503b1fab56d375f15ec58649c25`

Observer commit:

`fcfe161c12f4ed8ef37aff435554cc0660e477af`

Observer Git blob:

`f2dbdfe52661eca384897578ab272e602e36deac`

K2S helper SHA256:

`f741780e7199452e64b7c4a3d70f50f3e55ebc84296f69f288aa317582de84b8`

K2S helper Git blob:

`3a651fb508669bdcf72441a4869b863d6eee6c1f`

Model:

`state-spaces/mamba-130m-hf`

Revision:

`5708daa364c50b880e7bd92eab456e0d34492ee9`

Transformers:

`5.12.1`

Mamba source SHA256:

`23c7b410e204b5da01732566de10c94b70a8418ecb608e409754b00332eb2a41`

## 9. A0 / handoff binding

A0 commit:

`55debe94f0d19d16a334395e8561901fed6b52fa`

A0 model blob:

`f0ddc0eda64937de6fcd27943e30a296082c01d5`

A0 heads tree:

`68d26855aa511fcd41d6f395ae5f87177a162678`

Seed180 handoff ZIP SHA256:

`96859bad3e400613b4c981990e56aaf35b1d92baaaa930eb11448cf38a63b861`

Checkpoint SHA256:

`4f7ad019bddb988a534c477b58b36bdabe2775d6c9748331e8311653c07c864c`

Encoder canonical SHA256:

`48a7e9ac9dfa6c8c292090ee0fcb606bd4c85d13706bfc8a3e371af77c440597`

Encoder raw-concat SHA256:

`968c12c095a6aab883db5984f4c02ad5e893a5ff140781ffbda41b97970401ae`

## 10. Diagnostic runtime environment

Authorized runtime:

`LOCAL_CPU_SEQUENTIAL_MAMBA_ONLY`

GPU execution is not authorized.

Kaggle is not required.

The same validated sequential CPU runtime path must be used.

## 11. Frozen numerical diagnostic contract

The original frozen velocity tolerance remains:

`VELOCITY_ATOL = 1e-6`

`VELOCITY_RTOL = 1e-5`

For each of the nine layer-23 coordinates, R1 records:

- exact recurrence equality;
- frozen allclose pass/fail;
- max absolute residual;
- Frobenius residual;
- relative Frobenius residual;
- tolerance scale;
- max scaled tolerance residual;
- failing element count/fraction;
- relevant tensor norms;
- state-to-velocity norm ratio;
- maximum-residual element scalar audit;
- float64 snapshot diagnostic;
- recurrence-role SHA256 identities.

No replacement tolerance is computed.

## 12. Classification boundary

The only authorized classification hierarchy is:

1. `EXACT_RECURRENCE_OR_CAPTURE_FAILURE`;
2. `EXACT_RECURRENCE_INTACT_FLOAT32_REARRANGEMENT_TOLERANCE_FAILURE`;
3. `ORIGINAL_FAILURE_NOT_REPRODUCED_IN_FIXED_SINGLE_BRANCH_DIAGNOSTIC`.

The only authorized local support-profile labels are:

- `SINGLE_COORDINATE_FAILURE_WITHIN_FIXED_BRANCH`;
- `MULTI_COORDINATE_FAILURE_WITHIN_FIXED_BRANCH`;
- `NO_FAILURE_REPRODUCED_WITHIN_FIXED_BRANCH`.

These are diagnostic labels only.

They are not P1 scientific endpoint conclusions.

## 13. Non-layer-23 boundary

The collector may capture all registered layers only to satisfy the frozen capture-completeness contract.

Immediately after completeness validation:

- only the nine layer-23 record references may remain live;
- the full capture mapping must be cleared;
- no non-layer-23 numerical interpretation is authorized.

## 14. Raw-state persistence boundary

No raw recurrent tensor may be persisted.

The diagnostic artifact may persist only numerical diagnostics and SHA256 hashes for layer-23:

- `S_prev`;
- `G`;
- `W`;
- `S_post`.

## 15. Exact output artifact contract

A successful diagnostic run may create exactly one final file:

`numerical_support_diagnostic.json`

The output directory must be external to the repository and must not already exist.

Atomic finalization is required.

No P1 result-artifact filename may be reused.

## 16. One-attempt rule

This authority permits exactly one R1 diagnostic execution attempt.

`R1E_AUTHORIZED_DIAGNOSTIC_ATTEMPTS = 1`

If execution exits nonzero, crashes, is interrupted, or fails an internal contract:

`R1E_AUTOMATIC_RERUN_AUTHORIZED = NO`

Stop immediately.

Do not rerun the command under this authority.

Do not change tolerance values.

A new recovery authority is required before another scientific branch forward.

## 17. Launch-gate requirement

Before the diagnostic command is run, all of the following must pass at the frozen R1E authority HEAD:

1. branch is exactly:
   `longterm-k-series-native-state-kinematics`;
2. R1E authority file is tracked and worktree-identical to HEAD;
3. R1 runner/test bytes equal the frozen SHA256 values;
4. R1 implementation commit is the exact two-file child of the R1 protocol;
5. six P0 hashes match;
6. observer/K2S/A0/runtime provenance matches;
7. 38 R1 focused tests pass;
8. fabricated synthetic R1 preflight passes;
9. fabricated preflight confirms:
   - scientific population model forward:
     `false`;
   - scientific population recurrent-state read:
     `false`;
   - P1 endpoint computation:
     `false`;
   - replacement tolerance selected:
     `false`;
10. repository tracked state is clean;
11. only historical K1 untracked files remain;
12. future R1 diagnostic output directory is fresh.

The launch gate itself must not forward the frozen scientific item 0.

## 18. Exact command shape

After this authority is frozen and the launch gate passes, the authorized diagnostic invocation is exactly equivalent to:

`python scripts/longterm_k0_rvg_p1e_r1_numerical_support_diagnostic.py --execute-diagnostic --seed180-handoff <EXACT_HANDOFF_PATH> --execution-authority reports/longterm_k0_rvg_p1e_r1e_fixed_single_branch_numerical_support_diagnostic_execution_authority_candidate.md --output-dir <FRESH_OUTPUT_DIR>`

No additional CLI flag is authorized.

## 19. Scientific-result boundary

R1E is an instrument/root-cause diagnostic.

It does not establish:

- `X_turn`;
- `X_coh`;
- any phase-block score;
- sign-test significance;
- Holm significance;
- raw-vector scientific verdict;
- Branch A;
- Branch B;
- causal carry/write specialization;
- K4.

`SCIENTIFIC_CONCLUSION_FROM_R1E_AUTHORITY = NONE`

## 20. Interpretation boundary

A successful diagnostic process exit does not by itself establish the root cause.

After execution:

1. the single diagnostic artifact must be authenticated;
2. provenance must be validated;
3. diagnostic fields must be checked against the frozen schema;
4. only then may a numerical-support interpretation report be written.

`R1E_INTERPRETATION_BEFORE_ARTIFACT_VALIDATION = NO`

## 21. Branch and K4 boundary

`K_SUCCESSOR_BRANCH_SELECTION = DEFERRED`

`BRANCH_A_STATUS = PARKED_NOT_CLOSED`

`BRANCH_B_STATUS = PARKED_NOT_ACTIVE`

`K4_EXECUTION_AUTHORIZED = NO`

## 22. Next boundary after successful diagnostic execution

If the exact single diagnostic artifact is atomically finalized:

`NEXT_BOUNDARY = K0-RVG-P1E-R1_DIAGNOSTIC_ARTIFACT_VALIDATION`

Do not modify the P1 observer, P1 runner, R1 runner, or tolerance constants before validation.

## 23. Final authority markers

`P1E_R1E_EXECUTION_AUTHORITY_FROZEN = YES`

`R1_IMPLEMENTATION_COMMIT = 01ca5f5b11a4a5738ad8f77f97d33bdbf6c6bceb`

`R1_RUNNER_SHA256 = d5a8e8c5e3c7ef5855d12a0284e14b6eaef9f5991a21823025429b03757d5b26`

`R1_TEST_SHA256 = cc7b1c364fb050642fbe9f0b7e8c4f1f825322682ae4b430bfb09b9c93ee4979`

`R1_PROTOCOL_SHA256 = d9a10812c96d9453460356658d09f282a6f4da022b299ce9ceab99cb098f2fb0`

`OBSERVER_SHA256 = 12542e32d49b368e727de782b0cb833991464503b1fab56d375f15ec58649c25`

`P0_CANDIDATE_POOL_SHA256 = 743657411af4e143931e4d2c79f17043134bff3a370d30910588504c7f19f246`

`P0_GENERATED_SOURCE_SHA256 = 8137c0020a040faaf0c6be833b123e143dc5a0a09ae092e8e99530bb8671c1bf`

`P0_PHASE_PAIR_MAPPING_SHA256 = c5e1fa2ac946d153821896d5153354fae6e17038a55be87b3d31ab43d2921eda`

`P0_TOKEN_CONTRACTS_SHA256 = 6eb006f7deca28affa73318887421879e1279fd6897fab857b460873add9a998`

`P0_PROVISIONING_MANIFEST_SHA256 = feab9c60e3546ace3258f068e38d5bb577fc63b803b95349379fa8b4db425e52`

`P0_VALIDATION_REPORT_SHA256 = ae2fe4d1db13e1765415eab9c95263aec618fccf9299fe93fad0c8e31141b73a`

`R1_LOCAL_TEMPLATE_INDEX = 0`

`R1_BRANCH_ROLE = MATCHED_CORR`

`R1_PRIMARY_LAYER = 23`

`R1_TARGET_COORDINATE_COUNT = 9`

`R1_SCIENTIFIC_MODEL_FORWARD_AUTHORIZED = YES`

`R1_SCIENTIFIC_RECURRENT_STATE_READ_AUTHORIZED = YES`

`R1_SCIENTIFIC_ENDPOINT_COMPUTATION_AUTHORIZED = NO`

`R1_P1_ENDPOINT_COMPUTATION_AUTHORIZED = NO`

`R1_REPLACEMENT_TOLERANCE_SELECTION_AUTHORIZED = NO`

`R1_TOLERANCE_SWEEP_AUTHORIZED = NO`

`LOGITS_READ_AUTHORIZED = NO`

`CAUSAL_INTERVENTION_AUTHORIZED = NO`

`R1E_AUTHORIZED_DIAGNOSTIC_ATTEMPTS = 1`

`R1E_AUTOMATIC_RERUN_AUTHORIZED = NO`

`SCIENTIFIC_CONCLUSION_FROM_R1E_AUTHORITY = NONE`

`K_SUCCESSOR_BRANCH_SELECTION = DEFERRED`

`BRANCH_A_STATUS = PARKED_NOT_CLOSED`

`BRANCH_B_STATUS = PARKED_NOT_ACTIVE`

`K4_EXECUTION_AUTHORIZED = NO`

`NEXT_BOUNDARY = K0-RVG-P1E-R1_DIAGNOSTIC_ARTIFACT_VALIDATION`

This authority becomes active only after this exact document is committed and pushed as the immediate one-file child of `c00e95de1fc07290ec97a5c5f99e71f85c1dbc93`.
