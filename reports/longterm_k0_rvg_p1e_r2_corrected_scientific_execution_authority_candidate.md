# K0-RVG-P1E-R2 Corrected Scientific Raw-Vector Execution Authority Candidate

**Status:** one-time corrected scientific execution authority candidate.

**Date:** 2026-09-12

**Immediate parent / frozen R2 implementation-readiness report commit:**

`795c15d622d568377b36397f235f9e302f719f91`

**Frozen R2 implementation-readiness report:**

`reports/longterm_k0_rvg_p1e_r2_implementation_validation_readiness_report_candidate.md`

**Frozen R2 implementation-readiness report Git blob:**

`9f6bb66821cca21dc47dc01095f0272838e2e708`

**Corrected P1 implementation commit:**

`50a1daa781e47d1c0f1ba158beb445878e049a65`

**Frozen R2 numerical-guard correction authority commit:**

`d8d717a09516b8562f64f7c503d4bbdcc9c34c5d`

**Frozen R2 numerical-guard correction authority SHA256:**

`ae8e910e6ac34cb8550bb7e5b979068c409560bc0648f6ef16a686b2d999a8c3`

**Original failed P1E execution authority:**

`21c298b97c7f33a117ff2be667f640e054cc8125`

This document is the sole corrected authority that may reopen the frozen P0 scientific population to the corrected P1 runner after the R2 numerical-guard defect correction.

It authorizes exactly one new, provenance-distinct scientific execution attempt starting from item 0.

It does not authorize reuse, continuation, resumption, repair, or interpretation of the original failed P1E partial execution.

It does not authorize exploratory reruns, layer/window changes, tolerance changes, logits, learned geometry, causal intervention, training, hyperparameter tuning, K4, or branch promotion.

## 1. Corrected execution decision

The frozen R2 correction lineage has established:

- original fixed failure root cause:
  `FLOAT32_INTERMEDIATE_ROUNDING_UNDER_CANCELLING_REARRANGEMENT`;
- exact native recurrence remained valid for the fixed failure record;
- the prior hard rearrangement-allclose gate was numerically overconstraining;
- the primary raw velocity definition was not invalidated;
- post-observation tolerance retuning is forbidden;
- the corrected implementation changes exactly four authorized files;
- focused corrected test suite:
  `69 passed`;
- fabricated synthetic integration:
  `PASS_SYNTHETIC_P1_RAW_VECTOR_RUNNER`;
- independent corrected implementation review:
  `PASS`;
- protected scientific-estimand AST equivalence:
  `PASS`;
- fabricated endpoint compatibility:
  `PASS`;
- corrected implementation provenance:
  `PASS`;
- R2 implementation readiness:
  `READY_FOR_CORRECTED_P1E_EXECUTION_AUTHORITY_DRAFT = YES`.

Therefore, once this exact authority document is frozen as specified below:

`P1E_R2_EXECUTION_AUTHORITY_FROZEN = YES`

`SCIENTIFIC_EXECUTION_AUTHORIZED = YES`

`SCIENTIFIC_MODEL_FORWARD_AUTHORIZED = YES`

`SCIENTIFIC_RECURRENT_STATE_READ_AUTHORIZED = YES`

`SCIENTIFIC_ENDPOINT_COMPUTATION_AUTHORIZED = YES`

The authorization applies only to the exact frozen P0 population, the exact corrected implementation, and the exact provenance identities bound below.

## 2. Explicit prohibitions

`LOGITS_READ_AUTHORIZED = NO`

`LEARNED_OR_TUNED_GEOMETRY_AUTHORIZED = NO`

`OUTCOME_SELECTED_SUBSPACE_AUTHORIZED = NO`

`CAUSAL_INTERVENTION_AUTHORIZED = NO`

`TRAINING_AUTHORIZED = NO`

`HYPERPARAMETER_TUNING_AUTHORIZED = NO`

`K4_EXECUTION_AUTHORIZED = NO`

`POST_OBSERVATION_TOLERANCE_RETUNING_AUTHORIZED = NO`

`PARTIAL_FAILED_RUN_REUSE_AUTHORIZED = NO`

`SCIENTIFIC_RESUME_AUTHORIZED = NO`

No code modification is authorized by this execution authority.

No scientific parameter override is authorized.

No observed R1 or future P1 residual may be used to alter a threshold.

## 3. Exact corrected implementation binding

`P1_IMPLEMENTATION_COMMIT = 50a1daa781e47d1c0f1ba158beb445878e049a65`

`P1_OBSERVER_SHA256 = 9cb3b49a9b516def1fe50d90be1d57ddc160495743f5abff20e61684d2b34ffe`

`P1_OBSERVER_TEST_SHA256 = f766b8aa69e76da0bb4dc098b75b5fb4d5943e160693b6dbb0e7251703302013`

`P1_RUNNER_SHA256 = 7384ab0b208476a1d4a6941882af266f08020c93b1a2c9bdb780ff02a2777c6e`

`P1_RUNNER_TEST_SHA256 = 48736aeaecc0aa09920869fe9aa465c00275e537025ce0a4265bb90620400461`

`R2_CORRECTION_AUTHORITY_SHA256 = ae8e910e6ac34cb8550bb7e5b979068c409560bc0648f6ef16a686b2d999a8c3`

Corrected observer path:

`scripts/longterm_k0_rvg_raw_recurrence_observer.py`

Corrected observer-test path:

`tests/test_longterm_k0_rvg_raw_recurrence_observer.py`

Corrected P1 runner path:

`scripts/longterm_k0_rvg_p1_raw_vector_execution.py`

Corrected P1 runner-test path:

`tests/test_longterm_k0_rvg_p1_raw_vector_execution.py`

The implementation commit must remain the exact direct four-file child of:

`d8d717a09516b8562f64f7c503d4bbdcc9c34c5d`

All four implementation blobs must remain identical to the frozen implementation commit.

No later blob drift is permitted.

## 4. R2 readiness binding

R2 implementation-readiness commit:

`795c15d622d568377b36397f235f9e302f719f91`

R2 implementation-readiness report:

`reports/longterm_k0_rvg_p1e_r2_implementation_validation_readiness_report_candidate.md`

R2 implementation-readiness report Git blob:

`9f6bb66821cca21dc47dc01095f0272838e2e708`

The readiness report records:

`R2_CODE_CORRECTNESS = PASS_FOR_FROZEN_CORRECTION_CONTRACT`

`R2_FOCUSED_TEST_CONTRACT = PASS_69`

`R2_SYNTHETIC_EXECUTION_SUCCESS = YES`

`R2_INDEPENDENT_IMPLEMENTATION_REVIEW = PASS`

`R2_IMPLEMENTATION_PROVENANCE_VALID = YES`

`R2_SCIENTIFIC_ESTIMAND_PROTECTION = PASS`

`R2_IMPLEMENTATION_VALIDATED = YES`

`READY_FOR_CORRECTED_P1E_EXECUTION_AUTHORITY_DRAFT = YES`

This execution authority may not be used if that readiness lineage is not an ancestor of the execution-authority HEAD.

## 5. Exact P0 scientific population binding

Archive directory:

`reports/longterm_k0_rvg_p0_state_blind_provisioning_421d798_v1`

`P0_CANDIDATE_POOL_SHA256 = 743657411af4e143931e4d2c79f17043134bff3a370d30910588504c7f19f246`

`P0_GENERATED_SOURCE_SHA256 = 8137c0020a040faaf0c6be833b123e143dc5a0a09ae092e8e99530bb8671c1bf`

`P0_PHASE_PAIR_MAPPING_SHA256 = c5e1fa2ac946d153821896d5153354fae6e17038a55be87b3d31ab43d2921eda`

`P0_TOKEN_CONTRACTS_SHA256 = 6eb006f7deca28affa73318887421879e1279fd6897fab857b460873add9a998`

`P0_PROVISIONING_MANIFEST_SHA256 = feab9c60e3546ace3258f068e38d5bb577fc63b803b95349379fa8b4db425e52`

`P0_VALIDATION_REPORT_SHA256 = ae2fe4d1db13e1765415eab9c95263aec618fccf9299fe93fad0c8e31141b73a`

Authorized scientific population:

- item count:
  `336`;
- phase blocks:
  `168`;
- fresh template range:
  `[1236,1572)`;
- matched/swapped branch contract:
  exact frozen P0 token contracts;
- no item replacement;
- no item regeneration;
- no selective exclusion based on recurrence rearrangement diagnostic;
- no reuse of partial artifacts from the original failed P1E attempt.

## 6. Corrected recurrence-integrity contract

Exact native recurrence is the hard recurrence-integrity gate.

For every captured recurrence record:

`torch.equal(G * S_prev + W, S_post)`

must pass.

Exact recurrence mismatch remains a hard failure:

`RECURRENCE_EXACT_RECONSTRUCTION_FAILURE`

The corrected implementation must retain:

`V_raw = S_post - S_prev`

`V_carry = (G - 1) * S_prev`

`V_write = W`

`V_rearranged = V_carry + V_write`

Frozen descriptive tolerances remain exactly:

`VELOCITY_ATOL = 1e-6`

`VELOCITY_RTOL = 1e-5`

The rearrangement allclose result is diagnostic and nonblocking only after exact recurrence has passed.

Accepted rearrangement statuses are exactly:

`PASS_TOLERANCE`

or:

`DIAGNOSTIC_TOLERANCE_EXCEEDED`

Unknown rearrangement status must fail closed.

A valid, finite, exact-recurrence record must not be dropped, replaced, masked, or excluded solely because its rearrangement diagnostic exceeds the frozen tolerance.

`R2_EXACT_RECURRENCE_HARD_GATE = YES`

`R2_REARRANGEMENT_DIAGNOSTIC_BLOCKING = NO`

`R2_RECORD_DROP_ON_DIAGNOSTIC_EXCEEDANCE = NO`

## 7. Corrected recurrence audit contract

The scientific result artifact:

`recurrence_audit.json`

must use corrected explicit semantics compatible with:

`k0-rvg-p1-recurrence-audit-v2`

It must report, at minimum:

- hard integrity gate:
  `EXACT_NATIVE_RECURRENCE`;
- rearrangement blocking:
  `false`;
- exact-recurrence accepted count;
- rearrangement diagnostic pass count;
- rearrangement diagnostic exceedance count;
- frozen `atol`;
- frozen `rtol`;
- maximum absolute rearrangement residual;
- maximum relative Frobenius residual;
- maximum scaled tolerance residual;
- incoming common-state comparison count;
- incoming common-state failure count.

Required invariant:

`diagnostic_pass_count + diagnostic_exceedance_count = exact_recurrence_accepted_count`

No diagnostic exceedance may decrement scientific item count or block aggregation.

The numerical diagnostics are descriptive only and may not be interpreted as post-hoc selection thresholds.

## 8. K2S / model runtime binding

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

Mamba upstream blob:

`87987e3e6646d8d0f9f0048bdd8a155d99c845db`

## 9. A0 model provenance binding

A0 commit:

`55debe94f0d19d16a334395e8561901fed6b52fa`

A0 model source blob:

`f0ddc0eda64937de6fcd27943e30a296082c01d5`

A0 heads tree:

`68d26855aa511fcd41d6f395ae5f87177a162678`

Both frozen and current runtime source/tree identities must match before model execution.

## 10. Handoff/checkpoint/encoder binding

Seed180 handoff ZIP SHA256:

`96859bad3e400613b4c981990e56aaf35b1d92baaaa930eb11448cf38a63b861`

Checkpoint SHA256:

`4f7ad019bddb988a534c477b58b36bdabe2775d6c9748331e8311653c07c864c`

Common encoder canonical SHA256:

`48a7e9ac9dfa6c8c292090ee0fcb606bd4c85d13706bfc8a3e371af77c440597`

Common encoder raw-concat SHA256:

`968c12c095a6aab883db5984f4c02ad5e893a5ff140781ffbda41b97970401ae`

At launch, the handoff filesystem path may vary, but the exact ZIP bytes must match the frozen SHA256.

## 11. Scientific execution environment

Authorized execution environment:

`LOCAL_CPU_SEQUENTIAL_MAMBA_ONLY`

GPU execution is not authorized for this study.

Kaggle is not required and is not authorized for this P1 execution.

The validated sequential CPU Mamba path is the authoritative runtime path.

The scientific execution must be launched from the local ContraMamba-K repository only after the full launch gate passes.

## 12. Frozen primary measurement parameters

Primary layer:

`23`

Window:

`W=8`

Coordinates per branch:

`t_e-1, t_e, ..., t_e+7`

Raw recurrent-state shape:

`[1,1536,16]`

Raw recurrent-state dtype:

`torch.float32`

Primary geometry:

`RAW_FROBENIUS`

Branch order:

1. `matched_corr`
2. `matched_ctrl`
3. `swapped_corr`
4. `swapped_ctrl`

Common incoming-state equality remains mandatory.

No alternate layer, window, metric, projection, threshold, alpha, population, pair mapping, branch order, or state definition may be supplied at runtime.

## 13. Primary endpoints

Primary endpoint 1:

`TURNING`

Per-item score:

`X_turn = T_M - T_S`

Primary endpoint 2:

`RESPONSE_COHERENCE`

Per-item score:

`X_coh = C_M - C_S`

Both endpoints use exactly the protected frozen P1 formulas.

No diagnostic recurrence quantity enters either primary endpoint.

No other endpoint enters the primary multiplicity family.

## 14. Primary inference contract

Phase-block count:

`168`

Support threshold:

`160`

Sign classification:

- positive iff `B > 0`;
- negative iff `B < 0`;
- zero iff `B == 0`.

Sign-test denominator excludes exact zeros only.

Statistical test:

`EXACT_TWO_SIDED_SIGN_TEST`

Multiplicity:

`HOLM_M2`

Alpha:

`0.05`

No asymptotic substitution, rescue analysis, post-hoc subset, tolerance-selected population, or threshold retuning is authorized.

## 15. Corrected scientific launch gate

Before any scientific population command may run, all of the following must pass at the frozen corrected P1E authority HEAD:

1. branch is exactly:
   `longterm-k-series-native-state-kinematics`;
2. this corrected P1E authority file is tracked at HEAD and worktree-identical;
3. this authority commit is the immediate one-file child of:
   `795c15d622d568377b36397f235f9e302f719f91`;
4. corrected implementation commit is:
   `50a1daa781e47d1c0f1ba158beb445878e049a65`;
5. corrected implementation commit has direct parent:
   `d8d717a09516b8562f64f7c503d4bbdcc9c34c5d`;
6. corrected implementation commit changed exactly the four authorized files;
7. observer, observer-test, runner, and runner-test bytes equal the four frozen SHA256 markers in this authority;
8. R2 correction-authority SHA256 equals:
   `ae8e910e6ac34cb8550bb7e5b979068c409560bc0648f6ef16a686b2d999a8c3`;
9. all six frozen P0 archive hashes match;
10. K2S/A0/model/runtime provenance matches;
11. seed180 handoff/checkpoint/encoder identities match;
12. focused corrected tests pass:
   `69 passed`;
13. fabricated synthetic preflight returns:
   `PASS_SYNTHETIC_P1_RAW_VECTOR_RUNNER`;
14. fabricated preflight confirms:
   - scientific population model forward:
     `false`;
   - scientific population recurrent-state read:
     `false`;
   - scientific endpoint computation:
     `false`;
   - logits read:
     `false`;
   - causal intervention:
     `false`;
15. fabricated preflight confirms corrected recurrence accounting with:
   `rearrangement_blocking = false`;
16. repository tracked state is clean;
17. only historical K1 untracked files remain;
18. execution-authority authentication against this tracked HEAD passes before model construction;
19. the selected scientific output directory does not exist;
20. no previous corrected P1E scientific attempt under this authority has been launched.

The launch gate is not itself scientific-population execution.

Failure of any launch-gate item blocks scientific execution.

## 16. New provenance-distinct restart rule

The original failed P1E attempt is not resumable.

This corrected authority authorizes a new execution:

`P1E_R2_RESTART_FROM_ITEM = 0`

The new execution must recompute all `336` items prospectively under the corrected implementation.

No item, block, endpoint, recurrence audit value, state hash, partial output, or intermediate result from the original failed attempt may be reused.

The corrected execution must have a fresh output directory and a distinct execution manifest identifying the corrected implementation provenance.

## 17. One-attempt execution rule

This authority permits exactly one corrected scientific execution attempt.

`P1E_AUTHORIZED_EXECUTION_ATTEMPTS = 1`

If the scientific command exits nonzero, crashes, is interrupted, or fails any internal contract:

`AUTOMATIC_RERUN_AUTHORIZED = NO`

Stop immediately.

Do not modify code.

Do not rerun.

Do not reuse or promote partial output.

Do not interpret partial scientific endpoint values.

A separate recovery authority is required before any second corrected attempt.

## 18. Output directory naming

The final scientific output directory must be external to the repository.

Required naming pattern:

`C:\Users\Home1\Desktop\ContraMamba-K0-RVG-P1-Runs\p1-r2-scientific-<P1E_R2_AUTHORITY_SHORT_SHA>-v1`

After this document is frozen, `<P1E_R2_AUTHORITY_SHORT_SHA>` is replaced by the first seven characters of this corrected execution-authority commit.

The directory must not exist before launch.

No existing output directory may be reused.

## 19. Exact scientific command shape

The authorized runner invocation is exactly equivalent to:

`python scripts/longterm_k0_rvg_p1_raw_vector_execution.py --execute-scientific --seed180-handoff <EXACT_HANDOFF_PATH> --execution-authority reports/longterm_k0_rvg_p1e_r2_corrected_scientific_execution_authority_candidate.md --output-dir <FRESH_OUTPUT_DIR>`

No additional scientific CLI flags are authorized.

The exact handoff path and exact fresh output directory are supplied only after this authority is committed, pushed, authenticated, and the launch gate passes.

## 20. Expected scientific result artifacts

A successful corrected run must atomically finalize exactly six scientific result artifacts:

`item_metrics.jsonl`

`block_metrics.jsonl`

`endpoint_summary.json`

`recurrence_audit.json`

`state_hash_audit.jsonl`

`execution_manifest.json`

No seventh scientific result artifact is authorized.

A partial temporary directory is not a valid scientific result artifact set.

## 21. Corrected execution-manifest provenance

The corrected `execution_manifest.json` must identify:

- corrected implementation commit:
  `50a1daa781e47d1c0f1ba158beb445878e049a65`;
- corrected observer SHA256;
- corrected observer-test SHA256;
- corrected runner SHA256;
- corrected runner-test SHA256;
- R2 correction-authority identity;
- corrected execution-authority commit/blob;
- P0 identities;
- K2S identity;
- A0 identities;
- handoff/checkpoint/encoder identities;
- HF/runtime identities.

It must not falsely identify the historical pre-correction observer as the corrected runtime observer.

## 22. Raw-state persistence boundary

Full recurrent tensors must not be persisted.

The runner may persist only the frozen result metrics and deterministic layer-23 state-role SHA256 audit rows.

No raw `S_prev`, `G`, `W`, `S_post`, `V`, `V_carry`, `V_write`, `V_rearranged`, or other recurrent tensor dump is authorized.

## 23. Logits boundary

No logits are required for P1.

`LOGITS_READ_AUTHORIZED = NO`

If logits are read, the run is invalid.

## 24. Causal boundary

Carry/write decomposition remains observational.

`CAUSAL_INTERVENTION_AUTHORIZED = NO`

No state overwrite, channel ablation, retain/write intervention, causal mediation, or counterfactual recurrent-state mutation is authorized.

## 25. Result interpretation boundary

A successful process exit does not establish the scientific claim.

The following remain separate:

1. code correctness;
2. scientific execution completion;
3. artifact/provenance validity;
4. scientific interpretation.

Immediately after execution, the exact six output artifacts must undergo corrected P1 scientific artifact/provenance validation before any endpoint conclusion is accepted.

`SCIENTIFIC_RESULT_INTERPRETATION_BEFORE_ARTIFACT_VALIDATION = NO`

A rearrangement diagnostic exceedance is not by itself a scientific failure, exclusion criterion, endpoint, or interpretation result when exact recurrence passes.

## 26. Scientific artifact validation boundary

After successful execution, validation must confirm at minimum:

- exact six-artifact set;
- execution-manifest identity consistency;
- implementation/authority provenance;
- P0/handoff/runtime provenance;
- item count:
  `336`;
- phase-block count:
  `168`;
- exact recurrence accepted accounting;
- rearrangement diagnostic pass/exceedance accounting;
- recurrence audit invariant;
- common incoming-state failures:
  `0`;
- deterministic state-hash audit structure;
- endpoint support requirements;
- JSON/JSONL canonical validity;
- no NaN/Inf;
- no unexpected raw tensor persistence.

Only validated artifacts may enter scientific interpretation.

## 27. Branch-selection boundary

`K_SUCCESSOR_BRANCH_SELECTION = DEFERRED`

`BRANCH_A_STATUS = PARKED_NOT_CLOSED`

`BRANCH_B_STATUS = PARKED_NOT_ACTIVE`

Corrected P1E authorizes measurement only.

It does not authorize automatic Branch A or Branch B activation.

Any successor-branch decision must be based on validated corrected P1 scientific artifacts and a separately frozen interpretation report.

## 28. K4 boundary

`K4_EXECUTION_AUTHORIZED = NO`

This authority cannot be used as K4 authority.

## 29. Execution completion boundary

If the corrected scientific command completes successfully and emits the exact six-artifact set:

`NEXT_BOUNDARY = K0-RVG-P1-R2_SCIENTIFIC_ARTIFACT_VALIDATION`

Do not commit scientific result artifacts before provenance validation.

Do not interpret primary endpoint direction before provenance validation.

## 30. Failure boundary

If any of the following occurs:

- launch-gate failure;
- exact recurrence failure;
- metadata shape/dtype/device failure;
- nonfinite tensor rejection;
- unknown recurrence diagnostic status;
- common incoming-state failure;
- provenance mismatch;
- output collision;
- interrupted process;
- nonzero process exit;
- malformed or incomplete artifact set;

then:

`P1E_R2_EXECUTION_STATUS = BLOCKED_OR_FAILED`

`AUTOMATIC_RERUN_AUTHORIZED = NO`

Stop at the failure record.

Do not retune tolerances.

Do not patch code under this authority.

Do not replace an item.

Do not interpret partial endpoints.

A separately frozen recovery/root-cause authority is required.

## 31. Final authority markers

`P1E_R2_EXECUTION_AUTHORITY_FROZEN = YES`

`P1_IMPLEMENTATION_COMMIT = 50a1daa781e47d1c0f1ba158beb445878e049a65`

`P1_OBSERVER_SHA256 = 9cb3b49a9b516def1fe50d90be1d57ddc160495743f5abff20e61684d2b34ffe`

`P1_OBSERVER_TEST_SHA256 = f766b8aa69e76da0bb4dc098b75b5fb4d5943e160693b6dbb0e7251703302013`

`P1_RUNNER_SHA256 = 7384ab0b208476a1d4a6941882af266f08020c93b1a2c9bdb780ff02a2777c6e`

`P1_RUNNER_TEST_SHA256 = 48736aeaecc0aa09920869fe9aa465c00275e537025ce0a4265bb90620400461`

`R2_CORRECTION_AUTHORITY_SHA256 = ae8e910e6ac34cb8550bb7e5b979068c409560bc0648f6ef16a686b2d999a8c3`

`P0_CANDIDATE_POOL_SHA256 = 743657411af4e143931e4d2c79f17043134bff3a370d30910588504c7f19f246`

`P0_GENERATED_SOURCE_SHA256 = 8137c0020a040faaf0c6be833b123e143dc5a0a09ae092e8e99530bb8671c1bf`

`P0_PHASE_PAIR_MAPPING_SHA256 = c5e1fa2ac946d153821896d5153354fae6e17038a55be87b3d31ab43d2921eda`

`P0_TOKEN_CONTRACTS_SHA256 = 6eb006f7deca28affa73318887421879e1279fd6897fab857b460873add9a998`

`P0_PROVISIONING_MANIFEST_SHA256 = feab9c60e3546ace3258f068e38d5bb577fc63b803b95349379fa8b4db425e52`

`P0_VALIDATION_REPORT_SHA256 = ae2fe4d1db13e1765415eab9c95263aec618fccf9299fe93fad0c8e31141b73a`

`SCIENTIFIC_EXECUTION_AUTHORIZED = YES`

`SCIENTIFIC_MODEL_FORWARD_AUTHORIZED = YES`

`SCIENTIFIC_RECURRENT_STATE_READ_AUTHORIZED = YES`

`SCIENTIFIC_ENDPOINT_COMPUTATION_AUTHORIZED = YES`

`LOGITS_READ_AUTHORIZED = NO`

`LEARNED_OR_TUNED_GEOMETRY_AUTHORIZED = NO`

`OUTCOME_SELECTED_SUBSPACE_AUTHORIZED = NO`

`CAUSAL_INTERVENTION_AUTHORIZED = NO`

`TRAINING_AUTHORIZED = NO`

`HYPERPARAMETER_TUNING_AUTHORIZED = NO`

`POST_OBSERVATION_TOLERANCE_RETUNING_AUTHORIZED = NO`

`K_SUCCESSOR_BRANCH_SELECTION = DEFERRED`

`BRANCH_A_STATUS = PARKED_NOT_CLOSED`

`BRANCH_B_STATUS = PARKED_NOT_ACTIVE`

`K4_EXECUTION_AUTHORIZED = NO`

`P1E_R2_RESTART_FROM_ITEM = 0`

`P1E_AUTHORIZED_EXECUTION_ATTEMPTS = 1`

`AUTOMATIC_RERUN_AUTHORIZED = NO`

`NEXT_BOUNDARY = K0-RVG-P1-R2_SCIENTIFIC_ARTIFACT_VALIDATION`

This authority becomes active only after this exact document is committed and pushed as the immediate one-file child of `795c15d622d568377b36397f235f9e302f719f91`.

Until that freeze and subsequent launch-gate validation complete:

`SCIENTIFIC_EXECUTION_ACTIVE_NOW = NO`
