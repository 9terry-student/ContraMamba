# K0-RVG-P1E Scientific Raw-Vector Execution Authority Candidate

**Status:** one-time scientific execution authority candidate.

**Immediate parent / frozen P1 implementation validation report commit:**

`88da446196c6cda7ccd6f52726f546e5a5dde926`

**Validated P1 implementation commit:**

`2e6bb106d5d3081b7ae69ec4cde652e79d36070c`

**Frozen P1 specification commit:**

`c2b1990b649701cbf5ec71a360f03b4b7ff27465`

**P1 validation report SHA256:**

`85b95aab9ad35ab6c2cda93b9373f8c20265e39df2183da2e580532ca04ddbcf`

This document is the sole authority that may open the frozen P0 scientific population to the validated P1 runner.

It authorizes exactly one prospective scientific execution attempt under the constraints below.

It does not authorize exploratory reruns, layer/window changes, logits, learned geometry, causal intervention, K4, or branch promotion.

## 1. Execution decision

The frozen P1 implementation has passed:

- exact two-file implementation scope;
- 42 focused tests before and after commit;
- fabricated synthetic integration;
- recurrence/source/runtime provenance;
- P0 artifact authentication;
- A0 model/tree provenance;
- execution-authority fail-closed review.

Therefore:

`P1E_EXECUTION_AUTHORITY_FROZEN = YES`

`SCIENTIFIC_EXECUTION_AUTHORIZED = YES`

`SCIENTIFIC_MODEL_FORWARD_AUTHORIZED = YES`

`SCIENTIFIC_RECURRENT_STATE_READ_AUTHORIZED = YES`

`SCIENTIFIC_ENDPOINT_COMPUTATION_AUTHORIZED = YES`

The authorization applies only to the exact frozen P0 population and exact validated P1 implementation bound below.

## 2. Explicit prohibitions

`LOGITS_READ_AUTHORIZED = NO`

`LEARNED_OR_TUNED_GEOMETRY_AUTHORIZED = NO`

`OUTCOME_SELECTED_SUBSPACE_AUTHORIZED = NO`

`CAUSAL_INTERVENTION_AUTHORIZED = NO`

`TRAINING_AUTHORIZED = NO`

`HYPERPARAMETER_TUNING_AUTHORIZED = NO`

`K4_EXECUTION_AUTHORIZED = NO`

No code modification is authorized by this execution authority.

No scientific parameter override is authorized.

## 3. Exact P1 implementation binding

`P1_IMPLEMENTATION_COMMIT = 2e6bb106d5d3081b7ae69ec4cde652e79d36070c`

`P1_RUNNER_SHA256 = 2bd59a25a6303dc86c36c9438296b42197cc67019e9d4e9ee2fbbe1d832ec6eb`

`P1_TEST_SHA256 = 06c30c974029126fdc853a235071376738cc2769409369ccdb60f0878bac7706`

Runner path:

`scripts/longterm_k0_rvg_p1_raw_vector_execution.py`

Test path:

`tests/test_longterm_k0_rvg_p1_raw_vector_execution.py`

The implementation commit must remain the exact direct two-file child of:

`c2b1990b649701cbf5ec71a360f03b4b7ff27465`

No runner/test blob drift is permitted.

## 4. Exact P0 scientific population binding

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
- no regeneration.

## 5. Observer/runtime binding

`OBSERVER_SHA256 = 12542e32d49b368e727de782b0cb833991464503b1fab56d375f15ec58649c25`

Validated observer commit:

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

Mamba upstream blob:

`87987e3e6646d8d0f9f0048bdd8a155d99c845db`

## 6. A0 model provenance binding

A0 commit:

`55debe94f0d19d16a334395e8561901fed6b52fa`

A0 model source blob:

`f0ddc0eda64937de6fcd27943e30a296082c01d5`

A0 heads tree:

`68d26855aa511fcd41d6f395ae5f87177a162678`

Both frozen and current runtime source/tree identities must match before model execution.

## 7. Handoff/checkpoint/encoder binding

Seed180 handoff ZIP SHA256:

`96859bad3e400613b4c981990e56aaf35b1d92baaaa930eb11448cf38a63b861`

Checkpoint SHA256:

`4f7ad019bddb988a534c477b58b36bdabe2775d6c9748331e8311653c07c864c`

Common encoder canonical SHA256:

`48a7e9ac9dfa6c8c292090ee0fcb606bd4c85d13706bfc8a3e371af77c440597`

Common encoder raw-concat SHA256:

`968c12c095a6aab883db5984f4c02ad5e893a5ff140781ffbda41b97970401ae`

At launch, the exact handoff file path may vary, but its bytes must match the frozen ZIP SHA256.

## 8. Scientific execution environment

Authorized execution environment:

`LOCAL_CPU_SEQUENTIAL_MAMBA_ONLY`

GPU execution is not authorized for this study.

Kaggle is not required for this execution.

The validated sequential CPU Mamba path is the authoritative runtime path.

The user may execute on the current local workstation only after the launch gate in this document passes.

## 9. Frozen primary measurement parameters

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

Scalar accumulation:

`float64`

Primary geometry:

`RAW_FROBENIUS`

No alternate layer, window, metric, projection, threshold, alpha, population, or pair mapping may be supplied at runtime.

## 10. Primary endpoints

Primary endpoint 1:

`TURNING`

Per-item score:

`X_turn = T_M - T_S`

Primary endpoint 2:

`RESPONSE_COHERENCE`

Per-item score:

`X_coh = C_M - C_S`

Both endpoints use exactly the frozen P1 formulas.

No other endpoint enters the primary multiplicity family.

## 11. Primary inference contract

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

No asymptotic substitution or rescue analysis is authorized.

## 12. Scientific launch gate

Before the scientific command may be run, all of the following must pass at the frozen P1E authority HEAD:

1. branch is exactly:
   `longterm-k-series-native-state-kinematics`;
2. P1E authority file is tracked at HEAD and worktree-identical;
3. runner/test bytes equal the frozen SHA256 values;
4. P0 six-file archive hashes match;
5. observer/K2S/A0/runtime provenance matches;
6. `42` focused tests pass;
7. fabricated synthetic preflight returns:
   `PASS_SYNTHETIC_P1_RAW_VECTOR_RUNNER`;
8. fabricated preflight confirms:
   - scientific population model forward:
     `false`;
   - scientific population recurrent-state read:
     `false`;
   - scientific endpoint computation:
     `false`;
9. repository tracked state is clean;
10. only historical K1 untracked files remain.

The launch gate is not itself a scientific population execution.

## 13. One-attempt execution rule

This authority permits exactly one scientific execution attempt.

`P1E_AUTHORIZED_EXECUTION_ATTEMPTS = 1`

The scientific command must target a fresh output directory that does not already exist.

If the scientific command exits nonzero, crashes, is interrupted, or fails any internal contract:

`AUTOMATIC_RERUN_AUTHORIZED = NO`

Stop immediately.

Do not modify code.

Do not inspect or interpret partial scientific endpoint values.

A separate recovery authority is required before any second attempt.

## 14. Output directory naming

The final scientific output directory must be external to the repository.

Required naming pattern:

`C:\Users\Home1\Desktop\ContraMamba-K0-RVG-P1-Runs\p1-scientific-<P1E_AUTHORITY_SHORT_SHA>-v1`

After this P1E document is frozen, `<P1E_AUTHORITY_SHORT_SHA>` is replaced by the first seven characters of the P1E authority commit.

No existing directory may be reused.

## 15. Exact scientific command shape

The authorized runner invocation is exactly equivalent to:

`python scripts/longterm_k0_rvg_p1_raw_vector_execution.py --execute-scientific --seed180-handoff <EXACT_HANDOFF_PATH> --execution-authority reports/longterm_k0_rvg_p1e_scientific_raw_vector_execution_authority_candidate.md --output-dir <FRESH_OUTPUT_DIR>`

No additional scientific CLI flags are authorized.

The exact handoff path and exact fresh output directory are supplied only after this authority is frozen and the launch gate passes.

## 16. Expected scientific result artifacts

A successful run must atomically finalize exactly:

`item_metrics.jsonl`

`block_metrics.jsonl`

`endpoint_summary.json`

`recurrence_audit.json`

`state_hash_audit.jsonl`

`execution_manifest.json`

A partial temporary directory is not a scientific result artifact set.

## 17. Raw-state persistence boundary

Full recurrent tensors must not be persisted.

The runner may persist only the frozen result metrics and deterministic layer-23 state-role SHA256 audit rows.

No raw `S_prev`, `G`, `W`, `S_post`, `V`, or `R` tensor dump is authorized.

## 18. Logits boundary

No logits are required for P1.

`LOGITS_READ_AUTHORIZED = NO`

If logits are read, the run is invalid.

## 19. Causal boundary

Carry/write decomposition remains observational.

`CAUSAL_INTERVENTION_AUTHORIZED = NO`

No state overwrite, channel ablation, retain/write intervention, causal mediation, or counterfactual recurrent-state mutation is authorized.

## 20. Result interpretation boundary

A successful process exit does not establish the scientific claim.

The following remain separate:

1. code correctness;
2. scientific execution completion;
3. artifact/provenance validity;
4. scientific interpretation.

Immediately after execution, the six output artifacts must be validated before any branch-level scientific conclusion is accepted.

`SCIENTIFIC_RESULT_INTERPRETATION_BEFORE_ARTIFACT_VALIDATION = NO`

## 21. Branch-selection boundary

`K_SUCCESSOR_BRANCH_SELECTION = DEFERRED`

`BRANCH_A_STATUS = PARKED_NOT_CLOSED`

`BRANCH_B_STATUS = PARKED_NOT_ACTIVE`

P1E authorizes measurement only.

It does not authorize automatic Branch A or Branch B activation.

Any successor-branch decision must be based on validated P1 scientific artifacts and a separately frozen interpretation report.

## 22. K4 boundary

`K4_EXECUTION_AUTHORIZED = NO`

P1E cannot be used as K4 authority.

## 23. Execution completion boundary

If the scientific command completes successfully and emits the exact six-artifact set:

`NEXT_BOUNDARY = K0-RVG-P1_SCIENTIFIC_ARTIFACT_VALIDATION`

Do not commit scientific result artifacts before provenance validation.

Do not interpret primary endpoint direction before provenance validation.

## 24. Final authority markers

`P1E_EXECUTION_AUTHORITY_FROZEN = YES`

`P1_IMPLEMENTATION_COMMIT = 2e6bb106d5d3081b7ae69ec4cde652e79d36070c`

`P1_RUNNER_SHA256 = 2bd59a25a6303dc86c36c9438296b42197cc67019e9d4e9ee2fbbe1d832ec6eb`

`P1_TEST_SHA256 = 06c30c974029126fdc853a235071376738cc2769409369ccdb60f0878bac7706`

`P0_CANDIDATE_POOL_SHA256 = 743657411af4e143931e4d2c79f17043134bff3a370d30910588504c7f19f246`

`P0_GENERATED_SOURCE_SHA256 = 8137c0020a040faaf0c6be833b123e143dc5a0a09ae092e8e99530bb8671c1bf`

`P0_PHASE_PAIR_MAPPING_SHA256 = c5e1fa2ac946d153821896d5153354fae6e17038a55be87b3d31ab43d2921eda`

`P0_TOKEN_CONTRACTS_SHA256 = 6eb006f7deca28affa73318887421879e1279fd6897fab857b460873add9a998`

`P0_PROVISIONING_MANIFEST_SHA256 = feab9c60e3546ace3258f068e38d5bb577fc63b803b95349379fa8b4db425e52`

`P0_VALIDATION_REPORT_SHA256 = ae2fe4d1db13e1765415eab9c95263aec618fccf9299fe93fad0c8e31141b73a`

`OBSERVER_SHA256 = 12542e32d49b368e727de782b0cb833991464503b1fab56d375f15ec58649c25`

`SCIENTIFIC_EXECUTION_AUTHORIZED = YES`

`SCIENTIFIC_MODEL_FORWARD_AUTHORIZED = YES`

`SCIENTIFIC_RECURRENT_STATE_READ_AUTHORIZED = YES`

`SCIENTIFIC_ENDPOINT_COMPUTATION_AUTHORIZED = YES`

`LOGITS_READ_AUTHORIZED = NO`

`LEARNED_OR_TUNED_GEOMETRY_AUTHORIZED = NO`

`OUTCOME_SELECTED_SUBSPACE_AUTHORIZED = NO`

`CAUSAL_INTERVENTION_AUTHORIZED = NO`

`TRAINING_AUTHORIZED = NO`

`K_SUCCESSOR_BRANCH_SELECTION = DEFERRED`

`BRANCH_A_STATUS = PARKED_NOT_CLOSED`

`BRANCH_B_STATUS = PARKED_NOT_ACTIVE`

`K4_EXECUTION_AUTHORIZED = NO`

`P1E_AUTHORIZED_EXECUTION_ATTEMPTS = 1`

`AUTOMATIC_RERUN_AUTHORIZED = NO`

`NEXT_BOUNDARY = K0-RVG-P1_SCIENTIFIC_ARTIFACT_VALIDATION`

This authority becomes active only after this exact document is committed and pushed as the immediate one-file child of `88da446196c6cda7ccd6f52726f546e5a5dde926`.
