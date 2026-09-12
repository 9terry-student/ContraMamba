# K0-RVG-P1E-R2 Numerical-Guard Correction Implementation Validation / Readiness Report Candidate

**Status:** R2 corrected implementation validation/readiness candidate.

**Date:** 2026-09-12

**Frozen R2 correction-authority commit:**

`d8d717a09516b8562f64f7c503d4bbdcc9c34c5d`

**Frozen R2 correction-authority file:**

`reports/longterm_k0_rvg_p1e_r2_numerical_guard_correction_authority_spec_candidate.md`

**Frozen R2 correction-authority SHA256:**

`ae8e910e6ac34cb8550bb7e5b979068c409560bc0648f6ef16a686b2d999a8c3`

**Corrected implementation commit:**

`50a1daa781e47d1c0f1ba158beb445878e049a65`

**Corrected implementation direct parent:**

`d8d717a09516b8562f64f7c503d4bbdcc9c34c5d`

**Frozen R1 validated interpretation parent used for protected-function comparison:**

`37d7044b7111d7b02c7a44fdf7d8782c823afa5b`

This report validates only the bounded R2 numerical-guard correction implementation and its fabricated synthetic integration path.

It does not authorize scientific-population model forward.

It does not authorize scientific recurrent-state read.

It does not authorize scientific endpoint computation.

It does not authorize logits read, causal intervention, training, hyperparameter tuning, K4 execution, or any P1 scientific rerun.

## 1. Overall verdict

`R2_CODE_CORRECTNESS = PASS_FOR_FROZEN_CORRECTION_CONTRACT`

`R2_FOCUSED_TEST_CONTRACT = PASS_69`

`R2_SYNTHETIC_EXECUTION_SUCCESS = YES`

`R2_INDEPENDENT_IMPLEMENTATION_REVIEW = PASS`

`R2_IMPLEMENTATION_PROVENANCE_VALID = YES`

`R2_SCIENTIFIC_ESTIMAND_PROTECTION = PASS`

`R2_SCIENTIFIC_CONCLUSION = NONE`

`R2_IMPLEMENTATION_VALIDATED = YES`

`READY_FOR_CORRECTED_P1E_EXECUTION_AUTHORITY_DRAFT = YES`

The corrected implementation is ready for a separately frozen corrected scientific execution-authority review.

This report is not itself a scientific execution authority.

## 2. Exact corrected implementation scope

Independent remote comparison confirms:

- base:
  `d8d717a09516b8562f64f7c503d4bbdcc9c34c5d`
- head:
  `50a1daa781e47d1c0f1ba158beb445878e049a65`
- ahead by:
  `1`
- changed tracked files:
  exactly `4`

Exact modified tracked files:

`scripts/longterm_k0_rvg_raw_recurrence_observer.py`

`tests/test_longterm_k0_rvg_raw_recurrence_observer.py`

`scripts/longterm_k0_rvg_p1_raw_vector_execution.py`

`tests/test_longterm_k0_rvg_p1_raw_vector_execution.py`

No other tracked file was modified by the corrected implementation commit.

Historical K1 untracked files remained outside the commit:

`scripts/longterm_k1_native_state_kinematics.py`

`tests/test_longterm_k1_native_state_kinematics.py`

## 3. Frozen corrected implementation identities

Corrected observer:

`scripts/longterm_k0_rvg_raw_recurrence_observer.py`

SHA256:

`9cb3b49a9b516def1fe50d90be1d57ddc160495743f5abff20e61684d2b34ffe`

Corrected observer test:

`tests/test_longterm_k0_rvg_raw_recurrence_observer.py`

SHA256:

`f766b8aa69e76da0bb4dc098b75b5fb4d5943e160693b6dbb0e7251703302013`

Corrected P1 runner:

`scripts/longterm_k0_rvg_p1_raw_vector_execution.py`

SHA256:

`7384ab0b208476a1d4a6941882af266f08020c93b1a2c9bdb780ff02a2777c6e`

Corrected P1 runner test:

`tests/test_longterm_k0_rvg_p1_raw_vector_execution.py`

SHA256:

`48736aeaecc0aa09920869fe9aa465c00275e537025ce0a4265bb90620400461`

These identities were collected from the frozen post-commit worktree at implementation commit `50a1daa781e47d1c0f1ba158beb445878e049a65`.

## 4. Exact recurrence hard-integrity validation

The corrected observer retains exact native recurrence reconstruction as the blocking recurrence-integrity gate:

`torch.equal(G * S_prev + W, S_post)`

Exact mismatch still raises:

`RECURRENCE_EXACT_RECONSTRUCTION_FAILURE`

Result:

`R2_EXACT_RECURRENCE_HARD_GATE = PASS`

The correction did not weaken native recurrence integrity.

## 5. Rearrangement diagnostic correction

The corrected observer retains:

`V_raw = S_post - S_prev`

`V_carry = (G - 1) * S_prev`

`V_write = W`

`V_rearranged = V_carry + V_write`

Frozen tolerances remain exactly:

`VELOCITY_ATOL = 1e-6`

`VELOCITY_RTOL = 1e-5`

The frozen allclose remains a computed numerical diagnostic, but after exact recurrence has passed it no longer blocks the record.

The only accepted rearrangement statuses are:

`PASS_TOLERANCE`

and:

`DIAGNOSTIC_TOLERANCE_EXCEEDED`

An explicit boolean preserves the exact frozen-allclose outcome.

Unknown rearrangement status fails closed.

Result:

`R2_REARRANGEMENT_DIAGNOSTIC_NONBLOCKING = PASS`

`R2_POST_OBSERVATION_TOLERANCE_RETUNING = NONE`

## 6. No-record-drop validation

A deterministic fabricated cancellation fixture establishes all of the following simultaneously:

- exact recurrence:
  `PASS_EXACT`
- frozen rearrangement allclose:
  `false`
- observer status:
  `DIAGNOSTIC_TOLERANCE_EXCEEDED`
- observer does not raise;
- layer-23 validation does not raise;
- exact-recurrence accepted count increments;
- diagnostic exceedance count increments;
- branch capture retains all target primary records.

The fabricated fixture does not use the validated scientific token-49 scalar values.

Result:

`R2_DIAGNOSTIC_EXCEEDANCE_RECORD_RETENTION = PASS`

## 7. Exact-recurrence failure compatibility

A separate fabricated mismatch fixture confirms that exact recurrence failure still blocks before corrected diagnostic semantics can admit the record.

Result:

`R2_EXACT_RECURRENCE_FAILURE_STILL_BLOCKS = PASS`

## 8. Passing-record compatibility

A fabricated exact-recurrence record that satisfies the frozen rearrangement tolerance still returns:

`PASS_TOLERANCE`

with the original residual diagnostics preserved.

Result:

`R2_PASSING_RECORD_COMPATIBILITY = PASS`

## 9. Corrected audit semantics

The corrected audit independently tracks:

- exact-recurrence accepted count;
- rearrangement diagnostic pass count;
- rearrangement diagnostic exceedance count;
- maximum absolute rearrangement residual;
- maximum relative Frobenius residual;
- maximum scaled tolerance residual;
- incoming common-state comparison count;
- incoming common-state failure count.

The required invariant is enforced:

`diagnostic_pass_count + diagnostic_exceedance_count = exact_recurrence_accepted_count`

Result:

`R2_AUDIT_ACCOUNTING = PASS`

## 10. Recurrence audit artifact semantics

`recurrence_audit.json` remains one of the exact six P1 scientific result artifacts.

The corrected audit schema is:

`k0-rvg-p1-recurrence-audit-v2`

It explicitly records:

- hard integrity gate:
  `EXACT_NATIVE_RECURRENCE`
- rearrangement blocking:
  `false`
- exact-recurrence accepted count;
- diagnostic pass count;
- diagnostic exceedance count;
- frozen `atol`;
- frozen `rtol`;
- maximum residual diagnostics;
- incoming common-state diagnostics.

Result:

`R2_RECURRENCE_AUDIT_SCHEMA = PASS`

## 11. Protected scientific-estimand equivalence

The focused suite independently compares the corrected runner with parent commit:

`37d7044b7111d7b02c7a44fdf7d8782c823afa5b`

using AST dumps with location metadata excluded.

The protected functions include:

- `target_indices`
- `frobenius_cosine`
- `common_incoming_state`
- `compute_pair_metrics`
- `item_endpoint_metrics`
- `aggregate_blocks`
- `exact_two_sided_sign_test`
- `holm_m2`
- `endpoint_summary_one`
- `summarize_endpoints`
- `reconstruct_branch_texts`
- `revalidate_token_contract`

All protected functions matched exactly under this comparison.

Result:

`R2_SCIENTIFIC_ESTIMAND_AST_EQUIVALENCE = PASS`

## 12. Fabricated endpoint compatibility

The corrected suite directly loads the pre-correction parent runner from commit:

`37d7044b7111d7b02c7a44fdf7d8782c823afa5b`

and compares canonicalized fabricated:

- item metrics;
- block metrics;
- endpoint summaries.

The corrected implementation and parent behavior match on fabricated inputs where the old rearrangement guard passes.

Result:

`R2_FABRICATED_ENDPOINT_COMPATIBILITY = PASS`

This is a regression check only and is not a scientific P0 endpoint result.

## 13. Focused post-commit test result

At frozen implementation commit:

`50a1daa781e47d1c0f1ba158beb445878e049a65`

the focused corrected suite produced:

`69 passed`

Result:

`R2_POST_COMMIT_FOCUSED_TESTS = PASS_69`

## 14. Full fabricated synthetic integration

The frozen post-commit synthetic preflight produced:

`PASS_SYNTHETIC_P1_RAW_VECTOR_RUNNER`

Synthetic item count:

`2`

Synthetic branch forward count:

`12`

Layer-23 recurrence checks:

`72`

Rearrangement diagnostic pass count:

`72`

Rearrangement diagnostic exceedance count:

`0`

Rearrangement blocking:

`false`

Incoming common-state comparisons:

`16`

Incoming common-state failures:

`0`

State-hash rows:

`72`

Repeat-first-item identity:

`PASS_EXACT`

Zero-vector policy:

`PASS_UNDEFINED`

Exact sign-test fixture:

`PASS`

Holm `m=2` fixture:

`PASS`

Result:

`R2_FULL_FABRICATED_SYNTHETIC_INTEGRATION = PASS`

The synthetic preflight's zero diagnostic exceedances do not replace the dedicated fabricated exceedance/no-drop regression fixture in the focused suite.

## 15. Synthetic recurrence numerical diagnostics

Post-commit synthetic maximum absolute residual:

`3.814697265625e-06`

Post-commit synthetic maximum relative Frobenius residual:

`3.4151345205699163e-07`

Post-commit synthetic maximum scaled tolerance residual:

`0.8391106128692627`

These values are descriptive diagnostics only.

They did not alter:

`VELOCITY_ATOL = 1e-6`

or:

`VELOCITY_RTOL = 1e-5`

and are not used as selection thresholds.

## 16. Seed180 handoff and runtime identities

Authenticated seed180 handoff ZIP SHA256:

`96859bad3e400613b4c981990e56aaf35b1d92baaaa930eb11448cf38a63b861`

Authenticated checkpoint SHA256:

`4f7ad019bddb988a534c477b58b36bdabe2775d6c9748331e8311653c07c864c`

Encoder canonical digest:

`48a7e9ac9dfa6c8c292090ee0fcb606bd4c85d13706bfc8a3e371af77c440597`

Encoder raw-concat digest:

`968c12c095a6aab883db5984f4c02ad5e893a5ff140781ffbda41b97970401ae`

HF model:

`state-spaces/mamba-130m-hf`

HF revision:

`5708daa364c50b880e7bd92eab456e0d34492ee9`

Observed transformers version:

`5.12.1`

K2S helper SHA256:

`f741780e7199452e64b7c4a3d70f50f3e55ebc84296f69f288aa317582de84b8`

K2S helper Git blob:

`3a651fb508669bdcf72441a4869b863d6eee6c1f`

A0 commit:

`55debe94f0d19d16a334395e8561901fed6b52fa`

A0 model blob:

`f0ddc0eda64937de6fcd27943e30a296082c01d5`

A0 heads-tree SHA:

`68d26855aa511fcd41d6f395ae5f87177a162678`

## 17. Frozen P0 static artifact binding

The corrected synthetic validation statically authenticated the exact frozen P0 archive identities:

Candidate pool SHA256:

`743657411af4e143931e4d2c79f17043134bff3a370d30910588504c7f19f246`

Generated source SHA256:

`8137c0020a040faaf0c6be833b123e143dc5a0a09ae092e8e99530bb8671c1bf`

Phase-pair mapping SHA256:

`c5e1fa2ac946d153821896d5153354fae6e17038a55be87b3d31ab43d2921eda`

Token contracts SHA256:

`6eb006f7deca28affa73318887421879e1279fd6897fab857b460873add9a998`

Provisioning manifest SHA256:

`feab9c60e3546ace3258f068e38d5bb577fc63b803b95349379fa8b4db425e52`

P0 validation report SHA256:

`ae2fe4d1db13e1765415eab9c95263aec618fccf9299fe93fad0c8e31141b73a`

These artifacts were used for static authentication only.

## 18. Corrected provenance and future execution-authority gate

The corrected observer and P1 runner identify the frozen R2 correction authority:

`d8d717a09516b8562f64f7c503d4bbdcc9c34c5d`

The corrected implementation commit is its immediate child.

Future corrected implementation authentication requires:

1. direct parent equal to the frozen R2 authority commit;
2. changed files exactly the four authorized R2 implementation files;
3. exact observer blob;
4. exact observer-test blob;
5. exact runner blob;
6. exact runner-test blob;
7. no later blob drift for any of the four files.

Future corrected scientific execution authority must bind SHA256 identities for:

- corrected observer;
- corrected observer test;
- corrected P1 runner;
- corrected P1 runner test;
- frozen R2 correction authority;

as well as the existing P0/K2S/A0/handoff/runtime identities.

The corrected execution manifest no longer represents the pre-correction observer as the runtime observer.

Result:

`R2_CORRECTED_IMPLEMENTATION_PROVENANCE = PASS`

## 19. Independent implementation review

An independent verifier reviewed the frozen correction scope and reported:

`PASS`

and:

`PASS_READY_FOR_R2_IMPLEMENTATION_FREEZE_REVIEW`

The verifier independently checked:

- exact recurrence remains hard;
- rearrangement exceedance is nonblocking only after exact recurrence;
- frozen tolerances remain unchanged;
- audit semantics and v2 recurrence audit;
- protected AST equivalence;
- fabricated no-drop behavior;
- fabricated endpoint compatibility;
- four-file provenance and future execution-authority binding;
- absence of forbidden scientific drift;
- absence of scientific-population execution.

The independent verifier reran the focused suite:

`69 passed`

and confirmed:

`git diff --check = PASS`

No repository file was modified by the verifier.

Result:

`R2_INDEPENDENT_REVIEW = PASS`

## 20. Scientific execution boundary preserved

The frozen synthetic preflight reported:

`scientific_population_model_forward_executed = false`

`scientific_population_recurrent_state_read = false`

`scientific_endpoint_computed = false`

`logits_read = false`

`causal_intervention_executed = false`

No training occurred.

No hyperparameter tuning occurred.

No K4 execution occurred.

Result:

`R2_SCIENTIFIC_EXECUTION_BOUNDARY = PASS`

## 21. Scientific-result boundary

No frozen 336-item P0 scientific population execution occurred during R2 implementation or validation.

No P1 scientific endpoint result was produced.

No `X_turn` or `X_coh` scientific conclusion was established.

No sign-test or Holm result on the scientific population was produced.

Therefore:

`P1_SCIENTIFIC_CONCLUSION = NONE`

## 22. Readiness conclusion

All prerequisites required by the R2 correction authority for implementation validation are satisfied:

1. frozen R2 correction authority:
   `PASS`
2. exact four-file corrected implementation:
   `PASS`
3. focused corrected tests:
   `PASS_69`
4. fabricated synthetic correction validation:
   `PASS`
5. independent implementation review:
   `PASS`
6. corrected implementation validation/readiness evidence:
   `PASS`

Therefore:

`R2_IMPLEMENTATION_VALIDATED = YES`

`READY_FOR_CORRECTED_P1E_EXECUTION_AUTHORITY_DRAFT = YES`

A separate corrected scientific execution authority may now be drafted and reviewed.

Scientific execution remains:

`NOT_AUTHORIZED_BY_THIS_REPORT`
