# K0-RVG-P1E-R1 Numerical-Support Diagnostic Implementation Validation / Readiness Report Candidate

**Status:** R1 implementation validation/readiness candidate.

**Frozen R1 implementation commit:**

`01ca5f5b11a4a5738ad8f77f97d33bdbf6c6bceb`

**Immediate parent / frozen R1 protocol commit:**

`02f184a186aa6b90fc98ece717723de0adafc89c`

**R1 runner SHA256:**

`d5a8e8c5e3c7ef5855d12a0284e14b6eaef9f5991a21823025429b03757d5b26`

**R1 test SHA256:**

`cc7b1c364fb050642fbe9f0b7e8c4f1f825322682ae4b430bfb09b9c93ee4979`

This report validates only the bounded R1 numerical-support diagnostic implementation and fabricated synthetic integration path.

It does not authorize a second scientific-population model forward.

It does not authorize a second scientific recurrent-state read.

It does not authorize P1 endpoint computation, tolerance replacement, branch selection, causal intervention, or K4.

## 1. Overall verdict

`R1_CODE_CORRECTNESS = PASS_FOR_FROZEN_R1_CONTRACT`

`R1_SYNTHETIC_EXECUTION_SUCCESS = YES`

`R1_IMPLEMENTATION_PROVENANCE_VALID = YES`

`R1_SCIENTIFIC_POPULATION_BLINDING_PRESERVED = YES`

`R1_P1_ENDPOINT_COMPUTATION = NO`

`R1_REPLACEMENT_TOLERANCE_SELECTION = NO`

`R1_IMPLEMENTATION_VALIDATED = YES`

`READY_FOR_P1E_R1E_EXECUTION_AUTHORITY_DRAFT = YES`

`SCIENTIFIC_CONCLUSION_FROM_R1_VALIDATION = NONE`

The implementation is ready for a separately frozen fixed single-branch R1 diagnostic execution-authority review.

This report is not itself an execution authority.

## 2. Exact implementation scope

Independent remote comparison confirms:

- base:
  `02f184a186aa6b90fc98ece717723de0adafc89c`;
- head:
  `01ca5f5b11a4a5738ad8f77f97d33bdbf6c6bceb`;
- ahead by:
  `1`;
- changed files:
  exactly `2`.

Exact added files:

`scripts/longterm_k0_rvg_p1e_r1_numerical_support_diagnostic.py`

`tests/test_longterm_k0_rvg_p1e_r1_numerical_support_diagnostic.py`

No existing tracked repository file was modified.

Historical K1 untracked files remained outside the implementation commit.

## 3. Frozen implementation identities

Runner:

`scripts/longterm_k0_rvg_p1e_r1_numerical_support_diagnostic.py`

SHA256:

`d5a8e8c5e3c7ef5855d12a0284e14b6eaef9f5991a21823025429b03757d5b26`

Test:

`tests/test_longterm_k0_rvg_p1e_r1_numerical_support_diagnostic.py`

SHA256:

`cc7b1c364fb050642fbe9f0b7e8c4f1f825322682ae4b430bfb09b9c93ee4979`

The same exact bytes were verified:

- before staging;
- in the Git index;
- after commit.

## 4. Focused test result

Pre-commit v3 focused suite:

`38 passed`

Post-commit v3 focused suite:

`38 passed`

The frozen suite covers the required R1 implementation contract, including:

- frozen R1 protocol/provenance constants;
- fixed local item index `0`;
- fixed branch role `matched_corr`;
- fixed primary layer `23`;
- exact nine-coordinate target construction;
- exact recurrence diagnostic without raising;
- frozen float32 allclose diagnostic without raising;
- elementwise tolerance-scale diagnostics;
- maximum scaled residual;
- failing-element count and fraction;
- Frobenius residual diagnostics;
- state-to-velocity scale diagnostics;
- deterministic maximum-residual element audit;
- float64 snapshot diagnostics;
- fixed classification rules;
- fixed local support-profile rules;
- no tolerance sweep interface;
- no layer/window/item/branch override;
- no P1 endpoint implementation;
- JSON finite/null behavior;
- raw tensor non-persistence boundary;
- atomic single diagnostic artifact;
- R1E execution-authority fail-closed behavior;
- execution gate before model construction;
- exact R1 direct-parent/two-file implementation commit validation;
- non-layer-23 immediate discard;
- no import-time model execution.

Result:

`R1_FOCUSED_TEST_CONTRACT = PASS_38`

## 5. Fabricated synthetic integration result

Post-commit fabricated synthetic preflight:

`PASS_SYNTHETIC_R1_NUMERICAL_SUPPORT_DIAGNOSTIC`

Synthetic model forwards:

`2`

Synthetic branch role:

`matched_corr`

Coordinates per forward:

`9`

Repeated synthetic diagnostic identity:

`PASS_EXACT`

Scientific P0 population model forward:

`false`

Scientific P0 recurrent-state read:

`false`

P1 endpoint computation:

`false`

Replacement tolerance selection:

`false`

Logits read:

`false`

Causal intervention:

`false`

Scientific result interpretation:

`false`

## 6. Synthetic numerical-support result

The fabricated authentic-model branch produced:

`R1_SYNTHETIC_CLASSIFICATION = ORIGINAL_FAILURE_NOT_REPRODUCED_IN_FIXED_SINGLE_BRANCH_DIAGNOSTIC`

with:

- failing coordinates:
  `0`;
- passing coordinates:
  `9`;
- maximum scaled tolerance residual:
  `0.4528714716434479`;
- median scaled tolerance residual:
  `0.16030071675777435`.

This is a fabricated non-study result only.

It is not evidence about the frozen P0 scientific item 0.

## 7. Explicit frozen-tolerance failure fixture

The synthetic validation also contains a prospectively constructed algebraic fixture where:

- exact recurrence reconstruction:
  `PASS`;
- frozen allclose:
  `FAIL`;
- maximum scaled tolerance residual:
  `90909.09375`.

This verifies that the diagnostic can distinguish:

`EXACT_RECURRENCE_INTACT_FLOAT32_REARRANGEMENT_TOLERANCE_FAILURE`

without changing the frozen tolerance.

Result:

`R1_FAILURE_CLASSIFICATION_FIXTURE = PASS`

## 8. Frozen tolerance preservation

The implementation retains:

`VELOCITY_ATOL = 1e-6`

`VELOCITY_RTOL = 1e-5`

The implementation does not expose CLI overrides for either value.

It does not compute:

- minimum passing tolerance;
- percentile tolerance;
- adaptive tolerance;
- endpoint-conditioned tolerance;
- recommended replacement tolerance.

Result:

`R1_REPLACEMENT_TOLERANCE_SELECTION = NO`

## 9. Fixed diagnostic scope

The implementation freezes the future scientific diagnostic scope to:

`R1_LOCAL_TEMPLATE_INDEX = 0`

`R1_BRANCH_ROLE = MATCHED_CORR`

`R1_PRIMARY_LAYER = 23`

`R1_TARGET_COORDINATE_COUNT = 9`

The future R1E path has no CLI override for:

- item;
- branch;
- layer;
- window;
- atol;
- rtol;
- metric;
- population;
- threshold.

Result:

`R1_FIXED_SCOPE_IMPLEMENTATION = PASS`

## 10. Diagnostic quantities

For each of the nine layer-23 records the implementation records:

- token index;
- exact recurrence pass/fail;
- frozen allclose pass/fail;
- maximum absolute residual;
- residual Frobenius norm;
- raw-velocity Frobenius norm;
- rearranged-velocity Frobenius norm;
- relative Frobenius residual;
- tolerance-scale minimum and maximum;
- maximum scaled tolerance residual;
- failing-element count;
- element count;
- failing-element fraction;
- `S_prev` norm;
- `S_post` norm;
- `W` norm;
- carry norm;
- state-to-raw-velocity norm ratio;
- maximum-residual flattened index and scalar audit;
- float64 snapshot numerical diagnostics;
- hashes for `S_prev`, `G`, `W`, `S_post`.

These are numerical instrument diagnostics only.

## 11. Exact recurrence boundary

The implementation separately evaluates:

`torch.equal(G * S_prev + W, S_post)`

for each target coordinate.

The diagnostic does not silently reinterpret a frozen-tolerance failure as an exact recurrence failure.

The frozen classification hierarchy remains:

1. `EXACT_RECURRENCE_OR_CAPTURE_FAILURE`;
2. `EXACT_RECURRENCE_INTACT_FLOAT32_REARRANGEMENT_TOLERANCE_FAILURE`;
3. `ORIGINAL_FAILURE_NOT_REPRODUCED_IN_FIXED_SINGLE_BRANCH_DIAGNOSTIC`.

Result:

`R1_RECURRENCE_CLASSIFICATION_IMPLEMENTATION = PASS`

## 12. Local support-profile boundary

The diagnostic computes only within the fixed nine-coordinate branch:

- failing-coordinate count;
- passing-coordinate count;
- first failing token;
- maximum scaled residual;
- median scaled residual.

It may label:

- `SINGLE_COORDINATE_FAILURE_WITHIN_FIXED_BRANCH`;
- `MULTI_COORDINATE_FAILURE_WITHIN_FIXED_BRANCH`;
- `NO_FAILURE_REPRODUCED_WITHIN_FIXED_BRANCH`.

These labels do not estimate population-wide prevalence.

## 13. Float64 snapshot boundary

The implementation computes float64 diagnostics only by converting the captured float32 snapshots.

It does not:

- rerun the model in float64;
- redefine P1 primary velocity;
- redefine P1 geometry;
- promote float64 as a scientific endpoint metric.

Result:

`R1_FLOAT64_SNAPSHOT_DIAGNOSTIC = PASS_BOUNDARY_PRESERVED`

## 14. Data-minimization correction

Independent v2 review identified that capture-complete non-layer-23 records remained live during layer-23 diagnostic processing.

v3 corrected this by:

1. validating the exact full capture coordinate set;
2. retaining only the nine layer-23 record references;
3. immediately clearing the full 24-layer capture mapping;
4. then computing numerical diagnostics.

The focused suite explicitly tests this behavior.

Result:

`R1_NON_LAYER23_IMMEDIATE_DISCARD = PASS`

## 15. Raw-state persistence boundary

The future diagnostic artifact does not persist raw recurrent tensors.

Only numerical scalars and exact recurrence-role SHA256 identities may be persisted for layer 23.

No raw:

- `S_prev`;
- `G`;
- `W`;
- `S_post`;
- `V_raw`;
- `V_rearranged`;
- `D`

tensor dump is authorized.

## 16. P1 endpoint exclusion

The diagnostic implementation contains no scientific code path for:

- `X_turn`;
- `X_coh`;
- matched/swapped endpoint contrasts;
- phase-block aggregation;
- exact sign test;
- Holm correction;
- overall P1 raw-vector verdict.

Result:

`R1_P1_ENDPOINT_COMPUTATION = NO`

## 17. Future diagnostic artifact contract

A future R1E execution may produce exactly one external artifact:

`numerical_support_diagnostic.json`

No P1 scientific result artifact name is reused.

Atomic output finalization is required.

The artifact records explicit false flags for:

- P1 endpoints computed;
- logits read;
- causal intervention;
- tolerance sweep;
- replacement tolerance selected;
- scientific result interpretation.

Result:

`R1_DIAGNOSTIC_ARTIFACT_CONTRACT = PASS`

## 18. R1E execution-authority gate

The scientific diagnostic execution path is fail-closed without a later tracked R1E authority.

Before model construction, the implementation requires a tracked authority with exact markers for:

- R1E authority frozen:
  `YES`;
- R1 scientific model forward:
  `YES`;
- R1 scientific recurrent-state read:
  `YES`;
- P1 endpoint computation:
  `NO`;
- replacement tolerance selection:
  `NO`;
- local item:
  `0`;
- branch:
  `MATCHED_CORR`;
- logits:
  `NO`;
- causal intervention:
  `NO`.

It also validates exact R1 implementation commit ancestry/scope and runner/test identities.

Result:

`R1E_EXECUTION_AUTHORITY_GATE = PASS_FAIL_CLOSED`

## 19. Frozen dependency provenance

R1 remains bound to:

P1 implementation:

`2e6bb106d5d3081b7ae69ec4cde652e79d36070c`

P1 runner SHA256:

`2bd59a25a6303dc86c36c9438296b42197cc67019e9d4e9ee2fbbe1d832ec6eb`

P1 test SHA256:

`06c30c974029126fdc853a235071376738cc2769409369ccdb60f0878bac7706`

Observer commit:

`fcfe161c12f4ed8ef37aff435554cc0660e477af`

Observer SHA256:

`12542e32d49b368e727de782b0cb833991464503b1fab56d375f15ec58649c25`

Observer Git blob:

`f2dbdfe52661eca384897578ab272e602e36deac`

K2S SHA256:

`f741780e7199452e64b7c4a3d70f50f3e55ebc84296f69f288aa317582de84b8`

K2S Git blob:

`3a651fb508669bdcf72441a4869b863d6eee6c1f`

A0 commit:

`55debe94f0d19d16a334395e8561901fed6b52fa`

A0 model blob:

`f0ddc0eda64937de6fcd27943e30a296082c01d5`

A0 heads tree:

`68d26855aa511fcd41d6f395ae5f87177a162678`

## 20. P0 archive binding

The R1 diagnostic statically authenticates the exact six frozen P0 archive identities:

`candidate_pool.jsonl`

`743657411af4e143931e4d2c79f17043134bff3a370d30910588504c7f19f246`

`generated_source.jsonl`

`8137c0020a040faaf0c6be833b123e143dc5a0a09ae092e8e99530bb8671c1bf`

`phase_pair_mapping.json`

`c5e1fa2ac946d153821896d5153354fae6e17038a55be87b3d31ab43d2921eda`

`token_contracts.jsonl`

`6eb006f7deca28affa73318887421879e1279fd6897fab857b460873add9a998`

`provisioning_manifest.json`

`feab9c60e3546ace3258f068e38d5bb577fc63b803b95349379fa8b4db425e52`

`validation_report_candidate.md`

`ae2fe4d1db13e1765415eab9c95263aec618fccf9299fe93fad0c8e31141b73a`

During implementation validation these are static reads only.

No frozen P0 branch was model-forwarded.

## 21. Handoff/runtime provenance

Post-commit fabricated synthetic validation authenticated:

Handoff ZIP:

`96859bad3e400613b4c981990e56aaf35b1d92baaaa930eb11448cf38a63b861`

Checkpoint:

`4f7ad019bddb988a534c477b58b36bdabe2775d6c9748331e8311653c07c864c`

Encoder canonical:

`48a7e9ac9dfa6c8c292090ee0fcb606bd4c85d13706bfc8a3e371af77c440597`

Encoder raw-concat:

`968c12c095a6aab883db5984f4c02ad5e893a5ff140781ffbda41b97970401ae`

HF model:

`state-spaces/mamba-130m-hf`

HF revision:

`5708daa364c50b880e7bd92eab456e0d34492ee9`

Transformers:

`5.12.1`

## 22. Code correctness

`CODE_CORRECTNESS = PASS_FOR_FROZEN_R1_CONTRACT`

This establishes implementation conformance to the frozen R1 diagnostic protocol.

It does not establish the numerical mechanism on the frozen P0 scientific item.

## 23. Synthetic execution success

`SYNTHETIC_EXECUTION_SUCCESS = YES`

This establishes authentic model/checkpoint/observer integration on fabricated non-study text.

It does not establish R1 scientific diagnostic execution success.

## 24. Artifact / provenance validity

`IMPLEMENTATION_PROVENANCE_VALID = YES`

The R1 implementation commit, exact two-file scope, runner/test identities, R1 protocol, P0 archive, P1 implementation, observer, K2S, A0, handoff/checkpoint/encoder, and runtime identities are consistent.

## 25. Scientific conclusion

`SCIENTIFIC_CONCLUSION_FROM_R1_VALIDATION = NONE`

The implementation validation does not establish:

- exact cause of the original P1E failure;
- which scientific token coordinate failed;
- whether one or multiple scientific coordinates fail;
- whether cancellation is the full mechanism;
- any replacement tolerance;
- any P1 endpoint;
- Branch A;
- Branch B;
- K4.

## 26. Next-stage boundary

After this exact report is frozen as the immediate one-file child of:

`01ca5f5b11a4a5738ad8f77f97d33bdbf6c6bceb`

the next authorized stage is:

`K0-RVG-P1E-R1E — Fixed Single-Branch Numerical-Support Diagnostic Execution Authority Draft`

R1E may authorize exactly one scientific model forward / recurrent-state read for:

- local item:
  `0`;
- branch:
  `matched_corr`;
- primary diagnostic layer:
  `23`;
- target coordinates:
  exactly `9`.

R1E must continue to prohibit:

- matched control;
- swapped branches;
- items `1..335`;
- P1 endpoints;
- tolerance sweep;
- replacement tolerance selection;
- logits;
- causal intervention;
- K4.

This report does not itself set R1 scientific access to YES.

## 27. Branch state

`K_SUCCESSOR_BRANCH_SELECTION = DEFERRED`

`BRANCH_A_STATUS = PARKED_NOT_CLOSED`

`BRANCH_B_STATUS = PARKED_NOT_ACTIVE`

## 28. Final readiness markers

`R1_IMPLEMENTATION_VALIDATED = YES`

`R1_CODE_CORRECTNESS = PASS_FOR_FROZEN_R1_CONTRACT`

`R1_SYNTHETIC_EXECUTION_SUCCESS = YES`

`R1_IMPLEMENTATION_PROVENANCE_VALID = YES`

`R1_SCIENTIFIC_POPULATION_BLINDING_PRESERVED = YES`

`R1_NON_LAYER23_IMMEDIATE_DISCARD = PASS`

`READY_FOR_P1E_R1E_EXECUTION_AUTHORITY_DRAFT = YES`

`R1_SCIENTIFIC_MODEL_FORWARD_AUTHORIZED = NO`

`R1_SCIENTIFIC_RECURRENT_STATE_READ_AUTHORIZED = NO`

`R1_P1_ENDPOINT_COMPUTATION_AUTHORIZED = NO`

`R1_REPLACEMENT_TOLERANCE_SELECTION_AUTHORIZED = NO`

`SCIENTIFIC_CONCLUSION_FROM_R1_VALIDATION = NONE`

`LOGITS_READ_AUTHORIZED = NO`

`CAUSAL_INTERVENTION_AUTHORIZED = NO`

`K4_EXECUTION_AUTHORIZED = NO`

`NEXT_STAGE = K0-RVG-P1E-R1E_FIXED_SINGLE_BRANCH_NUMERICAL_SUPPORT_DIAGNOSTIC_EXECUTION_AUTHORITY_DRAFT`

Only after this report is frozen may R1E be drafted.
