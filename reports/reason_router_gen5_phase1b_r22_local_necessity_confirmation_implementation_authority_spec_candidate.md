# ContraMamba Gen5 Phase 1B — R22 Local Necessity Confirmation Implementation Authority

## 0. Status

PHASE =
`GEN5_PHASE1B_R22_LOCAL_NECESSITY_CONFIRMATION_IMPLEMENTATION`

STATUS =
`FROZEN_ON_COMMIT`

IMPLEMENTATION_ALLOWED_AFTER_FREEZE =
`YES_BOUNDED`

SCIENTIFIC_EXECUTION_ALLOWED =
`NO`

TRAINING_ALLOWED =
`NO`

BACKWARD_ALLOWED =
`NO`

KAGGLE_ALLOWED =
`NO`

MODEL_FORWARD_ALLOWED =
`NO`

README_UPDATE_REQUIRED =
`NO`

This authority permits only implementation and static/synthetic verification of
the already-frozen Phase 1B necessity-confirmation protocol.

It does not authorize execution on the necessity-confirmation population.

---

## 1. Parent scientific authority

PHASE1B_DESIGN_COMMIT =
`c8dc7a4bb69dd4e86f4bbbcf48b88940bc007cd8`

PHASE1B_DESIGN_PATH =
`reports/reason_router_gen5_phase1b_native_update_role_bridge_spec_candidate.md`

The implementation must preserve Section 15 local-necessity semantics exactly.

The parent design explicitly separates:

`IMPLEMENTATION`
->
`INDEPENDENT_VERIFICATION`
->
`EXECUTION_AUTHORITY`

and requires independent verification for hidden-state intervention semantics.

---

## 2. Frozen R22/C22 construction artifacts

CONSTRUCTION_ARTIFACT_FREEZE_COMMIT =
`1d3542013934870aa9181d1bbaf565ff4724112c`

CONSTRUCTION_ARTIFACT_ROOT =
`reports/reason_router_gen5_phase1b_r22_c22_construction_c9eca38_v1`

R22_PATH =
`reports/reason_router_gen5_phase1b_r22_c22_construction_c9eca38_v1/r22_basis.f64le`

R22_SHA256 =
`a69232900e8b5a91ec5248e36facee4d421eabd502829e6abf3d2719fdb02214`

C22_PATH =
`reports/reason_router_gen5_phase1b_r22_c22_construction_c9eca38_v1/c22_basis.f64le`

C22_SHA256 =
`c692d39a7387e32e9bf76fe4db7ce2d30d5f7d22a4389e8af363a80b394155c4`

RANK =
`2`

STATE_WIDTH =
`24576`

R22/C22 must be loaded from these exact frozen bytes.

Reconstruction, refit, reranking, alternate basis selection, or sign changes are
forbidden.

---

## 3. Frozen necessity-confirmation population

DATA_ROOT =
`data/reason_router_gen5_phase1b_xg1_necessity_confirmation_v1`

PAIR_RANGE =
`xg1_fact_8101..xg1_fact_8400`

PAIR_COUNT =
`300`

CHECKSUMS_SHA256 =
`c2413ead1bb0e55ab079c6f8832e5338ad14cc5fe7394ba4814b37040d93aaed`

STRUCTURED_SOURCE_FACTS_SHA256 =
`813534ffa5753dcf84637c43e177a6f96dac45230ceb3326747a528330e5b285`

SIX_CELL_ROWS_SHA256 =
`cf27ab041f77302562fd4cdd3fa4f2fc24e4b3f8f7367460512476b77efb1d14`

STRUCTURAL_MANIFEST_SHA256 =
`043fb36890b2663e430fc018b7a72fc1d9872b7bed15c019e2c9e8c626d98b58`

TOKENIZER_ANCHOR_MANIFEST_SHA256 =
`5ed5dc57b95df7121b301db2414c886655771dc77a53e0c6cbaad210d4683f88`

TOKENIZER_ELIGIBILITY_SUMMARY_SHA256 =
`752361c5367a68ea460f10dd554fd385d325d2c9434e998714366723c3004373`

Cohort replacement, filtering, row deletion, or response-guided exclusion are
forbidden.

The restoration-confirmation cohort must not be read by the implementation
except for path-name denial/static firewall checks.

---

## 4. Frozen model/checkpoint identity

REPRESENTATIVE_CHECKPOINT =
`reports/reason_router_gen3_grouped_factorial_runs/seed180/G3-GROUP-D-HALF/selected_checkpoint.pt`

REPRESENTATIVE_CHECKPOINT_SHA256 =
`1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f`

NATIVE_BACKBONE_SIGNATURE_SHA256 =
`81cd368d8a94932561e0ccd50f45a7db1f27941c00b3a08c8b816badaf25f415`

No checkpoint search or substitution is permitted.

---

## 5. Frozen necessity intervention

UPSTREAM_LAYER17_CONDITION =
`NATIVE`

TARGET_LAYER =
`22`

TARGET_OBJECT =
`WRITE22 = deltaB_u[:, :, target_token, :]`

CONDITIONS =
`NATIVE22 / R22_NEUTRALIZED / C22_COEFFICIENT_CONTROL`

For native write vector `w`:

`a = R22^T w`

R22 neutralization:

`N_R(w) = w - R22 a`

Matched C22 coefficient-transfer control:

`N_C(w) = w - C22 a`

The same exact R22-derived coefficient vector `a` must be used for both
interventions.

The implementation must enforce:

`||w - N_R(w)||_2 = ||w - N_C(w)||_2`

within the frozen numerical tolerance.

The remainder of the forward must remain unchanged.

---

## 6. Frozen broad endpoint

For each item and each layer-22 condition compute the unchanged frozen broad
endpoint:

`Q = E_XG2 - E_XG4`

yielding:

`Q0`
`QR`
`QC`

Then:

`A_R = Q0 - QR`

`A_C = Q0 - QC`

Primary necessity contrast:

`D_NEC22 = QC - QR`

The implementation may reuse already-frozen Gen4 XG2/XG4 broad-endpoint
machinery.

It must not alter:

- XG2/XG4 direction identity;
- direction count;
- target token;
- orientation convention;
- broad-endpoint aggregation;
- intervention magnitude;
- layer selection.

No alternative endpoint may be added.

---

## 7. Frozen confirmatory decision rule

Exactly one confirmatory p-value is authorized in the future execution stage.

Positive necessity requires all of:

1. all provenance/manipulation gates pass;
2. `mean(Q0) > 0`;
3. `mean(A_R) > 0`;
4. `mean(D_NEC22) > 0`;
5. one-sided one-sample Student t-test on `D_NEC22` gives `p < 0.05`.

Positive label:

`GEN5_R22_LOCAL_NECESSITY_OVER_MATCHED_C22_CONTROL_SUPPORTED`

Otherwise:

`GEN5_R22_LOCAL_NECESSITY_NOT_ESTABLISHED`

Implementation may encode and unit-test this frozen rule using synthetic inputs.

No real necessity-confirmation p-value may be computed during implementation or
verification.

---

## 8. Mandatory manipulation checks to implement

The future runner must fail closed on:

- exact checkpoint SHA256 mismatch;
- exact R22/C22 SHA256 mismatch;
- necessity population mismatch;
- target-token mismatch across conditions;
- R22/C22 shape mismatch;
- R22/C22 orthonormality failure;
- R22/C22 cross-orthogonality failure;
- coefficient-transfer norm inequality;
- unintended layer/token modification;
- incorrect layer-22 native-write site;
- incorrect post-state ordering relative to the modified recurrent write;
- parameter mutation;
- any training/backward/task-head optimization;
- response-guided row dropping.

The runner must also preserve descriptive native diagnostics without adding
confirmatory p-values.

---

## 9. Allowed implementation files

Exactly these new files may be created:

`scripts/reason_router_gen5_phase1b_r22_local_necessity_confirmation.py`

`scripts/verify_reason_router_gen5_phase1b_r22_local_necessity_confirmation.py`

`tests/test_reason_router_gen5_phase1b_r22_local_necessity_confirmation.py`

Existing production files must not be modified.

README must not be modified.

---

## 10. Implementation validation

Required before implementation freeze:

1. `git diff --check`
2. narrow pytest for the new test file
3. independent static verifier
4. zero model load
5. zero checkpoint load
6. zero model forward
7. zero CUDA execution
8. zero scientific execution

The static verifier must independently check at minimum:

- exact parent/frozen artifact identities;
- exact necessity population identities;
- no restoration-cohort scientific access;
- exact R22/C22 load/geometry contract;
- exact coefficient-transfer semantics;
- exact layer-22 write replacement semantics;
- exact frozen Q endpoint identities;
- exact single-p-value decision rule;
- artifact boundary;
- no hidden training/backward path.

---

## 11. Forbidden implementation changes

Forbidden:

- R22/C22 recomputation;
- rank search;
- alternative control construction;
- layer search;
- token search;
- checkpoint search;
- direction search;
- epsilon/magnitude tuning;
- confirmation-row filtering;
- extra confirmatory tests;
- restoration implementation;
- ownership implementation;
- training or evaluation execution;
- README changes;
- unrelated refactoring.

---

## 12. Stop conditions

Implementation must stop rather than improvise if:

- the frozen R22/C22 bytes are absent or mismatched;
- exact necessity cohort identity cannot be authenticated;
- the historical broad-Q implementation cannot be reused without changing its
  scientific semantics;
- exact native-write intervention cannot be implemented at the frozen layer-22
  source role;
- a required change falls outside the three-file whitelist;
- scientific execution would be required to validate correctness.

Any such issue requires a new explicit correction authority rather than silent
semantic relaxation.

---

## 13. Success boundary

Successful implementation plus independent verification establishes only:

`PASS_READY_FOR_GEN5_PHASE1B_NECESSITY_EXECUTION_AUTHORITY`

It does not establish local necessity.

It does not authorize necessity execution.

It does not authorize restoration.

It does not support the Phase 1B bridge claim.

STATUS =
`FROZEN_ON_COMMIT`
