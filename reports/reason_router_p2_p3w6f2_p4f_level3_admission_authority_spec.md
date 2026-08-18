# P3-W6-F2-P4-F Level-3 Admission Authority Specification

Decision scope: Level-3 phase-admission authority after Gate-6 Level-2 closure.

This document is specification-only. It creates no result artifact, runs no command, releases no training admission, and authorizes no execution by itself.

## A. Authority Freeze

Gate-6 closure authority:

- final closure commit: `08d233d53b9d33e055f04c833613edb313bd0cf5`
- final closure message: `Close P3W6F2P4E Gate-6 static verification`
- final Gate-6 decision token: `P3W6F2P4E_LEVEL2_RESULT_REVIEW_PASS`
- pair-dispositions SHA256: `cb8ca8b46e5440867cb1d22132ed5738e030bdc818dd685ffc8a0e6cc732cdce`
- current `training_admission_released`: `false`

Frozen Gate-6 Level-2 authority:

- P4-E specification path: `reports/reason_router_p2_p3w6f2_p4e_level2_result_review_spec.md`
- P4-E specification freeze commit: `d0e9f364ee5cab3456d9f902e42289de4bd09cd0`
- primary artifact freeze commit: `f9f074772ac6b4e2718eddee4588b3b8b57c4634`
- independent-verification attestation commit: `2d63c565eac34c9cd369ccefe8846c7c282d04ed`
- final Gate-6 artifact directory: `reports/reason_router_p2_p3w6f2_p4e_level2_result_review_d0e9f364ee5cab3456d9f902e42289de4bd09cd0/`
- final summary schema: `P3W6F2P4E_LEVEL2_RESULT_REVIEW_SUMMARY_V1`

Resolved P4-B/P4-D prerequisites:

- P4-B R1 regeneration specification: `reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_spec.md`
- P4-B R1 authority commit: `fcc3b9ccaf2bbee33ac18dcef10d50acff54aab4`
- P4-B R1 artifact directory: `reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/`
- regenerated dataset SHA256: `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`
- regenerated dataset semantic SHA256: `3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b`
- P4-D controlled-data integrity gate specification: `reports/reason_router_p2_p3w6f2_p4d_controlled_data_integrity_gate_spec.md`
- P4-D authority commit: `1be4050dbadb0cb5ed2f6b55a2391051f5f6c07e`
- official Gate-5 run/head/token: `p3w6f2-p4d-gate5-official-eced1d4`, `eced1d46e8788e4372eca14dcf090c2840649399`, `P3W6F2P4D_CONTROLLED_DATA_INTEGRITY_GATE_PASS`
- Gate-5 command SHA256: `b2e1efae4c06ee9a312b0b7e0ca0a8b40701eca4e461a05e629769f9c553eecd`
- imported Gate-5 ZIP/run-log/run-meta SHA256: `4e42868c437eb361292a9123e37fbab1be7e12a3fb36297228624b19cf965666`, `26161f680386a8048d942066accf5554aa887b694a04d6a2f1aeb1582484b58c`, `c74992a686d7952144b4220c303d0eecd42227ab55ca6326600074c78c72c910`

Resolved historical Stage184/Stage185 context:

- Stage184-A specification policy: `reports/stage184a_controlled_train_integrity_mask_policy.md`
- Stage184-A policy commit: `669be84ab929635bb247c2a51f49ec0edeef5bb0`
- Stage184-A closure: `reports/stage184a_integrity_sidecar_spec_ready_closure.md`
- Stage184-A closure JSON: `reports/stage184a_integrity_sidecar_spec_ready_closure.json`
- Stage184-A closure commit: `d89edf49454b02576733137f68f7f9309956487c`
- Stage184-A decision token: `STAGE184A_DETERMINISTIC_FAIL_CLOSED_INTEGRITY_SIDECAR_SPEC_READY`
- Stage184-A authorized next-stage token: `STAGE185_CONTROLLED_TRAIN_INTEGRITY_SIDECAR_BUILDER`
- Stage185-A policy: `reports/stage185a_controlled_train_integrity_sidecar_policy.md`
- Stage185-A policy commit: `d89edf49454b02576733137f68f7f9309956487c`
- Stage185-A closure: `reports/stage185a_integrity_sidecar_materialized_closure.md`
- Stage185-A closure JSON: `reports/stage185a_integrity_sidecar_materialized_closure.json`
- Stage185-A closure commit: `7940a72324ae15434649580ecf59f4140578c635`
- Stage185-A decision token: `STAGE185A_INTEGRITY_SIDECAR_BUILT_AND_POSITIVE_ELIGIBILITY_MATERIALIZED`
- authoritative historical Stage185-A sidecar directory: `reports/stage185a_controlled_train_integrity_sidecar_20260715_141914`
- authoritative historical Stage185-A sidecar semantic SHA256: `5bc03caa2a29f9b9176ab4eb0201db57ebad516352797546db1a18e6ec3373fc`
- Stage185-A authorized next-stage token: `STAGE186_COMPATIBLE_POSITIVE_MARGIN_FIXED_SPEC_AUDIT`

The Stage184/Stage185 lineage above remains historical authority context and may be a prerequisite where a future governing authority explicitly cites it. It does not itself resolve the first post-Gate-6 Level-3 action, does not automatically require a new Stage185-style execution, and does not authorize any execution against the P4-B R1 regenerated dataset.

Repository search found no established canonical P4-F or Level-3 admission token namespace. The only equivalent phrase found was the P4-B statement that a separate Level-3 training-admission decision gate is required. That statement is a requirement for a new authority, not an existing canonical token namespace. If a future reviewer finds an older applicable canonical namespace, the P4-F decision must be BLOCKED and the conflict reported.

Current earliest-action authority state:

`earliest_level3_action_status = UNRESOLVED_BY_FROZEN_AUTHORITY`

The current frozen repository authority does not uniquely identify the first post-Gate-6 Level-3 executable or static action. P4-F must not choose one.

## B. Level-3 Admission Definition

Level-3 admission has two distinct authority concepts:

1. Level-3 phase/admission authority determines whether prerequisites for entering a separately governed Level-3 decision process are valid.
2. Level-3 action-resolution authority is a separate frozen authority that must explicitly identify the exact earliest allowed action before P4-F can produce PASS.

P4-F is a Level-3 phase/admission specification. It is not itself the missing Level-3 action-resolution authority. It may define the contract that such an authority must satisfy, but it must not name a specific action absent separate governing evidence.

Level-3 phase admission is a research-phase authorization boundary after successful Gate-6 Level-2 closure. It means only that the repository may move from Level-2 remediation closure into a separately governed Level-3 decision process once the exact action-resolution authority exists.

Level-3 admission does not automatically mean:

- model training
- evaluation
- model loading
- checkpoint creation, use, or mutation
- promotion
- Kaggle GPU execution
- dataset replacement
- promotion-criteria changes

Current action-resolution state:

`earliest_level3_action_status = UNRESOLVED_BY_FROZEN_AUTHORITY`

The current frozen repository authority does not uniquely identify the first post-Gate-6 Level-3 executable or static action. P4-F must not manufacture that missing action-resolution authority, must not infer it from Stage184/Stage185 lineage, and must not choose one.

P4-F PASS is impossible while `earliest_level3_action_status = UNRESOLVED_BY_FROZEN_AUTHORITY`. Until a valid separate action-resolution authority exists, an actual P4-F admission evaluation must return `P3W6F2P4F_LEVEL3_ADMISSION_BLOCKED` with blocker `EARLIEST_LEVEL3_ACTION_AUTHORITY_UNRESOLVED` or a semantically equivalent blocker. This is an authority/provenance BLOCKED state, not a scientific FAIL.

## C. Stage185 Distinction

P4-F distinguishes three surfaces:

1. Historical Stage185 evidence and compatibility authority: P4-B artifacts 8-10 and Gate-6 verified Stage185 compatibility prerequisite PASS for the 119 authorized F2 pairs. This evidence preserves raw Stage185 observations and proves P4-B predicate-realization compatibility; it does not authorize a new Stage185 run.
2. Historical Stage184/Stage185 sidecar lineage: the Stage184-A/Stage185-A lineage records a fail-closed sidecar builder/static-audit history with exact row identity, one-to-one joins, historical seed-174 split replay, semantic SHA, and no model/Torch/training/checkpoint operations. This history does not itself resolve the first post-Gate-6 Level-3 action.
3. Model training/evaluation: any loss implementation, target-margin selection, nonzero weight, checkpoint-selection change, model loading, training, evaluation, promotion, or Kaggle GPU workload is outside P4-F admission and requires separate explicit authority.

Historical Stage185 compatibility PASS must not be converted into authorization to execute Stage185 again. Historical Stage185 sidecar authority does not automatically require a new Stage185-style execution. Predicate-realization compatibility does not authorize execution. Gate-5/P4-D integrity validation does not create a new Level-3 execution requirement. A future action-resolution authority may cite Stage184/Stage185, but P4-F must not infer the action solely from that lineage.

Seed 174 and dev ratio 0.2 are not normative requirements of the currently unresolved future Level-3 action. They remain documented only as historical/frozen parameters where they actually applied, including Stage185 split lineage and P4-D Gate-5 replay/validation. They become requirements of a future Level-3 action only if the separate action-resolution authority explicitly incorporates them.

## D. Training Admission Invariant

This specification preserves:

`training_admission_released = false`

Neither P4-F specification freeze, P4-F evaluation, P4-F PASS, future Level-3 phase admission, nor a future non-training earliest action may change that field to `true`. No such authority was resolved in this specification pass. Therefore, training admission remains false after Level-3 phase admission.

Any future `false -> true` transition requires a separate explicit authority and an auditable provenance contract.

## E. Exact Admission Prerequisites

Future P4-F Level-3 admission PASS requires all of the following:

- Gate-6 final closure commit exists and equals `08d233d53b9d33e055f04c833613edb313bd0cf5`.
- Gate-6 decision token equals `P3W6F2P4E_LEVEL2_RESULT_REVIEW_PASS`.
- Pair-dispositions SHA256 equals `cb8ca8b46e5440867cb1d22132ed5738e030bdc818dd685ffc8a0e6cc732cdce`.
- `verification_commit` equals `2d63c565eac34c9cd369ccefe8846c7c282d04ed` and is non-null.
- Primary artifact freeze commit equals `f9f074772ac6b4e2718eddee4588b3b8b57c4634`.
- P4-E specification freeze commit equals `d0e9f364ee5cab3456d9f902e42289de4bd09cd0`.
- Gate-6 counts are exactly 119 PASS, 0 FAIL, 0 BLOCKED, and 0 primary/verifier mismatch.
- Unresolved semantic issue count is 0.
- Unauthorized mutation is false.
- Authority/provenance ambiguity count is 0.
- `training_admission_released` is currently false.
- P4-B authority commit, artifact directory, regenerated dataset SHA256, and regenerated semantic SHA256 match Section A.
- P4-D authority commit, Gate-5 PASS token, Gate-5 run/head, command hash, and imported provenance hashes match Section A.
- Relevant Stage184-A historical authority is valid where cited as context: decision `STAGE184A_DETERMINISTIC_FAIL_CLOSED_INTEGRITY_SIDECAR_SPEC_READY`, authorized next stage `STAGE185_CONTROLLED_TRAIN_INTEGRITY_SIDECAR_BUILDER`, source dataset identity recorded, historical split contract preserved, sidecar/static audit only, and training false.
- Relevant Stage185-A authority is valid: decision `STAGE185A_INTEGRITY_SIDECAR_BUILT_AND_POSITIVE_ELIGIBILITY_MATERIALIZED`, historical sidecar semantic SHA exact, failed directory excluded, blocked invariants 0, Stage182 overlap regression passed, and training false.
- Repository HEAD/authority ancestry is coherent: P4-E spec, primary artifact freeze, independent attestation, and final closure commits are ancestors of the admission evaluation state.
- No conflicting newer authority exists.
- No existing canonical Level-3 admission namespace conflicts with Section G.
- No widened scientific population is used; the Gate-6 scientific population remains exactly 119 F2 pairs.
- No changed promotion criteria, split semantics, label semantics, dataset identity semantics, or external-label use is introduced.
- A separate earliest Level-3 action-resolution authority exists.
- The action-resolution authority path, full 40-character commit SHA, and canonical token/version are exact.
- The exact earliest Level-3 action is unambiguous.
- The exact earliest Level-3 action does not exceed its action-resolution authority.
- Any required action parameters are authority-bound by the separate action-resolution authority.
- Subsequent explicit workflow authorization exists for admission evaluation; independent verification and commit alone are insufficient.

P4-F PASS is impossible while `earliest_level3_action_status = UNRESOLVED_BY_FROZEN_AUTHORITY`.

A future separate action-resolution authority must supply, at minimum:

- exact authority path
- exact authority full 40-character commit SHA
- exact canonical token/version
- exact earliest action name
- exact action purpose
- whether the action is static/read-only
- whether the action is deterministic artifact generation
- whether the action is CPU execution
- whether the action is GPU execution
- whether the action is model/training/evaluation
- exact input dataset/artifact identities
- exact permitted outputs
- exact prohibited outputs/actions
- execution environment requirements, if any
- whether separate execution authorization is required
- `training_admission_released` effect, defaulting to false unless separately explicitly authorized
- split or seed parameters only if that action-resolution authority itself requires them

P4-F must fail closed if any of these are ambiguous.

## F. Admission Decision Artifact Contract

This specification defines future result artifacts but does not create them.

Future result filename:

`reports/reason_router_p2_p3w6f2_p4f_level3_admission_result_<admission_commit>.json`

`<admission_commit>` must be the full 40-character commit SHA of the future committed admission-evaluation state. The artifact must use deterministic UTF-8 JSON with stable keys and LF line endings.

Schema/version:

`P3W6F2P4F_LEVEL3_ADMISSION_RESULT_V1`

Minimum required fields:

- `schema_version`
- `decision_token`
- `gate6_closure_commit`
- `gate6_decision_token`
- `gate6_pair_dispositions_sha256`
- `gate6_verification_commit`
- `level3_action_resolution_status`
- `level3_action_authority_path`
- `level3_action_authority_commit`
- `level3_action_authority_token`
- `earliest_level3_action`
- `earliest_level3_action_type`
- `separate_execution_authorization_required`
- `training_admission_released`
- `prerequisite_statuses`
- `blockers`
- `failure_reasons`
- `admission_commit`
- `verification_commit`
- `p4f_spec_commit`
- `p4e_spec_commit`
- `primary_artifact_freeze_commit`
- `independent_attestation_commit`
- `p4b_authority_commit`
- `p4b_artifact_directory`
- `p4b_artifact_hashes`
- `p4d_authority_commit`
- `p4d_gate5_pass_token`
- `stage184_authority`
- `stage185_authority`
- `admission_scope`
- `prohibited_actions`

For the current unresolved state, `level3_action_authority_path`, `level3_action_authority_commit`, `level3_action_authority_token`, `earliest_level3_action`, and any unresolved action-type detail may be null; `level3_action_resolution_status` must express unresolved authority, and the decision must be `P3W6F2P4F_LEVEL3_ADMISSION_BLOCKED` if an admission evaluation were performed.

No model, checkpoint, training, evaluation, or promotion artifact is required or allowed by this contract.

## G. Decision Semantics

Because no equivalent canonical namespace was found, P4-F decision tokens are exactly:

- `P3W6F2P4F_LEVEL3_ADMISSION_PASS`
- `P3W6F2P4F_LEVEL3_ADMISSION_FAIL`
- `P3W6F2P4F_LEVEL3_ADMISSION_BLOCKED`

`P3W6F2P4F_LEVEL3_ADMISSION_PASS` is possible only after a separate exact action-resolution authority exists and all admission prerequisites pass. PASS means only that the exact action named by that separate authority becomes eligible for a subsequent explicit workflow/execution authorization. PASS does not execute the action.

`P3W6F2P4F_LEVEL3_ADMISSION_FAIL` means valid evidence establishes a substantive incompatibility with Level-3 admission criteria.

`P3W6F2P4F_LEVEL3_ADMISSION_BLOCKED` means required authority, action resolution, provenance, evidence, schema, workflow authorization, or earliest-action identity is absent, ambiguous, conflicting, malformed, or incomplete. The current absence of an exact earliest-action authority is BLOCKED with blocker `EARLIEST_LEVEL3_ACTION_AUTHORITY_UNRESOLVED` or a semantically equivalent blocker.

Missing authority must be BLOCKED, not FAIL.

## H. PASS Boundary

A future `P3W6F2P4F_LEVEL3_ADMISSION_PASS` may make eligible only the exact earliest Level-3 action named by a separate frozen Level-3 action-resolution authority.

It must not automatically authorize:

- training
- evaluation
- model loading
- checkpoint creation, use, or mutation
- promotion
- changing promotion criteria
- dataset replacement
- GPU use

If a future action-resolution authority names an executable or artifact-generating action, that action still requires subsequent explicit execution/workflow authorization. Any future materialization must occur under a new committed execution state and may not mutate historical datasets, P4-B/P4-D/P4-E artifacts, Stage184/185 scripts, model code, checkpoint state, or promotion criteria unless separately and explicitly authorized.

## I. Kaggle Boundary

P4-F does not assume Kaggle is needed because no earliest Level-3 action is resolved by frozen authority. Kaggle remains unauthorized.

If a later deterministic execution authority requires Kaggle, that future workflow still needs separate explicit authorization, a specific committed code state, checkout verification of the full commit SHA, and GPU OFF unless a separately authorized GPU workload actually requires it.

## J. No Hidden Execution Authority

Creating, fixing, or freezing this P4-F specification does not authorize:

- Level-3 action resolution
- Level-3 execution
- Stage185 execution
- Python
- pytest
- regeneration
- training
- evaluation
- Kaggle
- model or checkpoint operations

Future Level-3 admission evaluation may begin only after:

1. this specification is independently verified;
2. this specification is frozen/committed;
3. subsequent explicit workflow authorization is granted.

Independent verification plus commit alone are insufficient.

This P4-F specification is not the missing Level-3 action-resolution authority and does not create one.

## K. Fail-Closed Conditions

Future P4-F admission must return BLOCKED for:

- Gate-6 closure identity mismatch
- Gate-6 decision token other than `P3W6F2P4E_LEVEL2_RESULT_REVIEW_PASS`
- pair hash mismatch
- `verification_commit` missing or mismatched
- Level-2 counts mismatch
- unresolved semantic issues
- unauthorized mutation
- authority/provenance ambiguity
- ambiguous Stage184/Stage185 authority
- missing earliest Level-3 action-resolution authority
- ambiguous action-resolution authority path, commit, token, action, action type, input, output, prohibition, environment, execution-authorization, training-admission, split, seed, or parameter binding
- unclear earliest Level-3 action
- conflict with an existing admission authority or token namespace
- attempt to equate Level-3 admission with training admission without authority
- attempt to change `training_admission_released` to true
- attempt to infer training/evaluation authority from Gate-6 PASS
- attempt to infer Stage185 execution authority from historical Stage185 compatibility PASS
- missing subsequent explicit workflow authorization
- any widened scientific population
- any changed promotion criteria

FAIL is reserved for valid evidence of substantive incompatibility with the admission criteria. Missing, ambiguous, or conflicting authority remains BLOCKED.

## L. Independent Verification

This P4-F specification requires independent verification before freeze.

The future completed P4-F admission result artifact also requires independent verification before any `P3W6F2P4F_LEVEL3_ADMISSION_PASS` is treated as final authority.

The same review pass must not serve as both primary admission evaluation and independent verification.

## M. Scientific Boundary

P4-F is an authorization/admission layer only. It must not reopen or reinterpret:

- the 119-pair Gate-6 scientific result
- F2 remediation criteria
- P4-B regeneration
- Gate-5 integrity
- Stage185 compatibility result

This specification must not change:

- P4-E specification or artifacts
- P4-D files
- P4-B artifacts
- datasets
- Stage184/Stage185 scripts
- controlled-train artifacts
- model/training/evaluation code
- root `*.patch` files
- `reports/stage180a_pass2_annotations_completed.csv`

Final specification-readiness token:

`P3W6F2P4F_LEVEL3_ADMISSION_AUTHORITY_SPEC_ACTION_RESOLUTION_REPAIR_READY_FOR_INDEPENDENT_REVERIFICATION`
