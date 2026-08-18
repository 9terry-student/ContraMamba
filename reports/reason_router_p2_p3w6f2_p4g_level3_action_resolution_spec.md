# P3-W6-F2-P4-G Level-3 Action-Resolution Authority Specification

Decision scope: specification for a future static, read-only Level-3 action-resolution review.

This document is specification-only. It does not perform the action-resolution review, select an action, create a result artifact, run code, release training admission, authorize Kaggle, authorize GPU, authorize model loading, authorize training, or authorize evaluation.

## A. Authority Freeze

Frozen P4-F admission authority:

- P4-F specification path: `reports/reason_router_p2_p3w6f2_p4f_level3_admission_authority_spec.md`
- P4-F specification freeze commit: `8fd76261fbad2ef78e915e3442147c971e95aa33`

Gate-6 closure authority:

- P4-E/Gate-6 final closure commit: `08d233d53b9d33e055f04c833613edb313bd0cf5`
- Gate-6 decision token: `P3W6F2P4E_LEVEL2_RESULT_REVIEW_PASS`
- Gate-6 primary artifact freeze commit: `f9f074772ac6b4e2718eddee4588b3b8b57c4634`
- Gate-6 independent attestation commit: `2d63c565eac34c9cd369ccefe8846c7c282d04ed`
- Gate-6 pair-dispositions SHA256: `cb8ca8b46e5440867cb1d22132ed5738e030bdc818dd685ffc8a0e6cc732cdce`
- `training_admission_released`: `false`

Resolved P4-B/P4-D authorities:

- P4-B R1 regeneration authority commit: `fcc3b9ccaf2bbee33ac18dcef10d50acff54aab4`
- P4-D controlled-data integrity authority commit: `1be4050dbadb0cb5ed2f6b55a2391051f5f6c07e`

Relevant historical Stage184/Stage185 authorities:

- Stage184-A policy commit: `669be84ab929635bb247c2a51f49ec0edeef5bb0`
- Stage184-A closure commit: `d89edf49454b02576733137f68f7f9309956487c`
- Stage184-A decision token: `STAGE184A_DETERMINISTIC_FAIL_CLOSED_INTEGRITY_SIDECAR_SPEC_READY`
- Stage184-A authorized next-stage token: `STAGE185_CONTROLLED_TRAIN_INTEGRITY_SIDECAR_BUILDER`
- Stage185-A closure commit: `7940a72324ae15434649580ecf59f4140578c635`
- Stage185-A decision token: `STAGE185A_INTEGRITY_SIDECAR_BUILT_AND_POSITIVE_ELIGIBILITY_MATERIALIZED`
- Stage185-A authorized next-stage token: `STAGE186_COMPATIBLE_POSITIVE_MARGIN_FIXED_SPEC_AUDIT`

P4-G is subordinate to frozen P4-F for the admission contract. P4-G may resolve the missing action identity for future P4-F consumption, but it may not weaken P4-F training, execution, provenance, independent-verification, or workflow-authorization boundaries.

The historical Stage184/Stage185 authorities are preserved as historical context unless a current governing authority explicitly binds them to the post-Gate-6 P3-W6-F2 / P4-B R1 lineage.

## B. Current Unresolved State

Current action-resolution status:

`current_action_resolution_status = UNRESOLVED_BY_FROZEN_AUTHORITY`

No action is selected by this specification.

Creating, verifying, freezing, or committing this specification must not change `current_action_resolution_status` to resolved. It must remain unresolved until a later explicitly authorized P4-G action-resolution review produces a valid frozen result.

## C. Purpose Of Future P4-G Review

The future P4-G action-resolution review must answer:

`What is the exact earliest action, if any, that current frozen repository authority uniquely supports after Gate-6 Level-2 closure and before any training/evaluation authority?`

The review is authority resolution, not scientific experimentation.

It must not answer:

- What action seems useful?
- What should we try next?
- What historically came next?

The future review must determine exactly one of:

- one uniquely authority-supported earliest post-Gate-6 Level-3 action; or
- `BLOCKED` because no unique action can be validly resolved.

## D. Candidate-Action Discovery Contract

The future reviewer must exhaustively identify all materially plausible candidate actions supported by repository authority. Candidate discovery must inspect both repository contents and Git history.

Candidate discovery must inspect at least:

1. P4-F frozen authority.
2. P4-E/Gate-6 closure and boundaries.
3. P4-B clauses describing the later Level-3 gate.
4. P4-D/Gate-5 release boundary.
5. Applicable Stage184/Stage185 authority, including explicit authorized-next-stage tokens.
6. Any later committed authority that explicitly claims applicability to the same P3-W6-F2 / P4-B R1 post-Gate-6 lineage.
7. Repository `AGENTS.md` where applicable.
8. README/history only as non-authoritative context.

Candidate discovery must not be restricted to Stage185-related actions.

Candidate search evidence is required. Missing candidate search evidence, incomplete candidate universe, or a search restricted by historical chronology must produce `BLOCKED`.

## E. Candidate Admissibility Contract

A candidate action may be considered authority-supported only if its source authority provides or permits exact resolution of:

- authority path
- full 40-character authority commit
- canonical token/version
- action name
- action purpose
- applicability to the current P3-W6-F2 post-Gate-6 lineage
- input dataset/artifact identity
- action type
- permitted outputs
- prohibited actions
- environment requirements
- separate execution authorization requirement
- `training_admission_released` effect
- parameter bindings, if any

Historical chronology alone is insufficient.

An `authorized next stage` token from historical Stage184/185 is insufficient unless current authority explicitly binds that lineage/action to the current P4-B R1 regenerated dataset and post-Gate-6 state.

If any required candidate field is absent, ambiguous, conflicting, or merely inferred from convenience, the candidate is not eligible unless the correct decision is instead `BLOCKED` due to unresolved authority.

## F. Applicability To Current Lineage

Any selected candidate must have explicit proof that it applies to:

- P3-W6-F2
- P4-B R1 regenerated controlled dataset
- successful Gate-6 Level-2 closure
- current `training_admission_released=false` state

A historical action defined for the old controlled dataset or a different remediation lineage may not be silently transplanted.

If applicability to the current lineage is ambiguous, the future review must return `P3W6F2P4G_LEVEL3_ACTION_RESOLUTION_BLOCKED`.

## G. Precedence And Selection Rule

The future P4-G review may PASS only if exactly one earliest action is uniquely supported after all applicability and precedence checks.

Do not choose by:

- stage number alone
- commit date alone
- file name alone
- historical chronology alone
- implementation convenience
- compute cost
- presumed scientific usefulness

Selection must use authority applicability and explicit release/next-stage semantics.

If two or more non-dominated candidates remain, the result must be `BLOCKED`.

If zero candidates satisfy the contract, the result must be `BLOCKED`.

Do not ask the user to choose unless the remaining alternatives are both valid but represent genuinely different scientific objectives that frozen authority cannot order.

## H. Earliest-Action Definition

`earliest` means the narrowest action that must validly occur before any other currently authority-supported post-Gate-6 action can occur.

It does not mean:

- lowest stage number
- oldest historical successor
- cheapest action
- most conservative action

The selected action must be necessary or explicitly first according to applicable authority.

## I. Action Type

The future selected action must have exactly one resolved action type.

Allowed action-type enum:

- `STATIC_READ_ONLY`
- `DETERMINISTIC_CPU_ARTIFACT`
- `DETERMINISTIC_GPU_ARTIFACT`
- `MODEL_LOADING`
- `TRAINING`
- `EVALUATION`
- `OTHER_EXPLICITLY_AUTHORIZED`

This enum defines result vocabulary only. It does not imply that every enum value is currently allowable. The selected type must be the smallest exact type supported by the selected action's applicable authority.

## J. Training-Admission Boundary

Mandatory frozen state:

`training_admission_released = false`

P4-G PASS itself must not change this field.

If a candidate action is training, evaluation, model, checkpoint, promotion, or model-loading related, it can be selected only if an applicable current authority already explicitly authorizes that semantic transition. P4-G must not create such authority.

Absent exact current authority, such a candidate must be rejected or the review must return `BLOCKED`, depending on whether the absence eliminates only that candidate or leaves the candidate universe unresolved.

P4-G PASS does not itself authorize training or execution.

## K. Stage184 / Stage185 Boundary

The future review must explicitly preserve:

- historical Stage184/Stage185 evidence
- historical next-stage tokens
- P4-B Stage185 compatibility
- P4-D controlled-data integrity result

None is automatically the selected current action.

If a Stage184/185-derived candidate survives, the result must explain the exact authority bridge from historical lineage to current P4-B R1/Gate-6 state.

No bridge means the candidate is not selectable.

## L. Parameter Inheritance

No historical parameter may automatically propagate into the selected action.

This includes:

- seed `174`
- dev ratio `0.2`
- historical sidecar paths
- historical dataset hash
- historical output directory conventions

A parameter is binding only if the selected action's applicable authority explicitly requires it.

Parameter inheritance without authority must produce candidate rejection or `BLOCKED`, as appropriate.

## M. Future Result Artifact

This specification defines but does not create the future result artifact.

The future result must be created against a pre-existing reviewed repository
state:

`resolution_state_commit`

`resolution_state_commit` is the full 40-character SHA of the already-existing
committed repository state against which the future static P4-G
action-resolution review is performed.

`resolution_state_commit`:

- must exist before the result artifact is materialized;
- must contain the frozen P4-G specification;
- must be the exact repository authority state reviewed;
- binds all worktree/provenance conditions required by the future review to
  that pre-existing state;
- is not the result artifact freeze commit.

No result filename or result content may depend on a commit SHA that does not
exist yet.

Future result filename:

`reports/reason_router_p2_p3w6f2_p4g_level3_action_resolution_result_<resolution_state_commit>.json`

`<resolution_state_commit>` must be known before file creation and must be the
same full 40-character SHA recorded in the result as `resolution_state_commit`.

Schema/version:

`P3W6F2P4G_LEVEL3_ACTION_RESOLUTION_RESULT_V1`

The future result artifact must include at least:

- `schema_version`
- `decision_token`
- `verification_status`
- `resolution_state_commit`
- `p4g_spec_commit`
- `p4f_spec_commit`
- `gate6_closure_commit`
- `gate6_decision_token`
- `gate6_pair_dispositions_sha256`
- `training_admission_released`
- `current_action_resolution_status`
- `candidate_action_count`
- `candidate_actions`
- `eligible_candidate_count`
- `dominated_candidate_count`
- `unresolved_candidate_count`
- `selected_action`
- `selected_action_type`
- `selected_action_purpose`
- `selected_action_authority_path`
- `selected_action_authority_commit`
- `selected_action_authority_token`
- `selected_action_input_identities`
- `selected_action_permitted_outputs`
- `selected_action_prohibited_actions`
- `selected_action_environment`
- `selected_action_parameters`
- `separate_execution_authorization_required`
- `applicability_to_current_lineage_status`
- `blockers`
- `failure_reasons`

Each candidate record must include:

- `candidate_name`
- `source_authority_path`
- `source_authority_commit`
- `source_authority_token`
- `current_lineage_applicability_status`
- `authority_support_status`
- `precedence_status`
- `blocker_rejection_reasons`
- `action_type`
- `training_admission_effect`

For `BLOCKED` or `FAIL`, selected-action fields may be null only where the decision semantics require nullability and the blockers/failure reasons explain the unresolved or invalid field exactly.

`verification_status` in the primary result must be exactly `PENDING`. The
primary result may record a candidate `PASS`, `FAIL`, or `BLOCKED`
`decision_token`, but the primary result itself is not independently verified
or consumable authority until the separate verification attestation workflow
defined below is completed and frozen.

The primary result JSON must not require, contain, or forward-predict:

- the Git commit that first freezes the primary result artifact;
- the Git commit that first freezes a later independent verification
  attestation artifact.

## N. Primary Result Freeze And Verification Attestation

`result_freeze_commit` is the later Git commit that first freezes the completed
primary result artifact. This SHA is obtained from Git after the primary result
artifact is committed. It must not be predicted or embedded as a required
self-reference inside the primary result.

After `result_freeze_commit` exists, a separate independent verification
attestation artifact may be created at:

`reports/reason_router_p2_p3w6f2_p4g_level3_action_resolution_verification_attestation_<result_freeze_commit>.json`

The attestation filename uses `result_freeze_commit`, which already exists
before the attestation is created.

Verification attestation schema/version:

`P3W6F2P4G_LEVEL3_ACTION_RESOLUTION_VERIFICATION_ATTESTATION_V1`

The verification attestation artifact must include at least:

- `schema_version`
- `p4g_spec_commit`
- `resolution_state_commit`
- `result_path`
- `result_freeze_commit`
- `primary_result_sha256`
- `primary_decision_token`
- `independent_verification_status`
- `independent_verification_token`
- `authority_binding_status`
- `candidate_universe_verification_status`
- `selected_action_verification_status`
- `provenance_verification_status`
- `training_admission_released`
- `blockers`
- `failure_reasons`

The attestation file must not require, contain, or forward-predict the Git
commit that first contains the attestation itself.

`verification_attestation_commit` is the full 40-character SHA of the later
immutable Git commit that freezes the independent verification attestation.
This SHA becomes known only after that commit. It is external Git provenance
and must not be forward-predicted inside the attestation file.

P4-G does not require rewriting the primary result merely to insert
`result_freeze_commit` or `verification_attestation_commit`. No finalization
commit is required solely for SHA backfilling.

The immutable provenance tuple sufficient to identify the independently
verified P4-G result is exactly:

- P4-G spec commit
- `resolution_state_commit`
- primary result path
- primary result SHA256
- `result_freeze_commit`
- verification attestation path
- verification attestation SHA256
- `verification_attestation_commit`

## O. Resolution Decision Tokens

Repository and Git-history search found no existing equivalent canonical P4-G / Level-3 action-resolution namespace. The only equivalent content found was P4-F language requiring a separate action-resolution authority; that language is a prerequisite, not an existing P4-G namespace.

Because no existing applicable namespace was found, P4-G decision tokens are exactly:

- `P3W6F2P4G_LEVEL3_ACTION_RESOLUTION_PASS`
- `P3W6F2P4G_LEVEL3_ACTION_RESOLUTION_FAIL`
- `P3W6F2P4G_LEVEL3_ACTION_RESOLUTION_BLOCKED`

If a future reviewer discovers an existing applicable canonical namespace, the review must STOP and report the namespace conflict. A namespace conflict is `BLOCKED`, not PASS.

## P. Decision Semantics

`P3W6F2P4G_LEVEL3_ACTION_RESOLUTION_PASS` means:

- exactly one eligible non-dominated earliest action remains;
- all authority, applicability, and provenance fields are exact;
- the selected action applies to the current P3-W6-F2/Gate-6 lineage;
- the training-admission boundary remains valid;
- no unresolved conflict remains.

`P3W6F2P4G_LEVEL3_ACTION_RESOLUTION_FAIL` means valid evidence proves the action-resolution prerequisites themselves are substantively violated in a way that is not merely missing authority.

`P3W6F2P4G_LEVEL3_ACTION_RESOLUTION_BLOCKED` means one or more of:

- zero eligible actions
- multiple non-dominated eligible actions
- applicability ambiguity
- missing or conflicting authority
- unresolved path, commit, or token
- unresolved action type, scope, input, output, environment, or parameter binding
- current-lineage bridge missing
- token namespace collision
- provenance ambiguity

Missing authority is `BLOCKED`, not `FAIL`.

## Q. PASS Output Semantics

A primary result with decision token
`P3W6F2P4G_LEVEL3_ACTION_RESOLUTION_PASS` is only a candidate PASS until all of
the following exist:

1. primary result artifact frozen at `result_freeze_commit`;
2. independent verification attestation verdict PASS;
3. attestation frozen at `verification_attestation_commit`;
4. all artifact hashes and authority identities match;
5. no verification blocker exists.

Only then may the P4-G PASS result be treated as independently verified
action-resolution authority and supplied to a later separately authorized P4-F
admission evaluation.

Independent verification verdicts must be represented truthfully:

- verification PASS means the result may become consumable if the primary
  decision token is `P3W6F2P4G_LEVEL3_ACTION_RESOLUTION_PASS`;
- verification FAIL means the result is not consumable;
- verification BLOCKED means the result is not consumable.

The independent verification verdict must not be equated with the primary
`PASS`, `FAIL`, or `BLOCKED` scientific/authority-resolution decision.

Independently verified P4-G PASS establishes only:

`the exact earliest Level-3 action identity has been resolved.`

It may provide the selected authority path, commit, and token needed by P4-F.

P4-G PASS does not:

- execute the selected action
- make P4-F automatically PASS
- release training admission
- authorize Kaggle
- authorize GPU
- authorize Stage185 execution
- authorize model loading
- authorize checkpoints
- authorize training
- authorize evaluation
- authorize promotion
- replace datasets

After a valid frozen P4-G result, P4-F still requires its own separately authorized admission evaluation.

A valid verified P4-G PASS handoff to P4-F must provide:

- P4-G specification path/commit
- primary result path
- primary result SHA256
- `resolution_state_commit`
- `result_freeze_commit`
- P4-G decision token
- selected action identity/type
- selected action authority path/commit/token
- verification attestation path
- verification attestation SHA256
- `verification_attestation_commit`
- independent verification token
- `training_admission_released=false`

P4-F must reject or block a P4-G PASS handoff if any of these provenance
identities are missing or inconsistent.

## R. No Hidden Execution Authority

Creating, verifying, freezing, or committing this P4-G specification authorizes no action-resolution review by itself.

Future P4-G action-resolution review may begin only after all three:

1. P4-G specification independent verification PASS.
2. P4-G specification freeze/commit.
3. Subsequent explicit workflow authorization.

Verification plus commit alone are insufficient.

Likewise, a future P4-G PASS does not execute the selected action.

Future primary-result freeze, verification-attestation freeze, and any later
P4-F evaluation each remain separately controlled workflow steps. This
specification does not authorize them.

## S. Independent Verification

This P4-G specification requires independent verification before freeze.

The completed future P4-G result requires independent verification before its PASS may be consumed by P4-F.

The same review pass must not serve as both primary P4-G action-resolution review and independent verification.

## T. Fail-Closed Conditions

The future P4-G review must return `BLOCKED` at minimum for:

- P4-F authority mismatch
- Gate-6 closure mismatch
- Gate-6 decision other than `P3W6F2P4E_LEVEL2_RESULT_REVIEW_PASS`
- Gate-6 pair-dispositions SHA256 mismatch
- `training_admission_released` not equal to `false`
- missing candidate search evidence
- incomplete candidate universe
- historical chronology used as authority
- current-lineage applicability unresolved
- zero eligible candidates
- multiple non-dominated candidates
- selected authority path, commit, or token ambiguity
- action identity ambiguity
- action-type ambiguity
- input or output ambiguity
- environment ambiguity
- parameter inheritance without authority
- Stage184/185 bridge ambiguity
- training or evaluation inference without explicit authority
- namespace conflict
- missing subsequent workflow authorization

Any attempt to release training admission, select an action in this specification, modify a second file, change scientific scope, or treat this specification as execution authority must fail closed.

Consumption of a P4-G result by P4-F or any later workflow must be blocked at
minimum for:

- `resolution_state_commit` missing, nonexistent, or mismatched;
- result path does not encode the exact `resolution_state_commit`;
- primary result hash mismatch;
- `result_freeze_commit` does not contain the exact primary result;
- verification attestation path does not encode the exact
  `result_freeze_commit`;
- verification attestation hash mismatch;
- `verification_attestation_commit` does not contain the exact attestation;
- independent verification status is not PASS;
- verification authority bindings mismatch;
- any forward-predicted or self-referential SHA is required;
- P4-F handoff provenance tuple is incomplete.

## U. Scientific Boundary

P4-G is authority resolution only.

It must not reopen or reinterpret:

- Gate-6 119-pair result
- F2 remediation
- P4-B regeneration
- P4-D/Gate-5
- P4-E result
- Stage185 compatibility result

It must not change:

- P4-F spec
- P4-E artifacts
- P4-D
- P4-B
- datasets
- Stage184/Stage185 files
- model/training/evaluation code
- root `*.patch` files
- `reports/stage180a_pass2_annotations_completed.csv`

Final provenance-repair readiness token:

`P3W6F2P4G_LEVEL3_ACTION_RESOLUTION_SPEC_PROVENANCE_REPAIR_READY_FOR_INDEPENDENT_REVERIFICATION`
