# P3-W6-F2-P4-H A0 Current-Lineage Rebind Authority Specification

Decision scope: exact earliest Level-3 action authority after independently
verified P4-G action-resolution BLOCKED.

This document is specification-only. It does not perform the rebind audit,
modify implementation, release training admission, execute A0, load a model,
create a checkpoint, run evaluation, authorize Kaggle, or authorize GPU use.

## A. Authority Freeze

Current P4-G authority:

- P4-G specification:
  `reports/reason_router_p2_p3w6f2_p4g_level3_action_resolution_spec.md`
- P4-G specification freeze commit:
  `dc62e63da045435429dc3927f8dc5b2c0d4de59c`
- primary P4-G result:
  `reports/reason_router_p2_p3w6f2_p4g_level3_action_resolution_result_dc62e63da045435429dc3927f8dc5b2c0d4de59c.json`
- primary result freeze commit:
  `ebeb6f77c46a23c0b6bc29aeddb5fca4cc69aabf`
- primary result SHA256:
  `e841ea6fb76e54dfdd62e1a7412650164f342a26abe02d514ea8fbe6ee4ff57e`
- primary decision:
  `P3W6F2P4G_LEVEL3_ACTION_RESOLUTION_BLOCKED`
- verification attestation:
  `reports/reason_router_p2_p3w6f2_p4g_level3_action_resolution_verification_attestation_ebeb6f77c46a23c0b6bc29aeddb5fca4cc69aabf.json`
- verification attestation freeze commit:
  `9bec86eea8082d8d0dd68419542dc8565374c5e9`
- verification attestation SHA256:
  `3b447e7ff19d8fb9d9e929efa14a669015340384deea3b580cd5e80c512baf3b`
- independent verification:
  `PASS`

Frozen P4-F authority:

- specification:
  `reports/reason_router_p2_p3w6f2_p4f_level3_admission_authority_spec.md`
- freeze commit:
  `8fd76261fbad2ef78e915e3442147c971e95aa33`
- current earliest-action state:
  `UNRESOLVED_BY_FROZEN_AUTHORITY`

Gate-6 authority:

- final closure commit:
  `08d233d53b9d33e055f04c833613edb313bd0cf5`
- decision:
  `P3W6F2P4E_LEVEL2_RESULT_REVIEW_PASS`
- pair-dispositions SHA256:
  `cb8ca8b46e5440867cb1d22132ed5738e030bdc818dd685ffc8a0e6cc732cdce`

P4-B R1 authority:

- specification commit:
  `fcc3b9ccaf2bbee33ac18dcef10d50acff54aab4`
- artifact directory:
  `reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/`
- regenerated full-dataset SHA256:
  `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`
- regenerated semantic SHA256:
  `3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b`
- Stage185 compatibility artifacts 8-10 passed without mutating Stage185-v1.

P4-D authority:

- specification commit:
  `1be4050dbadb0cb5ed2f6b55a2391051f5f6c07e`
- Gate-5 decision:
  `P3W6F2P4D_CONTROLLED_DATA_INTEGRITY_GATE_PASS`
- Gate-5 execution head:
  `eced1d46e8788e4372eca14dcf090c2840649399`

Historical P3 execution intent:

- specification:
  `reports/reason_router_p2_p3_execution_spec.md`
- specification commit:
  `72cfdd3d551832e33799ca0399a6d6bf0c431901`
- historical A0 status:
  `P3_A0_PHASE_READY_FOR_EXECUTION`
- historical execution-order characterization:
  A0 is the first runnable Reason Router phase; A1-A3 remain blocked.
  This is descriptive authority evidence, not a separate canonical token.
- historical main dataset:
  `data/controlled_v5_v3_without_time_swap.jsonl`
- historical dataset SHA256:
  `f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640`
- historical Stage185 sidecar semantic SHA256:
  `5bc03caa2a29f9b9176ab4eb0201db57ebad516352797546db1a18e6ec3373fc`

The historical P3 specification establishes the downstream scientific intent
for A0, but its historical dataset/sidecar identities are not current P4-B R1
execution authority.

Current trainer binding evidence at the P4-H drafting state:

- `scripts/train_controlled_v6b_minimal.py` defines
  `_STAGE187_AUTHORITATIVE_DATA` as
  `data/controlled_v5_v3_without_time_swap.jsonl`;
- `_STAGE187_DATASET_SHA256` is fixed to
  `f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640`;
- `_STAGE187_SIDECAR_SEMANTIC_SHA256` is fixed to
  `5bc03caa2a29f9b9176ab4eb0201db57ebad516352797546db1a18e6ec3373fc`;
- static inspection confirms repeated path, dataset-SHA, source-dataset-SHA,
  authoritative-sidecar, and controlled-integrity-sidecar checks in current
  trainer/preflight paths.

Therefore the current implementation cannot be presumed to consume the P4-B
R1 regenerated dataset under the historical A0 execution contract without an
explicit authority/provenance rebind determination.

This static evidence supports the need for the rebind audit only. It does not
authorize trainer modification, sidecar generation, training, evaluation,
model loading, Kaggle, or GPU execution.

Mandatory invariant:

`training_admission_released = false`

## B. Exact Earliest Level-3 Action

Canonical P4-H authority/version:

`P3W6F2P4H_A0_CURRENT_LINEAGE_REBIND_AUTHORITY_V1`

The exact earliest Level-3 action is:

`P3W6F2P4H_A0_CURRENT_LINEAGE_EXECUTION_REBIND_AUDIT`

Action type:

`STATIC_READ_ONLY`

Purpose:

Determine the exact minimal authority/provenance/loader/manifest contract
required to rebind the historical P3 A0 Reason Router execution specification
from its historical controlled-dataset and Stage185-sidecar identities to the
current P3-W6-F2 P4-B R1 regenerated dataset and successfully closed Gate-6
lineage.

This action is earlier than A0 implementation or execution because current
authority proves the historical A0 scientific intent but does not prove that
its historical data/sidecar/runtime identity contract remains valid after
P4-B R1 remediation.

This action must occur before any current-lineage A0 training or evaluation.

## C. Why This Action Is Authority-Supported

The historical P3 execution specification explicitly resolved A0 as the
runnable first reason-router phase.

P4-B R1 subsequently created a new full controlled-dataset identity after F2
remediation.

P4-B, P4-D, P4-E, P4-F, and P4-G do not authorize silently substituting that
new dataset into the historical A0 command.

Historical Stage184/185/186/187/188/189 execution lineage is bound to the
historical dataset and/or historical Stage185 sidecar and therefore is not a
current-lineage substitute for this audit.

The narrow missing prerequisite is not a new training experiment. It is an
exact static determination of how the historical A0 execution contract must be
rebound, if at all, to current P4-B R1 authority.

## D. Exact Audit Questions

The future audit must determine all of the following:

1. Whether the current P2-capable trainer can consume the P4-B R1 regenerated
   full dataset without changing reason-router scientific semantics.

2. Which current trainer fail-fast constants, dataset identity checks, sidecar
   checks, manifest fields, preflight checks, and A0 reference-audit fields
   still encode historical `f552...` / `5bc...` authority.

3. Whether P4-B Stage185 predicate-realization compatibility artifacts 8-10
   are sufficient as an authority bridge for A0 metadata/supervision.

4. Whether a new current-lineage effective integrity sidecar or another
   deterministic compatibility artifact is required before A0 can run.

5. If a new derived artifact is required, its exact semantic purpose and
   minimum required input identity. The audit must not implement it.

6. The exact minimal future implementation/manifest delta, if any, required
   before current-lineage A0 execution can be authorized.

7. Whether the historical P3 A0 execution parameters remain valid under
   current authority or require a separate parameter authority.

8. Whether any current code path would consume historical malformed F2
   semantics despite use of the regenerated dataset.

9. Whether A0 can remain scientifically identical to the historical P3 A0
   definition after provenance rebinding.

The audit must not choose among multiple technically possible architectures
by convenience. If more than one non-dominated valid rebind architecture
remains and authority cannot order them, the audit must return BLOCKED.

## E. Exact Inputs

The audit authority inputs are:

- frozen P4-H specification itself, identified by its later immutable freeze
  commit;
- frozen P4-G specification, primary BLOCKED result, and verification
  attestation listed in Section A;
- frozen P4-F specification;
- P4-B R1 specification and artifacts 1-10;
- P4-D Gate-5 specification and validated authority;
- P4-E Gate-6 final closure;
- historical P3 execution specification at
  `72cfdd3d551832e33799ca0399a6d6bf0c431901`;
- current repository trainer and manifest/preflight code as present at the
  frozen P4-H specification commit;
- repository `AGENTS.md`.

The future `audit_state_commit` must equal the immutable P4-H specification
freeze commit exactly.

No moving branch name may substitute for `audit_state_commit`.

## F. Environment

Exact action environment:

`LOCAL_STATIC_READ_ONLY`

Allowed operations:

- repository file inspection;
- Git history inspection;
- static source inspection;
- hash/provenance comparison;
- `git status`;
- `git diff`;
- `git show`;
- `git log`;
- `git grep`;
- `git diff --check`;
- creation of the single authorized audit result artifact.

Not allowed:

- Python execution;
- pytest;
- validators;
- Stage185 execution;
- dataset regeneration;
- model import or model loading;
- Torch forward pass;
- checkpoint loading or mutation;
- training;
- evaluation;
- Kaggle;
- GPU use.

## G. Parameters

The rebind audit itself has no training or split parameters.

Historical values including:

- split seed `174`;
- dev ratio `0.2`;
- A0 seeds `180`, `181`, `182`;
- epochs `20`;
- learning rate `0.001`;

are historical P3 execution-contract evidence only during this audit.

They must not be silently promoted into current-lineage execution authority.

The audit may determine that an existing higher authority preserves a
parameter, but must cite the exact authority that does so.

## H. Permitted Output

The audit may create exactly one result artifact:

`reports/reason_router_p2_p3w6f2_p4h_a0_current_lineage_rebind_audit_<audit_state_commit>.json`

where `<audit_state_commit>` is the already-existing full 40-character P4-H
specification freeze commit.

Schema/version:

`P3W6F2P4H_A0_CURRENT_LINEAGE_REBIND_AUDIT_V1`

Required fields:

- `schema_version`
- `decision_token`
- `verification_status`
- `audit_state_commit`
- `p4h_authority_commit`
- `p4g_verification_attestation_commit`
- `p4f_spec_commit`
- `gate6_closure_commit`
- `historical_p3_spec_commit`
- `historical_p3_a0_status`
- `historical_dataset_sha256`
- `current_regenerated_dataset_sha256`
- `historical_sidecar_semantic_sha256`
- `trainer_identity_findings`
- `dataset_binding_findings`
- `sidecar_binding_findings`
- `compatibility_artifact_findings`
- `manifest_preflight_findings`
- `parameter_authority_findings`
- `required_future_delta`
- `current_lineage_a0_rebind_status`
- `training_admission_released`
- `blockers`
- `failure_reasons`

Primary audit result must use:

`verification_status = PENDING`

No future Git SHA may be predicted inside the primary result.

## I. Audit Decision Tokens

The audit decision tokens are:

- `P3W6F2P4H_A0_CURRENT_LINEAGE_REBIND_AUDIT_PASS`
- `P3W6F2P4H_A0_CURRENT_LINEAGE_REBIND_AUDIT_FAIL`
- `P3W6F2P4H_A0_CURRENT_LINEAGE_REBIND_AUDIT_BLOCKED`

PASS means:

- exactly one current-lineage A0 rebind contract is authority-supported;
- exact current dataset/provenance bindings are resolved;
- exact historical-sidecar treatment is resolved;
- exact future implementation/manifest delta is resolved, including a
  determination of no implementation delta if that is what authority proves;
- no scientific mechanism change is required.

PASS does not authorize implementation or A0 execution.

BLOCKED means required authority is missing, multiple non-dominated valid
rebind contracts remain, provenance is ambiguous, or a required identity
cannot be resolved.

Missing authority is BLOCKED, not FAIL.

FAIL means valid evidence shows the historical P3 A0 scientific contract
cannot be rebound to the current P4-B R1/Gate-6 lineage without changing the
approved scientific mechanism or violating a frozen invariant.

## J. Exact Prohibited Effects

Neither P4-H specification freeze nor future audit PASS may:

- set `training_admission_released=true`;
- execute A0;
- release A1/A2/A3;
- choose a reason-loss weight;
- change reason order;
- change gradient ownership semantics;
- change router semantics;
- change final 3-way CE ownership;
- change EMA semantics;
- change A0-A3 definitions;
- change E0;
- modify P4-B regenerated data;
- mutate historical Stage185-v1;
- reuse historical Stage185/186/187/188/189 execution authority as current
  execution authority;
- modify model/trainer code;
- create a checkpoint;
- authorize Kaggle or GPU.

## K. P4-G And P4-F Consumption Boundary

Freezing this P4-H specification does not retroactively change the frozen
P4-G BLOCKED result.

After this authority is independently verified and frozen, P4-G must be
separately re-evaluated against a new `resolution_state_commit` containing
this P4-H authority.

P4-G may select this action only if it independently verifies:

- exact P4-H authority path;
- exact P4-H freeze commit;
- canonical token/version;
- exact action name;
- exact action type `STATIC_READ_ONLY`;
- exact purpose;
- exact authority inputs;
- exact permitted output;
- exact prohibited actions;
- exact environment;
- no inherited execution parameters;
- `training_admission_released=false`;
- separate execution authorization required.

A future P4-G PASS still does not perform this audit.

After a verified P4-G PASS, frozen P4-F must receive a separate explicitly
authorized admission evaluation.

Only a later valid P4-F PASS may make this exact static audit eligible for
another explicit workflow authorization.

## L. Training Admission Invariant

Mandatory before, during, and after this authority:

`training_admission_released = false`

There is no `false -> true` transition in P4-H.

A future A0 training authority, if scientifically and procedurally justified,
must be separate and must follow completion of all required admission,
rebind, implementation, and validation authorities.

## M. Independent Verification

This P4-H specification requires independent static verification before
freeze because it resolves an authority/provenance boundary.

The verifier must confirm at minimum:

- original P3 A0 intent is represented faithfully;
- historical P3 execution identities are not treated as current execution
  authority;
- current R1/Gate-6 lineage is exact;
- this audit is narrower and earlier than current-lineage A0 execution;
- the action is uniquely and exactly specified;
- no hidden implementation/training authority exists;
- no historical seed/split/sidecar parameter is silently inherited;
- no namespace collision exists;
- `training_admission_released=false`.

Specification verification plus freeze does not execute the audit.

## N. Scientific Boundary

P4-H preserves the approved Reason Router mechanism:

- Conditional First-Blocker Reason Router;
- Reason-Specific Supervision;
- Explicit Gradient Ownership;
- FRAME > PREDICATE > SUFFICIENCY > AUTHORIZED;
- secondary reasons diagnostic-only;
- router-only final 3-way CE ownership;
- EMA observer/baseline-only;
- A0-A3 factorial definitions;
- E0 algebraic-equivalence check.

P4-H does not reinterpret the Gate-6 119-pair scientific result and does not
reopen F2 remediation.

Final readiness token:

`P3W6F2P4H_A0_CURRENT_LINEAGE_REBIND_AUTHORITY_SPEC_READY_FOR_INDEPENDENT_VERIFICATION`
