# P3-W6-F2-P4-M Current-Lineage Integrity Builder Implementation Authority Specification

Authority/version:

`P3W6F2P4M_CURRENT_LINEAGE_INTEGRITY_BUILDER_IMPLEMENTATION_AUTHORITY_V1`

This document is a frozen-authority candidate specification only. It becomes
canonical only after independent static verification PASS and immutable freeze.
Candidate creation does not authorize implementation, builder execution,
sidecar materialization, artifact/provenance production, trainer rebind,
parameter adoption, A0, A1, A2, A3, training, evaluation, Kaggle, GPU, or
promotion.

## A. Candidate Creation State

Candidate creation requires:

- HEAD exactly
  `80cb034792f03226cf6e22c196c1229ed4e6dd62`.
- Tracked worktree and index clean.
- Pre-existing untracked files, if any, must remain untouched.
- Exactly one new untracked file may be created by this task:
  `reports/reason_router_p2_p3w6f2_p4m_current_lineage_integrity_builder_implementation_authority_spec.md`.

Current-authority creation evidence:

- `git rev-parse HEAD` returned
  `80cb034792f03226cf6e22c196c1229ed4e6dd62`.
- `git diff --name-only` returned no tracked worktree modifications.
- `git diff --cached --name-only` returned no staged modifications.
- `git status --short` reported pre-existing untracked local files. Those files
  are not authority inputs and must remain untouched.

Warnings from inaccessible local pytest/cache directories during `git status`
or broad file discovery are environmental read warnings only. They are not
validation PASSes and do not authorize ignoring tracked or staged dirt.

If a verifier observes tracked worktree dirt, staged dirt, candidate path
pre-existence before this task, or any user-work overwrite, P4-M is BLOCKED.

## B. Namespace And History Search

Repository content and Git history were independently searched for:

- `P4-M`
- `P4M`
- `current-lineage integrity builder implementation authority`
- `equivalent builder implementation authority`

Current tracked content found no existing P4-M/P4M canonical namespace and no
existing authority that already authorizes this current-lineage integrity
builder implementation scope.

Git history search for the candidate path found no prior tracked version of
this file. Git content search found no applicable P4-M authority.

If independent verification finds an existing applicable implementation
authority, equivalent builder implementation authority, or namespace collision,
P4-M is BLOCKED.

## C. Authority Chain Consumed

P4-M consumes the frozen P4-L artifact contract:

- P4-L path:
  `reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_artifact_contract_authority_spec.md`
- P4-L freeze commit:
  `80cb034792f03226cf6e22c196c1229ed4e6dd62`
- P4-L authority/version:
  `P3W6F2P4L_CURRENT_LINEAGE_INTEGRITY_ARTIFACT_CONTRACT_AUTHORITY_V1`

P4-M also consumes the final verified P4-H chain:

- P4-H authority commit:
  `368d3b6991389aa6b6fd80f421c73565b562e290`
- P4-K freeze:
  `13e7b0d7e229aa678e791e06b2e1d7de26474414`
- P4-H result freeze:
  `b3626ae80ecf0664433821a772be28a56c6409da`
- P4-H verification attestation freeze:
  `703b861ab738b1cfdf73121de23ca07b6bbb9e48`

P4-L is the direct artifact contract authority. P4-M must not duplicate,
reinterpret, weaken, or supersede P4-L.

## D. Exact Authority Released By P4-M

Before independent verification and freeze, this candidate releases no
implementation authority.

After independent static verification PASS and immutable freeze, P4-M
authorizes only a later, explicitly scoped source-code implementation task to
add:

1. One deterministic builder for the frozen P4-L
   `CURRENT_LINEAGE_EFFECTIVE_INTEGRITY_SIDECAR_JSONL` contract.
2. Narrowly scoped helper/static validation code inside that builder if needed
   to prove that a future builder output would satisfy P4-L.
3. Unit/static test code at the exact path frozen in Section F, if consistent
   with repository conventions. P4-M itself does not authorize running those
   tests.

P4-M never authorizes:

- builder execution
- sidecar materialization
- artifact/provenance production
- trainer constant/path/SHA rebind
- checkpoint, manifest, or trainer changes
- parameter adoption
- A0, A1, A2, or A3
- training or evaluation
- Kaggle or GPU
- promotion

## E. Canonical Implementation Target

Repository inspection found the existing deterministic builder/materializer
naming pattern under `scripts/`, including:

- `scripts/build_stage185a_controlled_train_integrity_sidecar.py`
- `scripts/materialize_reason_router_p3w6f2_p4b_r1_stage185_compatibility.py`
- `scripts/validate_reason_router_p3w6f2_p4d_controlled_data_integrity_gate.py`

P4-M freezes exactly one canonical future builder source path:

`scripts/build_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar.py`

This builder must consume frozen P4-L directly and must not duplicate or weaken
the P4-L contract. It may define local deterministic helper functions inside
the same file. No separate validation module/script is authorized by P4-M
because the current repository conventions can support the needed static and
unit validation through the builder module plus tests.

The future builder may import existing deterministic helper functions from
historical modules read-only when semantically identical and authority-valid.
It must prefer a new P4-L/P4-M namespace over mutating historical Stage185
builder semantics.

## F. Exact Allowed Future Implementation Delta

After P4-M independent verification PASS and freeze, the smallest permissible
future implementation delta is:

- add new source file:
  `scripts/build_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar.py`
- add new test file, only if needed for static/unit contract proof:
  `tests/test_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar_builder.py`

No other source, trainer, model, manifest, checkpoint, dataset, artifact,
report, or configuration file is authorized by P4-M.

If future implementation requires trainer changes merely to build the artifact,
the implementation is BLOCKED. Trainer rebind belongs to a later separate
stage.

## G. Builder Input Contract

The future builder must require and validate these exact frozen inputs:

- P4-L authority path:
  `reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_artifact_contract_authority_spec.md`
- P4-L freeze commit:
  `80cb034792f03226cf6e22c196c1229ed4e6dd62`
- P4-L authority/version:
  `P3W6F2P4L_CURRENT_LINEAGE_INTEGRITY_ARTIFACT_CONTRACT_AUTHORITY_V1`
- P4-H authority commit:
  `368d3b6991389aa6b6fd80f421c73565b562e290`
- P4-K freeze:
  `13e7b0d7e229aa678e791e06b2e1d7de26474414`
- P4-H result freeze:
  `b3626ae80ecf0664433821a772be28a56c6409da`
- P4-H verification attestation freeze:
  `703b861ab738b1cfdf73121de23ca07b6bbb9e48`
- P4-B regenerated dataset path:
  `reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl`
- P4-B regenerated dataset physical SHA256:
  `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`
- P4-B regenerated dataset semantic SHA256:
  `3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b`
- P4-B compatibility rows artifact:
  `reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/p3w6f2_p4b_r1_stage185_predicate_realization_compatibility_rows.jsonl`
- P4-B compatibility rows SHA256:
  `59e4367d29e3c49152049e4a1b46e8783d5b81d1ebaa931ca9fd4ae0ac967b9f`
- P4-B compatibility summary artifact:
  `reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/p3w6f2_p4b_r1_stage185_predicate_realization_compatibility_summary.json`
- P4-B compatibility summary SHA256:
  `ce618d214dc4d660706d927a0d91ec5945d3a0edbffbf615eab8b9c9ff585aa8`
- P4-B compatibility provenance artifact:
  `reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/p3w6f2_p4b_r1_stage185_predicate_realization_compatibility_provenance_manifest.json`
- P4-B compatibility provenance SHA256:
  `09a56f1dca325e0749a0e4b0f822d68dad5d85fd90fa427e0efd6d617d41b2d6`
- historical Stage185-v1 sidecar path:
  `reports/stage185a_controlled_train_integrity_sidecar_20260715_141914/stage185a_controlled_train_integrity_sidecar.jsonl`
- historical Stage185-v1 sidecar semantic SHA256:
  `5bc03caa2a29f9b9176ab4eb0201db57ebad516352797546db1a18e6ec3373fc`
- historical Stage185 source script path:
  `scripts/build_stage185a_controlled_train_integrity_sidecar.py`
- historical Stage185 source script SHA256:
  `11e6ba89b8131c76eac4504b4273867eaa99a131abe23d3238eb65ecda207bbc`
- deterministic split rule from P4-L: sorted pair IDs, shuffle seed `174`,
  dev ratio `0.2`, pair-level split.
- deterministic canonical-row rule from P4-L: for controlled-v5 rows,
  canonical row is the same-pair row with `intervention_type == "none"`.

The builder must have no network access requirement and no model, torch,
checkpoint, GPU, Kaggle, trainer, or learned dependency.

## H. Builder Output Contract

The future implementation must be capable of producing exactly the P4-L
outputs, but P4-M does not authorize producing them.

Canonical sidecar namespace:

`P3W6F2P4L_CURRENT_LINEAGE_INTEGRITY_SIDECAR`

Canonical sidecar schema version:

`P3W6F2P4L_CURRENT_LINEAGE_EFFECTIVE_INTEGRITY_SIDECAR_V1`

Canonical sidecar path pattern:

`reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_<builder_commit>/p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl`

Canonical provenance manifest schema version:

`P3W6F2P4L_CURRENT_LINEAGE_INTEGRITY_SIDECAR_PROVENANCE_V1`

Canonical provenance manifest path pattern:

`reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_<builder_commit>/p3w6f2_p4l_current_lineage_effective_integrity_sidecar_provenance.json`

`<builder_commit>` must be the full 40-character source commit of the future
builder execution state. Builder source must never hardcode a future output SHA
or future execution commit.

Serialization and hash rules:

- UTF-8
- LF line endings
- no BOM
- no NaN, Infinity, or non-JSON values
- compact JSON for JSONL with exactly one JSON object per line
- final newline required
- physical SHA256 computed over exact artifact bytes
- sidecar semantic SHA256 exactly follows P4-L: parse ordered JSONL rows,
  preserve source row order, remove `created_at`, remove no identity/status/
  eligibility/reason/provenance hash field, serialize the ordered list with
  `sort_keys = true`, `separators = (",", ":")`, `ensure_ascii = false`, and
  hash the UTF-8 bytes.

The sidecar must contain exactly `3600` rows in exact P4-B regenerated source
order, one object per source row.

## I. Exact Semantic Preservation Contract

The future implementation must preserve P4-L exactly, including:

- `P2_SIDE_CAR_REQUIRED_FIELDS`
- `P2_SOURCE_REQUIRED_FIELDS`
- per-row `source_dataset_sha256` exactly
  `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`
- exact source row order and complete `row_id` coverage
- split seed `174` and dev ratio `0.2` solely as artifact-integrity
  compatibility semantics, not A0 parameter authority
- `canonical_row_id` semantics
- eligibility, status, and reason-code semantics
- P4-B 119-pair/357-member scoped compatibility semantics
- Stage185 historical/current bridge semantics
- fail-closed behavior
- semantic hashing

No reinterpretation is permitted. Historical dataset SHA
`f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640`
must never appear as current source identity. It may appear only in explicitly
named historical input or comparison fields.

P4-B artifacts 8-10 are scoped evidence only. They must not be treated as proof
of full 3600-row sidecar completeness, current row-order identity, current
source SHA binding, complete split identity, every P2 sidecar field, or A0
manifest/reference-audit identity.

## J. Implementation Style And Reuse

The future implementer must:

- inspect existing Stage185/integrity builders before editing.
- reuse deterministic helpers where semantically identical and authority-valid.
- avoid copy-pasting large divergent implementations.
- keep historical Stage185 artifacts/scripts immutable.
- import or reuse historical Stage185 code read-only only when this does not
  change historical behavior.
- prefer the new P4-L/P4-M namespace rather than mutating historical builder
  semantics.
- keep all implementation code deterministic and side-effect-minimal.

The historical Stage185-v1 sidecar and builder are evidence inputs, not current
lineage outputs.

## K. Fail-Closed Implementation Behavior

The future builder must fail before writing successful output when any global
contract fails, including:

- source path mismatch
- source physical SHA mismatch
- source semantic SHA mismatch
- source row count not exactly `3600`
- `row_id`, row order, or coverage mismatch
- historical/current join ambiguity
- P4-B compatibility artifact path/hash/schema/count mismatch
- split replay or canonical-row inconsistency
- required source fields unresolved
- required sidecar fields unresolved
- non-deterministic, missing, unsupported, or non-JSON values
- provenance identity mismatch
- historical Stage185 mutation requirement
- use of historical source identity as current source identity

Partial successful artifacts must not be treated as valid outputs.

If future execution/materialization is separately authorized, temporary-file or
staging-directory plus atomic-finalize behavior is required for successful
artifact publication, following existing repository practice in P4-B
compatibility materialization. Existing complete and byte-identical outputs may
be treated as idempotent only under a later authority that authorizes execution
or materialization.

## L. Future Validation Requirements

Before any execution/materialization authority, a later independent verifier
must statically inspect the implementation for:

- exact P4-L contract coverage
- no scientific semantic drift
- no trainer rebind
- no execution authorization
- deterministic serialization and hash implementation
- fail-closed behavior
- artifact paths and schema versions
- exact `3600`-row source-order contract
- P4-B 119-pair/357-member compatibility scoping
- no historical Stage185 mutation
- no hardcoded future output SHA
- no hardcoded future execution commit
- no network/model/Kaggle/GPU dependency

P4-M may require later Kaggle CPU/non-training tests as a prerequisite for a
future materialization authority, but P4-M does not authorize running them.
Validation PASS may not be reported unless the future command actually ran
successfully under a separate authority that permits that execution.

## M. Authority Flags

P4-M candidate itself:

`training_admission_released = false`

`builder_implementation_authorized = false`

`artifact_materialization_authorized = false`

`trainer_rebind_authorized = false`

`a0_execution_authorized = false`

`training_authorized = false`

`evaluation_authorized = false`

`kaggle_authorized = false`

`gpu_authorized = false`

Only after P4-M independent static verification PASS and immutable freeze may a
subsequent explicitly scoped implementation task have:

`builder_implementation_authorized = true`

All of the following remain false after P4-M freeze:

`artifact_materialization_authorized`

`trainer_rebind_authorized`

`a0_execution_authorized`

`training_authorized`

`evaluation_authorized`

`kaggle_authorized`

`gpu_authorized`

`training_admission_released`

## N. Scientific Boundary

P4-M preserves without modification:

- Conditional First-Blocker Reason Router
- Reason-Specific Supervision
- Explicit Gradient Ownership
- FRAME > PREDICATE > SUFFICIENCY > AUTHORIZED
- secondary reasons diagnostic-only
- router-only final 3-way CE
- detached F/P/S and polarity CE path
- EMA observer/baseline-only
- A0-A3
- E0

P4-M does not change dataset identity, split semantics, label semantics,
reason-router semantics, loss semantics, gradient ownership, promotion
criteria, or clean-vs-external evaluation separation.

## O. Stop Conditions

P4-M is BLOCKED if any of the following holds:

- namespace collision
- applicable implementation authority already exists
- P4-L contract is ambiguous
- more than one non-dominated builder architecture remains
- builder requires trainer modifications
- implementation cannot preserve historical Stage185 immutability
- deterministic byte/canonical output cannot be guaranteed
- scientific semantics would change
- materialization or execution must occur to define the implementation contract
- current source identity would need to use historical `f552...`
- P4-B artifacts 8-10 would need to be treated as proof of full 3600-row
  current-lineage sidecar completeness

Missing authority is BLOCKED, not scientific FAIL.

## P. Candidate Readiness

Candidate path:

`reports/reason_router_p2_p3w6f2_p4m_current_lineage_integrity_builder_implementation_authority_spec.md`

Candidate SHA256 is computed after file creation and is not predicted inside
this specification body.

Final candidate readiness token:

`P3W6F2P4M_CURRENT_LINEAGE_INTEGRITY_BUILDER_IMPLEMENTATION_AUTHORITY_CANDIDATE_READY_FOR_INDEPENDENT_VERIFICATION`
