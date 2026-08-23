# P3-W6-F2-P4-O POSIX Atomic Publication Compatibility Implementation Authority Specification

Authority/version:

`P3W6F2P4O_POSIX_ATOMIC_PUBLICATION_COMPATIBILITY_IMPLEMENTATION_AUTHORITY_V1`

This document is a candidate authority specification only. It becomes canonical
only after independent static verification PASS and immutable freeze. Candidate
creation does not authorize builder implementation, validation execution,
artifact materialization, trainer rebind, A0, training, evaluation, Kaggle, GPU,
commit, or push.

## A. Candidate Creation State

Candidate creation requires:

- HEAD exactly
  `b65f2c97e17eac66f00d5e7ae34555af6437188a`.
- Tracked worktree and index clean.
- Pre-existing untracked files, if any, must remain untouched.
- Exactly one new untracked file may be created by this task:
  `reports/reason_router_p2_p3w6f2_p4o_posix_atomic_publication_compatibility_implementation_authority_spec.md`.

Candidate creation evidence:

- `git rev-parse HEAD` returned
  `b65f2c97e17eac66f00d5e7ae34555af6437188a`.
- `git status --porcelain=v1 --untracked-files=no` returned no tracked
  worktree modifications.
- `git diff --cached --name-only` returned no staged modifications.
- `git status --short` reported pre-existing untracked local files only. Those
  files are not authority inputs and must remain untouched.
- The candidate path was absent before creation.
- `git diff --quiet 25a85015fb344552b48e57b6dd92f3b0320d37d1 -- scripts/build_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar.py tests/test_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar_builder.py`
  returned success, establishing that the current builder and test paths match
  the frozen builder implementation commit named below.

Warnings from inaccessible local pytest/cache directories during `git status`
or broad file discovery are environmental read warnings only. They are not
validation PASSes and do not authorize ignoring tracked or staged dirt.

If a verifier observes a HEAD mismatch, tracked worktree dirt, staged dirt,
candidate path pre-existence before this task, builder/test identity mismatch,
or user-work overwrite, P4-O is BLOCKED.

## B. Namespace And History Search

Repository content and Git history were searched for:

- `P4-O`
- `P4O`
- `POSIX atomic publication compatibility`
- `atomic no-replace builder publication`
- `equivalent current-lineage publication compatibility authority`

Scoped searches over `reports/`, `scripts/`, `tests/`, `AGENTS.md`, and
`README.md` found no existing P4-O/P4O namespace and no applicable existing
authority that already authorizes this current-lineage POSIX publication
compatibility implementation.

Git history searches over `reports/`, `scripts/`, and `tests/` found no
applicable prior authority for the same namespace or publication compatibility
scope.

An initial broad repository search hit old notebook/binary material unrelated
to an authority namespace. Those hits are not applicable authority collisions.

If independent verification finds an existing applicable frozen authority,
equivalent current-lineage publication compatibility authority, or namespace
collision, P4-O is BLOCKED.

## C. Authority Chain Consumed

P4-O consumes these frozen authorities and evidence:

- P4-L artifact contract:
  `reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_artifact_contract_authority_spec.md`
- P4-L freeze commit:
  `80cb034792f03226cf6e22c196c1229ed4e6dd62`
- P4-M builder implementation authority:
  `reports/reason_router_p2_p3w6f2_p4m_current_lineage_integrity_builder_implementation_authority_spec.md`
- P4-M freeze commit:
  `ee88ca5df65ff865e68007671d75f84ed97aa326`
- Frozen current builder implementation commit:
  `25a85015fb344552b48e57b6dd92f3b0320d37d1`
- P4-N CPU validation authority freeze commit:
  `1a1d7191dd5d5223366960240d49ea21e12264bd`
- P4-N PASS evidence commit:
  `b65f2c97e17eac66f00d5e7ae34555af6437188a`
- P4-N validated implementation commit:
  `25a85015fb344552b48e57b6dd92f3b0320d37d1`

P4-O does not duplicate, reinterpret, weaken, or supersede P4-L, P4-M, or
P4-N. It specifies a later bounded compatibility repair for the already-frozen
builder implementation.

## D. Exact Current POSIX Blocker

Direct source inspection of
`scripts/build_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar.py`
found that `finalize_payloads_atomic()` currently performs:

- an advisory pre-publication existence check for the canonical output path;
- staged writes into an attempt-owned sibling staging directory;
- staged pair completeness verification;
- a second advisory pre-publication existence check;
- `require(os.name == "nt", "P4L_ATOMIC_DIRECTORY_NOREPLACE_UNSUPPORTED")`;
- `staging_dir.rename(output_dir)`.

Therefore successful canonical publication is deliberately restricted to
`os.name == "nt"`. POSIX/Linux reaches
`P4L_ATOMIC_DIRECTORY_NOREPLACE_UNSUPPORTED` before publication. Kaggle is
Linux, so a future Kaggle materialization using this frozen implementation
would fail closed.

This is not a P4-N failure. P4-N runtime PASS establishes the narrow tested
behavior of the frozen implementation at commit
`25a85015fb344552b48e57b6dd92f3b0320d37d1`; it does not establish successful
canonical publication on Kaggle or any other POSIX/Linux environment. P4-N
evidence must not be reused as validation of a future modified builder.

## E. Exact Future Implementation Scope

Before P4-O independent verification PASS and immutable freeze, this candidate
authorizes no implementation.

After P4-O independent verification PASS and immutable freeze, a later
explicitly scoped implementation task may modify only:

1. `scripts/build_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar.py`
2. `tests/test_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar_builder.py`

No other source, report, dataset, trainer, model, manifest, checkpoint,
configuration, historical artifact, or generated artifact may be modified.

Implementation must remain stdlib-only.

## F. Required Publication Semantics

The future implementation must preserve the existing publication model:

1. Fully construct sidecar and provenance bytes.
2. Write both into one fresh attempt-owned sibling staging directory.
3. Verify the staged pair is complete.
4. Publish the entire staging directory as one directory-level operation.
5. Require the canonical final path not to previously exist.
6. Never overwrite, replace, move aside, back up, mutate, or delete any
   pre-existing canonical final path.
7. Cleanup may touch only attempt-owned staging.
8. Collision or race must fail closed.

The final publication primitive must provide atomic NO-REPLACE semantics.

## G. Linux/POSIX No-Replace Primitive Contract

For Linux, P4-O authorizes a narrow stdlib implementation using Python
`ctypes`, libc `renameat2()`, and `RENAME_NOREPLACE`, with `AT_FDCWD`
directory descriptors if needed by the libc signature.

The preferred contract is:

```text
renameat2(
    AT_FDCWD,
    staging_path,
    AT_FDCWD,
    final_path,
    RENAME_NOREPLACE
)
```

The implementation must not:

- hardcode architecture-specific syscall numbers;
- call shell `mv` or `cp`;
- use `os.replace()`;
- use `Path.replace()`;
- use plain POSIX `os.rename()` or `Path.rename()` as a no-replace primitive;
- delete an existing destination;
- temporarily rename or back up an existing destination;
- infer ownership of the final path from contents;
- use network or third-party packages.

If libc `renameat2` or `RENAME_NOREPLACE` semantics are unavailable or
unsupported, the implementation must fail closed with
`BuildBlocked("P4L_ATOMIC_DIRECTORY_NOREPLACE_UNSUPPORTED")` or an exactly
frozen-equivalent unsupported code. It must not silently fall back to an
overwrite-capable primitive.

Windows behavior may retain the currently verified fail-if-target-exists
directory rename path, provided existing safety semantics remain unchanged.

Unsupported non-Windows/non-Linux platforms must fail closed unless an equally
strong atomic no-replace primitive is statically demonstrated.

## H. TOCTOU And Collision Semantics

Initial path-existence checks are advisory only. Correctness must rely on the
final atomic NO-REPLACE publication primitive.

If another process creates the canonical final path after the final precheck
but before publication:

- publication must fail;
- the foreign path must remain unchanged;
- attempt-owned staging may be cleaned only if still owned by this attempt;
- no foreign path deletion or replacement is permitted.

Destination already exists or race collision must raise
`BuildBlocked("P4L_OUTPUT_PATH_PREEXISTING")` or the frozen-equivalent code.

## I. Cleanup And Failure Semantics

The future implementation must distinguish at least:

- destination already exists or race collision:
  `BuildBlocked("P4L_OUTPUT_PATH_PREEXISTING")` or frozen-equivalent code;
- atomic no-replace primitive unavailable:
  `BuildBlocked("P4L_ATOMIC_DIRECTORY_NOREPLACE_UNSUPPORTED")`;
- other publication OS error:
  fail closed and never reinterpret as success.

No cleanup failure may turn a failed publication into PASS. Cleanup may touch
only attempt-owned staging. Cleanup must not delete, replace, move aside, back
up, or mutate a pre-existing canonical final path.

## J. Required Regression-Test Surface

After P4-O freeze, future tests may add only narrow regressions exercising the
actual publication helpers or implementation. Tests must not merely search
source strings.

The future test delta must meaningfully cover:

1. Successful Linux/POSIX publication into an absent final path when
   `renameat2(RENAME_NOREPLACE)` is available:
   complete directory published; both expected files present; staged directory
   no longer present; no extra final files.
2. Destination created or existing at publication syscall boundary:
   fail closed; destination unchanged; no overwrite or delete; attempt-owned
   staging cleaned appropriately.
3. Existing empty directory remains protected.
4. Existing non-empty or unrelated directory remains protected.
5. Existing regular file remains protected.
6. Existing or broken symlink remains protected where the test environment
   permits.
7. Unsupported atomic primitive produces explicit fail-closed behavior with no
   fallback overwrite.
8. Windows branch remains behaviorally unchanged or is statically isolated from
   the POSIX delta.

## K. Preserved Builder And P4-L Semantics

The future implementation must not change:

- P4-B source dataset identity, path, or hashes;
- exact 3600-row order;
- split replay;
- canonical mapping;
- Stage185 historical/current separation;
- P4-B 119-pair/357-member scope;
- integrity composition;
- positive-margin eligibility formula;
- `FRAME > PREDICATE > SUFFICIENCY > AUTHORIZED`;
- P2 field contracts;
- serialization;
- semantic-hash algorithm;
- provenance schema semantics;
- trainer exclusion codes;
- CLI materialization boundary except platform-safe publication support;
- scientific mechanism.

## L. Builder-Source Identity And Hash Consequence

The future P4-O implementation will change builder source bytes. Therefore:

- `integrity_builder_sha256` in future sidecar rows will change;
- `builder_source_sha256` in future provenance will change;
- because P4-L semantic hashing removes only `created_at`, the future
  materialized sidecar semantic SHA may also change solely because
  builder-source provenance changed.

This is expected provenance behavior, not a scientific-semantic change. P4-O
does not predict any future builder SHA, artifact physical SHA, or artifact
semantic SHA.

## M. Future Implementation Commit Identity Rule

The current builder implementation commit
`25a85015fb344552b48e57b6dd92f3b0320d37d1` is the BASE implementation being
repaired.

After a future authorized P4-O implementation commit:

- that new exact commit becomes the builder-source implementation identity for
  future validation and materialization;
- future canonical output path must bind to that new builder implementation
  commit, not to `25a85015fb344552b48e57b6dd92f3b0320d37d1`;
- P4-N PASS evidence for `25a85015fb344552b48e57b6dd92f3b0320d37d1` must not be
  reused as runtime validation of the modified builder.

A separate later validation authority and run are required for the new
implementation.

## N. Candidate-Time Flags

At candidate time, all flags remain false:

- `builder_implementation_authorized = false`
- `posix_publication_compatibility_implementation_authorized = false`
- `validation_execution_authorized = false`
- `artifact_materialization_authorized = false`
- `trainer_rebind_authorized = false`
- `a0_execution_authorized = false`
- `training_admission_released = false`
- `training_authorized = false`
- `evaluation_authorized = false`
- `kaggle_authorized = false`
- `gpu_authorized = false`

## O. Post-Freeze Implementation Flags

After P4-O independent verification PASS and immutable freeze, a later
explicitly scoped implementation task may have:

- `builder_implementation_authorized = true`
- `posix_publication_compatibility_implementation_authorized = true`

All execution and research flags remain false:

- `validation_execution_authorized = false`
- `artifact_materialization_authorized = false`
- `trainer_rebind_authorized = false`
- `a0_execution_authorized = false`
- `training_admission_released = false`
- `training_authorized = false`
- `evaluation_authorized = false`
- `kaggle_authorized = false`
- `gpu_authorized = false`

P4-O freeze alone does not authorize Python, pytest, builder execution,
materialization, Kaggle execution, training, or evaluation.

## P. Independent Verification Requirements

Because this delta changes filesystem publication and provenance safety, P4-O
requires:

1. Independent static verification of this authority candidate before freeze.
2. After implementation, independent static verification of the builder/test
   delta before implementation commit.
3. A separate later runtime validation authority for the new implementation.

No materialization may occur before all required later gates are explicitly
authorized.

## Q. Scientific Boundary

P4-O preserves unchanged:

- Conditional First-Blocker Reason Router;
- Reason-Specific Supervision;
- Explicit Gradient Ownership;
- `FRAME > PREDICATE > SUFFICIENCY > AUTHORIZED`;
- secondary reasons diagnostic-only;
- router-only final 3-way CE;
- detached F/P/S and polarity CE path;
- EMA observer/baseline-only;
- A0-A3;
- E0.

P4-O makes no scientific claim and authorizes no training or evaluation.

## R. Stop Conditions

P4-O is BLOCKED if any of the following holds:

- namespace collision;
- current HEAD mismatch;
- tracked worktree or index dirt;
- builder/test identity mismatch;
- safe Linux atomic no-replace publication requires modifications outside the
  two authorized files;
- implementation would require third-party dependencies or shell commands;
- no fail-closed Linux no-replace primitive can be specified;
- P4-L schema or semantic contract would need modification;
- trainer or model changes are required;
- materialization is needed merely to create or verify this authority.

## S. Explicit Non-Authorization

Training/evaluation allowed: NO.

Materialization allowed: NO.

Kaggle execution: NO.

GPU: NO.

Commit/push: NO.

No Python, pytest, py_compile, builder execution, materialization, Kaggle
execution, training, evaluation, commit, or push is authorized by candidate
creation.

## T. Candidate Path And Candidate SHA256

Candidate path:

`reports/reason_router_p2_p3w6f2_p4o_posix_atomic_publication_compatibility_implementation_authority_spec.md`

Candidate SHA256:

`TO_BE_RECORDED_AFTER_FILE_CREATION`

## U. Blockers Or Failure Reasons

At candidate creation time, no blocker was found:

- required HEAD matched;
- tracked worktree and index were clean;
- scoped repository and Git history searches found no applicable P4-O namespace
  collision;
- current builder/test paths matched frozen implementation commit
  `25a85015fb344552b48e57b6dd92f3b0320d37d1`;
- the POSIX blocker was directly identified without running Python, pytest,
  builder execution, materialization, training, evaluation, Kaggle, GPU, commit,
  or push.

If independent verification finds contrary evidence, P4-O is BLOCKED.

## V. Final Git Status

Final git status must show no tracked worktree or index modifications and
exactly one new untracked P4-O candidate file attributable to this task, with
pre-existing untracked local files left untouched.

P3W6F2P4O_POSIX_ATOMIC_PUBLICATION_COMPATIBILITY_IMPLEMENTATION_AUTHORITY_CANDIDATE_READY_FOR_INDEPENDENT_VERIFICATION
