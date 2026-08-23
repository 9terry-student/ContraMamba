# P3-W6-F2-P4-N CPU Non-Training Builder Validation Execution Authority Specification

Authority/version:

`P3W6F2P4N_CPU_NON_TRAINING_BUILDER_VALIDATION_EXECUTION_AUTHORITY_V1`

This document is a frozen-authority candidate specification only. It becomes
canonical only after independent static verification PASS and immutable freeze.
Candidate creation does not authorize validation execution, Kaggle execution,
CPU execution, GPU use, builder materialization, sidecar or provenance artifact
production, trainer rebind, A0, training, evaluation, model execution,
checkpoint mutation, network use, or promotion.

## A. Candidate Creation State

Candidate creation requires:

- HEAD exactly
  `25a85015fb344552b48e57b6dd92f3b0320d37d1`.
- Tracked worktree and index clean.
- Pre-existing untracked files, if any, must remain untouched.
- Exactly one new untracked file may be created by this task:
  `reports/reason_router_p2_p3w6f2_p4n_cpu_non_training_builder_validation_execution_authority_spec.md`.

Current candidate creation evidence:

- `git rev-parse HEAD` returned
  `25a85015fb344552b48e57b6dd92f3b0320d37d1`.
- `git diff --name-only` returned no tracked worktree modifications.
- `git diff --cached --name-only` returned no staged modifications.
- `git status --short --branch` reported branch `main...origin/main` and
  pre-existing untracked local files. Those files are not authority inputs and
  must remain untouched.

Warnings from inaccessible local pytest/cache directories during `git status`
or broad repository search are environmental read warnings only. They are not
validation PASSes and do not authorize ignoring tracked or staged dirt.

If independent verification observes HEAD mismatch, tracked worktree dirt,
staged dirt, candidate path pre-existence before this task, or user-work
overwrite, P4-N is BLOCKED.

## B. Namespace And History Search

Repository content and Git history were searched for:

- `P4-N`
- `P4N`
- `CPU non-training builder validation execution authority`
- `builder runtime-validation authority`
- `runtime-validation authority`

The candidate path had no prior tracked history. Relevant tracked report,
script, test, and Markdown surfaces showed no existing P4-N/P4N authority and
no applicable equivalent builder runtime-validation authority.

The broad repository search encountered inaccessible local pytest/cache
directories and a very large notebook-text hit unrelated to this authority.
The authoritative namespace verdict is therefore based on the tracked relevant
surfaces and Git history searches above.

If independent verification finds an existing applicable authority, equivalent
builder runtime-validation authority, or namespace collision, P4-N is BLOCKED.

## C. Authority Chain Consumed

P4-N consumes the frozen P4-M implementation authority and the frozen P4-M
implementation:

- P4-M authority path:
  `reports/reason_router_p2_p3w6f2_p4m_current_lineage_integrity_builder_implementation_authority_spec.md`
- P4-M authority freeze commit:
  `ee88ca5df65ff865e68007671d75f84ed97aa326`
- P4-M implementation commit:
  `25a85015fb344552b48e57b6dd92f3b0320d37d1`
- builder source:
  `scripts/build_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar.py`
- builder test:
  `tests/test_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar_builder.py`

P4-N also preserves the frozen P4-L artifact contract:

- P4-L path:
  `reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_artifact_contract_authority_spec.md`
- P4-L freeze commit:
  `80cb034792f03226cf6e22c196c1229ed4e6dd62`
- P4-L authority/version:
  `P3W6F2P4L_CURRENT_LINEAGE_INTEGRITY_ARTIFACT_CONTRACT_AUTHORITY_V1`

P4-N does not duplicate, reinterpret, weaken, or supersede P4-L or P4-M.

## D. Exact Authority Purpose

P4-N distinguishes four separate questions:

1. code correctness
2. execution success
3. artifact/provenance validity
4. scientific conclusion

P4-N addresses only item 2: one bounded CPU/non-training validation execution
of the frozen P4-M builder implementation test surface.

A P4-N validation PASS does not establish:

- sidecar artifact validity
- provenance validity
- trainer compatibility under a materialized current-lineage artifact
- A0 correctness
- scientific mechanism evidence
- training or evaluation success

Any test failure is an implementation-validation FAIL, not automatically a
scientific FAIL.

## E. Frozen Implementation Identity

P4-N binds validation to exact commit:

`25a85015fb344552b48e57b6dd92f3b0320d37d1`

P4-N binds the exact source and test paths:

- `scripts/build_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar.py`
- `tests/test_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar_builder.py`

Future Kaggle validation execution must:

- use a tracked-clean worktree and clean index before and after execution.
- be detached at the exact full SHA
  `25a85015fb344552b48e57b6dd92f3b0320d37d1`.
- refuse dirty, staged, hash-mismatched, or non-detached state.
- use GPU OFF.

No validation run may be reused from another commit.

## F. Exact Validation Command

After P4-N independent static verification PASS and immutable freeze, the
canonical substantive CPU validation command is exactly:

```text
python -m pytest -q tests/test_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar_builder.py
```

This is the primary validation gate.

The command must:

- run only the authorized narrow test file.
- not invoke the builder CLI with `--materialize`.
- not write canonical
  `reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_*`
  outputs.
- not modify tracked repository files.
- use CPU only.
- avoid trainer, model, checkpoint, torch, network, and GPU dependencies.

Pytest-owned ephemeral temp files or directories outside canonical artifact
paths are allowed only as temporary validation fixtures and are not materialized
research artifacts.

P4-N does not authorize broadening to the full test suite. Full-suite execution
requires a later authority.

## G. Optional Preparatory Syntax Check

A separate syntax/import preparation check is materially useful because it can
fail fast on parser-level breakage in the frozen builder and test files without
materializing artifacts or replacing the substantive runtime validation.

After P4-N independent static verification PASS and immutable freeze, the only
optional preparatory Python command allowed by P4-N is exactly:

```text
python -m py_compile scripts/build_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar.py tests/test_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar_builder.py
```

This check:

- is CPU-only.
- is preparatory only.
- must not replace pytest as the substantive runtime validation.
- must not invoke builder materialization.
- must not modify tracked files.

P4-N authorizes no other Python command.

## H. Expected Narrow Test Surface

The P4-N validation must cover the already-frozen test surface, including at
minimum:

- deterministic split replay.
- canonical mapping.
- semantic hash `created_at`-only exclusion.
- exact-binary bool rejection.
- duplicate row-id rejection.
- invalid source identity rejection.
- `FRAME > PREDICATE > SUFFICIENCY > AUTHORIZED`.
- positive-margin eligibility.
- compact LF/no-BOM serialization.
- pre-existing unrelated output-directory collision fail-closed.
- pre-existing empty output-directory collision fail-closed.
- pre-existing output-path-as-file collision fail-closed.

P4-N does not invent additional scientific semantics.

## I. Side-Effect And Materialization Boundary

Validation must fail if any of the following occurs:

- the canonical P4-L output directory is created.
- sidecar or provenance artifact is materialized.
- tracked files change.
- the builder CLI materialization path is invoked.
- trainer, model, checkpoint, or network dependency is exercised.
- GPU is used.

Temporary pytest fixtures are permitted only when cleaned or confined to
ephemeral test temp locations outside canonical P4-L artifact paths.

The canonical P4-L output directory pattern is:

`reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_<builder_commit>`

The canonical sidecar/provenance filenames are:

- `p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl`
- `p3w6f2_p4l_current_lineage_effective_integrity_sidecar_provenance.json`

## J. Future Evidence And Result Contract

P4-N does not create validation evidence now.

Any future authorized P4-N validation result must record at minimum:

- execution commit SHA.
- exact command.
- Python version.
- pytest version.
- CPU/GPU state if available.
- exit code.
- collected test count.
- passed, failed, and skipped count.
- stdout/stderr or canonical textual log.
- pre/post tracked git status.
- confirmation the canonical P4-L artifact path was absent before and after.
- blockers or failure reasons.

Future validation evidence should use this deterministic namespace:

`reports/reason_router_p2_p3w6f2_p4n_cpu_non_training_builder_validation_execution_<execution_commit>/`

where `<execution_commit>` is the full 40-character execution commit SHA. P4-N
does not predict any future result hash.

## K. PASS / FAIL / BLOCKED Semantics

P4-N validation PASS requires:

- exact HEAD/hash provenance.
- exact authorized command.
- exit code `0`.
- all collected authorized tests pass.
- no unexpected skips or xpasses if relevant to the collected test set.
- tracked tree and index remain clean.
- no canonical sidecar/provenance materialization.
- GPU remains OFF.
- no authority-boundary violation.

P4-N validation FAIL applies when the authorized validation executes
meaningfully but the implementation-validation gate fails, including any test
failure or side-effect/materialization violation.

Environment/tooling failure that prevents meaningful execution is BLOCKED.
Missing authority is BLOCKED, not scientific FAIL.

## L. Future Result Verification

Any future execution result must be controller-reviewed for:

- exact commit.
- exact command.
- exit code and test counts.
- git cleanliness.
- absence of canonical materialized artifacts.
- GPU OFF.

P4-N does not require a new result/attestation meta-provenance loop unless an
applicable frozen higher authority explicitly requires one.

## M. Candidate-Time Authority Flags

Before independent verification and immutable freeze:

`validation_execution_authorized = false`

`kaggle_authorized = false`

`cpu_authorized = false`

`gpu_authorized = false`

`artifact_materialization_authorized = false`

`trainer_rebind_authorized = false`

`a0_execution_authorized = false`

`training_authorized = false`

`evaluation_authorized = false`

`training_admission_released = false`

## N. Post-Freeze Validation Authority Flags

After P4-N independent static verification PASS and immutable freeze, an
explicitly authorized validation run may have:

`validation_execution_authorized = true`

`kaggle_authorized = true`

`cpu_authorized = true`

`gpu_authorized = false`

All of the following remain false:

`training_admission_released`

`artifact_materialization_authorized`

`trainer_rebind_authorized`

`a0_execution_authorized`

`training_authorized`

`evaluation_authorized`

P4-N PASS may support a later separate authority for actual builder
materialization. P4-N PASS does not itself authorize materialization.

## O. Scientific Boundary

P4-N preserves without modification:

- Conditional First-Blocker Reason Router
- Reason-Specific Supervision
- Explicit Gradient Ownership
- `FRAME > PREDICATE > SUFFICIENCY > AUTHORIZED`
- secondary reasons diagnostic-only
- router-only final 3-way CE
- detached F/P/S and polarity CE path
- EMA observer/baseline-only
- A0-A3
- E0

P4-N does not change dataset identity, split semantics, label semantics,
reason-router semantics, loss semantics, gradient ownership, promotion
criteria, clean-vs-external evaluation separation, or scientific mechanism
claims.

## P. Stop Conditions

P4-N is BLOCKED if any of the following holds:

- namespace collision.
- applicable execution authority already exists.
- implementation commit mismatch.
- tracked worktree or index dirt.
- builder or test paths mismatch.
- narrow validation cannot be run without materialization.
- tests require trainer, model, GPU, or network.
- full-suite execution is required to interpret this authority.
- scientific semantics need modification.
- exact PASS/FAIL criteria cannot be resolved statically.

## Q. Candidate Readiness

Candidate path:

`reports/reason_router_p2_p3w6f2_p4n_cpu_non_training_builder_validation_execution_authority_spec.md`

Candidate SHA256 is computed after file creation and is not predicted inside
this specification body.

Final candidate readiness token:

`P3W6F2P4N_CPU_NON_TRAINING_BUILDER_VALIDATION_EXECUTION_AUTHORITY_CANDIDATE_READY_FOR_INDEPENDENT_VERIFICATION`
