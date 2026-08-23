# P3-W6-F2-P4-P POSIX Publication Runtime Validation Execution Authority Specification

Authority/version:

`P3W6F2P4P_POSIX_PUBLICATION_RUNTIME_VALIDATION_EXECUTION_AUTHORITY_V1`

This document is a candidate runtime-validation execution authority
specification only. It becomes executable authority only after independent
verification and immutable freeze. Candidate creation does not authorize
validation execution, Kaggle execution, CPU execution, GPU execution, artifact
materialization, trainer rebind, A0, training, evaluation, commit, or push.

## 1. HEAD And State Proof

Candidate creation requires:

- HEAD exactly
  `2f9e6076791358922e3ebd70e89533d9cb83b458`.
- Tracked worktree and index clean.
- Existing untracked files, if any, remain untouched.
- Exactly one new untracked file may be created by this task:
  `reports/reason_router_p2_p3w6f2_p4p_posix_publication_runtime_validation_execution_authority_spec.md`.

Candidate creation evidence:

- `git rev-parse HEAD` returned
  `2f9e6076791358922e3ebd70e89533d9cb83b458`.
- `git diff --quiet` returned success.
- `git diff --cached --quiet` returned success.
- `git status --short` reported existing untracked local files only before
  candidate creation.
- Warnings from inaccessible local pytest/cache directories during status are
  environmental read warnings only; they are not validation evidence.

If HEAD differs, tracked worktree/index dirt is present, or candidate creation
requires touching existing untracked work, P4-P is BLOCKED.

## 2. Namespace Verdict

Repository namespace search for P4-P/P4P/equivalent runtime authority terms
found no collision for:

- `P3W6F2P4P`
- `P4P_POSIX`
- `P4-P`
- `posix_publication_runtime_validation`
- `POSIX_PUBLICATION_RUNTIME_VALIDATION`
- `REAL_LINUX_RENAMEAT2_NOREPLACE`

One unrelated `P4-P8` content hit was observed in
`reports/stage196b2b6p9p2_separate_observability_instrumentations_spec.md`.
That hit is not this authority namespace and is not an applicable collision.

If independent verification finds an existing P4-P/P4P equivalent POSIX
publication runtime-validation execution authority, P4-P is BLOCKED.

## 3. Implementation Identity

P4-P binds exclusively to implementation commit:

`2f9e6076791358922e3ebd70e89533d9cb83b458`

Bound production builder:

`scripts/build_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar.py`

Bound test file:

`tests/test_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar_builder.py`

No mutable branch, latest, working-tree, or future commit semantics are
authorized. Previous P4-N validation applies only to old implementation
`25a85015fb344552b48e57b6dd92f3b0320d37d1` and must not be reused as
validation evidence for this implementation.

## 4. Pytest Gate

After independent verification and freeze, P4-P authorizes exactly this narrow
CPU-only pytest gate:

```bash
python -m pytest -q tests/test_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar_builder.py
```

Requirements:

- exact test file only;
- CPU only;
- no builder CLI;
- no `--materialize`;
- no canonical artifact output;
- no full suite;
- do not predict collected count.

Gate 1 PASS requires exit code 0, all collected narrow tests pass, and no
material skip or xpass.

If this command fails, P4-P result is implementation-validation FAIL.

## 5. Pytest-Only Limitation

The Linux-success unit test uses a mocked `renameat2` boundary and models the
directory move. This is correct unit coverage, but it does not establish that
actual Kaggle/Linux libc exposes `renameat2` or accepts `RENAME_NOREPLACE`.

Therefore P4-P requires one additional real-platform smoke gate.

## 6. Real Linux Smoke Contract

After independent verification and freeze, P4-P authorizes exactly one bounded
real-platform smoke script whose sole purpose is to exercise production:

`builder.atomic_publish_directory_noreplace(...)`

against disposable directories under a system temporary directory.

The smoke must not:

- call builder CLI;
- call `build_sidecar_artifacts`;
- access or write canonical P4-L reports output;
- consume the 3600-row dataset;
- create research artifacts;
- modify tracked files;
- monkeypatch `load_libc`, `renameat2`, platform detection, or errno.

The smoke must use `tempfile.TemporaryDirectory()` or equivalent temporary
location outside canonical artifact paths.

The exact authorized smoke command is:

```bash
python - <<'PY'
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path.cwd()))

from scripts import build_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar as builder

PASS_TOKEN = "P3W6F2P4P_REAL_LINUX_RENAMEAT2_NOREPLACE_SMOKE_PASS"

assert sys.platform.startswith("linux"), sys.platform
assert builder.running_on_linux() is True
assert builder.running_on_windows() is False

with tempfile.TemporaryDirectory(prefix="p3w6f2_p4p_renameat2_smoke_") as tmp:
    root = pathlib.Path(tmp)

    staging = root / "success_staging"
    final = root / "success_final"
    final_dir = final
    sentinel_a = staging / "sentinel_a.txt"
    sentinel_b = staging / "sentinel_b.txt"
    staging.mkdir()
    sentinel_a.write_text("alpha\n", encoding="utf-8")
    sentinel_b.write_text("beta\n", encoding="utf-8")

    assert not final_dir.exists()
    builder.atomic_publish_directory_noreplace(staging, final_dir)
    assert not staging.exists()
    assert final_dir.is_dir()
    assert sorted(p.name for p in final_dir.iterdir()) == sorted(
        [sentinel_a.name, sentinel_b.name]
    )
    assert (final_dir / sentinel_a.name).read_bytes() == b"alpha\n"
    assert (final_dir / sentinel_b.name).read_bytes() == b"beta\n"
    assert sorted(p.name for p in root.iterdir()) == [final_dir.name]

    collision_staging = root / "collision_staging"
    collision_final = root / "collision_final"
    collision_attempt = collision_staging / "attempt.txt"
    collision_foreign = collision_final / "foreign.txt"
    collision_staging.mkdir()
    collision_attempt.write_text("attempt\n", encoding="utf-8")
    collision_final.mkdir()
    collision_foreign.write_text("foreign\n", encoding="utf-8")
    collision_parent_entries_before = sorted(p.name for p in root.iterdir())
    collision_staging_inode = collision_staging.stat().st_ino
    collision_final_inode = collision_final.stat().st_ino

    try:
        builder.atomic_publish_directory_noreplace(collision_staging, collision_final)
    except builder.BuildBlocked as exc:
        assert str(exc) == "P4L_OUTPUT_PATH_PREEXISTING", str(exc)
    else:
        raise AssertionError("collision unexpectedly succeeded")

    assert sorted(p.name for p in root.iterdir()) == collision_parent_entries_before
    assert collision_staging.is_dir()
    assert collision_staging.stat().st_ino == collision_staging_inode
    assert collision_attempt.read_bytes() == b"attempt\n"
    assert collision_final.is_dir()
    assert collision_final.stat().st_ino == collision_final_inode
    assert collision_foreign.read_bytes() == b"foreign\n"
    assert sorted(p.name for p in collision_final.iterdir()) == [collision_foreign.name]
    assert not (collision_final / collision_attempt.name).exists()

print(PASS_TOKEN)
PY
```

## 7. Success-Case Semantics

The smoke success case must:

1. create a temporary staging directory;
2. put two sentinel files in it;
3. require final destination absent;
4. call actual `atomic_publish_directory_noreplace()` without monkeypatching
   libc loading, `renameat2`, platform detection, or errno;
5. require the call to return normally;
6. require staging no longer exists;
7. require final directory exists;
8. require final directory contains exactly the two expected sentinel entries
   and no extra entries;
9. require both sentinel contents are byte-for-byte unchanged;
10. require no foreign, backup, temporary, or move-aside artifact appears in
    the final destination or the controlled temporary parent scope.

## 8. Collision-Case Semantics

The smoke collision case must:

1. create another staging directory;
2. create final destination beforehand with foreign sentinel content;
3. call actual `atomic_publish_directory_noreplace()`;
4. require `builder.BuildBlocked`;
5. require exact message `P4L_OUTPUT_PATH_PREEXISTING`;
6. require foreign destination remains unchanged;
7. require staging remains present at helper level;
8. require attempted staging sentinel does not appear inside the foreign
   destination;
9. require no backup directory, move-aside destination, or extra sibling
   artifact attributable to the publication attempt is created in the
   controlled temporary parent scope;
10. require foreign destination is not renamed, replaced, or deleted;
11. require no overwrite or delete occurred.

`finalize_payloads_atomic` cleanup is separately covered by pytest and is not
part of this direct helper smoke.

## 9. BLOCKED Vs FAIL Mapping

If real libc symbol or kernel capability is unavailable and the helper raises
`P4L_ATOMIC_DIRECTORY_NOREPLACE_UNSUPPORTED`, P4-P result is BLOCKED, not
implementation FAIL, unless evidence proves the implementation incorrectly
handles a supported kernel/libc capability.

If the collision case overwrites, replaces, deletes, mutates, or incorrectly
succeeds against the foreign destination, P4-P result is FAIL.

If pytest fails, P4-P result is implementation-validation FAIL.

If the smoke cannot run without canonical materialization, P4-P is BLOCKED.

If tooling or environment prevents meaningful execution of either required
gate without indicating an implementation defect, P4-P is BLOCKED. Examples
include Python/pytest unavailable or unusable, temporary filesystem
unavailable, platform inspection impossible, execution environment preventing
the authorized smoke invocation, or required runtime observation unavailable.

This does not convert pytest assertion/test failure, actual no-clobber
violation, or implementation/runtime semantic defect into BLOCKED; those remain
FAIL as specified above.

## 10. Kaggle Preconditions

Before execution, the Kaggle environment must satisfy:

- HEAD exactly
  `2f9e6076791358922e3ebd70e89533d9cb83b458`;
- detached exact SHA, not a mutable branch checkout;
- tracked worktree/index clean;
- canonical P4-L output path for this implementation absent:
  `reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458`;
- GPU OFF.

## 11. Postconditions

After both gates:

- `git rev-parse HEAD` exactly
  `2f9e6076791358922e3ebd70e89533d9cb83b458`;
- tracked worktree/index clean;
- canonical P4-L path still absent;
- no research artifact materialization;
- GPU remains OFF;
- Python version recorded;
- pytest version recorded;
- actual platform recorded.

## 12. PASS Semantics

P4-P PASS requires all of the following:

Gate 1:

- exact narrow pytest command exited 0;
- all collected narrow tests passed;
- no material skip or xpass.

Gate 2:

- exact real-Linux smoke PASS token emitted:
  `P3W6F2P4P_REAL_LINUX_RENAMEAT2_NOREPLACE_SMOKE_PASS`;
- real production helper used;
- no mocks;
- successful no-replace publication proven in a temp directory;
- real collision preserved the foreign destination.

Global:

- exact commit provenance;
- tracked clean pre/post;
- canonical artifact absent pre/post;
- GPU OFF;
- no boundary violation.

## 13. Interpretation Boundary

P4-P distinguishes:

1. code correctness: prior static PASS;
2. execution success: P4-P may establish;
3. artifact/provenance validity: not established;
4. scientific conclusion: not established.

P4-P PASS establishes materialization-readiness of the publication primitive.
It does not authorize or validate the actual canonical sidecar artifact.

## 14. Authority Flags

At candidate creation time all flags are false:

- `validation_execution_authorized = false`
- `kaggle_authorized = false`
- `cpu_authorized = false`
- `gpu_authorized = false`
- `artifact_materialization_authorized = false`
- `trainer_rebind_authorized = false`
- `a0_execution_authorized = false`
- `training_admission_released = false`
- `training_authorized = false`
- `evaluation_authorized = false`

After independent verification and freeze, only the exact P4-P run may have:

- `validation_execution_authorized = true`
- `kaggle_authorized = true`
- `cpu_authorized = true`
- `gpu_authorized = false`

All materialization, training, rebind, A0, and evaluation flags remain false.

## 15. Evidence Namespace

Future P4-P evidence must use deterministic namespace:

`reports/reason_router_p2_p3w6f2_p4p_posix_publication_runtime_validation_execution_2f9e6076791358922e3ebd70e89533d9cb83b458/`

Evidence must record:

- authority freeze commit;
- exact implementation commit;
- exact pytest command;
- pytest exit, counts, and output;
- exact smoke script or command;
- smoke PASS/BLOCKED/FAIL output;
- Python version;
- pytest version;
- `sys.platform`;
- pre/post tracked status;
- canonical output absence pre/post;
- GPU observations;
- blockers or failure reasons.

No evidence is created by this candidate. No evidence hashes are predicted.

## 16. Post-PASS Boundary

P4-P PASS may support a later separate canonical materialization authority.

P4-P itself does not authorize:

- builder `--materialize`;
- canonical sidecar/provenance creation;
- trainer rebind;
- A0;
- training;
- evaluation.

## 17. Scientific Boundary

P4-P changes no frozen reason-router, loss, gradient, dataset, split, label,
promotion, trainer, checkpoint, or scientific semantics.

Training: NO.

Evaluation: NO.

Materialization: NO.

GPU: NO.

Commit/push: NO.

## 18. Candidate Path

`reports/reason_router_p2_p3w6f2_p4p_posix_publication_runtime_validation_execution_authority_spec.md`

## 19. Candidate SHA256

The candidate file SHA256 must be computed after file creation and reported in
the creator handoff. It is not embedded here because embedding it would mutate
the hashed bytes.

## 20. Blockers And Final Status Requirements

P4-P is BLOCKED on:

- namespace collision;
- HEAD/state mismatch;
- implementation identity mismatch;
- inability to run smoke without canonical materialization;
- smoke requiring production-code modification;
- real Linux primitive unsupported;
- materialization required merely to validate runtime readiness.

Final candidate handoff must report:

- final `git status --short`;
- candidate SHA256;
- whether exactly one new untracked spec file was created;
- whether Python, pytest, builder execution, Kaggle execution,
  materialization, commit, and push were not run.

Success token for candidate readiness:

`P3W6F2P4P_POSIX_PUBLICATION_RUNTIME_VALIDATION_EXECUTION_AUTHORITY_CANDIDATE_READY_FOR_INDEPENDENT_VERIFICATION`
