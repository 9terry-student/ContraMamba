# P3-W6-F2-P4-W Canonical P4-L Exact-Byte Restore Execution Authority Candidate

Authority/version:

`P3W6F2P4W_CANONICAL_P4L_EXACT_BYTE_RESTORE_EXECUTION_AUTHORITY_CANDIDATE_V1`

This document is a narrowly bounded execution-authority candidate only.
Candidate creation itself does not authorize Python execution, pytest, Kaggle
execution, CPU execution, GPU execution, builder execution, P4-Q
materialization rerun, reconstruction, artifact semantic modification,
training, evaluation, trainer rebind, A0, commit, push, or P4-V validator
execution.

Only after independent static verification PASS and immutable P4-W freeze may
this authority authorize exactly one future CPU-only recovery command whose
sole mutation is restoring two externally preserved original P4-Q byte copies
into the absent canonical P4-L destination path.

## 1. Authority Basis

Creation authority:

- Current controller instruction.
- Valid P4-V descendant authority freeze:
  `3e8fa6269a0728d615e03240ab4cd8f15418c178`.
- Historical P4-V candidate:
  `90c7a0cf713b79cadcc18a2f95b5b80f834272b6`.
- P4-V committed-copy SHA256:
  `e7ea41dc0a9bca4921ef9c0eccee40701bc82b81dac31b631e30863d85521a0c`.
- P4-V verification attestation SHA256:
  `cbd901e8027098ef01fbd1ec3e950026579cf50e5a7ffe499949a65d10a58db5`.
- Frozen P4-L:
  `80cb034792f03226cf6e22c196c1229ed4e6dd62`.
- Exact builder/execution HEAD:
  `2f9e6076791358922e3ebd70e89533d9cb83b458`.
- Observed current Kaggle HEAD:
  `8a841d3989c65c9cc575ea58b06734cf69d262b3`.
- Observed Kaggle state recorded by the controller:
  tracked worktree clean; index clean; target execution commit object present;
  P4-V freeze object present; canonical P4-L directory absent; canonical
  sidecar absent; canonical provenance absent; no staging sibling observed.
- Preserved original P4-Q byte identities:
  sidecar SHA256
  `2b8cffdf71d68a8abeb3b6eb3534eeb664bd012483bcebd9716c7a6645a487f1`;
  provenance SHA256
  `9d248df09ae8ba471966c468a1e06278ad046908cfe53da623ecc95d8da4cdf2`.
- `AGENTS.md`.

The controller named `04_KAGGLE_RUNBOOK.md`, `05_FAILURE_RECOVERY.md`, and
`06_NAMING_AND_PROVENANCE.md`; these are treated as required future workflow
context where available. This candidate does not supersede them.

Phase:

`P4-W CANONICAL P4-L EXACT-BYTE RESTORE EXECUTION AUTHORITY CANDIDATE CREATION ONLY`

## 2. Historical Disposition

P4-Q remains historical FAIL overall. Its original materialization bytes are
not invalidated by destination absence alone.

P4-R, P4-S, P4-T, and P4-U remain historical FAIL as already recorded.

P4-V runtime validation has NOT executed yet.

Canonical P4-L integrity remains NOT ESTABLISHED.

The absence of canonical files in the currently observed Kaggle worktree is an
execution-input availability problem, not by itself an artifact-content defect.

No scientific conclusion is established.

## 3. Restoration Source

Restoration MUST use externally preserved byte copies only.

Forbidden source mechanisms:

- builder execution;
- P4-Q materialization rerun;
- reconstruction from JSON semantics;
- line-ending conversion;
- parse-and-reserialize;
- manual editing;
- download from an unverified alternate artifact.

The source copies must be regular non-symlink files. They must remain outside
the canonical destination directory. Their physical SHA256 values must exactly
equal:

- sidecar:
  `2b8cffdf71d68a8abeb3b6eb3534eeb664bd012483bcebd9716c7a6645a487f1`;
- provenance:
  `9d248df09ae8ba471966c468a1e06278ad046908cfe53da623ecc95d8da4cdf2`.

No reliable historical byte lengths are asserted by this candidate. No byte
length guards are invented.

## 4. Runtime Interface

The future command requires exactly these workflow-supplied parameters:

- `P4W_AUTHORITY_FREEZE`
- `P4W_SIDECAR_SOURCE`
- `P4W_PROVENANCE_SOURCE`

`P4W_AUTHORITY_FREEZE` must be exactly a 40-character lowercase hex commit
that contains this P4-W authority path.

`P4W_SIDECAR_SOURCE` must identify a regular non-symlink file named exactly
`p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl`.

`P4W_PROVENANCE_SOURCE` must identify a regular non-symlink file named exactly
`p3w6f2_p4l_current_lineage_effective_integrity_sidecar_provenance.json`.

The two source paths must be distinct from each other, must not equal their
canonical destinations, and must not be inside the canonical destination
directory. Missing files, symlinks, wrong names, source equals destination, or
wrong hashes are FAIL-closed blockers.

## 5. Starting Kaggle State And HEAD Recovery Boundary

The exact currently observed starting HEAD is:

`8a841d3989c65c9cc575ea58b06734cf69d262b3`

The future restore command may authorize exactly one bounded tracked-worktree
identity transition:

```bash
git switch --detach 2f9e6076791358922e3ebd70e89533d9cb83b458
```

This transition is allowed only if BEFORE switching:

- current HEAD is exactly
  `8a841d3989c65c9cc575ea58b06734cf69d262b3`;
- tracked worktree is clean;
- index is clean;
- target commit object exists;
- P4-W freeze object exists and contains this authority path;
- canonical destination directory is absent;
- canonical destination sidecar is absent;
- canonical destination provenance is absent;
- no `.p4l-staging-*` sibling exists;
- source copies already pass exact physical SHA checks.

After switching:

- HEAD must be exactly
  `2f9e6076791358922e3ebd70e89533d9cb83b458`;
- tracked worktree and index must remain clean;
- canonical destination must still be absent;
- source hashes must remain unchanged.

P4-W does not authorize reset, clean, branch movement, pull, merge,
cherry-pick, rebase, fresh bootstrap, checkout of mutable `main`, or any
unbounded repository synchronization.

## 6. Exact Restore Mutation

P4-W authorizes creation of exactly this canonical directory:

`reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458/`

containing exactly:

- `p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl`;
- `p3w6f2_p4l_current_lineage_effective_integrity_sidecar_provenance.json`.

The destination MUST be absent before restore. Existing file, directory,
symlink, broken symlink, or any already existing canonical child path is a
FAIL-closed blocker.

The restore should use a same-parent staging directory plus no-overwrite final
publish. Staging is allowed only as attempt-owned scratch and must be removed
on failure. The final publish must not overwrite any existing destination.

No repository file other than this canonical output path may be created,
changed, deleted, renamed, backed up, moved aside, or chmodded. Source copies
must remain unchanged.

## 7. Post-Restore Validation

The future command must require:

- canonical directory is a normal directory, not a symlink;
- exactly two entries;
- both entries are regular non-symlink files;
- restored sidecar SHA256 exactly
  `2b8cffdf71d68a8abeb3b6eb3534eeb664bd012483bcebd9716c7a6645a487f1`;
- restored provenance SHA256 exactly
  `9d248df09ae8ba471966c468a1e06278ad046908cfe53da623ecc95d8da4cdf2`;
- source hashes unchanged;
- no `.p4l-staging-*` sibling remains;
- HEAD remains exact execution HEAD
  `2f9e6076791358922e3ebd70e89533d9cb83b458`;
- tracked worktree and index remain clean when untracked canonical artifacts
  are ignored;
- GPU remains off.

## 8. PASS Interpretation And P4-V Handoff Boundary

P4-W PASS establishes ONLY:

- exact known physical P4-Q bytes were restored at the required canonical path;
- execution-input availability was recovered.

P4-W PASS MUST NOT establish:

- canonical P4-L semantic/provenance integrity by itself;
- P4-V PASS;
- trainer rebind authority;
- A0 authority;
- training/evaluation authority;
- scientific conclusion.

After P4-W PASS, the only next authorized research action is the already
frozen P4-V read-only validator, exactly once, under:

`P4V_AUTHORITY_FREEZE=3e8fa6269a0728d615e03240ab4cd8f15418c178`

P4-W does not execute P4-V.

## 9. Exact Future CPU-Only Recovery Command

The complete exact future Kaggle recovery command is:

```bash
CUDA_VISIBLE_DEVICES="" python - <<'PY'
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from ctypes import CDLL, c_char_p, c_int, get_errno
from hashlib import sha256
from pathlib import Path, PurePosixPath

PASS_TOKEN = "P3W6F2P4W_CANONICAL_P4L_EXACT_BYTE_RESTORE_PASS"

START_HEAD = "8a841d3989c65c9cc575ea58b06734cf69d262b3"
EXECUTION_HEAD = "2f9e6076791358922e3ebd70e89533d9cb83b458"
P4W_AUTHORITY_PATH = "reports/reason_router_p2_p3w6f2_p4w_canonical_p4l_exact_byte_restore_execution_authority_spec.md"
OUTPUT_DIR_REL = "reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458"
SIDECAR_NAME = "p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl"
PROVENANCE_NAME = "p3w6f2_p4l_current_lineage_effective_integrity_sidecar_provenance.json"
EXPECTED_SIDECAR_SHA256 = "2b8cffdf71d68a8abeb3b6eb3534eeb664bd012483bcebd9716c7a6645a487f1"
EXPECTED_PROVENANCE_SHA256 = "9d248df09ae8ba471966c468a1e06278ad046908cfe53da623ecc95d8da4cdf2"
AT_FDCWD = -100
RENAME_NOREPLACE = 1
SYS_RENAMEAT2_X86_64 = 316


def fail(code: str) -> None:
    raise SystemExit(f"P4W_FAIL:{code}")


def require(condition: bool, code: str) -> None:
    if not condition:
        fail(code)


def run_git(args: list[str]) -> str:
    proc = subprocess.run(
        ["git", *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    require(proc.returncode == 0, "GIT_COMMAND_FAILED_" + "_".join(args[:2]).upper())
    return proc.stdout.strip()


def physical_sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def repo_relative_path(repo_root: Path, rel: str) -> Path:
    pure = PurePosixPath(rel)
    require(not pure.is_absolute(), "ABSOLUTE_REPO_RELATIVE_PATH")
    require(".." not in pure.parts, "DOTDOT_REPO_RELATIVE_PATH")
    return repo_root.joinpath(*pure.parts)


def require_normal_file(path: Path, expected_name: str, code: str) -> Path:
    require(path.name == expected_name, code + "_WRONG_NAME")
    require(path.exists(), code + "_MISSING")
    require(not path.is_symlink(), code + "_SYMLINK")
    require(path.is_file(), code + "_NOT_FILE")
    return path.resolve(strict=True)


def require_no_staging(parent: Path) -> None:
    for sibling in parent.glob(".p4l-staging-*"):
        fail("STAGING_SIBLING_PRESENT")


def require_destination_absent(output_dir: Path, sidecar_dest: Path, provenance_dest: Path) -> None:
    require(not output_dir.exists() and not output_dir.is_symlink(), "DESTINATION_PRESENT")
    require(not sidecar_dest.exists() and not sidecar_dest.is_symlink(), "SIDECAR_DESTINATION_PRESENT")
    require(not provenance_dest.exists() and not provenance_dest.is_symlink(), "PROVENANCE_DESTINATION_PRESENT")


def rename_noreplace(source: Path, dest: Path) -> None:
    require(sys.platform.startswith("linux"), "RENAME_NOREPLACE_REQUIRES_LINUX")
    libc = CDLL(None, use_errno=True)
    result = libc.syscall(
        c_int(SYS_RENAMEAT2_X86_64),
        c_int(AT_FDCWD),
        c_char_p(os.fsencode(source)),
        c_int(AT_FDCWD),
        c_char_p(os.fsencode(dest)),
        c_int(RENAME_NOREPLACE),
    )
    require(result == 0, "RENAME_NOREPLACE_FAILED_ERRNO_" + str(get_errno()))


def require_sources_outside_destination(source: Path, output_dir: Path, dest: Path, code: str) -> None:
    source_resolved = source.resolve(strict=True)
    dest_parent = output_dir.resolve(strict=False)
    require(source_resolved != dest.resolve(strict=False), code + "_SOURCE_EQUALS_DESTINATION")
    require(dest_parent not in source_resolved.parents, code + "_SOURCE_INSIDE_DESTINATION")


repo_root = Path.cwd().resolve(strict=True)

authority_freeze = os.environ.get("P4W_AUTHORITY_FREEZE", "")
sidecar_source_text = os.environ.get("P4W_SIDECAR_SOURCE", "")
provenance_source_text = os.environ.get("P4W_PROVENANCE_SOURCE", "")

require(re.fullmatch(r"[0-9a-f]{40}", authority_freeze or "") is not None, "P4W_AUTHORITY_FREEZE_INVALID")
require(sidecar_source_text != "", "SIDECAR_SOURCE_UNSET")
require(provenance_source_text != "", "PROVENANCE_SOURCE_UNSET")
require(os.environ.get("CUDA_VISIBLE_DEVICES", "") == "", "GPU_NOT_OFF")

run_git(["cat-file", "-e", f"{authority_freeze}^{{commit}}"])
authority_listing = run_git(["ls-tree", "-r", "--name-only", authority_freeze, "--", P4W_AUTHORITY_PATH])
require(authority_listing == P4W_AUTHORITY_PATH, "P4W_AUTHORITY_PATH_MISSING_AT_FREEZE")
run_git(["cat-file", "-e", f"{EXECUTION_HEAD}^{{commit}}"])

output_dir = repo_relative_path(repo_root, OUTPUT_DIR_REL)
sidecar_dest = output_dir / SIDECAR_NAME
provenance_dest = output_dir / PROVENANCE_NAME
require(output_dir.parent.exists(), "OUTPUT_PARENT_MISSING")
require(not output_dir.parent.is_symlink(), "OUTPUT_PARENT_SYMLINK")

sidecar_source = require_normal_file(Path(sidecar_source_text), SIDECAR_NAME, "SIDECAR_SOURCE")
provenance_source = require_normal_file(Path(provenance_source_text), PROVENANCE_NAME, "PROVENANCE_SOURCE")
require(sidecar_source != provenance_source, "SOURCES_IDENTICAL")
require_sources_outside_destination(sidecar_source, output_dir, sidecar_dest, "SIDECAR")
require_sources_outside_destination(provenance_source, output_dir, provenance_dest, "PROVENANCE")

sidecar_source_sha_before = physical_sha(sidecar_source)
provenance_source_sha_before = physical_sha(provenance_source)
require(sidecar_source_sha_before == EXPECTED_SIDECAR_SHA256, "SIDECAR_SOURCE_SHA_MISMATCH")
require(provenance_source_sha_before == EXPECTED_PROVENANCE_SHA256, "PROVENANCE_SOURCE_SHA_MISMATCH")

require(run_git(["rev-parse", "HEAD"]) == START_HEAD, "START_HEAD_MISMATCH")
require(run_git(["status", "--short", "--untracked-files=no"]) == "", "TRACKED_WORKTREE_DIRTY")
require(run_git(["diff", "--cached", "--name-status"]) == "", "INDEX_DIRTY")
require_destination_absent(output_dir, sidecar_dest, provenance_dest)
require_no_staging(output_dir.parent)

switch_proc = subprocess.run(
    ["git", "switch", "--detach", EXECUTION_HEAD],
    check=False,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
)
require(switch_proc.returncode == 0, "DETACH_SWITCH_FAILED")

require(run_git(["rev-parse", "HEAD"]) == EXECUTION_HEAD, "POST_SWITCH_HEAD_MISMATCH")
require(run_git(["status", "--short", "--untracked-files=no"]) == "", "POST_SWITCH_TRACKED_WORKTREE_DIRTY")
require(run_git(["diff", "--cached", "--name-status"]) == "", "POST_SWITCH_INDEX_DIRTY")
require_destination_absent(output_dir, sidecar_dest, provenance_dest)
require(physical_sha(sidecar_source) == sidecar_source_sha_before, "SIDECAR_SOURCE_SHA_CHANGED_AFTER_SWITCH")
require(physical_sha(provenance_source) == provenance_source_sha_before, "PROVENANCE_SOURCE_SHA_CHANGED_AFTER_SWITCH")

staging = output_dir.parent / f".p4l-staging-p4w-{os.getpid()}"
require(not staging.exists() and not staging.is_symlink(), "STAGING_ALREADY_EXISTS")

published = False
try:
    staging.mkdir(mode=0o755)
    require(not staging.is_symlink() and staging.is_dir(), "STAGING_NOT_NORMAL_DIR")
    staging_sidecar = staging / SIDECAR_NAME
    staging_provenance = staging / PROVENANCE_NAME
    shutil.copyfile(sidecar_source, staging_sidecar)
    shutil.copyfile(provenance_source, staging_provenance)
    require(not staging_sidecar.is_symlink() and staging_sidecar.is_file(), "STAGING_SIDECAR_NOT_FILE")
    require(not staging_provenance.is_symlink() and staging_provenance.is_file(), "STAGING_PROVENANCE_NOT_FILE")
    require(physical_sha(staging_sidecar) == EXPECTED_SIDECAR_SHA256, "STAGING_SIDECAR_SHA_MISMATCH")
    require(physical_sha(staging_provenance) == EXPECTED_PROVENANCE_SHA256, "STAGING_PROVENANCE_SHA_MISMATCH")
    require_destination_absent(output_dir, sidecar_dest, provenance_dest)
    rename_noreplace(staging, output_dir)
    published = True
finally:
    if not published and staging.exists() and staging.is_dir() and not staging.is_symlink():
        shutil.rmtree(staging)

require(not output_dir.is_symlink() and output_dir.is_dir(), "OUTPUT_DIR_NOT_NORMAL")
entries = [p.name for p in output_dir.iterdir()]
require(len(entries) == 2 and set(entries) == {SIDECAR_NAME, PROVENANCE_NAME}, "OUTPUT_DIR_ENTRIES_MISMATCH")
require(not sidecar_dest.is_symlink() and sidecar_dest.is_file(), "RESTORED_SIDECAR_NOT_FILE")
require(not provenance_dest.is_symlink() and provenance_dest.is_file(), "RESTORED_PROVENANCE_NOT_FILE")
require(physical_sha(sidecar_dest) == EXPECTED_SIDECAR_SHA256, "RESTORED_SIDECAR_SHA_MISMATCH")
require(physical_sha(provenance_dest) == EXPECTED_PROVENANCE_SHA256, "RESTORED_PROVENANCE_SHA_MISMATCH")
require(physical_sha(sidecar_source) == sidecar_source_sha_before, "SIDECAR_SOURCE_SHA_CHANGED")
require(physical_sha(provenance_source) == provenance_source_sha_before, "PROVENANCE_SOURCE_SHA_CHANGED")
require_no_staging(output_dir.parent)
require(run_git(["rev-parse", "HEAD"]) == EXECUTION_HEAD, "POST_HEAD_MISMATCH")
require(run_git(["status", "--short", "--untracked-files=no"]) == "", "POST_TRACKED_WORKTREE_DIRTY")
require(run_git(["diff", "--cached", "--name-status"]) == "", "POST_INDEX_DIRTY")
require(os.environ.get("CUDA_VISIBLE_DEVICES", "") == "", "POST_GPU_NOT_OFF")

print(PASS_TOKEN)
PY
```

The command prints the success token only after all postconditions pass:

`P3W6F2P4W_CANONICAL_P4L_EXACT_BYTE_RESTORE_PASS`

## 10. Static Audit Of Embedded Restore Command

Mutation boundaries:

- The only Git identity mutation is `git switch --detach
  2f9e6076791358922e3ebd70e89533d9cb83b458`, guarded by exact starting HEAD,
  clean tracked worktree, clean index, existing target commit, existing P4-W
  freeze, absent destination, absent staging, and verified source hashes.
- The only repository artifact mutation after the switch is creation of the
  exact canonical P4-L output directory and its two children.
- No reset, clean, pull, merge, cherry-pick, rebase, branch movement, builder,
  materializer, parser, serializer, training, evaluation, chmod, commit, push,
  or P4-V invocation appears in the command.

Hash guards:

- Source physical SHA256 is checked before the switch.
- Source physical SHA256 is rechecked after the switch.
- Staging physical SHA256 is checked before publish.
- Restored destination physical SHA256 is checked after publish.
- Source physical SHA256 is rechecked after publish.

Source/destination separation:

- Source basenames are exact.
- Source files must be normal files and not symlinks.
- Source paths must not equal destination paths.
- Source paths must not be inside the canonical destination directory.

Fail-closed behavior:

- Every precondition and postcondition uses an explicit failure token.
- Existing destination or staging path blocks restore.
- The final publish is same-parent staging rename and is attempted only after
  destination absence and staging hash checks.
- Attempt-owned staging is removed only on failed pre-publication attempt.
- No existing canonical path is overwritten, backed up, or moved aside.

No authority widening:

- GPU: NO.
- Builder/materialization regeneration: NO.
- Training: NO.
- Evaluation: NO.
- Trainer rebind: NO.
- A0: NO.
- P4-V validator execution inside P4-W: NO.
- Commit/push during execution: NO.

## 11. Candidate Creation Boundary

Candidate path:

`reports/reason_router_p2_p3w6f2_p4w_canonical_p4l_exact_byte_restore_execution_authority_spec.md`

Candidate creation final state requires:

- exactly one new untracked P4-W authority-spec candidate;
- existing untracked review patches untouched;
- `reports/stage180a_pass2_annotations_completed.csv` untouched;
- no restore execution;
- no Python, pytest, builder, materialization, validation, training,
  evaluation, Kaggle, commit, or push.

Static verification is required before immutable freeze. Candidate creation
itself does not authorize execution; only independent static verification PASS
plus immutable freeze does.

Final readiness token:

`P4W_CANONICAL_P4L_EXACT_BYTE_RESTORE_AUTHORITY_CANDIDATE_READY`
