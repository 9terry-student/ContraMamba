# P3-W6-F2-P4-L Current-HEAD Exact-Byte Provisioning Execution-Authority Candidate

Authority/version:

`P3W6F2P4L_CURRENT_HEAD_EXACT_BYTE_PROVISIONING_EXECUTION_AUTHORITY_CANDIDATE_V1`

## 1. Disposition and boundary

This document is a candidate execution authority only. Candidate creation does
not execute Python, Kaggle, provisioning, copying, builder code, P4-Q,
training, evaluation, validation, commit, push, or GPU work. It becomes
executable only after independent static verification and an explicit
immutable freeze under the current controller workflow.

If frozen, it authorizes exactly one future CPU-only provisioning attempt. The
attempt must begin and end at HEAD
`a1b614d0e659d2b34889cb55aef94e1824df2fd1`. The only permitted repository
artifact mutation is creation of the absent canonical P4-L directory and its
two exact-byte child files. No Git identity mutation is authorized.

The historical P4-W HEAD transition

`8a841d3989c65c9cc575ea58b06734cf69d262b3`
` -> `2f9e6076791358922e3ebd70e89533d9cb83b458`

is explicitly rejected for this workflow. P4-W is design precedent only;
neither its starting HEAD, target HEAD, `git switch`, nor its authority
freeze may be reused here.

## 2. Authority chain consumed

This candidate consumes, in precedence order:

1. The current controller instruction for P4-L current-HEAD candidate creation.
2. Current repository HEAD
   `a1b614d0e659d2b34889cb55aef94e1824df2fd1`.
3. `reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_artifact_contract_authority_spec.md`
   as the frozen/current P4-L artifact contract.
4. `reports/PRE_URP_HANDOFF.md`, `reports/RESEARCH_STATE.md`,
   `reports/ARTIFACT_INDEX.md`, `reports/artifact_manifest.json`,
   `docs/RESEARCH_OPERATIONS.md`, and `AGENTS.md` as current operational and
   repository-wide constraints.
5. `reports/reason_router_p2_p3w6f2_p4w_canonical_p4l_exact_byte_restore_execution_authority_spec.md`
   as historical precedent for fail-closed source checks, same-parent staging,
   and no-overwrite publication only.
6. Controller-supplied imported source-discovery and source-identity evidence.

The current P4-L contract defines the artifact and a future non-training
validation contract but does not itself authorize a validator, builder, or
materialization. P4-X/P4-Y are trainer-rebind context and do not replace a
future provisioning-validation authority.

## 3. Creation-state audit

Before this file was written, the following read-only checks were performed:

- `git rev-parse HEAD` returned
  `a1b614d0e659d2b34889cb55aef94e1824df2fd1`.
- `git diff --quiet` returned success: tracked worktree clean.
- `git diff --cached --quiet` returned success: index clean.
- The candidate path was absent.
- The canonical P4-L destination directory was absent.
- Pre-existing untracked review material and
  `reports/stage180a_pass2_annotations_completed.csv` were observed and were
  not modified.

Git emitted permission warnings while scanning unrelated local pytest/cache
directories. Those warnings are environmental and were not treated as a
cleanliness bypass; the direct tracked and index diff checks succeeded.

The two imported audit directories were attempted read-only and were not
accessible in this execution environment. No imported file was modified. The
controller-supplied evidence is therefore recorded as supplied evidence, not
as independently file-read evidence by this task.

## 4. Bound source identities

The future attempt MUST bind these exact Kaggle paths, with no discovery,
wildcard, guessed path, alternate download, reconstruction, or builder:

```text
/kaggle/input/datasets/terryterry9/proside/p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl
/kaggle/input/datasets/terryterry9/proside/p3w6f2_p4l_current_lineage_effective_integrity_sidecar_provenance.json
```

Expected identities:

- sidecar filename:
  `p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl`
- sidecar physical SHA256:
  `2b8cffdf71d68a8abeb3b6eb3534eeb664bd012483bcebd9716c7a6645a487f1`
- sidecar semantic SHA256 under the frozen P4-L algorithm:
  `0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08`
- sidecar row count: `3600`
- provenance filename:
  `p3w6f2_p4l_current_lineage_effective_integrity_sidecar_provenance.json`
- provenance physical SHA256:
  `9d248df09ae8ba471966c468a1e06278ad046908cfe53da623ecc95d8da4cdf2`

Both sources must be distinct regular non-symlink files, have exactly these
basenames, resolve outside the destination, and remain byte-unchanged.

## 5. Exact canonical destination

The future attempt MUST publish exactly this absent directory:

```text
reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458/
```

It must contain exactly these two entries and no others:

```text
p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl
p3w6f2_p4l_current_lineage_effective_integrity_sidecar_provenance.json
```

The directory, both child paths, and every `.p4l-staging-*` sibling must be
absent before the attempt. Existing destination or staging material is a
fail-closed blocker; nothing may be overwritten, moved aside, backed up, or
deleted by this authority. A failed provisioning attempt may leave one
attempt-created `.p4l-staging-*` directory containing zero, one, or two copied
candidate files. That retained staging directory is failure evidence, NOT
canonical P4-L, and its existence does not establish provisioning PASS, P4-L
PASS, or any scientific result. Its removal or disposition requires a
separately authorized bounded recovery/inspection action; this authority does
not authorize cleanup.

## 6. Required future preflight and mutation semantics

The future command must fail closed unless all of the following hold:

- HEAD is exactly the current commit above, before and after the attempt.
- Tracked worktree and index are clean.
- `CUDA_VISIBLE_DEVICES` is exactly the empty string; no GPU is used.
- Destination directory and both destination children are absent.
- No `.p4l-staging-*` sibling exists.
- Both exact source paths exist and are regular non-symlink files.
- Source names, distinctness, and source/destination separation are exact.
- All supplied source physical and semantic identities and row count match.

Copying must be raw byte copying only. It must not parse and reserialize JSON,
normalize line endings, reconstruct records, manually edit bytes, invoke a
builder or P4-Q, or run training/evaluation. Same-parent attempt-owned
staging is required. The final directory publication must be an atomic,
no-overwrite operation; if the platform cannot provide that operation, the
attempt must fail closed without publication.

The command must not run `git reset`, `clean`, `pull`, `merge`, `rebase`,
`cherry-pick`, `switch`, or `checkout`, and must not modify trainer, dataset,
manifest, index, branch, refs, or checkpoints.

## 7. Complete exact future CPU-only Kaggle provisioning command

Run this only after this candidate is independently verified and frozen, from
the repository root on Kaggle. This block is specified here and is **not run**
by this task.

```bash
CUDA_VISIBLE_DEVICES="" python - <<'PY'
import ctypes
import errno
import filecmp
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

EXPECTED_HEAD = "a1b614d0e659d2b34889cb55aef94e1824df2fd1"
SIDEcar_SHA = "2b8cffdf71d68a8abeb3b6eb3534eeb664bd012483bcebd9716c7a6645a487f1"
SIDEcar_SEMANTIC_SHA = "0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08"
PROVENANCE_SHA = "9d248df09ae8ba471966c468a1e06278ad046908cfe53da623ecc95d8da4cdf2"
SIDEcar_NAME = "p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl"
PROVENANCE_NAME = "p3w6f2_p4l_current_lineage_effective_integrity_sidecar_provenance.json"
SIDEcar_SOURCE = Path("/kaggle/input/datasets/terryterry9/proside") / SIDEcar_NAME
PROVENANCE_SOURCE = Path("/kaggle/input/datasets/terryterry9/proside") / PROVENANCE_NAME
OUTPUT_RELATIVE = Path("reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458")
OUTPUT = None
SIDEcar_DEST = None
PROVENANCE_DEST = None
staging = None
staging_created_by_this_attempt = False
staging_published = False

def fail(code):
    message = "P4L_PROVISION_BLOCKED:" + code
    if staging_created_by_this_attempt and not staging_published:
        message += ":STAGING_RETAINED_FOR_INSPECTION=" + os.fspath(staging)
    elif staging_published:
        message += ":CANONICAL_PROVISIONED_VALIDATION_FAILED=" + os.fspath(OUTPUT)
    raise SystemExit(message)

def require(condition, code):
    if not condition:
        fail(code)

def git(*args):
    result = subprocess.run(["git", *args], check=False, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    require(result.returncode == 0, "GIT_COMMAND_FAILED_" + "_".join(args))
    return result.stdout

def physical_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def semantic_sha(path):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            require(line.endswith("\n"), "SIDECAR_LINE_MISSING_FINAL_LF_" + str(line_number))
            value = json.loads(line)
            require(isinstance(value, dict), "SIDECAR_ROW_NOT_OBJECT_" + str(line_number))
            value.pop("created_at", None)
            rows.append(value)
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), len(rows)

def regular_non_symlink(path, name, code):
    require(path.name == name, code + "_NAME")
    require(path.exists() and not path.is_symlink() and path.is_file(), code + "_NOT_REGULAR_FILE")
    return path.resolve(strict=True)

def lexists(path):
    return os.path.lexists(os.fsencode(path))

def require_absent(path, code):
    require(not lexists(path), code)

def require_no_staging(parent, except_path=None):
    require(not any(p.name.startswith(".p4l-staging-") and p != except_path for p in parent.iterdir()), "STAGING_SIBLING_PRESENT")

def tracked_and_index_clean():
    require(git("status", "--short", "--untracked-files=no") == "", "TRACKED_WORKTREE_DIRTY")
    require(git("diff", "--cached", "--name-status") == "", "INDEX_DIRTY")

def all_untracked():
    output = git("status", "--porcelain=v1", "--untracked-files=all", "-z")
    paths = set()
    for entry in output.split("\0"):
        if not entry:
            continue
        require(entry.startswith("?? "), "UNEXPECTED_GIT_STATUS_ENTRY")
        relative = PurePosixPath(entry[3:])
        require(not relative.is_absolute() and ".." not in relative.parts,
                "GIT_STATUS_PATH_NOT_REPOSITORY_RELATIVE")
        paths.add(relative.as_posix())
    return paths

def rename_noreplace(source, destination):
    require(sys.platform == "linux", "ATOMIC_NOREPLACE_REQUIRES_LINUX")
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = libc.renameat2
    except (AttributeError, OSError):
        fail("ATOMIC_NOREPLACE_SYMBOL_UNAVAILABLE")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p,
                          ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    AT_FDCWD = -100
    RENAME_NOREPLACE = 1
    result = renameat2(AT_FDCWD, os.fsencode(source), AT_FDCWD,
                       os.fsencode(destination), RENAME_NOREPLACE)
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            fail("ATOMIC_NOREPLACE_DESTINATION_EXISTS")
        fail("ATOMIC_NOREPLACE_FAILED_ERRNO_" + str(error_number))

def establish_repo_root():
    raw_root = git("rev-parse", "--show-toplevel").strip()
    require(raw_root and "\n" not in raw_root and "\r" not in raw_root,
            "REPOSITORY_ROOT_INVALID")
    root = Path(raw_root)
    require(root.is_absolute(), "REPOSITORY_ROOT_NOT_ABSOLUTE")
    try:
        root = root.resolve(strict=True)
        current = Path.cwd().resolve(strict=True)
    except OSError:
        fail("REPOSITORY_ROOT_RESOLUTION_FAILED")
    require(root.is_dir() and not root.is_symlink(), "REPOSITORY_ROOT_NOT_NORMAL_DIR")
    require(current == root, "CWD_NOT_EXACT_REPOSITORY_ROOT")
    root_stat = os.lstat(root)
    require(stat.S_ISDIR(root_stat.st_mode) and not stat.S_ISLNK(root_stat.st_mode),
            "REPOSITORY_ROOT_ANCESTRY_INVALID")
    return root

def require_repo_relative(relative, code):
    require(not relative.is_absolute() and ".." not in relative.parts,
            code + "_ABSOLUTE_OR_PARENT_PATH")
    return relative

def verified_repo_directory(root, relative, code):
    relative = require_repo_relative(relative, code)
    current = root
    for part in relative.parts:
        current = current / part
        require(os.path.lexists(os.fsencode(current)), code + "_ANCESTOR_MISSING")
        entry_stat = os.lstat(current)
        require(stat.S_ISDIR(entry_stat.st_mode) and not stat.S_ISLNK(entry_stat.st_mode),
                code + "_ANCESTOR_NOT_NORMAL_DIR")
    try:
        resolved = current.resolve(strict=True)
    except OSError:
        fail(code + "_ANCESTOR_RESOLUTION_FAILED")
    require(resolved == current, code + "_ANCESTOR_REDIRECTED")
    require(os.path.commonpath((os.fspath(root), os.fspath(resolved))) == os.fspath(root),
            code + "_OUTSIDE_REPOSITORY")
    return current

require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "GPU_NOT_OFF")
require(git("rev-parse", "HEAD").strip() == EXPECTED_HEAD, "HEAD_MISMATCH")
tracked_and_index_clean()
before_untracked = all_untracked()
REPO_ROOT = establish_repo_root()
OUTPUT_RELATIVE = require_repo_relative(OUTPUT_RELATIVE, "OUTPUT")
OUTPUT = REPO_ROOT / OUTPUT_RELATIVE
require(OUTPUT == REPO_ROOT / Path("reports") / OUTPUT_RELATIVE.relative_to("reports"),
        "OUTPUT_PATH_NOT_CANONICAL")
SIDEcar_DEST = OUTPUT / SIDEcar_NAME
PROVENANCE_DEST = OUTPUT / PROVENANCE_NAME
verified_repo_directory(REPO_ROOT, OUTPUT_RELATIVE.parent, "OUTPUT_PARENT")
try:
    require(OUTPUT.parent.resolve(strict=True) == OUTPUT.parent, "OUTPUT_PARENT_REDIRECTED")
except OSError:
    fail("OUTPUT_PARENT_RESOLUTION_FAILED")
require(os.path.commonpath((os.fspath(REPO_ROOT), os.fspath(OUTPUT.parent.resolve(strict=True)))) == os.fspath(REPO_ROOT),
        "OUTPUT_PARENT_OUTSIDE_REPOSITORY")
require_absent(OUTPUT, "DESTINATION_PRESENT")
require_absent(SIDEcar_DEST, "SIDECAR_DESTINATION_PRESENT")
require_absent(PROVENANCE_DEST, "PROVENANCE_DESTINATION_PRESENT")
require_no_staging(OUTPUT.parent)

sidecar = regular_non_symlink(SIDEcar_SOURCE, SIDEcar_NAME, "SIDECAR_SOURCE")
provenance = regular_non_symlink(PROVENANCE_SOURCE, PROVENANCE_NAME, "PROVENANCE_SOURCE")
require(sidecar != provenance, "SOURCES_IDENTICAL")
output_resolved = OUTPUT.resolve(strict=False)
require(output_resolved not in sidecar.parents and output_resolved not in provenance.parents, "SOURCE_INSIDE_DESTINATION")
require(sidecar != SIDEcar_DEST.resolve(strict=False) and provenance != PROVENANCE_DEST.resolve(strict=False), "SOURCE_EQUALS_DESTINATION")
sidecar_sha_before = physical_sha(sidecar)
provenance_sha_before = physical_sha(provenance)
require(sidecar_sha_before == SIDEcar_SHA, "SIDECAR_SOURCE_SHA_MISMATCH")
require(provenance_sha_before == PROVENANCE_SHA, "PROVENANCE_SOURCE_SHA_MISMATCH")
semantic, row_count = semantic_sha(sidecar)
require(semantic == SIDEcar_SEMANTIC_SHA, "SIDECAR_SOURCE_SEMANTIC_SHA_MISMATCH")
require(row_count == 3600, "SIDECAR_SOURCE_ROW_COUNT_MISMATCH")

CANONICAL_RELATIVE = OUTPUT_RELATIVE.as_posix()
AUTHORIZED_NEW_UNTRACKED = {
    CANONICAL_RELATIVE,
    CANONICAL_RELATIVE + "/" + SIDEcar_NAME,
    CANONICAL_RELATIVE + "/" + PROVENANCE_NAME,
}
CANONICAL_CHILDREN_RELATIVE = {
    CANONICAL_RELATIVE + "/" + SIDEcar_NAME,
    CANONICAL_RELATIVE + "/" + PROVENANCE_NAME,
}
try:
    try:
        staging = Path(tempfile.mkdtemp(prefix=".p4l-staging-", dir=os.fspath(OUTPUT.parent)))
    except (OSError, TypeError):
        fail("STAGING_CREATION_FAILED")
    staging_created_by_this_attempt = True
    require(staging.parent == OUTPUT.parent, "STAGING_PARENT_INVALID")
    require(staging.name.startswith(".p4l-staging-"), "STAGING_PREFIX_INVALID")
    staging_stat = os.lstat(staging)
    require(stat.S_ISDIR(staging_stat.st_mode) and not stat.S_ISLNK(staging_stat.st_mode),
            "STAGING_NOT_NORMAL_DIR")
    try:
        staging_resolved = staging.resolve(strict=True)
    except OSError:
        fail("STAGING_RESOLUTION_FAILED")
    require(staging_resolved == staging and staging_resolved.parent == OUTPUT.parent,
            "STAGING_REDIRECTED")
    require(os.path.commonpath((os.fspath(OUTPUT.parent), os.fspath(staging_resolved))) == os.fspath(OUTPUT.parent),
            "STAGING_OUTSIDE_OUTPUT_PARENT")
    staging_sidecar = staging / SIDEcar_NAME
    staging_provenance = staging / PROVENANCE_NAME
    shutil.copyfile(sidecar, staging_sidecar)
    shutil.copyfile(provenance, staging_provenance)
    for path in (staging_sidecar, staging_provenance):
        require(path.is_file() and not path.is_symlink(), "STAGING_CHILD_INVALID")
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    require(physical_sha(staging_sidecar) == SIDEcar_SHA, "STAGING_SIDECAR_SHA_MISMATCH")
    require(physical_sha(staging_provenance) == PROVENANCE_SHA, "STAGING_PROVENANCE_SHA_MISMATCH")
    require_absent(OUTPUT, "DESTINATION_APPEARED_BEFORE_PUBLISH")
    require_no_staging(OUTPUT.parent, staging)
    rename_noreplace(staging, OUTPUT)
    staging_published = True
except SystemExit:
    raise
except Exception as error:
    fail("STAGING_ATTEMPT_FAILED_" + type(error).__name__)

require(git("rev-parse", "HEAD").strip() == EXPECTED_HEAD, "POST_HEAD_MISMATCH")
tracked_and_index_clean()
require(OUTPUT.is_dir() and not OUTPUT.is_symlink(), "DESTINATION_NOT_NORMAL_DIR")
require(sorted(p.name for p in OUTPUT.iterdir()) == sorted([SIDEcar_NAME, PROVENANCE_NAME]), "DESTINATION_ENTRY_SET_MISMATCH")
require(SIDEcar_DEST.is_file() and not SIDEcar_DEST.is_symlink(), "DESTINATION_SIDECAR_INVALID")
require(PROVENANCE_DEST.is_file() and not PROVENANCE_DEST.is_symlink(), "DESTINATION_PROVENANCE_INVALID")
require(physical_sha(SIDEcar_DEST) == SIDEcar_SHA, "DESTINATION_SIDECAR_SHA_MISMATCH")
require(physical_sha(PROVENANCE_DEST) == PROVENANCE_SHA, "DESTINATION_PROVENANCE_SHA_MISMATCH")
destination_semantic, destination_rows = semantic_sha(SIDEcar_DEST)
require(destination_semantic == SIDEcar_SEMANTIC_SHA, "DESTINATION_SIDECAR_SEMANTIC_SHA_MISMATCH")
require(destination_rows == 3600, "DESTINATION_SIDECAR_ROW_COUNT_MISMATCH")
require(filecmp.cmp(sidecar, SIDEcar_DEST, shallow=False), "SIDECAR_BYTES_DIFFER")
require(filecmp.cmp(provenance, PROVENANCE_DEST, shallow=False), "PROVENANCE_BYTES_DIFFER")
require(physical_sha(sidecar) == sidecar_sha_before and physical_sha(provenance) == provenance_sha_before, "SOURCE_BYTES_CHANGED")
require_no_staging(OUTPUT.parent)
after_untracked = all_untracked()
newly_untracked = after_untracked - before_untracked
require(newly_untracked in (
    {CANONICAL_RELATIVE},
    CANONICAL_CHILDREN_RELATIVE,
    AUTHORIZED_NEW_UNTRACKED,
), "UNEXPECTED_UNTRACKED_MUTATION")
require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "POST_GPU_NOT_OFF")
print("P3W6F2P4L_CURRENT_HEAD_EXACT_BYTE_PROVISION_PASS")
PY
```

The shell wrapper sets the Kaggle process environment to CPU-only and the
Python preflight still requires the resulting value to be exactly empty. The
command uses no JSON parsing for copying; JSON parsing is used only for the
frozen semantic-hash and row-count checks before and after raw copying.

If any failure occurs after unique staging creation and before publication, the
command stops and reports the exact staging path as intentionally retained for
inspection or separately authorized recovery. It performs no deletion or
rollback of that path. The retained directory may contain zero, one, or two
copied candidate files and is failure evidence only; it is not canonical P4-L
and does not establish provisioning PASS, P4-L PASS, or any scientific result.
The preflight rejects any pre-existing `.p4l-staging-*` sibling, so a retry
requires a separately authorized disposition of retained evidence.

Successful `renameat2(RENAME_NOREPLACE)` moves the unique staging directory to
the exact canonical destination. The old staging pathname must then be absent,
and post-publication validation requires that no staging sibling remains. No
cleanup operation is performed or authorized after success. If a
post-publication validation fails, the command reports the canonical directory
as provisioned-but-validation-failed; it does not delete, roll back, or recreate
either the canonical directory or the old staging pathname.

## 8. Post-provision interpretation

A provisioning PASS establishes only exact-byte canonical availability and
source/destination identity. It does not establish:

- formal A0 authority or A0 execution;
- training or evaluation success;
- scientific effectiveness;
- any A1, A2, or A3 result;
- trainer correctness beyond already-existing separate evidence;
- promotion or training admission.

The command must additionally demonstrate after publication:

- unchanged exact HEAD;
- clean tracked worktree and index, with the newly created untracked canonical
  artifact excluded from the tracked-clean interpretation;
- normal non-symlink destination directory with exactly two normal files;
- exact physical and semantic sidecar identities and row count;
- exact provenance identity;
- exact source/destination byte equality;
- unchanged source hashes;
- no staging sibling; and
- GPU still off.

## 9. Exact next validation boundary

Historical P4-V must not be blindly invoked. Its historical P4-W handoff was
bound to a different HEAD-transition workflow and is not the current execution
authority.

After a successful future provisioning attempt, the next required boundary is
a separately created, independently verified, and frozen **current-HEAD
P4-L provisioning-result validation authority**. That authority must run a
read-only validator against the newly provisioned directory and the two exact
source paths, verify every postcondition above plus the P4-L provenance/schema
contract, and record the provisioning command identity, current HEAD, source
identities, destination identities, and result hashes. It must not run a
builder, training, evaluation, or A0.

Only after that validation authority is independently frozen does the current
operational chain support creation and freeze of a separate P3-W7-A0 execution
authority, as stated by `reports/PRE_URP_HANDOFF.md`. The existing P4-L
specification says future validators are not authorized by P4-L itself; the
required validator authority therefore remains to be created rather than
inferred from P4-V.

## 10. Candidate stop conditions

This candidate is BLOCKED if, at future freeze or execution, any of the
following is true:

- HEAD differs from the exact current HEAD above;
- tracked worktree or index is dirty;
- the candidate already exists at creation time;
- an applicable newer authority conflicts with this candidate;
- imported evidence contradicts the controller-supplied identities;
- either exact source is missing, symlinked, non-regular, misnamed, relocated,
  or hash-inconsistent;
- the destination or any child already exists;
- any staging sibling exists;
- atomic no-overwrite publication is unavailable;
- source and destination separation cannot be proven;
- any mutation beyond the exact untracked canonical directory would be needed;
- the command would require HEAD movement, builder/P4-Q execution, parsing for
  copy, reconstruction, manual edit, training, evaluation, or GPU; or
- the required postcondition cannot be independently verified.

## 11. Provenance and non-claims

Controller-supplied discovery evidence:

- run: `p4l-external-source-discovery-a1b614d-retry2`
- HEAD: `a1b614d0e659d2b34889cb55aef94e1824df2fd1`
- command SHA256:
  `d0b42eb1abfda518a1b2997fdb0e3f0c2e6a68ef3e92386ddb8eadeff0b3a5bc`
- ZIP SHA256:
  `c5fb6d718eef8a8a717e7017cf01aaf4994e154326efef407395fec1a481b281`
- run-log SHA256:
  `3f1572ea37aeb8d71d88c083507bcd2ae4d6c51b0b188f5a3b8fbb8cb47089ca`
- run-meta SHA256:
  `77ba04ab1c8a217826c94a2b53b3474b2d55ccd7f68731cde130e5d6186aee57`

Controller-supplied identity-verification evidence:

- run: `p4l-source-identity-verification-a1b614d`
- HEAD: `a1b614d0e659d2b34889cb55aef94e1824df2fd1`
- command SHA256:
  `d450ecf05b831c39055114b2ac4389df45ae62bf2347101024d77ca24275a2c9`
- ZIP SHA256:
  `9ce7dbcbb294474a0e36ec1b85c687467911a7cbea977fae328c50149ee8b2df`
- run-log SHA256:
  `25f42c507bd31c5f85dba08e1f513f80c4cc29a4b3aa8e511d4f139652884bef`
- run-meta SHA256:
  `eee53c3186c0f9dab4075ef8d07e0dff009eecae97ec474eea50b85a70f9358a`

Imported audit directories were not independently readable in this task’s
environment. The source paths and hashes above are accepted here only as
controller-supplied bindings; independent verification remains required before
freeze/execution.

This candidate does not claim provisioning PASS, P4-L freeze, formal A0
authority, A0 execution, training/evaluation success, scientific effectiveness,
or any A1/A2/A3 result.
