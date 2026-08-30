# P3-W6-F2-P4-L Current Provisioning-Result Validation Execution-Authority Candidate

Authority/version:

`P3W6F2P4L_CURRENT_PROVISIONING_RESULT_VALIDATION_EXECUTION_AUTHORITY_CANDIDATE_V1`

## 1. Disposition and boundary

This document is a candidate execution authority only. Candidate creation does
not execute Kaggle, validation, provisioning, copying, regeneration, builder
code, P4-Q, P4-W, training, evaluation, A0, staging, cleanup, commit, push, or
GPU work. It becomes executable only after independent verification and
immutable freeze.

If frozen, it authorizes exactly one future CPU-only, strictly read-only P4-L
provisioning-result validation run. The run validates the current canonical
P4-L sidecar pair against the exact Kaggle input sources and the runtime HEAD
blob bytes. It must perform no creation, deletion, copy, staging, cleanup,
chmod, touch, temp-file creation, serialization write, Git mutation, checkpoint
mutation, dataset mutation, or artifact mutation.

The allowed creation delta for this task is exactly this one new untracked
candidate authority file:

```text
reports/reason_router_p2_p3w6f2_p4l_current_provisioning_result_validation_execution_authority_spec_candidate.md
```

## 2. Authority chain consumed

This candidate consumes, in precedence order:

1. The current controller instruction for P3-W6-F2-P4-L provisioning-result
   validation authority creation.
2. Repository HEAD `a93c291b79974f4aaa0b51f4578c807e8a5d6301`.
3. Successful provisioning evidence for run
   `p3w6f2-p4l-exact-byte-provision-retry1-23233cc`.
4. `reports/reason_router_p2_p3w6f2_p4l_current_exact_byte_provisioning_environment_binding_retry1_execution_authority_spec_candidate.md`.
5. `reports/reason_router_p2_p3w6f2_p4l_current_exact_byte_provisioning_corrected_execution_authority_spec_candidate.md`.
6. The canonical P4-L sidecar pair and provenance currently frozen at
   `a93c291b79974f4aaa0b51f4578c807e8a5d6301`.
7. `reports/PRE_URP_HANDOFF.md`, `docs/RESEARCH_OPERATIONS.md`, and
   `AGENTS.md`.

The older handoff blocker that stated the P4-L bytes were absent is superseded
only by the controller-supplied successful provisioning evidence and the
current HEAD tree containing the canonical pair. This supersession authorizes
only a post-provision read-only validator authority. It does not authorize A0,
training, evaluation, GPU use, scientific effectiveness claims, promotion, or
A1/A2/A3.

## 3. Successful provisioning evidence bound

The validation run must bind the following prior provisioning result exactly:

- run: `p3w6f2-p4l-exact-byte-provision-retry1-23233cc`
- execution HEAD: `23233cce48262979d2c39444724668cc86fadcc7`
- command SHA256:
  `bd65af1ff45c8ec36958ca51012b2ab1351b3e47c2c02ebd178a5028bb64cf2e`
- exit: `0`
- success token:
  `P3W6F2P4L_CURRENT_EXACT_BYTE_PROVISION_FREEZE_BOUND_PASS`
- run log SHA256:
  `b43f985116350b93a082eae945dff55198178721a8741a1600aed46a15d803c8`
- run meta SHA256:
  `be1c469f55701fcd7e59509334e39a3af89904963286ee3ae9ec9d8033beeadb`
- import ZIP SHA256:
  `2dff07af8bdc0d15497eda6bc692e40229da5058fe89ac9991441ccb41b1b7c5`
- import: `PASS`
- validated: `2`
- copied: `2`

These are provisioning-result inputs to this validator. They are not training,
evaluation, A0, A1, A2, A3, promotion, or scientific-effectiveness evidence.

## 4. Bound source and canonical identities

The future validator must bind these exact Kaggle sources, with no discovery,
wildcard, alternate path, download, reconstruction, copy, normalization, or
builder:

```text
/kaggle/input/datasets/terryterry9/proside/p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl
/kaggle/input/datasets/terryterry9/proside/p3w6f2_p4l_current_lineage_effective_integrity_sidecar_provenance.json
```

The future validator must bind these exact canonical repository paths:

```text
reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458/p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl
reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458/p3w6f2_p4l_current_lineage_effective_integrity_sidecar_provenance.json
```

Expected sidecar physical SHA256:

`2b8cffdf71d68a8abeb3b6eb3534eeb664bd012483bcebd9716c7a6645a487f1`

Expected provenance physical SHA256:

`9d248df09ae8ba471966c468a1e06278ad046908cfe53da623ecc95d8da4cdf2`

Expected sidecar semantic SHA256:

`0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08`

Expected sidecar row count:

`3600`

The canonical directory must be a normal non-symlink directory containing
exactly the two expected children. All four source/canonical files must be
regular non-symlink files. Source and canonical sidecar bytes must be exactly
equal. Source and canonical provenance bytes must be exactly equal. Canonical
working-tree bytes must be exactly equal to the bytes stored at runtime HEAD
for both canonical paths. Source and canonical bytes must remain unchanged
across validation.

## 5. Complete exact future CPU-only read-only Kaggle validation command

Run this only after this candidate is independently verified and frozen, from
the repository root on Kaggle/Linux. This block is specified here and is **not
run** by this task.

```bash
P4L_VALIDATION_AUTHORITY_FREEZE="$(git rev-parse HEAD)" CUDA_VISIBLE_DEVICES="" python - <<'PY'
import binascii
import filecmp
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

SUCCESS_TOKEN = "P3W6F2P4L_CURRENT_PROVISIONING_RESULT_VALIDATION_PASS"
PROVISIONING_RUN = "p3w6f2-p4l-exact-byte-provision-retry1-23233cc"
PROVISIONING_HEAD = "23233cce48262979d2c39444724668cc86fadcc7"
PROVISIONING_COMMAND_SHA256 = "bd65af1ff45c8ec36958ca51012b2ab1351b3e47c2c02ebd178a5028bb64cf2e"
PROVISIONING_RUN_LOG_SHA256 = "b43f985116350b93a082eae945dff55198178721a8741a1600aed46a15d803c8"
PROVISIONING_RUN_META_SHA256 = "be1c469f55701fcd7e59509334e39a3af89904963286ee3ae9ec9d8033beeadb"
PROVISIONING_IMPORT_ZIP_SHA256 = "2dff07af8bdc0d15497eda6bc692e40229da5058fe89ac9991441ccb41b1b7c5"
VALIDATION_AUTHORITY_RELATIVE = "reports/reason_router_p2_p3w6f2_p4l_current_provisioning_result_validation_execution_authority_spec_candidate.md"
RETRY1_AUTHORITY_RELATIVE = "reports/reason_router_p2_p3w6f2_p4l_current_exact_byte_provisioning_environment_binding_retry1_execution_authority_spec_candidate.md"
CORRECTED_AUTHORITY_RELATIVE = "reports/reason_router_p2_p3w6f2_p4l_current_exact_byte_provisioning_corrected_execution_authority_spec_candidate.md"
CANONICAL_DIR_RELATIVE = "reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458"
SIDECAR_NAME = "p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl"
PROVENANCE_NAME = "p3w6f2_p4l_current_lineage_effective_integrity_sidecar_provenance.json"
SIDECAR_SOURCE = Path("/kaggle/input/datasets/terryterry9/proside") / SIDECAR_NAME
PROVENANCE_SOURCE = Path("/kaggle/input/datasets/terryterry9/proside") / PROVENANCE_NAME
SIDECAR_RELATIVE = CANONICAL_DIR_RELATIVE + "/" + SIDECAR_NAME
PROVENANCE_RELATIVE = CANONICAL_DIR_RELATIVE + "/" + PROVENANCE_NAME
SIDECAR_SHA256 = "2b8cffdf71d68a8abeb3b6eb3534eeb664bd012483bcebd9716c7a6645a487f1"
PROVENANCE_SHA256 = "9d248df09ae8ba471966c468a1e06278ad046908cfe53da623ecc95d8da4cdf2"
SIDECAR_SEMANTIC_SHA256 = "0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08"
EXPECTED_ROW_COUNT = 3600
FREEZE = os.environ.get("P4L_VALIDATION_AUTHORITY_FREEZE")

def fail(code):
    raise SystemExit("P4L_PROVISIONING_RESULT_VALIDATION_BLOCKED:" + code)

def require(condition, code):
    if not condition:
        fail(code)

def git_text(*args):
    result = subprocess.run(["git", *args], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    require(result.returncode == 0, "GIT_COMMAND_FAILED_" + "_".join(args))
    return result.stdout.decode("utf-8")

def git_bytes(*args):
    result = subprocess.run(["git", *args], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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

def require_regular_non_symlink(path, name, code):
    require(path.name == name, code + "_NAME_MISMATCH")
    try:
        entry_stat = os.lstat(path)
    except OSError:
        fail(code + "_LSTAT_FAILED")
    require(stat.S_ISREG(entry_stat.st_mode) and not stat.S_ISLNK(entry_stat.st_mode), code + "_NOT_REGULAR_FILE")
    require(path.exists() and path.is_file() and not path.is_symlink(), code + "_PATH_CHECK_FAILED")
    return path.resolve(strict=True)

def require_normal_directory(path, code):
    try:
        entry_stat = os.lstat(path)
    except OSError:
        fail(code + "_LSTAT_FAILED")
    require(stat.S_ISDIR(entry_stat.st_mode) and not stat.S_ISLNK(entry_stat.st_mode), code + "_NOT_NORMAL_DIR")
    require(path.exists() and path.is_dir() and not path.is_symlink(), code + "_PATH_CHECK_FAILED")
    return path.resolve(strict=True)

def require_tree_path(relative, code):
    output = git_text("ls-tree", "-r", "--name-only", FREEZE, "--", relative).splitlines()
    require(output == [relative], code)

def require_blob_equal(relative, working_path, code):
    blob = git_bytes("show", FREEZE + ":" + relative)
    working = working_path.read_bytes()
    require(hashlib.sha256(blob).hexdigest() == hashlib.sha256(working).hexdigest(), code + "_SHA_MISMATCH")
    require(blob == working, code + "_BYTES_DIFFER")

require(sys.platform == "linux", "LINUX_ONLY")
require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "GPU_NOT_OFF")
require(FREEZE is not None and re.fullmatch(r"[0-9a-f]{40}", FREEZE) is not None, "FREEZE_NOT_LOWERCASE_40_HEX")
require(git_text("cat-file", "-t", FREEZE).strip() == "commit", "FREEZE_NOT_COMMIT")
runtime_head = git_text("rev-parse", "HEAD").strip()
require(runtime_head == FREEZE, "HEAD_MISMATCH_VALIDATION_FREEZE")
require(git_text("status", "--short", "--untracked-files=no") == "", "TRACKED_WORKTREE_DIRTY_BEFORE")
require(git_text("diff", "--cached", "--name-status") == "", "INDEX_DIRTY_BEFORE")
status_before = git_bytes("status", "--porcelain=v1", "--untracked-files=all", "-z")

for relative, code in (
    (VALIDATION_AUTHORITY_RELATIVE, "FREEZE_MISSING_VALIDATION_AUTHORITY"),
    (RETRY1_AUTHORITY_RELATIVE, "FREEZE_MISSING_RETRY1_AUTHORITY"),
    (CORRECTED_AUTHORITY_RELATIVE, "FREEZE_MISSING_CORRECTED_AUTHORITY"),
    (SIDECAR_RELATIVE, "FREEZE_MISSING_CANONICAL_SIDECAR"),
    (PROVENANCE_RELATIVE, "FREEZE_MISSING_CANONICAL_PROVENANCE"),
):
    require_tree_path(relative, code)

repo_root_raw = git_text("rev-parse", "--show-toplevel").strip()
require(repo_root_raw and "\n" not in repo_root_raw and "\r" not in repo_root_raw, "REPO_ROOT_INVALID")
repo_root = Path(repo_root_raw).resolve(strict=True)
require(Path.cwd().resolve(strict=True) == repo_root, "CWD_NOT_REPO_ROOT")
canonical_dir = repo_root / CANONICAL_DIR_RELATIVE
canonical_sidecar = repo_root / SIDECAR_RELATIVE
canonical_provenance = repo_root / PROVENANCE_RELATIVE

source_sidecar = require_regular_non_symlink(SIDECAR_SOURCE, SIDECAR_NAME, "SIDECAR_SOURCE")
source_provenance = require_regular_non_symlink(PROVENANCE_SOURCE, PROVENANCE_NAME, "PROVENANCE_SOURCE")
canonical_dir_resolved = require_normal_directory(canonical_dir, "CANONICAL_DIR")
canonical_sidecar_resolved = require_regular_non_symlink(canonical_sidecar, SIDECAR_NAME, "CANONICAL_SIDECAR")
canonical_provenance_resolved = require_regular_non_symlink(canonical_provenance, PROVENANCE_NAME, "CANONICAL_PROVENANCE")
require(canonical_sidecar_resolved.parent == canonical_dir_resolved, "CANONICAL_SIDECAR_PARENT_MISMATCH")
require(canonical_provenance_resolved.parent == canonical_dir_resolved, "CANONICAL_PROVENANCE_PARENT_MISMATCH")
require(sorted(path.name for path in canonical_dir.iterdir()) == sorted([SIDECAR_NAME, PROVENANCE_NAME]), "CANONICAL_DIR_CHILDREN_MISMATCH")

source_sidecar_before = physical_sha(source_sidecar)
source_provenance_before = physical_sha(source_provenance)
canonical_sidecar_before = physical_sha(canonical_sidecar_resolved)
canonical_provenance_before = physical_sha(canonical_provenance_resolved)
require(source_sidecar_before == SIDECAR_SHA256, "SIDECAR_SOURCE_SHA_MISMATCH")
require(source_provenance_before == PROVENANCE_SHA256, "PROVENANCE_SOURCE_SHA_MISMATCH")
require(canonical_sidecar_before == SIDECAR_SHA256, "CANONICAL_SIDECAR_SHA_MISMATCH")
require(canonical_provenance_before == PROVENANCE_SHA256, "CANONICAL_PROVENANCE_SHA_MISMATCH")

source_semantic, source_rows = semantic_sha(source_sidecar)
canonical_semantic, canonical_rows = semantic_sha(canonical_sidecar_resolved)
require(source_semantic == SIDECAR_SEMANTIC_SHA256, "SOURCE_SEMANTIC_SHA_MISMATCH")
require(canonical_semantic == SIDECAR_SEMANTIC_SHA256, "CANONICAL_SEMANTIC_SHA_MISMATCH")
require(source_rows == EXPECTED_ROW_COUNT, "SOURCE_ROW_COUNT_MISMATCH")
require(canonical_rows == EXPECTED_ROW_COUNT, "CANONICAL_ROW_COUNT_MISMATCH")
require(filecmp.cmp(source_sidecar, canonical_sidecar_resolved, shallow=False), "SIDECAR_SOURCE_CANONICAL_BYTES_DIFFER")
require(filecmp.cmp(source_provenance, canonical_provenance_resolved, shallow=False), "PROVENANCE_SOURCE_CANONICAL_BYTES_DIFFER")
require_blob_equal(SIDECAR_RELATIVE, canonical_sidecar_resolved, "HEAD_BLOB_SIDECAR")
require_blob_equal(PROVENANCE_RELATIVE, canonical_provenance_resolved, "HEAD_BLOB_PROVENANCE")

require(physical_sha(source_sidecar) == source_sidecar_before, "SIDECAR_SOURCE_BYTES_CHANGED")
require(physical_sha(source_provenance) == source_provenance_before, "PROVENANCE_SOURCE_BYTES_CHANGED")
require(physical_sha(canonical_sidecar_resolved) == canonical_sidecar_before, "CANONICAL_SIDECAR_BYTES_CHANGED")
require(physical_sha(canonical_provenance_resolved) == canonical_provenance_before, "CANONICAL_PROVENANCE_BYTES_CHANGED")
require(git_text("rev-parse", "HEAD").strip() == FREEZE, "POST_HEAD_MISMATCH")
require(git_text("status", "--short", "--untracked-files=no") == "", "TRACKED_WORKTREE_DIRTY_AFTER")
require(git_text("diff", "--cached", "--name-status") == "", "INDEX_DIRTY_AFTER")
status_after = git_bytes("status", "--porcelain=v1", "--untracked-files=all", "-z")
require(status_after == status_before, "GIT_STATUS_CHANGED")
require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "POST_GPU_NOT_OFF")

print("P4L_VALIDATION_STATUS_BEFORE_HEX=" + binascii.hexlify(status_before).decode("ascii"))
print("P4L_VALIDATION_STATUS_AFTER_HEX=" + binascii.hexlify(status_after).decode("ascii"))
print("P4L_VALIDATION_STATUS_SHA256=" + hashlib.sha256(status_before).hexdigest())
print("P4L_VALIDATION_RUN=" + PROVISIONING_RUN)
print("P4L_VALIDATION_PROVISIONING_HEAD=" + PROVISIONING_HEAD)
print("P4L_VALIDATION_PROVISIONING_COMMAND_SHA256=" + PROVISIONING_COMMAND_SHA256)
print("P4L_VALIDATION_PROVISIONING_RUN_LOG_SHA256=" + PROVISIONING_RUN_LOG_SHA256)
print("P4L_VALIDATION_PROVISIONING_RUN_META_SHA256=" + PROVISIONING_RUN_META_SHA256)
print("P4L_VALIDATION_PROVISIONING_IMPORT_ZIP_SHA256=" + PROVISIONING_IMPORT_ZIP_SHA256)
print(SUCCESS_TOKEN)
PY
```

## 6. Exact command identity and read-only contract

The exact command above is UTF-8 text with LF line endings. Its SHA256,
computed over the fenced command payload only, is:

`01661746e878162d002a951b3a924a0a34b0421b19007631bf77545cbc7866dd`

The command-local validation freeze is derived only by:

```bash
P4L_VALIDATION_AUTHORITY_FREEZE="$(git rev-parse HEAD)"
```

The command sets `CUDA_VISIBLE_DEVICES` exactly to the empty string and checks
that it remains empty before and after validation. It requires Linux. It
requires runtime HEAD to equal the command-local validation freeze, requires
the freeze to be lowercase 40-hex, and requires it to resolve to a commit.

The command requires the freeze tree to contain this validation authority, the
retry1 authority, the corrected provisioning authority, and the exact canonical
sidecar/provenance pair. It records the full raw bytes of
`git status --porcelain=v1 --untracked-files=all -z` before and after as hex on
stdout and requires byte-for-byte equality. It also requires tracked worktree
and index cleanliness before and after.

The command contains no write-capable operation. In particular, it does not
call file open in write/append/update mode, `mkdir`, `copy`, `rename`,
`replace`, `unlink`, `remove`, `rmdir`, `chmod`, `touch`, `tempfile`, `fsync`,
`git add`, `git commit`, `git checkout`, `git switch`, `git reset`,
`git clean`, `git pull`, `git merge`, `git rebase`, builder code, P4-Q, P4-W,
training, evaluation, A0, GPU work, or any write of serialized data.

Expected success token:

`P3W6F2P4L_CURRENT_PROVISIONING_RESULT_VALIDATION_PASS`

## 7. Interpretation boundary

A PASS establishes only current P4-L exact-byte provisioning-result validity:
the supplied sources, canonical working bytes, and runtime HEAD blobs agree
under the exact physical, semantic, row-count, tree-binding, and status
preservation contracts above.

A PASS does not authorize or establish A0, training, evaluation, GPU use,
scientific effectiveness, promotion, A1, A2, or A3.

## 8. Candidate stop conditions

This candidate is BLOCKED if, at future freeze or execution, any of the
following is true:

- HEAD differs from the command-local validation freeze;
- the freeze is not lowercase 40-hex or does not resolve to a commit;
- this validation authority path is absent from the freeze tree;
- the retry1 authority path is absent from the freeze tree;
- either canonical file is absent from the freeze tree;
- tracked worktree or index is dirty before or after;
- the raw full `git status --porcelain=v1 --untracked-files=all -z` bytes
  differ after validation;
- any source or canonical file is missing, symlinked, non-regular, hash
  inconsistent, semantic-hash inconsistent, row-count inconsistent, or
  byte-unequal to its counterpart;
- the canonical directory is missing, symlinked, non-normal, or contains any
  child other than the expected two files;
- canonical working bytes differ from runtime HEAD blob bytes;
- any source or canonical bytes change during validation;
- CUDA visibility is not exactly empty before and after;
- Linux-only execution is not available; or
- validation would require any creation, deletion, copy, staging, cleanup,
  chmod, touch, temp file, serialization write, provisioning, regeneration,
  builder/P4-Q/P4-W, training, evaluation, A0, GPU, commit, push, or Git/index
  mutation.

## 9. Final creation result

This candidate creation authorizes and runs nothing. The candidate SHA256 and
exact command SHA256 are computed after creation and reported by the creating
task.

Required readiness token:

`P4L_PROVISIONING_RESULT_VALIDATION_AUTHORITY_READY`
