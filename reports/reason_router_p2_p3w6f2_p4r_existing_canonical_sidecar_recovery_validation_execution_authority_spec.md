# P3-W6-F2-P4-R Existing Canonical Sidecar Recovery Validation Execution Authority Specification

Authority/version:

`P3W6F2P4R_EXISTING_CANONICAL_SIDECAR_RECOVERY_VALIDATION_EXECUTION_AUTHORITY_V1`

This document is a candidate authority specification only. Candidate creation
does not authorize recovery validation execution, Kaggle, CPU execution, GPU
execution, artifact materialization, builder execution, trainer rebind, A0,
training, evaluation, staging, commit, push, ZIP creation, evidence creation,
canonical rewrite, canonical deletion, or retry of P4-Q Gate 1.

After independent static verification PASS and immutable P4-R freeze, this
authority may authorize exactly one CPU-only, read-only recovery validation of
the already published canonical P4-L sidecar/provenance bytes. The already
published canonical bytes are immutable recovery input. P4-R never authorizes
builder execution, `--materialize`, canonical rewrite, canonical deletion,
P4-Q Gate 1 retry, trainer rebind, A0, training, evaluation, or GPU.

## 1. HEAD/State Proof

Candidate creation requires:

- HEAD exactly `79ec5fa764de30eaa04fb6de4c2a8228edf1a63a`.
- Tracked worktree clean.
- Index clean.
- Candidate path absent before creation:
  `reports/reason_router_p2_p3w6f2_p4r_existing_canonical_sidecar_recovery_validation_execution_authority_spec.md`.
- Unrelated untracked files untouched.
- Exactly one new untracked authority specification created.

Observed candidate creation state:

- `git rev-parse HEAD` returned
  `79ec5fa764de30eaa04fb6de4c2a8228edf1a63a`.
- `git status --short --untracked-files=no` returned no tracked changes.
- `git diff --cached --name-status` returned no staged changes.
- `Test-Path -LiteralPath '<candidate path>'` returned `False`.
- `git status --short` showed pre-existing unrelated untracked files and
  environmental permission warnings from local pytest/cache directories. These
  warnings are not execution evidence and do not authorize touching unrelated
  untracked files.

If local HEAD differs from the required value, candidate creation is BLOCKED
and no checkout/reset is authorized.

## 2. Namespace/History Verdict

The namespace/history collision check searched for:

- `P4-R`
- `P4R`
- `recovery validation`
- `existing canonical sidecar validation`
- `equivalent authority`

Search verdict:

- Reports namespace listing found no P4-R/P4R authority path.
- Git commit-message search found no applicable P4-R/P4R/recovery-validation
  collision.
- Targeted reports search found no applicable existing canonical sidecar
  validation authority.
- Broad content search encountered unrelated large notebook/runtime matches and
  permission warnings; no applicable P4-R authority collision was identified.

P4-R is BLOCKED if independent verification finds an applicable collision,
equivalent authority, or authority conflict.

## 3. P4-Q Gate 1 Disposition

P4-Q authority freeze:

`79ec5fa764de30eaa04fb6de4c2a8228edf1a63a`

P4-Q execution implementation:

`2f9e6076791358922e3ebd70e89533d9cb83b458`

P4-Q Gate 1 observed stdout:

```json
{"artifact_materialization_requested":true,"evaluation_authorized":false,"namespace":"P3W6F2P4L_CURRENT_LINEAGE_INTEGRITY_SIDECAR","provenance_physical_sha256":"9d248df09ae8ba471966c468a1e06278ad046908cfe53da623ecc95d8da4cdf2","publish_status":"PUBLISHED","row_count":3600,"sidecar_physical_sha256":"2b8cffdf71d68a8abeb3b6eb3534eeb664bd012483bcebd9716c7a6645a487f1","sidecar_semantic_sha256":"0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08","status":"PASS","training_authorized":false}
```

Observed Gate 1 exit: `0`.

Observed Gate 1 stderr: empty.

Therefore:

- canonical materialization execution = PASS
- publication status = PUBLISHED
- Gate 2 = NOT EXECUTED
- P4-Q overall = FAIL due authority-wrapper validation defect
- canonical artifact/provenance integrity = NOT YET ESTABLISHED

P4-R must not reinterpret the P4-Q overall FAIL as a builder/materialization
FAIL.

## 4. P4-Q Known Wrapper Defects

Defect 1: post-Gate1 shell check.

The frozen P4-Q wrapper sorts actual filenames but compares them to:

```text
PROVENANCE_NAME SIDECAR_NAME
```

The observed `LC_ALL=C` sorted order is:

```text
SIDECAR_NAME PROVENANCE_NAME
```

Therefore the wrapper emitted:

```text
P4Q_FAIL:CANONICAL_OUTPUT_ENTRIES_MISMATCH_AFTER_GATE1
```

despite the canonical directory containing exactly the intended two files.

Defect 2: embedded Gate 2 Python.

The embedded Gate 2 Python also computes:

```python
entries = sorted(p.name for p in OUTPUT_DIR.iterdir())
```

and compares against:

```python
[PROVENANCE_NAME, SIDECAR_NAME]
```

This is the same reversed-order defect. P4-R must not invoke or extract the
frozen P4-Q Gate 2 verbatim.

P4-R corrects this with exact two-file validation equivalent to:

```python
entries = [p.name for p in output_dir.iterdir()]
require(
    len(entries) == 2
    and set(entries) == {SIDECAR_NAME, PROVENANCE_NAME},
    "OUTPUT_DIR_ENTRIES_MISMATCH",
)
```

The exact-two-file requirement is not weakened.

## 5. Forensic Recovery-Input Identity

Canonical directory:

`reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458`

Observed state after failed P4-Q wrapper:

- execution HEAD:
  `2f9e6076791358922e3ebd70e89533d9cb83b458`
- tracked state clean
- `CUDA_VISIBLE_DEVICES = <unset>`
- canonical directory = normal directory
- exact entries:
  `p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl`
  `p3w6f2_p4l_current_lineage_effective_integrity_sidecar_provenance.json`
- both entries = normal files
- sidecar physical SHA256:
  `2b8cffdf71d68a8abeb3b6eb3534eeb664bd012483bcebd9716c7a6645a487f1`
- provenance physical SHA256:
  `9d248df09ae8ba471966c468a1e06278ad046908cfe53da623ecc95d8da4cdf2`
- sidecar line count: `3600`
- `.p4l-staging-*` siblings: `NONE`
- observed Gate 1 semantic SHA256:
  `0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08`

These are observed recovery inputs, not predicted hashes.

## 6. Immutable/No-Retry Boundary

P4-R explicitly forbids:

- `python scripts/build_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar.py ...`
- any builder CLI
- `--materialize`
- `build_sidecar_artifacts()`
- `finalize_payloads_atomic()`
- atomic publication calls
- delete/unlink/rmtree/rename/replace/mkdir/write/chmod of canonical artifacts
- reset/clean
- backup/move-aside
- canonical artifact regeneration
- P4-Q Gate 1 retry

The existing canonical directory is read-only validation input. On P4-R
validation FAIL, exact bytes must be preserved for diagnosis.

## 7. Execution Identity

P4-R execution must run on Kaggle at detached exact:

`2f9e6076791358922e3ebd70e89533d9cb83b458`

After P4-R freeze, the workflow controller must supply exactly one runtime
parameter:

`P4R_AUTHORITY_FREEZE`

P4-R requires:

- `P4R_AUTHORITY_FREEZE` set.
- `P4R_AUTHORITY_FREEZE` exactly 40 lowercase hex.
- `P4R_AUTHORITY_FREEZE` commit exists.
- frozen P4-R authority path exists at that commit.
- actual execution HEAD remains exact
  `2f9e6076791358922e3ebd70e89533d9cb83b458`.
- no mutable `main`/`latest` execution.

Builder source SHA remains:

`b7fad3672428c8199347c278be0521e8355a8ca1c1491b5d3036e6cdc45f3f6d`

The builder is inspected/read only; it is never executed.

## 8. Preconditions

Before recovery validation require:

- detached HEAD == `2f9e6076791358922e3ebd70e89533d9cb83b458`
- tracked worktree clean
- index clean
- GPU OFF
- canonical directory exists
- canonical directory normal and non-symlink
- exactly the two expected normal non-symlink files
- no `.p4l-staging-*` sibling
- no backup/move-aside artifact

Require physical SHA before validation:

- sidecar ==
  `2b8cffdf71d68a8abeb3b6eb3534eeb664bd012483bcebd9716c7a6645a487f1`
- provenance ==
  `9d248df09ae8ba471966c468a1e06278ad046908cfe53da623ecc95d8da4cdf2`

If either differs, P4-R FAILS due recovery-input identity mismatch and must not
rewrite anything.

## 9. Exact Validator Design

The future validator is a single stdlib-only Python program embedded below. It
is based on the independently reviewed P4-Q Gate 2 validation intent, but does
not copy the reversed filename-order defect. It does not import the builder
module, does not call builder functions, does not execute the builder, and does
not write repository files. It may print observations to stdout. No temporary
repository files are necessary.

The exact future command is:

```bash
CUDA_VISIBLE_DEVICES="" python - <<'PY'
from __future__ import annotations

import json
import math
import os
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path, PurePosixPath

AUTHORITY_VERSION = "P3W6F2P4R_EXISTING_CANONICAL_SIDECAR_RECOVERY_VALIDATION_EXECUTION_AUTHORITY_V1"
PASS_TOKEN = "P3W6F2P4R_EXISTING_CANONICAL_SIDECAR_RECOVERY_VALIDATION_PASS"

EXECUTION_HEAD = "2f9e6076791358922e3ebd70e89533d9cb83b458"
P4Q_AUTHORITY_FREEZE = "79ec5fa764de30eaa04fb6de4c2a8228edf1a63a"
P4Q_IMPLEMENTATION = EXECUTION_HEAD
P4L_AUTHORITY_COMMIT = "80cb034792f03226cf6e22c196c1229ed4e6dd62"
BUILDER_SOURCE_SHA256 = "b7fad3672428c8199347c278be0521e8355a8ca1c1491b5d3036e6cdc45f3f6d"

P4R_AUTHORITY_PATH = "reports/reason_router_p2_p3w6f2_p4r_existing_canonical_sidecar_recovery_validation_execution_authority_spec.md"
OUTPUT_DIR_REL = "reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458"
SIDECAR_NAME = "p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl"
PROVENANCE_NAME = "p3w6f2_p4l_current_lineage_effective_integrity_sidecar_provenance.json"
SOURCE_REL = "reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl"
P4B_ROWS_REL = "reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/p3w6f2_p4b_r1_stage185_predicate_realization_compatibility_rows.jsonl"
P4B_SUMMARY_REL = "reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/p3w6f2_p4b_r1_stage185_predicate_realization_compatibility_summary.json"
P4B_PROVENANCE_REL = "reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/p3w6f2_p4b_r1_stage185_predicate_realization_compatibility_provenance_manifest.json"
BUILDER_SOURCE_REL = "scripts/build_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar.py"

EXPECTED_SIDECAR_PHYSICAL = "2b8cffdf71d68a8abeb3b6eb3534eeb664bd012483bcebd9716c7a6645a487f1"
EXPECTED_SIDECAR_SEMANTIC = "0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08"
EXPECTED_PROVENANCE_PHYSICAL = "9d248df09ae8ba471966c468a1e06278ad046908cfe53da623ecc95d8da4cdf2"
EXPECTED_SOURCE_PHYSICAL = "eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3"
EXPECTED_SOURCE_SEMANTIC = "3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b"
EXPECTED_P4B_ROWS_PHYSICAL = "59e4367d29e3c49152049e4a1b46e8783d5b81d1ebaa931ca9fd4ae0ac967b9f"
EXPECTED_P4B_SUMMARY_PHYSICAL = "ce618d214dc4d660706d927a0d91ec5945d3a0edbffbf615eab8b9c9ff585aa8"
EXPECTED_P4B_PROVENANCE_PHYSICAL = "09a56f1dca325e0749a0e4b0f822d68dad5d85fd90fa427e0efd6d617d41b2d6"

SOURCE_SEMANTIC_KEYS = [
    "id",
    "pair_id",
    "claim",
    "evidence",
    "final_label",
    "frame_compatible_label",
    "predicate_covered_label",
    "sufficiency_label",
    "polarity_label",
    "primary_failure_type",
    "intervention_type",
]

P2_REQUIRED_FIELDS = {
    "row_id",
    "pair_id",
    "split",
    "canonical_row_id",
    "source_order_index",
    "integrity_status",
    "frame_compatible_label",
    "eligible_for_positive_margin",
    "reason_codes",
    "time_swap_status",
    "dataset_source_status",
}

STAGE185_COMPATIBLE_FIELDS = {
    "row_id",
    "pair_id",
    "split",
    "integrity_status",
    "eligible_for_positive_margin",
    "reason_codes",
    "canonical_row_id",
}


def fail(code: str) -> None:
    raise SystemExit(f"P4R_FAIL:{code}")


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


def reject_nonfinite(obj, where: str) -> None:
    if isinstance(obj, float):
        require(math.isfinite(obj), f"NONFINITE_{where}")
    elif isinstance(obj, dict):
        for key, value in obj.items():
            reject_nonfinite(value, f"{where}_{key}")
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            reject_nonfinite(value, f"{where}_{index}")


def reject_json_constants(value: str):
    fail("NONFINITE_JSON_CONSTANT_" + value)


def physical_sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_lf_utf8_bytes(path: Path, code_prefix: str) -> bytes:
    data = path.read_bytes()
    require(not data.startswith(b"\xef\xbb\xbf"), code_prefix + "_BOM")
    require(b"\r" not in data, code_prefix + "_CR_OR_CRLF")
    require(data.endswith(b"\n"), code_prefix + "_NO_FINAL_NEWLINE")
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        fail(code_prefix + "_UTF8_DECODE")
    return data


def strict_repo_relative(repo_root: Path, rel: str, expected_type: str) -> Path:
    pure = PurePosixPath(rel.replace("\\", "/"))
    require(not pure.is_absolute(), "ABSOLUTE_RELATIVE_PATH_" + rel)
    require(".." not in pure.parts, "DOTDOT_PATH_" + rel)
    current = repo_root
    for part in pure.parts:
        current = current / part
        require(not current.is_symlink(), "SYMLINK_COMPONENT_" + rel)
    try:
        resolved = current.resolve(strict=True)
    except FileNotFoundError:
        fail("MISSING_PATH_" + rel)
    require(resolved == repo_root or repo_root in resolved.parents, "PATH_ESCAPE_" + rel)
    require(not resolved.is_symlink(), "SYMLINK_FINAL_" + rel)
    if expected_type == "file":
        require(resolved.is_file(), "NOT_FILE_" + rel)
    elif expected_type == "dir":
        require(resolved.is_dir(), "NOT_DIR_" + rel)
    else:
        fail("BAD_EXPECTED_TYPE")
    return resolved


def load_jsonl(path: Path, expected_count: int, code_prefix: str) -> list[dict]:
    data = require_lf_utf8_bytes(path, code_prefix)
    lines = data.decode("utf-8").splitlines()
    require(len(lines) == expected_count, code_prefix + "_COUNT")
    rows = []
    for index, line in enumerate(lines):
        require(line.strip() != "", code_prefix + "_EMPTY_LINE")
        obj = json.loads(line, parse_constant=reject_json_constants)
        require(isinstance(obj, dict), code_prefix + "_NON_OBJECT_LINE")
        reject_nonfinite(obj, f"{code_prefix}_{index}")
        rows.append(obj)
    return rows


def load_json_object(path: Path, code_prefix: str) -> tuple[dict, bytes]:
    data = require_lf_utf8_bytes(path, code_prefix)
    obj = json.loads(data.decode("utf-8"), parse_constant=reject_json_constants)
    require(isinstance(obj, dict), code_prefix + "_NOT_OBJECT")
    reject_nonfinite(obj, code_prefix)
    expected = (
        json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    require(expected == data, code_prefix + "_DETERMINISTIC_BYTES")
    return obj, data


repo_root = Path.cwd().resolve(strict=True)

authority_freeze = os.environ.get("P4R_AUTHORITY_FREEZE", "")
require(re.fullmatch(r"[0-9a-f]{40}", authority_freeze or "") is not None, "P4R_AUTHORITY_FREEZE_INVALID")
run_git(["cat-file", "-e", f"{authority_freeze}^{{commit}}"])
authority_listing = run_git(["ls-tree", "-r", "--name-only", authority_freeze, "--", P4R_AUTHORITY_PATH])
require(authority_listing == P4R_AUTHORITY_PATH, "P4R_AUTHORITY_PATH_MISSING_AT_FREEZE")

require(run_git(["rev-parse", "HEAD"]) == EXECUTION_HEAD, "HEAD_MISMATCH")
require(run_git(["status", "--short", "--untracked-files=no"]) == "", "TRACKED_WORKTREE_DIRTY")
require(run_git(["diff", "--cached", "--name-status"]) == "", "INDEX_DIRTY")
require(os.environ.get("CUDA_VISIBLE_DEVICES", "") == "", "GPU_NOT_OFF")

output_dir = strict_repo_relative(repo_root, OUTPUT_DIR_REL, "dir")
sidecar_path = strict_repo_relative(repo_root, OUTPUT_DIR_REL + "/" + SIDECAR_NAME, "file")
provenance_path = strict_repo_relative(repo_root, OUTPUT_DIR_REL + "/" + PROVENANCE_NAME, "file")
source_path = strict_repo_relative(repo_root, SOURCE_REL, "file")
p4b_rows_path = strict_repo_relative(repo_root, P4B_ROWS_REL, "file")
p4b_summary_path = strict_repo_relative(repo_root, P4B_SUMMARY_REL, "file")
p4b_provenance_path = strict_repo_relative(repo_root, P4B_PROVENANCE_REL, "file")
builder_source_path = strict_repo_relative(repo_root, BUILDER_SOURCE_REL, "file")

entries = [p.name for p in output_dir.iterdir()]
require(len(entries) == 2 and set(entries) == {SIDECAR_NAME, PROVENANCE_NAME}, "OUTPUT_DIR_ENTRIES_MISMATCH")
for sibling in output_dir.parent.glob(".p4l-staging-*"):
    fail("STAGING_SIBLING_PRESENT")
for sibling in output_dir.parent.iterdir():
    name = sibling.name.lower()
    if name.startswith(output_dir.name.lower()) and any(token in name for token in ("backup", "bak", "move-aside", "moved")):
        fail("BACKUP_OR_MOVE_ASIDE_PRESENT")

sidecar_physical_before = physical_sha(sidecar_path)
provenance_physical_before = physical_sha(provenance_path)
require(sidecar_physical_before == EXPECTED_SIDECAR_PHYSICAL, "SIDECAR_RECOVERY_INPUT_SHA_MISMATCH")
require(provenance_physical_before == EXPECTED_PROVENANCE_PHYSICAL, "PROVENANCE_RECOVERY_INPUT_SHA_MISMATCH")
require(physical_sha(source_path) == EXPECTED_SOURCE_PHYSICAL, "SOURCE_PHYSICAL_SHA_MISMATCH")
require(physical_sha(p4b_rows_path) == EXPECTED_P4B_ROWS_PHYSICAL, "P4B_ROWS_PHYSICAL_SHA_MISMATCH")
require(physical_sha(p4b_summary_path) == EXPECTED_P4B_SUMMARY_PHYSICAL, "P4B_SUMMARY_PHYSICAL_SHA_MISMATCH")
require(physical_sha(p4b_provenance_path) == EXPECTED_P4B_PROVENANCE_PHYSICAL, "P4B_PROVENANCE_PHYSICAL_SHA_MISMATCH")
require(physical_sha(builder_source_path) == BUILDER_SOURCE_SHA256, "BUILDER_SOURCE_SHA_MISMATCH")

sidecar_rows = load_jsonl(sidecar_path, 3600, "SIDECAR")
source_rows = load_jsonl(source_path, 3600, "SOURCE")
provenance, provenance_bytes = load_json_object(provenance_path, "PROVENANCE")

source_semantic_bytes = b"".join(
    (
        json.dumps(
            {key: row.get(key) for key in SOURCE_SEMANTIC_KEYS},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")
    for row in source_rows
)
require(sha256(source_semantic_bytes).hexdigest() == EXPECTED_SOURCE_SEMANTIC, "SOURCE_SEMANTIC_SHA_MISMATCH")

source_ids = [row.get("id") for row in source_rows]
sidecar_ids = [row.get("row_id") for row in sidecar_rows]
require(sidecar_ids == source_ids, "ROW_ID_SEQUENCE_MISMATCH")
require(all(isinstance(x, str) and x for x in sidecar_ids), "ROW_ID_EMPTY_OR_NONSTRING")
require(len(set(sidecar_ids)) == 3600, "ROW_ID_DUPLICATE")
require(all(row.get("source_order_index") == i for i, row in enumerate(sidecar_rows)), "SOURCE_ORDER_INDEX_MISMATCH")

for index, row in enumerate(sidecar_rows):
    missing_p2 = P2_REQUIRED_FIELDS - set(row)
    missing_185 = STAGE185_COMPATIBLE_FIELDS - set(row)
    require(not missing_p2, "MISSING_P4L_P2_FIELDS")
    require(not missing_185, "MISSING_STAGE185_COMPATIBLE_FIELDS")
    frame_value = row.get("frame_compatible_label")
    require(type(frame_value) is int and frame_value in (0, 1), "FRAME_COMPATIBLE_LABEL_TYPE")
    require(type(row.get("eligible_for_positive_margin")) is bool, "ELIGIBLE_FOR_POSITIVE_MARGIN_TYPE")
    reason_codes = row.get("reason_codes")
    require(isinstance(reason_codes, list), "REASON_CODES_NOT_LIST")
    require(reason_codes == sorted(reason_codes) and len(reason_codes) == len(set(reason_codes)), "REASON_CODES_NOT_SORTED_UNIQUE")

pairs = sorted({row["pair_id"] for row in source_rows})
shuffled = list(pairs)
random.Random(174).shuffle(shuffled)
dev_count = min(len(shuffled) - 1, max(1, round(len(shuffled) * 0.2)))
dev_pairs = set(shuffled[:dev_count])
expected_split_by_pair = {pair: ("dev" if pair in dev_pairs else "train") for pair in pairs}
for row in sidecar_rows:
    require(row.get("split") == expected_split_by_pair.get(row.get("pair_id")), "SPLIT_REPLAY_MISMATCH")

source_by_id = {row["id"]: row for row in source_rows}
same_pair_none = defaultdict(list)
for row in source_rows:
    if row.get("intervention_type") == "none":
        same_pair_none[row["pair_id"]].append(row["id"])
require(all(len(ids) == 1 for ids in same_pair_none.values()), "CANONICAL_NONE_COUNT_MISMATCH")

for sidecar, source in zip(sidecar_rows, source_rows):
    pair_id = source["pair_id"]
    canonical_id = same_pair_none[pair_id][0]
    require(sidecar.get("canonical_row_id") == canonical_id, "CANONICAL_ROW_ID_MISMATCH")
    canonical_sidecar = sidecar_rows[sidecar_ids.index(canonical_id)]
    require(canonical_sidecar["pair_id"] == pair_id, "CANONICAL_PAIR_MISMATCH")
    require(canonical_sidecar["split"] == sidecar["split"], "CANONICAL_SPLIT_MISMATCH")
    if source.get("intervention_type") == "none":
        require(sidecar["row_id"] == sidecar["canonical_row_id"], "CANONICAL_SELF_ANCHOR_MISMATCH")

def primary_reason(source: dict) -> str:
    if source.get("frame_compatible_label") == 0:
        return "FRAME"
    if source.get("predicate_covered_label") == 0:
        return "PREDICATE"
    if source.get("sufficiency_label") == 0:
        return "SUFFICIENCY"
    return "AUTHORIZED"

integrity_counts = Counter()
positive_margin_counts = Counter()
reason_supervision_counts = Counter()
unresolved_status_counts = Counter()
unresolved_reason_counts = Counter()
unresolved_count = 0
for sidecar, source in zip(sidecar_rows, source_rows):
    expected_reason = primary_reason(source)
    if "p2_primary_reason" in sidecar:
        require(sidecar["p2_primary_reason"] == expected_reason, "PRIMARY_REASON_REPLAY_MISMATCH")
    expected_positive = (
        sidecar.get("integrity_status") == "ELIGIBLE"
        and sidecar.get("split") == "train"
        and sidecar.get("frame_compatible_label") == 1
        and sidecar.get("time_swap_status") == "PASS"
        and sidecar.get("dataset_source_status") == "PASS"
    )
    require(sidecar.get("eligible_for_positive_margin") == expected_positive, "POSITIVE_MARGIN_REPLAY_MISMATCH")
    integrity_counts[sidecar.get("integrity_status")] += 1
    positive_margin_counts[str(sidecar.get("eligible_for_positive_margin")).lower()] += 1
    if "p2_reason_supervision_eligible" in sidecar:
        reason_supervision_counts[str(sidecar.get("p2_reason_supervision_eligible")).lower()] += 1
    status_text = str(sidecar.get("integrity_status", ""))
    if status_text not in {"ELIGIBLE", "PASS"}:
        unresolved_count += 1
        unresolved_status_counts[status_text] += 1
        unresolved_reason_counts[str(sidecar.get("unresolved_reason", sidecar.get("p2_primary_reason", "MISSING")))] += 1

p4b_rows = load_jsonl(p4b_rows_path, 357, "P4B_ROWS")
p4b_summary, _ = load_json_object(p4b_summary_path, "P4B_SUMMARY")
p4b_provenance, _ = load_json_object(p4b_provenance_path, "P4B_PROVENANCE")
require(len({row.get("pair_id") for row in p4b_rows}) == 119, "P4B_PAIR_SCOPE_MISMATCH")
require(len(p4b_rows) == 357, "P4B_MEMBER_SCOPE_MISMATCH")
require(p4b_summary.get("pair_count", p4b_summary.get("pairs")) in (119, "119"), "P4B_SUMMARY_PAIR_SCOPE_MISMATCH")
require(p4b_summary.get("member_count", p4b_summary.get("members")) in (357, "357"), "P4B_SUMMARY_MEMBER_SCOPE_MISMATCH")
require(p4b_provenance.get("pair_count", p4b_provenance.get("pairs")) in (119, "119"), "P4B_PROVENANCE_PAIR_SCOPE_MISMATCH")
require(p4b_provenance.get("member_count", p4b_provenance.get("members")) in (357, "357"), "P4B_PROVENANCE_MEMBER_SCOPE_MISMATCH")

semantic_rows = []
for row in sidecar_rows:
    semantic_row = dict(row)
    semantic_row.pop("created_at", None)
    semantic_rows.append(semantic_row)
sidecar_semantic_bytes = b"".join(
    (
        json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    for row in semantic_rows
)
sidecar_semantic_sha = sha256(sidecar_semantic_bytes).hexdigest()
require(sidecar_semantic_sha == EXPECTED_SIDECAR_SEMANTIC, "SIDECAR_SEMANTIC_SHA_MISMATCH")

required_provenance = {
    "sidecar_physical_sha256": EXPECTED_SIDECAR_PHYSICAL,
    "sidecar_semantic_sha256": EXPECTED_SIDECAR_SEMANTIC,
    "schema_version": "P3W6F2P4L_CURRENT_LINEAGE_INTEGRITY_SIDECAR_PROVENANCE_V1",
    "authority_version": "P3W6F2P4L_CURRENT_LINEAGE_INTEGRITY_ARTIFACT_CONTRACT_AUTHORITY_V1",
    "p4l_authority_commit": P4L_AUTHORITY_COMMIT,
    "builder_source_commit": EXECUTION_HEAD,
    "builder_source_sha256": BUILDER_SOURCE_SHA256,
    "source_dataset_path": SOURCE_REL,
    "source_dataset_sha256": EXPECTED_SOURCE_PHYSICAL,
    "source_dataset_semantic_sha256": EXPECTED_SOURCE_SEMANTIC,
    "source_physical_sha256": EXPECTED_SOURCE_PHYSICAL,
    "source_semantic_sha256": EXPECTED_SOURCE_SEMANTIC,
    "p4b_compatibility_rows_path": P4B_ROWS_REL,
    "p4b_compatibility_rows_sha256": EXPECTED_P4B_ROWS_PHYSICAL,
    "p4b_compatibility_summary_path": P4B_SUMMARY_REL,
    "p4b_compatibility_summary_sha256": EXPECTED_P4B_SUMMARY_PHYSICAL,
    "p4b_compatibility_provenance_path": P4B_PROVENANCE_REL,
    "p4b_compatibility_provenance_sha256": EXPECTED_P4B_PROVENANCE_PHYSICAL,
    "canonical_sidecar_path": OUTPUT_DIR_REL + "/" + SIDECAR_NAME,
}
for key, expected in required_provenance.items():
    require(provenance.get(key) == expected, "PROVENANCE_" + key.upper() + "_MISMATCH")

scope = provenance.get("p4b_compatibility_authorized_scope")
require(isinstance(scope, dict), "PROVENANCE_P4B_COMPATIBILITY_AUTHORIZED_SCOPE_NOT_OBJECT")
require(scope.get("pair_count") == 119, "PROVENANCE_P4B_COMPATIBILITY_AUTHORIZED_SCOPE_PAIR_COUNT_MISMATCH")
require(scope.get("member_count") == 357, "PROVENANCE_P4B_COMPATIBILITY_AUTHORIZED_SCOPE_MEMBER_COUNT_MISMATCH")
require(provenance.get("provenance_physical_sha256_self_certified") is False, "PROVENANCE_SELF_CERTIFIED_FLAG_MISMATCH")
require(provenance.get("row_count") == 3600, "PROVENANCE_ROW_COUNT_MISMATCH")
require(provenance.get("one_to_one_row_coverage") is True, "PROVENANCE_ONE_TO_ONE_ROW_COVERAGE_MISMATCH")
require(provenance.get("unique_row_id") is True, "PROVENANCE_UNIQUE_ROW_ID_MISMATCH")
require(provenance.get("historical_stage185_used_as_current_source_identity") is False, "HISTORICAL_STAGE185_FLAG_MISMATCH")
require(provenance.get("blockers") == [], "PROVENANCE_BLOCKERS_NOT_EMPTY")
require(provenance.get("failure_reasons") == [], "PROVENANCE_FAILURE_REASONS_NOT_EMPTY")

for flag in [
    "artifact_materialization_authorized_by_p4l",
    "kaggle_authorized",
    "gpu_authorized",
    "training_admission_released",
    "a0_execution_authorized",
    "training_authorized",
    "evaluation_authorized",
]:
    require(provenance.get(flag) is False, "PROVENANCE_FLAG_NOT_FALSE_" + flag.upper())

require(physical_sha(sidecar_path) == EXPECTED_SIDECAR_PHYSICAL, "SIDECAR_PHYSICAL_SHA_CHANGED")
require(physical_sha(provenance_path) == EXPECTED_PROVENANCE_PHYSICAL, "PROVENANCE_PHYSICAL_SHA_CHANGED")
require(run_git(["rev-parse", "HEAD"]) == EXECUTION_HEAD, "POST_HEAD_MISMATCH")
require(run_git(["status", "--short", "--untracked-files=no"]) == "", "POST_TRACKED_WORKTREE_DIRTY")
require(run_git(["diff", "--cached", "--name-status"]) == "", "POST_INDEX_DIRTY")
require(os.environ.get("CUDA_VISIBLE_DEVICES", "") == "", "POST_GPU_NOT_OFF")
require(not output_dir.is_symlink() and output_dir.is_dir(), "POST_OUTPUT_DIR_NOT_NORMAL")
entries_after = [p.name for p in output_dir.iterdir()]
require(len(entries_after) == 2 and set(entries_after) == {SIDECAR_NAME, PROVENANCE_NAME}, "POST_OUTPUT_DIR_ENTRIES_MISMATCH")
for sibling in output_dir.parent.glob(".p4l-staging-*"):
    fail("POST_STAGING_SIBLING_PRESENT")

observations = {
    "authority_version": AUTHORITY_VERSION,
    "python_version": sys.version,
    "sys_platform": sys.platform,
    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
    "sidecar_physical_sha256": sidecar_physical_before,
    "sidecar_semantic_sha256": sidecar_semantic_sha,
    "provenance_physical_sha256": provenance_physical_before,
    "integrity_status_counts": dict(sorted(integrity_counts.items())),
    "positive_margin_counts": dict(sorted(positive_margin_counts.items())),
    "p2_reason_supervision_eligible_counts": dict(sorted(reason_supervision_counts.items())),
    "unresolved_row_count": unresolved_count,
    "unresolved_status_summary": dict(sorted(unresolved_status_counts.items())),
    "unresolved_reason_summary": dict(sorted(unresolved_reason_counts.items())),
}
print(json.dumps(observations, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
print(PASS_TOKEN)
PY
```

## 10. Corrected Exact-Two-File Check

Correct validation is order independent and requires exactly two entries:

```python
entries = [p.name for p in output_dir.iterdir()]
require(
    len(entries) == 2 and set(entries) == {SIDECAR_NAME, PROVENANCE_NAME},
    "OUTPUT_DIR_ENTRIES_MISMATCH",
)
```

It must not compare sorted actual names to an unsorted expected list.

## 11. Path/Symlink Safety

The validator uses:

```python
repo_root = Path.cwd().resolve(strict=True)
```

It uses a strict repo-relative read helper that:

- rejects absolute relative paths
- rejects `..`
- rejects symlink components
- strict-resolves the target
- proves the resolved path remains below `repo_root`
- requires expected normal file/dir type

The helper applies to:

- canonical directory
- sidecar
- provenance
- regenerated source dataset
- P4-B compatibility rows
- P4-B compatibility summary
- P4-B compatibility provenance
- builder source

No unchecked external symlink target is allowed.

## 12. Serialization Checks

Sidecar requirements:

- regular non-symlink file
- UTF-8
- no BOM
- LF only
- no CR/CRLF
- final newline
- exactly 3600 non-empty JSONL objects
- explicit NaN/Infinity rejection

Provenance requirements:

- regular non-symlink file
- UTF-8
- no BOM
- LF only
- final newline
- exactly one JSON object
- explicit non-finite rejection
- exact deterministic builder bytes:

```python
json.dumps(
    provenance,
    sort_keys=True,
    indent=2,
    ensure_ascii=False,
    allow_nan=False,
) + "\n"
```

UTF-8 encoded. Reserialized bytes must equal actual bytes.

## 13. Source Identity Checks

Exact source path:

`reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl`

Physical SHA:

`eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`

Semantic SHA:

`3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b`

The validator recomputes semantic SHA from ordered rows with exactly:

- `id`
- `pair_id`
- `claim`
- `evidence`
- `final_label`
- `frame_compatible_label`
- `predicate_covered_label`
- `sufficiency_label`
- `polarity_label`
- `primary_failure_type`
- `intervention_type`

Canonical serialization:

- `sort_keys=True`
- `separators=(",", ":")`
- `ensure_ascii=False`
- UTF-8

Source count must equal 3600.

## 14. Row Universe/Order/Schema

Require:

- sidecar count == 3600
- source count == 3600
- sidecar `row_id` sequence == source `id` sequence exactly
- unique non-empty `row_id`
- no missing/extra/duplicate
- `source_order_index == 0..3599` when emitted
- all P4-L required P2 fields
- all P4-L Stage185-compatible fields

Types:

- `frame_compatible_label` exact int 0/1, bool rejected
- `eligible_for_positive_margin` exact bool
- `reason_codes` list, sorted, unique

Current source path/physical/semantic bindings must be exact.
`historical_stage185_used_as_current_source_identity == false`.
Historical `f552...` is never current source SHA.

## 15. Split Replay

Exact algorithm:

```python
pairs = sorted(unique pair IDs)
shuffled = list(pairs)
random.Random(174).shuffle(shuffled)

dev_count = min(
    len(shuffled) - 1,
    max(1, round(len(shuffled) * 0.2)),
)

dev_pairs = set(shuffled[:dev_count])
```

Pair split is `dev` iff pair in `dev_pairs`, else `train`. Every sidecar split
must match. Python round semantics are intentional.

## 16. Canonical Replay

For each pair, require exactly one source row with:

`intervention_type == "none"`

Every emitted `canonical_row_id` must match that row. The canonical target must
exist, be in the same pair, be in the same split, and self-anchor when the row
itself is the canonical source row.

## 17. Reason/Eligibility Replay

Primary reason:

```text
frame == 0 -> FRAME
else predicate == 0 -> PREDICATE
else sufficiency == 0 -> SUFFICIENCY
else AUTHORIZED
```

Require `p2_primary_reason` where emitted.

Positive-margin formula exactly:

```text
integrity_status == "ELIGIBLE"
and split == "train"
and frame_compatible_label == 1
and time_swap_status == "PASS"
and dataset_source_status == "PASS"
```

Require exact emitted equality.

Report:

- integrity_status counts
- positive-margin true/false counts
- `p2_reason_supervision_eligible` counts when present
- unresolved row count
- unresolved status/reason summaries

Do not require unresolved count == 0 unless frozen P4-L requires it.

## 18. P4-B Scope

Recompute exact physical SHAs:

- rows:
  `59e4367d29e3c49152049e4a1b46e8783d5b81d1ebaa931ca9fd4ae0ac967b9f`
- summary:
  `ce618d214dc4d660706d927a0d91ec5945d3a0edbffbf615eab8b9c9ff585aa8`
- provenance:
  `09a56f1dca325e0749a0e4b0f822d68dad5d85fd90fa427e0efd6d617d41b2d6`

Require 119 pairs / 357 members.

No full-3600 compatibility inference is authorized.

## 19. Sidecar Physical/Semantic Hashes

Recompute physical SHA and require exact:

`2b8cffdf71d68a8abeb3b6eb3534eeb664bd012483bcebd9716c7a6645a487f1`

Also require `provenance.sidecar_physical_sha256` equals this.

Recompute semantic SHA exactly:

- preserve row order
- remove `created_at` only
- retain all other fields
- `sort_keys=True`
- `separators=(",", ":")`
- `ensure_ascii=False`
- UTF-8 SHA256

Require exact:

`0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08`

Also require `provenance.sidecar_semantic_sha256` equals this.

## 20. Provenance Physical Hash

External `SHA256(exact provenance bytes)` must equal exact:

`9d248df09ae8ba471966c468a1e06278ad046908cfe53da623ecc95d8da4cdf2`

Require:

`provenance_physical_sha256_self_certified == false`

Do not require a recursive/self physical hash field.

## 21. Provenance Identities/Flags

Require exact frozen P4-L identities:

- `schema_version =
  P3W6F2P4L_CURRENT_LINEAGE_INTEGRITY_SIDECAR_PROVENANCE_V1`
- `authority_version =
  P3W6F2P4L_CURRENT_LINEAGE_INTEGRITY_ARTIFACT_CONTRACT_AUTHORITY_V1`
- `p4l_authority_commit =
  80cb034792f03226cf6e22c196c1229ed4e6dd62`
- `builder_source_commit =
  2f9e6076791358922e3ebd70e89533d9cb83b458`
- `builder_source_sha256 =
  b7fad3672428c8199347c278be0521e8355a8ca1c1491b5d3036e6cdc45f3f6d`
- source identities exact
- canonical sidecar path exact
- `row_count == 3600`
- `one_to_one_row_coverage == true`
- `unique_row_id == true`
- P4-B identities and 119/357 scope exact
- `historical_stage185_used_as_current_source_identity == false`
- `blockers == []`
- `failure_reasons == []`

Intrinsic P4-L flags remain frozen:

- `artifact_materialization_authorized_by_p4l == false`
- `kaggle_authorized == false`
- `gpu_authorized == false`
- `training_admission_released == false`
- `a0_execution_authorized == false`
- `training_authorized == false`
- `evaluation_authorized == false`

P4-R must not rewrite intrinsic flags.

## 22. PASS Token/Postconditions

Only after every read-only assertion passes, print:

`P3W6F2P4R_EXISTING_CANONICAL_SIDECAR_RECOVERY_VALIDATION_PASS`

Before that, print one compact observations JSON containing:

- Python version
- `sys.platform`
- `CUDA_VISIBLE_DEVICES`
- sidecar physical SHA
- sidecar semantic SHA
- provenance physical SHA
- integrity_status counts
- positive-margin counts
- `p2_reason_supervision_eligible` counts
- unresolved row count
- unresolved reason/status summary

After validation require again:

- HEAD exact `2f9e6076791358922e3ebd70e89533d9cb83b458`
- tracked/index clean
- canonical directory normal/non-symlink
- exact two normal files
- sidecar physical SHA unchanged
- provenance physical SHA unchanged
- no staging sibling
- GPU OFF

P4-R performs no artifact cleanup on PASS or FAIL.

## 23. PASS/BLOCKED/FAIL Mapping

PASS:

All recovery preconditions, validator assertions, and postconditions pass.

BLOCKED:

- authority freeze unavailable
- execution HEAD/state mismatch
- tooling/environment prevents safe read-only validation
- GPU cannot be kept off

FAIL:

- existing recovery input hashes differ
- serialized artifact mismatch
- source identity/order mismatch
- split/canonical/reason mismatch
- P4-B scope mismatch
- hash/provenance mismatch
- artifact mutation detected

The known P4-Q filename-order authority defect itself is not an artifact FAIL.

## 24. Interpretation Boundary

On P4-R PASS:

1. code correctness = PASS
2. POSIX runtime publication = PASS
3. P4-Q Gate 1 canonical materialization = PASS
4. existing canonical P4-L artifact/provenance integrity = ESTABLISHED
5. P4-Q original combined execution = FAIL due authority-wrapper defect
6. trainer rebind = NOT AUTHORIZED
7. A0 = NOT AUTHORIZED
8. training/evaluation = NOT AUTHORIZED
9. scientific conclusion = NOT ESTABLISHED

P4-R does not retroactively make the defective P4-Q wrapper PASS. It recovers
validation of the already published immutable bytes.

## 25. Authority Flags

Candidate-time flags are all false:

- `recovery_validation_execution_authorized = false`
- `kaggle_authorized = false`
- `cpu_authorized = false`
- `gpu_authorized = false`
- `artifact_materialization_authorized = false`
- `trainer_rebind_authorized = false`
- `a0_execution_authorized = false`
- `training_admission_released = false`
- `training_authorized = false`
- `evaluation_authorized = false`

After independent verification plus immutable P4-R freeze, only the exact P4-R
run may set:

- `recovery_validation_execution_authorized = true`
- `kaggle_authorized = true`
- `cpu_authorized = true`
- `gpu_authorized = false`

Artifact materialization remains false.

## 26. Evidence Boundary

Future evidence namespace:

`reports/reason_router_p2_p3w6f2_p4r_existing_canonical_sidecar_recovery_validation_execution_2f9e6076791358922e3ebd70e89533d9cb83b458/`

Future evidence must preserve:

- P4-R authority freeze
- P4-Q authority freeze
- implementation commit
- exact observed P4-Q Gate 1 stdout
- exact P4-Q wrapper failure:
  `P4Q_FAIL:CANONICAL_OUTPUT_ENTRIES_MISMATCH_AFTER_GATE1`
- forensic capture facts
- exact P4-R validator command
- validator output/token
- pre/post hashes and state
- status/unresolved summaries
- blockers/failure reasons

No evidence is created now. No ZIP is created now. No run ID is invented. No
new artifact hashes are predicted.

## 27. Scientific Boundary

P4-R makes no change to:

- reason-router mechanism
- supervision
- gradient ownership
- reason order
- secondary diagnostic semantics
- router-only CE/detach contract
- EMA role
- A0-A3
- E0

No trainer/model/config/checkpoint edits are authorized.

## 28. Candidate Path

Candidate path:

`reports/reason_router_p2_p3w6f2_p4r_existing_canonical_sidecar_recovery_validation_execution_authority_spec.md`

Expected delta:

Exactly one new untracked authority spec.

No Python, pytest, builder, materialization, Kaggle execution, validation
execution, staging, commit, or push is authorized during candidate creation.

## 29. Candidate SHA256

Candidate SHA256 is computed after file creation with a read-only filesystem
hash command and reported in the candidate creation summary. The hash is a
property of this authority-spec file only; it is not a predicted artifact hash
and does not authorize execution.

## 30. Blockers/Final Git Status

Current blockers: none identified during candidate creation.

Independent static verification is required before freeze. The verifier must
inspect:

- no builder/materialization path
- exact recovery hashes
- corrected two-file comparison
- read-only path safety
- serialization algorithms
- exact split/canonical replay
- semantic hash
- provenance interpretation
- no retroactive P4-Q PASS
- no trainer/A0/training widening

Final git status must show the candidate as the only newly created P4-R
authority spec, with pre-existing unrelated untracked files untouched and no
tracked/index changes beyond the intended untracked candidate.

Success token:

`P3W6F2P4R_EXISTING_CANONICAL_SIDECAR_RECOVERY_VALIDATION_EXECUTION_AUTHORITY_CANDIDATE_READY_FOR_INDEPENDENT_VERIFICATION`
