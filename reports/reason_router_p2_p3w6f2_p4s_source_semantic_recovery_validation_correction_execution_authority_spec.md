# P3-W6-F2-P4-S Source-Semantic Recovery Validation Correction Execution Authority Specification

Authority/version:

`P3W6F2P4S_SOURCE_SEMANTIC_RECOVERY_VALIDATION_CORRECTION_EXECUTION_AUTHORITY_V1`

This document is a candidate authority specification only. Candidate creation
does not authorize recovery validation execution, Kaggle, CPU execution, GPU
execution, artifact materialization, builder execution, trainer rebind, A0,
training, evaluation, staging, commit, push, ZIP creation, evidence creation,
canonical rewrite, canonical deletion, P4-Q Gate 1 retry, or P4-R exact
validator retry.

After independent static verification PASS and immutable P4-S freeze, this
authority may authorize exactly one CPU-only, read-only recovery validation of
the already published immutable canonical P4-L sidecar/provenance bytes,
correcting the source semantic SHA recomputation defect in frozen P4-R and the
P4-S candidate sidecar semantic SHA recomputation defect. P4-S does not repair
or retroactively modify P4-R.

## 1. HEAD/State Proof

Candidate creation requires:

- HEAD exactly `d80ed289273763c1c90f2fba14ca796c604c9529`.
- Tracked worktree clean.
- Index clean.
- Candidate path absent before creation:
  `reports/reason_router_p2_p3w6f2_p4s_source_semantic_recovery_validation_correction_execution_authority_spec.md`.
- Unrelated untracked files untouched.
- Exactly one new untracked authority specification created.

Observed candidate creation state:

- `git rev-parse HEAD` returned
  `d80ed289273763c1c90f2fba14ca796c604c9529`.
- `git diff --quiet` exited `0`.
- `git diff --cached --quiet` exited `0`.
- `Test-Path -LiteralPath '<candidate path>'` returned `False`.
- `git status --short` showed pre-existing unrelated untracked files and
  environmental permission warnings from local pytest/cache directories. These
  were untouched and are not execution evidence.

If local HEAD or state differs from the required values, candidate creation is
BLOCKED and no checkout/reset is authorized.

## 2. Namespace/History Verdict

The namespace/history collision check searched for:

- `P4-S`
- `P4S`
- `source-semantic recovery`
- `SOURCE_SEMANTIC_RECOVERY`
- `SOURCE_SEMANTIC_RECOVERY_VALIDATION`
- `P3W6F2P4S`

Search verdict:

- Targeted report/agent search found no applicable P4-S/P4S/source-semantic
  recovery authority collision.
- Frozen P4-R authority was found and is the required basis for this corrected
  authority, not a collision.

P4-S is BLOCKED if independent verification finds an applicable collision,
equivalent authority, or authority conflict.

## 3. P4-Q Disposition

P4-Q authority freeze:

`79ec5fa764de30eaa04fb6de4c2a8228edf1a63a`

P4-Q Gate 1:

`PASS`

P4-Q overall:

`FAIL due filename-order authority-wrapper defect`

P4-S must not retroactively relabel P4-Q as PASS.

## 4. P4-R Disposition

P4-R authority freeze:

`d80ed289273763c1c90f2fba14ca796c604c9529`

P4-R execution:

`FAIL`

Exact observed P4-R failure:

`P4R_FAIL:SOURCE_SEMANTIC_SHA_MISMATCH`

P4-R did not establish artifact/provenance integrity. P4-S must not call P4-R
PASS and must not call the canonical artifact defective based solely on this
failure.

## 5. Static Root-Cause Proof

At exact builder commit `2f9e6076791358922e3ebd70e89533d9cb83b458`, source
semantic SHA is computed as:

```python
semantic_dataset_sha256(rows) == canonical_sha256(
    [ordered_dataset_row(row) for row in rows]
)
```

The builder defines:

```python
DATASET_FIELDS = (
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
)

def canonical_json_bytes(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")

def canonical_sha256(value):
    return sha256_bytes(canonical_json_bytes(value))

def ordered_dataset_row(row):
    missing = [field for field in DATASET_FIELDS if field not in row]
    require(not missing, ...)
    return {field: row[field] for field in DATASET_FIELDS}

def semantic_dataset_sha256(rows):
    return canonical_sha256([ordered_dataset_row(row) for row in rows])
```

Therefore the top-level serialized value is one JSON list. There is no per-row
newline, no JSONL concatenation, no final newline, no row sorting, no row-by-row
hashing, and no `row.get()` defaulting.

Frozen P4-R instead computes an incompatible byte stream equivalent to:

```python
b"".join(
    (
        json.dumps(
            projected_row,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    for row in source_rows
)
```

The exact P4-R validator source used `row.get(key)` and newline-delimited
compact row serialization:

```python
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
```

That is compact JSONL-like concatenation and is not the builder/P4-L semantic
hash algorithm. Therefore:

`P4R_FAIL:SOURCE_SEMANTIC_SHA_MISMATCH`

is a validator false-fail attributable to authority/executable logic. It does
not establish a source dataset semantic defect.

## 6. Immutable Recovery Input

Canonical directory:

`reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458`

Observed immutable hashes remain:

- sidecar physical:
  `2b8cffdf71d68a8abeb3b6eb3534eeb664bd012483bcebd9716c7a6645a487f1`
- sidecar semantic expected:
  `0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08`
- provenance physical:
  `9d248df09ae8ba471966c468a1e06278ad046908cfe53da623ecc95d8da4cdf2`
- source physical:
  `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`
- correct source semantic expected:
  `3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b`

No existing canonical byte may be changed.

## 7. No-Retry/No-Mutation Boundary

P4-S explicitly forbids:

- builder CLI
- `--materialize`
- `build_sidecar_artifacts()`
- `finalize_payloads_atomic()`
- atomic publication
- P4-Q Gate 1 retry
- P4-R exact validator rerun
- canonical regeneration
- unlink/delete/rmtree/rename/replace
- mkdir affecting repository artifacts
- write/append/update artifact opens
- chmod
- `git reset`
- `git clean`
- backup/move-aside

The existing canonical directory is read-only validation input. P4-S performs
no artifact cleanup on PASS or FAIL.

## 8. Execution Identity

Future execution must run detached exact:

`2f9e6076791358922e3ebd70e89533d9cb83b458`

After P4-S freeze, the workflow controller must supply exactly one runtime
parameter:

`P4S_AUTHORITY_FREEZE`

P4-S requires:

- `P4S_AUTHORITY_FREEZE` set.
- `P4S_AUTHORITY_FREEZE` exactly 40 lowercase hex.
- The commit exists.
- The P4-S authority path exists at that commit.
- The validator does not checkout the P4-S freeze.
- No mutable `main`/`latest` execution.
- GPU OFF.
- CPU only.

Builder source:

`scripts/build_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar.py`

Builder source SHA256:

`b7fad3672428c8199347c278be0521e8355a8ca1c1491b5d3036e6cdc45f3f6d`

The builder is inspected/read only; it is never executed.

## 9. Corrected Validator

The future validator is one stdlib-only, read-only Python program embedded
below. It uses the independently reviewed frozen P4-R executable validator as
its basis. It preserves every P4-R assertion that was previously independently
verified PASS, including recovery physical-hash preconditions,
authority/execution identity, builder SHA read-only check, corrected
order-independent exact-two-file check, repo containment/symlink checks,
sidecar serialization, provenance deterministic pretty serialization, explicit
non-finite rejection, source physical SHA, 3600/order/schema checks, split
replay, canonical replay, reason replay, positive-margin replay, unresolved
reporting, P4-B external hashes and 119/357, P4-B provenance bindings,
current-source provenance bindings, sidecar physical hash, sidecar semantic
hash removing `created_at` only, external provenance physical hash, no
provenance recursive self-hash, all P4-L provenance identities/intrinsic
flags, postconditions, and no retroactive P4-Q/P4-R PASS.

The substantive validator algorithm corrections are source semantic SHA
recomputation from frozen P4-R and sidecar semantic SHA recomputation from the
pre-repair P4-S candidate.

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

AUTHORITY_VERSION = "P3W6F2P4S_SOURCE_SEMANTIC_RECOVERY_VALIDATION_CORRECTION_EXECUTION_AUTHORITY_V1"
PASS_TOKEN = "P3W6F2P4S_SOURCE_SEMANTIC_RECOVERY_VALIDATION_CORRECTION_PASS"

EXECUTION_HEAD = "2f9e6076791358922e3ebd70e89533d9cb83b458"
P4R_AUTHORITY_FREEZE = "d80ed289273763c1c90f2fba14ca796c604c9529"
P4Q_AUTHORITY_FREEZE = "79ec5fa764de30eaa04fb6de4c2a8228edf1a63a"
P4Q_IMPLEMENTATION = EXECUTION_HEAD
P4L_AUTHORITY_COMMIT = "80cb034792f03226cf6e22c196c1229ed4e6dd62"
BUILDER_SOURCE_SHA256 = "b7fad3672428c8199347c278be0521e8355a8ca1c1491b5d3036e6cdc45f3f6d"

P4S_AUTHORITY_PATH = "reports/reason_router_p2_p3w6f2_p4s_source_semantic_recovery_validation_correction_execution_authority_spec.md"
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
    raise SystemExit(f"P4S_FAIL:{code}")


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

authority_freeze = os.environ.get("P4S_AUTHORITY_FREEZE", "")
require(re.fullmatch(r"[0-9a-f]{40}", authority_freeze or "") is not None, "P4S_AUTHORITY_FREEZE_INVALID")
run_git(["cat-file", "-e", f"{authority_freeze}^{{commit}}"])
authority_listing = run_git(["ls-tree", "-r", "--name-only", authority_freeze, "--", P4S_AUTHORITY_PATH])
require(authority_listing == P4S_AUTHORITY_PATH, "P4S_AUTHORITY_PATH_MISSING_AT_FREEZE")

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
source_physical_before = physical_sha(source_path)
require(sidecar_physical_before == EXPECTED_SIDECAR_PHYSICAL, "SIDECAR_RECOVERY_INPUT_SHA_MISMATCH")
require(provenance_physical_before == EXPECTED_PROVENANCE_PHYSICAL, "PROVENANCE_RECOVERY_INPUT_SHA_MISMATCH")
require(source_physical_before == EXPECTED_SOURCE_PHYSICAL, "SOURCE_PHYSICAL_SHA_MISMATCH")
require(physical_sha(p4b_rows_path) == EXPECTED_P4B_ROWS_PHYSICAL, "P4B_ROWS_PHYSICAL_SHA_MISMATCH")
require(physical_sha(p4b_summary_path) == EXPECTED_P4B_SUMMARY_PHYSICAL, "P4B_SUMMARY_PHYSICAL_SHA_MISMATCH")
require(physical_sha(p4b_provenance_path) == EXPECTED_P4B_PROVENANCE_PHYSICAL, "P4B_PROVENANCE_PHYSICAL_SHA_MISMATCH")
require(physical_sha(builder_source_path) == BUILDER_SOURCE_SHA256, "BUILDER_SOURCE_SHA_MISMATCH")

sidecar_rows = load_jsonl(sidecar_path, 3600, "SIDECAR")
source_rows = load_jsonl(source_path, 3600, "SOURCE")
provenance, provenance_bytes = load_json_object(provenance_path, "PROVENANCE")

source_projection = [
    {
        key: row[key]
        for key in SOURCE_SEMANTIC_KEYS
    }
    for row in source_rows
]
# Byte-for-byte equivalent to builder semantic_dataset_sha256()/canonical_sha256().
source_semantic_bytes = json.dumps(
    source_projection,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
    allow_nan=False,
).encode("utf-8")
source_semantic_sha = sha256(source_semantic_bytes).hexdigest()
require(source_semantic_sha == EXPECTED_SOURCE_SEMANTIC, "SOURCE_SEMANTIC_SHA_MISMATCH")

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

semantic_rows = [
    {
        key: row[key]
        for key in sorted(row)
        if key != "created_at"
    }
    for row in sidecar_rows
]
# Byte-for-byte equivalent to builder semantic_sidecar_sha256()/canonical_sha256().
sidecar_semantic_bytes = json.dumps(
    semantic_rows,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
    allow_nan=False,
).encode("utf-8")
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
require(physical_sha(source_path) == EXPECTED_SOURCE_PHYSICAL, "SOURCE_PHYSICAL_SHA_CHANGED")
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
    "source_physical_sha256": source_physical_before,
    "source_semantic_sha256": source_semantic_sha,
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

## 10. Corrected Whole-List Source Semantic Algorithm

The executable validator must compute exactly:

```python
source_projection = [
    {
        key: row[key]
        for key in SOURCE_SEMANTIC_KEYS
    }
    for row in source_rows
]

source_semantic_bytes = json.dumps(
    source_projection,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
    allow_nan=False,
).encode("utf-8")

source_semantic_sha256 = sha256(source_semantic_bytes).hexdigest()
```

Then require:

`source_semantic_sha256 == 3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b`

This is byte-for-byte equivalent to builder
`semantic_dataset_sha256()` / `canonical_sha256()`.

## 11. Fail-Closed Row Semantics

The corrected validator uses `row[key]`, not `row.get(key)`. Missing semantic
fields fail closed. It preserves physical source row order and preserves the
exact `SOURCE_SEMANTIC_KEYS` projection:

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

## 12. No-Newline/No-JSONL Proof

The corrected top-level object is a JSON list serialized once. There is:

- no per-row newline
- no newline between rows
- no final newline
- no JSONL semantic hashing
- no concatenation of independently serialized row objects
- no row sorting

`sort_keys=True` applies inside JSON objects only.

## 13. Source Physical/Semantic Checks

Exact source path:

`reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl`

Required source physical SHA:

`eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`

Required corrected source semantic SHA:

`3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b`

Historical `f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640`
is not accepted as current identity.

## 14. Provenance Source Bindings

After successful corrected recomputation, canonical provenance must require:

- `source_dataset_path` equals the exact P4-B R1 path.
- `source_dataset_sha256 ==
  eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`.
- `source_dataset_semantic_sha256 ==
  3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b`.
- `source_physical_sha256 ==
  eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`.
- `source_semantic_sha256 ==
  3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b`.
- `historical_stage185_used_as_current_source_identity is False`.

## 15. P4-B Bindings/Scope

P4-S preserves P4-R's repaired executable assertions for exact:

- `p4b_compatibility_rows_path`
- `p4b_compatibility_rows_sha256`
- `p4b_compatibility_summary_path`
- `p4b_compatibility_summary_sha256`
- `p4b_compatibility_provenance_path`
- `p4b_compatibility_provenance_sha256`

It independently recomputes external P4-B hashes:

- rows:
  `59e4367d29e3c49152049e4a1b46e8783d5b81d1ebaa931ca9fd4ae0ac967b9f`
- summary:
  `ce618d214dc4d660706d927a0d91ec5945d3a0edbffbf615eab8b9c9ff585aa8`
- provenance:
  `09a56f1dca325e0749a0e4b0f822d68dad5d85fd90fa427e0efd6d617d41b2d6`

Require `p4b_compatibility_authorized_scope` is a dict, `pair_count == 119`,
and `member_count == 357`.

## 16. Preservation of Other P4-R Assertions

P4-S preserves all other P4-R read-only assertions, including:

- recovery physical-hash preconditions
- authority/execution identity
- builder SHA read-only check
- corrected exact-two-file check
- repo containment/symlink checks
- sidecar JSONL serialization checks
- provenance deterministic pretty serialization
- explicit non-finite rejection
- source physical SHA
- 3600/order/schema checks
- split replay
- canonical replay
- reason replay
- positive-margin replay
- unresolved reporting
- sidecar physical SHA
- sidecar semantic SHA removing `created_at` only
- external provenance physical SHA
- no provenance recursive self-hash
- all P4-L provenance identities/intrinsic flags
- postconditions
- no retroactive P4-Q/P4-R PASS

## 17. PASS Token

Only after every corrected recovery validation assertion and postcondition
passes, print:

`P3W6F2P4S_SOURCE_SEMANTIC_RECOVERY_VALIDATION_CORRECTION_PASS`

Immediately before token print, print compact observations JSON including:

- Python version
- `sys.platform`
- `CUDA_VISIBLE_DEVICES`
- source physical SHA
- corrected recomputed source semantic SHA
- sidecar physical SHA
- sidecar semantic SHA
- provenance physical SHA
- integrity status counts
- positive-margin counts
- reason-supervision eligibility counts
- unresolved count/status/reason summary

## 18. Postconditions

Recheck after validator:

- HEAD exact `2f9e6076791358922e3ebd70e89533d9cb83b458`.
- Tracked worktree clean.
- Index clean.
- Canonical directory non-symlink.
- Exact same two files.
- Sidecar physical SHA unchanged.
- Provenance physical SHA unchanged.
- Source physical SHA unchanged.
- No `.p4l-staging-*` sibling.
- GPU OFF.

No cleanup on PASS or FAIL.

## 19. PASS/BLOCKED/FAIL Mapping

PASS:

All corrected read-only assertions and postconditions pass.

BLOCKED:

- P4-S authority unavailable.
- HEAD/state mismatch.
- Safe read-only validation unavailable.
- GPU cannot remain off.

FAIL:

- Immutable recovery-input physical hashes differ.
- Correct builder-equivalent source semantic hash differs from frozen expected.
- Artifact serialization/order/schema mismatch.
- Split/canonical/reason/P4-B mismatch.
- Sidecar/provenance hash or identity mismatch.
- Artifact mutation detected.

Do not classify frozen P4-R's known JSONL-style source-semantic algorithm defect
as an artifact FAIL.

## 20. Interpretation Boundary

On P4-S PASS:

1. code correctness = PASS
2. POSIX publication runtime = PASS
3. P4-Q Gate 1 materialization = PASS
4. P4-Q overall = FAIL due wrapper defect
5. P4-R overall = FAIL due source-semantic validator defect
6. canonical P4-L artifact/provenance integrity = ESTABLISHED
7. trainer rebind = NOT AUTHORIZED
8. A0 = NOT AUTHORIZED
9. training/evaluation = NOT AUTHORIZED
10. scientific conclusion = NOT ESTABLISHED

P4-S must not retroactively relabel P4-Q or P4-R as PASS.

## 21. Authority Flags

Candidate-time flags are all false:

- `recovery_validation_execution_authorized = false`
- `artifact_materialization_authorized = false`
- `kaggle_authorized = false`
- `cpu_authorized = false`
- `gpu_authorized = false`
- `trainer_rebind_authorized = false`
- `a0_execution_authorized = false`
- `training_admission_released = false`
- `training_authorized = false`
- `evaluation_authorized = false`

After independent verification plus immutable P4-S freeze, only the exact P4-S
run may set:

- `recovery_validation_execution_authorized = true`
- `kaggle_authorized = true`
- `cpu_authorized = true`
- `gpu_authorized = false`

Artifact materialization remains false. All downstream flags remain false.

## 22. Evidence Boundary

Future evidence namespace:

`reports/reason_router_p2_p3w6f2_p4s_source_semantic_recovery_validation_correction_execution_2f9e6076791358922e3ebd70e89533d9cb83b458/`

Future evidence must preserve:

- P4-S authority freeze
- P4-R authority freeze
- P4-Q authority freeze
- exact P4-R failure token
- static root-cause proof
- corrected source semantic algorithm
- corrected sidecar semantic algorithm
- immutable input hashes
- exact P4-S validator output/token
- post-state

No evidence is created now. No ZIP is created now. No run ID is invented. No
new artifact hash is predicted.

## 23. Scientific Boundary

P4-S changes nothing about:

- Reason Router mechanism
- supervision semantics
- gradient ownership
- primary reason ordering
- secondary diagnostic semantics
- router-only CE/detach
- EMA
- A0-A3
- E0

No trainer/model/config/checkpoint modification, training, or evaluation is
authorized.

## 24. Independent Verification Requirement

Because P4-S can establish canonical artifact/provenance validity, independent
static verification is mandatory before freeze.

Verifier must compare the corrected source and sidecar semantic implementations
directly against exact builder commit
`2f9e6076791358922e3ebd70e89533d9cb83b458`, not against prose alone.

Specifically verify:

- builder:
  `semantic_dataset_sha256(rows) == canonical_sha256([ordered_dataset_row(row) for row in rows])`
- P4-S validator:
  whole ordered projected list serialized exactly once using the same canonical
  JSON settings
- builder:
  `semantic_sidecar_sha256(rows) == canonical_sha256([{key: row[key] for key in sorted(row) if key != "created_at"} for row in rows])`
- P4-S validator:
  whole ordered sidecar list with only `created_at` removed, serialized exactly
  once using the same canonical JSON settings

Reject:

- JSONL
- newline-concatenated rows
- row-by-row hashing
- row sorting
- `row.get` defaults
- omitted fields
- final newline

Also recheck all other P4-R assertions for regressions.

## 25. Stop Conditions

BLOCK candidate creation if:

- local HEAD/state mismatch
- namespace collision
- correction cannot be made exact
- existing bytes need mutation
- builder/materialization would be required
- GPU would be required

Commit/push: NO.

Training/evaluation: NO.

## 26. Candidate Path

Candidate path:

`reports/reason_router_p2_p3w6f2_p4s_source_semantic_recovery_validation_correction_execution_authority_spec.md`

Expected delta:

Exactly one new untracked authority spec.

No Python, pytest, builder, materialization, Kaggle execution, recovery
validation execution, staging, commit, or push is authorized during candidate
creation.

## 27. Candidate SHA256

Candidate SHA256 is computed after file creation with a read-only filesystem
hash command and reported in the candidate creation summary. The hash is a
property of this authority-spec file only; it is not a predicted artifact hash
and does not authorize execution.

## 28. Git Diff Check

`git diff --check` is required after creation. It is a static whitespace check
only and does not stage, commit, execute Python, run pytest, run a builder,
materialize artifacts, or validate recovery.

## 29. Blockers

Current candidate-creation blockers: none identified.

Independent static verification is required before freeze.

## 30. Final Git Status

Final git status must show the candidate as the only newly created P4-S
authority spec, with pre-existing unrelated untracked files untouched and no
tracked/index changes.

Success token:

`P3W6F2P4S_SOURCE_SEMANTIC_RECOVERY_VALIDATION_CORRECTION_EXECUTION_AUTHORITY_CANDIDATE_READY_FOR_INDEPENDENT_VERIFICATION`
