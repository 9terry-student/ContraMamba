# P3-W6-F2-P4-U Current-Source Provenance Schema Correction Execution Authority Candidate

Authority/version:

`P3W6F2P4U_CURRENT_SOURCE_PROVENANCE_SCHEMA_CORRECTION_EXECUTION_AUTHORITY_CANDIDATE_V1`

This document is a bounded execution-authority candidate only. Candidate
creation itself does not authorize Python execution, pytest, Kaggle execution,
CPU execution, GPU execution, builder execution, materialization, artifact
mutation, training, evaluation, staging, commit, push, or retroactive relabeling
of historical P4-T.

Only after independent static verification PASS and immutable P4-U freeze may
this authority authorize exactly one future CPU-only, read-only validation of
the existing canonical P4-L bytes.

## 1. Authority Basis

Creation authority:

- Current controller instruction.
- Frozen P4-U candidate:
  `4a5d2a582848bc64a6742d4c81946cd276777a74`.
- Frozen candidate SHA256:
  `558fb710f97a53fa27454e02a8993393cda946a13b1c1747a6a017e9fd792746`.
- P4-T freeze:
  `89cc8ad374b1b9656e2a7333a4fa412916e007c9`.
- Exact execution HEAD:
  `2f9e6076791358922e3ebd70e89533d9cb83b458`.
- `AGENTS.md`.

Phase:

`P4-U EXECUTION AUTHORITY CANDIDATE CREATION ONLY`

## 2. Historical Disposition

P4-T remains FAIL with exact observed token:

`P4T_FAIL:PROVENANCE_SOURCE_PHYSICAL_SHA256_MISMATCH`

P4-U does not retroactively relabel P4-T PASS and does not assert that the
canonical P4-L artifact is valid until the future authorized validator actually
prints the P4-U PASS token after freeze.

The P4-T failure is preserved as a validator-schema failure: P4-T inherited
nonexistent current-source alias requirements while correcting the P4-B
119/357 schema defect.

## 3. Scope

P4-U starts from the P4-T validator semantics and preserves every valid P4-T
check, including:

- corrected P4-B 119/357 checks;
- P4-B rows, summary, and provenance physical hashes;
- source semantic algorithm;
- sidecar semantic algorithm;
- sidecar physical and semantic hashes;
- provenance physical hash;
- read-only boundaries;
- exact detached execution HEAD
  `2f9e6076791358922e3ebd70e89533d9cb83b458`;
- CPU-only/GPU-off constraints;
- historical P4-T FAIL status.

P4-U corrects only:

- authority identity, environment variable, authority path, and PASS token to
  the P4-U namespace;
- current-source provenance bindings to exactly:
  `source_dataset_path`, `source_dataset_sha256`,
  `source_dataset_semantic_sha256`.

P4-U must not require the alias provenance keys `source_physical_sha256` or
`source_semantic_sha256`; absence of either alias must not be treated as a
failure.

## 4. Runtime Authority Boundary

After independent verification PASS and immutable P4-U freeze, the workflow
controller must supply exactly one runtime parameter:

`P4U_AUTHORITY_FREEZE`

The future validator must require:

- `P4U_AUTHORITY_FREEZE` is set.
- `P4U_AUTHORITY_FREEZE` is exactly 40 lowercase hex characters.
- `git cat-file -e "${P4U_AUTHORITY_FREEZE}^{commit}"` succeeds.
- this P4-U execution-authority path exists at that commit.
- the validator does not check out, fetch, reset, clean, or otherwise
  materialize the P4-U freeze.
- `git rev-parse HEAD ==
  2f9e6076791358922e3ebd70e89533d9cb83b458`.
- CPU-only operation; GPU disabled before and after.

The sole future validation is read-only. It must not write, unlink, rename,
replace, chmod, clean, move-aside, create repository artifact paths, invoke the
builder, run training, or run evaluation.

## 5. Exact Future Kaggle Validator Command

The complete exact future Kaggle validator command is:

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

AUTHORITY_VERSION = "P3W6F2P4U_CURRENT_SOURCE_PROVENANCE_SCHEMA_CORRECTION_EXECUTION_AUTHORITY_V1"
PASS_TOKEN = "P3W6F2P4U_CURRENT_SOURCE_PROVENANCE_SCHEMA_CORRECTION_PASS"

EXECUTION_HEAD = "2f9e6076791358922e3ebd70e89533d9cb83b458"
P4U_SCHEMA_CORRECTION_FREEZE = "4a5d2a582848bc64a6742d4c81946cd276777a74"
P4T_AUTHORITY_FREEZE = "89cc8ad374b1b9656e2a7333a4fa412916e007c9"
P4R_AUTHORITY_FREEZE = "d80ed289273763c1c90f2fba14ca796c604c9529"
P4Q_AUTHORITY_FREEZE = "79ec5fa764de30eaa04fb6de4c2a8228edf1a63a"
P4L_AUTHORITY_COMMIT = "80cb034792f03226cf6e22c196c1229ed4e6dd62"
BUILDER_SOURCE_SHA256 = "b7fad3672428c8199347c278be0521e8355a8ca1c1491b5d3036e6cdc45f3f6d"

P4U_AUTHORITY_PATH = "reports/reason_router_p2_p3w6f2_p4u_current_source_provenance_schema_correction_execution_authority_spec.md"
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

P4B_COMPATIBILITY_RULE_VERSION = "P3W6F2P4B_R1_STAGE185_PREDICATE_REALIZATION_COMPATIBILITY_V1"

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
    raise SystemExit(f"P4U_FAIL:{code}")


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

authority_freeze = os.environ.get("P4U_AUTHORITY_FREEZE", "")
require(re.fullmatch(r"[0-9a-f]{40}", authority_freeze or "") is not None, "P4U_AUTHORITY_FREEZE_INVALID")
run_git(["cat-file", "-e", f"{authority_freeze}^{{commit}}"])
authority_listing = run_git(["ls-tree", "-r", "--name-only", authority_freeze, "--", P4U_AUTHORITY_PATH])
require(authority_listing == P4U_AUTHORITY_PATH, "P4U_AUTHORITY_PATH_MISSING_AT_FREEZE")

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
require(len(p4b_rows) == 357, "P4B_MEMBER_SCOPE_MISMATCH")
require(len({row.get("member_id") for row in p4b_rows}) == 357, "P4B_MEMBER_ID_SCOPE_MISMATCH")
require(len({row.get("pair_id") for row in p4b_rows}) == 119, "P4B_PAIR_SCOPE_MISMATCH")
for row in p4b_rows:
    require(row.get("schema_version") == "P3W6F2P4B_R1_STAGE185_COMPATIBILITY_ROW_V1", "P4B_ROW_SCHEMA_VERSION_MISMATCH")
    require(row.get("compatibility_rule_version") == P4B_COMPATIBILITY_RULE_VERSION, "P4B_ROW_RULE_VERSION_MISMATCH")
    require(row.get("effective_compatibility_status") == "PASS", "P4B_ROW_STATUS_MISMATCH")
    effect = row.get("training_admission_effect")
    require(isinstance(effect, dict), "P4B_ROW_TRAINING_ADMISSION_EFFECT_NOT_OBJECT")
    require(effect.get("training_admission_released") is False, "P4B_ROW_TRAINING_ADMISSION_RELEASED")
    require(isinstance(row.get("member_id"), str) and row.get("member_id"), "P4B_ROW_MEMBER_ID_EMPTY")
    require(isinstance(row.get("pair_id"), str) and row.get("pair_id"), "P4B_ROW_PAIR_ID_EMPTY")

require(p4b_summary.get("schema_version") == "P3W6F2P4B_R1_STAGE185_COMPATIBILITY_SUMMARY_V1", "P4B_SUMMARY_SCHEMA_VERSION_MISMATCH")
require(p4b_summary.get("compatibility_rule_version") == P4B_COMPATIBILITY_RULE_VERSION, "P4B_SUMMARY_RULE_VERSION_MISMATCH")
require(p4b_summary.get("row_count") == 357, "P4B_SUMMARY_ROW_COUNT_MISMATCH")
require(p4b_summary.get("authorized_pair_count") == 119, "P4B_SUMMARY_AUTHORIZED_PAIR_COUNT_MISMATCH")
require(p4b_summary.get("authorized_member_count") == 357, "P4B_SUMMARY_AUTHORIZED_MEMBER_COUNT_MISMATCH")
require(p4b_summary.get("compatibility_gate_status") == "PASS", "P4B_SUMMARY_GATE_STATUS_MISMATCH")
require(p4b_summary.get("training_admission_released") is False, "P4B_SUMMARY_TRAINING_ADMISSION_RELEASED")

require(p4b_provenance.get("schema_version") == "P3W6F2P4B_R1_STAGE185_COMPATIBILITY_PROVENANCE_V1", "P4B_PROVENANCE_SCHEMA_VERSION_MISMATCH")
require(p4b_provenance.get("compatibility_rows_sha256") == EXPECTED_P4B_ROWS_PHYSICAL, "P4B_PROVENANCE_ROWS_SHA_MISMATCH")
require(p4b_provenance.get("compatibility_summary_sha256") == EXPECTED_P4B_SUMMARY_PHYSICAL, "P4B_PROVENANCE_SUMMARY_SHA_MISMATCH")
require(p4b_provenance.get("regenerated_dataset_path") == SOURCE_REL, "P4B_PROVENANCE_REGENERATED_DATASET_PATH_MISMATCH")
require(p4b_provenance.get("regenerated_dataset_sha256") == EXPECTED_SOURCE_PHYSICAL, "P4B_PROVENANCE_REGENERATED_DATASET_SHA_MISMATCH")
require(p4b_provenance.get("stage185_source_script_sha256") == "11e6ba89b8131c76eac4504b4273867eaa99a131abe23d3238eb65ecda207bbc", "P4B_PROVENANCE_STAGE185_SOURCE_SCRIPT_SHA_MISMATCH")

semantic_rows = [
    {
        key: row[key]
        for key in sorted(row)
        if key != "created_at"
    }
    for row in sidecar_rows
]
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
    "authority_freeze": authority_freeze,
    "authority_version": AUTHORITY_VERSION,
    "execution_head": EXECUTION_HEAD,
    "python_version": sys.version,
    "sys_platform": sys.platform,
    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
    "p4b_summary_schema_version": p4b_summary.get("schema_version"),
    "p4b_summary_row_count": p4b_summary.get("row_count"),
    "p4b_summary_authorized_pair_count": p4b_summary.get("authorized_pair_count"),
    "p4b_summary_authorized_member_count": p4b_summary.get("authorized_member_count"),
    "p4b_provenance_schema_version": p4b_provenance.get("schema_version"),
    "p4b_compatibility_row_count": len(p4b_rows),
    "p4b_compatibility_unique_pair_count": len({row.get("pair_id") for row in p4b_rows}),
    "canonical_p4l_p4b_authorized_scope": scope,
    "source_dataset_path": provenance.get("source_dataset_path"),
    "source_dataset_sha256": provenance.get("source_dataset_sha256"),
    "source_dataset_semantic_sha256": provenance.get("source_dataset_semantic_sha256"),
    "source_dataset_semantic_recomputed_sha256": source_semantic_sha,
    "sidecar_physical_sha256": sidecar_physical_before,
    "sidecar_semantic_sha256": sidecar_semantic_sha,
    "provenance_physical_sha256": provenance_physical_before,
    "integrity_status_counts": dict(sorted(integrity_counts.items())),
    "positive_margin_counts": dict(sorted(positive_margin_counts.items())),
    "p2_reason_supervision_eligible_counts": dict(sorted(reason_supervision_counts.items())),
    "unresolved_row_count": unresolved_count,
    "unresolved_status_summary": dict(sorted(unresolved_status_counts.items())),
    "unresolved_reason_summary": dict(sorted(unresolved_reason_counts.items())),
    "post_head": run_git(["rev-parse", "HEAD"]),
    "post_tracked_state": run_git(["status", "--short", "--untracked-files=no"]),
    "post_index_state": run_git(["diff", "--cached", "--name-status"]),
}
print(json.dumps(observations, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
print(PASS_TOKEN)
PY
```

## 6. Required Current-Source Provenance Binding

The embedded validator requires exactly these current-source provenance fields:

```python
required_provenance = {
    ...
    "source_dataset_path": SOURCE_REL,
    "source_dataset_sha256": EXPECTED_SOURCE_PHYSICAL,
    "source_dataset_semantic_sha256": EXPECTED_SOURCE_SEMANTIC,
    ...
}
```

No executable assertion requires `source_physical_sha256` or
`source_semantic_sha256`.

## 7. P4-B 119/357 Preservation

The embedded validator preserves the P4-T corrected P4-B checks:

- rows physical SHA:
  `59e4367d29e3c49152049e4a1b46e8783d5b81d1ebaa931ca9fd4ae0ac967b9f`;
- summary physical SHA:
  `ce618d214dc4d660706d927a0d91ec5945d3a0edbffbf615eab8b9c9ff585aa8`;
- provenance physical SHA:
  `09a56f1dca325e0749a0e4b0f822d68dad5d85fd90fa427e0efd6d617d41b2d6`;
- exactly 357 compatibility row objects;
- exactly 119 unique non-empty `pair_id` values;
- exactly 357 unique non-empty `member_id` values;
- summary `row_count == 357`;
- summary `authorized_pair_count == 119`;
- summary `authorized_member_count == 357`;
- canonical provenance `p4b_compatibility_authorized_scope` has
  `pair_count == 119` and `member_count == 357`.

The embedded validator does not use generic fallback P4-B field assertions such
as `summary.get("pair_count", summary.get("pairs"))`.

## 8. PASS Token And Interpretation

Only after every read-only assertion and postcondition passes may the future
validator print:

`P3W6F2P4U_CURRENT_SOURCE_PROVENANCE_SCHEMA_CORRECTION_PASS`

On P4-U PASS:

- canonical P4-L artifact/provenance integrity is established only for the
  frozen P4-L contract and immutable canonical bytes;
- P4-T remains historical FAIL;
- trainer rebind remains NOT AUTHORIZED;
- A0 remains NOT AUTHORIZED;
- training remains NOT AUTHORIZED;
- evaluation remains NOT AUTHORIZED;
- scientific conclusion remains NOT ESTABLISHED.

## 9. Candidate Creation Boundary

Candidate creation final state requires:

- HEAD unchanged:
  `4a5d2a582848bc64a6742d4c81946cd276777a74`;
- tracked worktree clean;
- index clean;
- exactly one new untracked P4-U execution-authority spec;
- frozen P4-U candidate unchanged;
- no Python, pytest, builder, materialization, validation, training,
  evaluation, Kaggle, commit, or push.

Candidate path:

`reports/reason_router_p2_p3w6f2_p4u_current_source_provenance_schema_correction_execution_authority_spec.md`

Independent static verification is required before immutable freeze. Candidate
creation itself does not authorize execution; only independent verification
PASS plus immutable freeze does.

Success token:

`P4U_EXECUTION_AUTHORITY_CANDIDATE_READY_FOR_VERIFICATION`
