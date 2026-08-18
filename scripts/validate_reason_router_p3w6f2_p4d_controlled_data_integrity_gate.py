#!/usr/bin/env python3
"""Read-only P3-W6-F2 P4-D controlled-data integrity Gate 5 validator."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, replace
import hashlib
import json
import random
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence


P4D_SPEC_AUTHORITY_COMMIT = "1be4050dbadb0cb5ed2f6b55a2391051f5f6c07e"
P4B_PARENT_AUTHORITY_COMMIT = "fcc3b9ccaf2bbee33ac18dcef10d50acff54aab4"
P4B_EXECUTION_COMMIT = "4122078ab7962042e3d6bf89f8b4eb5cec463458"
REPORT_SCHEMA_VERSION = "P3W6F2P4D_CONTROLLED_DATA_INTEGRITY_GATE_REPORT_V1"
VALIDATOR_CONTRACT_VERSION = "P3W6F2P4D_CONTROLLED_DATA_INTEGRITY_GATE_VALIDATOR_V1"

PASS_TOKEN = "P3W6F2P4D_CONTROLLED_DATA_INTEGRITY_GATE_PASS"
FAIL_TOKEN = "P3W6F2P4D_CONTROLLED_DATA_INTEGRITY_GATE_FAIL"
BLOCKED_TOKEN = "P3W6F2P4D_CONTROLLED_DATA_INTEGRITY_GATE_BLOCKED"

HISTORICAL_DATASET_PATH = "data/controlled_v5_v3_without_time_swap.jsonl"
HISTORICAL_DATASET_SHA256 = "f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640"
P4B_ARTIFACT_DIR = (
    "reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_"
    "4122078ab7962042e3d6bf89f8b4eb5cec463458"
)
REGENERATED_DATASET_NAME = "controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl"
REGENERATED_DATASET_SHA256 = "eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3"
REGENERATED_DATASET_SEMANTIC_SHA256 = "3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b"
STAGE185_SOURCE_SCRIPT = "scripts/build_stage185a_controlled_train_integrity_sidecar.py"
STAGE185_SOURCE_SHA256 = "11e6ba89b8131c76eac4504b4273867eaa99a131abe23d3238eb65ecda207bbc"
STRUCTURED_SOURCE_PRODUCER = "scripts/build_controlled_v5.py::fact_templates_for_count"
STRUCTURED_SOURCE_PATH = "scripts/build_controlled_v5.py"
STRUCTURED_SOURCE_SHA256 = "9fbd94a151c4d83a5e824412d7c0837062fedd20628f4f198116b2d08b679410"
PREDICATE_MAPPING_SHA256 = "617ce712753bc09282b4bb6792154cbe1daa7713021895160c01ce1a839cc309"
STAGE185_SPLIT_SEED = 174
STAGE185_DEV_RATIO = 0.2

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
LABEL_FIELDS = (
    "final_label",
    "frame_compatible_label",
    "predicate_covered_label",
    "sufficiency_label",
    "polarity_label",
    "primary_failure_type",
)
ADMITTED_FAMILIES = frozenset(
    {
        "entity_swap",
        "event_swap",
        "evidence_deletion",
        "evidence_truncation",
        "irrelevant_evidence",
        "location_swap",
        "none",
        "paraphrase",
        "polarity_flip",
        "predicate_swap",
        "role_swap",
        "title_name_swap",
    }
)
FINAL_LABELS = {"SUPPORT", "NOT_ENTITLED", "REFUTE"}
POLARITY_LABELS = {"SUPPORT", "NONE", "REFUTE"}
PRIMARY_FAILURE_TYPES = {"none", "frame", "predicate", "sufficiency", "polarity"}
AUTHORIZED_F2_INTERVENTIONS = frozenset({"none", "paraphrase", "polarity_flip"})
AUTHORIZED_CHANGED_INTERVENTIONS = frozenset({"none", "paraphrase"})
AUTHORIZED_PAIR_COUNT = 119
AUTHORIZED_MEMBER_COUNT = 357
AUTHORIZED_CHANGED_ROW_COUNT = 238
PAIR_COUNT = 300
ROWS_PER_PAIR = 12
ROW_COUNT = 3600

MEMBERS_NAME = "p3w6f2_p4b_r1_regenerated_members.jsonl"
AUDIT_NAME = "p3w6f2_p4b_r1_regeneration_audit.jsonl"
SUMMARY_NAME = "p3w6f2_p4b_r1_regeneration_summary.json"
ISOLATION_NAME = "p3w6f2_p4b_r1_full_output_isolation.json"
INVOCATION_NAME = "p3w6f2_p4b_r1_deterministic_generator_invocation.json"
COVERAGE_NAME = "p3w6f2_p4b_r1_base_form_coverage.json"
COMPAT_ROWS_NAME = "p3w6f2_p4b_r1_stage185_predicate_realization_compatibility_rows.jsonl"
COMPAT_SUMMARY_NAME = "p3w6f2_p4b_r1_stage185_predicate_realization_compatibility_summary.json"
COMPAT_PROVENANCE_NAME = "p3w6f2_p4b_r1_stage185_predicate_realization_compatibility_provenance_manifest.json"

P4B_ARTIFACT_HASHES = {
    REGENERATED_DATASET_NAME: REGENERATED_DATASET_SHA256,
    MEMBERS_NAME: "3dd91d6e2888d50ccd45acb8243d2b8d47bd3476e2a76f5c2b9e7cd93b82bbf3",
    AUDIT_NAME: "17eaae3e20779fc6bbfe730222bd4e410d5bd03b108aaf7ea98214eaeb8d77a1",
    SUMMARY_NAME: "e09bcd09207d78a211a4b63a94af8db2a93b6c4c6b1d618e2a673168618f1157",
    ISOLATION_NAME: "c4b342d200757b4e330fd7b4bfb1b5550b3c74933bca5c99323eeac9c87ebb7e",
    INVOCATION_NAME: "2cbd5057cf89c3ba0a01bbaa1d6168b3bd595a1704bbcec50902de44066494f0",
    COVERAGE_NAME: "5d130c529e0ebcca9fa3f7137620222488a9eb8d2db137e5a7283c345a277bb3",
    COMPAT_ROWS_NAME: "59e4367d29e3c49152049e4a1b46e8783d5b81d1ebaa931ca9fd4ae0ac967b9f",
    COMPAT_SUMMARY_NAME: "ce618d214dc4d660706d927a0d91ec5945d3a0edbffbf615eab8b9c9ff585aa8",
    COMPAT_PROVENANCE_NAME: "09a56f1dca325e0749a0e4b0f822d68dad5d85fd90fa427e0efd6d617d41b2d6",
}
EXPECTED_ARTIFACT_NAMES = tuple(P4B_ARTIFACT_HASHES)

COMPAT_ROW_FIELDS = (
    "schema_version",
    "compatibility_rule_version",
    "pair_id",
    "member_id",
    "intervention_type",
    "raw_stage185_changed_axes",
    "raw_stage185_expected_axes",
    "raw_stage185_statuses",
    "historical_semantic_predicate",
    "regenerated_negative_base_surface",
    "structured_fact",
    "semantic_slot_preservation",
    "permitted_predicate_realization_delta",
    "effective_compatibility_status",
    "effective_reason_codes",
    "training_admission_effect",
)
COMPAT_SUMMARY_FIELDS = (
    "schema_version",
    "compatibility_rule_version",
    "row_count",
    "authorized_pair_count",
    "authorized_member_count",
    "raw_stage185_predicate_axis_observation_count",
    "permitted_predicate_realization_delta_count",
    "compatibility_pass_count",
    "compatibility_fail_count",
    "compatibility_unresolved_count",
    "stage185_v1_mutated",
    "historical_authority_weakened",
    "training_admission_released",
    "compatibility_gate_status",
    "failure_reasons",
)
COMPAT_PROVENANCE_FIELDS = (
    "schema_version",
    "compatibility_rule_version",
    "stage185_source_script",
    "stage185_source_script_sha256",
    "historical_stage185_authority",
    "historical_stage185_authority_sha256",
    "regenerated_dataset_path",
    "regenerated_dataset_sha256",
    "structured_source_producer",
    "structured_source_producer_sha256",
    "base_form_coverage_path",
    "base_form_coverage_sha256",
    "compatibility_rows_path",
    "compatibility_rows_sha256",
    "compatibility_summary_path",
    "compatibility_summary_sha256",
    "created_at_utc",
)

STATUS_FIELDS = (
    "schema_status",
    "row_id_status",
    "row_order_status",
    "pair_topology_status",
    "time_swap_status",
    "split_replay_status",
    "split_identity_status",
    "label_integrity_status",
    "canonical_linkage_status",
    "delta_isolation_status",
    "polarity_flip_status",
    "non_f2_identity_status",
    "compatibility_artifact_status",
    "raw_stage185_observation_status",
    "historical_stage185_immutability_status",
    "determinism_status",
    "provenance_status",
)


class GateBlocked(RuntimeError):
    """A frozen P4-D controlled-data integrity requirement failed closed."""


def require(condition: bool, code: str) -> None:
    if not condition:
        raise GateBlocked(code)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GateBlocked(f"DUPLICATE_JSON_KEY:{key}")
        result[key] = value
    return result


def reject_json_constant(value: str) -> None:
    raise GateBlocked(f"NON_STANDARD_JSON_CONSTANT:{value}")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def validate_text_bytes(path: Path, *, jsonl: bool) -> bytes:
    require(path.is_file(), f"INPUT_MISSING:{path}")
    payload = path.read_bytes()
    try:
        payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise GateBlocked(f"NON_UTF8:{path}") from exc
    require(b"\r" not in payload, f"NON_LF_LINE_ENDING:{path}")
    if payload:
        require(payload.endswith(b"\n"), f"MISSING_FINAL_LF:{path}")
    if jsonl:
        require(all(line.strip() for line in payload.splitlines()), f"BLANK_JSONL_LINE:{path}")
    return payload


def load_json(path: Path) -> dict[str, Any]:
    payload = validate_text_bytes(path, jsonl=False)
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=reject_json_constant,
        )
    except GateBlocked:
        raise
    except Exception as exc:
        raise GateBlocked(f"MALFORMED_JSON:{path}") from exc
    require(isinstance(value, dict), f"JSON_NOT_OBJECT:{path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    payload = validate_text_bytes(path, jsonl=True)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
        try:
            value = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=reject_json_constant,
            )
        except GateBlocked:
            raise
        except Exception as exc:
            raise GateBlocked(f"MALFORMED_JSONL:{path}:{line_number}") from exc
        require(isinstance(value, dict), f"JSONL_ROW_NOT_OBJECT:{path}:{line_number}")
        rows.append(value)
    return rows


def deterministic_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def canonical_dataset_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    ordered = [{field: row[field] for field in DATASET_FIELDS} for row in rows]
    payload = json.dumps(ordered, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256_bytes(payload.encode("utf-8"))


@dataclass(frozen=True)
class FrozenInputs:
    historical_dataset_path: str = HISTORICAL_DATASET_PATH
    historical_dataset_sha256: str = HISTORICAL_DATASET_SHA256
    regenerated_dataset_name: str = REGENERATED_DATASET_NAME
    regenerated_dataset_sha256: str = REGENERATED_DATASET_SHA256
    regenerated_dataset_semantic_sha256: str = REGENERATED_DATASET_SEMANTIC_SHA256
    p4b_artifact_dir: str = P4B_ARTIFACT_DIR
    p4b_artifact_hashes: Mapping[str, str] | None = None
    stage185_source_script: str = STAGE185_SOURCE_SCRIPT
    stage185_source_script_sha256: str = STAGE185_SOURCE_SHA256
    structured_source_path: str = STRUCTURED_SOURCE_PATH
    structured_source_producer: str = STRUCTURED_SOURCE_PRODUCER
    structured_source_sha256: str = STRUCTURED_SOURCE_SHA256
    predicate_mapping_sha256: str = PREDICATE_MAPPING_SHA256
    p4d_authority_commit: str = P4D_SPEC_AUTHORITY_COMMIT
    p4b_parent_authority_commit: str = P4B_PARENT_AUTHORITY_COMMIT

    def artifact_hashes(self) -> Mapping[str, str]:
        return self.p4b_artifact_hashes or P4B_ARTIFACT_HASHES


def resolve_under_repo(repo_root: Path, path: str | Path) -> Path:
    root = repo_root.resolve()
    candidate = Path(path)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise GateBlocked(f"PATH_OUTSIDE_REPO:{path}") from exc
    return resolved


def current_head(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "UNKNOWN"


def validate_artifact_set(artifact_dir: Path, hashes: Mapping[str, str]) -> dict[str, str]:
    require(set(hashes) == set(EXPECTED_ARTIFACT_NAMES), "P4B_ARTIFACT_SET_AMBIGUITY")
    require(artifact_dir.name == f"reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_{P4B_EXECUTION_COMMIT}", "P4B_ARTIFACT_DIR_AMBIGUITY")
    observed_names = {entry.name for entry in artifact_dir.iterdir() if entry.is_file()}
    require(observed_names == set(EXPECTED_ARTIFACT_NAMES), "P4B_ARTIFACT_SET_MISSING_OR_EXTRA")
    observed: dict[str, str] = {}
    for name in EXPECTED_ARTIFACT_NAMES:
        path = artifact_dir / name
        observed[name] = file_sha256(path)
        require(observed[name] == hashes[name], f"P4B_ARTIFACT_HASH_MISMATCH:{name}")
    return observed


def validate_authority_inputs(repo_root: Path, frozen: FrozenInputs) -> dict[str, str]:
    hashes = frozen.artifact_hashes()
    artifact_dir = resolve_under_repo(repo_root, frozen.p4b_artifact_dir)
    observed = validate_artifact_set(artifact_dir, hashes)
    historical_path = resolve_under_repo(repo_root, frozen.historical_dataset_path)
    require(frozen.historical_dataset_path == HISTORICAL_DATASET_PATH, "HISTORICAL_DATASET_PATH_MISMATCH")
    historical_sha = file_sha256(historical_path)
    require(historical_sha == frozen.historical_dataset_sha256, "HISTORICAL_DATASET_SHA_MISMATCH")
    require(hashes[frozen.regenerated_dataset_name] == frozen.regenerated_dataset_sha256, "REGENERATED_DATASET_HASH_AMBIGUITY")
    stage185_path = resolve_under_repo(repo_root, frozen.stage185_source_script)
    require(frozen.stage185_source_script == STAGE185_SOURCE_SCRIPT, "STAGE185_SOURCE_PATH_MISMATCH")
    require(file_sha256(stage185_path) == frozen.stage185_source_script_sha256, "STAGE185_SOURCE_SHA_MISMATCH")
    structured_path = resolve_under_repo(repo_root, frozen.structured_source_path)
    require(file_sha256(structured_path) == frozen.structured_source_sha256, "STRUCTURED_SOURCE_SHA_MISMATCH")
    observed[frozen.historical_dataset_path] = historical_sha
    observed[frozen.stage185_source_script] = frozen.stage185_source_script_sha256
    observed[frozen.structured_source_path] = frozen.structured_source_sha256
    return observed


def validate_row_schema(row: Mapping[str, Any], index: int) -> None:
    require(tuple(row.keys()) == DATASET_FIELDS, f"DATASET_SCHEMA_FIELD_ORDER_MISMATCH:{index}")
    for field in ("id", "pair_id", "claim", "evidence", "final_label", "polarity_label", "primary_failure_type", "intervention_type"):
        require(isinstance(row[field], str) and bool(row[field]), f"DATASET_FIELD_TYPE_MISMATCH:{index}:{field}")
    for field in ("frame_compatible_label", "predicate_covered_label", "sufficiency_label"):
        require(type(row[field]) is int and row[field] in (0, 1), f"DATASET_FIELD_TYPE_MISMATCH:{index}:{field}")
    require(row["final_label"] in FINAL_LABELS, f"DATASET_FINAL_LABEL_ENUM_MISMATCH:{index}")
    require(row["polarity_label"] in POLARITY_LABELS, f"DATASET_POLARITY_LABEL_ENUM_MISMATCH:{index}")
    require(row["primary_failure_type"] in PRIMARY_FAILURE_TYPES, f"DATASET_PRIMARY_FAILURE_ENUM_MISMATCH:{index}")
    require(row["intervention_type"] in ADMITTED_FAMILIES, f"DATASET_INTERVENTION_ENUM_MISMATCH:{index}")


def validate_dataset_structure(rows: Sequence[Mapping[str, Any]], *, label: str) -> None:
    require(len(rows) == ROW_COUNT, f"{label}_ROW_COUNT_MISMATCH")
    for index, row in enumerate(rows):
        validate_row_schema(row, index)
    ids = [str(row["id"]) for row in rows]
    require(all(ids), f"{label}_EMPTY_ROW_ID")
    require(len(ids) == len(set(ids)), f"{label}_DUPLICATE_ROW_ID")
    require(all(row["intervention_type"] != "time_swap" for row in rows), f"{label}_TIME_SWAP_PRESENT")
    pairs: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    pair_interventions: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row["pair_id"]), str(row["intervention_type"]))
        require(key not in pair_interventions, f"{label}_DUPLICATE_PAIR_INTERVENTION:{key[0]}:{key[1]}")
        pair_interventions.add(key)
        pairs[str(row["pair_id"])].append(row)
    require(len(pairs) == PAIR_COUNT, f"{label}_PAIR_COUNT_MISMATCH")
    families = {str(row["intervention_type"]) for row in rows}
    require(families == ADMITTED_FAMILIES, f"{label}_FAMILY_SET_MISMATCH")
    for pair_id, members in pairs.items():
        require(len(members) == ROWS_PER_PAIR, f"{label}_ROWS_PER_PAIR_MISMATCH:{pair_id}")
        member_families = {str(row["intervention_type"]) for row in members}
        require(member_families == ADMITTED_FAMILIES, f"{label}_NONRECTANGULAR_TOPOLOGY:{pair_id}")
        require(sum(row["intervention_type"] == "none" for row in members) == 1, f"{label}_CANONICAL_COUNT_MISMATCH:{pair_id}")


def row_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    mapped = {str(row["id"]): row for row in rows}
    require(len(mapped) == len(rows), "DUPLICATE_ROW_ID")
    return mapped


def changed_fields(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[str]:
    return [field for field in DATASET_FIELDS if before.get(field) != after.get(field)]


def replay_stage185_split(rows: Sequence[Mapping[str, Any]], *, seed: int, ratio: float) -> dict[str, str]:
    require(seed == STAGE185_SPLIT_SEED, f"STAGE185_SPLIT_SEED_REJECTED:{seed}")
    require(ratio == STAGE185_DEV_RATIO, f"STAGE185_DEV_RATIO_REJECTED:{ratio}")
    pair_ids = sorted({str(row["pair_id"]) for row in rows})
    require(len(pair_ids) == PAIR_COUNT, "STAGE185_SPLIT_PAIR_COUNT_MISMATCH")
    random.Random(seed).shuffle(pair_ids)
    dev_count = min(len(pair_ids) - 1, max(1, round(len(pair_ids) * ratio)))
    dev_ids = set(pair_ids[:dev_count])
    return {str(row["id"]): ("dev" if str(row["pair_id"]) in dev_ids else "train") for row in rows}


def validate_split_identity(
    historical_rows: Sequence[Mapping[str, Any]],
    regenerated_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    ratio: float,
) -> None:
    historical_split = replay_stage185_split(historical_rows, seed=seed, ratio=ratio)
    regenerated_split = replay_stage185_split(regenerated_rows, seed=seed, ratio=ratio)
    require(historical_split == regenerated_split, "STAGE185_SPLIT_DRIFT")


def _authorized_pair_ids_from_artifacts(artifact_dir: Path) -> set[str]:
    summary = load_json(artifact_dir / SUMMARY_NAME)
    isolation = load_json(artifact_dir / ISOLATION_NAME)
    require(summary.get("authorized_pair_count") == AUTHORIZED_PAIR_COUNT, "P4B_SUMMARY_AUTHORIZED_PAIR_COUNT_MISMATCH")
    require(summary.get("authorized_member_count") == AUTHORIZED_MEMBER_COUNT, "P4B_SUMMARY_AUTHORIZED_MEMBER_COUNT_MISMATCH")
    changed_ids = isolation.get("authorized_changed_row_ids")
    unchanged_ids = isolation.get("authorized_unchanged_row_ids")
    require(isinstance(changed_ids, list) and isinstance(unchanged_ids, list), "P4B_ISOLATION_AUTHORIZED_IDS_MALFORMED")
    pair_ids = {str(row_id).rsplit("__", 1)[0] for row_id in [*changed_ids, *unchanged_ids]}
    require(len(pair_ids) == AUTHORIZED_PAIR_COUNT, "AUTHORIZED_F2_PAIR_COUNT_MISMATCH")
    return pair_ids


def validate_identity_label_linkage_and_deltas(
    historical_rows: Sequence[Mapping[str, Any]],
    regenerated_rows: Sequence[Mapping[str, Any]],
    *,
    authorized_pair_ids: set[str],
) -> dict[str, Any]:
    require([row["id"] for row in historical_rows] == [row["id"] for row in regenerated_rows], "ROW_ORDER_DRIFT")
    historical_by_id = row_map(historical_rows)
    regenerated_by_id = row_map(regenerated_rows)
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in regenerated_rows:
        groups[str(row["pair_id"])].append(row)
    changed_rows: list[str] = []
    changed_pairs: set[str] = set()
    authorized_members: set[str] = set()
    non_f2_changed: list[str] = []
    polarity_changed: list[str] = []
    for row_id, before in historical_by_id.items():
        after = regenerated_by_id[row_id]
        require(before["pair_id"] == after["pair_id"], f"PAIR_ID_DRIFT:{row_id}")
        require(before["intervention_type"] == after["intervention_type"], f"INTERVENTION_DRIFT:{row_id}")
        require(before["claim"] == after["claim"], f"CLAIM_DRIFT:{row_id}")
        for field in LABEL_FIELDS:
            require(before[field] == after[field], f"LABEL_DRIFT:{row_id}:{field}")
        pair_id = str(before["pair_id"])
        intervention = str(before["intervention_type"])
        is_f2 = pair_id in authorized_pair_ids
        if is_f2 and intervention in AUTHORIZED_F2_INTERVENTIONS:
            authorized_members.add(row_id)
            if intervention in {"none", "paraphrase"}:
                require(after["final_label"] == "REFUTE", f"AUTHORIZED_F2_FINAL_LABEL_MISMATCH:{row_id}")
                require(after["polarity_label"] == "REFUTE", f"AUTHORIZED_F2_POLARITY_LABEL_MISMATCH:{row_id}")
                require(after["primary_failure_type"] == "none", f"AUTHORIZED_F2_PRIMARY_FAILURE_MISMATCH:{row_id}")
            if intervention == "polarity_flip":
                require(after["final_label"] == "SUPPORT", f"AUTHORIZED_F2_FINAL_LABEL_MISMATCH:{row_id}")
                require(after["polarity_label"] == "SUPPORT", f"AUTHORIZED_F2_POLARITY_LABEL_MISMATCH:{row_id}")
                require(after["primary_failure_type"] == "polarity", f"AUTHORIZED_F2_PRIMARY_FAILURE_MISMATCH:{row_id}")
            require(after["frame_compatible_label"] == 1, f"AUTHORIZED_F2_FRAME_MISMATCH:{row_id}")
            require(after["predicate_covered_label"] == 1, f"AUTHORIZED_F2_PREDICATE_MISMATCH:{row_id}")
            require(after["sufficiency_label"] == 1, f"AUTHORIZED_F2_SUFFICIENCY_MISMATCH:{row_id}")
        deltas = changed_fields(before, after)
        if deltas:
            changed_rows.append(row_id)
            changed_pairs.add(pair_id)
        if is_f2 and intervention in AUTHORIZED_CHANGED_INTERVENTIONS:
            require(deltas in ([], ["evidence"]), f"UNAUTHORIZED_F2_FIELD_DELTA:{row_id}:{deltas}")
        elif is_f2 and intervention == "polarity_flip":
            if deltas:
                polarity_changed.append(row_id)
        elif deltas:
            non_f2_changed.append(row_id)
    require(len(authorized_members) == AUTHORIZED_MEMBER_COUNT, "AUTHORIZED_F2_MEMBER_COUNT_MISMATCH")
    require(len(changed_rows) == AUTHORIZED_CHANGED_ROW_COUNT, "CHANGED_ROW_COUNT_MISMATCH")
    require(len(changed_pairs) == AUTHORIZED_PAIR_COUNT, "CHANGED_PAIR_COUNT_MISMATCH")
    require(not polarity_changed, f"F2_POLARITY_FLIP_MUTATION:{polarity_changed[:3]}")
    require(not non_f2_changed, f"NON_F2_MUTATION:{non_f2_changed[:3]}")
    for pair_id, members in groups.items():
        canonical = [row for row in members if row["intervention_type"] == "none"]
        require(len(canonical) == 1, f"CANONICAL_LINKAGE_COUNT_MISMATCH:{pair_id}")
        canonical_id = str(canonical[0]["id"])
        require(canonical_id == f"{pair_id}__none", f"CANONICAL_ROW_ID_MISMATCH:{pair_id}")
        for row in members:
            require(str(row["pair_id"]) == pair_id, f"CANONICAL_PAIR_LINKAGE_MISMATCH:{row['id']}")
            require(row["claim"] == canonical[0]["claim"], f"CANONICAL_CLAIM_LINKAGE_MISMATCH:{row['id']}")
    return {
        "changed_row_count": len(changed_rows),
        "changed_pair_count": len(changed_pairs),
        "authorized_member_count": len(authorized_members),
    }


def require_fields(value: Mapping[str, Any], fields: Sequence[str], code: str) -> None:
    require(set(value.keys()) == set(fields), code)


def validate_compatibility_artifacts(artifact_dir: Path, frozen: FrozenInputs) -> None:
    rows = load_jsonl(artifact_dir / COMPAT_ROWS_NAME)
    summary = load_json(artifact_dir / COMPAT_SUMMARY_NAME)
    provenance = load_json(artifact_dir / COMPAT_PROVENANCE_NAME)
    require(len(rows) == AUTHORIZED_MEMBER_COUNT, "COMPATIBILITY_ROWS_COUNT_MISMATCH")
    for index, row in enumerate(rows):
        require_fields(row, COMPAT_ROW_FIELDS, f"COMPATIBILITY_ROW_SCHEMA_MISMATCH:{index}")
        require(row["schema_version"] == "P3W6F2P4B_R1_STAGE185_COMPATIBILITY_ROW_V1", f"COMPATIBILITY_ROW_SCHEMA_TOKEN_MISMATCH:{index}")
        require(row["effective_compatibility_status"] == "PASS", f"COMPATIBILITY_ROW_NOT_PASS:{index}")
        require(row["intervention_type"] in AUTHORIZED_F2_INTERVENTIONS, f"COMPATIBILITY_ROW_INTERVENTION_MISMATCH:{index}")
        require(row.get("training_admission_effect", {}).get("training_admission_released") is False, f"COMPATIBILITY_ROW_TRAINING_RELEASED:{index}")
        if row["intervention_type"] in AUTHORIZED_CHANGED_INTERVENTIONS:
            require("predicate" in row["raw_stage185_changed_axes"], f"RAW_STAGE185_PREDICATE_OBSERVATION_MISSING:{index}")
        else:
            require(row["raw_stage185_changed_axes"] == [], f"POLARITY_FLIP_COMPATIBILITY_RAW_DELTA:{index}")
    require_fields(summary, COMPAT_SUMMARY_FIELDS, "COMPATIBILITY_SUMMARY_SCHEMA_MISMATCH")
    expected_summary = {
        "schema_version": "P3W6F2P4B_R1_STAGE185_COMPATIBILITY_SUMMARY_V1",
        "authorized_pair_count": AUTHORIZED_PAIR_COUNT,
        "authorized_member_count": AUTHORIZED_MEMBER_COUNT,
        "row_count": AUTHORIZED_MEMBER_COUNT,
        "compatibility_pass_count": AUTHORIZED_MEMBER_COUNT,
        "compatibility_fail_count": 0,
        "compatibility_unresolved_count": 0,
        "compatibility_gate_status": "PASS",
        "raw_stage185_predicate_axis_observation_count": AUTHORIZED_CHANGED_ROW_COUNT,
        "stage185_v1_mutated": False,
        "historical_authority_weakened": False,
        "training_admission_released": False,
    }
    for key, expected in expected_summary.items():
        require(summary.get(key) == expected, f"COMPATIBILITY_SUMMARY_VALUE_MISMATCH:{key}")
    require(summary.get("failure_reasons") == [], "COMPATIBILITY_SUMMARY_FAILURE_REASONS")
    require_fields(provenance, COMPAT_PROVENANCE_FIELDS, "COMPATIBILITY_PROVENANCE_SCHEMA_MISMATCH")
    require(provenance["schema_version"] == "P3W6F2P4B_R1_STAGE185_COMPATIBILITY_PROVENANCE_V1", "COMPATIBILITY_PROVENANCE_SCHEMA_TOKEN_MISMATCH")
    require(provenance["stage185_source_script"] == frozen.stage185_source_script, "COMPATIBILITY_STAGE185_SOURCE_PATH_MISMATCH")
    require(provenance["stage185_source_script_sha256"] == frozen.stage185_source_script_sha256, "COMPATIBILITY_STAGE185_SOURCE_SHA_MISMATCH")
    require(provenance["historical_stage185_authority_sha256"] == frozen.stage185_source_script_sha256, "COMPATIBILITY_HISTORICAL_STAGE185_SHA_MISMATCH")
    require(provenance["regenerated_dataset_sha256"] == frozen.regenerated_dataset_sha256, "COMPATIBILITY_REGENERATED_DATASET_SHA_MISMATCH")
    require(provenance["structured_source_producer"] == frozen.structured_source_producer, "COMPATIBILITY_STRUCTURED_SOURCE_IDENTITY_MISMATCH")
    require(provenance["structured_source_producer_sha256"] == frozen.structured_source_sha256, "COMPATIBILITY_STRUCTURED_SOURCE_SHA_MISMATCH")
    require(provenance["base_form_coverage_sha256"] == frozen.artifact_hashes()[COVERAGE_NAME], "COMPATIBILITY_COVERAGE_SHA_MISMATCH")
    require(provenance["compatibility_rows_sha256"] == frozen.artifact_hashes()[COMPAT_ROWS_NAME], "COMPATIBILITY_ROWS_SHA_SPOOF")
    require(provenance["compatibility_summary_sha256"] == frozen.artifact_hashes()[COMPAT_SUMMARY_NAME], "COMPATIBILITY_SUMMARY_SHA_SPOOF")


def validate_p4b_summary_and_provenance(artifact_dir: Path, frozen: FrozenInputs) -> None:
    summary = load_json(artifact_dir / SUMMARY_NAME)
    isolation = load_json(artifact_dir / ISOLATION_NAME)
    invocation = load_json(artifact_dir / INVOCATION_NAME)
    coverage = load_json(artifact_dir / COVERAGE_NAME)
    require(summary.get("schema_version") == "P3W6F2P4B_R1_REGENERATION_SUMMARY_V1", "P4B_SUMMARY_SCHEMA_MISMATCH")
    require(summary.get("p4b_spec_authority_commit") == frozen.p4b_parent_authority_commit, "P4B_PARENT_AUTHORITY_COMMIT_MISMATCH")
    require(summary.get("historical_dataset_path") == frozen.historical_dataset_path, "P4B_SUMMARY_HISTORICAL_PATH_MISMATCH")
    require(summary.get("historical_dataset_sha256") == frozen.historical_dataset_sha256, "P4B_SUMMARY_HISTORICAL_SHA_MISMATCH")
    require(summary.get("regenerated_dataset_path") == frozen.regenerated_dataset_name, "P4B_SUMMARY_REGENERATED_PATH_MISMATCH")
    require(summary.get("regenerated_dataset_sha256") == frozen.regenerated_dataset_sha256, "P4B_SUMMARY_REGENERATED_SHA_MISMATCH")
    require(summary.get("regenerated_dataset_semantic_sha256") == frozen.regenerated_dataset_semantic_sha256, "P4B_SUMMARY_SEMANTIC_SHA_MISMATCH")
    require(summary.get("authorized_pair_count") == AUTHORIZED_PAIR_COUNT, "P4B_SUMMARY_AUTHORIZED_PAIR_COUNT_MISMATCH")
    require(summary.get("authorized_member_count") == AUTHORIZED_MEMBER_COUNT, "P4B_SUMMARY_AUTHORIZED_MEMBER_COUNT_MISMATCH")
    require(summary.get("changed_pair_count") == AUTHORIZED_PAIR_COUNT, "P4B_SUMMARY_CHANGED_PAIR_COUNT_MISMATCH")
    require(summary.get("changed_member_count") == AUTHORIZED_CHANGED_ROW_COUNT, "P4B_SUMMARY_CHANGED_MEMBER_COUNT_MISMATCH")
    require(summary.get("canonical_changed_member_count") == AUTHORIZED_PAIR_COUNT, "P4B_SUMMARY_CANONICAL_CHANGED_COUNT_MISMATCH")
    require(summary.get("paraphrase_changed_member_count") == AUTHORIZED_PAIR_COUNT, "P4B_SUMMARY_PARAPHRASE_CHANGED_COUNT_MISMATCH")
    require(summary.get("polarity_flip_changed_member_count") == 0, "P4B_SUMMARY_POLARITY_CHANGED_COUNT_MISMATCH")
    require(summary.get("structured_source_producer") == frozen.structured_source_producer, "P4B_SUMMARY_STRUCTURED_SOURCE_MISMATCH")
    require(summary.get("structured_source_producer_sha256") == frozen.structured_source_sha256, "P4B_SUMMARY_STRUCTURED_SOURCE_SHA_MISMATCH")
    require(summary.get("predicate_base_mapping_sha256") == frozen.predicate_mapping_sha256, "P4B_SUMMARY_MAPPING_SHA_MISMATCH")
    require(summary.get("artifact_set_complete") is True, "P4B_SUMMARY_ARTIFACT_SET_NOT_COMPLETE")
    require(summary.get("fail_closed_status") == "PASS", "P4B_SUMMARY_FAIL_CLOSED_STATUS_MISMATCH")
    require(isolation.get("schema_version") == "P3W6F2P4B_R1_FULL_OUTPUT_ISOLATION_V1", "P4B_ISOLATION_SCHEMA_MISMATCH")
    require(isolation.get("row_count_historical") == ROW_COUNT, "P4B_ISOLATION_HISTORICAL_ROW_COUNT_MISMATCH")
    require(isolation.get("row_count_regenerated") == ROW_COUNT, "P4B_ISOLATION_REGENERATED_ROW_COUNT_MISMATCH")
    require(isolation.get("authorized_pair_count") == AUTHORIZED_PAIR_COUNT, "P4B_ISOLATION_AUTHORIZED_PAIR_COUNT_MISMATCH")
    require(isolation.get("authorized_member_count") == AUTHORIZED_MEMBER_COUNT, "P4B_ISOLATION_AUTHORIZED_MEMBER_COUNT_MISMATCH")
    require(isolation.get("field_delta_counts") == {"evidence": AUTHORIZED_CHANGED_ROW_COUNT}, "P4B_ISOLATION_FIELD_DELTA_COUNTS_MISMATCH")
    require(isolation.get("row_order_identical") is True, "P4B_ISOLATION_ROW_ORDER_NOT_IDENTICAL")
    require(isolation.get("non_f2_rows_byte_identical") is True, "P4B_ISOLATION_NON_F2_NOT_IDENTICAL")
    require(isolation.get("unauthorized_changed_row_ids") == [], "P4B_ISOLATION_UNAUTHORIZED_CHANGES")
    require(isolation.get("isolation_status") == "PASS", "P4B_ISOLATION_STATUS_MISMATCH")
    require(len(isolation.get("authorized_changed_row_ids", [])) == AUTHORIZED_CHANGED_ROW_COUNT, "P4B_ISOLATION_CHANGED_ROW_COUNT_MISMATCH")
    require(len(isolation.get("authorized_unchanged_row_ids", [])) == AUTHORIZED_PAIR_COUNT, "P4B_ISOLATION_UNCHANGED_ROW_COUNT_MISMATCH")
    require(invocation.get("schema_version") == "P3W6F2P4B_R1_INVOCATION_V1", "P4B_INVOCATION_SCHEMA_MISMATCH")
    require(stage185_seed_authority() == (STAGE185_SPLIT_SEED, STAGE185_DEV_RATIO), "STAGE185_AUTHORITY_PARAMETERS_AMBIGUOUS")
    require(coverage.get("schema_version") == "P3W6F2P4B_R1_BASE_FORM_COVERAGE_V1", "P4B_COVERAGE_SCHEMA_MISMATCH")
    require(coverage.get("mapping_source_symbol") == "_BASE_PREDICATE_BY_INFLECTED", "P4B_COVERAGE_MAPPING_SYMBOL_MISMATCH")
    require(coverage.get("mapping_source_sha256") == frozen.structured_source_sha256, "P4B_COVERAGE_SOURCE_SHA_MISMATCH")
    require(coverage.get("coverage_status") == "PASS", "P4B_COVERAGE_STATUS_MISMATCH")


def stage185_seed_authority() -> tuple[int, float]:
    return STAGE185_SPLIT_SEED, STAGE185_DEV_RATIO


def validate_gate(
    repo_root: Path,
    *,
    historical_dataset: str | Path = HISTORICAL_DATASET_PATH,
    p4b_artifact_dir: str | Path = P4B_ARTIFACT_DIR,
    stage185_split_seed: int = STAGE185_SPLIT_SEED,
    stage185_dev_ratio: float = STAGE185_DEV_RATIO,
    frozen: FrozenInputs | None = None,
) -> dict[str, Any]:
    frozen = frozen or FrozenInputs()
    frozen = replace(frozen, historical_dataset_path=str(historical_dataset).replace("\\", "/"), p4b_artifact_dir=str(p4b_artifact_dir).replace("\\", "/"))
    statuses = {field: "BLOCKED" for field in STATUS_FIELDS}
    failures: list[str] = []
    row_count_historical = 0
    row_count_regenerated = 0
    try:
        require(frozen.p4d_authority_commit == P4D_SPEC_AUTHORITY_COMMIT, "P4D_AUTHORITY_COMMIT_AMBIGUITY")
        validate_authority_inputs(repo_root, frozen)
        artifact_dir = resolve_under_repo(repo_root, frozen.p4b_artifact_dir)
        historical_path = resolve_under_repo(repo_root, frozen.historical_dataset_path)
        regenerated_path = artifact_dir / frozen.regenerated_dataset_name
        statuses["provenance_status"] = "PASS"
        historical_rows = load_jsonl(historical_path)
        regenerated_rows = load_jsonl(regenerated_path)
        row_count_historical = len(historical_rows)
        row_count_regenerated = len(regenerated_rows)
        validate_dataset_structure(historical_rows, label="HISTORICAL")
        validate_dataset_structure(regenerated_rows, label="REGENERATED")
        statuses["schema_status"] = "PASS"
        statuses["row_id_status"] = "PASS"
        statuses["pair_topology_status"] = "PASS"
        statuses["time_swap_status"] = "PASS"
        require([row["id"] for row in historical_rows] == [row["id"] for row in regenerated_rows], "ROW_ORDER_DRIFT")
        statuses["row_order_status"] = "PASS"
        require(canonical_dataset_sha256(regenerated_rows) == frozen.regenerated_dataset_semantic_sha256, "REGENERATED_SEMANTIC_SHA_MISMATCH")
        validate_p4b_summary_and_provenance(artifact_dir, frozen)
        authorized_pair_ids = _authorized_pair_ids_from_artifacts(artifact_dir)
        validate_split_identity(historical_rows, regenerated_rows, seed=stage185_split_seed, ratio=stage185_dev_ratio)
        statuses["split_replay_status"] = "PASS"
        statuses["split_identity_status"] = "PASS"
        validate_identity_label_linkage_and_deltas(
            historical_rows,
            regenerated_rows,
            authorized_pair_ids=authorized_pair_ids,
        )
        statuses["label_integrity_status"] = "PASS"
        statuses["canonical_linkage_status"] = "PASS"
        statuses["delta_isolation_status"] = "PASS"
        statuses["polarity_flip_status"] = "PASS"
        statuses["non_f2_identity_status"] = "PASS"
        validate_compatibility_artifacts(artifact_dir, frozen)
        statuses["compatibility_artifact_status"] = "PASS"
        statuses["raw_stage185_observation_status"] = "PASS"
        statuses["historical_stage185_immutability_status"] = "PASS"
        statuses["determinism_status"] = "PASS"
    except GateBlocked as exc:
        failures.append(str(exc))
    decision = PASS_TOKEN if not failures and all(status == "PASS" for status in statuses.values()) else BLOCKED_TOKEN
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "decision_token": decision,
        "validator_contract_version": VALIDATOR_CONTRACT_VERSION,
        "validator_commit": "UNCOMMITTED_IMPLEMENTATION_STATIC_ONLY",
        "authority_commit": P4D_SPEC_AUTHORITY_COMMIT,
        "current_head": current_head(repo_root),
        "phase": "P4-D IMPLEMENTATION ONLY - NO OFFICIAL GATE 5 EXECUTION",
        "training_admission_released": False,
        "historical_dataset_path": frozen.historical_dataset_path,
        "historical_dataset_sha256": frozen.historical_dataset_sha256,
        "regenerated_dataset_path": frozen.regenerated_dataset_name,
        "regenerated_dataset_sha256": frozen.regenerated_dataset_sha256,
        "regenerated_dataset_semantic_sha256": frozen.regenerated_dataset_semantic_sha256,
        "p4b_artifact_directory": frozen.p4b_artifact_dir,
        "p4b_artifact_hashes": dict(sorted(frozen.artifact_hashes().items())),
        "stage185_source_script": frozen.stage185_source_script,
        "stage185_source_script_sha256": frozen.stage185_source_script_sha256,
        "structured_source_producer": frozen.structured_source_producer,
        "structured_source_producer_sha256": frozen.structured_source_sha256,
        "stage185_split_seed": stage185_split_seed,
        "stage185_dev_ratio": stage185_dev_ratio,
        "row_count_historical": row_count_historical,
        "row_count_regenerated": row_count_regenerated,
        **statuses,
        "failure_reasons": failures,
        "created_at_utc": "DETERMINISTIC_STATIC_REPORT_NO_WALL_CLOCK",
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--p4b-artifact-dir", type=Path, required=True)
    parser.add_argument("--historical-dataset", type=Path, required=True)
    parser.add_argument("--stage185-split-seed", type=int, default=STAGE185_SPLIT_SEED)
    parser.add_argument("--stage185-dev-ratio", type=float, default=STAGE185_DEV_RATIO)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = validate_gate(
        args.repo_root.resolve(),
        historical_dataset=args.historical_dataset,
        p4b_artifact_dir=args.p4b_artifact_dir,
        stage185_split_seed=args.stage185_split_seed,
        stage185_dev_ratio=args.stage185_dev_ratio,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["decision_token"] == PASS_TOKEN else 2


if __name__ == "__main__":
    raise SystemExit(main())
