#!/usr/bin/env python3
"""Build the P4-L current-lineage effective integrity sidecar.

This module is a deterministic, fail-closed builder for the frozen P4-L
sidecar contract. It imports no trainer, model, torch, checkpoint, network, or
Kaggle dependency. Historical Stage185 is consumed as read-only bridge evidence
only; current-lineage source identity is always the P4-B regenerated dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


AUTHORITY_VERSION = "P3W6F2P4L_CURRENT_LINEAGE_INTEGRITY_ARTIFACT_CONTRACT_AUTHORITY_V1"
NAMESPACE = "P3W6F2P4L_CURRENT_LINEAGE_INTEGRITY_SIDECAR"
SIDECAR_SCHEMA_VERSION = "P3W6F2P4L_CURRENT_LINEAGE_EFFECTIVE_INTEGRITY_SIDECAR_V1"
PROVENANCE_SCHEMA_VERSION = "P3W6F2P4L_CURRENT_LINEAGE_INTEGRITY_SIDECAR_PROVENANCE_V1"
RULE_VERSION = "P3W6F2P4L_CURRENT_LINEAGE_EFFECTIVE_INTEGRITY_SIDECAR_V1"

P4L_AUTHORITY_PATH = "reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_artifact_contract_authority_spec.md"
P4L_AUTHORITY_COMMIT = "80cb034792f03226cf6e22c196c1229ed4e6dd62"
P4H_AUTHORITY_COMMIT = "368d3b6991389aa6b6fd80f421c73565b562e290"
P4K_FREEZE_COMMIT = "13e7b0d7e229aa678e791e06b2e1d7de26474414"
P4H_RESULT_FREEZE_COMMIT = "b3626ae80ecf0664433821a772be28a56c6409da"
P4H_VERIFICATION_ATTESTATION_FREEZE_COMMIT = "703b861ab738b1cfdf73121de23ca07b6bbb9e48"

BUILDER_SOURCE_PATH = "scripts/build_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar.py"
SOURCE_DATASET_PATH = (
    "reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_"
    "4122078ab7962042e3d6bf89f8b4eb5cec463458/"
    "controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl"
)
SOURCE_DATASET_SHA256 = "eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3"
SOURCE_DATASET_SEMANTIC_SHA256 = "3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b"
HISTORICAL_DATASET_SHA256 = "f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640"
EXPECTED_ROW_COUNT = 3600
SPLIT_SEED = 174
DEV_RATIO = 0.2

P4B_COMPATIBILITY_ROWS_PATH = (
    "reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_"
    "4122078ab7962042e3d6bf89f8b4eb5cec463458/"
    "p3w6f2_p4b_r1_stage185_predicate_realization_compatibility_rows.jsonl"
)
P4B_COMPATIBILITY_ROWS_SHA256 = "59e4367d29e3c49152049e4a1b46e8783d5b81d1ebaa931ca9fd4ae0ac967b9f"
P4B_COMPATIBILITY_SUMMARY_PATH = (
    "reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_"
    "4122078ab7962042e3d6bf89f8b4eb5cec463458/"
    "p3w6f2_p4b_r1_stage185_predicate_realization_compatibility_summary.json"
)
P4B_COMPATIBILITY_SUMMARY_SHA256 = "ce618d214dc4d660706d927a0d91ec5945d3a0edbffbf615eab8b9c9ff585aa8"
P4B_COMPATIBILITY_PROVENANCE_PATH = (
    "reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_"
    "4122078ab7962042e3d6bf89f8b4eb5cec463458/"
    "p3w6f2_p4b_r1_stage185_predicate_realization_compatibility_provenance_manifest.json"
)
P4B_COMPATIBILITY_PROVENANCE_SHA256 = "09a56f1dca325e0749a0e4b0f822d68dad5d85fd90fa427e0efd6d617d41b2d6"
P4B_COMPATIBILITY_RULE_VERSION = "P3W6F2P4B_R1_STAGE185_PREDICATE_REALIZATION_COMPATIBILITY_V1"
P4B_AUTHORIZED_PAIR_COUNT = 119
P4B_AUTHORIZED_MEMBER_COUNT = 357

HISTORICAL_STAGE185_SIDECAR_PATH = (
    "reports/stage185a_controlled_train_integrity_sidecar_20260715_141914/"
    "stage185a_controlled_train_integrity_sidecar.jsonl"
)
HISTORICAL_STAGE185_SIDECAR_SEMANTIC_SHA256 = "5bc03caa2a29f9b9176ab4eb0201db57ebad516352797546db1a18e6ec3373fc"
HISTORICAL_STAGE185_SOURCE_PATH = "scripts/build_stage185a_controlled_train_integrity_sidecar.py"
HISTORICAL_STAGE185_SOURCE_SHA256 = "11e6ba89b8131c76eac4504b4273867eaa99a131abe23d3238eb65ecda207bbc"

SIDECAR_NAME = "p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl"
PROVENANCE_NAME = "p3w6f2_p4l_current_lineage_effective_integrity_sidecar_provenance.json"
EXPECTED_OUTPUT_NAMES = {SIDECAR_NAME, PROVENANCE_NAME}

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
P2_SOURCE_REQUIRED_FIELDS = (
    "id",
    "pair_id",
    "intervention_type",
    "final_label",
    "frame_compatible_label",
    "predicate_covered_label",
    "sufficiency_label",
    "polarity_label",
    "primary_failure_type",
)
P2_SIDE_CAR_REQUIRED_FIELDS = (
    "row_id",
    "split",
    "pair_id",
    "canonical_row_id",
    "canonical_status",
    "intervention_contract_status",
    "integrity_status",
    "schema_status",
    "dataset_source_status",
    "grammar_status",
    "polarity_contamination_status",
    "time_swap_status",
    "reason_codes",
    "source_dataset_path",
    "source_dataset_sha256",
    "frame_compatible_label",
)
P4L_STAGE185_COMPATIBLE_FIELDS = (
    "intervention_type",
    "eligible_for_positive_margin",
    "family_contract_id",
    "rule_version",
    "generator_source_sha256",
    "integrity_builder_sha256",
    "created_at",
)
HISTORICAL_AUDIT_FIELDS = (
    "audit_changed_axes",
    "audit_expected_axes",
    "audit_pair_failure_scope",
    "audit_preserved_axes",
    "generator_source_path",
    "stage182a_report_sha256",
    "stage184a_report_sha256",
)
COMPONENT_STATUS_FIELDS = (
    "schema_status",
    "dataset_source_status",
    "grammar_status",
    "canonical_status",
    "intervention_contract_status",
    "polarity_contamination_status",
    "time_swap_status",
)
STATUS_ENUM = {"PASS", "FAIL", "UNRESOLVED", "NOT_APPLICABLE"}
INTEGRITY_ENUM = {"ELIGIBLE", "INELIGIBLE", "UNRESOLVED"}
FINAL_LABELS = {"REFUTE", "NOT_ENTITLED", "SUPPORT"}
PRIMARY_FAILURE_TYPES = {"none", "frame", "predicate", "sufficiency", "polarity"}
P2_REASON_CLASS_ORDER = ("FRAME", "PREDICATE", "SUFFICIENCY", "AUTHORIZED")
P2_DIRECTIONAL_LABELS = {"REFUTE", "SUPPORT"}
P2_EXCLUSION_CODES = (
    "P2_NON_CANONICAL_SOURCE",
    "P2_SIDECAR_MISSING",
    "P2_SPLIT_MISMATCH",
    "P2_CANONICAL_ROW_ID_MISMATCH",
    "P2_SIDECAR_SOURCE_BINARY_MISMATCH",
    "P2_POLARITY_INTERVENTION_CONTRACT_FAIL",
    "P2_INTEGRITY_SOURCE_REQUIRED",
    "P2_GENERATOR_STATUS_DEFECT",
    "P2_PRIMARY_REASON_AXIS_CONFLICT",
    "P2_FAILURE_FINAL_LABEL_MISMATCH",
    "P2_AUTHORIZED_FINAL_LABEL_MISMATCH",
    "P2_POLARITY_TARGET_FINAL_MISMATCH",
)


class BuildBlocked(RuntimeError):
    """Fail-closed P4-L builder rejection."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BuildBlocked(message)


def repo_path(repo_root: Path, relative: str) -> Path:
    path = repo_root / relative
    resolved_root = repo_root.resolve()
    resolved_path = path.resolve()
    require(
        resolved_path == resolved_root or resolved_root in resolved_path.parents,
        f"PATH_OUTSIDE_REPO:{relative}",
    )
    return resolved_path


def repo_relative(repo_root: Path, path: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def deterministic_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def compact_jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    materialized = list(rows)
    lines = [
        json.dumps(dict(row), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        for row in materialized
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON_NOT_OBJECT:{path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            require(line.endswith("\n"), f"JSONL_LINE_WITHOUT_LF:{path}:{line_number}")
            if not line.strip():
                continue
            value = json.loads(line)
            require(isinstance(value, dict), f"JSONL_LINE_NOT_OBJECT:{path}:{line_number}")
            rows.append(value)
    return rows


def ordered_dataset_row(row: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in DATASET_FIELDS if field not in row]
    require(not missing, f"DATASET_ROW_MISSING_FIELDS:{row.get('id', '<unknown>')}:{missing}")
    return {field: row[field] for field in DATASET_FIELDS}


def semantic_dataset_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return canonical_sha256([ordered_dataset_row(row) for row in rows])


def semantic_sidecar_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    canonical = [
        {key: row[key] for key in sorted(row) if key != "created_at"}
        for row in rows
    ]
    return canonical_sha256(canonical)


def validate_path_and_sha(repo_root: Path, relative: str, expected_sha256: str) -> Path:
    path = repo_path(repo_root, relative)
    require(path.is_file(), f"REQUIRED_INPUT_MISSING:{relative}")
    observed = file_sha256(path)
    require(observed == expected_sha256, f"INPUT_SHA256_MISMATCH:{relative}:{observed}")
    return path


def validate_frozen_inputs(repo_root: Path) -> dict[str, Path]:
    inputs = {
        "source_dataset": validate_path_and_sha(repo_root, SOURCE_DATASET_PATH, SOURCE_DATASET_SHA256),
        "p4b_rows": validate_path_and_sha(repo_root, P4B_COMPATIBILITY_ROWS_PATH, P4B_COMPATIBILITY_ROWS_SHA256),
        "p4b_summary": validate_path_and_sha(repo_root, P4B_COMPATIBILITY_SUMMARY_PATH, P4B_COMPATIBILITY_SUMMARY_SHA256),
        "p4b_provenance": validate_path_and_sha(repo_root, P4B_COMPATIBILITY_PROVENANCE_PATH, P4B_COMPATIBILITY_PROVENANCE_SHA256),
        "historical_stage185": repo_path(repo_root, HISTORICAL_STAGE185_SIDECAR_PATH),
        "historical_stage185_source": validate_path_and_sha(repo_root, HISTORICAL_STAGE185_SOURCE_PATH, HISTORICAL_STAGE185_SOURCE_SHA256),
        "builder_source": repo_path(repo_root, BUILDER_SOURCE_PATH),
        "p4l_authority": repo_path(repo_root, P4L_AUTHORITY_PATH),
    }
    require(inputs["historical_stage185"].is_file(), "HISTORICAL_STAGE185_SIDECAR_MISSING")
    return inputs


def exact_binary(row: Mapping[str, Any], field: str, row_id: str) -> int:
    value = row.get(field)
    if isinstance(value, bool) or type(value) is not int or value not in (0, 1):
        raise BuildBlocked(f"EXACT_BINARY_VALIDATION_FAILED:{row_id}:{field}:{value!r}")
    return value


def canonical_polarity(value: Any) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return {0: "NONE", 1: "REFUTE", 2: "SUPPORT"}.get(value, "UNKNOWN")
    return str(value).strip().upper()


def normalize_final_label(value: Any) -> str:
    return str(value).strip().upper()


def validate_source_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    require(len(rows) == EXPECTED_ROW_COUNT, f"SOURCE_ROW_COUNT_MISMATCH:{len(rows)}")
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        missing = [field for field in P2_SOURCE_REQUIRED_FIELDS if field not in row]
        require(not missing, f"P2_SOURCE_REQUIRED_FIELDS_MISSING:{index}:{missing}")
        row_id = row.get("id")
        pair_id = row.get("pair_id")
        intervention_type = row.get("intervention_type")
        require(isinstance(row_id, str) and row_id.strip(), f"SOURCE_ROW_ID_INVALID:{index}")
        require(row_id not in seen_ids, f"DUPLICATE_ROW_ID:{row_id}")
        seen_ids.add(row_id)
        require(isinstance(pair_id, str) and pair_id.strip(), f"SOURCE_PAIR_ID_INVALID:{row_id}")
        require(isinstance(intervention_type, str) and intervention_type.strip(), f"SOURCE_INTERVENTION_INVALID:{row_id}")
        final_label = normalize_final_label(row.get("final_label"))
        require(final_label in FINAL_LABELS, f"FINAL_LABEL_INVALID:{row_id}:{row.get('final_label')!r}")
        frame = exact_binary(row, "frame_compatible_label", row_id)
        predicate = exact_binary(row, "predicate_covered_label", row_id)
        sufficiency = exact_binary(row, "sufficiency_label", row_id)
        primary_failure = str(row.get("primary_failure_type", "")).strip().lower()
        require(primary_failure in PRIMARY_FAILURE_TYPES, f"PRIMARY_FAILURE_TYPE_INVALID:{row_id}:{primary_failure!r}")
        polarity = canonical_polarity(row.get("polarity_label"))
        require(polarity in {"NONE", "REFUTE", "SUPPORT", "NOT_ENTITLED"}, f"POLARITY_LABEL_INVALID:{row_id}:{row.get('polarity_label')!r}")
        normalized.append(
            {
                **dict(row),
                "final_label": final_label,
                "frame_compatible_label": frame,
                "predicate_covered_label": predicate,
                "sufficiency_label": sufficiency,
                "primary_failure_type": primary_failure,
            }
        )
    return normalized


def validate_source_dataset(repo_root: Path, path: Path) -> list[dict[str, Any]]:
    require(repo_relative(repo_root, path) == SOURCE_DATASET_PATH, "SOURCE_DATASET_PATH_MISMATCH")
    require(file_sha256(path) == SOURCE_DATASET_SHA256, "SOURCE_DATASET_PHYSICAL_SHA_MISMATCH")
    rows = validate_source_rows(read_jsonl(path))
    observed_semantic = semantic_dataset_sha256(rows)
    require(observed_semantic == SOURCE_DATASET_SEMANTIC_SHA256, "SOURCE_DATASET_SEMANTIC_SHA_MISMATCH")
    return rows


def deterministic_pair_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = SPLIT_SEED,
    dev_ratio: float = DEV_RATIO,
) -> dict[str, str]:
    pair_ids = sorted({str(row["pair_id"]) for row in rows})
    require(pair_ids, "NO_PAIR_IDS")
    shuffled = list(pair_ids)
    random.Random(seed).shuffle(shuffled)
    dev_count = min(len(shuffled) - 1, max(1, round(len(shuffled) * dev_ratio)))
    dev_ids = set(shuffled[:dev_count])
    return {pair_id: "dev" if pair_id in dev_ids else "train" for pair_id in pair_ids}


def canonical_row_ids(rows: Sequence[Mapping[str, Any]], split_by_pair: Mapping[str, str]) -> dict[str, str]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["pair_id"])].append(row)
    canonical_by_pair: dict[str, str] = {}
    for pair_id, group in grouped.items():
        canonical = [row for row in group if str(row.get("intervention_type", "")) == "none"]
        require(len(canonical) == 1, f"CANONICAL_TARGET_COUNT_INVALID:{pair_id}:{len(canonical)}")
        canonical_id = str(canonical[0].get("id", ""))
        require(canonical_id, f"CANONICAL_ROW_ID_EMPTY:{pair_id}")
        require(str(canonical[0].get("pair_id", "")) == pair_id, f"CANONICAL_PAIR_MISMATCH:{pair_id}")
        require(pair_id in split_by_pair, f"CANONICAL_SPLIT_MISSING:{pair_id}")
        canonical_by_pair[pair_id] = canonical_id
    return {str(row["id"]): canonical_by_pair[str(row["pair_id"])] for row in rows}


def validate_canonical_lineage(
    rows: Sequence[Mapping[str, Any]],
    split_by_pair: Mapping[str, str],
    canonical_by_row: Mapping[str, str],
) -> None:
    by_id = {str(row["id"]): row for row in rows}
    for row in rows:
        row_id = str(row["id"])
        pair_id = str(row["pair_id"])
        canonical_id = canonical_by_row.get(row_id)
        require(isinstance(canonical_id, str) and canonical_id, f"CANONICAL_ROW_ID_INVALID:{row_id}")
        target = by_id.get(canonical_id)
        require(target is not None, f"CANONICAL_TARGET_MISSING:{row_id}:{canonical_id}")
        require(str(target["pair_id"]) == pair_id, f"CANONICAL_TARGET_PAIR_MISMATCH:{row_id}")
        require(split_by_pair[str(target["pair_id"])] == split_by_pair[pair_id], f"CANONICAL_TARGET_SPLIT_MISMATCH:{row_id}")
        require(canonical_by_row.get(canonical_id) == canonical_id, f"CANONICAL_TARGET_NOT_SELF_ANCHORED:{row_id}")


def load_historical_stage185(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    observed = semantic_sidecar_sha256(rows)
    require(observed == HISTORICAL_STAGE185_SIDECAR_SEMANTIC_SHA256, "HISTORICAL_STAGE185_SEMANTIC_SHA_MISMATCH")
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = row.get("row_id")
        require(isinstance(row_id, str) and row_id, "HISTORICAL_STAGE185_ROW_ID_INVALID")
        require(row_id not in by_id, f"HISTORICAL_STAGE185_DUPLICATE_ROW_ID:{row_id}")
        by_id[row_id] = row
    return by_id


def load_p4b_compatibility(repo_root: Path, rows_path: Path, summary_path: Path, provenance_path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(rows_path)
    summary = read_json(summary_path)
    provenance = read_json(provenance_path)
    require(summary.get("schema_version") == "P3W6F2P4B_R1_STAGE185_COMPATIBILITY_SUMMARY_V1", "P4B_SUMMARY_SCHEMA_MISMATCH")
    require(summary.get("compatibility_rule_version") == P4B_COMPATIBILITY_RULE_VERSION, "P4B_SUMMARY_RULE_MISMATCH")
    require(summary.get("row_count") == P4B_AUTHORIZED_MEMBER_COUNT, "P4B_SUMMARY_ROW_COUNT_MISMATCH")
    require(summary.get("authorized_pair_count") == P4B_AUTHORIZED_PAIR_COUNT, "P4B_SUMMARY_PAIR_COUNT_MISMATCH")
    require(summary.get("authorized_member_count") == P4B_AUTHORIZED_MEMBER_COUNT, "P4B_SUMMARY_MEMBER_COUNT_MISMATCH")
    require(summary.get("compatibility_gate_status") == "PASS", "P4B_COMPATIBILITY_GATE_NOT_PASS")
    require(summary.get("training_admission_released") is False, "P4B_TRAINING_ADMISSION_RELEASED")
    require(provenance.get("schema_version") == "P3W6F2P4B_R1_STAGE185_COMPATIBILITY_PROVENANCE_V1", "P4B_PROVENANCE_SCHEMA_MISMATCH")
    require(provenance.get("compatibility_rows_sha256") == P4B_COMPATIBILITY_ROWS_SHA256, "P4B_PROVENANCE_ROWS_SHA_MISMATCH")
    require(provenance.get("compatibility_summary_sha256") == P4B_COMPATIBILITY_SUMMARY_SHA256, "P4B_PROVENANCE_SUMMARY_SHA_MISMATCH")
    require(provenance.get("regenerated_dataset_path") == SOURCE_DATASET_PATH, "P4B_PROVENANCE_SOURCE_PATH_MISMATCH")
    require(provenance.get("regenerated_dataset_sha256") == SOURCE_DATASET_SHA256, "P4B_PROVENANCE_SOURCE_SHA_MISMATCH")
    require(provenance.get("stage185_source_script_sha256") == HISTORICAL_STAGE185_SOURCE_SHA256, "P4B_STAGE185_SOURCE_SHA_MISMATCH")
    require(file_sha256(rows_path) == P4B_COMPATIBILITY_ROWS_SHA256, "P4B_ROWS_PHYSICAL_SHA_MISMATCH")
    require(file_sha256(summary_path) == P4B_COMPATIBILITY_SUMMARY_SHA256, "P4B_SUMMARY_PHYSICAL_SHA_MISMATCH")
    require(file_sha256(provenance_path) == P4B_COMPATIBILITY_PROVENANCE_SHA256, "P4B_PROVENANCE_PHYSICAL_SHA_MISMATCH")
    require(repo_relative(repo_root, rows_path) == P4B_COMPATIBILITY_ROWS_PATH, "P4B_ROWS_PATH_MISMATCH")
    require(repo_relative(repo_root, summary_path) == P4B_COMPATIBILITY_SUMMARY_PATH, "P4B_SUMMARY_PATH_MISMATCH")
    require(repo_relative(repo_root, provenance_path) == P4B_COMPATIBILITY_PROVENANCE_PATH, "P4B_PROVENANCE_PATH_MISMATCH")
    by_member: dict[str, dict[str, Any]] = {}
    pair_ids: set[str] = set()
    for row in rows:
        member_id = row.get("member_id")
        pair_id = row.get("pair_id")
        require(isinstance(member_id, str) and member_id, "P4B_MEMBER_ID_INVALID")
        require(member_id not in by_member, f"P4B_DUPLICATE_MEMBER_ID:{member_id}")
        require(row.get("schema_version") == "P3W6F2P4B_R1_STAGE185_COMPATIBILITY_ROW_V1", f"P4B_ROW_SCHEMA_MISMATCH:{member_id}")
        require(row.get("compatibility_rule_version") == P4B_COMPATIBILITY_RULE_VERSION, f"P4B_ROW_RULE_MISMATCH:{member_id}")
        require(row.get("effective_compatibility_status") == "PASS", f"P4B_ROW_NOT_PASS:{member_id}")
        require(row.get("training_admission_effect", {}).get("training_admission_released") is False, f"P4B_ROW_TRAINING_RELEASED:{member_id}")
        require(isinstance(pair_id, str) and pair_id, f"P4B_PAIR_ID_INVALID:{member_id}")
        by_member[member_id] = row
        pair_ids.add(pair_id)
    require(len(by_member) == P4B_AUTHORIZED_MEMBER_COUNT, "P4B_MEMBER_COUNT_MISMATCH")
    require(len(pair_ids) == P4B_AUTHORIZED_PAIR_COUNT, "P4B_PAIR_COUNT_MISMATCH")
    return by_member


def status_from_required_components(statuses: Sequence[str]) -> str:
    require(all(status in STATUS_ENUM for status in statuses), f"STATUS_ENUM_INVALID:{statuses}")
    required = [status for status in statuses if status != "NOT_APPLICABLE"]
    if any(status == "FAIL" for status in required):
        return "INELIGIBLE"
    if any(status == "UNRESOLVED" for status in required):
        return "UNRESOLVED"
    return "ELIGIBLE"


def normalized_generator_status(sidecar: Mapping[str, Any]) -> str:
    statuses = [sidecar.get(field) for field in COMPONENT_STATUS_FIELDS]
    if any(status is None or not isinstance(status, str) for status in statuses):
        return "UNRESOLVED"
    if any(status not in {"PASS", "FAIL"} for status in statuses):
        return "UNRESOLVED"
    if all(status == "PASS" for status in statuses):
        return "CLEAN"
    return "DEFECT"


def primary_reason_from_axes(frame: int, predicate: int, sufficiency: int) -> str:
    if frame == 0:
        return "FRAME"
    if predicate == 0:
        return "PREDICATE"
    if sufficiency == 0:
        return "SUFFICIENCY"
    return "AUTHORIZED"


def expected_primary_from_record(record: Mapping[str, Any]) -> str | None:
    raw = str(record.get("primary_failure_type", "")).strip().lower()
    if raw == "frame":
        return "FRAME"
    if raw == "predicate":
        return "PREDICATE"
    if raw == "sufficiency":
        return "SUFFICIENCY"
    if raw in {"none", "polarity"}:
        return "AUTHORIZED"
    return None


def p2_reason_exclusion_codes(
    *,
    source_row: Mapping[str, Any],
    sidecar_row: Mapping[str, Any],
    split: str,
    expected_canonical_row_id: str,
    source_label: str = "clean_main",
) -> list[str]:
    row_id = str(source_row.get("id", ""))
    frame = exact_binary(source_row, "frame_compatible_label", row_id)
    predicate = exact_binary(source_row, "predicate_covered_label", row_id)
    sufficiency = exact_binary(source_row, "sufficiency_label", row_id)
    raw_primary = str(source_row.get("primary_failure_type", "")).strip().lower()
    require(raw_primary in PRIMARY_FAILURE_TYPES, f"P2_UNKNOWN_PRIMARY_FAILURE_TYPE:{row_id}:{raw_primary!r}")
    derived = primary_reason_from_axes(frame, predicate, sufficiency)
    expected = expected_primary_from_record(source_row)
    final_label = normalize_final_label(source_row.get("final_label"))
    directional = final_label in P2_DIRECTIONAL_LABELS
    polarity_label = canonical_polarity(source_row.get("polarity_label"))
    intervention_type = str(source_row.get("intervention_type", "")).strip().lower()
    codes: list[str] = []
    if source_label != "clean_main":
        codes.append("P2_NON_CANONICAL_SOURCE")
    if sidecar_row.get("split") != split:
        codes.append("P2_SPLIT_MISMATCH")
    if sidecar_row.get("canonical_row_id") != expected_canonical_row_id:
        codes.append("P2_CANONICAL_ROW_ID_MISMATCH")
    sidecar_frame = exact_binary(sidecar_row, "frame_compatible_label", row_id)
    if sidecar_frame != frame:
        codes.append("P2_SIDECAR_SOURCE_BINARY_MISMATCH")
    if sidecar_row.get("intervention_contract_status") != "PASS":
        codes.append("P2_POLARITY_INTERVENTION_CONTRACT_FAIL")
    generator_status = normalized_generator_status(sidecar_row)
    if generator_status == "UNRESOLVED":
        codes.append("P2_INTEGRITY_SOURCE_REQUIRED")
    elif generator_status != "CLEAN":
        codes.append("P2_GENERATOR_STATUS_DEFECT")
    if expected != derived:
        codes.append("P2_PRIMARY_REASON_AXIS_CONFLICT")
    if derived in {"FRAME", "PREDICATE", "SUFFICIENCY"} and final_label != "NOT_ENTITLED":
        codes.append("P2_FAILURE_FINAL_LABEL_MISMATCH")
    if raw_primary in {"none", "polarity"} and not directional:
        codes.append("P2_AUTHORIZED_FINAL_LABEL_MISMATCH")
    if directional:
        if polarity_label != final_label:
            codes.append("P2_POLARITY_TARGET_FINAL_MISMATCH")
    elif polarity_label not in {"NONE", "NOT_ENTITLED"}:
        codes.append("P2_POLARITY_TARGET_FINAL_MISMATCH")
    if raw_primary == "polarity" and (not directional or intervention_type != "polarity_flip"):
        codes.append("P2_POLARITY_INTERVENTION_CONTRACT_FAIL")
    return sorted(set(codes), key=lambda code: P2_EXCLUSION_CODES.index(code) if code in P2_EXCLUSION_CODES else len(P2_EXCLUSION_CODES))


def bridge_historical_statuses(
    *,
    source_row: Mapping[str, Any],
    historical_row: Mapping[str, Any] | None,
    p4b_row: Mapping[str, Any] | None,
) -> dict[str, Any]:
    row_id = str(source_row["id"])
    if historical_row is None:
        return {
            "canonical_status": "UNRESOLVED",
            "intervention_contract_status": "UNRESOLVED",
            "schema_status": "PASS",
            "grammar_status": "UNRESOLVED",
            "polarity_contamination_status": "UNRESOLVED",
            "time_swap_status": "UNRESOLVED",
            "reason_codes": ["HISTORICAL_STAGE185_ROW_MISSING"],
            "historical_bridge_status": "UNRESOLVED",
        }
    require(historical_row.get("row_id") == row_id, f"HISTORICAL_BRIDGE_ROW_ID_MISMATCH:{row_id}")
    require(historical_row.get("pair_id") == source_row.get("pair_id"), f"HISTORICAL_BRIDGE_PAIR_ID_MISMATCH:{row_id}")
    require(historical_row.get("intervention_type") == source_row.get("intervention_type"), f"HISTORICAL_BRIDGE_INTERVENTION_MISMATCH:{row_id}")
    bridged: dict[str, Any] = {
        "canonical_status": historical_row.get("canonical_status", "UNRESOLVED"),
        "intervention_contract_status": historical_row.get("intervention_contract_status", "UNRESOLVED"),
        "schema_status": "PASS",
        "grammar_status": historical_row.get("grammar_status", "UNRESOLVED"),
        "polarity_contamination_status": historical_row.get("polarity_contamination_status", "UNRESOLVED"),
        "time_swap_status": historical_row.get("time_swap_status", "UNRESOLVED"),
        "reason_codes": list(historical_row.get("reason_codes", [])) if isinstance(historical_row.get("reason_codes"), list) else [],
        "family_contract_id": f"{RULE_VERSION}:{source_row['intervention_type']}",
        "historical_bridge_status": "PASS",
    }
    for field in HISTORICAL_AUDIT_FIELDS:
        if field in historical_row:
            key = field if field.startswith("generator_") or field.startswith("stage") else f"historical_{field}"
            bridged[key] = historical_row[field]
    if p4b_row is not None:
        require(p4b_row.get("member_id") == row_id, f"P4B_MEMBER_ROW_MISMATCH:{row_id}")
        require(p4b_row.get("pair_id") == source_row.get("pair_id"), f"P4B_MEMBER_PAIR_MISMATCH:{row_id}")
        require(p4b_row.get("intervention_type") == source_row.get("intervention_type"), f"P4B_MEMBER_INTERVENTION_MISMATCH:{row_id}")
        status = p4b_row.get("effective_compatibility_status")
        if status == "PASS":
            bridged["grammar_status"] = "PASS"
            bridged["p4b_effective_compatibility_status"] = "PASS"
            bridged["p4b_effective_reason_codes"] = list(p4b_row.get("effective_reason_codes", []))
            bridged["reason_codes"] = [
                code
                for code in bridged["reason_codes"]
                if code not in {"GRAMMAR_ANOMALY", "DID_NOT_INFLECTED_PREDICATE"}
            ]
        else:
            bridged["grammar_status"] = "FAIL" if status == "FAIL" else "UNRESOLVED"
            bridged["p4b_effective_compatibility_status"] = status or "UNRESOLVED"
            bridged["reason_codes"].append("P4B_EFFECTIVE_COMPATIBILITY_NOT_PASS")
    return bridged


def compose_integrity_status(row: Mapping[str, Any]) -> str:
    return status_from_required_components([str(row.get(field, "UNRESOLVED")) for field in COMPONENT_STATUS_FIELDS])


def positive_margin_eligible(row: Mapping[str, Any]) -> bool:
    return (
        row.get("integrity_status") == "ELIGIBLE"
        and row.get("split") == "train"
        and row.get("frame_compatible_label") == 1
        and row.get("time_swap_status") == "PASS"
        and row.get("dataset_source_status") == "PASS"
    )


def assemble_sidecar_rows(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    split_by_pair: Mapping[str, str],
    canonical_by_row: Mapping[str, str],
    historical_by_id: Mapping[str, Mapping[str, Any]],
    p4b_by_member: Mapping[str, Mapping[str, Any]],
    builder_source_sha256: str,
    created_at: str,
) -> list[dict[str, Any]]:
    sidecar: list[dict[str, Any]] = []
    for index, source_row in enumerate(source_rows):
        row_id = str(source_row["id"])
        pair_id = str(source_row["pair_id"])
        split = split_by_pair[pair_id]
        canonical_id = canonical_by_row[row_id]
        p4b_row = p4b_by_member.get(row_id)
        bridged = bridge_historical_statuses(
            source_row=source_row,
            historical_row=historical_by_id.get(row_id),
            p4b_row=p4b_row,
        )
        row: dict[str, Any] = {
            "namespace": NAMESPACE,
            "schema_version": SIDECAR_SCHEMA_VERSION,
            "source_order_index": index,
            "row_id": row_id,
            "split": split,
            "pair_id": pair_id,
            "canonical_row_id": canonical_id,
            "canonical_status": bridged.get("canonical_status", "UNRESOLVED"),
            "intervention_contract_status": bridged.get("intervention_contract_status", "UNRESOLVED"),
            "integrity_status": "UNRESOLVED",
            "schema_status": bridged.get("schema_status", "PASS"),
            "dataset_source_status": "PASS",
            "grammar_status": bridged.get("grammar_status", "UNRESOLVED"),
            "polarity_contamination_status": bridged.get("polarity_contamination_status", "UNRESOLVED"),
            "time_swap_status": bridged.get("time_swap_status", "UNRESOLVED"),
            "reason_codes": [],
            "source_dataset_path": SOURCE_DATASET_PATH,
            "source_dataset_sha256": SOURCE_DATASET_SHA256,
            "source_dataset_semantic_sha256": SOURCE_DATASET_SEMANTIC_SHA256,
            "frame_compatible_label": source_row["frame_compatible_label"],
            "intervention_type": source_row["intervention_type"],
            "eligible_for_positive_margin": False,
            "family_contract_id": bridged.get("family_contract_id", f"{RULE_VERSION}:{source_row['intervention_type']}"),
            "rule_version": RULE_VERSION,
            "generator_source_sha256": bridged.get("generator_source_sha256", "HISTORICAL_OBSERVATION_ONLY"),
            "integrity_builder_sha256": builder_source_sha256,
            "created_at": created_at,
            "p2_primary_reason": primary_reason_from_axes(
                source_row["frame_compatible_label"],
                source_row["predicate_covered_label"],
                source_row["sufficiency_label"],
            ),
            "p2_secondary_reasons_3": [
                1 - source_row["frame_compatible_label"],
                1 - source_row["predicate_covered_label"],
                1 - source_row["sufficiency_label"],
            ],
            "historical_stage185_bridge_status": bridged.get("historical_bridge_status", "UNRESOLVED"),
            "historical_stage185_used_as_current_source_identity": False,
        }
        for key, value in bridged.items():
            if key not in row and key not in {"reason_codes"}:
                row[key] = value
        row["integrity_status"] = compose_integrity_status(row)
        p2_codes = p2_reason_exclusion_codes(
            source_row=source_row,
            sidecar_row=row,
            split=split,
            expected_canonical_row_id=canonical_id,
        )
        row["p2_reason_supervision_eligible"] = len(p2_codes) == 0
        row["p2_reason_exclusion_codes"] = p2_codes
        row["reason_codes"] = sorted(set(list(bridged.get("reason_codes", [])) + p2_codes))
        row["eligible_for_positive_margin"] = positive_margin_eligible(row)
        missing_sidecar = [field for field in P2_SIDE_CAR_REQUIRED_FIELDS + P4L_STAGE185_COMPATIBLE_FIELDS if field not in row]
        require(not missing_sidecar, f"SIDECAR_REQUIRED_FIELDS_MISSING:{row_id}:{missing_sidecar}")
        require(row["source_dataset_sha256"] == SOURCE_DATASET_SHA256, f"SIDECAR_SOURCE_SHA_MISMATCH:{row_id}")
        require(HISTORICAL_DATASET_SHA256 not in {row.get("source_dataset_sha256"), row.get("source_dataset_semantic_sha256")}, f"HISTORICAL_SHA_USED_AS_CURRENT:{row_id}")
        require(row["integrity_status"] in INTEGRITY_ENUM, f"INTEGRITY_STATUS_INVALID:{row_id}")
        require(isinstance(row["eligible_for_positive_margin"], bool), f"ELIGIBILITY_NOT_BOOL:{row_id}")
        sidecar.append(row)
    return sidecar


def validate_sidecar_rows(source_rows: Sequence[Mapping[str, Any]], sidecar_rows: Sequence[Mapping[str, Any]]) -> None:
    require(len(sidecar_rows) == len(source_rows) == EXPECTED_ROW_COUNT, "SIDECAR_ROW_COUNT_MISMATCH")
    source_ids = [str(row["id"]) for row in source_rows]
    sidecar_ids = [str(row["row_id"]) for row in sidecar_rows]
    require(source_ids == sidecar_ids, "SIDECAR_SOURCE_ORDER_MISMATCH")
    require(len(set(sidecar_ids)) == len(sidecar_ids), "SIDECAR_DUPLICATE_ROW_ID")
    for row in sidecar_rows:
        row_id = str(row["row_id"])
        require(row.get("reason_codes") == sorted(set(row.get("reason_codes", []))), f"REASON_CODES_NOT_SORTED_UNIQUE:{row_id}")
        require(row.get("source_dataset_path") == SOURCE_DATASET_PATH, f"SIDECAR_SOURCE_PATH_MISMATCH:{row_id}")
        require(row.get("source_dataset_sha256") == SOURCE_DATASET_SHA256, f"SIDECAR_CURRENT_SOURCE_SHA_MISMATCH:{row_id}")
        require(HISTORICAL_DATASET_SHA256 != row.get("source_dataset_sha256"), f"HISTORICAL_SOURCE_SHA_LEAK:{row_id}")
        require(row.get("historical_stage185_used_as_current_source_identity") is False, f"HISTORICAL_IDENTITY_FLAG_INVALID:{row_id}")


def resolve_builder_commit(repo_root: Path, supplied: str | None) -> str:
    if supplied:
        require(len(supplied) == 40 and all(ch in "0123456789abcdef" for ch in supplied), "BUILDER_COMMIT_INVALID")
        return supplied
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )
    commit = result.stdout.strip()
    require(len(commit) == 40, "BUILDER_COMMIT_RESOLUTION_FAILED")
    return commit


def canonical_output_dir(repo_root: Path, builder_commit: str) -> Path:
    return repo_path(
        repo_root,
        f"reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_{builder_commit}",
    )


def build_provenance(
    *,
    builder_commit: str,
    builder_source_sha256: str,
    output_dir: Path,
    sidecar_physical_sha256: str,
    sidecar_semantic_sha256: str,
    blockers: Sequence[str] = (),
    failure_reasons: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "authority_version": AUTHORITY_VERSION,
        "p4l_authority_path": P4L_AUTHORITY_PATH,
        "p4l_authority_commit": P4L_AUTHORITY_COMMIT,
        "builder_source_path": BUILDER_SOURCE_PATH,
        "builder_source_commit": builder_commit,
        "builder_source_sha256": builder_source_sha256,
        "p4h_authority_commit": P4H_AUTHORITY_COMMIT,
        "p4k_freeze_commit": P4K_FREEZE_COMMIT,
        "p4h_result_freeze_commit": P4H_RESULT_FREEZE_COMMIT,
        "p4h_verification_attestation_freeze_commit": P4H_VERIFICATION_ATTESTATION_FREEZE_COMMIT,
        "source_dataset_path": SOURCE_DATASET_PATH,
        "source_dataset_sha256": SOURCE_DATASET_SHA256,
        "source_dataset_semantic_sha256": SOURCE_DATASET_SEMANTIC_SHA256,
        "sidecar_path": f"{output_dir.as_posix()}/{SIDECAR_NAME}",
        "sidecar_physical_sha256": sidecar_physical_sha256,
        "sidecar_semantic_sha256": sidecar_semantic_sha256,
        "sidecar_schema_version": SIDECAR_SCHEMA_VERSION,
        "provenance_path": f"{output_dir.as_posix()}/{PROVENANCE_NAME}",
        "provenance_physical_sha256_self_certified": False,
        "row_count": EXPECTED_ROW_COUNT,
        "row_order_rule": "exact P4-B regenerated JSONL physical row order",
        "one_to_one_row_coverage": True,
        "unique_row_id": True,
        "split_rule": {"pair_level": True, "sorted_pair_ids": True, "shuffle_seed": SPLIT_SEED, "dev_ratio": DEV_RATIO},
        "canonical_row_rule": "same-pair intervention_type == 'none'; canonical target self-anchors",
        "p4b_compatibility_rows_path": P4B_COMPATIBILITY_ROWS_PATH,
        "p4b_compatibility_rows_sha256": P4B_COMPATIBILITY_ROWS_SHA256,
        "p4b_compatibility_summary_path": P4B_COMPATIBILITY_SUMMARY_PATH,
        "p4b_compatibility_summary_sha256": P4B_COMPATIBILITY_SUMMARY_SHA256,
        "p4b_compatibility_provenance_path": P4B_COMPATIBILITY_PROVENANCE_PATH,
        "p4b_compatibility_provenance_sha256": P4B_COMPATIBILITY_PROVENANCE_SHA256,
        "p4b_compatibility_authorized_scope": {"pair_count": P4B_AUTHORIZED_PAIR_COUNT, "member_count": P4B_AUTHORIZED_MEMBER_COUNT},
        "historical_stage185_sidecar_path": HISTORICAL_STAGE185_SIDECAR_PATH,
        "historical_stage185_sidecar_semantic_sha256": HISTORICAL_STAGE185_SIDECAR_SEMANTIC_SHA256,
        "historical_stage185_source_path": HISTORICAL_STAGE185_SOURCE_PATH,
        "historical_stage185_source_sha256": HISTORICAL_STAGE185_SOURCE_SHA256,
        "historical_stage185_used_as_current_source_identity": False,
        "training_admission_released": False,
        "implementation_authorized": True,
        "artifact_materialization_authorized_by_p4l": False,
        "a0_execution_authorized": False,
        "training_authorized": False,
        "evaluation_authorized": False,
        "kaggle_authorized": False,
        "gpu_authorized": False,
        "blockers": list(blockers),
        "failure_reasons": list(failure_reasons),
    }


def sidecar_payload_and_hashes(sidecar_rows: Sequence[Mapping[str, Any]]) -> tuple[bytes, str, str]:
    payload = compact_jsonl_bytes(sidecar_rows)
    require(payload.endswith(b"\n"), "SIDECAR_FINAL_NEWLINE_MISSING")
    require(not payload.startswith(b"\xef\xbb\xbf"), "SIDECAR_BOM_PRESENT")
    return payload, sha256_bytes(payload), semantic_sidecar_sha256(sidecar_rows)


def path_exists_or_symlink(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def finalize_payloads_atomic(output_dir: Path, payloads: Mapping[str, bytes]) -> str:
    require(set(payloads) == EXPECTED_OUTPUT_NAMES, "OUTPUT_PAYLOAD_SET_MISMATCH")
    require(not path_exists_or_symlink(output_dir), "P4L_OUTPUT_PATH_PREEXISTING")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir.parent / f".{output_dir.name}.p4l-staging-{uuid.uuid4().hex}"
    require(not path_exists_or_symlink(staging_dir), "P4L_STAGING_EXISTS")
    staging_created = False
    published = False
    try:
        staging_dir.mkdir()
        staging_created = True
        for name, payload in payloads.items():
            (staging_dir / name).write_bytes(payload)
        require(all((staging_dir / name).is_file() for name in EXPECTED_OUTPUT_NAMES), "P4L_STAGING_SET_MISMATCH")
        require(not path_exists_or_symlink(output_dir), "P4L_OUTPUT_PATH_PREEXISTING")
        require(os.name == "nt", "P4L_ATOMIC_DIRECTORY_NOREPLACE_UNSUPPORTED")
        try:
            staging_dir.rename(output_dir)
        except FileExistsError as exc:
            raise BuildBlocked("P4L_OUTPUT_PATH_PREEXISTING") from exc
        published = True
    except Exception:
        if staging_created and not published and staging_dir.is_dir() and not staging_dir.is_symlink():
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    require(all((output_dir / name).is_file() for name in EXPECTED_OUTPUT_NAMES), "P4L_OUTPUT_SET_MISMATCH")
    return "PUBLISHED"


def build_sidecar_artifacts(
    *,
    repo_root: Path,
    builder_commit: str | None = None,
    created_at: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], bytes, bytes]:
    inputs = validate_frozen_inputs(repo_root)
    source_rows = validate_source_dataset(repo_root, inputs["source_dataset"])
    split_by_pair = deterministic_pair_split(source_rows)
    canonical_by_row = canonical_row_ids(source_rows, split_by_pair)
    validate_canonical_lineage(source_rows, split_by_pair, canonical_by_row)
    historical_by_id = load_historical_stage185(inputs["historical_stage185"])
    p4b_by_member = load_p4b_compatibility(
        repo_root,
        inputs["p4b_rows"],
        inputs["p4b_summary"],
        inputs["p4b_provenance"],
    )
    commit = resolve_builder_commit(repo_root, builder_commit)
    output_dir = canonical_output_dir(repo_root, commit)
    builder_sha = file_sha256(inputs["builder_source"])
    timestamp = created_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    sidecar_rows = assemble_sidecar_rows(
        source_rows=source_rows,
        split_by_pair=split_by_pair,
        canonical_by_row=canonical_by_row,
        historical_by_id=historical_by_id,
        p4b_by_member=p4b_by_member,
        builder_source_sha256=builder_sha,
        created_at=timestamp,
    )
    validate_sidecar_rows(source_rows, sidecar_rows)
    sidecar_payload, sidecar_physical, sidecar_semantic = sidecar_payload_and_hashes(sidecar_rows)
    provenance = build_provenance(
        builder_commit=commit,
        builder_source_sha256=builder_sha,
        output_dir=Path(repo_relative(repo_root, output_dir)),
        sidecar_physical_sha256=sidecar_physical,
        sidecar_semantic_sha256=sidecar_semantic,
    )
    provenance_payload = deterministic_json_bytes(provenance)
    return sidecar_rows, provenance, sidecar_payload, provenance_payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--builder-commit", default=None)
    parser.add_argument("--created-at", default=None)
    parser.add_argument("--materialize", action="store_true", help="Write outputs atomically when a later authority permits execution.")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    sidecar_rows, provenance, sidecar_payload, provenance_payload = build_sidecar_artifacts(
        repo_root=repo_root,
        builder_commit=args.builder_commit,
        created_at=args.created_at,
    )
    result = {
        "namespace": NAMESPACE,
        "status": "PASS",
        "row_count": len(sidecar_rows),
        "sidecar_physical_sha256": sha256_bytes(sidecar_payload),
        "sidecar_semantic_sha256": semantic_sidecar_sha256(sidecar_rows),
        "provenance_physical_sha256": sha256_bytes(provenance_payload),
        "training_authorized": False,
        "evaluation_authorized": False,
        "artifact_materialization_requested": bool(args.materialize),
    }
    if args.materialize:
        output_dir = canonical_output_dir(repo_root, provenance["builder_source_commit"])
        result["publish_status"] = finalize_payloads_atomic(
            output_dir,
            {SIDECAR_NAME: sidecar_payload, PROVENANCE_NAME: provenance_payload},
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = run(parse_args(argv))
    except BuildBlocked as exc:
        raise SystemExit(f"P3W6F2P4L_CURRENT_LINEAGE_INTEGRITY_SIDECAR_BUILD_BLOCKED:{exc}") from exc
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
