from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Iterable


P1_AUTHORITY_COMMIT = "030fbf18e24a57caeaeb67e5a612445877094728"
P3W4_EXECUTION_COMMIT = "ca99038d812696467a4330cffc1c4c5b5f72cfe2"
P3W4_RESULT_AUTHORITY_COMMIT = "f0a9afddc5b93c54aa72b0335c5a1a2f517cf934"
P3W5_AUTHORITY_COMMIT = "01d983f8d09cacf0eddefd2014fc81a28771cf5e"
EXPECTED_HEAD = P1_AUTHORITY_COMMIT

P1_MANIFEST_PATH = "reports/reason_router_p2_p3w6f2p1_manual_review_execution_manifest.json"
DEFAULT_WIP_PATH = str(Path.home() / "p3w6f2_review_wip.jsonl")
EXECUTION_SCRIPT_REPO_PATH = "scripts/reason_router_p3w6f2_manual_review.py"

EXPECTED_PAIR_COUNT = 119
EXPECTED_MEMBER_COUNT = 357
P3W5_REVIEW_PROTOCOL_VERSION = "P3W5_F2_MANUAL_REVIEW_V1"
REVIEW_PROTOCOL_VERSION = "P3W6F2_HYBRID_HUMAN_REVIEW_V2"
COMPATIBILITY_MATRIX_VERSION = "F2_REVIEW_COMPATIBILITY_V1"
SOURCE_HASH_VERSION = "F2_SOURCE_RECORD_HASH_V1"
EXECUTION_STATUS_COMPLETE = "P3W6F2_HYBRID_HUMAN_REVIEW_EXECUTION_COMPLETE_PENDING_RESULT_REVIEW"
EXECUTION_DECISION_COMPLETE = "P3W6F2_HYBRID_HUMAN_REVIEW_LEVEL1_COMPLETE_PENDING_RESULT_REVIEW"
F2_OUTPUT_SHA256_CONTRACT = "NON_SELF_REFERENTIAL_ARTIFACT_SHA256_MAP_V1"
STRUCTURAL_GATE_VERSION = "P3W6F2_STRUCTURAL_COHORT_GATE_V1"
AI_PRESCREEN_PROTOCOL_VERSION = "P3W6F2_AI_ASSISTED_PRESCREEN_V1"
XLSX_CONFIRMED_IMPORT_INTERMEDIATE_PROTOCOL_VERSION = "P3W6F2_XLSX_CONFIRMED_IMPORT_INTERMEDIATE_V1"
COHORT_CONFIRMATION_PROTOCOL_VERSION = "P3W6F2_STRUCTURAL_COHORT_CONFIRMATION_V1"
STRUCTURAL_AUDIT_SCHEMA_VERSION = "reason_router_p3w6f2_structural_cohort_audit_v1"
COHORT_CONFIRMATION_SCHEMA_VERSION = "reason_router_p3w6f2_structural_cohort_confirmation_v1"
INDIVIDUAL_REVIEW_METHOD = "INDIVIDUAL_TRIPLE_REVIEW"
STRUCTURAL_COHORT_METHOD = "STRUCTURAL_COHORT_CONFIRMATION"
NO_COHORT_LINKAGE = ""
NOT_CAPTURED_IN_XLSX = "NOT_CAPTURED_IN_XLSX"
CAPTURED_IN_RECORD = "CAPTURED_IN_RECORD"
NOT_APPLICABLE_STRUCTURAL_COHORT = "NOT_APPLICABLE_STRUCTURAL_COHORT"
CLI_INDIVIDUAL_RECORD = "CLI_INDIVIDUAL_RECORD"
XLSX_CONFIRMED_IMPORT = "XLSX_CONFIRMED_IMPORT"
STRUCTURAL_COHORT_RECORD_ORIGIN = "STRUCTURAL_COHORT_CONFIRMATION"
COHORT_CONFIRM_ACTION = "COHORT_CONFIRM"

INDIVIDUALLY_REVIEWED_AUDIT_PAIR_IDS = [
    "generated_fact_152",
    "generated_fact_154",
    "generated_fact_156",
    "generated_fact_157",
    "generated_fact_159",
    "generated_fact_160",
    "generated_fact_162",
    "generated_fact_163",
    "generated_fact_164",
    "generated_fact_167",
    "generated_fact_168",
    "generated_fact_169",
    "generated_fact_170",
    "generated_fact_171",
    "generated_fact_173",
    "generated_fact_175",
    "generated_fact_177",
    "generated_fact_178",
    "generated_fact_179",
    "generated_fact_180",
]
EXPECTED_AUDIT_PREDICATE_COVERAGE = {
    "restored": 4,
    "selected": 4,
    "approved": 4,
    "delivered": 3,
    "published": 2,
    "opened": 2,
    "launched": 1,
}

SEMANTIC_ENUMS = frozenset({"VALID", "INVALID", "UNCLEAR"})
GRAMMAR_ENUMS = frozenset(
    {
        "CANONICAL_ONLY_DEFECT",
        "MULTI_MEMBER_DEFECT",
        "NO_REPRODUCIBLE_DEFECT",
        "UNCLEAR",
    }
)
AUTHORITY_DECISION_ENUMS = frozenset(
    {
        "CANONICAL_TEXTUAL_REPAIR_CANDIDATE",
        "CANONICAL_REGENERATION_REQUIRED",
        "SEMANTIC_CONFLICT",
        "INSUFFICIENT_EVIDENCE_KEEP_BLOCKED",
        "NO_REPRODUCIBLE_DEFECT_KEEP_BLOCKED",
    }
)
REVIEW_METHOD_ENUMS = frozenset({INDIVIDUAL_REVIEW_METHOD, STRUCTURAL_COHORT_METHOD})
HUMAN_REVIEW_TIME_PROVENANCE_ENUMS = frozenset({CAPTURED_IN_RECORD, NOT_CAPTURED_IN_XLSX, NOT_APPLICABLE_STRUCTURAL_COHORT})
RECORD_ORIGIN_ENUMS = frozenset({CLI_INDIVIDUAL_RECORD, XLSX_CONFIRMED_IMPORT, STRUCTURAL_COHORT_RECORD_ORIGIN})
AI_PRESCREEN_SUGGESTION_FIELDS = [
    "ai_canonical_semantics_suggestion",
    "ai_paraphrase_semantics_suggestion",
    "ai_polarity_flip_semantics_suggestion",
    "ai_grammar_validity_suggestion",
]
AI_PRESCREEN_FIELDS = [
    "pair_id",
    "source_record_sha256",
    *AI_PRESCREEN_SUGGESTION_FIELDS,
    "ai_triage_status",
    "ai_prescreen_protocol_version",
    "ai_prescreen_model_or_system_id",
    "ai_prescreen_created_at_utc",
]
AI_PRESCREEN_FIELD_SET = frozenset(AI_PRESCREEN_FIELDS)
XLSX_IMPORT_INTERMEDIATE_FIELDS = [
    "schema_version",
    "import_intermediate_protocol_version",
    "source_workbook_sha256",
    "source_workbook_name",
    "conversion_created_at_utc",
    "row_count",
    "records",
]
XLSX_IMPORT_INTERMEDIATE_FIELD_SET = frozenset(XLSX_IMPORT_INTERMEDIATE_FIELDS)
XLSX_IMPORT_RECORD_FIELDS = [
    "pair_id",
    "source_record_sha256",
    "ai_canonical_semantics_suggestion",
    "ai_paraphrase_semantics_suggestion",
    "ai_polarity_flip_semantics_suggestion",
    "ai_grammar_validity_suggestion",
    "ai_triage_status",
    "human_review_action",
    "human_override_canonical_semantics",
    "human_override_paraphrase_semantics",
    "human_override_polarity_flip_semantics",
    "human_override_grammar_validity",
    "derived_human_canonical_semantics",
    "derived_human_paraphrase_semantics",
    "derived_human_polarity_flip_semantics",
    "derived_human_grammar_validity",
    "derived_human_authority_decision",
    "review_status",
]
XLSX_IMPORT_RECORD_FIELD_SET = frozenset(XLSX_IMPORT_RECORD_FIELDS)
COHORT_CONFIRMATION_FIELDS = [
    "schema_version",
    "cohort_confirmation_protocol_version",
    "cohort_confirmation_id",
    "confirmation_payload_sha256",
    "authority_recorded_at_utc",
    "reviewer_id",
    "structural_audit_sha256",
    "structural_audit_path_identity",
    "ai_prescreen_artifact_sha256",
    "validated_individual_wip_state_sha256",
    "eligible_pair_count",
    "exception_pair_count",
    "eligible_pair_ids",
    "exception_pair_ids",
    "individually_reviewed_pair_ids",
    "cohort_confirmed_pair_ids",
    "structural_gate_version",
    "structural_gate_result",
    "ai_prescreen_protocol_version",
    "ai_prescreen_result_summary",
    "source_authority_identity",
    "human_action",
]
COHORT_CONFIRMATION_FIELD_SET = frozenset(COHORT_CONFIRMATION_FIELDS)
COHORT_CONFIRMATION_PAYLOAD_FIELDS = [
    field
    for field in COHORT_CONFIRMATION_FIELDS
    if field not in {"cohort_confirmation_id", "confirmation_payload_sha256"}
]
STRUCTURAL_AUDIT_FIELDS = [
    "schema_version",
    "structural_gate_version",
    "audit_created_at_utc",
    "source_authority_identity",
    "authorized_pair_count",
    "authorized_member_count",
    "source_record_sha256_by_pair",
    "ai_prescreen_artifact_sha256",
    "ai_prescreen_protocol_version",
    "validated_individual_wip_state_sha256",
    "required_individual_audit_pair_ids",
    "eligible_pair_ids",
    "exception_pair_ids",
    "structural_gate_result_by_pair",
    "overall_structural_gate_result",
    "audit_payload_sha256",
]
STRUCTURAL_AUDIT_FIELD_SET = frozenset(STRUCTURAL_AUDIT_FIELDS)

SOURCE_FIELDS = [
    "pair_id",
    "canonical_none_row_id",
    "paraphrase_row_id",
    "polarity_flip_row_id",
    "canonical_final_label",
    "paraphrase_final_label",
    "polarity_flip_final_label",
    "canonical_claim",
    "paraphrase_claim",
    "polarity_flip_claim",
    "canonical_evidence",
    "paraphrase_evidence",
    "polarity_flip_evidence",
    "canonical_grammar_status",
    "paraphrase_grammar_status",
    "polarity_flip_grammar_status",
    "canonical_reason_codes",
    "paraphrase_reason_codes",
    "polarity_flip_reason_codes",
    "canonical_claim_text_diff_summary",
    "paraphrase_claim_text_diff_summary",
    "polarity_flip_claim_text_diff_summary",
    "canonical_evidence_text_diff_summary",
    "paraphrase_evidence_text_diff_summary",
    "polarity_flip_evidence_text_diff_summary",
    "automatic_root_cause_class",
    "automatic_evidence",
]
HUMAN_FIELDS = [
    "human_canonical_semantics",
    "human_paraphrase_semantics",
    "human_polarity_flip_semantics",
    "human_grammar_validity",
    "human_authority_decision",
    "human_notes",
]
P3W5_PROVENANCE_FIELDS = [
    "source_record_sha256",
    "reviewer_id",
    "review_protocol_version",
    "reviewed_at_utc",
]
PROVENANCE_FIELDS = [
    "source_record_sha256",
    "reviewer_id",
    "review_protocol_version",
    "reviewed_at_utc",
    "authority_recorded_at_utc",
    "human_review_time_provenance",
    "record_origin",
    "review_method",
    "cohort_confirmation_id",
]
WIP_FIELDS = [
    "pair_id",
    "source_record_sha256",
    "human_canonical_semantics",
    "human_paraphrase_semantics",
    "human_polarity_flip_semantics",
    "human_grammar_validity",
    "human_authority_decision",
    "human_notes",
    "reviewer_id",
    "review_protocol_version",
    "reviewed_at_utc",
    "authority_recorded_at_utc",
    "human_review_time_provenance",
    "record_origin",
    "review_method",
    "cohort_confirmation_id",
]
WIP_FIELD_SET = frozenset(WIP_FIELDS)
COMPLETED_CSV_FIELDS = [*SOURCE_FIELDS, *HUMAN_FIELDS, *PROVENANCE_FIELDS]
P3W5_COMPLETED_CSV_FIELDS = [*SOURCE_FIELDS, *HUMAN_FIELDS, *P3W5_PROVENANCE_FIELDS]
DECISIONS_JSONL_FIELDS = [
    "pair_id",
    "source_record_sha256",
    "human_canonical_semantics",
    "human_paraphrase_semantics",
    "human_polarity_flip_semantics",
    "human_grammar_validity",
    "human_authority_decision",
    "human_notes",
    "reviewer_id",
    "review_protocol_version",
    "reviewed_at_utc",
    "authority_recorded_at_utc",
    "human_review_time_provenance",
    "record_origin",
    "review_method",
    "cohort_confirmation_id",
    "compatibility_matrix_version",
    "compatibility_matrix_expected_decision",
    "compatibility_matrix_match",
    "review_record_valid",
    "ordered_validation_errors",
]

VALIDATION_FAILURE_CODES = [
    "AUTHORITY_PAIR_UNIVERSE_MISMATCH",
    "SOURCE_SCHEMA_MISMATCH",
    "SOURCE_RECORD_HASH_MISMATCH",
    "SOURCE_FIELD_MUTATION",
    "MISSING_HUMAN_FIELD",
    "INVALID_SEMANTIC_ENUM",
    "INVALID_GRAMMAR_ENUM",
    "INVALID_AUTHORITY_DECISION_ENUM",
    "COMPATIBILITY_MATRIX_MISMATCH",
    "MISSING_REQUIRED_NOTES",
    "MISSING_REVIEWER_ID",
    "INVALID_REVIEW_PROTOCOL_VERSION",
    "INVALID_REVIEW_TIMESTAMP",
    "INVALID_REVIEW_METHOD",
    "INVALID_REVIEW_TIME_PROVENANCE",
    "INVALID_COHORT_CONFIRMATION_LINKAGE",
    "INVALID_COHORT_CONFIRMATION_ARTIFACT",
    "CONFIRMATION_PAYLOAD_HASH_MISMATCH",
    "COHORT_CONFIRMATION_ID_MISMATCH",
    "INVALID_AI_PRESCREEN_ARTIFACT",
    "STRUCTURAL_COHORT_GATE_FAILED",
    "MISSING_INDIVIDUAL_AUDIT_EVIDENCE",
    "AUDIT_PREDICATE_COVERAGE_MISMATCH",
    "DUPLICATE_PAIR_ID",
    "MISSING_PAIR_ID",
    "UNAUTHORIZED_PAIR_ID",
    "WIP_SCHEMA_MISMATCH",
    "COUNT_ARRAY_ASYMMETRY",
    "DECISION_PARTITION_MISMATCH",
    "DIRTY_TRACKED_EXECUTION_STATE",
    "UNTRACKED_EXECUTION_SCRIPT",
    "OUTPUT_PATH_MISMATCH",
]

COUNT_ARRAY_PAIRS = {
    "authorized_F2_pair_count": "authorized_F2_pair_ids",
    "reviewed_pair_count": "reviewed_pair_ids",
    "unreviewed_pair_count": "unreviewed_pair_ids",
    "textual_repair_candidate_count": "textual_repair_candidate_pair_ids",
    "regeneration_required_count": "regeneration_required_pair_ids",
    "semantic_conflict_count": "semantic_conflict_pair_ids",
    "insufficient_evidence_count": "insufficient_evidence_pair_ids",
    "no_reproducible_defect_keep_blocked_count": "no_reproducible_defect_keep_blocked_pair_ids",
    "invalid_review_count": "invalid_review_pair_ids",
    "invalid_combination_count": "invalid_combination_pair_ids",
    "missing_reviewer_provenance_count": "missing_reviewer_provenance_pair_ids",
    "source_hash_mismatch_count": "source_hash_mismatch_pair_ids",
    "missing_human_field_count": "missing_human_field_pair_ids",
    "invalid_enum_count": "invalid_enum_pair_ids",
    "missing_required_notes_count": "missing_required_notes_pair_ids",
    "completed_decision_pair_count": "completed_decision_pair_ids",
    "individual_review_pair_count": "individual_review_pair_ids",
    "structural_cohort_confirmation_pair_count": "structural_cohort_confirmation_pair_ids",
}

DECISION_TO_SUMMARY_ARRAY = {
    "CANONICAL_TEXTUAL_REPAIR_CANDIDATE": "textual_repair_candidate_pair_ids",
    "CANONICAL_REGENERATION_REQUIRED": "regeneration_required_pair_ids",
    "SEMANTIC_CONFLICT": "semantic_conflict_pair_ids",
    "INSUFFICIENT_EVIDENCE_KEEP_BLOCKED": "insufficient_evidence_pair_ids",
    "NO_REPRODUCIBLE_DEFECT_KEEP_BLOCKED": "no_reproducible_defect_keep_blocked_pair_ids",
}


class ReviewInfrastructureError(ValueError):
    pass


@dataclass(frozen=True)
class Authority:
    repo_root: Path
    manifest: dict[str, Any]
    source_rows: list[dict[str, str]]
    pair_records: list[dict[str, Any]]
    summary: dict[str, Any]
    source_sha256_by_pair_id: dict[str, str]
    p3w4_artifact_commit: str
    input_artifact_paths: dict[str, str]
    input_artifact_sha256: dict[str, str]
    source_template_fields: list[str]

    @property
    def ordered_pair_ids(self) -> list[str]:
        return [row["pair_id"] for row in self.source_rows]

    @property
    def row_by_pair_id(self) -> dict[str, dict[str, str]]:
        return {row["pair_id"]: row for row in self.source_rows}


@dataclass(frozen=True)
class CohortConfirmationAuthority:
    cohort_confirmation_id: str
    cohort_confirmed_pair_ids: frozenset[str]
    individually_reviewed_pair_ids: frozenset[str]
    structural_audit_sha256: str
    structural_gate_version: str
    source_authority_identity: dict[str, Any]
    ai_prescreen_artifact_sha256: str
    validated_individual_wip_state_sha256: str


def require(condition: bool, code: str, detail: str = "") -> None:
    if not condition:
        raise ReviewInfrastructureError(code if not detail else f"{code}: {detail}")


def require_no_validation_errors(errors: list[str]) -> None:
    if errors:
        raise ReviewInfrastructureError(f"{errors[0]}: {','.join(errors)}")


def resolve_repo_root(start: Path | None = None) -> Path:
    cwd = (start or Path.cwd()).resolve()
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return Path(result.stdout.decode("utf-8").strip()).resolve()


def git_object_bytes(repo_root: Path, commit: str, path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ReviewInfrastructureError(f"AUTHORITY_PAIR_UNIVERSE_MISMATCH: git object missing {commit}:{path}: {detail}")
    return result.stdout


def require_commit(repo_root: Path, commit: str) -> None:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    require(result.returncode == 0, "AUTHORITY_PAIR_UNIVERSE_MISMATCH", f"missing commit {commit}")


def current_head(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ReviewInfrastructureError(f"DIRTY_TRACKED_EXECUTION_STATE: unable to resolve HEAD: {detail}")
    return result.stdout.decode("utf-8").strip()


def require_git_quiet(repo_root: Path, args: list[str], code: str, detail: str) -> None:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise ReviewInfrastructureError(f"{code}: {detail}" if not stderr else f"{code}: {detail}: {stderr}")


def require_tracked_execution_state(repo_root: Path) -> str:
    head = current_head(repo_root)
    result = subprocess.run(
        ["git", "cat-file", "-e", f"HEAD:{EXECUTION_SCRIPT_REPO_PATH}"],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    require(result.returncode == 0, "UNTRACKED_EXECUTION_SCRIPT", EXECUTION_SCRIPT_REPO_PATH)
    require_git_quiet(repo_root, ["diff", "--quiet", "HEAD", "--"], "DIRTY_TRACKED_EXECUTION_STATE", "unstaged tracked changes")
    require_git_quiet(repo_root, ["diff", "--cached", "--quiet", "HEAD", "--"], "DIRTY_TRACKED_EXECUTION_STATE", "staged tracked changes")
    return head


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def parse_json_object(payload: bytes, path: str) -> dict[str, Any]:
    value = json.loads(payload.decode("utf-8"))
    require(isinstance(value, dict), "AUTHORITY_PAIR_UNIVERSE_MISMATCH", f"{path} is not a JSON object")
    return value


def parse_jsonl_objects(payload: bytes, path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    text = payload.decode("utf-8")
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        require(isinstance(value, dict), "AUTHORITY_PAIR_UNIVERSE_MISMATCH", f"{path}:{line_number} is not an object")
        records.append(value)
    return records


def parse_source_csv(payload: bytes, source_fields: list[str]) -> list[dict[str, str]]:
    text = payload.decode("utf-8")
    reader = csv.DictReader(StringIO(text), restkey="__extra__", restval=None)
    fieldnames = reader.fieldnames or []
    allowed_template_fields = [*source_fields, *HUMAN_FIELDS]
    require(
        fieldnames == allowed_template_fields,
        "SOURCE_SCHEMA_MISMATCH",
        "authority template must be exact 27 source + 6 empty human fields",
    )
    rows: list[dict[str, str]] = []
    for row_number, row in enumerate(reader, 2):
        require("__extra__" not in row or row["__extra__"] is None, "SOURCE_SCHEMA_MISMATCH", f"extra fields at CSV row {row_number}")
        require(all(row.get(field) is not None for field in fieldnames), "SOURCE_SCHEMA_MISMATCH", f"missing field at CSV row {row_number}")
        require(all(row[field] == "" for field in HUMAN_FIELDS), "SOURCE_FIELD_MUTATION", f"nonempty human template field at CSV row {row_number}")
        rows.append({field: row[field] for field in source_fields})
    return rows


def compute_source_record_sha256(row: dict[str, str], source_fields: Iterable[str] = SOURCE_FIELDS) -> str:
    values = [row[field] for field in source_fields]
    payload = json.dumps(values, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def derive_authority_decision(
    human_canonical_semantics: str,
    human_paraphrase_semantics: str,
    human_polarity_flip_semantics: str,
    human_grammar_validity: str,
) -> str:
    semantics = [human_canonical_semantics, human_paraphrase_semantics, human_polarity_flip_semantics]
    if any(value not in SEMANTIC_ENUMS for value in semantics):
        raise ReviewInfrastructureError("INVALID_SEMANTIC_ENUM")
    if human_grammar_validity not in GRAMMAR_ENUMS:
        raise ReviewInfrastructureError("INVALID_GRAMMAR_ENUM")
    if "UNCLEAR" in semantics or human_grammar_validity == "UNCLEAR":
        return "INSUFFICIENT_EVIDENCE_KEEP_BLOCKED"
    if "INVALID" in semantics:
        return "SEMANTIC_CONFLICT"
    if all(value == "VALID" for value in semantics):
        if human_grammar_validity == "CANONICAL_ONLY_DEFECT":
            return "CANONICAL_TEXTUAL_REPAIR_CANDIDATE"
        if human_grammar_validity == "MULTI_MEMBER_DEFECT":
            return "CANONICAL_REGENERATION_REQUIRED"
        if human_grammar_validity == "NO_REPRODUCIBLE_DEFECT":
            return "NO_REPRODUCIBLE_DEFECT_KEEP_BLOCKED"
    raise ReviewInfrastructureError("COMPATIBILITY_MATRIX_MISMATCH")


def notes_required(record: dict[str, Any]) -> bool:
    decision = record.get("human_authority_decision", "")
    semantics = [
        record.get("human_canonical_semantics", ""),
        record.get("human_paraphrase_semantics", ""),
        record.get("human_polarity_flip_semantics", ""),
    ]
    return (
        "UNCLEAR" in semantics
        or record.get("human_grammar_validity") == "UNCLEAR"
        or decision
        in {
            "SEMANTIC_CONFLICT",
            "INSUFFICIENT_EVIDENCE_KEEP_BLOCKED",
            "NO_REPRODUCIBLE_DEFECT_KEEP_BLOCKED",
        }
    )


def validate_reviewer_provenance(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    reviewer_id = record.get("reviewer_id", "")
    if not isinstance(reviewer_id, str) or reviewer_id == "":
        errors.append("MISSING_REVIEWER_ID")
    elif reviewer_id != reviewer_id.strip():
        errors.append("MISSING_REVIEWER_ID")
    if record.get("review_protocol_version") != REVIEW_PROTOCOL_VERSION:
        errors.append("INVALID_REVIEW_PROTOCOL_VERSION")
    if not valid_rfc3339_utc_z(record.get("reviewed_at_utc", "")):
        errors.append("INVALID_REVIEW_TIMESTAMP")
    if not valid_rfc3339_utc_z(record.get("authority_recorded_at_utc", "")):
        errors.append("INVALID_REVIEW_TIMESTAMP")
    if record.get("reviewed_at_utc") != record.get("authority_recorded_at_utc"):
        errors.append("INVALID_REVIEW_TIMESTAMP")
    if record.get("human_review_time_provenance") not in HUMAN_REVIEW_TIME_PROVENANCE_ENUMS:
        errors.append("INVALID_REVIEW_TIME_PROVENANCE")
    return errors


def validate_review_method_provenance(
    record: dict[str, Any],
    cohort_confirmations: dict[str, CohortConfirmationAuthority] | None = None,
) -> list[str]:
    errors: list[str] = []
    method = record.get("review_method")
    linkage = record.get("cohort_confirmation_id")
    provenance = record.get("human_review_time_provenance")
    origin = record.get("record_origin")
    if method not in REVIEW_METHOD_ENUMS:
        return ["INVALID_REVIEW_METHOD"]
    if origin not in RECORD_ORIGIN_ENUMS:
        errors.append("INVALID_REVIEW_TIME_PROVENANCE")
    if method == INDIVIDUAL_REVIEW_METHOD:
        if linkage != NO_COHORT_LINKAGE:
            errors.append("INVALID_COHORT_CONFIRMATION_LINKAGE")
        allowed = {
            CLI_INDIVIDUAL_RECORD: CAPTURED_IN_RECORD,
            XLSX_CONFIRMED_IMPORT: NOT_CAPTURED_IN_XLSX,
        }
        if allowed.get(origin) != provenance:
            errors.append("INVALID_REVIEW_TIME_PROVENANCE")
    if method == STRUCTURAL_COHORT_METHOD:
        if not isinstance(linkage, str) or linkage == "":
            errors.append("INVALID_COHORT_CONFIRMATION_LINKAGE")
        elif cohort_confirmations is None or linkage not in cohort_confirmations:
            errors.append("INVALID_COHORT_CONFIRMATION_LINKAGE")
        else:
            artifact = cohort_confirmations[linkage]
            pair_id = record.get("pair_id")
            if pair_id not in artifact.cohort_confirmed_pair_ids:
                errors.append("INVALID_COHORT_CONFIRMATION_LINKAGE")
            if pair_id in artifact.individually_reviewed_pair_ids:
                errors.append("INVALID_COHORT_CONFIRMATION_LINKAGE")
            if artifact.structural_gate_version != STRUCTURAL_GATE_VERSION:
                errors.append("INVALID_COHORT_CONFIRMATION_LINKAGE")
            if artifact.source_authority_identity == {}:
                errors.append("INVALID_COHORT_CONFIRMATION_LINKAGE")
        if provenance != NOT_APPLICABLE_STRUCTURAL_COHORT:
            errors.append("INVALID_REVIEW_TIME_PROVENANCE")
        if origin != STRUCTURAL_COHORT_RECORD_ORIGIN:
            errors.append("INVALID_REVIEW_TIME_PROVENANCE")
    return errors


def valid_rfc3339_utc_z(value: Any) -> bool:
    if not isinstance(value, str) or not value.endswith("Z"):
        return False
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return False
    return parsed.utcoffset() is not None and parsed.utcoffset().total_seconds() == 0


def utc_timestamp(clock: Callable[[], datetime] | None = None) -> str:
    now = (clock or (lambda: datetime.now(UTC)))()
    if now.tzinfo is None or now.utcoffset() is None:
        raise ReviewInfrastructureError("INVALID_REVIEW_TIMESTAMP: clock returned naive datetime")
    return now.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def validate_review_record(
    record: dict[str, Any],
    authority: Authority,
    duplicate_pair_ids: set[str] | None = None,
    cohort_confirmations: dict[str, CohortConfirmationAuthority] | None = None,
) -> list[str]:
    errors: list[str] = []
    pair_id = record.get("pair_id", "")
    authorized = authority.row_by_pair_id
    duplicate_pair_ids = duplicate_pair_ids or set()

    if not pair_id:
        errors.append("MISSING_PAIR_ID")
    elif pair_id not in authorized:
        errors.append("UNAUTHORIZED_PAIR_ID")
    if pair_id in duplicate_pair_ids:
        errors.append("DUPLICATE_PAIR_ID")

    if any(field in record for field in SOURCE_FIELDS if field != "pair_id"):
        errors.append("SOURCE_FIELD_MUTATION")

    if pair_id in authorized and record.get("source_record_sha256") != authority.source_sha256_by_pair_id[pair_id]:
        errors.append("SOURCE_RECORD_HASH_MISMATCH")

    for field in HUMAN_FIELDS:
        if field not in record or record[field] is None:
            errors.append("MISSING_HUMAN_FIELD")

    errors.extend(validate_reviewer_provenance(record))
    errors.extend(validate_review_method_provenance(record, cohort_confirmations))

    semantics = [
        record.get("human_canonical_semantics"),
        record.get("human_paraphrase_semantics"),
        record.get("human_polarity_flip_semantics"),
    ]
    if any(value not in SEMANTIC_ENUMS for value in semantics):
        errors.append("INVALID_SEMANTIC_ENUM")
    if record.get("human_grammar_validity") not in GRAMMAR_ENUMS:
        errors.append("INVALID_GRAMMAR_ENUM")
    if record.get("human_authority_decision") not in AUTHORITY_DECISION_ENUMS:
        errors.append("INVALID_AUTHORITY_DECISION_ENUM")

    if not any(code in errors for code in ("INVALID_SEMANTIC_ENUM", "INVALID_GRAMMAR_ENUM", "INVALID_AUTHORITY_DECISION_ENUM")):
        expected = derive_authority_decision(
            str(record["human_canonical_semantics"]),
            str(record["human_paraphrase_semantics"]),
            str(record["human_polarity_flip_semantics"]),
            str(record["human_grammar_validity"]),
        )
        if record.get("human_authority_decision") != expected:
            errors.append("COMPATIBILITY_MATRIX_MISMATCH")

    notes = record.get("human_notes", "")
    if notes_required(record) and (not isinstance(notes, str) or notes.strip() == ""):
        errors.append("MISSING_REQUIRED_NOTES")

    return order_errors(errors)


def order_errors(errors: Iterable[str]) -> list[str]:
    rank = {code: index for index, code in enumerate(VALIDATION_FAILURE_CODES)}
    return sorted(set(errors), key=lambda code: (rank.get(code, len(rank)), code))


def load_authority(repo_root: Path | None = None, p3w4_artifact_commit: str = P3W4_RESULT_AUTHORITY_COMMIT) -> Authority:
    repo = resolve_repo_root(repo_root)
    for commit in (P1_AUTHORITY_COMMIT, P3W4_EXECUTION_COMMIT, P3W4_RESULT_AUTHORITY_COMMIT, P3W5_AUTHORITY_COMMIT):
        require_commit(repo, commit)

    manifest = parse_json_object(git_object_bytes(repo, P1_AUTHORITY_COMMIT, P1_MANIFEST_PATH), P1_MANIFEST_PATH)
    require(manifest.get("review_protocol_version") == P3W5_REVIEW_PROTOCOL_VERSION, "INVALID_REVIEW_PROTOCOL_VERSION")
    require(manifest.get("compatibility_matrix_version") == COMPATIBILITY_MATRIX_VERSION, "COMPATIBILITY_MATRIX_MISMATCH")
    require(manifest.get("source_hash_version") == SOURCE_HASH_VERSION, "SOURCE_SCHEMA_MISMATCH")

    source_fields = list(manifest.get("immutable_source_columns", []))
    require(source_fields == SOURCE_FIELDS, "SOURCE_SCHEMA_MISMATCH", "P1 immutable source columns differ from implementation constants")
    require(list(manifest.get("human_review_fields", [])) == HUMAN_FIELDS, "SOURCE_SCHEMA_MISMATCH", "P1 human fields differ")
    require(list(manifest.get("reviewer_provenance_fields", [])) == P3W5_PROVENANCE_FIELDS, "SOURCE_SCHEMA_MISMATCH", "P1 provenance fields differ")
    require(list(manifest.get("completed_csv_schema_order", [])) == P3W5_COMPLETED_CSV_FIELDS, "SOURCE_SCHEMA_MISMATCH", "P1 completed schema differs")

    input_artifacts = manifest["input_artifact_authority"]
    template_path = input_artifacts["template_path"]
    pair_path = input_artifacts["pair_authority_path"]
    summary_path = input_artifacts["summary_authority_path"]

    template_bytes = git_object_bytes(repo, p3w4_artifact_commit, template_path)
    pair_bytes = git_object_bytes(repo, p3w4_artifact_commit, pair_path)
    summary_bytes = git_object_bytes(repo, p3w4_artifact_commit, summary_path)
    observed_sha = {
        template_path: sha256_bytes(template_bytes),
        pair_path: sha256_bytes(pair_bytes),
        summary_path: sha256_bytes(summary_bytes),
    }
    expected_sha = {
        template_path: input_artifacts["template_git_lf_sha256"],
        pair_path: input_artifacts["pair_authority_sha256"],
        summary_path: input_artifacts["summary_authority_sha256"],
    }
    for path, expected in expected_sha.items():
        require(observed_sha[path] == expected, "SOURCE_RECORD_HASH_MISMATCH", f"{path} Git-object SHA mismatch")
    require(
        observed_sha[template_path] != input_artifacts.get("template_historical_original_execution_crlf_sha256"),
        "SOURCE_RECORD_HASH_MISMATCH",
        "template matched historical CRLF worktree SHA instead of Git/LF authority",
    )

    source_rows = parse_source_csv(template_bytes, source_fields)
    require(len(source_rows) == EXPECTED_PAIR_COUNT, "AUTHORITY_PAIR_UNIVERSE_MISMATCH", "unexpected source row count")
    pair_ids = [row["pair_id"] for row in source_rows]
    require(len(set(pair_ids)) == EXPECTED_PAIR_COUNT, "AUTHORITY_PAIR_UNIVERSE_MISMATCH", "unexpected unique pair count")

    pair_records = parse_jsonl_objects(pair_bytes, pair_path)
    summary = parse_json_object(summary_bytes, summary_path)
    verify_pair_membership(source_rows, pair_records, summary)
    source_sha256_by_pair_id = {row["pair_id"]: compute_source_record_sha256(row, source_fields) for row in source_rows}
    return Authority(
        repo_root=repo,
        manifest=manifest,
        source_rows=source_rows,
        pair_records=pair_records,
        summary=summary,
        source_sha256_by_pair_id=source_sha256_by_pair_id,
        p3w4_artifact_commit=p3w4_artifact_commit,
        input_artifact_paths={"template": template_path, "pairs": pair_path, "summary": summary_path},
        input_artifact_sha256=observed_sha,
        source_template_fields=[*source_fields, *HUMAN_FIELDS],
    )


def verify_pair_membership(source_rows: list[dict[str, str]], pair_records: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    f2_records = [record for record in pair_records if record.get("family") == "F2"]
    require(len(f2_records) == EXPECTED_PAIR_COUNT, "AUTHORITY_PAIR_UNIVERSE_MISMATCH", "P3-W4 pair authority F2 count mismatch")
    f2_by_pair = {str(record.get("pair_id", "")): record for record in f2_records}
    require(len(f2_by_pair) == EXPECTED_PAIR_COUNT, "AUTHORITY_PAIR_UNIVERSE_MISMATCH", "P3-W4 duplicate F2 pair IDs")
    expected_members = 0
    for row in source_rows:
        pair_record = f2_by_pair.get(row["pair_id"])
        require(pair_record is not None, "AUTHORITY_PAIR_UNIVERSE_MISMATCH", f"missing pair {row['pair_id']}")
        members = pair_record.get("members", {})
        canonical = members.get("canonical", {}).get("source_row", {})
        paraphrase = members.get("paraphrase", {}).get("source_row", {})
        polarity = members.get("polarity_flip", {}).get("source_row", {})
        checks = [
            (row["canonical_none_row_id"], canonical.get("id")),
            (row["paraphrase_row_id"], paraphrase.get("id")),
            (row["polarity_flip_row_id"], polarity.get("id")),
            (row["canonical_final_label"], canonical.get("final_label")),
            (row["paraphrase_final_label"], paraphrase.get("final_label")),
            (row["polarity_flip_final_label"], polarity.get("final_label")),
        ]
        require(all(left == right for left, right in checks), "AUTHORITY_PAIR_UNIVERSE_MISMATCH", f"member mismatch for {row['pair_id']}")
        expected_members += 3
    require(expected_members == EXPECTED_MEMBER_COUNT, "AUTHORITY_PAIR_UNIVERSE_MISMATCH", "F2 member count mismatch")
    family_counts = summary.get("aggregates", {}).get("family_counts", {})
    require(family_counts.get("F2_pair_count") == EXPECTED_PAIR_COUNT, "AUTHORITY_PAIR_UNIVERSE_MISMATCH", "summary F2 pair count mismatch")
    require(family_counts.get("F2_complete_triple_members") == EXPECTED_MEMBER_COUNT, "AUTHORITY_PAIR_UNIVERSE_MISMATCH", "summary F2 member count mismatch")
    require(set(f2_by_pair).isdisjoint({record.get("pair_id") for record in pair_records if record.get("family") == "F1"}), "AUTHORITY_PAIR_UNIVERSE_MISMATCH", "F1/F2 pair IDs overlap")


def resolved_policy_path(path: Path) -> Path:
    if path.exists():
        return path.resolve()
    return path.parent.resolve() / path.name


def require_wip_path_outside_repo(repo_root: Path, path: Path) -> None:
    repo = repo_root.resolve()
    resolved = resolved_policy_path(path)
    require(resolved != repo and repo not in resolved.parents, "WIP_SCHEMA_MISMATCH", f"WIP path must be outside repository root: {resolved}")


def load_wip(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    if not path.exists():
        return [], set()
    records: list[dict[str, Any]] = []
    duplicate_pair_ids: set[str] = set()
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            require(line.strip() != "", "WIP_SCHEMA_MISMATCH", f"WIP line {line_number} is blank")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReviewInfrastructureError(f"WIP_SCHEMA_MISMATCH: malformed WIP JSONL line {line_number}: {exc}") from exc
            require(isinstance(value, dict), "MISSING_PAIR_ID", f"WIP line {line_number} is not an object")
            require(frozenset(value.keys()) == WIP_FIELD_SET, "WIP_SCHEMA_MISMATCH", f"WIP line {line_number} schema mismatch")
            pair_id = value.get("pair_id", "")
            if pair_id in seen:
                duplicate_pair_ids.add(pair_id)
            seen.add(pair_id)
            records.append(value)
    return records, duplicate_pair_ids


def strict_load_wip(
    authority: Authority,
    path: Path,
    cohort_confirmations: dict[str, CohortConfirmationAuthority] | None = None,
) -> tuple[list[dict[str, Any]], set[str]]:
    require_wip_path_outside_repo(authority.repo_root, path)
    records, duplicate_pair_ids = load_wip(path)
    errors = validate_wip_records(authority, records, duplicate_pair_ids, cohort_confirmations)
    require_no_validation_errors(errors)
    return records, duplicate_pair_ids


def validate_wip_records(
    authority: Authority,
    records: list[dict[str, Any]],
    duplicate_pair_ids: set[str] | None = None,
    cohort_confirmations: dict[str, CohortConfirmationAuthority] | None = None,
) -> list[str]:
    duplicate_pair_ids = duplicate_pair_ids or set()
    errors: list[str] = []
    authorized = authority.row_by_pair_id
    for record in records:
        if frozenset(record.keys()) != WIP_FIELD_SET:
            errors.append("WIP_SCHEMA_MISMATCH")
        errors.extend(validate_review_record(record, authority, duplicate_pair_ids, cohort_confirmations))
        pair_id = record.get("pair_id", "")
        if pair_id not in authorized:
            errors.append("UNAUTHORIZED_PAIR_ID")
    return order_errors(errors)


def write_wip_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
                handle.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def canonical_pair_id_list(values: Any) -> list[str]:
    require(isinstance(values, list), "INVALID_COHORT_CONFIRMATION_ARTIFACT", "pair-id field must be a list")
    require(all(isinstance(pair_id, str) for pair_id in values), "INVALID_COHORT_CONFIRMATION_ARTIFACT", "pair-id field must contain strings")
    return sorted(values)


def canonical_confirmation_payload(artifact: dict[str, Any]) -> dict[str, Any]:
    payload = {field: artifact[field] for field in COHORT_CONFIRMATION_PAYLOAD_FIELDS}
    for field in ("eligible_pair_ids", "exception_pair_ids", "individually_reviewed_pair_ids", "cohort_confirmed_pair_ids"):
        payload[field] = canonical_pair_id_list(payload[field])
    return payload


def confirmation_payload_sha256(artifact: dict[str, Any]) -> str:
    return canonical_json_sha256(canonical_confirmation_payload(artifact))


def derive_cohort_confirmation_id_from_payload_sha256(payload_sha256: str) -> str:
    require(isinstance(payload_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", payload_sha256), "INVALID_COHORT_CONFIRMATION_ARTIFACT")
    return f"P3W6F2_COHORT_CONFIRMATION_{payload_sha256}"


def path_sha256(path: Path) -> str:
    require(path.exists(), "INVALID_AI_PRESCREEN_ARTIFACT", f"missing artifact: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_external_json_object(path: Path, code: str) -> dict[str, Any]:
    require(path.exists(), code, f"missing external artifact: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReviewInfrastructureError(f"{code}: malformed JSON: {exc}") from exc
    require(isinstance(value, dict), code, "artifact must be a JSON object")
    return value


def load_ai_prescreen(path: Path, authority: Authority) -> dict[str, dict[str, Any]]:
    require(path.exists(), "INVALID_AI_PRESCREEN_ARTIFACT", f"missing AI prescreen artifact: {path}")
    text = path.read_text(encoding="utf-8")
    stripped = text.strip()
    require(stripped != "", "INVALID_AI_PRESCREEN_ARTIFACT", "empty AI prescreen artifact")
    records: list[dict[str, Any]]
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        records = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReviewInfrastructureError(f"INVALID_AI_PRESCREEN_ARTIFACT: malformed JSONL line {line_number}: {exc}") from exc
            require(isinstance(value, dict), "INVALID_AI_PRESCREEN_ARTIFACT", f"line {line_number} is not an object")
            records.append(value)
    else:
        if isinstance(parsed, list):
            records = parsed
        elif isinstance(parsed, dict) and isinstance(parsed.get("records"), list):
            records = parsed["records"]
        elif isinstance(parsed, dict) and frozenset(parsed.keys()) == AI_PRESCREEN_FIELD_SET:
            records = [parsed]
        else:
            raise ReviewInfrastructureError("INVALID_AI_PRESCREEN_ARTIFACT: expected JSONL, array, or object.records")

    out: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for record in records:
        require(isinstance(record, dict), "INVALID_AI_PRESCREEN_ARTIFACT", "AI prescreen record is not an object")
        require(frozenset(record.keys()) == AI_PRESCREEN_FIELD_SET, "INVALID_AI_PRESCREEN_ARTIFACT", "AI prescreen schema mismatch")
        pair_id = str(record.get("pair_id", ""))
        require(pair_id in authority.row_by_pair_id, "INVALID_AI_PRESCREEN_ARTIFACT", f"unauthorized pair {pair_id}")
        require(pair_id not in seen, "INVALID_AI_PRESCREEN_ARTIFACT", f"duplicate pair {pair_id}")
        seen.add(pair_id)
        require(record.get("source_record_sha256") == authority.source_sha256_by_pair_id[pair_id], "SOURCE_RECORD_HASH_MISMATCH", pair_id)
        require(record.get("ai_prescreen_protocol_version") == AI_PRESCREEN_PROTOCOL_VERSION, "INVALID_AI_PRESCREEN_ARTIFACT", pair_id)
        require(valid_rfc3339_utc_z(record.get("ai_prescreen_created_at_utc", "")), "INVALID_AI_PRESCREEN_ARTIFACT", pair_id)
        require(isinstance(record.get("ai_prescreen_model_or_system_id"), str) and record["ai_prescreen_model_or_system_id"].strip() != "", "INVALID_AI_PRESCREEN_ARTIFACT", pair_id)
        out[pair_id] = record
    require(set(out) == set(authority.ordered_pair_ids), "INVALID_AI_PRESCREEN_ARTIFACT", "AI prescreen must cover exactly the authorized F2 universe")
    return out


def load_xlsx_confirmed_import_intermediate(path: Path, authority: Authority) -> dict[str, dict[str, Any]]:
    require_wip_path_outside_repo(authority.repo_root, path)
    artifact = load_external_json_object(path, "WIP_SCHEMA_MISMATCH")
    require(frozenset(artifact.keys()) == XLSX_IMPORT_INTERMEDIATE_FIELD_SET, "WIP_SCHEMA_MISMATCH", "XLSX import intermediate schema mismatch")
    require(
        artifact.get("schema_version") == XLSX_CONFIRMED_IMPORT_INTERMEDIATE_PROTOCOL_VERSION,
        "WIP_SCHEMA_MISMATCH",
        "XLSX import intermediate schema_version mismatch",
    )
    require(
        artifact.get("import_intermediate_protocol_version") == XLSX_CONFIRMED_IMPORT_INTERMEDIATE_PROTOCOL_VERSION,
        "WIP_SCHEMA_MISMATCH",
        "XLSX import intermediate protocol mismatch",
    )
    workbook_sha = artifact.get("source_workbook_sha256")
    require(isinstance(workbook_sha, str) and re.fullmatch(r"[0-9a-f]{64}", workbook_sha), "WIP_SCHEMA_MISMATCH", "invalid source workbook SHA-256")
    workbook_name = artifact.get("source_workbook_name")
    require(isinstance(workbook_name, str) and workbook_name.strip() != "", "WIP_SCHEMA_MISMATCH", "missing source workbook name")
    require(valid_rfc3339_utc_z(artifact.get("conversion_created_at_utc", "")), "INVALID_REVIEW_TIMESTAMP", "invalid conversion timestamp")
    require(artifact.get("row_count") == len(INDIVIDUALLY_REVIEWED_AUDIT_PAIR_IDS), "WIP_SCHEMA_MISMATCH", "XLSX import row count mismatch")
    records = artifact.get("records")
    require(isinstance(records, list), "WIP_SCHEMA_MISMATCH", "XLSX import records must be a list")
    require(len(records) == len(INDIVIDUALLY_REVIEWED_AUDIT_PAIR_IDS), "WIP_SCHEMA_MISMATCH", "XLSX import records length mismatch")

    out: dict[str, dict[str, Any]] = {}
    expected_pairs = set(INDIVIDUALLY_REVIEWED_AUDIT_PAIR_IDS)
    for index, record in enumerate(records, 1):
        require(isinstance(record, dict), "WIP_SCHEMA_MISMATCH", f"XLSX import record {index} is not an object")
        require(frozenset(record.keys()) == XLSX_IMPORT_RECORD_FIELD_SET, "WIP_SCHEMA_MISMATCH", f"XLSX import record {index} schema mismatch")
        pair_id = str(record.get("pair_id", ""))
        require(pair_id in expected_pairs, "UNAUTHORIZED_PAIR_ID", pair_id)
        require(pair_id in authority.row_by_pair_id, "UNAUTHORIZED_PAIR_ID", pair_id)
        require(pair_id not in out, "DUPLICATE_PAIR_ID", pair_id)
        require(record.get("source_record_sha256") == authority.source_sha256_by_pair_id[pair_id], "SOURCE_RECORD_HASH_MISMATCH", pair_id)
        expected_ai = {
            "ai_canonical_semantics_suggestion": "V",
            "ai_paraphrase_semantics_suggestion": "V",
            "ai_polarity_flip_semantics_suggestion": "V",
            "ai_grammar_validity_suggestion": "M",
            "ai_triage_status": "CLEAR_SUGGESTION",
        }
        require(all(record.get(key) == value for key, value in expected_ai.items()), "WIP_SCHEMA_MISMATCH", f"AI suggestion mismatch for {pair_id}")
        require(record.get("human_review_action") == "CONFIRM", "WIP_SCHEMA_MISMATCH", f"non-CONFIRM action for {pair_id}")
        override_fields = [
            "human_override_canonical_semantics",
            "human_override_paraphrase_semantics",
            "human_override_polarity_flip_semantics",
            "human_override_grammar_validity",
        ]
        require(all(record.get(field) == "" for field in override_fields), "WIP_SCHEMA_MISMATCH", f"override field must be empty for {pair_id}")
        expected_human = {
            "derived_human_canonical_semantics": "VALID",
            "derived_human_paraphrase_semantics": "VALID",
            "derived_human_polarity_flip_semantics": "VALID",
            "derived_human_grammar_validity": "MULTI_MEMBER_DEFECT",
        }
        require(all(record.get(key) == value for key, value in expected_human.items()), "WIP_SCHEMA_MISMATCH", f"derived human value mismatch for {pair_id}")
        expected_decision = derive_authority_decision("VALID", "VALID", "VALID", "MULTI_MEMBER_DEFECT")
        require(record.get("derived_human_authority_decision") == expected_decision, "COMPATIBILITY_MATRIX_MISMATCH", pair_id)
        require(record.get("review_status") == "READY_TO_IMPORT", "WIP_SCHEMA_MISMATCH", f"review status mismatch for {pair_id}")
        out[pair_id] = record
    require(set(out) == expected_pairs, "WIP_SCHEMA_MISMATCH", "XLSX import must contain exactly the required 20 pair IDs")
    return out


def validate_cohort_confirmation_artifact(
    artifact: dict[str, Any],
    authority: Authority,
    expected_audit: dict[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    if frozenset(artifact.keys()) != COHORT_CONFIRMATION_FIELD_SET:
        errors.append("INVALID_COHORT_CONFIRMATION_ARTIFACT")
    if artifact.get("schema_version") != COHORT_CONFIRMATION_SCHEMA_VERSION:
        errors.append("INVALID_COHORT_CONFIRMATION_ARTIFACT")
    if artifact.get("cohort_confirmation_protocol_version") != COHORT_CONFIRMATION_PROTOCOL_VERSION:
        errors.append("INVALID_COHORT_CONFIRMATION_ARTIFACT")
    if not isinstance(artifact.get("cohort_confirmation_id"), str) or artifact.get("cohort_confirmation_id") == "":
        errors.append("INVALID_COHORT_CONFIRMATION_ARTIFACT")
    if not valid_rfc3339_utc_z(artifact.get("authority_recorded_at_utc", "")):
        errors.append("INVALID_REVIEW_TIMESTAMP")
    if not isinstance(artifact.get("reviewer_id"), str) or artifact.get("reviewer_id", "").strip() == "":
        errors.append("MISSING_REVIEWER_ID")
    if artifact.get("human_action") != COHORT_CONFIRM_ACTION:
        errors.append("INVALID_COHORT_CONFIRMATION_ARTIFACT")
    if artifact.get("structural_gate_version") != STRUCTURAL_GATE_VERSION:
        errors.append("STRUCTURAL_COHORT_GATE_FAILED")
    if artifact.get("structural_gate_result") not in {"PASS", "PASS_WITH_EXCEPTIONS"}:
        errors.append("STRUCTURAL_COHORT_GATE_FAILED")
    if artifact.get("ai_prescreen_protocol_version") != AI_PRESCREEN_PROTOCOL_VERSION:
        errors.append("INVALID_AI_PRESCREEN_ARTIFACT")
    if artifact.get("individually_reviewed_pair_ids") != INDIVIDUALLY_REVIEWED_AUDIT_PAIR_IDS:
        errors.append("MISSING_INDIVIDUAL_AUDIT_EVIDENCE")
    for sha_field in ("structural_audit_sha256", "ai_prescreen_artifact_sha256", "validated_individual_wip_state_sha256", "confirmation_payload_sha256"):
        value = artifact.get(sha_field)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            errors.append("INVALID_COHORT_CONFIRMATION_ARTIFACT")
    if not isinstance(artifact.get("structural_audit_path_identity"), str) or artifact.get("structural_audit_path_identity") == "":
        errors.append("INVALID_COHORT_CONFIRMATION_ARTIFACT")
    confirmed = artifact.get("cohort_confirmed_pair_ids")
    eligible = artifact.get("eligible_pair_ids")
    exceptions = artifact.get("exception_pair_ids")
    if not isinstance(confirmed, list) or any(not isinstance(pair_id, str) for pair_id in confirmed):
        errors.append("INVALID_COHORT_CONFIRMATION_ARTIFACT")
    elif not set(confirmed).issubset(set(authority.ordered_pair_ids)):
        errors.append("UNAUTHORIZED_PAIR_ID")
    if not isinstance(eligible, list) or any(not isinstance(pair_id, str) for pair_id in eligible):
        errors.append("INVALID_COHORT_CONFIRMATION_ARTIFACT")
    if not isinstance(exceptions, list) or any(not isinstance(pair_id, str) for pair_id in exceptions):
        errors.append("INVALID_COHORT_CONFIRMATION_ARTIFACT")
    if isinstance(confirmed, list) and isinstance(eligible, list) and isinstance(exceptions, list):
        if confirmed != eligible:
            errors.append("INVALID_COHORT_CONFIRMATION_ARTIFACT")
        if len(set(confirmed)) != len(confirmed) or len(set(eligible)) != len(eligible) or len(set(exceptions)) != len(exceptions):
            errors.append("INVALID_COHORT_CONFIRMATION_ARTIFACT")
        if set(confirmed).intersection(INDIVIDUALLY_REVIEWED_AUDIT_PAIR_IDS):
            errors.append("INVALID_COHORT_CONFIRMATION_ARTIFACT")
        if set(confirmed).intersection(exceptions):
            errors.append("INVALID_COHORT_CONFIRMATION_ARTIFACT")
        if not set(eligible).issubset(set(authority.ordered_pair_ids)) or not set(exceptions).issubset(set(authority.ordered_pair_ids)):
            errors.append("UNAUTHORIZED_PAIR_ID")
    if artifact.get("eligible_pair_count") != len(confirmed if isinstance(confirmed, list) else []):
        errors.append("INVALID_COHORT_CONFIRMATION_ARTIFACT")
    if artifact.get("exception_pair_count") != len(exceptions if isinstance(exceptions, list) else []):
        errors.append("INVALID_COHORT_CONFIRMATION_ARTIFACT")
    if expected_audit is not None:
        if artifact.get("structural_audit_sha256") != expected_audit.get("audit_payload_sha256"):
            errors.append("INVALID_COHORT_CONFIRMATION_ARTIFACT")
        if artifact.get("ai_prescreen_artifact_sha256") != expected_audit.get("ai_prescreen_artifact_sha256"):
            errors.append("INVALID_AI_PRESCREEN_ARTIFACT")
        if artifact.get("validated_individual_wip_state_sha256") != expected_audit.get("validated_individual_wip_state_sha256"):
            errors.append("MISSING_INDIVIDUAL_AUDIT_EVIDENCE")
        if artifact.get("eligible_pair_ids") != expected_audit.get("eligible_pair_ids"):
            errors.append("INVALID_COHORT_CONFIRMATION_ARTIFACT")
        if artifact.get("exception_pair_ids") != expected_audit.get("exception_pair_ids"):
            errors.append("INVALID_COHORT_CONFIRMATION_ARTIFACT")
        if artifact.get("structural_gate_result") != expected_audit.get("overall_structural_gate_result"):
            errors.append("STRUCTURAL_COHORT_GATE_FAILED")
    elif not set(confirmed if isinstance(confirmed, list) else []).issubset(set(authority.ordered_pair_ids)):
        errors.append("UNAUTHORIZED_PAIR_ID")
    source_identity = artifact.get("source_authority_identity")
    if not isinstance(source_identity, dict):
        errors.append("INVALID_COHORT_CONFIRMATION_ARTIFACT")
    else:
        expected_identity = source_authority_identity(authority)
        for key, expected in expected_identity.items():
            if source_identity.get(key) != expected:
                errors.append("INVALID_COHORT_CONFIRMATION_ARTIFACT")
                break
    summary = artifact.get("ai_prescreen_result_summary")
    if not isinstance(summary, dict) or summary.get("ai_prescreen_protocol_version") != AI_PRESCREEN_PROTOCOL_VERSION:
        errors.append("INVALID_AI_PRESCREEN_ARTIFACT")
    if frozenset(artifact.keys()) == COHORT_CONFIRMATION_FIELD_SET:
        try:
            recomputed_payload_sha256 = confirmation_payload_sha256(artifact)
        except (KeyError, ReviewInfrastructureError):
            errors.append("INVALID_COHORT_CONFIRMATION_ARTIFACT")
        else:
            if artifact.get("confirmation_payload_sha256") != recomputed_payload_sha256:
                errors.append("CONFIRMATION_PAYLOAD_HASH_MISMATCH")
            expected_id = derive_cohort_confirmation_id_from_payload_sha256(recomputed_payload_sha256)
            if artifact.get("cohort_confirmation_id") != expected_id:
                errors.append("COHORT_CONFIRMATION_ID_MISMATCH")
    return order_errors(errors)


def cohort_authority_from_artifact(artifact: dict[str, Any]) -> CohortConfirmationAuthority:
    return CohortConfirmationAuthority(
        cohort_confirmation_id=str(artifact["cohort_confirmation_id"]),
        cohort_confirmed_pair_ids=frozenset(str(pair_id) for pair_id in artifact["cohort_confirmed_pair_ids"]),
        individually_reviewed_pair_ids=frozenset(str(pair_id) for pair_id in artifact["individually_reviewed_pair_ids"]),
        structural_audit_sha256=str(artifact["structural_audit_sha256"]),
        structural_gate_version=str(artifact["structural_gate_version"]),
        source_authority_identity=dict(artifact["source_authority_identity"]),
        ai_prescreen_artifact_sha256=str(artifact["ai_prescreen_artifact_sha256"]),
        validated_individual_wip_state_sha256=str(artifact["validated_individual_wip_state_sha256"]),
    )


def load_cohort_confirmations(path: Path | None, authority: Authority) -> dict[str, CohortConfirmationAuthority] | None:
    if path is None:
        return None
    require_wip_path_outside_repo(authority.repo_root, path)
    artifact = load_external_json_object(path, "INVALID_COHORT_CONFIRMATION_ARTIFACT")
    require_no_validation_errors(validate_cohort_confirmation_artifact(artifact, authority))
    authority_artifact = cohort_authority_from_artifact(artifact)
    return {authority_artifact.cohort_confirmation_id: authority_artifact}


def source_authority_identity(authority: Authority) -> dict[str, Any]:
    return {
        "authorized_F2_pair_count": EXPECTED_PAIR_COUNT,
        "authorized_F2_member_count": EXPECTED_MEMBER_COUNT,
        "P1_authority_commit": P1_AUTHORITY_COMMIT,
        "P3W4_result_authority_commit": P3W4_RESULT_AUTHORITY_COMMIT,
        "P3W4_artifact_commit": authority.p3w4_artifact_commit,
        "input_authority_paths": authority.input_artifact_paths,
        "input_authority_sha256": authority.input_artifact_sha256,
    }


INDIVIDUAL_AUDIT_STATE_FIELDS = [
    "pair_id",
    "source_record_sha256",
    "review_method",
    "human_canonical_semantics",
    "human_paraphrase_semantics",
    "human_polarity_flip_semantics",
    "human_grammar_validity",
    "human_authority_decision",
    "reviewer_id",
    "review_protocol_version",
    "reviewed_at_utc",
    "authority_recorded_at_utc",
    "human_review_time_provenance",
    "record_origin",
]


def validated_individual_wip_state_sha256(authority: Authority, records: list[dict[str, Any]]) -> str:
    require_no_validation_errors(validate_individual_audit_evidence(authority, records))
    by_pair = {record["pair_id"]: record for record in records if record.get("pair_id") in INDIVIDUALLY_REVIEWED_AUDIT_PAIR_IDS}
    payload = [
        {field: by_pair[pair_id][field] for field in INDIVIDUAL_AUDIT_STATE_FIELDS}
        for pair_id in INDIVIDUALLY_REVIEWED_AUDIT_PAIR_IDS
    ]
    return canonical_json_sha256(payload)


def structural_audit_payload(
    authority: Authority,
    records: list[dict[str, Any]],
    ai_by_pair_id: dict[str, dict[str, Any]],
    ai_prescreen_artifact_sha256: str,
) -> dict[str, Any]:
    require_no_validation_errors(validate_individual_audit_evidence(authority, records))
    eligible, gate_failures = gate_structural_cohort(authority, ai_by_pair_id)
    existing_by_pair = {record["pair_id"]: record for record in records}
    required_individual = set(INDIVIDUALLY_REVIEWED_AUDIT_PAIR_IDS)
    eligible_pair_ids = [
        pair_id
        for pair_id in eligible
        if pair_id not in existing_by_pair and pair_id not in required_individual
    ]
    exception_pair_ids = [
        pair_id
        for pair_id in authority.ordered_pair_ids
        if pair_id not in set(eligible_pair_ids) and pair_id not in existing_by_pair and pair_id not in required_individual
    ]
    gate_by_pair = {
        pair_id: {
            "eligible": pair_id in set(eligible),
            "errors": gate_failures.get(pair_id, []),
        }
        for pair_id in authority.ordered_pair_ids
    }
    return {
        "schema_version": STRUCTURAL_AUDIT_SCHEMA_VERSION,
        "structural_gate_version": STRUCTURAL_GATE_VERSION,
        "source_authority_identity": source_authority_identity(authority),
        "authorized_pair_count": EXPECTED_PAIR_COUNT,
        "authorized_member_count": EXPECTED_MEMBER_COUNT,
        "source_record_sha256_by_pair": {pair_id: authority.source_sha256_by_pair_id[pair_id] for pair_id in authority.ordered_pair_ids},
        "ai_prescreen_artifact_sha256": ai_prescreen_artifact_sha256,
        "ai_prescreen_protocol_version": AI_PRESCREEN_PROTOCOL_VERSION,
        "validated_individual_wip_state_sha256": validated_individual_wip_state_sha256(authority, records),
        "required_individual_audit_pair_ids": INDIVIDUALLY_REVIEWED_AUDIT_PAIR_IDS,
        "eligible_pair_ids": eligible_pair_ids,
        "exception_pair_ids": exception_pair_ids,
        "structural_gate_result_by_pair": gate_by_pair,
        "overall_structural_gate_result": "PASS" if not exception_pair_ids else "PASS_WITH_EXCEPTIONS",
    }


def structural_audit_artifact(
    authority: Authority,
    records: list[dict[str, Any]],
    ai_by_pair_id: dict[str, dict[str, Any]],
    ai_prescreen_artifact_sha256: str,
    audit_created_at_utc: str,
) -> dict[str, Any]:
    payload = structural_audit_payload(authority, records, ai_by_pair_id, ai_prescreen_artifact_sha256)
    artifact = {
        **payload,
        "audit_created_at_utc": audit_created_at_utc,
        "audit_payload_sha256": canonical_json_sha256(payload),
    }
    return {field: artifact[field] for field in STRUCTURAL_AUDIT_FIELDS}


def validate_structural_audit_artifact(artifact: dict[str, Any], authority: Authority) -> list[str]:
    errors: list[str] = []
    if frozenset(artifact.keys()) != STRUCTURAL_AUDIT_FIELD_SET:
        errors.append("INVALID_COHORT_CONFIRMATION_ARTIFACT")
    if artifact.get("schema_version") != STRUCTURAL_AUDIT_SCHEMA_VERSION:
        errors.append("INVALID_COHORT_CONFIRMATION_ARTIFACT")
    if artifact.get("structural_gate_version") != STRUCTURAL_GATE_VERSION:
        errors.append("STRUCTURAL_COHORT_GATE_FAILED")
    if not valid_rfc3339_utc_z(artifact.get("audit_created_at_utc", "")):
        errors.append("INVALID_REVIEW_TIMESTAMP")
    if artifact.get("source_authority_identity") != source_authority_identity(authority):
        errors.append("INVALID_COHORT_CONFIRMATION_ARTIFACT")
    if artifact.get("authorized_pair_count") != EXPECTED_PAIR_COUNT or artifact.get("authorized_member_count") != EXPECTED_MEMBER_COUNT:
        errors.append("AUTHORITY_PAIR_UNIVERSE_MISMATCH")
    if artifact.get("ai_prescreen_protocol_version") != AI_PRESCREEN_PROTOCOL_VERSION:
        errors.append("INVALID_AI_PRESCREEN_ARTIFACT")
    if artifact.get("required_individual_audit_pair_ids") != INDIVIDUALLY_REVIEWED_AUDIT_PAIR_IDS:
        errors.append("MISSING_INDIVIDUAL_AUDIT_EVIDENCE")
    source_hashes = artifact.get("source_record_sha256_by_pair")
    if source_hashes != {pair_id: authority.source_sha256_by_pair_id[pair_id] for pair_id in authority.ordered_pair_ids}:
        errors.append("SOURCE_RECORD_HASH_MISMATCH")
    for key in ("eligible_pair_ids", "exception_pair_ids"):
        values = artifact.get(key)
        if not isinstance(values, list) or any(not isinstance(pair_id, str) for pair_id in values):
            errors.append("INVALID_COHORT_CONFIRMATION_ARTIFACT")
    eligible = artifact.get("eligible_pair_ids", [])
    exceptions = artifact.get("exception_pair_ids", [])
    if isinstance(eligible, list) and isinstance(exceptions, list):
        if len(set(eligible)) != len(eligible) or len(set(exceptions)) != len(exceptions):
            errors.append("INVALID_COHORT_CONFIRMATION_ARTIFACT")
        if set(eligible).intersection(exceptions):
            errors.append("INVALID_COHORT_CONFIRMATION_ARTIFACT")
        if set(eligible).intersection(INDIVIDUALLY_REVIEWED_AUDIT_PAIR_IDS) or set(exceptions).intersection(INDIVIDUALLY_REVIEWED_AUDIT_PAIR_IDS):
            errors.append("INVALID_COHORT_CONFIRMATION_ARTIFACT")
        if not set(eligible).issubset(set(authority.ordered_pair_ids)) or not set(exceptions).issubset(set(authority.ordered_pair_ids)):
            errors.append("UNAUTHORIZED_PAIR_ID")
    payload = {field: artifact[field] for field in STRUCTURAL_AUDIT_FIELDS if field not in {"audit_created_at_utc", "audit_payload_sha256"} and field in artifact}
    if artifact.get("audit_payload_sha256") != canonical_json_sha256(payload):
        errors.append("INVALID_COHORT_CONFIRMATION_ARTIFACT")
    return order_errors(errors)


def load_structural_audit_artifact(path: Path, authority: Authority, expected_sha256: str) -> dict[str, Any]:
    require_wip_path_outside_repo(authority.repo_root, path)
    require(expected_sha256 != "", "INVALID_COHORT_CONFIRMATION_ARTIFACT", "missing --expected-audit-sha256")
    artifact = load_external_json_object(path, "INVALID_COHORT_CONFIRMATION_ARTIFACT")
    require_no_validation_errors(validate_structural_audit_artifact(artifact, authority))
    require(artifact["audit_payload_sha256"] == expected_sha256, "INVALID_COHORT_CONFIRMATION_ARTIFACT", "audit SHA mismatch")
    return artifact


def did_not_predicate(text: str) -> str:
    match = re.search(r"\bdid not ([A-Za-z]+)\b", text)
    return match.group(1) if match else ""


def predicate_coverage_for_pair_ids(authority: Authority, pair_ids: list[str]) -> dict[str, int]:
    coverage: dict[str, int] = {}
    rows = authority.row_by_pair_id
    for pair_id in pair_ids:
        predicate = did_not_predicate(rows[pair_id]["canonical_evidence"])
        coverage[predicate] = coverage.get(predicate, 0) + 1
    return coverage


def structural_gate_errors(row: dict[str, str], authority: Authority, ai_record: dict[str, Any] | None) -> list[str]:
    errors: list[str] = []
    if len(authority.source_rows) != EXPECTED_PAIR_COUNT or len(authority.ordered_pair_ids) != EXPECTED_PAIR_COUNT:
        errors.append("AUTHORITY_PAIR_UNIVERSE_MISMATCH")
    family_counts = authority.summary.get("aggregates", {}).get("family_counts", {})
    if family_counts and family_counts.get("F2_complete_triple_members") != EXPECTED_MEMBER_COUNT:
        errors.append("AUTHORITY_PAIR_UNIVERSE_MISMATCH")
    pair_record = next((record for record in authority.pair_records if record.get("family") == "F2" and record.get("pair_id") == row["pair_id"]), None)
    if pair_record is not None:
        members = pair_record.get("members", {})
        if sorted(members.keys()) != ["canonical", "paraphrase", "polarity_flip"]:
            errors.append("STRUCTURAL_COHORT_GATE_FAILED")
    if row["canonical_final_label"] != "REFUTE" or row["paraphrase_final_label"] != "REFUTE" or row["polarity_flip_final_label"] != "SUPPORT":
        errors.append("STRUCTURAL_COHORT_GATE_FAILED")
    if not (row["canonical_claim"] == row["paraphrase_claim"] == row["polarity_flip_claim"]):
        errors.append("STRUCTURAL_COHORT_GATE_FAILED")
    predicate = did_not_predicate(row["canonical_evidence"])
    paraphrase_predicate = did_not_predicate(row["paraphrase_evidence"])
    if predicate == "" or paraphrase_predicate == "" or predicate != paraphrase_predicate:
        errors.append("STRUCTURAL_COHORT_GATE_FAILED")
    claim = row["canonical_claim"]
    if predicate and predicate not in claim:
        errors.append("STRUCTURAL_COHORT_GATE_FAILED")
    if row["polarity_flip_evidence"] != claim:
        errors.append("STRUCTURAL_COHORT_GATE_FAILED")
    if did_not_predicate(row["polarity_flip_evidence"]):
        errors.append("STRUCTURAL_COHORT_GATE_FAILED")
    if (row["canonical_grammar_status"], row["paraphrase_grammar_status"], row["polarity_flip_grammar_status"]) != ("FAIL", "FAIL", "PASS"):
        errors.append("STRUCTURAL_COHORT_GATE_FAILED")
    if row["canonical_reason_codes"] != '["DID_NOT_INFLECTED_PREDICATE","GRAMMAR_TEMPLATE_FAIL"]':
        errors.append("STRUCTURAL_COHORT_GATE_FAILED")
    if row["paraphrase_reason_codes"] != '["CANONICAL_ROW_KNOWN_GENERATOR_DEFECT","DID_NOT_INFLECTED_PREDICATE","GRAMMAR_TEMPLATE_FAIL"]':
        errors.append("STRUCTURAL_COHORT_GATE_FAILED")
    if row["polarity_flip_reason_codes"] != '["CANONICAL_ROW_KNOWN_GENERATOR_DEFECT"]':
        errors.append("STRUCTURAL_COHORT_GATE_FAILED")
    if ai_record is None:
        errors.append("INVALID_AI_PRESCREEN_ARTIFACT")
    else:
        if ai_record.get("source_record_sha256") != authority.source_sha256_by_pair_id[row["pair_id"]]:
            errors.append("SOURCE_RECORD_HASH_MISMATCH")
        expected_ai = {
            "ai_canonical_semantics_suggestion": "V",
            "ai_paraphrase_semantics_suggestion": "V",
            "ai_polarity_flip_semantics_suggestion": "V",
            "ai_grammar_validity_suggestion": "M",
            "ai_triage_status": "CLEAR_SUGGESTION",
        }
        if any(ai_record.get(key) != value for key, value in expected_ai.items()):
            errors.append("STRUCTURAL_COHORT_GATE_FAILED")
    return order_errors(errors)


def gate_structural_cohort(authority: Authority, ai_by_pair_id: dict[str, dict[str, Any]]) -> tuple[list[str], dict[str, list[str]]]:
    eligible: list[str] = []
    failures: dict[str, list[str]] = {}
    for row in authority.source_rows:
        errors = structural_gate_errors(row, authority, ai_by_pair_id.get(row["pair_id"]))
        if errors:
            failures[row["pair_id"]] = errors
        else:
            eligible.append(row["pair_id"])
    return eligible, failures


def validate_individual_audit_evidence(authority: Authority, records: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    by_pair = {record.get("pair_id"): record for record in records}
    for pair_id in INDIVIDUALLY_REVIEWED_AUDIT_PAIR_IDS:
        record = by_pair.get(pair_id)
        if record is None:
            errors.append("MISSING_INDIVIDUAL_AUDIT_EVIDENCE")
            continue
        record_errors = validate_review_record(record, authority)
        if record_errors:
            errors.append("MISSING_INDIVIDUAL_AUDIT_EVIDENCE")
        expected_values = {
            "human_canonical_semantics": "VALID",
            "human_paraphrase_semantics": "VALID",
            "human_polarity_flip_semantics": "VALID",
            "human_grammar_validity": "MULTI_MEMBER_DEFECT",
            "human_authority_decision": "CANONICAL_REGENERATION_REQUIRED",
            "review_method": INDIVIDUAL_REVIEW_METHOD,
            "cohort_confirmation_id": NO_COHORT_LINKAGE,
        }
        if any(record.get(key) != value for key, value in expected_values.items()):
            errors.append("MISSING_INDIVIDUAL_AUDIT_EVIDENCE")
    if predicate_coverage_for_pair_ids(authority, INDIVIDUALLY_REVIEWED_AUDIT_PAIR_IDS) != EXPECTED_AUDIT_PREDICATE_COVERAGE:
        errors.append("AUDIT_PREDICATE_COVERAGE_MISMATCH")
    return order_errors(errors)


def make_structural_cohort_record(
    authority: Authority,
    pair_id: str,
    reviewer_id: str,
    cohort_confirmation_id: str,
    authority_recorded_at_utc: str,
) -> dict[str, Any]:
    require(pair_id in authority.row_by_pair_id, "UNAUTHORIZED_PAIR_ID", pair_id)
    require(cohort_confirmation_id != "", "INVALID_COHORT_CONFIRMATION_LINKAGE")
    decision = derive_authority_decision("VALID", "VALID", "VALID", "MULTI_MEMBER_DEFECT")
    record = {
        "pair_id": pair_id,
        "source_record_sha256": authority.source_sha256_by_pair_id[pair_id],
        "human_canonical_semantics": "VALID",
        "human_paraphrase_semantics": "VALID",
        "human_polarity_flip_semantics": "VALID",
        "human_grammar_validity": "MULTI_MEMBER_DEFECT",
        "human_authority_decision": decision,
        "human_notes": "",
        "reviewer_id": reviewer_id,
        "review_protocol_version": REVIEW_PROTOCOL_VERSION,
        "reviewed_at_utc": authority_recorded_at_utc,
        "authority_recorded_at_utc": authority_recorded_at_utc,
        "human_review_time_provenance": NOT_APPLICABLE_STRUCTURAL_COHORT,
        "record_origin": STRUCTURAL_COHORT_RECORD_ORIGIN,
        "review_method": STRUCTURAL_COHORT_METHOD,
        "cohort_confirmation_id": cohort_confirmation_id,
    }
    confirmation = CohortConfirmationAuthority(
        cohort_confirmation_id=cohort_confirmation_id,
        cohort_confirmed_pair_ids=frozenset({pair_id}),
        individually_reviewed_pair_ids=frozenset(INDIVIDUALLY_REVIEWED_AUDIT_PAIR_IDS),
        structural_audit_sha256="0" * 64,
        structural_gate_version=STRUCTURAL_GATE_VERSION,
        source_authority_identity=source_authority_identity(authority),
        ai_prescreen_artifact_sha256="0" * 64,
        validated_individual_wip_state_sha256="0" * 64,
    )
    require_no_validation_errors(validate_review_record(record, authority, cohort_confirmations={cohort_confirmation_id: confirmation}))
    return record


def cohort_confirmation_artifact(
    authority: Authority,
    audit_artifact: dict[str, Any],
    audit_path: Path,
    reviewer_id: str,
    authority_recorded_at_utc: str,
) -> dict[str, Any]:
    eligible_pair_ids = list(audit_artifact["eligible_pair_ids"])
    exception_pair_ids = list(audit_artifact["exception_pair_ids"])
    payload = {
        "schema_version": COHORT_CONFIRMATION_SCHEMA_VERSION,
        "cohort_confirmation_protocol_version": COHORT_CONFIRMATION_PROTOCOL_VERSION,
        "authority_recorded_at_utc": authority_recorded_at_utc,
        "reviewer_id": reviewer_id,
        "structural_audit_sha256": audit_artifact["audit_payload_sha256"],
        "structural_audit_path_identity": str(audit_path.resolve()),
        "ai_prescreen_artifact_sha256": audit_artifact["ai_prescreen_artifact_sha256"],
        "validated_individual_wip_state_sha256": audit_artifact["validated_individual_wip_state_sha256"],
        "eligible_pair_count": len(eligible_pair_ids),
        "exception_pair_count": len(exception_pair_ids),
        "eligible_pair_ids": eligible_pair_ids,
        "exception_pair_ids": exception_pair_ids,
        "individually_reviewed_pair_ids": INDIVIDUALLY_REVIEWED_AUDIT_PAIR_IDS,
        "cohort_confirmed_pair_ids": eligible_pair_ids,
        "structural_gate_version": STRUCTURAL_GATE_VERSION,
        "structural_gate_result": audit_artifact["overall_structural_gate_result"],
        "ai_prescreen_protocol_version": AI_PRESCREEN_PROTOCOL_VERSION,
        "ai_prescreen_result_summary": {
            "ai_prescreen_protocol_version": AI_PRESCREEN_PROTOCOL_VERSION,
            "required_values": "V/V/V/M",
            "required_triage_status": "CLEAR_SUGGESTION",
            "eligible_pair_count": len(eligible_pair_ids),
            "exception_pair_count": len(exception_pair_ids),
            "ai_prescreen_artifact_sha256": audit_artifact["ai_prescreen_artifact_sha256"],
        },
        "source_authority_identity": source_authority_identity(authority),
        "human_action": COHORT_CONFIRM_ACTION,
    }
    payload_sha256 = canonical_json_sha256(canonical_confirmation_payload(payload))
    return {
        "cohort_confirmation_id": derive_cohort_confirmation_id_from_payload_sha256(payload_sha256),
        "confirmation_payload_sha256": payload_sha256,
        **payload,
    }


def make_review_record(
    authority: Authority,
    pair_id: str,
    reviewer_id: str,
    canonical_semantics: str,
    paraphrase_semantics: str,
    polarity_flip_semantics: str,
    grammar_validity: str,
    notes: str,
    clock: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    require(pair_id in authority.row_by_pair_id, "UNAUTHORIZED_PAIR_ID", pair_id)
    require(reviewer_id != "", "MISSING_REVIEWER_ID")
    require(reviewer_id == reviewer_id.strip(), "MISSING_REVIEWER_ID", "reviewer_id has leading/trailing whitespace")
    decision = derive_authority_decision(canonical_semantics, paraphrase_semantics, polarity_flip_semantics, grammar_validity)
    authority_recorded_at = utc_timestamp(clock)
    record = {
        "pair_id": pair_id,
        "source_record_sha256": authority.source_sha256_by_pair_id[pair_id],
        "human_canonical_semantics": canonical_semantics,
        "human_paraphrase_semantics": paraphrase_semantics,
        "human_polarity_flip_semantics": polarity_flip_semantics,
        "human_grammar_validity": grammar_validity,
        "human_authority_decision": decision,
        "human_notes": notes,
        "reviewer_id": reviewer_id,
        "review_protocol_version": REVIEW_PROTOCOL_VERSION,
        "reviewed_at_utc": authority_recorded_at,
        "authority_recorded_at_utc": authority_recorded_at,
        "human_review_time_provenance": CAPTURED_IN_RECORD,
        "record_origin": CLI_INDIVIDUAL_RECORD,
        "review_method": INDIVIDUAL_REVIEW_METHOD,
        "cohort_confirmation_id": NO_COHORT_LINKAGE,
    }
    errors = validate_review_record(record, authority)
    require_no_validation_errors(errors)
    return record


def make_xlsx_imported_individual_record(
    authority: Authority,
    pair_id: str,
    reviewer_id: str,
    authority_recorded_at_utc: str,
) -> dict[str, Any]:
    require(pair_id in authority.row_by_pair_id, "UNAUTHORIZED_PAIR_ID", pair_id)
    require(reviewer_id != "", "MISSING_REVIEWER_ID")
    require(reviewer_id == reviewer_id.strip(), "MISSING_REVIEWER_ID", "reviewer_id has leading/trailing whitespace")
    require(valid_rfc3339_utc_z(authority_recorded_at_utc), "INVALID_REVIEW_TIMESTAMP")
    decision = derive_authority_decision("VALID", "VALID", "VALID", "MULTI_MEMBER_DEFECT")
    record = {
        "pair_id": pair_id,
        "source_record_sha256": authority.source_sha256_by_pair_id[pair_id],
        "human_canonical_semantics": "VALID",
        "human_paraphrase_semantics": "VALID",
        "human_polarity_flip_semantics": "VALID",
        "human_grammar_validity": "MULTI_MEMBER_DEFECT",
        "human_authority_decision": decision,
        "human_notes": "",
        "reviewer_id": reviewer_id,
        "review_protocol_version": REVIEW_PROTOCOL_VERSION,
        "reviewed_at_utc": authority_recorded_at_utc,
        "authority_recorded_at_utc": authority_recorded_at_utc,
        "human_review_time_provenance": NOT_CAPTURED_IN_XLSX,
        "record_origin": XLSX_CONFIRMED_IMPORT,
        "review_method": INDIVIDUAL_REVIEW_METHOD,
        "cohort_confirmation_id": NO_COHORT_LINKAGE,
    }
    require_no_validation_errors(validate_review_record(record, authority))
    return record


def upsert_wip_record(
    authority: Authority,
    path: Path,
    record: dict[str, Any],
    cohort_confirmations: dict[str, CohortConfirmationAuthority] | None = None,
) -> None:
    require_wip_path_outside_repo(authority.repo_root, path)
    records, duplicate_pair_ids = load_wip(path)
    errors = validate_wip_records(authority, records, duplicate_pair_ids, cohort_confirmations)
    require_no_validation_errors(errors)
    record_errors = validate_review_record(record, authority, cohort_confirmations=cohort_confirmations)
    require_no_validation_errors(record_errors)
    replaced = False
    next_records: list[dict[str, Any]] = []
    for existing in records:
        if existing.get("pair_id") == record["pair_id"]:
            next_records.append(record)
            replaced = True
        else:
            next_records.append(existing)
    if not replaced:
        next_records.append(record)
    write_wip_atomic(path, next_records)


def decision_record(
    record: dict[str, Any],
    authority: Authority,
    duplicate_pair_ids: set[str] | None = None,
    cohort_confirmations: dict[str, CohortConfirmationAuthority] | None = None,
) -> dict[str, Any]:
    expected_decision = ""
    errors = validate_review_record(record, authority, duplicate_pair_ids, cohort_confirmations)
    try:
        expected_decision = derive_authority_decision(
            str(record.get("human_canonical_semantics", "")),
            str(record.get("human_paraphrase_semantics", "")),
            str(record.get("human_polarity_flip_semantics", "")),
            str(record.get("human_grammar_validity", "")),
        )
    except ReviewInfrastructureError:
        expected_decision = ""
    out = {field: record.get(field, "") for field in WIP_FIELDS}
    out.update(
        {
            "compatibility_matrix_version": COMPATIBILITY_MATRIX_VERSION,
            "compatibility_matrix_expected_decision": expected_decision,
            "compatibility_matrix_match": bool(expected_decision and record.get("human_authority_decision") == expected_decision),
            "review_record_valid": not errors,
            "ordered_validation_errors": errors,
        }
    )
    return {field: out[field] for field in DECISIONS_JSONL_FIELDS}


def compute_summary(
    authority: Authority,
    records: list[dict[str, Any]],
    duplicate_pair_ids: set[str] | None = None,
    cohort_confirmations: dict[str, CohortConfirmationAuthority] | None = None,
) -> dict[str, Any]:
    duplicate_pair_ids = duplicate_pair_ids or set()
    authorized = authority.ordered_pair_ids
    authorized_set = set(authorized)
    reviewed_ids = [record.get("pair_id", "") for record in records if record.get("pair_id", "") in authorized_set]
    reviewed_unique = [pair_id for pair_id in authorized if pair_id in set(reviewed_ids)]
    unreviewed = [pair_id for pair_id in authorized if pair_id not in set(reviewed_ids)]
    decision_records = [decision_record(record, authority, duplicate_pair_ids, cohort_confirmations) for record in records]
    invalid_by_pair: dict[str, list[str]] = {}
    for record in decision_records:
        pair_id = str(record.get("pair_id", ""))
        if pair_id in authorized_set and record["ordered_validation_errors"]:
            invalid_by_pair[pair_id] = record["ordered_validation_errors"]
    invalid_ids = [pair_id for pair_id in authorized if pair_id in invalid_by_pair]
    completed = [pair_id for pair_id in reviewed_unique if pair_id not in set(invalid_ids)]

    summary: dict[str, Any] = {
        "schema_version": "reason_router_p3w6f2_manual_review_summary_v1",
        "F2_execution_status": "",
        "authorized_F2_pair_ids": authorized,
        "reviewed_pair_ids": reviewed_unique,
        "unreviewed_pair_ids": unreviewed,
        "textual_repair_candidate_pair_ids": [],
        "regeneration_required_pair_ids": [],
        "semantic_conflict_pair_ids": [],
        "insufficient_evidence_pair_ids": [],
        "no_reproducible_defect_keep_blocked_pair_ids": [],
        "invalid_review_pair_ids": invalid_ids,
        "invalid_combination_pair_ids": ids_with_error(decision_records, "COMPATIBILITY_MATRIX_MISMATCH", authorized),
        "missing_reviewer_provenance_pair_ids": ids_with_any_error(
            decision_records,
            {"MISSING_REVIEWER_ID", "INVALID_REVIEW_PROTOCOL_VERSION", "INVALID_REVIEW_TIMESTAMP"},
            authorized,
        ),
        "source_hash_mismatch_pair_ids": ids_with_error(decision_records, "SOURCE_RECORD_HASH_MISMATCH", authorized),
        "missing_human_field_pair_ids": ids_with_error(decision_records, "MISSING_HUMAN_FIELD", authorized),
        "invalid_enum_pair_ids": ids_with_any_error(
            decision_records,
            {"INVALID_SEMANTIC_ENUM", "INVALID_GRAMMAR_ENUM", "INVALID_AUTHORITY_DECISION_ENUM"},
            authorized,
        ),
        "missing_required_notes_pair_ids": ids_with_error(decision_records, "MISSING_REQUIRED_NOTES", authorized),
        "completed_decision_pair_ids": completed,
        "individual_review_pair_ids": [],
        "structural_cohort_confirmation_pair_ids": [],
        "duplicate_pair_ids": sorted(duplicate_pair_ids),
        "unauthorized_pair_ids": sorted({str(record.get("pair_id", "")) for record in records if record.get("pair_id", "") not in authorized_set}),
        "missing_pair_ids": unreviewed,
        "review_protocol_version": REVIEW_PROTOCOL_VERSION,
        "compatibility_matrix_version": COMPATIBILITY_MATRIX_VERSION,
        "source_hash_version": SOURCE_HASH_VERSION,
        "P1_authority_commit": P1_AUTHORITY_COMMIT,
        "P3W4_execution_commit": P3W4_EXECUTION_COMMIT,
        "P3W4_result_authority_commit": P3W4_RESULT_AUTHORITY_COMMIT,
        "P3W4_artifact_commit": authority.p3w4_artifact_commit,
        "P3W5_authority_commit": P3W5_AUTHORITY_COMMIT,
        "input_authority_paths": authority.input_artifact_paths,
        "input_authority_sha256": authority.input_artifact_sha256,
    }
    records_by_pair = {record.get("pair_id"): record for record in records if record.get("pair_id") in authorized_set}
    for pair_id in completed:
        record = records_by_pair[pair_id]
        decision = record["human_authority_decision"]
        summary[DECISION_TO_SUMMARY_ARRAY[decision]].append(pair_id)
        if record.get("review_method") == INDIVIDUAL_REVIEW_METHOD:
            summary["individual_review_pair_ids"].append(pair_id)
        if record.get("review_method") == STRUCTURAL_COHORT_METHOD:
            summary["structural_cohort_confirmation_pair_ids"].append(pair_id)
    for count_key, array_key in COUNT_ARRAY_PAIRS.items():
        summary[count_key] = len(summary[array_key])
    partition_errors = validate_summary_partitions(summary)
    summary["summary_validation_errors"] = partition_errors
    summary["completion_gate_passed"] = completion_gate(summary)
    if summary["completion_gate_passed"]:
        summary["F2_execution_status"] = EXECUTION_STATUS_COMPLETE
    return summary


def ids_with_error(decision_records: list[dict[str, Any]], code: str, authority_order: list[str]) -> list[str]:
    bad = {record["pair_id"] for record in decision_records if code in record["ordered_validation_errors"]}
    return [pair_id for pair_id in authority_order if pair_id in bad]


def ids_with_any_error(decision_records: list[dict[str, Any]], codes: set[str], authority_order: list[str]) -> list[str]:
    bad = {record["pair_id"] for record in decision_records if codes.intersection(record["ordered_validation_errors"])}
    return [pair_id for pair_id in authority_order if pair_id in bad]


def validate_summary_partitions(summary: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for count_key, array_key in COUNT_ARRAY_PAIRS.items():
        if summary.get(count_key) != len(summary.get(array_key, [])):
            errors.append("COUNT_ARRAY_ASYMMETRY")
    authorized = set(summary["authorized_F2_pair_ids"])
    reviewed = set(summary["reviewed_pair_ids"])
    unreviewed = set(summary["unreviewed_pair_ids"])
    invalid = set(summary["invalid_review_pair_ids"])
    completed = set(summary["completed_decision_pair_ids"])
    decision_arrays = [set(summary[array_key]) for array_key in DECISION_TO_SUMMARY_ARRAY.values()]
    if reviewed.intersection(unreviewed) or reviewed.union(unreviewed) != authorized:
        errors.append("DECISION_PARTITION_MISMATCH")
    if not invalid.issubset(reviewed):
        errors.append("DECISION_PARTITION_MISMATCH")
    if completed != reviewed - invalid:
        errors.append("DECISION_PARTITION_MISMATCH")
    union: set[str] = set()
    for values in decision_arrays:
        if union.intersection(values):
            errors.append("DECISION_PARTITION_MISMATCH")
        union.update(values)
    if union != completed:
        errors.append("DECISION_PARTITION_MISMATCH")
    method_union = set(summary.get("individual_review_pair_ids", [])) | set(summary.get("structural_cohort_confirmation_pair_ids", []))
    if method_union != completed:
        errors.append("DECISION_PARTITION_MISMATCH")
    if set(summary.get("individual_review_pair_ids", [])).intersection(set(summary.get("structural_cohort_confirmation_pair_ids", []))):
        errors.append("DECISION_PARTITION_MISMATCH")
    return order_errors(errors)


def completion_gate(summary: dict[str, Any]) -> bool:
    return (
        summary.get("authorized_F2_pair_count") == EXPECTED_PAIR_COUNT
        and summary.get("reviewed_pair_count") == EXPECTED_PAIR_COUNT
        and summary.get("unreviewed_pair_count") == 0
        and summary.get("invalid_review_count") == 0
        and summary.get("completed_decision_pair_count") == EXPECTED_PAIR_COUNT
        and not summary.get("duplicate_pair_ids")
        and not summary.get("unauthorized_pair_ids")
        and not summary.get("summary_validation_errors")
    )


def next_unreviewed_pair_id(authority: Authority, records: list[dict[str, Any]]) -> str | None:
    reviewed = {record.get("pair_id") for record in records}
    for pair_id in authority.ordered_pair_ids:
        if pair_id not in reviewed:
            return pair_id
    return None


def final_output_dir(repo_root: Path, execution_commit: str) -> Path:
    return repo_root / "reports" / f"reason_router_p2_p3w6f2_hybrid_human_review_execution_{execution_commit[:8]}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def final_artifact_paths(repo_root: Path, target: Path) -> dict[str, str]:
    return {
        "p3w6f2_hybrid_review_completed.csv": (target / "p3w6f2_hybrid_review_completed.csv").relative_to(repo_root).as_posix(),
        "p3w6f2_hybrid_review_summary.json": (target / "p3w6f2_hybrid_review_summary.json").relative_to(repo_root).as_posix(),
        "p3w6f2_hybrid_review_decisions.jsonl": (target / "p3w6f2_hybrid_review_decisions.jsonl").relative_to(repo_root).as_posix(),
    }


def finalize_artifacts(
    authority: Authority,
    wip_path: Path,
    cohort_confirmations: dict[str, CohortConfirmationAuthority] | None = None,
) -> Path:
    records, duplicate_pair_ids = strict_load_wip(authority, wip_path, cohort_confirmations)
    summary = compute_summary(authority, records, duplicate_pair_ids, cohort_confirmations)
    require(summary["completion_gate_passed"], "DECISION_PARTITION_MISMATCH", "completion gate failed")
    execution_commit = require_tracked_execution_state(authority.repo_root)
    summary["F2_execution_commit"] = execution_commit
    summary["F2_execution_decision"] = EXECUTION_DECISION_COMPLETE
    target = final_output_dir(authority.repo_root, execution_commit)
    require(target == final_output_dir(authority.repo_root, execution_commit), "OUTPUT_PATH_MISMATCH", str(target))
    require(not target.exists(), "OUTPUT_PATH_MISMATCH", f"output directory exists: {target}")
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=str(target.parent)))
    try:
        records_by_pair = {record["pair_id"]: record for record in records}
        completed_csv = staging / "p3w6f2_hybrid_review_completed.csv"
        with completed_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COMPLETED_CSV_FIELDS, lineterminator="\n")
            writer.writeheader()
            for source_row in authority.source_rows:
                review = records_by_pair[source_row["pair_id"]]
                writer.writerow({**source_row, **{field: review[field] for field in HUMAN_FIELDS + PROVENANCE_FIELDS}})

        decisions_path = staging / "p3w6f2_hybrid_review_decisions.jsonl"
        with decisions_path.open("w", encoding="utf-8", newline="\n") as handle:
            for source_row in authority.source_rows:
                record = decision_record(records_by_pair[source_row["pair_id"]], authority, cohort_confirmations=cohort_confirmations)
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
                handle.write("\n")

        summary["F2_artifact_paths"] = final_artifact_paths(authority.repo_root, target)
        summary["F2_input_sha256"] = authority.input_artifact_sha256
        summary["F2_output_sha256_contract"] = F2_OUTPUT_SHA256_CONTRACT
        summary["summary_physical_sha256_embedded"] = False
        summary["summary_physical_sha256_authority"] = "external_result_review"
        summary["F2_output_sha256"] = {
            "p3w6f2_hybrid_review_completed.csv": file_sha256(completed_csv),
            "p3w6f2_hybrid_review_decisions.jsonl": file_sha256(decisions_path),
        }
        summary_path = staging / "p3w6f2_hybrid_review_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        expected_files = {
            "p3w6f2_hybrid_review_completed.csv",
            "p3w6f2_hybrid_review_summary.json",
            "p3w6f2_hybrid_review_decisions.jsonl",
        }
        require({path.name for path in staging.iterdir()} == expected_files, "OUTPUT_PATH_MISMATCH", "staging output set mismatch")
        os.rename(staging, target)
        return target
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def print_pair(
    authority: Authority,
    pair_id: str,
    wip_path: Path | None = None,
    cohort_confirmations: dict[str, CohortConfirmationAuthority] | None = None,
) -> None:
    require(pair_id in authority.row_by_pair_id, "UNAUTHORIZED_PAIR_ID", pair_id)
    row = authority.row_by_pair_id[pair_id]
    print(f"pair_id: {pair_id}")
    for prefix, title in (("canonical", "canonical"), ("paraphrase", "paraphrase"), ("polarity_flip", "polarity_flip")):
        label_field = "canonical_final_label" if prefix == "canonical" else f"{prefix}_final_label"
        print(f"\n{title}:")
        print(f"label: {row[label_field]}")
        print(f"claim: {row[f'{prefix}_claim']}")
        print(f"evidence: {row[f'{prefix}_evidence']}")
        print(f"grammar status: {row[f'{prefix}_grammar_status']}")
        print(f"reason codes: {row[f'{prefix}_reason_codes']}")
    print("\nclaim/evidence diff summaries:")
    for field in (
        "canonical_claim_text_diff_summary",
        "paraphrase_claim_text_diff_summary",
        "polarity_flip_claim_text_diff_summary",
        "canonical_evidence_text_diff_summary",
        "paraphrase_evidence_text_diff_summary",
        "polarity_flip_evidence_text_diff_summary",
    ):
        print(f"{field}: {row[field]}")
    print(f"\nautomatic_root_cause_class: {row['automatic_root_cause_class']}")
    print(f"automatic_evidence: {row['automatic_evidence']}")
    print(f"source_record_sha256: {authority.source_sha256_by_pair_id[pair_id]}")
    if wip_path:
        records, duplicate_pair_ids = strict_load_wip(authority, wip_path, cohort_confirmations)
        matches = [record for record in records if record.get("pair_id") == pair_id]
        if matches:
            print("\nexisting WIP review status:")
            print(json.dumps(decision_record(matches[-1], authority, duplicate_pair_ids, cohort_confirmations), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print("\nexisting WIP review status: UNREVIEWED")


def command_show(args: argparse.Namespace) -> int:
    authority = load_authority()
    cohort_confirmations = load_cohort_confirmations(args.cohort_confirmation_path, authority)
    print_pair(authority, args.pair_id, args.wip_path, cohort_confirmations)
    return 0


def command_record(args: argparse.Namespace) -> int:
    require(args.ack_complete_triple_reviewed, "MISSING_HUMAN_FIELD", "missing --ack-complete-triple-reviewed")
    authority = load_authority()
    record = make_review_record(
        authority=authority,
        pair_id=args.pair_id,
        reviewer_id=args.reviewer_id,
        canonical_semantics=args.canonical_semantics,
        paraphrase_semantics=args.paraphrase_semantics,
        polarity_flip_semantics=args.polarity_flip_semantics,
        grammar_validity=args.grammar_validity,
        notes=args.notes,
    )
    upsert_wip_record(authority, args.wip_path, record)
    print(json.dumps(decision_record(record, authority), ensure_ascii=False, sort_keys=True))
    return 0


def command_import_confirmed_individual(args: argparse.Namespace) -> int:
    authority = load_authority()
    require(args.reviewer_id != "", "MISSING_REVIEWER_ID")
    require(args.reviewer_id == args.reviewer_id.strip(), "MISSING_REVIEWER_ID", "reviewer_id has leading/trailing whitespace")
    require_wip_path_outside_repo(authority.repo_root, args.wip_path)
    require_wip_path_outside_repo(authority.repo_root, args.import_intermediate_path)
    records, duplicate_pair_ids = load_wip(args.wip_path)
    require_no_validation_errors(validate_wip_records(authority, records, duplicate_pair_ids))
    import_by_pair = load_xlsx_confirmed_import_intermediate(args.import_intermediate_path, authority)
    existing_pair_ids = {record.get("pair_id") for record in records}
    collisions = [pair_id for pair_id in INDIVIDUALLY_REVIEWED_AUDIT_PAIR_IDS if pair_id in existing_pair_ids]
    require(not collisions, "DUPLICATE_PAIR_ID", ",".join(collisions))

    timestamp = utc_timestamp()
    imported_records = [
        make_xlsx_imported_individual_record(
            authority=authority,
            pair_id=pair_id,
            reviewer_id=args.reviewer_id,
            authority_recorded_at_utc=timestamp,
        )
        for pair_id in INDIVIDUALLY_REVIEWED_AUDIT_PAIR_IDS
    ]
    require(set(import_by_pair) == {record["pair_id"] for record in imported_records}, "WIP_SCHEMA_MISMATCH", "import pair mismatch")
    next_records = [*records, *imported_records]
    require_no_validation_errors(validate_wip_records(authority, next_records))
    require_no_validation_errors(validate_individual_audit_evidence(authority, next_records))
    write_wip_atomic(args.wip_path, next_records)
    print(
        json.dumps(
            {
                "imported_individual_record_count": len(imported_records),
                "imported_pair_ids": [record["pair_id"] for record in imported_records],
                "record_origin": XLSX_CONFIRMED_IMPORT,
                "review_method": INDIVIDUAL_REVIEW_METHOD,
                "human_review_time_provenance": NOT_CAPTURED_IN_XLSX,
                "authority_recorded_at_utc": timestamp,
                "historical_visual_review_time_claimed": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def wip_jsonl_bytes(records: list[dict[str, Any]]) -> bytes:
    lines = [
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for record in records
    ]
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")


def write_bytes_fsynced(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def promote_staged_file(staged: Path, target: Path) -> None:
    os.replace(staged, target)


def restore_target(path: Path, existed: bool, payload: bytes) -> None:
    if existed:
        write_bytes_fsynced(path, payload)
    elif path.exists():
        path.unlink()


def write_cohort_transaction(
    authority: Authority,
    cohort_confirmation_path: Path,
    cohort_artifact: dict[str, Any],
    wip_path: Path,
    records: list[dict[str, Any]],
) -> None:
    require_wip_path_outside_repo(authority.repo_root, cohort_confirmation_path)
    require_wip_path_outside_repo(authority.repo_root, wip_path)
    require(not cohort_confirmation_path.exists(), "INVALID_COHORT_CONFIRMATION_ARTIFACT", f"output exists: {cohort_confirmation_path}")
    confirmation = cohort_authority_from_artifact(cohort_artifact)
    future_confirmations = {confirmation.cohort_confirmation_id: confirmation}
    require_no_validation_errors(validate_cohort_confirmation_artifact(cohort_artifact, authority))
    require_no_validation_errors(validate_wip_records(authority, records, cohort_confirmations=future_confirmations))

    prior_cohort_exists = cohort_confirmation_path.exists()
    prior_cohort = cohort_confirmation_path.read_bytes() if prior_cohort_exists else b""
    prior_wip_exists = wip_path.exists()
    prior_wip = wip_path.read_bytes() if prior_wip_exists else b""
    cohort_bytes = json.dumps(cohort_artifact, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    wip_bytes = wip_jsonl_bytes(records)
    transaction_id = canonical_json_sha256(
        {
            "cohort_confirmation_path": str(cohort_confirmation_path.resolve()),
            "cohort_confirmation_sha256": hashlib.sha256(cohort_bytes).hexdigest(),
            "wip_path": str(wip_path.resolve()),
            "wip_sha256": hashlib.sha256(wip_bytes).hexdigest(),
        }
    )
    staging_root = cohort_confirmation_path.parent / f".{cohort_confirmation_path.name}.{transaction_id[:16]}.txn"
    require(not staging_root.exists(), "INVALID_COHORT_CONFIRMATION_ARTIFACT", f"transaction staging exists: {staging_root}")
    staging_root.mkdir(parents=True)
    try:
        staged_cohort = staging_root / "cohort_confirmation.json"
        staged_wip = staging_root / "wip.jsonl"
        staged_manifest = staging_root / "transaction_manifest.json"
        write_bytes_fsynced(staged_cohort, cohort_bytes)
        write_bytes_fsynced(staged_wip, wip_bytes)
        write_bytes_fsynced(
            staged_manifest,
            json.dumps(
                {
                    "schema_version": "p3w6f2_cohort_confirmation_transaction_v1",
                    "transaction_id": transaction_id,
                    "target_paths": {
                        "cohort_confirmation": str(cohort_confirmation_path.resolve()),
                        "wip": str(wip_path.resolve()),
                    },
                    "staged_sha256": {
                        "cohort_confirmation": path_sha256(staged_cohort),
                        "wip": path_sha256(staged_wip),
                    },
                    "prior_state": {
                        "cohort_confirmation_existed": prior_cohort_exists,
                        "cohort_confirmation_sha256": hashlib.sha256(prior_cohort).hexdigest() if prior_cohort_exists else "",
                        "wip_existed": prior_wip_exists,
                        "wip_sha256": hashlib.sha256(prior_wip).hexdigest() if prior_wip_exists else "",
                    },
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ).encode("utf-8")
            + b"\n",
        )
        staged_artifact = load_external_json_object(staged_cohort, "INVALID_COHORT_CONFIRMATION_ARTIFACT")
        require_no_validation_errors(validate_cohort_confirmation_artifact(staged_artifact, authority))
        staged_records, staged_duplicates = load_wip(staged_wip)
        require_no_validation_errors(validate_wip_records(authority, staged_records, staged_duplicates, future_confirmations))
        promote_staged_file(staged_cohort, cohort_confirmation_path)
        try:
            promote_staged_file(staged_wip, wip_path)
        except Exception:
            restore_target(cohort_confirmation_path, prior_cohort_exists, prior_cohort)
            raise
        require(path_sha256(cohort_confirmation_path) == hashlib.sha256(cohort_bytes).hexdigest(), "INVALID_COHORT_CONFIRMATION_ARTIFACT", "cohort final hash mismatch")
        require(path_sha256(wip_path) == hashlib.sha256(wip_bytes).hexdigest(), "WIP_SCHEMA_MISMATCH", "WIP final hash mismatch")
    except Exception:
        if cohort_confirmation_path.exists() and path_sha256(cohort_confirmation_path) == hashlib.sha256(cohort_bytes).hexdigest():
            restore_target(cohort_confirmation_path, prior_cohort_exists, prior_cohort)
        if wip_path.exists() and path_sha256(wip_path) == hashlib.sha256(wip_bytes).hexdigest():
            restore_target(wip_path, prior_wip_exists, prior_wip)
        raise
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def command_cohort_audit(args: argparse.Namespace) -> int:
    authority = load_authority()
    require_wip_path_outside_repo(authority.repo_root, args.wip_path)
    require_wip_path_outside_repo(authority.repo_root, args.audit_output_path)
    require(not args.audit_output_path.exists(), "INVALID_COHORT_CONFIRMATION_ARTIFACT", f"output exists: {args.audit_output_path}")
    ai_sha256 = path_sha256(args.ai_prescreen_path)
    ai_by_pair_id = load_ai_prescreen(args.ai_prescreen_path, authority)
    records, _duplicate_pair_ids = strict_load_wip(authority, args.wip_path)
    artifact = structural_audit_artifact(
        authority=authority,
        records=records,
        ai_by_pair_id=ai_by_pair_id,
        ai_prescreen_artifact_sha256=ai_sha256,
        audit_created_at_utc=utc_timestamp(),
    )
    require_no_validation_errors(validate_structural_audit_artifact(artifact, authority))
    write_json_atomic(args.audit_output_path, artifact)
    print(
        json.dumps(
            {
                "audit_output_path": str(args.audit_output_path),
                "audit_payload_sha256": artifact["audit_payload_sha256"],
                "source_authority_identity": artifact["source_authority_identity"],
                "ai_prescreen_artifact_sha256": artifact["ai_prescreen_artifact_sha256"],
                "validated_individual_wip_state_sha256": artifact["validated_individual_wip_state_sha256"],
                "individual_audit_count": len(artifact["required_individual_audit_pair_ids"]),
                "eligible_structural_count": len(artifact["eligible_pair_ids"]),
                "exception_count": len(artifact["exception_pair_ids"]),
                "exception_pair_ids": artifact["exception_pair_ids"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def command_cohort_confirm(args: argparse.Namespace) -> int:
    require(args.ack_structural_cohort_confirm, "INVALID_COHORT_CONFIRMATION_ARTIFACT", "missing --ack-structural-cohort-confirm")
    authority = load_authority()
    require_wip_path_outside_repo(authority.repo_root, args.wip_path)
    require_wip_path_outside_repo(authority.repo_root, args.cohort_confirmation_path)
    reviewed_audit = load_structural_audit_artifact(args.audit_path, authority, args.expected_audit_sha256)
    ai_sha256 = path_sha256(args.ai_prescreen_path)
    ai_by_pair_id = load_ai_prescreen(args.ai_prescreen_path, authority)
    records, duplicate_pair_ids = strict_load_wip(authority, args.wip_path)
    fresh_payload = structural_audit_payload(authority, records, ai_by_pair_id, ai_sha256)
    reviewed_payload = {
        field: reviewed_audit[field]
        for field in STRUCTURAL_AUDIT_FIELDS
        if field not in {"audit_created_at_utc", "audit_payload_sha256"}
    }
    require(fresh_payload == reviewed_payload, "INVALID_COHORT_CONFIRMATION_ARTIFACT", "reviewed structural audit is stale")

    timestamp = utc_timestamp()
    artifact = cohort_confirmation_artifact(
        authority=authority,
        audit_artifact=reviewed_audit,
        audit_path=args.audit_path,
        reviewer_id=args.reviewer_id,
        authority_recorded_at_utc=timestamp,
    )
    require_no_validation_errors(validate_cohort_confirmation_artifact(artifact, authority, reviewed_audit))
    cohort_confirmation_id = artifact["cohort_confirmation_id"]
    confirmation = cohort_authority_from_artifact(artifact)
    future_confirmations = {confirmation.cohort_confirmation_id: confirmation}
    cohort_records = [
        make_structural_cohort_record(
            authority=authority,
            pair_id=pair_id,
            reviewer_id=args.reviewer_id,
            cohort_confirmation_id=cohort_confirmation_id,
            authority_recorded_at_utc=timestamp,
        )
        for pair_id in artifact["cohort_confirmed_pair_ids"]
    ]
    next_records = [*records, *cohort_records]
    require_no_validation_errors(validate_wip_records(authority, next_records, duplicate_pair_ids, future_confirmations))
    write_cohort_transaction(authority, args.cohort_confirmation_path, artifact, args.wip_path, next_records)
    print(
        json.dumps(
            {
                "cohort_confirmation_path": str(args.cohort_confirmation_path),
                "cohort_confirmation_id": cohort_confirmation_id,
                "structural_audit_sha256": artifact["structural_audit_sha256"],
                "created_structural_record_count": len(cohort_records),
                "exception_pair_count": artifact["exception_pair_count"],
                "cohort_records_do_not_claim_individual_inspection": True,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def command_status(args: argparse.Namespace) -> int:
    authority = load_authority()
    cohort_confirmations = load_cohort_confirmations(args.cohort_confirmation_path, authority)
    records, duplicate_pair_ids = strict_load_wip(authority, args.wip_path, cohort_confirmations)
    summary = compute_summary(authority, records, duplicate_pair_ids, cohort_confirmations)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def command_next(args: argparse.Namespace) -> int:
    authority = load_authority()
    cohort_confirmations = load_cohort_confirmations(args.cohort_confirmation_path, authority)
    records, _duplicates = strict_load_wip(authority, args.wip_path, cohort_confirmations)
    pair_id = next_unreviewed_pair_id(authority, records)
    if pair_id is None:
        print("NO_UNREVIEWED_PAIR")
        return 0
    if args.show:
        print_pair(authority, pair_id, args.wip_path, cohort_confirmations)
    else:
        print(pair_id)
    return 0


def command_finalize(args: argparse.Namespace) -> int:
    authority = load_authority()
    cohort_confirmations = load_cohort_confirmations(args.cohort_confirmation_path, authority)
    target = finalize_artifacts(authority, args.wip_path, cohort_confirmations)
    print(str(target))
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P3-W6-F2 manual review infrastructure")
    sub = parser.add_subparsers(dest="command", required=True)

    show = sub.add_parser("show")
    show.add_argument("--pair-id", required=True)
    show.add_argument("--wip-path", type=Path)
    show.add_argument("--cohort-confirmation-path", type=Path)
    show.set_defaults(func=command_show)

    record = sub.add_parser("record")
    record.add_argument("--pair-id", required=True)
    record.add_argument("--reviewer-id", required=True)
    record.add_argument("--canonical-semantics", required=True)
    record.add_argument("--paraphrase-semantics", required=True)
    record.add_argument("--polarity-flip-semantics", required=True)
    record.add_argument("--grammar-validity", required=True)
    record.add_argument("--notes", required=True)
    record.add_argument("--wip-path", type=Path, default=Path(DEFAULT_WIP_PATH))
    record.add_argument("--ack-complete-triple-reviewed", action="store_true")
    record.set_defaults(func=command_record)

    import_confirmed = sub.add_parser("import-confirmed-individual")
    import_confirmed.add_argument("--wip-path", type=Path, default=Path(DEFAULT_WIP_PATH))
    import_confirmed.add_argument("--import-intermediate-path", type=Path, required=True)
    import_confirmed.add_argument("--reviewer-id", required=True)
    import_confirmed.set_defaults(func=command_import_confirmed_individual)

    audit = sub.add_parser("cohort-audit")
    audit.add_argument("--wip-path", type=Path, default=Path(DEFAULT_WIP_PATH))
    audit.add_argument("--ai-prescreen-path", type=Path, required=True)
    audit.add_argument("--audit-output-path", type=Path, required=True)
    audit.set_defaults(func=command_cohort_audit)

    cohort = sub.add_parser("cohort-confirm")
    cohort.add_argument("--wip-path", type=Path, default=Path(DEFAULT_WIP_PATH))
    cohort.add_argument("--ai-prescreen-path", type=Path, required=True)
    cohort.add_argument("--audit-path", type=Path, required=True)
    cohort.add_argument("--expected-audit-sha256", required=True)
    cohort.add_argument("--cohort-confirmation-path", type=Path, required=True)
    cohort.add_argument("--reviewer-id", required=True)
    cohort.add_argument("--ack-structural-cohort-confirm", action="store_true")
    cohort.set_defaults(func=command_cohort_confirm)

    status = sub.add_parser("status")
    status.add_argument("--wip-path", type=Path, default=Path(DEFAULT_WIP_PATH))
    status.add_argument("--cohort-confirmation-path", type=Path)
    status.set_defaults(func=command_status)

    next_cmd = sub.add_parser("next")
    next_cmd.add_argument("--wip-path", type=Path, default=Path(DEFAULT_WIP_PATH))
    next_cmd.add_argument("--cohort-confirmation-path", type=Path)
    next_cmd.add_argument("--show", action="store_true")
    next_cmd.set_defaults(func=command_next)

    finalize = sub.add_parser("finalize")
    finalize.add_argument("--wip-path", type=Path, default=Path(DEFAULT_WIP_PATH))
    finalize.add_argument("--cohort-confirmation-path", type=Path)
    finalize.set_defaults(func=command_finalize)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ReviewInfrastructureError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
