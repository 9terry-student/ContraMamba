from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
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
REVIEW_PROTOCOL_VERSION = "P3W5_F2_MANUAL_REVIEW_V1"
COMPATIBILITY_MATRIX_VERSION = "F2_REVIEW_COMPATIBILITY_V1"
SOURCE_HASH_VERSION = "F2_SOURCE_RECORD_HASH_V1"
EXECUTION_STATUS_COMPLETE = "P3W5_F2_MANUAL_REVIEW_EXECUTION_COMPLETE_PENDING_RESULT_REVIEW"
EXECUTION_DECISION_COMPLETE = "P3W5_F2_MANUAL_REVIEW_LEVEL1_COMPLETE_PENDING_RESULT_REVIEW"
F2_OUTPUT_SHA256_CONTRACT = "NON_SELF_REFERENTIAL_ARTIFACT_SHA256_MAP_V1"

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
PROVENANCE_FIELDS = [
    "source_record_sha256",
    "reviewer_id",
    "review_protocol_version",
    "reviewed_at_utc",
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
]
WIP_FIELD_SET = frozenset(WIP_FIELDS)
COMPLETED_CSV_FIELDS = [*SOURCE_FIELDS, *HUMAN_FIELDS, *PROVENANCE_FIELDS]
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


def require(condition: bool, code: str, detail: str = "") -> None:
    if not condition:
        raise ReviewInfrastructureError(code if not detail else f"{code}: {detail}")


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
    require(manifest.get("review_protocol_version") == REVIEW_PROTOCOL_VERSION, "INVALID_REVIEW_PROTOCOL_VERSION")
    require(manifest.get("compatibility_matrix_version") == COMPATIBILITY_MATRIX_VERSION, "COMPATIBILITY_MATRIX_MISMATCH")
    require(manifest.get("source_hash_version") == SOURCE_HASH_VERSION, "SOURCE_SCHEMA_MISMATCH")

    source_fields = list(manifest.get("immutable_source_columns", []))
    require(source_fields == SOURCE_FIELDS, "SOURCE_SCHEMA_MISMATCH", "P1 immutable source columns differ from implementation constants")
    require(list(manifest.get("human_review_fields", [])) == HUMAN_FIELDS, "SOURCE_SCHEMA_MISMATCH", "P1 human fields differ")
    require(list(manifest.get("reviewer_provenance_fields", [])) == PROVENANCE_FIELDS, "SOURCE_SCHEMA_MISMATCH", "P1 provenance fields differ")
    require(list(manifest.get("completed_csv_schema_order", [])) == COMPLETED_CSV_FIELDS, "SOURCE_SCHEMA_MISMATCH", "P1 completed schema differs")

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
    require([row["pair_id"] for row in source_rows] == [row["pair_id"] for row in source_rows], "AUTHORITY_PAIR_UNIVERSE_MISMATCH")
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


def strict_load_wip(authority: Authority, path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    require_wip_path_outside_repo(authority.repo_root, path)
    records, duplicate_pair_ids = load_wip(path)
    errors = validate_wip_records(authority, records, duplicate_pair_ids)
    require(not errors, errors[0], ",".join(errors))
    return records, duplicate_pair_ids


def validate_wip_records(authority: Authority, records: list[dict[str, Any]], duplicate_pair_ids: set[str] | None = None) -> list[str]:
    duplicate_pair_ids = duplicate_pair_ids or set()
    errors: list[str] = []
    authorized = authority.row_by_pair_id
    for record in records:
        if frozenset(record.keys()) != WIP_FIELD_SET:
            errors.append("WIP_SCHEMA_MISMATCH")
        errors.extend(validate_review_record(record, authority, duplicate_pair_ids))
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
        "reviewed_at_utc": utc_timestamp(clock),
    }
    errors = validate_review_record(record, authority)
    require(not errors, errors[0], ",".join(errors))
    return record


def upsert_wip_record(authority: Authority, path: Path, record: dict[str, Any]) -> None:
    require_wip_path_outside_repo(authority.repo_root, path)
    records, duplicate_pair_ids = load_wip(path)
    errors = validate_wip_records(authority, records, duplicate_pair_ids)
    require(not errors, errors[0], ",".join(errors))
    record_errors = validate_review_record(record, authority)
    require(not record_errors, record_errors[0], ",".join(record_errors))
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


def decision_record(record: dict[str, Any], authority: Authority, duplicate_pair_ids: set[str] | None = None) -> dict[str, Any]:
    expected_decision = ""
    errors = validate_review_record(record, authority, duplicate_pair_ids)
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


def compute_summary(authority: Authority, records: list[dict[str, Any]], duplicate_pair_ids: set[str] | None = None) -> dict[str, Any]:
    duplicate_pair_ids = duplicate_pair_ids or set()
    authorized = authority.ordered_pair_ids
    authorized_set = set(authorized)
    reviewed_ids = [record.get("pair_id", "") for record in records if record.get("pair_id", "") in authorized_set]
    reviewed_unique = [pair_id for pair_id in authorized if pair_id in set(reviewed_ids)]
    unreviewed = [pair_id for pair_id in authorized if pair_id not in set(reviewed_ids)]
    decision_records = [decision_record(record, authority, duplicate_pair_ids) for record in records]
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
        decision = records_by_pair[pair_id]["human_authority_decision"]
        summary[DECISION_TO_SUMMARY_ARRAY[decision]].append(pair_id)
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
    return repo_root / "reports" / f"reason_router_p2_p3w5_f2_manual_review_execution_{execution_commit[:8]}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def final_artifact_paths(repo_root: Path, target: Path) -> dict[str, str]:
    return {
        "p3w5_f2_review_completed.csv": (target / "p3w5_f2_review_completed.csv").relative_to(repo_root).as_posix(),
        "p3w5_f2_review_summary.json": (target / "p3w5_f2_review_summary.json").relative_to(repo_root).as_posix(),
        "p3w5_f2_review_decisions.jsonl": (target / "p3w5_f2_review_decisions.jsonl").relative_to(repo_root).as_posix(),
    }


def finalize_artifacts(authority: Authority, wip_path: Path) -> Path:
    records, duplicate_pair_ids = strict_load_wip(authority, wip_path)
    summary = compute_summary(authority, records, duplicate_pair_ids)
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
        completed_csv = staging / "p3w5_f2_review_completed.csv"
        with completed_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COMPLETED_CSV_FIELDS, lineterminator="\n")
            writer.writeheader()
            for source_row in authority.source_rows:
                review = records_by_pair[source_row["pair_id"]]
                writer.writerow({**source_row, **{field: review[field] for field in HUMAN_FIELDS + PROVENANCE_FIELDS}})

        decisions_path = staging / "p3w5_f2_review_decisions.jsonl"
        with decisions_path.open("w", encoding="utf-8", newline="\n") as handle:
            for source_row in authority.source_rows:
                record = decision_record(records_by_pair[source_row["pair_id"]], authority)
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
                handle.write("\n")

        summary["F2_artifact_paths"] = final_artifact_paths(authority.repo_root, target)
        summary["F2_input_sha256"] = authority.input_artifact_sha256
        summary["F2_output_sha256_contract"] = F2_OUTPUT_SHA256_CONTRACT
        summary["summary_physical_sha256_embedded"] = False
        summary["summary_physical_sha256_authority"] = "external_result_review"
        summary["F2_output_sha256"] = {
            "p3w5_f2_review_completed.csv": file_sha256(completed_csv),
            "p3w5_f2_review_decisions.jsonl": file_sha256(decisions_path),
        }
        summary_path = staging / "p3w5_f2_review_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        expected_files = {
            "p3w5_f2_review_completed.csv",
            "p3w5_f2_review_summary.json",
            "p3w5_f2_review_decisions.jsonl",
        }
        require({path.name for path in staging.iterdir()} == expected_files, "OUTPUT_PATH_MISMATCH", "staging output set mismatch")
        os.rename(staging, target)
        return target
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def print_pair(authority: Authority, pair_id: str, wip_path: Path | None = None) -> None:
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
        records, duplicate_pair_ids = strict_load_wip(authority, wip_path)
        matches = [record for record in records if record.get("pair_id") == pair_id]
        if matches:
            print("\nexisting WIP review status:")
            print(json.dumps(decision_record(matches[-1], authority, duplicate_pair_ids), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print("\nexisting WIP review status: UNREVIEWED")


def command_show(args: argparse.Namespace) -> int:
    authority = load_authority()
    print_pair(authority, args.pair_id, args.wip_path)
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


def command_status(args: argparse.Namespace) -> int:
    authority = load_authority()
    records, duplicate_pair_ids = strict_load_wip(authority, args.wip_path)
    summary = compute_summary(authority, records, duplicate_pair_ids)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def command_next(args: argparse.Namespace) -> int:
    authority = load_authority()
    records, _duplicates = strict_load_wip(authority, args.wip_path)
    pair_id = next_unreviewed_pair_id(authority, records)
    if pair_id is None:
        print("NO_UNREVIEWED_PAIR")
        return 0
    if args.show:
        print_pair(authority, pair_id, args.wip_path)
    else:
        print(pair_id)
    return 0


def command_finalize(args: argparse.Namespace) -> int:
    authority = load_authority()
    target = finalize_artifacts(authority, args.wip_path)
    print(str(target))
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P3-W6-F2 manual review infrastructure")
    sub = parser.add_subparsers(dest="command", required=True)

    show = sub.add_parser("show")
    show.add_argument("--pair-id", required=True)
    show.add_argument("--wip-path", type=Path)
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

    status = sub.add_parser("status")
    status.add_argument("--wip-path", type=Path, default=Path(DEFAULT_WIP_PATH))
    status.set_defaults(func=command_status)

    next_cmd = sub.add_parser("next")
    next_cmd.add_argument("--wip-path", type=Path, default=Path(DEFAULT_WIP_PATH))
    next_cmd.add_argument("--show", action="store_true")
    next_cmd.set_defaults(func=command_next)

    finalize = sub.add_parser("finalize")
    finalize.add_argument("--wip-path", type=Path, default=Path(DEFAULT_WIP_PATH))
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
