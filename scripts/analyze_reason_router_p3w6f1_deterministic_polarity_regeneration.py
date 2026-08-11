from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from scripts import build_stage185a_controlled_train_integrity_sidecar as stage185_builder
from scripts import build_controlled_v5 as generator
from scripts import analyze_reason_router_p3w4_canonical_grammar_authority as p3w4_authority


SCHEMA_VERSION = "reason_router_p3w6f1_deterministic_polarity_regeneration_v4"
TARGET_SCOPE_MEMBERSHIP_UNRESOLVED = "TARGET_SCOPE_MEMBERSHIP_UNRESOLVED"
F1_AUTHORITY_CARDINALITY_MISMATCH = "F1_AUTHORITY_CARDINALITY_MISMATCH"
BASE_FORM_COVERAGE_UNRESOLVED = "BASE_FORM_COVERAGE_UNRESOLVED"
STAGE185_PROVENANCE_UNRESOLVED = "STAGE185_PROVENANCE_UNRESOLVED"
STAGE185_TRANSITION_FAILED = "STAGE185_TRANSITION_FAILED"
FULL_OUTPUT_ISOLATION_FAILED = "FULL_OUTPUT_ISOLATION_FAILED"
PROVENANCE_UNRESOLVED = "PROVENANCE_UNRESOLVED"
PROVENANCE_IDENTITY_MISMATCH = "PROVENANCE_IDENTITY_MISMATCH"
PASS_STATUS = "DETERMINISTIC_POLARITY_REPAIR_PASS"
MANUAL_STATUS = "MANUAL_REVIEW_REQUIRED"
REJECTED_STATUS = "REJECTED"
SEMANTIC_AUTHORITY_UNRESOLVED = "SEMANTIC_AUTHORITY_UNRESOLVED"
BASE_FORM_METHOD = "generator_owned_explicit_mapping"
BASE_FORM_SYMBOL = "_BASE_PREDICATE_BY_INFLECTED"
GENERATOR_SOURCE_PATH = "scripts/build_controlled_v5.py"
GRAMMAR_VALIDATOR_SOURCE_PATH = "scripts/build_stage185a_controlled_train_integrity_sidecar.py"
GRAMMAR_VALIDATOR_SOURCE = GRAMMAR_VALIDATOR_SOURCE_PATH + "::stage185_contract_sidecar"
STAGE185_RULE_VERSION = "stage185a_v1"
P3W6F1_STAGE185_SPLIT_SEED = 174
P3W6F1_STAGE185_DEV_RATIO = 0.2
P3W6F1_EXPECTED_TRAIN_ROWS = 2880
P3W6F1_EXPECTED_DEV_ROWS = 720
P3W6F1_AUTHORITATIVE_DATA_PATH = "data/controlled_v5_v3_without_time_swap.jsonl"
P3W6F1_AUTHORITATIVE_DATA_SHA256 = "f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640"
P3W6F1_BASELINE_SIDECAR_PATH = (
    "reports/stage185a_controlled_train_integrity_sidecar_20260715_141914/"
    "stage185a_controlled_train_integrity_sidecar.jsonl"
)
P3W6F1_BASELINE_SIDECAR_SEMANTIC_SHA256 = (
    "5bc03caa2a29f9b9176ab4eb0201db57ebad516352797546db1a18e6ec3373fc"
)
P3W6F1_P3W4_SUMMARY_PATH = (
    "reports/reason_router_p2_p3w4_canonical_grammar_authority_execution_ca99038d/"
    "p3w4_canonical_grammar_authority_summary.json"
)
P3W6F1_P3W4_SUMMARY_SHA256 = "7c0cc383dde38a1c564dae445a78eaf9171b8648d0720de3a2acc0ba68e68e80"
P3W6F1_P3W4_PAIRS_PATH = (
    "reports/reason_router_p2_p3w4_canonical_grammar_authority_execution_ca99038d/"
    "p3w4_canonical_grammar_authority_pairs.jsonl"
)
P3W6F1_P3W4_PAIRS_SHA256 = "850ac6e8924fe334fa7f18659d204f6e0546381b1c3d3eb601f893f3eb00a493"
P3W6F1_P3W5_MANIFEST_PATH = "reports/reason_router_p2_p3w5_separate_remediation_manifest.json"
P3W6F1_P3W5_AUTHORITY_COMMIT = "01d983f8d09cacf0eddefd2014fc81a28771cf5e"
P3W6F1_TRUSTED_STAGE185_DEPENDENCY_COMMIT = "ff6929bf33693fb4e70bd9528551053f4402fe1c"
P3W6F1_STAGE184_CONTRACT_MATRIX_PATH = (
    "reports/stage184a_controlled_train_integrity_mask_spec_20260715_134538/"
    "stage184a_family_contract_matrix.csv"
)
P3W6F1_STAGE185_BUILDER_PATH = "scripts/build_stage185a_controlled_train_integrity_sidecar.py"
P3W6F1_STAGE182_ANALYZER_PATH = "scripts/analyze_stage182a_controlled_intervention_integrity.py"
EXPECTED_F1_TARGET_COUNT = 121
REQUIRED_STAGE185_BEFORE = {
    "grammar_status": "FAIL",
    "integrity_status": "INELIGIBLE",
    "canonical_status": "PASS",
}
REQUIRED_STAGE185_AFTER = {
    "dataset_source_status": "PASS",
    "schema_status": "PASS",
    "intervention_contract_status": "PASS",
    "grammar_status": "PASS",
    "integrity_status": "ELIGIBLE",
    "canonical_status": "PASS",
    "polarity_contamination_status": "PASS",
    "time_swap_status": "PASS",
    "audit_expected_axes": ["polarity"],
    "audit_changed_axes": ["polarity"],
    "audit_pair_failure_scope": "none",
}
REQUIRED_STAGE185_SCHEMA_FIELDS = (
    "row_id",
    "pair_id",
    "split",
    "intervention_type",
    "frame_compatible_label",
    "grammar_status",
    "intervention_contract_status",
    "polarity_contamination_status",
    "schema_status",
    "canonical_status",
    "time_swap_status",
    "dataset_source_status",
    "integrity_status",
    "eligible_for_positive_margin",
    "reason_codes",
    "canonical_row_id",
    "family_contract_id",
    "rule_version",
    "source_dataset_path",
    "source_dataset_sha256",
    "generator_source_path",
    "generator_source_sha256",
    "stage182a_report_sha256",
    "stage184a_report_sha256",
    "integrity_builder_sha256",
    "created_at",
    "audit_changed_axes",
    "audit_preserved_axes",
    "audit_expected_axes",
    "audit_pair_failure_scope",
)
STAGE185_SEMANTIC_IDENTITY_FIELDS = (
    "split",
    "intervention_type",
    "frame_compatible_label",
    "grammar_status",
    "intervention_contract_status",
    "polarity_contamination_status",
    "schema_status",
    "canonical_status",
    "time_swap_status",
    "dataset_source_status",
    "integrity_status",
    "eligible_for_positive_margin",
    "reason_codes",
    "canonical_row_id",
    "family_contract_id",
    "rule_version",
    "audit_changed_axes",
    "audit_preserved_axes",
    "audit_expected_axes",
    "audit_pair_failure_scope",
)
BLOCKER_ORDER = (
    F1_AUTHORITY_CARDINALITY_MISMATCH,
    TARGET_SCOPE_MEMBERSHIP_UNRESOLVED,
    BASE_FORM_COVERAGE_UNRESOLVED,
    STAGE185_PROVENANCE_UNRESOLVED,
    STAGE185_TRANSITION_FAILED,
    FULL_OUTPUT_ISOLATION_FAILED,
    SEMANTIC_AUTHORITY_UNRESOLVED,
    PROVENANCE_IDENTITY_MISMATCH,
    PROVENANCE_UNRESOLVED,
)
ALL_ACCEPTED_DECISION = "P3W5_F1_REGENERATION_COMPLETE_ALL_CANDIDATES_ACCEPTED_PENDING_RESULT_REVIEW"
BLOCKERS_DECISION = "P3W5_F1_REGENERATION_COMPLETE_WITH_BLOCKERS_PENDING_RESULT_REVIEW"

SUMMARY_PAIR_FIELDS = (
    ("F1_target_pair_count", "F1_target_pair_ids"),
    ("F1_generated_candidate_count", "F1_generated_candidate_pair_ids"),
    ("F1_accepted_candidate_count", "F1_accepted_candidate_pair_ids"),
    ("F1_manual_review_required_count", "F1_manual_review_required_pair_ids"),
    ("F1_rejected_candidate_count", "F1_rejected_candidate_pair_ids"),
    ("F1_missing_candidate_count", "F1_missing_candidate_pair_ids"),
    ("F1_unauthorized_candidate_count", "F1_unauthorized_candidate_pair_ids"),
)

AUDIT_REQUIRED_FIELDS = (
    "pair_id",
    "original_row_id",
    "regenerated_row_id",
    "intervention_type",
    "original_text",
    "regenerated_text",
    "original_final_label",
    "regenerated_final_label",
    "canonical_row_id",
    "generator_source_path",
    "generator_source_sha256",
    "generator_commit",
    "fact_identity",
    "grammar_validator_source",
    "grammar_validator_sha256",
    "grammar_before",
    "grammar_after",
    "sidecar_before",
    "sidecar_after",
    "lineage_preserved",
    "semantic_validation_status",
    "semantic_polarity_preserved",
    "candidate_accepted",
    "ordered_rejection_codes",
    "inflected_predicate_surface",
    "expected_base_predicate",
    "base_form_derivation_method",
    "base_form_derivation_source_path",
    "base_form_derivation_source_sha256",
    "base_form_source_symbol",
    "authorized_replacement_span",
    "outside_span_byte_identity",
    "canonical_text",
    "original_defective_text",
    "normalized_changed_span",
    "negation_markers_added",
    "negation_markers_removed",
    "auxiliary_verb_changes",
    "predicate_inflection_changes",
    "duplicate_or_missing_tokens",
    "semantic_validation_method",
    "semantic_validation_evidence",
)

FULL_OUTPUT_ISOLATION_FIELDS = (
    "changed_ids",
    "structural_negative_polarity_flip_row_ids",
    "authorized_F1_row_ids",
    "non_authorized_structural_negative_polarity_flip_row_ids",
    "repair_consumed_row_ids",
    "non_authorized_structural_negative_polarity_flip_changed_row_ids",
    "missing_ids",
    "added_ids",
    "duplicate_ids",
    "unauthorized_changed_row_ids",
    "F2_changed_row_ids",
    "unaffected_changed_row_ids",
    "canonical_changed_row_ids",
    "paraphrase_changed_row_ids",
    "evidence_changed_row_ids",
    "claim_changed_row_ids",
    "non_text_field_changed_row_ids",
    "baseline_generator_commit",
    "baseline_generator_source_path",
    "baseline_generator_source_sha256",
    "repaired_generator_commit",
    "repaired_generator_source_path",
    "repaired_generator_source_sha256",
    "deterministic_generator_invocation",
    "generator_configuration_identity",
    "baseline_complete_output_sha256",
    "repaired_complete_output_sha256",
    "baseline_row_count",
    "repaired_row_count",
    "baseline_id_sequence",
    "repaired_id_sequence",
    "baseline_id_sequence_sha256",
    "repaired_id_sequence_sha256",
    "row_order_changed",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_repo_path(repo_root: Path, relative_path: str) -> Path:
    return (repo_root / relative_path).resolve()


def path_is_repo_internal(repo_root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(repo_root.resolve())
        return True
    except ValueError:
        return False


def is_git_tracked(repo_root: Path, relative_path: str) -> bool:
    try:
        subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "--error-unmatch", relative_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def require_canonical_repo_file(
    repo_root: Path,
    caller_path: Path,
    canonical_relative_path: str,
    *,
    expected_sha256: str | None = None,
    require_git_tracked: bool = True,
    git_tracked_checker: Any | None = None,
) -> Path:
    canonical = resolve_repo_path(repo_root, canonical_relative_path)
    if caller_path.resolve() != canonical:
        raise ValueError(PROVENANCE_IDENTITY_MISMATCH)
    if not path_is_repo_internal(repo_root, canonical):
        raise ValueError(PROVENANCE_IDENTITY_MISMATCH)
    if require_git_tracked:
        tracked = git_tracked_checker or (lambda relative_path: is_git_tracked(repo_root, relative_path))
        if not tracked(canonical_relative_path):
            raise ValueError(PROVENANCE_IDENTITY_MISMATCH)
    if expected_sha256 is not None and file_sha256(canonical) != expected_sha256:
        raise ValueError(PROVENANCE_IDENTITY_MISMATCH)
    return canonical


def sidecar_source_path_matches(source_dataset_path: Any, expected_source_path: Path, repo_root: Path | None = None) -> bool:
    if not isinstance(source_dataset_path, str) or not source_dataset_path.strip():
        return False
    raw = source_dataset_path.replace("\\", "/")
    expected = expected_source_path.resolve()
    if raw.startswith("/kaggle/working/ContraMamba/"):
        raw = raw.removeprefix("/kaggle/working/ContraMamba/")
    candidate = Path(raw)
    if not candidate.is_absolute() and repo_root is not None:
        candidate = repo_root.resolve() / candidate
    try:
        return candidate.resolve() == expected
    except OSError:
        return False


def singleton_sidecar_value(sidecar_rows: Sequence[Mapping[str, Any]], field: str) -> str:
    values = {str(row.get(field, "")) for row in sidecar_rows}
    if len(values) != 1 or "" in values:
        raise ValueError(PROVENANCE_IDENTITY_MISMATCH)
    return next(iter(values))


def trusted_blob_identity(
    *,
    repo_root: Path,
    relative_path: str,
    trusted_commit: str = P3W6F1_TRUSTED_STAGE185_DEPENDENCY_COMMIT,
    git_tracked_checker: Any | None = None,
    git_blob_reader: Any | None = None,
) -> dict[str, Any]:
    path = require_canonical_repo_file(
        repo_root,
        resolve_repo_path(repo_root, relative_path),
        relative_path,
        git_tracked_checker=git_tracked_checker,
    )
    blob_reader = git_blob_reader or (lambda commit, blob_path: git_blob_bytes(repo_root, commit, blob_path))
    trusted_blob = blob_reader(trusted_commit, relative_path)
    if trusted_blob is None or path.read_bytes() != trusted_blob:
        raise ValueError(STAGE185_PROVENANCE_UNRESOLVED)
    return {
        "path": relative_path,
        "trusted_commit": trusted_commit,
        "source_sha256": sha256_bytes(trusted_blob),
    }


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON file is not an object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            require(isinstance(value, dict), f"JSONL line is not an object: {path}:{line_number}")
            rows.append(value)
    return rows


def stage185_semantic_sidecar_sha256(sidecar_rows: Sequence[Mapping[str, Any]]) -> str:
    return stage185_builder.semantic_sidecar_sha([dict(row) for row in sidecar_rows])


def validate_p3w6f1_input_authority_identity(
    *,
    repo_root: Path,
    baseline_jsonl_path: Path,
    baseline_sidecar_jsonl_path: Path,
    p3w4_summary_json_path: Path,
    p3w4_pairs_jsonl_path: Path,
    p3w5_manifest_json_path: Path,
    f1_input_sha256: str,
    git_tracked_checker: Any | None = None,
    git_blob_reader: Any | None = None,
) -> dict[str, Any]:
    root = repo_root.resolve()
    baseline_path = require_canonical_repo_file(
        root,
        baseline_jsonl_path,
        P3W6F1_AUTHORITATIVE_DATA_PATH,
        expected_sha256=P3W6F1_AUTHORITATIVE_DATA_SHA256,
        git_tracked_checker=git_tracked_checker,
    )
    if f1_input_sha256 != P3W6F1_AUTHORITATIVE_DATA_SHA256:
        raise ValueError(PROVENANCE_IDENTITY_MISMATCH)
    baseline_sidecar_path = require_canonical_repo_file(
        root,
        baseline_sidecar_jsonl_path,
        P3W6F1_BASELINE_SIDECAR_PATH,
        git_tracked_checker=git_tracked_checker,
    )
    baseline_sidecar_rows = load_jsonl(baseline_sidecar_path)
    observed_sidecar_semantic_sha = stage185_semantic_sidecar_sha256(baseline_sidecar_rows)
    if observed_sidecar_semantic_sha != P3W6F1_BASELINE_SIDECAR_SEMANTIC_SHA256:
        raise ValueError(PROVENANCE_IDENTITY_MISMATCH)
    baseline_generator_source_sha256 = singleton_sidecar_value(baseline_sidecar_rows, "generator_source_sha256")
    baseline_integrity_builder_sha256 = singleton_sidecar_value(baseline_sidecar_rows, "integrity_builder_sha256")
    baseline_generator_paths = {str(row.get("generator_source_path", "")) for row in baseline_sidecar_rows}
    if not baseline_generator_paths or any(
        not sidecar_source_path_matches(path, resolve_repo_path(root, GENERATOR_SOURCE_PATH), root)
        for path in baseline_generator_paths
    ):
        raise ValueError(PROVENANCE_IDENTITY_MISMATCH)
    require_canonical_repo_file(
        root,
        p3w4_summary_json_path,
        P3W6F1_P3W4_SUMMARY_PATH,
        expected_sha256=P3W6F1_P3W4_SUMMARY_SHA256,
        git_tracked_checker=git_tracked_checker,
    )
    require_canonical_repo_file(
        root,
        p3w4_pairs_jsonl_path,
        P3W6F1_P3W4_PAIRS_PATH,
        expected_sha256=P3W6F1_P3W4_PAIRS_SHA256,
        git_tracked_checker=git_tracked_checker,
    )
    p3w5_manifest_path = require_canonical_repo_file(
        root,
        p3w5_manifest_json_path,
        P3W6F1_P3W5_MANIFEST_PATH,
        git_tracked_checker=git_tracked_checker,
    )
    blob_reader = git_blob_reader or (lambda commit, path: git_blob_bytes(root, commit, path))
    expected_manifest_blob = blob_reader(P3W6F1_P3W5_AUTHORITY_COMMIT, P3W6F1_P3W5_MANIFEST_PATH)
    if expected_manifest_blob is None or p3w5_manifest_path.read_bytes() != expected_manifest_blob:
        raise ValueError(PROVENANCE_IDENTITY_MISMATCH)
    return {
        "input_authority_identity_pass": True,
        "input_authority_identity_status": "PASS",
        "baseline_jsonl_path": str(baseline_path),
        "baseline_jsonl_sha256": P3W6F1_AUTHORITATIVE_DATA_SHA256,
        "baseline_sidecar_path": str(baseline_sidecar_path),
        "baseline_sidecar_semantic_sha256": observed_sidecar_semantic_sha,
        "baseline_generator_source_sha256": baseline_generator_source_sha256,
        "baseline_integrity_builder_sha256": baseline_integrity_builder_sha256,
        "p3w4_summary_path": P3W6F1_P3W4_SUMMARY_PATH,
        "p3w4_pairs_path": P3W6F1_P3W4_PAIRS_PATH,
        "p3w5_manifest_path": P3W6F1_P3W5_MANIFEST_PATH,
        "p3w5_authority_commit": P3W6F1_P3W5_AUTHORITY_COMMIT,
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def tokenise(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]", text)


def token_count(text: str, token: str) -> int:
    return sum(1 for value in tokenise(text) if value.lower() == token.lower())


def extract_decision_supporting_pair_ids(p3w4_summary: Mapping[str, Any], p3w5_manifest: Mapping[str, Any] | None = None) -> set[str]:
    value = p3w4_summary.get("decision_supporting_pair_ids")
    if value is None and p3w5_manifest is not None:
        value = p3w5_manifest.get("decision_supporting_pair_ids")
    require(isinstance(value, list), "decision_supporting_pair_ids missing")
    result = {str(item) for item in value}
    require(len(result) == len(value), "duplicate decision_supporting_pair_ids")
    return result


def polarity_member(record: Mapping[str, Any]) -> Mapping[str, Any]:
    members = record.get("members")
    require(isinstance(members, Mapping), "authority record missing members")
    member = members.get("polarity_flip") or members.get("polarity")
    require(isinstance(member, Mapping), "authority record missing polarity member")
    return member


def extract_authorized_f1_targets(
    pair_records: Iterable[Mapping[str, Any]],
    decision_supporting_pair_ids: set[str],
) -> dict[str, Any]:
    target_pair_ids: list[str] = []
    target_row_ids: list[str] = []
    for record in pair_records:
        pair_id = str(record.get("pair_id", ""))
        if record.get("family") != "F1":
            continue
        if record.get("automatic_root_cause_class") != "F1_TRUE_POLARITY_GENERATION_DEFECT":
            continue
        if record.get("remediation_state") != "REGENERATION_REQUIRED":
            continue
        if pair_id not in decision_supporting_pair_ids:
            continue
        source_row = polarity_member(record).get("source_row")
        require(isinstance(source_row, Mapping), "polarity member missing source_row")
        row_id = str(source_row.get("id", ""))
        require(row_id.endswith("__polarity_flip"), "authorized F1 row is not polarity_flip")
        require(source_row.get("intervention_type") == "polarity_flip", "authorized F1 row intervention mismatch")
        target_pair_ids.append(pair_id)
        target_row_ids.append(row_id)
    require(len(target_pair_ids) == len(set(target_pair_ids)), "duplicate authorized F1 pair")
    require(len(target_row_ids) == len(set(target_row_ids)), "duplicate authorized F1 row")
    return {
        "F1_target_pair_count": len(target_pair_ids),
        "F1_target_pair_ids": sorted(target_pair_ids),
        "F1_target_row_count": len(target_row_ids),
        "authorized_F1_row_ids": sorted(target_row_ids),
    }


def ordered_blockers(codes: Iterable[str]) -> list[str]:
    observed = {str(code) for code in codes if code}
    ordered = [code for code in BLOCKER_ORDER if code in observed]
    ordered.extend(sorted(observed - set(ordered)))
    return ordered


def validate_authority_cardinality(
    targets: Mapping[str, Any], expected: int = EXPECTED_F1_TARGET_COUNT
) -> dict[str, Any]:
    row_ids = targets.get("authorized_F1_row_ids")
    pair_ids = targets.get("F1_target_pair_ids")
    passed = (
        targets.get("F1_target_pair_count") == expected
        and targets.get("F1_target_row_count") == expected
        and isinstance(row_ids, list)
        and isinstance(pair_ids, list)
        and len(row_ids) == expected
        and len(pair_ids) == expected
    )
    return {
        "authority_cardinality_pass": passed,
        "authority_cardinality_status": "PASS" if passed else F1_AUTHORITY_CARDINALITY_MISMATCH,
        "expected_F1_target_pair_count": expected,
        "expected_F1_target_row_count": expected,
        "observed_F1_target_pair_count": targets.get("F1_target_pair_count"),
        "observed_F1_target_row_count": targets.get("F1_target_row_count"),
        "observed_authorized_F1_row_id_count": len(row_ids) if isinstance(row_ids, list) else None,
    }

def extract_f2_row_ids(pair_records: Iterable[Mapping[str, Any]]) -> set[str]:
    row_ids: set[str] = set()
    for record in pair_records:
        if record.get("family") != "F2":
            continue
        members = record.get("members")
        require(isinstance(members, Mapping), "F2 authority record missing members")
        for member in members.values():
            require(isinstance(member, Mapping), "F2 member malformed")
            source_row = member.get("source_row")
            require(isinstance(source_row, Mapping), "F2 member missing source_row")
            row_ids.add(str(source_row.get("id", "")))
    row_ids.discard("")
    return row_ids

def required_f1_inflected_predicate_surfaces(
    pair_records: Iterable[Mapping[str, Any]],
    authorized_pair_ids: set[str],
) -> set[str]:
    surfaces: set[str] = set()
    for record in pair_records:
        if str(record.get("pair_id", "")) not in authorized_pair_ids:
            continue
        member = polarity_member(record)
        proof = member.get("grammar_rule_reproduction") or {}
        if isinstance(proof, Mapping) and proof.get("fact_predicate"):
            surfaces.add(str(proof["fact_predicate"]))
            continue
        span = str(proof.get("matched_surface_span", "")) if isinstance(proof, Mapping) else ""
        if span.lower().startswith("did not "):
            surfaces.add(span[8:])
    return surfaces


def generator_owned_predicate_surfaces(module: Any = generator) -> set[str]:
    surfaces: set[str] = set()
    for fact in getattr(module, "FACT_TEMPLATES", []):
        surfaces.add(str(fact["predicate"]))
        surfaces.add(str(fact["alternate_predicate"]))
    for predicate, alternate_predicate in getattr(module, "_GENERATED_PREDICATES", []):
        surfaces.add(str(predicate))
        surfaces.add(str(alternate_predicate))
    return surfaces


def validate_base_form_coverage(
    required_surfaces: Iterable[str],
    mapping_entries: Mapping[str, str] | Iterable[tuple[str, str]],
) -> dict[str, Any]:
    values_by_surface: dict[str, set[str]] = {}
    items = mapping_entries.items() if isinstance(mapping_entries, Mapping) else mapping_entries
    for inflected, base in items:
        values_by_surface.setdefault(str(inflected), set()).add(str(base))
    required = {str(surface) for surface in required_surfaces}
    ambiguous = sorted(
        surface for surface, values in values_by_surface.items()
        if len(values) != 1 and surface in required
    )
    missing = sorted(surface for surface in required if surface not in values_by_surface)
    covered = sorted(surface for surface in required if surface in values_by_surface and surface not in ambiguous)
    passed = not missing and not ambiguous
    return {
        "base_form_derivation_method": BASE_FORM_METHOD,
        "base_form_source_symbol": BASE_FORM_SYMBOL,
        "required_F1_inflected_predicate_surfaces": sorted(required),
        "missing_base_form_surfaces": missing,
        "ambiguous_base_form_surfaces": ambiguous,
        "covered_surfaces": covered,
        "coverage_pass": passed,
        "coverage_status": "PASS" if passed else BASE_FORM_COVERAGE_UNRESOLVED,
    }


def negative_polarity_flip_generated_ids(records: Iterable[Mapping[str, Any]]) -> set[str]:
    result: set[str] = set()
    for row in records:
        if row.get("intervention_type") == "polarity_flip" and re.search(r"\bdid\s+not\b", str(row.get("evidence", "")), re.IGNORECASE):
            result.add(str(row.get("id", "")))
    return result


def structural_negative_polarity_flip_row_ids_for_pair_count(pair_count: int) -> set[str]:
    templates = generator.fact_templates_for_count(pair_count)
    return set(generator._negative_polarity_flip_row_ids(templates))


def baseline_pair_count(baseline_rows: Sequence[Mapping[str, Any]]) -> int:
    pair_ids = [str(row.get("pair_id", "")) for row in baseline_rows]
    unique_pair_ids = {pair_id for pair_id in pair_ids if pair_id}
    require(len(unique_pair_ids) > 0, "baseline topology missing pair IDs")
    return len(unique_pair_ids)


def project_replay_to_baseline_topology(
    replay_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    replay_duplicates = set(duplicate_ids(replay_rows))
    baseline_ids = [str(row.get("id", "")) for row in baseline_rows]
    require(not duplicate_ids(baseline_rows), "baseline topology duplicate IDs")
    replay_by_id = row_map(replay_rows)
    missing = [row_id for row_id in baseline_ids if row_id not in replay_by_id]
    if missing or replay_duplicates:
        raise ValueError(PROVENANCE_IDENTITY_MISMATCH)
    return [dict(replay_by_id[row_id]) for row_id in baseline_ids]


def actual_repaired_generator_replay(
    baseline_rows: Sequence[Mapping[str, Any]],
    authorized_f1_row_ids: Iterable[str],
) -> dict[str, Any]:
    pair_count = baseline_pair_count(baseline_rows)
    authorized = sorted({str(row_id) for row_id in authorized_f1_row_ids})
    replay_rows, audit = generator.build_controlled_records_with_f1_polarity_repair_audit(
        pair_count,
        set(authorized),
    )
    projected_rows = project_replay_to_baseline_topology(replay_rows, baseline_rows)
    consumed = sorted(str(row_id) for row_id in audit.get("repair_consumed_row_ids", []))
    invocation_identity = {
        "pair_count": pair_count,
        "authorized_F1_row_ids_sha256": canonical_sha256(authorized),
        "repair_api": "build_controlled_records_with_f1_polarity_repair_audit",
        "baseline_id_sequence_sha256": id_sequence_sha256([str(row.get("id", "")) for row in baseline_rows]),
        "projection_policy": "baseline_id_sequence",
        "repair_mode": "f1_authorized_polarity_negative_only",
    }
    configuration_identity = {
        "generator_source_path": GENERATOR_SOURCE_PATH,
        "pair_count": pair_count,
        "authorized_F1_row_count": len(authorized),
        "structural_negative_polarity_flip_row_count": len(structural_negative_polarity_flip_row_ids_for_pair_count(pair_count)),
        "baseline_topology_row_count": len(baseline_rows),
        "baseline_id_sequence_sha256": invocation_identity["baseline_id_sequence_sha256"],
    }
    return {
        "pair_count": pair_count,
        "replayed_records": projected_rows,
        "actual_generator_repair_consumed_row_ids": consumed,
        "deterministic_generator_invocation": invocation_identity,
        "generator_configuration_identity": configuration_identity,
    }


def validate_repaired_output_replay_identity(
    baseline_rows: Sequence[Mapping[str, Any]],
    repaired_rows: Sequence[Mapping[str, Any]],
    authorized_f1_row_ids: Iterable[str],
) -> dict[str, Any]:
    try:
        replay = actual_repaired_generator_replay(baseline_rows, authorized_f1_row_ids)
    except Exception as exc:
        return {
            "generator_replay_identity_pass": False,
            "generator_replay_identity_status": PROVENANCE_IDENTITY_MISMATCH,
            "generator_replay_mismatch_row_ids": [],
            "generator_replay_error": str(exc),
            "replayed_records": [],
            "actual_generator_repair_consumed_row_ids": [],
            "deterministic_generator_invocation": {},
            "generator_configuration_identity": {},
        }
    replayed_records = replay["replayed_records"]
    authorized = sorted({str(row_id) for row_id in authorized_f1_row_ids})
    mismatch_ids = [
        str(expected.get("id", ""))
        for expected, observed in zip(replayed_records, repaired_rows)
        if dict(expected) != dict(observed)
    ]
    if len(replayed_records) != len(repaired_rows):
        mismatch_ids.append("__row_count__")
    consumed_mismatch = replay["actual_generator_repair_consumed_row_ids"] != authorized
    if consumed_mismatch:
        mismatch_ids.append("__repair_consumed_row_ids__")
    passed = not mismatch_ids
    return {
        "generator_replay_identity_pass": passed,
        "generator_replay_identity_status": "PASS" if passed else PROVENANCE_IDENTITY_MISMATCH,
        "generator_replay_mismatch_row_ids": sorted(set(mismatch_ids)),
        **replay,
    }


def validate_target_scope_membership(
    structural_generated_ids: Iterable[str],
    authorized_f1_row_ids: Iterable[str],
) -> dict[str, Any]:
    structural = sorted({str(row_id) for row_id in structural_generated_ids})
    authorized = sorted({str(row_id) for row_id in authorized_f1_row_ids})
    missing = sorted(set(authorized) - set(structural))
    non_authorized = sorted(set(structural) - set(authorized))
    passed = not missing
    return {
        "target_scope_membership_pass": passed,
        "target_scope_status": "PASS" if passed else TARGET_SCOPE_MEMBERSHIP_UNRESOLVED,
        "structural_negative_polarity_flip_row_ids": structural,
        "authorized_F1_row_ids": authorized,
        "non_authorized_structural_negative_polarity_flip_row_ids": non_authorized,
        "missing_authorized_structural_row_ids": missing,
    }


def span_record(text: str, span: str) -> dict[str, Any] | None:
    matches = list(re.finditer(re.escape(span), text))
    if len(matches) != 1:
        return None
    match = matches[0]
    return {"text": span, "start": match.start(), "end": match.end()}


def outside_span_identity(original: str, regenerated: str, original_span: dict[str, Any], regenerated_span: dict[str, Any]) -> bool:
    return (
        original[: original_span["start"]] == regenerated[: regenerated_span["start"]]
        and original[original_span["end"] :] == regenerated[regenerated_span["end"] :]
    )


def sidecar_grammar_status(sidecar: Mapping[str, Any] | None) -> str | None:
    return str(sidecar.get("grammar_status")) if isinstance(sidecar, Mapping) and "grammar_status" in sidecar else None


def sidecar_by_row_id(rows: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        row_id = str(row.get("row_id", ""))
        require(row_id, "sidecar row missing row_id")
        require(row_id not in result, "duplicate sidecar row_id")
        result[row_id] = row
    return result


def derive_stage185_dev_ids(source_rows: Sequence[Mapping[str, Any]]) -> set[str]:
    train_rows, dev_rows, dev_ids = stage185_builder.split_by_pair(
        [dict(row) for row in source_rows],
        P3W6F1_STAGE185_SPLIT_SEED,
        P3W6F1_STAGE185_DEV_RATIO,
    )
    if len(source_rows) == 3600:
        if len(train_rows) != P3W6F1_EXPECTED_TRAIN_ROWS or len(dev_rows) != P3W6F1_EXPECTED_DEV_ROWS:
            raise ValueError(STAGE185_PROVENANCE_UNRESOLVED)
    return {str(pair_id) for pair_id in dev_ids}


def load_authoritative_stage185_contracts(
    source_rows: Sequence[Mapping[str, Any]],
    *,
    repo_root: Path | None = None,
    contract_matrix_path: Path | None = None,
    git_tracked_checker: Any | None = None,
    runtime_authority: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    root = (repo_root or Path.cwd()).resolve()
    families = {str(row.get("intervention_type", "")) for row in source_rows}
    try:
        authority = runtime_authority or validate_stage185_runtime_authority_identity(
            repo_root=root,
            git_tracked_checker=git_tracked_checker,
        )
        if (
            not authority.get("stage185_runtime_authority_pass")
            or authority.get("stage184_contract_matrix_path") != P3W6F1_STAGE184_CONTRACT_MATRIX_PATH
        ):
            raise ValueError(STAGE185_PROVENANCE_UNRESOLVED)
        matrix = require_canonical_repo_file(
            root,
            contract_matrix_path or resolve_repo_path(root, P3W6F1_STAGE184_CONTRACT_MATRIX_PATH),
            P3W6F1_STAGE184_CONTRACT_MATRIX_PATH,
            git_tracked_checker=git_tracked_checker,
        )
        return stage185_builder.load_contracts(matrix, families)
    except Exception as exc:
        raise ValueError(STAGE185_PROVENANCE_UNRESOLVED) from exc


def generator_expected_rows_and_facts_for_source(
    source_rows: Sequence[Mapping[str, Any]],
    *,
    expected_generator_rows: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    pair_count = baseline_pair_count(source_rows)
    generated = expected_generator_rows if expected_generator_rows is not None else generator.build_controlled_records(pair_count)
    projected = project_replay_to_baseline_topology(generated, source_rows)
    facts = {
        str(fact["pair_id"]): fact
        for fact in generator.fact_templates_for_count(pair_count)
    }
    return {str(row["id"]): dict(row) for row in projected}, facts


def derive_stage185_expected_sidecar(
    source_rows: Sequence[Mapping[str, Any]],
    *,
    actual_source_dataset_sha256: str,
    actual_source_dataset_path: Path,
    actual_integrity_builder_sha256: str,
    expected_generator_rows: Sequence[Mapping[str, Any]] | None = None,
    repo_root: Path | None = None,
    contract_matrix_path: Path | None = None,
    git_tracked_checker: Any | None = None,
    runtime_authority: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    expected_rows, facts = generator_expected_rows_and_facts_for_source(
        source_rows,
        expected_generator_rows=expected_generator_rows,
    )
    root = repo_root or Path.cwd()
    try:
        source_dataset_path = actual_source_dataset_path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        source_dataset_path = str(actual_source_dataset_path.resolve()).replace("\\", "/")
    provenance = {
        "source_dataset_path": source_dataset_path,
        "source_dataset_sha256": actual_source_dataset_sha256,
        "generator_source_path": "/kaggle/working/ContraMamba/scripts/build_controlled_v5.py",
        "generator_source_sha256": "",
        "stage182a_report_sha256": "",
        "stage184a_report_sha256": "",
        "integrity_builder_sha256": actual_integrity_builder_sha256,
    }
    return stage185_builder.build_sidecar(
        [dict(row) for row in source_rows],
        load_authoritative_stage185_contracts(
            source_rows,
            repo_root=repo_root,
            contract_matrix_path=contract_matrix_path,
            git_tracked_checker=git_tracked_checker,
            runtime_authority=runtime_authority,
        ),
        expected_rows,
        facts,
        derive_stage185_dev_ids(source_rows),
        provenance,
        STAGE185_RULE_VERSION,
        "",
    )


def validate_stage185_sidecar_provenance(
    source_rows: Sequence[Mapping[str, Any]],
    sidecar_rows: Sequence[Mapping[str, Any]],
    *,
    actual_source_dataset_sha256: str,
    actual_source_dataset_path: Path,
    actual_integrity_builder_sha256: str,
    expected_generator_rows: Sequence[Mapping[str, Any]] | None = None,
    repo_root: Path | None = None,
    runtime_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    expected_semantic_by_id: dict[str, Mapping[str, Any]] = {}
    if expected_generator_rows is None:
        failures.append("stage185_expected_generator_unavailable")
    else:
        try:
            expected_sidecar = derive_stage185_expected_sidecar(
                source_rows,
                actual_source_dataset_sha256=actual_source_dataset_sha256,
                actual_source_dataset_path=actual_source_dataset_path,
                actual_integrity_builder_sha256=actual_integrity_builder_sha256,
                expected_generator_rows=expected_generator_rows,
                repo_root=repo_root,
                runtime_authority=runtime_authority,
            )
            expected_semantic_by_id = sidecar_by_row_id(expected_sidecar)
        except Exception:
            failures.append("stage185_semantic_derivation")
    if len(source_rows) != len(sidecar_rows):
        failures.append("source_sidecar_row_count")
    source_ids = [str(row.get("id", "")) for row in source_rows]
    sidecar_ids = [str(row.get("row_id", "")) for row in sidecar_rows]
    if source_ids != sidecar_ids:
        failures.append("source_sidecar_row_id_sequence")
    source_by_id: dict[str, Mapping[str, Any]] = {}
    for row in source_rows:
        row_id = str(row.get("id", ""))
        if not row_id or row_id in source_by_id:
            failures.append("source_row_id_identity")
        source_by_id[row_id] = row
    canonical_by_pair = {
        str(row.get("pair_id", "")): str(row.get("id", ""))
        for row in source_rows
        if row.get("intervention_type") == "none"
    }
    for index, sidecar in enumerate(sidecar_rows):
        source = source_rows[index] if index < len(source_rows) else source_by_id.get(str(sidecar.get("row_id", "")))
        missing_schema = [field for field in REQUIRED_STAGE185_SCHEMA_FIELDS if field not in sidecar]
        if missing_schema:
            failures.append("required_stage185_schema_fields")
        if not isinstance(source, Mapping):
            failures.append("source_order_join")
            continue
        source_row_id = str(source.get("id", ""))
        source_pair_id = str(source.get("pair_id", ""))
        if sidecar.get("row_id") != source_row_id:
            failures.append("sidecar_row_id")
        if sidecar.get("pair_id") != source_pair_id:
            failures.append("sidecar_pair_id")
        if sidecar.get("intervention_type") != source.get("intervention_type"):
            failures.append("sidecar_intervention_type")
        if sidecar.get("source_dataset_sha256") != actual_source_dataset_sha256:
            failures.append("source_dataset_sha256")
        if not sidecar_source_path_matches(sidecar.get("source_dataset_path"), actual_source_dataset_path, repo_root):
            failures.append("source_dataset_path")
        if sidecar.get("integrity_builder_sha256") != actual_integrity_builder_sha256:
            failures.append("integrity_builder_sha256")
        canonical_row_id = str(sidecar.get("canonical_row_id", ""))
        canonical_source = source_by_id.get(canonical_row_id)
        if (
            not canonical_source
            or canonical_source.get("pair_id") != source_pair_id
            or canonical_source.get("intervention_type") != "none"
            or canonical_by_pair.get(source_pair_id) != canonical_row_id
        ):
            failures.append("canonical_row_id_lineage")
        expected_semantic = expected_semantic_by_id.get(source_row_id)
        if expected_semantic is None:
            failures.append("stage185_semantic_identity")
        else:
            for field in STAGE185_SEMANTIC_IDENTITY_FIELDS:
                if sidecar.get(field) != expected_semantic.get(field):
                    failures.append("stage185_semantic_identity")
                    break
    failures = sorted(set(failures))
    passed = not failures
    return {
        "stage185_provenance_pass": passed,
        "stage185_provenance_status": "PASS" if passed else STAGE185_PROVENANCE_UNRESOLVED,
        "stage185_provenance_failures": failures,
        "source_row_count": len(source_rows),
        "sidecar_row_count": len(sidecar_rows),
        "source_row_id_sequence": source_ids,
        "sidecar_row_id_sequence": sidecar_ids,
        "actual_source_dataset_sha256": actual_source_dataset_sha256,
        "actual_source_dataset_path": str(actual_source_dataset_path),
        "actual_integrity_builder_sha256": actual_integrity_builder_sha256,
    }


def validate_stage185_runtime_authority_identity(
    *,
    repo_root: Path,
    baseline_integrity_builder_sha256: str | None = None,
    git_tracked_checker: Any | None = None,
    git_blob_reader: Any | None = None,
) -> dict[str, Any]:
    try:
        builder = trusted_blob_identity(
            repo_root=repo_root,
            relative_path=P3W6F1_STAGE185_BUILDER_PATH,
            git_tracked_checker=git_tracked_checker,
            git_blob_reader=git_blob_reader,
        )
        stage182 = trusted_blob_identity(
            repo_root=repo_root,
            relative_path=P3W6F1_STAGE182_ANALYZER_PATH,
            git_tracked_checker=git_tracked_checker,
            git_blob_reader=git_blob_reader,
        )
        matrix = trusted_blob_identity(
            repo_root=repo_root,
            relative_path=P3W6F1_STAGE184_CONTRACT_MATRIX_PATH,
            git_tracked_checker=git_tracked_checker,
            git_blob_reader=git_blob_reader,
        )
        grammar_validator = p3w4_authority.load_production_grammar_validator(repo_root)
        resolved_path = str(grammar_validator["validator_source_path"])
        validator_blob = trusted_blob_identity(
            repo_root=repo_root,
            relative_path=resolved_path,
            git_tracked_checker=git_tracked_checker,
            git_blob_reader=git_blob_reader,
        )
        if baseline_integrity_builder_sha256 is not None and builder["source_sha256"] != baseline_integrity_builder_sha256:
            raise ValueError(STAGE185_PROVENANCE_UNRESOLVED)
    except Exception as exc:
        raise ValueError(STAGE185_PROVENANCE_UNRESOLVED) from exc
    return {
        "stage185_runtime_authority_pass": True,
        "stage185_runtime_authority_status": "PASS",
        "resolved_grammar_validator_source_path": resolved_path,
        "resolved_grammar_validator_source_sha256": validator_blob["source_sha256"],
        "stage185_integrity_builder_source_path": P3W6F1_STAGE185_BUILDER_PATH,
        "stage185_integrity_builder_source_sha256": builder["source_sha256"],
        "stage182_analyzer_source_path": P3W6F1_STAGE182_ANALYZER_PATH,
        "stage182_analyzer_source_sha256": stage182["source_sha256"],
        "stage184_contract_matrix_path": P3W6F1_STAGE184_CONTRACT_MATRIX_PATH,
        "stage184_contract_matrix_sha256": matrix["source_sha256"],
        "grammar_validator_definition_kind": grammar_validator.get("validator_definition_kind"),
    }


def stage185_transition(
    before: Mapping[str, Any] | None,
    after: Mapping[str, Any] | None,
) -> dict[str, Any]:
    failures: list[str] = []
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        failures.append("STAGE185_SIDECAR_MISSING")
    else:
        for field, expected in REQUIRED_STAGE185_BEFORE.items():
            if before.get(field) != expected:
                failures.append(f"before:{field}")
        for field, expected in REQUIRED_STAGE185_AFTER.items():
            if after.get(field) != expected:
                failures.append(f"after:{field}")
    passed = not failures
    return {
        "stage185_transition_status": "PASS" if passed else STAGE185_TRANSITION_FAILED,
        "stage185_transition_pass": passed,
        "F1_integrity_transition": "INELIGIBLE_TO_ELIGIBLE" if passed else "UNRESOLVED",
        "stage185_transition_failures": sorted(failures),
    }


def sidecar_value(row: Mapping[str, Any] | None, field: str) -> Any:
    return row.get(field) if isinstance(row, Mapping) else None


def semantic_audit_record(
    original: Mapping[str, Any],
    regenerated: Mapping[str, Any],
    canonical: Mapping[str, Any],
    *,
    sidecar_before: Mapping[str, Any] | None = None,
    sidecar_after: Mapping[str, Any] | None = None,
    fact_identity: Mapping[str, Any] | None = None,
    inflected_predicate_surface: str | None = None,
    expected_base_predicate: str | None = None,
    generator_source_sha256: str = "",
    generator_commit: str = "",
    grammar_validator_source: str = GRAMMAR_VALIDATOR_SOURCE,
    grammar_validator_sha256: str = "",
) -> dict[str, Any]:
    original_text = str(original.get("evidence", ""))
    regenerated_text = str(regenerated.get("evidence", ""))
    inflected = inflected_predicate_surface or str((fact_identity or {}).get("predicate", ""))
    expected_base = expected_base_predicate or generator._BASE_PREDICATE_BY_INFLECTED.get(inflected)
    original_span_text = f"did not {inflected}" if inflected else ""
    regenerated_span_text = f"did not {expected_base}" if expected_base else ""
    original_span = span_record(original_text, original_span_text) if original_span_text else None
    regenerated_span = span_record(regenerated_text, regenerated_span_text) if regenerated_span_text else None
    transition = stage185_transition(sidecar_before, sidecar_after)
    rejection_codes: list[str] = []
    status = PASS_STATUS
    polarity_preserved: bool | None = True
    accepted = True

    protected_fields = set(original) | set(regenerated)
    protected_fields.discard("evidence")
    protected_changes = sorted(field for field in protected_fields if original.get(field) != regenerated.get(field))
    if protected_changes:
        status = REJECTED_STATUS
        polarity_preserved = False
        accepted = False
        rejection_codes.append("PROTECTED_FIELD_CHANGED")
    if original.get("id") != regenerated.get("id"):
        status = REJECTED_STATUS
        polarity_preserved = False
        accepted = False
        rejection_codes.append("ROW_IDENTITY_CHANGED")
    if original.get("claim") != regenerated.get("claim"):
        status = REJECTED_STATUS
        polarity_preserved = False
        accepted = False
        rejection_codes.append("CLAIM_CHANGED")
    if original.get("final_label") != regenerated.get("final_label") or regenerated.get("final_label") != "REFUTE":
        status = REJECTED_STATUS
        polarity_preserved = False
        accepted = False
        rejection_codes.append("LABEL_IDENTITY_CHANGED")
    if regenerated.get("polarity_label") != "REFUTE":
        status = REJECTED_STATUS
        polarity_preserved = False
        accepted = False
        rejection_codes.append("POLARITY_LABEL_CONTRADICTION")

    span_identity = False
    if status != REJECTED_STATUS:
        if not expected_base or original_span is None or regenerated_span is None:
            status = MANUAL_STATUS
            polarity_preserved = None
            accepted = False
            rejection_codes.append(SEMANTIC_AUTHORITY_UNRESOLVED)
        else:
            span_identity = outside_span_identity(original_text, regenerated_text, original_span, regenerated_span)
            if not span_identity:
                status = REJECTED_STATUS
                polarity_preserved = False
                accepted = False
                rejection_codes.append("OUTSIDE_SPAN_CHANGED")
            if token_count(regenerated_text, "did") != 1 or token_count(regenerated_text, "not") != 1:
                status = REJECTED_STATUS
                polarity_preserved = False
                accepted = False
                rejection_codes.append("DID_NOT_COUNT_CONTRACT_FAILED")

    if not transition["stage185_transition_pass"]:
        if status == PASS_STATUS:
            status = MANUAL_STATUS
            polarity_preserved = None
        accepted = False
        rejection_codes.append(STAGE185_TRANSITION_FAILED)
    if status == MANUAL_STATUS and SEMANTIC_AUTHORITY_UNRESOLVED not in rejection_codes:
        rejection_codes.append(SEMANTIC_AUTHORITY_UNRESOLVED)

    normalized_changed_span = {
        "from": original_span_text,
        "to": regenerated_span_text,
    }
    if expected_base:
        predicate_inflection_changes: list[dict[str, str]] = [{"from": inflected, "to": expected_base}]
    else:
        predicate_inflection_changes = []

    return {
        "pair_id": str(original.get("pair_id", "")),
        "original_row_id": str(original.get("id", "")),
        "regenerated_row_id": str(regenerated.get("id", "")),
        "intervention_type": str(original.get("intervention_type", "")),
        "original_text": original_text,
        "regenerated_text": regenerated_text,
        "original_final_label": original.get("final_label"),
        "regenerated_final_label": regenerated.get("final_label"),
        "canonical_row_id": str(canonical.get("id", "")),
        "generator_source_path": GENERATOR_SOURCE_PATH,
        "generator_source_sha256": generator_source_sha256,
        "generator_commit": generator_commit,
        "fact_identity": dict(fact_identity or {}),
        "grammar_validator_source": grammar_validator_source,
        "grammar_validator_sha256": grammar_validator_sha256,
        "grammar_before": sidecar_value(sidecar_before, "grammar_status"),
        "grammar_after": sidecar_value(sidecar_after, "grammar_status"),
        "sidecar_before": dict(sidecar_before or {}),
        "sidecar_after": dict(sidecar_after or {}),
        "lineage_preserved": original.get("id") == regenerated.get("id") == f"{original.get('pair_id')}__polarity_flip",
        "semantic_validation_status": status,
        "semantic_polarity_preserved": polarity_preserved,
        "candidate_accepted": accepted,
        "ordered_rejection_codes": ordered_blockers(dict.fromkeys(rejection_codes)),
        "inflected_predicate_surface": inflected,
        "expected_base_predicate": expected_base,
        "base_form_derivation_method": BASE_FORM_METHOD,
        "base_form_derivation_source_path": GENERATOR_SOURCE_PATH,
        "base_form_derivation_source_sha256": generator_source_sha256,
        "base_form_source_symbol": BASE_FORM_SYMBOL,
        "authorized_replacement_span": regenerated_span,
        "outside_span_byte_identity": span_identity,
        "canonical_text": str(canonical.get("evidence", "")),
        "original_defective_text": original_text,
        "normalized_changed_span": normalized_changed_span,
        "negation_markers_added": [],
        "negation_markers_removed": [],
        "auxiliary_verb_changes": {"from": [], "to": []},
        "predicate_inflection_changes": predicate_inflection_changes,
        "duplicate_or_missing_tokens": {
            "did": token_count(regenerated_text, "did"),
            "not": token_count(regenerated_text, "not"),
        },
        "semantic_validation_method": "stage185_contract_audit",
        "semantic_validation_evidence": {
            "original_authorized_span": original_span,
            "regenerated_authorized_span": regenerated_span,
            "claim_identity": original.get("claim") == regenerated.get("claim"),
            "label_identity": original.get("final_label") == regenerated.get("final_label"),
            "protected_field_changes": protected_changes,
            **transition,
        },
        "baseline_source_row": dict(original),
        "regenerated_source_row": dict(regenerated),
    }


def row_map(rows: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        row_id = str(row.get("id", ""))
        require(row_id, "row missing id")
        require(row_id not in result, "duplicate id")
        result[row_id] = row
    return result


def duplicate_ids(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    counts = Counter(str(row.get("id", "")) for row in rows)
    return sorted(row_id for row_id, count in counts.items() if count > 1)


def complete_output_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return canonical_sha256([dict(row) for row in rows])


def id_sequence_sha256(ids: Sequence[str]) -> str:
    return canonical_sha256(list(ids))


def changed_fields(a: Mapping[str, Any], b: Mapping[str, Any]) -> list[str]:
    return sorted(field for field in (set(a) | set(b)) if a.get(field) != b.get(field))


def full_output_isolation(
    baseline_rows: Sequence[Mapping[str, Any]],
    repaired_rows: Sequence[Mapping[str, Any]],
    *,
    authorized_f1_row_ids: Iterable[str],
    structural_negative_polarity_flip_row_ids: Iterable[str] = (),
    repair_consumed_row_ids: Iterable[str] = (),
    f2_row_ids: Iterable[str] = (),
    baseline_generator_commit: str = "",
    baseline_generator_source_path: str = GENERATOR_SOURCE_PATH,
    baseline_generator_source_sha256: str = "",
    repaired_generator_commit: str = "",
    repaired_generator_source_path: str = GENERATOR_SOURCE_PATH,
    repaired_generator_source_sha256: str = "",
    deterministic_generator_invocation: Mapping[str, Any] | None = None,
    generator_configuration_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    baseline_by_id = row_map(baseline_rows)
    repaired_by_id = row_map(repaired_rows)
    baseline_ids = [str(row.get("id", "")) for row in baseline_rows]
    repaired_ids = [str(row.get("id", "")) for row in repaired_rows]
    authorized = sorted({str(row_id) for row_id in authorized_f1_row_ids})
    structural_negative = sorted({str(row_id) for row_id in structural_negative_polarity_flip_row_ids})
    repair_consumed = sorted({str(row_id) for row_id in repair_consumed_row_ids})
    non_authorized_structural = sorted(set(structural_negative) - set(authorized))
    f2_ids = {str(row_id) for row_id in f2_row_ids}
    missing = sorted(set(baseline_by_id) - set(repaired_by_id))
    added = sorted(set(repaired_by_id) - set(baseline_by_id))
    changed = sorted(row_id for row_id in set(baseline_by_id) & set(repaired_by_id) if dict(baseline_by_id[row_id]) != dict(repaired_by_id[row_id]))
    evidence_changed = sorted(row_id for row_id in changed if baseline_by_id[row_id].get("evidence") != repaired_by_id[row_id].get("evidence"))
    claim_changed = sorted(row_id for row_id in changed if baseline_by_id[row_id].get("claim") != repaired_by_id[row_id].get("claim"))
    non_text_changed = sorted(
        row_id
        for row_id in changed
        if any(field not in {"claim", "evidence"} for field in changed_fields(baseline_by_id[row_id], repaired_by_id[row_id]))
    )
    canonical_changed = sorted(row_id for row_id in changed if baseline_by_id[row_id].get("intervention_type") == "none")
    paraphrase_changed = sorted(row_id for row_id in changed if baseline_by_id[row_id].get("intervention_type") == "paraphrase")
    unauthorized_changed = sorted(set(changed) - set(authorized))
    non_authorized_structural_changed = sorted(set(changed) & set(non_authorized_structural))
    return {
        "changed_ids": changed,
        "structural_negative_polarity_flip_row_ids": structural_negative,
        "authorized_F1_row_ids": authorized,
        "non_authorized_structural_negative_polarity_flip_row_ids": non_authorized_structural,
        "repair_consumed_row_ids": repair_consumed,
        "non_authorized_structural_negative_polarity_flip_changed_row_ids": non_authorized_structural_changed,
        "missing_ids": missing,
        "added_ids": added,
        "duplicate_ids": sorted(set(duplicate_ids(baseline_rows)) | set(duplicate_ids(repaired_rows))),
        "unauthorized_changed_row_ids": unauthorized_changed,
        "F2_changed_row_ids": sorted(set(changed) & f2_ids),
        "unaffected_changed_row_ids": sorted(set(changed) - set(authorized) - f2_ids),
        "canonical_changed_row_ids": canonical_changed,
        "paraphrase_changed_row_ids": paraphrase_changed,
        "evidence_changed_row_ids": evidence_changed,
        "claim_changed_row_ids": claim_changed,
        "non_text_field_changed_row_ids": non_text_changed,
        "baseline_generator_commit": baseline_generator_commit,
        "baseline_generator_source_path": baseline_generator_source_path,
        "baseline_generator_source_sha256": baseline_generator_source_sha256,
        "repaired_generator_commit": repaired_generator_commit,
        "repaired_generator_source_path": repaired_generator_source_path,
        "repaired_generator_source_sha256": repaired_generator_source_sha256,
        "deterministic_generator_invocation": dict(deterministic_generator_invocation or {}),
        "generator_configuration_identity": dict(generator_configuration_identity or {}),
        "baseline_complete_output_sha256": complete_output_sha256(baseline_rows),
        "repaired_complete_output_sha256": complete_output_sha256(repaired_rows),
        "baseline_row_count": len(baseline_rows),
        "repaired_row_count": len(repaired_rows),
        "baseline_id_sequence": baseline_ids,
        "repaired_id_sequence": repaired_ids,
        "baseline_id_sequence_sha256": id_sequence_sha256(baseline_ids),
        "repaired_id_sequence_sha256": id_sequence_sha256(repaired_ids),
        "row_order_changed": baseline_ids != repaired_ids,
    }


def validate_full_output_isolation(isolation: Mapping[str, Any]) -> dict[str, Any]:
    authorized = sorted(isolation.get("authorized_F1_row_ids", []))
    failures: list[str] = []
    expected_arrays = {
        "changed_ids": authorized,
        "repair_consumed_row_ids": authorized,
        "non_authorized_structural_negative_polarity_flip_changed_row_ids": [],
        "evidence_changed_row_ids": authorized,
        "missing_ids": [],
        "added_ids": [],
        "duplicate_ids": [],
        "unauthorized_changed_row_ids": [],
        "F2_changed_row_ids": [],
        "unaffected_changed_row_ids": [],
        "canonical_changed_row_ids": [],
        "paraphrase_changed_row_ids": [],
        "claim_changed_row_ids": [],
        "non_text_field_changed_row_ids": [],
    }
    for field, expected in expected_arrays.items():
        if sorted(isolation.get(field, [])) != expected:
            failures.append(field)
    if isolation.get("baseline_id_sequence") != isolation.get("repaired_id_sequence"):
        failures.append("baseline_id_sequence")
    if isolation.get("baseline_id_sequence_sha256") != isolation.get("repaired_id_sequence_sha256"):
        failures.append("baseline_id_sequence_sha256")
    if isolation.get("row_order_changed") is not False:
        failures.append("row_order_changed")
    passed = not failures
    return {
        "full_output_isolation_pass": passed,
        "full_output_isolation_status": "PASS" if passed else FULL_OUTPUT_ISOLATION_FAILED,
        "full_output_isolation_failures": sorted(failures),
    }


def git_blob_bytes(repo_root: Path, commit: str, source_path: str) -> bytes | None:
    try:
        subprocess.run(
            ["git", "-C", str(repo_root), "cat-file", "-e", f"{commit}^{{commit}}"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "show", f"{commit}:{source_path}"],
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def current_git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def validate_execution_provenance_identity(
    provenance: Mapping[str, Any],
    *,
    baseline_jsonl_path: Path | None = None,
    repaired_jsonl_path: Path | None = None,
    actual_deterministic_generator_invocation: Mapping[str, Any] | None = None,
    actual_generator_configuration_identity: Mapping[str, Any] | None = None,
    input_authority: Mapping[str, Any] | None = None,
    repo_root: Path | None = None,
    git_blob_reader: Any | None = None,
    current_source_reader: Any | None = None,
    current_commit_resolver: Any | None = None,
) -> dict[str, Any]:
    required = (
        "baseline_generator_commit",
        "baseline_generator_source_path",
        "baseline_generator_source_sha256",
        "repaired_generator_commit",
        "repaired_generator_source_path",
        "repaired_generator_source_sha256",
        "deterministic_generator_invocation",
        "generator_configuration_identity",
        "F1_input_sha256",
        "F1_execution_commit",
        "F1_output_sha256",
    )
    missing = [field for field in required if provenance.get(field) in (None, "", {}, [])]
    mismatches: list[str] = []
    unresolved: list[str] = []
    if not missing:
        if provenance.get("baseline_generator_source_path") != GENERATOR_SOURCE_PATH:
            mismatches.append("baseline_generator_source_path")
        if input_authority is not None and provenance.get("baseline_generator_source_sha256") != input_authority.get("baseline_generator_source_sha256"):
            mismatches.append("baseline_generator_source_sha256")
        if provenance.get("repaired_generator_source_path") != GENERATOR_SOURCE_PATH:
            mismatches.append("repaired_generator_source_path")
        if actual_deterministic_generator_invocation is not None and dict(provenance.get("deterministic_generator_invocation") or {}) != dict(actual_deterministic_generator_invocation):
            mismatches.append("deterministic_generator_invocation")
        if actual_generator_configuration_identity is not None and dict(provenance.get("generator_configuration_identity") or {}) != dict(actual_generator_configuration_identity):
            mismatches.append("generator_configuration_identity")
        if provenance.get("repaired_generator_commit") != provenance.get("F1_execution_commit"):
            mismatches.append("repaired_generator_commit")
        if baseline_jsonl_path is not None:
            if provenance.get("F1_input_sha256") != file_sha256(baseline_jsonl_path):
                mismatches.append("F1_input_sha256")
        if repaired_jsonl_path is not None:
            if provenance.get("F1_output_sha256") != file_sha256(repaired_jsonl_path):
                mismatches.append("F1_output_sha256")
        root = repo_root or Path.cwd()
        blob_reader = git_blob_reader or (lambda commit, path: git_blob_bytes(root, commit, path))
        current_reader = current_source_reader or (lambda path: (root / path).read_bytes())
        commit_resolver = current_commit_resolver or (lambda: current_git_commit(root))
        for prefix in ("baseline", "repaired"):
            commit = str(provenance[f"{prefix}_generator_commit"])
            source_path = str(provenance[f"{prefix}_generator_source_path"])
            expected_sha = str(provenance[f"{prefix}_generator_source_sha256"])
            blob = blob_reader(commit, source_path)
            if blob is None:
                unresolved.append(f"{prefix}_generator_commit")
                continue
            if sha256_bytes(blob) != expected_sha:
                mismatches.append(f"{prefix}_generator_source_sha256")
            if prefix == "repaired":
                try:
                    current_bytes = current_reader(source_path)
                except OSError:
                    unresolved.append("repaired_generator_source_path")
                else:
                    if sha256_bytes(current_bytes) != expected_sha:
                        mismatches.append("repaired_generator_actual_source_sha256")
        current_commit = commit_resolver()
        if current_commit is None:
            unresolved.append("F1_execution_commit")
        elif str(provenance["F1_execution_commit"]) != str(current_commit):
            mismatches.append("F1_execution_commit")
        else:
            execution_blob = blob_reader(str(provenance["F1_execution_commit"]), GENERATOR_SOURCE_PATH)
            if execution_blob is None:
                unresolved.append("F1_execution_commit_generator_blob")
            else:
                try:
                    current_bytes = current_reader(GENERATOR_SOURCE_PATH)
                except OSError:
                    unresolved.append("current_generator_source_path")
                else:
                    expected_repaired_sha = str(provenance["repaired_generator_source_sha256"])
                    if sha256_bytes(execution_blob) != expected_repaired_sha:
                        mismatches.append("execution_commit_generator_source_sha256")
                    if sha256_bytes(current_bytes) != sha256_bytes(execution_blob):
                        mismatches.append("current_generator_bytes")
    passed = not missing and not unresolved and not mismatches
    status = "PASS"
    if mismatches:
        status = PROVENANCE_IDENTITY_MISMATCH
    elif missing or unresolved:
        status = PROVENANCE_UNRESOLVED
    return {
        "execution_provenance_pass": passed,
        "execution_provenance_status": status,
        "missing_execution_provenance_fields": missing,
        "unresolved_execution_provenance_fields": sorted(set(unresolved)),
        "provenance_identity_mismatches": sorted(set(mismatches)),
    }


def all_stage185_transitions_pass(audit_rows: Iterable[Mapping[str, Any]]) -> bool:
    return all(
        (row.get("semantic_validation_evidence") or {}).get("stage185_transition_pass") is True
        for row in audit_rows
    )

def pair_ids_for_status(audit_rows: Iterable[Mapping[str, Any]], status: str) -> list[str]:
    return sorted({str(row.get("pair_id", "")) for row in audit_rows if row.get("semantic_validation_status") == status})


def accepted_candidate_pair_ids(audit_rows: Iterable[Mapping[str, Any]]) -> list[str]:
    return sorted({
        str(row.get("pair_id", ""))
        for row in audit_rows
        if row.get("candidate_accepted") is True
        and row.get("semantic_validation_status") == PASS_STATUS
        and row.get("semantic_polarity_preserved") is True
    })


def final_acceptance_blockers(
    *,
    full_output_isolation_validation: Mapping[str, Any],
    stage185_provenance_validation: Mapping[str, Any],
    execution_provenance_validation: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if full_output_isolation_validation.get("full_output_isolation_pass") is not True:
        blockers.append(FULL_OUTPUT_ISOLATION_FAILED)
    if stage185_provenance_validation.get("stage185_provenance_pass") is not True:
        blockers.append(STAGE185_PROVENANCE_UNRESOLVED)
    if execution_provenance_validation.get("execution_provenance_pass") is not True:
        status = execution_provenance_validation.get("execution_provenance_status")
        blockers.append(PROVENANCE_IDENTITY_MISMATCH if status == PROVENANCE_IDENTITY_MISMATCH else PROVENANCE_UNRESOLVED)
    return ordered_blockers(blockers)


def finalize_candidate_acceptance(
    audit_rows: Iterable[Mapping[str, Any]],
    *,
    full_output_isolation_validation: Mapping[str, Any],
    stage185_provenance_validation: Mapping[str, Any],
    execution_provenance_validation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    global_blockers = final_acceptance_blockers(
        full_output_isolation_validation=full_output_isolation_validation,
        stage185_provenance_validation=stage185_provenance_validation,
        execution_provenance_validation=execution_provenance_validation,
    )
    finalized: list[dict[str, Any]] = []
    for row in audit_rows:
        candidate = dict(row)
        candidate_claims_pass = (
            candidate.get("semantic_validation_status") == PASS_STATUS
            and candidate.get("semantic_polarity_preserved") is True
            and candidate.get("candidate_accepted") is True
        )
        local_pass = (
            candidate_claims_pass
            and (candidate.get("semantic_validation_evidence") or {}).get("stage185_transition_pass") is True
        )
        status_only_pass = candidate.get("semantic_validation_status") == PASS_STATUS and not candidate_claims_pass
        if status_only_pass or (candidate_claims_pass and (not local_pass or global_blockers)):
            rejection_codes = list(candidate.get("ordered_rejection_codes") or [])
            rejection_codes.append(SEMANTIC_AUTHORITY_UNRESOLVED)
            if candidate_claims_pass and not local_pass:
                rejection_codes.append(STAGE185_TRANSITION_FAILED)
            rejection_codes.extend(global_blockers)
            candidate["semantic_validation_status"] = MANUAL_STATUS
            candidate["semantic_polarity_preserved"] = None
            candidate["candidate_accepted"] = False
            candidate["ordered_rejection_codes"] = ordered_blockers(dict.fromkeys(rejection_codes))
        finalized.append(candidate)
    return finalized


def audit_authorized_candidates(
    baseline_rows: Sequence[Mapping[str, Any]],
    repaired_rows: Sequence[Mapping[str, Any]],
    pair_records: Iterable[Mapping[str, Any]],
    authorized_f1_row_ids: Iterable[str],
    baseline_sidecar_by_id: Mapping[str, Mapping[str, Any]],
    repaired_sidecar_by_id: Mapping[str, Mapping[str, Any]],
    *,
    generator_source_sha256: str = "",
    generator_commit: str = "",
    grammar_validator_source: str = GRAMMAR_VALIDATOR_SOURCE,
    grammar_validator_sha256: str = "",
) -> list[dict[str, Any]]:
    baseline_by_id = row_map(baseline_rows)
    repaired_by_id = row_map(repaired_rows)
    authorized = {str(row_id) for row_id in authorized_f1_row_ids}
    records_by_pair = {str(record.get("pair_id", "")): record for record in pair_records}
    audit_rows: list[dict[str, Any]] = []
    for row_id in sorted(authorized & set(repaired_by_id)):
        pair_id = row_id.rsplit("__", 1)[0]
        authority = records_by_pair[pair_id]
        member = polarity_member(authority)
        canonical_member = authority.get("members", {}).get("canonical") if isinstance(authority.get("members"), Mapping) else None
        require(isinstance(canonical_member, Mapping), "authority record missing canonical member")
        canonical = canonical_member.get("source_row")
        require(isinstance(canonical, Mapping), "canonical member missing source_row")
        proof = member.get("grammar_rule_reproduction") or {}
        require(isinstance(proof, Mapping), "polarity member missing grammar proof")
        inflected = str(proof.get("fact_predicate", ""))
        expected_base = generator._BASE_PREDICATE_BY_INFLECTED.get(inflected)
        audit_rows.append(
            semantic_audit_record(
                baseline_by_id[row_id],
                repaired_by_id[row_id],
                canonical,
                sidecar_before=baseline_sidecar_by_id.get(row_id),
                sidecar_after=repaired_sidecar_by_id.get(row_id),
                fact_identity={
                    "pair_id": pair_id,
                    "predicate": inflected,
                    "alternate_predicate": str(proof.get("fact_alternate_predicate", "")),
                },
                inflected_predicate_surface=inflected,
                expected_base_predicate=expected_base,
                generator_source_sha256=generator_source_sha256,
                generator_commit=generator_commit,
                grammar_validator_source=grammar_validator_source,
                grammar_validator_sha256=grammar_validator_sha256,
            )
        )
    return audit_rows

def validate_summary_accounting(summary: Mapping[str, Any]) -> None:
    for count_field, array_field in SUMMARY_PAIR_FIELDS:
        values = summary.get(array_field)
        require(isinstance(values, list), f"summary field is not an array: {array_field}")
        require(summary.get(count_field) == len(values), f"summary count mismatch: {count_field}")
    accepted = set(summary.get("F1_accepted_candidate_pair_ids", []))
    manual = set(summary.get("F1_manual_review_required_pair_ids", []))
    rejected = set(summary.get("F1_rejected_candidate_pair_ids", []))
    require(not (accepted & manual), "accepted/manual partitions overlap")
    require(not (accepted & rejected), "accepted/rejected partitions overlap")
    require(not (manual & rejected), "manual/rejected partitions overlap")
    generated_authorized = set(summary.get("F1_generated_candidate_pair_ids", [])) & set(summary.get("F1_target_pair_ids", []))
    require(accepted | manual | rejected == generated_authorized, "candidate partition union mismatch")


def derive_execution_decision(summary: Mapping[str, Any]) -> str:
    all_accepted = (
        summary.get("F1_execution_blockers") == []
        and summary.get("F1_target_pair_count") == EXPECTED_F1_TARGET_COUNT
        and summary.get("F1_generated_candidate_count") == EXPECTED_F1_TARGET_COUNT
        and summary.get("F1_accepted_candidate_count") == EXPECTED_F1_TARGET_COUNT
        and summary.get("F1_manual_review_required_count") == 0
        and summary.get("F1_rejected_candidate_count") == 0
        and summary.get("F1_missing_candidate_count") == 0
        and summary.get("F1_unauthorized_candidate_count") == 0
        and (summary.get("authority_cardinality") or {}).get("authority_cardinality_pass") is True
        and (summary.get("target_scope_membership") or {}).get("target_scope_membership_pass") is True
        and (summary.get("base_form_coverage") or {}).get("coverage_pass") is True
        and (summary.get("stage185_provenance_validation") or {}).get("stage185_provenance_pass") is True
        and (summary.get("full_output_isolation_validation") or {}).get("full_output_isolation_pass") is True
        and (summary.get("execution_provenance_validation") or {}).get("execution_provenance_pass") is True
    )
    return ALL_ACCEPTED_DECISION if all_accepted else BLOCKERS_DECISION

def build_summary(
    authorized_target_pair_ids: Iterable[str],
    generated_candidate_pair_ids: Iterable[str],
    audit_rows: Iterable[Mapping[str, Any]],
    *,
    authority_cardinality: Mapping[str, Any] | None = None,
    target_scope: Mapping[str, Any] | None = None,
    base_form_coverage: Mapping[str, Any] | None = None,
    stage185_provenance_validation: Mapping[str, Any] | None = None,
    full_output_validation: Mapping[str, Any] | None = None,
    provenance_validation: Mapping[str, Any] | None = None,
    execution_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    audit_materialized = list(audit_rows)
    target = sorted({str(pair_id) for pair_id in authorized_target_pair_ids})
    generated = sorted({str(pair_id) for pair_id in generated_candidate_pair_ids})
    accepted = accepted_candidate_pair_ids(audit_materialized)
    manual = pair_ids_for_status(audit_materialized, MANUAL_STATUS)
    rejected = pair_ids_for_status(audit_materialized, REJECTED_STATUS)
    missing = sorted(set(target) - set(generated))
    unauthorized = sorted(set(generated) - set(target))
    blockers: list[str] = []
    if not authority_cardinality or not authority_cardinality.get("authority_cardinality_pass"):
        blockers.append(F1_AUTHORITY_CARDINALITY_MISMATCH)
    if not target_scope or not target_scope.get("target_scope_membership_pass"):
        blockers.append(TARGET_SCOPE_MEMBERSHIP_UNRESOLVED)
    if not base_form_coverage or not base_form_coverage.get("coverage_pass"):
        blockers.append(BASE_FORM_COVERAGE_UNRESOLVED)
    if not stage185_provenance_validation or not stage185_provenance_validation.get("stage185_provenance_pass"):
        blockers.append(STAGE185_PROVENANCE_UNRESOLVED)
    if not all_stage185_transitions_pass(audit_materialized):
        blockers.append(STAGE185_TRANSITION_FAILED)
    if not full_output_validation or not full_output_validation.get("full_output_isolation_pass"):
        blockers.append(FULL_OUTPUT_ISOLATION_FAILED)
    if any(row.get("semantic_validation_status") == MANUAL_STATUS for row in audit_materialized):
        blockers.append(SEMANTIC_AUTHORITY_UNRESOLVED)
    if not provenance_validation or not provenance_validation.get("execution_provenance_pass"):
        status = (provenance_validation or {}).get("execution_provenance_status")
        blockers.append(PROVENANCE_IDENTITY_MISMATCH if status == PROVENANCE_IDENTITY_MISMATCH else PROVENANCE_UNRESOLVED)
    if len(target) != EXPECTED_F1_TARGET_COUNT or len(generated) != EXPECTED_F1_TARGET_COUNT or len(accepted) != EXPECTED_F1_TARGET_COUNT:
        blockers.append(F1_AUTHORITY_CARDINALITY_MISMATCH)
    if rejected or missing or unauthorized:
        blockers.append(SEMANTIC_AUTHORITY_UNRESOLVED)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "F1_target_pair_ids": target,
        "F1_generated_candidate_pair_ids": generated,
        "F1_accepted_candidate_pair_ids": accepted,
        "F1_manual_review_required_pair_ids": manual,
        "F1_rejected_candidate_pair_ids": rejected,
        "F1_missing_candidate_pair_ids": missing,
        "F1_unauthorized_candidate_pair_ids": unauthorized,
        "F1_execution_blockers": ordered_blockers(blockers),
        "authority_cardinality": dict(authority_cardinality or {}),
        "target_scope_membership": dict(target_scope or {}),
        "base_form_coverage": dict(base_form_coverage or {}),
        "stage185_provenance_validation": dict(stage185_provenance_validation or {}),
        "full_output_isolation_validation": dict(full_output_validation or {}),
        "execution_provenance_validation": dict(provenance_validation or {}),
        "F1_artifact_paths": {
            "summary": "p3w6f1_regeneration_summary.json",
            "regenerated_rows": "p3w6f1_regenerated_rows.jsonl",
            "audit": "p3w6f1_regeneration_audit.jsonl",
            "full_output_isolation": "p3w6f1_full_output_isolation.json",
        },
        "F1_input_sha256": (execution_provenance or {}).get("F1_input_sha256", ""),
        "F1_execution_commit": (execution_provenance or {}).get("F1_execution_commit", ""),
        "F1_output_sha256": (execution_provenance or {}).get("F1_output_sha256", ""),
    }
    for count_field, array_field in SUMMARY_PAIR_FIELDS:
        summary[count_field] = len(summary[array_field])
    execution_decision = derive_execution_decision(summary)
    summary["F1_execution_status"] = execution_decision
    summary["F1_execution_decision"] = execution_decision
    validate_summary_accounting(summary)
    return summary

def assert_artifact_schemas(
    summary: Mapping[str, Any],
    audit_rows: Iterable[Mapping[str, Any]],
    regenerated_records: Iterable[Mapping[str, Any]],
    isolation: Mapping[str, Any],
) -> None:
    for count_field, array_field in SUMMARY_PAIR_FIELDS:
        require(count_field in summary and array_field in summary, f"summary missing {count_field}/{array_field}")
    for name, rows in (("audit", audit_rows), ("regenerated", regenerated_records)):
        for row in rows:
            missing = [field for field in AUDIT_REQUIRED_FIELDS if field not in row]
            require(not missing, f"{name} row missing fields: {missing}")
    missing_isolation = [field for field in FULL_OUTPUT_ISOLATION_FIELDS if field not in isolation]
    require(not missing_isolation, f"isolation missing fields: {missing_isolation}")


def parse_json_arg(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    require(isinstance(parsed, dict) and bool(parsed), "JSON argument must be a non-empty object")
    return parsed

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P3-W6-F1 deterministic polarity regeneration analyzer")
    parser.add_argument("--p3w4-summary-json", required=True)
    parser.add_argument("--p3w4-pairs-jsonl", required=True)
    parser.add_argument("--p3w5-manifest-json", required=True)
    parser.add_argument("--baseline-jsonl", required=True)
    parser.add_argument("--repaired-jsonl", required=True)
    parser.add_argument("--baseline-sidecar-jsonl", required=True)
    parser.add_argument("--repaired-sidecar-jsonl", required=True)
    parser.add_argument("--baseline-generator-commit", required=True)
    parser.add_argument("--baseline-generator-source-path", required=True)
    parser.add_argument("--baseline-generator-source-sha256", required=True)
    parser.add_argument("--repaired-generator-commit", required=True)
    parser.add_argument("--repaired-generator-source-path", required=True)
    parser.add_argument("--repaired-generator-source-sha256", required=True)
    parser.add_argument("--deterministic-generator-invocation-json", required=True)
    parser.add_argument("--generator-configuration-identity-json", required=True)
    parser.add_argument("--f1-input-sha256", required=True)
    parser.add_argument("--f1-execution-commit", required=True)
    parser.add_argument("--f1-output-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path.cwd()
    input_authority = validate_p3w6f1_input_authority_identity(
        repo_root=repo_root,
        baseline_jsonl_path=Path(args.baseline_jsonl),
        baseline_sidecar_jsonl_path=Path(args.baseline_sidecar_jsonl),
        p3w4_summary_json_path=Path(args.p3w4_summary_json),
        p3w4_pairs_jsonl_path=Path(args.p3w4_pairs_jsonl),
        p3w5_manifest_json_path=Path(args.p3w5_manifest_json),
        f1_input_sha256=args.f1_input_sha256,
    )
    p3w4_summary = load_json(Path(args.p3w4_summary_json))
    p3w5_manifest = load_json(Path(args.p3w5_manifest_json))
    pair_records = load_jsonl(Path(args.p3w4_pairs_jsonl))
    supporting = extract_decision_supporting_pair_ids(p3w4_summary, p3w5_manifest)
    targets = extract_authorized_f1_targets(pair_records, supporting)
    cardinality = validate_authority_cardinality(targets)
    required_surfaces = required_f1_inflected_predicate_surfaces(
        pair_records, set(targets["F1_target_pair_ids"])
    )
    coverage = validate_base_form_coverage(required_surfaces, generator._BASE_PREDICATE_BY_INFLECTED)
    baseline = load_jsonl(Path(args.baseline_jsonl))
    repaired = load_jsonl(Path(args.repaired_jsonl))
    baseline_sidecar_rows = load_jsonl(Path(args.baseline_sidecar_jsonl))
    repaired_sidecar_rows = load_jsonl(Path(args.repaired_sidecar_jsonl))
    baseline_sidecar = sidecar_by_row_id(baseline_sidecar_rows)
    repaired_sidecar = sidecar_by_row_id(repaired_sidecar_rows)
    structural_ids = structural_negative_polarity_flip_row_ids_for_pair_count(
        len({str(row["pair_id"]) for row in baseline})
    )
    target_scope = validate_target_scope_membership(
        structural_ids, targets["authorized_F1_row_ids"]
    )
    invocation = parse_json_arg(args.deterministic_generator_invocation_json)
    config_identity = parse_json_arg(args.generator_configuration_identity_json)
    replay_validation = validate_repaired_output_replay_identity(
        baseline,
        repaired,
        targets["authorized_F1_row_ids"],
    )
    execution_provenance = {
        "baseline_generator_commit": args.baseline_generator_commit,
        "baseline_generator_source_path": args.baseline_generator_source_path,
        "baseline_generator_source_sha256": args.baseline_generator_source_sha256,
        "repaired_generator_commit": args.repaired_generator_commit,
        "repaired_generator_source_path": args.repaired_generator_source_path,
        "repaired_generator_source_sha256": args.repaired_generator_source_sha256,
        "deterministic_generator_invocation": invocation,
        "generator_configuration_identity": config_identity,
        "F1_input_sha256": args.f1_input_sha256,
        "F1_execution_commit": args.f1_execution_commit,
        "F1_output_sha256": args.f1_output_sha256,
    }
    isolation = full_output_isolation(
        baseline,
        repaired,
        authorized_f1_row_ids=targets["authorized_F1_row_ids"],
        structural_negative_polarity_flip_row_ids=target_scope["structural_negative_polarity_flip_row_ids"],
        repair_consumed_row_ids=replay_validation["actual_generator_repair_consumed_row_ids"],
        f2_row_ids=extract_f2_row_ids(pair_records),
        baseline_generator_commit=args.baseline_generator_commit,
        baseline_generator_source_path=args.baseline_generator_source_path,
        baseline_generator_source_sha256=args.baseline_generator_source_sha256,
        repaired_generator_commit=args.repaired_generator_commit,
        repaired_generator_source_path=args.repaired_generator_source_path,
        repaired_generator_source_sha256=args.repaired_generator_source_sha256,
        deterministic_generator_invocation=replay_validation["deterministic_generator_invocation"],
        generator_configuration_identity=replay_validation["generator_configuration_identity"],
    )
    isolation_validation = validate_full_output_isolation(isolation)
    runtime_authority = validate_stage185_runtime_authority_identity(
        repo_root=repo_root,
        baseline_integrity_builder_sha256=input_authority["baseline_integrity_builder_sha256"],
    )
    baseline_expected_rows, _baseline_facts = generator_expected_rows_and_facts_for_source(baseline)
    stage185_baseline_provenance = validate_stage185_sidecar_provenance(
        baseline,
        baseline_sidecar_rows,
        actual_source_dataset_sha256=file_sha256(Path(args.baseline_jsonl)),
        actual_source_dataset_path=Path(args.baseline_jsonl),
        actual_integrity_builder_sha256=runtime_authority["stage185_integrity_builder_source_sha256"],
        expected_generator_rows=list(baseline_expected_rows.values()),
        repo_root=repo_root,
        runtime_authority=runtime_authority,
    )
    stage185_repaired_provenance = validate_stage185_sidecar_provenance(
        repaired,
        repaired_sidecar_rows,
        actual_source_dataset_sha256=file_sha256(Path(args.repaired_jsonl)),
        actual_source_dataset_path=Path(args.repaired_jsonl),
        actual_integrity_builder_sha256=runtime_authority["stage185_integrity_builder_source_sha256"],
        expected_generator_rows=replay_validation["replayed_records"] if replay_validation["generator_replay_identity_pass"] else None,
        repo_root=repo_root,
        runtime_authority=runtime_authority,
    )
    stage185_provenance_validation = {
        "stage185_provenance_pass": (
            stage185_baseline_provenance["stage185_provenance_pass"]
            and stage185_repaired_provenance["stage185_provenance_pass"]
        ),
        "stage185_provenance_status": (
            "PASS"
            if stage185_baseline_provenance["stage185_provenance_pass"]
            and stage185_repaired_provenance["stage185_provenance_pass"]
            else STAGE185_PROVENANCE_UNRESOLVED
        ),
        "baseline_stage185_provenance": stage185_baseline_provenance,
        "repaired_stage185_provenance": stage185_repaired_provenance,
    }
    local_audit_rows = audit_authorized_candidates(
        baseline,
        repaired,
        pair_records,
        targets["authorized_F1_row_ids"],
        baseline_sidecar,
        repaired_sidecar,
        generator_source_sha256=args.repaired_generator_source_sha256,
        generator_commit=args.repaired_generator_commit,
        grammar_validator_source=runtime_authority["resolved_grammar_validator_source_path"],
        grammar_validator_sha256=runtime_authority["resolved_grammar_validator_source_sha256"],
    )
    provenance_validation = validate_execution_provenance_identity(
        execution_provenance,
        baseline_jsonl_path=Path(args.baseline_jsonl),
        repaired_jsonl_path=Path(args.repaired_jsonl),
        actual_deterministic_generator_invocation=replay_validation["deterministic_generator_invocation"],
        actual_generator_configuration_identity=replay_validation["generator_configuration_identity"],
        input_authority=input_authority,
        repo_root=repo_root,
    )
    if not replay_validation["generator_replay_identity_pass"]:
        provenance_validation = {
            **provenance_validation,
            "execution_provenance_pass": False,
            "execution_provenance_status": PROVENANCE_IDENTITY_MISMATCH,
            "provenance_identity_mismatches": sorted(
                set(provenance_validation.get("provenance_identity_mismatches", []))
                | {"generator_replay_identity"}
            ),
        }
    provenance_validation["generator_replay_identity"] = {
        key: value
        for key, value in replay_validation.items()
        if key != "replayed_records"
    }
    provenance_validation["stage185_runtime_authority"] = runtime_authority
    audit_rows = finalize_candidate_acceptance(
        local_audit_rows,
        full_output_isolation_validation=isolation_validation,
        stage185_provenance_validation=stage185_provenance_validation,
        execution_provenance_validation=provenance_validation,
    )
    generated_pair_ids = [
        row_id.rsplit("__", 1)[0]
        for row_id in set(targets["authorized_F1_row_ids"]) & {str(row.get("id", "")) for row in repaired}
    ]
    summary = build_summary(
        targets["F1_target_pair_ids"],
        generated_pair_ids,
        audit_rows,
        authority_cardinality=cardinality,
        target_scope=target_scope,
        base_form_coverage=coverage,
        stage185_provenance_validation=stage185_provenance_validation,
        full_output_validation=isolation_validation,
        provenance_validation=provenance_validation,
        execution_provenance=execution_provenance,
    )
    regenerated_records = audit_rows
    assert_artifact_schemas(summary, audit_rows, regenerated_records, isolation)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "p3w6f1_regeneration_summary.json", summary)
    write_json(output_dir / "p3w6f1_full_output_isolation.json", isolation)
    write_jsonl(output_dir / "p3w6f1_regenerated_rows.jsonl", regenerated_records)
    write_jsonl(output_dir / "p3w6f1_regeneration_audit.jsonl", audit_rows)
    return summary

def main() -> None:
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
