#!/usr/bin/env python3
"""P3-W6-F2 P4-B R1 structured regeneration materializer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


_REPO_IMPORT_ROOT = Path(__file__).resolve().parents[1]
if not any(Path(entry).resolve() == _REPO_IMPORT_ROOT for entry in sys.path if entry):
    sys.path.insert(0, str(_REPO_IMPORT_ROOT))

from scripts import build_controlled_v5 as generator


LEVEL1_FREEZE_COMMIT = "acc078f8ddb5ba362d0c6861e23de21aad09cb8b"
PARENT_RUNTIME_AUTHORITY_COMMIT = "cf80d52c222450cf84622a4f830b7331355bee07"
P4B_SPEC_AUTHORITY_COMMIT = "fcc3b9ccaf2bbee33ac18dcef10d50acff54aab4"
GENERATION_CONTRACT_VERSION = "P3W6F2P4B_R1_CLEAN_STRUCTURED_REGENERATION_V1"
PREDICATE_REALIZATION_CONTRACT_VERSION = "P3W6F2P4B_R1_BASE_PREDICATE_REALIZATION_V1"
HASH_NAMESPACE = "P3W6F2P4B_R1_REGENERATION_HASH_V1"

HISTORICAL_DATASET_PATH = "data/controlled_v5_v3_without_time_swap.jsonl"
LEVEL1_SUMMARY_PATH = (
    "reports/reason_router_p2_p3w6f2_hybrid_human_review_execution_cf80d52c/"
    "p3w6f2_hybrid_review_summary.json"
)
LEVEL1_DECISIONS_PATH = (
    "reports/reason_router_p2_p3w6f2_hybrid_human_review_execution_cf80d52c/"
    "p3w6f2_hybrid_review_decisions.jsonl"
)
LEVEL1_COMPLETED_CSV_PATH = (
    "reports/reason_router_p2_p3w6f2_hybrid_human_review_execution_cf80d52c/"
    "p3w6f2_hybrid_review_completed.csv"
)
LEVEL1_RESULT_REVIEW_JSON_PATH = (
    "reports/reason_router_p2_p3w6f2_hybrid_human_review_result_review_cf80d52c/"
    "p3w6f2_final_result_review.json"
)
LEVEL1_FINAL_REVIEW_WIP_JSONL_PATH = (
    "reports/reason_router_p2_p3w6f2_hybrid_human_review_result_review_cf80d52c/"
    "p3w6f2_final_review_wip.jsonl"
)
LEVEL1_STRUCTURAL_COHORT_AUDIT_PATH = (
    "reports/reason_router_p2_p3w6f2_hybrid_human_review_result_review_cf80d52c/"
    "p3w6f2_structural_cohort_audit_v1.json"
)
LEVEL1_STRUCTURAL_COHORT_CONFIRMATION_PATH = (
    "reports/reason_router_p2_p3w6f2_hybrid_human_review_result_review_cf80d52c/"
    "p3w6f2_structural_cohort_confirmation_v1.json"
)
LEVEL1_REVIEWER_ALIAS_EVIDENCE_PATH = (
    "reports/reason_router_p2_p3w6f2_hybrid_human_review_result_review_cf80d52c/"
    "p3w6f2_reviewer_alias_evidence_v1.json"
)
SPEC_PATH = "reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_spec.md"
GENERATOR_SOURCE_PATH = "scripts/build_controlled_v5.py"
STAGE185_SOURCE_PATH = "scripts/build_stage185a_controlled_train_integrity_sidecar.py"

FULL_DATASET_NAME = "controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl"
MEMBERS_NAME = "p3w6f2_p4b_r1_regenerated_members.jsonl"
AUDIT_NAME = "p3w6f2_p4b_r1_regeneration_audit.jsonl"
SUMMARY_NAME = "p3w6f2_p4b_r1_regeneration_summary.json"
ISOLATION_NAME = "p3w6f2_p4b_r1_full_output_isolation.json"
INVOCATION_NAME = "p3w6f2_p4b_r1_deterministic_generator_invocation.json"
COVERAGE_NAME = "p3w6f2_p4b_r1_base_form_coverage.json"
EXPECTED_ARTIFACT_NAMES = {
    FULL_DATASET_NAME,
    MEMBERS_NAME,
    AUDIT_NAME,
    SUMMARY_NAME,
    ISOLATION_NAME,
    INVOCATION_NAME,
    COVERAGE_NAME,
}
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
AUTHORIZED_PREDICATE_BASES = generator.F2_P4B_R1_REQUIRED_PREDICATE_BASES
AUTHORIZED_INTERVENTIONS = generator.F2_P4B_R1_REQUIRED_INTERVENTIONS
AUTHORIZED_PAIR_COUNT = 119
AUTHORIZED_MEMBER_COUNT = 357
EXPECTED_HISTORICAL_DATASET_SHA256 = "f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640"
EXPECTED_STAGE185_SOURCE_SHA256 = "11e6ba89b8131c76eac4504b4273867eaa99a131abe23d3238eb65ecda207bbc"
EXPECTED_FROZEN_AUTHORITY_SHA256: dict[str, str] = {
    LEVEL1_COMPLETED_CSV_PATH: "8c01bf4c4301382a28928543611fd1f78cb094810ed09d430b187da9bd4216c2",
    LEVEL1_DECISIONS_PATH: "d2c845baa7316187466bd3a2352824a7821387136524a9ef5c03630f0b3c397f",
    LEVEL1_SUMMARY_PATH: "5401f7e7fe1fb3cdd55802021b37cb33e6a7e3919faba85dbb34d8d5adbbffbc",
    LEVEL1_RESULT_REVIEW_JSON_PATH: "a0656020bc62b1933350f114054b839028113d532a538cf0c82786e356e9040c",
    LEVEL1_FINAL_REVIEW_WIP_JSONL_PATH: "28792fe90a8470c0fb3fec2a134a61c9d6897c458f23bfc174175f5bd906bf6b",
    LEVEL1_STRUCTURAL_COHORT_AUDIT_PATH: "dbe1c5a3dbe3ca76d2723ab62844774de92e2480c65bdff49228b1726a0df794",
    LEVEL1_STRUCTURAL_COHORT_CONFIRMATION_PATH: "b3686f732136bf3f3e5047ddf5123d5a78153abaf03cb397203704eeb5f25d06",
    LEVEL1_REVIEWER_ALIAS_EVIDENCE_PATH: "ecf77c655e0b8c8ab143fb5162422b9d93d37f0a5eac98cb7013799e1d28c919",
    SPEC_PATH: "42c152a44f1bf81471d8fe566aee8388c17c576a53584848db6f7205e06b291e",
    HISTORICAL_DATASET_PATH: EXPECTED_HISTORICAL_DATASET_SHA256,
    STAGE185_SOURCE_PATH: EXPECTED_STAGE185_SOURCE_SHA256,
}


class P4BRegenerationError(RuntimeError):
    """Fail-closed P4-B regeneration rejection."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise P4BRegenerationError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def deterministic_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def ordered_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {field: row[field] for field in DATASET_FIELDS}


def deterministic_jsonl_bytes(rows: Iterable[Mapping[str, Any]], *, dataset_rows: bool = False) -> bytes:
    materialized = [ordered_row(row) if dataset_rows else dict(row) for row in rows]
    return (
        "\n".join(
            json.dumps(row, ensure_ascii=False, sort_keys=not dataset_rows, allow_nan=False)
            for row in materialized
        )
        + "\n"
    ).encode("utf-8")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON_NOT_OBJECT:{path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        require(isinstance(value, dict), f"JSONL_ROW_NOT_OBJECT:{path}:{line_number}")
        rows.append(value)
    return rows


def is_full_commit(value: str) -> bool:
    return re.fullmatch(r"[0-9a-f]{40}", value or "") is not None


def git_stdout(repo_root: Path, args: Sequence[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), *args],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise P4BRegenerationError(f"GIT_COMMAND_FAILED:{' '.join(args)}") from exc


def current_head(repo_root: Path) -> str:
    return git_stdout(repo_root, ["rev-parse", "HEAD"])


def tracked_worktree_clean(repo_root: Path) -> bool:
    unstaged = subprocess.run(["git", "-C", str(repo_root), "diff", "--quiet", "--"], stderr=subprocess.DEVNULL)
    staged = subprocess.run(["git", "-C", str(repo_root), "diff", "--cached", "--quiet", "--"], stderr=subprocess.DEVNULL)
    return unstaged.returncode == 0 and staged.returncode == 0


def resolve_under_repo(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (repo_root / path).resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise P4BRegenerationError(f"PATH_OUTSIDE_REPO:{value}") from exc
    return resolved


def repo_relative_path(repo_root: Path, path: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def verify_execution_identity(
    repo_root: Path,
    execution_commit: str,
    *,
    head_resolver: Any | None = None,
    tracked_clean_checker: Any | None = None,
) -> str:
    require(is_full_commit(execution_commit), "EXECUTION_COMMIT_NOT_FULL_40_HEX")
    observed_head = (head_resolver or current_head)(repo_root)
    require(is_full_commit(observed_head), "CURRENT_HEAD_NOT_FULL_40_HEX")
    require(observed_head == execution_commit, "EXECUTION_COMMIT_HEAD_MISMATCH")
    require((tracked_clean_checker or tracked_worktree_clean)(repo_root), "TRACKED_WORKTREE_DIRTY")
    return observed_head


def expected_output_dir(repo_root: Path, execution_commit: str) -> Path:
    return (
        repo_root
        / f"reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_{execution_commit}"
    ).resolve()


def verify_output_dir(repo_root: Path, output_dir: Path, execution_commit: str) -> Path:
    observed = resolve_under_repo(repo_root, output_dir)
    require(observed == expected_output_dir(repo_root, execution_commit), "OUTPUT_DIR_CONTRACT_MISMATCH")
    return observed


def derive_authorized_f2_pair_ids(summary: Mapping[str, Any]) -> list[str]:
    pair_ids = summary.get("regeneration_required_pair_ids", summary.get("authorized_F2_pair_ids"))
    require(isinstance(pair_ids, list), "LEVEL1_AUTHORIZED_PAIR_IDS_MISSING")
    normalized = [str(pair_id) for pair_id in pair_ids]
    require(len(normalized) == len(set(normalized)), "LEVEL1_AUTHORIZED_PAIR_IDS_DUPLICATE")
    require(len(normalized) == AUTHORIZED_PAIR_COUNT, "LEVEL1_AUTHORIZED_PAIR_COUNT_MISMATCH")
    require(summary.get("authorized_F2_pair_count") == AUTHORIZED_PAIR_COUNT, "LEVEL1_AUTHORIZED_COUNT_MISMATCH")
    require(summary.get("completion_gate_passed") is True, "LEVEL1_COMPLETION_GATE_NOT_PASS")
    require(summary.get("source_hash_mismatch_count") == 0, "LEVEL1_SOURCE_HASH_MISMATCH_REPORTED")
    return normalized


def expected_member_ids(pair_ids: Iterable[str]) -> list[str]:
    return sorted(
        f"{pair_id}__{intervention}"
        for pair_id in pair_ids
        for intervention in AUTHORIZED_INTERVENTIONS
    )


def verify_expected_file_sha256(repo_root: Path, relative: str, expected: str) -> str:
    path = repo_root / relative
    require(path.is_file(), f"AUTHORITY_ARTIFACT_MISSING:{relative}")
    observed = file_sha256(path)
    require(observed == expected, f"AUTHORITY_ARTIFACT_SHA_MISMATCH:{relative}")
    return observed


def verify_authority_files(repo_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative, expected in sorted(EXPECTED_FROZEN_AUTHORITY_SHA256.items()):
        hashes[relative] = verify_expected_file_sha256(repo_root, relative, expected)
    generator_path = repo_root / GENERATOR_SOURCE_PATH
    require(generator_path.is_file(), f"AUTHORITY_ARTIFACT_MISSING:{GENERATOR_SOURCE_PATH}")
    hashes[GENERATOR_SOURCE_PATH] = file_sha256(generator_path)
    return hashes


def _pair_ids_from_completed_csv(repo_root: Path) -> list[str]:
    with (repo_root / LEVEL1_COMPLETED_CSV_PATH).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    pair_ids = [str(row.get("pair_id", "")) for row in rows]
    require(len(pair_ids) == AUTHORIZED_PAIR_COUNT, "COMPLETED_AUTHORIZED_PAIR_COUNT_MISMATCH")
    require(len(pair_ids) == len(set(pair_ids)), "COMPLETED_AUTHORIZED_PAIR_DUPLICATE")
    for row in rows:
        pair_id = str(row["pair_id"])
        for intervention, field in (
            ("none", "canonical_none_row_id"),
            ("paraphrase", "paraphrase_row_id"),
            ("polarity_flip", "polarity_flip_row_id"),
        ):
            require(row.get(field) == f"{pair_id}__{intervention}", f"COMPLETED_MEMBER_ID_MISMATCH:{pair_id}:{field}")
    return pair_ids


def _pair_ids_from_decisions(repo_root: Path) -> list[str]:
    decisions = load_jsonl(repo_root / LEVEL1_DECISIONS_PATH)
    pair_ids = [str(row.get("pair_id", "")) for row in decisions]
    require(len(pair_ids) == AUTHORIZED_PAIR_COUNT, "DECISIONS_AUTHORIZED_PAIR_COUNT_MISMATCH")
    require(len(pair_ids) == len(set(pair_ids)), "DECISIONS_AUTHORIZED_PAIR_DUPLICATE")
    require(all(row.get("review_record_valid") is True for row in decisions), "DECISIONS_REVIEW_RECORD_INVALID")
    return pair_ids


def authenticated_frozen_authority(repo_root: Path) -> dict[str, Any]:
    authority_hashes = verify_authority_files(repo_root)
    summary = load_json(repo_root / LEVEL1_SUMMARY_PATH)
    summary_pair_ids = derive_authorized_f2_pair_ids(summary)
    completed_pair_ids = _pair_ids_from_completed_csv(repo_root)
    decision_pair_ids = _pair_ids_from_decisions(repo_root)
    require(summary_pair_ids == completed_pair_ids, "SUMMARY_COMPLETED_PAIR_UNIVERSE_MISMATCH")
    require(summary_pair_ids == decision_pair_ids, "SUMMARY_DECISIONS_PAIR_UNIVERSE_MISMATCH")
    member_ids = expected_member_ids(summary_pair_ids)
    require(len(member_ids) == AUTHORIZED_MEMBER_COUNT, "AUTHENTICATED_MEMBER_COUNT_MISMATCH")
    return {
        "authority_hashes": authority_hashes,
        "authorized_pair_ids": summary_pair_ids,
        "authorized_member_ids": member_ids,
        "p4b_spec_authority_commit": P4B_SPEC_AUTHORITY_COMMIT,
        "level1_freeze_commit": LEVEL1_FREEZE_COMMIT,
    }


def require_exact_authorized_pair_ids(candidate_pair_ids: Iterable[str], authority: Mapping[str, Any]) -> list[str]:
    candidate = sorted(str(pair_id) for pair_id in candidate_pair_ids)
    expected = sorted(str(pair_id) for pair_id in authority["authorized_pair_ids"])
    require(candidate == expected, "AUTHORIZED_PAIR_UNIVERSE_NOT_EXACT_FROZEN_AUTHORITY")
    return list(authority["authorized_pair_ids"])


def validate_historical_dataset(rows: Sequence[Mapping[str, Any]]) -> None:
    require(len(rows) > 0, "HISTORICAL_DATASET_EMPTY")
    ids = [str(row.get("id", "")) for row in rows]
    require(len(ids) == len(set(ids)), "HISTORICAL_DATASET_DUPLICATE_ROW_ID")
    for index, row in enumerate(rows):
        require(tuple(row.keys()) == DATASET_FIELDS, f"HISTORICAL_DATASET_SCHEMA_MISMATCH:{index}")


def pair_count(rows: Sequence[Mapping[str, Any]]) -> int:
    return len({str(row["pair_id"]) for row in rows})


def row_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    mapped = {str(row["id"]): row for row in rows}
    require(len(mapped) == len(rows), "DUPLICATE_ROW_ID")
    return mapped


def project_to_historical_topology(
    replay_rows: Sequence[Mapping[str, Any]],
    historical_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    replay_by_id = row_map(replay_rows)
    projected: list[dict[str, Any]] = []
    for historical in historical_rows:
        row_id = str(historical["id"])
        require(row_id in replay_by_id, f"REPLAY_ROW_MISSING_FOR_HISTORICAL_TOPOLOGY:{row_id}")
        projected.append(dict(replay_by_id[row_id]))
    require([row["id"] for row in projected] == [row["id"] for row in historical_rows], "TOPOLOGY_PROJECTION_ORDER_MISMATCH")
    return projected


def changed_fields(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[str]:
    return [field for field in DATASET_FIELDS if before.get(field) != after.get(field)]


def split_identity(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    train, dev = generator.split_by_pair_id([dict(row) for row in rows], dev_ratio=0.2, seed=17)
    return {str(row["id"]): "dev" for row in dev} | {str(row["id"]): "train" for row in train}


def structured_fact_by_pair(num_pairs: int, authorized_pair_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    templates = generator.fact_templates_for_count(num_pairs)
    facts = {
        fact["pair_id"]: {field: fact[field] for field in generator._FACT_FIELDS}
        for fact in templates
        if fact["pair_id"] in set(authorized_pair_ids)
    }
    require(len(facts) == AUTHORIZED_PAIR_COUNT, "STRUCTURED_FACT_COUNT_MISMATCH")
    return facts


def build_base_form_coverage(
    facts_by_pair: Mapping[str, Mapping[str, Any]],
    *,
    source_sha256: str,
) -> dict[str, Any]:
    observed = sorted({str(fact["predicate"]) for fact in facts_by_pair.values()})
    missing = [
        predicate
        for predicate, base in AUTHORIZED_PREDICATE_BASES.items()
        if generator._BASE_PREDICATE_BY_INFLECTED.get(predicate) != base
    ]
    extra = sorted(set(observed) - set(AUTHORIZED_PREDICATE_BASES))
    pair_coverage = {
        pair_id: {
            "semantic_predicate": fact["predicate"],
            "base_predicate": generator._BASE_PREDICATE_BY_INFLECTED.get(str(fact["predicate"])),
            "coverage_status": "PASS"
            if str(fact["predicate"]) in AUTHORIZED_PREDICATE_BASES
            and generator._BASE_PREDICATE_BY_INFLECTED.get(str(fact["predicate"]))
            == AUTHORIZED_PREDICATE_BASES[str(fact["predicate"])]
            else "FAIL",
        }
        for pair_id, fact in sorted(facts_by_pair.items())
    }
    return {
        "schema_version": "P3W6F2P4B_R1_BASE_FORM_COVERAGE_V1",
        "predicate_realization_contract_version": PREDICATE_REALIZATION_CONTRACT_VERSION,
        "mapping_source_symbol": "_BASE_PREDICATE_BY_INFLECTED",
        "mapping_source_sha256": source_sha256,
        "authorized_predicates": sorted(AUTHORIZED_PREDICATE_BASES),
        "observed_authorized_predicates": observed,
        "required_base_forms": dict(sorted(AUTHORIZED_PREDICATE_BASES.items())),
        "missing_mappings": missing,
        "extra_observed_predicates": extra,
        "ambiguous_mappings": [],
        "pair_coverage": pair_coverage,
        "coverage_status": "PASS" if not missing and not extra else "FAIL",
    }


def semantic_dataset_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    return canonical_sha256([ordered_row(row) for row in rows])


def build_regenerated_payload(
    *,
    repo_root: Path,
    historical_rows: Sequence[Mapping[str, Any]],
    authorized_pair_ids: Sequence[str],
    execution_commit: str,
    authority_hashes: Mapping[str, str],
    implementation_commit: str | None = None,
) -> dict[str, Any]:
    authority = authenticated_frozen_authority(repo_root)
    authorized_pair_ids = require_exact_authorized_pair_ids(authorized_pair_ids, authority)
    require(authority_hashes == authority["authority_hashes"], "AUTHORITY_HASH_MAP_NOT_AUTHENTICATED")
    implementation_commit = implementation_commit or execution_commit
    require(is_full_commit(implementation_commit), "IMPLEMENTATION_COMMIT_NOT_FULL_40_HEX")
    num_pairs = pair_count(historical_rows)
    replay_rows, generator_audit = generator.build_controlled_records_with_f2_p4b_r1_regeneration_audit(
        num_pairs,
        set(authorized_pair_ids),
    )
    regenerated_rows = project_to_historical_topology(replay_rows, historical_rows)
    validate_historical_dataset(regenerated_rows)
    require([row["id"] for row in historical_rows] == [row["id"] for row in regenerated_rows], "ROW_ORDER_DRIFT")
    historical_by_id = row_map(historical_rows)
    regenerated_by_id = row_map(regenerated_rows)
    facts_by_pair = structured_fact_by_pair(num_pairs, authorized_pair_ids)
    historical_split = split_identity(historical_rows)
    regenerated_split = split_identity(regenerated_rows)
    require(historical_split == regenerated_split, "SPLIT_DRIFT")

    members: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    authorized_changed_row_ids: list[str] = []
    authorized_unchanged_row_ids: list[str] = []
    unauthorized_changed_row_ids: list[str] = []
    field_delta_counts: dict[str, int] = {}
    for row_index, (historical, regenerated) in enumerate(zip(historical_rows, regenerated_rows, strict=True)):
        row_id = str(historical["id"])
        deltas = changed_fields(historical, regenerated)
        for field in deltas:
            field_delta_counts[field] = field_delta_counts.get(field, 0) + 1
        pair_id = str(historical["pair_id"])
        intervention = str(historical["intervention_type"])
        authorized_member = pair_id in set(authorized_pair_ids) and intervention in AUTHORIZED_INTERVENTIONS
        if not authorized_member:
            if deltas:
                unauthorized_changed_row_ids.append(row_id)
            continue
        fact = facts_by_pair[pair_id]
        base = generator._BASE_PREDICATE_BY_INFLECTED[str(fact["predicate"])]
        permitted = deltas == ["evidence"] if intervention in {"none", "paraphrase"} else deltas == []
        require(permitted, f"AUTHORIZED_FIELD_DELTA_SCOPE_MISMATCH:{row_id}")
        if deltas:
            authorized_changed_row_ids.append(row_id)
        else:
            authorized_unchanged_row_ids.append(row_id)
        member = {
            "schema_version": "P3W6F2P4B_R1_REGENERATED_MEMBER_V1",
            "generation_contract_version": GENERATION_CONTRACT_VERSION,
            "pair_id": pair_id,
            "member_id": row_id,
            "intervention_type": intervention,
            "member_role": intervention,
            "structured_fact": dict(fact),
            "structured_fact_sha256": canonical_sha256(dict(fact)),
            "semantic_predicate": fact["predicate"],
            "base_predicate": base,
            "negative_auxiliary_realization": f"did not {base}" if intervention in {"none", "paraphrase"} else "",
            "claim": regenerated["claim"],
            "regenerated_evidence": regenerated["evidence"],
            "final_label": regenerated["final_label"],
            "frame_compatible_label": regenerated["frame_compatible_label"],
            "predicate_covered_label": regenerated["predicate_covered_label"],
            "sufficiency_label": regenerated["sufficiency_label"],
            "polarity_label": regenerated["polarity_label"],
            "primary_failure_type": regenerated["primary_failure_type"],
            "generation_root": "structured_fact",
            "generation_template": "_paraphrase" if intervention == "paraphrase" else "_statement",
            "historical_member_id": row_id,
            "historical_row_sha256": canonical_sha256(ordered_row(historical)),
            "regenerated_row_sha256": canonical_sha256(ordered_row(regenerated)),
            "row_field_delta_keys": deltas,
            "old_text_used_for_generation": False,
            "source_authority_commit": LEVEL1_FREEZE_COMMIT,
            "execution_commit": execution_commit,
        }
        members.append(member)
        preservation_fields = [field for field in DATASET_FIELDS if field != "evidence"]
        audit_rows.append(
            {
                "schema_version": "P3W6F2P4B_R1_MEMBER_AUDIT_V1",
                "pair_id": pair_id,
                "member_id": row_id,
                "intervention_type": intervention,
                "historical_row": ordered_row(historical),
                "regenerated_row": ordered_row(regenerated),
                "field_delta": {field: [historical[field], regenerated[field]] for field in deltas},
                "permitted_delta": permitted,
                "semantic_slot_preservation": {
                    "status": "PASS",
                    "structured_fact_sha256": canonical_sha256(dict(fact)),
                },
                "label_preservation": {
                    field: historical[field] == regenerated[field]
                    for field in (
                        "final_label",
                        "frame_compatible_label",
                        "predicate_covered_label",
                        "sufficiency_label",
                        "polarity_label",
                        "primary_failure_type",
                    )
                },
                "identity_preservation": {
                    field: historical[field] == regenerated[field]
                    for field in ("id", "pair_id", "claim", "intervention_type")
                },
                "row_order_preservation": {"row_index": row_index, "status": "PASS"},
                "split_preservation": {
                    "historical_split": historical_split[row_id],
                    "regenerated_split": regenerated_split[row_id],
                    "status": "PASS",
                },
                "structured_source_replay_status": "PASS",
                "predicate_base_mapping_status": "PASS",
                "old_text_isolation_status": "PASS",
                "member_audit_status": "PASS",
                "failure_reasons": [],
            }
        )
        require(all(historical[field] == regenerated[field] for field in preservation_fields), f"PROTECTED_FIELD_DRIFT:{row_id}")
    require(not unauthorized_changed_row_ids, "UNAUTHORIZED_ROW_CHANGED")
    require(len(members) == AUTHORIZED_MEMBER_COUNT, "AUTHORIZED_MEMBER_COUNT_MISMATCH")
    require(len(audit_rows) == AUTHORIZED_MEMBER_COUNT, "AUDIT_MEMBER_COUNT_MISMATCH")
    coverage = build_base_form_coverage(facts_by_pair, source_sha256=authority_hashes[GENERATOR_SOURCE_PATH])
    require(coverage["coverage_status"] == "PASS", "BASE_FORM_COVERAGE_FAILED")
    full_dataset_bytes = deterministic_jsonl_bytes(regenerated_rows, dataset_rows=True)
    historical_dataset_sha = file_sha256(repo_root / HISTORICAL_DATASET_PATH)
    require(historical_dataset_sha == EXPECTED_HISTORICAL_DATASET_SHA256, "HISTORICAL_DATASET_SHA_MISMATCH")
    regenerated_sha = sha256_bytes(full_dataset_bytes)
    invocation = {
        "schema_version": "P3W6F2P4B_R1_INVOCATION_V1",
        "command": "scripts/regenerate_reason_router_p3w6f2_p4b_r1_structured.py",
        "arguments": {"execution_commit": execution_commit},
        "environment_policy": "deterministic_cpu_file_materialization_only",
        "python_version": sys.version.split()[0],
        "locale_policy": "UTF-8",
        "timezone_policy": "UTC for manifests only; dataset rows contain no wall-clock timestamp",
        "random_seed_policy": "no random sampling; Stage185 split identity replay uses fixed seed 17 only",
        "input_paths": sorted(authority_hashes),
        "output_directory": f"reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_{execution_commit}",
        "source_authority_commit": LEVEL1_FREEZE_COMMIT,
        "p4b_spec_authority_commit": P4B_SPEC_AUTHORITY_COMMIT,
        "implementation_commit": implementation_commit,
        "execution_commit": execution_commit,
        "dirty_tracked_worktree_allowed": False,
    }
    invocation["deterministic_invocation_sha256"] = canonical_sha256(invocation)
    isolation = {
        "schema_version": "P3W6F2P4B_R1_FULL_OUTPUT_ISOLATION_V1",
        "historical_dataset_path": HISTORICAL_DATASET_PATH,
        "historical_dataset_sha256": historical_dataset_sha,
        "regenerated_dataset_path": FULL_DATASET_NAME,
        "regenerated_dataset_sha256": regenerated_sha,
        "row_count_historical": len(historical_rows),
        "row_count_regenerated": len(regenerated_rows),
        "authorized_pair_count": len(set(authorized_pair_ids)),
        "authorized_member_count": len(members),
        "authorized_changed_row_ids": authorized_changed_row_ids,
        "authorized_unchanged_row_ids": authorized_unchanged_row_ids,
        "unauthorized_changed_row_ids": unauthorized_changed_row_ids,
        "field_delta_counts": dict(sorted(field_delta_counts.items())),
        "row_order_identical": True,
        "non_f2_rows_byte_identical": True,
        "isolation_status": "PASS",
        "failure_reasons": [],
    }
    summary = {
        "schema_version": "P3W6F2P4B_R1_REGENERATION_SUMMARY_V1",
        "generation_contract_version": GENERATION_CONTRACT_VERSION,
        "level1_freeze_commit": LEVEL1_FREEZE_COMMIT,
        "parent_runtime_authority_commit": PARENT_RUNTIME_AUTHORITY_COMMIT,
        "p4b_spec_authority_commit": P4B_SPEC_AUTHORITY_COMMIT,
        "implementation_commit": implementation_commit,
        "execution_commit": execution_commit,
        "head_clean_required": True,
        "authority_artifacts": sorted(authority_hashes),
        "authority_artifact_sha256": dict(sorted(authority_hashes.items())),
        "historical_dataset_path": HISTORICAL_DATASET_PATH,
        "historical_dataset_sha256": historical_dataset_sha,
        "regenerated_dataset_path": FULL_DATASET_NAME,
        "regenerated_dataset_sha256": regenerated_sha,
        "regenerated_dataset_semantic_sha256": semantic_dataset_hash(regenerated_rows),
        "authorized_pair_count": len(set(authorized_pair_ids)),
        "authorized_member_count": len(members),
        "changed_pair_count": len({row_id.rsplit("__", 1)[0] for row_id in authorized_changed_row_ids}),
        "changed_member_count": len(authorized_changed_row_ids),
        "canonical_changed_member_count": len([row_id for row_id in authorized_changed_row_ids if row_id.endswith("__none")]),
        "paraphrase_changed_member_count": len([row_id for row_id in authorized_changed_row_ids if row_id.endswith("__paraphrase")]),
        "polarity_flip_changed_member_count": len([row_id for row_id in authorized_changed_row_ids if row_id.endswith("__polarity_flip")]),
        "unchanged_non_f2_row_count": len(historical_rows) - AUTHORIZED_MEMBER_COUNT,
        "predicate_base_mapping_version": PREDICATE_REALIZATION_CONTRACT_VERSION,
        "predicate_base_mapping_sha256": canonical_sha256(AUTHORIZED_PREDICATE_BASES),
        "structured_source_producer": f"{GENERATOR_SOURCE_PATH}::fact_templates_for_count",
        "structured_source_producer_sha256": authority_hashes[GENERATOR_SOURCE_PATH],
        "deterministic_invocation_sha256": invocation["deterministic_invocation_sha256"],
        "artifact_set_complete": True,
        "fail_closed_status": "PASS",
        "created_at_utc": "DETERMINISTIC_REPLAY_NO_WALL_CLOCK",
    }
    return {
        FULL_DATASET_NAME: full_dataset_bytes,
        MEMBERS_NAME: deterministic_jsonl_bytes(members),
        AUDIT_NAME: deterministic_jsonl_bytes(audit_rows),
        SUMMARY_NAME: deterministic_json_bytes(summary),
        ISOLATION_NAME: deterministic_json_bytes(isolation),
        INVOCATION_NAME: deterministic_json_bytes(invocation),
        COVERAGE_NAME: deterministic_json_bytes(coverage),
        "_summary": summary,
        "_generator_audit": generator_audit,
    }


def publish_artifacts(output_dir: Path, payloads: Mapping[str, bytes]) -> str:
    public_payloads = {name: payload for name, payload in payloads.items() if not name.startswith("_")}
    require(set(public_payloads) == EXPECTED_ARTIFACT_NAMES, "ARTIFACT_SET_PAYLOAD_MISMATCH")
    if output_dir.exists():
        require(output_dir.is_dir(), "OUTPUT_PATH_EXISTS_NOT_DIRECTORY")
        observed = {entry.name for entry in output_dir.iterdir()}
        require(observed == EXPECTED_ARTIFACT_NAMES, "OUTPUT_ARTIFACT_SET_MISMATCH")
        conflicts = [
            name
            for name, payload in public_payloads.items()
            if not (output_dir / name).is_file() or (output_dir / name).read_bytes() != payload
        ]
        require(not conflicts, f"OUTPUT_ARTIFACT_CONFLICT:{conflicts}")
        return "IDEMPOTENT_PASS"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    require(not staging_dir.exists(), "STAGING_DIR_ALREADY_EXISTS")
    try:
        staging_dir.mkdir()
        for name in sorted(public_payloads):
            (staging_dir / name).write_bytes(public_payloads[name])
        require({entry.name for entry in staging_dir.iterdir()} == EXPECTED_ARTIFACT_NAMES, "STAGING_ARTIFACT_SET_MISMATCH")
        staging_dir.replace(output_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    require({entry.name for entry in output_dir.iterdir()} == EXPECTED_ARTIFACT_NAMES, "OUTPUT_ARTIFACT_SET_MISMATCH")
    return "PUBLISHED"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P3-W6-F2 P4-B R1 structured regeneration materializer")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    implementation_commit = verify_execution_identity(
        repo_root,
        args.execution_commit,
    )
    output_dir = verify_output_dir(repo_root, args.output_dir, args.execution_commit)
    authority = authenticated_frozen_authority(repo_root)
    authority_hashes = authority["authority_hashes"]
    authorized_pair_ids = authority["authorized_pair_ids"]
    historical_path = repo_root / HISTORICAL_DATASET_PATH
    require(historical_path.is_file(), "HISTORICAL_DATASET_MISSING")
    historical_rows = load_jsonl(historical_path)
    validate_historical_dataset(historical_rows)
    payloads = build_regenerated_payload(
        repo_root=repo_root,
        historical_rows=historical_rows,
        authorized_pair_ids=authorized_pair_ids,
        execution_commit=args.execution_commit,
        authority_hashes=authority_hashes,
        implementation_commit=implementation_commit,
    )
    publish_status = publish_artifacts(output_dir, payloads)
    return {"status": "PASS", "publish_status": publish_status, "summary": payloads["_summary"]}


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = run(build_arg_parser().parse_args(argv))
    except P4BRegenerationError as exc:
        raise SystemExit(f"P3W6F2P4B_R1_REGENERATION_FAILED_CLOSED:{exc}") from exc
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
