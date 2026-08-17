#!/usr/bin/env python3
"""Read-only validator for P3-W6-F2 P4-B R1 regeneration artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


_REPO_IMPORT_ROOT = Path(__file__).resolve().parents[1]
if not any(Path(entry).resolve() == _REPO_IMPORT_ROOT for entry in sys.path if entry):
    sys.path.insert(0, str(_REPO_IMPORT_ROOT))

from scripts import build_controlled_v5 as generator
from scripts import regenerate_reason_router_p3w6f2_p4b_r1_structured as regen


ANALYZER_SCHEMA_VERSION = "P3W6F2P4B_R1_REGENERATION_ANALYSIS_V1"
OPTIONAL_STAGE185_COMPATIBILITY_ARTIFACT_NAMES = {
    "p3w6f2_p4b_r1_stage185_predicate_realization_compatibility_rows.jsonl",
    "p3w6f2_p4b_r1_stage185_predicate_realization_compatibility_summary.json",
    "p3w6f2_p4b_r1_stage185_predicate_realization_compatibility_provenance_manifest.json",
}
MEMBER_FIELDS = (
    "schema_version",
    "generation_contract_version",
    "pair_id",
    "member_id",
    "intervention_type",
    "member_role",
    "structured_fact",
    "structured_fact_sha256",
    "semantic_predicate",
    "base_predicate",
    "negative_auxiliary_realization",
    "claim",
    "regenerated_evidence",
    "final_label",
    "frame_compatible_label",
    "predicate_covered_label",
    "sufficiency_label",
    "polarity_label",
    "primary_failure_type",
    "generation_root",
    "generation_template",
    "historical_member_id",
    "historical_row_sha256",
    "regenerated_row_sha256",
    "row_field_delta_keys",
    "old_text_used_for_generation",
    "source_authority_commit",
    "execution_commit",
)
AUDIT_FIELDS = (
    "schema_version",
    "pair_id",
    "member_id",
    "intervention_type",
    "historical_row",
    "regenerated_row",
    "field_delta",
    "permitted_delta",
    "semantic_slot_preservation",
    "label_preservation",
    "identity_preservation",
    "row_order_preservation",
    "split_preservation",
    "structured_source_replay_status",
    "predicate_base_mapping_status",
    "old_text_isolation_status",
    "member_audit_status",
    "failure_reasons",
)


class P4BAnalysisError(RuntimeError):
    """Fail-closed analyzer rejection."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise P4BAnalysisError(message)


def load_required_artifacts(execution_dir: Path) -> dict[str, Any]:
    require(execution_dir.is_dir(), "EXECUTION_DIR_MISSING")
    observed = {entry.name for entry in execution_dir.iterdir()}
    require(regen.EXPECTED_ARTIFACT_NAMES <= observed, "ARTIFACT_SET_INCOMPLETE_OR_UNEXPECTED")
    unexpected = observed - regen.EXPECTED_ARTIFACT_NAMES - OPTIONAL_STAGE185_COMPATIBILITY_ARTIFACT_NAMES
    require(not unexpected, f"ARTIFACT_SET_UNEXPECTED:{sorted(unexpected)}")
    return {
        regen.FULL_DATASET_NAME: regen.load_jsonl(execution_dir / regen.FULL_DATASET_NAME),
        regen.MEMBERS_NAME: regen.load_jsonl(execution_dir / regen.MEMBERS_NAME),
        regen.AUDIT_NAME: regen.load_jsonl(execution_dir / regen.AUDIT_NAME),
        regen.SUMMARY_NAME: regen.load_json(execution_dir / regen.SUMMARY_NAME),
        regen.ISOLATION_NAME: regen.load_json(execution_dir / regen.ISOLATION_NAME),
        regen.INVOCATION_NAME: regen.load_json(execution_dir / regen.INVOCATION_NAME),
        regen.COVERAGE_NAME: regen.load_json(execution_dir / regen.COVERAGE_NAME),
    }


def require_exact_keys(row: Mapping[str, Any], fields: Sequence[str], message: str) -> None:
    require(set(row.keys()) == set(fields), message)


def validate_member_universe(
    members: Sequence[Mapping[str, Any]],
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    pair_ids = sorted({str(row.get("pair_id", "")) for row in members})
    member_ids = [str(row.get("member_id", "")) for row in members]
    expected_pairs = sorted(str(pair_id) for pair_id in authority["authorized_pair_ids"])
    expected_members = sorted(str(member_id) for member_id in authority["authorized_member_ids"])
    require(pair_ids == expected_pairs, "AUTHORIZED_PAIR_UNIVERSE_NOT_EXACT_FROZEN_AUTHORITY")
    require(sorted(member_ids) == expected_members, "AUTHORIZED_MEMBER_UNIVERSE_NOT_EXACT_FROZEN_AUTHORITY")
    require(len(member_ids) == len(set(member_ids)), "DUPLICATE_AUTHORIZED_MEMBER")
    for row in members:
        require_exact_keys(row, MEMBER_FIELDS, f"MEMBER_SCHEMA_MISMATCH:{row.get('member_id')}")
        require(row.get("schema_version") == "P3W6F2P4B_R1_REGENERATED_MEMBER_V1", f"MEMBER_SCHEMA_VERSION_MISMATCH:{row.get('member_id')}")
    for pair_id in pair_ids:
        interventions = sorted(
            str(row.get("intervention_type", ""))
            for row in members
            if str(row.get("pair_id", "")) == pair_id
        )
        require(interventions == sorted(regen.AUTHORIZED_INTERVENTIONS), f"INTERVENTION_TRIPLE_MISMATCH:{pair_id}")
    return {"authorized_pair_ids": pair_ids, "authorized_member_ids": sorted(member_ids)}


def validate_base_form_coverage(coverage: Mapping[str, Any], members: Sequence[Mapping[str, Any]]) -> None:
    require(coverage.get("schema_version") == "P3W6F2P4B_R1_BASE_FORM_COVERAGE_V1", "BASE_FORM_SCHEMA_MISMATCH")
    require(coverage.get("required_base_forms") == dict(sorted(regen.AUTHORIZED_PREDICATE_BASES.items())), "BASE_FORM_AUTHORITY_MISMATCH")
    observed = sorted({str(row.get("semantic_predicate", "")) for row in members})
    require(observed == sorted(regen.AUTHORIZED_PREDICATE_BASES), "OBSERVED_PREDICATE_SET_MISMATCH")
    for row in members:
        predicate = str(row["semantic_predicate"])
        require(row.get("base_predicate") == regen.AUTHORIZED_PREDICATE_BASES[predicate], f"MEMBER_BASE_PREDICATE_MISMATCH:{row.get('member_id')}")
    missing = [
        predicate
        for predicate, base in regen.AUTHORIZED_PREDICATE_BASES.items()
        if generator._BASE_PREDICATE_BY_INFLECTED.get(predicate) != base
    ]
    require(not missing, f"BASE_FORM_MISSING_MAPPING:{missing}")
    require(coverage.get("coverage_status") == "PASS", "BASE_FORM_COVERAGE_PRODUCER_FLAG_MISMATCH")


def validate_hash_contracts(
    repo_root: Path,
    execution_dir: Path,
    artifacts: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> None:
    summary = artifacts[regen.SUMMARY_NAME]
    isolation = artifacts[regen.ISOLATION_NAME]
    invocation = artifacts[regen.INVOCATION_NAME]
    dataset_rows = artifacts[regen.FULL_DATASET_NAME]
    observed_authority = regen.verify_authority_files(repo_root)
    require(observed_authority == authority["authority_hashes"], "AUTHORITY_HASH_MAP_MISMATCH")
    require(summary.get("authority_artifact_sha256") == dict(sorted(observed_authority.items())), "SUMMARY_AUTHORITY_HASH_MISMATCH")
    require(summary.get("historical_dataset_sha256") == regen.EXPECTED_HISTORICAL_DATASET_SHA256, "SUMMARY_HISTORICAL_DATASET_SHA_MISMATCH")
    require(invocation.get("p4b_spec_authority_commit") == regen.P4B_SPEC_AUTHORITY_COMMIT, "INVOCATION_P4B_SPEC_AUTHORITY_MISMATCH")
    require(summary.get("p4b_spec_authority_commit") == regen.P4B_SPEC_AUTHORITY_COMMIT, "SUMMARY_P4B_SPEC_AUTHORITY_MISMATCH")
    require(regen.is_full_commit(str(invocation.get("implementation_commit", ""))), "INVOCATION_IMPLEMENTATION_COMMIT_NOT_FULL")
    require(regen.is_full_commit(str(invocation.get("execution_commit", ""))), "INVOCATION_EXECUTION_COMMIT_NOT_FULL")
    require(invocation.get("implementation_commit") == invocation.get("execution_commit"), "IMPLEMENTATION_EXECUTION_COMMIT_MISMATCH")
    require(summary.get("implementation_commit") == summary.get("execution_commit"), "SUMMARY_IMPLEMENTATION_EXECUTION_COMMIT_MISMATCH")
    require(
        execution_dir.name
        == f"reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_{invocation['execution_commit']}",
        "EXECUTION_DIR_CONTRACT_MISMATCH",
    )
    require(invocation.get("dirty_tracked_worktree_allowed") is False, "DIRTY_WORKTREE_BYPASS_RECORDED")
    require(summary.get("regenerated_dataset_sha256") == regen.file_sha256(execution_dir / regen.FULL_DATASET_NAME), "REGENERATED_DATASET_SHA_MISMATCH")
    require(summary.get("regenerated_dataset_semantic_sha256") == regen.semantic_dataset_hash(dataset_rows), "REGENERATED_DATASET_SEMANTIC_SHA_MISMATCH")
    require(isolation.get("regenerated_dataset_sha256") == summary.get("regenerated_dataset_sha256"), "ISOLATION_REGENERATED_SHA_MISMATCH")
    expected_invocation_sha = regen.canonical_sha256({key: value for key, value in invocation.items() if key != "deterministic_invocation_sha256"})
    require(invocation.get("deterministic_invocation_sha256") == expected_invocation_sha, "INVOCATION_SHA_MISMATCH")
    require(summary.get("deterministic_invocation_sha256") == invocation.get("deterministic_invocation_sha256"), "SUMMARY_INVOCATION_SHA_MISMATCH")


def validate_replay_and_isolation(
    repo_root: Path,
    artifacts: Mapping[str, Any],
    member_universe: Mapping[str, Any],
) -> dict[str, Any]:
    historical_rows = regen.load_jsonl(repo_root / regen.HISTORICAL_DATASET_PATH)
    regenerated_rows = artifacts[regen.FULL_DATASET_NAME]
    regen.validate_historical_dataset(historical_rows)
    regen.validate_historical_dataset(regenerated_rows)
    require(regen.file_sha256(repo_root / regen.HISTORICAL_DATASET_PATH) == regen.EXPECTED_HISTORICAL_DATASET_SHA256, "HISTORICAL_DATASET_SHA_MISMATCH")
    require(len(historical_rows) == len(regenerated_rows), "FULL_DATASET_ROW_COUNT_MISMATCH")
    require([row["id"] for row in historical_rows] == [row["id"] for row in regenerated_rows], "ROW_ORDER_MISMATCH")
    historical_by_id = regen.row_map(historical_rows)
    regenerated_by_id = regen.row_map(regenerated_rows)
    authorized_member_ids = set(member_universe["authorized_member_ids"])
    unauthorized_changed: list[str] = []
    authorized_changed: list[str] = []
    protected_drift: list[str] = []
    for row_id, historical in historical_by_id.items():
        regenerated = regenerated_by_id[row_id]
        deltas = regen.changed_fields(historical, regenerated)
        if row_id not in authorized_member_ids:
            if deltas:
                unauthorized_changed.append(row_id)
            continue
        intervention = str(historical["intervention_type"])
        expected = ["evidence"] if intervention in {"none", "paraphrase"} else []
        if deltas != expected:
            protected_drift.append(row_id)
        if deltas:
            authorized_changed.append(row_id)
    require(not unauthorized_changed, f"UNAUTHORIZED_CHANGED_ROWS:{unauthorized_changed[:5]}")
    require(not protected_drift, f"AUTHORIZED_PROTECTED_FIELD_DRIFT:{protected_drift[:5]}")
    isolation = artifacts[regen.ISOLATION_NAME]
    require(isolation.get("unauthorized_changed_row_ids") == [], "ISOLATION_REPORTS_UNAUTHORIZED_CHANGE")
    require(sorted(isolation.get("authorized_changed_row_ids", [])) == sorted(authorized_changed), "ISOLATION_AUTHORIZED_CHANGE_MISMATCH")
    replay_rows, replay_audit = generator.build_controlled_records_with_f2_p4b_r1_regeneration_audit(
        regen.pair_count(historical_rows),
        set(member_universe["authorized_pair_ids"]),
    )
    projected_replay = regen.project_to_historical_topology(replay_rows, historical_rows)
    require(projected_replay == regenerated_rows, "STRUCTURED_SOURCE_REPLAY_MISMATCH")
    require(len(replay_audit["regenerated_members"]) == regen.AUTHORIZED_MEMBER_COUNT, "REPLAY_MEMBER_COUNT_MISMATCH")
    members_by_id = {str(row["member_id"]): row for row in artifacts[regen.MEMBERS_NAME]}
    for replay_member in replay_audit["regenerated_members"]:
        member_id = str(replay_member["member_id"])
        member = members_by_id[member_id]
        row = regenerated_by_id[member_id]
        require(member.get("regenerated_evidence") == row["evidence"], f"MEMBER_EVIDENCE_REPLAY_MISMATCH:{member_id}")
        require(member.get("structured_fact") == replay_member["structured_fact"], f"MEMBER_STRUCTURED_FACT_REPLAY_MISMATCH:{member_id}")
        require(member.get("old_text_used_for_generation") is False, f"OLD_TEXT_PROVENANCE_NOT_FALSE:{member_id}")
    return {
        "historical_row_count": len(historical_rows),
        "regenerated_row_count": len(regenerated_rows),
        "authorized_changed_row_count": len(authorized_changed),
        "unauthorized_changed_row_count": len(unauthorized_changed),
    }


def expected_member_from_rows(
    *,
    member: Mapping[str, Any],
    historical: Mapping[str, Any],
    regenerated: Mapping[str, Any],
    replay_member: Mapping[str, Any],
) -> dict[str, Any]:
    member_id = str(regenerated["id"])
    intervention = str(regenerated["intervention_type"])
    predicate = str(replay_member["semantic_predicate"])
    base = regen.AUTHORIZED_PREDICATE_BASES[predicate]
    row_delta = regen.changed_fields(historical, regenerated)
    expected_delta = ["evidence"] if intervention in {"none", "paraphrase"} else []
    require(row_delta == expected_delta, f"MEMBER_ROW_DELTA_SCOPE_MISMATCH:{member_id}")
    if intervention == "polarity_flip":
        require(
            regen.canonical_sha256(regen.ordered_row(historical))
            == regen.canonical_sha256(regen.ordered_row(regenerated)),
            f"POLARITY_FLIP_ROW_SHA_NOT_IDENTICAL:{member_id}",
        )
    template = "_paraphrase" if intervention == "paraphrase" else "_statement"
    negative = f"did not {base}" if intervention in {"none", "paraphrase"} else ""
    return {
        "schema_version": "P3W6F2P4B_R1_REGENERATED_MEMBER_V1",
        "generation_contract_version": regen.GENERATION_CONTRACT_VERSION,
        "pair_id": regenerated["pair_id"],
        "member_id": member_id,
        "intervention_type": intervention,
        "member_role": intervention,
        "structured_fact": replay_member["structured_fact"],
        "structured_fact_sha256": regen.canonical_sha256(replay_member["structured_fact"]),
        "semantic_predicate": predicate,
        "base_predicate": base,
        "negative_auxiliary_realization": negative,
        "claim": regenerated["claim"],
        "regenerated_evidence": regenerated["evidence"],
        "final_label": regenerated["final_label"],
        "frame_compatible_label": regenerated["frame_compatible_label"],
        "predicate_covered_label": regenerated["predicate_covered_label"],
        "sufficiency_label": regenerated["sufficiency_label"],
        "polarity_label": regenerated["polarity_label"],
        "primary_failure_type": regenerated["primary_failure_type"],
        "generation_root": "structured_fact",
        "generation_template": template,
        "historical_member_id": member_id,
        "historical_row_sha256": regen.canonical_sha256(regen.ordered_row(historical)),
        "regenerated_row_sha256": regen.canonical_sha256(regen.ordered_row(regenerated)),
        "row_field_delta_keys": row_delta,
        "old_text_used_for_generation": False,
        "source_authority_commit": regen.LEVEL1_FREEZE_COMMIT,
        "execution_commit": member["execution_commit"],
    }


def validate_member_artifact_independence(
    *,
    members: Sequence[Mapping[str, Any]],
    historical_rows: Sequence[Mapping[str, Any]],
    regenerated_rows: Sequence[Mapping[str, Any]],
    replay_members: Sequence[Mapping[str, Any]],
) -> None:
    historical_by_id = regen.row_map(historical_rows)
    regenerated_by_id = regen.row_map(regenerated_rows)
    replay_by_id = {str(row["member_id"]): row for row in replay_members}
    require(len(replay_by_id) == len(replay_members), "REPLAY_MEMBER_DUPLICATE_ID")
    for member in members:
        member_id = str(member["member_id"])
        require_exact_keys(member, MEMBER_FIELDS, f"MEMBER_SCHEMA_MISMATCH:{member_id}")
        require(member_id in historical_by_id, f"MEMBER_HISTORICAL_ROW_MISSING:{member_id}")
        require(member_id in regenerated_by_id, f"MEMBER_REGENERATED_ROW_MISSING:{member_id}")
        require(member_id in replay_by_id, f"MEMBER_REPLAY_ROW_MISSING:{member_id}")
        expected = expected_member_from_rows(
            member=member,
            historical=historical_by_id[member_id],
            regenerated=regenerated_by_id[member_id],
            replay_member=replay_by_id[member_id],
        )
        for field, expected_value in expected.items():
            require(
                member.get(field) == expected_value,
                f"MEMBER_ARTIFACT_FIELD_MISMATCH:{member_id}:{field}",
            )


def validate_audit_rows(
    audit_rows: Sequence[Mapping[str, Any]],
    members: Sequence[Mapping[str, Any]],
) -> None:
    member_ids = {str(row["member_id"]) for row in members}
    audit_ids = {str(row.get("member_id", "")) for row in audit_rows}
    require(audit_ids == member_ids, "AUDIT_MEMBER_ID_MISMATCH")
    members_by_id = {str(row["member_id"]): row for row in members}
    for row in audit_rows:
        member_id = str(row.get("member_id", ""))
        require_exact_keys(row, AUDIT_FIELDS, f"AUDIT_SCHEMA_MISMATCH:{member_id}")
        require(row.get("schema_version") == "P3W6F2P4B_R1_MEMBER_AUDIT_V1", f"AUDIT_SCHEMA_VERSION_MISMATCH:{member_id}")
        member = members_by_id[member_id]
        historical = row.get("historical_row")
        regenerated = row.get("regenerated_row")
        require(isinstance(historical, Mapping) and isinstance(regenerated, Mapping), f"AUDIT_ROW_PAIR_MISSING:{member_id}")
        deltas = regen.changed_fields(historical, regenerated)
        expected = ["evidence"] if row.get("intervention_type") in {"none", "paraphrase"} else []
        require(deltas == expected, f"AUDIT_INDEPENDENT_FIELD_DELTA_MISMATCH:{member_id}")
        require(member.get("regenerated_evidence") == regenerated.get("evidence"), f"AUDIT_MEMBER_EVIDENCE_MISMATCH:{member_id}")
        require(row.get("field_delta") == {field: [historical[field], regenerated[field]] for field in deltas}, f"AUDIT_FIELD_DELTA_REPORT_MISMATCH:{member_id}")


def validate_summary(summary: Mapping[str, Any], member_universe: Mapping[str, Any]) -> None:
    require(summary.get("schema_version") == "P3W6F2P4B_R1_REGENERATION_SUMMARY_V1", "SUMMARY_SCHEMA_MISMATCH")
    require(summary.get("level1_freeze_commit") == regen.LEVEL1_FREEZE_COMMIT, "LEVEL1_FREEZE_COMMIT_MISMATCH")
    require(summary.get("parent_runtime_authority_commit") == regen.PARENT_RUNTIME_AUTHORITY_COMMIT, "PARENT_RUNTIME_AUTHORITY_MISMATCH")
    require(summary.get("authorized_pair_count") == regen.AUTHORIZED_PAIR_COUNT, "SUMMARY_PAIR_COUNT_MISMATCH")
    require(summary.get("authorized_member_count") == regen.AUTHORIZED_MEMBER_COUNT, "SUMMARY_MEMBER_COUNT_MISMATCH")
    require(summary.get("polarity_flip_changed_member_count") == 0, "POLARITY_FLIP_ROW_DELTA_NOT_ZERO")
    require(len(member_universe["authorized_pair_ids"]) == summary.get("authorized_pair_count"), "SUMMARY_PAIR_UNIVERSE_MISMATCH")
    require(summary.get("artifact_set_complete") is True, "SUMMARY_ARTIFACT_SET_PRODUCER_FLAG_MISMATCH")
    require(summary.get("fail_closed_status") == "PASS", "SUMMARY_FAIL_CLOSED_PRODUCER_FLAG_MISMATCH")


def analyze_execution_dir(repo_root: Path, execution_dir: Path) -> dict[str, Any]:
    authority = regen.authenticated_frozen_authority(repo_root)
    artifacts = load_required_artifacts(execution_dir)
    member_universe = validate_member_universe(artifacts[regen.MEMBERS_NAME], authority)
    validate_summary(artifacts[regen.SUMMARY_NAME], member_universe)
    validate_base_form_coverage(artifacts[regen.COVERAGE_NAME], artifacts[regen.MEMBERS_NAME])
    validate_hash_contracts(repo_root, execution_dir, artifacts, authority)
    replay = validate_replay_and_isolation(repo_root, artifacts, member_universe)
    historical_rows = regen.load_jsonl(repo_root / regen.HISTORICAL_DATASET_PATH)
    _replay_rows, replay_audit = generator.build_controlled_records_with_f2_p4b_r1_regeneration_audit(
        regen.pair_count(historical_rows),
        set(member_universe["authorized_pair_ids"]),
    )
    validate_member_artifact_independence(
        members=artifacts[regen.MEMBERS_NAME],
        historical_rows=historical_rows,
        regenerated_rows=artifacts[regen.FULL_DATASET_NAME],
        replay_members=replay_audit["regenerated_members"],
    )
    validate_audit_rows(artifacts[regen.AUDIT_NAME], artifacts[regen.MEMBERS_NAME])
    return {
        "schema_version": ANALYZER_SCHEMA_VERSION,
        "analysis_status": "PASS",
        "execution_dir": regen.repo_relative_path(repo_root, execution_dir),
        "artifact_set_complete": True,
        "authorized_pair_count": regen.AUTHORIZED_PAIR_COUNT,
        "authorized_member_count": regen.AUTHORIZED_MEMBER_COUNT,
        "authorized_pair_ids": member_universe["authorized_pair_ids"],
        "authorized_member_ids": member_universe["authorized_member_ids"],
        "replay": replay,
        "training_admission_released": False,
        "failure_reasons": [],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze P3-W6-F2 P4-B R1 regeneration artifacts")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--execution-dir", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    execution_dir = regen.resolve_under_repo(repo_root, args.execution_dir)
    return analyze_execution_dir(repo_root, execution_dir)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = run(build_arg_parser().parse_args(argv))
    except (P4BAnalysisError, regen.P4BRegenerationError) as exc:
        raise SystemExit(f"P3W6F2P4B_R1_ANALYSIS_FAILED_CLOSED:{exc}") from exc
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
