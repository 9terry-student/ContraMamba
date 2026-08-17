#!/usr/bin/env python3
"""Scoped P4-B Stage185 predicate-realization compatibility materializer."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence


_REPO_IMPORT_ROOT = Path(__file__).resolve().parents[1]
if not any(Path(entry).resolve() == _REPO_IMPORT_ROOT for entry in sys.path if entry):
    sys.path.insert(0, str(_REPO_IMPORT_ROOT))

from scripts import analyze_reason_router_p3w6f2_p4b_r1_regeneration as analyzer
from scripts import build_stage185a_controlled_train_integrity_sidecar as stage185
from scripts import regenerate_reason_router_p3w6f2_p4b_r1_structured as regen


COMPATIBILITY_RULE_VERSION = "P3W6F2P4B_R1_STAGE185_PREDICATE_REALIZATION_COMPATIBILITY_V1"
ROWS_NAME = "p3w6f2_p4b_r1_stage185_predicate_realization_compatibility_rows.jsonl"
SUMMARY_NAME = "p3w6f2_p4b_r1_stage185_predicate_realization_compatibility_summary.json"
PROVENANCE_NAME = "p3w6f2_p4b_r1_stage185_predicate_realization_compatibility_provenance_manifest.json"
EXPECTED_ARTIFACT_NAMES = {ROWS_NAME, SUMMARY_NAME, PROVENANCE_NAME}
STAGE185_SOURCE_SCRIPT = "scripts/build_stage185a_controlled_train_integrity_sidecar.py"


class Stage185CompatibilityError(RuntimeError):
    """Fail-closed Stage185 compatibility rejection."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage185CompatibilityError(message)


def historical_malformed_span(predicate: str) -> str:
    return f"did not {predicate}"


def regenerated_base_span(base_predicate: str) -> str:
    return f"did not {base_predicate}"


def verify_stage185_source(repo_root: Path) -> str:
    observed = regen.file_sha256(repo_root / STAGE185_SOURCE_SCRIPT)
    require(observed == regen.EXPECTED_STAGE185_SOURCE_SHA256, "STAGE185_SOURCE_SHA_MISMATCH")
    return observed


def raw_stage185_changed_axes(
    *,
    historical: Mapping[str, Any],
    regenerated: Mapping[str, Any],
    fact: Mapping[str, Any],
    intended: set[str],
) -> list[str]:
    return sorted(
        stage185.changed_axes(
            dict(regenerated),
            dict(historical),
            dict(fact),
            intended,
        )
    )


def build_compatibility_rows(
    *,
    members: Sequence[Mapping[str, Any]],
    audit_rows: Sequence[Mapping[str, Any]],
    authorized_member_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    audit_by_id = {str(row["member_id"]): row for row in audit_rows}
    rows: list[dict[str, Any]] = []
    for member in members:
        member_id = str(member["member_id"])
        intervention = str(member["intervention_type"])
        if authorized_member_ids is not None:
            require(member_id in authorized_member_ids, f"COMPATIBILITY_UNAUTHORIZED_MEMBER:{member_id}")
        audit = audit_by_id.get(member_id)
        require(isinstance(audit, Mapping), f"AUDIT_ROW_MISSING:{member_id}")
        historical = audit["historical_row"]
        regenerated = audit["regenerated_row"]
        fact = member["structured_fact"]
        predicate = str(member["semantic_predicate"])
        base = str(member["base_predicate"])
        require(str(member["pair_id"]) == str(historical["pair_id"]), f"COMPATIBILITY_PAIR_ID_MISMATCH:{member_id}")
        require(member_id == str(historical["id"]), f"COMPATIBILITY_MEMBER_ID_MISMATCH:{member_id}")
        require(intervention in regen.AUTHORIZED_INTERVENTIONS, f"COMPATIBILITY_UNAUTHORIZED_INTERVENTION:{member_id}")
        raw_expected_axes = [] if intervention in {"none", "paraphrase"} else ["polarity"]
        raw_changed_axes = raw_stage185_changed_axes(
            historical=historical,
            regenerated=regenerated,
            fact=fact,
            intended=set(raw_expected_axes),
        )
        conditions: list[str] = []
        failures: list[str] = []
        if predicate != fact.get("predicate"):
            failures.append("structured_semantic_predicate_changed")
        if regen.AUTHORIZED_PREDICATE_BASES.get(predicate) != base:
            failures.append("authorized_base_mapping_mismatch")
        if intervention in {"none", "paraphrase"}:
            if "predicate" not in raw_changed_axes:
                failures.append("raw_stage185_predicate_axis_not_observed")
            if historical_malformed_span(predicate) not in str(historical["evidence"]):
                failures.append("historical_malformed_span_missing")
            if regenerated_base_span(base) not in str(regenerated["evidence"]):
                failures.append("regenerated_base_span_missing")
        else:
            if raw_changed_axes:
                failures.append("polarity_flip_raw_stage185_delta")
            if historical != regenerated:
                failures.append("polarity_flip_row_delta")
        for field in regen.DATASET_FIELDS:
            if field == "evidence" and intervention in {"none", "paraphrase"}:
                continue
            if historical.get(field) != regenerated.get(field):
                failures.append(f"unauthorized_field_delta:{field}")
        semantic_preserved = not any(
            code
            for code in failures
            if code
            in {
                "structured_semantic_predicate_changed",
                "authorized_base_mapping_mismatch",
            }
            or code.startswith("unauthorized_field_delta:")
        )
        permitted_delta = len(failures) == 0
        effective_status = "PASS" if permitted_delta else "FAIL"
        if intervention in {"none", "paraphrase"}:
            conditions.extend(
                [
                    "raw_stage185_predicate_axis_observation_retained",
                    "historical_did_not_inflected_predicate_observed",
                    "regenerated_did_not_base_predicate_observed",
                    "structured_semantic_predicate_preserved",
                ]
            )
        else:
            conditions.extend(
                [
                    "raw_stage185_polarity_observation_retained",
                    "polarity_flip_affirmative_row_byte_identical",
                    "structured_semantic_predicate_preserved",
                ]
            )
        rows.append(
            {
                "schema_version": "P3W6F2P4B_R1_STAGE185_COMPATIBILITY_ROW_V1",
                "compatibility_rule_version": COMPATIBILITY_RULE_VERSION,
                "pair_id": member["pair_id"],
                "member_id": member_id,
                "intervention_type": intervention,
                "raw_stage185_changed_axes": raw_changed_axes,
                "raw_stage185_expected_axes": raw_expected_axes,
                "raw_stage185_statuses": {
                    "predicate_axis_observed": "predicate" in raw_changed_axes,
                    "raw_observation_preserved": True,
                    "stage185_v1_mutated": False,
                },
                "historical_semantic_predicate": predicate,
                "regenerated_negative_base_surface": regenerated_base_span(base) if intervention in {"none", "paraphrase"} else "",
                "structured_fact": dict(fact),
                "semantic_slot_preservation": {
                    "status": "PASS" if semantic_preserved else "FAIL",
                    "semantic_predicate": predicate,
                    "base_predicate": base,
                },
                "permitted_predicate_realization_delta": permitted_delta,
                "effective_compatibility_status": effective_status,
                "effective_reason_codes": conditions if permitted_delta else failures,
                "training_admission_effect": {
                    "training_admission_released": False,
                    "level3_admission_released": False,
                },
            }
        )
    return rows


def build_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pass_rows = [row for row in rows if row["effective_compatibility_status"] == "PASS"]
    fail_rows = [row for row in rows if row["effective_compatibility_status"] == "FAIL"]
    unresolved_rows = [row for row in rows if row["effective_compatibility_status"] not in {"PASS", "FAIL"}]
    raw_predicate_count = sum(
        1 for row in rows if "predicate" in row.get("raw_stage185_changed_axes", [])
    )
    permitted_count = sum(
        1 for row in rows if row.get("permitted_predicate_realization_delta") is True
    )
    status = "PASS" if len(pass_rows) == len(rows) and len(rows) == regen.AUTHORIZED_MEMBER_COUNT else "FAIL"
    return {
        "schema_version": "P3W6F2P4B_R1_STAGE185_COMPATIBILITY_SUMMARY_V1",
        "compatibility_rule_version": COMPATIBILITY_RULE_VERSION,
        "row_count": len(rows),
        "authorized_pair_count": len({row["pair_id"] for row in rows}),
        "authorized_member_count": len(rows),
        "raw_stage185_predicate_axis_observation_count": raw_predicate_count,
        "permitted_predicate_realization_delta_count": permitted_count,
        "compatibility_pass_count": len(pass_rows),
        "compatibility_fail_count": len(fail_rows),
        "compatibility_unresolved_count": len(unresolved_rows),
        "stage185_v1_mutated": False,
        "historical_authority_weakened": False,
        "training_admission_released": False,
        "compatibility_gate_status": status,
        "failure_reasons": [] if status == "PASS" else ["compatibility_rows_not_all_pass"],
    }


def build_provenance(
    *,
    repo_root: Path,
    execution_dir: Path,
    output_dir: Path,
    rows_sha256: str,
    summary_sha256: str,
    coverage_path: Path,
) -> dict[str, Any]:
    stage185_sha = verify_stage185_source(repo_root)
    return {
        "schema_version": "P3W6F2P4B_R1_STAGE185_COMPATIBILITY_PROVENANCE_V1",
        "compatibility_rule_version": COMPATIBILITY_RULE_VERSION,
        "stage185_source_script": STAGE185_SOURCE_SCRIPT,
        "stage185_source_script_sha256": stage185_sha,
        "historical_stage185_authority": "STAGE185_V1_RAW_OBSERVATION_RETAINED_NOT_MUTATED",
        "historical_stage185_authority_sha256": stage185_sha,
        "regenerated_dataset_path": regen.repo_relative_path(repo_root, execution_dir / regen.FULL_DATASET_NAME),
        "regenerated_dataset_sha256": regen.file_sha256(execution_dir / regen.FULL_DATASET_NAME),
        "structured_source_producer": f"{regen.GENERATOR_SOURCE_PATH}::fact_templates_for_count",
        "structured_source_producer_sha256": regen.file_sha256(repo_root / regen.GENERATOR_SOURCE_PATH),
        "base_form_coverage_path": regen.repo_relative_path(repo_root, coverage_path),
        "base_form_coverage_sha256": regen.file_sha256(coverage_path),
        "compatibility_rows_path": regen.repo_relative_path(repo_root, output_dir / ROWS_NAME),
        "compatibility_rows_sha256": rows_sha256,
        "compatibility_summary_path": regen.repo_relative_path(repo_root, output_dir / SUMMARY_NAME),
        "compatibility_summary_sha256": summary_sha256,
        "created_at_utc": "DETERMINISTIC_REPLAY_NO_WALL_CLOCK",
    }


def validate_compatibility_payloads(payloads: Mapping[str, bytes]) -> None:
    require(set(payloads) == EXPECTED_ARTIFACT_NAMES, "COMPATIBILITY_PAYLOAD_SET_MISMATCH")
    row_lines = payloads[ROWS_NAME].decode("utf-8").splitlines()
    require(bool(row_lines), "COMPATIBILITY_ROWS_EMPTY")
    rows = [json.loads(line) for line in row_lines]
    summary = json.loads(payloads[SUMMARY_NAME].decode("utf-8"))
    provenance = json.loads(payloads[PROVENANCE_NAME].decode("utf-8"))
    require(all(row.get("schema_version") == "P3W6F2P4B_R1_STAGE185_COMPATIBILITY_ROW_V1" for row in rows), "COMPATIBILITY_ROWS_SCHEMA_MISMATCH")
    require(summary.get("schema_version") == "P3W6F2P4B_R1_STAGE185_COMPATIBILITY_SUMMARY_V1", "COMPATIBILITY_SUMMARY_SCHEMA_MISMATCH")
    require(provenance.get("schema_version") == "P3W6F2P4B_R1_STAGE185_COMPATIBILITY_PROVENANCE_V1", "COMPATIBILITY_PROVENANCE_SCHEMA_MISMATCH")
    require(summary.get("compatibility_gate_status") == "PASS", "COMPATIBILITY_GATE_NOT_PASS")
    require(summary.get("training_admission_released") is False, "COMPATIBILITY_TRAINING_ADMISSION_RELEASED")
    require(all(row.get("training_admission_effect", {}).get("training_admission_released") is False for row in rows), "COMPATIBILITY_ROW_TRAINING_ADMISSION_RELEASED")
    require(provenance.get("compatibility_rows_sha256") == regen.sha256_bytes(payloads[ROWS_NAME]), "COMPATIBILITY_ROWS_SHA_MISMATCH")
    require(provenance.get("compatibility_summary_sha256") == regen.sha256_bytes(payloads[SUMMARY_NAME]), "COMPATIBILITY_SUMMARY_SHA_MISMATCH")
    require(provenance.get("stage185_source_script") == STAGE185_SOURCE_SCRIPT, "COMPATIBILITY_STAGE185_SOURCE_IDENTITY_MISMATCH")
    require(provenance.get("stage185_source_script_sha256") == regen.EXPECTED_STAGE185_SOURCE_SHA256, "COMPATIBILITY_STAGE185_SOURCE_SHA_MISMATCH")


def publish_artifacts(
    output_dir: Path,
    payloads: Mapping[str, bytes],
    *,
    staging_dir_name: str | None = None,
    promote_directory: Any | None = None,
    remove_tree: Any | None = None,
) -> str:
    validate_compatibility_payloads(payloads)
    existing = {name for name in EXPECTED_ARTIFACT_NAMES if (output_dir / name).exists()}
    if existing:
        require(existing == EXPECTED_ARTIFACT_NAMES, "COMPATIBILITY_OUTPUT_PARTIAL_PREEXISTING")
        conflicts = [
            name
            for name, payload in payloads.items()
            if (output_dir / name).read_bytes() != payload
        ]
        require(not conflicts, f"COMPATIBILITY_OUTPUT_CONFLICT:{conflicts}")
        return "IDEMPOTENT_PASS"

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        require(output_dir.is_dir(), "COMPATIBILITY_OUTPUT_PATH_NOT_DIRECTORY")
        unexpected = {
            entry.name
            for entry in output_dir.iterdir()
            if entry.name in EXPECTED_ARTIFACT_NAMES
        }
        require(not unexpected, "COMPATIBILITY_OUTPUT_PARTIAL_PREEXISTING")
        noncompat = {entry.name for entry in output_dir.iterdir()}
        require(noncompat <= regen.EXPECTED_ARTIFACT_NAMES, f"COMPATIBILITY_OUTPUT_UNEXPECTED_BASE_ARTIFACTS:{sorted(noncompat - regen.EXPECTED_ARTIFACT_NAMES)}")
    staging_dir = output_dir.parent / (
        staging_dir_name
        or f".{output_dir.name}.p4b-stage185-compatibility-staging-{uuid.uuid4().hex}"
    )
    backup_dir = output_dir.parent / f".{output_dir.name}.p4b-stage185-compatibility-backup-{uuid.uuid4().hex}"
    require(not staging_dir.exists(), "COMPATIBILITY_STAGING_EXISTS")
    require(not backup_dir.exists(), "COMPATIBILITY_BACKUP_EXISTS")
    remover = remove_tree or shutil.rmtree
    promoter = promote_directory or (lambda source, target: source.replace(target))
    try:
        staging_dir.mkdir()
        if output_dir.exists():
            for entry in output_dir.iterdir():
                target = staging_dir / entry.name
                if entry.is_dir():
                    shutil.copytree(entry, target)
                else:
                    shutil.copy2(entry, target)
        for name, payload in payloads.items():
            (staging_dir / name).write_bytes(payload)
        staged_compat = {name for name in EXPECTED_ARTIFACT_NAMES if (staging_dir / name).is_file()}
        require(staged_compat == EXPECTED_ARTIFACT_NAMES, "COMPATIBILITY_STAGING_SET_MISMATCH")
        validate_compatibility_payloads({name: (staging_dir / name).read_bytes() for name in EXPECTED_ARTIFACT_NAMES})
        if output_dir.exists():
            output_dir.replace(backup_dir)
        try:
            promoter(staging_dir, output_dir)
        except Exception:
            if output_dir.exists():
                remover(output_dir, ignore_errors=True)
            if backup_dir.exists():
                backup_dir.replace(output_dir)
            raise
        if backup_dir.exists():
            remover(backup_dir, ignore_errors=True)
    except Exception:
        if staging_dir.exists():
            remover(staging_dir, ignore_errors=True)
        raise
    require(all((output_dir / name).is_file() for name in EXPECTED_ARTIFACT_NAMES), "COMPATIBILITY_OUTPUT_SET_MISMATCH")
    if regen.EXPECTED_ARTIFACT_NAMES <= {entry.name for entry in output_dir.iterdir()}:
        require(
            {entry.name for entry in output_dir.iterdir()}
            == regen.EXPECTED_ARTIFACT_NAMES | EXPECTED_ARTIFACT_NAMES,
            "COMPATIBILITY_COMPLETE_OUTPUT_SET_MISMATCH",
        )
    return "PUBLISHED"


def materialize(repo_root: Path, execution_dir: Path, output_dir: Path) -> dict[str, Any]:
    verify_stage185_source(repo_root)
    analysis = analyzer.analyze_execution_dir(repo_root, execution_dir)
    require(analysis["analysis_status"] == "PASS", "REGENERATION_ANALYSIS_NOT_PASS")
    artifacts = analyzer.load_required_artifacts(execution_dir)
    rows = build_compatibility_rows(
        members=artifacts[regen.MEMBERS_NAME],
        audit_rows=artifacts[regen.AUDIT_NAME],
        authorized_member_ids=set(analysis["authorized_member_ids"]),
    )
    summary = build_summary(rows)
    require(summary["training_admission_released"] is False, "TRAINING_ADMISSION_RELEASED")
    rows_bytes = regen.deterministic_jsonl_bytes(rows)
    summary_bytes = regen.deterministic_json_bytes(summary)
    provenance = build_provenance(
        repo_root=repo_root,
        execution_dir=execution_dir,
        output_dir=output_dir,
        rows_sha256=regen.sha256_bytes(rows_bytes),
        summary_sha256=regen.sha256_bytes(summary_bytes),
        coverage_path=execution_dir / regen.COVERAGE_NAME,
    )
    payloads = {
        ROWS_NAME: rows_bytes,
        SUMMARY_NAME: summary_bytes,
        PROVENANCE_NAME: regen.deterministic_json_bytes(provenance),
    }
    publish_status = publish_artifacts(output_dir, payloads)
    return {
        "status": summary["compatibility_gate_status"],
        "publish_status": publish_status,
        "summary": summary,
        "training_admission_released": False,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize P4-B R1 Stage185 compatibility artifacts")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--execution-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    execution_dir = regen.resolve_under_repo(repo_root, args.execution_dir)
    output_dir = regen.resolve_under_repo(repo_root, args.output_dir)
    return materialize(repo_root, execution_dir, output_dir)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = run(build_arg_parser().parse_args(argv))
    except (Stage185CompatibilityError, analyzer.P4BAnalysisError, regen.P4BRegenerationError) as exc:
        raise SystemExit(f"P3W6F2P4B_R1_STAGE185_COMPATIBILITY_FAILED_CLOSED:{exc}") from exc
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
