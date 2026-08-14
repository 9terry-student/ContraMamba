from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from scripts import reason_router_p3w6f2_manual_review as f2


def source_row(pair_id: str = "pair_a") -> dict[str, str]:
    return {
        "pair_id": pair_id,
        "canonical_none_row_id": f"{pair_id}__none",
        "paraphrase_row_id": f"{pair_id}__paraphrase",
        "polarity_flip_row_id": f"{pair_id}__polarity_flip",
        "canonical_final_label": "REFUTE",
        "paraphrase_final_label": "REFUTE",
        "polarity_flip_final_label": "SUPPORT",
        "canonical_claim": "A restored B.",
        "paraphrase_claim": "A restored B.",
        "polarity_flip_claim": "A restored B.",
        "canonical_evidence": "A did not restored B.",
        "paraphrase_evidence": "B was not restored by A.",
        "polarity_flip_evidence": "A restored B.",
        "canonical_grammar_status": "FAIL",
        "paraphrase_grammar_status": "FAIL",
        "polarity_flip_grammar_status": "PASS",
        "canonical_reason_codes": '["DID_NOT_INFLECTED_PREDICATE","GRAMMAR_TEMPLATE_FAIL"]',
        "paraphrase_reason_codes": '["CANONICAL_ROW_KNOWN_GENERATOR_DEFECT","DID_NOT_INFLECTED_PREDICATE","GRAMMAR_TEMPLATE_FAIL"]',
        "polarity_flip_reason_codes": '["CANONICAL_ROW_KNOWN_GENERATOR_DEFECT"]',
        "canonical_claim_text_diff_summary": "no text difference",
        "paraphrase_claim_text_diff_summary": "no text difference",
        "polarity_flip_claim_text_diff_summary": "no text difference",
        "canonical_evidence_text_diff_summary": "negation-only difference",
        "paraphrase_evidence_text_diff_summary": "multiple changes",
        "polarity_flip_evidence_text_diff_summary": "no text difference",
        "automatic_root_cause_class": "F2_CANONICAL_DEFECT_PROPAGATED_TO_DERIVATIVES",
        "automatic_evidence": "mechanical fixture",
    }


def authority(rows: list[dict[str, str]] | None = None, tmp_path: Path | None = None) -> f2.Authority:
    rows = rows or [source_row()]
    repo_root = tmp_path or Path(".")
    return f2.Authority(
        repo_root=repo_root,
        manifest={},
        source_rows=rows,
        pair_records=[],
        summary={},
        source_sha256_by_pair_id={row["pair_id"]: f2.compute_source_record_sha256(row) for row in rows},
        p3w4_artifact_commit=f2.P3W4_RESULT_AUTHORITY_COMMIT,
        input_artifact_paths={"template": "template.csv", "pairs": "pairs.jsonl", "summary": "summary.json"},
        input_artifact_sha256={"template.csv": "a" * 64, "pairs.jsonl": "b" * 64, "summary.json": "c" * 64},
        source_template_fields=[*f2.SOURCE_FIELDS, *f2.HUMAN_FIELDS],
    )


def pair_authority_record(row: dict[str, str], family: str = "F2") -> dict[str, object]:
    return {
        "pair_id": row["pair_id"],
        "family": family,
        "members": {
            "canonical": {
                "source_row": {
                    "id": row["canonical_none_row_id"],
                    "final_label": row["canonical_final_label"],
                }
            },
            "paraphrase": {
                "source_row": {
                    "id": row["paraphrase_row_id"],
                    "final_label": row["paraphrase_final_label"],
                }
            },
            "polarity_flip": {
                "source_row": {
                    "id": row["polarity_flip_row_id"],
                    "final_label": row["polarity_flip_final_label"],
                }
            },
        },
    }


def pair_summary(pair_count: int, member_count: int) -> dict[str, object]:
    return {
        "aggregates": {
            "family_counts": {
                "F2_pair_count": pair_count,
                "F2_complete_triple_members": member_count,
            }
        }
    }


def external_wip_path(tmp_path: Path) -> Path:
    return tmp_path.parent / f"{tmp_path.name}_p3w6f2_review_wip.jsonl"


def prepare_reports_dir(repo_root: Path) -> None:
    (repo_root / "reports").mkdir(exist_ok=True)


def review_record(auth: f2.Authority, pair_id: str = "pair_a", **overrides: str) -> dict[str, str]:
    record = f2.make_review_record(
        auth,
        pair_id=pair_id,
        reviewer_id=overrides.pop("reviewer_id", "reviewer"),
        canonical_semantics=overrides.pop("human_canonical_semantics", "VALID"),
        paraphrase_semantics=overrides.pop("human_paraphrase_semantics", "VALID"),
        polarity_flip_semantics=overrides.pop("human_polarity_flip_semantics", "VALID"),
        grammar_validity=overrides.pop("human_grammar_validity", "CANONICAL_ONLY_DEFECT"),
        notes=overrides.pop("human_notes", ""),
        clock=lambda: datetime(2026, 8, 14, 0, 0, tzinfo=UTC),
    )
    record.update(overrides)
    return record


def structural_row(pair_id: str = "pair_a", predicate: str = "restored") -> dict[str, str]:
    row = source_row(pair_id)
    claim = f"A {predicate} B."
    row.update(
        {
            "canonical_claim": claim,
            "paraphrase_claim": claim,
            "polarity_flip_claim": claim,
            "canonical_evidence": f"A did not {predicate} B.",
            "paraphrase_evidence": f"During March, A did not {predicate} B.",
            "polarity_flip_evidence": claim,
            "canonical_reason_codes": '["DID_NOT_INFLECTED_PREDICATE","GRAMMAR_TEMPLATE_FAIL"]',
            "paraphrase_reason_codes": '["CANONICAL_ROW_KNOWN_GENERATOR_DEFECT","DID_NOT_INFLECTED_PREDICATE","GRAMMAR_TEMPLATE_FAIL"]',
            "polarity_flip_reason_codes": '["CANONICAL_ROW_KNOWN_GENERATOR_DEFECT"]',
        }
    )
    return row


def ai_prescreen_record(auth: f2.Authority, pair_id: str = "pair_a", **overrides: str) -> dict[str, str]:
    record = {
        "pair_id": pair_id,
        "source_record_sha256": auth.source_sha256_by_pair_id[pair_id],
        "ai_canonical_semantics_suggestion": "V",
        "ai_paraphrase_semantics_suggestion": "V",
        "ai_polarity_flip_semantics_suggestion": "V",
        "ai_grammar_validity_suggestion": "M",
        "ai_triage_status": "CLEAR_SUGGESTION",
        "ai_prescreen_protocol_version": f2.AI_PRESCREEN_PROTOCOL_VERSION,
        "ai_prescreen_model_or_system_id": "ai-prescreen-test",
        "ai_prescreen_created_at_utc": "2026-08-14T00:00:00Z",
    }
    record.update(overrides)
    return record


def audit_artifact(auth: f2.Authority, eligible: list[str] | None = None, exceptions: list[str] | None = None) -> dict[str, object]:
    eligible = eligible if eligible is not None else ["pair_a"]
    exceptions = exceptions if exceptions is not None else []
    payload = {
        "schema_version": f2.STRUCTURAL_AUDIT_SCHEMA_VERSION,
        "structural_gate_version": f2.STRUCTURAL_GATE_VERSION,
        "source_authority_identity": f2.source_authority_identity(auth),
        "authorized_pair_count": f2.EXPECTED_PAIR_COUNT,
        "authorized_member_count": f2.EXPECTED_MEMBER_COUNT,
        "source_record_sha256_by_pair": {pair_id: auth.source_sha256_by_pair_id[pair_id] for pair_id in auth.ordered_pair_ids},
        "ai_prescreen_artifact_sha256": "a" * 64,
        "ai_prescreen_protocol_version": f2.AI_PRESCREEN_PROTOCOL_VERSION,
        "validated_individual_wip_state_sha256": "b" * 64,
        "required_individual_audit_pair_ids": f2.INDIVIDUALLY_REVIEWED_AUDIT_PAIR_IDS,
        "eligible_pair_ids": eligible,
        "exception_pair_ids": exceptions,
        "structural_gate_result_by_pair": {
            pair_id: {"eligible": pair_id in set(eligible), "errors": [] if pair_id in set(eligible) else ["STRUCTURAL_COHORT_GATE_FAILED"]}
            for pair_id in auth.ordered_pair_ids
        },
        "overall_structural_gate_result": "PASS" if not exceptions else "PASS_WITH_EXCEPTIONS",
    }
    artifact = {
        **payload,
        "audit_created_at_utc": "2026-08-14T00:00:00Z",
        "audit_payload_sha256": f2.canonical_json_sha256(payload),
    }
    return {field: artifact[field] for field in f2.STRUCTURAL_AUDIT_FIELDS}


def cohort_artifact(auth: f2.Authority, confirmed: list[str] | None = None, exceptions: list[str] | None = None) -> dict[str, object]:
    audit = audit_artifact(auth, confirmed, exceptions)
    return f2.cohort_confirmation_artifact(
        authority=auth,
        audit_artifact=audit,
        audit_path=auth.repo_root.parent / "audit.json",
        reviewer_id="reviewer",
        authority_recorded_at_utc="2026-08-14T00:00:00Z",
    )


def cohort_id(auth: f2.Authority, confirmed: list[str] | None = None, exceptions: list[str] | None = None) -> str:
    return str(cohort_artifact(auth, confirmed, exceptions)["cohort_confirmation_id"])


def cohort_confirmations(auth: f2.Authority, confirmed: list[str] | None = None, exceptions: list[str] | None = None) -> dict[str, f2.CohortConfirmationAuthority]:
    artifact = cohort_artifact(auth, confirmed, exceptions)
    authority_artifact = f2.cohort_authority_from_artifact(artifact)
    return {authority_artifact.cohort_confirmation_id: authority_artifact}


def audit_authority(tmp_path: Path) -> f2.Authority:
    predicates: list[str] = []
    for predicate, count in f2.EXPECTED_AUDIT_PREDICATE_COVERAGE.items():
        predicates.extend([predicate] * count)
    rows = [structural_row(pair_id, predicate) for pair_id, predicate in zip(f2.INDIVIDUALLY_REVIEWED_AUDIT_PAIR_IDS, predicates, strict=True)]
    rows.append(structural_row("generated_fact_999", "restored"))
    auth = authority(rows, tmp_path)
    return f2.Authority(
        repo_root=auth.repo_root,
        manifest=auth.manifest,
        source_rows=auth.source_rows,
        pair_records=[pair_authority_record(row) for row in auth.source_rows],
        summary=pair_summary(pair_count=len(auth.source_rows), member_count=len(auth.source_rows) * 3),
        source_sha256_by_pair_id=auth.source_sha256_by_pair_id,
        p3w4_artifact_commit=auth.p3w4_artifact_commit,
        input_artifact_paths=auth.input_artifact_paths,
        input_artifact_sha256=auth.input_artifact_sha256,
        source_template_fields=auth.source_template_fields,
    )


def write_full_ai_prescreen(path: Path, auth: f2.Authority, **overrides: str) -> None:
    records = []
    for pair_id in auth.ordered_pair_ids:
        record = ai_prescreen_record(auth, pair_id)
        if pair_id in overrides:
            record["ai_triage_status"] = overrides[pair_id]
        records.append(record)
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def write_template(rows: list[dict[str, str]], fields: list[str], human_values: dict[str, str] | None = None) -> bytes:
    human_values = human_values or {}
    from io import StringIO

    handle = StringIO()
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        out = {field: row.get(field, human_values.get(field, "")) for field in fields}
        writer.writerow(out)
    return handle.getvalue().encode("utf-8")


def test_source_record_hash_v1_exact_known_authority_first_row_fixture():
    literal_values = [
        "generated_fact_152",
        "generated_fact_152__none",
        "generated_fact_152__paraphrase",
        "generated_fact_152__polarity_flip",
        "REFUTE",
        "REFUTE",
        "SUPPORT",
        "Dr Bianca Flores, the research lead, restored the Meridian initiative 152 in Beacon Harbor during March.",
        "Dr Bianca Flores, the research lead, restored the Meridian initiative 152 in Beacon Harbor during March.",
        "Dr Bianca Flores, the research lead, restored the Meridian initiative 152 in Beacon Harbor during March.",
        "Dr Bianca Flores, the research lead, did not restored the Meridian initiative 152 in Beacon Harbor during March.",
        "During March in Beacon Harbor, Dr Bianca Flores acting as research lead did not restored the Meridian initiative 152.",
        "Dr Bianca Flores, the research lead, restored the Meridian initiative 152 in Beacon Harbor during March.",
        "FAIL",
        "FAIL",
        "PASS",
        '["DID_NOT_INFLECTED_PREDICATE","GRAMMAR_TEMPLATE_FAIL"]',
        '["CANONICAL_ROW_KNOWN_GENERATOR_DEFECT","DID_NOT_INFLECTED_PREDICATE","GRAMMAR_TEMPLATE_FAIL"]',
        '["CANONICAL_ROW_KNOWN_GENERATOR_DEFECT"]',
        "no text difference",
        "no text difference",
        "no text difference",
        "no text difference",
        "multiple changes",
        "negation-only difference",
        "F2_CANONICAL_DEFECT_PROPAGATED_TO_DERIVATIVES",
        "canonical grammar failure propagates to paraphrase canonical_status UNRESOLVED",
    ]
    row = dict(zip(f2.SOURCE_FIELDS, literal_values, strict=True))
    assert f2.compute_source_record_sha256(row) == "b4127d435bb063fc8dca2213a753d23382f528014703736bd0d4c56ade836154"


def test_production_f2_static_authority_constants_are_not_miniaturized():
    assert f2.EXPECTED_PAIR_COUNT == 119
    assert f2.EXPECTED_MEMBER_COUNT == 357
    assert f2.INDIVIDUALLY_REVIEWED_AUDIT_PAIR_IDS == [
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
    assert f2.EXPECTED_AUDIT_PREDICATE_COVERAGE == {
        "restored": 4,
        "selected": 4,
        "approved": 4,
        "delivered": 3,
        "published": 2,
        "opened": 2,
        "launched": 1,
    }


def test_make_review_record_valid_empty_error_path_succeeds():
    auth = authority()
    record = f2.make_review_record(
        auth,
        pair_id="pair_a",
        reviewer_id="reviewer",
        canonical_semantics="VALID",
        paraphrase_semantics="VALID",
        polarity_flip_semantics="VALID",
        grammar_validity="CANONICAL_ONLY_DEFECT",
        notes="",
        clock=lambda: datetime(2026, 8, 14, 0, 0, tzinfo=UTC),
    )

    assert f2.validate_review_record(record, auth) == []
    assert record["human_authority_decision"] == "CANONICAL_TEXTUAL_REPAIR_CANDIDATE"
    assert record["review_method"] == f2.INDIVIDUAL_REVIEW_METHOD
    assert record["cohort_confirmation_id"] == f2.NO_COHORT_LINKAGE
    assert record["reviewed_at_utc"] == record["authority_recorded_at_utc"]
    assert record["human_review_time_provenance"] == f2.CAPTURED_IN_RECORD
    assert record["record_origin"] == f2.CLI_INDIVIDUAL_RECORD


@pytest.mark.parametrize("field", ["human_canonical_semantics", "human_paraphrase_semantics", "human_polarity_flip_semantics"])
def test_semantic_enum_rejection(field):
    auth = authority()
    record = review_record(auth)
    record[field] = "MAYBE"
    assert "INVALID_SEMANTIC_ENUM" in f2.validate_review_record(record, auth)


def test_grammar_enum_rejection():
    auth = authority()
    record = review_record(auth)
    record["human_grammar_validity"] = "BAD_GRAMMAR"
    assert "INVALID_GRAMMAR_ENUM" in f2.validate_review_record(record, auth)


@pytest.mark.parametrize("canonical", sorted(f2.SEMANTIC_ENUMS))
@pytest.mark.parametrize("paraphrase", sorted(f2.SEMANTIC_ENUMS))
@pytest.mark.parametrize("polarity", sorted(f2.SEMANTIC_ENUMS))
@pytest.mark.parametrize("grammar", sorted(f2.GRAMMAR_ENUMS))
def test_full_compatibility_matrix_behavior(canonical, paraphrase, polarity, grammar):
    decision = f2.derive_authority_decision(canonical, paraphrase, polarity, grammar)
    semantics = [canonical, paraphrase, polarity]
    if "UNCLEAR" in semantics or grammar == "UNCLEAR":
        assert decision == "INSUFFICIENT_EVIDENCE_KEEP_BLOCKED"
    elif "INVALID" in semantics:
        assert decision == "SEMANTIC_CONFLICT"
    elif grammar == "CANONICAL_ONLY_DEFECT":
        assert decision == "CANONICAL_TEXTUAL_REPAIR_CANDIDATE"
    elif grammar == "MULTI_MEMBER_DEFECT":
        assert decision == "CANONICAL_REGENERATION_REQUIRED"
    else:
        assert decision == "NO_REPRODUCIBLE_DEFECT_KEEP_BLOCKED"


def test_unclear_precedence_over_invalid():
    assert f2.derive_authority_decision("UNCLEAR", "INVALID", "VALID", "CANONICAL_ONLY_DEFECT") == "INSUFFICIENT_EVIDENCE_KEEP_BLOCKED"


def test_invalid_precedence_over_all_valid_grammar_routing():
    assert f2.derive_authority_decision("INVALID", "VALID", "VALID", "CANONICAL_ONLY_DEFECT") == "SEMANTIC_CONFLICT"


@pytest.mark.parametrize(
    "decision,notes_required",
    [
        ("SEMANTIC_CONFLICT", True),
        ("INSUFFICIENT_EVIDENCE_KEEP_BLOCKED", True),
        ("NO_REPRODUCIBLE_DEFECT_KEEP_BLOCKED", True),
        ("CANONICAL_TEXTUAL_REPAIR_CANDIDATE", False),
        ("CANONICAL_REGENERATION_REQUIRED", False),
    ],
)
def test_notes_required_cases(decision, notes_required):
    record = {
        "human_canonical_semantics": "VALID",
        "human_paraphrase_semantics": "VALID",
        "human_polarity_flip_semantics": "VALID",
        "human_grammar_validity": "CANONICAL_ONLY_DEFECT",
        "human_authority_decision": decision,
    }
    assert f2.notes_required(record) is notes_required


def test_notes_optional_positive_candidates():
    auth = authority()
    record = review_record(auth, human_notes="")
    assert f2.validate_review_record(record, auth) == []


def test_reviewer_id_empty_rejection():
    auth = authority()
    with pytest.raises(f2.ReviewInfrastructureError, match="MISSING_REVIEWER_ID"):
        f2.make_review_record(auth, "pair_a", "", "VALID", "VALID", "VALID", "CANONICAL_ONLY_DEFECT", "")


def test_reviewer_id_untrimmed_rejection():
    auth = authority()
    with pytest.raises(f2.ReviewInfrastructureError, match="MISSING_REVIEWER_ID"):
        f2.make_review_record(auth, "pair_a", " reviewer", "VALID", "VALID", "VALID", "CANONICAL_ONLY_DEFECT", "")


def test_review_protocol_mismatch():
    auth = authority()
    record = review_record(auth)
    record["review_protocol_version"] = "OLD"
    assert "INVALID_REVIEW_PROTOCOL_VERSION" in f2.validate_review_record(record, auth)


def test_v1_protocol_record_is_not_silently_accepted_as_v2():
    auth = authority()
    record = review_record(auth)
    record["review_protocol_version"] = f2.P3W5_REVIEW_PROTOCOL_VERSION
    assert "INVALID_REVIEW_PROTOCOL_VERSION" in f2.validate_review_record(record, auth)


def test_v2_record_requires_review_method():
    auth = authority()
    record = review_record(auth)
    del record["review_method"]
    assert "INVALID_REVIEW_METHOD" in f2.validate_review_record(record, auth)


def test_invalid_review_method_rejected():
    auth = authority()
    record = review_record(auth)
    record["review_method"] = "AUTO_ACCEPT"
    assert "INVALID_REVIEW_METHOD" in f2.validate_review_record(record, auth)


def test_individual_record_rejects_structural_linkage():
    auth = authority()
    record = review_record(auth)
    record["cohort_confirmation_id"] = "cohort"
    assert "INVALID_COHORT_CONFIRMATION_LINKAGE" in f2.validate_review_record(record, auth)


def test_structural_record_requires_valid_cohort_linkage():
    auth = authority()
    record = f2.make_structural_cohort_record(auth, "pair_a", "reviewer", "cohort", "2026-08-14T00:00:00Z")
    assert record["review_method"] == f2.STRUCTURAL_COHORT_METHOD
    assert record["human_review_time_provenance"] == f2.NOT_APPLICABLE_STRUCTURAL_COHORT
    assert record["record_origin"] == f2.STRUCTURAL_COHORT_RECORD_ORIGIN
    record["cohort_confirmation_id"] = ""
    assert "INVALID_COHORT_CONFIRMATION_LINKAGE" in f2.validate_review_record(record, auth)


def test_structural_record_cannot_use_individual_time_provenance():
    auth = authority()
    record = f2.make_structural_cohort_record(auth, "pair_a", "reviewer", cohort_id(auth, ["pair_a"]), "2026-08-14T00:00:00Z")
    record["human_review_time_provenance"] = f2.CAPTURED_IN_RECORD
    assert "INVALID_REVIEW_TIME_PROVENANCE" in f2.validate_review_record(record, auth, cohort_confirmations=cohort_confirmations(auth, ["pair_a"]))


def test_xlsx_imported_individual_record_accepts_truthful_missing_visual_review_time():
    auth = authority()
    record = review_record(
        auth,
        human_grammar_validity="MULTI_MEMBER_DEFECT",
        human_review_time_provenance=f2.NOT_CAPTURED_IN_XLSX,
        record_origin=f2.XLSX_CONFIRMED_IMPORT,
    )
    assert f2.validate_review_record(record, auth) == []


def test_xlsx_imported_individual_record_rejects_fabricated_captured_time():
    auth = authority()
    record = review_record(
        auth,
        human_grammar_validity="MULTI_MEMBER_DEFECT",
        human_review_time_provenance=f2.CAPTURED_IN_RECORD,
        record_origin=f2.XLSX_CONFIRMED_IMPORT,
    )
    assert "INVALID_REVIEW_TIME_PROVENANCE" in f2.validate_review_record(record, auth)


def test_timestamp_invalid_non_z_rejection():
    auth = authority()
    record = review_record(auth)
    record["reviewed_at_utc"] = "2026-08-14T00:00:00+00:00"
    assert "INVALID_REVIEW_TIMESTAMP" in f2.validate_review_record(record, auth)


def test_timestamp_provenance_does_not_permit_separate_visual_review_time():
    auth = authority()
    record = review_record(auth)
    record["authority_recorded_at_utc"] = "2026-08-14T00:00:01Z"
    assert "INVALID_REVIEW_TIMESTAMP" in f2.validate_review_record(record, auth)


def test_source_hash_mismatch():
    auth = authority()
    record = review_record(auth)
    record["source_record_sha256"] = "0" * 64
    assert "SOURCE_RECORD_HASH_MISMATCH" in f2.validate_review_record(record, auth)


def test_unauthorized_pair_id():
    auth = authority()
    record = review_record(auth)
    record["pair_id"] = "not_authorized"
    assert "UNAUTHORIZED_PAIR_ID" in f2.validate_review_record(record, auth)


def test_duplicate_wip_pair(tmp_path):
    path = external_wip_path(tmp_path)
    auth = authority(tmp_path=tmp_path)
    first = review_record(auth)
    f2.write_wip_atomic(path, [first, first])
    records, duplicates = f2.load_wip(path)
    assert duplicates == {"pair_a"}
    assert "DUPLICATE_PAIR_ID" in f2.validate_wip_records(auth, records, duplicates)


def test_correction_replaces_rather_than_duplicates(tmp_path):
    path = external_wip_path(tmp_path)
    auth = authority(tmp_path=tmp_path)
    first = review_record(auth)
    second = review_record(auth, human_grammar_validity="MULTI_MEMBER_DEFECT")
    f2.upsert_wip_record(auth, path, first)
    f2.upsert_wip_record(auth, path, second)
    records, duplicates = f2.load_wip(path)
    assert duplicates == set()
    assert len(records) == 1
    assert records[0]["human_authority_decision"] == "CANONICAL_REGENERATION_REQUIRED"


def test_strict_load_wip_and_upsert_valid_empty_error_paths_succeed(tmp_path):
    path = external_wip_path(tmp_path)
    auth = authority(tmp_path=tmp_path)
    record = review_record(auth)

    f2.upsert_wip_record(auth, path, record)
    records, duplicates = f2.strict_load_wip(auth, path)

    assert duplicates == set()
    assert records == [record]


def test_correction_updates_timestamp_semantics_through_clock():
    auth = authority()
    first = f2.make_review_record(auth, "pair_a", "reviewer", "VALID", "VALID", "VALID", "CANONICAL_ONLY_DEFECT", "", clock=lambda: datetime(2026, 8, 14, tzinfo=UTC))
    second = f2.make_review_record(auth, "pair_a", "reviewer", "VALID", "VALID", "VALID", "MULTI_MEMBER_DEFECT", "", clock=lambda: datetime(2026, 8, 14, tzinfo=UTC) + timedelta(seconds=1))
    assert first["reviewed_at_utc"] != second["reviewed_at_utc"]


def test_source_authority_order_preservation():
    rows = [source_row("pair_b"), source_row("pair_a")]
    auth = authority(rows)
    assert auth.ordered_pair_ids == ["pair_b", "pair_a"]
    assert f2.next_unreviewed_pair_id(auth, [review_record(auth, "pair_b")]) == "pair_a"


def test_verify_pair_membership_rejects_missing_f2_pair(monkeypatch):
    rows = [source_row("pair_a"), source_row("pair_b")]
    wrong_extra_row = source_row("pair_c")
    monkeypatch.setattr(f2, "EXPECTED_PAIR_COUNT", 2)
    monkeypatch.setattr(f2, "EXPECTED_MEMBER_COUNT", 6)

    with pytest.raises(f2.ReviewInfrastructureError, match="missing pair pair_b"):
        f2.verify_pair_membership(
            rows,
            [pair_authority_record(rows[0]), pair_authority_record(wrong_extra_row)],
            pair_summary(pair_count=2, member_count=6),
        )


def test_structural_gate_passes_exact_literal_and_ai_contract(monkeypatch):
    row = structural_row()
    auth = authority([row])
    monkeypatch.setattr(f2, "EXPECTED_PAIR_COUNT", 1)
    monkeypatch.setattr(f2, "EXPECTED_MEMBER_COUNT", 3)
    assert f2.structural_gate_errors(row, auth, ai_prescreen_record(auth)) == []


@pytest.mark.parametrize(
    "mutator",
    [
        lambda row: row.update({"canonical_final_label": "SUPPORT"}),
        lambda row: row.update({"paraphrase_claim": "A selected B."}),
        lambda row: row.update({"canonical_evidence": "A restored B."}),
        lambda row: row.update({"paraphrase_evidence": "A restored B."}),
        lambda row: row.update({"polarity_flip_evidence": "A did not restored B."}),
        lambda row: row.update({"canonical_grammar_status": "PASS"}),
        lambda row: row.update({"canonical_reason_codes": "[]"}),
    ],
)
def test_structural_gate_fails_closed_on_structural_mismatch(monkeypatch, mutator):
    row = structural_row()
    mutator(row)
    auth = authority([row])
    monkeypatch.setattr(f2, "EXPECTED_PAIR_COUNT", 1)
    monkeypatch.setattr(f2, "EXPECTED_MEMBER_COUNT", 3)
    assert "STRUCTURAL_COHORT_GATE_FAILED" in f2.structural_gate_errors(row, auth, ai_prescreen_record(auth))


def test_structural_gate_fails_on_ai_human_review_required(monkeypatch):
    row = structural_row()
    auth = authority([row])
    monkeypatch.setattr(f2, "EXPECTED_PAIR_COUNT", 1)
    monkeypatch.setattr(f2, "EXPECTED_MEMBER_COUNT", 3)
    ai = ai_prescreen_record(auth, ai_triage_status="HUMAN_REVIEW_REQUIRED")
    assert "STRUCTURAL_COHORT_GATE_FAILED" in f2.structural_gate_errors(row, auth, ai)


def test_structural_gate_fails_on_ai_non_vvvm(monkeypatch):
    row = structural_row()
    auth = authority([row])
    monkeypatch.setattr(f2, "EXPECTED_PAIR_COUNT", 1)
    monkeypatch.setattr(f2, "EXPECTED_MEMBER_COUNT", 3)
    ai = ai_prescreen_record(auth, ai_grammar_validity_suggestion="C")
    assert "STRUCTURAL_COHORT_GATE_FAILED" in f2.structural_gate_errors(row, auth, ai)


def test_structural_gate_fails_on_ai_source_hash_mismatch(monkeypatch):
    row = structural_row()
    auth = authority([row])
    monkeypatch.setattr(f2, "EXPECTED_PAIR_COUNT", 1)
    monkeypatch.setattr(f2, "EXPECTED_MEMBER_COUNT", 3)
    ai = ai_prescreen_record(auth, source_record_sha256="0" * 64)
    assert "SOURCE_RECORD_HASH_MISMATCH" in f2.structural_gate_errors(row, auth, ai)


def test_missing_or_invalid_twenty_pair_audit_blocks_cohort_confirm(tmp_path):
    auth = audit_authority(tmp_path)
    records = [review_record(auth, pair_id, human_grammar_validity="MULTI_MEMBER_DEFECT") for pair_id in f2.INDIVIDUALLY_REVIEWED_AUDIT_PAIR_IDS[:-1]]
    assert "MISSING_INDIVIDUAL_AUDIT_EVIDENCE" in f2.validate_individual_audit_evidence(auth, records)


def test_twenty_pair_audit_accepts_truthful_xlsx_import_time_provenance(tmp_path):
    auth = audit_authority(tmp_path)
    records = [
        review_record(
            auth,
            pair_id,
            human_grammar_validity="MULTI_MEMBER_DEFECT",
            human_review_time_provenance=f2.NOT_CAPTURED_IN_XLSX,
            record_origin=f2.XLSX_CONFIRMED_IMPORT,
        )
        for pair_id in f2.INDIVIDUALLY_REVIEWED_AUDIT_PAIR_IDS
    ]
    assert f2.validate_individual_audit_evidence(auth, records) == []


def test_audit_pair_override_blocks_cohort_confirm(tmp_path):
    auth = audit_authority(tmp_path)
    records = [review_record(auth, pair_id, human_grammar_validity="MULTI_MEMBER_DEFECT") for pair_id in f2.INDIVIDUALLY_REVIEWED_AUDIT_PAIR_IDS]
    records[0]["human_canonical_semantics"] = "INVALID"
    records[0]["human_authority_decision"] = "SEMANTIC_CONFLICT"
    assert "MISSING_INDIVIDUAL_AUDIT_EVIDENCE" in f2.validate_individual_audit_evidence(auth, records)


def test_incorrect_seven_predicate_audit_coverage_blocks(tmp_path):
    auth = audit_authority(tmp_path)
    bad_row = dict(auth.source_rows[0])
    bad_row["canonical_evidence"] = "A did not selected B."
    rows = [bad_row, *auth.source_rows[1:]]
    bad_auth = authority(rows, tmp_path)
    records = [review_record(bad_auth, pair_id, human_grammar_validity="MULTI_MEMBER_DEFECT") for pair_id in f2.INDIVIDUALLY_REVIEWED_AUDIT_PAIR_IDS]
    assert "AUDIT_PREDICATE_COVERAGE_MISMATCH" in f2.validate_individual_audit_evidence(bad_auth, records)


def test_partial_review_never_reaches_completion():
    rows = [source_row("pair_a"), source_row("pair_b")]
    auth = authority(rows)
    summary = f2.compute_summary(auth, [review_record(auth, "pair_a")])
    assert not summary["completion_gate_passed"]
    assert summary["unreviewed_pair_ids"] == ["pair_b"]


def test_completed_category_partition_logic():
    auth = authority()
    record = review_record(auth)
    summary = f2.compute_summary(auth, [record])
    assert summary["completed_decision_pair_ids"] == ["pair_a"]
    assert summary["textual_repair_candidate_pair_ids"] == ["pair_a"]
    assert summary["individual_review_pair_ids"] == ["pair_a"]
    assert summary["structural_cohort_confirmation_pair_ids"] == []
    assert summary["summary_validation_errors"] == []


def test_summary_method_counts_and_arrays_are_exact():
    rows = [structural_row("pair_a"), structural_row("pair_b")]
    auth = authority(rows)
    individual = review_record(auth, "pair_a", human_grammar_validity="MULTI_MEMBER_DEFECT")
    structural = f2.make_structural_cohort_record(auth, "pair_b", "reviewer", cohort_id(auth, ["pair_b"]), "2026-08-14T00:00:00Z")
    summary = f2.compute_summary(auth, [individual, structural], cohort_confirmations=cohort_confirmations(auth, ["pair_b"]))
    assert summary["individual_review_pair_count"] == 1
    assert summary["individual_review_pair_ids"] == ["pair_a"]
    assert summary["structural_cohort_confirmation_pair_count"] == 1
    assert summary["structural_cohort_confirmation_pair_ids"] == ["pair_b"]
    assert summary["individual_review_pair_count"] + summary["structural_cohort_confirmation_pair_count"] == summary["reviewed_pair_count"]


def test_structural_wip_requires_loaded_cohort_artifact_identity():
    auth = authority([structural_row()])
    record = f2.make_structural_cohort_record(auth, "pair_a", "reviewer", "cohort", "2026-08-14T00:00:00Z")
    assert "INVALID_COHORT_CONFIRMATION_LINKAGE" in f2.validate_wip_records(auth, [record])
    record["cohort_confirmation_id"] = cohort_id(auth, ["pair_a"])
    assert f2.validate_wip_records(auth, [record], cohort_confirmations=cohort_confirmations(auth, ["pair_a"])) == []


def test_structural_wip_rejects_valid_cohort_id_when_pair_absent():
    auth = authority([structural_row("pair_a"), structural_row("pair_b")])
    record = f2.make_structural_cohort_record(auth, "pair_b", "reviewer", cohort_id(auth, ["pair_a"]), "2026-08-14T00:00:00Z")
    errors = f2.validate_wip_records(auth, [record], cohort_confirmations=cohort_confirmations(auth, ["pair_a"]))
    assert "INVALID_COHORT_CONFIRMATION_LINKAGE" in errors


def test_structural_wip_rejects_individual_audit_pair_as_structural(tmp_path):
    auth = audit_authority(tmp_path)
    pair_id = f2.INDIVIDUALLY_REVIEWED_AUDIT_PAIR_IDS[0]
    record = f2.make_structural_cohort_record(auth, "generated_fact_999", "reviewer", cohort_id(auth, [pair_id]), "2026-08-14T00:00:00Z")
    record["pair_id"] = pair_id
    record["source_record_sha256"] = auth.source_sha256_by_pair_id[pair_id]
    errors = f2.validate_wip_records(auth, [record], cohort_confirmations=cohort_confirmations(auth, [pair_id]))
    assert "INVALID_COHORT_CONFIRMATION_LINKAGE" in errors


def test_cohort_confirmation_id_alone_is_insufficient_for_wip_acceptance():
    auth = authority([structural_row()])
    record = f2.make_structural_cohort_record(auth, "pair_a", "reviewer", cohort_id(auth, ["pair_a"]), "2026-08-14T00:00:00Z")
    assert "INVALID_COHORT_CONFIRMATION_LINKAGE" in f2.validate_wip_records(auth, [record], cohort_confirmations={"cohort-1": f2.CohortConfirmationAuthority(
        cohort_confirmation_id="cohort-1",
        cohort_confirmed_pair_ids=frozenset(),
        individually_reviewed_pair_ids=frozenset(f2.INDIVIDUALLY_REVIEWED_AUDIT_PAIR_IDS),
        structural_audit_sha256="a" * 64,
        structural_gate_version=f2.STRUCTURAL_GATE_VERSION,
        source_authority_identity=f2.source_authority_identity(auth),
        ai_prescreen_artifact_sha256="b" * 64,
        validated_individual_wip_state_sha256="c" * 64,
    )})


def test_count_array_symmetry_detection():
    auth = authority()
    summary = f2.compute_summary(auth, [review_record(auth)])
    summary["reviewed_pair_count"] = 99
    assert "COUNT_ARRAY_ASYMMETRY" in f2.validate_summary_partitions(summary)


def test_wip_inside_repo_rejected(tmp_path):
    auth = authority(tmp_path=tmp_path)
    with pytest.raises(f2.ReviewInfrastructureError, match="WIP_SCHEMA_MISMATCH"):
        f2.require_wip_path_outside_repo(auth.repo_root, tmp_path / "wip.jsonl")


def test_wip_authority_file_path_rejected(tmp_path):
    auth = authority(tmp_path=tmp_path)
    with pytest.raises(f2.ReviewInfrastructureError, match="WIP_SCHEMA_MISMATCH"):
        f2.require_wip_path_outside_repo(auth.repo_root, tmp_path / "reports" / "reason_router_p2_p3w6f2p1_manual_review_execution_manifest.json")


def test_wip_inside_reports_rejected(tmp_path):
    auth = authority(tmp_path=tmp_path)
    with pytest.raises(f2.ReviewInfrastructureError, match="WIP_SCHEMA_MISMATCH"):
        f2.require_wip_path_outside_repo(auth.repo_root, tmp_path / "reports" / "wip.jsonl")


def test_wip_outside_repo_accepted(tmp_path):
    auth = authority(tmp_path=tmp_path)
    f2.require_wip_path_outside_repo(auth.repo_root, external_wip_path(tmp_path))


def test_load_ai_prescreen_requires_artifact(tmp_path):
    auth = authority(tmp_path=tmp_path)
    with pytest.raises(f2.ReviewInfrastructureError, match="INVALID_AI_PRESCREEN_ARTIFACT"):
        f2.load_ai_prescreen(tmp_path / "missing.jsonl", auth)


def test_load_ai_prescreen_rejects_source_hash_mismatch(tmp_path):
    auth = authority(tmp_path=tmp_path)
    path = tmp_path / "ai.jsonl"
    record = ai_prescreen_record(auth, source_record_sha256="0" * 64)
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(f2.ReviewInfrastructureError, match="SOURCE_RECORD_HASH_MISMATCH"):
        f2.load_ai_prescreen(path, auth)


def test_load_ai_prescreen_accepts_strict_jsonl(tmp_path):
    auth = authority(tmp_path=tmp_path)
    path = tmp_path / "ai.jsonl"
    record = ai_prescreen_record(auth)
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    loaded = f2.load_ai_prescreen(path, auth)
    assert loaded == {"pair_a": record}


def test_cohort_confirmation_artifact_strict_schema_validation(tmp_path):
    auth = authority([structural_row()], tmp_path)
    artifact = cohort_artifact(auth)
    assert f2.validate_cohort_confirmation_artifact(artifact, auth, audit_artifact(auth)) == []
    artifact["extra"] = "not allowed"
    assert "INVALID_COHORT_CONFIRMATION_ARTIFACT" in f2.validate_cohort_confirmation_artifact(artifact, auth, audit_artifact(auth))


def test_cohort_confirmation_artifact_binds_audit_identity(tmp_path):
    auth = authority([structural_row()], tmp_path)
    audit = audit_artifact(auth)
    artifact = f2.cohort_confirmation_artifact(auth, audit, tmp_path.parent / "audit.json", "reviewer", "2026-08-14T00:00:00Z")
    assert artifact["structural_audit_sha256"] == audit["audit_payload_sha256"]
    assert artifact["ai_prescreen_artifact_sha256"] == audit["ai_prescreen_artifact_sha256"]
    assert artifact["validated_individual_wip_state_sha256"] == audit["validated_individual_wip_state_sha256"]
    assert artifact["confirmation_payload_sha256"] == f2.confirmation_payload_sha256(artifact)
    assert artifact["cohort_confirmation_id"] == f2.derive_cohort_confirmation_id_from_payload_sha256(artifact["confirmation_payload_sha256"])
    assert f2.validate_cohort_confirmation_artifact(artifact, auth, audit) == []


@pytest.mark.parametrize(
    "mutator,expected",
    [
        (lambda artifact: artifact.update({"ai_prescreen_artifact_sha256": "0" * 64}), "INVALID_AI_PRESCREEN_ARTIFACT"),
        (lambda artifact: artifact.update({"validated_individual_wip_state_sha256": "0" * 64}), "MISSING_INDIVIDUAL_AUDIT_EVIDENCE"),
        (lambda artifact: artifact.update({"source_authority_identity": {}}), "INVALID_COHORT_CONFIRMATION_ARTIFACT"),
        (lambda artifact: artifact.update({"eligible_pair_count": 2}), "INVALID_COHORT_CONFIRMATION_ARTIFACT"),
        (lambda artifact: artifact.update({"exception_pair_ids": ["pair_a"]}), "INVALID_COHORT_CONFIRMATION_ARTIFACT"),
    ],
)
def test_cohort_confirmation_artifact_identity_mismatches_rejected(tmp_path, mutator, expected):
    auth = authority([structural_row()], tmp_path)
    audit = audit_artifact(auth)
    artifact = f2.cohort_confirmation_artifact(auth, audit, tmp_path.parent / "audit.json", "reviewer", "2026-08-14T00:00:00Z")
    mutator(artifact)
    assert expected in f2.validate_cohort_confirmation_artifact(artifact, auth, audit)


@pytest.mark.parametrize(
    "mutator,expected",
    [
        (lambda artifact: artifact.update({"cohort_confirmation_id": "forged-nonempty-id"}), "COHORT_CONFIRMATION_ID_MISMATCH"),
        (lambda artifact: artifact.update({"cohort_confirmation_id": f2.derive_cohort_confirmation_id_from_payload_sha256("0" * 64)}), "COHORT_CONFIRMATION_ID_MISMATCH"),
        (lambda artifact: artifact.update({"structural_audit_sha256": "0" * 64}), "CONFIRMATION_PAYLOAD_HASH_MISMATCH"),
        (lambda artifact: artifact.update({"reviewer_id": "other-reviewer"}), "CONFIRMATION_PAYLOAD_HASH_MISMATCH"),
        (lambda artifact: artifact.update({"authority_recorded_at_utc": "2026-08-14T00:00:01Z"}), "CONFIRMATION_PAYLOAD_HASH_MISMATCH"),
        (lambda artifact: artifact.update({"human_action": "CONFIRM"}), "CONFIRMATION_PAYLOAD_HASH_MISMATCH"),
        (lambda artifact: artifact.update({"ai_prescreen_artifact_sha256": "0" * 64}), "CONFIRMATION_PAYLOAD_HASH_MISMATCH"),
        (lambda artifact: artifact.update({"validated_individual_wip_state_sha256": "0" * 64}), "CONFIRMATION_PAYLOAD_HASH_MISMATCH"),
        (lambda artifact: artifact.update({"cohort_confirmed_pair_ids": []}), "CONFIRMATION_PAYLOAD_HASH_MISMATCH"),
        (lambda artifact: artifact.update({"eligible_pair_ids": [], "exception_pair_ids": ["pair_a"]}), "CONFIRMATION_PAYLOAD_HASH_MISMATCH"),
        (lambda artifact: artifact.update({"source_authority_identity": {**artifact["source_authority_identity"], "P3W4_artifact_commit": "0" * 40}}), "CONFIRMATION_PAYLOAD_HASH_MISMATCH"),
    ],
)
def test_loaded_confirmation_identity_rejects_stale_or_forged_payload(tmp_path, mutator, expected):
    auth = authority([structural_row()], tmp_path)
    artifact = cohort_artifact(auth)
    mutator(artifact)
    errors = f2.validate_cohort_confirmation_artifact(artifact, auth)
    assert expected in errors


def test_loaded_confirmation_accepts_exact_recomputed_payload_identity(tmp_path):
    auth = authority([structural_row()], tmp_path)
    artifact = cohort_artifact(auth)
    assert f2.validate_cohort_confirmation_artifact(artifact, auth) == []


def test_confirmation_payload_sha_is_independent_of_dict_insertion_order(tmp_path):
    auth = authority([structural_row()], tmp_path)
    artifact = cohort_artifact(auth)
    reordered = {key: artifact[key] for key in reversed(list(artifact.keys()))}
    assert f2.confirmation_payload_sha256(reordered) == artifact["confirmation_payload_sha256"]


def test_confirmation_payload_pair_lists_are_canonicalized(tmp_path):
    auth = authority([structural_row("pair_a"), structural_row("pair_b")], tmp_path)
    artifact = cohort_artifact(auth, ["pair_b", "pair_a"])
    reordered = dict(artifact)
    reordered["eligible_pair_ids"] = ["pair_a", "pair_b"]
    reordered["cohort_confirmed_pair_ids"] = ["pair_a", "pair_b"]
    assert f2.confirmation_payload_sha256(reordered) == artifact["confirmation_payload_sha256"]


def test_confirmation_id_derivation_is_repeatable_and_materially_sensitive(tmp_path):
    auth = authority([structural_row()], tmp_path)
    artifact = cohort_artifact(auth)
    payload_sha = f2.confirmation_payload_sha256(artifact)
    assert f2.derive_cohort_confirmation_id_from_payload_sha256(payload_sha) == f2.derive_cohort_confirmation_id_from_payload_sha256(payload_sha)
    changed = dict(artifact)
    changed["reviewer_id"] = "other-reviewer"
    changed_sha = f2.confirmation_payload_sha256(changed)
    assert changed_sha != payload_sha
    assert f2.derive_cohort_confirmation_id_from_payload_sha256(changed_sha) != artifact["cohort_confirmation_id"]


def test_confirmation_payload_hash_excludes_self_hash_and_id_fields(tmp_path):
    auth = authority([structural_row()], tmp_path)
    artifact = cohort_artifact(auth)
    changed = dict(artifact)
    changed["cohort_confirmation_id"] = "forged"
    changed["confirmation_payload_sha256"] = "0" * 64
    assert f2.confirmation_payload_sha256(changed) == artifact["confirmation_payload_sha256"]


def test_status_rejects_forged_loaded_confirmation_before_wip_authority(tmp_path, monkeypatch):
    auth = authority([structural_row()], tmp_path)
    monkeypatch.setattr(f2, "load_authority", lambda: auth)
    wip = external_wip_path(tmp_path)
    cohort_path = external_wip_path(tmp_path).with_name("cohort.json")
    artifact = cohort_artifact(auth)
    forged_id = f2.derive_cohort_confirmation_id_from_payload_sha256("0" * 64)
    artifact["cohort_confirmation_id"] = forged_id
    record = f2.make_structural_cohort_record(auth, "pair_a", "reviewer", forged_id, "2026-08-14T00:00:00Z")
    f2.write_wip_atomic(wip, [record])
    f2.write_json_atomic(cohort_path, artifact)
    args = f2.build_arg_parser().parse_args(
        [
            "status",
            "--wip-path",
            str(wip),
            "--cohort-confirmation-path",
            str(cohort_path),
        ]
    )
    with pytest.raises(f2.ReviewInfrastructureError, match="COHORT_CONFIRMATION_ID_MISMATCH"):
        f2.command_status(args)


def test_created_confirmation_round_trips_through_strict_loader(tmp_path):
    auth = authority([structural_row()], tmp_path)
    cohort_path = external_wip_path(tmp_path).with_name("cohort.json")
    artifact = cohort_artifact(auth)
    f2.write_json_atomic(cohort_path, artifact)
    loaded = f2.load_cohort_confirmations(cohort_path, auth)
    assert loaded is not None
    assert list(loaded) == [artifact["cohort_confirmation_id"]]


def test_wip_traversal_resolving_inside_repo_rejected(tmp_path):
    auth = authority(tmp_path=tmp_path)
    outside_then_back_in = tmp_path / ".." / tmp_path.name / "wip.jsonl"
    with pytest.raises(f2.ReviewInfrastructureError, match="WIP_SCHEMA_MISMATCH"):
        f2.require_wip_path_outside_repo(auth.repo_root, outside_then_back_in)


def test_strict_wip_extra_field_rejected(tmp_path):
    auth = authority(tmp_path=tmp_path)
    path = external_wip_path(tmp_path)
    record = review_record(auth)
    record["extra"] = "not allowed"
    f2.write_wip_atomic(path, [record])
    with pytest.raises(f2.ReviewInfrastructureError, match="WIP_SCHEMA_MISMATCH"):
        f2.strict_load_wip(auth, path)


def test_strict_wip_missing_field_rejected(tmp_path):
    auth = authority(tmp_path=tmp_path)
    path = external_wip_path(tmp_path)
    record = review_record(auth)
    del record["reviewed_at_utc"]
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(f2.ReviewInfrastructureError, match="WIP_SCHEMA_MISMATCH"):
        f2.strict_load_wip(auth, path)


def test_strict_wip_malformed_jsonl_rejected(tmp_path):
    auth = authority(tmp_path=tmp_path)
    path = external_wip_path(tmp_path)
    path.write_text("{bad json\n", encoding="utf-8")
    with pytest.raises(f2.ReviewInfrastructureError, match="WIP_SCHEMA_MISMATCH"):
        f2.strict_load_wip(auth, path)


@pytest.mark.parametrize(
    "mutator,expected",
    [
        (lambda record: record.update({"human_canonical_semantics": "MAYBE"}), "INVALID_SEMANTIC_ENUM"),
        (lambda record: record.update({"source_record_sha256": "0" * 64}), "SOURCE_RECORD_HASH_MISMATCH"),
        (lambda record: record.update({"pair_id": "not_authorized"}), "UNAUTHORIZED_PAIR_ID"),
        (lambda record: record.update({"reviewer_id": ""}), "MISSING_REVIEWER_ID"),
        (lambda record: record.update({"human_authority_decision": "SEMANTIC_CONFLICT"}), "COMPATIBILITY_MATRIX_MISMATCH"),
        (lambda record: record.update({"human_canonical_semantics": "UNCLEAR", "human_authority_decision": "INSUFFICIENT_EVIDENCE_KEEP_BLOCKED", "human_notes": ""}), "MISSING_REQUIRED_NOTES"),
    ],
)
def test_invalid_wip_blocks_next(tmp_path, mutator, expected):
    auth = authority(tmp_path=tmp_path)
    path = external_wip_path(tmp_path)
    record = review_record(auth)
    mutator(record)
    f2.write_wip_atomic(path, [record])
    with pytest.raises(f2.ReviewInfrastructureError, match=expected):
        f2.strict_load_wip(auth, path)


def test_duplicate_wip_blocks_next(tmp_path):
    auth = authority(tmp_path=tmp_path)
    path = external_wip_path(tmp_path)
    record = review_record(auth)
    f2.write_wip_atomic(path, [record, record])
    with pytest.raises(f2.ReviewInfrastructureError, match="DUPLICATE_PAIR_ID"):
        f2.strict_load_wip(auth, path)


def test_valid_partial_wip_selects_first_truly_unreviewed(tmp_path):
    rows = [source_row("pair_a"), source_row("pair_b"), source_row("pair_c")]
    auth = authority(rows, tmp_path)
    path = external_wip_path(tmp_path)
    f2.write_wip_atomic(path, [review_record(auth, "pair_a")])
    records, _duplicates = f2.strict_load_wip(auth, path)
    assert f2.next_unreviewed_pair_id(auth, records) == "pair_b"


def test_status_valid_partial_succeeds_but_not_complete(tmp_path):
    rows = [source_row("pair_a"), source_row("pair_b")]
    auth = authority(rows, tmp_path)
    path = external_wip_path(tmp_path)
    f2.write_wip_atomic(path, [review_record(auth, "pair_a")])
    records, duplicates = f2.strict_load_wip(auth, path)
    summary = f2.compute_summary(auth, records, duplicates)
    assert summary["reviewed_pair_count"] == 1
    assert not summary["completion_gate_passed"]


@pytest.mark.parametrize(
    "mutator",
    [
        lambda record: record.update({"pair_id": "not_authorized"}),
        lambda record: record.update({"extra": "not allowed"}),
        lambda record: record.pop("reviewed_at_utc"),
    ],
)
def test_status_invalid_wip_fails_closed(tmp_path, mutator):
    auth = authority(tmp_path=tmp_path)
    path = external_wip_path(tmp_path)
    record = review_record(auth)
    mutator(record)
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(f2.ReviewInfrastructureError):
        f2.strict_load_wip(auth, path)


def test_strict_33_column_template_required():
    row = source_row()
    payload = write_template([row], [*f2.SOURCE_FIELDS, *f2.HUMAN_FIELDS])
    parsed = f2.parse_source_csv(payload, f2.SOURCE_FIELDS)
    assert parsed == [row]


def test_prepopulated_template_human_field_rejected():
    row = source_row()
    fields = [*f2.SOURCE_FIELDS, *f2.HUMAN_FIELDS]
    payload = write_template([row], fields, {"human_notes": "already populated"})
    with pytest.raises(f2.ReviewInfrastructureError, match="SOURCE_FIELD_MUTATION"):
        f2.parse_source_csv(payload, f2.SOURCE_FIELDS)


def test_27_column_only_authority_template_rejected():
    payload = write_template([source_row()], f2.SOURCE_FIELDS)
    with pytest.raises(f2.ReviewInfrastructureError, match="SOURCE_SCHEMA_MISMATCH"):
        f2.parse_source_csv(payload, f2.SOURCE_FIELDS)


def test_37_column_authority_template_rejected():
    payload = write_template([source_row()], f2.COMPLETED_CSV_FIELDS)
    with pytest.raises(f2.ReviewInfrastructureError, match="SOURCE_SCHEMA_MISMATCH"):
        f2.parse_source_csv(payload, f2.SOURCE_FIELDS)


def test_finalization_refuses_incomplete_wip_and_produces_no_target(tmp_path, monkeypatch):
    rows = [source_row("pair_a"), source_row("pair_b")]
    auth = authority(rows, tmp_path)
    prepare_reports_dir(tmp_path)
    wip = external_wip_path(tmp_path)
    f2.write_wip_atomic(wip, [review_record(auth, "pair_a")])
    monkeypatch.setattr(f2, "require_tracked_execution_state", lambda repo_root: "f" * 40)
    with pytest.raises(f2.ReviewInfrastructureError):
        f2.finalize_artifacts(auth, wip)
    assert not f2.final_output_dir(tmp_path, "f" * 40).exists()


def finalized_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[f2.Authority, Path]:
    auth = authority(tmp_path=tmp_path)
    prepare_reports_dir(tmp_path)
    wip = external_wip_path(tmp_path)
    f2.write_wip_atomic(wip, [review_record(auth)])
    monkeypatch.setattr(f2, "EXPECTED_PAIR_COUNT", 1)
    monkeypatch.setattr(f2, "require_tracked_execution_state", lambda repo_root: "f" * 40)
    out = f2.finalize_artifacts(auth, wip)
    return auth, out


def test_finalization_produces_exactly_three_artifacts_for_complete_valid_synthetic_fixture(tmp_path, monkeypatch):
    _auth, out = finalized_fixture(tmp_path, monkeypatch)
    assert sorted(path.name for path in out.iterdir()) == [
        "p3w6f2_hybrid_review_completed.csv",
        "p3w6f2_hybrid_review_decisions.jsonl",
        "p3w6f2_hybrid_review_summary.json",
    ]


def test_final_csv_exact_37_column_order(tmp_path, monkeypatch):
    _auth, out = finalized_fixture(tmp_path, monkeypatch)
    with (out / "p3w6f2_hybrid_review_completed.csv").open(newline="", encoding="utf-8") as handle:
        assert next(csv.reader(handle)) == f2.COMPLETED_CSV_FIELDS


def test_final_csv_source_fields_equal_immutable_authority_fields(tmp_path, monkeypatch):
    auth, out = finalized_fixture(tmp_path, monkeypatch)
    with (out / "p3w6f2_hybrid_review_completed.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert {field: row[field] for field in f2.SOURCE_FIELDS} == auth.source_rows[0]


def test_decisions_jsonl_required_fields_and_valid_state_values(tmp_path, monkeypatch):
    _auth, out = finalized_fixture(tmp_path, monkeypatch)
    record = json.loads((out / "p3w6f2_hybrid_review_decisions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert list(record) == sorted(f2.DECISIONS_JSONL_FIELDS)
    assert record["compatibility_matrix_match"] is True
    assert record["review_record_valid"] is True
    assert record["ordered_validation_errors"] == []


def test_final_output_existing_directory_refusal(tmp_path, monkeypatch):
    auth = authority(tmp_path=tmp_path)
    prepare_reports_dir(tmp_path)
    wip = external_wip_path(tmp_path)
    f2.write_wip_atomic(wip, [review_record(auth)])
    monkeypatch.setattr(f2, "EXPECTED_PAIR_COUNT", 1)
    monkeypatch.setattr(f2, "require_tracked_execution_state", lambda repo_root: "f" * 40)
    f2.final_output_dir(tmp_path, "f" * 40).mkdir()
    with pytest.raises(f2.ReviewInfrastructureError, match="OUTPUT_PATH_MISMATCH"):
        f2.finalize_artifacts(auth, wip)


def test_final_output_target_is_derived_from_head_short_sha(tmp_path, monkeypatch):
    _auth, out = finalized_fixture(tmp_path, monkeypatch)
    assert out == tmp_path / "reports" / "reason_router_p2_p3w6f2_hybrid_human_review_execution_ffffffff"


def test_finalize_uses_exact_structural_pair_membership(tmp_path, monkeypatch):
    auth = authority([structural_row()], tmp_path)
    prepare_reports_dir(tmp_path)
    wip = external_wip_path(tmp_path)
    record = f2.make_structural_cohort_record(auth, "pair_a", "reviewer", cohort_id(auth, ["pair_a"]), "2026-08-14T00:00:00Z")
    f2.write_wip_atomic(wip, [record])
    monkeypatch.setattr(f2, "EXPECTED_PAIR_COUNT", 1)
    monkeypatch.setattr(f2, "require_tracked_execution_state", lambda repo_root: "f" * 40)
    with pytest.raises(f2.ReviewInfrastructureError, match="INVALID_COHORT_CONFIRMATION_LINKAGE"):
        f2.finalize_artifacts(auth, wip, cohort_confirmations(auth, []))
    out = f2.finalize_artifacts(auth, wip, cohort_confirmations(auth, ["pair_a"]))
    assert out.exists()


def test_cli_cannot_select_fake_final_output_namespace():
    parser = f2.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["finalize", "--output-dir", "fake"])


def test_individual_record_parser_has_no_structural_method_path():
    parser = f2.build_arg_parser()
    args = parser.parse_args(
        [
            "record",
            "--pair-id",
            "pair_a",
            "--reviewer-id",
            "reviewer",
            "--canonical-semantics",
            "VALID",
            "--paraphrase-semantics",
            "VALID",
            "--polarity-flip-semantics",
            "VALID",
            "--grammar-validity",
            "MULTI_MEMBER_DEFECT",
            "--notes",
            "",
            "--ack-complete-triple-reviewed",
        ]
    )
    assert not hasattr(args, "review_method")
    assert not hasattr(args, "cohort_confirmation_id")


def test_record_parser_rejects_structural_cohort_method_flag():
    parser = f2.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "record",
                "--pair-id",
                "pair_a",
                "--reviewer-id",
                "reviewer",
                "--canonical-semantics",
                "VALID",
                "--paraphrase-semantics",
                "VALID",
                "--polarity-flip-semantics",
                "VALID",
                "--grammar-validity",
                "MULTI_MEMBER_DEFECT",
                "--notes",
                "",
                "--ack-complete-triple-reviewed",
                "--review-method",
                f2.STRUCTURAL_COHORT_METHOD,
            ]
        )


def test_cohort_command_uses_structural_ack_not_individual_ack():
    parser = f2.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "cohort-confirm",
                "--wip-path",
                "wip.jsonl",
                "--ai-prescreen-path",
                "ai.jsonl",
                "--audit-path",
                "audit.json",
                "--expected-audit-sha256",
                "a" * 64,
                "--cohort-confirmation-path",
                "cohort.json",
                "--reviewer-id",
                "reviewer",
                "--ack-complete-triple-reviewed",
            ]
        )


def test_cohort_audit_command_writes_stable_artifact_and_no_wip_authority(tmp_path, monkeypatch):
    auth = audit_authority(tmp_path)
    monkeypatch.setattr(f2, "load_authority", lambda: auth)
    wip = external_wip_path(tmp_path)
    ai_path = tmp_path / "ai.jsonl"
    audit_path = external_wip_path(tmp_path).with_name("audit.json")
    write_full_ai_prescreen(ai_path, auth)
    audit_records = [review_record(auth, pair_id, human_grammar_validity="MULTI_MEMBER_DEFECT") for pair_id in f2.INDIVIDUALLY_REVIEWED_AUDIT_PAIR_IDS]
    f2.write_wip_atomic(wip, audit_records)
    args = f2.build_arg_parser().parse_args(
        [
            "cohort-audit",
            "--wip-path",
            str(wip),
            "--ai-prescreen-path",
            str(ai_path),
            "--audit-output-path",
            str(audit_path),
        ]
    )
    assert f2.command_cohort_audit(args) == 0
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["audit_payload_sha256"] == f2.canonical_json_sha256({field: audit[field] for field in f2.STRUCTURAL_AUDIT_FIELDS if field not in {"audit_created_at_utc", "audit_payload_sha256"}})
    loaded, duplicates = f2.strict_load_wip(auth, wip)
    assert duplicates == set()
    assert loaded == audit_records


def test_cohort_confirm_requires_audit_artifact_and_expected_sha(tmp_path, monkeypatch):
    auth = audit_authority(tmp_path)
    monkeypatch.setattr(f2, "load_authority", lambda: auth)
    wip = external_wip_path(tmp_path)
    ai_path = tmp_path / "ai.jsonl"
    write_full_ai_prescreen(ai_path, auth)
    records = [review_record(auth, pair_id, human_grammar_validity="MULTI_MEMBER_DEFECT") for pair_id in f2.INDIVIDUALLY_REVIEWED_AUDIT_PAIR_IDS]
    f2.write_wip_atomic(wip, records)
    args = f2.build_arg_parser().parse_args(
        [
            "cohort-confirm",
            "--wip-path",
            str(wip),
            "--ai-prescreen-path",
            str(ai_path),
            "--audit-path",
            str(tmp_path / "missing_audit.json"),
            "--expected-audit-sha256",
            "a" * 64,
            "--cohort-confirmation-path",
            str(external_wip_path(tmp_path).with_name("cohort.json")),
            "--reviewer-id",
            "reviewer",
            "--ack-structural-cohort-confirm",
        ]
    )
    with pytest.raises(f2.ReviewInfrastructureError, match="INVALID_COHORT_CONFIRMATION_ARTIFACT"):
        f2.command_cohort_confirm(args)


def test_cohort_confirm_rejects_wrong_expected_audit_sha(tmp_path, monkeypatch):
    auth = audit_authority(tmp_path)
    monkeypatch.setattr(f2, "load_authority", lambda: auth)
    wip = external_wip_path(tmp_path)
    ai_path = tmp_path / "ai.jsonl"
    audit_path = external_wip_path(tmp_path).with_name("audit.json")
    write_full_ai_prescreen(ai_path, auth)
    records = [review_record(auth, pair_id, human_grammar_validity="MULTI_MEMBER_DEFECT") for pair_id in f2.INDIVIDUALLY_REVIEWED_AUDIT_PAIR_IDS]
    f2.write_wip_atomic(wip, records)
    audit = f2.structural_audit_artifact(auth, records, f2.load_ai_prescreen(ai_path, auth), f2.path_sha256(ai_path), "2026-08-14T00:00:00Z")
    f2.write_json_atomic(audit_path, audit)
    args = f2.build_arg_parser().parse_args(
        [
            "cohort-confirm",
            "--wip-path",
            str(wip),
            "--ai-prescreen-path",
            str(ai_path),
            "--audit-path",
            str(audit_path),
            "--expected-audit-sha256",
            "0" * 64,
            "--cohort-confirmation-path",
            str(external_wip_path(tmp_path).with_name("cohort.json")),
            "--reviewer-id",
            "reviewer",
            "--ack-structural-cohort-confirm",
        ]
    )
    with pytest.raises(f2.ReviewInfrastructureError, match="audit SHA mismatch"):
        f2.command_cohort_confirm(args)


@pytest.mark.parametrize("drift", ["source", "ai", "wip", "eligible"])
def test_cohort_confirm_recomputes_and_rejects_stale_reviewed_audit(tmp_path, monkeypatch, drift):
    auth = audit_authority(tmp_path)
    monkeypatch.setattr(f2, "load_authority", lambda: auth)
    wip = external_wip_path(tmp_path)
    ai_path = tmp_path / "ai.jsonl"
    audit_path = external_wip_path(tmp_path).with_name("audit.json")
    write_full_ai_prescreen(ai_path, auth)
    records = [review_record(auth, pair_id, human_grammar_validity="MULTI_MEMBER_DEFECT") for pair_id in f2.INDIVIDUALLY_REVIEWED_AUDIT_PAIR_IDS]
    f2.write_wip_atomic(wip, records)
    audit = f2.structural_audit_artifact(auth, records, f2.load_ai_prescreen(ai_path, auth), f2.path_sha256(ai_path), "2026-08-14T00:00:00Z")
    f2.write_json_atomic(audit_path, audit)
    if drift == "source":
        changed_rows = [dict(row) for row in auth.source_rows]
        changed_rows[-1]["canonical_evidence"] = "A restored B."
        changed_auth = f2.Authority(
            repo_root=auth.repo_root,
            manifest=auth.manifest,
            source_rows=changed_rows,
            pair_records=auth.pair_records,
            summary=auth.summary,
            source_sha256_by_pair_id={row["pair_id"]: f2.compute_source_record_sha256(row) for row in changed_rows},
            p3w4_artifact_commit=auth.p3w4_artifact_commit,
            input_artifact_paths=auth.input_artifact_paths,
            input_artifact_sha256=auth.input_artifact_sha256,
            source_template_fields=auth.source_template_fields,
        )
        monkeypatch.setattr(f2, "load_authority", lambda: changed_auth)
    elif drift == "ai":
        write_full_ai_prescreen(ai_path, auth, generated_fact_999="HUMAN_REVIEW_REQUIRED")
    elif drift == "wip":
        records[0]["reviewer_id"] = "different"
        f2.write_wip_atomic(wip, records)
    elif drift == "eligible":
        audit["eligible_pair_ids"] = []
        audit["audit_payload_sha256"] = f2.canonical_json_sha256({field: audit[field] for field in f2.STRUCTURAL_AUDIT_FIELDS if field not in {"audit_created_at_utc", "audit_payload_sha256"}})
        f2.write_json_atomic(audit_path, audit)
    args = f2.build_arg_parser().parse_args(
        [
            "cohort-confirm",
            "--wip-path",
            str(wip),
            "--ai-prescreen-path",
            str(ai_path),
            "--audit-path",
            str(audit_path),
            "--expected-audit-sha256",
            audit["audit_payload_sha256"],
            "--cohort-confirmation-path",
            str(external_wip_path(tmp_path).with_name("cohort.json")),
            "--reviewer-id",
            "reviewer",
            "--ack-structural-cohort-confirm",
        ]
    )
    with pytest.raises(f2.ReviewInfrastructureError):
        f2.command_cohort_confirm(args)


def test_cohort_command_does_not_overwrite_existing_individual_record(tmp_path, monkeypatch):
    auth = audit_authority(tmp_path)
    monkeypatch.setattr(f2, "load_authority", lambda: auth)
    wip = external_wip_path(tmp_path)
    ai_path = tmp_path / "ai.jsonl"
    audit_path = external_wip_path(tmp_path).with_name("audit.json")
    output = external_wip_path(tmp_path).with_name("cohort.json")
    write_full_ai_prescreen(ai_path, auth)
    records = [review_record(auth, pair_id, human_grammar_validity="MULTI_MEMBER_DEFECT") for pair_id in f2.INDIVIDUALLY_REVIEWED_AUDIT_PAIR_IDS]
    existing = review_record(auth, "generated_fact_999", human_grammar_validity="MULTI_MEMBER_DEFECT")
    records.append(existing)
    f2.write_wip_atomic(wip, records)
    audit = f2.structural_audit_artifact(auth, records, f2.load_ai_prescreen(ai_path, auth), f2.path_sha256(ai_path), "2026-08-14T00:00:00Z")
    f2.write_json_atomic(audit_path, audit)
    args = f2.build_arg_parser().parse_args(
        [
            "cohort-confirm",
            "--wip-path",
            str(wip),
            "--ai-prescreen-path",
            str(ai_path),
            "--audit-path",
            str(audit_path),
            "--expected-audit-sha256",
            audit["audit_payload_sha256"],
            "--cohort-confirmation-path",
            str(output),
            "--reviewer-id",
            "reviewer",
            "--ack-structural-cohort-confirm",
        ]
    )
    assert f2.command_cohort_confirm(args) == 0
    loaded, duplicates = f2.strict_load_wip(auth, wip)
    assert duplicates == set()
    assert [record for record in loaded if record["pair_id"] == "generated_fact_999"] == [existing]


def test_fresh_exact_audit_confirmation_creates_structural_records(tmp_path, monkeypatch):
    auth = audit_authority(tmp_path)
    monkeypatch.setattr(f2, "load_authority", lambda: auth)
    wip = external_wip_path(tmp_path)
    ai_path = tmp_path / "ai.jsonl"
    audit_path = external_wip_path(tmp_path).with_name("audit.json")
    cohort_path = external_wip_path(tmp_path).with_name("cohort.json")
    write_full_ai_prescreen(ai_path, auth)
    records = [review_record(auth, pair_id, human_grammar_validity="MULTI_MEMBER_DEFECT") for pair_id in f2.INDIVIDUALLY_REVIEWED_AUDIT_PAIR_IDS]
    f2.write_wip_atomic(wip, records)
    audit = f2.structural_audit_artifact(auth, records, f2.load_ai_prescreen(ai_path, auth), f2.path_sha256(ai_path), "2026-08-14T00:00:00Z")
    f2.write_json_atomic(audit_path, audit)
    args = f2.build_arg_parser().parse_args(
        [
            "cohort-confirm",
            "--wip-path",
            str(wip),
            "--ai-prescreen-path",
            str(ai_path),
            "--audit-path",
            str(audit_path),
            "--expected-audit-sha256",
            audit["audit_payload_sha256"],
            "--cohort-confirmation-path",
            str(cohort_path),
            "--reviewer-id",
            "reviewer",
            "--ack-structural-cohort-confirm",
        ]
    )
    assert f2.command_cohort_confirm(args) == 0
    cohort_loaded = json.loads(cohort_path.read_text(encoding="utf-8"))
    confirmations = f2.load_cohort_confirmations(cohort_path, auth)
    loaded, duplicates = f2.strict_load_wip(auth, wip, confirmations)
    assert duplicates == set()
    assert cohort_loaded["cohort_confirmed_pair_ids"] == ["generated_fact_999"]
    assert [record["pair_id"] for record in loaded if record["review_method"] == f2.STRUCTURAL_COHORT_METHOD] == ["generated_fact_999"]


def test_transaction_failure_after_first_promotion_rolls_back_old_state(tmp_path, monkeypatch):
    auth = authority([structural_row("pair_a"), structural_row("pair_b")], tmp_path)
    cohort_path = external_wip_path(tmp_path).with_name("cohort.json")
    wip = external_wip_path(tmp_path)
    old_records = [review_record(auth, "pair_b")]
    f2.write_wip_atomic(wip, old_records)
    audit = audit_artifact(auth)
    artifact = f2.cohort_confirmation_artifact(auth, audit, tmp_path.parent / "audit.json", "reviewer", "2026-08-14T00:00:00Z")
    structural = f2.make_structural_cohort_record(auth, "pair_a", "reviewer", str(artifact["cohort_confirmation_id"]), "2026-08-14T00:00:00Z")
    original_promote = f2.promote_staged_file
    calls = {"count": 0}

    def fail_second(staged: Path, target: Path) -> None:
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("injected second promotion failure")
        original_promote(staged, target)

    monkeypatch.setattr(f2, "promote_staged_file", fail_second)
    with pytest.raises(OSError, match="injected second promotion failure"):
        f2.write_cohort_transaction(auth, cohort_path, artifact, wip, [old_records[0], structural])
    assert not cohort_path.exists()
    assert f2.load_wip(wip)[0] == old_records


def test_transaction_staged_wip_validation_failure_leaves_old_state(tmp_path):
    auth = authority([structural_row("pair_a"), structural_row("pair_b")], tmp_path)
    cohort_path = external_wip_path(tmp_path).with_name("cohort.json")
    wip = external_wip_path(tmp_path)
    old_records = [review_record(auth, "pair_b")]
    f2.write_wip_atomic(wip, old_records)
    audit = audit_artifact(auth)
    artifact = f2.cohort_confirmation_artifact(auth, audit, tmp_path.parent / "audit.json", "reviewer", "2026-08-14T00:00:00Z")
    bad_structural = f2.make_structural_cohort_record(auth, "pair_a", "reviewer", str(artifact["cohort_confirmation_id"]), "2026-08-14T00:00:00Z")
    bad_structural["source_record_sha256"] = "0" * 64
    with pytest.raises(f2.ReviewInfrastructureError, match="SOURCE_RECORD_HASH_MISMATCH"):
        f2.write_cohort_transaction(auth, cohort_path, artifact, wip, [old_records[0], bad_structural])
    assert not cohort_path.exists()
    assert f2.load_wip(wip)[0] == old_records


def test_transaction_success_establishes_matching_confirmation_and_wip(tmp_path):
    auth = authority([structural_row()], tmp_path)
    cohort_path = external_wip_path(tmp_path).with_name("cohort.json")
    wip = external_wip_path(tmp_path)
    audit = audit_artifact(auth)
    artifact = f2.cohort_confirmation_artifact(auth, audit, tmp_path.parent / "audit.json", "reviewer", "2026-08-14T00:00:00Z")
    structural = f2.make_structural_cohort_record(auth, "pair_a", "reviewer", str(artifact["cohort_confirmation_id"]), "2026-08-14T00:00:00Z")
    f2.write_cohort_transaction(auth, cohort_path, artifact, wip, [structural])
    confirmations = f2.load_cohort_confirmations(cohort_path, auth)
    assert f2.strict_load_wip(auth, wip, confirmations)[0] == [structural]


def test_no_summary_self_hash_and_f2_namespace_hashes_are_physical(tmp_path, monkeypatch):
    _auth, out = finalized_fixture(tmp_path, monkeypatch)
    summary = json.loads((out / "p3w6f2_hybrid_review_summary.json").read_text(encoding="utf-8"))
    completed_path = out / "p3w6f2_hybrid_review_completed.csv"
    decisions_path = out / "p3w6f2_hybrid_review_decisions.jsonl"
    assert "p3w6f2_hybrid_review_summary.json" not in summary["F2_output_sha256"]
    assert summary["F2_output_sha256"]["p3w6f2_hybrid_review_completed.csv"] == hashlib.sha256(completed_path.read_bytes()).hexdigest()
    assert summary["F2_output_sha256"]["p3w6f2_hybrid_review_decisions.jsonl"] == hashlib.sha256(decisions_path.read_bytes()).hexdigest()
    assert summary["F2_output_sha256_contract"] == "NON_SELF_REFERENTIAL_ARTIFACT_SHA256_MAP_V1"
    assert summary["summary_physical_sha256_embedded"] is False


def test_f2_artifact_paths_use_frozen_relative_namespace(tmp_path, monkeypatch):
    _auth, out = finalized_fixture(tmp_path, monkeypatch)
    summary = json.loads((out / "p3w6f2_hybrid_review_summary.json").read_text(encoding="utf-8"))
    assert summary["F2_artifact_paths"] == {
        "p3w6f2_hybrid_review_completed.csv": "reports/reason_router_p2_p3w6f2_hybrid_human_review_execution_ffffffff/p3w6f2_hybrid_review_completed.csv",
        "p3w6f2_hybrid_review_summary.json": "reports/reason_router_p2_p3w6f2_hybrid_human_review_execution_ffffffff/p3w6f2_hybrid_review_summary.json",
        "p3w6f2_hybrid_review_decisions.jsonl": "reports/reason_router_p2_p3w6f2_hybrid_human_review_execution_ffffffff/p3w6f2_hybrid_review_decisions.jsonl",
    }


def test_summary_status_remains_level1_only(tmp_path, monkeypatch):
    _auth, out = finalized_fixture(tmp_path, monkeypatch)
    summary = json.loads((out / "p3w6f2_hybrid_review_summary.json").read_text(encoding="utf-8"))
    assert summary["F2_execution_status"] == "P3W6F2_HYBRID_HUMAN_REVIEW_EXECUTION_COMPLETE_PENDING_RESULT_REVIEW"
    assert "REMEDIATION_COMPLETE" not in json.dumps(summary)
    assert "training admission" not in json.dumps(summary).lower()
    assert "polarity supervision" not in json.dumps(summary).lower()


def init_git_repo(repo: Path, track_script: bool = True) -> None:
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if track_script:
        script = repo / f2.EXECUTION_SCRIPT_REPO_PATH
        script.parent.mkdir(parents=True)
        script.write_text("print('tracked')\n", encoding="utf-8")
    else:
        (repo / "README.md").write_text("tracked\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "initial"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_clean_tracked_repository_permits_provenance_gate(tmp_path):
    repo = tmp_path / "repo"
    init_git_repo(repo)
    assert f2.require_tracked_execution_state(repo)


def test_unstaged_tracked_modification_causes_finalize_refusal(tmp_path):
    repo = tmp_path / "repo"
    init_git_repo(repo)
    (repo / f2.EXECUTION_SCRIPT_REPO_PATH).write_text("print('dirty')\n", encoding="utf-8")
    with pytest.raises(f2.ReviewInfrastructureError, match="DIRTY_TRACKED_EXECUTION_STATE"):
        f2.require_tracked_execution_state(repo)


def test_staged_tracked_modification_causes_finalize_refusal(tmp_path):
    repo = tmp_path / "repo"
    init_git_repo(repo)
    (repo / f2.EXECUTION_SCRIPT_REPO_PATH).write_text("print('staged')\n", encoding="utf-8")
    subprocess.run(["git", "add", f2.EXECUTION_SCRIPT_REPO_PATH], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    with pytest.raises(f2.ReviewInfrastructureError, match="DIRTY_TRACKED_EXECUTION_STATE"):
        f2.require_tracked_execution_state(repo)


def test_untracked_wip_does_not_cause_tracked_provenance_failure(tmp_path):
    repo = tmp_path / "repo"
    init_git_repo(repo)
    (repo / "untracked_wip.jsonl").write_text("{}\n", encoding="utf-8")
    assert f2.require_tracked_execution_state(repo)


def test_execution_script_must_be_tracked_at_head(tmp_path):
    repo = tmp_path / "repo"
    init_git_repo(repo, track_script=False)
    with pytest.raises(f2.ReviewInfrastructureError, match="UNTRACKED_EXECUTION_SCRIPT"):
        f2.require_tracked_execution_state(repo)


def test_dirty_provenance_failure_produces_no_target(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    init_git_repo(repo)
    prepare_reports_dir(repo)
    (repo / f2.EXECUTION_SCRIPT_REPO_PATH).write_text("print('dirty')\n", encoding="utf-8")
    auth = authority(tmp_path=repo)
    wip = tmp_path / "external_wip.jsonl"
    f2.write_wip_atomic(wip, [review_record(auth)])
    monkeypatch.setattr(f2, "EXPECTED_PAIR_COUNT", 1)
    with pytest.raises(f2.ReviewInfrastructureError, match="DIRTY_TRACKED_EXECUTION_STATE"):
        f2.finalize_artifacts(auth, wip)
    assert not any((repo / "reports").glob("reason_router_p2_p3w6f2_hybrid_human_review_execution_*"))


def test_git_object_authority_loader_does_not_accept_crlf_hash(monkeypatch, tmp_path):
    manifest = {
        "review_protocol_version": f2.P3W5_REVIEW_PROTOCOL_VERSION,
        "compatibility_matrix_version": f2.COMPATIBILITY_MATRIX_VERSION,
        "source_hash_version": f2.SOURCE_HASH_VERSION,
        "immutable_source_columns": f2.SOURCE_FIELDS,
        "human_review_fields": f2.HUMAN_FIELDS,
        "reviewer_provenance_fields": f2.P3W5_PROVENANCE_FIELDS,
        "completed_csv_schema_order": f2.P3W5_COMPLETED_CSV_FIELDS,
        "input_artifact_authority": {
            "template_path": "template.csv",
            "template_git_lf_sha256": "lf",
            "template_historical_original_execution_crlf_sha256": "crlf",
            "pair_authority_path": "pairs.jsonl",
            "pair_authority_sha256": "pair",
            "summary_authority_path": "summary.json",
            "summary_authority_sha256": "summary",
        },
    }
    monkeypatch.setattr(f2, "resolve_repo_root", lambda repo_root=None: tmp_path)
    monkeypatch.setattr(f2, "require_commit", lambda repo_root, commit: None)
    monkeypatch.setattr(f2, "sha256_bytes", lambda payload: {"template": "crlf", "pairs": "pair", "summary": "summary"}[payload.decode()])
    monkeypatch.setattr(
        f2,
        "git_object_bytes",
        lambda repo_root, commit, path: json.dumps(manifest).encode() if path == f2.P1_MANIFEST_PATH else path.split(".")[0].encode(),
    )
    with pytest.raises(f2.ReviewInfrastructureError, match="SOURCE_RECORD_HASH_MISMATCH"):
        f2.load_authority(tmp_path)


def test_production_authority_contract_loads_actual_tracked_f2_authority():
    auth = f2.load_authority()
    assert len(auth.source_template_fields) == 33
    assert auth.source_template_fields[:27] == f2.SOURCE_FIELDS
    assert auth.source_template_fields[27:] == f2.HUMAN_FIELDS
    assert len(auth.source_rows) == 119
    assert len(set(auth.ordered_pair_ids)) == 119
    assert len(auth.ordered_pair_ids) == 119
    assert auth.summary["aggregates"]["family_counts"]["F2_pair_count"] == 119
    assert auth.summary["aggregates"]["family_counts"]["F2_complete_triple_members"] == 357
    assert auth.input_artifact_sha256[auth.input_artifact_paths["template"]] == "ccc539e743d1a4226391cdca1422bb0a1054c53fd7c53a4210a54271d1e9e8a5"
    assert auth.input_artifact_sha256[auth.input_artifact_paths["pairs"]] == "850ac6e8924fe334fa7f18659d204f6e0546381b1c3d3eb601f893f3eb00a493"
    assert auth.input_artifact_sha256[auth.input_artifact_paths["summary"]] == "7c0cc383dde38a1c564dae445a78eaf9171b8648d0720de3a2acc0ba68e68e80"
    assert auth.p3w4_artifact_commit == f2.P3W4_RESULT_AUTHORITY_COMMIT
    assert f2.P3W4_EXECUTION_COMMIT == "ca99038d812696467a4330cffc1c4c5b5f72cfe2"
