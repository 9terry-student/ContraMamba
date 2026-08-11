from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

from scripts import analyze_reason_router_p3w6f1_deterministic_polarity_regeneration as f1
from scripts import build_controlled_v5 as generator


def fact(predicate: str = "approved") -> dict:
    return {
        "pair_id": "p",
        "title": "Dr",
        "name": "Mira Chen",
        "alternate_title": "Mr",
        "alternate_name": "Jon Bell",
        "role": "director",
        "alternate_role": "auditor",
        "predicate": predicate,
        "alternate_predicate": "reviewed",
        "object": "the Orion project",
        "alternate_object": "the Vega project",
        "time": "Monday",
        "alternate_time": "Tuesday",
        "location": "Seoul",
        "alternate_location": "Busan",
    }


def row(pair_id: str, intervention: str, evidence: str, final: str = "REFUTE") -> dict:
    return {
        "id": f"{pair_id}__{intervention}",
        "pair_id": pair_id,
        "claim": f"{pair_id} claim",
        "evidence": evidence,
        "final_label": final,
        "frame_compatible_label": 1,
        "predicate_covered_label": 1,
        "sufficiency_label": 1,
        "polarity_label": final if final in {"SUPPORT", "REFUTE"} else "NONE",
        "primary_failure_type": "polarity" if intervention == "polarity_flip" else "none",
        "intervention_type": intervention,
    }


def stage185_sidecar(
    row_id: str,
    pair_id: str,
    *,
    before: bool = False,
    source_sha: str = "source-sha",
    builder_sha: str = "builder-sha",
    override: dict | None = None,
) -> dict:
    intervention = row_id.rsplit("__", 1)[1]
    data = {
        "row_id": row_id,
        "pair_id": pair_id,
        "split": "train",
        "intervention_type": intervention,
        "frame_compatible_label": 1,
        "grammar_status": "FAIL" if before else "PASS",
        "integrity_status": "INELIGIBLE" if before else "ELIGIBLE",
        "canonical_status": "PASS",
        "dataset_source_status": "PASS",
        "schema_status": "PASS",
        "intervention_contract_status": "PASS",
        "polarity_contamination_status": "PASS",
        "time_swap_status": "PASS",
        "audit_expected_axes": ["polarity"],
        "audit_changed_axes": ["polarity"],
        "audit_pair_failure_scope": "none",
        "eligible_for_positive_margin": not before,
        "reason_codes": ["DID_NOT_INFLECTED_PREDICATE"] if before else ["ELIGIBLE_CLEAN_COMPATIBLE"],
        "canonical_row_id": f"{pair_id}__none",
        "family_contract_id": f"stage185a_v1:{intervention}",
        "rule_version": "stage185a_v1",
        "source_dataset_path": "data/controlled_v5_v3_without_time_swap.jsonl",
        "source_dataset_sha256": source_sha,
        "generator_source_path": "/kaggle/working/ContraMamba/scripts/build_controlled_v5.py",
        "generator_source_sha256": "generator-sha",
        "stage182a_report_sha256": "stage182-sha",
        "stage184a_report_sha256": "stage184-sha",
        "integrity_builder_sha256": builder_sha,
        "created_at": "2026-08-10T00:00:00Z",
        "audit_preserved_axes": ["location", "name", "object", "predicate", "role", "time", "title"],
    }
    if override:
        data.update(override)
    return data


def pass_audit(pair_id: str) -> dict:
    return {
        "pair_id": pair_id,
        "semantic_validation_status": "DETERMINISTIC_POLARITY_REPAIR_PASS",
        "semantic_polarity_preserved": True,
        "candidate_accepted": True,
        "ordered_rejection_codes": [],
        "semantic_validation_evidence": {"stage185_transition_pass": True},
    }


def stage185_runtime_authority_stub() -> dict:
    return {
        "stage185_runtime_authority_pass": True,
        "stage184_contract_matrix_path": f1.P3W6F1_STAGE184_CONTRACT_MATRIX_PATH,
        "stage185_integrity_builder_source_sha256": "builder-sha",
        "resolved_grammar_validator_source_path": f1.P3W6F1_STAGE182_ANALYZER_PATH,
        "resolved_grammar_validator_source_sha256": "grammar-sha",
    }

def authority_record(pair_id: str = "p", family: str = "F1", predicate: str = "approved") -> dict:
    return {
        "pair_id": pair_id,
        "family": family,
        "automatic_root_cause_class": "F1_TRUE_POLARITY_GENERATION_DEFECT",
        "remediation_state": "REGENERATION_REQUIRED",
        "members": {
            "polarity_flip": {
                "source_row": row(pair_id, "polarity_flip", f"A did not {predicate} B."),
                "grammar_rule_reproduction": {
                    "fact_predicate": predicate,
                    "matched_surface_span": f"did not {predicate}",
                },
            }
        },
    }


def test_analyzer_ast_parse_syntax_passes():
    ast.parse(Path(f1.__file__).read_text(encoding="utf-8"))


def test_override_none_preserves_old_behavior():
    assert generator._statement(fact(), negative=True) == "Dr Mira Chen, the director, did not approved the Orion project in Seoul during Monday."


def test_negative_true_valid_override_uses_supplied_base_surface():
    assert generator._statement(fact(), negative=True, predicate_surface_override="approve") == "Dr Mira Chen, the director, did not approve the Orion project in Seoul during Monday."


def test_negative_false_override_fails_closed():
    with pytest.raises(ValueError, match="predicate_surface_override requires negative=True"):
        generator._statement(fact(), predicate_surface_override="approve")


def test_canonical_paraphrase_and_positive_polarity_flip_do_not_use_override():
    records = generator._build_records([fact(), fact("released") | {"pair_id": "q"}])
    by_id = {record["id"]: record for record in records}
    assert "did not approved" not in by_id["p__none"]["evidence"]
    assert "did not approved" not in by_id["p__paraphrase"]["evidence"]
    assert "did not approved" in by_id["p__polarity_flip"]["evidence"]
    assert "did not release" not in by_id["p__polarity_flip"]["evidence"]


def test_positive_polarity_flip_never_uses_override_under_repair_gate():
    templates = [fact("approved"), fact("released") | {"pair_id": "q"}]
    records = generator._build_records(templates, authorized_negative_polarity_flip_row_ids={"p__polarity_flip"})
    by_id = {record["id"]: record for record in records}
    assert "did not approve" in by_id["p__polarity_flip"]["evidence"]
    assert "did not release" not in by_id["q__polarity_flip"]["evidence"]
    assert by_id["q__polarity_flip"]["final_label"] == "SUPPORT"


def test_generator_membership_superset_repairs_only_authorized_negative_rows():
    templates = [
        fact("approved") | {"pair_id": "p"},
        fact("released") | {"pair_id": "q"},
        fact("opened") | {"pair_id": "r"},
        fact("won") | {"pair_id": "s"},
    ]
    consumed: set[str] = set()
    records = generator._build_records(
        templates,
        authorized_negative_polarity_flip_row_ids={"p__polarity_flip"},
        repair_consumed_row_ids=consumed,
    )
    by_id = {record["id"]: record for record in records}
    assert "did not approve" in by_id["p__polarity_flip"]["evidence"]
    assert "did not released" in by_id["q__polarity_flip"]["evidence"]
    assert consumed == {"p__polarity_flip"}


def test_generator_membership_missing_authorized_row_fails_before_consumption():
    consumed: set[str] = set()
    with pytest.raises(ValueError, match="TARGET_SCOPE_MEMBERSHIP_UNRESOLVED"):
        generator._build_records(
            [fact("approved") | {"pair_id": "p"}, fact("released") | {"pair_id": "q"}],
            authorized_negative_polarity_flip_row_ids={"missing__polarity_flip"},
            repair_consumed_row_ids=consumed,
        )
    assert consumed == set()


def test_f1_public_repair_wrapper_cardinality_120_and_122_blocked():
    ids_120 = {f"p{i:03d}__polarity_flip" for i in range(120)}
    ids_122 = {f"p{i:03d}__polarity_flip" for i in range(122)}
    with pytest.raises(ValueError, match="F1_AUTHORITY_CARDINALITY_MISMATCH"):
        generator.build_controlled_records_with_f1_polarity_repair(300, ids_120)
    with pytest.raises(ValueError, match="F1_AUTHORITY_CARDINALITY_MISMATCH"):
        generator.build_controlled_records_with_f1_polarity_repair(300, ids_122)


def test_f1_public_repair_audit_observes_exact_121_consumed_rows():
    structural = sorted(generator._negative_polarity_flip_row_ids(generator.fact_templates_for_count(300)))
    authorized = set(structural[:121])
    records, audit = generator.build_controlled_records_with_f1_polarity_repair_audit(300, authorized)
    assert len(records) > 0
    assert audit["repair_consumed_row_ids"] == sorted(authorized)


def replay_case() -> tuple[list[dict], list[dict], set[str]]:
    pair_count = 300
    baseline = generator.build_controlled_records(pair_count)
    structural = sorted(generator._negative_polarity_flip_row_ids(generator.fact_templates_for_count(pair_count)))
    authorized = set(structural[:121])
    repaired, _audit = generator.build_controlled_records_with_f1_polarity_repair_audit(pair_count, authorized)
    return baseline, repaired, authorized


def test_repair_consumption_spoof_not_authoritative_in_analyzer_source():
    source = Path(f1.__file__).read_text(encoding="utf-8")
    assert 'invocation.get("repair_consumed_row_ids")' not in source


def test_positive_repaired_output_replay_identity_returns_generator_owned_consumption():
    baseline, repaired, authorized = replay_case()
    replay = f1.validate_repaired_output_replay_identity(baseline, repaired, authorized)
    assert replay["generator_replay_identity_pass"] is True
    assert replay["actual_generator_repair_consumed_row_ids"] == sorted(authorized)


def test_repaired_output_replay_identity_rejects_protected_field_tamper():
    baseline, repaired, authorized = replay_case()
    tampered = [dict(row) for row in repaired]
    tampered[0] = dict(tampered[0]) | {"claim": tampered[0]["claim"] + " tampered"}
    replay = f1.validate_repaired_output_replay_identity(baseline, tampered, authorized)
    assert replay["generator_replay_identity_status"] == "PROVENANCE_IDENTITY_MISMATCH"


def test_repaired_output_replay_identity_rejects_one_evidence_byte_tamper():
    baseline, repaired, authorized = replay_case()
    tampered = [dict(row) for row in repaired]
    tampered[0] = dict(tampered[0]) | {"evidence": tampered[0]["evidence"] + " "}
    replay = f1.validate_repaired_output_replay_identity(baseline, tampered, authorized)
    assert replay["generator_replay_identity_status"] == "PROVENANCE_IDENTITY_MISMATCH"


def test_required_f1_surface_coverage():
    required = {"approved", "released", "opened", "won"}
    coverage = f1.validate_base_form_coverage(required, generator._BASE_PREDICATE_BY_INFLECTED)
    assert coverage["missing_base_form_surfaces"] == []
    assert coverage["ambiguous_base_form_surfaces"] == []
    assert set(coverage["covered_surfaces"]) >= required


def test_missing_surface_detected():
    coverage = f1.validate_base_form_coverage({"not_in_mapping"}, generator._BASE_PREDICATE_BY_INFLECTED)
    assert coverage["missing_base_form_surfaces"] == ["not_in_mapping"]
    assert coverage["coverage_pass"] is False


def test_ambiguous_conflicting_mapping_rejected():
    coverage = f1.validate_base_form_coverage({"approved"}, [("approved", "approve"), ("approved", "approv")])
    assert coverage["ambiguous_base_form_surfaces"] == ["approved"]
    assert coverage["coverage_pass"] is False


def test_mapping_provenance_deterministic():
    first = f1.validate_base_form_coverage({"approved"}, generator._BASE_PREDICATE_BY_INFLECTED)
    second = f1.validate_base_form_coverage({"approved"}, generator._BASE_PREDICATE_BY_INFLECTED)
    assert first == second
    assert first["base_form_derivation_method"] == "generator_owned_explicit_mapping"
    assert first["base_form_source_symbol"] == "_BASE_PREDICATE_BY_INFLECTED"


def test_exact_authority_extraction_and_f2_not_included():
    records = [authority_record("p"), authority_record("f2", family="F2")]
    targets = f1.extract_authorized_f1_targets(records, {"p", "f2"})
    assert targets["F1_target_pair_count"] == 1
    assert targets["F1_target_pair_ids"] == ["p"]
    assert targets["authorized_F1_row_ids"] == ["p__polarity_flip"]


def test_121_target_pair_row_expectation_shape():
    records = [authority_record(f"p{i:03d}") for i in range(121)]
    ids = {f"p{i:03d}" for i in range(121)}
    targets = f1.extract_authorized_f1_targets(records, ids)
    assert targets["F1_target_pair_count"] == 121
    assert targets["F1_target_row_count"] == 121


def test_target_scope_membership_allows_structural_superset_and_reports_remainder():
    structural = {"p__polarity_flip", "q__polarity_flip"}
    membership = f1.validate_target_scope_membership(structural, {"p__polarity_flip"})
    assert membership["target_scope_membership_pass"] is True
    assert membership["non_authorized_structural_negative_polarity_flip_row_ids"] == ["q__polarity_flip"]
    mismatch = f1.validate_target_scope_membership(structural, {"other__polarity_flip"})
    assert mismatch["target_scope_status"] == "TARGET_SCOPE_MEMBERSHIP_UNRESOLVED"


def test_required_inflected_surfaces_from_authority():
    records = [authority_record("p", predicate="released")]
    assert f1.required_f1_inflected_predicate_surfaces(records, {"p"}) == {"released"}


def test_valid_deterministic_repair_pass_contract():
    original = row("p", "polarity_flip", "A did not approved B.")
    regenerated = row("p", "polarity_flip", "A did not approve B.")
    canonical = row("p", "none", "A approved B.", "SUPPORT")
    audit = f1.semantic_audit_record(
        original,
        regenerated,
        canonical,
        sidecar_before=stage185_sidecar("p__polarity_flip", "p", before=True),
        sidecar_after=stage185_sidecar("p__polarity_flip", "p"),
        inflected_predicate_surface="approved",
        expected_base_predicate="approve",
    )
    assert audit["semantic_validation_status"] == "DETERMINISTIC_POLARITY_REPAIR_PASS"
    assert audit["semantic_polarity_preserved"] is True
    assert audit["candidate_accepted"] is True
    assert audit["duplicate_or_missing_tokens"] == {"did": 1, "not": 1}
    assert audit["outside_span_byte_identity"] is True
    assert audit["semantic_validation_evidence"]["claim_identity"] is True
    assert audit["semantic_validation_evidence"]["label_identity"] is True


def test_missing_or_ambiguous_authority_manual_review_required():
    original = row("p", "polarity_flip", "A did not approved B.")
    regenerated = row("p", "polarity_flip", "A did not approved B.")
    canonical = row("p", "none", "A approved B.", "SUPPORT")
    audit = f1.semantic_audit_record(
        original,
        regenerated,
        canonical,
        sidecar_before=stage185_sidecar("p__polarity_flip", "p", before=True),
        sidecar_after=stage185_sidecar("p__polarity_flip", "p"),
        inflected_predicate_surface="missing_surface",
        expected_base_predicate=None,
    )
    assert audit["semantic_validation_status"] == "MANUAL_REVIEW_REQUIRED"
    assert audit["semantic_polarity_preserved"] is None
    assert audit["candidate_accepted"] is False
    assert "SEMANTIC_AUTHORITY_UNRESOLVED" in audit["ordered_rejection_codes"]


def test_explicit_contradiction_rejected():
    original = row("p", "polarity_flip", "A did not approved B.")
    regenerated = row("p", "polarity_flip", "A did not approve B.", "SUPPORT")
    canonical = row("p", "none", "A approved B.", "SUPPORT")
    audit = f1.semantic_audit_record(
        original,
        regenerated,
        canonical,
        sidecar_before=stage185_sidecar("p__polarity_flip", "p", before=True),
        sidecar_after=stage185_sidecar("p__polarity_flip", "p"),
        inflected_predicate_surface="approved",
        expected_base_predicate="approve",
    )
    assert audit["semantic_validation_status"] == "REJECTED"
    assert audit["semantic_polarity_preserved"] is False
    assert audit["candidate_accepted"] is False


def locally_valid_deterministic_repair_audit(pair_id: str = "p") -> dict:
    return f1.semantic_audit_record(
        row(pair_id, "polarity_flip", "A did not approved B."),
        row(pair_id, "polarity_flip", "A did not approve B."),
        row(pair_id, "none", "A approved B.", "SUPPORT"),
        sidecar_before=stage185_sidecar(f"{pair_id}__polarity_flip", pair_id, before=True),
        sidecar_after=stage185_sidecar(f"{pair_id}__polarity_flip", pair_id),
        inflected_predicate_surface="approved",
        expected_base_predicate="approve",
    )


def finalized_single_candidate(
    *,
    full_output_pass: bool = True,
    stage185_pass: bool = True,
    execution_pass: bool = True,
    execution_status: str = "PASS",
) -> dict:
    rows = f1.finalize_candidate_acceptance(
        [locally_valid_deterministic_repair_audit()],
        full_output_isolation_validation={"full_output_isolation_pass": full_output_pass},
        stage185_provenance_validation={"stage185_provenance_pass": stage185_pass},
        execution_provenance_validation={
            "execution_provenance_pass": execution_pass,
            "execution_provenance_status": execution_status,
        },
    )
    return rows[0]


def test_candidate_acceptance_blocked_when_full_output_isolation_fails():
    finalized = finalized_single_candidate(full_output_pass=False)
    assert finalized["candidate_accepted"] is False
    assert finalized["semantic_validation_status"] != "DETERMINISTIC_POLARITY_REPAIR_PASS"
    assert finalized["semantic_polarity_preserved"] is None
    assert "SEMANTIC_AUTHORITY_UNRESOLVED" in finalized["ordered_rejection_codes"]
    assert "FULL_OUTPUT_ISOLATION_FAILED" in finalized["ordered_rejection_codes"]


def test_candidate_acceptance_blocked_when_stage185_provenance_fails():
    finalized = finalized_single_candidate(stage185_pass=False)
    assert finalized["candidate_accepted"] is False
    assert finalized["semantic_validation_status"] != "DETERMINISTIC_POLARITY_REPAIR_PASS"
    assert finalized["semantic_polarity_preserved"] is None
    assert "STAGE185_PROVENANCE_UNRESOLVED" in finalized["ordered_rejection_codes"]


def test_candidate_acceptance_blocked_when_execution_provenance_fails():
    finalized = finalized_single_candidate(
        execution_pass=False,
        execution_status="PROVENANCE_IDENTITY_MISMATCH",
    )
    assert finalized["candidate_accepted"] is False
    assert finalized["semantic_validation_status"] != "DETERMINISTIC_POLARITY_REPAIR_PASS"
    assert finalized["semantic_polarity_preserved"] is None
    assert "PROVENANCE_IDENTITY_MISMATCH" in finalized["ordered_rejection_codes"]


def test_candidate_acceptance_requires_all_global_authority_gates():
    finalized = finalized_single_candidate()
    assert finalized["candidate_accepted"] is True
    assert finalized["semantic_validation_status"] == "DETERMINISTIC_POLARITY_REPAIR_PASS"
    assert finalized["semantic_polarity_preserved"] is True


def test_finalizer_never_upgrades_manual_or_rejected_candidate():
    manual = locally_valid_deterministic_repair_audit("m") | {
        "semantic_validation_status": "MANUAL_REVIEW_REQUIRED",
        "semantic_polarity_preserved": None,
        "candidate_accepted": False,
        "ordered_rejection_codes": ["SEMANTIC_AUTHORITY_UNRESOLVED"],
    }
    rejected = locally_valid_deterministic_repair_audit("r") | {
        "semantic_validation_status": "REJECTED",
        "semantic_polarity_preserved": False,
        "candidate_accepted": False,
        "ordered_rejection_codes": ["OUTSIDE_SPAN_CHANGED"],
    }
    finalized = f1.finalize_candidate_acceptance(
        [manual, rejected],
        full_output_isolation_validation={"full_output_isolation_pass": True},
        stage185_provenance_validation={"stage185_provenance_pass": True},
        execution_provenance_validation={"execution_provenance_pass": True},
    )
    assert [row["semantic_validation_status"] for row in finalized] == ["MANUAL_REVIEW_REQUIRED", "REJECTED"]
    assert [row["candidate_accepted"] for row in finalized] == [False, False]


def test_summary_accepted_count_uses_finalized_candidate_acceptance():
    downgraded = finalized_single_candidate(full_output_pass=False)
    summary = f1.build_summary(
        ["p"],
        ["p"],
        [downgraded],
        authority_cardinality={"authority_cardinality_pass": True},
        target_scope={"target_scope_membership_pass": True},
        base_form_coverage={"coverage_pass": True},
        stage185_provenance_validation={"stage185_provenance_pass": True},
        full_output_validation={"full_output_isolation_pass": False},
        provenance_validation={"execution_provenance_pass": True},
        execution_provenance={"F1_input_sha256": "in", "F1_execution_commit": "c", "F1_output_sha256": "out"},
    )
    assert summary["F1_accepted_candidate_pair_ids"] == []
    assert summary["F1_accepted_candidate_count"] == 0
    assert summary["F1_manual_review_required_pair_ids"] == ["p"]


def test_full_output_isolation_contract_fields():
    baseline = [
        row("p", "none", "A approved B.", "SUPPORT"),
        row("p", "polarity_flip", "A did not approved B."),
        row("f2", "none", "A did not released B."),
        row("u", "paraphrase", "A released B.", "SUPPORT"),
    ]
    repaired = [dict(item) for item in baseline]
    repaired[1] = dict(repaired[1]) | {"evidence": "A did not approve B."}
    isolation = f1.full_output_isolation(
        baseline,
        repaired,
        authorized_f1_row_ids={"p__polarity_flip"},
        structural_negative_polarity_flip_row_ids={"p__polarity_flip", "u__polarity_flip"},
        repair_consumed_row_ids={"p__polarity_flip"},
        f2_row_ids={"f2__none"},
    )
    assert isolation["baseline_row_count"] == isolation["repaired_row_count"]
    assert isolation["baseline_id_sequence"] == isolation["repaired_id_sequence"]
    assert isolation["row_order_changed"] is False
    assert isolation["changed_ids"] == ["p__polarity_flip"]
    assert isolation["repair_consumed_row_ids"] == ["p__polarity_flip"]
    assert isolation["non_authorized_structural_negative_polarity_flip_row_ids"] == ["u__polarity_flip"]
    assert isolation["non_authorized_structural_negative_polarity_flip_changed_row_ids"] == []
    assert isolation["evidence_changed_row_ids"] == ["p__polarity_flip"]
    assert isolation["claim_changed_row_ids"] == []
    assert isolation["non_text_field_changed_row_ids"] == []
    assert isolation["F2_changed_row_ids"] == []
    assert isolation["canonical_changed_row_ids"] == []
    assert isolation["paraphrase_changed_row_ids"] == []
    assert isolation["unaffected_changed_row_ids"] == []
    assert f1.validate_full_output_isolation(isolation)["full_output_isolation_pass"] is True


def test_unauthorized_structural_negative_change_blocks_post_isolation():
    baseline = [
        row("p", "polarity_flip", "A did not approved B."),
        row("u", "polarity_flip", "A did not released B."),
    ]
    repaired = [dict(item) for item in baseline]
    repaired[0] = dict(repaired[0]) | {"evidence": "A did not approve B."}
    repaired[1] = dict(repaired[1]) | {"evidence": "A did not release B."}
    isolation = f1.full_output_isolation(
        baseline,
        repaired,
        authorized_f1_row_ids={"p__polarity_flip"},
        structural_negative_polarity_flip_row_ids={"p__polarity_flip", "u__polarity_flip"},
        repair_consumed_row_ids={"p__polarity_flip"},
    )
    validation = f1.validate_full_output_isolation(isolation)
    assert isolation["non_authorized_structural_negative_polarity_flip_changed_row_ids"] == ["u__polarity_flip"]
    assert validation["full_output_isolation_status"] == "FULL_OUTPUT_ISOLATION_FAILED"


def test_accounting_count_symmetry_partition_and_decision():
    audit_rows = [
        pass_audit("p"),
        {"pair_id": "q", "semantic_validation_status": "MANUAL_REVIEW_REQUIRED"},
        {"pair_id": "r", "semantic_validation_status": "REJECTED"},
    ]
    summary = f1.build_summary(["p", "q", "r", "m"], ["p", "q", "r", "u"], audit_rows)
    f1.validate_summary_accounting(summary)
    assert summary["F1_target_pair_count"] == len(summary["F1_target_pair_ids"])
    assert summary["F1_missing_candidate_pair_ids"] == ["m"]
    assert summary["F1_unauthorized_candidate_pair_ids"] == ["u"]
    assert summary["F1_execution_decision"] == "P3W5_F1_REGENERATION_COMPLETE_WITH_BLOCKERS_PENDING_RESULT_REVIEW"


def test_all_accepted_decision_derivation():
    pair_ids = [f"p{i:03d}" for i in range(121)]
    audit_rows = [pass_audit(pair_id) for pair_id in pair_ids]
    summary = f1.build_summary(
        pair_ids,
        pair_ids,
        audit_rows,
        authority_cardinality={"authority_cardinality_pass": True},
        target_scope={"target_scope_membership_pass": True},
        base_form_coverage={"coverage_pass": True},
        stage185_provenance_validation={"stage185_provenance_pass": True},
        full_output_validation={"full_output_isolation_pass": True},
        provenance_validation={"execution_provenance_pass": True},
        execution_provenance={"F1_input_sha256": "in", "F1_execution_commit": "c", "F1_output_sha256": "out"},
    )
    assert summary["F1_execution_blockers"] == []
    assert summary["F1_execution_decision"] == "P3W5_F1_REGENERATION_COMPLETE_ALL_CANDIDATES_ACCEPTED_PENDING_RESULT_REVIEW"


def test_all_accepted_execution_status_and_decision_use_exact_authority_enum():
    pair_ids = [f"p{i:03d}" for i in range(121)]
    summary = f1.build_summary(
        pair_ids,
        pair_ids,
        [pass_audit(pair_id) for pair_id in pair_ids],
        authority_cardinality={"authority_cardinality_pass": True},
        target_scope={"target_scope_membership_pass": True},
        base_form_coverage={"coverage_pass": True},
        stage185_provenance_validation={"stage185_provenance_pass": True},
        full_output_validation={"full_output_isolation_pass": True},
        provenance_validation={"execution_provenance_pass": True},
        execution_provenance={"F1_input_sha256": "in", "F1_execution_commit": "c", "F1_output_sha256": "out"},
    )
    expected = "P3W5_F1_REGENERATION_COMPLETE_ALL_CANDIDATES_ACCEPTED_PENDING_RESULT_REVIEW"
    assert summary["F1_execution_status"] == expected
    assert summary["F1_execution_decision"] == expected


def test_blocked_execution_status_and_decision_use_exact_authority_enum():
    pair_ids = [f"p{i:03d}" for i in range(121)]
    summary = f1.build_summary(
        pair_ids,
        pair_ids,
        [pass_audit(pair_id) for pair_id in pair_ids],
        authority_cardinality={"authority_cardinality_pass": True},
        target_scope={"target_scope_membership_pass": True},
        base_form_coverage={"coverage_pass": True},
        stage185_provenance_validation={"stage185_provenance_pass": True},
        full_output_validation={"full_output_isolation_pass": False},
        provenance_validation={"execution_provenance_pass": True},
        execution_provenance={"F1_input_sha256": "in", "F1_execution_commit": "c", "F1_output_sha256": "out"},
    )
    expected = "P3W5_F1_REGENERATION_COMPLETE_WITH_BLOCKERS_PENDING_RESULT_REVIEW"
    assert summary["F1_execution_status"] == expected
    assert summary["F1_execution_decision"] == expected


def test_deprecated_generic_complete_pending_result_review_status_absent():
    source = Path(f1.__file__).read_text(encoding="utf-8")
    deprecated = "_".join(["COMPLETE", "PENDING", "RESULT", "REVIEW"])
    assert deprecated not in source


def test_authority_cardinality_120_targets_blocked():
    targets = {
        "F1_target_pair_count": 120,
        "F1_target_row_count": 120,
        "F1_target_pair_ids": [f"p{i:03d}" for i in range(120)],
        "authorized_F1_row_ids": [f"p{i:03d}__polarity_flip" for i in range(120)],
    }
    result = f1.validate_authority_cardinality(targets)
    assert result["authority_cardinality_status"] == "F1_AUTHORITY_CARDINALITY_MISMATCH"


def test_authority_cardinality_121_targets_passes():
    targets = {
        "F1_target_pair_count": 121,
        "F1_target_row_count": 121,
        "F1_target_pair_ids": [f"p{i:03d}" for i in range(121)],
        "authorized_F1_row_ids": [f"p{i:03d}__polarity_flip" for i in range(121)],
    }
    assert f1.validate_authority_cardinality(targets)["authority_cardinality_pass"] is True


def test_authority_cardinality_122_targets_blocked():
    targets = {
        "F1_target_pair_count": 122,
        "F1_target_row_count": 122,
        "F1_target_pair_ids": [f"p{i:03d}" for i in range(122)],
        "authorized_F1_row_ids": [f"p{i:03d}__polarity_flip" for i in range(122)],
    }
    result = f1.validate_authority_cardinality(targets)
    assert result["authority_cardinality_status"] == "F1_AUTHORITY_CARDINALITY_MISMATCH"


def test_actual_authority_artifact_base_form_preflight():
    root = Path(__file__).resolve().parents[1]
    summary = f1.load_json(root / "reports/reason_router_p2_p3w4_canonical_grammar_authority_execution_ca99038d/p3w4_canonical_grammar_authority_summary.json")
    pairs = f1.load_jsonl(root / "reports/reason_router_p2_p3w4_canonical_grammar_authority_execution_ca99038d/p3w4_canonical_grammar_authority_pairs.jsonl")
    supporting = f1.extract_decision_supporting_pair_ids(summary)
    targets = f1.extract_authorized_f1_targets(pairs, supporting)
    required = f1.required_f1_inflected_predicate_surfaces(pairs, set(targets["F1_target_pair_ids"]))
    coverage = f1.validate_base_form_coverage(required, generator._BASE_PREDICATE_BY_INFLECTED)
    structural = f1.structural_negative_polarity_flip_row_ids_for_pair_count(300)
    assert required
    assert targets["F1_target_row_count"] == 121
    assert set(targets["authorized_F1_row_ids"]) <= structural
    assert len(structural) > len(targets["authorized_F1_row_ids"])
    assert coverage["missing_base_form_surfaces"] == []
    assert coverage["ambiguous_base_form_surfaces"] == []
    assert coverage["coverage_pass"] is True


def summary_with_isolation(isolation_validation: dict) -> dict:
    pair_ids = [f"p{i:03d}" for i in range(121)]
    return f1.build_summary(
        pair_ids,
        pair_ids,
        [pass_audit(pair_id) for pair_id in pair_ids],
        authority_cardinality={"authority_cardinality_pass": True},
        target_scope={"target_scope_membership_pass": True},
        base_form_coverage={"coverage_pass": True},
        stage185_provenance_validation={"stage185_provenance_pass": True},
        full_output_validation=isolation_validation,
        provenance_validation={"execution_provenance_pass": True},
        execution_provenance={"F1_input_sha256": "in", "F1_execution_commit": "c", "F1_output_sha256": "out"},
    )


def test_full_output_f2_change_blocks_all_accepted():
    summary = summary_with_isolation({"full_output_isolation_pass": False, "full_output_isolation_failures": ["F2_changed_row_ids"]})
    assert "FULL_OUTPUT_ISOLATION_FAILED" in summary["F1_execution_blockers"]
    assert summary["F1_execution_decision"] == "P3W5_F1_REGENERATION_COMPLETE_WITH_BLOCKERS_PENDING_RESULT_REVIEW"


def test_full_output_canonical_change_blocks_all_accepted():
    summary = summary_with_isolation({"full_output_isolation_pass": False, "full_output_isolation_failures": ["canonical_changed_row_ids"]})
    assert "FULL_OUTPUT_ISOLATION_FAILED" in summary["F1_execution_blockers"]


def test_full_output_paraphrase_change_blocks_all_accepted():
    summary = summary_with_isolation({"full_output_isolation_pass": False, "full_output_isolation_failures": ["paraphrase_changed_row_ids"]})
    assert "FULL_OUTPUT_ISOLATION_FAILED" in summary["F1_execution_blockers"]


def test_full_output_unaffected_change_blocks_all_accepted():
    summary = summary_with_isolation({"full_output_isolation_pass": False, "full_output_isolation_failures": ["unaffected_changed_row_ids"]})
    assert "FULL_OUTPUT_ISOLATION_FAILED" in summary["F1_execution_blockers"]


def test_full_output_row_order_change_blocks_all_accepted():
    summary = summary_with_isolation({"full_output_isolation_pass": False, "full_output_isolation_failures": ["row_order_changed"]})
    assert "FULL_OUTPUT_ISOLATION_FAILED" in summary["F1_execution_blockers"]


def test_generated_candidate_accounting_uses_authorized_candidates_not_structural_universe():
    target_pair_ids = [f"p{i:03d}" for i in range(121)]
    structural_pair_ids = target_pair_ids + [f"u{i:03d}" for i in range(29)]
    summary = f1.build_summary(
        target_pair_ids,
        target_pair_ids,
        [pass_audit(pair_id) for pair_id in target_pair_ids],
        authority_cardinality={"authority_cardinality_pass": True},
        target_scope={"target_scope_membership_pass": True},
        base_form_coverage={"coverage_pass": True},
        stage185_provenance_validation={"stage185_provenance_pass": True},
        full_output_validation={"full_output_isolation_pass": True},
        provenance_validation={"execution_provenance_pass": True},
        execution_provenance={"F1_input_sha256": "in", "F1_execution_commit": "c", "F1_output_sha256": "out"},
    )
    assert summary["F1_generated_candidate_pair_ids"] != structural_pair_ids
    assert summary["F1_generated_candidate_count"] == 121


def test_stage185_field_failure_prevents_candidate_acceptance():
    original = row("p", "polarity_flip", "A did not approved B.")
    regenerated = row("p", "polarity_flip", "A did not approve B.")
    canonical = row("p", "none", "A approved B.", "SUPPORT")
    audit = f1.semantic_audit_record(
        original,
        regenerated,
        canonical,
        sidecar_before=stage185_sidecar("p__polarity_flip", "p", before=True),
        sidecar_after=stage185_sidecar("p__polarity_flip", "p", override={"canonical_status": "UNRESOLVED"}),
        inflected_predicate_surface="approved",
        expected_base_predicate="approve",
    )
    assert audit["candidate_accepted"] is False
    assert "STAGE185_TRANSITION_FAILED" in audit["ordered_rejection_codes"]


def test_stage185_positive_case_checks_exact_transition():
    original = row("p", "polarity_flip", "A did not approved B.")
    regenerated = row("p", "polarity_flip", "A did not approve B.")
    canonical = row("p", "none", "A approved B.", "SUPPORT")
    audit = f1.semantic_audit_record(
        original,
        regenerated,
        canonical,
        sidecar_before=stage185_sidecar("p__polarity_flip", "p", before=True),
        sidecar_after=stage185_sidecar("p__polarity_flip", "p"),
        inflected_predicate_surface="approved",
        expected_base_predicate="approve",
    )
    evidence = audit["semantic_validation_evidence"]
    assert evidence["stage185_transition_pass"] is True
    assert evidence["F1_integrity_transition"] == "INELIGIBLE_TO_ELIGIBLE"


def stage185_source_and_sidecar() -> tuple[list[dict], list[dict]]:
    templates = [fact("approved") | {"pair_id": "p"}, fact("released") | {"pair_id": "q"}]
    source = [
        record for record in generator._build_records(templates)
        if record["id"] in {"p__none", "p__polarity_flip", "q__none", "q__polarity_flip"}
    ]
    sidecar = f1.derive_stage185_expected_sidecar(
        source,
        actual_source_dataset_sha256="source-sha",
        actual_source_dataset_path=Path(f1.P3W6F1_AUTHORITATIVE_DATA_PATH),
        actual_integrity_builder_sha256="builder-sha",
        expected_generator_rows=source,
        runtime_authority=stage185_runtime_authority_stub(),
    )
    return source, sidecar


def assert_stage185_provenance_failure(sidecar_mutation):
    source, sidecar = stage185_source_and_sidecar()
    sidecar_mutation(source, sidecar)
    result = f1.validate_stage185_sidecar_provenance(
        source,
        sidecar,
        actual_source_dataset_sha256="source-sha",
        actual_source_dataset_path=Path(f1.P3W6F1_AUTHORITATIVE_DATA_PATH),
        actual_integrity_builder_sha256="builder-sha",
        expected_generator_rows=source,
        runtime_authority=stage185_runtime_authority_stub(),
    )
    assert result["stage185_provenance_status"] == "STAGE185_PROVENANCE_UNRESOLVED"
    assert result["stage185_provenance_pass"] is False


def test_stage185_provenance_positive_case():
    source, sidecar = stage185_source_and_sidecar()
    result = f1.validate_stage185_sidecar_provenance(
        source,
        sidecar,
        actual_source_dataset_sha256="source-sha",
        actual_source_dataset_path=Path(f1.P3W6F1_AUTHORITATIVE_DATA_PATH),
        actual_integrity_builder_sha256="builder-sha",
        expected_generator_rows=source,
        runtime_authority=stage185_runtime_authority_stub(),
    )
    assert result["stage185_provenance_pass"] is True


def test_stage185_provenance_row_count_mismatch_blocks_acceptance():
    assert_stage185_provenance_failure(lambda source, sidecar: sidecar.pop())


def test_stage185_provenance_row_order_mismatch_blocks_acceptance():
    assert_stage185_provenance_failure(lambda source, sidecar: sidecar.reverse())


def test_stage185_provenance_row_id_mismatch_blocks_acceptance():
    assert_stage185_provenance_failure(lambda source, sidecar: sidecar[1].update({"row_id": "wrong"}))


def test_stage185_provenance_pair_id_mismatch_blocks_acceptance():
    assert_stage185_provenance_failure(lambda source, sidecar: sidecar[1].update({"pair_id": "wrong"}))


def test_stage185_provenance_canonical_lineage_mismatch_blocks_acceptance():
    assert_stage185_provenance_failure(lambda source, sidecar: sidecar[1].update({"canonical_row_id": "missing__none"}))


def test_stage185_provenance_source_dataset_sha_mismatch_blocks_acceptance():
    assert_stage185_provenance_failure(lambda source, sidecar: sidecar[1].update({"source_dataset_sha256": "wrong"}))


def test_stage185_provenance_integrity_builder_sha_mismatch_blocks_acceptance():
    assert_stage185_provenance_failure(lambda source, sidecar: sidecar[1].update({"integrity_builder_sha256": "wrong"}))


def test_stage185_provenance_required_schema_missing_blocks_acceptance():
    assert_stage185_provenance_failure(lambda source, sidecar: sidecar[1].pop("schema_status"))


def test_stage185_expected_sidecar_preserves_production_dev_split():
    source = generator.build_controlled_records(300)
    _train_rows, _dev_rows, dev_ids = f1.stage185_builder.split_by_pair(
        [dict(row) for row in source],
        f1.P3W6F1_STAGE185_SPLIT_SEED,
        f1.P3W6F1_STAGE185_DEV_RATIO,
    )
    sidecar = f1.derive_stage185_expected_sidecar(
        source,
        actual_source_dataset_sha256=f1.P3W6F1_AUTHORITATIVE_DATA_SHA256,
        actual_source_dataset_path=Path(f1.P3W6F1_AUTHORITATIVE_DATA_PATH),
        actual_integrity_builder_sha256="builder-sha",
        expected_generator_rows=source,
        runtime_authority=stage185_runtime_authority_stub(),
    )
    dev_row = next(row for row in sidecar if row["pair_id"] in dev_ids)
    assert dev_row["split"] == "dev"


def test_stage185_expected_sidecar_preserves_production_train_split():
    source = generator.build_controlled_records(300)
    _train_rows, _dev_rows, dev_ids = f1.stage185_builder.split_by_pair(
        [dict(row) for row in source],
        f1.P3W6F1_STAGE185_SPLIT_SEED,
        f1.P3W6F1_STAGE185_DEV_RATIO,
    )
    sidecar = f1.derive_stage185_expected_sidecar(
        source,
        actual_source_dataset_sha256=f1.P3W6F1_AUTHORITATIVE_DATA_SHA256,
        actual_source_dataset_path=Path(f1.P3W6F1_AUTHORITATIVE_DATA_PATH),
        actual_integrity_builder_sha256="builder-sha",
        expected_generator_rows=source,
        runtime_authority=stage185_runtime_authority_stub(),
    )
    train_row = next(row for row in sidecar if row["pair_id"] not in dev_ids)
    assert train_row["split"] == "train"


def test_stage185_split_tamper_fails_provenance():
    source, sidecar = stage185_source_and_sidecar()
    sidecar[0] = dict(sidecar[0]) | {"split": "dev" if sidecar[0]["split"] == "train" else "train"}
    result = f1.validate_stage185_sidecar_provenance(
        source,
        sidecar,
        actual_source_dataset_sha256="source-sha",
        actual_source_dataset_path=Path(f1.P3W6F1_AUTHORITATIVE_DATA_PATH),
        actual_integrity_builder_sha256="builder-sha",
        expected_generator_rows=source,
        runtime_authority=stage185_runtime_authority_stub(),
    )
    assert result["stage185_provenance_status"] == "STAGE185_PROVENANCE_UNRESOLVED"
    assert "stage185_semantic_identity" in result["stage185_provenance_failures"]


def test_stage185_contracts_are_loaded_from_authoritative_source(monkeypatch, tmp_path: Path):
    observed: dict = {}
    matrix = tmp_path / "stage184a_family_contract_matrix.csv"
    matrix.write_text("family,intended_changed_axes,intended_preserved_axes,structured_provenance_available,sidecar_implementation_readiness\n", encoding="utf-8")
    monkeypatch.setattr(f1, "require_canonical_repo_file", lambda repo_root, caller_path, canonical_relative_path, **kwargs: matrix)
    def fake_load_contracts(path, families):
        observed["path"] = path
        observed["families"] = set(families)
        return {family: {"changed_axes": [], "preserved_axes": []} for family in families}
    monkeypatch.setattr(f1.stage185_builder, "load_contracts", fake_load_contracts)
    contracts = f1.load_authoritative_stage185_contracts(
        [row("p", "none", "A approved B.", "SUPPORT")],
        repo_root=tmp_path,
        runtime_authority=stage185_runtime_authority_stub(),
    )
    assert observed["path"] == matrix
    assert observed["families"] == {"none"}
    assert set(contracts) == {"none"}


def test_analyzer_local_contract_map_cannot_be_stage185_authority():
    source = Path(f1.__file__).read_text(encoding="utf-8")
    assert "STAGE185_CONTRACT_AXES" not in source
    assert "stage185_contracts_for_rows" not in source


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("grammar_status", "PASS"),
        ("integrity_status", "ELIGIBLE"),
        ("canonical_status", "UNRESOLVED"),
        ("audit_changed_axes", []),
    ],
)
def test_stage185_semantic_tamper_fails_provenance_before_transition(field: str, value):
    source, sidecar = stage185_source_and_sidecar()
    sidecar[1] = dict(sidecar[1]) | {field: value}
    result = f1.validate_stage185_sidecar_provenance(
        source,
        sidecar,
        actual_source_dataset_sha256="source-sha",
        actual_source_dataset_path=Path(f1.P3W6F1_AUTHORITATIVE_DATA_PATH),
        actual_integrity_builder_sha256="builder-sha",
        expected_generator_rows=source,
        runtime_authority=stage185_runtime_authority_stub(),
    )
    assert result["stage185_provenance_status"] == "STAGE185_PROVENANCE_UNRESOLVED"
    assert "stage185_semantic_identity" in result["stage185_provenance_failures"]


def repaired_stage185_replay_case() -> tuple[list[dict], list[dict], set[str], dict]:
    baseline, repaired, authorized = replay_case()
    replay = f1.validate_repaired_output_replay_identity(baseline, repaired, authorized)
    return baseline, repaired, authorized, replay


def test_repaired_stage185_expected_sidecar_uses_authorized_repaired_generator_replay():
    _baseline, repaired, authorized, replay = repaired_stage185_replay_case()
    sidecar = f1.derive_stage185_expected_sidecar(
        repaired,
        actual_source_dataset_sha256="repaired-sha",
        actual_source_dataset_path=Path("reports/p3w6f1_repaired.jsonl"),
        actual_integrity_builder_sha256="builder-sha",
        expected_generator_rows=replay["replayed_records"],
        runtime_authority=stage185_runtime_authority_stub(),
    )
    repaired_member = next(row for row in sidecar if row["row_id"] in authorized)
    assert replay["generator_replay_identity_pass"] is True
    assert repaired_member["grammar_status"] == "PASS"
    assert repaired_member["integrity_status"] == "ELIGIBLE"
    assert repaired_member["dataset_source_status"] == "PASS"


def test_repaired_stage185_expected_sidecar_does_not_use_unrepaired_baseline_expected_rows():
    baseline, repaired, _authorized, replay = repaired_stage185_replay_case()
    observed_sidecar = f1.derive_stage185_expected_sidecar(
        repaired,
        actual_source_dataset_sha256="repaired-sha",
        actual_source_dataset_path=Path("reports/p3w6f1_repaired.jsonl"),
        actual_integrity_builder_sha256="builder-sha",
        expected_generator_rows=replay["replayed_records"],
        runtime_authority=stage185_runtime_authority_stub(),
    )
    result = f1.validate_stage185_sidecar_provenance(
        repaired,
        observed_sidecar,
        actual_source_dataset_sha256="repaired-sha",
        actual_source_dataset_path=Path("reports/p3w6f1_repaired.jsonl"),
        actual_integrity_builder_sha256="builder-sha",
        expected_generator_rows=baseline,
        runtime_authority=stage185_runtime_authority_stub(),
    )
    assert result["stage185_provenance_status"] == "STAGE185_PROVENANCE_UNRESOLVED"
    assert "stage185_semantic_identity" in result["stage185_provenance_failures"]


def test_repaired_stage185_source_dataset_path_is_not_baseline_path():
    _baseline, repaired, _authorized, replay = repaired_stage185_replay_case()
    repaired_path = Path("reports/p3w6f1_repaired.jsonl")
    observed_sidecar = f1.derive_stage185_expected_sidecar(
        repaired,
        actual_source_dataset_sha256="repaired-sha",
        actual_source_dataset_path=repaired_path,
        actual_integrity_builder_sha256="builder-sha",
        expected_generator_rows=replay["replayed_records"],
        runtime_authority=stage185_runtime_authority_stub(),
    )
    result = f1.validate_stage185_sidecar_provenance(
        repaired,
        observed_sidecar,
        actual_source_dataset_sha256="repaired-sha",
        actual_source_dataset_path=Path(f1.P3W6F1_AUTHORITATIVE_DATA_PATH),
        actual_integrity_builder_sha256="builder-sha",
        expected_generator_rows=replay["replayed_records"],
        runtime_authority=stage185_runtime_authority_stub(),
    )
    assert result["stage185_provenance_status"] == "STAGE185_PROVENANCE_UNRESOLVED"
    assert "source_dataset_path" in result["stage185_provenance_failures"]


def test_repaired_stage185_replay_mismatch_blocks_provenance():
    baseline, repaired, authorized, replay = repaired_stage185_replay_case()
    tampered = [dict(row) for row in repaired]
    for row_value in tampered:
        if row_value["id"] in authorized:
            row_value["evidence"] = row_value["evidence"] + " tamper"
            break
    mismatch = f1.validate_repaired_output_replay_identity(baseline, tampered, authorized)
    observed_sidecar = f1.derive_stage185_expected_sidecar(
        tampered,
        actual_source_dataset_sha256="tampered-repaired-sha",
        actual_source_dataset_path=Path("reports/p3w6f1_repaired.jsonl"),
        actual_integrity_builder_sha256="builder-sha",
        expected_generator_rows=replay["replayed_records"],
        runtime_authority=stage185_runtime_authority_stub(),
    )
    result = f1.validate_stage185_sidecar_provenance(
        tampered,
        observed_sidecar,
        actual_source_dataset_sha256="tampered-repaired-sha",
        actual_source_dataset_path=Path("reports/p3w6f1_repaired.jsonl"),
        actual_integrity_builder_sha256="builder-sha",
        expected_generator_rows=mismatch["replayed_records"] if mismatch["generator_replay_identity_pass"] else None,
        runtime_authority=stage185_runtime_authority_stub(),
    )
    assert mismatch["generator_replay_identity_pass"] is False
    assert result["stage185_provenance_status"] == "STAGE185_PROVENANCE_UNRESOLVED"
    assert "stage185_expected_generator_unavailable" in result["stage185_provenance_failures"]


def install_stage185_runtime_authority_tree(monkeypatch, tmp_path: Path, *, drift_path: str | None = None):
    current_bytes = {
        f1.P3W6F1_STAGE185_BUILDER_PATH: b"stage185-builder",
        f1.P3W6F1_STAGE182_ANALYZER_PATH: b"stage182-analyzer",
        f1.P3W6F1_STAGE184_CONTRACT_MATRIX_PATH: b"stage184-contract-matrix",
    }
    for relative_path, payload in current_bytes.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    trusted_bytes = dict(current_bytes)
    if drift_path is not None:
        trusted_bytes[drift_path] = b"trusted-blob-drift"
    monkeypatch.setattr(
        f1.p3w4_authority,
        "load_production_grammar_validator",
        lambda root: {
            "validator_source_path": f1.P3W6F1_STAGE182_ANALYZER_PATH,
            "validator_definition_kind": "stage182_imported",
        },
    )
    return (
        lambda commit, path: trusted_bytes.get(path)
        if commit == f1.P3W6F1_TRUSTED_STAGE185_DEPENDENCY_COMMIT
        else None,
        hashlib.sha256(current_bytes[f1.P3W6F1_STAGE185_BUILDER_PATH]).hexdigest(),
    )


def test_stage185_runtime_authority_rejects_builder_blob_drift(monkeypatch, tmp_path: Path):
    blob_reader, builder_sha = install_stage185_runtime_authority_tree(
        monkeypatch,
        tmp_path,
        drift_path=f1.P3W6F1_STAGE185_BUILDER_PATH,
    )
    with pytest.raises(ValueError, match="STAGE185_PROVENANCE_UNRESOLVED"):
        f1.validate_stage185_runtime_authority_identity(
            repo_root=tmp_path,
            baseline_integrity_builder_sha256=builder_sha,
            git_tracked_checker=lambda relative_path: True,
            git_blob_reader=blob_reader,
        )


def test_stage185_runtime_authority_rejects_stage182_blob_drift(monkeypatch, tmp_path: Path):
    blob_reader, builder_sha = install_stage185_runtime_authority_tree(
        monkeypatch,
        tmp_path,
        drift_path=f1.P3W6F1_STAGE182_ANALYZER_PATH,
    )
    with pytest.raises(ValueError, match="STAGE185_PROVENANCE_UNRESOLVED"):
        f1.validate_stage185_runtime_authority_identity(
            repo_root=tmp_path,
            baseline_integrity_builder_sha256=builder_sha,
            git_tracked_checker=lambda relative_path: True,
            git_blob_reader=blob_reader,
        )


def test_stage185_runtime_authority_rejects_stage184_contract_matrix_blob_drift(monkeypatch, tmp_path: Path):
    blob_reader, builder_sha = install_stage185_runtime_authority_tree(
        monkeypatch,
        tmp_path,
        drift_path=f1.P3W6F1_STAGE184_CONTRACT_MATRIX_PATH,
    )
    with pytest.raises(ValueError, match="STAGE185_PROVENANCE_UNRESOLVED"):
        f1.validate_stage185_runtime_authority_identity(
            repo_root=tmp_path,
            baseline_integrity_builder_sha256=builder_sha,
            git_tracked_checker=lambda relative_path: True,
            git_blob_reader=blob_reader,
        )


def test_resolved_grammar_validator_identity_is_used_in_candidate_audit():
    original = row("p", "polarity_flip", "A did not approved B.")
    regenerated = row("p", "polarity_flip", "A did not approve B.")
    canonical = row("p", "none", "A approved B.", "SUPPORT")
    audit = f1.semantic_audit_record(
        original,
        regenerated,
        canonical,
        sidecar_before=stage185_sidecar("p__polarity_flip", "p", before=True),
        sidecar_after=stage185_sidecar("p__polarity_flip", "p"),
        inflected_predicate_surface="approved",
        expected_base_predicate="approve",
        grammar_validator_source=f1.P3W6F1_STAGE182_ANALYZER_PATH,
        grammar_validator_sha256="g" * 64,
    )
    assert audit["grammar_validator_source"] == f1.P3W6F1_STAGE182_ANALYZER_PATH
    assert audit["grammar_validator_sha256"] == "g" * 64


def test_regenerated_candidate_record_schema_contains_explicit_fields():
    original = row("p", "polarity_flip", "A did not approved B.")
    regenerated = row("p", "polarity_flip", "A did not approve B.")
    canonical = row("p", "none", "A approved B.", "SUPPORT")
    candidate = f1.semantic_audit_record(
        original,
        regenerated,
        canonical,
        sidecar_before=stage185_sidecar("p__polarity_flip", "p", before=True),
        sidecar_after=stage185_sidecar("p__polarity_flip", "p"),
        inflected_predicate_surface="approved",
        expected_base_predicate="approve",
    )
    isolation = f1.full_output_isolation([original], [regenerated], authorized_f1_row_ids={"p__polarity_flip"})
    summary = f1.build_summary(["p"], ["p"], [candidate])
    f1.assert_artifact_schemas(summary, [candidate], [candidate], isolation)


def make_authority_tree(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "baseline": tmp_path / f1.P3W6F1_AUTHORITATIVE_DATA_PATH,
        "sidecar": tmp_path / f1.P3W6F1_BASELINE_SIDECAR_PATH,
        "summary": tmp_path / f1.P3W6F1_P3W4_SUMMARY_PATH,
        "pairs": tmp_path / f1.P3W6F1_P3W4_PAIRS_PATH,
        "manifest": tmp_path / f1.P3W6F1_P3W5_MANIFEST_PATH,
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    paths["manifest"].write_bytes(b"p3w5-manifest")
    return paths


def install_authority_identity_mocks(monkeypatch, paths: dict[str, Path], *, wrong_sha_path: Path | None = None, sidecar_semantic_sha: str | None = None):
    expected_by_path = {
        paths["baseline"].resolve(): f1.P3W6F1_AUTHORITATIVE_DATA_SHA256,
        paths["summary"].resolve(): f1.P3W6F1_P3W4_SUMMARY_SHA256,
        paths["pairs"].resolve(): f1.P3W6F1_P3W4_PAIRS_SHA256,
    }
    def fake_file_sha256(path: Path) -> str:
        if wrong_sha_path is not None and path.resolve() == wrong_sha_path.resolve():
            return "0" * 64
        return expected_by_path.get(path.resolve(), "irrelevant")
    monkeypatch.setattr(f1, "file_sha256", fake_file_sha256)
    monkeypatch.setattr(
        f1,
        "load_jsonl",
        lambda path: [{
            "row_id": "p__none",
            "created_at": "ignored",
            "generator_source_path": "/kaggle/working/ContraMamba/scripts/build_controlled_v5.py",
            "generator_source_sha256": "baseline-generator-sha",
            "integrity_builder_sha256": "baseline-builder-sha",
        }],
    )
    monkeypatch.setattr(
        f1,
        "stage185_semantic_sidecar_sha256",
        lambda rows: sidecar_semantic_sha or f1.P3W6F1_BASELINE_SIDECAR_SEMANTIC_SHA256,
    )


def validate_authority_tree(tmp_path: Path, paths: dict[str, Path], **kwargs):
    return f1.validate_p3w6f1_input_authority_identity(
        repo_root=tmp_path,
        baseline_jsonl_path=kwargs.get("baseline", paths["baseline"]),
        baseline_sidecar_jsonl_path=kwargs.get("sidecar", paths["sidecar"]),
        p3w4_summary_json_path=kwargs.get("summary", paths["summary"]),
        p3w4_pairs_jsonl_path=kwargs.get("pairs", paths["pairs"]),
        p3w5_manifest_json_path=kwargs.get("manifest", paths["manifest"]),
        f1_input_sha256=kwargs.get("f1_input_sha256", f1.P3W6F1_AUTHORITATIVE_DATA_SHA256),
        git_tracked_checker=kwargs.get("git_tracked_checker", lambda relative_path: True),
        git_blob_reader=kwargs.get(
            "git_blob_reader",
            lambda commit, path: b"p3w5-manifest"
            if commit == f1.P3W6F1_P3W5_AUTHORITY_COMMIT and path == f1.P3W6F1_P3W5_MANIFEST_PATH
            else None,
        ),
    )


def test_wrong_baseline_path_fails_input_authority(monkeypatch, tmp_path: Path):
    paths = make_authority_tree(tmp_path)
    install_authority_identity_mocks(monkeypatch, paths)
    alternate = tmp_path / "data/alternate.jsonl"
    alternate.parent.mkdir(parents=True, exist_ok=True)
    alternate.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="PROVENANCE_IDENTITY_MISMATCH"):
        validate_authority_tree(tmp_path, paths, baseline=alternate)


def test_self_hashed_alternate_baseline_cannot_become_authority(monkeypatch, tmp_path: Path):
    paths = make_authority_tree(tmp_path)
    install_authority_identity_mocks(monkeypatch, paths)
    alternate = tmp_path / "data/alternate.jsonl"
    alternate.parent.mkdir(parents=True, exist_ok=True)
    alternate.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(f1, "file_sha256", lambda path: "a" * 64 if path.resolve() == alternate.resolve() else f1.P3W6F1_AUTHORITATIVE_DATA_SHA256)
    with pytest.raises(ValueError, match="PROVENANCE_IDENTITY_MISMATCH"):
        validate_authority_tree(tmp_path, paths, baseline=alternate, f1_input_sha256="a" * 64)


def test_wrong_baseline_sha_fails_input_authority(monkeypatch, tmp_path: Path):
    paths = make_authority_tree(tmp_path)
    install_authority_identity_mocks(monkeypatch, paths, wrong_sha_path=paths["baseline"])
    with pytest.raises(ValueError, match="PROVENANCE_IDENTITY_MISMATCH"):
        validate_authority_tree(tmp_path, paths)


def test_wrong_baseline_sidecar_path_fails_input_authority(monkeypatch, tmp_path: Path):
    paths = make_authority_tree(tmp_path)
    install_authority_identity_mocks(monkeypatch, paths)
    with pytest.raises(ValueError, match="PROVENANCE_IDENTITY_MISMATCH"):
        validate_authority_tree(tmp_path, paths, sidecar=tmp_path / "reports/other_sidecar.jsonl")


def test_wrong_baseline_sidecar_semantic_sha_fails_input_authority(monkeypatch, tmp_path: Path):
    paths = make_authority_tree(tmp_path)
    install_authority_identity_mocks(monkeypatch, paths, sidecar_semantic_sha="0" * 64)
    with pytest.raises(ValueError, match="PROVENANCE_IDENTITY_MISMATCH"):
        validate_authority_tree(tmp_path, paths)


def test_wrong_p3w4_summary_path_fails_input_authority(monkeypatch, tmp_path: Path):
    paths = make_authority_tree(tmp_path)
    install_authority_identity_mocks(monkeypatch, paths)
    with pytest.raises(ValueError, match="PROVENANCE_IDENTITY_MISMATCH"):
        validate_authority_tree(tmp_path, paths, summary=tmp_path / "reports/synthetic_summary.json")


def test_wrong_p3w4_summary_sha_fails_input_authority(monkeypatch, tmp_path: Path):
    paths = make_authority_tree(tmp_path)
    install_authority_identity_mocks(monkeypatch, paths, wrong_sha_path=paths["summary"])
    with pytest.raises(ValueError, match="PROVENANCE_IDENTITY_MISMATCH"):
        validate_authority_tree(tmp_path, paths)


def test_wrong_p3w4_pairs_path_fails_input_authority(monkeypatch, tmp_path: Path):
    paths = make_authority_tree(tmp_path)
    install_authority_identity_mocks(monkeypatch, paths)
    with pytest.raises(ValueError, match="PROVENANCE_IDENTITY_MISMATCH"):
        validate_authority_tree(tmp_path, paths, pairs=tmp_path / "reports/synthetic_121_pairs.jsonl")


def test_wrong_p3w4_pairs_sha_fails_input_authority(monkeypatch, tmp_path: Path):
    paths = make_authority_tree(tmp_path)
    install_authority_identity_mocks(monkeypatch, paths, wrong_sha_path=paths["pairs"])
    with pytest.raises(ValueError, match="PROVENANCE_IDENTITY_MISMATCH"):
        validate_authority_tree(tmp_path, paths)


def test_synthetic_structurally_valid_121_target_file_cannot_redefine_authority(monkeypatch, tmp_path: Path):
    paths = make_authority_tree(tmp_path)
    install_authority_identity_mocks(monkeypatch, paths)
    synthetic = tmp_path / "reports/reason_router_p2_p3w4_canonical_grammar_authority_execution_ca99038d/synthetic_121_pairs.jsonl"
    synthetic.parent.mkdir(parents=True, exist_ok=True)
    synthetic.write_text("\n".join("{}" for _ in range(121)), encoding="utf-8")
    with pytest.raises(ValueError, match="PROVENANCE_IDENTITY_MISMATCH"):
        validate_authority_tree(tmp_path, paths, pairs=synthetic)


def test_wrong_p3w5_manifest_path_fails_input_authority(monkeypatch, tmp_path: Path):
    paths = make_authority_tree(tmp_path)
    install_authority_identity_mocks(monkeypatch, paths)
    with pytest.raises(ValueError, match="PROVENANCE_IDENTITY_MISMATCH"):
        validate_authority_tree(tmp_path, paths, manifest=tmp_path / "reports/p3w5_synthetic_manifest.json")


def test_wrong_p3w5_manifest_blob_fails_input_authority(monkeypatch, tmp_path: Path):
    paths = make_authority_tree(tmp_path)
    install_authority_identity_mocks(monkeypatch, paths)
    with pytest.raises(ValueError, match="PROVENANCE_IDENTITY_MISMATCH"):
        validate_authority_tree(tmp_path, paths, git_blob_reader=lambda commit, path: b"different")


def test_baseline_generator_sha_is_derived_from_frozen_baseline_sidecar(monkeypatch, tmp_path: Path):
    paths = make_authority_tree(tmp_path)
    install_authority_identity_mocks(monkeypatch, paths)
    result = validate_authority_tree(tmp_path, paths)
    assert result["baseline_generator_source_sha256"] == "baseline-generator-sha"
    assert result["baseline_integrity_builder_sha256"] == "baseline-builder-sha"


def test_input_authority_validation_precedes_target_extraction(monkeypatch):
    calls: list[str] = []
    def fail_authority(**kwargs):
        calls.append("authority")
        raise ValueError("PROVENANCE_IDENTITY_MISMATCH")
    def target_extraction(*args, **kwargs):
        calls.append("target")
        return set()
    monkeypatch.setattr(f1, "validate_p3w6f1_input_authority_identity", fail_authority)
    monkeypatch.setattr(f1, "extract_decision_supporting_pair_ids", target_extraction)
    args = type("Args", (), {
        "p3w4_summary_json": "summary",
        "p3w4_pairs_jsonl": "pairs",
        "p3w5_manifest_json": "manifest",
        "baseline_jsonl": "baseline",
        "repaired_jsonl": "repaired",
        "baseline_sidecar_jsonl": "sidecar",
        "repaired_sidecar_jsonl": "repaired_sidecar",
        "baseline_generator_commit": "b",
        "baseline_generator_source_path": f1.GENERATOR_SOURCE_PATH,
        "baseline_generator_source_sha256": "s",
        "repaired_generator_commit": "c",
        "repaired_generator_source_path": f1.GENERATOR_SOURCE_PATH,
        "repaired_generator_source_sha256": "s",
        "deterministic_generator_invocation_json": "{}",
        "generator_configuration_identity_json": "{}",
        "f1_input_sha256": f1.P3W6F1_AUTHORITATIVE_DATA_SHA256,
        "f1_execution_commit": "c",
        "f1_output_sha256": "o",
        "output_dir": "out",
    })()
    with pytest.raises(ValueError, match="PROVENANCE_IDENTITY_MISMATCH"):
        f1.run(args)
    assert calls == ["authority"]


def run_order_args(tmp_path: Path):
    return type("Args", (), {
        "p3w4_summary_json": str(tmp_path / "summary.json"),
        "p3w4_pairs_jsonl": str(tmp_path / "pairs.jsonl"),
        "p3w5_manifest_json": str(tmp_path / "manifest.json"),
        "baseline_jsonl": str(tmp_path / "baseline.jsonl"),
        "repaired_jsonl": str(tmp_path / "repaired.jsonl"),
        "baseline_sidecar_jsonl": str(tmp_path / "baseline_sidecar.jsonl"),
        "repaired_sidecar_jsonl": str(tmp_path / "repaired_sidecar.jsonl"),
        "baseline_generator_commit": "base",
        "baseline_generator_source_path": f1.GENERATOR_SOURCE_PATH,
        "baseline_generator_source_sha256": "base-sha",
        "repaired_generator_commit": "exec",
        "repaired_generator_source_path": f1.GENERATOR_SOURCE_PATH,
        "repaired_generator_source_sha256": "repair-sha",
        "deterministic_generator_invocation_json": "{\"actual\": true}",
        "generator_configuration_identity_json": "{\"actual\": true}",
        "f1_input_sha256": f1.P3W6F1_AUTHORITATIVE_DATA_SHA256,
        "f1_execution_commit": "exec",
        "f1_output_sha256": "out",
        "output_dir": str(tmp_path / "out"),
    })()


def install_run_trace_mocks(
    monkeypatch,
    tmp_path: Path,
    *,
    isolation_pass: bool = True,
    stage185_pass: bool = True,
    execution_pass: bool = True,
) -> tuple[list[str], dict[str, list[list[dict]]]]:
    calls: list[str] = []
    persisted: dict[str, list[list[dict]]] = {}
    baseline_rows = [row("p", "polarity_flip", "A did not approved B.")]
    repaired_rows = [row("p", "polarity_flip", "A did not approve B.")]
    sidecar_rows = [{"row_id": "p__polarity_flip", "pair_id": "p"}]

    monkeypatch.setattr(f1, "validate_p3w6f1_input_authority_identity", lambda **kwargs: {
        "baseline_integrity_builder_sha256": "builder-sha",
        "baseline_generator_source_sha256": "base-sha",
    })
    monkeypatch.setattr(f1, "load_json", lambda path: {"decision_supporting_pair_ids": ["p"]})
    def fake_load_jsonl(path: Path):
        name = Path(path).name
        if name == "pairs.jsonl":
            return [authority_record("p")]
        if name == "baseline.jsonl":
            return baseline_rows
        if name == "repaired.jsonl":
            return repaired_rows
        return sidecar_rows
    monkeypatch.setattr(f1, "load_jsonl", fake_load_jsonl)
    monkeypatch.setattr(f1, "extract_decision_supporting_pair_ids", lambda summary, manifest: {"p"})
    monkeypatch.setattr(f1, "extract_authorized_f1_targets", lambda records, supporting: {
        "F1_target_pair_count": 121,
        "F1_target_pair_ids": ["p"],
        "F1_target_row_count": 121,
        "authorized_F1_row_ids": ["p__polarity_flip"],
    })
    monkeypatch.setattr(f1, "validate_authority_cardinality", lambda targets: {"authority_cardinality_pass": True})
    monkeypatch.setattr(f1, "required_f1_inflected_predicate_surfaces", lambda records, pairs: {"approved"})
    monkeypatch.setattr(f1, "validate_base_form_coverage", lambda surfaces, mapping: {"coverage_pass": True})
    monkeypatch.setattr(f1, "structural_negative_polarity_flip_row_ids_for_pair_count", lambda pair_count: {"p__polarity_flip"})
    monkeypatch.setattr(f1, "validate_target_scope_membership", lambda structural, authorized: {
        "target_scope_membership_pass": True,
        "structural_negative_polarity_flip_row_ids": ["p__polarity_flip"],
    })
    def fake_replay(*args, **kwargs):
        calls.append("validate_repaired_output_replay_identity")
        return {
            "generator_replay_identity_pass": True,
            "actual_generator_repair_consumed_row_ids": ["p__polarity_flip"],
            "replayed_records": repaired_rows,
            "deterministic_generator_invocation": {"actual": True},
            "generator_configuration_identity": {"actual": True},
        }
    monkeypatch.setattr(f1, "validate_repaired_output_replay_identity", fake_replay)
    def fake_isolation(*args, **kwargs):
        calls.append("full_output_isolation")
        return {}
    monkeypatch.setattr(f1, "full_output_isolation", fake_isolation)
    def fake_validate_isolation(isolation):
        calls.append("validate_full_output_isolation")
        return {"full_output_isolation_pass": isolation_pass}
    monkeypatch.setattr(f1, "validate_full_output_isolation", fake_validate_isolation)
    def fake_runtime(**kwargs):
        calls.append("validate_stage185_runtime_authority_identity")
        return {
            "stage185_runtime_authority_pass": True,
            "stage185_integrity_builder_source_sha256": "builder-sha",
            "resolved_grammar_validator_source_path": f1.P3W6F1_STAGE182_ANALYZER_PATH,
            "resolved_grammar_validator_source_sha256": "grammar-sha",
        }
    monkeypatch.setattr(f1, "validate_stage185_runtime_authority_identity", fake_runtime)
    monkeypatch.setattr(f1, "generator_expected_rows_and_facts_for_source", lambda rows: ({row_value["id"]: row_value for row_value in rows}, {}))
    monkeypatch.setattr(f1, "file_sha256", lambda path: "sha")
    def fake_stage185(*args, **kwargs):
        calls.append("validate_stage185_sidecar_provenance")
        return {"stage185_provenance_pass": stage185_pass}
    monkeypatch.setattr(f1, "validate_stage185_sidecar_provenance", fake_stage185)
    def fake_audit(*args, **kwargs):
        calls.append("audit_authorized_candidates")
        return [pass_audit("p")]
    monkeypatch.setattr(f1, "audit_authorized_candidates", fake_audit)
    def fake_execution(*args, **kwargs):
        calls.append("validate_execution_provenance_identity")
        return {
            "execution_provenance_pass": execution_pass,
            "execution_provenance_status": "PASS" if execution_pass else "PROVENANCE_IDENTITY_MISMATCH",
        }
    monkeypatch.setattr(f1, "validate_execution_provenance_identity", fake_execution)
    original_finalize = f1.finalize_candidate_acceptance
    def tracing_finalize(*args, **kwargs):
        calls.append("finalize_candidate_acceptance")
        return original_finalize(*args, **kwargs)
    monkeypatch.setattr(f1, "finalize_candidate_acceptance", tracing_finalize)
    def fake_build_summary(*args, **kwargs):
        calls.append("build_summary")
        return {"schema_version": f1.SCHEMA_VERSION}
    monkeypatch.setattr(f1, "build_summary", fake_build_summary)
    monkeypatch.setattr(f1, "assert_artifact_schemas", lambda *args, **kwargs: None)
    monkeypatch.setattr(f1, "write_json", lambda path, value: None)
    monkeypatch.setattr(f1, "write_jsonl", lambda path, rows: persisted.setdefault(Path(path).name, []).append(list(rows)))
    return calls, persisted


def test_run_orders_full_output_isolation_before_stage185_provenance(monkeypatch, tmp_path: Path):
    calls, _persisted = install_run_trace_mocks(monkeypatch, tmp_path)
    f1.run(run_order_args(tmp_path))
    expected = [
        "validate_repaired_output_replay_identity",
        "full_output_isolation",
        "validate_full_output_isolation",
        "validate_stage185_runtime_authority_identity",
        "validate_stage185_sidecar_provenance",
        "validate_stage185_sidecar_provenance",
        "audit_authorized_candidates",
        "validate_execution_provenance_identity",
        "finalize_candidate_acceptance",
        "build_summary",
    ]
    assert calls == expected


def test_persisted_audit_rows_are_finalized_not_local_only(monkeypatch, tmp_path: Path):
    _calls, persisted = install_run_trace_mocks(monkeypatch, tmp_path, isolation_pass=False)
    f1.run(run_order_args(tmp_path))
    audit_rows = persisted["p3w6f1_regeneration_audit.jsonl"][0]
    regenerated_rows = persisted["p3w6f1_regenerated_rows.jsonl"][0]
    assert audit_rows[0]["candidate_accepted"] is False
    assert audit_rows[0]["semantic_validation_status"] == "MANUAL_REVIEW_REQUIRED"
    assert "FULL_OUTPUT_ISOLATION_FAILED" in audit_rows[0]["ordered_rejection_codes"]
    assert regenerated_rows == audit_rows


def execution_provenance(tmp_path: Path) -> tuple[dict, Path, Path, bytes, bytes]:
    baseline_path = tmp_path / "baseline.jsonl"
    repaired_path = tmp_path / "repaired.jsonl"
    baseline_path.write_text("{\"id\":\"a\"}\n", encoding="utf-8")
    repaired_path.write_text("{\"id\":\"b\"}\n", encoding="utf-8")
    baseline_source = b"baseline generator"
    repaired_source = b"repaired generator"
    provenance = {
        "baseline_generator_commit": "base-commit",
        "baseline_generator_source_path": "scripts/build_controlled_v5.py",
        "baseline_generator_source_sha256": hashlib.sha256(baseline_source).hexdigest(),
        "repaired_generator_commit": "exec-commit",
        "repaired_generator_source_path": "scripts/build_controlled_v5.py",
        "repaired_generator_source_sha256": hashlib.sha256(repaired_source).hexdigest(),
        "deterministic_generator_invocation": {"num_pairs": 300, "repair_consumed_row_ids": []},
        "generator_configuration_identity": {"num_pairs": 300},
        "F1_input_sha256": f1.file_sha256(baseline_path),
        "F1_execution_commit": "exec-commit",
        "F1_output_sha256": f1.file_sha256(repaired_path),
    }
    return provenance, baseline_path, repaired_path, baseline_source, repaired_source


def validate_fake_execution_provenance(
    provenance: dict,
    baseline_path: Path,
    repaired_path: Path,
    baseline_source: bytes,
    repaired_source: bytes,
    *,
    current_commit: str = "exec-commit",
    current_source: bytes | None = None,
    extra_blobs: dict[tuple[str, str], bytes] | None = None,
    input_authority: dict | None = None,
) -> dict:
    blobs = {
        ("base-commit", "scripts/build_controlled_v5.py"): baseline_source,
        ("exec-commit", "scripts/build_controlled_v5.py"): repaired_source,
    }
    if extra_blobs:
        blobs.update(extra_blobs)
    return f1.validate_execution_provenance_identity(
        provenance,
        baseline_jsonl_path=baseline_path,
        repaired_jsonl_path=repaired_path,
        git_blob_reader=lambda commit, path: blobs.get((commit, path)),
        current_source_reader=lambda path: repaired_source if current_source is None else current_source,
        current_commit_resolver=lambda: current_commit,
        input_authority=input_authority,
    )


def test_empty_execution_provenance_is_not_pass(tmp_path: Path):
    provenance, baseline_path, repaired_path, baseline_source, repaired_source = execution_provenance(tmp_path)
    provenance["baseline_generator_commit"] = ""
    result = validate_fake_execution_provenance(provenance, baseline_path, repaired_path, baseline_source, repaired_source)
    assert result["execution_provenance_status"] == "PROVENANCE_UNRESOLVED"


def test_execution_provenance_positive_identity_case(tmp_path: Path):
    provenance, baseline_path, repaired_path, baseline_source, repaired_source = execution_provenance(tmp_path)
    result = validate_fake_execution_provenance(provenance, baseline_path, repaired_path, baseline_source, repaired_source)
    assert result["execution_provenance_pass"] is True


def test_arbitrary_historical_baseline_generator_blob_cannot_redefine_authority(tmp_path: Path):
    provenance, baseline_path, repaired_path, baseline_source, repaired_source = execution_provenance(tmp_path)
    result = validate_fake_execution_provenance(
        provenance,
        baseline_path,
        repaired_path,
        baseline_source,
        repaired_source,
        input_authority={"baseline_generator_source_sha256": "authoritative-frozen-sidecar-sha"},
    )
    assert result["execution_provenance_status"] == "PROVENANCE_IDENTITY_MISMATCH"
    assert "baseline_generator_source_sha256" in result["provenance_identity_mismatches"]


def test_baseline_generator_matching_authoritative_source_blob_passes(tmp_path: Path):
    provenance, baseline_path, repaired_path, baseline_source, repaired_source = execution_provenance(tmp_path)
    result = validate_fake_execution_provenance(
        provenance,
        baseline_path,
        repaired_path,
        baseline_source,
        repaired_source,
        input_authority={"baseline_generator_source_sha256": hashlib.sha256(baseline_source).hexdigest()},
    )
    assert result["execution_provenance_pass"] is True


@pytest.mark.parametrize(
    "field",
    [
        "F1_input_sha256",
        "F1_output_sha256",
        "baseline_generator_source_sha256",
        "repaired_generator_source_sha256",
        "F1_execution_commit",
    ],
)
def test_execution_provenance_identity_mismatches_block_all_accepted(tmp_path: Path, field: str):
    provenance, baseline_path, repaired_path, baseline_source, repaired_source = execution_provenance(tmp_path)
    provenance[field] = "wrong"
    result = validate_fake_execution_provenance(provenance, baseline_path, repaired_path, baseline_source, repaired_source)
    assert result["execution_provenance_status"] == "PROVENANCE_IDENTITY_MISMATCH"


def test_wrong_baseline_generator_source_path_is_mismatch(tmp_path: Path):
    provenance, baseline_path, repaired_path, baseline_source, repaired_source = execution_provenance(tmp_path)
    provenance["baseline_generator_source_path"] = "scripts/other.py"
    result = validate_fake_execution_provenance(
        provenance,
        baseline_path,
        repaired_path,
        baseline_source,
        repaired_source,
        extra_blobs={("base-commit", "scripts/other.py"): baseline_source},
    )
    assert result["execution_provenance_status"] == "PROVENANCE_IDENTITY_MISMATCH"


def test_wrong_repaired_generator_source_path_is_mismatch(tmp_path: Path):
    provenance, baseline_path, repaired_path, baseline_source, repaired_source = execution_provenance(tmp_path)
    provenance["repaired_generator_source_path"] = "scripts/other.py"
    result = validate_fake_execution_provenance(
        provenance,
        baseline_path,
        repaired_path,
        baseline_source,
        repaired_source,
        extra_blobs={("exec-commit", "scripts/other.py"): repaired_source},
    )
    assert result["execution_provenance_status"] == "PROVENANCE_IDENTITY_MISMATCH"


def test_repaired_generator_commit_must_equal_execution_commit(tmp_path: Path):
    provenance, baseline_path, repaired_path, baseline_source, repaired_source = execution_provenance(tmp_path)
    provenance["repaired_generator_commit"] = "other-repair-commit"
    result = validate_fake_execution_provenance(
        provenance,
        baseline_path,
        repaired_path,
        baseline_source,
        repaired_source,
        extra_blobs={("other-repair-commit", "scripts/build_controlled_v5.py"): repaired_source},
    )
    assert result["execution_provenance_status"] == "PROVENANCE_IDENTITY_MISMATCH"


def test_execution_commit_must_equal_current_head(tmp_path: Path):
    provenance, baseline_path, repaired_path, baseline_source, repaired_source = execution_provenance(tmp_path)
    result = validate_fake_execution_provenance(
        provenance,
        baseline_path,
        repaired_path,
        baseline_source,
        repaired_source,
        current_commit="different-head",
    )
    assert result["execution_provenance_status"] == "PROVENANCE_IDENTITY_MISMATCH"


def test_current_generator_bytes_must_equal_execution_commit_blob(tmp_path: Path):
    provenance, baseline_path, repaired_path, baseline_source, repaired_source = execution_provenance(tmp_path)
    result = validate_fake_execution_provenance(
        provenance,
        baseline_path,
        repaired_path,
        baseline_source,
        repaired_source,
        current_source=b"different current generator",
    )
    assert result["execution_provenance_status"] == "PROVENANCE_IDENTITY_MISMATCH"


def test_invocation_and_config_spoof_do_not_pass(tmp_path: Path):
    provenance, baseline_path, repaired_path, baseline_source, repaired_source = execution_provenance(tmp_path)
    result = f1.validate_execution_provenance_identity(
        provenance,
        baseline_jsonl_path=baseline_path,
        repaired_jsonl_path=repaired_path,
        actual_deterministic_generator_invocation={"pair_count": 300, "repair_api": "actual"},
        actual_generator_configuration_identity={"pair_count": 300, "authorized_F1_row_count": 121},
        git_blob_reader=lambda commit, path: {
            ("base-commit", "scripts/build_controlled_v5.py"): baseline_source,
            ("exec-commit", "scripts/build_controlled_v5.py"): repaired_source,
        }.get((commit, path)),
        current_source_reader=lambda path: repaired_source,
        current_commit_resolver=lambda: "exec-commit",
    )
    assert result["execution_provenance_status"] == "PROVENANCE_IDENTITY_MISMATCH"


def test_missing_execution_provenance_shape_is_not_pass():
    result = f1.validate_execution_provenance_identity({
        "baseline_generator_commit": "",
        "baseline_generator_source_path": "scripts/build_controlled_v5.py",
        "baseline_generator_source_sha256": "sha",
        "repaired_generator_commit": "commit",
        "repaired_generator_source_path": "scripts/build_controlled_v5.py",
        "repaired_generator_source_sha256": "sha",
        "deterministic_generator_invocation": {},
        "generator_configuration_identity": {"num_pairs": 300},
        "F1_input_sha256": "in",
        "F1_execution_commit": "commit",
        "F1_output_sha256": "out",
    })
    assert result["execution_provenance_status"] == "PROVENANCE_UNRESOLVED"


def summary_with_base_form_coverage(base_form_coverage: dict) -> dict:
    pair_ids = [f"p{i:03d}" for i in range(121)]
    return f1.build_summary(
        pair_ids,
        pair_ids,
        [pass_audit(pair_id) for pair_id in pair_ids],
        authority_cardinality={"authority_cardinality_pass": True},
        target_scope={"target_scope_membership_pass": True},
        base_form_coverage=base_form_coverage,
        stage185_provenance_validation={"stage185_provenance_pass": True},
        full_output_validation={"full_output_isolation_pass": True},
        provenance_validation={"execution_provenance_pass": True},
        execution_provenance={"F1_input_sha256": "in", "F1_execution_commit": "c", "F1_output_sha256": "out"},
    )


def test_missing_base_form_coverage_blocks_execution_decision():
    coverage = f1.validate_base_form_coverage(
        {"approved", "missing_required_surface"},
        generator._BASE_PREDICATE_BY_INFLECTED,
    )
    summary = summary_with_base_form_coverage(coverage)
    assert coverage["missing_base_form_surfaces"] != []
    assert coverage["coverage_pass"] is False
    assert "BASE_FORM_COVERAGE_UNRESOLVED" in summary["F1_execution_blockers"]
    assert summary["F1_execution_decision"] == "P3W5_F1_REGENERATION_COMPLETE_WITH_BLOCKERS_PENDING_RESULT_REVIEW"


def test_complete_base_form_coverage_does_not_create_blocker():
    coverage = f1.validate_base_form_coverage(
        {"approved", "released"},
        generator._BASE_PREDICATE_BY_INFLECTED,
    )
    summary = f1.build_summary(
        ["p000"],
        ["p000"],
        [pass_audit("p000")],
        authority_cardinality={"authority_cardinality_pass": False},
        target_scope={"target_scope_membership_pass": True},
        base_form_coverage=coverage,
        stage185_provenance_validation={"stage185_provenance_pass": True},
        full_output_validation={"full_output_isolation_pass": False},
        provenance_validation={"execution_provenance_pass": True},
        execution_provenance={"F1_input_sha256": "in", "F1_execution_commit": "c", "F1_output_sha256": "out"},
    )
    assert coverage["missing_base_form_surfaces"] == []
    assert coverage["ambiguous_base_form_surfaces"] == []
    assert coverage["coverage_pass"] is True
    assert "BASE_FORM_COVERAGE_UNRESOLVED" not in summary["F1_execution_blockers"]


def test_ambiguous_base_form_coverage_blocks_execution_decision():
    coverage = f1.validate_base_form_coverage(
        {"approved"},
        [("approved", "approve"), ("approved", "approv")],
    )
    summary = summary_with_base_form_coverage(coverage)
    assert coverage["ambiguous_base_form_surfaces"] != []
    assert coverage["coverage_pass"] is False
    assert "BASE_FORM_COVERAGE_UNRESOLVED" in summary["F1_execution_blockers"]
    assert summary["F1_execution_decision"] == "P3W5_F1_REGENERATION_COMPLETE_WITH_BLOCKERS_PENDING_RESULT_REVIEW"
