from __future__ import annotations

import ast
import copy
import hashlib
from functools import lru_cache
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
        "integrity_status": "INELIGIBLE",
        "canonical_status": "PASS",
        "dataset_source_status": "PASS",
        "schema_status": "PASS",
        "intervention_contract_status": "PASS" if before else "FAIL",
        "polarity_contamination_status": "PASS",
        "time_swap_status": "PASS",
        "audit_expected_axes": ["polarity"],
        "audit_changed_axes": ["polarity"] if before else ["polarity", "predicate"],
        "audit_pair_failure_scope": "none",
        "eligible_for_positive_margin": False,
        "reason_codes": ["DID_NOT_INFLECTED_PREDICATE", "GRAMMAR_TEMPLATE_FAIL"] if before else ["INTERVENTION_CONTRACT_FAIL"],
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
        "semantic_validation_evidence": {
            "stage185_transition_pass": True,
            "compatibility_record": {
                "row_id": f"{pair_id}__polarity_flip",
                "compatibility_status": "PASS",
                "effective_F1_repair_integrity_status": "COMPATIBILITY_ELIGIBLE",
            },
        },
    }



def compatibility_base_identity(status: str = "PASS") -> dict:
    return {
        "base_form_source_identity_status": status,
        "base_form_source_path": f1.GENERATOR_SOURCE_PATH,
        "base_form_source_sha256": "a" * 64 if status == "PASS" else None,
        "base_form_source_symbol": f1.BASE_FORM_SYMBOL,
    }


def resolved_compatibility_prerequisites() -> dict:
    return {
        "generator_replay_identity_status": "PASS",
        "repair_consumption_status": "PASS",
        "full_output_isolation_status": "PASS",
        "stage185_v1_runtime_authority_status": "PASS",
        "baseline_stage185_v1_provenance_status": "PASS",
        "repaired_stage185_v1_provenance_status": "PASS",
    }


def compatibility_record(audit: dict) -> dict:
    return audit["semantic_validation_evidence"]["compatibility_record"]
def stage185_runtime_authority_stub() -> dict:
    return {
        "stage185_runtime_authority_pass": True,
        "stage184_contract_matrix_path": f1.P3W6F1_STAGE184_CONTRACT_MATRIX_PATH,
        "stage185_integrity_builder_source_sha256": "builder-sha",
        "resolved_grammar_validator_source_path": f1.P3W6F1_STAGE182_ANALYZER_PATH,
        "resolved_grammar_validator_source_sha256": "grammar-sha",
    }


def authoritative_clean_baseline_rows() -> list[dict]:
    root = Path(f1.__file__).resolve().parents[1]
    return f1.load_jsonl(root / f1.P3W6F1_AUTHORITATIVE_DATA_PATH)


def authoritative_clean_baseline_expected_generator_rows() -> list[dict]:
    baseline = authoritative_clean_baseline_rows()
    pair_count = f1.baseline_pair_count(baseline)
    raw_generator_rows = generator.build_controlled_records(pair_count)
    return f1.project_replay_to_baseline_topology(raw_generator_rows, baseline)


def actual_authorized_f1_row_ids() -> set[str]:
    root = Path(f1.__file__).resolve().parents[1]
    summary = f1.load_json(root / f1.P3W6F1_P3W4_SUMMARY_PATH)
    pairs = f1.load_jsonl(root / f1.P3W6F1_P3W4_PAIRS_PATH)
    manifest = f1.load_json(root / f1.P3W6F1_P3W5_MANIFEST_PATH)
    supporting = f1.extract_decision_supporting_pair_ids(summary, manifest)
    targets = f1.extract_authorized_f1_targets(pairs, supporting)
    assert targets["F1_target_row_count"] == 121
    return set(targets["authorized_F1_row_ids"])


@lru_cache(maxsize=1)
def _cached_stage185_baseline_case() -> tuple[list[dict], list[dict], list[dict]]:
    source = authoritative_clean_baseline_rows()
    expected_generator_rows = authoritative_clean_baseline_expected_generator_rows()
    source_families = {row["intervention_type"] for row in source}
    expected_families = {row["intervention_type"] for row in expected_generator_rows}
    assert len(source) == 3600
    assert len(expected_generator_rows) == 3600
    assert source_families == expected_families
    assert "time_swap" not in source_families
    assert {"none", "paraphrase", "polarity_flip"}.issubset(source_families)
    sidecar = f1.derive_stage185_expected_sidecar(
        source,
        actual_source_dataset_sha256=f1.P3W6F1_AUTHORITATIVE_DATA_SHA256,
        actual_source_dataset_path=Path(f1.P3W6F1_AUTHORITATIVE_DATA_PATH),
        actual_integrity_builder_sha256="builder-sha",
        expected_generator_rows=expected_generator_rows,
        runtime_authority=stage185_runtime_authority_stub(),
    )
    return source, expected_generator_rows, sidecar


def stage185_source_and_sidecar() -> tuple[list[dict], list[dict]]:
    source, _expected_generator_rows, sidecar = _cached_stage185_baseline_case()
    return copy.deepcopy(source), copy.deepcopy(sidecar)


def stage185_source_expected_and_sidecar() -> tuple[list[dict], list[dict], list[dict]]:
    source, expected_generator_rows, sidecar = _cached_stage185_baseline_case()
    return copy.deepcopy(source), copy.deepcopy(expected_generator_rows), copy.deepcopy(sidecar)


def authorized_defective_f1_sidecar_index(sidecar: list[dict]) -> int:
    authorized = actual_authorized_f1_row_ids()
    matches = [
        index
        for index, row_value in enumerate(sidecar)
        if row_value["row_id"] in authorized
    ]
    assert len(matches) == 121

    target_index = matches[0]
    target = sidecar[target_index]
    assert target["row_id"] in authorized
    assert target["intervention_type"] == "polarity_flip"
    assert target["grammar_status"] == "FAIL"
    assert target["integrity_status"] == "INELIGIBLE"
    assert target["canonical_status"] == "PASS"
    assert target["audit_changed_axes"] == ["polarity"]
    return target_index


@lru_cache(maxsize=1)
def _cached_repaired_stage185_replay_case() -> tuple[list[dict], list[dict], set[str], dict]:
    baseline = authoritative_clean_baseline_rows()
    authorized = actual_authorized_f1_row_ids()
    replay = f1.actual_repaired_generator_replay(baseline, authorized)
    repaired = replay["replayed_records"]
    assert len(baseline) == 3600
    assert len(repaired) == 3600
    assert len(authorized) == 121
    assert "time_swap" not in {row["intervention_type"] for row in repaired}
    assert set(replay["actual_generator_repair_consumed_row_ids"]) == authorized
    return baseline, repaired, authorized, replay


def repaired_stage185_replay_case() -> tuple[list[dict], list[dict], set[str], dict]:
    baseline, repaired, authorized, replay = _cached_repaired_stage185_replay_case()
    return copy.deepcopy(baseline), copy.deepcopy(repaired), set(authorized), copy.deepcopy(replay)

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


def test_extract_authorized_targets_sorted_api_output_but_compatibility_uses_baseline_order():
    records = [authority_record("b"), authority_record("a")]
    targets = f1.extract_authorized_f1_targets(records, {"a", "b"})
    assert targets["F1_target_pair_ids"] == ["a", "b"]
    assert targets["authorized_F1_row_ids"] == ["a__polarity_flip", "b__polarity_flip"]

    authorized = targets["authorized_F1_row_ids"] + [f"x{i:03d}__polarity_flip" for i in range(119)]
    baseline_order = ["b__polarity_flip", "a__polarity_flip"] + [f"x{i:03d}__polarity_flip" for i in range(119)]
    accounting = f1.build_compatibility_accounting(
        authorized,
        [compatibility_case(pair_id="a"), compatibility_case(pair_id="b")],
        baseline_row_order=baseline_order,
    )
    assert accounting["compatibility_checked_row_ids"] == ["b__polarity_flip", "a__polarity_flip"]
    assert accounting["pass_row_ids"] == ["b__polarity_flip", "a__polarity_flip"]

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
        authorized_f1_row_ids={"p__polarity_flip"},
        base_form_source_identity=compatibility_base_identity(),
        **resolved_compatibility_prerequisites(),
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
        authorized_f1_row_ids={"p__polarity_flip"},
        base_form_source_identity=compatibility_base_identity(),
        **resolved_compatibility_prerequisites(),
    )
    assert audit["semantic_validation_status"] == "REJECTED"
    assert audit["semantic_polarity_preserved"] is False
    assert audit["candidate_accepted"] is False
    assert "ORIGINAL_AUTHORIZED_SPAN_MISSING_OR_AMBIGUOUS" in audit["ordered_rejection_codes"]


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
        authorized_f1_row_ids={"p__polarity_flip"},
        base_form_source_identity=compatibility_base_identity(),
        **resolved_compatibility_prerequisites(),
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
        authorized_f1_row_ids={f"{pair_id}__polarity_flip"},
        base_form_source_identity=compatibility_base_identity(),
        **resolved_compatibility_prerequisites(),
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
        authorized_f1_row_ids={"p__polarity_flip"},
        base_form_source_identity=compatibility_base_identity(),
        **resolved_compatibility_prerequisites(),
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
        authorized_f1_row_ids={"p__polarity_flip"},
        base_form_source_identity=compatibility_base_identity(),
        **resolved_compatibility_prerequisites(),
    )
    evidence = audit["semantic_validation_evidence"]
    assert evidence["raw_stage185_transition"]["stage185_transition_pass"] is True
    assert evidence["raw_stage185_transition"]["F1_integrity_transition"] == "RAW_INELIGIBLE_TO_RAW_INELIGIBLE"
    assert evidence["stage185_transition_pass"] is True
    assert evidence["F1_effective_transition"] == "RAW_INELIGIBLE_TO_COMPATIBILITY_ELIGIBLE"
    assert "F1_integrity_transition" not in {
        key
        for key in evidence
        if key != "raw_stage185_transition"
    }
    assert evidence["compatibility_record"]["effective_F1_repair_integrity_status"] == "COMPATIBILITY_ELIGIBLE"


def test_manual_compatibility_effective_transition_is_unresolved():
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
        authorized_f1_row_ids={"p__polarity_flip"},
        base_form_source_identity=compatibility_base_identity("FAIL"),
        **resolved_compatibility_prerequisites(),
    )
    evidence = audit["semantic_validation_evidence"]
    assert evidence["compatibility_record"]["compatibility_status"] == f1.COMPATIBILITY_MANUAL_STATUS
    assert evidence["raw_stage185_transition"]["F1_integrity_transition"] == "RAW_INELIGIBLE_TO_RAW_INELIGIBLE"
    assert evidence["F1_effective_transition"] == "UNRESOLVED"


def test_rejected_compatibility_effective_transition_is_unresolved():
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
        authorized_f1_row_ids={"other__polarity_flip"},
        base_form_source_identity=compatibility_base_identity(),
        **resolved_compatibility_prerequisites(),
    )
    evidence = audit["semantic_validation_evidence"]
    assert evidence["compatibility_record"]["compatibility_status"] == f1.COMPATIBILITY_REJECTED_STATUS
    assert evidence["raw_stage185_transition"]["F1_integrity_transition"] == "RAW_INELIGIBLE_TO_RAW_INELIGIBLE"
    assert evidence["F1_effective_transition"] == "UNRESOLVED"


def assert_stage185_provenance_failure(sidecar_mutation):
    source, expected_generator_rows, sidecar = stage185_source_expected_and_sidecar()
    sidecar_mutation(source, sidecar)
    result = f1.validate_stage185_sidecar_provenance(
        source,
        sidecar,
        actual_source_dataset_sha256=f1.P3W6F1_AUTHORITATIVE_DATA_SHA256,
        actual_source_dataset_path=Path(f1.P3W6F1_AUTHORITATIVE_DATA_PATH),
        actual_integrity_builder_sha256="builder-sha",
        expected_generator_rows=expected_generator_rows,
        runtime_authority=stage185_runtime_authority_stub(),
    )
    assert result["stage185_provenance_status"] == "STAGE185_PROVENANCE_UNRESOLVED"
    assert result["stage185_provenance_pass"] is False


def test_stage185_provenance_positive_case():
    source, expected_generator_rows, sidecar = stage185_source_expected_and_sidecar()
    result = f1.validate_stage185_sidecar_provenance(
        source,
        sidecar,
        actual_source_dataset_sha256=f1.P3W6F1_AUTHORITATIVE_DATA_SHA256,
        actual_source_dataset_path=Path(f1.P3W6F1_AUTHORITATIVE_DATA_PATH),
        actual_integrity_builder_sha256="builder-sha",
        expected_generator_rows=expected_generator_rows,
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


def test_stage185_test_fixture_uses_complete_authoritative_clean_family_universe():
    baseline = authoritative_clean_baseline_rows()
    source, _sidecar = stage185_source_and_sidecar()
    baseline_families = {row["intervention_type"] for row in baseline}
    source_families = {row["intervention_type"] for row in source}
    assert len(source) == 3600
    assert source_families == baseline_families
    assert {"none", "paraphrase", "polarity_flip"}.issubset(source_families)


def test_stage185_test_fixture_excludes_time_swap_by_authoritative_baseline_topology():
    baseline = authoritative_clean_baseline_rows()
    projected_expected_rows = authoritative_clean_baseline_expected_generator_rows()
    baseline_families = {row["intervention_type"] for row in baseline}
    projected_families = {row["intervention_type"] for row in projected_expected_rows}
    assert len(baseline) == 3600
    assert len(projected_expected_rows) == 3600
    assert baseline_families == projected_families
    assert "time_swap" not in baseline_families


def test_repaired_stage185_fixture_uses_exact_121_authority_rows_on_clean_topology():
    baseline, repaired, authorized, replay = repaired_stage185_replay_case()
    repaired_families = {row["intervention_type"] for row in repaired}
    assert len(baseline) == 3600
    assert len(repaired) == 3600
    assert len(authorized) == 121
    assert len(replay["actual_generator_repair_consumed_row_ids"]) == 121
    assert set(replay["actual_generator_repair_consumed_row_ids"]) == authorized
    assert "time_swap" not in repaired_families


def test_stage185_expected_sidecar_preserves_production_dev_split():
    source = authoritative_clean_baseline_rows()
    expected_generator_rows = authoritative_clean_baseline_expected_generator_rows()
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
        expected_generator_rows=expected_generator_rows,
        runtime_authority=stage185_runtime_authority_stub(),
    )
    dev_row = next(row for row in sidecar if row["pair_id"] in dev_ids)
    assert dev_row["split"] == "dev"


def test_stage185_expected_sidecar_preserves_production_train_split():
    source = authoritative_clean_baseline_rows()
    expected_generator_rows = authoritative_clean_baseline_expected_generator_rows()
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
        expected_generator_rows=expected_generator_rows,
        runtime_authority=stage185_runtime_authority_stub(),
    )
    train_row = next(row for row in sidecar if row["pair_id"] not in dev_ids)
    assert train_row["split"] == "train"


def test_stage185_split_tamper_fails_provenance():
    source, expected_generator_rows, sidecar = stage185_source_expected_and_sidecar()
    sidecar[0] = dict(sidecar[0]) | {"split": "dev" if sidecar[0]["split"] == "train" else "train"}
    result = f1.validate_stage185_sidecar_provenance(
        source,
        sidecar,
        actual_source_dataset_sha256=f1.P3W6F1_AUTHORITATIVE_DATA_SHA256,
        actual_source_dataset_path=Path(f1.P3W6F1_AUTHORITATIVE_DATA_PATH),
        actual_integrity_builder_sha256="builder-sha",
        expected_generator_rows=expected_generator_rows,
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


def test_stage185_semantic_tamper_fixture_targets_authorized_defective_f1_row():
    _source, _expected_generator_rows, sidecar = stage185_source_expected_and_sidecar()
    authorized = actual_authorized_f1_row_ids()
    authorized_rows = [row_value for row_value in sidecar if row_value["row_id"] in authorized]
    target = sidecar[authorized_defective_f1_sidecar_index(sidecar)]
    assert len(authorized_rows) == 121
    assert target["row_id"] in authorized
    assert target["intervention_type"] == "polarity_flip"
    assert target["grammar_status"] == "FAIL"
    assert target["integrity_status"] == "INELIGIBLE"
    assert target["canonical_status"] == "PASS"
    assert target["audit_changed_axes"] == ["polarity"]


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
    source, expected_generator_rows, sidecar = stage185_source_expected_and_sidecar()
    target_index = authorized_defective_f1_sidecar_index(sidecar)
    assert sidecar[target_index][field] != value
    sidecar[target_index] = dict(sidecar[target_index]) | {field: value}
    result = f1.validate_stage185_sidecar_provenance(
        source,
        sidecar,
        actual_source_dataset_sha256=f1.P3W6F1_AUTHORITATIVE_DATA_SHA256,
        actual_source_dataset_path=Path(f1.P3W6F1_AUTHORITATIVE_DATA_PATH),
        actual_integrity_builder_sha256="builder-sha",
        expected_generator_rows=expected_generator_rows,
        runtime_authority=stage185_runtime_authority_stub(),
    )
    assert result["stage185_provenance_status"] == "STAGE185_PROVENANCE_UNRESOLVED"
    assert result["stage185_provenance_pass"] is False
    assert "stage185_semantic_identity" in result["stage185_provenance_failures"]


def test_repaired_stage185_expected_sidecar_uses_authorized_repaired_generator_replay():
    baseline, repaired, authorized, _raw_replay = repaired_stage185_replay_case()
    replay_validation = f1.validate_repaired_output_replay_identity(
        baseline,
        repaired,
        authorized,
    )
    assert replay_validation["generator_replay_identity_pass"] is True
    assert replay_validation["generator_replay_identity_status"] == "PASS"
    assert replay_validation["replayed_records"] == repaired
    assert set(replay_validation["actual_generator_repair_consumed_row_ids"]) == authorized
    sidecar = f1.derive_stage185_expected_sidecar(
        repaired,
        actual_source_dataset_sha256="repaired-sha",
        actual_source_dataset_path=Path("reports/p3w6f1_repaired.jsonl"),
        actual_integrity_builder_sha256="builder-sha",
        expected_generator_rows=replay_validation["replayed_records"],
        runtime_authority=stage185_runtime_authority_stub(),
    )
    repaired_member = next(row for row in sidecar if row["row_id"] in authorized)
    assert repaired_member["grammar_status"] == "PASS"
    assert repaired_member["integrity_status"] == "INELIGIBLE"
    assert repaired_member["intervention_contract_status"] == "FAIL"
    assert repaired_member["audit_changed_axes"] == ["polarity", "predicate"]
    assert repaired_member["dataset_source_status"] == "PASS"


def test_stage185_expected_generator_evidence_does_not_replace_replay_identity_authority():
    baseline, repaired, authorized, _raw_replay = repaired_stage185_replay_case()
    replay_validation = f1.validate_repaired_output_replay_identity(
        baseline,
        repaired,
        authorized,
    )
    assert replay_validation["generator_replay_identity_pass"] is True
    unrepaired_expected_rows = authoritative_clean_baseline_expected_generator_rows()
    observed_sidecar = f1.derive_stage185_expected_sidecar(
        repaired,
        actual_source_dataset_sha256="repaired-sha",
        actual_source_dataset_path=Path("reports/p3w6f1_repaired.jsonl"),
        actual_integrity_builder_sha256="builder-sha",
        expected_generator_rows=replay_validation["replayed_records"],
        runtime_authority=stage185_runtime_authority_stub(),
    )
    result = f1.validate_stage185_sidecar_provenance(
        repaired,
        observed_sidecar,
        actual_source_dataset_sha256="repaired-sha",
        actual_source_dataset_path=Path("reports/p3w6f1_repaired.jsonl"),
        actual_integrity_builder_sha256="builder-sha",
        expected_generator_rows=unrepaired_expected_rows,
        runtime_authority=stage185_runtime_authority_stub(),
    )
    assert result["stage185_provenance_pass"] is True
    assert result["stage185_provenance_status"] == "PASS"


def test_stage185_expected_generator_label_mismatch_fails_semantic_identity():
    baseline, repaired, authorized, _raw_replay = repaired_stage185_replay_case()
    replay_validation = f1.validate_repaired_output_replay_identity(
        baseline,
        repaired,
        authorized,
    )
    assert replay_validation["generator_replay_identity_pass"] is True
    observed_sidecar = f1.derive_stage185_expected_sidecar(
        repaired,
        actual_source_dataset_sha256="repaired-sha",
        actual_source_dataset_path=Path("reports/p3w6f1_repaired.jsonl"),
        actual_integrity_builder_sha256="builder-sha",
        expected_generator_rows=replay_validation["replayed_records"],
        runtime_authority=stage185_runtime_authority_stub(),
    )
    tampered_expected_rows = copy.deepcopy(authoritative_clean_baseline_expected_generator_rows())
    target_row_id = sorted(authorized)[0]
    target_expected = next(row_value for row_value in tampered_expected_rows if row_value["id"] == target_row_id)
    replacement = "SUPPORT" if target_expected["polarity_label"] != "SUPPORT" else "REFUTE"
    assert target_expected["polarity_label"] != replacement
    target_expected["polarity_label"] = replacement
    result = f1.validate_stage185_sidecar_provenance(
        repaired,
        observed_sidecar,
        actual_source_dataset_sha256="repaired-sha",
        actual_source_dataset_path=Path("reports/p3w6f1_repaired.jsonl"),
        actual_integrity_builder_sha256="builder-sha",
        expected_generator_rows=tampered_expected_rows,
        runtime_authority=stage185_runtime_authority_stub(),
    )
    assert result["stage185_provenance_pass"] is False
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
        authorized_f1_row_ids={"p__polarity_flip"},
        base_form_source_identity=compatibility_base_identity(),
        grammar_validator_source=f1.P3W6F1_STAGE182_ANALYZER_PATH,
        grammar_validator_sha256="g" * 64,
        **resolved_compatibility_prerequisites(),
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
        authorized_f1_row_ids={"p__polarity_flip"},
        base_form_source_identity=compatibility_base_identity(),
        **resolved_compatibility_prerequisites(),
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
    monkeypatch.setattr(f1, "validate_compatibility_base_form_source_identity", lambda **kwargs: compatibility_base_identity())
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
        return {"stage185_provenance_pass": stage185_pass, "stage185_provenance_status": "PASS" if stage185_pass else f1.STAGE185_PROVENANCE_UNRESOLVED}
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
        return {"schema_version": f1.SCHEMA_VERSION, "F1_execution_decision": f1.ALL_ACCEPTED_DECISION, "F1_execution_blockers": [], "F1_artifact_paths": {}}
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

def compatibility_case(
    *,
    original_text: str = "A did not approved B.",
    regenerated_text: str = "A did not approve B.",
    pair_id: str = "p",
    intervention: str = "polarity_flip",
    final: str = "REFUTE",
    sidecar_before_override: dict | None = None,
    sidecar_after_override: dict | None = None,
    authorized: set[str] | None = None,
    f2_ids: set[str] | None = None,
    **kwargs,
) -> dict:
    original = row(pair_id, intervention, original_text)
    regenerated = row(pair_id, intervention, regenerated_text, final)
    canonical = row(pair_id, "none", "A approved B.", "SUPPORT")
    for key, value in resolved_compatibility_prerequisites().items():
        kwargs.setdefault(key, value)
    return f1.semantic_audit_record(
        original,
        regenerated,
        canonical,
        sidecar_before=stage185_sidecar(f"{pair_id}__{intervention}", pair_id, before=True, override=sidecar_before_override),
        sidecar_after=stage185_sidecar(f"{pair_id}__{intervention}", pair_id, override=sidecar_after_override),
        inflected_predicate_surface="approved",
        expected_base_predicate=kwargs.pop("expected_base_predicate", "approve"),
        authorized_f1_row_ids=authorized or {f"{pair_id}__{intervention}"},
        f2_row_ids=f2_ids,
        base_form_source_identity=kwargs.pop("base_form_source_identity", compatibility_base_identity()),
        **kwargs,
    )



def test_exact_valid_121_authorized_rows_derive_compatibility_pass_fixture():
    baseline, repaired, authorized, replay = repaired_stage185_replay_case()
    _source, _expected_generator_rows, baseline_sidecar_rows = stage185_source_expected_and_sidecar()
    repaired_sidecar_rows = f1.derive_stage185_expected_sidecar(
        repaired,
        actual_source_dataset_sha256="repaired-sha",
        actual_source_dataset_path=Path("reports/p3w6f1_repaired.jsonl"),
        actual_integrity_builder_sha256="builder-sha",
        expected_generator_rows=replay["replayed_records"],
        runtime_authority=stage185_runtime_authority_stub(),
    )
    root = Path(f1.__file__).resolve().parents[1]
    pairs = f1.load_jsonl(root / f1.P3W6F1_P3W4_PAIRS_PATH)
    audit_rows = f1.audit_authorized_candidates(
        baseline,
        repaired,
        pairs,
        authorized,
        f1.sidecar_by_row_id(baseline_sidecar_rows),
        f1.sidecar_by_row_id(repaired_sidecar_rows),
        base_form_source_identity=compatibility_base_identity(),
        generator_replay_identity_status="PASS",
        repair_consumption_status="PASS",
        full_output_isolation_status="PASS",
        stage185_v1_runtime_authority_status="PASS",
        baseline_stage185_v1_provenance_status="PASS",
        repaired_stage185_v1_provenance_status="PASS",
    )
    assert len(audit_rows) == 121
    assert all(compatibility_record(row)["compatibility_status"] == "PASS" for row in audit_rows)
    assert [row["original_row_id"] for row in audit_rows] == [row["id"] for row in baseline if row["id"] in authorized]
def test_compatibility_record_pass_preserves_raw_stage185_and_derives_effective_state():
    audit = compatibility_case()
    record = compatibility_record(audit)
    assert audit["semantic_validation_status"] == f1.PASS_STATUS
    assert record["compatibility_status"] == "PASS"
    assert record["label_identity_status"] == "PASS"
    assert audit["original_final_label"] == "REFUTE"
    assert audit["regenerated_final_label"] == "REFUTE"
    assert "POLARITY_LABEL_CONTRADICTION" not in record["ordered_compatibility_blockers"]
    assert record["compatibility_pass"] is True
    assert record["repaired_stage185_v1_intervention_contract_status"] == "FAIL"
    assert record["repaired_stage185_v1_integrity_status"] == "INELIGIBLE"
    assert record["repaired_stage185_v1_audit_changed_axes"] == ["polarity", "predicate"]
    assert record["effective_F1_repair_integrity_status"] == "COMPATIBILITY_ELIGIBLE"
    assert record["effective_semantic_changed_axes"] == ["polarity"]
    assert record["ordered_compatibility_blockers"] == []
    assert all(record[field] is not None for field in f1.COMPATIBILITY_ROW_FIELDS)


@pytest.mark.parametrize(
    ("kwargs", "blocker"),
    [
        ({"expected_base_predicate": "approves", "regenerated_text": "A did not approves B."}, "EXPECTED_BASE_PREDICATE_MISMATCH"),
        ({"authorized": {"other__polarity_flip"}}, "UNAUTHORIZED_F1_ROW"),
        ({"regenerated_text": "A did not reviewed B.", "expected_base_predicate": "reviewed"}, "EXPECTED_BASE_PREDICATE_MISMATCH"),
        ({"regenerated_text": "A did not inspect B.", "expected_base_predicate": "inspect"}, "EXPECTED_BASE_PREDICATE_MISMATCH"),
        ({"original_text": "A did not approved B and did not approved C."}, "ORIGINAL_AUTHORIZED_SPAN_MISSING_OR_AMBIGUOUS"),
        ({"original_text": "A approved B."}, "ORIGINAL_AUTHORIZED_SPAN_MISSING_OR_AMBIGUOUS"),
        ({"regenerated_text": "A approved B."}, "REGENERATED_AUTHORIZED_SPAN_MISSING_OR_AMBIGUOUS"),
        ({"regenerated_text": "A did not approve C."}, "OUTSIDE_SPAN_CHANGED"),
        ({"final": "SUPPORT"}, "LABEL_IDENTITY_CHANGED"),
        ({"intervention": "predicate_swap"}, "INTERVENTION_TYPE_NOT_POLARITY_FLIP"),
        ({"f2_ids": {"p__polarity_flip"}}, "F2_COMPATIBILITY_SCOPE_REJECTED"),
        ({"sidecar_after_override": {"grammar_status": "FAIL"}}, "REPAIRED_STAGE185_SIGNATURE_MISMATCH"),
        ({"sidecar_after_override": {"audit_changed_axes": ["predicate"]}}, "REPAIRED_STAGE185_SIGNATURE_MISMATCH"),
        ({"sidecar_after_override": {"audit_changed_axes": ["polarity", "predicate", "time"]}}, "REPAIRED_STAGE185_SIGNATURE_MISMATCH"),
    ],
)
def test_compatibility_concrete_contradictions_rejected(kwargs: dict, blocker: str):
    audit = compatibility_case(**kwargs)
    record = compatibility_record(audit)
    assert audit["semantic_validation_status"] == f1.REJECTED_STATUS
    assert record["compatibility_status"] == "REJECTED"
    assert record["effective_F1_repair_integrity_status"] == "COMPATIBILITY_INELIGIBLE"
    assert blocker in record["ordered_compatibility_blockers"]


@pytest.mark.parametrize(
    ("kwargs", "blocker"),
    [
        ({"base_form_source_identity": compatibility_base_identity("FAIL")}, "BASE_FORM_SOURCE_IDENTITY_UNRESOLVED"),
        ({"generator_replay_identity_status": "FAIL"}, "GENERATOR_REPLAY_IDENTITY_UNRESOLVED"),
        ({"repair_consumption_status": "FAIL"}, "REPAIR_CONSUMPTION_UNRESOLVED"),
        ({"full_output_isolation_status": "FAIL"}, "FULL_OUTPUT_ISOLATION_FAILED"),
        ({"baseline_stage185_v1_provenance_status": "FAIL"}, "BASELINE_STAGE185_PROVENANCE_UNRESOLVED"),
        ({"repaired_stage185_v1_provenance_status": "FAIL"}, "REPAIRED_STAGE185_PROVENANCE_UNRESOLVED"),
        ({"stage185_v1_runtime_authority_status": "FAIL"}, "STAGE185_RUNTIME_AUTHORITY_UNRESOLVED"),
    ],
)
def test_compatibility_unresolved_authority_manual_not_rejected(kwargs: dict, blocker: str):
    audit = compatibility_case(**kwargs)
    record = compatibility_record(audit)
    assert audit["semantic_validation_status"] == f1.MANUAL_STATUS
    assert record["compatibility_status"] == "MANUAL_REVIEW_REQUIRED"
    assert record["effective_F1_repair_integrity_status"] == "COMPATIBILITY_BLOCKED"
    assert blocker in record["ordered_compatibility_blockers"]



def test_compatibility_claim_and_non_evidence_mutations_rejected():
    original = row("p", "polarity_flip", "A did not approved B.")
    claim_changed = row("p", "polarity_flip", "A did not approve B.") | {"claim": "changed claim"}
    non_evidence_changed = row("p", "polarity_flip", "A did not approve B.") | {"primary_failure_type": "predicate"}
    canonical = row("p", "none", "A approved B.", "SUPPORT")
    claim_audit = f1.semantic_audit_record(
        original,
        claim_changed,
        canonical,
        sidecar_before=stage185_sidecar("p__polarity_flip", "p", before=True),
        sidecar_after=stage185_sidecar("p__polarity_flip", "p"),
        inflected_predicate_surface="approved",
        expected_base_predicate="approve",
        authorized_f1_row_ids={"p__polarity_flip"},
        base_form_source_identity=compatibility_base_identity(),
        **resolved_compatibility_prerequisites(),
    )
    non_evidence_audit = f1.semantic_audit_record(
        original,
        non_evidence_changed,
        canonical,
        sidecar_before=stage185_sidecar("p__polarity_flip", "p", before=True),
        sidecar_after=stage185_sidecar("p__polarity_flip", "p"),
        inflected_predicate_surface="approved",
        expected_base_predicate="approve",
        authorized_f1_row_ids={"p__polarity_flip"},
        base_form_source_identity=compatibility_base_identity(),
        **resolved_compatibility_prerequisites(),
    )
    assert compatibility_record(claim_audit)["compatibility_status"] == "REJECTED"
    assert "CLAIM_CHANGED" in compatibility_record(claim_audit)["ordered_compatibility_blockers"]
    assert compatibility_record(non_evidence_audit)["compatibility_status"] == "REJECTED"
    assert "NON_EVIDENCE_FIELD_CHANGED" in compatibility_record(non_evidence_audit)["ordered_compatibility_blockers"]

def test_unresolved_raw_stage185_binary_fields_are_null_never_not_run():
    audit = compatibility_case(repaired_stage185_v1_provenance_status="FAIL")
    record = compatibility_record(audit)
    assert record["compatibility_status"] == "MANUAL_REVIEW_REQUIRED"
    assert record["repaired_stage185_v1_canonical_status"] is None
    assert record["repaired_stage185_v1_canonical_status"] != "NOT_RUN"


def test_raw_stage185_binary_not_run_is_rejected_when_observed():
    audit = compatibility_case(sidecar_after_override={"canonical_status": "NOT_RUN"})
    record = compatibility_record(audit)
    assert record["compatibility_status"] == "REJECTED"
    assert "RAW_STAGE185_BINARY_NOT_RUN" in record["ordered_compatibility_blockers"]


def test_authorized_row_accounting_excludes_unauthorized_rejected_invocation():
    authorized = ["p__polarity_flip", "q__polarity_flip"]
    good = compatibility_case(pair_id="p")
    unauthorized = compatibility_case(pair_id="u", authorized=set(authorized))
    accounting = f1.build_compatibility_accounting(
        authorized,
        [good, unauthorized],
        baseline_row_order=["q__polarity_flip", "p__polarity_flip", "u__polarity_flip"],
    )
    assert accounting["compatibility_checked_row_ids"] == ["p__polarity_flip"]
    assert accounting["pass_row_ids"] == ["p__polarity_flip"]
    assert accounting["rejected_row_ids"] == []
    assert accounting["missing_row_ids"] == ["q__polarity_flip"]
    assert accounting["unauthorized_row_ids"] == ["u__polarity_flip"]


def test_required_compatibility_keys_present_under_manual_and_rejected():
    manual = compatibility_record(compatibility_case(generator_replay_identity_status="FAIL"))
    rejected = compatibility_record(compatibility_case(regenerated_text="A did not inspect B.", expected_base_predicate="inspect"))
    assert all(field in manual for field in f1.COMPATIBILITY_ROW_FIELDS)
    assert all(field in rejected for field in f1.COMPATIBILITY_ROW_FIELDS)


def test_finalizer_does_not_upgrade_genuine_compatibility_rejection():
    rejected = compatibility_case(regenerated_text="A did not inspect B.", expected_base_predicate="inspect")
    finalized = f1.finalize_candidate_acceptance(
        [rejected],
        full_output_isolation_validation={"full_output_isolation_pass": True},
        stage185_provenance_validation={"stage185_provenance_pass": True},
        execution_provenance_validation={"execution_provenance_pass": True},
    )
    assert finalized[0]["semantic_validation_status"] == f1.REJECTED_STATUS
    assert compatibility_record(finalized[0])["compatibility_status"] == "REJECTED"

def direct_no_authority_case(**kwargs) -> dict:
    original = row("p", "polarity_flip", "A did not approved B.")
    regenerated = row("p", "polarity_flip", "A did not approve B.")
    canonical = row("p", "none", "A approved B.", "SUPPORT")
    for key, value in resolved_compatibility_prerequisites().items():
        kwargs.setdefault(key, value)
    return f1.semantic_audit_record(
        original,
        regenerated,
        canonical,
        sidecar_before=stage185_sidecar("p__polarity_flip", "p", before=True),
        sidecar_after=stage185_sidecar("p__polarity_flip", "p", override=kwargs.pop("sidecar_after_override", None)),
        inflected_predicate_surface="approved",
        expected_base_predicate="approve",
        **kwargs,
    )


def test_no_authorized_authority_supplied_cannot_self_authorize_or_pass():
    audit = direct_no_authority_case(base_form_source_identity=compatibility_base_identity())
    record = compatibility_record(audit)
    assert record["compatibility_status"] == "MANUAL_REVIEW_REQUIRED"
    assert record["authorized_F1_membership"] is None
    assert "AUTHORIZED_F1_AUTHORITY_UNRESOLVED" in record["ordered_compatibility_blockers"]


def test_no_base_form_source_identity_supplied_cannot_default_to_pass():
    audit = direct_no_authority_case(authorized_f1_row_ids={"p__polarity_flip"})
    record = compatibility_record(audit)
    assert record["compatibility_status"] == "MANUAL_REVIEW_REQUIRED"
    assert record["base_form_source_identity_status"] == "NOT_RUN"
    assert "BASE_FORM_SOURCE_IDENTITY_UNRESOLVED" in record["ordered_compatibility_blockers"]


def test_untrusted_repaired_sidecar_contradiction_remains_manual_under_failed_provenance():
    audit = compatibility_case(
        sidecar_after_override={"grammar_status": "FAIL", "canonical_status": "NOT_RUN"},
        repaired_stage185_v1_provenance_status="FAIL",
    )
    record = compatibility_record(audit)
    assert record["compatibility_status"] == "MANUAL_REVIEW_REQUIRED"
    assert "REPAIRED_STAGE185_SIGNATURE_MISMATCH" not in record["ordered_compatibility_blockers"]
    assert record["repaired_stage185_v1_grammar_status"] is None


def test_untrusted_baseline_sidecar_contradiction_remains_manual_under_failed_provenance():
    original = row("p", "polarity_flip", "A did not approved B.")
    regenerated = row("p", "polarity_flip", "A did not approve B.")
    canonical = row("p", "none", "A approved B.", "SUPPORT")
    audit = f1.semantic_audit_record(
        original,
        regenerated,
        canonical,
        sidecar_before=stage185_sidecar("p__polarity_flip", "p", before=True, override={"grammar_status": "PASS"}),
        sidecar_after=stage185_sidecar("p__polarity_flip", "p"),
        inflected_predicate_surface="approved",
        expected_base_predicate="approve",
        authorized_f1_row_ids={"p__polarity_flip"},
        base_form_source_identity=compatibility_base_identity(),
        generator_replay_identity_status="PASS",
        repair_consumption_status="PASS",
        full_output_isolation_status="PASS",
        stage185_v1_runtime_authority_status="PASS",
        baseline_stage185_v1_provenance_status="FAIL",
        repaired_stage185_v1_provenance_status="PASS",
    )
    record = compatibility_record(audit)
    assert record["compatibility_status"] == "MANUAL_REVIEW_REQUIRED"
    assert "BASELINE_STAGE185_SIGNATURE_MISMATCH" not in record["ordered_compatibility_blockers"]
    assert record["baseline_stage185_v1_grammar_status"] is None


def test_partial_provenance_trusted_baseline_valid_untrusted_repaired_manual():
    audit = compatibility_case(repaired_stage185_v1_provenance_status="FAIL")
    record = compatibility_record(audit)
    assert record["compatibility_status"] == "MANUAL_REVIEW_REQUIRED"
    assert "BASELINE_STAGE185_SIGNATURE_MISMATCH" not in record["ordered_compatibility_blockers"]
    assert "REPAIRED_STAGE185_SIGNATURE_MISMATCH" not in record["ordered_compatibility_blockers"]
    assert record["baseline_stage185_v1_grammar_status"] == "FAIL"
    assert record["repaired_stage185_v1_grammar_status"] is None


def test_partial_provenance_trusted_baseline_contradiction_rejected_even_if_repaired_untrusted():
    audit = compatibility_case(
        sidecar_before_override={"grammar_status": "PASS"},
        repaired_stage185_v1_provenance_status="FAIL",
    )
    record = compatibility_record(audit)
    assert record["compatibility_status"] == "REJECTED"
    assert "BASELINE_STAGE185_SIGNATURE_MISMATCH" in record["ordered_compatibility_blockers"]
    assert "REPAIRED_STAGE185_SIGNATURE_MISMATCH" not in record["ordered_compatibility_blockers"]
    assert record["repaired_stage185_v1_grammar_status"] is None


def test_partial_provenance_untrusted_baseline_trusted_repaired_valid_manual():
    audit = compatibility_case(baseline_stage185_v1_provenance_status="FAIL")
    record = compatibility_record(audit)
    assert record["compatibility_status"] == "MANUAL_REVIEW_REQUIRED"
    assert "BASELINE_STAGE185_SIGNATURE_MISMATCH" not in record["ordered_compatibility_blockers"]
    assert "REPAIRED_STAGE185_SIGNATURE_MISMATCH" not in record["ordered_compatibility_blockers"]
    assert record["baseline_stage185_v1_grammar_status"] is None
    assert record["repaired_stage185_v1_grammar_status"] == "PASS"


def test_partial_provenance_trusted_repaired_contradiction_rejected_even_if_baseline_untrusted():
    audit = compatibility_case(
        sidecar_after_override={"grammar_status": "FAIL"},
        baseline_stage185_v1_provenance_status="FAIL",
    )
    record = compatibility_record(audit)
    assert record["compatibility_status"] == "REJECTED"
    assert "BASELINE_STAGE185_SIGNATURE_MISMATCH" not in record["ordered_compatibility_blockers"]
    assert "REPAIRED_STAGE185_SIGNATURE_MISMATCH" in record["ordered_compatibility_blockers"]
    assert record["baseline_stage185_v1_grammar_status"] is None

def test_identical_but_wrong_final_labels_are_rejected():
    original = row("p", "polarity_flip", "A did not approved B.", "SUPPORT")
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
        authorized_f1_row_ids={"p__polarity_flip"},
        base_form_source_identity=compatibility_base_identity(),
        **resolved_compatibility_prerequisites(),
    )
    record = compatibility_record(audit)
    assert record["compatibility_status"] == "REJECTED"
    assert record["label_identity_status"] == "PASS"
    assert "FINAL_LABEL_NOT_REFUTE" in record["ordered_compatibility_blockers"]
    assert "LABEL_IDENTITY_CHANGED" not in record["ordered_compatibility_blockers"]


def test_wrong_repaired_polarity_label_rejected():
    original = row("p", "polarity_flip", "A did not approved B.")
    regenerated = row("p", "polarity_flip", "A did not approve B.") | {"polarity_label": "SUPPORT"}
    canonical = row("p", "none", "A approved B.", "SUPPORT")
    audit = f1.semantic_audit_record(
        original,
        regenerated,
        canonical,
        sidecar_before=stage185_sidecar("p__polarity_flip", "p", before=True),
        sidecar_after=stage185_sidecar("p__polarity_flip", "p"),
        inflected_predicate_surface="approved",
        expected_base_predicate="approve",
        authorized_f1_row_ids={"p__polarity_flip"},
        base_form_source_identity=compatibility_base_identity(),
        **resolved_compatibility_prerequisites(),
    )
    record = compatibility_record(audit)
    assert record["compatibility_status"] == "REJECTED"
    assert "POLARITY_LABEL_CONTRADICTION" in record["ordered_compatibility_blockers"]


def test_label_identity_mutation_uses_specific_label_blockers():
    audit = compatibility_case(final="SUPPORT")
    record = compatibility_record(audit)
    assert record["compatibility_status"] == "REJECTED"
    assert "LABEL_IDENTITY_CHANGED" in record["ordered_compatibility_blockers"]
    assert "FINAL_LABEL_NOT_REFUTE" in record["ordered_compatibility_blockers"]
    assert audit["semantic_validation_status"] == "REJECTED"
    assert audit["semantic_polarity_preserved"] is False
    assert audit["candidate_accepted"] is False


def test_finalizer_does_not_upgrade_label_contract_rejections():
    identity_mutation = compatibility_case(final="SUPPORT")

    original_wrong = row("p", "polarity_flip", "A did not approved B.", "SUPPORT")
    regenerated_wrong = row("p", "polarity_flip", "A did not approve B.", "SUPPORT")
    same_wrong = f1.semantic_audit_record(
        original_wrong,
        regenerated_wrong,
        row("p", "none", "A approved B.", "SUPPORT"),
        sidecar_before=stage185_sidecar("p__polarity_flip", "p", before=True),
        sidecar_after=stage185_sidecar("p__polarity_flip", "p"),
        inflected_predicate_surface="approved",
        expected_base_predicate="approve",
        authorized_f1_row_ids={"p__polarity_flip"},
        base_form_source_identity=compatibility_base_identity(),
        **resolved_compatibility_prerequisites(),
    )

    wrong_polarity = f1.semantic_audit_record(
        row("p", "polarity_flip", "A did not approved B."),
        row("p", "polarity_flip", "A did not approve B.") | {"polarity_label": "SUPPORT"},
        row("p", "none", "A approved B.", "SUPPORT"),
        sidecar_before=stage185_sidecar("p__polarity_flip", "p", before=True),
        sidecar_after=stage185_sidecar("p__polarity_flip", "p"),
        inflected_predicate_surface="approved",
        expected_base_predicate="approve",
        authorized_f1_row_ids={"p__polarity_flip"},
        base_form_source_identity=compatibility_base_identity(),
        **resolved_compatibility_prerequisites(),
    )

    finalized = f1.finalize_candidate_acceptance(
        [identity_mutation, same_wrong, wrong_polarity],
        full_output_isolation_validation={"full_output_isolation_pass": True},
        stage185_provenance_validation={"stage185_provenance_pass": True},
        execution_provenance_validation={"execution_provenance_pass": True},
    )
    assert [candidate["semantic_validation_status"] for candidate in finalized] == ["REJECTED", "REJECTED", "REJECTED"]
    assert [candidate["semantic_polarity_preserved"] for candidate in finalized] == [False, False, False]
    assert [candidate["candidate_accepted"] for candidate in finalized] == [False, False, False]
    assert "LABEL_IDENTITY_CHANGED" in compatibility_record(finalized[0])["ordered_compatibility_blockers"]
    assert "FINAL_LABEL_NOT_REFUTE" in compatibility_record(finalized[1])["ordered_compatibility_blockers"]
    assert "POLARITY_LABEL_CONTRADICTION" in compatibility_record(finalized[2])["ordered_compatibility_blockers"]

def test_compatibility_accounting_uses_compatibility_status_after_global_downgrade():
    candidate = compatibility_case(pair_id="p")
    finalized = f1.finalize_candidate_acceptance(
        [candidate],
        full_output_isolation_validation={"full_output_isolation_pass": False},
        stage185_provenance_validation={"stage185_provenance_pass": True},
        execution_provenance_validation={"execution_provenance_pass": True},
    )
    assert finalized[0]["semantic_validation_status"] == f1.MANUAL_STATUS
    accounting = f1.build_compatibility_accounting(["p__polarity_flip"], finalized, baseline_row_order=["p__polarity_flip"])
    assert accounting["pass_row_ids"] == ["p__polarity_flip"]
    assert accounting["manual_row_ids"] == []


def test_interleaved_compatibility_accounting_preserves_authoritative_order():
    passed = compatibility_case(pair_id="p")
    manual = compatibility_case(pair_id="m", generator_replay_identity_status="FAIL")
    rejected = compatibility_case(pair_id="r", regenerated_text="A did not inspect B.", expected_base_predicate="inspect")
    authorized = ["p__polarity_flip", "m__polarity_flip", "r__polarity_flip"] + [f"x{i:03d}__polarity_flip" for i in range(118)]
    order = ["m__polarity_flip", "p__polarity_flip", "r__polarity_flip"] + [f"x{i:03d}__polarity_flip" for i in range(118)]
    accounting = f1.build_compatibility_accounting(authorized, [passed, manual, rejected], baseline_row_order=order)
    assert accounting["compatibility_checked_row_ids"] == ["m__polarity_flip", "p__polarity_flip", "r__polarity_flip"]
    assert accounting["manual_row_ids"] == ["m__polarity_flip"]
    assert accounting["pass_row_ids"] == ["p__polarity_flip"]
    assert accounting["rejected_row_ids"] == ["r__polarity_flip"]


def test_compatibility_execution_ready_requires_all_121_pass_partition():
    authorized = [f"p{i:03d}__polarity_flip" for i in range(121)]
    audit_rows = [pass_audit(f"p{i:03d}") for i in range(121)]
    accounting = f1.build_compatibility_accounting(authorized, audit_rows, baseline_row_order=authorized)
    readiness = f1.compatibility_execution_ready(accounting)
    assert accounting["compatibility_checked_count"] == 121
    assert accounting["compatibility_pass_count"] == 121
    assert accounting["compatibility_manual_count"] == 0
    assert accounting["compatibility_rejected_count"] == 0
    assert accounting["missing_count"] == 0
    assert accounting["unauthorized_count"] == 0
    assert readiness["compatibility_execution_ready"] is True
    assert readiness["compatibility_execution_readiness_status"] == "PASS"


@pytest.mark.parametrize(
    "audit_rows",
    [
        [],
        [compatibility_case(pair_id="p", generator_replay_identity_status="FAIL")],
        [compatibility_case(pair_id="p", regenerated_text="A did not inspect B.", expected_base_predicate="inspect")],
    ],
)
def test_missing_manual_rejected_accounting_blocks_execution_readiness(audit_rows):
    authorized = ["p__polarity_flip"] + [f"x{i:03d}__polarity_flip" for i in range(120)]
    accounting = f1.build_compatibility_accounting(authorized, audit_rows, baseline_row_order=authorized)
    assert accounting["compatibility_accounting_validation"]["compatibility_accounting_pass"] is True
    readiness = f1.compatibility_execution_ready(accounting)
    assert readiness["compatibility_execution_ready"] is False
    assert readiness["compatibility_execution_readiness_status"] == f1.COMPATIBILITY_ACCOUNTING_UNRESOLVED


def test_structurally_valid_all_missing_accounting_blocks_main_execution_readiness():
    authorized = [f"p{i:03d}__polarity_flip" for i in range(121)]
    summary = f1.build_summary(
        [f"p{i:03d}" for i in range(121)],
        [],
        [],
        authority_cardinality={"authority_cardinality_pass": True},
        target_scope={"target_scope_membership_pass": True},
        base_form_coverage={"coverage_pass": True},
        stage185_provenance_validation={"stage185_provenance_pass": True},
        full_output_validation={"full_output_isolation_pass": True},
        provenance_validation={"execution_provenance_pass": True},
        authorized_target_row_ids=authorized,
        baseline_row_order=authorized,
    )
    accounting = summary["compatibility_accounting"]
    assert accounting["compatibility_accounting_validation"]["compatibility_accounting_pass"] is True
    assert accounting["compatibility_checked_count"] == 0
    assert accounting["missing_count"] == 121
    assert f1.COMPATIBILITY_ACCOUNTING_UNRESOLVED in summary["F1_execution_blockers"]
    assert summary["F1_execution_decision"] == f1.BLOCKERS_DECISION

def test_missing_authorized_target_keeps_target_count_121_and_visible_missing():
    authorized = ["p__polarity_flip"] + [f"x{i:03d}__polarity_flip" for i in range(120)]
    accounting = f1.build_compatibility_accounting(authorized, [compatibility_case(pair_id="p")], baseline_row_order=authorized)
    assert accounting["target_count"] == 121
    assert accounting["compatibility_checked_row_ids"] == ["p__polarity_flip"]
    assert len(accounting["missing_row_ids"]) == 120


def test_authorized_target_absent_from_baseline_order_records_authority_failure():
    authorized = ["p__polarity_flip"] + [f"x{i:03d}__polarity_flip" for i in range(120)]
    baseline_order = authorized[:-1]
    accounting = f1.build_compatibility_accounting(authorized, [compatibility_case(pair_id="p")], baseline_row_order=baseline_order)
    assert accounting["target_count"] == 121
    assert authorized[-1] in accounting["missing_row_ids"]
    validation = accounting["compatibility_accounting_validation"]
    assert validation["compatibility_accounting_pass"] is False
    assert "authorized_target_missing_from_baseline_order" in validation["compatibility_accounting_failures"]


@pytest.mark.parametrize(
    ("mutation", "failure"),
    [
        (lambda acc: acc | {"target_count": 120}, "target_count_not_121"),
        (lambda acc: acc | {"compatibility_checked_count": 2}, "compatibility_checked_count_mismatch"),
        (lambda acc: acc | {"missing_row_ids": ["p__polarity_flip"]}, "checked_missing_overlap"),
        (lambda acc: acc | {"manual_row_ids": ["p__polarity_flip"]}, "status_partition_overlap"),
        (lambda acc: acc | {"compatibility_checked_row_ids": ["q__polarity_flip", "p__polarity_flip"]}, "checked_order_not_authoritative"),
        (lambda acc: acc | {"unauthorized_row_ids": ["p__polarity_flip"], "unauthorized_count": 1}, "unauthorized_authorized_overlap"),
        (lambda acc: acc | {"pass_row_ids": ["p__polarity_flip", "p__polarity_flip"], "compatibility_pass_count": 2}, "pass_row_ids_duplicates"),
    ],
)
def test_compatibility_accounting_validator_negative_cases(mutation, failure: str):
    authorized = ["p__polarity_flip", "q__polarity_flip"] + [f"x{i:03d}__polarity_flip" for i in range(119)]
    accounting = f1.build_compatibility_accounting(authorized, [compatibility_case(pair_id="p")], baseline_row_order=authorized)
    result = f1.validate_compatibility_accounting(mutation(accounting), authorized)
    assert result["compatibility_accounting_pass"] is False
    assert failure in result["compatibility_accounting_failures"]


def test_duplicate_authorized_compatibility_result_is_detected():
    authorized = ["p__polarity_flip"] + [f"x{i:03d}__polarity_flip" for i in range(120)]
    accounting = f1.build_compatibility_accounting(
        authorized,
        [compatibility_case(pair_id="p"), compatibility_case(pair_id="p")],
        baseline_row_order=authorized,
    )
    validation = accounting["compatibility_accounting_validation"]
    assert validation["compatibility_accounting_pass"] is False
    assert accounting["duplicate_checked_compatibility_row_ids"] == ["p__polarity_flip"]
    assert "duplicate_checked_compatibility_row_id" in validation["compatibility_accounting_failures"]

def test_provisional_pass_null_required_field_downgrades_and_clears_positive_proof():
    audit = compatibility_case(base_form_source_identity={
        "base_form_source_identity_status": "PASS",
        "base_form_source_path": f1.GENERATOR_SOURCE_PATH,
        "base_form_source_symbol": f1.BASE_FORM_SYMBOL,
    })
    record = compatibility_record(audit)
    assert record["compatibility_status"] == "MANUAL_REVIEW_REQUIRED"
    assert record["predicate_semantic_identity_preserved"] is None
    assert record["surface_realization_changed"] is None
    assert record["compatibility_explained_stage185_axes"] is None
    assert record["effective_semantic_changed_axes"] is None
    assert record["effective_F1_repair_integrity_status"] == "COMPATIBILITY_BLOCKED"


def test_compatibility_artifact_names_and_row_schema(tmp_path: Path):
    authorized = ["p__polarity_flip"] + [f"x{i:03d}__polarity_flip" for i in range(120)]
    baseline_sidecar_path = tmp_path / "baseline.jsonl"
    repaired_sidecar_path = tmp_path / "repaired.jsonl"
    baseline_sidecar_path.write_text("{}\n", encoding="utf-8")
    repaired_sidecar_path.write_text("{}\n", encoding="utf-8")
    paths = f1.write_compatibility_artifacts(
        tmp_path,
        audit_rows=[compatibility_case(pair_id="p")],
        authorized_f1_row_ids=authorized,
        baseline_row_order=authorized,
        execution_decision="BLOCKED",
        base_form_source_identity=compatibility_base_identity(),
        runtime_authority={"stage185_integrity_builder_source_path": f1.P3W6F1_STAGE185_BUILDER_PATH, "stage185_integrity_builder_source_sha256": "b" * 64},
        baseline_sidecar_path=baseline_sidecar_path,
        repaired_sidecar_path=repaired_sidecar_path,
        replay_validation={"actual_generator_repair_consumed_row_ids": authorized},
        isolation={},
    )
    assert set(paths) == {
        "compatibility_rows_jsonl",
        "compatibility_rows_csv",
        "compatibility_summary_json",
        "compatibility_report_md",
        "compatibility_provenance_manifest_json",
    }
    row_payload = f1.load_jsonl(tmp_path / f1.COMPATIBILITY_JSONL_NAME)
    assert row_payload
    assert list(row_payload[0]) == sorted(f1.COMPATIBILITY_ROW_FIELDS)
    assert set(row_payload[0]) == set(f1.COMPATIBILITY_ROW_FIELDS)
    assert (tmp_path / f1.COMPATIBILITY_CSV_NAME).read_text(encoding="utf-8").splitlines()[0].split(",") == list(f1.COMPATIBILITY_ROW_FIELDS)
    manifest = f1.load_json(tmp_path / f1.COMPATIBILITY_PROVENANCE_NAME)
    assert set(manifest["compatibility_artifact_sha256"]) == set(f1.REQUIRED_COMPATIBILITY_ARTIFACT_SHA_KEYS)
    assert f1.validate_compatibility_provenance_manifest_payload(manifest)["compatibility_provenance_manifest_payload_pass"] is True


def test_invalid_compatibility_provenance_blocks_successful_execution_decision(tmp_path: Path):
    authorized = ["p__polarity_flip"] + [f"x{i:03d}__polarity_flip" for i in range(120)]
    f1.write_compatibility_artifacts(
        tmp_path,
        audit_rows=[compatibility_case(pair_id="p")],
        authorized_f1_row_ids=authorized,
        baseline_row_order=authorized,
        execution_decision=f1.ALL_ACCEPTED_DECISION,
        base_form_source_identity=compatibility_base_identity(),
        runtime_authority={"stage185_integrity_builder_source_path": f1.P3W6F1_STAGE185_BUILDER_PATH, "stage185_integrity_builder_source_sha256": "b" * 64},
        baseline_sidecar_path=tmp_path / "missing-baseline.jsonl",
        repaired_sidecar_path=tmp_path / "missing-repaired.jsonl",
        replay_validation={"actual_generator_repair_consumed_row_ids": authorized},
        isolation={},
    )
    summary = f1.load_json(tmp_path / f1.COMPATIBILITY_SUMMARY_NAME)
    manifest = f1.load_json(tmp_path / f1.COMPATIBILITY_PROVENANCE_NAME)
    assert summary["compatibility_execution_decision"] == f1.BLOCKERS_DECISION
    assert manifest["compatibility_provenance_validation"]["compatibility_provenance_manifest_pass"] is False


def test_compatibility_provenance_manifest_required_bindings(tmp_path: Path):
    authorized = ["p__polarity_flip"] + [f"x{i:03d}__polarity_flip" for i in range(120)]
    baseline_sidecar_path = tmp_path / "baseline.jsonl"
    repaired_sidecar_path = tmp_path / "repaired.jsonl"
    baseline_sidecar_path.write_text("{}\n", encoding="utf-8")
    repaired_sidecar_path.write_text("{}\n", encoding="utf-8")
    f1.write_compatibility_artifacts(
        tmp_path,
        audit_rows=[compatibility_case(pair_id="p")],
        authorized_f1_row_ids=authorized,
        baseline_row_order=authorized,
        execution_decision="BLOCKED",
        base_form_source_identity=compatibility_base_identity(),
        runtime_authority={"stage185_integrity_builder_source_path": f1.P3W6F1_STAGE185_BUILDER_PATH, "stage185_integrity_builder_source_sha256": "b" * 64},
        baseline_sidecar_path=baseline_sidecar_path,
        repaired_sidecar_path=repaired_sidecar_path,
        replay_validation={"actual_generator_repair_consumed_row_ids": authorized},
        isolation={},
    )
    manifest = f1.load_json(tmp_path / f1.COMPATIBILITY_PROVENANCE_NAME)
    for field in [
        "compatibility_rule_id",
        "compatibility_rule_version",
        "immediate_authority_commit",
        "audit_report_path",
        "audit_manifest_path",
        "base_form_source_path",
        "base_form_source_sha256",
        "base_form_source_symbol",
        "stage185_v1_runtime_dependency_id",
        "stage185_v1_runtime_dependency_sha256",
        "generator_replay_artifact_sha256",
        "repair_consumption_artifact_sha256",
        "full_output_isolation_artifact_sha256",
        "compatibility_artifact_sha256",
        "compatibility_provenance_manifest_json_payload_sha256",
    ]:
        assert field in manifest
    assert set(manifest["compatibility_artifact_sha256"]) == set(f1.REQUIRED_COMPATIBILITY_ARTIFACT_SHA_KEYS)
    validation = f1.validate_compatibility_provenance_manifest(manifest)
    assert validation["compatibility_provenance_manifest_pass"] is True


def complete_compatibility_manifest_fixture() -> dict:
    return {
        "compatibility_rule_id": f1.COMPATIBILITY_RULE_ID,
        "compatibility_rule_version": f1.COMPATIBILITY_RULE_VERSION,
        "immediate_authority_commit": f1.COMPATIBILITY_IMMEDIATE_AUTHORITY_COMMIT,
        "audit_report_path": f1.COMPATIBILITY_AUDIT_REPORT_PATH,
        "audit_manifest_path": f1.COMPATIBILITY_AUDIT_MANIFEST_PATH,
        "base_form_source_path": f1.GENERATOR_SOURCE_PATH,
        "base_form_source_sha256": "a" * 64,
        "base_form_source_symbol": f1.BASE_FORM_SYMBOL,
        "stage185_v1_runtime_dependency_id": f1.P3W6F1_STAGE185_BUILDER_PATH,
        "stage185_v1_runtime_dependency_sha256": "b" * 64,
        "baseline_stage185_v1_sidecar_sha256": "c" * 64,
        "repaired_stage185_v1_sidecar_sha256": "d" * 64,
        "generator_replay_artifact_sha256": "e" * 64,
        "repair_consumption_artifact_sha256": "f" * 64,
        "full_output_isolation_artifact_sha256": "0" * 64,
        "compatibility_artifact_sha256": {
            "compatibility_rows_jsonl": "1" * 64,
            "compatibility_rows_csv": "2" * 64,
            "compatibility_summary_json": "3" * 64,
            "compatibility_report_md": "4" * 64,
        },
    }


def complete_compatibility_manifest_with_payload_fixture() -> dict:
    manifest = complete_compatibility_manifest_fixture()
    manifest["compatibility_provenance_validation"] = f1.validate_compatibility_provenance_manifest(manifest)
    manifest["compatibility_provenance_manifest_json_payload_sha256"] = f1.canonical_sha256(manifest)
    return manifest


@pytest.mark.parametrize(
    "field",
    [
        "compatibility_rule_id",
        "compatibility_rule_version",
        "immediate_authority_commit",
        "audit_report_path",
        "audit_manifest_path",
        "base_form_source_path",
        "base_form_source_symbol",
        "stage185_v1_runtime_dependency_id",
    ],
)
def test_compatibility_provenance_manifest_validator_rejects_wrong_fixed_identity(field: str):
    manifest = complete_compatibility_manifest_fixture()
    manifest[field] = "wrong-non-empty-value"
    result = f1.validate_compatibility_provenance_manifest(manifest)
    assert result["compatibility_provenance_manifest_pass"] is False
    assert f"{field}_identity_mismatch" in result["compatibility_provenance_manifest_failures"]

def test_compatibility_provenance_manifest_validator_accepts_complete_manifest():
    result = f1.validate_compatibility_provenance_manifest(complete_compatibility_manifest_fixture())
    assert result["compatibility_provenance_manifest_pass"] is True
    assert result["compatibility_provenance_manifest_failures"] == []


def test_compatibility_provenance_requires_all_generated_artifact_hashes():
    for key in f1.REQUIRED_COMPATIBILITY_ARTIFACT_SHA_KEYS:
        manifest = complete_compatibility_manifest_fixture()
        manifest["compatibility_artifact_sha256"] = {
            artifact_key: value
            for artifact_key, value in manifest["compatibility_artifact_sha256"].items()
            if artifact_key != key
        }
        result = f1.validate_compatibility_provenance_manifest(manifest)
        assert result["compatibility_provenance_manifest_pass"] is False
        assert f"compatibility_artifact_sha256:{key}_missing" in result["compatibility_provenance_manifest_failures"]


def test_compatibility_provenance_rejects_unexpected_artifact_hash_key():
    manifest = complete_compatibility_manifest_fixture()
    manifest["compatibility_artifact_sha256"]["compatibility_provenance_manifest_json"] = "5" * 64
    result = f1.validate_compatibility_provenance_manifest(manifest)
    assert result["compatibility_provenance_manifest_pass"] is False
    assert "compatibility_artifact_sha256:compatibility_provenance_manifest_json_unexpected" in result["compatibility_provenance_manifest_failures"]


def test_compatibility_provenance_rejects_malformed_artifact_hash():
    manifest = complete_compatibility_manifest_fixture()
    manifest["compatibility_artifact_sha256"]["compatibility_rows_csv"] = "ABC"
    result = f1.validate_compatibility_provenance_manifest(manifest)
    assert result["compatibility_provenance_manifest_pass"] is False
    assert "compatibility_artifact_sha256:compatibility_rows_csv_malformed_sha256" in result["compatibility_provenance_manifest_failures"]


def test_compatibility_provenance_payload_sha_validates():
    manifest = complete_compatibility_manifest_with_payload_fixture()
    result = f1.validate_compatibility_provenance_manifest_payload(manifest)
    assert result["compatibility_provenance_manifest_payload_pass"] is True
    assert result["compatibility_provenance_manifest_payload_failures"] == []


def test_compatibility_provenance_payload_sha_detects_tampering():
    manifest = complete_compatibility_manifest_with_payload_fixture()
    assert f1.validate_compatibility_provenance_manifest_payload(manifest)["compatibility_provenance_manifest_payload_pass"] is True
    manifest["base_form_source_sha256"] = "9" * 64
    result = f1.validate_compatibility_provenance_manifest_payload(manifest)
    assert result["compatibility_provenance_manifest_payload_pass"] is False
    assert "compatibility_provenance_manifest_json_payload_sha256_mismatch" in result["compatibility_provenance_manifest_payload_failures"]


@pytest.mark.parametrize(
    "digest",
    [None, "", "ABC", "g" * 64, "1" * 63],
)
def test_compatibility_provenance_payload_sha_missing_or_malformed_fails(digest):
    manifest = complete_compatibility_manifest_with_payload_fixture()
    manifest["compatibility_provenance_manifest_json_payload_sha256"] = digest
    result = f1.validate_compatibility_provenance_manifest_payload(manifest)
    assert result["compatibility_provenance_manifest_payload_pass"] is False


@pytest.mark.parametrize(
    ("mutation", "failure"),
    [
        (lambda manifest: manifest | {"baseline_stage185_v1_sidecar_sha256": None}, "baseline_stage185_v1_sidecar_sha256_missing"),
        (lambda manifest: manifest | {"base_form_source_sha256": "ABC"}, "base_form_source_sha256_malformed_sha256"),
        (lambda manifest: {key: value for key, value in manifest.items() if key != "base_form_source_symbol"}, "base_form_source_symbol_missing"),
    ],
)
def test_compatibility_provenance_manifest_validator_rejects_missing_or_malformed_bindings(mutation, failure: str):
    result = f1.validate_compatibility_provenance_manifest(mutation(complete_compatibility_manifest_fixture()))
    assert result["compatibility_provenance_manifest_pass"] is False
    assert failure in result["compatibility_provenance_manifest_failures"]


def accepted_compatibility_summary_fixture() -> dict:
    pair_ids = [f"p{i:03d}" for i in range(121)]
    return f1.build_summary(
        pair_ids,
        pair_ids,
        [pass_audit(pair_id) for pair_id in pair_ids],
        authority_cardinality={"authority_cardinality_pass": True},
        target_scope={"target_scope_membership_pass": True},
        base_form_coverage={"coverage_pass": True},
        stage185_provenance_validation={"stage185_provenance_pass": True},
        full_output_validation={"full_output_isolation_pass": True},
        provenance_validation={"execution_provenance_pass": True},
        authorized_target_row_ids=[f"p{i:03d}__polarity_flip" for i in range(121)],
        baseline_row_order=[f"p{i:03d}__polarity_flip" for i in range(121)],
    )


def test_compatibility_provenance_failure_propagates_to_main_execution_decision():
    summary = accepted_compatibility_summary_fixture()
    assert summary["F1_execution_decision"] == f1.ALL_ACCEPTED_DECISION
    updated = f1.apply_compatibility_provenance_validation(
        summary,
        {
            "compatibility_provenance_manifest_pass": False,
            "compatibility_provenance_manifest_status": "FAIL",
            "compatibility_provenance_manifest_failures": ["base_form_source_sha256_missing"],
        },
    )
    assert updated["F1_execution_decision"] == f1.BLOCKERS_DECISION
    assert f1.COMPATIBILITY_PROVENANCE_UNRESOLVED in updated["F1_execution_blockers"]


def test_valid_compatibility_provenance_does_not_downgrade_main_execution_decision():
    summary = accepted_compatibility_summary_fixture()
    updated = f1.apply_compatibility_provenance_validation(
        summary,
        {
            "compatibility_provenance_manifest_pass": True,
            "compatibility_provenance_manifest_status": "PASS",
            "compatibility_provenance_manifest_failures": [],
        },
    )
    assert updated["F1_execution_decision"] == f1.ALL_ACCEPTED_DECISION
    assert f1.COMPATIBILITY_PROVENANCE_UNRESOLVED not in updated["F1_execution_blockers"]
