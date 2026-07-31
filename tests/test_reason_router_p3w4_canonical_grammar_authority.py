from __future__ import annotations

import ast
import csv
import json
import types
from pathlib import Path

import pytest

from scripts import analyze_reason_router_p3w4_canonical_grammar_authority as p3w4


def source_row(pair: str, intervention: str, final: str, evidence: str | None = None) -> dict:
    row_id = f"{pair}__{intervention}"
    text = evidence if evidence is not None else f"{pair} {intervention} evidence"
    return {
        "id": row_id,
        "pair_id": pair,
        "claim": f"{pair} claim",
        "evidence": text,
        "final_label": final,
        "frame_compatible_label": 1,
        "predicate_covered_label": 1,
        "sufficiency_label": 1,
        "polarity_label": final if final in {"SUPPORT", "REFUTE"} else "NONE",
        "primary_failure_type": "polarity" if intervention == "polarity_flip" else "none",
        "intervention_type": intervention,
    }


def sidecar(row: dict, *, grammar: str = "PASS", canonical: str = "PASS", eligible: bool = False, codes: list[str] | None = None) -> dict:
    return {
        "row_id": row["id"],
        "pair_id": row["pair_id"],
        "canonical_row_id": f"{row['pair_id']}__none",
        "split": "train",
        "source_dataset_sha256": "sha",
        "source_dataset_path": "data/controlled_v5_v3_without_time_swap.jsonl",
        "reason_codes": codes or [],
        "schema_status": "PASS",
        "dataset_source_status": "PASS",
        "grammar_status": grammar,
        "canonical_status": canonical,
        "intervention_contract_status": "PASS",
        "polarity_contamination_status": "PASS",
        "time_swap_status": "PASS",
        "integrity_status": "ELIGIBLE" if eligible else "INELIGIBLE",
    }


def validator_fixture(validator, *, result_source: str = "src") -> dict:
    return {
        "validator": validator,
        "validator_source_path": result_source,
        "validator_function": "grammar_anomaly",
        "validator_source_sha256": "a" * 64,
        "validator_authority_source": result_source,
        "validator_authority_function": "grammar_anomaly",
        "validator_definition_kind": "stage185_local",
        "validator_call_site_function": "build_sidecar",
        "validator_call_site_lineno": 10,
        "validator_call_site_reachable_from_run": True,
        "validator_call_site_authorized": True,
        "validator_call_chain_verified": True,
        "validator_authorized_call_sites": [
            {
                "function_name": "build_sidecar",
                "scope_path": ["function:build_sidecar"],
                "lineno": 10,
                "col_offset": 12,
                "reachable_from_run_or_main": True,
                "authorized_sidecar_construction_function": True,
                "nested_scope": False,
                "class_scope": False,
                "module_level": False,
                "definition_time_expression": False,
                "scope_kind": "sync_function_body",
                "context_path": ["function_body:build_sidecar"],
                "authorized": True,
            }
        ],
        "validator_authorized_call_site_count": 1,
        "validator_callable_source_path": result_source,
        "validator_signature": "(row, fact)",
    }

def tiny_family(monkeypatch):
    monkeypatch.setattr(p3w4, "EXPECTED_F1_PAIRS", 1)
    monkeypatch.setattr(p3w4, "EXPECTED_F2_PAIRS", 1)
    monkeypatch.setattr(p3w4, "EXPECTED_AFFECTED_PAIRS", 2)
    f1_none = source_row("f1", "none", "SUPPORT", "A approved B.")
    f1_pol = source_row("f1", "polarity_flip", "REFUTE", "A did not approved B.")
    f2_none = source_row("f2", "none", "REFUTE", "A did not approved B.")
    f2_para = source_row("f2", "paraphrase", "REFUTE", "B was not approved by A.")
    f2_pol = source_row("f2", "polarity_flip", "SUPPORT", "A approved B.")
    rows = [f1_none, f1_pol, f2_none, f2_para, f2_pol]
    sidecars = {
        f1_none["id"]: sidecar(f1_none, eligible=True),
        f1_pol["id"]: sidecar(f1_pol, grammar="FAIL", codes=["DID_NOT_INFLECTED_PREDICATE", "GRAMMAR_TEMPLATE_FAIL"]),
        f2_none["id"]: sidecar(f2_none, grammar="FAIL", codes=["DID_NOT_INFLECTED_PREDICATE", "GRAMMAR_TEMPLATE_FAIL"]),
        f2_para["id"]: sidecar(
            f2_para,
            grammar="FAIL",
            canonical="UNRESOLVED",
            codes=["CANONICAL_ROW_KNOWN_GENERATOR_DEFECT", "DID_NOT_INFLECTED_PREDICATE", "GRAMMAR_TEMPLATE_FAIL"],
        ),
        f2_pol["id"]: sidecar(f2_pol, canonical="UNRESOLVED", codes=["CANONICAL_ROW_KNOWN_GENERATOR_DEFECT"]),
    }
    p3rows = [
        {"row_id": f1_pol["id"], "pair_id": "f1", "intervention_type": "polarity_flip", "final_label": "REFUTE", "canonical_row_id": f1_none["id"], "canonical_counterpart_row_id": f1_none["id"], "canonical_counterpart_final_label": "SUPPORT", "canonical_counterpart_eligibility": True, "ordered_exclusion_codes": ["P2_GENERATOR_STATUS_DEFECT"], "generator_evidence_class": "INDEPENDENT_GENERATOR_OR_SEMANTIC_DEFECT_EVIDENCE", "generator_source_sha256": "gen", "integrity_builder_sha256": "builder"},
        {"row_id": f2_none["id"], "pair_id": "f2", "intervention_type": "none", "final_label": "REFUTE", "canonical_row_id": f2_none["id"], "canonical_counterpart_row_id": f2_none["id"], "canonical_counterpart_final_label": "REFUTE", "canonical_counterpart_eligibility": False, "ordered_exclusion_codes": ["P2_GENERATOR_STATUS_DEFECT"], "generator_evidence_class": "INDEPENDENT_GENERATOR_OR_SEMANTIC_DEFECT_EVIDENCE", "generator_source_sha256": "gen", "integrity_builder_sha256": "builder"},
        {"row_id": f2_para["id"], "pair_id": "f2", "intervention_type": "paraphrase", "final_label": "REFUTE", "canonical_row_id": f2_none["id"], "canonical_counterpart_row_id": f2_none["id"], "canonical_counterpart_final_label": "REFUTE", "canonical_counterpart_eligibility": False, "ordered_exclusion_codes": ["P2_INTEGRITY_SOURCE_REQUIRED"], "generator_evidence_class": "AMBIGUOUS_INTEGRITY_EVIDENCE", "generator_source_sha256": "gen", "integrity_builder_sha256": "builder"},
    ]
    return rows, sidecars, p3rows



def test_analyzer_ast_parse_syntax_passes():
    ast.parse(Path(p3w4.__file__).read_text(encoding="utf-8"))
def test_exact_f1_reconstruction(monkeypatch):
    rows, sidecars, p3rows = tiny_family(monkeypatch)
    f1, _f2 = p3w4.reconstruct_families(rows, sidecars, p3rows, {"u8_support_row_ids": {"f1__none"}})
    assert set(f1) == {"f1"}


def test_exact_f2_reconstruction(monkeypatch):
    rows, sidecars, p3rows = tiny_family(monkeypatch)
    _f1, f2 = p3w4.reconstruct_families(rows, sidecars, p3rows, {"u8_support_row_ids": {"f1__none"}})
    assert set(f2) == {"f2"}


def test_f1_f2_sets_are_disjoint(monkeypatch):
    rows, sidecars, p3rows = tiny_family(monkeypatch)
    f1, f2 = p3w4.reconstruct_families(rows, sidecars, p3rows, {"u8_support_row_ids": {"f1__none"}})
    assert set(f1).isdisjoint(f2)


def test_unexpected_family_structure_fails_closed(monkeypatch):
    rows, sidecars, p3rows = tiny_family(monkeypatch)
    rows[0]["final_label"] = "REFUTE"
    with pytest.raises(ValueError, match="P3W4_UNEXPECTED_INTERVENTION_COMPOSITION"):
        p3w4.reconstruct_families(rows, sidecars, p3rows, {"u8_support_row_ids": {"f1__none"}})


def test_missing_canonical_row_fails_closed(monkeypatch):
    rows, sidecars, p3rows = tiny_family(monkeypatch)
    with pytest.raises(ValueError, match="P3W4_MISSING_PAIR_MEMBER"):
        p3w4.reconstruct_families(rows[1:], sidecars, p3rows, {"u8_support_row_ids": {"f1__none"}})


def test_wrong_canonical_lineage_fails_closed(monkeypatch):
    rows, sidecars, p3rows = tiny_family(monkeypatch)
    sidecars["f2__polarity_flip"]["canonical_row_id"] = "wrong"
    with pytest.raises(ValueError, match="P3W4_CANONICAL_LINEAGE_INCONSISTENCY"):
        p3w4.reconstruct_families(rows, sidecars, p3rows, {"u8_support_row_ids": {"f1__none"}})


def test_f2_paraphrase_without_defective_canonical_fails_closed(monkeypatch):
    rows, sidecars, p3rows = tiny_family(monkeypatch)
    sidecars["f2__none"]["grammar_status"] = "PASS"
    with pytest.raises(ValueError, match="P3W4_F2_CANONICAL_SIDECAR_CONTRACT_MISMATCH"):
        p3w4.reconstruct_families(rows, sidecars, p3rows, {"u8_support_row_ids": {"f1__none"}})


def test_row_suffix_alone_cannot_define_family(monkeypatch):
    rows, sidecars, p3rows = tiny_family(monkeypatch)
    p3rows[0]["ordered_exclusion_codes"] = ["P2_INTEGRITY_SOURCE_REQUIRED"]
    with pytest.raises(ValueError, match="P3W4_UNEXPECTED_INTERVENTION_COMPOSITION"):
        p3w4.reconstruct_families(rows, sidecars, p3rows, {"u8_support_row_ids": {"f1__none"}})


def test_exact_reason_code_combinations_preserved(monkeypatch):
    rows, sidecars, p3rows = tiny_family(monkeypatch)
    f1, f2 = p3w4.reconstruct_families(rows, sidecars, p3rows, {"u8_support_row_ids": {"f1__none"}})
    records = [p3w4.pair_record("f1", "F1", f1["f1"], sidecars), p3w4.pair_record("f2", "F2", f2["f2"], sidecars)]
    combos = p3w4.aggregate_pairs(records)["reason_code_combinations"]
    assert "DID_NOT_INFLECTED_PREDICATE|GRAMMAR_TEMPLATE_FAIL" in combos
    assert "CANONICAL_ROW_KNOWN_GENERATOR_DEFECT|DID_NOT_INFLECTED_PREDICATE|GRAMMAR_TEMPLATE_FAIL" in combos


def test_grammar_rule_provenance_resolver_finds_real_repository_functions():
    provenance = p3w4.resolve_grammar_rule_provenance(Path(__file__).resolve().parents[1])
    assert provenance["DID_NOT_INFLECTED_PREDICATE"]["function"] in {"audit_item", "build_sidecar"}
    assert provenance["GRAMMAR_TEMPLATE_FAIL"]["source_file"].endswith("build_stage185a_controlled_train_integrity_sidecar.py")


def test_missing_provenance_source_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="missing provenance source"):
        p3w4.resolve_grammar_rule_provenance(tmp_path)


def test_no_difference_text_classification():
    assert p3w4.text_diagnostics("A approved B.", "A approved B.")["pattern"] == "no text difference"


def test_negation_only_classification():
    assert p3w4.text_diagnostics("A approved B.", "A did not approved B.")["pattern"] == "negation-only difference"


def test_predicate_inflection_only_classification():
    assert p3w4.text_diagnostics("A approve B.", "A approved B.")["pattern"] == "predicate-inflection-only difference"


def test_insertion_deletion_classification():
    assert p3w4.text_diagnostics("A approved B.", "A quickly approved B.")["pattern"] == "token insertion"
    assert p3w4.text_diagnostics("A quickly approved B.", "A approved B.")["pattern"] == "token deletion"


def test_multi_change_classification():
    assert p3w4.text_diagnostics("A approved B in Seoul.", "C rejected D in Busan.")["pattern"] == "multiple changes"


def test_deterministic_text_diagnostics():
    first = p3w4.text_diagnostics("A approved B.", "A did not approved B.")
    second = p3w4.text_diagnostics("A approved B.", "A did not approved B.")
    assert first == second


def test_manual_review_csv_contains_one_f2_row_per_pair(monkeypatch, tmp_path):
    rows, sidecars, p3rows = tiny_family(monkeypatch)
    f1, f2 = p3w4.reconstruct_families(rows, sidecars, p3rows, {"u8_support_row_ids": {"f1__none"}})
    records = [p3w4.pair_record("f1", "F1", f1["f1"], sidecars), p3w4.pair_record("f2", "F2", f2["f2"], sidecars)]
    path = tmp_path / "review.csv"
    p3w4.write_review_csv(path, records)
    with path.open(newline="", encoding="utf-8") as handle:
        loaded = list(csv.DictReader(handle))
    assert len(loaded) == 1
    assert loaded[0]["pair_id"] == "f2"


def test_human_fields_remain_empty(monkeypatch, tmp_path):
    rows, sidecars, p3rows = tiny_family(monkeypatch)
    _f1, f2 = p3w4.reconstruct_families(rows, sidecars, p3rows, {"u8_support_row_ids": {"f1__none"}})
    path = tmp_path / "review.csv"
    p3w4.write_review_csv(path, [p3w4.pair_record("f2", "F2", f2["f2"], sidecars)])
    loaded = list(csv.DictReader(path.open(newline="", encoding="utf-8")))[0]
    assert all(loaded[column] == "" for column in p3w4.REVIEW_COLUMNS if column.startswith("human_"))


def test_pair_text_is_not_truncated(monkeypatch):
    rows, sidecars, p3rows = tiny_family(monkeypatch)
    rows[2]["evidence"] = "x" * 5000
    _f1, f2 = p3w4.reconstruct_families(rows, sidecars, p3rows, {"u8_support_row_ids": {"f1__none"}})
    record = p3w4.pair_record("f2", "F2", f2["f2"], sidecars)
    assert len(record["members"]["canonical"]["source_row"]["evidence"]) == 5000


def test_f2_propagation_graph_exactness(monkeypatch):
    rows, sidecars, p3rows = tiny_family(monkeypatch)
    _f1, f2 = p3w4.reconstruct_families(rows, sidecars, p3rows, {"u8_support_row_ids": {"f1__none"}})
    record = p3w4.pair_record("f2", "F2", f2["f2"], sidecars)
    graph = p3w4.aggregate_pairs([record])["f2_propagation_patterns"]
    assert len(graph) == 1
    assert "paraphrase:UNRESOLVED" in next(iter(graph))


def test_scenario_r0_does_not_mutate_production_state():
    assert p3w4.scenario_diagnostics([])["R0"]["diagnostic_only"] is True
    assert p3w4.EXECUTION_ISOLATION["production_behavior_modified"] is False


def test_r1_r5_remain_diagnostic():
    scenarios = p3w4.scenario_diagnostics([])
    assert all(value["diagnostic_only"] for name, value in scenarios.items() if name in {"R1", "R2", "R3", "R4", "R5"})


def test_minimum_count_readiness_calculation_is_exact():
    readiness = p3w4.potential_authority_yield([])["minimum_50_count_readiness_by_scenario"]
    assert readiness["R0"] is False
    assert readiness["R1"] is False


def test_automatic_recovery_requires_explicit_deterministic_proof(monkeypatch):
    rows, sidecars, p3rows = tiny_family(monkeypatch)
    _f1, f2 = p3w4.reconstruct_families(rows, sidecars, p3rows, {"u8_support_row_ids": {"f1__none"}})
    record = p3w4.pair_record("f2", "F2", f2["f2"], sidecars)
    assert record["automatic_root_cause_class"] != "F2_CANONICAL_REFUTE_SEMANTICALLY_VALID_SURFACE_DEFECT"


def test_ambiguous_rows_route_to_review():
    cls, _evidence = p3w4.classify_pair("F1", {"polarity_flip": {"id": "x"}}, {"x": {"grammar_status": "PASS", "reason_codes": []}})
    assert cls == "F1_AMBIGUOUS_REQUIRES_REVIEW"


def test_semantic_conflict_cannot_be_marked_recoverable(monkeypatch):
    rows, sidecars, p3rows = tiny_family(monkeypatch)
    rows[2]["final_label"] = "SUPPORT"
    with pytest.raises(ValueError, match="P3W4_UNEXPECTED_INTERVENTION_COMPOSITION"):
        p3w4.reconstruct_families(rows, sidecars, p3rows, {"u8_support_row_ids": {"f1__none"}})


def test_p3w3_artifact_sha_mismatch_fails_closed(monkeypatch, tmp_path):
    artifact = tmp_path / "summary.json"
    artifact.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(p3w4, "file_sha256", lambda path: "actual")
    with pytest.raises(ValueError, match="P3-W3 artifact SHA mismatch"):
        p3w4.verify_artifact_sha(artifact, "expected")


def test_p3w3_family_count_mismatch_fails_closed(monkeypatch):
    rows, sidecars, p3rows = tiny_family(monkeypatch)
    monkeypatch.setattr(p3w4, "EXPECTED_F1_PAIRS", 2)
    with pytest.raises(ValueError, match="P3-W3 pair-family count mismatch"):
        p3w4.reconstruct_families(rows, sidecars, p3rows, {"u8_support_row_ids": {"f1__none"}})


def test_data_sidecar_split_commit_dirty_tree_gates_fail_closed(tmp_path):
    row = source_row("p", "none", "SUPPORT")
    p3w4.validate_source_rows([row], {"none"})
    with pytest.raises(ValueError, match="split/count/identity"):
        p3w4.validate_split_contract([row], 174, 0.2, 999, "wrong")
    with pytest.raises(ValueError, match="sidecar/source identity"):
        p3w4.validate_sidecar([row], [sidecar(row) | {"source_dataset_sha256": "wrong"}], "sha")


def test_no_model_tokenizer_forward_backward_training_or_external_api_path():
    source = Path(p3w4.__file__).read_text(encoding="utf-8")
    forbidden = ["import torch", "AutoTokenizer", "AutoModel", ".forward(", ".backward(", "optimizer.step", "requests.", "urllib.request"]
    assert not any(token in source for token in forbidden)
    assert all(value is False for value in p3w4.EXECUTION_ISOLATION.values())


def test_required_cli_arguments_are_registered():
    parser = p3w4.build_arg_parser()
    actions = {action.dest for action in parser._actions}
    for option in [
        "data",
        "controlled_integrity_sidecar_path",
        "p3w3_summary_json",
        "p3w3_refute_jsonl",
        "expected_data_sha256",
        "expected_sidecar_semantic_sha256",
        "expected_p3w3_summary_sha256",
        "expected_p3w3_refute_jsonl_sha256",
        "split_seed",
        "dev_ratio",
        "expected_train_row_count",
        "expected_train_row_identity_hash",
        "expected_p3w3_execution_commit",
        "execution_commit",
        "output_json",
        "output_pair_jsonl",
        "output_review_csv",
    ]:
        assert option in actions


def test_pair_jsonl_round_trips(monkeypatch, tmp_path):
    rows, sidecars, p3rows = tiny_family(monkeypatch)
    f1, _f2 = p3w4.reconstruct_families(rows, sidecars, p3rows, {"u8_support_row_ids": {"f1__none"}})
    record = p3w4.pair_record("f1", "F1", f1["f1"], sidecars)
    path = tmp_path / "pairs.jsonl"
    p3w4.write_pair_jsonl(path, [record])
    loaded = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert loaded[0]["pair_id"] == "f1"


def strict_summary() -> dict:
    return {
        "schema_version": p3w4.P3W3_SCHEMA_VERSION,
        "status": p3w4.P3W3_STATUS,
        "decision": p3w4.P3W3_DECISION,
        "execution_commit": "8a587a6f28a84a01237d81a47898ec4d5597ffc4",
        "refute_row_count_exported": 359,
        "pair_level_canonical_comparison": {
            "refute_row_count": 359,
            "unique_refute_pair_count": 240,
            "multi_refute_row_pair_count": 119,
        },
        "final_label_overview": {
            "eligible_REFUTE_polarity_targets": 0,
            "eligible_SUPPORT_polarity_targets": 242,
        },
        "generator_evidence_class_counts": {
            "AMBIGUOUS_INTEGRITY_EVIDENCE": 119,
            "INDEPENDENT_GENERATOR_OR_SEMANTIC_DEFECT_EVIDENCE": 240,
        },
        "sidecar_semantic_interpretation_audit": {
            "generator_evidence_proof_contract_available": False,
        },
        "counterfactual_eligibility_results": {"C5": {"newly_admitted_row_count": 0}},
        "candidate_universe_counts": {
            "U8_final_polarity_applicable_rows": {
                "REFUTE": 0,
                "SUPPORT": 242,
                "row_ids": ["f1__none", *[f"u8_support_{i}" for i in range(241)]],
            }
        },
        "A1_A3_released": False,
        "polarity_supervision_released": False,
    }


def p3_refute_row(row_id: str, pair_id: str, intervention: str, final: str, counterpart: str, counterpart_final: str, counterpart_eligible: bool, exclusion: str, evidence_class: str) -> dict:
    return {
        "row_id": row_id,
        "pair_id": pair_id,
        "intervention_type": intervention,
        "final_label": final,
        "canonical_row_id": f"{pair_id}__none",
        "canonical_counterpart_row_id": counterpart,
        "canonical_counterpart_final_label": counterpart_final,
        "canonical_counterpart_eligibility": counterpart_eligible,
        "ordered_exclusion_codes": [exclusion],
        "generator_evidence_class": evidence_class,
        "generator_source_sha256": "gen",
        "integrity_builder_sha256": "builder",
    }


def refute_rows_for_partition():
    rows = []
    for i in range(121):
        rows.append(p3_refute_row(
            f"f1_{i}__polarity_flip",
            f"f1_{i}",
            "polarity_flip",
            "REFUTE",
            f"f1_{i}__none",
            "SUPPORT",
            True,
            "P2_GENERATOR_STATUS_DEFECT",
            "INDEPENDENT_GENERATOR_OR_SEMANTIC_DEFECT_EVIDENCE",
        ))
    for i in range(119):
        rows.append(p3_refute_row(
            f"f2_{i}__none",
            f"f2_{i}",
            "none",
            "REFUTE",
            f"f2_{i}__none",
            "REFUTE",
            False,
            "P2_GENERATOR_STATUS_DEFECT",
            "INDEPENDENT_GENERATOR_OR_SEMANTIC_DEFECT_EVIDENCE",
        ))
        rows.append(p3_refute_row(
            f"f2_{i}__paraphrase",
            f"f2_{i}",
            "paraphrase",
            "REFUTE",
            f"f2_{i}__none",
            "REFUTE",
            False,
            "P2_INTEGRITY_SOURCE_REQUIRED",
            "AMBIGUOUS_INTEGRITY_EVIDENCE",
        ))
    return rows

def test_p3w3_commit_may_differ_from_p3w4_head(monkeypatch):
    monkeypatch.setattr(p3w4, "git_output", lambda args, root: "new_head" if args[:2] == ["rev-parse", "HEAD"] else "")
    assert p3w4.verify_git_gates(Path("."), "new_head")["git_head"] == "new_head"


def test_current_head_mismatch_rejects(monkeypatch):
    monkeypatch.setattr(p3w4, "git_output", lambda args, root: "actual" if args[:2] == ["rev-parse", "HEAD"] else "")
    with pytest.raises(ValueError, match="current execution commit mismatch"):
        p3w4.verify_git_gates(Path("."), "expected")


def test_dirty_tracked_tree_rejects(monkeypatch):
    def fake(args, root):
        return "head" if args[:2] == ["rev-parse", "HEAD"] else " M tracked.py"
    monkeypatch.setattr(p3w4, "git_output", fake)
    with pytest.raises(ValueError, match="dirty tracked tree"):
        p3w4.verify_git_gates(Path("."), "head")


def test_strict_summary_schema_status_decision_validation():
    summary = strict_summary()
    summary["status"] = "BAD"
    with pytest.raises(ValueError, match="wrong schema/status/decision"):
        p3w4.validate_p3w3_artifacts(summary, refute_rows_for_partition(), "8a587a6f28a84a01237d81a47898ec4d5597ffc4")


def test_missing_summary_commit_rejects():
    summary = strict_summary()
    del summary["execution_commit"]
    with pytest.raises(ValueError, match="missing P3-W3 summary execution_commit"):
        p3w4.validate_p3w3_artifacts(summary, refute_rows_for_partition(), "8a587a6f28a84a01237d81a47898ec4d5597ffc4")


def test_wrong_summary_commit_rejects():
    summary = strict_summary()
    summary["execution_commit"] = "0" * 40
    with pytest.raises(ValueError, match="P3-W3 summary commit mismatch"):
        p3w4.validate_p3w3_artifacts(summary, refute_rows_for_partition(), "8a587a6f28a84a01237d81a47898ec4d5597ffc4")


def test_wrong_evidence_counts_reject():
    summary = strict_summary()
    summary["generator_evidence_class_counts"]["AMBIGUOUS_INTEGRITY_EVIDENCE"] = 118
    with pytest.raises(ValueError, match="wrong evidence counts"):
        p3w4.validate_p3w3_artifacts(summary, refute_rows_for_partition(), "8a587a6f28a84a01237d81a47898ec4d5597ffc4")


def test_exact_359_row_family_partition():
    authority = p3w4.validate_p3w3_artifacts(strict_summary(), refute_rows_for_partition(), "8a587a6f28a84a01237d81a47898ec4d5597ffc4")
    assert len(authority["u8_support_row_ids"]) == 242
    assert "f1__none" in authority["u8_support_row_ids"]


def test_pair_with_three_refute_rows_rejects():
    rows = refute_rows_for_partition()
    rows.append({"row_id": "extra", "pair_id": "f2_0"})
    with pytest.raises(ValueError, match="P3-W3 row count mismatch"):
        p3w4.validate_p3w3_artifacts(strict_summary(), rows, "8a587a6f28a84a01237d81a47898ec4d5597ffc4")


def test_duplicate_refute_row_id_rejects():
    rows = refute_rows_for_partition()
    rows[1]["row_id"] = rows[0]["row_id"]
    with pytest.raises(ValueError, match="duplicate REFUTE row ID"):
        p3w4.validate_p3w3_artifacts(strict_summary(), rows, "8a587a6f28a84a01237d81a47898ec4d5597ffc4")


def test_sidecar_order_independence(monkeypatch):
    rows, sidecars, _ = tiny_family(monkeypatch)
    sidecar_rows = list(reversed(list(sidecars.values())))
    assert set(p3w4.validate_sidecar(rows, sidecar_rows, "sha")) == {row["id"] for row in rows}


def test_duplicate_sidecar_row_id_rejects(monkeypatch):
    rows, sidecars, _ = tiny_family(monkeypatch)
    sc = list(sidecars.values())
    sc[1] = dict(sc[0])
    with pytest.raises(ValueError, match="duplicate sidecar row ID"):
        p3w4.validate_sidecar(rows, sc, "sha")


def test_duplicate_pair_intervention_rejection(monkeypatch):
    rows, _sidecars, _ = tiny_family(monkeypatch)
    rows.append(dict(rows[0], id="duplicate_none"))
    with pytest.raises(ValueError, match="P3W4_DUPLICATE_PAIR_INTERVENTION_MEMBER"):
        p3w4.validate_source_rows(rows, {"none", "paraphrase", "polarity_flip"})


def test_independent_grammar_rule_reproduction():
    member = {"source_row": {"evidence": "A did not approved B."}, "sidecar": {"reason_codes": ["DID_NOT_INFLECTED_PREDICATE", "GRAMMAR_TEMPLATE_FAIL"]}}
    validator = validator_fixture(lambda row, fact: True)
    proof = p3w4.reproduce_grammar_rule(member, {"pair_id": "p", "predicate": "approved", "alternate_predicate": "rejected"}, validator, True)
    assert proof["production_rule_reproduction_result"] is True
    assert proof["matched_surface_span"] == "did not approved"

def test_missing_fact_authority_routes_ambiguous(monkeypatch):
    rows, sidecars, p3rows = tiny_family(monkeypatch)
    f1, _ = p3w4.reconstruct_families(rows, sidecars, p3rows, {"u8_support_row_ids": {"f1__none"}})
    record = p3w4.pair_record("f1", "F1", f1["f1"], sidecars, {}, {"DID_NOT_INFLECTED_PREDICATE": {}})
    assert record["automatic_root_cause_class"] == "F1_AMBIGUOUS_REQUIRES_REVIEW"


def test_claim_and_evidence_diagnostics_separate(monkeypatch):
    rows, sidecars, p3rows = tiny_family(monkeypatch)
    f1, _ = p3w4.reconstruct_families(rows, sidecars, p3rows, {"u8_support_row_ids": {"f1__none"}})
    record = p3w4.pair_record("f1", "F1", f1["f1"], sidecars)
    member = record["members"]["polarity_flip"]
    assert "claim_diagnostics_vs_canonical_claim" in member
    assert "evidence_diagnostics_vs_canonical_evidence" in member


def test_review_csv_contains_all_three_evidence_texts(monkeypatch, tmp_path):
    rows, sidecars, p3rows = tiny_family(monkeypatch)
    _f1, f2 = p3w4.reconstruct_families(rows, sidecars, p3rows, {"u8_support_row_ids": {"f1__none"}})
    path = tmp_path / "review.csv"
    p3w4.write_review_csv(path, [p3w4.pair_record("f2", "F2", f2["f2"], sidecars)])
    loaded = list(csv.DictReader(path.open(newline="", encoding="utf-8")))[0]
    assert loaded["canonical_evidence"] and loaded["paraphrase_evidence"] and loaded["polarity_flip_evidence"]


def test_dynamic_scenario_arithmetic(monkeypatch):
    rows, sidecars, p3rows = tiny_family(monkeypatch)
    f1, f2 = p3w4.reconstruct_families(rows, sidecars, p3rows, {"u8_support_row_ids": {"f1__none"}})
    records = [p3w4.pair_record("f1", "F1", f1["f1"], sidecars), p3w4.pair_record("f2", "F2", f2["f2"], sidecars)]
    scenarios = p3w4.scenario_diagnostics(records)
    assert scenarios["R2"]["conditional_REFUTE_contribution"] == 2
    assert scenarios["R2"]["conditional_SUPPORT_contribution"] == 1
    assert scenarios["R4"]["rows_requiring_regeneration"] == 4
    assert scenarios["R5"]["actually_available_new_REFUTE_rows"] == "unknown"
    assert scenarios["R3"]["potential_total_REFUTE"] == "unresolved"


def test_row_level_authority_yield(monkeypatch):
    rows, sidecars, p3rows = tiny_family(monkeypatch)
    _f1, f2 = p3w4.reconstruct_families(rows, sidecars, p3rows, {"u8_support_row_ids": {"f1__none"}})
    record = p3w4.pair_record("f2", "F2", f2["f2"], sidecars)
    yield_report = p3w4.potential_authority_yield([record])
    assert yield_report["manual_review_pair_count"] == 1
    assert yield_report["manual_review_REFUTE_row_count"] == 2


def test_fresh_output_namespace_and_existing_rejection(tmp_path):
    parent = tmp_path / "fresh"
    out = parent / "summary.json"
    pair = parent / "pairs.jsonl"
    csv_path = parent / "review.csv"
    assert p3w4.validate_output_namespace(out, pair, csv_path) == parent
    parent.mkdir()
    with pytest.raises(ValueError, match="output namespace already exists"):
        p3w4.validate_output_namespace(out, pair, csv_path)


def test_missing_refute_jsonl_field_rejects():
    rows = refute_rows_for_partition()
    del rows[0]["canonical_counterpart_row_id"]
    with pytest.raises(ValueError, match="missing REFUTE JSONL authority field"):
        p3w4.validate_p3w3_artifacts(strict_summary(), rows, "8a587a6f28a84a01237d81a47898ec4d5597ffc4")


def test_actual_u8_and_generator_evidence_paths():
    authority = p3w4.validate_p3w3_artifacts(strict_summary(), refute_rows_for_partition(), "8a587a6f28a84a01237d81a47898ec4d5597ffc4")
    assert len(authority["u8_support_row_ids"]) == 242
    assert "f1__none" in authority["u8_support_row_ids"]


def test_grammatical_did_not_base_form_not_falsely_marked_defect():
    member = {"source_row": {"evidence": "A did not approve B."}, "sidecar": {"reason_codes": ["DID_NOT_INFLECTED_PREDICATE", "GRAMMAR_TEMPLATE_FAIL"]}}
    validator = validator_fixture(lambda row, fact: False)
    proof = p3w4.reproduce_grammar_rule(member, {"pair_id": "p", "predicate": "approved", "alternate_predicate": "rejected"}, validator, True)
    assert proof["production_rule_reproduction_result"] is False


def test_f2_propagation_alone_remains_manual_review(monkeypatch):
    rows, sidecars, p3rows = tiny_family(monkeypatch)
    _f1, f2 = p3w4.reconstruct_families(rows, sidecars, p3rows, {"u8_support_row_ids": {"f1__none"}})
    record = p3w4.pair_record("f2", "F2", f2["f2"], sidecars)
    assert record["automatic_root_cause_class"] == "F2_CANONICAL_DEFECT_PROPAGATED_TO_DERIVATIVES"
    assert record["remediation_state"] == "MANUAL_REVIEW_REQUIRED"


def test_action_sets_disjoint_and_potential_separated(monkeypatch):
    rows, sidecars, p3rows = tiny_family(monkeypatch)
    f1, f2 = p3w4.reconstruct_families(rows, sidecars, p3rows, {"u8_support_row_ids": {"f1__none"}})
    records = [p3w4.pair_record("f1", "F1", f1["f1"], sidecars), p3w4.pair_record("f2", "F2", f2["f2"], sidecars)]
    r2 = p3w4.scenario_diagnostics(records)["R2"]
    action = set(r2["action_review_row_ids"]) | set(r2["action_regenerate_row_ids"]) | set(r2["action_exclude_row_ids"]) | set(r2["current_retained_row_ids"])
    assert len(action) == 4
    assert set(r2["conditional_potential_admitted_row_ids"]).issubset(action)


def test_baseline_and_total_scenario_counts(monkeypatch):
    rows, sidecars, p3rows = tiny_family(monkeypatch)
    f1, f2 = p3w4.reconstruct_families(rows, sidecars, p3rows, {"u8_support_row_ids": {"f1__none"}})
    records = [p3w4.pair_record("f1", "F1", f1["f1"], sidecars), p3w4.pair_record("f2", "F2", f2["f2"], sidecars)]
    scenarios = p3w4.scenario_diagnostics(records)
    assert scenarios["R2"]["potential_total_SUPPORT"] == 243
    assert scenarios["R2"]["potential_total_REFUTE"] == 2
    assert scenarios["R4"]["rows_requiring_regeneration"] == 4
    assert scenarios["R4"]["potential_total_SUPPORT"] == 243
    assert scenarios["R4"]["potential_total_REFUTE"] == 3
    assert scenarios["R3"]["confirmed_REFUTE_contribution"] == 0
    assert scenarios["R3"]["conditional_REFUTE_contribution_if_label_contract_preserved"] == 1
    assert scenarios["R3"]["polarity_minimum_50_could_be_met"] == "unresolved"


def test_exported_rows_359_vs_affected_members_478_namespaces():
    assert p3w4.EXPECTED_REFUTE_ROWS == 359
    assert p3w4.EXPECTED_AFFECTED_MEMBER_ROWS == 478


def test_runtime_schema_v3():
    assert p3w4.SCHEMA_VERSION == "reason_router_p3w4_canonical_grammar_authority_audit_v3"


def test_runtime_blockers_truthful_field_names():
    source = Path(p3w4.__file__).read_text(encoding="utf-8")
    runtime_tail = source[source.index('"remaining_blockers"'):]
    assert "P3W4_RESULT_STATIC_REVIEW_PENDING" in runtime_tail
    assert "P3W4_CANONICAL_GRAMMAR_AUDIT_NOT_EXECUTED" not in runtime_tail
    assert "P3W4_STATIC_REVIEW_PENDING" not in runtime_tail


def test_generator_source_sha_mismatch_reject(monkeypatch, tmp_path):
    monkeypatch.setattr(p3w4, "file_sha256", lambda path: "actual")
    with pytest.raises(ValueError, match="P3W4_GENERATOR_SOURCE_AUTHORITY_MISMATCH"):
        p3w4.verify_generator_source_authority(tmp_path, [{"generator_source_sha256": "other"}])


def test_complete_fact_authority_then_affected_filter(monkeypatch, tmp_path):
    class Module:
        INTERVENTION_TYPES = ["none"]
        @staticmethod
        def fact_templates_for_count(count):
            assert count == 3
            return [{"pair_id": "a"}, {"pair_id": "b"}, {"pair_id": "c"}]
    monkeypatch.setattr(p3w4, "load_intervention_authority", lambda root: {"module": Module, "values": frozenset({"none"}), "source": "s"})
    facts = p3w4.load_fact_authority(tmp_path, {"a", "b", "c"}, {"b"})
    assert set(facts) == {"b"}


def test_untracked_p3w3_artifact_rejected(monkeypatch, tmp_path):
    path = tmp_path / "artifact.json"
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(p3w4, "file_sha256", lambda p: "sha")
    def fake_git(args, root):
        if args[:2] == ["ls-files", "--error-unmatch"]:
            raise ValueError("untracked")
        return ""
    monkeypatch.setattr(p3w4, "git_output", fake_git)
    with pytest.raises(ValueError, match="P3W3_ARTIFACT_NOT_GIT_TRACKED"):
        p3w4.verify_tracked_artifact(path, tmp_path, "sha")


def test_external_repository_artifact_path_rejected(tmp_path):
    root = tmp_path / "root"
    other = tmp_path / "other" / "artifact.json"
    root.mkdir()
    other.parent.mkdir()
    other.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="external-repository artifact path rejected"):
        p3w4.relative_to_root(other, root)

def test_missing_intervention_final_or_counterpart_rejects():
    rows = refute_rows_for_partition()
    rows[0]["intervention_type"] = ""
    with pytest.raises(ValueError, match="missing REFUTE JSONL authority field"):
        p3w4.validate_p3w3_artifacts(strict_summary(), rows, "8a587a6f28a84a01237d81a47898ec4d5597ffc4")

    rows = refute_rows_for_partition()
    rows[0]["final_label"] = ""
    with pytest.raises(ValueError, match="missing REFUTE JSONL authority field"):
        p3w4.validate_p3w3_artifacts(strict_summary(), rows, "8a587a6f28a84a01237d81a47898ec4d5597ffc4")

    rows = refute_rows_for_partition()
    rows[0]["canonical_counterpart_final_label"] = ""
    with pytest.raises(ValueError, match="missing REFUTE JSONL authority field"):
        p3w4.validate_p3w3_artifacts(strict_summary(), rows, "8a587a6f28a84a01237d81a47898ec4d5597ffc4")


def test_refute_jsonl_contract_mismatch_rejects():
    rows = refute_rows_for_partition()
    rows[0]["canonical_counterpart_final_label"] = "REFUTE"
    with pytest.raises(ValueError, match="P3-W3 exact REFUTE partition mismatch"):
        p3w4.validate_p3w3_artifacts(strict_summary(), rows, "8a587a6f28a84a01237d81a47898ec4d5597ffc4")


def observed_scenario_records() -> list[dict]:
    records = []
    for i in range(121):
        records.append({
            "family": "F1",
            "members": {"polarity_flip": {"source_row": {"id": f"f1_{i}__polarity_flip", "final_label": "REFUTE"}}},
        })
    for i in range(119):
        records.append({
            "family": "F2",
            "members": {
                "canonical": {"source_row": {"id": f"f2_{i}__none", "final_label": "REFUTE"}},
                "paraphrase": {"source_row": {"id": f"f2_{i}__paraphrase", "final_label": "REFUTE"}},
                "polarity_flip": {"source_row": {"id": f"f2_{i}__polarity_flip", "final_label": "SUPPORT"}},
            },
        })
    return records


def test_observed_r2_and_r4_totals_from_member_records():
    scenarios = p3w4.scenario_diagnostics(observed_scenario_records())
    assert scenarios["R2"]["conditional_REFUTE_contribution"] == 238
    assert scenarios["R2"]["conditional_SUPPORT_contribution"] == 119
    assert scenarios["R2"]["potential_total_SUPPORT"] == 361
    assert scenarios["R2"]["potential_total_REFUTE"] == 238
    assert scenarios["R4"]["rows_requiring_regeneration"] == 478
    assert scenarios["R4"]["potential_total_SUPPORT"] == 361
    assert scenarios["R4"]["potential_total_REFUTE"] == 359
    assert scenarios["R3"]["confirmed_REFUTE_contribution"] == 0
    assert scenarios["R3"]["conditional_REFUTE_contribution_if_label_contract_preserved"] == 119
    assert scenarios["R3"]["polarity_minimum_50_could_be_met"] == "unresolved"

def test_repository_internal_file_wrong_sha_reaches_sha_gate(monkeypatch, tmp_path):
    path = tmp_path / "artifact.json"
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(p3w4, "file_sha256", lambda p: "actual")
    with pytest.raises(ValueError, match="P3-W3 artifact SHA mismatch"):
        p3w4.verify_tracked_artifact(path, tmp_path, "expected")


def test_tracked_but_absent_at_head_rejected(monkeypatch, tmp_path):
    path = tmp_path / "artifact.json"
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(p3w4, "file_sha256", lambda p: "sha")
    def fake_git(args, root):
        if args[:2] == ["cat-file", "-e"]:
            raise ValueError("absent")
        return ""
    monkeypatch.setattr(p3w4, "git_output", fake_git)
    with pytest.raises(ValueError, match="P3W3_ARTIFACT_NOT_HEAD_ADDRESSABLE"):
        p3w4.verify_tracked_artifact(path, tmp_path, "sha")


def test_u8_duplicate_missing_and_overlap_reject():
    summary = strict_summary()
    summary["candidate_universe_counts"]["U8_final_polarity_applicable_rows"]["row_ids"] = ["dup"] * 242
    with pytest.raises(ValueError, match="wrong evidence counts"):
        p3w4.validate_p3w3_artifacts(summary, refute_rows_for_partition(), "8a587a6f28a84a01237d81a47898ec4d5597ffc4")

    summary = strict_summary()
    summary["candidate_universe_counts"]["U8_final_polarity_applicable_rows"]["row_ids"] = [f"u8_{i}" for i in range(241)]
    with pytest.raises(ValueError, match="wrong evidence counts"):
        p3w4.validate_p3w3_artifacts(summary, refute_rows_for_partition(), "8a587a6f28a84a01237d81a47898ec4d5597ffc4")

    summary = strict_summary()
    ids = summary["candidate_universe_counts"]["U8_final_polarity_applicable_rows"]["row_ids"]
    ids[0] = "f1_0__polarity_flip"
    with pytest.raises(ValueError, match="wrong evidence counts"):
        p3w4.validate_p3w3_artifacts(summary, refute_rows_for_partition(), "8a587a6f28a84a01237d81a47898ec4d5597ffc4")


def test_per_row_evidence_class_contracts_reject():
    rows = refute_rows_for_partition()
    rows[0]["generator_evidence_class"] = "AMBIGUOUS_INTEGRITY_EVIDENCE"
    with pytest.raises(ValueError, match="P3-W3 exact REFUTE partition mismatch"):
        p3w4.validate_p3w3_artifacts(strict_summary(), rows, "8a587a6f28a84a01237d81a47898ec4d5597ffc4")

    rows = refute_rows_for_partition()
    f2_none = next(row for row in rows if row["intervention_type"] == "none")
    f2_none["generator_evidence_class"] = "AMBIGUOUS_INTEGRITY_EVIDENCE"
    with pytest.raises(ValueError, match="P3-W3 exact REFUTE partition mismatch"):
        p3w4.validate_p3w3_artifacts(strict_summary(), rows, "8a587a6f28a84a01237d81a47898ec4d5597ffc4")

    rows = refute_rows_for_partition()
    f2_para = next(row for row in rows if row["intervention_type"] == "paraphrase")
    f2_para["generator_evidence_class"] = "INDEPENDENT_GENERATOR_OR_SEMANTIC_DEFECT_EVIDENCE"
    with pytest.raises(ValueError, match="P3-W3 exact REFUTE partition mismatch"):
        p3w4.validate_p3w3_artifacts(strict_summary(), rows, "8a587a6f28a84a01237d81a47898ec4d5597ffc4")


def test_canonical_row_id_mismatch_rejects():
    rows = refute_rows_for_partition()
    rows[0]["canonical_row_id"] = "wrong"
    with pytest.raises(ValueError, match="P3-W3 exact REFUTE partition mismatch"):
        p3w4.validate_p3w3_artifacts(strict_summary(), rows, "8a587a6f28a84a01237d81a47898ec4d5597ffc4")


def test_row_derived_evidence_count_mismatch_rejects():
    summary = strict_summary()
    summary["generator_evidence_class_counts"]["INDEPENDENT_GENERATOR_OR_SEMANTIC_DEFECT_EVIDENCE"] = 239
    with pytest.raises(ValueError, match="wrong evidence counts"):
        p3w4.validate_p3w3_artifacts(summary, refute_rows_for_partition(), "8a587a6f28a84a01237d81a47898ec4d5597ffc4")


def test_integrity_builder_sha_mismatch_reject(monkeypatch, tmp_path):
    monkeypatch.setattr(p3w4, "file_sha256", lambda path: "actual")
    with pytest.raises(ValueError, match="P3W4_INTEGRITY_BUILDER_AUTHORITY_MISMATCH"):
        p3w4.verify_integrity_builder_authority(tmp_path, [{"integrity_builder_sha256": "other"}])


def test_source_git_blob_mismatch_rejects(monkeypatch):
    def fake_git(args, root):
        if args[0] == "rev-parse" and args[1].startswith("p3w3:"):
            return "old"
        if args[0] == "rev-parse" and args[1].startswith("HEAD:"):
            return "new"
        return ""
    monkeypatch.setattr(p3w4, "git_output", fake_git)
    with pytest.raises(ValueError, match="P3W4_SOURCE_BLOB_IDENTITY_MISMATCH"):
        p3w4.verify_source_blob_identities(Path("."), "p3w3")


def test_grammar_validator_call_chain_verification(monkeypatch, tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "build_stage185a_controlled_train_integrity_sidecar.py").write_text("from analyze_stage182a_controlled_intervention_integrity import grammar_anomaly\ndef build_sidecar(row, fact):\n    return grammar_anomaly(row, fact)\n", encoding="utf-8")
    (scripts / "analyze_stage182a_controlled_intervention_integrity.py").write_text("def grammar_anomaly(row, fact):\n    return True\n", encoding="utf-8")
    stage185 = types.SimpleNamespace()
    stage182 = types.SimpleNamespace(grammar_anomaly=lambda row, fact: True)
    def fake_load(path, name):
        return stage185 if "stage185" in name else stage182
    monkeypatch.setattr(p3w4, "load_module_from_path", fake_load)
    monkeypatch.setattr(p3w4, "file_sha256", lambda path: "a" * 64)
    monkeypatch.setattr(p3w4, "callable_source_relative_path", lambda validator, root: "scripts/analyze_stage182a_controlled_intervention_integrity.py")
    validator = p3w4.load_production_grammar_validator(tmp_path)
    assert validator["validator"] is stage182.grammar_anomaly
    assert validator["validator_call_chain_verified"] is True


def test_builder_local_validator_used(monkeypatch, tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "build_stage185a_controlled_train_integrity_sidecar.py").write_text("def grammar_anomaly(row, fact):\n    return True\ndef build_sidecar(row, fact):\n    return grammar_anomaly(row, fact)\n", encoding="utf-8")
    (scripts / "analyze_stage182a_controlled_intervention_integrity.py").write_text("", encoding="utf-8")
    local = lambda row, fact: True
    monkeypatch.setattr(p3w4, "load_module_from_path", lambda path, name: types.SimpleNamespace(grammar_anomaly=local))
    monkeypatch.setattr(p3w4, "file_sha256", lambda path: "a" * 64)
    monkeypatch.setattr(p3w4, "callable_source_relative_path", lambda validator, root: "scripts/build_stage185a_controlled_train_integrity_sidecar.py")
    validator = p3w4.load_production_grammar_validator(tmp_path)
    assert validator["validator"] is local
    assert validator["validator_authority_source"] == "scripts/build_stage185a_controlled_train_integrity_sidecar.py"


def test_unverified_call_chain_keeps_f1_ambiguous(monkeypatch, tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "build_stage185a_controlled_train_integrity_sidecar.py").write_text("# no validator\n", encoding="utf-8")
    (scripts / "analyze_stage182a_controlled_intervention_integrity.py").write_text("def grammar_anomaly(row, fact):\n    return True\n", encoding="utf-8")
    monkeypatch.setattr(p3w4, "load_module_from_path", lambda path, name: types.SimpleNamespace())
    validator = p3w4.load_production_grammar_validator(tmp_path)
    proof = {"sidecar_rule_claimed_failure": True, "generator_source_sha_matches": True, "validator_call_chain_verified": validator["validator_call_chain_verified"], "validator_source_path": None, "fact_pair_id": "p", "production_rule_reproduction_result": True, "exact_row_evidence_supplied": True}
    cls, _ = p3w4.classify_pair("F1", {"polarity_flip": {"id": "x"}}, {"x": {}}, proof)
    assert cls == "F1_AMBIGUOUS_REQUIRES_REVIEW"


def test_wrong_callable_signature_fails_closed(monkeypatch, tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "build_stage185a_controlled_train_integrity_sidecar.py").write_text("def grammar_anomaly():\n    return True\ndef build_sidecar(row, fact):\n    return grammar_anomaly(row, fact)\n", encoding="utf-8")
    (scripts / "analyze_stage182a_controlled_intervention_integrity.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(p3w4, "load_module_from_path", lambda path, name: types.SimpleNamespace(grammar_anomaly=lambda: True))
    monkeypatch.setattr(p3w4, "callable_source_relative_path", lambda validator, root: "scripts/build_stage185a_controlled_train_integrity_sidecar.py")
    with pytest.raises(ValueError, match="wrong callable signature"):
        p3w4.load_production_grammar_validator(tmp_path)


def test_complete_fact_set_exact_equality(monkeypatch, tmp_path):
    class Module:
        INTERVENTION_TYPES = ["none"]
        @staticmethod
        def fact_templates_for_count(count):
            return [{"pair_id": "a"}, {"pair_id": "b"}]
    monkeypatch.setattr(p3w4, "load_intervention_authority", lambda root: {"module": Module, "values": frozenset({"none"}), "source": "s"})
    assert set(p3w4.load_fact_authority(tmp_path, {"a", "b"}, {"a"})) == {"a"}

    with pytest.raises(ValueError, match="complete fact authority identity mismatch"):
        p3w4.load_fact_authority(tmp_path, {"a", "b", "c"}, {"a"})

    with pytest.raises(ValueError, match="complete fact authority identity mismatch"):
        p3w4.load_fact_authority(tmp_path, {"a"}, {"a"})


def test_f2_remediation_drives_decision_and_blocking_ids():
    records = [
        {"pair_id": "f2", "family": "F2", "automatic_root_cause_class": "F2_CANONICAL_DEFECT_PROPAGATED_TO_DERIVATIVES", "remediation_state": "MANUAL_REVIEW_REQUIRED"},
    ]
    decision, criteria = p3w4.provisional_decision(records)
    assert decision != "P3W4_F2_CANONICAL_ROWS_REQUIRE_REGENERATION"
    assert criteria["F2_manual_review_required_count"] == 1
    assert criteria["F2_regeneration_required_count"] == 0


def test_f1_regeneration_and_f2_manual_review_separate_decision():
    records = [
        {"pair_id": "f1", "family": "F1", "remediation_state": "REGENERATION_REQUIRED"},
        {"pair_id": "f2", "family": "F2", "remediation_state": "MANUAL_REVIEW_REQUIRED"},
    ]
    decision, criteria = p3w4.provisional_decision(records)
    assert decision == "P3W4_F1_AND_F2_REQUIRE_SEPARATE_REMEDIATION"
    assert criteria["F1_action"] == "regeneration"
    assert criteria["F2_action"] == "manual textual/semantic review"
    assert criteria["F2_regeneration_approved"] is False


def test_r3_unresolved_regeneration_yield_from_observed_records():
    r3 = p3w4.scenario_diagnostics(observed_scenario_records())["R3"]
    assert r3["confirmed_REFUTE_contribution"] == 0
    assert r3["conditional_REFUTE_contribution_if_label_contract_preserved"] == 119
    assert r3["potential_total_REFUTE"] == "unresolved"
    assert r3["polarity_minimum_50_could_be_met"] == "unresolved"

def test_duplicate_fact_pair_rejects(monkeypatch, tmp_path):
    class Module:
        INTERVENTION_TYPES = ["none"]
        @staticmethod
        def fact_templates_for_count(count):
            return [{"pair_id": "a"}, {"pair_id": "a"}]
    monkeypatch.setattr(p3w4, "load_intervention_authority", lambda root: {"module": Module, "values": frozenset({"none"}), "source": "s"})
    with pytest.raises(ValueError, match="duplicate fact pair ID"):
        p3w4.load_fact_authority(tmp_path, {"a"}, {"a"})


def test_manual_f2_blocking_and_supporting_id_policy():
    records = [
        {"pair_id": "f1", "family": "F1", "remediation_state": "REGENERATION_REQUIRED"},
        {"pair_id": "f2", "family": "F2", "remediation_state": "MANUAL_REVIEW_REQUIRED"},
    ]
    partition = p3w4.partition_decision_pair_ids(records)
    assert partition["blocking"] == ["f2"]
    assert "f2" not in partition["supporting"]

def test_no_unresolved_global_helper_references():
    tree = ast.parse(Path(p3w4.__file__).read_text(encoding="utf-8"))
    defined = {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    called = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    for helper in {"canonical_counterpart_of", "intervention_of", "final_label_of"}:
        assert helper in defined
        assert helper in called


def test_f1_without_paraphrase_passes(monkeypatch):
    rows, sidecars, p3rows = tiny_family(monkeypatch)
    rows = [row for row in rows if row["pair_id"] != "f1" or row["intervention_type"] != "paraphrase"]
    f1, _ = p3w4.reconstruct_families(rows, sidecars, p3rows, {"u8_support_row_ids": {"f1__none"}})
    assert set(f1["f1"]) >= {"canonical", "polarity_flip"}
    assert "paraphrase" not in f1["f1"]


def test_f1_optional_paraphrase_present_passes(monkeypatch):
    rows, sidecars, p3rows = tiny_family(monkeypatch)
    f1_para = source_row("f1", "paraphrase", "SUPPORT", "A approved B, in other words.")
    rows.append(f1_para)
    sidecars[f1_para["id"]] = sidecar(f1_para)
    f1, _ = p3w4.reconstruct_families(rows, sidecars, p3rows, {"u8_support_row_ids": {"f1__none"}})
    assert "paraphrase" in f1["f1"]


def test_f1_missing_polarity_flip_rejects(monkeypatch):
    rows, sidecars, p3rows = tiny_family(monkeypatch)
    rows = [row for row in rows if row["id"] != "f1__polarity_flip"]
    with pytest.raises(ValueError, match="P3W4_MISSING_PAIR_MEMBER"):
        p3w4.reconstruct_families(rows, sidecars, p3rows, {"u8_support_row_ids": {"f1__none"}})


def test_f2_missing_paraphrase_or_polarity_rejects(monkeypatch):
    rows, sidecars, p3rows = tiny_family(monkeypatch)
    no_para = [row for row in rows if row["id"] != "f2__paraphrase"]
    with pytest.raises(ValueError, match="P3W4_MISSING_PAIR_MEMBER"):
        p3w4.reconstruct_families(no_para, sidecars, p3rows, {"u8_support_row_ids": {"f1__none"}})
    no_pol = [row for row in rows if row["id"] != "f2__polarity_flip"]
    with pytest.raises(ValueError, match="P3W4_MISSING_PAIR_MEMBER"):
        p3w4.reconstruct_families(no_pol, sidecars, p3rows, {"u8_support_row_ids": {"f1__none"}})


def test_f1_sidecar_exact_contract(monkeypatch):
    rows, sidecars, p3rows = tiny_family(monkeypatch)
    sidecars["f1__polarity_flip"]["grammar_status"] = "PASS"
    with pytest.raises(ValueError, match="P3W4_F1_SIDECAR_CONTRACT_MISMATCH"):
        p3w4.reconstruct_families(rows, sidecars, p3rows, {"u8_support_row_ids": {"f1__none"}})
    rows, sidecars, p3rows = tiny_family(monkeypatch)
    sidecars["f1__polarity_flip"]["canonical_status"] = "UNRESOLVED"
    with pytest.raises(ValueError, match="P3W4_F1_SIDECAR_CONTRACT_MISMATCH"):
        p3w4.reconstruct_families(rows, sidecars, p3rows, {"u8_support_row_ids": {"f1__none"}})
    rows, sidecars, p3rows = tiny_family(monkeypatch)
    sidecars["f1__polarity_flip"]["reason_codes"] = ["GRAMMAR_TEMPLATE_FAIL"]
    with pytest.raises(ValueError, match="P3W4_F1_SIDECAR_CONTRACT_MISMATCH"):
        p3w4.reconstruct_families(rows, sidecars, p3rows, {"u8_support_row_ids": {"f1__none"}})


def test_f2_sidecar_exact_contracts(monkeypatch):
    rows, sidecars, p3rows = tiny_family(monkeypatch)
    sidecars["f2__none"]["reason_codes"] = ["GRAMMAR_TEMPLATE_FAIL"]
    with pytest.raises(ValueError, match="P3W4_F2_CANONICAL_SIDECAR_CONTRACT_MISMATCH"):
        p3w4.reconstruct_families(rows, sidecars, p3rows, {"u8_support_row_ids": {"f1__none"}})
    rows, sidecars, p3rows = tiny_family(monkeypatch)
    sidecars["f2__paraphrase"]["canonical_status"] = "PASS"
    with pytest.raises(ValueError, match="P3W4_F2_PARAPHRASE_SIDECAR_CONTRACT_MISMATCH"):
        p3w4.reconstruct_families(rows, sidecars, p3rows, {"u8_support_row_ids": {"f1__none"}})
    rows, sidecars, p3rows = tiny_family(monkeypatch)
    sidecars["f2__paraphrase"]["reason_codes"] = ["GRAMMAR_TEMPLATE_FAIL"]
    with pytest.raises(ValueError, match="P3W4_F2_PARAPHRASE_SIDECAR_CONTRACT_MISMATCH"):
        p3w4.reconstruct_families(rows, sidecars, p3rows, {"u8_support_row_ids": {"f1__none"}})


def test_imported_validator_cannot_be_mislabeled_local(monkeypatch, tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "build_stage185a_controlled_train_integrity_sidecar.py").write_text("from analyze_stage182a_controlled_intervention_integrity import grammar_anomaly\ndef build_sidecar(row, fact):\n    return grammar_anomaly(row, fact)\n", encoding="utf-8")
    (scripts / "analyze_stage182a_controlled_intervention_integrity.py").write_text("def grammar_anomaly(row, fact):\n    return True\n", encoding="utf-8")
    stage185 = types.SimpleNamespace(grammar_anomaly=lambda row, fact: True)
    stage182 = types.SimpleNamespace(grammar_anomaly=lambda row, fact: True)
    monkeypatch.setattr(p3w4, "load_module_from_path", lambda path, name: stage185 if "stage185" in name else stage182)
    monkeypatch.setattr(p3w4, "callable_source_relative_path", lambda validator, root: "scripts/analyze_stage182a_controlled_intervention_integrity.py")
    monkeypatch.setattr(p3w4, "file_sha256", lambda path: "a" * 64)
    validator = p3w4.load_production_grammar_validator(tmp_path)
    assert validator["validator_definition_kind"] == "stage182_imported"


def test_validator_attribute_without_call_site_rejected(monkeypatch, tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "build_stage185a_controlled_train_integrity_sidecar.py").write_text("def grammar_anomaly(row, fact):\n    return True\n", encoding="utf-8")
    (scripts / "analyze_stage182a_controlled_intervention_integrity.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(p3w4, "load_module_from_path", lambda path, name: types.SimpleNamespace(grammar_anomaly=lambda row, fact: True))
    assert p3w4.load_production_grammar_validator(tmp_path)["validator_call_chain_verified"] is False


def test_validator_signature_exact_binding_rejects_required_arguments():
    with pytest.raises(ValueError, match="wrong callable signature"):
        p3w4.validate_validator_signature(lambda row, fact, required: True)
    def keyword_required(row, fact, *, required):
        return True
    with pytest.raises(ValueError, match="wrong callable signature"):
        p3w4.validate_validator_signature(keyword_required)


def test_validator_invocation_type_error_converted():
    def bad_validator(row, fact):
        raise TypeError("internal")
    member = {"source_row": {"evidence": "A did not approved B."}, "sidecar": {"reason_codes": ["DID_NOT_INFLECTED_PREDICATE", "GRAMMAR_TEMPLATE_FAIL"]}}
    validator = validator_fixture(bad_validator)
    with pytest.raises(ValueError, match="P3W4_GRAMMAR_VALIDATOR_INVOCATION_FAILED"):
        p3w4.reproduce_grammar_rule(member, {"pair_id": "p", "predicate": "approved"}, validator, True)


def test_remediation_driven_potential_yield_and_conflict_action():
    records = [
        {"pair_id": "a", "family": "F2", "members": {"canonical": {"source_row": {"id": "a_none", "final_label": "REFUTE"}}, "paraphrase": {"source_row": {"id": "a_para", "final_label": "REFUTE"}}, "polarity_flip": {"source_row": {"id": "a_pol", "final_label": "SUPPORT"}}}, "remediation_state": "TEXTUALLY_RECOVERABLE"},
        {"pair_id": "b", "family": "F2", "members": {"canonical": {"source_row": {"id": "b_none", "final_label": "REFUTE"}}, "paraphrase": {"source_row": {"id": "b_para", "final_label": "REFUTE"}}, "polarity_flip": {"source_row": {"id": "b_pol", "final_label": "SUPPORT"}}}, "remediation_state": "SEMANTIC_CONFLICT"},
    ]
    report = p3w4.potential_authority_yield(records)
    assert report["automatic_recoverable_pair_count"] == 1
    assert report["semantic_conflict_pair_count"] == 1
    decision, criteria = p3w4.provisional_decision(records)
    assert decision == "P3W4_AUDIT_BLOCKED"
    assert criteria["F2_action"] == ["textual review/recovery candidate", "semantic conflict resolution"]


def test_shared_supporting_blocking_partition_helper():
    records = [
        {"pair_id": "manual", "remediation_state": "MANUAL_REVIEW_REQUIRED"},
        {"pair_id": "conflict", "remediation_state": "SEMANTIC_CONFLICT"},
        {"pair_id": "text", "remediation_state": "TEXTUALLY_RECOVERABLE"},
        {"pair_id": "regen", "remediation_state": "REGENERATION_REQUIRED"},
    ]
    assert p3w4.partition_decision_pair_ids(records) == {"blocking": ["conflict", "manual"], "supporting": ["regen", "text"]}

def test_validator_record_schema_malformed_rejects():
    record = validator_fixture(lambda row, fact: True)
    del record["validator_definition_kind"]
    with pytest.raises(ValueError, match="P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED"):
        p3w4.validate_grammar_validator_record(record)
    record = validator_fixture(lambda row, fact: True)
    del record["validator_callable_source_path"]
    with pytest.raises(ValueError, match="P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED"):
        p3w4.validate_grammar_validator_record(record)
    record = validator_fixture(lambda row, fact: True)
    record["validator"] = "not-callable"
    with pytest.raises(ValueError, match="P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED"):
        p3w4.validate_grammar_validator_record(record)


def test_authorized_build_sidecar_call_site_passes():
    source = "def grammar_anomaly(row, fact):\n    return True\ndef build_sidecar(row, fact):\n    return grammar_anomaly(row, fact)\n"
    authority = p3w4.stage185_validator_ast_authority(source)
    assert authority["call_site_function"] == "build_sidecar"
    assert authority["call_site_authorized"] is True


def test_actual_repository_authority_function_call_site_static():
    source = Path("scripts/build_stage185a_controlled_train_integrity_sidecar.py").read_text(encoding="utf-8")
    authority = p3w4.stage185_validator_ast_authority(source)
    assert authority["call_site_function"] == "build_sidecar"
    assert authority["call_site_authorized"] is True


def test_debug_helper_only_call_site_rejected():
    source = "def grammar_anomaly(row, fact):\n    return True\ndef debug_check(row, fact):\n    return grammar_anomaly(row, fact)\n"
    with pytest.raises(ValueError, match="P3W4_GRAMMAR_VALIDATOR_CALL_SITE_NOT_AUTHORIZED"):
        p3w4.stage185_validator_ast_authority(source)


def test_module_level_and_dead_nested_call_sites_rejected():
    module_level = "def grammar_anomaly(row, fact):\n    return True\ngrammar_anomaly({}, {})\n"
    with pytest.raises(ValueError, match="P3W4_GRAMMAR_VALIDATOR_CALL_SITE_NOT_AUTHORIZED"):
        p3w4.stage185_validator_ast_authority(module_level)
    dead_nested = "def grammar_anomaly(row, fact):\n    return True\ndef build_sidecar(row, fact):\n    def nested():\n        return grammar_anomaly(row, fact)\n    return False\n"
    with pytest.raises(ValueError, match="P3W4_GRAMMAR_VALIDATOR_CALL_SITE_NOT_AUTHORIZED"):
        p3w4.stage185_validator_ast_authority(dead_nested)


def test_local_imported_validator_source_ambiguity_rejects():
    source = "from analyze_stage182a_controlled_intervention_integrity import grammar_anomaly\ndef grammar_anomaly(row, fact):\n    return True\ndef build_sidecar(row, fact):\n    return grammar_anomaly(row, fact)\n"
    with pytest.raises(ValueError, match="P3W4_GRAMMAR_VALIDATOR_SOURCE_AMBIGUOUS"):
        p3w4.stage185_validator_ast_authority(source)


def test_mixed_remediation_decision_matrix():
    decision, criteria = p3w4.provisional_decision([
        {"pair_id": "f1", "family": "F1", "remediation_state": "REGENERATION_REQUIRED"},
        {"pair_id": "f2", "family": "F2", "remediation_state": "TEXTUALLY_RECOVERABLE"},
    ])
    assert decision == "P3W4_F1_AND_F2_REQUIRE_SEPARATE_REMEDIATION"
    decision, criteria = p3w4.provisional_decision([
        {"pair_id": "f1", "family": "F1", "remediation_state": "MANUAL_REVIEW_REQUIRED"},
        {"pair_id": "f2", "family": "F2", "remediation_state": "REGENERATION_REQUIRED"},
    ])
    assert decision == "P3W4_F1_AND_F2_REQUIRE_SEPARATE_REMEDIATION"
    decision, criteria = p3w4.provisional_decision([
        {"pair_id": "m", "family": "F2", "remediation_state": "MANUAL_REVIEW_REQUIRED"},
        {"pair_id": "c", "family": "F2", "remediation_state": "SEMANTIC_CONFLICT"},
    ])
    assert criteria["F2_remediation_mixed"] is True
    assert criteria["F2_action"] == ["manual textual/semantic review", "semantic conflict resolution"]


def test_single_family_f2_decisions_allowed():
    decision, criteria = p3w4.provisional_decision([{"pair_id": "f2", "family": "F2", "remediation_state": "TEXTUALLY_RECOVERABLE"}])
    assert decision == "P3W4_EXISTING_F2_AUTHORITY_RECOVERABLE_BY_TEXTUAL_REVIEW"
    assert criteria["F2_action"] == "textual review/recovery candidate"
    decision, criteria = p3w4.provisional_decision([{"pair_id": "f2", "family": "F2", "remediation_state": "REGENERATION_REQUIRED"}])
    assert decision == "P3W4_F2_CANONICAL_ROWS_REQUIRE_REGENERATION"
    assert criteria["F2_action"] == "regeneration"


def test_validator_summary_authority_and_pair_metadata_match():
    authority = p3w4.grammar_validator_summary_authority(validator_fixture(lambda row, fact: True))
    record = {"members": {"canonical": {"grammar_rule_reproduction": dict(authority)}}}
    p3w4.validate_pair_validator_metadata([record], authority)
    record["members"]["canonical"]["grammar_rule_reproduction"]["validator_signature"] = "wrong"
    with pytest.raises(ValueError, match="P3W4_GRAMMAR_VALIDATOR_AUTHORITY_MISMATCH"):
        p3w4.validate_pair_validator_metadata([record], authority)


def test_resolved_stage185_validator_blob_identity(monkeypatch):
    record = validator_fixture(lambda row, fact: True, result_source="scripts/build_stage185a_controlled_train_integrity_sidecar.py")
    monkeypatch.setattr(p3w4, "git_blob_identity", lambda root, commit, rel_path: f"blob:{rel_path}")
    identity = p3w4.verify_resolved_validator_blob_identity(Path("."), "p3w3", record)
    assert identity["path"] == "scripts/build_stage185a_controlled_train_integrity_sidecar.py"


def test_resolved_stage182_validator_blob_identity(monkeypatch):
    record = validator_fixture(lambda row, fact: True, result_source="scripts/analyze_stage182a_controlled_intervention_integrity.py")
    record["validator_definition_kind"] = "stage182_imported"
    monkeypatch.setattr(p3w4, "git_blob_identity", lambda root, commit, rel_path: f"blob:{rel_path}")
    identity = p3w4.verify_resolved_validator_blob_identity(Path("."), "p3w3", record)
    assert identity["path"] == "scripts/analyze_stage182a_controlled_intervention_integrity.py"


def test_resolved_validator_blob_mismatch_rejects(monkeypatch):
    record = validator_fixture(lambda row, fact: True, result_source="scripts/build_stage185a_controlled_train_integrity_sidecar.py")
    def fake_blob(root, commit, rel_path):
        return "old" if commit == "p3w3" else "new"
    monkeypatch.setattr(p3w4, "git_blob_identity", fake_blob)
    with pytest.raises(ValueError, match="P3W4_GRAMMAR_VALIDATOR_BLOB_IDENTITY_MISMATCH"):
        p3w4.verify_resolved_validator_blob_identity(Path("."), "p3w3", record)


def test_validator_blob_source_mismatch_rejects():
    authority = p3w4.grammar_validator_summary_authority(validator_fixture(lambda row, fact: True, result_source="scripts/build_stage185a_controlled_train_integrity_sidecar.py"))
    blob = {"path": "scripts/analyze_stage182a_controlled_intervention_integrity.py", "matches": True}
    with pytest.raises(ValueError, match="P3W4_GRAMMAR_VALIDATOR_AUTHORITY_MISMATCH"):
        p3w4.validate_runtime_grammar_validator_authority(authority, blob, [])


def test_multiple_authorized_build_sidecar_calls_pass():
    source = "def grammar_anomaly(row, fact):\n    return True\ndef build_sidecar(row, fact):\n    a = grammar_anomaly(row, fact)\n    b = grammar_anomaly(row, fact)\n    return a or b\n"
    authority = p3w4.stage185_validator_ast_authority(source)
    assert authority["authorized_call_site_count"] == 2
    assert all(site["authorized"] for site in authority["authorized_call_sites"])


def test_authorized_plus_unauthorized_call_site_inventory_rejects():
    source = "def grammar_anomaly(row, fact):\n    return True\ndef build_sidecar(row, fact):\n    return grammar_anomaly(row, fact)\ndef debug_check(row, fact):\n    return grammar_anomaly(row, fact)\n"
    with pytest.raises(ValueError, match="P3W4_GRAMMAR_VALIDATOR_CALL_SITE_NOT_AUTHORIZED"):
        p3w4.stage185_validator_ast_authority(source)


def test_main_reachable_debug_call_rejected():
    source = "def grammar_anomaly(row, fact):\n    return True\ndef debug_check(row, fact):\n    return grammar_anomaly(row, fact)\ndef main():\n    return debug_check({}, {})\n"
    with pytest.raises(ValueError, match="P3W4_GRAMMAR_VALIDATOR_CALL_SITE_NOT_AUTHORIZED"):
        p3w4.stage185_validator_ast_authority(source)


def test_assignment_form_module_level_call_rejected():
    source = "def grammar_anomaly(row, fact):\n    return True\nRESULT = grammar_anomaly({}, {})\n"
    with pytest.raises(ValueError, match="P3W4_GRAMMAR_VALIDATOR_CALL_SITE_NOT_AUTHORIZED"):
        p3w4.stage185_validator_ast_authority(source)


def test_no_call_site_returns_unverified_empty_authority():
    source = "def grammar_anomaly(row, fact):\n    return True\n"
    authority = p3w4.stage185_validator_ast_authority(source)
    assert authority["authorized_call_site_count"] == 0
    assert authority["call_site_authorized"] is False


def test_verified_validator_record_semantic_rejections():
    record = validator_fixture(lambda row, fact: True)
    record["validator_call_site_authorized"] = False
    with pytest.raises(ValueError, match="P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED"):
        p3w4.validate_grammar_validator_record(record)
    record = validator_fixture(lambda row, fact: True)
    record["validator_source_path"] = None
    with pytest.raises(ValueError, match="P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED"):
        p3w4.validate_grammar_validator_record(record)
    record = validator_fixture(lambda row, fact: True)
    record["validator_call_site_lineno"] = None
    with pytest.raises(ValueError, match="P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED"):
        p3w4.validate_grammar_validator_record(record)
    record = validator_fixture(lambda row, fact: True)
    record["validator_callable_source_path"] = "other.py"
    with pytest.raises(ValueError, match="P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED"):
        p3w4.validate_grammar_validator_record(record)
    record = validator_fixture(lambda row, fact: True)
    record["validator_authority_function"] = "other"
    with pytest.raises(ValueError, match="P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED"):
        p3w4.validate_grammar_validator_record(record)
    record = validator_fixture(lambda row, fact: True)
    record["validator_source_sha256"] = "sha"
    with pytest.raises(ValueError, match="P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED"):
        p3w4.validate_grammar_validator_record(record)


def test_half_verified_validator_record_rejected():
    record = p3w4.empty_validator_record()
    record["validator_source_path"] = "scripts/build_stage185a_controlled_train_integrity_sidecar.py"
    with pytest.raises(ValueError, match="P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED"):
        p3w4.validate_grammar_validator_record(record)


def test_full_pair_metadata_comparison_includes_authorization_inventory():
    authority = p3w4.grammar_validator_summary_authority(validator_fixture(lambda row, fact: True))
    record = {"members": {"canonical": {"grammar_rule_reproduction": dict(authority)}}}
    p3w4.validate_runtime_grammar_validator_authority(authority, {"path": authority["validator_authority_source"]}, [record])
    record["members"]["canonical"]["grammar_rule_reproduction"]["validator_call_site_authorized"] = False
    with pytest.raises(ValueError, match="P3W4_GRAMMAR_VALIDATOR_AUTHORITY_MISMATCH"):
        p3w4.validate_runtime_grammar_validator_authority(authority, {"path": authority["validator_authority_source"]}, [record])


def test_single_authorized_build_sidecar_call_count_exactly_one():
    source = "def grammar_anomaly(row, fact):\n    return True\ndef build_sidecar(row, fact):\n    return grammar_anomaly(row, fact)\n"
    authority = p3w4.stage185_validator_ast_authority(source)
    assert authority["authorized_call_site_count"] == 1
    assert authority["authorized_call_sites"][0]["scope_path"] == ["function:build_sidecar"]


def test_top_level_class_method_call_rejected():
    source = "def grammar_anomaly(row, fact):\n    return True\ndef build_sidecar(row, fact):\n    return grammar_anomaly(row, fact)\nclass Debug:\n    def check(self, row, fact):\n        return grammar_anomaly(row, fact)\n"
    with pytest.raises(ValueError, match="P3W4_GRAMMAR_VALIDATOR_CALL_SITE_NOT_AUTHORIZED"):
        p3w4.stage185_validator_ast_authority(source)


def test_nested_class_method_call_rejected():
    source = "def grammar_anomaly(row, fact):\n    return True\ndef build_sidecar(row, fact):\n    class Debug:\n        def check(self):\n            return grammar_anomaly(row, fact)\n    return False\n"
    with pytest.raises(ValueError, match="P3W4_GRAMMAR_VALIDATOR_CALL_SITE_NOT_AUTHORIZED"):
        p3w4.stage185_validator_ast_authority(source)


def test_decorator_and_default_argument_grammar_calls_rejected():
    decorator_source = "def grammar_anomaly(row, fact):\n    return True\n@grammar_anomaly({}, {})\ndef build_sidecar(row, fact):\n    return False\n"
    with pytest.raises(ValueError, match="P3W4_GRAMMAR_VALIDATOR_CALL_SITE_NOT_AUTHORIZED"):
        p3w4.stage185_validator_ast_authority(decorator_source)
    default_source = "def grammar_anomaly(row, fact):\n    return True\ndef build_sidecar(row=grammar_anomaly({}, {}), fact=None):\n    return False\n"
    with pytest.raises(ValueError, match="P3W4_GRAMMAR_VALIDATOR_CALL_SITE_NOT_AUTHORIZED"):
        p3w4.stage185_validator_ast_authority(default_source)


def test_same_line_two_calls_have_distinct_col_offset_identities():
    source = "def grammar_anomaly(row, fact):\n    return True\ndef build_sidecar(row, fact):\n    return grammar_anomaly(row, fact) or grammar_anomaly(row, fact)\n"
    authority = p3w4.stage185_validator_ast_authority(source)
    identities = [(tuple(site["scope_path"]), site["lineno"], site["col_offset"]) for site in authority["authorized_call_sites"]]
    assert authority["authorized_call_site_count"] == 2
    assert len(set(identities)) == 2


def test_malformed_inventory_entries_rejected():
    record = validator_fixture(lambda row, fact: True)
    record["validator_authorized_call_sites"][0]["authorized"] = False
    with pytest.raises(ValueError, match="P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED"):
        p3w4.validate_grammar_validator_record(record)
    record = validator_fixture(lambda row, fact: True)
    record["validator_authorized_call_sites"][0]["function_name"] = "debug_check"
    with pytest.raises(ValueError, match="P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED"):
        p3w4.validate_grammar_validator_record(record)


def test_duplicate_inventory_identity_rejected():
    record = validator_fixture(lambda row, fact: True)
    record["validator_authorized_call_sites"].append(dict(record["validator_authorized_call_sites"][0]))
    record["validator_authorized_call_site_count"] = 2
    with pytest.raises(ValueError, match="P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED"):
        p3w4.validate_grammar_validator_record(record)


def test_representative_function_line_mismatch_rejected():
    record = validator_fixture(lambda row, fact: True)
    record["validator_call_site_function"] = "debug_check"
    with pytest.raises(ValueError, match="P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED"):
        p3w4.validate_grammar_validator_record(record)
    record = validator_fixture(lambda row, fact: True)
    record["validator_call_site_lineno"] = 999
    with pytest.raises(ValueError, match="P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED"):
        p3w4.validate_grammar_validator_record(record)


def test_nested_function_definition_expressions_rejected():
    default_source = "def grammar_anomaly(row, fact):\n    return True\ndef build_sidecar(row, fact):\n    def inner(x=grammar_anomaly(row, fact)):\n        return x\n    return False\n"
    with pytest.raises(ValueError, match="P3W4_GRAMMAR_VALIDATOR_CALL_SITE_NOT_AUTHORIZED"):
        p3w4.stage185_validator_ast_authority(default_source)
    decorator_source = "def grammar_anomaly(row, fact):\n    return True\ndef build_sidecar(row, fact):\n    @grammar_anomaly(row, fact)\n    def inner():\n        return False\n    return False\n"
    with pytest.raises(ValueError, match="P3W4_GRAMMAR_VALIDATOR_CALL_SITE_NOT_AUTHORIZED"):
        p3w4.stage185_validator_ast_authority(decorator_source)


def test_nested_class_definition_expressions_rejected():
    decorator_source = "def grammar_anomaly(row, fact):\n    return True\ndef build_sidecar(row, fact):\n    @grammar_anomaly(row, fact)\n    class Debug:\n        pass\n    return False\n"
    with pytest.raises(ValueError, match="P3W4_GRAMMAR_VALIDATOR_CALL_SITE_NOT_AUTHORIZED"):
        p3w4.stage185_validator_ast_authority(decorator_source)
    base_source = "def grammar_anomaly(row, fact):\n    return True\ndef build_sidecar(row, fact):\n    class Debug(grammar_anomaly(row, fact)):\n        pass\n    return False\n"
    with pytest.raises(ValueError, match="P3W4_GRAMMAR_VALIDATOR_CALL_SITE_NOT_AUTHORIZED"):
        p3w4.stage185_validator_ast_authority(base_source)
    keyword_source = "def grammar_anomaly(row, fact):\n    return True\ndef build_sidecar(row, fact):\n    class Debug(metaclass=grammar_anomaly(row, fact)):\n        pass\n    return False\n"
    with pytest.raises(ValueError, match="P3W4_GRAMMAR_VALIDATOR_CALL_SITE_NOT_AUTHORIZED"):
        p3w4.stage185_validator_ast_authority(keyword_source)


def test_lambda_definition_and_body_calls_rejected():
    default_source = "def grammar_anomaly(row, fact):\n    return True\ndef build_sidecar(row, fact):\n    fn = lambda x=grammar_anomaly(row, fact): x\n    return False\n"
    with pytest.raises(ValueError, match="P3W4_GRAMMAR_VALIDATOR_CALL_SITE_NOT_AUTHORIZED"):
        p3w4.stage185_validator_ast_authority(default_source)
    body_source = "def grammar_anomaly(row, fact):\n    return True\ndef build_sidecar(row, fact):\n    fn = lambda: grammar_anomaly(row, fact)\n    return False\n"
    with pytest.raises(ValueError, match="P3W4_GRAMMAR_VALIDATOR_CALL_SITE_NOT_AUTHORIZED"):
        p3w4.stage185_validator_ast_authority(body_source)


def test_function_annotation_inventory_rejects():
    parameter_source = "def grammar_anomaly(row, fact):\n    return True\ndef build_sidecar(row: grammar_anomaly({}, {}), fact=None):\n    return False\n"
    with pytest.raises(ValueError, match="P3W4_GRAMMAR_VALIDATOR_CALL_SITE_NOT_AUTHORIZED"):
        p3w4.stage185_validator_ast_authority(parameter_source)
    return_source = "def grammar_anomaly(row, fact):\n    return True\ndef build_sidecar(row, fact) -> grammar_anomaly({}, {}):\n    return False\n"
    with pytest.raises(ValueError, match="P3W4_GRAMMAR_VALIDATOR_CALL_SITE_NOT_AUTHORIZED"):
        p3w4.stage185_validator_ast_authority(return_source)
    nested_source = "def grammar_anomaly(row, fact):\n    return True\ndef build_sidecar(row, fact):\n    def inner(value: grammar_anomaly(row, fact)):\n        return value\n    return False\n"
    with pytest.raises(ValueError, match="P3W4_GRAMMAR_VALIDATOR_CALL_SITE_NOT_AUTHORIZED"):
        p3w4.stage185_validator_ast_authority(nested_source)


def test_async_build_sidecar_body_call_rejected():
    source = "def grammar_anomaly(row, fact):\n    return True\nasync def build_sidecar(row, fact):\n    return grammar_anomaly(row, fact)\n"
    with pytest.raises(ValueError, match="P3W4_GRAMMAR_VALIDATOR_CALL_SITE_NOT_AUTHORIZED"):
        p3w4.stage185_validator_ast_authority(source)


def test_call_site_entry_reachability_schema_rejections():
    record = validator_fixture(lambda row, fact: True)
    del record["validator_authorized_call_sites"][0]["reachable_from_run_or_main"]
    with pytest.raises(ValueError, match="P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED"):
        p3w4.validate_grammar_validator_record(record)
    record = validator_fixture(lambda row, fact: True)
    record["validator_authorized_call_sites"][0]["reachable_from_run_or_main"] = "yes"
    with pytest.raises(ValueError, match="P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED"):
        p3w4.validate_grammar_validator_record(record)
    record = validator_fixture(lambda row, fact: True)
    record["validator_call_site_reachable_from_run"] = "yes"
    with pytest.raises(ValueError, match="P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED"):
        p3w4.validate_grammar_validator_record(record)
    record = validator_fixture(lambda row, fact: True)
    record["validator_call_site_reachable_from_run"] = False
    with pytest.raises(ValueError, match="P3W4_GRAMMAR_VALIDATOR_RECORD_MALFORMED"):
        p3w4.validate_grammar_validator_record(record)
