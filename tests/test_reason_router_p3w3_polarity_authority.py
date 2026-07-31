from __future__ import annotations

import copy
import inspect
from pathlib import Path

import pytest

from scripts import analyze_reason_router_p3w3_polarity_authority as audit

EXEC = "e" * 40


def _record(row_id: str, pair_id: str, kind: str, *, split: str = "train") -> dict:
    values = {
        "support": (1, 1, 1, "SUPPORT", "SUPPORT", "none", "none"),
        "refute": (1, 1, 1, "REFUTE", "REFUTE", "polarity", "polarity_flip"),
        "frame": (0, 1, 1, "NOT_ENTITLED", "NONE", "frame", "entity_swap"),
    }
    frame, pred, suff, final, polarity, primary, intervention = values[kind]
    return {
        "id": row_id,
        "pair_id": pair_id,
        "intervention_type": intervention,
        "final_label": final,
        "frame_compatible_label": frame,
        "predicate_covered_label": pred,
        "sufficiency_label": suff,
        "polarity_label": polarity,
        "primary_failure_type": primary,
        "split": split,
    }


def _sidecar(record: dict, canonical: str, *, split: str = "train") -> dict:
    return {
        "row_id": record["id"],
        "split": split,
        "pair_id": record["pair_id"],
        "canonical_row_id": canonical,
        "canonical_status": "PASS",
        "intervention_contract_status": "PASS",
        "integrity_status": "ELIGIBLE",
        "schema_status": "PASS",
        "dataset_source_status": "PASS",
        "grammar_status": "PASS",
        "polarity_contamination_status": "PASS",
        "time_swap_status": "PASS",
        "reason_codes": ["ELIGIBLE_CLEAN_COMPATIBLE"],
        "source_dataset_path": "data/controlled_v5_v3_without_time_swap.jsonl",
        "source_dataset_sha256": audit.EXPECTED_DATA_SHA256,
        "frame_compatible_label": record["frame_compatible_label"],
        "audit_changed_axes": [],
        "audit_expected_axes": [],
        "audit_preserved_axes": ["location", "name", "object", "polarity", "predicate", "role", "time", "title"],
        "generator_source_path": "/kaggle/working/ContraMamba/scripts/build_controlled_v5.py",
        "generator_source_sha256": "g" * 64,
        "integrity_builder_sha256": "i" * 64,
        "stage182a_report_sha256": "a" * 64,
        "stage184a_report_sha256": "b" * 64,
    }


def _fixture(*, grammar = "PASS", contract_status: str = "PASS", unresolved: bool = False, semantic_defect: bool = False, proven: bool = False):
    support = _record("p0__none", "p0", "support")
    refute = _record("p0__polarity_flip", "p0", "refute")
    if semantic_defect:
        refute["polarity_label"] = "SUPPORT"
    records = [support, refute]
    sidecars = {r["id"]: _sidecar(r, "p0__none") for r in records}
    rside = sidecars["p0__polarity_flip"]
    rside["grammar_status"] = grammar
    rside["intervention_contract_status"] = contract_status
    rside["audit_changed_axes"] = ["polarity"]
    rside["audit_expected_axes"] = ["polarity"]
    rside["audit_preserved_axes"] = ["location", "name", "object", "predicate", "role", "time", "title"]
    if proven:
        rside["intervention_contract_status"] = "FAIL"
        rside["reason_codes"] = ["EXPECTED_POLARITY_INTERVENTION_MISMATCH"]
    if unresolved:
        del rside["grammar_status"]
    return records, [], sidecars


def _states(**kwargs):
    records, dev, sidecars = _fixture(**kwargs)
    return audit.build_row_states(train_records=records, dev_records=None, sidecar_by_id=sidecars)


def _summary(states):
    return audit.analyze_states(
        states,
        execution_commit=EXEC,
        dataset={"path": "data", "sha256": audit.EXPECTED_DATA_SHA256},
        sidecar={"path": "sidecar", "semantic_sha256": audit.EXPECTED_SIDECAR_SEMANTIC_SHA256},
        split={"split_seed": 174, "dev_ratio": 0.2, "train_rows": len(states), "dev_rows": 0, "ordered_train_identity": "synthetic"},
    )


def test_production_helper_output_is_row_authority_exactly():
    records, _, sidecars = _fixture(grammar="FAIL")
    annotated, supervision_audit, _ = audit.apply_production_reason_supervision(records, sidecars)
    states = audit.build_row_states(train_records=records, dev_records=None, sidecar_by_id=sidecars)
    fields = [
        "p2_primary_reason",
        "p2_primary_reason_target_4",
        "p2_reason_supervision_eligible",
        "p2_reason_exclusion_codes",
        "p2_frame_applicable",
        "p2_predicate_applicable",
        "p2_sufficiency_applicable",
        "p2_polarity_applicable",
        "p2_polarity_target_2",
        "intervention_contract_pass",
        "generator_integrity_status",
    ]
    by_id = {state["row_id"]: state for state in states}
    for row in annotated:
        state = by_id[row["id"]]
        assert state["derived_primary_reason"] == row["p2_primary_reason"]
        assert state["p2_primary_reason_target_4"] == row["p2_primary_reason_target_4"]
        assert state["p2_reason_supervision_eligible"] == row["p2_reason_supervision_eligible"]
        assert state["ordered_exclusion_codes"] == row["p2_reason_exclusion_codes"]
        assert state["p2_frame_applicable"] == row["p2_frame_applicable"]
        assert state["p2_predicate_applicable"] == row["p2_predicate_applicable"]
        assert state["p2_sufficiency_applicable"] == row["p2_sufficiency_applicable"]
        assert state["p2_polarity_applicable"] == row["p2_polarity_applicable"]
        assert state["p2_polarity_target_2"] == row["p2_polarity_target_2"]
        assert state["intervention_contract_pass"] == row["intervention_contract_pass"]
        assert state["normalized_generator_integrity_status"] == row["generator_integrity_status"]
    assert fields
    assert "train_reason_counts" in supervision_audit
    assert "target_class_counts" in supervision_audit
    assert "train_exclusion_counts" in supervision_audit


def test_clean_support_and_clean_refute_produce_nondegenerate_polarity_readiness():
    summary = _summary(_states())
    assert summary["final_label_overview"]["eligible_SUPPORT_polarity_targets"] == 1
    assert summary["final_label_overview"]["eligible_REFUTE_polarity_targets"] == 1
    assert summary["polarity_readiness_under_each_counterfactual"]["C0"] is True


def test_support_only_produces_current_degenerate_readiness():
    support = _record("p0__none", "p0", "support")
    states = audit.build_row_states(train_records=[support], dev_records=None, sidecar_by_id={support["id"]: _sidecar(support, support["id"])})
    summary = _summary(states)
    assert summary["final_label_overview"]["eligible_SUPPORT_polarity_targets"] == 1
    assert summary["final_label_overview"]["eligible_REFUTE_polarity_targets"] == 0
    assert summary["polarity_readiness_under_each_counterfactual"]["C0"] is False


def test_refute_excluded_only_by_generator_defect_is_counted_exactly():
    summary = _summary(_states(grammar="FAIL"))
    assert summary["REFUTE_exclusion_marginals"]["P2_GENERATOR_STATUS_DEFECT"] == 1
    assert summary["REFUTE_exact_exclusion_combinations"]["P2_GENERATOR_STATUS_DEFECT"] == 1


def test_refute_excluded_only_by_intervention_contract_failure_is_counted_exactly():
    states = _states(proven=True)
    summary = _summary(states)
    assert summary["REFUTE_exclusion_marginals"]["P2_POLARITY_INTERVENTION_CONTRACT_FAIL"] == 1
    assert "P2_POLARITY_INTERVENTION_CONTRACT_FAIL" in next(s for s in states if s["final_label"] == "REFUTE")["ordered_exclusion_codes"]


def test_refute_excluded_only_by_unresolved_integrity_is_counted_exactly():
    summary = _summary(_states(unresolved=True))
    assert summary["REFUTE_exclusion_marginals"]["P2_INTEGRITY_SOURCE_REQUIRED"] == 1
    assert summary["REFUTE_exact_exclusion_combinations"]["P2_INTEGRITY_SOURCE_REQUIRED"] == 1


def test_overlapping_exclusion_codes_preserve_exact_combination_counts():
    summary = _summary(_states(grammar="FAIL", proven=True))
    assert summary["REFUTE_exclusion_marginals"]["P2_POLARITY_INTERVENTION_CONTRACT_FAIL"] == 1
    assert summary["REFUTE_exclusion_marginals"]["P2_GENERATOR_STATUS_DEFECT"] == 1


def test_exclusion_marginal_counts_and_combination_counts_are_both_correct():
    summary = _summary(_states(grammar="FAIL", proven=True))
    assert sum(summary["REFUTE_exact_exclusion_combinations"].values()) == 1
    assert summary["REFUTE_exclusion_marginals"]["P2_POLARITY_INTERVENTION_CONTRACT_FAIL"] == 1


def test_canonical_support_clean_refute_ineligible_pair_is_classified_correctly():
    summary = _summary(_states(grammar="FAIL"))
    pair = summary["pair_level_canonical_comparison"]
    assert pair["refute_row_count"] == 1
    assert pair["unique_refute_pair_count"] == 1
    assert pair["pairs_with_eligible_canonical_SUPPORT_row"] == 1
    assert pair["pairs_where_REFUTE_is_ineligible_but_canonical_SUPPORT_is_eligible"] == 1
    assert pair["pairs_with_independent_REFUTE_defect_evidence"] == 1


def test_missing_sidecar_fails_closed():
    records, _, sidecars = _fixture()
    del sidecars["p0__polarity_flip"]
    with pytest.raises(ValueError, match="P2_SIDECAR_MISSING"):
        audit.build_row_states(train_records=records, dev_records=None, sidecar_by_id=sidecars)


def test_canonical_lineage_corruption_fails_closed():
    records, _, sidecars = _fixture()
    sidecars["p0__polarity_flip"]["canonical_row_id"] = "missing"
    with pytest.raises(ValueError):
        audit.build_row_states(train_records=records, dev_records=None, sidecar_by_id=sidecars)


def test_duplicate_row_id_fails_closed():
    records, _, _ = _fixture()
    with pytest.raises(ValueError, match="P2_DUPLICATE_ROW_ID"):
        audit.validate_split_contract([records[0], copy.deepcopy(records[0])], [], 2, audit.ordered_train_identity_hash([records[0], records[0]]))


def test_train_dev_pair_leakage_fails_closed():
    records, _, _ = _fixture()
    with pytest.raises(ValueError, match="P2_TRAIN_DEV_PAIR_ID_LEAKAGE"):
        audit.validate_split_contract([records[0]], [records[1]], 1, audit.ordered_train_identity_hash([records[0]]))


def test_dataset_sha_mismatch_fails_closed(tmp_path: Path):
    data = tmp_path / "d.jsonl"
    side = tmp_path / "s.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    side.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dataset SHA mismatch"):
        audit.load_reason_sidecar(data_path=data, source_records=[], sidecar_path=side, expected_data_sha256="0" * 64, expected_semantic_sha256="0" * 64)

def test_sidecar_semantic_sha_mismatch_fails_closed(tmp_path: Path):
    record = _record("p0__none", "p0", "support")
    data = tmp_path / "d.jsonl"
    side = tmp_path / "s.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    side.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="semantic SHA"):
        audit.load_reason_sidecar(data_path=data, source_records=[record], sidecar_path=side, expected_data_sha256=audit._file_sha256(data), expected_semantic_sha256="0" * 64)


def test_execution_commit_mismatch_fails_closed(monkeypatch):
    monkeypatch.setattr(audit, "_git_output", lambda args: "a" * 40)
    with pytest.raises(ValueError, match="EXECUTION_COMMIT"):
        audit.validate_git_authority("b" * 40)


def test_ordered_train_identity_mismatch_fails_closed():
    records, _, _ = _fixture()
    with pytest.raises(ValueError, match="ORDERED_TRAIN_IDENTITY"):
        audit.validate_split_contract(records, [], len(records), "x" * 64)


def test_unknown_nonempty_intervention_type_fails_closed():
    records, _, sidecars = _fixture()
    records[1]["intervention_type"] = "unknown_nonempty_value"
    with pytest.raises(ValueError, match="P2_UNKNOWN_INTERVENTION_TYPE"):
        audit.build_row_states(train_records=records, dev_records=None, sidecar_by_id=sidecars)


def test_empty_intervention_type_fails_closed():
    records, _, sidecars = _fixture()
    records[1]["intervention_type"] = ""
    with pytest.raises(ValueError, match="P2_UNKNOWN_INTERVENTION_TYPE"):
        audit.build_row_states(train_records=records, dev_records=None, sidecar_by_id=sidecars)


def test_every_registered_intervention_type_is_accepted_by_metadata_validator():
    for intervention_type in audit.AUTHORIZED_INTERVENTION_TYPES:
        record = _record(f"p0__{intervention_type}", "p0", "support")
        record["intervention_type"] = intervention_type
        audit._validate_source_metadata(record)

def test_non_nested_synthetic_input_cannot_produce_negative_attrition():
    stages = audit._stage_counts(_states(grammar="FAIL"))
    funnel = audit._attrition_funnel(stages)
    for row in funnel[1:]:
        assert row["support_attrition_from_previous_stage"] >= 0
        assert row["refute_attrition_from_previous_stage"] >= 0


def test_every_universe_is_subset_of_previous_universe():
    stages = audit._stage_counts(_states(grammar="FAIL"))
    previous = None
    for payload in stages.values():
        current = set(payload["row_ids"])
        if previous is not None:
            assert current <= previous
        previous = current


def test_c1_c5_counterfactuals_do_not_mutate_c0():
    states = _states(grammar="FAIL")
    cfs = audit.counterfactual_results(states)
    assert cfs["C0"]["eligible_refute_count"] == 0
    assert cfs["C1"]["eligible_refute_count"] == 1
    assert cfs["C5"]["eligible_refute_count"] == 0
    assert states[1]["ordered_exclusion_codes"] == ["P2_GENERATOR_STATUS_DEFECT"]


def test_counterfactuals_admit_only_rows_whose_ignored_exclusions_match_definition():
    cfs = audit.counterfactual_results(_states(proven=True))
    assert cfs["C1"]["eligible_refute_count"] == 0
    assert cfs["C4"]["eligible_refute_count"] == 1


def test_grammar_defect_is_not_recoverable_and_not_admitted_by_c5():
    states = _states(grammar="FAIL")
    assert states[1]["generator_evidence_class"] == audit.INDEPENDENT_GENERATOR_OR_SEMANTIC_DEFECT_EVIDENCE
    decision, _ = audit.decide(states, audit.counterfactual_results(states))
    assert decision in {"P3W3_NEW_REFUTE_AUTHORITY_REQUIRED", "P3W3_MIXED_REMEDIATION_REQUIRED"}
    assert audit.counterfactual_results(states)["C5"]["eligible_refute_count"] == 0


def test_ambiguous_generator_defect_requires_reannotation_and_is_not_admitted_by_c5():
    states = _states(proven=True)
    states[1]["sidecar_reason_codes"] = ["UNEXPLAINED_DEFECT"]
    evidence_class, evidence_reasons, proof_meta = audit.classify_generator_evidence(states[1])
    states[1]["generator_evidence_class"] = evidence_class
    states[1]["generator_evidence_reasons"] = evidence_reasons
    states[1].update(proof_meta)
    assert states[1]["generator_evidence_class"] == audit.AMBIGUOUS_INTEGRITY_EVIDENCE
    decision, _ = audit.decide(states, audit.counterfactual_results(states))
    assert decision == "P3W3_EXISTING_AUTHORITY_REQUIRES_INTEGRITY_REANNOTATION"
    assert audit.counterfactual_results(states)["C5"]["eligible_refute_count"] == 0


def test_default_generator_defect_with_insufficient_provenance_cannot_prove_narrow_repair():
    states = _states(proven=True)
    assert states[1]["generator_evidence_class"] == audit.AMBIGUOUS_INTEGRITY_EVIDENCE
    cfs = audit.counterfactual_results(states)
    assert cfs["C5"]["eligible_refute_count"] == 0
    decision, _ = audit.decide(states, cfs)
    assert decision == "P3W3_EXISTING_AUTHORITY_REQUIRES_INTEGRITY_REANNOTATION"


def test_explicit_injected_proof_authority_allows_narrow_repair_and_c5():
    states = _states(proven=True)
    proof = {
        "available": True,
        "source": "synthetic-test-only",
        "clause": "synthetic polarity-only clause",
        "accepted_reason_codes": ["EXPECTED_POLARITY_INTERVENTION_MISMATCH"],
        "required_preserved_axes": ["location", "name", "object", "predicate", "role", "time", "title"],
        "allowed_nonpass_fields": ["intervention_contract_status"],
    }
    evidence_class, evidence_reasons, proof_meta = audit.classify_generator_evidence(states[1], proof_authority=proof)
    states[1]["generator_evidence_class"] = evidence_class
    states[1]["generator_evidence_reasons"] = evidence_reasons
    states[1].update(proof_meta)
    assert states[1]["generator_evidence_class"] == audit.PROVEN_EXPECTED_POLARITY_INTERVENTION_MISMATCH_ONLY
    cfs = audit.counterfactual_results(states)
    assert cfs["C0"]["eligible_refute_count"] == 0
    assert cfs["C5"]["eligible_refute_count"] == 1
    assert cfs["C5"]["newly_admitted_rows"][0]["proof_contract_available"] is True
    assert cfs["C5"]["newly_admitted_rows"][0]["proof_contract_source"] == "synthetic-test-only"
    decision, _ = audit.decide(states, cfs)
    assert decision == "P3W3_EXISTING_AUTHORITY_RECOVERABLE_BY_NARROW_CONTRACT_REPAIR"


def test_genuine_semantic_defect_yields_new_authority_or_mixed_remediation_decision():
    states = _states(semantic_defect=True)
    decision, _ = audit.decide(states, audit.counterfactual_results(states))
    assert decision in {"P3W3_NEW_REFUTE_AUTHORITY_REQUIRED", "P3W3_MIXED_REMEDIATION_REQUIRED"}


def test_pair_level_unique_aggregation_handles_multi_refute_rows():
    records, _, sidecars = _fixture(grammar="FAIL")
    extra = _record("p0__polarity_flip_extra", "p0", "refute")
    records.append(extra)
    sidecars[extra["id"]] = _sidecar(extra, "p0__none")
    sidecars[extra["id"]]["grammar_status"] = "FAIL"
    sidecars[extra["id"]]["audit_changed_axes"] = ["polarity"]
    sidecars[extra["id"]]["audit_expected_axes"] = ["polarity"]
    states = audit.build_row_states(train_records=records, dev_records=None, sidecar_by_id=sidecars)
    pair = audit.pair_level_canonical_comparison(states)
    assert pair["refute_row_count"] == 2
    assert pair["unique_refute_pair_count"] == 1
    assert pair["multi_refute_row_pair_count"] == 1


def test_refute_export_contains_provenance_and_generator_evidence():
    states = _states(grammar="FAIL")
    row = audit.refute_export_record(states[1], states)
    for key in (
        "generator_source_path",
        "generator_source_sha256",
        "integrity_builder_sha256",
        "stage182a_report_sha256",
        "stage184a_report_sha256",
        "generator_evidence_class",
        "generator_evidence_reasons",
    ):
        assert key in row


def test_summary_contains_sidecar_interpretation_and_evidence_aggregates():
    summary = _summary(_states(grammar="FAIL"))
    assert "sidecar_semantic_interpretation_audit" in summary
    assert "generator_evidence_class_counts" in summary
    assert "generator_evidence_class_by_exact_exclusion_combination" in summary


def test_no_model_tokenizer_optimizer_scheduler_references_in_analyzer_source():
    source = inspect.getsource(audit)
    forbidden = [
        "AutoTokenizer.from_pretrained",
        "ContraMambaV6BMinimal(",
        "AdamW(",
        "get_linear_schedule_with_warmup",
        ".forward(",
        ".backward(",
        "cuda.is_available",
    ]
    for token in forbidden:
        assert token not in source


def test_analyzer_uses_only_production_metadata_helper_for_reason_supervision():
    source = inspect.getsource(audit.apply_production_reason_supervision)
    assert "_p2_prepare_reason_supervision_train_only" in source
    assert "torch.device(\"cpu\")" in source
    assert "train_inputs={}" in source
    assert "require_min_counts=False" in source


def test_execution_isolation_audit_flags_remain_false():
    assert all(value is False for value in audit.EXECUTION_ISOLATION_AUDIT.values())

def test_production_and_row_derived_primary_binary_and_exclusion_counts_match():
    records, _, sidecars = _fixture(grammar="FAIL")
    annotated, supervision_audit, _ = audit.apply_production_reason_supervision(records, sidecars)
    expected_canonical = audit.validate_train_sidecar_identity_and_lineage(records, sidecars)
    states = [
        audit._build_row_state(
            record,
            split="train",
            source_label="clean_main",
            sidecar_by_id=sidecars,
            expected_canonical_by_pair=expected_canonical,
        )
        for record in annotated
    ]
    result = audit.verify_production_audit_reconstruction(states, supervision_audit)
    assert result["checked"] is True
    assert result["row_derived"]["train_reason_counts"] == supervision_audit["train_reason_counts"]
    assert result["row_derived"]["target_class_counts"]["train_applicable_binary"] == supervision_audit["target_class_counts"]["train_applicable_binary"]
    assert result["row_derived"]["train_exclusion_counts"] == supervision_audit["train_exclusion_counts"]

def test_forced_production_audit_mismatch_rejects():
    states = _states(grammar="FAIL")
    row_counts = audit.row_reconstruction_counts(states)
    row_counts["train_reason_counts"]["AUTHORIZED"] += 1
    with pytest.raises(ValueError, match="P3W3_PRODUCTION_AUDIT_RECONSTRUCTION_MISMATCH"):
        audit.verify_production_audit_reconstruction(states, row_counts)


def test_dirty_tracked_tree_fails_closed(monkeypatch):
    def fake_git(args):
        if args == ["git", "rev-parse", "HEAD"]:
            return EXEC
        if args == ["git", "status", "--porcelain", "--untracked-files=no"]:
            return " M scripts/analyze_reason_router_p3w3_polarity_authority.py"
        raise AssertionError(args)
    monkeypatch.setattr(audit, "_git_output", fake_git)
    with pytest.raises(ValueError, match="P3W3_DIRTY_TRACKED_WORKING_TREE"):
        audit.validate_git_authority(EXEC)


def test_no_axis_authorized_refute_decides_new_authority_required():
    support = _record("p0__none", "p0", "support")
    states = audit.build_row_states(train_records=[support], dev_records=None, sidecar_by_id={support["id"]: _sidecar(support, support["id"])})
    decision, criteria = audit.decide(states, audit.counterfactual_results(states))
    assert decision == "P3W3_NEW_REFUTE_AUTHORITY_REQUIRED"
    assert criteria["axis_authorized_refute_count"] == 0


def test_all_axis_authorized_refute_already_eligible_blocks_as_not_reproduced():
    states = _states()
    decision, criteria = audit.decide(states, audit.counterfactual_results(states))
    assert decision == "P3W3_AUDIT_BLOCKED"
    assert criteria["reason"] == "polarity supervision blocker is not reproduced"


def test_missing_refute_subset_uses_evidence_classifier_decision():
    states = _states(proven=True)
    decision, criteria = audit.decide(states, audit.counterfactual_results(states))
    assert decision == "P3W3_EXISTING_AUTHORITY_REQUIRES_INTEGRITY_REANNOTATION"
    assert criteria["missing_axis_authorized_refute_count"] == 1


def test_pair_eligibility_and_evidence_cleanliness_are_counted_separately():
    states = _states(grammar="FAIL")
    states[0]["generator_evidence_class"] = audit.AMBIGUOUS_INTEGRITY_EVIDENCE
    pair = audit.pair_level_canonical_comparison(states)
    assert pair["pairs_with_eligible_canonical_SUPPORT_row"] == 1
    assert pair["pairs_with_generator_evidence_clean_canonical_SUPPORT_row"] == 0


def test_repository_default_proof_contract_is_unavailable():
    proof = audit.resolve_generator_evidence_proof_authority()
    assert proof["available"] is False
    assert proof["source"] is None
    source = inspect.getsource(audit)
    assert "PROVEN_POLARITY_REASON_CODES" not in source
    assert "REQUIRED_POLARITY_PRESERVED_AXES" not in source

def test_generator_status_none_is_ambiguous_not_independent():
    states = _states(grammar=None)
    assert states[1]["generator_evidence_class"] == audit.AMBIGUOUS_INTEGRITY_EVIDENCE


def test_generator_status_unknown_is_ambiguous_not_independent():
    states = _states(grammar="UNKNOWN")
    assert states[1]["generator_evidence_class"] == audit.AMBIGUOUS_INTEGRITY_EVIDENCE


def test_generator_status_non_string_is_ambiguous_not_independent():
    states = _states(grammar=0)
    assert states[1]["generator_evidence_class"] == audit.AMBIGUOUS_INTEGRITY_EVIDENCE


def test_generator_status_exact_fail_is_independent_defect():
    states = _states(grammar="FAIL")
    assert states[1]["generator_evidence_class"] == audit.INDEPENDENT_GENERATOR_OR_SEMANTIC_DEFECT_EVIDENCE


def test_intervention_contract_fail_only_with_no_proof_is_ambiguous():
    states = _states(proven=True)
    assert states[1]["generator_evidence_class"] == audit.AMBIGUOUS_INTEGRITY_EVIDENCE


def test_executed_summary_uses_runtime_blockers_not_preexecution_blockers():
    summary = _summary(_states(grammar="FAIL"))
    dumped = str(summary)
    assert summary["status"] == "P3W3_AUDIT_EXECUTION_COMPLETE_PENDING_RESULT_REVIEW"
    assert summary["audit_execution_completed"] is True
    assert summary["result_static_review_completed"] is False
    assert summary["production_behavior_modified"] is False
    assert summary["polarity_supervision_released"] is False
    assert summary["A1_A3_released"] is False
    assert "P3W3_POLARITY_AUTHORITY_AUDIT_NOT_EXECUTED" not in dumped
    assert "P3W3_RESULT_STATIC_REVIEW_PENDING" in summary["remaining_blockers"]


def _missing_refute_state_with_class(evidence_class: str):
    state = copy.deepcopy(_states(proven=True)[1])
    state["generator_evidence_class"] = evidence_class
    state["generator_evidence_reasons"] = ["synthetic decision test"]
    state["p2_polarity_applicable"] = False
    state["p2_polarity_target_2"] = -100
    return state


def test_decision_independent_only_requires_new_refute_authority():
    states = [_missing_refute_state_with_class(audit.INDEPENDENT_GENERATOR_OR_SEMANTIC_DEFECT_EVIDENCE)]
    decision, criteria = audit.decide(states, audit.counterfactual_results(states))
    assert decision == "P3W3_NEW_REFUTE_AUTHORITY_REQUIRED"
    assert criteria["required_remediation_classes"] == ["NEW_REFUTE_AUTHORITY"]


def test_decision_ambiguous_only_requires_integrity_reannotation():
    states = [_missing_refute_state_with_class(audit.AMBIGUOUS_INTEGRITY_EVIDENCE)]
    decision, criteria = audit.decide(states, audit.counterfactual_results(states))
    assert decision == "P3W3_EXISTING_AUTHORITY_REQUIRES_INTEGRITY_REANNOTATION"
    assert criteria["required_remediation_classes"] == ["INTEGRITY_REANNOTATION"]


def test_decision_proven_only_requires_narrow_contract_repair():
    states = [_missing_refute_state_with_class(audit.PROVEN_EXPECTED_POLARITY_INTERVENTION_MISMATCH_ONLY)]
    decision, criteria = audit.decide(states, audit.counterfactual_results(states))
    assert decision == "P3W3_EXISTING_AUTHORITY_RECOVERABLE_BY_NARROW_CONTRACT_REPAIR"
    assert criteria["required_remediation_classes"] == ["NARROW_CONTRACT_REPAIR"]


def test_decision_independent_and_ambiguous_is_mixed():
    states = [
        _missing_refute_state_with_class(audit.INDEPENDENT_GENERATOR_OR_SEMANTIC_DEFECT_EVIDENCE),
        _missing_refute_state_with_class(audit.AMBIGUOUS_INTEGRITY_EVIDENCE),
    ]
    decision, criteria = audit.decide(states, audit.counterfactual_results(states))
    assert decision == "P3W3_MIXED_REMEDIATION_REQUIRED"
    assert criteria["mixed_remediation_required"] is True


def test_decision_independent_and_proven_is_mixed():
    states = [
        _missing_refute_state_with_class(audit.INDEPENDENT_GENERATOR_OR_SEMANTIC_DEFECT_EVIDENCE),
        _missing_refute_state_with_class(audit.PROVEN_EXPECTED_POLARITY_INTERVENTION_MISMATCH_ONLY),
    ]
    decision, criteria = audit.decide(states, audit.counterfactual_results(states))
    assert decision == "P3W3_MIXED_REMEDIATION_REQUIRED"
    assert criteria["mixed_remediation_required"] is True


def test_decision_ambiguous_and_proven_is_mixed():
    states = [
        _missing_refute_state_with_class(audit.AMBIGUOUS_INTEGRITY_EVIDENCE),
        _missing_refute_state_with_class(audit.PROVEN_EXPECTED_POLARITY_INTERVENTION_MISMATCH_ONLY),
    ]
    decision, criteria = audit.decide(states, audit.counterfactual_results(states))
    assert decision == "P3W3_MIXED_REMEDIATION_REQUIRED"
    assert criteria["mixed_remediation_required"] is True


def test_decision_clean_missing_refute_is_audit_blocked():
    states = [_missing_refute_state_with_class(audit.CLEAN_GENERATOR_EVIDENCE)]
    decision, criteria = audit.decide(states, audit.counterfactual_results(states))
    assert decision == "P3W3_AUDIT_BLOCKED"
    assert criteria["reason"] == "clean missing REFUTE state is inconsistent with production eligibility"
