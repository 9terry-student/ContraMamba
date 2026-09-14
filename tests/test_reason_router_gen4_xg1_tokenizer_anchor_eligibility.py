from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import reason_router_gen4_xg1_tokenizer_anchor_eligibility as gate


class FakeTokenizer:
    def __init__(self, mapping):
        self.mapping = mapping

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        ids, offsets = self.mapping[text]
        return SimpleNamespace(ids=list(ids), offsets=list(offsets))


def sample_fact():
    return {
        "schema_version": "GEN4_XG1_STRUCTURED_SOURCE_FACT_V1",
        "generator_family": "xg1_independent_structured_records_v1",
        "pair_id": "xg1_fact_001",
        "title": "Envoy",
        "name": "Talia Voss",
        "role": "archive custodian",
        "predicate": "certified",
        "object": "the Quasar registry unit 001",
        "time": "the first winter week",
        "location": "Juniper Reach",
        "alternate_title": "Warden",
        "alternate_name": "Damon Rusk",
        "alternate_role": "program marshal",
        "alternate_predicate": "reclassified",
    }


def _simple_word_tokenization(text):
    ids = []
    offsets = []
    cursor = 0
    token_id = 1
    for word in text.split(" "):
        start = text.index(word, cursor)
        end = start + len(word)
        ids.append(token_id)
        offsets.append((start, end))
        token_id += 1
        cursor = end
    return ids, offsets


def test_frozen_contract_constants():
    assert gate.STRUCTURAL_FREEZE_COMMIT == (
        "d9029801fd47636c155b1c846c433fc561424c8f"
    )
    assert gate.EXPECTED_SOURCE_FACTS_SHA256 == (
        "fccd6821eeb97194d5b898aca4911eaba71e893df7fe27c910aa37255a5695e0"
    )
    assert gate.EXPECTED_ROWS_SHA256 == (
        "6ea0484517e0ae7479ad7f3b0a74af4d75f7f7353d29586c597f2a9fee1e649f"
    )
    assert gate.EXPECTED_STRUCTURAL_MANIFEST_SHA256 == (
        "f7c881dd1a4a400e600eea4b03b46e05a48bf0da07d29905462fc3543d8af822"
    )
    assert gate.XG1_BUILDER_BLOB == "c830026935a6c9f4990c6a3315c75fd5580e7264"
    assert gate.CANONICAL_ANALYSIS_TOKENIZER_REFERENCE == (
        "40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37"
    )
    assert gate.EXPECTED_TOKENIZERS_VERSION == "0.22.2"
    assert gate.CLAIM_BUDGET == 63
    assert gate.EVIDENCE_BUDGET == 64
    assert gate.EOS_TOKEN_ID == 0
    assert gate.MAX_LENGTH == 128


def test_anchor_plan_matches_frozen_prospective_contract():
    assert sum(len(v) for v in gate.ANCHOR_PLAN.values()) == 6
    assert gate.EXPECTED_ANCHOR_ROWS == 1800
    assert gate.ANCHOR_EXPECTED_COUNTS == {
        "A_IDENTITY": 1200,
        "A_NAME": 600,
    }
    assert gate.ANCHOR_PLAN["C0_SHAM"] == ("A_IDENTITY", "A_NAME")
    assert gate.ANCHOR_PLAN["C1_TITLE"] == ("A_IDENTITY",)
    assert gate.ANCHOR_PLAN["C2_NAME"] == ("A_IDENTITY", "A_NAME")
    assert gate.ANCHOR_PLAN["C5_TITLE_NAME"] == ("A_IDENTITY",)
    assert gate.ANCHOR_PLAN["C3_ROLE"] == ()
    assert gate.ANCHOR_PLAN["C4_PREDICATE"] == ()


def test_xg1_realized_statement_and_spans_follow_new_renderer():
    fact = sample_fact()
    rendered, spans = gate.realized_statement_and_spans(
        fact,
        {"title": fact["alternate_title"], "name": fact["alternate_name"]},
    )
    assert rendered.startswith(
        "During the first winter week, records from Juniper Reach identify "
        "Warden Damon Rusk as archive custodian; this person certified "
    )
    assert rendered[slice(*spans["A_TITLE"])] == "Warden"
    assert rendered[slice(*spans["A_NAME"])] == "Damon Rusk"
    assert rendered[slice(*spans["A_ROLE"])] == "archive custodian"
    assert rendered[slice(*spans["A_PREDICATE"])] == "certified"
    assert rendered[slice(*spans["A_IDENTITY"])] == "Warden Damon Rusk"
    assert spans["A_IDENTITY"][1] == spans["A_NAME"][1]


def test_overrides_are_exactly_xg1_cell_spec():
    fact = sample_fact()
    assert gate._overrides_for_cell(fact, "C0_SHAM") == {}
    assert gate._overrides_for_cell(fact, "C1_TITLE") == {
        "title": fact["alternate_title"]
    }
    assert gate._overrides_for_cell(fact, "C2_NAME") == {
        "name": fact["alternate_name"]
    }
    assert gate._overrides_for_cell(fact, "C5_TITLE_NAME") == {
        "title": fact["alternate_title"],
        "name": fact["alternate_name"],
    }


def test_final_overlapping_token_accepts_leading_space_offset():
    offsets = [(0, 2), (2, 8), (8, 14)]
    index, error = gate.final_overlapping_token_index(offsets, 3, 14)
    assert index == 2
    assert error is None


def test_final_overlapping_token_rejects_incomplete_span():
    index, error = gate.final_overlapping_token_index(
        [(5, 8), (8, 10)],
        4,
        10,
    )
    assert index is None
    assert error == "SPAN_MAPPING_INCOMPLETE"


def test_final_overlapping_token_rejects_missing_span():
    index, error = gate.final_overlapping_token_index([(0, 2)], 3, 5)
    assert index is None
    assert error == "SPAN_NOT_MAPPED"


def test_row_analysis_maps_xg1_anchor_and_preserves_identity_name_terminal():
    fact = sample_fact()
    evidence, _ = gate.realized_statement_and_spans(fact, {})
    claim = "A compact claim for testing"
    claim_ids, claim_offsets = _simple_word_tokenization(claim)
    evidence_ids, evidence_offsets = _simple_word_tokenization(evidence)

    tokenizer = FakeTokenizer({
        claim: (claim_ids, claim_offsets),
        evidence: (evidence_ids, evidence_offsets),
    })
    row = {
        "source_pair_id": fact["pair_id"],
        "row_id": fact["pair_id"] + "__masked_slot_substitution_v1__c0_sham",
        "contrast_cell_id": "C0_SHAM",
        "claim": claim,
        "evidence": evidence,
    }

    results = gate.analyze_required_anchors_for_row(row, fact, tokenizer)
    assert [r["anchor_name"] for r in results] == ["A_IDENTITY", "A_NAME"]
    assert results[0]["absolute_anchor_token_index"] == (
        results[1]["absolute_anchor_token_index"]
    )
    for result in results:
        assert result["post4_rule"] == "a+4 <= terminal_index-1"
        assert result["absolute_anchor_token_index"] is not None


def test_post4_ineligibility_is_not_shortened_or_rescued():
    fact = sample_fact()
    overrides = {"name": fact["alternate_name"]}
    evidence, spans = gate.realized_statement_and_spans(fact, overrides)
    claim = "claim"

    identity_start, identity_end = spans["A_IDENTITY"]
    name_start, _name_end = spans["A_NAME"]
    tokenizer = FakeTokenizer({
        claim: ([1], [(0, len(claim))]),
        evidence: (
            [10, 11],
            [
                (identity_start, name_start),
                (name_start, identity_end),
            ],
        ),
    })
    row = {
        "source_pair_id": fact["pair_id"],
        "row_id": "target-row",
        "contrast_cell_id": "C2_NAME",
        "claim": claim,
        "evidence": evidence,
    }

    results = gate.analyze_required_anchors_for_row(row, fact, tokenizer)
    assert [r["anchor_name"] for r in results] == ["A_IDENTITY", "A_NAME"]
    for result in results:
        assert result["post4_eligible"] is False
        assert result["exclusion_code"] == "POST4_PREFIX_INELIGIBLE"
        assert result["post4_end_index"] == (
            result["absolute_anchor_token_index"] + 4
        )


def test_anchor_not_consumed_after_evidence_truncation_is_ineligible():
    fact = sample_fact()
    evidence, spans = gate.realized_statement_and_spans(
        fact,
        {"title": fact["alternate_title"]},
    )
    claim = "claim"
    filler = [(0, 1)] * 64
    identity_offset = spans["A_IDENTITY"]
    offsets = filler + [identity_offset]
    tokenizer = FakeTokenizer({
        claim: ([1], [(0, len(claim))]),
        evidence: (list(range(100, 165)), offsets),
    })
    row = {
        "source_pair_id": fact["pair_id"],
        "row_id": "identity-row",
        "contrast_cell_id": "C1_TITLE",
        "claim": claim,
        "evidence": evidence,
    }
    result = gate.analyze_required_anchors_for_row(row, fact, tokenizer)[0]
    assert result["anchor_evidence_token_index"] == 64
    assert result["absolute_anchor_token_index"] is None
    assert result["post4_eligible"] is False
    assert result["exclusion_code"] == (
        "ANCHOR_NOT_CONSUMED_AFTER_EVIDENCE_TRUNCATION"
    )


def test_structural_manifest_contract_rejects_tokenizer_boundary_drift():
    manifest = {
        "schema_version": "GEN4_XG1_STRUCTURAL_MANIFEST_V1",
        "result": "PASS_XG1_STRUCTURAL_INDEPENDENCE",
        "design_freeze_commit": gate.DESIGN_FREEZE_COMMIT,
        "design_sha256":
            "21ce2dd133721766a9a18d924774b8f02aa026364e1102809ea839c3985bf403",
        "generator_family": "xg1_independent_structured_records_v1",
        "source_pair_count": 300,
        "row_count": 1800,
        "rows_per_pair": 6,
        "pair_id_first": "xg1_fact_001",
        "pair_id_last": "xg1_fact_300",
        "source_file_sha256": gate.EXPECTED_SOURCE_FACTS_SHA256,
        "row_file_sha256": gate.EXPECTED_ROWS_SHA256,
        "exact_inventory_overlap_count": 0,
        "casefold_inventory_overlap_count": 0,
        "discovery_claim_overlap_count": 0,
        "discovery_evidence_overlap_count": 0,
        "prior_holdout_claim_overlap_count": 0,
        "prior_holdout_evidence_overlap_count": 0,
        "deterministic_byte_regeneration": True,
        "production_builder_uses_historical_generator": False,
        "labels_present": False,
        "model_geometry_present": False,
        "endpoint_values_present": False,
        "response_fields_present": False,
        "tokenizer_executed": False,
        "model_executed": False,
        "cuda_executed": False,
    }
    gate.validate_structural_manifest(manifest)
    manifest["model_executed"] = True
    with pytest.raises(gate.EligibilityError, match="model_executed"):
        gate.validate_structural_manifest(manifest)


def test_anchor_manifest_serialization_is_deterministic():
    rows = [{
        "schema_version": gate.ANCHOR_MANIFEST_SCHEMA,
        "source_pair_id": "p",
        "row_id": "r",
        "contrast_cell_id": "C0_SHAM",
        "anchor_name": "A_NAME",
        "post4_eligible": True,
    }]
    first = gate.serialize_anchor_manifest(rows)
    second = gate.serialize_anchor_manifest(rows)
    assert first == second
    assert first.endswith(b"\n")
    assert gate.sha256_bytes(first) == gate.sha256_bytes(second)


def test_repository_frozen_xg1_inputs_are_exact():
    root = Path(__file__).resolve().parents[1]
    if not (root / gate.ROWS_PATH).is_file():
        pytest.skip("frozen XG1 cohort not present")

    facts, rows, manifest = gate.load_frozen_xg1_inputs(root)
    assert len(facts) == 300
    assert len(rows) == 1800
    assert manifest["result"] == "PASS_XG1_STRUCTURAL_INDEPENDENCE"


def test_full_canonical_tokenizer_gate_when_snapshot_is_available():
    root = Path(__file__).resolve().parents[1]
    snapshot = gate.canonical_tokenizer_snapshot_dir()
    if not snapshot.is_dir():
        pytest.skip("canonical tokenizer snapshot not locally available")

    anchor_rows, summary = gate.compute_eligibility(
        root=root,
        tokenizer_snapshot=snapshot,
    )
    assert len(anchor_rows) == 1800
    assert summary["model_forward_count"] == 0
    assert summary["checkpoint_load_count"] == 0
    assert summary["gpu_used"] is False
    assert summary["required_anchor_counts"] == gate.ANCHOR_EXPECTED_COUNTS
    assert summary["primary_complete_pair_prefix_feasibility"] in {
        "PASS_300_OF_300",
        "BLOCKED_TOKENIZER_ANCHOR_INELIGIBILITY",
    }


def test_write_outputs_refuses_collision(tmp_path):
    rows = [{
        "schema_version": gate.ANCHOR_MANIFEST_SCHEMA,
        "source_pair_id": "p",
        "row_id": "r",
        "contrast_cell_id": "C0_SHAM",
        "anchor_name": "A_NAME",
        "post4_eligible": True,
    }]
    raw = gate.serialize_anchor_manifest(rows)
    summary = {"anchor_manifest_sha256": gate.sha256_bytes(raw)}
    out = tmp_path / "anchors.jsonl"
    report = tmp_path / "summary.json"

    gate.write_outputs(
        anchor_rows=rows,
        summary=summary,
        output_jsonl=out,
        summary_path=report,
    )
    assert json.loads(report.read_text(encoding="utf-8"))[
        "anchor_manifest_sha256"
    ] == gate.sha256_bytes(raw)

    with pytest.raises(gate.EligibilityError, match="OUTPUT_COLLISION"):
        gate.write_outputs(
            anchor_rows=rows,
            summary=summary,
            output_jsonl=out,
            summary_path=tmp_path / "other.json",
        )


def test_gate_source_has_no_model_or_cuda_dependency():
    source = Path(gate.__file__).read_text(encoding="utf-8")
    for token in (
        "import torch",
        "torch.cuda",
        "load_representative_model",
        "selected_checkpoint.pt",
        "capture_branch(",
    ):
        assert token not in source
