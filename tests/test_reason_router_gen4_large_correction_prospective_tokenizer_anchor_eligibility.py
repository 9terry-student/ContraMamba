from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import (
    reason_router_gen4_large_correction_prospective_tokenizer_anchor_eligibility
    as gate,
)


class FakeTokenizer:
    def __init__(self, mapping):
        self.mapping = mapping

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        ids, offsets = self.mapping[text]
        return SimpleNamespace(ids=list(ids), offsets=list(offsets))


class FakeMaterializer:
    @staticmethod
    def _overrides_for_cell(fact, cell_id):
        mapping = {
            "C0_SHAM": {},
            "C1_TITLE": {"title": fact["alternate_title"]},
            "C2_NAME": {"name": fact["alternate_name"]},
            "C3_ROLE": {"role": fact["alternate_role"]},
            "C4_PREDICATE": {"predicate": fact["alternate_predicate"]},
            "C5_TITLE_NAME": {
                "title": fact["alternate_title"],
                "name": fact["alternate_name"],
            },
        }
        return mapping[cell_id]


def sample_fact():
    return {
        "pair_id": "generated_fact_301",
        "title": "Dr",
        "name": "Avery Stone",
        "alternate_title": "Professor",
        "alternate_name": "Farah Qureshi",
        "role": "program director",
        "alternate_role": "operations chief",
        "predicate": "approved",
        "alternate_predicate": "reviewed",
        "object": "the Meridian initiative 301",
        "time": "January",
        "location": "Aurora City",
    }


def test_frozen_contract_constants():
    assert gate.PROSPECTIVE_HOLDOUT_FREEZE_COMMIT == (
        "c5abe9e3fd551c49ab67f68e2cd8fb3100a7c91e"
    )
    assert gate.EXPECTED_PROSPECTIVE_ROWS_SHA256 == (
        "f4289173ba3837fad217728830b8aefb8baa7dce4f45dde7c58441021b436bdc"
    )
    assert gate.CANONICAL_ANALYSIS_TOKENIZER_REFERENCE == (
        "40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37"
    )
    assert gate.CANONICAL_ANALYSIS_TOKENIZER_IS_CLAIMED_HISTORICAL_SNAPSHOT is False
    assert gate.EXPECTED_TOKENIZERS_VERSION == "0.22.2"
    assert gate.CLAIM_BUDGET == 63
    assert gate.EVIDENCE_BUDGET == 64
    assert gate.EOS_TOKEN_ID == 0
    assert gate.MAX_LENGTH == 128


def test_anchor_plan_matches_closed_transport_event_contract():
    per_pair = sum(len(v) for v in gate.ANCHOR_PLAN.values())
    assert per_pair == 6
    assert gate.EXPECTED_ANCHOR_ROWS == 300 * 6
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


def test_realized_statement_and_spans_are_structural_not_substring_search():
    fact = sample_fact()
    rendered, spans = gate.realized_statement_and_spans(
        fact,
        {"title": fact["alternate_title"], "name": fact["alternate_name"]},
    )
    assert rendered.startswith("Professor Farah Qureshi, the program director, approved ")
    assert rendered[slice(*spans["A_TITLE"])] == "Professor"
    assert rendered[slice(*spans["A_NAME"])] == "Farah Qureshi"
    assert rendered[slice(*spans["A_ROLE"])] == "program director"
    assert rendered[slice(*spans["A_PREDICATE"])] == "approved"
    assert rendered[slice(*spans["A_IDENTITY"])] == "Professor Farah Qureshi"
    assert spans["A_IDENTITY"][1] == spans["A_NAME"][1]


def test_final_overlapping_token_accepts_leading_space_offset():
    offsets = [(0, 2), (2, 8), (8, 14)]
    index, error = gate.final_overlapping_token_index(offsets, 3, 14)
    assert index == 2
    assert error is None


def test_final_overlapping_token_rejects_incomplete_span_coverage():
    offsets = [(5, 8), (8, 10)]
    index, error = gate.final_overlapping_token_index(offsets, 4, 10)
    assert index is None
    assert error == "SPAN_MAPPING_INCOMPLETE"


def test_final_overlapping_token_rejects_missing_span():
    index, error = gate.final_overlapping_token_index([(0, 2)], 3, 5)
    assert index is None
    assert error == "SPAN_NOT_MAPPED"


def test_validate_materialization_manifest_rejects_outcome_boundary_drift():
    manifest = {
        "schema_version": "GEN4_LARGE_CORRECTION_PROSPECTIVE_COHORT_V1",
        "phase": "PROSPECTIVE_HOLDOUT_MATERIALIZATION",
        "branch": gate.EXPECTED_BRANCH,
        "discovery_freeze_commit": gate.DISCOVERY_FREEZE_COMMIT,
        "generator_source_blob": gate.GENERATOR_SOURCE_BLOB,
        "materializer_source_blob": gate.MATERIALIZER_SOURCE_BLOB,
        "mechanism_id": "masked_slot_substitution_v1",
        "validation_pair_count": 300,
        "validation_row_count": 1800,
        "first_validation_pair_id": "generated_fact_301",
        "last_validation_pair_id": "generated_fact_600",
        "rows_sha256": gate.EXPECTED_PROSPECTIVE_ROWS_SHA256,
        "validation_pair_ids_sha256": gate.EXPECTED_VALIDATION_PAIR_IDS_SHA256,
        "tokenizer_invoked": False,
        "model_forward_count": 0,
        "training_executed": False,
        "evaluation_executed": False,
        "scientific_outcomes_observed": False,
        "validation_pair_index_range": [300, 600],
        "discovery_pair_index_range": [0, 300],
        "contrast_cells": list(gate.CANONICAL_CELLS),
    }
    gate.validate_materialization_manifest(manifest)
    manifest["model_forward_count"] = 1
    with pytest.raises(gate.EligibilityError, match="model_forward_count"):
        gate.validate_materialization_manifest(manifest)


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


def test_row_analysis_maps_anchor_into_serialized_coordinate_and_applies_post4():
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

    results = gate.analyze_required_anchors_for_row(
        row,
        fact,
        FakeMaterializer,
        tokenizer,
    )
    assert [r["anchor_name"] for r in results] == ["A_IDENTITY", "A_NAME"]
    for result in results:
        assert result["absolute_anchor_token_index"] is not None
        assert result["post4_rule"] == "a+4 <= terminal_index-1"
    assert results[0]["evidence_start"] == len(claim_ids) + 1


def test_post4_ineligibility_is_not_shortened_or_rescued():
    fact = sample_fact()
    overrides = {"name": fact["alternate_name"]}
    evidence, spans = gate.realized_statement_and_spans(fact, overrides)
    claim = "claim"
    # Make the common identity/name end the final consumed token.
    identity_end = spans["A_IDENTITY"][1]
    name_start = spans["A_NAME"][0]
    tokenizer = FakeTokenizer({
        claim: ([1], [(0, len(claim))]),
        evidence: (
            [10, 11],
            [
                (spans["A_IDENTITY"][0], name_start),
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
    results = gate.analyze_required_anchors_for_row(
        row,
        fact,
        FakeMaterializer,
        tokenizer,
    )
    assert [r["anchor_name"] for r in results] == ["A_IDENTITY", "A_NAME"]
    for result in results:
        assert result["post4_eligible"] is False
        assert result["exclusion_code"] == "POST4_PREFIX_INELIGIBLE"
        assert result["post4_end_index"] == result["absolute_anchor_token_index"] + 4
    assert (
        results[0]["absolute_anchor_token_index"]
        == results[1]["absolute_anchor_token_index"]
    )

def test_anchor_not_consumed_after_evidence_truncation_is_ineligible():
    fact = sample_fact()
    overrides = {"title": fact["alternate_title"]}
    evidence, spans = gate.realized_statement_and_spans(fact, overrides)
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
    result = gate.analyze_required_anchors_for_row(
        row,
        fact,
        FakeMaterializer,
        tokenizer,
    )[0]
    assert result["anchor_name"] == "A_IDENTITY"
    assert result["anchor_evidence_token_index"] == 64
    assert result["absolute_anchor_token_index"] is None
    assert result["post4_eligible"] is False
    assert result["exclusion_code"] == "ANCHOR_NOT_CONSUMED_AFTER_EVIDENCE_TRUNCATION"

def test_anchor_manifest_serialization_is_deterministic():
    rows = [
        {
            "schema_version": gate.ANCHOR_MANIFEST_SCHEMA,
            "source_pair_id": "p",
            "row_id": "r",
            "contrast_cell_id": "C0_SHAM",
            "anchor_name": "A_NAME",
            "post4_eligible": True,
        }
    ]
    first = gate.serialize_anchor_manifest(rows)
    second = gate.serialize_anchor_manifest(rows)
    assert first == second
    assert first.endswith(b"\n")
    assert gate.sha256_bytes(first) == gate.sha256_bytes(second)


def test_repository_frozen_holdout_re_materializes_byte_exact():
    root = Path(__file__).resolve().parents[1]
    if not (root / gate.PROSPECTIVE_DATA_PATH).is_file():
        pytest.skip("repository prospective holdout is not present")
    gate.authenticate_repository(root)
    rows, raw, _manifest = gate.load_frozen_prospective_rows(root)
    facts, regenerated, _materializer = (
        gate.reconstruct_validation_facts_and_verify_bytes(raw, root=root)
    )
    assert len(rows) == 1800
    assert len(facts) == 300
    assert len(regenerated) == 1800


def test_full_canonical_tokenizer_gate_when_snapshot_is_available():
    root = Path(__file__).resolve().parents[1]
    if not (root / gate.PROSPECTIVE_DATA_PATH).is_file():
        pytest.skip("repository prospective holdout is not present")
    snapshot = gate.canonical_tokenizer_snapshot_dir()
    if not snapshot.is_dir():
        pytest.skip("canonical tokenizer snapshot is not locally available")

    anchor_rows, summary = gate.compute_eligibility(
        root=root,
        tokenizer_snapshot=snapshot,
    )
    assert len(anchor_rows) == 1800
    assert summary["model_forward_count"] == 0
    assert summary["checkpoint_load_count"] == 0
    assert summary["gpu_used"] is False
    assert summary["required_anchor_counts"] == gate.ANCHOR_EXPECTED_COUNTS
    assert summary["primary_complete_pair_prefix_feasibility"] == "PASS_300_OF_300"
    assert summary["complete_source_pair_count"] == 300
    assert summary["exclusion_counts"] == {}
    assert summary["target_identity_name_mismatch_count"] == 0


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
    assert json.loads(report.read_text(encoding="utf-8"))["anchor_manifest_sha256"] == (
        gate.sha256_bytes(raw)
    )
    with pytest.raises(gate.EligibilityError, match="OUTPUT_COLLISION"):
        gate.write_outputs(
            anchor_rows=rows,
            summary=summary,
            output_jsonl=out,
            summary_path=tmp_path / "other.json",
        )
