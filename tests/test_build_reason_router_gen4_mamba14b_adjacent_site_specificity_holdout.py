from __future__ import annotations

import json

from scripts import (
    build_reason_router_gen4_mamba14b_adjacent_site_specificity_holdout
    as m,
)


def test_constants() -> None:
    assert m.FIRST_PAIR == 5101
    assert m.LAST_PAIR == 5400
    assert m.PAIR_COUNT == 300
    assert m.ROW_COUNT == 1800
    assert m.CANONICAL_TRIPLET == (33, 34, 35)
    assert m.ADJACENT_TRIPLET == (34, 35, 36)
    assert m.SELECTED_CAUSAL_CANDIDATE == "P5"
    assert m.EPSILON == 0.025


def test_population_is_deterministic_and_outcome_blind() -> None:
    facts_a, rows_a = m.build_population()
    facts_b, rows_b = m.build_population()

    assert m.jsonl_bytes(facts_a) == m.jsonl_bytes(facts_b)
    assert m.jsonl_bytes(rows_a) == m.jsonl_bytes(rows_b)
    assert len(facts_a) == 300
    assert len(rows_a) == 1800
    assert facts_a[0]["pair_id"] == "xg1_fact_5101"
    assert facts_a[-1]["pair_id"] == "xg1_fact_5400"

    for row in [*facts_a, *rows_a]:
        assert not (m.FORBIDDEN_OUTCOME_FIELDS & set(row))


def test_payload_freshness_and_boundary() -> None:
    payload = m.build_payload()
    manifest = payload["manifest"]

    assert manifest["result"] == m.RESULT
    assert manifest["pair_id_first"] == "xg1_fact_5101"
    assert manifest["pair_id_last"] == "xg1_fact_5400"
    assert manifest["prior_pair_range"] == "xg1_fact_001..xg1_fact_5100"
    assert manifest["pair_id_overlap_with_prior"] == 0
    assert manifest["claim_overlap_with_prior"] == 0
    assert manifest["evidence_overlap_with_prior"] == 0
    assert manifest["canonical_triplet"] == [33, 34, 35]
    assert manifest["adjacent_triplet"] == [34, 35, 36]
    assert manifest["selected_causal_candidate_frozen"] == "P5"

    for key in (
        "response_fields_present",
        "endpoint_values_present",
        "selection_allowed",
        "layer_sweep_allowed",
        "second_adjacent_site_allowed",
        "token_sweep_allowed",
        "epsilon_sweep_allowed",
        "response_based_control_selection_allowed",
        "tokenizer_executed",
        "checkpoint_loaded",
        "model_executed",
        "cuda_executed",
    ):
        assert manifest[key] is False


def test_write_and_validate_round_trip(tmp_path) -> None:
    out = tmp_path / "cohort"
    result = m.write_holdout(out)
    assert result["result"] == m.RESULT
    manifest = m.validate_written(out)
    assert manifest["source_pair_count"] == 300
    assert manifest["row_count"] == 1800

    sums = (out / m.CHECKSUM_FILE).read_text(encoding="utf-8")
    assert m.SOURCE_FILE in sums
    assert m.ROW_FILE in sums
    assert m.MANIFEST_FILE in sums

    parsed = json.loads((out / m.MANIFEST_FILE).read_text(encoding="utf-8"))
    assert parsed == manifest
