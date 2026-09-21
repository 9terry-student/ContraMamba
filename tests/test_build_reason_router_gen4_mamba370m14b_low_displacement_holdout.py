from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import (
    build_reason_router_gen4_mamba370m14b_low_displacement_holdout
    as m,
)


def test_constants() -> None:
    assert m.FIRST_PAIR == 5401
    assert m.LAST_PAIR == 5700
    assert m.PAIR_COUNT == 300
    assert m.ROW_COUNT == 1800
    assert m.TARGET_CELLS == ("C0_SHAM", "C2_NAME")
    assert m.SCALES == ("mamba370m", "mamba14b")
    assert m.BEHAVIORAL_ALPHAS == (0.5, 0.25)
    assert m.PRIMARY_ALPHA == 0.25
    assert m.SELECTED_PLANES == {
        "mamba370m": "P3",
        "mamba14b": "P5",
    }
    assert m.CONTROL_PLANES == {
        "mamba370m": "P5",
        "mamba14b": "P4",
    }


def test_population_is_deterministic_and_outcome_blind() -> None:
    facts_a, rows_a = m.build_population()
    facts_b, rows_b = m.build_population()

    assert m.jsonl_bytes(facts_a) == m.jsonl_bytes(facts_b)
    assert m.jsonl_bytes(rows_a) == m.jsonl_bytes(rows_b)
    assert len(facts_a) == 300
    assert len(rows_a) == 1800
    assert facts_a[0]["pair_id"] == "xg1_fact_5401"
    assert facts_a[-1]["pair_id"] == "xg1_fact_5700"
    assert len({str(row["claim"]) for row in rows_a}) == 300
    assert len({str(row["evidence"]) for row in rows_a}) == 1800

    for row in [*facts_a, *rows_a]:
        assert not (m.FORBIDDEN_OUTCOME_FIELDS & set(row))


def test_payload_boundary_and_freshness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        m,
        "prior_inventory_through_5400",
        lambda: (
            set(m.previous.expected_pair_ids(1, 5400)),
            {"prior_claim"},
            {"prior_evidence"},
        ),
    )

    payload = m.build_payload()
    manifest = payload["manifest"]

    assert manifest["result"] == m.RESULT
    assert manifest["pair_id_first"] == "xg1_fact_5401"
    assert manifest["pair_id_last"] == "xg1_fact_5700"
    assert manifest["prior_pair_range"] == "xg1_fact_001..xg1_fact_5400"
    assert manifest["pair_id_overlap_with_prior"] == 0
    assert manifest["claim_overlap_with_prior"] == 0
    assert manifest["evidence_overlap_with_prior"] == 0
    assert manifest["behavioral_alphas"] == [0.5, 0.25]
    assert manifest["primary_alpha"] == 0.25

    for key in (
        "response_fields_present",
        "endpoint_values_present",
        "selection_allowed",
        "cohort_replacement_allowed",
        "row_filtering_allowed",
        "alpha_sweep_allowed",
        "scale_specific_alpha_allowed",
        "layer_sweep_allowed",
        "token_sweep_allowed",
        "response_based_control_selection_allowed",
        "tokenizer_executed",
        "checkpoint_loaded",
        "model_executed",
        "cuda_executed",
    ):
        assert manifest[key] is False


def test_prior_inventory_extends_through_5400(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior_pairs = set(m.previous.expected_pair_ids(1, 5100))
    prior_claims = {"claim_prior"}
    prior_evidence = {"evidence_prior"}

    monkeypatch.setattr(
        m.previous,
        "prior_inventory_through_5100",
        lambda: (
            set(prior_pairs),
            set(prior_claims),
            set(prior_evidence),
        ),
    )

    facts, rows = m.build_population(5101, 5400)
    source_raw = m.jsonl_bytes(facts)
    row_raw = m.jsonl_bytes(rows)
    manifest = {
        "result": m.previous.RESULT,
        "pair_id_first": "xg1_fact_5101",
        "pair_id_last": "xg1_fact_5400",
        "source_file_sha256": m.sha256_bytes(source_raw),
        "row_file_sha256": m.sha256_bytes(row_raw),
    }

    by_path = {
        (m.PREVIOUS_DIR / m.SOURCE_FILE).as_posix(): source_raw,
        (m.PREVIOUS_DIR / m.ROW_FILE).as_posix(): row_raw,
        (m.PREVIOUS_DIR / m.previous.MANIFEST_FILE).as_posix():
            (json.dumps(manifest) + "\n").encode("utf-8"),
    }

    monkeypatch.setattr(
        m,
        "git_blob_bytes",
        lambda path: by_path[path.as_posix()],
    )

    pairs, claims, evidence = m.prior_inventory_through_5400()

    assert pairs == set(m.previous.expected_pair_ids(1, 5400))
    assert prior_claims <= claims
    assert prior_evidence <= evidence


def test_write_and_validate_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        m,
        "authenticate_repo",
        lambda: None,
    )
    monkeypatch.setattr(
        m,
        "prior_inventory_through_5400",
        lambda: (
            set(m.previous.expected_pair_ids(1, 5400)),
            {"prior_claim"},
            {"prior_evidence"},
        ),
    )

    out = tmp_path / "cohort"
    result = m.write_holdout(out)

    assert result["result"] == m.RESULT
    manifest = m.validate_written(out)

    assert manifest["source_pair_count"] == 300
    assert manifest["row_count"] == 1800
    assert manifest["target_cells"] == ["C0_SHAM", "C2_NAME"]
    assert manifest["scale_order"] == ["mamba370m", "mamba14b"]

    sums = (
        out / m.CHECKSUM_FILE
    ).read_text(encoding="utf-8")
    assert m.SOURCE_FILE in sums
    assert m.ROW_FILE in sums
    assert m.MANIFEST_FILE in sums
