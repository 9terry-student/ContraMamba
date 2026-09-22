from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import build_reason_router_gen4_mamba1_five_scale_ladder_holdouts as subject


def _inventory(payload):
    facts = subject._parse_jsonl_bytes(payload["source"], "SOURCE")
    rows = subject._parse_jsonl_bytes(payload["rows"], "ROWS")
    return subject._sets_for_rows(facts, rows)


def test_ranges_and_execution_order_are_exact() -> None:
    assert subject.COHORTS == (
        ("mamba790m", "discovery", 6901, 7200),
        ("mamba790m", "confirmation", 7201, 7500),
        ("mamba790m", "readout", 7501, 7800),
        ("mamba28b", "discovery", 6001, 6300),
        ("mamba28b", "confirmation", 6301, 6600),
        ("mamba28b", "readout", 6601, 6900),
    )


def test_prior_inventory_extends_exactly_through_6000(monkeypatch: pytest.MonkeyPatch) -> None:
    prior = (
        set(subject.expected_pair_ids(1, 5700)),
        {"prior_claim"},
        {"prior_evidence"},
    )
    monkeypatch.setattr(subject.previous, "prior_inventory_through_5700", lambda: prior)
    facts, rows = subject.previous.build_population(5701, 6000)
    source_raw = subject.previous.jsonl_bytes(facts)
    row_raw = subject.previous.jsonl_bytes(rows)
    manifest = {
        "result": subject.previous.RESULT,
        "pair_id_first": "xg1_fact_5701",
        "pair_id_last": "xg1_fact_6000",
        "source_file_sha256": subject.sha256_bytes(source_raw),
        "row_file_sha256": subject.sha256_bytes(row_raw),
    }
    by_path = {
        (subject.PREVIOUS_DIR / subject.SOURCE_FILE).as_posix(): source_raw,
        (subject.PREVIOUS_DIR / subject.ROW_FILE).as_posix(): row_raw,
        (subject.PREVIOUS_DIR / subject.MANIFEST_FILE).as_posix():
            subject.canonical_manifest_bytes(manifest),
    }
    monkeypatch.setattr(subject, "PREVIOUS_SOURCE_SHA256", subject.sha256_bytes(source_raw))
    monkeypatch.setattr(subject, "PREVIOUS_ROWS_SHA256", subject.sha256_bytes(row_raw))
    monkeypatch.setattr(
        subject, "PREVIOUS_MANIFEST_SHA256",
        subject.sha256_bytes(subject.canonical_manifest_bytes(manifest))
    )
    monkeypatch.setattr(subject, "git_blob_bytes", lambda path: by_path[path.as_posix()])
    pairs, claims, evidence = subject.prior_inventory_through_6000()
    assert pairs == set(subject.expected_pair_ids(1, 6000))
    assert "prior_claim" in claims
    assert "prior_evidence" in evidence


def test_all_six_payloads_are_deterministic_disjoint_and_outcome_blind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        subject,
        "prior_inventory_through_6000",
        lambda: (
            set(subject.expected_pair_ids(1, 6000)),
            {"prior_claim"},
            {"prior_evidence"},
        ),
    )
    a = subject.build_holdout_payloads()
    b = subject.build_holdout_payloads()
    assert a == b
    inventories = []
    for key, payload in a.items():
        manifest = payload["manifest"]
        assert manifest["labels_present"] is False
        assert manifest["response_fields_present"] is False
        assert manifest["endpoint_values_present"] is False
        assert manifest["tokenizer_executed"] is False
        assert manifest["checkpoint_loaded"] is False
        assert manifest["model_executed"] is False
        assert manifest["cuda_executed"] is False
        assert manifest["cohort_replacement_allowed"] is False
        json.dumps(manifest, sort_keys=True, allow_nan=False)
        inventories.append(_inventory(payload))
    for i, left in enumerate(inventories):
        for right in inventories[i + 1:]:
            assert not (left[0] & right[0])
            assert not (left[1] & right[1])
            assert not (left[2] & right[2])


def test_role_boundaries_are_prospective(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subject,
        "prior_inventory_through_6000",
        lambda: (
            set(subject.expected_pair_ids(1, 6000)),
            {"prior_claim"},
            {"prior_evidence"},
        ),
    )
    payloads = subject.build_holdout_payloads()
    d = payloads[("mamba790m", "discovery")]["manifest"]
    c = payloads[("mamba790m", "confirmation")]["manifest"]
    r = payloads[("mamba790m", "readout")]["manifest"]
    assert d["selection_allowed"] is True
    assert d["confirmation_response_access_allowed"] is False
    assert d["readout_response_access_allowed"] is False
    assert c["selection_allowed"] is False
    assert c["planned_primary_test"]["n"] == 300
    assert c["readout_response_access_allowed"] is False
    assert r["selection_allowed"] is False
    assert r["selected_component_fixed_before_readout"] is True
    assert r["response_blind_control_fixed_before_readout"] is True


def test_write_round_trip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(subject, "authenticate_repo", lambda: None)
    monkeypatch.setattr(
        subject,
        "prior_inventory_through_6000",
        lambda: (
            set(subject.expected_pair_ids(1, 6000)),
            {"prior_claim"},
            {"prior_evidence"},
        ),
    )
    monkeypatch.setattr(
        subject,
        "OUTPUT_DIRS",
        {key: Path(key[0] + "_" + key[1]) for key in subject.OUTPUT_DIRS},
    )
    monkeypatch.setattr(subject, "ROOT", tmp_path)
    result = subject.write_holdouts()
    assert result["result"] == subject.RESULT
    assert len(result["cohorts"]) == 6
