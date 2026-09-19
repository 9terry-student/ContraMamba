from __future__ import annotations

import json
from pathlib import Path

from scripts import build_reason_router_gen4_mamba14b_xg1_holdouts as subject


def _inventory(payload):
    facts = subject.m370.prior.read_jsonl_bytes(
        payload["source"],
        "TEST_SOURCE",
    )
    rows = subject.m370.prior.read_jsonl_bytes(
        payload["rows"],
        "TEST_ROWS",
    )
    return subject._sets_for_rows(facts, rows)


def test_ranges_are_exact_and_outcome_blind():
    specs = (
        (3901, 4200),
        (4201, 4500),
        (4501, 4800),
    )
    for first, last in specs:
        facts, rows = subject.build_population(first, last)
        assert facts[0]["pair_id"] == f"xg1_fact_{first:03d}"
        assert facts[-1]["pair_id"] == f"xg1_fact_{last:03d}"
        assert len(facts) == 300
        assert len(rows) == 1800

        for row in [*facts, *rows]:
            assert not (
                subject.FORBIDDEN_OUTCOME_FIELDS & set(row)
            )


def test_prior_inventory_is_exact_through_3900():
    pairs, claims, evidence = (
        subject.prior_inventory_through_3900()
    )
    assert pairs == set(subject.expected_pair_ids(1, 3900))
    assert len(pairs) == 3900
    assert claims
    assert evidence


def test_all_three_payloads_are_deterministic_and_disjoint():
    a = subject.build_holdout_payloads()
    b = subject.build_holdout_payloads()

    assert a == b

    d = _inventory(a["discovery"])
    c = _inventory(a["confirmation"])
    r = _inventory(a["residual_characterization"])

    for left, right in ((d, c), (d, r), (c, r)):
        assert not (left[0] & right[0])
        assert not (left[1] & right[1])
        assert not (left[2] & right[2])

    prior = subject.prior_inventory_through_3900()
    for current in (d, c, r):
        assert not (prior[0] & current[0])
        assert not (prior[1] & current[1])
        assert not (prior[2] & current[2])

    all_pairs = prior[0] | d[0] | c[0] | r[0]
    assert all_pairs == set(subject.expected_pair_ids(1, 4800))


def test_manifest_role_boundaries_are_prospective():
    payloads = subject.build_holdout_payloads()

    d = payloads["discovery"]["manifest"]
    assert d["role"] == "discovery"
    assert d["pair_id_first"] == "xg1_fact_3901"
    assert d["pair_id_last"] == "xg1_fact_4200"
    assert d["all_five_planes_required"] == [
        "P1", "P2", "P3", "P4", "P5"
    ]
    assert d["unique_argmax_required"] is True
    assert d["positivity_gate"] is False
    assert d["response_blind_control_from_geometry_only"] is True
    assert d["selection_allowed"] is True
    assert d["p_value_count"] == 0
    assert d["confirmation_response_access_allowed"] is False
    assert d["residual_response_access_allowed"] is False

    c = payloads["confirmation"]["manifest"]
    assert c["role"] == "confirmation"
    assert c["pair_id_first"] == "xg1_fact_4201"
    assert c["pair_id_last"] == "xg1_fact_4500"
    assert c["selection_allowed"] is False
    assert c["discovery_raw_response_access_allowed"] is False
    assert c["residual_response_access_allowed"] is False
    assert c["planned_primary_test"] == {
        "test": "one_sample_student_t",
        "alternative": "greater",
        "alpha": 0.05,
        "n": 300,
        "primary_p_value_count": 1,
    }

    r = payloads["residual_characterization"]["manifest"]
    assert r["role"] == "residual_characterization"
    assert r["pair_id_first"] == "xg1_fact_4501"
    assert r["pair_id_last"] == "xg1_fact_4800"
    assert r["dominant_plane_preselected"] is False
    assert r["residual_plane_set_preselected"] is False
    assert r["p3_preselected"] is False
    assert r["p5_predesignated"] is False
    assert r["formal_inference_allowed"] is False
    assert r["p_value_count"] == 0
    assert r["selection_allowed"] is False
    assert r["rescue_of_failed_core_test"] is False
    assert r["planned_symbolic_endpoints"] == {
        "S_k": "Q_native-Q_k-neutralized for k != k*",
        "S_RES": "Q_native-Q_all-residual-neutralized",
        "S_SUM": "sum_{k != k*} S_k",
        "I_RES": "S_RES-S_SUM",
        "C_k": "mean_branch[a_k^2+b_k^2]",
    }

    for manifest in (d, c, r):
        assert manifest["labels_present"] is False
        assert manifest["response_fields_present"] is False
        assert manifest["endpoint_values_present"] is False
        assert manifest["tokenizer_executed"] is False
        assert manifest["checkpoint_loaded"] is False
        assert manifest["model_executed"] is False
        assert manifest["cuda_executed"] is False
        json.dumps(manifest, sort_keys=True, allow_nan=False)


def test_write_and_validate_three_cohorts(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(subject, "authenticate_repo", lambda: None)

    result = subject.write_holdouts(
        discovery_dir=tmp_path / "d",
        confirmation_dir=tmp_path / "c",
        residual_dir=tmp_path / "r",
    )

    assert result["result"] == subject.RESULT

    expected = {
        "discovery": ("xg1_fact_3901", "xg1_fact_4200"),
        "confirmation": ("xg1_fact_4201", "xg1_fact_4500"),
        "residual_characterization":
            ("xg1_fact_4501", "xg1_fact_4800"),
    }
    for role, (first, last) in expected.items():
        manifest = result[role]["manifest"]
        assert manifest["pair_id_first"] == first
        assert manifest["pair_id_last"] == last
        assert len(result[role]["artifact_sha256"]) == 3
