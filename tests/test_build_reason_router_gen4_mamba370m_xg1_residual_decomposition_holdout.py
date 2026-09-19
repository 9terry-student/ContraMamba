from __future__ import annotations

import json

from scripts import (
    build_reason_router_gen4_mamba370m_xg1_residual_decomposition_holdout
    as subject,
)


def test_range_and_outcome_blindness():
    facts, rows = subject.build_population()

    assert facts[0]["pair_id"] == "xg1_fact_3601"
    assert facts[-1]["pair_id"] == "xg1_fact_3900"
    assert len(facts) == 300
    assert len(rows) == 1800

    for row in [*facts, *rows]:
        assert not (
            subject.FORBIDDEN_OUTCOME_FIELDS
            & set(row)
        )


def test_prior_inventory_is_exact_through_3600():
    pairs, claims, evidence = (
        subject.prior_inventory_through_3600()
    )

    assert pairs == set(
        subject.prior.expected_pair_ids(1, 3600)
    )
    assert len(pairs) == 3600
    assert claims
    assert evidence


def test_payload_is_deterministic_and_disjoint():
    a = subject.build_payload()
    b = subject.build_payload()

    assert a["source"] == b["source"]
    assert a["rows"] == b["rows"]
    assert a["manifest"] == b["manifest"]

    facts = subject.prior.prior.read_jsonl_bytes(
        a["source"],
        "TEST_SOURCE",
    )
    rows = subject.prior.prior.read_jsonl_bytes(
        a["rows"],
        "TEST_ROWS",
    )

    new_pairs, new_claims, new_evidence = (
        subject.prior._sets_for_rows(facts, rows)
    )

    old_pairs, old_claims, old_evidence = (
        subject.prior_inventory_through_3600()
    )

    assert not (new_pairs & old_pairs)
    assert not (new_claims & old_claims)
    assert not (new_evidence & old_evidence)


def test_manifest_freezes_exploratory_decomposition_only():
    payload = subject.build_payload()
    manifest = payload["manifest"]

    assert manifest["schema_version"] == subject.SCHEMA
    assert manifest["result"] == subject.RESULT
    assert manifest["role"] == "exploratory_residual_decomposition"
    assert manifest["pair_id_first"] == "xg1_fact_3601"
    assert manifest["pair_id_last"] == "xg1_fact_3900"
    assert manifest["source_pair_count"] == 300
    assert manifest["row_count"] == 1800

    assert (
        manifest["planned_condition_order"]
        == list(subject.CONDITIONS)
    )
    assert (
        manifest["residual_planes_frozen"]
        == ["P1", "P2", "P4", "P5"]
    )
    assert (
        manifest["selected_dominant_candidate_frozen"]
        == "P3"
    )

    assert manifest["formal_inference_allowed"] is False
    assert manifest["p_value_count"] == 0
    assert manifest["selection_allowed"] is False
    assert (
        manifest["rescue_of_failed_cross_backbone_claim"]
        is False
    )
    assert manifest["layer_sweep_allowed"] is False
    assert manifest["token_sweep_allowed"] is False
    assert manifest["epsilon_sweep_allowed"] is False

    assert manifest["labels_present"] is False
    assert manifest["response_fields_present"] is False
    assert manifest["endpoint_values_present"] is False
    assert manifest["model_executed"] is False
    assert manifest["cuda_executed"] is False

    json.dumps(
        manifest,
        sort_keys=True,
        allow_nan=False,
    )


def test_pair_level_endpoint_definitions_are_frozen():
    manifest = subject.build_payload()["manifest"]
    endpoints = manifest["planned_pair_level_endpoints"]

    assert endpoints == {
        "S_P1": "Q_native-Q_p1_neutralized",
        "S_P2": "Q_native-Q_p2_neutralized",
        "S_P4": "Q_native-Q_p4_neutralized",
        "S_P5": "Q_native-Q_p5_neutralized",
        "S_RES": "Q_native-Q_residual_all_neutralized",
        "I_RES": "S_RES-(S_P1+S_P2+S_P4+S_P5)",
    }
