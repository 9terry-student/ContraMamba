from __future__ import annotations

import json

from scripts import (
    build_reason_router_gen4_mamba370m_xg1_holdouts
    as subject,
)


def test_ranges_are_exact_and_outcome_blind():
    discovery_facts, discovery_rows = (
        subject.build_population(3001, 3300)
    )
    confirmation_facts, confirmation_rows = (
        subject.build_population(3301, 3600)
    )

    assert discovery_facts[0]["pair_id"] == "xg1_fact_3001"
    assert discovery_facts[-1]["pair_id"] == "xg1_fact_3300"
    assert confirmation_facts[0]["pair_id"] == "xg1_fact_3301"
    assert confirmation_facts[-1]["pair_id"] == "xg1_fact_3600"

    assert len(discovery_facts) == 300
    assert len(discovery_rows) == 1800
    assert len(confirmation_facts) == 300
    assert len(confirmation_rows) == 1800

    for row in [
        *discovery_facts,
        *discovery_rows,
        *confirmation_facts,
        *confirmation_rows,
    ]:
        assert not (
            subject.FORBIDDEN_OUTCOME_FIELDS
            & set(row)
        )


def test_prior_inventory_is_exact_through_3000():
    pair_ids, claims, evidence = (
        subject.prior_inventory_through_3000()
    )

    assert pair_ids == set(
        subject.expected_pair_ids(1, 3000)
    )
    assert len(pair_ids) == 3000
    assert claims
    assert evidence


def test_prospective_holdout_payloads_are_disjoint():
    payloads = subject.build_holdout_payloads()

    discovery = payloads["discovery"]
    confirmation = payloads["confirmation"]

    d_manifest = discovery["manifest"]
    c_manifest = confirmation["manifest"]

    assert d_manifest["role"] == "discovery"
    assert d_manifest["pair_id_first"] == "xg1_fact_3001"
    assert d_manifest["pair_id_last"] == "xg1_fact_3300"
    assert (
        d_manifest["prospective_use"]
        == "plane_selection_only"
    )
    assert d_manifest["selection_response_access_allowed"] is True
    assert (
        d_manifest[
            "confirmation_response_access_allowed_before_selection_freeze"
        ]
        is False
    )

    assert c_manifest["role"] == "confirmation"
    assert c_manifest["pair_id_first"] == "xg1_fact_3301"
    assert c_manifest["pair_id_last"] == "xg1_fact_3600"
    assert (
        c_manifest["prospective_use"]
        == "confirmatory_only_after_selection_freeze"
    )
    assert c_manifest["selection_response_access_allowed"] is False
    assert (
        c_manifest[
            "confirmation_response_access_allowed_before_selection_freeze"
        ]
        is False
    )

    d_facts = subject.prior.read_jsonl_bytes(
        discovery["source"],
        "DISCOVERY_SOURCE",
    )
    d_rows = subject.prior.read_jsonl_bytes(
        discovery["rows"],
        "DISCOVERY_ROWS",
    )
    c_facts = subject.prior.read_jsonl_bytes(
        confirmation["source"],
        "CONFIRMATION_SOURCE",
    )
    c_rows = subject.prior.read_jsonl_bytes(
        confirmation["rows"],
        "CONFIRMATION_ROWS",
    )

    d_pairs, d_claims, d_evidence = (
        subject._sets_for_rows(d_facts, d_rows)
    )
    c_pairs, c_claims, c_evidence = (
        subject._sets_for_rows(c_facts, c_rows)
    )

    assert not (d_pairs & c_pairs)
    assert not (d_claims & c_claims)
    assert not (d_evidence & c_evidence)

    for manifest in (d_manifest, c_manifest):
        assert manifest["labels_present"] is False
        assert manifest["response_fields_present"] is False
        assert manifest["endpoint_values_present"] is False
        assert manifest["tokenizer_executed"] is False
        assert manifest["checkpoint_loaded"] is False
        assert manifest["model_executed"] is False
        assert manifest["cuda_executed"] is False
        assert manifest["rescue_policy"] == "none"

        # Must remain JSON-serializable with no NaN allowance.
        json.dumps(
            manifest,
            sort_keys=True,
            allow_nan=False,
        )
