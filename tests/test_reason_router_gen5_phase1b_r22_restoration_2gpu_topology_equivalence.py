from __future__ import annotations

import pytest

from scripts import (
    reason_router_gen5_phase1b_r22_restoration_2gpu_topology_equivalence
    as gate
)


def synthetic_item(pair: str, pair_index: int, delta: float = 0.0):
    return {
        "source_pair_id": pair,
        "pair_index": pair_index,
        "family_key": "xg1",
        "row_dropped": False,
        "scientific_model_forward_count_this_run": 160,
        "direction_order": ["xg2_0", "xg4_0"],
        "metrics": {
            "Q_B": 1.0 + delta,
            "Q_RR": 1.25 + delta,
            "Q_RC": 1.10 + delta,
            "target_token_index": 42,
            "donor_write_sha256": "reference-only-provenance",
        },
    }


def test_gate_pair_and_shard_plan_is_exact():
    assert gate.REFERENCE_PAIRS == (
        "xg1_fact_7801",
        "xg1_fact_7802",
        "xg1_fact_7803",
        "xg1_fact_7804",
    )
    assert gate.CANDIDATE0_PAIRS == (
        "xg1_fact_7801",
        "xg1_fact_7802",
    )
    assert gate.CANDIDATE1_PAIRS == (
        "xg1_fact_7803",
        "xg1_fact_7804",
    )
    assert set(gate.CANDIDATE0_PAIRS).isdisjoint(gate.CANDIDATE1_PAIRS)


def test_frozen_forward_accounting():
    assert gate.FORWARDS_PER_PAIR == 160
    assert gate.REFERENCE_FORWARD_BUDGET == 640
    assert gate.CANDIDATE0_FORWARD_BUDGET == 320
    assert gate.CANDIDATE1_FORWARD_BUDGET == 320
    assert gate.CANDIDATE_TOTAL_FORWARD_BUDGET == 640
    assert gate.TOTAL_GATE_FORWARD_BUDGET == 1280
    assert gate.CPU_SCIENTIFIC_MODEL_FORWARD_BUDGET == 0


def test_candidate_merge_is_canonical_and_budget_exact():
    c0 = [
        synthetic_item(pair, i)
        for i, pair in enumerate(gate.CANDIDATE0_PAIRS)
    ]
    c1 = [
        synthetic_item(pair, i + 2)
        for i, pair in enumerate(gate.CANDIDATE1_PAIRS)
    ]
    merged = gate.merge_candidate(c0, c1)
    assert tuple(x["source_pair_id"] for x in merged) == gate.REFERENCE_PAIRS
    assert sum(x["scientific_model_forward_count_this_run"] for x in merged) == 640


def test_candidate_merge_rejects_reordered_shard():
    c0 = [
        synthetic_item(pair, i)
        for i, pair in enumerate(gate.CANDIDATE0_PAIRS)
    ][::-1]
    c1 = [
        synthetic_item(pair, i + 2)
        for i, pair in enumerate(gate.CANDIDATE1_PAIRS)
    ]
    with pytest.raises(gate.TopologyEquivalenceError):
        gate.merge_candidate(c0, c1)


def test_equivalence_accepts_small_float_delta_and_skips_tensor_hash():
    ref = synthetic_item("xg1_fact_7801", 0)
    cand = synthetic_item("xg1_fact_7801", 0, delta=1e-8)
    cand["metrics"]["donor_write_sha256"] = "different-gpu-provenance"
    row = gate.compare_pair(ref, cand)
    assert row["equivalence_pass"] is True
    assert row["float_leaf_count"] == 3
    assert row["skipped_tensor_sha256_leaf_count"] == 1
    assert row["max_float_bound_usage_ratio"] < 1.0


def test_equivalence_rejects_large_float_delta():
    ref = synthetic_item("xg1_fact_7801", 0)
    cand = synthetic_item("xg1_fact_7801", 0, delta=1e-3)
    with pytest.raises(gate.TopologyEquivalenceError):
        gate.compare_pair(ref, cand)


def test_equivalence_rejects_discrete_mismatch():
    ref = synthetic_item("xg1_fact_7801", 0)
    cand = synthetic_item("xg1_fact_7801", 1)
    with pytest.raises(gate.TopologyEquivalenceError):
        gate.compare_pair(ref, cand)


def test_float_tolerance_is_frozen():
    assert gate.FLOAT_ATOL == 1e-9
    assert gate.FLOAT_RTOL == 1e-7
