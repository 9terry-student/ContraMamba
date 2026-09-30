from __future__ import annotations

import math

import pytest
import torch

from scripts import reason_router_gen5_phase1b_r22_restoration_confirmation as restoration


def orthogonal_bases():
    r22 = torch.zeros((restoration.STATE_WIDTH, 2), dtype=torch.float64)
    c22 = torch.zeros((restoration.STATE_WIDTH, 2), dtype=torch.float64)
    r22[0, 0] = 1.0
    r22[1, 1] = 1.0
    c22[2, 0] = 1.0
    c22[3, 1] = 1.0
    return r22, c22


def test_shard_partition_is_exact_and_disjoint():
    s0 = restoration.shard_pairs(0)
    s1 = restoration.shard_pairs(1)
    assert len(s0) == len(s1) == 150
    assert s0[0] == "xg1_fact_8401"
    assert s0[-1] == "xg1_fact_8550"
    assert s1[0] == "xg1_fact_8551"
    assert s1[-1] == "xg1_fact_8700"
    assert set(s0).isdisjoint(s1)
    assert s0 + s1 == restoration.expected_pairs()


def test_restoration_vector_semantics_and_matched_norm():
    r22, c22 = orthogonal_bases()
    donor = torch.zeros(restoration.STATE_SHAPE, dtype=torch.float32)
    background = torch.zeros(restoration.STATE_SHAPE, dtype=torch.float32)
    flat = donor.reshape(-1)
    flat[0] = 3.0
    flat[1] = -4.0
    background.reshape(-1)[10] = 2.0

    b = restoration.restoration_vectors(background, donor, r22, c22, "B")
    rr = restoration.restoration_vectors(background, donor, r22, c22, "RR")
    rc = restoration.restoration_vectors(background, donor, r22, c22, "RC")

    assert torch.equal(b["modified"], b["background"])
    assert torch.allclose(rr["a_native"], torch.tensor([3.0, -4.0], dtype=torch.float64))
    assert torch.allclose(rc["a_native"], rr["a_native"])
    assert rr["r_add_l2"] == pytest.approx(5.0)
    assert rc["c_add_l2"] == pytest.approx(5.0)
    assert rr["matched_addition_norm_residual"] == pytest.approx(0.0)
    assert rc["matched_addition_norm_residual"] == pytest.approx(0.0)


def test_endpoint_identity():
    row = restoration.endpoint(1.0, 1.4, 1.1)
    assert row["S_R"] == pytest.approx(0.4)
    assert row["S_C"] == pytest.approx(0.1)
    assert row["D_SUF22"] == pytest.approx(0.3)


def make_item(pair: str, i: int, d: float = 0.2):
    q_b = 1.0 + 0.001 * i
    q_rc = q_b + 0.1 + 1e-5 * i
    q_rr = q_rc + d + 1e-6 * i
    return {
        "source_pair_id": pair,
        **restoration.endpoint(q_b, q_rr, q_rc),
        "scientific_model_forward_count_this_run": restoration.FORWARDS_PER_PAIR,
        "row_dropped": False,
    }


def test_merge_requires_canonical_order_and_exact_forward_budget():
    s0 = [make_item(p, i) for i, p in enumerate(restoration.shard_pairs(0))]
    s1 = [
        make_item(p, i + restoration.SHARD_PAIR_COUNT)
        for i, p in enumerate(restoration.shard_pairs(1))
    ]
    merged = restoration.merge_shards(s0, s1)
    assert len(merged) == 300
    assert sum(x["scientific_model_forward_count_this_run"] for x in merged) == 48000

    with pytest.raises(restoration.RestorationError):
        restoration.merge_shards(list(reversed(s0)), s1)


def test_confirmatory_decision_uses_one_p_value_after_full_merge():
    items = [make_item(p, i) for i, p in enumerate(restoration.expected_pairs())]
    decision = restoration.confirmatory_decision(items)
    assert decision["confirmatory_p_value_count"] == 1
    assert decision["confirmatory_test"]["endpoint"] == "D_SUF22"
    assert decision["mean_Q_RR"] > 0
    assert decision["mean_S_R"] > 0
    assert decision["mean_D_SUF22"] > 0
    assert decision["confirmatory_test"]["p_one_sided_greater"] < 0.05
    assert decision["label"] == restoration.LABEL_SUPPORTED


def test_negative_restoration_is_not_established():
    items = []
    for i, pair in enumerate(restoration.expected_pairs()):
        q_b = 1.0
        q_rr = 0.9 - i * 1e-6
        q_rc = 1.0
        items.append({
            "source_pair_id": pair,
            **restoration.endpoint(q_b, q_rr, q_rc),
            "scientific_model_forward_count_this_run": restoration.FORWARDS_PER_PAIR,
            "row_dropped": False,
        })
    decision = restoration.confirmatory_decision(items)
    assert decision["confirmatory_p_value_count"] == 1
    assert decision["label"] == restoration.LABEL_NOT_ESTABLISHED


def test_frozen_budget_constants():
    assert restoration.FORWARDS_PER_PAIR == 160
    assert restoration.SHARD_PAIR_COUNT == 150
    assert restoration.SHARD_MODEL_FORWARD_BUDGET == 24000
    assert restoration.FULL_MODEL_FORWARD_BUDGET == 48000
    assert restoration.CPU_SCIENTIFIC_MODEL_FORWARD_BUDGET == 0
