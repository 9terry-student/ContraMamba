from __future__ import annotations

import math

import numpy as np

from scripts import (
    reason_router_gen4_mamba370m_native_coefficient_census_fast_cuda
    as subject,
)


def test_protocol_budget_and_evidence_pin():
    subject.validate_protocol()

    assert subject.PAIR_IDS[0] == "xg1_fact_3601"
    assert subject.PAIR_IDS[-1] == "xg1_fact_3900"
    assert subject.RESIDUAL_PLANES == ("P1", "P2", "P4", "P5")

    assert subject.FORWARDS_PER_PAIR == 2
    assert subject.FORWARDS_PER_SHARD == 300
    assert subject.TOTAL_FORWARD_BUDGET == 600

    assert (
        subject.REQUIRED_EVIDENCE_FREEZE_COMMIT
        == "b92ed1aded48de898cfac9cc85da373cc675b484"
    )
    assert (
        subject.DECOMP_ITEMS_SHA256
        == "89c9fa60efbb90a4c6fd993d5f71dbb7e10ee1c1f5a52932172566acd75b97e9"
    )


def _items():
    items = []

    for i in range(subject.PAIR_COUNT):
        energy = {
            "P1": 100.0 + 0.2 * i,
            "P2": 20.0 + 0.1 * i,
            "P4": 5.0 + 0.02 * i,
            "P5": 15.0 + 0.05 * i,
        }
        effects = {
            "P1": 1.0e-8 + 1.0e-11 * i,
            "P2": -2.0e-8 - 2.0e-11 * i,
            "P4": 3.0e-8 + 3.0e-11 * i,
            "P5": -6.0e-8 - 4.0e-11 * i,
        }
        items.append({
            "source_pair_id": subject.PAIR_IDS[i],
            "pair_mean_coefficient_energy": energy,
            "frozen_residual_effects": effects,
        })

    return items


def test_summary_energy_shares_sum_to_one():
    summary = subject.summarize_items(_items())

    total = sum(
        summary["plane_summary"][p][
            "energy_share_of_residual_mean_total"
        ]
        for p in subject.RESIDUAL_PLANES
    )

    assert math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12)


def test_summary_preserves_effect_signs_and_coupling():
    summary = subject.summarize_items(_items())

    assert summary["plane_summary"]["P1"]["frozen_effect_mean"] > 0
    assert summary["plane_summary"]["P2"]["frozen_effect_mean"] < 0
    assert summary["plane_summary"]["P4"]["frozen_effect_mean"] > 0
    assert summary["plane_summary"]["P5"]["frozen_effect_mean"] < 0

    assert (
        summary["plane_summary"]["P5"][
            "mean_effect_over_mean_energy"
        ]
        < 0
    )


def test_largest_energy_need_not_equal_largest_effect():
    summary = subject.summarize_items(_items())

    # Synthetic fixture: P1 always has the most coefficient energy,
    # but P5 always has the largest absolute effect.
    assert (
        summary["plane_summary"]["P1"]["largest_energy_fraction"]
        == 1.0
    )
    assert (
        summary["plane_summary"]["P5"][
            "largest_abs_effect_fraction"
        ]
        == 1.0
    )
    assert (
        summary[
            "largest_energy_and_largest_abs_effect_same_plane"
        ]["fraction"]
        == 0.0
    )


def test_no_inference_boundary():
    summary = subject.summarize_items(_items())

    assert summary["formal_inference_performed"] is False
    assert summary["p_value_count"] == 0
    assert summary["selection_performed"] is False
    assert summary["scientific_conclusion_established"] is False
    assert summary["rescue_performed"] is False

    for plane in subject.RESIDUAL_PLANES:
        row = summary["plane_summary"][plane]
        assert np.isfinite(
            row["mean_effect_over_mean_energy"]
        )
        assert np.isfinite(row["corr_energy_effect"])
        assert np.isfinite(row["corr_energy_abs_effect"])
