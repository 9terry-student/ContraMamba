from __future__ import annotations

import math

import pytest
import torch

from scripts import (
    reason_router_gen4_mamba370m_residual_decomposition_fast_cuda
    as subject,
)


def _orthogonal_planes():
    planes = {}
    for index, plane in enumerate(subject.PLANE_ORDER):
        plus = torch.zeros(subject.DIM, dtype=torch.float64)
        minus = torch.zeros(subject.DIM, dtype=torch.float64)
        plus[2 * index] = 1.0
        minus[2 * index + 1] = 1.0
        planes[plane] = {
            "plus": plus,
            "minus": minus,
        }
    return planes


def test_protocol_and_budget():
    subject.validate_protocol()

    assert subject.PAIR_IDS[0] == "xg1_fact_3601"
    assert subject.PAIR_IDS[-1] == "xg1_fact_3900"
    assert subject.PAIR_COUNT == 300

    assert subject.SELECTED_DOMINANT == "P3"
    assert subject.RESIDUAL_PLANES == (
        "P1",
        "P2",
        "P4",
        "P5",
    )
    assert subject.CONDITION_ORDER == (
        "native",
        "p1_neutralized",
        "p2_neutralized",
        "p4_neutralized",
        "p5_neutralized",
        "residual_all_neutralized",
    )

    assert subject.FORWARDS_PER_CONDITION == 40
    assert subject.FORWARDS_PER_PAIR == 240
    assert subject.FORWARDS_PER_SHARD == 36000
    assert subject.TOTAL_FORWARD_BUDGET == 72000


def test_individual_neutralization_is_exact_and_preserves_p3():
    planes = _orthogonal_planes()
    h = torch.arange(
        1,
        subject.DIM + 1,
        dtype=torch.float64,
    ) / 101.0

    before_p3 = (
        float(torch.dot(h, planes["P3"]["plus"])),
        float(torch.dot(h, planes["P3"]["minus"])),
    )

    for plane in subject.RESIDUAL_PLANES:
        result = subject.condition_correction(
            h,
            condition=f"{plane.lower()}_neutralized",
            planes=planes,
        )

        post = h + result["correction"]

        assert (
            abs(float(torch.dot(post, planes[plane]["plus"])))
            <= subject.TOL
        )
        assert (
            abs(float(torch.dot(post, planes[plane]["minus"])))
            <= subject.TOL
        )

        after_p3 = (
            float(torch.dot(post, planes["P3"]["plus"])),
            float(torch.dot(post, planes["P3"]["minus"])),
        )
        assert after_p3 == pytest.approx(
            before_p3,
            abs=subject.TOL,
        )


def test_all_residual_neutralization_is_exact_and_preserves_p3():
    planes = _orthogonal_planes()
    h = torch.arange(
        1,
        subject.DIM + 1,
        dtype=torch.float64,
    ) / 79.0

    result = subject.condition_correction(
        h,
        condition="residual_all_neutralized",
        planes=planes,
    )

    post = h + result["correction"]

    for plane in subject.RESIDUAL_PLANES:
        assert (
            abs(float(torch.dot(post, planes[plane]["plus"])))
            <= subject.TOL
        )
        assert (
            abs(float(torch.dot(post, planes[plane]["minus"])))
            <= subject.TOL
        )

    assert result["p3_coefficient_drift_max_abs"] <= subject.TOL


def _item(
    index: int,
    *,
    p1: float,
    p2: float,
    p4: float,
    p5: float,
    s_res: float,
):
    total = p1 + p2 + p4 + p5
    return {
        "source_pair_id": subject.PAIR_IDS[index],
        "S_P1": p1,
        "S_P2": p2,
        "S_P4": p4,
        "S_P5": p5,
        "S_RES": s_res,
        "S_INDIVIDUAL_SUM": total,
        "I_RES": s_res - total,
        "Q_native": 1.0 + 0.001 * index,
        "Q_residual_all_neutralized":
            1.0 + 0.001 * index - s_res,
    }


def test_summary_recovers_same_pair_interaction_residual():
    items = [
        _item(
            index,
            p1=0.10 + index * 1e-4,
            p2=-0.20 - index * 2e-4,
            p4=0.30 + index * 3e-4,
            p5=-0.50 - index * 4e-4,
            s_res=(
                0.10 + index * 1e-4
                - 0.20 - index * 2e-4
                + 0.30 + index * 3e-4
                - 0.50 - index * 4e-4
                + 0.07
            ),
        )
        for index in range(subject.PAIR_COUNT)
    ]

    summary = subject.summarize_items(items)

    assert (
        summary["interaction_residual"]["mean"]
        == pytest.approx(0.07)
    )
    assert (
        summary["same_pair_additivity"][
            "corr_S_RES_vs_S_INDIVIDUAL_SUM"
        ]
        == pytest.approx(1.0)
    )
    assert summary["formal_inference_performed"] is False
    assert summary["p_value_count"] == 0
    assert summary["selection_performed"] is False
    assert summary["scientific_conclusion_established"] is False


def test_no_inference_or_selection_boundary_is_frozen():
    assert (
        subject.REQUIRED_HOLDOUT_FREEZE_COMMIT
        == "1429c61e3ac5f4bdd8f18cd789a385bad8832acd"
    )
    assert (
        subject.SOURCE_SHA256
        == "2eb3c50a95b325f2bb0a3f478443bb2c2be3f2dab0c134a0b760649db65bc9e6"
    )
    assert (
        subject.ROWS_SHA256
        == "9caaff06c9f4ae02015542ac5f7f193a6d97666f1dc3ae96e126b7eb4df9590b"
    )
    assert (
        subject.STRUCTURAL_MANIFEST_SHA256
        == "0ede4e6095dbe1194164d4249e4a47982688a06f3406c94ec6c14b01f2d655d3"
    )

    state = torch.zeros(
        1,
        subject.geom.INTERMEDIATE_SIZE,
        subject.geom.STATE_SIZE,
        dtype=torch.float32,
    )
    flat = subject.flatten_state(state)

    assert flat.shape == (32768,)
    assert math.isfinite(float(flat.sum()))
