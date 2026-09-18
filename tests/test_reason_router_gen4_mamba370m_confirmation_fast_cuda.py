from __future__ import annotations

import math

import pytest
import torch

from scripts import (
    reason_router_gen4_mamba370m_confirmation_fast_cuda
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


def test_protocol_and_forward_budget_are_frozen():
    subject.validate_protocol()

    assert subject.PAIR_IDS[0] == "xg1_fact_3301"
    assert subject.PAIR_IDS[-1] == "xg1_fact_3600"
    assert subject.PAIR_COUNT == 300

    assert subject.SELECTED_PLANE == "P3"
    assert subject.CONTROL_PLANE == "P5"
    assert subject.RESIDUAL_PLANES == ("P1", "P2", "P4", "P5")

    assert subject.EPS == 0.025
    assert subject.FORWARDS_PER_CONDITION == 40
    assert subject.FORWARDS_PER_PAIR == 160
    assert subject.FORWARDS_PER_SHARD == 24000
    assert subject.TOTAL_FORWARD_BUDGET == 48000


def test_dominant_restored_is_exact_round_trip_and_control_is_matched():
    planes = _orthogonal_planes()
    h = torch.arange(
        1,
        subject.DIM + 1,
        dtype=torch.float64,
    ) / 100.0

    restored = subject.condition_correction(
        h,
        condition="dominant_restored",
        planes=planes,
    )
    assert torch.equal(
        restored["correction"],
        torch.zeros_like(h),
    )
    assert (
        restored["dominant_restoration_round_trip_max_abs"]
        == 0.0
    )

    control = subject.condition_correction(
        h,
        condition="dominant_control",
        planes=planes,
    )
    post = h + control["correction"]

    p3 = planes["P3"]
    assert abs(float(torch.dot(post, p3["plus"]))) <= subject.TOL
    assert abs(float(torch.dot(post, p3["minus"]))) <= subject.TOL

    assert (
        control["dominant_matched_norm_abs_mismatch"]
        <= subject.TOL
    )


def test_residual_neutralization_excludes_selected_p3():
    planes = _orthogonal_planes()
    h = torch.arange(
        1,
        subject.DIM + 1,
        dtype=torch.float64,
    ) / 37.0

    before_p3 = (
        float(torch.dot(h, planes["P3"]["plus"])),
        float(torch.dot(h, planes["P3"]["minus"])),
    )

    result = subject.condition_correction(
        h,
        condition="residual_neutralized",
        planes=planes,
    )
    post = h + result["correction"]

    after_p3 = (
        float(torch.dot(post, planes["P3"]["plus"])),
        float(torch.dot(post, planes["P3"]["minus"])),
    )

    assert after_p3 == pytest.approx(before_p3, abs=subject.TOL)

    for plane in subject.RESIDUAL_PLANES:
        assert (
            abs(float(torch.dot(post, planes[plane]["plus"])))
            <= subject.TOL
        )
        assert (
            abs(float(torch.dot(post, planes[plane]["minus"])))
            <= subject.TOL
        )


def test_holm_two_hypothesis_family():
    adjusted = subject.holm_adjust({
        "H_DOM": 0.01,
        "H_RES": 0.04,
    })

    assert adjusted["H_DOM"]["p_holm"] == pytest.approx(0.02)
    assert adjusted["H_DOM"]["reject_h0"] is True
    assert adjusted["H_RES"]["p_holm"] == pytest.approx(0.04)
    assert adjusted["H_RES"]["reject_h0"] is True


def test_holm_failure_of_one_hypothesis_blocks_joint_success():
    adjusted = subject.holm_adjust({
        "H_DOM": 0.01,
        "H_RES": 0.08,
    })

    assert adjusted["H_DOM"]["reject_h0"] is True
    assert adjusted["H_RES"]["reject_h0"] is False


def _mock_item(index: int, d_dom: float, s_res: float):
    return {
        "source_pair_id": subject.PAIR_IDS[index],
        "D_DOM": d_dom,
        "S_RES": s_res,
    }


def test_inference_success_requires_both_holm_rejections():
    items = [
        _mock_item(
            index,
            1.0 + (index % 7) * 0.01,
            0.8 + (index % 11) * 0.01,
        )
        for index in range(subject.PAIR_COUNT)
    ]

    result = subject.infer_confirmation(items)

    assert result["H_DOM"]["reject_h0"] is True
    assert result["H_RES"]["reject_h0"] is True
    assert result["both_holm_reject"] is True
    assert result["scientific_conclusion"] == subject.SUCCESS_LABEL


def test_inference_failure_label_when_residual_endpoint_is_negative():
    items = [
        _mock_item(
            index,
            1.0 + (index % 7) * 0.01,
            -0.8 - (index % 11) * 0.01,
        )
        for index in range(subject.PAIR_COUNT)
    ]

    result = subject.infer_confirmation(items)

    assert result["H_DOM"]["reject_h0"] is True
    assert result["H_RES"]["reject_h0"] is False
    assert result["both_holm_reject"] is False
    assert result["scientific_conclusion"] == subject.FAILURE_LABEL
    assert (
        result["scientific_conclusion"]
        == "CROSS_BACKBONE_QUALITATIVE_RECURRENCE_NOT_ESTABLISHED"
    )


def test_confirmation_identity_and_no_selection_reopen():
    assert (
        subject.DISCOVERY_FREEZE_COMMIT
        == "22e0b01a2eaf5ff9dec34c3e19b51e288db9267c"
    )
    assert (
        subject.DISCOVERY_SELECTION_SHA256
        == "0a66905f7b2cb2456a90401b598b0958fc1572f096246f3fcb8dd7078fb54b20"
    )
    assert (
        subject.CONFIRMATION_SOURCE_SHA256
        == "5827ee4f8b60717348ad588c0d4f29772b90d02bd2d13e36599042fcf5a9d4b0"
    )
    assert (
        subject.CONFIRMATION_ROWS_SHA256
        == "14e216a0de7105ae7cf7863577f773cf72b4c2344ee31160cf0d1e2defa04be0"
    )
    assert (
        subject.CONFIRMATION_MANIFEST_SHA256
        == "80bb2817d5a2489f28603f433db703d91429606e93053fc9cd948b215b74597c"
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
