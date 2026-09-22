from __future__ import annotations

import math

import pytest
import torch

from scripts import (
    reason_router_gen4_mamba790m_confirmation_fast_cuda
    as subject,
)


def _orthogonal_planes():
    planes = {}

    for index, plane in enumerate(subject.PLANE_ORDER):
        plus = torch.zeros(
            subject.DIM,
            dtype=torch.float64,
        )
        minus = torch.zeros(
            subject.DIM,
            dtype=torch.float64,
        )

        plus[2 * index] = 1.0
        minus[2 * index + 1] = 1.0

        planes[plane] = {
            "plus": plus,
            "minus": minus,
        }

    return planes


def _mock_item(index: int, value: float):
    return {
        "source_pair_id": subject.PAIR_IDS[index],
        "D_CORE": value,
    }


def test_protocol_and_budget_are_exact():
    subject.validate_protocol()

    assert subject.PAIR_IDS[0] == "xg1_fact_7201"
    assert subject.PAIR_IDS[-1] == "xg1_fact_7500"

    assert subject.PAIR_COUNT == 300
    assert subject.SELECTED_PLANE == "P2"
    assert subject.CONTROL_PLANE == "P5"

    assert subject.CONDITION_ORDER == (
        "dominant_restored",
        "dominant_control",
    )

    assert subject.DIM == 975
    assert subject.EPS == 0.025

    assert subject.FORWARDS_PER_CONDITION == 40
    assert subject.FORWARDS_PER_PAIR == 80
    assert subject.FORWARDS_PER_SHARD == 12000
    assert subject.TOTAL_FORWARD_BUDGET == 24000

    assert subject.SHARDS[0]["pair_first"] == "xg1_fact_7201"
    assert subject.SHARDS[0]["pair_last"] == "xg1_fact_7350"
    assert subject.SHARDS[1]["pair_first"] == "xg1_fact_7351"
    assert subject.SHARDS[1]["pair_last"] == "xg1_fact_7500"

    assert subject.ALPHA == 0.05


def test_frozen_selection_and_confirmation_population():
    selection = subject.load_frozen_selection()

    assert (
        selection["selected_dominant_candidate"]
        == "P2"
    )
    assert (
        selection["response_blind_control_plane"]
        == "P5"
    )
    assert selection["selection_unique"] is True
    assert selection["positivity_gate_applied"] is False
    assert (
        selection["control_selection_uses_response"]
        is False
    )
    assert selection["confirmation_accessed"] is False

    facts, rows = subject.load_confirmation_population()

    assert len(facts) == 300
    assert len(rows) == 1800

    assert facts[0]["pair_id"] == "xg1_fact_7201"
    assert facts[-1]["pair_id"] == "xg1_fact_7500"


def test_restored_round_trip_and_p5_matched_control():
    planes = _orthogonal_planes()

    h = (
        torch.arange(
            1,
            subject.DIM + 1,
            dtype=torch.float64,
        )
        / 100.0
    )

    restored = subject.condition_correction(
        h,
        condition="dominant_restored",
        planes=planes,
    )

    assert torch.equal(
        restored["correction"],
        torch.zeros_like(h),
    )

    control = subject.condition_correction(
        h,
        condition="dominant_control",
        planes=planes,
    )

    post = h + control["correction"]

    selected = planes["P2"]

    assert abs(
        float(
            torch.dot(
                post,
                selected["plus"],
            )
        )
    ) <= subject.TOL

    assert abs(
        float(
            torch.dot(
                post,
                selected["minus"],
            )
        )
    ) <= subject.TOL

    assert (
        control[
            "dominant_matched_norm_abs_mismatch"
        ]
        <= subject.TOL
    )


def test_exactly_one_primary_p_value_positive_support():
    items = [
        _mock_item(
            i,
            1.0 + (i % 7) * 0.01,
        )
        for i in range(subject.PAIR_COUNT)
    ]

    result = subject.infer_confirmation(items)
    primary = result["primary_test"]

    assert primary["endpoint"] == (
        "D_CORE=Q_restored(P2)-Q_control(P5)"
    )
    assert primary["p_value_count"] == 1
    assert primary["test"] == "one_sample_student_t"
    assert primary["tail"] == "greater"
    assert primary["alternative"] == "greater"

    assert primary["mean"] > 0.0
    assert primary["p_raw"] < 0.05

    assert result["core_supported"] is True
    assert result["selection_reopened"] is False
    assert result["readout_response_accessed"] is False
    assert result["rescue_performed"] is False
    assert (
        result["additional_p_values_executed"]
        is False
    )


def test_negative_effect_not_supported_no_rescue():
    items = [
        _mock_item(
            i,
            -1.0 - (i % 11) * 0.01,
        )
        for i in range(subject.PAIR_COUNT)
    ]

    result = subject.infer_confirmation(items)

    assert result["core_supported"] is False
    assert result["selection_reopened"] is False
    assert result["readout_response_accessed"] is False
    assert result["rescue_performed"] is False
    assert (
        result["additional_p_values_executed"]
        is False
    )


def test_frozen_identities_are_exact():
    assert subject.GEOMETRY_FREEZE_COMMIT == (
        "ba4c73aa9a5e63108f3582e053678b6344990d2c"
    )

    assert (
        subject.CONFIRMATION_HOLDOUT_FREEZE_COMMIT
        == "6c0989a45382db31c9052b89b088e999b3e7cf59"
    )

    assert subject.DISCOVERY_FREEZE_COMMIT == (
        "e6ba3c466d3063ff69112ca8f57f0d569ee63d06"
    )

    assert subject.DISCOVERY_SELECTION_SHA256 == (
        "854324f4799d9011bcde734fb8b4959f2ff1aa765a9f82572aa9ede1ea3c188e"
    )

    assert subject.CONFIRMATION_SOURCE_SHA256 == (
        "8da2bf45693ee4d2224db3d3379af1f17ec082f8210da8b988a84559ef96671b"
    )

    assert subject.CONFIRMATION_ROWS_SHA256 == (
        "e1e1e93272d244b4db91bcd7dc7eeb20a48575e36853509a6b5ff9944c11e011"
    )

    assert subject.CONFIRMATION_MANIFEST_SHA256 == (
        "0db2bb28c41dd40037466da25892cdb6b6e5cc268bf7819fd7f1a9642b23aaf9"
    )


def test_state_width_matches_mamba790m():
    state = torch.zeros(
        1,
        subject.geom.INTERMEDIATE_SIZE,
        subject.geom.STATE_SIZE,
        dtype=torch.float32,
    )

    flat = subject.flatten_state(state)

    assert flat.shape == (49152,)
    assert subject.STATE_WIDTH == 49152
    assert math.isfinite(float(flat.sum()))


def test_confirmation_does_not_reopen_selection_or_readout():
    assert subject.SELECTED_PLANE == "P2"
    assert subject.CONTROL_PLANE == "P5"

    assert subject.SUCCESS_LABEL == (
        "MAMBA790M_CORE_CONFIRMATION_SUPPORTED"
    )
    assert subject.FAILURE_LABEL == (
        "MAMBA790M_CORE_CONFIRMATION_NOT_SUPPORTED"
    )

    assert subject.RESULT_PASS == (
        "PASS_MAMBA790M_CORE_CONFIRMATION"
    )
