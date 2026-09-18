from __future__ import annotations

import math

import pytest
import torch

from scripts import (
    reason_router_gen4_mamba370m_discovery_fast_cuda
    as subject,
)


def _orthogonal_planes():
    planes = {}
    for index, plane in enumerate(subject.PLANE_ORDER):
        plus = torch.zeros(subject.DIM, dtype=torch.float64)
        minus = torch.zeros(subject.DIM, dtype=torch.float64)
        plus[2 * index] = 1.0
        minus[2 * index + 1] = 1.0
        planes[plane] = {"plus": plus, "minus": minus}
    return planes


def test_frozen_discovery_protocol_and_budget():
    subject.validate_protocol()

    assert subject.PAIR_IDS[0] == "xg1_fact_3001"
    assert subject.PAIR_IDS[-1] == "xg1_fact_3300"
    assert subject.PAIR_COUNT == 300
    assert subject.EPS == 0.025
    assert subject.PLANE_ORDER == ("P1", "P2", "P3", "P4", "P5")

    assert subject.FORWARDS_PER_CONDITION == 40
    assert subject.FORWARDS_PER_PAIR == 240
    assert subject.FORWARDS_PER_SHARD == 36000
    assert subject.TOTAL_FORWARD_BUDGET == 72000

    assert subject.SHARDS[0]["pair_first"] == "xg1_fact_3001"
    assert subject.SHARDS[0]["pair_last"] == "xg1_fact_3150"
    assert subject.SHARDS[1]["pair_first"] == "xg1_fact_3151"
    assert subject.SHARDS[1]["pair_last"] == "xg1_fact_3300"


def test_plane_neutralization_and_restoration_round_trip():
    planes = _orthogonal_planes()

    h = torch.arange(
        1,
        subject.DIM + 1,
        dtype=torch.float64,
    ) / 100.0

    restored = subject.condition_correction(
        h,
        condition="restored",
        planes=planes,
    )
    assert restored["selected_plane"] is None
    assert torch.equal(
        restored["correction"],
        torch.zeros_like(h),
    )
    assert restored["restoration_round_trip_max_abs"] == 0.0

    for plane in subject.PLANE_ORDER:
        neutral = subject.condition_correction(
            h,
            condition=f"{plane.lower()}_neutralized",
            planes=planes,
        )
        post = h + neutral["correction"]
        plus = planes[plane]["plus"]
        minus = planes[plane]["minus"]

        assert abs(float(torch.dot(post, plus))) <= subject.TOL
        assert abs(float(torch.dot(post, minus))) <= subject.TOL
        assert neutral["selected_plane"] == plane


def _item(pair_index: int, effects: dict[str, float]):
    return {
        "source_pair_id": subject.PAIR_IDS[pair_index],
        "plane_effects": {
            plane: {
                "Q_restored": 1.0,
                "Q_neutralized": 1.0 - effects[plane],
                "S": effects[plane],
            }
            for plane in subject.PLANE_ORDER
        },
    }


def test_selection_has_no_positivity_gate_and_control_is_geometry_only():
    effects = {
        "P1": -0.5,
        "P2": -0.4,
        "P3": -0.1,
        "P4": -0.3,
        "P5": -0.2,
    }
    items = [
        _item(index, effects)
        for index in range(subject.PAIR_COUNT)
    ]

    selected, stats = subject.select_dominant(items)

    # All means are negative, but the frozen protocol still selects the
    # unique argmax. There is no positivity gate.
    assert selected == "P3"
    assert stats["P3"]["mean"] == pytest.approx(-0.1)

    lambdas = {
        "P1": 0.1,
        "P2": 0.2,
        "P3": 0.3,
        "P4": 0.4,
        "P5": 0.5,
    }
    assert (
        subject.select_response_blind_control(
            selected,
            lambdas,
        )
        == "P5"
    )
    assert (
        subject.select_response_blind_control(
            "P5",
            lambdas,
        )
        == "P4"
    )


def test_selection_tie_fails_closed():
    effects = {
        "P1": 0.1,
        "P2": 0.2,
        "P3": 0.3,
        "P4": 0.3,
        "P5": 0.0,
    }
    items = [
        _item(index, effects)
        for index in range(subject.PAIR_COUNT)
    ]

    with pytest.raises(
        subject.DiscoveryError,
        match="SELECTION_NOT_UNIQUE",
    ):
        subject.select_dominant(items)


def test_control_tie_fails_closed():
    lambdas = {
        "P1": 0.1,
        "P2": 0.2,
        "P3": 0.3,
        "P4": 0.9,
        "P5": 0.9,
    }
    with pytest.raises(
        subject.DiscoveryError,
        match="CONTROL_SELECTION_NOT_UNIQUE",
    ):
        subject.select_response_blind_control(
            "P1",
            lambdas,
        )


def test_flatten_state_uses_370m_state_width():
    state = torch.zeros(
        1,
        subject.geom.INTERMEDIATE_SIZE,
        subject.geom.STATE_SIZE,
        dtype=torch.float32,
    )
    out = subject.flatten_state(state)

    assert out.shape == (subject.STATE_WIDTH,)
    assert subject.STATE_WIDTH == 32768
    assert math.isfinite(float(out.sum()))


def test_frozen_hashes_and_confirmation_boundary():
    assert (
        subject.GEOMETRY_FREEZE_COMMIT
        == "9803aab542d08a257511cf409bf292428b11e0f8"
    )
    assert (
        subject.DISCOVERY_SOURCE_SHA256
        == "246b5c262c6d018a1e6047ae61ed2f7b3041154d234ea2817dbf5f7da120bc80"
    )
    assert (
        subject.DISCOVERY_ROWS_SHA256
        == "831452947cdf8136376a20ebeca873f260449bcebcd0f6b83c4361dae3a90697"
    )
    assert (
        subject.DISCOVERY_MANIFEST_SHA256
        == "28e5e5aaf8c17a803d8db9c45b5747c90c7fec883a59bd189a875e60fc0d475f"
    )
    assert subject.RESULT_PASS == "PASS_MAMBA370M_DISCOVERY_SELECTION_FREEZE"
    assert "confirmation" not in subject.DISCOVERY_ROOT.as_posix().lower()
