from __future__ import annotations

import math

import pytest
import torch

from scripts import (
    reason_router_gen4_mamba14b_discovery_fast_cuda
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


def test_prospective_discovery_protocol_and_budget():
    subject.validate_protocol()

    assert subject.PAIR_IDS[0] == "xg1_fact_3901"
    assert subject.PAIR_IDS[-1] == "xg1_fact_4200"
    assert subject.PAIR_COUNT == 300
    assert subject.EPS == 0.025
    assert subject.PLANE_ORDER == ("P1", "P2", "P3", "P4", "P5")

    assert subject.FORWARDS_PER_CONDITION == 40
    assert subject.FORWARDS_PER_PAIR == 240
    assert subject.FORWARDS_PER_SHARD == 36000
    assert subject.TOTAL_FORWARD_BUDGET == 72000

    assert subject.SHARDS[0]["pair_first"] == "xg1_fact_3901"
    assert subject.SHARDS[0]["pair_last"] == "xg1_fact_4050"
    assert subject.SHARDS[1]["pair_first"] == "xg1_fact_4051"
    assert subject.SHARDS[1]["pair_last"] == "xg1_fact_4200"


def test_frozen_discovery_population_is_exact_and_response_bounded():
    facts, rows = subject.load_discovery_population()

    assert len(facts) == 300
    assert len(rows) == 1800
    assert facts[0]["pair_id"] == "xg1_fact_3901"
    assert facts[-1]["pair_id"] == "xg1_fact_4200"


def test_frozen_geometry_identity_and_width():
    frozen = subject.load_frozen_geometry()

    assert subject.DIM == 829
    assert len(frozen["strong_indices"]) == 829
    assert subject.geom.strong_index_sha256(
        frozen["strong_indices"]
    ) == "ceaebe6046c747e09a169e8b5c0e59d68c9d82185dc5a9017373f497cc0a96c7"

    assert frozen["bases"]["xg2"].shape == (829, 5)
    assert frozen["bases"]["xg4"].shape == (829, 5)
    assert frozen["summary"]["result"] == (
        "PASS_MAMBA14B_GEOMETRY_PREPARATION"
    )

    expected = {
        "P1": 0.8890412240357709,
        "P2": 0.9779875748459493,
        "P3": 0.9938177403811372,
        "P4": 0.999560709817956,
        "P5": 0.999841883307528,
    }
    assert frozen["lambda_plus"] == pytest.approx(expected)


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
        assert abs(
            float(torch.dot(post, planes[plane]["plus"]))
        ) <= subject.TOL
        assert abs(
            float(torch.dot(post, planes[plane]["minus"]))
        ) <= subject.TOL
        assert neutral["selected_plane"] == plane


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
    assert selected == "P3"
    assert stats["P3"]["mean"] == pytest.approx(-0.1)

    lambdas = {
        "P1": 0.1,
        "P2": 0.2,
        "P3": 0.3,
        "P4": 0.4,
        "P5": 0.5,
    }
    assert subject.select_response_blind_control(
        selected, lambdas
    ) == "P5"
    assert subject.select_response_blind_control(
        "P5", lambdas
    ) == "P4"


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
            "P1", lambdas
        )


def test_flatten_state_uses_14b_state_width():
    state = torch.zeros(
        1,
        subject.geom.INTERMEDIATE_SIZE,
        subject.geom.STATE_SIZE,
        dtype=torch.float32,
    )
    out = subject.flatten_state(state)

    assert out.shape == (subject.STATE_WIDTH,)
    assert subject.STATE_WIDTH == 65536
    assert math.isfinite(float(out.sum()))


def test_frozen_hashes_and_confirmation_boundary():
    assert subject.GEOMETRY_FREEZE_COMMIT == (
        "f97b597fb4da08a8d360ed07e48c6727e15ae0be"
    )
    assert subject.DISCOVERY_HOLDOUT_FREEZE_COMMIT == (
        "e988cf3e20990a9397b8238df0a1edaf03674522"
    )
    assert subject.GEOMETRY_SUMMARY_SHA256 == (
        "8a248c3f9747015a90b4a35d6082d216be9b4c4f2a0f784b5a40c6ae572909d4"
    )
    assert subject.DISCOVERY_SOURCE_SHA256 == (
        "7bdfce4701295ac316f183c2bdab2856180269c7a0b0e892655f9d2f217aecdd"
    )
    assert subject.DISCOVERY_ROWS_SHA256 == (
        "b3d62037eb2bdfe880d8517adc971123db90733eb0806e6985cbc33343d4eddb"
    )
    assert subject.DISCOVERY_MANIFEST_SHA256 == (
        "6ab69496e13de33bc657f83396cbc96b06f067753c2c0b865b58aba10000ec27"
    )
    assert subject.RESULT_PASS == (
        "PASS_MAMBA14B_DISCOVERY_SELECTION_FREEZE"
    )
    assert "confirmation" not in (
        subject.DISCOVERY_ROOT.as_posix().lower()
    )
    assert "residual" not in (
        subject.DISCOVERY_ROOT.as_posix().lower()
    )
