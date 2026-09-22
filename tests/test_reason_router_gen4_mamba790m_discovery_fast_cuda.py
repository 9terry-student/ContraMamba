from __future__ import annotations

import math

import pytest
import torch

from scripts import (
    reason_router_gen4_mamba790m_discovery_fast_cuda
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

    assert subject.PAIR_IDS[0] == "xg1_fact_6901"
    assert subject.PAIR_IDS[-1] == "xg1_fact_7200"
    assert subject.PAIR_COUNT == 300
    assert subject.EPS == 0.025
    assert subject.PLANE_ORDER == ("P1", "P2", "P3", "P4", "P5")

    assert subject.FORWARDS_PER_CONDITION == 40
    assert subject.FORWARDS_PER_PAIR == 240
    assert subject.FORWARDS_PER_SHARD == 36000
    assert subject.TOTAL_FORWARD_BUDGET == 72000

    assert subject.SHARDS[0]["pair_first"] == "xg1_fact_6901"
    assert subject.SHARDS[0]["pair_last"] == "xg1_fact_7050"
    assert subject.SHARDS[1]["pair_first"] == "xg1_fact_7051"
    assert subject.SHARDS[1]["pair_last"] == "xg1_fact_7200"


def test_frozen_discovery_population_is_exact_and_response_bounded():
    facts, rows = subject.load_discovery_population()

    assert len(facts) == 300
    assert len(rows) == 1800
    assert facts[0]["pair_id"] == "xg1_fact_6901"
    assert facts[-1]["pair_id"] == "xg1_fact_7200"


def test_frozen_geometry_identity_and_width():
    frozen = subject.load_frozen_geometry()

    assert subject.DIM == 975
    assert len(frozen["strong_indices"]) == 975
    assert subject.geom.strong_index_sha256(
        frozen["strong_indices"]
    ) == "4178cd58881257f6b6cd453a35c18a499dd64f86e811a05abc8ea0a6ade405c3"

    assert frozen["bases"]["xg2"].shape == (975, 5)
    assert frozen["bases"]["xg4"].shape == (975, 5)
    assert frozen["summary"]["result"] == (
        "PASS_MAMBA790M_GEOMETRY_PREPARATION"
    )

    expected = {
        "P1": 0.8929611457472746,
        "P2": 0.9661968257027739,
        "P3": 0.9886096436448485,
        "P4": 0.9914984692804061,
        "P5": 0.9977346617630234,
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


def test_flatten_state_uses_790m_state_width():
    state = torch.zeros(
        1,
        subject.geom.INTERMEDIATE_SIZE,
        subject.geom.STATE_SIZE,
        dtype=torch.float32,
    )
    out = subject.flatten_state(state)

    assert out.shape == (subject.STATE_WIDTH,)
    assert subject.STATE_WIDTH == 49152
    assert math.isfinite(float(out.sum()))


def test_frozen_hashes_and_confirmation_boundary():
    assert subject.GEOMETRY_FREEZE_COMMIT == (
        "ba4c73aa9a5e63108f3582e053678b6344990d2c"
    )
    assert subject.DISCOVERY_HOLDOUT_FREEZE_COMMIT == (
        "6c0989a45382db31c9052b89b088e999b3e7cf59"
    )
    assert subject.GEOMETRY_SUMMARY_SHA256 == (
        "7ac0c2d904d851c13727956df8f75c8be57b71af7a962f372585879a4470b8cb"
    )
    assert subject.DISCOVERY_SOURCE_SHA256 == (
        "a276845b66116d168b0c6065ea56cf07b229aa550071376d3bca46950b967cce"
    )
    assert subject.DISCOVERY_ROWS_SHA256 == (
        "1bc6acb464d6e66da8ba5a065eff1060a19cc28397b962ad15c36329acf4e46a"
    )
    assert subject.DISCOVERY_MANIFEST_SHA256 == (
        "a37e3e25bc6f3d8853c49f849f774ec6ee61ed9c37dcbcf7c8ff4f2ae513d1f4"
    )
    assert subject.RESULT_PASS == (
        "PASS_MAMBA790M_DISCOVERY_SELECTION_FREEZE"
    )
    assert "confirmation" not in (
        subject.DISCOVERY_ROOT.as_posix().lower()
    )
    assert "residual" not in (
        subject.DISCOVERY_ROOT.as_posix().lower()
    )
