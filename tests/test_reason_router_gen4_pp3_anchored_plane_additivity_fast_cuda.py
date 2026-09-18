from __future__ import annotations

import math

import torch

from scripts import (
    reason_router_gen4_pp3_anchored_plane_additivity_fast_cuda
    as m
)


def synthetic_planes() -> dict[str, torch.Tensor]:
    out = {}
    mapping = {
        "P1": (0, 1),
        "P2": (2, 3),
        "P3": (4, 5),
        "P4": (6, 7),
        "P5": (8, 9),
    }
    for plane, (a, b) in mapping.items():
        plus = torch.zeros(m.DIM, dtype=torch.float64)
        minus = torch.zeros(m.DIM, dtype=torch.float64)
        plus[a] = 1.0
        minus[b] = 1.0
        if plane == "P3":
            out["pp3_plus"] = plus
            out["pp3_minus"] = minus
        else:
            key = plane.lower()
            out[f"{key}_plus"] = plus
            out[f"{key}_minus"] = minus
    return out


def test_budget_contract():
    assert m.F_CONDITION == 40
    assert m.F_PAIR == 360
    assert m.F_TOTAL == 108000
    assert sum(s["forward_budget"] for s in m.SHARDS) == 108000
    m.validate_shards()


def test_condition_design_is_main_plus_p3_pairs_only():
    assert m.CONDITIONS == (
        "p1_neutralized",
        "p2_neutralized",
        "p3_neutralized",
        "p4_neutralized",
        "p5_neutralized",
        "p3_p1_joint_neutralized",
        "p3_p2_joint_neutralized",
        "p3_p4_joint_neutralized",
        "p3_p5_joint_neutralized",
    )
    assert len(m.CONDITIONS) == 9
    assert m.PAIR_PARTNERS == ("P1", "P2", "P4", "P5")


def test_single_neutralization_zeroes_only_selected_plane():
    planes = synthetic_planes()
    h = torch.arange(
        1,
        m.DIM + 1,
        dtype=torch.float64,
    ) / 100.0

    out = m.condition_correction(
        h,
        "p2_neutralized",
        planes,
    )

    assert out["selected_planes"] == ["P2"]
    assert out["selected_plane_residual_max_abs"] <= m.TOL
    assert out["unselected_plane_drift_max_abs"] <= m.TOL

    post = h + out["d"]
    assert post[2].item() == 0.0
    assert post[3].item() == 0.0
    assert post[0].item() == h[0].item()
    assert post[4].item() == h[4].item()


def test_joint_neutralization_zeroes_p3_and_partner_only():
    planes = synthetic_planes()
    h = torch.arange(
        1,
        m.DIM + 1,
        dtype=torch.float64,
    ) / 100.0

    out = m.condition_correction(
        h,
        "p3_p4_joint_neutralized",
        planes,
    )

    assert out["selected_planes"] == ["P3", "P4"]
    assert out["selected_plane_residual_max_abs"] <= m.TOL
    assert out["unselected_plane_drift_max_abs"] <= m.TOL

    post = h + out["d"]
    for index in (4, 5, 6, 7):
        assert post[index].item() == 0.0
    for index in (0, 1, 2, 3, 8, 9):
        assert post[index].item() == h[index].item()


def test_interaction_algebra():
    q0 = 10.0
    q = {
        "p1_neutralized": 9.0,
        "p2_neutralized": 8.0,
        "p3_neutralized": 7.0,
        "p4_neutralized": 9.5,
        "p5_neutralized": 8.5,
        "p3_p1_joint_neutralized": 5.5,
        "p3_p2_joint_neutralized": 4.0,
        "p3_p4_joint_neutralized": 6.0,
        "p3_p5_joint_neutralized": 5.0,
    }

    out = m.endpoints_from_q(q0, q)

    assert out["main_effects"] == {
        "P1": 1.0,
        "P2": 2.0,
        "P3": 3.0,
        "P4": 0.5,
        "P5": 1.5,
    }

    p1 = out["pp3_pair_interactions"]["P1"]
    assert p1["joint_effect"] == 4.5
    assert p1["additive_effect_prediction"] == 4.0
    assert p1["interaction_effect"] == 0.5
    assert p1["Q_additive_prediction"] == 6.0
    assert p1["Q_interaction_residual"] == -0.5

    for partner in m.PAIR_PARTNERS:
        row = out["pp3_pair_interactions"][partner]
        assert math.isclose(
            row["interaction_effect"],
            -row["Q_interaction_residual"],
            rel_tol=0.0,
            abs_tol=0.0,
        )


def test_interaction_algebra_accepts_float_roundoff():
    q0 = -2.0313360211095804e-07
    q3 = -2.8861277617966834e-07
    q5 = -2.591794889855885e-07
    q35 = -2.283140354174052e-07

    q = {
        "p1_neutralized": q0,
        "p2_neutralized": q0,
        "p3_neutralized": q3,
        "p4_neutralized": q0,
        "p5_neutralized": q5,
        "p3_p1_joint_neutralized": q3,
        "p3_p2_joint_neutralized": q3,
        "p3_p4_joint_neutralized": q3,
        "p3_p5_joint_neutralized": q35,
    }

    out = m.endpoints_from_q(q0, q)
    row = out["pp3_pair_interactions"]["P5"]

    # This case intentionally exercises non-exact binary floating arithmetic.
    assert (
        row["interaction_effect"]
        != -row["Q_interaction_residual"]
    )

    algebra_ulp = max(
        math.ulp(value)
        for value in (
            q0,
            q35,
            row["joint_effect"],
            row["additive_effect_prediction"],
            row["interaction_effect"],
            row["Q_additive_prediction"],
            row["Q_interaction_residual"],
        )
    )

    assert (
        abs(
            row["interaction_effect"]
            + row["Q_interaction_residual"]
        )
        <= m.FLOAT_IDENTITY_ULPS * algebra_ulp
    )


def test_stage2_baseline_identity_constants():
    assert m.STAGE2_RESULT_COMMIT == (
        "fb6498e52d0410c64d458896b555f4cbdbf5e407"
    )
    assert m.STAGE2_ITEMS_SHA256 == (
        "8db506d872ff81e72b08d44f9ff0af907cb1a65086c0071e3e365a75bd166e17"
    )
    assert m.STAGE2_SUMMARY_SHA256 == (
        "7f9e01dd28ffb2d2a286b464aa4701c639e68f265e11de9eef67ca1e3c1e448e"
    )


def test_raw_runner_has_no_inferential_decision_constants():
    forbidden = {
        name
        for name in vars(m)
        if "ALPHA" in name
        or "P_VALUE" in name
        or "EQUIVALENCE_MARGIN" in name
        or "INTERACTION_THRESHOLD" in name
    }
    assert forbidden == set()
