from __future__ import annotations

import math

from scripts import (
    reason_router_gen4_finite_epsilon_five_plane_decomposition_fast_cuda
    as m,
)


def test_forward_budget_contract():
    assert m.N == 300
    assert m.F_DIRECTION == 4
    assert m.F_PAIR == 40
    assert m.F_TOTAL == 12000
    assert sum(s["forward_budget"] for s in m.SHARDS) == 12000
    m.validate_shards()


def test_direction_order_is_full_five_plane_basis():
    assert m.PLANE_ORDER == ("P1", "P2", "P3", "P4", "P5")
    assert m.DIRECTION_ORDER == (
        "P1_plus", "P1_minus",
        "P2_plus", "P2_minus",
        "P3_plus", "P3_minus",
        "P4_plus", "P4_minus",
        "P5_plus", "P5_minus",
    )


def test_decomposition_formula_exact_synthetic():
    j = {
        key: float(index + 1) / 10.0
        for index, key in enumerate(m.DIRECTION_ORDER)
    }
    eigenvalues = (0.2, 0.3, 0.4, 0.5, 0.6)

    expected_contributions = {}
    for index, plane in enumerate(m.PLANE_ORDER):
        jp = j[f"{plane}_plus"]
        jm = j[f"{plane}_minus"]
        expected_contributions[plane] = (
            eigenvalues[index] * (jp * jp - jm * jm) / 5
        )

    q_principal = math.fsum(expected_contributions.values())
    q0 = q_principal + 0.125

    out = m.decomposition_from_j(j, eigenvalues, q0)

    assert out["plane_contributions"] == expected_contributions
    assert out["Q_principal"] == q_principal
    assert out["reconstruction_residual"] == q0 - q_principal
    assert out["absolute_reconstruction_residual"] == abs(q0 - q_principal)
    assert (
        out["relative_reconstruction_residual_to_Q0"]
        == (q0 - q_principal) / q0
    )


def test_zero_q0_relative_residual_is_none():
    j = {key: 0.0 for key in m.DIRECTION_ORDER}
    out = m.decomposition_from_j(
        j,
        m.EXPECTED_EIGENVALUES,
        0.0,
    )
    assert out["Q_principal"] == 0.0
    assert out["reconstruction_residual"] == 0.0
    assert out["relative_reconstruction_residual_to_Q0"] is None
    assert out["absolute_relative_reconstruction_residual_to_Q0"] is None


def test_principal_geometry_matches_frozen_spectrum():
    planes, observed = m.principal_geometry()
    assert set(planes) >= {
        "p1_plus", "p1_minus",
        "p2_plus", "p2_minus",
        "pp3_plus", "pp3_minus",
        "p4_plus", "p4_minus",
        "p5_plus", "p5_minus",
    }
    assert len(observed) == 5
    for actual, expected in zip(
        observed,
        m.EXPECTED_EIGENVALUES,
        strict=True,
    ):
        assert abs(actual - expected) <= 2.0e-12


def test_prior_result_identity_constants():
    assert m.PRIOR_EXECUTION_HEAD == (
        "92efdd06974f3db96937d07f8df00b8ca0fea6ff"
    )
    assert m.PRIOR_ITEMS_SHA256 == (
        "9d5dabbef82a8fcfaccc4e610bbb4d91f1e7ee9ea2f2627a92f999c88034ac49"
    )
    assert m.PRIOR_SUMMARY_SHA256 == (
        "e38d8c3943594e934535ecf1682788868f815e28630e4ac44cc92864f44bd7d9"
    )


def test_no_inferential_threshold_is_defined():
    forbidden = {
        name
        for name in vars(m)
        if "ALPHA" in name
        or "P_VALUE" in name
        or "SUPPORT_THRESHOLD" in name
    }
    assert forbidden == set()
