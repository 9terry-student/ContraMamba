import math

import numpy as np
import torch

from scripts import (
    reason_router_gen4_k_directional_alignment_transport_core
    as core,
)


def test_alignment_delta_preserves_norm_and_hits_target():
    torch.manual_seed(7)

    x = torch.randn(
        core.EXPECTED_STRONG_COUNT,
        dtype=torch.float64,
    )
    y = torch.randn(
        core.EXPECTED_STRONG_COUNT,
        dtype=torch.float64,
    )

    target = 0.35
    delta, audit = core.alignment_delta(
        x,
        y,
        target,
    )
    realized_y = y + delta

    assert math.isclose(
        torch.linalg.vector_norm(realized_y).item(),
        torch.linalg.vector_norm(y).item(),
        rel_tol=0,
        abs_tol=1e-10,
    )
    assert abs(
        core.cosine(x, realized_y) - target
    ) < 1e-10
    assert abs(
        audit["realized_C"] - target
    ) < 1e-10


def test_magnitude_delta_hits_targets_and_preserves_cosine():
    torch.manual_seed(11)

    x = torch.randn(
        core.EXPECTED_STRONG_COUNT,
        dtype=torch.float64,
    )
    y = torch.randn(
        core.EXPECTED_STRONG_COUNT,
        dtype=torch.float64,
    )

    delta, audit = core.magnitude_delta(
        x,
        y,
        3.0,
        4.0,
    )

    assert audit["target_A"] == 3.0
    assert audit["target_B"] == 4.0
    assert math.isclose(
        audit["realized_A"],
        3.0,
        rel_tol=0,
        abs_tol=core.VECTOR_TOL,
    )
    assert math.isclose(
        audit["realized_B"],
        4.0,
        rel_tol=0,
        abs_tol=core.VECTOR_TOL,
    )
    assert (
        audit["A_target_abs_residual"]
        <= core.VECTOR_TOL
    )
    assert (
        audit["B_target_abs_residual"]
        <= core.VECTOR_TOL
    )
    assert abs(
        audit["realized_C"]
        - audit["baseline_C"]
    ) < 1e-10
    assert torch.isfinite(delta).all()


def test_branch_symmetric_pair():
    torch.manual_seed(13)

    plus = torch.randn(
        core.EXPECTED_STRONG_COUNT,
        dtype=torch.float64,
    )
    minus = torch.randn(
        core.EXPECTED_STRONG_COUNT,
        dtype=torch.float64,
    )
    delta = torch.randn(
        core.EXPECTED_STRONG_COUNT,
        dtype=torch.float64,
    )

    plus2, minus2, residual = (
        core.branch_symmetric_pair(
            plus,
            minus,
            delta,
        )
    )

    assert residual <= core.VECTOR_TOL
    assert torch.allclose(
        plus2 - minus2,
        plus - minus + delta,
        atol=1e-12,
        rtol=0,
    )


def test_post4_path_efficiency_straight_path():
    states = [
        np.asarray(
            [float(i), 0.0],
            dtype=np.float32,
        )
        for i in range(8)
    ]

    assert (
        core.post4_path_efficiency(
            states,
            2,
        )
        == 1.0
    )


def test_negative_frozen_phenotype_reduction_sign():
    result = core.causal_reductions(
        baseline_plus=0.40,
        baseline_minus=0.50,
        alignment_plus=0.45,
        alignment_minus=0.50,
        magnitude_plus=0.42,
        magnitude_minus=0.50,
    )

    assert math.isclose(
        result["delta_baseline"],
        -0.10,
    )
    assert math.isclose(
        result["R_ALIGN"],
        0.05,
    )
    assert math.isclose(
        result["R_MAG"],
        0.02,
    )
    assert result["ALIGNMENT_SPECIFICITY"] > 0
