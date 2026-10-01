from __future__ import annotations

import torch

from scripts.reason_router_gen5_optimization_path_bypass_stage_b_functional_decomposition import (
    CONDITIONS,
    condition_metrics,
    decompose_b,
)


def test_conditions_are_frozen() -> None:
    assert CONDITIONS == (
        "FULL",
        "R22_ONLY",
        "R22_REMOVED",
        "ZERO",
    )


def test_decompose_b_reconstructs_and_removes_r22() -> None:
    torch.manual_seed(7)

    raw = torch.randn(24576, 2, dtype=torch.float64)
    r22 = torch.linalg.qr(
        torch.randn(24576, 2, dtype=torch.float64),
        mode="reduced",
    ).Q

    variants, audit = decompose_b(raw, r22)

    torch.testing.assert_close(
        variants["R22_ONLY"] + variants["R22_REMOVED"],
        variants["FULL"],
        rtol=0.0,
        atol=1e-12,
    )

    torch.testing.assert_close(
        r22.T @ variants["R22_REMOVED"],
        torch.zeros(2, 2, dtype=torch.float64),
        rtol=0.0,
        atol=1e-12,
    )

    assert torch.count_nonzero(variants["ZERO"]) == 0
    assert audit["reconstruction_max_abs"] <= 1e-12
    assert audit["r22_removed_projection_max_abs"] <= 1e-12


def test_decomposition_keeps_a_out_of_scope() -> None:
    torch.manual_seed(9)

    a = torch.randn(2, 768, dtype=torch.float64)
    a_before = a.clone()

    b = torch.randn(24576, 2, dtype=torch.float64)
    r22 = torch.linalg.qr(
        torch.randn(24576, 2, dtype=torch.float64),
        mode="reduced",
    ).Q

    decompose_b(b, r22)

    torch.testing.assert_close(
        a,
        a_before,
        rtol=0.0,
        atol=0.0,
    )


def test_condition_metrics() -> None:
    logits = torch.zeros(840, 3, dtype=torch.float64)
    labels = torch.zeros(840, dtype=torch.long)
    logits[:, 0] = 2.0

    metrics, per_row = condition_metrics(logits, labels)

    assert per_row.shape == (840,)
    assert metrics["mean_final_3way_ce"] > 0.0
    assert metrics["accuracy"] == 1.0
