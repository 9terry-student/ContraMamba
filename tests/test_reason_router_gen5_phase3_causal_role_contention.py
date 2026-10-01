from __future__ import annotations

import pytest
import torch

from contramamba.gen5_phase3_causal_role_contention import (
    INTERMEDIATE_SIZE,
    STRONG_DIM,
    apply_stressor_batch,
    contention_fractions,
    pressure_delta,
    validate_plane_geometry,
)


def synthetic_planes():
    eye = torch.eye(STRONG_DIM, dtype=torch.float64)
    return {
        "pp3_plus": eye[:, 0].clone(),
        "pp3_minus": eye[:, 1].clone(),
        "pp5_plus": eye[:, 2].clone(),
        "pp5_minus": eye[:, 3].clone(),
    }


def mask():
    out = torch.zeros(INTERMEDIATE_SIZE, dtype=torch.bool)
    out[:STRONG_DIM] = True
    return out


def test_plane_geometry_exact():
    g = validate_plane_geometry(synthetic_planes())
    assert g["pp3_plus_minus_abs_dot"] == 0.0
    assert g["pp5_plus_minus_abs_dot"] == 0.0
    assert g["pp3_pp5_cross_max_abs"] == 0.0


def test_pr_exact_neutralization():
    planes = synthetic_planes()
    h = torch.zeros(3, STRONG_DIM, dtype=torch.float64)
    h[:, 0] = torch.tensor([2.0, -3.0, 0.5])
    h[:, 1] = torch.tensor([-1.0, 4.0, 2.0])
    h[:, 7] = 9.0
    d = pressure_delta(h, pressure="PR", planes=planes)
    post = h + d["delta"]
    assert torch.equal(post[:, 0], torch.zeros(3, dtype=torch.float64))
    assert torch.equal(post[:, 1], torch.zeros(3, dtype=torch.float64))
    assert torch.equal(post[:, 7], h[:, 7])


def test_pc_uses_native_pp3_coefficients_and_matches_pr_norm():
    planes = synthetic_planes()
    h = torch.zeros(2, STRONG_DIM, dtype=torch.float64)
    h[0, 0], h[0, 1] = 3.0, 4.0
    h[1, 0], h[1, 1] = -5.0, 12.0
    pr = pressure_delta(h, pressure="PR", planes=planes)
    pc = pressure_delta(h, pressure="PC", planes=planes)
    assert torch.equal(pr["a"], pc["a"])
    assert torch.equal(pr["b"], pc["b"])
    assert torch.allclose(
        torch.linalg.vector_norm(pr["delta"], dim=1),
        torch.linalg.vector_norm(pc["delta"], dim=1),
        atol=0.0,
        rtol=0.0,
    )
    assert torch.equal(pc["delta"][:, 2], -pc["a"])
    assert torch.equal(pc["delta"][:, 3], -pc["b"])


def test_p0_exact_identity():
    x = torch.randn(4, 8, 2 * INTERMEDIATE_SIZE)
    active = torch.tensor([True, False, True, False])
    targets = torch.tensor([2, -1, 6, -1])
    y, audit = apply_stressor_batch(
        x,
        pressure="P0",
        strong_mask=mask(),
        active_rows=active,
        target_indices=targets,
        planes=synthetic_planes(),
    )
    assert y is x
    assert audit["active_row_count"] == 2


def test_mixed_batch_changes_only_active_target_strong_hidden_channels():
    planes = synthetic_planes()
    strong = mask()
    x = torch.zeros(4, 7, 2 * INTERMEDIATE_SIZE, dtype=torch.float64)
    x[0, 2, 0], x[0, 2, 1] = 3.0, 4.0
    x[2, 5, 0], x[2, 5, 1] = -2.0, 7.0
    x[:, :, STRONG_DIM + 10] = 11.0
    x[:, :, INTERMEDIATE_SIZE + 3] = 13.0

    active = torch.tensor([True, False, True, False])
    targets = torch.tensor([2, -1, 5, -1])

    y, audit = apply_stressor_batch(
        x,
        pressure="PR",
        strong_mask=strong,
        active_rows=active,
        target_indices=targets,
        planes=planes,
    )
    assert audit["active_row_count"] == 2
    assert y[0, 2, 0].item() == 0.0
    assert y[0, 2, 1].item() == 0.0
    assert y[2, 5, 0].item() == 0.0
    assert y[2, 5, 1].item() == 0.0
    assert torch.equal(y[1], x[1])
    assert torch.equal(y[3], x[3])
    assert torch.equal(y[:, :, INTERMEDIATE_SIZE:], x[:, :, INTERMEDIATE_SIZE:])
    assert torch.equal(
        y[:, :, :INTERMEDIATE_SIZE][:, :, ~strong],
        x[:, :, :INTERMEDIATE_SIZE][:, :, ~strong],
    )
    for row, target in ((0, 2), (2, 5)):
        keep = [i for i in range(x.shape[1]) if i != target]
        assert torch.equal(y[row, keep], x[row, keep])


def test_mixed_batch_matches_rowwise_reference():
    planes = synthetic_planes()
    strong = mask()
    torch.manual_seed(7)
    x = torch.randn(5, 6, 2 * INTERMEDIATE_SIZE, dtype=torch.float64)
    active = torch.tensor([True, False, True, True, False])
    targets = torch.tensor([1, -1, 3, 4, -1])

    batch, _ = apply_stressor_batch(
        x,
        pressure="PC",
        strong_mask=strong,
        active_rows=active,
        target_indices=targets,
        planes=planes,
    )

    pieces = []
    for i in range(x.shape[0]):
        out, _ = apply_stressor_batch(
            x[i:i+1],
            pressure="PC",
            strong_mask=strong,
            active_rows=torch.tensor([bool(active[i])]),
            target_indices=torch.tensor([int(targets[i])]),
            planes=planes,
        )
        pieces.append(out)
    assert torch.equal(batch, torch.cat(pieces, dim=0))


def test_stressor_delta_owns_no_gradient():
    x = torch.randn(
        2,
        4,
        2 * INTERMEDIATE_SIZE,
        dtype=torch.float64,
        requires_grad=True,
    )
    y, _ = apply_stressor_batch(
        x,
        pressure="PR",
        strong_mask=mask(),
        active_rows=torch.tensor([True, False]),
        target_indices=torch.tensor([2, -1]),
        planes=synthetic_planes(),
    )
    y.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


def test_contention_fractions_matches_materialized_map():
    torch.manual_seed(11)
    state_width, rank, hidden = 17, 2, 5
    q, _ = torch.linalg.qr(torch.randn(state_width, 4, dtype=torch.float64))
    r22 = q[:, :2].contiguous()
    c22 = q[:, 2:4].contiguous()
    a = torch.randn(rank, hidden, dtype=torch.float64)
    b = torch.randn(state_width, rank, dtype=torch.float64)

    got = contention_fractions(a, b, r22, c22)
    m = b @ a
    denominator = torch.linalg.matrix_norm(m).pow(2).item()
    expected_r = torch.linalg.matrix_norm(r22.T @ m).pow(2).item() / denominator
    expected_c = torch.linalg.matrix_norm(c22.T @ m).pow(2).item() / denominator

    assert got["F_R"] == pytest.approx(expected_r, rel=1e-12, abs=1e-12)
    assert got["F_C"] == pytest.approx(expected_c, rel=1e-12, abs=1e-12)


def test_contention_fraction_rejects_zero_map():
    q, _ = torch.linalg.qr(torch.randn(8, 4, dtype=torch.float64))
    with pytest.raises(Exception, match="ZERO_CORRECTION_MAP"):
        contention_fractions(
            torch.ones(2, 3, dtype=torch.float64),
            torch.zeros(8, 2, dtype=torch.float64),
            q[:, :2],
            q[:, 2:4],
        )


def test_float32_runtime_cast_residual_is_bounded():
    planes = synthetic_planes()
    strong = mask()

    x = torch.zeros(
        2,
        6,
        2 * INTERMEDIATE_SIZE,
        dtype=torch.float32,
    )
    x[0, 2, 0] = 1.2345679
    x[0, 2, 1] = -0.9876543
    x[1, 4, 0] = -2.3456788
    x[1, 4, 1] = 3.4567890

    y, audit = apply_stressor_batch(
        x,
        pressure="PR",
        strong_mask=strong,
        active_rows=torch.tensor([True, True]),
        target_indices=torch.tensor([2, 4]),
        planes=planes,
    )

    assert audit["applied_correction_max_abs_residual"] <= 5e-6
    assert audit["actual_post_pp3_plus_max_abs"] <= 5e-6
    assert audit["actual_post_pp3_minus_max_abs"] <= 5e-6

    assert abs(float(y[0, 2, 0])) <= 5e-6
    assert abs(float(y[0, 2, 1])) <= 5e-6
    assert abs(float(y[1, 4, 0])) <= 5e-6
    assert abs(float(y[1, 4, 1])) <= 5e-6
