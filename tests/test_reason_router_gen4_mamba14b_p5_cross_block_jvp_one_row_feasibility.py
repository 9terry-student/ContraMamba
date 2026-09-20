from __future__ import annotations

import inspect

import torch

from scripts import (
    reason_router_gen4_mamba14b_p5_cross_block_jvp_one_row_feasibility
    as gate,
)


def test_protocol_is_exact_bounded_gate() -> None:
    gate.validate_protocol()
    assert gate.FAMILY == "xg2"
    assert gate.PAIR == "xg2_fact_301"
    assert gate.CELL == "C2_NAME"
    assert gate.ANCHOR_NAME == "A_IDENTITY"
    assert gate.TARGET_OFFSET == 2
    assert gate.SOURCE_BLOCK == 35
    assert gate.TARGET_BLOCK == 36
    assert gate.INTERMEDIATE_SIZE == 4096
    assert gate.SOURCE_PLANE == "P5"
    assert gate.BASIS_NAMES == ("plus", "minus")


def test_scatter_strong_vector_preserves_coordinate_and_norm() -> None:
    value = torch.tensor([3.0, 4.0], dtype=torch.float64)
    out = gate.scatter_strong_vector(
        value,
        [1, 4],
        ambient_dim=6,
    )
    assert out.dtype == torch.float64
    assert out.tolist() == [0.0, 3.0, 0.0, 0.0, 4.0, 0.0]
    assert torch.linalg.vector_norm(out).item() == 5.0


def test_replace_target_content_changes_only_content_half_at_target() -> None:
    baseline = torch.arange(
        1 * 3 * gate.PROJECTED_WIDTH,
        dtype=torch.float32,
    ).reshape(1, 3, gate.PROJECTED_WIDTH)
    target = torch.full(
        (gate.INTERMEDIATE_SIZE,),
        -7.0,
        dtype=torch.float32,
    )
    out = gate.replace_target_content(
        baseline,
        target,
        target_abs=1,
    )

    assert tuple(out.shape) == tuple(baseline.shape)
    assert torch.equal(out[:, 0, :], baseline[:, 0, :])
    assert torch.equal(out[:, 2, :], baseline[:, 2, :])
    assert torch.equal(
        out[0, 1, : gate.INTERMEDIATE_SIZE],
        target,
    )
    assert torch.equal(
        out[0, 1, gate.INTERMEDIATE_SIZE :],
        baseline[0, 1, gate.INTERMEDIATE_SIZE :],
    )


def test_numerical_rank_two_accepts_independent_vectors() -> None:
    a = torch.zeros(gate.INTERMEDIATE_SIZE, dtype=torch.float64)
    b = torch.zeros(gate.INTERMEDIATE_SIZE, dtype=torch.float64)
    a[0] = 1.0
    b[1] = 2.0
    info = gate.numerical_rank_two([a, b])
    assert info["rank"] == 2
    assert info["rank_tolerance"] > 0.0
    assert len(info["singular_values"]) == 2


def test_numerical_rank_two_detects_degeneracy() -> None:
    a = torch.zeros(gate.INTERMEDIATE_SIZE, dtype=torch.float64)
    a[0] = 1.0
    info = gate.numerical_rank_two([a, 3.0 * a])
    assert info["rank"] == 1


def test_local_map_uses_exact_kernel_functions_and_forward_jvp() -> None:
    local_source = inspect.getsource(gate.local_block35_to_block36_map)
    jvp_source = inspect.getsource(gate.compute_one_direction_jvp)

    assert 'kernels["causal_conv1d_fn"]' in local_source
    assert 'kernels["selective_scan_fn"]' in local_source
    assert "torch.func.jvp" in jvp_source
    assert "slow_forward" not in local_source
    assert "finite_difference" not in jvp_source.lower()


def test_gate_source_has_no_scientific_transport_comparison() -> None:
    source = inspect.getsource(gate)
    assert "principal_angles_computed" in source
    assert "projector_overlap_computed" in source
    assert "procrustes_alignment_computed" in source
    assert '"scientific_conclusion": None' in source
    assert '"p_value_count_added": 0' in source
    assert "scipy" not in source.lower()
    assert "ttest" not in source.lower()
