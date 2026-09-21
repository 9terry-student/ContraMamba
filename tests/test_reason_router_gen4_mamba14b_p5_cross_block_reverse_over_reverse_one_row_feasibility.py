from __future__ import annotations

import inspect

import torch

from scripts import (
    reason_router_gen4_mamba14b_p5_cross_block_reverse_over_reverse_one_row_feasibility
    as gate,
)


def test_protocol_is_exact_bounded_fallback_gate() -> None:
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


def test_reverse_over_reverse_identity_matches_linear_jvp() -> None:
    matrix = torch.tensor(
        [[1.0, 2.0, -1.0], [0.5, -3.0, 4.0], [2.0, 0.0, 1.5]],
        dtype=torch.float64,
    )
    base = torch.tensor([0.3, -0.4, 1.2], dtype=torch.float64)
    direction = torch.tensor([2.0, -1.0, 0.5], dtype=torch.float64)

    def phi(x: torch.Tensor) -> torch.Tensor:
        return matrix @ x

    primal, transported = gate.reverse_over_reverse_jvp(phi, base, direction)
    assert torch.allclose(primal, matrix @ base, atol=0.0, rtol=0.0)
    assert torch.allclose(transported, matrix @ direction, atol=1e-12, rtol=1e-12)


def test_reverse_over_reverse_identity_handles_nonlinear_map() -> None:
    base = torch.tensor([0.2, -0.7, 1.1], dtype=torch.float64)
    direction = torch.tensor([1.5, -0.25, 0.8], dtype=torch.float64)

    def phi(x: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                x[0] * x[1],
                torch.sin(x[1]) + x[2] ** 2,
                torch.exp(x[0] - x[2]),
            ]
        )

    _, transported = gate.reverse_over_reverse_jvp(phi, base, direction)
    x = base.detach().clone().requires_grad_(True)
    _, reference = torch.autograd.functional.jvp(
        phi, x, direction, create_graph=False, strict=True
    )
    assert torch.allclose(transported, reference, atol=1e-12, rtol=1e-12)


def test_gate_reuses_frozen_local_map_and_not_direct_forward_jvp() -> None:
    source = inspect.getsource(gate.compute_one_direction_reverse_over_reverse)
    helper = inspect.getsource(gate.reverse_over_reverse_jvp)
    assert "direct.replace_target_content" in source
    assert "direct.local_block35_to_block36_map" in source
    assert helper.count("torch.autograd.grad") == 2
    assert "create_graph=True" in helper
    assert "torch.func.jvp" not in source
    assert "torch.func.jvp" not in helper
    assert "finite_difference" not in source.lower()
    assert "finite_difference" not in helper.lower()


def test_gate_pins_program_closure_and_direct_script() -> None:
    source = inspect.getsource(gate.authenticate_repo)
    assert "REQUIRED_PROGRAM_ANCESTOR" in source
    assert "PROGRAM_PATH" in source
    assert "DIRECT_JVP_CLOSURE_PATH" in source
    assert "DIRECT_JVP_SCRIPT_PATH" in source


def test_gate_source_has_no_scientific_transport_comparison() -> None:
    source = inspect.getsource(gate)
    assert "principal_angles_computed" in source
    assert "projector_overlap_computed" in source
    assert "procrustes_alignment_computed" in source
    assert '"scientific_conclusion": None' in source
    assert '"p_value_count_added": 0' in source
    assert '"finite_difference_used": False' in source
    assert '"direct_torch_func_jvp_used": False' in source
    assert "scipy" not in source.lower()
    assert "ttest" not in source.lower()


def test_result_and_accounting_are_technical_only() -> None:
    assert gate.RESULT_PASS == (
        "PASS_MAMBA14B_P5_CROSS_BLOCK_ONE_ROW_REVERSE_OVER_REVERSE_FEASIBILITY"
    )
    run_source = inspect.getsource(gate.run_gate)
    assert '"local_exact_jv_count": 2' in run_source
    assert '"first_reverse_grad_count": 2' in run_source
    assert '"second_reverse_grad_count": 2' in run_source
    assert '"direct_forward_jvp_count": 0' in run_source
    assert '"scientific_model_forward_count": 0' in run_source
