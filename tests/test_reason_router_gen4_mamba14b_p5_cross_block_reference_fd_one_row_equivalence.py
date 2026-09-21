from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import torch

from scripts import (
    reason_router_gen4_mamba14b_p5_cross_block_reference_fd_one_row_equivalence
    as gate,
)


def test_protocol_freezes_reference_fd_route() -> None:
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
    assert gate.EPSILON == 0.025
    assert gate.PRIMAL_ATOL == 1e-4
    assert gate.PRIMAL_RTOL == 1e-4
    assert gate.DIRECTION_COSINE_MIN == 0.9998
    assert gate.DIRECTION_NORM_RATIO_MIN == 0.98
    assert gate.DIRECTION_NORM_RATIO_MAX == 1.02
    assert gate.DIRECTION_REL_L2_MAX == 0.02


def test_transformers_reference_identity_is_frozen() -> None:
    assert gate.TRANSFORMERS_REFERENCE_REPO == "huggingface/transformers"
    assert gate.TRANSFORMERS_REFERENCE_TAG == "v5.0.0"
    assert (
        gate.TRANSFORMERS_REFERENCE_COMMIT
        == "08810b1e278938278c50153ee1edfd7a20a759da"
    )
    assert (
        gate.TRANSFORMERS_MAMBA_SOURCE_GIT_BLOB
        == "ae80aa74f651f15a82bfc41ece60ba51ed5bb206"
    )


def test_reference_map_is_unfused_and_does_not_call_fast_kernels() -> None:
    source = inspect.getsource(
        gate.reference_slow_block35_to_block36_map
    )
    assert "mixer35.conv1d" in source
    assert "F.softplus" in source
    assert "discrete_a" in source
    assert "ssm_state" in source
    assert "torch.matmul" in source
    assert "mixer35.out_proj" in source
    assert "layer36.norm" in source
    assert "layer36.mixer.in_proj" in source
    assert "causal_conv1d_fn" not in source
    assert "selective_scan_fn" not in source


def test_reference_exact_jvp_matches_linear_map() -> None:
    matrix = torch.tensor(
        [
            [1.0, 2.0, -1.0],
            [0.5, -3.0, 4.0],
            [2.0, 0.0, 1.5],
        ],
        dtype=torch.float64,
    )
    base = torch.tensor([0.3, -0.4, 1.2], dtype=torch.float64)
    direction = torch.tensor([2.0, -1.0, 0.5], dtype=torch.float64)

    def phi(x: torch.Tensor) -> torch.Tensor:
        return matrix @ x

    primal, transported = gate.exact_reference_jvp(
        phi,
        base,
        direction,
    )
    assert torch.allclose(primal, matrix @ base, atol=0.0, rtol=0.0)
    assert torch.allclose(
        transported,
        matrix @ direction,
        atol=1e-12,
        rtol=1e-12,
    )


def test_fixed_fd_matches_linear_reference_exactly() -> None:
    matrix = torch.tensor(
        [
            [1.0, 2.0],
            [-3.0, 0.5],
        ],
        dtype=torch.float64,
    )
    base = torch.tensor([0.25, -0.75], dtype=torch.float64)
    direction = torch.tensor([0.8, 1.3], dtype=torch.float64)

    def phi(x: torch.Tensor) -> torch.Tensor:
        return matrix @ x

    _, reference = gate.exact_reference_jvp(phi, base, direction)
    _, _, fd = gate.fast_symmetric_fd(phi, base, direction)

    assert torch.allclose(fd, reference, atol=1e-12, rtol=1e-12)


def test_direction_comparison_thresholds_accept_equivalence() -> None:
    original_dim = gate.INTERMEDIATE_SIZE
    try:
        gate.INTERMEDIATE_SIZE = 3
        ref = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        obs = ref * 1.005
        result = gate.compare_direction(ref, obs)
    finally:
        gate.INTERMEDIATE_SIZE = original_dim

    assert result["pass"] is True
    assert result["cosine_similarity"] >= gate.DIRECTION_COSINE_MIN
    assert result["relative_l2_error"] <= gate.DIRECTION_REL_L2_MAX


def test_fast_fd_is_fixed_and_has_no_epsilon_argument() -> None:
    signature = inspect.signature(gate.fast_symmetric_fd)
    assert "epsilon" not in signature.parameters
    source = inspect.getsource(gate.fast_symmetric_fd)
    assert "EPSILON" in source
    assert "torch.inference_mode" in source
    assert "torch.autograd" not in source
    assert "torch.func" not in source


def test_gate_does_not_access_scientific_transport_outputs() -> None:
    source = inspect.getsource(gate)
    assert '"experiment5_xg1_response_accessed": False' in source
    assert '"population_transport_executed": False' in source
    assert '"adjacent_p5_accessed": False' in source
    assert '"principal_angles_computed": False' in source
    assert '"projector_overlap_computed": False' in source
    assert '"procrustes_alignment_computed": False' in source
    assert '"scientific_conclusion": None' in source
    assert '"p_value_count_added": 0' in source
    assert "scipy" not in source.lower()
    assert "ttest" not in source.lower()


def test_standalone_help_imports_from_repo_root() -> None:
    repo_root = Path(gate.__file__).resolve().parents[1]
    script_path = (
        repo_root
        / "scripts"
        / "reason_router_gen4_mamba14b_p5_cross_block_reference_fd_one_row_equivalence.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout
    assert "--expected-head" in completed.stdout
    assert "--snapshot" in completed.stdout
