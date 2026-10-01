from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts"
    / "reason_router_gen5_optimization_path_bypass_stage_c_step0_gradient_geometry.py"
)

spec = importlib.util.spec_from_file_location("stage_c", SCRIPT)
assert spec is not None and spec.loader is not None
stage_c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage_c)


def test_gpu_queues_cover_exact_matrix_once() -> None:
    flat = [
        cell
        for worker_id in (0, 1)
        for cell in stage_c.GPU_QUEUES[worker_id]
    ]
    assert len(flat) == 9
    assert len(set(flat)) == 9
    assert set(flat) == set(stage_c.CELLS)


def test_effective_rank_float32_convention_detects_rank2_and_rank1() -> None:
    x2 = torch.zeros(8, 2, dtype=torch.float32)
    x2[0, 0] = 2.0
    x2[1, 1] = 1.0

    rank2, _q2, s2, tol2 = stage_c._effective_rank_basis(x2)
    assert rank2 == 2
    assert s2[0] == 2.0
    assert s2[1] == 1.0
    assert tol2 > 0.0

    x1 = torch.zeros(8, 2, dtype=torch.float32)
    x1[0, 0] = 2.0
    x1[0, 1] = 1.0

    rank1, _q1, _s1, _tol1 = stage_c._effective_rank_basis(x1)
    assert rank1 == 1


def test_projected_energy_fraction_exact() -> None:
    basis = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    value = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    observed = stage_c.projected_energy_fraction(value, basis)
    assert abs(observed - 0.5) < 1e-12


def test_subspace_geometry_is_basis_invariant() -> None:
    x = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    rotation = torch.tensor(
        [
            [0.0, -1.0],
            [1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    y = x @ rotation

    geometry = stage_c.subspace_geometry(x, y)

    assert geometry["left_effective_rank"] == 2
    assert geometry["right_effective_rank"] == 2
    assert geometry["rank2_interpretation_valid"] is True
    assert abs(geometry["affinity"] - 1.0) < 1e-12
    assert max(geometry["principal_angles_deg"]) < 1e-5


def test_rank_collapse_marks_rank2_interpretation_invalid() -> None:
    rank1 = torch.tensor(
        [
            [1.0, 2.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    rank2 = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    geometry = stage_c.subspace_geometry(rank1, rank2)

    assert geometry["left_effective_rank"] == 1
    assert geometry["right_effective_rank"] == 2
    assert geometry["rank2_interpretation_valid"] is False
    assert len(geometry["principal_cosines"]) == 1
