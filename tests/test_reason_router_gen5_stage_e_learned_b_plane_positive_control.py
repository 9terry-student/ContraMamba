from __future__ import annotations

from pathlib import Path

import pytest
import torch

from contramamba.gen5_phase2_state_update_ownership import FROZEN_SHAPE
from contramamba.gen5_stage_e_causal_plane_bottleneck import (
    EXPECTED_TRAINABLE_NUMEL,
    FixedPlaneStateWriteCorrection,
)
from contramamba.gen5_stage_e_learned_b_plane_positive_control import (
    ARM,
    SOURCE_CORRECTIONS,
    TRAINING_SEEDS,
    LearnedBPlaneCorrection,
    _canonical_qr,
    learned_b_fixed_plane_geometry,
    load_seed_matched_learned_plane,
    tensor_sha256,
)
from scripts import (
    train_reason_router_gen5_stage_e_learned_b_plane_positive_control as runner,
)


ROOT = Path(__file__).resolve().parents[1]


def _q() -> torch.Tensor:
    q = torch.zeros(
        FROZEN_SHAPE.state_width,
        FROZEN_SHAPE.rank,
        dtype=torch.float64,
    )
    q[0, 0] = 1.0
    q[1, 1] = 1.0
    return q


def _r22_c22() -> tuple[torch.Tensor, torch.Tensor]:
    r = _q()
    c = torch.zeros_like(r)
    c[2, 0] = 1.0
    c[3, 1] = 1.0
    return r, c


def test_contract_constants() -> None:
    assert ARM == "E-BFREE"
    assert TRAINING_SEEDS == (6201, 6202, 6203)
    assert set(SOURCE_CORRECTIONS) == {6201, 6202, 6203}
    assert EXPECTED_TRAINABLE_NUMEL == 1540


def test_canonical_qr_reconstructs_and_canonicalizes_signs() -> None:
    b = torch.zeros(
        FROZEN_SHAPE.state_width,
        FROZEN_SHAPE.rank,
        dtype=torch.float64,
    )
    b[0, 0] = -2.0
    b[1, 0] = 0.5
    b[2, 1] = -3.0
    b[3, 1] = 0.25
    q, r = _canonical_qr(b)
    assert torch.all(torch.diagonal(r) >= 0)
    assert torch.allclose(q.T @ q, torch.eye(2, dtype=torch.float64), atol=1e-12, rtol=0)
    assert torch.allclose(q @ r, b, atol=1e-12, rtol=1e-12)


def test_canonical_qr_is_deterministic() -> None:
    b = torch.randn(
        FROZEN_SHAPE.state_width,
        FROZEN_SHAPE.rank,
        generator=torch.Generator().manual_seed(99),
        dtype=torch.float64,
    )
    q1, r1 = _canonical_qr(b)
    q2, r2 = _canonical_qr(b)
    assert torch.equal(q1, q2)
    assert torch.equal(r1, r2)
    assert tensor_sha256(q1) == tensor_sha256(q2)


def test_canonical_qr_rejects_rank_one() -> None:
    b = torch.zeros(
        FROZEN_SHAPE.state_width,
        FROZEN_SHAPE.rank,
        dtype=torch.float64,
    )
    b[0, :] = 1.0
    with pytest.raises(Exception):
        _canonical_qr(b)


@pytest.mark.parametrize("seed", TRAINING_SEEDS)
def test_a_initialization_exactly_matches_stage_e(seed: int) -> None:
    r22, c22 = _r22_c22()
    reference = FixedPlaneStateWriteCorrection(
        arm="E-R22",
        r22=r22,
        c22=c22,
        seed=seed,
    )
    candidate = LearnedBPlaneCorrection(
        q_bfree=r22,
        seed=seed,
    )
    assert torch.equal(
        reference.A_theta.weight,
        candidate.A_theta.weight,
    )
    assert torch.count_nonzero(candidate.M_theta.weight).item() == 0


def test_trainable_parameter_contract() -> None:
    correction = LearnedBPlaneCorrection(
        q_bfree=_q(),
        seed=6201,
    )
    trainable = [
        (name, parameter)
        for name, parameter in correction.named_parameters()
        if parameter.requires_grad
    ]
    assert [name for name, _ in trainable] == [
        "A_theta.weight",
        "M_theta.weight",
    ]
    assert sum(p.numel() for _, p in trainable) == 1540


def test_geometry_stays_in_fixed_plane() -> None:
    correction = LearnedBPlaneCorrection(
        q_bfree=_q(),
        seed=6201,
    )
    with torch.no_grad():
        correction.M_theta.weight.copy_(
            torch.tensor([[1.0, 0.2], [0.1, 0.8]])
        )
    geometry = learned_b_fixed_plane_geometry(correction)
    assert geometry["effective_output_rank"] == 2
    assert geometry["effective_operator_rank"] == 2
    assert geometry["fixed_plane_residual_max_abs"] < 1e-12


def test_forward_has_expected_shape() -> None:
    correction = LearnedBPlaneCorrection(
        q_bfree=_q(),
        seed=6201,
    )
    x = torch.randn(2, 3, FROZEN_SHAPE.hidden_size)
    y = correction(x)
    assert y.shape == (2, 3, FROZEN_SHAPE.state_width)


def test_zero_m_means_zero_initial_correction() -> None:
    correction = LearnedBPlaneCorrection(
        q_bfree=_q(),
        seed=6201,
    )
    x = torch.randn(2, 3, FROZEN_SHAPE.hidden_size)
    y = correction(x)
    assert torch.count_nonzero(y).item() == 0


@pytest.mark.parametrize("seed", TRAINING_SEEDS)
def test_actual_source_checkpoint_authentication(seed: int) -> None:
    q, metadata = load_seed_matched_learned_plane(ROOT, seed)
    assert q.shape == (FROZEN_SHAPE.state_width, FROZEN_SHAPE.rank)
    assert metadata["source_file_sha256"] == SOURCE_CORRECTIONS[seed].file_sha256
    assert metadata["source_B_rank"] == 2
    assert metadata["qr_reconstruction_relative"] <= 1e-12
    assert metadata["projector_residual_relative"] <= 1e-12


def test_worker_assignment_is_exact_three_seed_matrix() -> None:
    flattened = [
        seed
        for worker in runner.MATRIX_WORKER_SEEDS
        for seed in worker
    ]
    assert sorted(flattened) == [6201, 6202, 6203]
    assert len(flattened) == len(set(flattened)) == 3


def test_static_mode_rejects_runtime_authority_arguments() -> None:
    parser = runner.build_parser()
    args = parser.parse_args([
        "--static-verify-only",
        "--expected-head",
        runner.IMPLEMENTATION_AUTHORITY_COMMIT,
        "--implementation-freeze-commit",
        "x",
    ])
    with pytest.raises(Exception):
        runner.validate_mode_args(args)


def test_runtime_mode_requires_execution_authority() -> None:
    parser = runner.build_parser()
    args = parser.parse_args([
        "--run-matrix",
        "--expected-head",
        runner.IMPLEMENTATION_AUTHORITY_COMMIT,
        "--checkpoint",
        "parent.pt",
        "--output-root",
        "out",
    ])
    with pytest.raises(Exception):
        runner.validate_mode_args(args)


def test_frozen_stage_e_recovery_reference_is_exact() -> None:
    assert runner.FROZEN_STAGE_E_RECOVERY[6201]["E-R22"] == 0.001975318561968087
    assert runner.FROZEN_STAGE_E_RECOVERY[6202]["E-C22"] == 0.0020609486561624537
    assert runner.FROZEN_STAGE_E_RECOVERY[6203]["E-R22"] == 0.0024332976314431222
