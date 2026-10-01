from __future__ import annotations

from pathlib import Path

import pytest
import torch

from contramamba.gen5_phase2_state_update_ownership import FROZEN_SHAPE
from contramamba.gen5_stage_e_causal_plane_bottleneck import (
    EXPECTED_TRAINABLE_NUMEL,
)
from contramamba.gen5_stage_e_learned_b_plane_ainit_control import (
    ARM,
    SOURCE_CORRECTIONS,
    TRAINING_SEEDS,
    LearnedBPlaneAInitCorrection,
    factorized_operator_representability,
    load_seed_matched_ainit_source,
    tensor_sha256,
    trainable_parameter_audit,
    zero_output_firewall,
)
from scripts import (
    train_reason_router_gen5_stage_e_learned_b_plane_ainit_control as runner,
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


def _a() -> torch.Tensor:
    return torch.randn(
        FROZEN_SHAPE.rank,
        FROZEN_SHAPE.hidden_size,
        generator=torch.Generator().manual_seed(123),
        dtype=torch.float32,
    )


def test_contract_constants() -> None:
    assert ARM == "E-BFREE-AINIT"
    assert TRAINING_SEEDS == (6201, 6202, 6203)
    assert set(SOURCE_CORRECTIONS) == {6201, 6202, 6203}
    assert EXPECTED_TRAINABLE_NUMEL == 1540
    assert runner.BFREE_MEAN_RECOVERY == 0.07142508872336398


def test_ainit_correction_copies_source_a_exactly() -> None:
    a = _a()
    correction = LearnedBPlaneAInitCorrection(
        q_bfree=_q(),
        a_free=a,
        seed=6201,
    )
    assert torch.equal(correction.A_theta.weight, a)
    assert tensor_sha256(correction.A_theta.weight) == tensor_sha256(a)
    assert torch.count_nonzero(correction.M_theta.weight).item() == 0


def test_zero_output_firewall_is_exact() -> None:
    correction = LearnedBPlaneAInitCorrection(
        q_bfree=_q(),
        a_free=_a(),
        seed=6201,
    )
    result = zero_output_firewall(correction)
    assert result["M_nonzero_count"] == 0
    assert result["correction_output_nonzero_count"] == 0
    assert result["correction_output_exact_zero"] is True


def test_trainable_parameter_contract() -> None:
    correction = LearnedBPlaneAInitCorrection(
        q_bfree=_q(),
        a_free=_a(),
        seed=6201,
    )
    audit = trainable_parameter_audit(correction)
    assert audit["trainable_tensor_names"] == [
        "A_theta.weight",
        "M_theta.weight",
    ]
    assert audit["trainable_tensor_count"] == 2
    assert audit["trainable_numel"] == 1540


def test_factorized_representability_exact_qr() -> None:
    a = _a().to(torch.float64)
    q = _q()
    r = torch.tensor(
        [[2.0, -0.3], [0.0, 1.7]],
        dtype=torch.float64,
    )
    b = q @ r
    result = factorized_operator_representability(
        a_free=a,
        b_free=b,
        q_bfree=q,
        r_bfree=r,
    )
    assert result["factorized_operator_residual_sq"] == 0.0
    assert result["factorized_operator_denominator_sq"] > 0.0
    assert result["factorized_operator_representability_relative"] == 0.0


def test_factorized_representability_rejects_wrong_factorization() -> None:
    a = _a().to(torch.float64)
    q = _q()
    r = torch.eye(2, dtype=torch.float64)
    b = q @ torch.tensor(
        [[2.0, 0.0], [0.0, 1.0]],
        dtype=torch.float64,
    )
    with pytest.raises(Exception):
        factorized_operator_representability(
            a_free=a,
            b_free=b,
            q_bfree=q,
            r_bfree=r,
        )


@pytest.mark.parametrize("seed", TRAINING_SEEDS)
def test_actual_source_checkpoint_ainit_authentication(seed: int) -> None:
    q, a_free, metadata = load_seed_matched_ainit_source(ROOT, seed)
    assert q.shape == (FROZEN_SHAPE.state_width, FROZEN_SHAPE.rank)
    assert a_free.shape == (FROZEN_SHAPE.rank, FROZEN_SHAPE.hidden_size)
    assert metadata["source_file_sha256"] == SOURCE_CORRECTIONS[seed].file_sha256
    assert metadata["source_A_tensor_sha256"] == tensor_sha256(a_free)
    assert metadata["source_A_row_rank"] in (1, 2)
    assert metadata["source_B_rank"] == 2
    assert metadata["qr_reconstruction_relative"] <= 1e-12
    assert metadata["factorized_operator_representability_relative"] <= 1e-12


def test_worker_assignment_is_exact_three_seed_matrix() -> None:
    flattened = [
        seed
        for worker in runner.MATRIX_WORKER_SEEDS
        for seed in worker
    ]
    assert sorted(flattened) == [6201, 6202, 6203]
    assert len(flattened) == len(set(flattened)) == 3


def test_bfree_recovery_reference_is_exact() -> None:
    assert runner.BFREE_RECOVERY_BY_SEED == {
        6201: 0.0690419274517625,
        6202: 0.0737218360466545,
        6203: 0.0715115026716749,
    }


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


def test_execution_authority_tokens_are_ainit_specific() -> None:
    source = (
        Path(runner.__file__)
        .read_text(encoding="utf-8")
    )
    assert (
        "SCIENTIFIC_EXECUTION_ALLOWED="
        "YES_STAGE_E_BFREE_AINIT_THREE_CELL_DIAGNOSTIC"
    ) in source
    assert (
        "TRAINING_ALLOWED="
        "YES_EXACT_STAGE_E_BFREE_AINIT_THREE_CELL"
    ) in source


def test_static_authentication_allows_only_three_new_files() -> None:
    assert runner.IMPLEMENTATION_PATHS == frozenset({
        "src/contramamba/gen5_stage_e_learned_b_plane_ainit_control.py",
        "scripts/train_reason_router_gen5_stage_e_learned_b_plane_ainit_control.py",
        "tests/test_reason_router_gen5_stage_e_learned_b_plane_ainit_control.py",
    })
