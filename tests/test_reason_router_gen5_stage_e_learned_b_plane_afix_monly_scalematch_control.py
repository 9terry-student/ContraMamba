from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from contramamba.gen5_phase2_state_update_ownership import FROZEN_SHAPE
from contramamba.gen5_stage_e_learned_b_plane_afix_monly_scalematch_control import (
    ARM,
    EXPECTED_TRAINABLE_NUMEL,
    SOURCE_CORRECTIONS,
    TRAINING_SEEDS,
    LearnedBPlaneAFixMOnlyScaleMatchCorrection,
    load_seed_matched_afix_monly_scalematch_source,
    tensor_sha256,
    trainable_parameter_audit,
    zero_output_firewall,
)
from scripts import (
    train_reason_router_gen5_stage_e_learned_b_plane_afix_monly_scalematch_control
    as runner,
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
    assert ARM == "E-BFREE-AFIX-MONLY-SCALEMATCH"
    assert TRAINING_SEEDS == (6201, 6202, 6203)
    assert set(SOURCE_CORRECTIONS) == {6201, 6202, 6203}
    assert EXPECTED_TRAINABLE_NUMEL == 4
    assert runner.PRESSURE == "P0"
    assert runner.TOTAL_OPTIMIZER_STEPS == 20


def test_scalematch_factor_is_dimension_derived() -> None:
    assert runner.UNRESTRICTED_B_TRAINABLE_NUMEL == 49152
    assert runner.M_TRAINABLE_NUMEL == 4
    assert runner.SCALEMATCH_FACTOR == math.sqrt(49152 / 4)
    assert runner.SCALEMATCH_FACTOR == math.sqrt(12288)


def test_scalematch_learning_rate_is_exact_dimension_formula() -> None:
    assert runner.ORIGINAL_UNRESTRICTED_B_LEARNING_RATE == 0.001
    expected = 0.001 * math.sqrt(49152 / 4)
    assert expected == 0.11085125168440814
    assert runner.LEARNING_RATE == expected
    assert runner.LEARNING_RATE == runner.EXPECTED_SCALEMATCH_LEARNING_RATE


def test_scalematch_lr_does_not_encode_target_r_norm() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "R_B" not in source
    assert "3.206321" not in source
    assert "3.320166" not in source
    assert "3.265662" not in source


def test_correction_copies_and_freezes_source_a() -> None:
    a = _a()
    correction = LearnedBPlaneAFixMOnlyScaleMatchCorrection(
        q_bfree=_q(),
        a_free=a,
        seed=6201,
    )
    assert torch.equal(correction.A_theta.weight, a)
    assert tensor_sha256(correction.A_theta.weight) == tensor_sha256(a)
    assert correction.A_theta.weight.requires_grad is False
    assert correction.M_theta.weight.requires_grad is True
    assert torch.count_nonzero(correction.M_theta.weight).item() == 0
    assert correction.arm == ARM


def test_zero_output_firewall_is_exact() -> None:
    correction = LearnedBPlaneAFixMOnlyScaleMatchCorrection(
        q_bfree=_q(),
        a_free=_a(),
        seed=6201,
    )
    result = zero_output_firewall(correction)
    assert result["M_nonzero_count"] == 0
    assert result["correction_output_nonzero_count"] == 0
    assert result["correction_output_exact_zero"] is True
    assert result["A_requires_grad"] is False
    assert result["M_requires_grad"] is True


def test_trainable_parameter_contract_is_m_only_four_params() -> None:
    correction = LearnedBPlaneAFixMOnlyScaleMatchCorrection(
        q_bfree=_q(),
        a_free=_a(),
        seed=6201,
    )
    audit = trainable_parameter_audit(correction)
    assert audit["trainable_tensor_names"] == ["M_theta.weight"]
    assert audit["trainable_tensor_count"] == 1
    assert audit["trainable_numel"] == 4
    assert audit["A_requires_grad"] is False
    assert audit["M_requires_grad"] is True


@pytest.mark.parametrize("seed", TRAINING_SEEDS)
def test_actual_source_checkpoint_authentication(seed: int) -> None:
    q, a_free, metadata = load_seed_matched_afix_monly_scalematch_source(
        ROOT, seed
    )
    assert q.shape == (FROZEN_SHAPE.state_width, FROZEN_SHAPE.rank)
    assert a_free.shape == (FROZEN_SHAPE.rank, FROZEN_SHAPE.hidden_size)
    assert metadata["source_file_sha256"] == SOURCE_CORRECTIONS[seed].file_sha256
    assert metadata["source_A_tensor_sha256"] == tensor_sha256(a_free)
    assert metadata["source_B_rank"] == 2
    assert metadata["qr_reconstruction_relative"] <= 1e-12
    assert metadata["factorized_operator_representability_relative"] <= 1e-12


def test_worker_assignment_is_exact_three_seed_matrix() -> None:
    flattened = [
        seed
        for worker in runner.MATRIX_WORKER_SEEDS
        for seed in worker
    ]
    assert runner.MATRIX_WORKER_SEEDS == ((6201, 6203), (6202,))
    assert sorted(flattened) == [6201, 6202, 6203]
    assert len(flattened) == len(set(flattened)) == 3


def test_afix_baseline_recovery_is_frozen() -> None:
    assert runner.AFIX_MONLY_RECOVERY_BY_SEED == {
        6201: 0.063109275770733,
        6202: 0.0610331932890345,
        6203: 0.0648228579611734,
    }
    assert runner.AFIX_MONLY_MEAN_RECOVERY == 0.0629884423403136


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


def test_runtime_mode_requires_future_execution_authority() -> None:
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


def test_execution_authority_tokens_are_scalematch_specific() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert (
        "SCIENTIFIC_EXECUTION_ALLOWED="
        "YES_STAGE_E_BFREE_AFIX_MONLY_SCALEMATCH_THREE_CELL_DIAGNOSTIC"
    ) in source
    assert (
        "TRAINING_ALLOWED="
        "YES_EXACT_STAGE_E_BFREE_AFIX_MONLY_SCALEMATCH_THREE_CELL"
    ) in source
    assert (
        "OPTIMIZER_ALLOWED="
        "YES_ADAMW_20_STEPS_M_ONLY_SCALEMATCH_LR"
    ) in source


def test_implementation_scope_contains_only_three_new_files() -> None:
    assert runner.IMPLEMENTATION_PATHS == frozenset({
        "src/contramamba/gen5_stage_e_learned_b_plane_afix_monly_scalematch_control.py",
        "scripts/train_reason_router_gen5_stage_e_learned_b_plane_afix_monly_scalematch_control.py",
        "tests/test_reason_router_gen5_stage_e_learned_b_plane_afix_monly_scalematch_control.py",
    })
    assert (
        "src/contramamba/gen5_stage_e_learned_b_plane_afix_monly_control.py"
        not in runner.IMPLEMENTATION_PATHS
    )
    assert (
        "scripts/train_reason_router_gen5_stage_e_learned_b_plane_afix_monly_control.py"
        not in runner.IMPLEMENTATION_PATHS
    )
    assert (
        "tests/test_reason_router_gen5_stage_e_learned_b_plane_afix_monly_control.py"
        not in runner.IMPLEMENTATION_PATHS
    )


def test_static_contract_rejects_any_non_lr_drift() -> None:
    runner._validate_frozen_contract()
    assert runner.WEIGHT_DECAY == 0.0001
    assert runner.GRADIENT_CLIP_NORM == 5.0
    assert runner.TOTAL_OPTIMIZER_STEPS == 20
    assert runner.EXPECTED_TRAINABLE_NUMEL == 4
