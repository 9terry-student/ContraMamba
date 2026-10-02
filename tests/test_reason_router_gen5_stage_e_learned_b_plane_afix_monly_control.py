from __future__ import annotations

from pathlib import Path

import pytest
import torch

from contramamba.gen5_phase2_state_update_ownership import FROZEN_SHAPE
from contramamba.gen5_stage_e_learned_b_plane_afix_monly_control import (
    ARM,
    EXPECTED_TRAINABLE_NUMEL,
    SOURCE_CORRECTIONS,
    TRAINING_SEEDS,
    LearnedBPlaneAFixMOnlyCorrection,
    load_seed_matched_afix_monly_source,
    tensor_sha256,
    trainable_parameter_audit,
    zero_output_firewall,
)
from scripts import (
    train_reason_router_gen5_stage_e_learned_b_plane_afix_monly_control as runner,
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
    assert ARM == "E-BFREE-AFIX-MONLY"
    assert TRAINING_SEEDS == (6201, 6202, 6203)
    assert set(SOURCE_CORRECTIONS) == {6201, 6202, 6203}
    assert EXPECTED_TRAINABLE_NUMEL == 4
    assert runner.AINIT_MEAN_RECOVERY == 0.133576230985308


def test_afix_monly_correction_copies_and_freezes_source_a() -> None:
    a = _a()
    correction = LearnedBPlaneAFixMOnlyCorrection(
        q_bfree=_q(),
        a_free=a,
        seed=6201,
    )
    assert torch.equal(correction.A_theta.weight, a)
    assert tensor_sha256(correction.A_theta.weight) == tensor_sha256(a)
    assert correction.A_theta.weight.requires_grad is False
    assert correction.M_theta.weight.requires_grad is True
    assert torch.count_nonzero(correction.M_theta.weight).item() == 0


def test_zero_output_firewall_is_exact_and_a_is_frozen() -> None:
    correction = LearnedBPlaneAFixMOnlyCorrection(
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
    correction = LearnedBPlaneAFixMOnlyCorrection(
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


def test_m_only_optimizer_parameter_helper_excludes_a() -> None:
    correction = LearnedBPlaneAFixMOnlyCorrection(
        q_bfree=_q(),
        a_free=_a(),
        seed=6201,
    )

    class Wrapper:
        pass

    wrapper = Wrapper()
    wrapper.correction = correction
    params = runner._m_only_optimizer_parameters(wrapper)
    assert len(params) == 1
    assert params[0] is correction.M_theta.weight
    assert all(p is not correction.A_theta.weight for p in params)
    assert sum(p.numel() for p in params) == 4


@pytest.mark.parametrize("seed", TRAINING_SEEDS)
def test_actual_source_checkpoint_authentication(seed: int) -> None:
    q, a_free, metadata = load_seed_matched_afix_monly_source(ROOT, seed)
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


def test_frozen_ainit_recovery_reference_is_exact() -> None:
    assert runner.AINIT_RECOVERY_BY_SEED == {
        6201: 0.135931570756102,
        6202: 0.124898380318522,
        6203: 0.139898741881301,
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


def test_execution_authority_tokens_are_afix_monly_specific() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert (
        "SCIENTIFIC_EXECUTION_ALLOWED="
        "YES_STAGE_E_BFREE_AFIX_MONLY_THREE_CELL_DIAGNOSTIC"
    ) in source
    assert (
        "TRAINING_ALLOWED="
        "YES_EXACT_STAGE_E_BFREE_AFIX_MONLY_THREE_CELL"
    ) in source
    assert "OPTIMIZER_ALLOWED=YES_ADAMW_20_STEPS_M_ONLY" in source


def test_runtime_gradient_firewall_requires_no_a_grad() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "require(a_grad is None" in source
    assert "A_MUTATION_AFTER_STEP" in source
    assert "M_THETA_WEIGHT_ONLY" in source


def test_static_authentication_allows_only_three_new_files() -> None:
    assert runner.IMPLEMENTATION_PATHS == frozenset({
        "src/contramamba/gen5_stage_e_learned_b_plane_afix_monly_control.py",
        "scripts/train_reason_router_gen5_stage_e_learned_b_plane_afix_monly_control.py",
        "tests/test_reason_router_gen5_stage_e_learned_b_plane_afix_monly_control.py",
    })
