from __future__ import annotations

import argparse
import inspect
import math

import pytest
import torch

from scripts import audit_reason_router_gen5_ainit_temporal_birth as mod
from scripts import train_reason_router_gen5_ainit_rng_causal_intervention as base


def _args(**overrides):
    values = {
        "static_verify_only": False,
        "cuda_preflight_only": False,
        "replay_auth_recovery_only": False,
        "replay_auth_recovery_worker": False,
        "run_replay_matrix": False,
        "run_replay_worker": False,
        "expected_head": mod.AUTHORITY_COMMIT,
        "allow_opening_worktree": False,
        "implementation_freeze_commit": None,
        "model_snapshot": None,
        "tokenizer_snapshot": None,
        "checkpoint": None,
        "output_root": None,
        "scratch_root": None,
        "worker_id": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_full_factorial_grid_and_exact_two_gpu_parity_sharding():
    assert mod.FACTOR_SEEDS == (6201, 6202, 6203)
    assert mod.FULL_FACTORIAL_CELLS == (
        (6201, 6201),
        (6201, 6202),
        (6201, 6203),
        (6202, 6201),
        (6202, 6202),
        (6202, 6203),
        (6203, 6201),
        (6203, 6202),
        (6203, 6203),
    )
    assert mod.GPU_QUEUES == {
        0: (
            (6201, 6201),
            (6201, 6203),
            (6202, 6202),
            (6203, 6201),
            (6203, 6203),
        ),
        1: (
            (6201, 6202),
            (6202, 6201),
            (6202, 6203),
            (6203, 6202),
        ),
    }
    mod.validate_gpu_queues()


@pytest.mark.parametrize("cell", mod.FULL_FACTORIAL_CELLS)
def test_low_level_factor_validation_accepts_full_3x3(cell):
    base.validate_factor_cell(*cell)
    mod.validate_full_factorial_cell(*cell)


@pytest.mark.parametrize("seed", mod.FACTOR_SEEDS)
def test_historical_offdiagonal_validation_still_rejects_diagonal(seed):
    with pytest.raises(
        base.AInitRNGCausalInterventionError,
        match="NOT_AUTHORIZED_OFFDIAGONAL",
    ):
        base.validate_offdiagonal_cell(seed, seed)


def test_runtime_model_low_level_uses_full_factor_validation():
    source = inspect.getsource(base._prepare_runtime_model)
    executable_lines = [
        line.strip()
        for line in source.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert "validate_factor_cell(a_init_seed, training_rng_seed)" in executable_lines
    assert not any(
        line.startswith("validate_offdiagonal_cell(")
        for line in executable_lines
    )


def test_exact_frozen_checkpoint_source_matrix():
    assert set(mod.FROZEN_FINAL_SOURCES) == set(mod.FULL_FACTORIAL_CELLS)
    assert len(mod.FROZEN_FINAL_SOURCES) == 9
    expected_hashes = {
        "157ca1c945c7f70b03638ef7e06750a4f4105e5504904272529ecd82bfba5ddf",
        "ef03fbbedf3fab8efb255f2ccb6ec33cdb65ee6881a92e6b4d43fe1e719e40d4",
        "15582eda034befb1c8d202f04c494f7fd5882bf9fd60ddc963232761057b3df7",
        "3c217b43eb164583980bee91c39d53a7a3dc20d181e0341db35ffe8ece162ed3",
        "1aa16196aa5aa338c30ac71dd20c7a6fb36a62eafdbe52bcf5696cfc3b87c214",
        "d1c478b448c5f53e2a97552196a454f63f9d9081de614a6d9a5cd4e389e30ee2",
        "d9e1062baf554867b212da57c9d30fe8efeccafc77255ba869affac691606359",
        "32943df3558ef72eb7f7b7bfb70c6a1a03185a1ea6ca4dd83b9677cd59d13698",
        "c536464dd8541423d18a2bfbfee40211f068885baa37730315dbf7cf0a784770",
    }
    assert {row[1] for row in mod.FROZEN_FINAL_SOURCES.values()} == expected_hashes


def test_operator_inner_matches_explicit_matrix_product():
    generator = torch.Generator().manual_seed(123)
    a1 = torch.randn((2, 5), generator=generator)
    b1 = torch.randn((7, 2), generator=generator)
    a2 = torch.randn((2, 5), generator=generator)
    b2 = torch.randn((7, 2), generator=generator)

    explicit1 = b1.to(torch.float64) @ a1.to(torch.float64)
    explicit2 = b2.to(torch.float64) @ a2.to(torch.float64)

    expected_inner = float(torch.sum(explicit1 * explicit2).item())
    expected_distance_sq = float(
        torch.sum((explicit1 - explicit2) ** 2).item()
    )

    assert math.isclose(
        mod.operator_inner(a1, b1, a2, b2),
        expected_inner,
        rel_tol=0.0,
        abs_tol=1e-10,
    )
    assert math.isclose(
        mod.operator_distance_sq(a1, b1, a2, b2),
        expected_distance_sq,
        rel_tol=0.0,
        abs_tol=1e-10,
    )


def test_operator_snapshot_singular_values_match_explicit_product():
    generator = torch.Generator().manual_seed(321)
    a = torch.randn((2, 5), generator=generator)
    b = torch.randn((7, 2), generator=generator)
    observed = mod.operator_snapshot_stats(a, b)
    expected = torch.linalg.svdvals(
        b.to(torch.float64) @ a.to(torch.float64)
    )
    assert observed["nonzero_singular_values"] == pytest.approx(
        expected[:2].tolist(),
        rel=1e-12,
        abs=1e-12,
    )
    assert expected[2:].tolist() == pytest.approx(
        [0.0] * int(expected.numel() - 2),
        rel=0.0,
        abs=1e-12,
    )
    assert observed["frobenius_norm"] == pytest.approx(
        float(torch.linalg.vector_norm(
            b.to(torch.float64) @ a.to(torch.float64)
        ).item()),
        rel=1e-12,
        abs=1e-12,
    )


def test_factorial_energy_detects_pure_a_effect():
    cells = list(mod.FULL_FACTORIAL_CELLS)
    # Complete grid with identical values within each A level.
    complete = {
        (a, r): torch.tensor([[float(a - 6200)]], dtype=torch.float64)
        for a, r in cells
    }
    ordered, gram = mod._tensor_gram(complete)
    result = mod.factorial_energy_from_gram(ordered, gram)
    fractions = result["normalized_fractions"]
    assert fractions is not None
    assert fractions["A"] == pytest.approx(1.0, abs=1e-12)
    assert fractions["R"] == pytest.approx(0.0, abs=1e-12)
    assert fractions["AR"] == pytest.approx(0.0, abs=1e-12)
    assert result["closure_abs"] <= 1e-10


def test_zero_grid_has_no_normalized_pair_residual_or_factor_fraction():
    values = {
        cell: torch.zeros((3, 2), dtype=torch.float32)
        for cell in mod.FULL_FACTORIAL_CELLS
    }
    ordered, gram = mod._tensor_gram(values)
    grouped = mod.grouped_pair_stats_from_gram(ordered, gram)
    factorial = mod.factorial_energy_from_gram(ordered, gram)

    assert (
        grouped["same_training_rng_different_a"]["mean_squared_distance"]
        == 0.0
    )
    assert (
        grouped["same_training_rng_different_a"]["mean_normalized_residual"]
        is None
    )
    assert (
        grouped["same_a_different_training_rng"]["mean_normalized_residual"]
        is None
    )
    assert grouped["A_over_R_pair_distance_ratio"] is None
    assert factorial["normalized_fractions"] is None



def test_replay_auth_recovery_cells_cover_two_historical_lineages():
    assert mod.REPLAY_AUTH_RECOVERY_CELLS == {0: (6201, 6201), 1: (6201, 6202)}


@pytest.mark.parametrize("cell", [(6201, 6201), (6201, 6202)])
def test_replay_auth_recovery_historical_trace_is_frozen(cell):
    trace = mod.load_historical_training_trace(*cell)
    assert len(trace["training_losses"]) == mod.TOTAL_OPTIMIZER_STEPS
    assert len(trace["gradient_norms_before_clip"]) == mod.TOTAL_OPTIMIZER_STEPS
    assert trace["file_sha256"] == mod.RECOVERY_TRAINING_REPORT_SHA256[cell]


def test_replay_instrumentation_occurs_after_historical_sync():
    source = inspect.getsource(mod._run_replay_cell)
    assert source.index("optimizer.step()") < source.index("torch.cuda.synchronize()") < source.index("a_grad_cpu =") < source.index("public, private = snapshot_state(")
    assert "torch.linalg.vector_norm(a_grad).detach().cpu()" not in source
    assert "torch.linalg.vector_norm(b_grad).detach().cpu()" not in source


def test_replay_auth_recovery_mode_forbids_outputs():
    args = _args(replay_auth_recovery_only=True, implementation_freeze_commit="freeze", checkpoint="parent.pt", output_root="forbidden")
    with pytest.raises(mod.TemporalBirthError, match="RECOVERY_OUTPUT_FORBIDDEN"):
        mod.validate_args(args)


def test_replay_auth_recovery_worker_requires_fixed_id():
    args = _args(replay_auth_recovery_worker=True, implementation_freeze_commit="freeze", checkpoint="parent.pt", worker_id=3)
    with pytest.raises(mod.TemporalBirthError, match="RECOVERY_WORKER_ID_REQUIRED"):
        mod.validate_args(args)

def test_static_mode_forbids_runtime_fields():
    args = _args(
        static_verify_only=True,
        implementation_freeze_commit="freeze",
    )
    with pytest.raises(mod.TemporalBirthError, match="STATIC_RUNTIME_ARG"):
        mod.validate_args(args)


def test_runtime_requires_exact_implementation_freeze_argument():
    args = _args(
        run_replay_matrix=True,
        checkpoint="parent.pt",
        output_root="reports/reason_router_gen5_ainit_temporal_birth_replay_runs/x",
    )
    with pytest.raises(mod.TemporalBirthError, match="IMPLEMENTATION_FREEZE_REQUIRED"):
        mod.validate_args(args)


def test_authority_identity_and_collection_policy_are_frozen():
    assert mod.AUTHORITY_COMMIT == "20ae761dbff10ad70853b10910cbe12e51e0666a"
    assert mod.GPU_TOPOLOGY == "TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP"
    authority_text = (
        mod.ROOT / mod.AUTHORITY_PATH
    ).read_text(encoding="utf-8")
    assert "PREFLIGHT_COLLECTION=FORBIDDEN" in authority_text
    assert "FAILED_RUN_COLLECTION=FORBIDDEN" in authority_text
    assert "SUCCESSFUL_SCIENTIFIC_RUN_COLLECTION=REQUIRED" in authority_text
