from __future__ import annotations

import argparse

import pytest
import torch

from scripts import train_reason_router_gen5_ainit_rng_causal_intervention as mod
from scripts import train_reason_router_gen5_phase3a_contention as p3a


def _args(**overrides):
    values = {
        "static_verify_only": False,
        "cuda_preflight_only": False,
        "run_cell": False,
        "run_matrix": False,
        "expected_head": mod.IMPLEMENTATION_AUTHORITY_COMMIT,
        "allow_opening_worktree": False,
        "implementation_freeze_commit": None,
        "execution_authority_commit": None,
        "a_init_seed": None,
        "training_rng_seed": None,
        "tokenizer_snapshot": None,
        "model_snapshot": None,
        "checkpoint": None,
        "preflight_output": None,
        "output_root": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_exact_factor_levels_and_offdiagonal_cells():
    assert mod.FACTOR_SEEDS == (6201, 6202, 6203)
    assert mod.OFFDIAGONAL_CELLS == (
        (6201, 6202),
        (6201, 6203),
        (6202, 6201),
        (6202, 6203),
        (6203, 6201),
        (6203, 6202),
    )
    assert len(mod.OFFDIAGONAL_CELLS) == 6
    assert all(a != r for a, r in mod.OFFDIAGONAL_CELLS)


def test_diagonal_cells_are_reused_not_scheduled():
    diagonal = {(s, s) for s in mod.FACTOR_SEEDS}
    assert diagonal.isdisjoint(set(mod.OFFDIAGONAL_CELLS))
    assert set(mod.DIAGONAL_SOURCES) == set(mod.FACTOR_SEEDS)


def test_exact_two_gpu_worker_assignment_and_balance():
    assert mod.MATRIX_GPU_COUNT == 2
    assert mod.GPU_TOPOLOGY == "TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP"
    assert mod.MATRIX_WORKER_CELLS == (
        ((6201, 6202), (6202, 6203), (6203, 6201)),
        ((6201, 6203), (6202, 6201), (6203, 6202)),
    )
    flattened = [
        cell
        for worker in mod.MATRIX_WORKER_CELLS
        for cell in worker
    ]
    assert set(flattened) == set(mod.OFFDIAGONAL_CELLS)
    assert len(set(flattened)) == 6
    for worker in mod.MATRIX_WORKER_CELLS:
        assert {a for a, _ in worker} == set(mod.FACTOR_SEEDS)
        assert {r for _, r in worker} == set(mod.FACTOR_SEEDS)


@pytest.mark.parametrize("seed", mod.FACTOR_SEEDS)
def test_deterministic_a_initialization_hash(seed):
    first = mod.reconstruct_a_init(seed)
    second = mod.reconstruct_a_init(seed)
    assert torch.equal(first, second)
    assert mod.tensor_sha256(first) == mod.A_INIT_SHA256[seed]


def test_all_three_a_initializations_are_distinct():
    hashes = {
        mod.tensor_sha256(mod.reconstruct_a_init(seed))
        for seed in mod.FACTOR_SEEDS
    }
    assert len(hashes) == 3


@pytest.mark.parametrize("a_init_seed", mod.FACTOR_SEEDS)
def test_training_rng_seed_does_not_change_reconstructed_a(a_init_seed):
    reference = mod.reconstruct_a_init(a_init_seed)
    for training_rng_seed in mod.FACTOR_SEEDS:
        # A construction has no training_rng_seed input by design.
        candidate = mod.reconstruct_a_init(a_init_seed)
        assert torch.equal(reference, candidate)
        assert (
            mod.tensor_sha256(candidate)
            == mod.A_INIT_SHA256[a_init_seed]
        )


def test_changing_a_seed_changes_a_while_b_is_exact_zero():
    a_values = [
        mod.reconstruct_a_init(seed)
        for seed in mod.FACTOR_SEEDS
    ]
    assert not torch.equal(a_values[0], a_values[1])
    assert not torch.equal(a_values[0], a_values[2])
    assert not torch.equal(a_values[1], a_values[2])

    b = mod.zero_b_init()
    assert tuple(b.shape) == (24576, 2)
    assert b.dtype == torch.float32
    assert int(torch.count_nonzero(b).item()) == 0


def test_inherited_phase3a_training_contract_is_exact():
    assert mod.PRESSURE == "P0"
    assert mod.ARM == "G5-C0"
    assert mod.TRAIN_ROWS == p3a.TRAIN_ROWS == 3360
    assert mod.DEV_ROWS == p3a.DEV_ROWS == 840
    assert mod.SPLIT_SEED == p3a.SPLIT_SEED == 16384
    assert mod.TOTAL_OPTIMIZER_STEPS == p3a.TOTAL_OPTIMIZER_STEPS == 20
    assert mod.LEARNING_RATE == p3a.LEARNING_RATE == 0.001
    assert mod.WEIGHT_DECAY == p3a.WEIGHT_DECAY == 0.0001
    assert mod.GRADIENT_CLIP_NORM == p3a.GRADIENT_CLIP_NORM == 5.0
    assert mod.CONFIRMATORY_9601_9900_ALLOWED is False


def test_exact_frozen_diagonal_sources():
    expected = {
        6201: (
            "reports/reason_router_gen5_phase3a_training_runs/"
            "gen5-phase3a-contention-qualification-9cell-d58e894-retry3/"
            "cells/seed6201/P0/final_correction.pt",
            "157ca1c945c7f70b03638ef7e06750a4f4105e5504904272529ecd82bfba5ddf",
        ),
        6202: (
            "reports/reason_router_gen5_phase3a_training_runs/"
            "gen5-phase3a-contention-qualification-9cell-d58e894-retry3/"
            "cells/seed6202/P0/final_correction.pt",
            "1aa16196aa5aa338c30ac71dd20c7a6fb36a62eafdbe52bcf5696cfc3b87c214",
        ),
        6203: (
            "reports/reason_router_gen5_phase3a_training_runs/"
            "gen5-phase3a-contention-qualification-9cell-d58e894-retry3/"
            "cells/seed6203/P0/final_correction.pt",
            "c536464dd8541423d18a2bfbfee40211f068885baa37730315dbf7cf0a784770",
        ),
    }
    assert mod.DIAGONAL_SOURCES == expected


@pytest.mark.parametrize("cell", mod.OFFDIAGONAL_CELLS)
def test_authorized_offdiagonal_cells_validate(cell):
    mod.validate_offdiagonal_cell(*cell)


@pytest.mark.parametrize("seed", mod.FACTOR_SEEDS)
def test_diagonal_cells_are_rejected(seed):
    with pytest.raises(
        mod.AInitRNGCausalInterventionError,
        match="NOT_AUTHORIZED_OFFDIAGONAL",
    ):
        mod.validate_offdiagonal_cell(seed, seed)


def test_static_mode_forbids_runtime_authority_arguments():
    args = _args(
        static_verify_only=True,
        implementation_freeze_commit="deadbeef",
    )
    with pytest.raises(
        mod.AInitRNGCausalInterventionError,
        match="STATIC_RUNTIME_ARG_FORBIDDEN",
    ):
        mod.validate_mode_args(args)


def test_runtime_mode_blocked_without_future_authority():
    args = _args(
        run_matrix=True,
        checkpoint="checkpoint.pt",
        output_root="out",
    )
    with pytest.raises(
        mod.AInitRNGCausalInterventionError,
        match="RUNTIME_IMPLEMENTATION_FREEZE_REQUIRED",
    ):
        mod.validate_mode_args(args)


def test_matrix_mode_rejects_individual_seed_arguments():
    args = _args(
        run_matrix=True,
        implementation_freeze_commit="freeze",
        execution_authority_commit="authority",
        checkpoint="checkpoint.pt",
        output_root="out",
        a_init_seed=6201,
    )
    with pytest.raises(
        mod.AInitRNGCausalInterventionError,
        match="MATRIX_A_INIT_SEED_FORBIDDEN",
    ):
        mod.validate_mode_args(args)


def test_cell_name_is_unambiguous():
    assert mod.cell_name(6201, 6202) == "A6201-R6202"
    assert mod.cell_name(6203, 6201) == "A6203-R6201"


def test_static_contract_validation_passes_without_runtime():
    mod._validate_contract()
