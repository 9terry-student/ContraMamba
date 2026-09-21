from __future__ import annotations

import inspect
import math
import subprocess
import sys
from pathlib import Path

import torch

from scripts import (
    reason_router_gen4_mamba14b_transported_p5_adjacent_response_raw_fast_cuda
    as transported,
)


def test_protocol_freezes_followup_before_response() -> None:
    transported.validate_protocol()
    assert transported.PAIR_FIRST == 5101
    assert transported.PAIR_LAST == 5400
    assert transported.PAIR_COUNT == 300
    assert transported.CELL == "C2_NAME"
    assert transported.ANCHOR_NAME == "A_IDENTITY"
    assert transported.TARGET_OFFSET == 2
    assert transported.SOURCE_BLOCK == 35
    assert transported.TARGET_BLOCK == 36
    assert transported.EPSILON == 0.025
    assert transported.ADJACENT_CONTROL_PLANE == "P4"
    assert transported.PLANNED_PRIMARY_ENDPOINT == "G=D_TRANSPORT-D_ADJ"
    assert (
        transported.PLANNED_PRIMARY_TEST
        == "paired_one_sample_student_t_greater"
    )
    assert transported.PLANNED_ALPHA == 0.05
    assert transported.PLANNED_PRIMARY_P_VALUE_COUNT == 1
    assert transported.PLANNED_SIGN_GATE == "mean(D_TRANSPORT)>0"
    assert transported.D_CAN_ROLE == "descriptive_only"
    assert (
        transported.CONTROL_EXTENSION_RULE
        == (
            "orthogonal_residualization_of_frozen_adjacent_P4_against_"
            "row_conditioned_transported_P5"
        )
    )


def test_forward_budget_is_one_adjacent_response_plus_transport() -> None:
    assert transported.RESPONSE_FORWARDS_PER_PAIR == 80
    assert transported.TOTAL_RESPONSE_FORWARDS == 24000
    assert transported.TOTAL_TRANSPORT_BOUNDARY_FORWARDS == 300
    assert transported.TOTAL_LOCAL_FD_FORWARDS == 1200
    assert transported.TOTAL_FULL_MODEL_FORWARDS == 24300
    assert transported.PAIRS_PER_SHARD == 150
    assert len(transported.SHARDS) == 2


def test_strong_projection_perfect_retention() -> None:
    old_ambient = transported.AMBIENT_DIM
    old_strong = transported.ADJACENT_STRONG_DIM
    try:
        transported.AMBIENT_DIM = 4
        transported.ADJACENT_STRONG_DIM = 2

        plus = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        minus = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
        plane, diag = transported.restrict_transported_plane_to_adjacent_strong(
            plus,
            minus,
            [0, 1],
        )
    finally:
        transported.AMBIENT_DIM = old_ambient
        transported.ADJACENT_STRONG_DIM = old_strong

    matrix = torch.stack([plane["plus"], plane["minus"]], dim=1)
    assert torch.allclose(
        matrix.T @ matrix,
        torch.eye(2, dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )
    assert diag["ambient_w_rank"] == 2
    assert diag["strong_projection_rank"] == 2
    assert all(
        math.isclose(value, 1.0, abs_tol=1e-12)
        for value in diag["strong_projection_singular_values"]
    )


def test_strong_projection_rank_loss_is_blocking() -> None:
    old_ambient = transported.AMBIENT_DIM
    old_strong = transported.ADJACENT_STRONG_DIM
    try:
        transported.AMBIENT_DIM = 4
        transported.ADJACENT_STRONG_DIM = 2

        plus = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        minus = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float64)

        try:
            transported.restrict_transported_plane_to_adjacent_strong(
                plus,
                minus,
                [0, 1],
            )
        except transported.TransportedResponseError as exc:
            assert "STRONG_PROJECTED_RANK" in str(exc)
        else:
            raise AssertionError("expected rank-loss block")
    finally:
        transported.AMBIENT_DIM = old_ambient
        transported.ADJACENT_STRONG_DIM = old_strong


def test_pair_planes_residualizes_nonorthogonal_p4_without_reselection() -> None:
    old_strong = transported.ADJACENT_STRONG_DIM
    try:
        transported.ADJACENT_STRONG_DIM = 4
        root2 = math.sqrt(2.0)
        frozen = {
            "P4": {
                "plus": torch.tensor(
                    [1.0 / root2, 0.0, 1.0 / root2, 0.0],
                    dtype=torch.float64,
                ),
                "minus": torch.tensor(
                    [0.0, 1.0 / root2, 0.0, 1.0 / root2],
                    dtype=torch.float64,
                ),
            },
            "P5": {
                "plus": torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float64),
                "minus": torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64),
            },
        }
        selected = {
            "plus": torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
            "minus": torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64),
        }

        out, diag = transported.pair_planes(frozen, selected)
        selected_matrix = torch.stack(
            [out["P5"]["plus"], out["P5"]["minus"]],
            dim=1,
        )
        control_matrix = torch.stack(
            [out["P4"]["plus"], out["P4"]["minus"]],
            dim=1,
        )

        assert torch.equal(out["P5"]["plus"], selected["plus"])
        assert torch.equal(out["P5"]["minus"], selected["minus"])
        assert torch.allclose(
            control_matrix.T @ control_matrix,
            torch.eye(2, dtype=torch.float64),
            atol=1e-12,
            rtol=0.0,
        )
        assert torch.allclose(
            selected_matrix.T @ control_matrix,
            torch.zeros((2, 2), dtype=torch.float64),
            atol=1e-12,
            rtol=0.0,
        )
        assert diag["control_source_plane"] == "P4"
        assert diag["control_residual_rank"] == 2
        assert diag["selected_control_cross_max_abs_before"] > 0.0
        assert diag["selected_control_cross_max_abs_after"] <= 1e-12
    finally:
        transported.ADJACENT_STRONG_DIM = old_strong


def test_pair_planes_is_identity_for_already_orthogonal_p4() -> None:
    old_strong = transported.ADJACENT_STRONG_DIM
    try:
        transported.ADJACENT_STRONG_DIM = 4
        selected = {
            "plus": torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
            "minus": torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64),
        }
        frozen = {
            "P4": {
                "plus": torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float64),
                "minus": torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64),
            },
            "P5": selected,
        }

        out, diag = transported.pair_planes(frozen, selected)
        assert torch.allclose(
            out["P4"]["plus"],
            frozen["P4"]["plus"],
            atol=1e-12,
            rtol=0.0,
        )
        assert torch.allclose(
            out["P4"]["minus"],
            frozen["P4"]["minus"],
            atol=1e-12,
            rtol=0.0,
        )
        assert diag["selected_control_cross_max_abs_before"] <= 1e-12
        assert diag["selected_control_cross_max_abs_after"] <= 1e-12
    finally:
        transported.ADJACENT_STRONG_DIM = old_strong


def test_pair_planes_blocks_if_p4_loses_rank_after_selected_residualization() -> None:
    old_strong = transported.ADJACENT_STRONG_DIM
    try:
        transported.ADJACENT_STRONG_DIM = 4
        selected = {
            "plus": torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
            "minus": torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64),
        }
        frozen = {
            "P4": selected,
            "P5": selected,
        }

        try:
            transported.pair_planes(frozen, selected)
        except transported.TransportedResponseError as exc:
            assert "CONTROL_RESIDUAL_RANK" in str(exc)
        else:
            raise AssertionError("expected control residual rank-loss block")
    finally:
        transported.ADJACENT_STRONG_DIM = old_strong


def test_transport_plane_uses_validated_fixed_fd() -> None:
    source = inspect.getsource(transported.build_pair_transported_plane)
    assert "ref_fd.fast_symmetric_fd" in source
    assert "for name in BASIS_NAMES" in source
    assert "direct.capture_baseline_boundary" in source
    assert "CELL" in source
    assert "exact_reference_jvp" not in source
    assert "torch.func.jvp" not in source


def test_raw_execution_does_not_read_historical_response_values() -> None:
    source = inspect.getsource(transported.run_raw)
    worker = inspect.getsource(transported.worker_run)
    combined = source + worker

    assert "HISTORICAL_ITEMS_PATH.read" not in combined
    assert "HISTORICAL_SUMMARY_PATH.read" not in combined
    assert "scipy" not in combined.lower()
    assert "ttest" not in combined.lower()
    assert '"inferential_test_performed": False' in source


def test_same_pair_plane_is_passed_once_to_both_response_branches() -> None:
    source = inspect.getsource(transported.worker_run)
    assert "planes, control_diag = pair_planes" in source
    assert "planes=planes" in source
    assert "core.run_pair" in source
    assert "transported_plane" in source


def test_standalone_help_imports_from_repo_root() -> None:
    repo_root = Path(transported.__file__).resolve().parents[1]
    script_path = (
        repo_root
        / "scripts"
        / "reason_router_gen4_mamba14b_transported_p5_adjacent_response_raw_fast_cuda.py"
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
    assert "5101..5400" in completed.stdout
    assert "no p-value" in completed.stdout
