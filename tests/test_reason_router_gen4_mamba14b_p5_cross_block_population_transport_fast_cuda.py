from __future__ import annotations

import inspect
import math
import subprocess
import sys
from pathlib import Path

import torch

from scripts import (
    reason_router_gen4_mamba14b_p5_cross_block_population_transport_fast_cuda
    as population,
)


def test_protocol_is_exact_frozen_population() -> None:
    population.validate_protocol()
    assert population.FAMILIES == ("xg2", "xg4")
    assert population.SOURCE_PAIR_FIRST == 301
    assert population.SOURCE_PAIR_LAST == 600
    assert population.SOURCE_PAIR_COUNT_PER_FAMILY == 300
    assert population.TOTAL_ROW_COUNT == 600
    assert population.CELL == "C2_NAME"
    assert population.ANCHOR_NAME == "A_IDENTITY"
    assert population.TARGET_OFFSET == 2
    assert population.SOURCE_BLOCK == 35
    assert population.TARGET_BLOCK == 36
    assert population.AMBIENT_DIM == 4096
    assert population.SOURCE_PLANE == "P5"
    assert population.TARGET_PLANE == "P5"
    assert population.EPSILON == 0.025
    assert population.GPU_COUNT == 2


def test_pins_validated_reference_fd_evidence_and_adjacent_plane() -> None:
    assert (
        population.REFERENCE_FD_SCRIPT_GIT_BLOB
        == "3dc94d833f81d829e81c85c0c3641fce861e8eb7"
    )
    assert (
        population.REFERENCE_FD_REPORT_GIT_BLOB
        == "11ecd4b0122f5d9db126e3ab3dd6fef5f048b8ad"
    )
    assert (
        population.ADJACENT_STRONG_GIT_BLOB
        == "dc5f8f0af4d0375a88fbee93c8769bf4b64492cd"
    )
    assert (
        population.ADJACENT_P5_PLUS_GIT_BLOB
        == "6d83d1fc64bcbb7b1c9a779399dd405d82a063a6"
    )
    assert (
        population.ADJACENT_P5_MINUS_GIT_BLOB
        == "0ff81176cd5be099eb0da344293621d74b3ce39b"
    )


def test_expected_pair_order() -> None:
    xg2 = population.expected_pairs("xg2")
    xg4 = population.expected_pairs("xg4")
    assert len(xg2) == len(xg4) == 300
    assert xg2[0] == "xg2_fact_301"
    assert xg2[-1] == "xg2_fact_600"
    assert xg4[0] == "xg4_fact_301"
    assert xg4[-1] == "xg4_fact_600"


def test_transport_metrics_perfect_alignment() -> None:
    e1 = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    e2 = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
    target = torch.stack([e1, e2], dim=1)

    result = population.transport_metrics(e1, e2, target)

    assert result["w_rank"] == 2
    assert math.isclose(result["w_condition_number"], 1.0, abs_tol=1e-12)
    assert result["qt_u36_singular_values"] == [1.0, 1.0]
    assert all(abs(v) <= 1e-12 for v in result["principal_angles_radians"])
    assert math.isclose(result["projector_overlap"], 1.0, abs_tol=1e-12)
    assert math.isclose(result["procrustes_residual_fro"], 0.0, abs_tol=1e-12)


def test_transport_metrics_orthogonal_subspaces() -> None:
    w1 = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    w2 = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
    u1 = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float64)
    u2 = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
    target = torch.stack([u1, u2], dim=1)

    result = population.transport_metrics(w1, w2, target)

    assert result["w_rank"] == 2
    assert all(abs(v) <= 1e-12 for v in result["qt_u36_singular_values"])
    assert all(
        math.isclose(v, 90.0, abs_tol=1e-10)
        for v in result["principal_angles_degrees"]
    )
    assert math.isclose(result["projector_overlap"], 0.0, abs_tol=1e-12)
    assert math.isclose(result["procrustes_residual_fro"], 2.0, abs_tol=1e-12)


def test_transport_metrics_rank_collapse_keeps_alignment_undefined() -> None:
    w1 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    w2 = 2.0 * w1
    target = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=torch.float64,
    )

    result = population.transport_metrics(w1, w2, target)

    assert result["w_rank"] == 1
    assert result["w_condition_number"] is None
    assert result["qt_u36_singular_values"] is None
    assert result["principal_angles_radians"] is None
    assert result["projector_overlap"] is None
    assert result["procrustes_residual_fro"] is None


def test_runner_uses_validated_fd_exactly_twice_per_row() -> None:
    source = inspect.getsource(population.run_family)
    assert "ref_fd.fast_symmetric_fd" in source
    assert "for basis_name in BASIS_NAMES" in source
    assert "local_perturbed_forward_count += 2" in source
    assert "exact_reference_jvp" not in source
    assert "torch.func.jvp" not in source
    assert "EPSILON" not in inspect.signature(
        population.ref_fd.fast_symmetric_fd
    ).parameters


def test_measurement_has_no_response_or_inferential_branch() -> None:
    source = inspect.getsource(population)
    lower = source.lower()
    assert '"xg1_experiment5_response_accessed": False' in source
    assert '"statistical_testing_performed": False' in source
    assert '"p_value_count_added": 0' in source
    assert '"interpretation_threshold_applied": False' in source
    assert '"preserved_reoriented_collapse_label_emitted": False' in source
    assert "scipy" not in lower
    assert "ttest" not in lower
    assert "holm" not in lower


def test_summary_is_descriptive_only() -> None:
    rows = [
        {
            "w_rank": 2,
            "jv_plus_l2": 1.0,
            "jv_minus_l2": 2.0,
            "w_condition_number": 2.0,
            "projector_overlap": 0.75,
            "procrustes_residual_fro": 0.5,
            "procrustes_residual_normalized": 0.5 / math.sqrt(2.0),
            "w_singular_values": [2.0, 1.0],
            "qt_u36_singular_values": [0.9, 0.8],
            "principal_angles_degrees": [10.0, 20.0],
        },
        {
            "w_rank": 1,
            "jv_plus_l2": 1.5,
            "jv_minus_l2": 2.5,
            "w_condition_number": None,
            "projector_overlap": None,
            "procrustes_residual_fro": None,
            "procrustes_residual_normalized": None,
            "w_singular_values": [2.5, 0.0],
            "qt_u36_singular_values": None,
            "principal_angles_degrees": None,
        },
    ]
    summary = population.summarize_rows(rows)
    assert summary["row_count"] == 2
    assert summary["rank_counts"] == {"0": 0, "1": 1, "2": 1}
    assert summary["projector_overlap"]["count"] == 1
    assert summary["w_condition_number"]["count"] == 1


def test_standalone_help_imports_from_repo_root() -> None:
    repo_root = Path(population.__file__).resolve().parents[1]
    script_path = (
        repo_root
        / "scripts"
        / "reason_router_gen4_mamba14b_p5_cross_block_population_transport_fast_cuda.py"
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
    assert "N=600" in completed.stdout
    assert "no p-values" in completed.stdout
