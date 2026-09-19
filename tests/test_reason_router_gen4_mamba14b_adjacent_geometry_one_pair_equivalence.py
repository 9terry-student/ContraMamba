from __future__ import annotations

import inspect

import pytest
import torch

from scripts import (
    reason_router_gen4_mamba14b_adjacent_geometry_one_pair_equivalence
    as m,
)


def test_protocol_constants() -> None:
    assert m.FAMILIES == ("xg2", "xg4")
    assert m.PAIR_BY_FAMILY == {
        "xg2": "xg2_fact_301",
        "xg4": "xg4_fact_301",
    }
    assert m.GEOMETRY_ATOL == 1.0e-4
    assert m.GEOMETRY_RTOL == 1.0e-4
    assert m.CPU_MODEL_FORWARD_COUNT == 8
    assert m.GPU_MODEL_FORWARD_COUNT == 8
    assert m.TOTAL_MODEL_FORWARD_COUNT == 16


def test_adjacent_site_configuration() -> None:
    m.adjacent._configure_canonical_module()
    m.adjacent.validate_protocol_constants()
    assert m.canonical.SOURCE_BLOCK == 34
    assert m.canonical.TARGET_RESIDUAL_LAYER == 35
    assert m.canonical.INTERVENTION_LAYER == 36
    assert m.canonical.TARGET_OFFSET == 2


def test_tensor_comparator_accepts_frozen_tolerance() -> None:
    cpu = torch.tensor([1.0, 2.0], dtype=torch.float64)
    gpu = torch.tensor(
        [1.0 + 5e-5, 2.0 - 5e-5],
        dtype=torch.float64,
    )
    assert m._compare_tensor(cpu, gpu, label="fixture") <= 5.1e-5


def test_tensor_comparator_blocks_large_mismatch() -> None:
    cpu = torch.tensor([1.0], dtype=torch.float64)
    gpu = torch.tensor([1.1], dtype=torch.float64)
    with pytest.raises(
        m.AdjacentGeometryEquivalenceError,
        match="EQUIVALENCE_FAILURE",
    ):
        m._compare_tensor(cpu, gpu, label="fixture")


def test_scalar_comparator_accepts_frozen_tolerance() -> None:
    assert (
        m._compare_scalar(1.0, 1.00005, label="fixture")
        <= 5.1e-5
    )


def test_cpu_capture_is_explicitly_slow() -> None:
    source = inspect.getsource(m.capture_geometry_branch_cpu_slow)
    assert "cuda_kernels_forward" not in source
    assert 'parameter.device.type == "cpu"' in source
    assert "fast_path_calls" in source


def test_gpu_capture_reuses_canonical_fast_path() -> None:
    source = inspect.getsource(m.run_gate)
    assert "canonical.capture_geometry_branch" in source
    assert "canonical.validate_fast_runtime_for_device(0)" in source
    assert "kernel_compat.validate_transformers_kernel_bindings" in source


def test_no_scientific_response_or_inference_path() -> None:
    source = inspect.getsource(m)
    required = (
        '"xg1_accessed": False',
        '"causal_response_observed": False',
        '"plane_selection_performed": False',
        '"control_selection_performed": False',
        '"statistical_testing_performed": False',
        '"scientific_conclusion": None',
        '"scientific_model_forward_count": 0',
    )
    for token in required:
        assert token in source

    forbidden = (
        "scipy",
        "ttest",
        "binomtest",
        "holm",
        "multipletests",
        "xg1_fact_5101",
    )
    lower = source.lower()
    for token in forbidden:
        assert token.lower() not in lower


def test_report_does_not_authorize_full_run() -> None:
    source = inspect.getsource(m)
    assert (
        '"full_adjacent_geometry_execution_authorized_by_this_artifact": False'
        in source
    )

def test_authenticate_repo_accepts_detached_exact_clean(monkeypatch) -> None:
    expected = "a" * 40

    def fake_git(*args: str) -> str:
        if args == ("branch", "--show-current"):
            return ""
        if args == ("rev-parse", "HEAD"):
            return expected
        if args == ("status", "--porcelain"):
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(m, "git", fake_git)
    monkeypatch.setattr(m, "git_rc", lambda *args: 0)

    m.authenticate_repo(expected)


def test_authenticate_repo_rejects_unexpected_named_branch(monkeypatch) -> None:
    expected = "b" * 40

    def fake_git(*args: str) -> str:
        if args == ("branch", "--show-current"):
            return "wrong-branch"
        if args == ("rev-parse", "HEAD"):
            return expected
        if args == ("status", "--porcelain"):
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(m, "git", fake_git)
    monkeypatch.setattr(m, "git_rc", lambda *args: 0)

    with pytest.raises(
        m.AdjacentGeometryEquivalenceError,
        match="BRANCH_MISMATCH:wrong-branch",
    ):
        m.authenticate_repo(expected)
