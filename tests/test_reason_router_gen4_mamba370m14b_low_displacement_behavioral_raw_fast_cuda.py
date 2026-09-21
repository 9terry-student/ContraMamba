from __future__ import annotations

import inspect

import torch

from scripts import (
    reason_router_gen4_mamba370m14b_low_displacement_behavioral_raw_fast_cuda
    as runner,
)


def test_protocol_is_exact() -> None:
    runner.validate_protocol()
    assert runner.PAIR_FIRST == 5401
    assert runner.PAIR_LAST == 5700
    assert runner.PAIR_COUNT == 300
    assert runner.TARGET_CELLS == ("C0_SHAM", "C2_NAME")
    assert runner.ALPHAS == (0.5, 0.25)
    assert runner.PRIMARY_ALPHA == 0.25
    assert runner.CONDITIONS == (
        "restored_native",
        "control_alpha_0_5",
        "control_alpha_0_25",
    )
    assert runner.ROWS_PER_SCALE == 600
    assert runner.FORWARDS_PER_ROW == 3
    assert runner.FORWARDS_PER_SCALE == 1800
    assert runner.TOTAL_FORWARD_BUDGET == 3600
    assert runner.TOTAL_STATE_ROWS == 1200


def test_scale_mapping_and_frozen_planes() -> None:
    assert runner.SCALE_TO_PHYSICAL_GPU == {
        "mamba370m": 0,
        "mamba14b": 1,
    }
    m370 = runner.scale_spec("mamba370m")
    m14 = runner.scale_spec("mamba14b")
    assert m370["selected_plane"] == "P3"
    assert m370["control_plane"] == "P5"
    assert m14["selected_plane"] == "P5"
    assert m14["control_plane"] == "P4"


def test_scaled_full_correction_is_exact_for_binary_alphas() -> None:
    base = torch.tensor(
        [1.0, -2.0, 0.125, -0.0625],
        dtype=torch.float32,
    )
    half = runner.scaled_full_correction(base, 0.5)
    quarter = runner.scaled_full_correction(base, 0.25)
    assert torch.equal(half, base * 0.5)
    assert torch.equal(quarter, base * 0.25)


def test_no_alpha1_behavioral_condition_exists() -> None:
    assert 1.0 not in runner.ALPHAS
    assert all("1_0" not in x for x in runner.CONDITIONS)
    source = inspect.getsource(runner.run_three_forward_row)
    assert "ALPHAS" in source
    assert "FORWARDS_PER_ROW" in source


def test_raw_runner_contains_no_inference_or_rescue_logic() -> None:
    source = inspect.getsource(runner).lower()
    assert "scipy" not in source
    assert "ttest" not in source
    assert "holm" not in source
    assert '"primary_inference_executed": false' in source
    assert '"p_value_count_executed": 0' in source
    assert '"rescue_performed": false' in source


def test_state_capture_adds_no_extra_forward_or_backward() -> None:
    source = inspect.getsource(runner.run_three_forward_row)
    assert source.count("adapter.historical_forward(") == 2
    # One native call appears explicitly; two alpha calls share the loop body.
    assert "for alpha, condition in zip(" in source
    assert "torch.autograd.grad" not in source
    assert ".backward(" not in source


def test_required_raw_file_contracts() -> None:
    assert runner.BEHAVIOR_ITEM_FILE == "low_displacement_behavioral_items.jsonl"
    assert runner.BEHAVIOR_SUMMARY_FILE == "raw_behavioral_summary.json"
    assert runner.BEHAVIOR_MANIFEST_FILE == "artifact_manifest.json"
    assert runner.STATE_FILE == "low_displacement_states.npz"
    assert runner.STATE_INDEX_FILE == "capture_index.jsonl"
    assert runner.STATE_MANIFEST_FILE == "artifact_manifest.json"
    assert runner.CHECKSUM_FILE == "SHA256SUMS.txt"


def test_technical_gate_is_numeric_outcome_blind() -> None:
    source = inspect.getsource(runner.technical_gate_worker)
    assert "numeric_margin_retained" in source
    assert "numeric_behavioral_endpoint_retained" in source
    assert "numeric_displacement_metric_retained" in source
    assert "correct_class_logit_margin" not in source
    assert "D_BEH" not in source


def test_repository_auth_allows_detached_head_contract() -> None:
    source = inspect.getsource(runner.authenticate_repo)
    assert 'branch in ("", EXPECTED_BRANCH)' in source


def test_raw_information_boundary_does_not_read_historical_results() -> None:
    source = inspect.getsource(runner)
    forbidden = (
        "readout_alignment_analysis_v1",
        "readout_behavior_pair_merge_v1",
        "intervention_manifold_deviation",
        "behavioral_bridge_analysis",
        "study_a_adjacent",
    )
    for token in forbidden:
        assert token not in source
