from __future__ import annotations

import inspect

import torch

from scripts import (
    reason_router_gen4_factor2_small_alpha_behavioral_raw_fast_cuda
    as runner,
)


def test_protocol_is_exact() -> None:
    runner.validate_protocol()
    assert runner.PAIR_FIRST == 5701
    assert runner.PAIR_LAST == 6000
    assert runner.PAIR_COUNT == 300
    assert runner.TARGET_CELLS == ("C0_SHAM", "C2_NAME")
    assert runner.ALPHAS == (0.25, 0.125, 0.0625, 0.03125)
    assert runner.PRIMARY_CURVE == "K(alpha)"
    assert runner.CONDITIONS == (
        "restored_native",
        "control_alpha_0_25",
        "control_alpha_0_125",
        "control_alpha_0_0625",
        "control_alpha_0_03125",
    )
    assert runner.ROWS_PER_SCALE == 600
    assert runner.FORWARDS_PER_ROW == 5
    assert runner.FORWARDS_PER_SCALE == 3000
    assert runner.TOTAL_FORWARD_BUDGET == 6000
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


def test_scaled_full_correction_is_exact_for_frozen_positive_grid() -> None:
    base = torch.tensor(
        [1.0, -2.0, 0.125, -0.0625],
        dtype=torch.float32,
    )
    for alpha in runner.ALPHAS:
        actual = runner.scaled_full_correction(base, alpha)
        assert torch.equal(actual, base * alpha)


def test_no_alpha1_behavioral_condition_exists() -> None:
    assert 1.0 not in runner.ALPHAS
    assert all("1_0" not in x for x in runner.CONDITIONS)
    source = inspect.getsource(runner.run_five_forward_row)
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
    source = inspect.getsource(runner.run_five_forward_row)
    assert source.count("adapter.historical_forward(") == 2
    # One native call appears explicitly; four alpha calls share the loop body.
    assert "for alpha, condition in zip(" in source
    assert "torch.autograd.grad" not in source
    assert ".backward(" not in source


def test_required_raw_file_contracts() -> None:
    assert runner.BEHAVIOR_ITEM_FILE == "factor2_small_alpha_behavioral_items.jsonl"
    assert runner.BEHAVIOR_SUMMARY_FILE == "raw_behavioral_summary.json"
    assert runner.BEHAVIOR_MANIFEST_FILE == "artifact_manifest.json"
    assert runner.STATE_FILE == "factor2_small_alpha_states.npz"
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

def test_factor2_behavior_is_prospective_and_predictor_blind() -> None:
    source = inspect.getsource(runner)
    assert runner.REQUIRED_ANCESTOR == (
        "069b48dee6cb39909623defd862965b3f940e9d9"
    )
    assert "reason_router_gen4_factor2_small_alpha_runtime_inputs" in source
    forbidden = (
        "factor2_small_alpha_native_readout",
        "native_readout_raw_v1",
        "Delta_L_row",
        "pair_level_contrasts.jsonl",
        "primary_analysis.json",
    )
    for token in forbidden:
        assert token not in source


def test_frozen_small_alpha_grid_has_no_negative_or_adaptive_arm() -> None:
    assert runner.ALPHAS == (0.25, 0.125, 0.0625, 0.03125)
    assert all(alpha > 0.0 for alpha in runner.ALPHAS)
    assert 1.0 not in runner.ALPHAS
    source = inspect.getsource(runner).lower()
    assert '"negative_alpha_arm_executed": false' in source
    assert '"adaptive_rerun_executed": false' in source


def test_five_forward_budget_and_state_capture_contract() -> None:
    assert runner.FORWARDS_PER_ROW == 5
    assert runner.FORWARDS_PER_SCALE == 3000
    assert runner.TOTAL_FORWARD_BUDGET == 6000
    assert runner.TOTAL_STATE_ROWS == 1200
    source = inspect.getsource(runner.run_five_forward_row)
    assert source.count("adapter.historical_forward(") == 2
    assert "for alpha, condition in zip(" in source
    assert "torch.autograd.grad" not in source
    assert ".backward(" not in source
