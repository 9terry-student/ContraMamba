from __future__ import annotations

import inspect
import sys
import types


# Minimal stubs keep candidate tests isolated from the full repository imports.
lowdisp_stub = types.ModuleType(
    "scripts.reason_router_gen4_mamba370m14b_low_displacement_behavioral_raw_fast_cuda"
)
study_b_stub = types.ModuleType(
    "scripts.reason_router_gen4_mamba370m14b_readout_alignment_raw_fast_cuda"
)
sys.modules[
    "scripts.reason_router_gen4_mamba370m14b_low_displacement_behavioral_raw_fast_cuda"
] = lowdisp_stub
sys.modules[
    "scripts.reason_router_gen4_mamba370m14b_readout_alignment_raw_fast_cuda"
] = study_b_stub

from scripts import (  # noqa: E402
    reason_router_gen4_mamba370m14b_low_displacement_native_readout_raw_fast_cuda
    as runner,
)


def test_protocol_constants_are_exact() -> None:
    assert runner.REQUIRED_BASE_COMMIT == (
        "1f9c42cd502bcf204c05baa2c6cbbc7e9274bd7c"
    )
    assert runner.PAIR_FIRST == 5401
    assert runner.PAIR_LAST == 5700
    assert runner.PAIR_COUNT == 300
    assert runner.PAIR_IDS == tuple(
        f"xg1_fact_{i}" for i in range(5401, 5701)
    )
    assert runner.TARGET_CELLS == ("C0_SHAM", "C2_NAME")
    assert runner.ROWS_PER_SCALE == 600
    assert runner.TOTAL_ROWS == 1200


def test_frozen_scale_local_planes_and_checkpoints() -> None:
    assert runner.EXPECTED_SELECTED == {
        "mamba370m": "P3",
        "mamba14b": "P5",
    }
    assert runner.EXPECTED_CONTROL == {
        "mamba370m": "P5",
        "mamba14b": "P4",
    }
    assert runner.EXPECTED_CHECKPOINT == {
        "mamba370m":
            "9d8e3db22af4636938679aac6a8a97dd45344937d434fab29eac2ddc41a52a72",
        "mamba14b":
            "915c9de38d9dc7ee9da26ba4328e74549864c6bd29723f3b7a4b4e0050efce0a",
    }


def test_frozen_lowdisp_input_hashes_are_pinned() -> None:
    assert runner.STRUCTURAL_ROW_SHA256 == (
        "1d0f21ba44b4a282f42edb7e56626bb7d522ce09e25e1611bf1424ead4da2b28"
    )
    assert runner.STRUCTURAL_SOURCE_SHA256 == (
        "eddd6a264130e6451c72aeab010758dac43de82d9521898dfd1489c86717d11a"
    )
    assert runner.TOKEN_GATE_CROSS_SCALE_SHA256 == (
        "6657df0844a4fc8f41488b6fe119322ffa99583c83c7fdb4de683e9f7bb20b00"
    )


def test_gradient_semantics_reuse_study_b_exact_row_measurement() -> None:
    source = inspect.getsource(runner.run_one_row)
    assert "study_b.run_native_gradient_row" in source
    assert "normalize_item" in source


def test_raw_has_exact_forward_backward_budget() -> None:
    source = inspect.getsource(runner.run_raw)
    assert '"scientific_full_model_forward_count": TOTAL_ROWS' in source
    assert '"local_backward_count": TOTAL_ROWS' in source
    assert '"parameter_gradient_count": 0' in source
    assert runner.TOTAL_ROWS == 1200


def test_raw_information_boundary_is_behavior_outcome_blind() -> None:
    source = inspect.getsource(runner).lower()
    forbidden = (
        "pair_level_contrasts.jsonl",
        "primary_analysis.json",
        "low_displacement_behavioral_analysis_v1",
        "low_displacement_behavioral_items.jsonl",
        "raw_behavioral_summary.json",
        "pearson",
        "spearman",
        "rmse",
        "mae",
        "calibration",
    )
    for token in forbidden:
        assert token not in source

    assert '"behavioral_response_accessed": false' in source
    assert '"predicted_behavior_computed": false' in source
    assert '"behavioral_merge_computed": false' in source


def test_raw_contains_no_inferential_statistics() -> None:
    source = inspect.getsource(runner).lower()
    assert "scipy" not in source
    assert "ttest" not in source
    assert '"p_value_count_executed": 0' in source
    assert '"inferential_test_performed": false' in source


def test_technical_gate_retains_no_numeric_scientific_values() -> None:
    source = inspect.getsource(runner.technical_gate_worker)
    assert '"numeric_margin_retained": False' in source
    assert '"numeric_gradient_retained": False' in source
    assert '"numeric_Delta_L_retained": False' in source
    assert '"behavioral_response_accessed": False' in source


def test_output_contract_is_four_files() -> None:
    assert runner.ITEM_FILE == "low_displacement_native_readout_items.jsonl"
    assert runner.SUMMARY_FILE == "raw_readout_summary.json"
    assert runner.MANIFEST_FILE == "artifact_manifest.json"
    assert runner.CHECKSUM_FILE == "SHA256SUMS.txt"


def test_repository_auth_allows_detached_kaggle_head() -> None:
    source = inspect.getsource(runner.authenticate_repo)
    assert 'branch in ("", EXPECTED_BRANCH)' in source


def test_cli_has_gate_and_raw_modes() -> None:
    source = inspect.getsource(runner.parse_args)
    assert 'choices=("gate", "raw")' in source
    assert "--expected-head" in source
    assert "--mamba370m-snapshot" in source
    assert "--mamba14b-snapshot" in source
