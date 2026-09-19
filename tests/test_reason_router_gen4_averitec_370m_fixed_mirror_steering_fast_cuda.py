from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from scripts import (
    reason_router_gen4_averitec_370m_fixed_mirror_steering_fast_cuda
    as runner,
)


def synthetic_planes():
    e0 = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    e1 = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
    e2 = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float64)
    e3 = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
    return {
        "P3": {"plus": e0, "minus": e1},
        "P5": {"plus": e2, "minus": e3},
    }


def test_protocol_freeze() -> None:
    runner.validate_protocol()
    assert runner.REQUIRED_ANCESTOR == (
        "d4296763ac1110b9b8cd2f6c05c92b6bc56e6479"
    )
    assert runner.N == 2799
    assert runner.CONDITIONS == (
        "native",
        "p3_mirror_steer",
        "p5_matched_control",
    )
    assert runner.SELECTED_PLANE == "P3"
    assert runner.CONTROL_PLANE == "P5"
    assert runner.INTERVENTION_LAYER == 35
    assert runner.ANCHOR_NAME == "A_CLAIM_EVIDENCE_BOUNDARY"
    assert runner.TARGET_OFFSET == 2
    assert runner.EXPECTED_FORWARD_BUDGET == 8397
    assert runner.SHARDS == (
        {
            "shard_id": 0,
            "physical_device": 0,
            "start": 0,
            "end": 1400,
            "example_count": 1400,
            "forward_budget": 4200,
        },
        {
            "shard_id": 1,
            "physical_device": 1,
            "start": 1400,
            "end": 2799,
            "example_count": 1399,
            "forward_budget": 4197,
        },
    )


def test_frozen_provenance_constants() -> None:
    assert runner.DESIGN_ARTIFACT_GIT_BLOB == (
        "0d35f9aebb37e6203bea40304d3a55ab2361b684"
    )
    assert runner.SOURCE_ELIGIBILITY_CORRECTION_GIT_BLOB == (
        "3a55da349d7c9780f8e51de76e3fd9abd67531aa"
    )
    assert runner.COHORT_GIT_BLOB == (
        "f6c8edcfa37c10feb7dfa66ecd36648bd1cc2bf1"
    )
    assert runner.MANIFEST_GIT_BLOB == (
        "c31b20880eac0db2c3330a977fa7a5f66a788ddd"
    )
    assert runner.COHORT_SUMS_GIT_BLOB == (
        "6a480e21178de09a7e24415413626a6576479693"
    )
    assert runner.COHORT_SHA256 == (
        "ce271c1b57b33e12399bc180f06d417ceef6e4e92ad9e5383e2d8187829a2810"
    )
    assert runner.MANIFEST_SHA256 == (
        "7fef7f8c9544f46fbdeb1184bee313259bd52154181c346c4ae6d367d08da738"
    )
    assert runner.BRIDGE_SCRIPT_GIT_BLOB == (
        "07a6ccd97e7f616de4400053fbd442408b158398"
    )
    assert runner.CONFIRMATION_SCRIPT_GIT_BLOB == (
        "ba9aaa6de002759f1f8b5f42c6cfb0cedbc67984"
    )
    assert runner.GEOMETRY_SCRIPT_GIT_BLOB == (
        "12cfc608ddca1f16892ebaed1e1f05f09b4c452c"
    )


def test_frozen_cohort_validates() -> None:
    rows, manifest = runner.validate_cohort()
    assert len(rows) == 2799
    assert manifest["result"] == "PASS_2799_OF_2799"
    assert manifest["fresh_cohort_label_counts"] == {
        "Not Enough Evidence": 267,
        "Refuted": 1727,
        "Supported": 805,
    }

    indices = [row["averitec_train_index"] for row in rows]
    assert 438 in indices
    assert 1948 not in indices

    assert all(row["token_gate_pass"] is True for row in rows)
    assert all(row["token_gate_reasons"] == [] for row in rows)
    assert all(
        row["target_intervention_token_index"]
        == row["absolute_anchor_token_index"] + 2
        for row in rows
    )


def test_model_rows_do_not_change_text() -> None:
    cohort, _ = runner.validate_cohort()
    rows = runner.model_rows(cohort)

    assert len(rows) == 2799
    for source, row in zip(cohort, rows, strict=True):
        assert row["claim"] == source["claim"]
        assert row["evidence"] == source["evidence"]
        assert row["row_id"] == source["example_id"]
        assert row["source_pair_id"] == source["example_id"]
        assert row["contrast_cell_id"] == "AVERITEC_STEERING"


def test_mirror_algebra_is_exact_symmetric_extension_of_historical_control():
    planes = synthetic_planes()
    h = torch.tensor([2.0, -1.0, 0.5, 0.25], dtype=torch.float64)

    steer = runner.mirror_condition_correction(
        h,
        condition="p3_mirror_steer",
        planes=planes,
        selected_plane="P3",
        control_plane="P5",
        dim=4,
        tol=1e-12,
    )
    control = runner.mirror_condition_correction(
        h,
        condition="p5_matched_control",
        planes=planes,
        selected_plane="P3",
        control_plane="P5",
        dim=4,
        tol=1e-12,
    )
    historical = runner.bridge.condition_correction(
        h,
        condition="dominant_control",
        planes=planes,
        selected_plane="P3",
        control_plane="P5",
        dim=4,
        tol=1e-12,
    )

    expected_delta = torch.tensor(
        [2.0, -1.0, -2.0, 1.0],
        dtype=torch.float64,
    )

    assert torch.equal(steer["correction"], expected_delta)
    assert torch.equal(control["correction"], -expected_delta)
    assert torch.equal(
        control["correction"],
        historical["correction"],
    )
    assert torch.equal(
        steer["correction"] + control["correction"],
        torch.zeros(4, dtype=torch.float64),
    )

    assert steer["mirror_sign"] == 1
    assert control["mirror_sign"] == -1
    assert steer["delta_l2"] == control["delta_l2"]
    assert steer["correction_l2"] == control["correction_l2"]
    assert control["selected_post_max_abs_projection"] <= 1e-12


def test_mirror_hook_changes_only_target_strong_hidden_channels():
    planes = synthetic_planes()
    strong_mask = torch.tensor(
        [True, True, True, True, False, False],
        dtype=torch.bool,
    )
    intermediate_size = 6
    output = torch.zeros(
        (1, 4, 2 * intermediate_size),
        dtype=torch.float64,
    )
    output[0, 2, :4] = torch.tensor(
        [2.0, -1.0, 0.5, 0.25],
        dtype=torch.float64,
    )

    steer_audit = {}
    control_audit = {}

    steer = runner.steering_hook(
        output,
        token_index=2,
        strong_mask=strong_mask,
        condition="p3_mirror_steer",
        planes=planes,
        selected_plane="P3",
        control_plane="P5",
        dim=4,
        intermediate_size=intermediate_size,
        tol=1e-12,
        cast_tol=1e-12,
        audit=steer_audit,
    )
    control = runner.steering_hook(
        output,
        token_index=2,
        strong_mask=strong_mask,
        condition="p5_matched_control",
        planes=planes,
        selected_plane="P3",
        control_plane="P5",
        dim=4,
        intermediate_size=intermediate_size,
        tol=1e-12,
        cast_tol=1e-12,
        audit=control_audit,
    )

    steer_delta = steer - output
    control_delta = control - output

    assert torch.equal(steer_delta, -control_delta)
    assert torch.equal(steer_delta[:, :2, :], torch.zeros_like(steer_delta[:, :2, :]))
    assert torch.equal(steer_delta[:, 3:, :], torch.zeros_like(steer_delta[:, 3:, :]))
    assert torch.equal(
        steer_delta[:, :, intermediate_size:],
        torch.zeros_like(steer_delta[:, :, intermediate_size:]),
    )
    assert torch.equal(
        steer_delta[0, 2, 4:intermediate_size],
        torch.zeros(intermediate_size - 4, dtype=torch.float64),
    )
    assert steer_audit["mirror_sign"] == 1
    assert control_audit["mirror_sign"] == -1
    assert steer_audit["free_steering_coefficient_present"] is False
    assert control_audit["free_steering_coefficient_present"] is False
    assert steer_audit["response_dependent_strength_present"] is False


def test_correct_margin() -> None:
    assert runner.correct_margin([3.0, 1.0, 0.0], 0) == 2.0
    assert runner.correct_margin([3.0, 1.0, 5.0], 1) == -4.0


def test_summary_contains_no_response_derived_analysis() -> None:
    fake_identity = {
        "compact_checkpoint_sha256": "a",
        "source_full_checkpoint_sha256": "b",
        "mamba_state_canonical_sha256": "c",
        "downstream_state_canonical_sha256": "d",
        "full_state_canonical_sha256": "e",
    }
    shard_meta = [
        {
            "shard": dict(runner.SHARDS[0]),
            "model_provenance": dict(fake_identity),
            "tokenizer": {"fake": True},
            "raw_row_count": 4200,
        },
        {
            "shard": dict(runner.SHARDS[1]),
            "model_provenance": dict(fake_identity),
            "tokenizer": {"fake": True},
            "raw_row_count": 4197,
        },
    ]

    summary = runner.build_summary(
        execution_head="deadbeef",
        shard_meta=shard_meta,
    )

    assert summary["result"] == runner.RESULT_PASS
    assert summary["fresh_cohort_count"] == 2799
    assert summary["full_model_forward_budget"] == 8397
    assert summary["raw_row_count"] == 8397
    assert summary["primary_inference_executed"] is False
    assert summary["p_value_count_added"] == 0
    assert summary["scientific_conclusion"] is None
    assert summary["response_based_selection_performed"] is False
    assert summary["rescue_performed"] is False
    assert summary["magnitude_search_performed"] is False
    assert "accuracy" not in summary
    assert "correction_count" not in summary
    assert "damage_count" not in summary


def test_runner_contains_no_statistical_inference_or_tuning_path() -> None:
    source = inspect.getsource(runner).lower()

    forbidden = (
        "import scipy",
        "binomtest",
        "mcnemar",
        "ttest",
        "multipletests",
        "steering_multiplier",
        "lambda_grid",
        "magnitude_grid",
        "epsilon_grid",
        "response_trigger",
    )
    for token in forbidden:
        assert token not in source

    assert "primary_inference_executed = false" in source
    assert "p_value_count_added = 0" in source


def test_direct_cli_help() -> None:
    repo_root = Path(runner.__file__).resolve().parents[1]
    script = (
        repo_root
        / "scripts"
        / "reason_router_gen4_averitec_370m_fixed_mirror_steering_fast_cuda.py"
    )

    import subprocess
    import sys

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--expected-head" in completed.stdout
    assert "--mamba370m-snapshot" in completed.stdout
    assert "--output-dir" in completed.stdout
