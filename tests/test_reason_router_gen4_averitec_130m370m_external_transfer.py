from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from scripts import (
    reason_router_gen4_averitec_130m370m_external_transfer_fast_cuda
    as runner,
)


def test_protocol_freeze() -> None:
    runner.validate_protocol()
    assert runner.SCALE_ORDER == ("mamba130m", "mamba370m")
    assert runner.CONDITIONS == (
        "native",
        "dominant_neutralized",
        "dominant_control",
    )
    assert runner.N == 462
    assert runner.FORWARDS_PER_SCALE == 1386
    assert runner.TOTAL_FORWARD_BUDGET == 2772
    assert runner.GPU_BY_SCALE == {
        "mamba130m": 0,
        "mamba370m": 1,
    }


def test_frozen_cohort_validates() -> None:
    rows, manifest = runner.validate_cohort()
    assert len(rows) == 462
    assert manifest["result"] == "PASS_462_OF_462"
    assert all(row["token_gate_pass"] is True for row in rows)
    assert all(
        row["target_intervention_token_index"]
        == row["absolute_anchor_token_index"] + 2
        for row in rows
    )


def test_130m_release_asset_identity_is_exact() -> None:
    assert runner.MAMBA130_CHECKPOINT_RELEASE_TAG == (
        "gen4-r5-evaluator-checkpoints-cf08261"
    )
    assert runner.MAMBA130_CHECKPOINT_RELEASE_ID == 387669220
    assert runner.MAMBA130_CHECKPOINT_ASSET_ID == 559773417
    assert runner.MAMBA130_CHECKPOINT_ASSET_NAME == (
        "seed181__G3-GROUP-D-HALF__selected_checkpoint.pt"
    )
    assert runner.MAMBA130_CHECKPOINT_BYTES == 518270455
    assert runner.MAMBA130_CHECKPOINT_SHA256 == (
        "afc55ef0bf6a250dadc16dfa85ae2350505dd1289e781e109519c6bc8009422f"
    )


def test_130m_and_370m_causal_objects_are_frozen() -> None:
    assert runner.MAMBA130_SELECTED == "P3"
    assert runner.MAMBA130_CONTROL == "P5"
    assert runner.MAMBA130_INTERVENTION_LAYER == 17
    assert runner.MAMBA370_SELECTED == "P3"
    assert runner.MAMBA370_CONTROL == "P5"
    assert runner.MAMBA370_INTERVENTION_LAYER == 35


def test_130m_condition_mapping_preserves_historical_semantics() -> None:
    source = inspect.getsource(runner.run_condition_130)
    assert '"dominant_neutralized": "pp3_neutralized"' in source
    assert '"dominant_control": "pp5_replacement"' in source


def test_runner_contains_no_primary_inference() -> None:
    source = inspect.getsource(runner).lower()
    assert "scipy" not in source
    assert "ttest_1samp" not in source
    assert "confirmation_inference.json" not in source
    assert "primary_inference_executed = false" in source
    assert "p_value_count_added = 0" in source


def test_model_rows_do_not_change_text() -> None:
    cohort, _ = runner.validate_cohort()
    rows = runner.model_rows(cohort)
    assert len(rows) == 462
    for source, row in zip(cohort, rows, strict=True):
        assert row["claim"] == source["claim"]
        assert row["evidence"] == source["evidence"]
        assert row["row_id"] == source["example_id"]
        assert row["source_pair_id"] == source["example_id"]
        assert row["contrast_cell_id"] == "AVERITEC"


def test_correct_margin() -> None:
    assert runner.correct_margin([3.0, 1.0, 0.0], 0) == 2.0
    assert runner.correct_margin([3.0, 1.0, 5.0], 1) == -4.0
