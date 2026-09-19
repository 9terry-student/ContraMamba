from __future__ import annotations

import inspect
from pathlib import Path

import torch

from scripts import (
    build_reason_router_gen4_mamba370m14b_behavioral_bridge_holdout
    as holdout,
)
from scripts import (
    reason_router_gen4_mamba370m14b_behavioral_bridge_fast_cuda
    as runner,
)


def synthetic_planes(dim: int = 4):
    out = {}
    for plane, base in (("P3", 0), ("P5", 2)):
        plus = torch.zeros(dim, dtype=torch.float64)
        minus = torch.zeros(dim, dtype=torch.float64)
        plus[base] = 1.0
        minus[base + 1] = 1.0
        out[plane] = {"plus": plus, "minus": minus}
    return out


def test_holdout_range_and_outcome_blind_population() -> None:
    facts, rows = holdout.build_population()
    assert len(facts) == 300
    assert len(rows) == 1800
    assert facts[0]["pair_id"] == "xg1_fact_4801"
    assert facts[-1]["pair_id"] == "xg1_fact_5100"
    assert not any(
        holdout.FORBIDDEN_OUTCOME_FIELDS & set(row)
        for row in rows
    )
    holdout.validate_semantic_labels(facts, rows)


def test_latest_frozen_tail_regenerates_exactly() -> None:
    facts, rows = holdout.build_population(4501, 4800)
    source = holdout.prior.m370.prior.git_blob_bytes(
        holdout.FROZEN_TAIL_DIR / holdout.SOURCE_FILE
    )
    materialized = holdout.prior.m370.prior.git_blob_bytes(
        holdout.FROZEN_TAIL_DIR / holdout.ROW_FILE
    )
    assert holdout.jsonl_bytes(facts) == source
    assert holdout.jsonl_bytes(rows) == materialized


def test_runner_protocol_and_scale_identities() -> None:
    assert runner.PAIR_FIRST == 4801
    assert runner.PAIR_LAST == 5100
    assert runner.CONDITIONS == (
        "native",
        "dominant_neutralized",
        "dominant_restored",
        "dominant_control",
    )
    assert runner.FORWARDS_PER_PAIR == 8
    assert runner.FORWARDS_PER_SHARD == 1200
    assert runner.TOTAL_FORWARD_BUDGET == 2400

    m370 = runner.scale_spec("mamba370m")
    m14 = runner.scale_spec("mamba14b")
    assert m370["selected_plane"] == "P3"
    assert m370["control_plane"] == "P5"
    assert m14["selected_plane"] == "P5"
    assert m14["control_plane"] == "P4"


def test_label_contract_is_exact() -> None:
    assert runner.LABEL_ID_BY_CELL == {
        "C0_SHAM": 2,
        "C2_NAME": 1,
    }
    assert runner.LABEL_NAME_BY_CELL == {
        "C0_SHAM": "SUPPORT",
        "C2_NAME": "NOT_ENTITLED",
    }


def test_correct_margin() -> None:
    assert runner.correct_margin([1.0, 2.0, 5.0], 2) == 3.0
    assert runner.correct_margin([1.0, 2.0, 5.0], 1) == -3.0


def test_generic_hook_neutralized_restored_and_control() -> None:
    planes = synthetic_planes()
    width = 8
    dim = 4
    mask = torch.zeros(width, dtype=torch.bool)
    mask[:dim] = True

    before = torch.zeros((1, 3, 2 * width), dtype=torch.float32)
    before[0, 1, 0] = 3.0
    before[0, 1, 1] = -4.0

    neutral_audit = {}
    neutral = runner.behavior_hook(
        before,
        token_index=1,
        strong_mask=mask,
        condition="dominant_neutralized",
        planes=planes,
        selected_plane="P3",
        control_plane="P5",
        dim=dim,
        intermediate_size=width,
        tol=1e-12,
        cast_tol=1e-6,
        audit=neutral_audit,
    )
    assert neutral[0, 1, 0].item() == 0.0
    assert neutral[0, 1, 1].item() == 0.0

    restored_audit = {}
    restored = runner.behavior_hook(
        before,
        token_index=1,
        strong_mask=mask,
        condition="dominant_restored",
        planes=planes,
        selected_plane="P3",
        control_plane="P5",
        dim=dim,
        intermediate_size=width,
        tol=1e-12,
        cast_tol=1e-6,
        audit=restored_audit,
    )
    assert torch.equal(restored, before)
    assert restored_audit["correction_l2"] == 0.0

    control_audit = {}
    control = runner.behavior_hook(
        before,
        token_index=1,
        strong_mask=mask,
        condition="dominant_control",
        planes=planes,
        selected_plane="P3",
        control_plane="P5",
        dim=dim,
        intermediate_size=width,
        tol=1e-12,
        cast_tol=1e-6,
        audit=control_audit,
    )
    assert control[0, 1, 0].item() == 0.0
    assert control[0, 1, 1].item() == 0.0
    assert control[0, 1, 2].item() == 3.0
    assert control[0, 1, 3].item() == -4.0
    assert control_audit["probe_correction_l2"] == 0.0


def test_raw_runner_contains_no_inference_or_rescue() -> None:
    source = inspect.getsource(runner).lower()
    assert "scipy" not in source
    assert "ttest" not in source
    assert '"primary_inference_executed": false' in source
    assert '"confirmation_inference_accessed": false' in source
    assert '"rescue_performed": false' in source


def test_full_model_path_uses_historical_forward() -> None:
    source = inspect.getsource(runner.run_condition)
    assert "adapter.historical_forward" in source
    assert "model.mamba(" not in source


def test_shards_are_exact_and_disjoint() -> None:
    a = set(
        runner.PAIR_IDS[
            runner.SHARDS[0]["start_index"]:
            runner.SHARDS[0]["end_index"]
        ]
    )
    b = set(
        runner.PAIR_IDS[
            runner.SHARDS[1]["start_index"]:
            runner.SHARDS[1]["end_index"]
        ]
    )
    assert not (a & b)
    assert sorted(a | b) == list(runner.PAIR_IDS)


def test_combined_runner_is_available_for_single_pinned_execution() -> None:
    source = inspect.getsource(runner.run_both)
    assert 'scale="mamba370m"' in source
    assert 'scale="mamba14b"' in source
    assert "authenticate_checkout=False" in source
