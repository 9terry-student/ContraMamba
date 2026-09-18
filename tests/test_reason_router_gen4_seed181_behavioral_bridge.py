from __future__ import annotations

import inspect
import math
from pathlib import Path

import torch

from scripts import build_reason_router_gen4_seed181_behavioral_bridge_holdout as holdout
from scripts import reason_router_gen4_seed181_behavioral_restoration_bridge_fast_cuda as runner


def synthetic_planes() -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, index in (
        ("pp3_plus", 0),
        ("pp3_minus", 1),
        ("pp5_plus", 2),
        ("pp5_minus", 3),
    ):
        value = torch.zeros(runner.restoration.DIM, dtype=torch.float64)
        value[index] = 1.0
        out[name] = value
    return out


def test_design_identity_and_budget() -> None:
    assert runner.DESIGN_COMMIT == "9ea1617f4485fa0b6093df0c70aa88a42260c710"
    assert runner.SEED == 181
    assert runner.ARM == "G3-GROUP-D-HALF"
    assert runner.TARGET_CELLS == ("C0_SHAM", "C2_NAME")
    assert runner.CONDITIONS == (
        "native",
        "pp3_neutralized",
        "pp3_restored",
        "pp5_replacement",
    )
    assert runner.SHARD_RANGES == {0: (2701, 2850), 1: (2851, 3000)}
    assert runner.FORWARDS_PER_SHARD == 1200
    assert runner.TOTAL_FORWARD_BUDGET == 2400
    assert runner.MIN_PHYSICAL_GPU_COUNT == 2


def test_holdout_global_continuation_ids_and_label_free_rows() -> None:
    facts, rows = holdout.build_population()
    assert len(facts) == 300
    assert len(rows) == 1800
    assert facts[0]["pair_id"] == "xg1_fact_2701"
    assert facts[-1]["pair_id"] == "xg1_fact_3000"
    assert not any(holdout.base.FORBIDDEN_FIELDS & set(row) for row in rows)
    holdout.validate_semantic_labels(facts, rows)


def test_holdout_regenerates_latest_frozen_continuation_exactly() -> None:
    facts, rows = holdout.build_population(2401, 2700)
    latest = Path(holdout.PRIOR_COHORTS[-1][0])
    assert holdout.jsonl_bytes(facts) == holdout.git_blob_bytes(
        latest / holdout.SOURCE_FILE
    )
    assert holdout.jsonl_bytes(rows) == holdout.git_blob_bytes(
        latest / holdout.ROW_FILE
    )


def test_prior_inventory_is_git_blob_bound_and_covers_001_2700() -> None:
    pair_ids, claims, evidence = holdout.prior_inventory()

    assert len(pair_ids) == 2700
    assert "xg1_fact_001" in pair_ids
    assert "xg1_fact_2700" in pair_ids
    assert claims
    assert evidence


def test_label_contract_is_frozen() -> None:
    assert runner.LABEL_ID_BY_CELL == {"C0_SHAM": 2, "C2_NAME": 1}
    assert runner.LABEL_NAME_BY_CELL == {
        "C0_SHAM": "SUPPORT",
        "C2_NAME": "NOT_ENTITLED",
    }


def test_correct_margin() -> None:
    logits = [1.0, 2.0, 5.0]
    assert runner.correct_margin(logits, 2) == 3.0
    assert runner.correct_margin(logits, 1) == -3.0


def test_behavior_hook_has_no_finite_epsilon_probe() -> None:
    p = synthetic_planes()
    runtime = runner.restoration.holdout.phase1.base.prevalence_eq
    width = runtime.core.INTERMEDIATE_SIZE
    before = torch.zeros((1, 4, 2 * width), dtype=torch.float32)
    before[0, 2, 0] = 3.0
    before[0, 2, 1] = -4.0
    mask = torch.zeros(width, dtype=torch.bool)
    mask[: runner.restoration.DIM] = True

    audit: dict = {}
    after = runner.behavior_hook(
        before,
        token_index=2,
        strong_mask=mask,
        condition="pp5_replacement",
        planes=p,
        audit=audit,
    )
    realized = (after[0, 2, :width][mask] - before[0, 2, :width][mask]).to(torch.float64)
    expected = torch.zeros(runner.restoration.DIM, dtype=torch.float64)
    expected[0] = -3.0
    expected[1] = 4.0
    expected[2] = 3.0
    expected[3] = -4.0
    assert torch.equal(realized, expected.to(torch.float32).to(torch.float64))
    assert audit["probe_correction_l2"] == 0.0
    assert audit["coefficient_source"] == "native_seed181_P3_coordinates"


def test_restored_hook_is_exact_zero_correction() -> None:
    p = synthetic_planes()
    runtime = runner.restoration.holdout.phase1.base.prevalence_eq
    width = runtime.core.INTERMEDIATE_SIZE
    before = torch.randn((1, 3, 2 * width), dtype=torch.float32)
    mask = torch.zeros(width, dtype=torch.bool)
    mask[: runner.restoration.DIM] = True
    audit: dict = {}
    after = runner.behavior_hook(
        before,
        token_index=1,
        strong_mask=mask,
        condition="pp3_restored",
        planes=p,
        audit=audit,
    )
    assert torch.equal(after, before)
    assert audit["correction_l2"] == 0.0
    assert audit["probe_correction_l2"] == 0.0


def test_runner_contains_no_primary_inference() -> None:
    source = inspect.getsource(runner).lower()
    assert "scipy" not in source
    assert "ttest" not in source
    assert "one_sided" not in source
    assert '"primary_inference_executed": false' in source
    assert '"scientific_conclusion": none' in source


def test_runner_full_model_path_uses_historical_forward() -> None:
    source = inspect.getsource(runner.run_condition)
    assert "adapter.historical_forward" in source
    assert "model.mamba(" not in source


def test_shard_pair_partition_is_exact_and_disjoint() -> None:
    a = set(runner.shard_pairs(0))
    b = set(runner.shard_pairs(1))
    assert not (a & b)
    assert sorted(a | b) == [f"xg1_fact_{i:03d}" for i in range(2701, 3001)]
