from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

from scripts import reason_router_gen4_generator_family_prevalence_fast_cuda_full as full


def _items(n_large: int):
    rows = []
    for index in range(full.SOURCE_PAIR_COUNT):
        rows.append(
            {
                "source_pair_id": f"xg2_fact_{index + 1:03d}",
                "regime": (
                    full.REGIME_LARGE
                    if index < n_large
                    else full.REGIME_SMALL
                ),
            }
        )
    return rows


def test_frozen_execution_constants():
    assert full.R5_FREEZE_COMMIT == "71f4e5106e36d36003e49e7424ab98d3df0dbe90"
    assert full.R5_EXECUTION_HEAD == "e646fc900d12b42621a6a58ee31eff255bb52bfd"
    assert full.SOURCE_PAIR_COUNT == 300
    assert full.BASELINE_FORWARDS_PER_PAIR == 4
    assert full.FULL_BASELINE_FORWARD_BUDGET == 1200
    assert full.ALIGNMENT_SHIFT_THRESHOLD == 0.11228626366380845
    assert full.MIN_GROUP_SIZE == 30


def test_only_frozen_eligible_families_are_supported():
    assert set(full.R5_GATE_ARTIFACTS) == {"xg2", "xg4"}
    assert set(full.eq.FAMILY_CONFIG) == {"xg2", "xg4"}


def test_r5_gate_artifact_hashes_are_frozen():
    assert (
        full.R5_GATE_ARTIFACTS["xg2"]["sha256"]
        == "1395b2ca9c20d501e21058250a972a7af5648b57d900b79d2cc04f960f5032fa"
    )
    assert (
        full.R5_GATE_ARTIFACTS["xg4"]["sha256"]
        == "d3d18a90c44207654a39a01e6ed38bebc50b096e1b12c9bb55d70c0b4a10bf4d"
    )


def test_threshold_boundary_is_large():
    assert full.classify_regime(full.ALIGNMENT_SHIFT_THRESHOLD) == full.REGIME_LARGE
    assert (
        full.classify_regime(full.ALIGNMENT_SHIFT_THRESHOLD - 1e-12)
        == full.REGIME_SMALL
    )


def test_prevalence_summary_viable_at_30_270():
    summary = full.summarize_prevalence(_items(30))
    assert summary["n_LARGE"] == 30
    assert summary["n_SMALL"] == 270
    assert summary["p_LARGE"] == pytest.approx(0.1)
    assert summary["VIABLE"] is True
    assert summary["family_level_classification"] == "VIABLE"


def test_prevalence_summary_not_viable_at_29_271():
    summary = full.summarize_prevalence(_items(29))
    assert summary["n_LARGE"] == 29
    assert summary["n_SMALL"] == 271
    assert summary["VIABLE"] is False
    assert summary["family_level_classification"] == "NOT_VIABLE"


def test_prevalence_summary_requires_exact_300_unique_pairs():
    with pytest.raises(full.PrevalenceFullBaselineError, match="ITEM_COUNT"):
        full.summarize_prevalence(_items(30)[:-1])

    rows = _items(30)
    rows[-1]["source_pair_id"] = rows[0]["source_pair_id"]
    with pytest.raises(full.PrevalenceFullBaselineError, match="PAIR_ID_UNIQUENESS"):
        full.summarize_prevalence(rows)


def test_full_runner_has_no_intervention_or_response_execution_path():
    source = inspect.getsource(full.run_full_baseline)

    forbidden_source = (
        "_run_alignment_pair",
        "analyze_responses",
        ".backward(",
        "optimizer",
    )
    for token in forbidden_source:
        assert token not in source

    tree = ast.parse(textwrap.dedent(source))
    identifiers = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
    }
    identifiers.update(
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
    )

    assert "task_head" not in identifiers
    assert "logits" not in identifiers


def test_global_classification_remains_incomplete():
    assert full.GLOBAL_PRIMARY_CLASSIFICATION == "PREVALENCE_TRANSPORTABILITY_INCOMPLETE"
