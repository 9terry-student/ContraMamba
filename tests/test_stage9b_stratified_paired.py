from __future__ import annotations

import math

import pytest

from scripts.analyze_stage9b_stratified_paired import (
    STRATIFIED_METRICS,
    mcnemar_exact,
    paired_bootstrap,
    pairwise_summaries,
    stratified_metrics,
)


def _auditor(example_id: str, gate: float) -> dict:
    return {
        "id": example_id,
        "frame_prob": gate,
        "predicate_coverage_prob": gate,
        "sufficiency_prob": gate,
        "entitlement_prob": gate,
        "polarity_margin": 1.0,
    }


def test_stratified_grouping_downgrades_and_zero_denominators() -> None:
    records = [
        {"id": "a", "pair_id": "p1", "intervention_type": "none", "gold_final_label": "SUPPORT"},
        {"id": "b", "pair_id": "p2", "intervention_type": "none", "gold_final_label": "SUPPORT"},
        {"id": "c", "pair_id": "p1", "intervention_type": "predicate_swap", "gold_final_label": "NOT_ENTITLED"},
        {"id": "d", "pair_id": "p2", "intervention_type": "predicate_swap", "gold_final_label": "NOT_ENTITLED"},
    ]
    classifier = {"a": "SUPPORT", "b": "SUPPORT", "c": "SUPPORT", "d": "NOT_ENTITLED"}
    routed = {"a": "SUPPORT", "b": "NOT_ENTITLED", "c": "NOT_ENTITLED", "d": "NOT_ENTITLED"}
    auditors = {
        "a": (_auditor("a", 0.9),), "b": (_auditor("b", 0.2),),
        "c": (_auditor("c", 0.2),), "d": (_auditor("d", 0.2),),
    }
    rows = stratified_metrics(records, classifier, routed, auditors, 0.5)
    by_type = {row["intervention_type"]: row for row in rows}

    assert set(by_type) == {"none", "predicate_swap"}
    assert by_type["none"]["n_examples"] == 2
    assert by_type["none"]["downgraded_count"] == 1
    assert by_type["predicate_swap"]["downgraded_count"] == 1
    # Predicate-swap stratum has no gold SUPPORT examples, so recall is a
    # documented numeric zero rather than NaN or a blank.
    assert by_type["predicate_swap"]["support_recall_pre_router"] == 0.0
    assert by_type["predicate_swap"]["support_recall_post_router"] == 0.0
    assert by_type["predicate_swap"]["support_precision_pre_router"] == 0.0
    assert by_type["predicate_swap"]["support_precision_post_router"] == 0.0
    for row in rows:
        assert set(STRATIFIED_METRICS) <= set(row)
        assert all(math.isfinite(float(row[metric])) for metric in STRATIFIED_METRICS)


def test_pairwise_intervention_success_is_computed_per_pair() -> None:
    records = [
        {"id": "n1", "pair_id": "p1", "intervention_type": "none"},
        {"id": "s1", "pair_id": "p1", "intervention_type": "predicate_swap"},
        {"id": "n2", "pair_id": "p2", "intervention_type": "none"},
        {"id": "s2", "pair_id": "p2", "intervention_type": "predicate_swap"},
    ]
    systems = {
        "raw_classifier_only": {"n1": "SUPPORT", "s1": "SUPPORT", "n2": "REFUTE", "s2": "NOT_ENTITLED"},
        "conservative_balanced_router": {"n1": "SUPPORT", "s1": "NOT_ENTITLED", "n2": "REFUTE", "s2": "NOT_ENTITLED"},
    }
    summary = pairwise_summaries(records, systems)["predicate_swap"]
    assert summary["raw_classifier_only"] == (1, 2, 0.5)
    assert summary["conservative_balanced_router"] == (2, 2, 1.0)


def test_mcnemar_contingency_counts() -> None:
    records = [
        {"id": "a", "gold_final_label": "SUPPORT"},
        {"id": "b", "gold_final_label": "SUPPORT"},
        {"id": "c", "gold_final_label": "REFUTE"},
        {"id": "d", "gold_final_label": "NOT_ENTITLED"},
    ]
    classifier = {"a": "SUPPORT", "b": "SUPPORT", "c": "SUPPORT", "d": "SUPPORT"}
    router = {"a": "SUPPORT", "b": "NOT_ENTITLED", "c": "REFUTE", "d": "REFUTE"}
    result = mcnemar_exact(records, classifier, router)
    assert result["both_correct"] == 1
    assert result["classifier_only_correct"] == 1
    assert result["router_only_correct"] == 1
    assert result["both_wrong"] == 1
    assert result["p_value"] == 1.0
    assert result["accuracy_delta"] == 0.0


def test_pair_id_bootstrap_returns_finite_intervals() -> None:
    records = [
        {"id": "a", "pair_id": "p1", "gold_final_label": "SUPPORT"},
        {"id": "b", "pair_id": "p1", "gold_final_label": "NOT_ENTITLED"},
        {"id": "c", "pair_id": "p2", "gold_final_label": "REFUTE"},
        {"id": "d", "pair_id": "p2", "gold_final_label": "NOT_ENTITLED"},
    ]
    classifier = {"a": "SUPPORT", "b": "SUPPORT", "c": "REFUTE", "d": "NOT_ENTITLED"}
    router = {"a": "SUPPORT", "b": "NOT_ENTITLED", "c": "REFUTE", "d": "NOT_ENTITLED"}
    auditors = {
        "a": (_auditor("a", 0.9),), "b": (_auditor("b", 0.2),),
        "c": (_auditor("c", 0.9),), "d": (_auditor("d", 0.2),),
    }
    rows = paired_bootstrap(
        records, classifier, router, auditors, 0.5, n_bootstrap=100, seed=3
    )
    assert {row["metric"] for row in rows} == {
        "accuracy_delta", "macro_f1_delta", "support_precision_gain",
        "support_recall_drop", "downgrade_rate",
        "pre_router_candidate_gate_fail_rate",
    }
    for row in rows:
        assert row["n_bootstrap"] == 100
        assert row["resampling_unit"] == "pair_id"
        assert math.isfinite(row["estimate"])
        assert math.isfinite(row["ci_low"])
        assert math.isfinite(row["ci_high"])
        assert row["ci_low"] <= row["ci_high"]
