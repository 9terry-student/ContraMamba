from __future__ import annotations

import math

import pytest

from scripts.analyze_stage9d_within_gate_correlation import (
    FIELDS,
    analyze_correlations,
    correlation_summary,
    render_markdown,
    residualize_by_intervention,
)


def _row(intervention: str, value: float) -> dict:
    return {
        "intervention_type": intervention,
        "frame_prob": value,
        "predicate_coverage_prob": value,
        "sufficiency_prob": value,
        "entitlement_prob": value,
        "polarity_margin": value,
    }


def test_constant_columns_are_safe_numeric_zero() -> None:
    summary = correlation_summary([_row("none", 0.5), _row("none", 0.5)])
    assert summary["pearson_mean_abs_off_diagonal"] == 0.0
    assert summary["spearman_mean_abs_off_diagonal"] == 0.0
    assert all(math.isfinite(float(value)) for value in summary.values())


def test_residualization_subtracts_intervention_means() -> None:
    rows = [
        _row("none", 0.2),
        _row("none", 0.6),
        _row("time_swap", 0.7),
        _row("time_swap", 0.9),
    ]
    residuals = residualize_by_intervention(rows)
    for intervention in ("none", "time_swap"):
        group = [row for row in residuals if row["intervention_type"] == intervention]
        for field in FIELDS:
            assert sum(float(row[field]) for row in group) / len(group) == pytest.approx(0.0)


def test_analysis_contains_all_required_scopes() -> None:
    rows = [_row("none", 0.2), _row("none", 0.4), _row("time_swap", 0.7), _row("time_swap", 0.9)]
    output = analyze_correlations(rows, seed=1)
    assert {row["scope"] for row in output} == {
        "global",
        "within_intervention",
        "between_intervention_means",
        "residualized",
    }


def test_markdown_warns_against_global_only_mechanism_claims() -> None:
    markdown = render_markdown([])
    assert "within-stratum analysis" in markdown
    assert "Do not claim four independent gates or one effective gate" in markdown
    assert "time_swap is temporal-specific or a broader same-type" in markdown
