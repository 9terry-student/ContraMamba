from __future__ import annotations

import math
from pathlib import Path

import pytest

from scripts import (
    reason_router_gen4_precursor_v4_stage_a_primary_analysis
    as analysis,
)


def test_protocol_is_single_frozen_primary_test() -> None:
    analysis.validate_protocol()
    assert analysis.N == 800
    assert analysis.GROUP_ORDER == ("unsupported", "supported")
    assert analysis.MIN_GROUP_N == 30
    assert analysis.ALPHA == 0.05
    assert analysis.PRIMARY_ENDPOINT == "z_i"
    assert analysis.P_VALUE_COUNT == 1


def test_frozen_raw_inputs_are_exact_and_uninferred() -> None:
    summary, rows = analysis.validate_raw_inputs()
    assert len(rows) == 800
    assert summary["result"] == "PASS_PRECURSOR_V4_ANALYTIC_STAGE_A_RAW"
    assert summary["supported_unsupported_group_counts"] == {
        "supported": 393,
        "unsupported": 407,
    }
    assert summary["scientific_inference_executed"] is False
    assert summary["primary_test_executed"] is False
    assert summary["p_value_count_added"] == 0
    assert summary["scientific_conclusion"] is None


def test_regularized_beta_reference_values() -> None:
    assert analysis.regularized_incomplete_beta(
        2.0, 3.0, 0.4
    ) == pytest.approx(0.5248, abs=1e-14)
    assert analysis.regularized_incomplete_beta(
        0.5, 0.5, 0.5
    ) == pytest.approx(0.5, abs=1e-14)


def test_welch_reference_matches_independent_reference() -> None:
    first = [1.0, 2.0, 3.0, 4.0]
    second = [1.0, 1.5, 2.0, 2.5, 3.0]
    out = analysis.welch_test(first, second)

    assert out["t_statistic"] == pytest.approx(
        0.6793662204867574,
        rel=0.0,
        abs=1e-13,
    )
    assert out["degrees_of_freedom"] == pytest.approx(
        4.749414519906323,
        rel=0.0,
        abs=1e-13,
    )
    assert out["p_value_two_sided"] == pytest.approx(
        0.528591698528773,
        rel=0.0,
        abs=1e-13,
    )


def test_student_t_two_sided_symmetry_and_range() -> None:
    p1 = analysis.student_t_two_sided_p_value(2.25, 17.5)
    p2 = analysis.student_t_two_sided_p_value(-2.25, 17.5)
    assert p1 == pytest.approx(p2, abs=1e-15)
    assert 0.0 < p1 < 1.0
    assert analysis.student_t_two_sided_p_value(
        0.0, 10.0
    ) == pytest.approx(1.0, abs=1e-15)


def test_not_estimable_boundary_executes_no_test() -> None:
    rows = []
    for i in range(29):
        rows.append({
            "example_id": f"u{i}",
            "primary_stage_a_group": "unsupported",
            "z_i": float(i),
            "primary_inference_executed": False,
            "p_value_count_added": 0,
        })
    for i in range(40):
        rows.append({
            "example_id": f"s{i}",
            "primary_stage_a_group": "supported",
            "z_i": float(i),
            "primary_inference_executed": False,
            "p_value_count_added": 0,
        })

    out = analysis.analyze(rows)
    assert out["estimable"] is False
    assert out["primary_test_executed"] is False
    assert out["p_value_count_added"] == 0
    assert out["scientific_conclusion"] == (
        "PRECURSOR_V4_STAGE_A_NOT_ESTIMABLE"
    )


def test_source_contains_no_selection_or_extra_p_values() -> None:
    source = Path(analysis.__file__).read_text(encoding="utf-8")
    assert 'P_VALUE_COUNT = 1' in source
    assert 'PRIMARY_ENDPOINT = "z_i"' in source
    assert 'GROUP_ORDER = ("unsupported", "supported")' in source
    assert "holm" not in source.lower()
    assert "bonferroni" not in source.lower()
    assert "offset_scan" not in source.lower()
    assert "subgroup_scan" not in source.lower()
