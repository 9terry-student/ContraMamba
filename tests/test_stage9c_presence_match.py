from __future__ import annotations

import math

import pytest

from scripts.analyze_stage9c_presence_match import (
    classify_gate_state,
    correlation_rows,
    interpret_auc,
    opportunity_rows,
    raw_auc,
    render_markdown,
    signal_rows,
)


def _prediction(
    example_id: str,
    intervention: str,
    gold: str,
    predicted: str,
    gate: float,
    polarity: float = 1.0,
) -> dict:
    return {
        "id": example_id,
        "pair_id": "pair",
        "intervention_type": intervention,
        "gold_final_label": gold,
        "pred_final_label": predicted,
        "frame_prob": gate,
        "predicate_coverage_prob": gate,
        "sufficiency_prob": gate,
        "entitlement_prob": gate,
        "polarity_margin": polarity,
    }


def test_opportunity_count_arithmetic() -> None:
    predictions = [
        _prediction("a", "time_swap", "NOT_ENTITLED", "SUPPORT", 0.9),
        _prediction("b", "time_swap", "NOT_ENTITLED", "NOT_ENTITLED", 0.1),
        _prediction("c", "time_swap", "SUPPORT", "SUPPORT", 0.8),
    ]
    row = opportunity_rows(predictions, seed=1)[0]
    assert row["n"] == 3
    assert row["gold_NOT_ENTITLED"] == 2
    assert row["pred_SUPPORT"] == 2
    assert row["pred_NOT_ENTITLED"] == 1
    assert row["classifier_entitled"] == 2
    assert row["classifier_error"] == 1
    assert row["false_entitled"] == 1
    assert row["false_support"] == 1
    assert row["false_entitled_rate"] == 1 / 3


def test_gate_state_categories() -> None:
    assert classify_gate_state(0, 10, 0.9) == "no_opportunity"
    assert classify_gate_state(10, 0, 0.1) == "no_opportunity"
    assert classify_gate_state(10, 5, 0.1) == "correct_rejection"
    assert classify_gate_state(10, 5, 0.5) == "uncertain_no_signal"
    assert classify_gate_state(10, 5, 0.9) == "confidently_inverted"
    assert classify_gate_state(10, 5, 0.3) == "mixed"


def test_expected_direction_auc_does_not_hide_inversion() -> None:
    auc, defined = raw_auc([0.1, 0.9], [True, False])
    assert auc == 0.0
    assert defined == 1
    assert interpret_auc(auc, mean_pass_positive=0.9, auc_defined=defined) == "confidently_inverted"

    rows = signal_rows(
        [
            _prediction("bad", "time_swap", "NOT_ENTITLED", "SUPPORT", 0.9),
            _prediction("good", "time_swap", "NOT_ENTITLED", "NOT_ENTITLED", 0.1),
        ],
        seed=1,
    )
    frame = next(
        row for row in rows
        if row["target"] == "classifier_error" and row["score"] == "frame_fail_score"
    )
    assert frame["raw_auc"] == 0.0
    assert frame["expected_direction_auc"] == 0.0
    assert frame["inverted_auc"] == 1.0
    assert frame["direction_interpretation"] == "confidently_inverted"


def test_single_class_auc_is_numeric_and_uninformative() -> None:
    auc, defined = raw_auc([0.1, 0.2], [False, False])
    assert auc == 0.5
    assert defined == 0
    assert interpret_auc(auc, 0.0, defined) == "uninformative"


def test_gate_correlation_output_is_finite() -> None:
    predictions = [
        _prediction("a", "none", "SUPPORT", "SUPPORT", 0.1, -1.0),
        _prediction("b", "none", "SUPPORT", "SUPPORT", 0.5, 0.0),
        _prediction("c", "none", "SUPPORT", "SUPPORT", 0.9, 1.0),
    ]
    rows = correlation_rows(predictions, seed=1)
    global_row = next(row for row in rows if row["intervention_type"] == "__GLOBAL__")
    assert global_row["pearson_max_abs_off_diagonal"] == pytest.approx(1.0)
    assert global_row["spearman_max_abs_off_diagonal"] == pytest.approx(1.0)
    assert all(
        math.isfinite(float(value))
        for key, value in global_row.items()
        if key not in {"seed", "intervention_type"}
    )


def test_markdown_has_required_mechanism_sections() -> None:
    markdown = render_markdown([], [])
    assert "Main mechanism question: evidence presence or claim-evidence match?" in markdown
    assert "Evidence-presence diagnostic" in markdown
    assert "Confidently inverted diagnostic" in markdown
    assert "Polarity diagnostic" in markdown
    assert "Gate-correlation and independence diagnostic" in markdown
