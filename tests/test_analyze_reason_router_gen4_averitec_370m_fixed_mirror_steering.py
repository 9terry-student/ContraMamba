from __future__ import annotations

import inspect
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from scripts import (
    analyze_reason_router_gen4_averitec_370m_fixed_mirror_steering
    as analysis,
)


def test_protocol_freeze() -> None:
    analysis.validate_protocol()
    assert analysis.REQUIRED_ANCESTOR == (
        "f6209a5721a16519148880e8a7f0a7734d34c787"
    )
    assert analysis.RAW_EXECUTION_HEAD == (
        "25a626dd52592c42318c80e14aa13c18b74fded9"
    )
    assert analysis.N == 2799
    assert analysis.RAW_ROW_COUNT == 8397
    assert analysis.CONDITIONS == (
        "native",
        "p3_mirror_steer",
        "p5_matched_control",
    )
    assert analysis.PRIMARY_ALPHA == 0.05
    assert analysis.HARM_THRESHOLD == 0.05


def test_frozen_raw_identity_constants() -> None:
    assert analysis.RAW_ROWS_GIT_BLOB == (
        "52e07a7c93f96eb6d10f7b87e38a0aee7fb6ac2d"
    )
    assert analysis.RAW_SUMMARY_GIT_BLOB == (
        "b35b4ed1aa5b53e896abd45fb8f22cb273c6660f"
    )
    assert analysis.RAW_SUMS_GIT_BLOB == (
        "6c5d1825378f0207c0bd560e968b9c9360cb093d"
    )
    assert analysis.RAW_ROWS_SHA256 == (
        "912fb91bdfc5e778e4d2f99cb9a8f7a5bdb8a4a1389b39de1d47947eb9058d0c"
    )
    assert analysis.RAW_SUMMARY_SHA256 == (
        "ad9497956b8db7b9d3930e13437d7d6daf5c33e3b800999e0d3ee2787cca4f7e"
    )
    assert analysis.RAW_SUMS_SHA256 == (
        "372d243e5bb6acc1f0e4ad1309fb446f474e813acbdedb9466fbd56940614465"
    )


def test_frozen_raw_artifacts_validate_without_analysis() -> None:
    rows, summary = analysis.validate_raw_artifacts()
    assert len(rows) == 8397
    assert summary["fresh_cohort_count"] == 2799
    assert summary["primary_inference_executed"] is False
    assert summary["p_value_count_added"] == 0
    assert summary["scientific_conclusion"] is None


@pytest.mark.parametrize(
    ("c", "d", "expected"),
    [
        (1, 0, Fraction(1, 2)),
        (2, 0, Fraction(1, 4)),
        (3, 0, Fraction(1, 8)),
        (2, 1, Fraction(1, 2)),
        (1, 2, Fraction(7, 8)),
        (5, 0, Fraction(1, 32)),
    ],
)
def test_exact_one_sided_binomial_upper(
    c: int,
    d: int,
    expected: Fraction,
) -> None:
    result = analysis.exact_one_sided_binomial_upper(c, d)
    observed = Fraction(
        int(result["exact_tail_numerator"]),
        int(result["exact_tail_denominator"]),
    )
    assert observed == expected
    assert result["p_value"] == float(expected)
    assert result["discordant_count"] == c + d


def make_row(
    *,
    example_id: str,
    condition: str,
    gold_id: int,
    prediction_id: int,
    margin: float,
) -> dict[str, Any]:
    labels = ("REFUTE", "NOT_ENTITLED", "SUPPORT")
    # Only analysis-consumed fields are needed by analyze_rows.
    return {
        "example_id": example_id,
        "condition": condition,
        "correct_label_id": gold_id,
        "correct_label": labels[gold_id],
        "prediction_id": prediction_id,
        "prediction": labels[prediction_id],
        "is_correct": prediction_id == gold_id,
        "correct_class_logit_margin": margin,
    }


def synthetic_triplet(
    index: int,
    *,
    native_correct: bool,
    steer_correct: bool,
    control_correct: bool = False,
    gold_id: int = 0,
) -> list[dict[str, Any]]:
    wrong = 1 if gold_id != 1 else 2
    native_pred = gold_id if native_correct else wrong
    steer_pred = gold_id if steer_correct else wrong
    control_pred = gold_id if control_correct else wrong

    return [
        make_row(
            example_id=f"e{index}",
            condition="native",
            gold_id=gold_id,
            prediction_id=native_pred,
            margin=1.0 if native_correct else -1.0,
        ),
        make_row(
            example_id=f"e{index}",
            condition="p3_mirror_steer",
            gold_id=gold_id,
            prediction_id=steer_pred,
            margin=2.0 if steer_correct else -2.0,
        ),
        make_row(
            example_id=f"e{index}",
            condition="p5_matched_control",
            gold_id=gold_id,
            prediction_id=control_pred,
            margin=0.5 if control_correct else -1.5,
        ),
    ]


def patched_analysis_for_small_n(monkeypatch, groups):
    n = len(groups)
    monkeypatch.setattr(analysis, "N", n)
    monkeypatch.setattr(analysis, "RAW_ROW_COUNT", 3 * n)
    rows = [row for group in groups for row in group]
    return analysis.analyze_rows(rows, analysis_head="test-head")


def test_success_rule_supported(monkeypatch) -> None:
    # Six corrections, zero damages -> p=1/64 < .05.
    groups = [
        synthetic_triplet(i, native_correct=False, steer_correct=True)
        for i in range(6)
    ]
    groups.extend(
        synthetic_triplet(
            100 + i,
            native_correct=True,
            steer_correct=True,
            control_correct=True,
        )
        for i in range(20)
    )
    result = patched_analysis_for_small_n(monkeypatch, groups)

    assert result["primary_endpoint"]["C_corrections"] == 6
    assert result["primary_endpoint"]["D_damages"] == 0
    assert result["primary_endpoint"]["p_value_count_added"] == 1
    assert result["primary_endpoint"]["test"]["p_value"] == 1 / 64
    assert result["preservation_gate"]["damage_rate_on_native_correct"] == 0
    assert result["success_rule"]["all_four_pass"] is True
    assert result["result"] == analysis.SUPPORTED_RESULT


def test_success_rule_fails_harm_gate_even_if_primary_is_positive(monkeypatch) -> None:
    groups = [
        synthetic_triplet(i, native_correct=False, steer_correct=True)
        for i in range(10)
    ]
    # One damage among ten native-correct => 10% harm, above 5%.
    groups.extend(
        synthetic_triplet(
            100 + i,
            native_correct=True,
            steer_correct=(i != 0),
            control_correct=True,
        )
        for i in range(10)
    )
    result = patched_analysis_for_small_n(monkeypatch, groups)

    assert result["primary_endpoint"]["C_corrections"] == 10
    assert result["primary_endpoint"]["D_damages"] == 1
    assert result["primary_endpoint"]["test"]["p_value"] < 0.05
    assert result["primary_endpoint"]["C_greater_than_D_gate_pass"] is True
    assert result["preservation_gate"]["damage_rate_on_native_correct"] == 0.1
    assert result["preservation_gate"]["pass"] is False
    assert result["success_rule"]["all_four_pass"] is False
    assert result["result"] == analysis.NOT_ESTABLISHED_RESULT


def test_zero_discordance_adds_no_p_value(monkeypatch) -> None:
    groups = [
        synthetic_triplet(
            i,
            native_correct=True,
            steer_correct=True,
            control_correct=True,
        )
        for i in range(5)
    ]
    result = patched_analysis_for_small_n(monkeypatch, groups)

    assert result["primary_endpoint"]["discordant_count"] == 0
    assert result["primary_endpoint"]["estimable"] is False
    assert result["primary_endpoint"]["test"] is None
    assert result["primary_endpoint"]["p_value_count_added"] == 0
    assert result["result"] == analysis.NOT_ESTABLISHED_RESULT


def count_p_value_keys(value: Any) -> int:
    if isinstance(value, dict):
        return sum(
            (1 if key == "p_value" else 0) + count_p_value_keys(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return sum(count_p_value_keys(child) for child in value)
    return 0


def test_exactly_one_p_value_key_when_estimable(monkeypatch) -> None:
    groups = [
        synthetic_triplet(i, native_correct=False, steer_correct=True)
        for i in range(6)
    ]
    groups.extend(
        synthetic_triplet(
            100 + i,
            native_correct=True,
            steer_correct=True,
            control_correct=True,
        )
        for i in range(10)
    )
    result = patched_analysis_for_small_n(monkeypatch, groups)

    assert result["primary_endpoint"]["p_value_count_added"] == 1
    assert result["analysis_execution"]["additional_p_value_count"] == 0
    assert count_p_value_keys(result) == 1


def test_runner_contains_no_scipy_or_multiplicity_or_tuning() -> None:
    source = inspect.getsource(analysis).lower()
    forbidden = (
        "import scipy",
        "binomtest",
        "mcnemar(",
        "ttest",
        "multipletests",
        "holm",
        "lambda_grid",
        "magnitude_grid",
        "epsilon_grid",
        "steering_multiplier",
    )
    for token in forbidden:
        assert token not in source


def test_direct_cli_help() -> None:
    import subprocess
    import sys

    repo_root = Path(analysis.__file__).resolve().parents[1]
    script = (
        repo_root
        / "scripts"
        / "analyze_reason_router_gen4_averitec_370m_fixed_mirror_steering.py"
    )

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
    assert "--output-dir" in completed.stdout
