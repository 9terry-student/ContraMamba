from __future__ import annotations

import inspect
import math
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import (
    reason_router_gen4_pre_emission_forced_decisive_stage_a_static_analysis
    as subject,
)


def test_protocol_constants_are_exact():
    subject.validate_protocol_constants()

    assert subject.REQUIRED_ANCESTOR == (
        "880eab834c442054642773935a88fa60a31287c3"
    )
    assert subject.RAW_SUMS_GIT_BLOB == (
        "a7c41fb72d08bc8f5083f45d96bf39aca693bd64"
    )
    assert subject.RAW_SUMMARY_GIT_BLOB == (
        "725e5454a824adc18814d3a20931bb2ab65209f7"
    )
    assert subject.RAW_ROWS_GIT_BLOB == (
        "1691f8e9582f786760df19bee778f247d53b238b"
    )
    assert subject.RAW_SUMMARY_SHA256 == (
        "0dc283213dda376b9684ee34e093de3ad6e09a425ea4fa4884390fa8b14555c5"
    )
    assert subject.RAW_ROWS_SHA256 == (
        "5cc15d2626b350108c2c532ab8502d87aacb0d5253f82905ab83dc438aff5fcd"
    )
    assert subject.PARTITION_SHA256 == (
        "871fb5c1e2c62f247c284ceae409ef9ec19acc75f63ffc832829c2664db46311"
    )
    assert subject.FORCED_GRAMMAR_SHA256 == (
        "62c9c53871f68fcc0f57d38c96b75d2391ee6ddc473a4325a522f91c9249bd00"
    )

    assert subject.PRIMARY_PARTITION == "confirmatory"
    assert subject.GROUP_ORDER == ("unsupported", "supported")
    assert subject.OFFSETS == (-4, -3, -2, -1)
    assert subject.SIGNAL == "p3_component_l2"
    assert subject.TEST == "two_sided_welch_t"
    assert subject.MULTIPLICITY == "holm"
    assert subject.P_VALUE_COUNT == 4
    assert subject.ALPHA == 0.05


@pytest.mark.parametrize(
    "x,a,b,expected",
    [
        (0.5, 1.0, 1.0, 0.5),
        (0.25, 0.5, 0.5, 1.0 / 3.0),
        (0.75, 0.5, 0.5, 2.0 / 3.0),
    ],
)
def test_regularized_incomplete_beta_known_values(
    x,
    a,
    b,
    expected,
):
    actual = subject.regularized_incomplete_beta(x, a, b)
    assert math.isclose(
        actual,
        expected,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def test_student_t_two_sided_p_known_case():
    # Welch comparison:
    # x=[1,2,3,4,5], y=[2,4,6,8]
    # Reference values from an independent Student-t implementation.
    t = -1.3587324409735149
    df = 4.749414519906323
    expected_p = 0.23519411138940557

    actual = subject.student_t_two_sided_p(t, df)

    assert math.isclose(
        actual,
        expected_p,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def test_welch_t_test_known_case_and_orientation():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 4.0, 6.0, 8.0]

    result = subject.welch_t_test(x, y)

    assert result["n_unsupported"] == 5
    assert result["n_supported"] == 4
    assert math.isclose(
        result["mean_unsupported"],
        3.0,
        rel_tol=0.0,
        abs_tol=0.0,
    )
    assert math.isclose(
        result["mean_supported"],
        5.0,
        rel_tol=0.0,
        abs_tol=0.0,
    )
    assert math.isclose(
        result["mean_difference_unsupported_minus_supported"],
        -2.0,
        rel_tol=0.0,
        abs_tol=0.0,
    )
    assert math.isclose(
        result["t_statistic"],
        -1.3587324409735149,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
    assert math.isclose(
        result["degrees_of_freedom"],
        4.749414519906323,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
    assert math.isclose(
        result["raw_p_value"],
        0.23519411138940557,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
    assert result["direction"] == "unsupported_less"


def test_welch_fails_closed_on_zero_standard_error():
    with pytest.raises(
        subject.StaticAnalysisError,
        match="WELCH_ZERO_OR_NONFINITE_STANDARD_ERROR",
    ):
        subject.welch_t_test(
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        )


def test_holm_adjust_exact_step_down_and_original_order():
    raw = [0.04, 0.001, 0.03, 0.20]
    adjusted = subject.holm_adjust(raw)

    assert adjusted == pytest.approx([
        0.09,
        0.004,
        0.09,
        0.20,
    ])


def test_holm_enforces_exact_family_size():
    with pytest.raises(
        subject.StaticAnalysisError,
        match="HOLM_FAMILY_SIZE",
    ):
        subject.holm_adjust([0.01, 0.02, 0.03])


def _synthetic_rows():
    rows = []

    # 169 confirmatory unsupported + 62 confirmatory supported.
    for i in range(169):
        rows.append({
            "example_id": f"u_{i}",
            "partition": "confirmatory",
            "primary_stage_a_group": "unsupported",
            "observations": [
                {
                    "relative_offset": offset,
                    "p3_component_l2":
                        float(i + 1 + (offset + 4) * 0.25),
                }
                for offset in subject.OFFSETS
            ],
        })

    for i in range(62):
        rows.append({
            "example_id": f"s_{i}",
            "partition": "confirmatory",
            "primary_stage_a_group": "supported",
            "observations": [
                {
                    "relative_offset": offset,
                    "p3_component_l2":
                        float(i + 2 + (offset + 4) * 0.25),
                }
                for offset in subject.OFFSETS
            ],
        })

    # Calibration rows are structurally present but must not affect inference.
    for i in range(231):
        rows.append({
            "example_id": f"c_{i}",
            "partition": "calibration",
            "primary_stage_a_group":
                "unsupported" if i % 2 == 0 else "supported",
            "observations": [
                {
                    "relative_offset": offset,
                    "p3_component_l2": 1.0e12,
                }
                for offset in subject.OFFSETS
            ],
        })

    assert len(rows) == 462
    return rows


def test_collect_primary_values_uses_confirmatory_only():
    values = subject.collect_primary_values(
        _synthetic_rows()
    )

    for offset in subject.OFFSETS:
        assert len(values[offset]["unsupported"]) == 169
        assert len(values[offset]["supported"]) == 62
        assert max(values[offset]["unsupported"]) < 1.0e12
        assert max(values[offset]["supported"]) < 1.0e12


def test_analyze_adds_exactly_four_p_values_and_no_extra_family():
    result = subject.analyze(_synthetic_rows())

    assert len(result["tests"]) == 4
    assert result["primary_family"]["p_value_count"] == 4
    assert result["p_value_count_added"] == 4
    assert [
        row["relative_offset"]
        for row in result["tests"]
    ] == [-4, -3, -2, -1]

    for row in result["tests"]:
        assert "raw_p_value" in row
        assert "holm_adjusted_p_value" in row
        assert 0.0 <= row["raw_p_value"] <= 1.0
        assert 0.0 <= row["holm_adjusted_p_value"] <= 1.0

    assert result["training_executed"] is False
    assert result["backward_executed"] is False
    assert result["selection_reopened"] is False
    assert result["layer_scan_executed"] is False
    assert result["rescue_performed"] is False
    assert result["stage_b_executed"] is False
    assert result["stage_c_executed"] is False


def test_render_report_contains_only_bounded_claim_scope():
    result = subject.analyze(_synthetic_rows())
    report = subject.render_report(result)

    assert "forced-decisive two-class finite grammar only" in report
    assert "spontaneous hallucination prediction" in report
    assert "Stage B" in report
    assert "Stage C" in report


def test_production_source_has_no_scipy_or_exploratory_inference():
    source = inspect.getsource(subject).lower()

    assert "import scipy" not in source
    assert "mannwhitney" not in source
    assert "bootstrap" not in source
    assert "permutation" not in source
    assert "subgroup" not in source
    assert "p5_component_l2" not in source
    assert "m47_support_minus_refute" not in source

    assert "p_value_count_added" in source
    assert "holm_adjust" in source
    assert "welch_t_test" in source


def test_direct_script_entrypoint_help():
    repo_root = Path(subject.__file__).resolve().parents[1]
    script = (
        repo_root
        / "scripts"
        / (
            "reason_router_gen4_pre_emission_forced_decisive_"
            "stage_a_static_analysis.py"
        )
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
    assert "--output-dir" in completed.stdout
    assert "--expected-head" in completed.stdout
