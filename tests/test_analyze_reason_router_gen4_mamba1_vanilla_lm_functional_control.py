from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "analyze_reason_router_gen4_mamba1_vanilla_lm_functional_control.py"

spec = importlib.util.spec_from_file_location("vanilla_control_analysis", PATH)
assert spec is not None and spec.loader is not None
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


def test_static_contract_has_no_inference() -> None:
    assert subject.RAW_FREEZE_HEAD == "f78c56418902dc208adaef6c9a188378f9a7fa45"
    assert subject.PLAN_FREEZE_COMMIT == "1fe9a198a15c9cea0e5451d918cd949bc21bf7e0"
    assert subject.SCALE_ORDER == ("370M", "790M", "1.4B", "2.8B")
    assert subject.N == 300


def test_descriptive_and_rank_helpers() -> None:
    d = subject.descriptive([1.0, 2.0, 3.0, 4.0])
    assert d["n"] == 4
    assert d["mean"] == 2.5
    assert d["median"] == 2.5
    assert d["fraction_positive"] == 1.0

    assert subject.mean_sign(1.0) == "+"
    assert subject.mean_sign(-1.0) == "-"
    assert subject.mean_sign(0.0) == "0"

    assert np.isclose(subject.pearson([1, 2, 3], [2, 4, 6]), 1.0)
    assert np.isclose(subject.spearman([3, 1, 2], [30, 10, 20]), 1.0)


def test_full_frozen_analysis_pattern() -> None:
    analysis, pairs = subject.analyze_all()

    assert len(pairs) == 1200
    assert analysis["inferential_test_performed"] is False
    assert analysis["p_value_count"] == 0
    assert analysis["row_filter_performed"] is False
    assert analysis["rescue_performed"] is False
    assert analysis["selection_reopened"] is False

    pattern = analysis["pattern_diagnostics"]
    assert pattern["TASK_MATCHED_SIGN_VECTOR_LM"] == ["-", "-", "+", "+"]
    assert pattern["FROZEN_CONTRAMAMBA_SIGN_VECTOR"] == ["+", "+", "-", "-"]
    assert pattern["FULL_TASK_MATCHED_SIGN_CONCORDANCE"] is False
    assert pattern["POSITIVE_SCALE_SIGN_PRESERVATION"] is False
    assert pattern["NEGATIVE_SCALE_SIGN_PRESERVATION"] is False
    assert pattern["aggregate_sign_opposite_at_all_four_sampled_scales"] is True

    expected_means = {
        "370M": -0.001975319110073159,
        "790M": -0.015964364288833865,
        "1.4B": 0.00783318280382364,
        "2.8B": 0.01644840225237525,
    }
    for scale, expected in expected_means.items():
        observed = analysis["scales"][scale]["task_matched"]["pair_endpoint"]["mean"]
        assert np.isclose(observed, expected, rtol=0.0, atol=1e-15)

    expected_common = {
        "370M": -0.001975319110073159,
        "790M": -0.004439468895178745,
        "1.4B": -0.00990855313213648,
        "2.8B": 0.01644840225237525,
    }
    for scale, expected in expected_common.items():
        observed = analysis["scales"][scale]["common_p3_p5"]["pair_endpoint"]["mean"]
        assert np.isclose(observed, expected, rtol=0.0, atol=1e-15)
