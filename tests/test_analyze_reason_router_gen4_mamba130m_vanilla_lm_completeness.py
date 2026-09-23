from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PATH = (
    ROOT
    / "scripts"
    / "analyze_reason_router_gen4_mamba130m_vanilla_lm_completeness.py"
)

spec = importlib.util.spec_from_file_location(
    "mamba130m_vanilla_completeness",
    PATH,
)
assert spec is not None and spec.loader is not None
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


def test_authority_and_boundaries() -> None:
    assert (
        subject.RAW_FREEZE_HEAD
        == "9bbea9cb03327d4dca86cdb148eec44a4dd27b34"
    )
    assert (
        subject.EXTENSION_PLAN_FREEZE
        == "e567338f1dcd99d61ed7465ec39d441566f71fa3"
    )
    assert (
        subject.PRIOR_FOUR_SCALE_ANALYSIS_FREEZE
        == "88a6d6c469d44071a070b494485efe56db4faa58"
    )


def test_helpers() -> None:
    d = subject.descriptive([1.0, 2.0, 3.0, 4.0])
    assert d["n"] == 4
    assert d["mean"] == 2.5
    assert d["median"] == 2.5
    assert subject.sign(-1.0) == "-"
    assert subject.sign(1.0) == "+"
    assert np.isclose(
        subject.pearson([1, 2, 3], [2, 4, 6]),
        1.0,
    )
    assert np.isclose(
        subject.spearman([3, 1, 2], [30, 10, 20]),
        1.0,
    )


def test_frozen_extension_result() -> None:
    analysis, pairs = subject.analyze()

    assert len(pairs) == 300
    assert analysis["inferential_test_performed"] is False
    assert analysis["p_value_count"] == 0
    assert analysis["row_filter_performed"] is False
    assert analysis["rescue_performed"] is False
    assert analysis["selection_reopened"] is False

    x = analysis["mamba130m"]
    assert np.isclose(
        x["task_matched_pair"]["mean"],
        -0.056416846386448054,
        rtol=0.0,
        atol=1e-15,
    )
    assert np.isclose(
        x["by_cell"]["C0_SHAM"]["mean"],
        -0.037406762206373935,
        rtol=0.0,
        atol=1e-15,
    )
    assert np.isclose(
        x["by_cell"]["C2_NAME"]["mean"],
        -0.07542693056652212,
        rtol=0.0,
        atol=1e-15,
    )
    assert np.isclose(
        x["contra_comparison"]["contra_forward_equivalent"]["mean"],
        0.002240732073064253,
        rtol=0.0,
        atol=1e-15,
    )
    assert np.isclose(
        x["contra_comparison"]["pearson"],
        0.32079831963669675,
        rtol=0.0,
        atol=1e-12,
    )
    assert np.isclose(
        x["contra_comparison"]["spearman"],
        0.20438093756597295,
        rtol=0.0,
        atol=1e-12,
    )
    assert np.isclose(
        x["contra_comparison"]["pair_sign_agreement_fraction"],
        0.5566666666666666,
        rtol=0.0,
        atol=1e-12,
    )

    ext = analysis["extension_diagnostics"]
    assert ext["M130_TASK_MATCHED_SIGN_LM"] == "-"
    assert ext["M130_CONTRAMAMBA_SIGN"] == "+"
    assert ext["M130_SIGN_CONCORDANCE"] is False

    five = analysis["five_scale_descriptive_completeness"]
    assert five["vanilla_lm_task_matched_sign_vector"] == [
        "-", "-", "-", "+", "+"
    ]
    assert five["contramamba_forward_equivalent_sign_vector"] == [
        "+", "+", "+", "-", "-"
    ]
    assert five["aggregate_sign_opposite_at_all_five_sampled_scales"] is True
