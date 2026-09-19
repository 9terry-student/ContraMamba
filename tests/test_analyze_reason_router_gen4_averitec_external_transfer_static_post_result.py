from __future__ import annotations

import inspect

import pytest

from scripts import (
    analyze_reason_router_gen4_averitec_external_transfer_static_post_result
    as analysis,
)


def test_descriptive_basic() -> None:
    got = analysis.descriptive([-1.0, 0.0, 1.0, 2.0])
    assert got["n"] == 4
    assert got["mean"] == pytest.approx(0.5)
    assert got["median"] == pytest.approx(0.5)
    assert got["fraction_positive"] == pytest.approx(0.5)
    assert got["fraction_negative"] == pytest.approx(0.25)
    assert got["fraction_zero"] == pytest.approx(0.25)


def test_pearson_basic() -> None:
    assert analysis.pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)
    assert analysis.pearson([1.0, 1.0], [2.0, 3.0]) is None


def test_no_inferential_test_call_sites() -> None:
    source = inspect.getsource(analysis).lower()
    assert "ttest" not in source
    assert "mannwhitney" not in source
    assert "wilcoxon" not in source
    assert "fisher_exact" not in source
    assert "chi2" not in source
    assert "permutation" not in source
    assert "bootstrap" not in source


def test_static_contract_is_zero_forward_zero_pvalue() -> None:
    source = inspect.getsource(analysis.analyze)
    assert '"new_model_forward_count": 0' in source
    assert '"new_p_value_count": 0' in source
    assert '"primary_family_reopened": False' in source
    assert '"selection_reopened": False' in source
    assert '"rescue_performed": False' in source
    assert '"subset_promotion_performed": False' in source


def test_expected_frozen_source_identity() -> None:
    assert analysis.EXPECTED_EXECUTION_HEAD == (
        "c5b470f0601c0d287218bf39cefc0e681b3f81fd"
    )
    assert analysis.EXPECTED_RECOVERY_RUN.endswith("postcheck-recovery2")
    assert analysis.EXPECTED_ANALYSIS_SHA256 == (
        "0c8e5fff026fbc243f7344d9e2af57cd00a6597ce7c468f0fd8b604dbe7d2a7f"
    )


def test_cross_scale_simple() -> None:
    r130 = [
        {"example_id": "a", "D_EXT": 1.0, "source_label": "Refuted"},
        {"example_id": "b", "D_EXT": -1.0, "source_label": "Supported"},
    ]
    r370 = [
        {"example_id": "a", "D_EXT": 2.0, "source_label": "Refuted"},
        {"example_id": "b", "D_EXT": 3.0, "source_label": "Supported"},
    ]
    got = analysis.cross_scale(r130, r370)
    assert got["n"] == 2
    assert got["both_positive_fraction"] == pytest.approx(0.5)
    assert got["opposite_sign_fraction"] == pytest.approx(0.5)
