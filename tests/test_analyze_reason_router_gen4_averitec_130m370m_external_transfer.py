from __future__ import annotations

import inspect
import math

import pytest

from scripts import (
    analyze_reason_router_gen4_averitec_130m370m_external_transfer
    as analysis,
)


def test_holm_two_scale_family() -> None:
    got = analysis.holm_adjust({
        "mamba130m": 0.01,
        "mamba370m": 0.04,
    })
    assert got["mamba130m"] == pytest.approx(0.02)
    assert got["mamba370m"] == pytest.approx(0.04)


def test_holm_order_invariant() -> None:
    a = analysis.holm_adjust({
        "mamba130m": 0.20,
        "mamba370m": 0.01,
    })
    b = analysis.holm_adjust({
        "mamba370m": 0.01,
        "mamba130m": 0.20,
    })
    assert a == b
    assert a["mamba370m"] == pytest.approx(0.02)
    assert a["mamba130m"] == pytest.approx(0.20)


def test_primary_test_is_one_sided_greater() -> None:
    source = inspect.getsource(analysis.one_sample_greater)
    assert "stats.ttest_1samp" in source
    assert 'alternative="greater"' in source
    assert "popmean=0.0" in source


def test_exactly_two_primary_scale_tests_by_construction() -> None:
    source = inspect.getsource(analysis.analyze)
    assert analysis.SCALES == ("mamba130m", "mamba370m")
    assert 'scale="mamba130m"' in source
    assert 'scale="mamba370m"' in source
    assert '"mamba14b_in_family": False' in source
    assert 'primary_p_value_count": 2' in source


def test_conclusion_labels_are_frozen() -> None:
    source = inspect.getsource(analysis.analyze)
    assert (
        "CROSS_SCALE_AVERITEC_GOLD_EVIDENCE_CAUSAL_TRANSFER_"
        in source
    )
    assert "SCALE_SPECIFIC_AVERITEC_GOLD_EVIDENCE_CAUSAL_TRANSFER_ONLY" in source
    assert "AVERITEC_GOLD_EVIDENCE_CAUSAL_TRANSFER_NOT_ESTABLISHED" in source


def test_analyzer_has_only_one_reusable_ttest_call_site() -> None:
    source = inspect.getsource(analysis)
    assert source.count("stats.ttest_1samp(") == 1


def test_no_rescue_or_extra_primary_family() -> None:
    source = inspect.getsource(analysis.analyze)
    assert '"rescue_performed": False' in source
    assert '"additional_primary_p_values": False' in source
    assert '"mamba14b_in_family": False' in source
