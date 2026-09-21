from __future__ import annotations

import numpy as np
from scipy import stats

from scripts import (
    analyze_reason_router_gen4_mamba14b_transported_p5_adjacent_response
    as subject,
)


def test_primary_contract_is_exact() -> None:
    assert subject.N == 300
    assert subject.ALPHA == 0.05
    assert subject.PRIMARY_ENDPOINT == "G=D_TRANSPORT-D_ADJ"
    assert subject.PRIMARY_TEST == "paired_one_sample_student_t_greater"
    assert subject.PRIMARY_P_VALUE_COUNT == 1
    assert subject.SIGN_GATE == "mean(D_TRANSPORT)>0"


def test_sign_gate_distinguishes_relative_shift_from_positive_restoration() -> None:
    d_adj = np.linspace(-3.0, -1.0, 300)
    d_transport = d_adj + np.linspace(0.5, 1.0, 300)
    g = d_transport - d_adj
    test = stats.ttest_1samp(g, 0.0, alternative="greater")
    assert float(test.pvalue) < 0.05
    assert float(g.mean()) > 0.0
    assert float(d_transport.mean()) < 0.0
    # This is the Study-A distinction: relative upward shift can pass while
    # positive transported response restoration remains unestablished.


def test_conclusion_labels_are_distinct() -> None:
    assert subject.CONCLUSION_RESTORED != subject.CONCLUSION_SHIFT_ONLY
    assert subject.CONCLUSION_SHIFT_ONLY != subject.CONCLUSION_NOT
