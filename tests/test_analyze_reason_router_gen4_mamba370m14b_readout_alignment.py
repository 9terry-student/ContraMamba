from __future__ import annotations

import numpy as np
from scripts import analyze_reason_router_gen4_mamba370m14b_readout_alignment as subject

def test_primary_support_requires_contrast_and_both_sign_gates() -> None:
    a = np.linspace(0.2, 0.4, 300)
    b = np.linspace(-0.35, -0.25, 300)
    out = subject.primary_inference(a, b)
    assert out["primary_test"]["p_value_count"] == 1
    assert out["sign_reversal_supported"] is True
    assert out["result"] == subject.RESULT_SUPPORTED

def test_positive_cross_scale_separation_without_negative_14b_is_not_reversal() -> None:
    a = np.linspace(0.4, 0.6, 300)
    b = np.linspace(0.1, 0.2, 300)
    out = subject.primary_inference(a, b)
    assert out["primary_test"]["p_value"] < 0.05
    assert out["sign_gates"]["gate_1.4B_negative"] is False
    assert out["sign_reversal_supported"] is False
    assert out["result"] == subject.RESULT_SEPARATED

def test_negative_or_null_contrast_not_supported() -> None:
    a = np.linspace(-0.1, 0.1, 300)
    b = np.linspace(0.2, 0.35, 300)
    out = subject.primary_inference(a, b)
    assert out["sign_reversal_supported"] is False
    assert out["result"] == subject.RESULT_NOT
