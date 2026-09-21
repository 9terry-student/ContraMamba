from __future__ import annotations

import inspect
import json
import math
import subprocess
import sys
import types

import numpy as np
import pytest


from scripts import (
    analyze_reason_router_gen4_averitec_mamba14b_negative_sign_transfer
    as analysis,
)


def test_frozen_raw_identity_constants() -> None:
    assert analysis.RAW_FREEZE_COMMIT == (
        "59c24ae696700dd722b8e1ca628fe495cfa65738"
    )
    assert analysis.RAW_GIT_BLOBS[
        analysis.RAW_ROWS_FILE.as_posix()
    ] == "87c03efb0a5f74cf7f89a069bf5edfb166caaac0"
    assert analysis.RAW_SHA256["external_transfer_rows.jsonl"] == (
        "e2ef452a6dd3bb076924db5e4892741948e4a3c8c5ff11913a2c848cbc72980a"
    )


def test_primary_contract_is_exactly_one_less_test() -> None:
    source = inspect.getsource(analysis.one_sample_less)
    assert "stats.ttest_1samp" in source
    assert 'alternative="less"' in source
    assert "popmean=0.0" in source

    whole = inspect.getsource(analysis)
    assert whole.count("stats.ttest_1samp(") == 1
    assert analysis.N == 462
    assert analysis.DF == 461
    assert analysis.ALPHA == 0.05


def test_primary_less_statistics_on_synthetic_negative_values(monkeypatch) -> None:
    monkeypatch.setattr(analysis, "N", 4)
    monkeypatch.setattr(analysis, "DF", 3)
    result = analysis.one_sample_less([-1.0, -2.0, -3.0, -4.0])
    assert result["n"] == 4
    assert result["df"] == 3
    assert result["mean"] == pytest.approx(-2.5)
    assert result["sd_sample"] > 0.0
    assert result["t_statistic"] < 0.0
    assert 0.0 <= result["p_one_sided_less"] <= 1.0
    assert result["fraction_negative"] == 1.0
    assert result["fraction_positive"] == 0.0


def _synthetic_rows():
    rows = []
    specs = [
        ("a", 1, "Refuted", "REFUTE", -1.0, -0.5, 0.0),
        ("b", 2, "Supported", "SUPPORT", -2.0, -1.0, 0.0),
        ("c", 3, "Not Enough Evidence", "NOT_ENTITLED", -3.0, -1.5, 0.0),
        ("d", 4, "Refuted", "REFUTE", -4.0, -2.0, 0.0),
    ]
    for item_id, dev_index, source_label, correct_label, mn, mz, mc in specs:
        for condition, margin, pred in (
            ("native", mn, 0),
            ("dominant_neutralized", mz, 1),
            ("dominant_control", mc, 2),
        ):
            rows.append({
                "example_id": item_id,
                "averitec_dev_index": dev_index,
                "source_label": source_label,
                "correct_label": correct_label,
                "condition": condition,
                "correct_class_logit_margin": margin,
                "prediction_id": pred,
                "is_correct": False,
            })
    return rows


def test_analyze_reconstructs_endpoint_and_support_rule(monkeypatch) -> None:
    monkeypatch.setattr(analysis, "N", 4)
    monkeypatch.setattr(analysis, "DF", 3)
    monkeypatch.setattr(
        analysis,
        "validate_raw",
        lambda: (
            _synthetic_rows(),
            {"execution_head": "raw-head"},
            {},
        ),
    )
    result, items = analysis.analyze()
    assert len(items) == 4
    assert [row["D_EXT_14B"] for row in items] == [-1.0, -2.0, -3.0, -4.0]
    assert result["primary"]["mean"] == pytest.approx(-2.5)
    assert result["negative_mean_sign_gate"] is True
    assert result["primary_p_value_count"] == 1
    assert result["secondary_p_value_count"] == 0
    assert result["historical_p_values_in_new_family"] == 0
    assert result["row_filter_performed"] is False
    assert result["rescue_performed"] is False
    assert result["model_forward_count"] == 0


def test_conclusion_requires_both_negative_mean_and_alpha_gate(monkeypatch) -> None:
    monkeypatch.setattr(analysis, "N", 4)
    monkeypatch.setattr(analysis, "DF", 3)
    monkeypatch.setattr(
        analysis,
        "validate_raw",
        lambda: (
            _synthetic_rows(),
            {"execution_head": "raw-head"},
            {},
        ),
    )
    result, _ = analysis.analyze()
    expected = (
        analysis.SUPPORTED
        if (
            result["primary"]["mean"] < 0.0
            and result["primary"]["p_one_sided_less"] < analysis.ALPHA
        )
        else analysis.NOT_ESTABLISHED
    )
    assert result["scientific_conclusion"] == expected


def test_no_holm_no_second_test_no_historical_family_reopen() -> None:
    source = inspect.getsource(analysis).lower()
    assert "holm_adjust" not in source
    assert "alternative=\"greater\"" not in source
    assert "pearson" not in source
    assert "spearman" not in source
    assert '"historical_130m370m_averitec_family_reopened": false' in source
    assert '"historical_p_values_in_new_family": 0' in source
    assert '"primary_p_value_count": 1' in source
    assert '"secondary_p_value_count": 0' in source


def test_static_analyzer_has_no_model_execution() -> None:
    source = inspect.getsource(analysis).lower()
    forbidden = (
        "torch",
        "historical_forward",
        "cuda",
        "reconstruct_model",
        "automodel",
        "training_step",
    )
    for token in forbidden:
        assert token not in source


def test_output_contract_is_four_files() -> None:
    assert analysis.ANALYSIS_FILE == "external_transfer_analysis.json"
    assert analysis.ITEM_FILE == "external_transfer_items.jsonl"
    assert analysis.MANIFEST_FILE == "artifact_manifest.json"
    assert analysis.CHECKSUM_FILE == "SHA256SUMS.txt"


def test_result_labels_are_frozen() -> None:
    assert analysis.RESULT_PASS == (
        "PASS_GEN4_MAMBA14B_AVERITEC_NEGATIVE_SIGN_STATIC_ANALYSIS"
    )
    assert analysis.SUPPORTED == (
        "MAMBA14B_AVERITEC_NEGATIVE_SIGN_TRANSFER_SUPPORTED"
    )
    assert analysis.NOT_ESTABLISHED == (
        "MAMBA14B_AVERITEC_NEGATIVE_SIGN_TRANSFER_NOT_ESTABLISHED"
    )
