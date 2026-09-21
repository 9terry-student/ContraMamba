from __future__ import annotations

import numpy as np

from scripts import (
    analyze_reason_router_gen4_mamba370m14b_low_displacement_behavioral
    as subject,
)


def _rows(d370_05=0.20, d370_025=0.10, d14_05=-0.20, d14_025=-0.10):
    rows = []
    for i, pair in enumerate(subject.PAIR_IDS):
        variation = (i - 149.5) * 1e-5
        for scale, d05, d025 in (
            ("mamba370m", d370_05, d370_025),
            ("mamba14b", d14_05, d14_025),
        ):
            for cell in subject.CELLS:
                restored = 1.0
                local_d05 = d05 + variation
                local_d025 = d025 + 0.5 * variation
                for condition, alpha, d in (
                    ("restored_native", 0.0, 0.0),
                    ("control_alpha_0_5", 0.5, local_d05),
                    ("control_alpha_0_25", 0.25, local_d025),
                ):
                    margin = restored if condition == "restored_native" else restored - d
                    rows.append({
                        "scale": scale,
                        "source_pair_id": pair,
                        "contrast_cell_id": cell,
                        "condition": condition,
                        "alpha": alpha,
                        "correct_class_logit_margin": margin,
                    })
    return rows


def test_holm_two_test_adjustment() -> None:
    out = subject.holm_adjust({
        "mamba370m": 0.01,
        "mamba14b": 0.04,
    })
    assert out == {
        "mamba370m": 0.02,
        "mamba14b": 0.04,
    }


def test_opposite_direction_primary_family_can_preserve_pattern() -> None:
    result, pairs = subject.analyze_rows(
        _rows(),
        raw_execution_head="raw",
    )
    assert len(pairs) == 300
    assert result["primary_family"]["primary_p_value_count"] == 2
    assert result["primary_by_scale"]["mamba370m"]["alternative"] == "greater"
    assert result["primary_by_scale"]["mamba14b"]["alternative"] == "less"
    assert result["support_by_scale"] == {
        "mamba370m": True,
        "mamba14b": True,
    }
    assert result["scientific_conclusion"] == subject.RESULT_BOTH
    assert result["total_p_value_count"] == 2
    assert result["alpha_0_5_p_value_count"] == 0
    assert result["cross_scale_p_value_count"] == 0


def test_partial_label_when_only_370m_passes() -> None:
    result, _ = subject.analyze_rows(
        _rows(d14_025=0.10),
        raw_execution_head="raw",
    )
    assert result["support_by_scale"]["mamba370m"] is True
    assert result["support_by_scale"]["mamba14b"] is False
    assert result["scientific_conclusion"] == subject.RESULT_PARTIAL


def test_pair_level_r_is_370m_minus_14b() -> None:
    _, pairs = subject.analyze_rows(
        _rows(d370_05=0.2, d14_05=-0.3),
        raw_execution_head="raw",
    )
    assert np.isclose(pairs[0]["R_q_alpha_0_5"], 0.5)


def test_no_alpha1_behavioral_analysis() -> None:
    assert subject.ALPHAS == (0.5, 0.25)
    assert 1.0 not in subject.ALPHAS
