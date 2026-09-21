from __future__ import annotations

import inspect

import numpy as np

from scripts import (
    analyze_reason_router_gen4_mamba130m_readout_alignment
    as subject,
)


def _rows(delta_values: np.ndarray) -> list[dict]:
    assert delta_values.shape == (300,)
    rows = []
    for i, pair in enumerate(subject.PAIR_IDS):
        for cell, offset in (("C0_SHAM", -0.01), ("C2_NAME", 0.01)):
            delta = float(delta_values[i] + offset)
            rows.append({
                "scale": "mamba130m",
                "source_pair_id": pair,
                "contrast_cell_id": cell,
                "Delta_L_row": delta,
                "gradient_full_l2": 2.0,
                "selected_projection_l2": 0.5,
                "control_projection_l2": 0.4,
                "selected_projection_fraction": 0.25,
                "control_projection_fraction": 0.20,
                "selected_component_l2": 1.0,
                "control_component_l2": 1.0,
                "L_selected": delta + 0.3,
                "L_control": 0.3,
                "selected_directional_coordinates": [0.2, -0.1],
                "control_directional_coordinates": [0.1, -0.05],
                "cosine_gradient_selected_component": 0.2,
                "cosine_gradient_control_component": 0.1,
                "active_wrong_class_id": 0,
            })
    return rows


def test_pair_aggregation_is_exact_two_cell_mean() -> None:
    d = np.linspace(0.1, 0.2, 300)
    pairs = subject.pair_values(_rows(d))
    assert len(pairs) == 300
    observed = np.asarray([x["Delta_L"] for x in pairs])
    assert np.allclose(observed, d, rtol=0.0, atol=1e-15)


def test_positive_primary_support() -> None:
    d = np.linspace(0.05, 0.15, 300)
    out = subject.primary_inference(d)
    assert out["p_value_count"] == 1
    assert out["alternative"] == "greater"
    assert out["mean"] > 0.0
    assert out["p_value"] < 0.05
    assert out["positive_mean_sign_gate_pass"] is True
    assert out["positive_alignment_supported"] is True


def test_negative_primary_not_supported() -> None:
    d = np.linspace(-0.15, -0.05, 300)
    out = subject.primary_inference(d)
    assert out["positive_mean_sign_gate_pass"] is False
    assert out["positive_alignment_supported"] is False


def test_sign_gate_is_required_even_if_function_is_called() -> None:
    d = np.concatenate([
        np.full(150, -1.0),
        np.full(150, 0.9),
    ])
    out = subject.primary_inference(d)
    assert out["mean"] < 0.0
    assert out["positive_mean_sign_gate_pass"] is False
    assert out["positive_alignment_supported"] is False


def test_analyze_rows_executes_one_p_value_only() -> None:
    d = np.linspace(0.05, 0.15, 300)
    result, pairs = subject.analyze_rows(_rows(d))
    assert len(pairs) == 300
    assert result["total_p_value_count"] == 1
    assert result["primary"]["p_value_count"] == 1
    assert result["behavioral_merge_performed"] is False
    assert result["three_scale_synthesis_performed"] is False
    assert result["row_filter_performed"] is False
    assert result["rescue_performed"] is False


def test_secondary_reports_zero_gradient_and_no_extra_inference() -> None:
    d = np.linspace(0.05, 0.15, 300)
    rows = _rows(d)
    rows[0]["gradient_full_l2"] = 0.0
    result, _ = subject.analyze_rows(rows)
    assert result["secondary_descriptive"]["zero_gradient_count"] == 1
    assert result["secondary_descriptive"]["wrong_class_exact_tie_count"] == 0
    assert result["secondary_descriptive"]["nonfinite_count"] == 0
    assert result["total_p_value_count"] == 1


def test_raw_bundle_identity_constants_are_exact() -> None:
    assert subject.RAW_FREEZE_COMMIT == (
        "32eb3baf678a44946f4ecd102f917832929624a7"
    )
    assert subject.RAW_EXECUTION_HEAD == (
        "ccbfe14389655e3a900882505a10df0fb645217b"
    )
    assert subject.EXPECTED_RAW_SHA256[subject.ITEM_FILE] == (
        "27c428fbcc3f9e89cc4dad566666e4d5307727637cf8bf1dd0c347744da18c85"
    )
    assert subject.PLAN_SHA256 == (
        "a8a2cb2af1ed35b82f404a713c15ebf176cd645f653c6eacf73722b6e34058b3"
    )


def test_source_has_no_behavioral_merge_or_three_scale_inference() -> None:
    src = inspect.getsource(subject)
    forbidden = (
        "behavioral_restoration_bridge_analysis",
        "mamba370m14b_readout_alignment_analysis",
        "spearmanr(",
        "pearsonr(",
        "ttest_ind(",
        "ttest_rel(",
    )
    for token in forbidden:
        assert token not in src
