from __future__ import annotations

import inspect

import numpy as np

from scripts import (
    analyze_reason_router_gen4_factor2_gradient_consistency
    as analyzer,
)


def test_origin_slope_recovers_1x_and_2x() -> None:
    x = np.asarray([1.0, -2.0, 3.0, 4.0])
    assert analyzer.origin_slope(x, x) == 1.0
    assert analyzer.origin_slope(x, 2.0 * x) == 2.0


def test_rank_average_and_coefficients() -> None:
    ranks = analyzer.rank_average(
        [10.0, 20.0, 20.0, 5.0]
    )
    assert np.allclose(
        ranks,
        [2.0, 3.5, 3.5, 1.0],
    )
    x = np.asarray([1.0, 2.0, 3.0, 4.0])
    y = 3.0 * x
    assert np.isclose(
        analyzer.pearson_coefficient(x, y),
        1.0,
    )
    assert np.isclose(
        analyzer.spearman_coefficient(x, y),
        1.0,
    )


def test_vector_summary_distinguishes_1x_from_2x() -> None:
    x = np.linspace(-2.0, 2.0, 64)
    one = analyzer.vector_summary(x, x)
    two = analyzer.vector_summary(x, 2.0 * x)
    assert np.isclose(
        one["origin_slope_estimate_on_frozen_Delta_L"],
        1.0,
    )
    assert np.isclose(
        one["rmse_vs_1x_frozen_Delta_L"],
        0.0,
    )
    assert np.isclose(
        two["origin_slope_estimate_on_frozen_Delta_L"],
        2.0,
    )
    assert np.isclose(
        two["rmse_vs_2x_frozen_Delta_L"],
        0.0,
    )


def test_derive_one_epsilon_uses_fixed_native_wrong_margin() -> None:
    epsilon = 0.03125
    delta_l = 2.0
    native = [0.0, 3.0, 5.0]
    # correct=2, frozen wrong=1. Perturb only fixed margin linearly.
    plus = [0.0, 3.0, 5.0 - epsilon * delta_l]
    minus = [0.0, 3.0, 5.0 + epsilon * delta_l]
    row = {
        "frozen_Delta_L_row": delta_l,
        "correct_label_id": 2,
        "frozen_active_wrong_class_id": 1,
        "native": {"final_logits": native},
        "epsilon_results": {
            str(epsilon): {
                "plus": {"final_logits": plus},
                "minus": {"final_logits": minus},
            }
        },
    }
    result = analyzer.derive_one_epsilon(
        row,
        epsilon=epsilon,
    )
    assert np.isclose(
        result["fixed_central_difference"],
        delta_l,
    )
    assert np.isclose(
        result["fixed_positive_one_sided_estimate"],
        delta_l,
    )
    assert np.isclose(
        result["fixed_negative_one_sided_estimate"],
        delta_l,
    )


def test_static_analyzer_is_descriptive_only() -> None:
    source = inspect.getsource(analyzer).lower()
    assert "scipy" not in source
    assert "ttest" not in source
    assert "spearmanr" not in source
    assert '"primary_p_value_count_added": 0' in source
    assert '"secondary_p_value_count_added": 0' in source
    assert '"inference_performed": false' in source
    assert '"scientific_conclusion": none' in source


def test_static_analyzer_reads_frozen_raw_via_git_blob() -> None:
    source = inspect.getsource(analyzer)
    assert '["git", "show", f"HEAD:{rel}"]' in source


def test_output_contract() -> None:
    assert analyzer.ROW_DERIVATIVE_FILE == (
        "row_level_derivatives.jsonl"
    )
    assert analyzer.PAIR_DERIVATIVE_FILE == (
        "pair_level_derivatives.jsonl"
    )
    assert analyzer.ANALYSIS_FILE == (
        "gradient_consistency_analysis.json"
    )
    assert analyzer.MANIFEST_FILE == "artifact_manifest.json"
    assert analyzer.SUMS_FILE == "SHA256SUMS.txt"
