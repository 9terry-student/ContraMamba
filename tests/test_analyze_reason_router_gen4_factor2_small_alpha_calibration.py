from __future__ import annotations

import inspect

import numpy as np

from scripts import (
    analyze_reason_router_gen4_factor2_small_alpha_calibration
    as analyzer,
)


def test_protocol_is_exact() -> None:
    analyzer.validate_protocol()
    assert analyzer.PAIR_COUNT == 300
    assert analyzer.CELLS == ("C0_SHAM", "C2_NAME")
    assert analyzer.SCALES == ("mamba370m", "mamba14b")
    assert analyzer.ALPHAS == (0.25, 0.125, 0.0625, 0.03125)
    assert all(alpha > 0.0 for alpha in analyzer.ALPHAS)
    assert 1.0 not in analyzer.ALPHAS


def test_origin_slope_recovers_exact_factor() -> None:
    x = np.asarray([1.0, 2.0, -3.0, 4.0], dtype=np.float64)
    assert analyzer.origin_slope(x, x) == 1.0
    assert analyzer.origin_slope(x, 2.0 * x) == 2.0


def test_rank_average_handles_ties() -> None:
    ranks = analyzer.rank_average([10.0, 20.0, 20.0, 5.0])
    assert np.allclose(ranks, [2.0, 3.5, 3.5, 1.0])


def test_coefficients_are_coefficient_only() -> None:
    x = np.asarray([1.0, 2.0, 3.0, 4.0])
    y = np.asarray([2.0, 4.0, 6.0, 8.0])
    assert np.isclose(analyzer.pearson_coefficient(x, y), 1.0)
    assert np.isclose(analyzer.spearman_coefficient(x, y), 1.0)


def test_rmse_reference_identity() -> None:
    x = np.asarray([0.1, -0.2, 0.3])
    assert analyzer.rmse(x, x) == 0.0
    assert analyzer.rmse(2.0 * x, 2.0 * x) == 0.0


def test_calibration_summary_K1_and_K2() -> None:
    d = np.linspace(-1.5, 1.5, analyzer.PAIR_COUNT)
    for alpha in analyzer.ALPHAS:
        first = analyzer.summarize_calibration(
            d,
            alpha * d,
            alpha=alpha,
        )
        factor2 = analyzer.summarize_calibration(
            d,
            2.0 * alpha * d,
            alpha=alpha,
        )
        assert np.isclose(first["origin_slope_K"], 1.0)
        assert np.isclose(first["rmse_vs_alpha_Delta_L"], 0.0)
        assert np.isclose(factor2["origin_slope_K"], 2.0)
        assert np.isclose(factor2["rmse_vs_2alpha_Delta_L"], 0.0)
        assert first["p_value_count"] == 0
        assert first["inference_performed"] is False


def test_displacement_relative_metric_scales_with_alpha() -> None:
    rng = np.random.default_rng(7)
    native = rng.normal(size=(analyzer.PAIR_COUNT, 6))
    delta = rng.normal(size=(analyzer.PAIR_COUNT, 6))
    strong = np.asarray([1, 3, 5], dtype=np.int64)
    native_strong = native[:, strong]

    a = analyzer.summarize_displacement(
        native,
        native_strong,
        delta,
        strong,
        alpha=0.25,
    )
    b = analyzer.summarize_displacement(
        native,
        native_strong,
        delta,
        strong,
        alpha=0.125,
    )
    assert np.isclose(
        b["R_rel_full"]["mean"],
        0.5 * a["R_rel_full"]["mean"],
    )
    assert np.isclose(
        b["R_rel_strong"]["mean"],
        0.5 * a["R_rel_strong"]["mean"],
    )


def test_static_analyzer_contains_no_inferential_or_model_execution_dependency() -> None:
    source = inspect.getsource(analyzer).lower()
    assert "scipy" not in source
    assert "spearmanr" not in source
    assert "ttest" not in source
    assert "p_value_count_added" in source
    assert "torch" not in source
    assert "transformers" not in source
    assert '"scientific_conclusion": none' in source


def test_frozen_inputs_are_read_from_git_blobs() -> None:
    source = inspect.getsource(analyzer)
    assert '["git", "show", f"HEAD:{rel}"]' in source
    assert analyzer.INPUT_FREEZE_COMMIT == (
        "a88b71070310906b664c578c9627d9af3ad441d1"
    )


def test_output_contract() -> None:
    assert analyzer.PAIR_FILE == "pair_level_merge.jsonl"
    assert analyzer.CALIBRATION_FILE == "calibration_analysis.json"
    assert analyzer.DISPLACEMENT_FILE == "displacement_context.json"
    assert analyzer.MANIFEST_FILE == "artifact_manifest.json"
    assert analyzer.SUMS_FILE == "SHA256SUMS.txt"
