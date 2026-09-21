from __future__ import annotations

import inspect

import torch

from scripts import (
    reason_router_gen4_factor2_gradient_consistency_raw_fast_cuda
    as runner,
)


EXPECTED_SUBSET = (
    "xg1_fact_5716",
    "xg1_fact_5719",
    "xg1_fact_5724",
    "xg1_fact_5726",
    "xg1_fact_5739",
    "xg1_fact_5741",
    "xg1_fact_5758",
    "xg1_fact_5777",
    "xg1_fact_5795",
    "xg1_fact_5796",
    "xg1_fact_5802",
    "xg1_fact_5812",
    "xg1_fact_5822",
    "xg1_fact_5835",
    "xg1_fact_5838",
    "xg1_fact_5841",
    "xg1_fact_5869",
    "xg1_fact_5870",
    "xg1_fact_5874",
    "xg1_fact_5891",
    "xg1_fact_5909",
    "xg1_fact_5921",
    "xg1_fact_5940",
    "xg1_fact_5951",
    "xg1_fact_5952",
    "xg1_fact_5961",
    "xg1_fact_5966",
    "xg1_fact_5973",
    "xg1_fact_5979",
    "xg1_fact_5982",
    "xg1_fact_5988",
    "xg1_fact_5990",
)


def test_protocol_is_exact_and_outcome_independent() -> None:
    runner.validate_protocol()
    assert runner.SUBSET_SELECTION_SALT == (
        "factor2-gradient-consistency-v1"
    )
    assert runner.SUBSET_PAIRS == EXPECTED_SUBSET
    assert runner.SUBSET_SIZE == 32
    assert runner.EPSILONS == (
        0.03125,
        0.015625,
        0.0078125,
    )
    assert runner.SIGNED_EPSILONS == (
        0.03125,
        -0.03125,
        0.015625,
        -0.015625,
        0.0078125,
        -0.0078125,
    )
    assert runner.ROWS_PER_SCALE == 64
    assert runner.TOTAL_ROWS == 128
    assert runner.FORWARDS_PER_ROW == 7
    assert runner.FORWARDS_PER_SCALE == 448
    assert runner.TOTAL_FORWARD_BUDGET == 896


def test_signed_correction_supports_exact_positive_and_negative_grid() -> None:
    base = torch.tensor(
        [1.0, -2.0, 0.125, -0.0625],
        dtype=torch.float32,
    )
    for value in runner.SIGNED_EPSILONS:
        actual = runner.signed_full_correction(
            base,
            value,
        )
        assert torch.equal(actual, base * value)


def test_fixed_margin_freezes_wrong_class_identity() -> None:
    logits = [1.0, 4.0, 3.0]
    assert runner.fixed_margin(
        logits,
        correct_label_id=0,
        wrong_class_id=2,
    ) == -2.0
    dynamic, wrong = runner.dynamic_margin(
        logits,
        correct_label_id=0,
    )
    assert dynamic == -3.0
    assert wrong == 1


def test_central_difference_identity_for_matching_linear_derivative() -> None:
    # Let dM/dalpha = -Delta_L, so the sign-convention central
    # difference (M(-eps)-M(+eps))/(2 eps) equals Delta_L.
    epsilon = 0.03125
    delta_l = 2.5
    native = 7.0
    plus = native - epsilon * delta_l
    minus = native + epsilon * delta_l
    metrics = runner.central_difference_metrics(
        native_margin=native,
        plus_margin=plus,
        minus_margin=minus,
        epsilon=epsilon,
        frozen_delta_l=delta_l,
    )
    assert abs(
        metrics["central_difference_Delta_L"]
        - delta_l
    ) < 1e-12
    assert abs(
        metrics["central_over_frozen_Delta_L"]
        - 1.0
    ) < 1e-12


def test_even_second_difference_detects_quadratic_curvature() -> None:
    epsilon = 0.015625
    delta_l = 1.2
    curvature = 3.5
    native = 2.0

    def margin(alpha: float) -> float:
        return (
            native
            - delta_l * alpha
            + 0.5 * curvature * alpha * alpha
        )

    metrics = runner.central_difference_metrics(
        native_margin=native,
        plus_margin=margin(epsilon),
        minus_margin=margin(-epsilon),
        epsilon=epsilon,
        frozen_delta_l=delta_l,
    )
    assert abs(
        metrics["central_difference_Delta_L"]
        - delta_l
    ) < 1e-12
    assert abs(
        metrics["even_second_difference"]
        - curvature
    ) < 1e-10


def test_raw_runner_uses_frozen_autograd_and_no_new_backward() -> None:
    source = inspect.getsource(runner)
    lower = source.lower()
    assert "factor2_small_alpha_native_readout_raw_v1" in source
    assert "torch.autograd" not in source
    assert ".backward(" not in source
    assert '"local_backward_count": 0' in source
    assert '"frozen_autograd_readout_reused": true' in lower


def test_raw_runner_does_not_read_factor2_behavior_or_static_result() -> None:
    source = inspect.getsource(runner)
    forbidden = (
        "factor2_small_alpha_behavioral_raw_v1",
        "factor2_small_alpha_static_calibration_v1",
        "calibration_analysis.json",
        "pair_level_merge.jsonl",
    )
    for token in forbidden:
        assert token not in source


def test_technical_gate_is_numeric_outcome_blind() -> None:
    source = inspect.getsource(runner.technical_gate_worker)
    assert "numeric_logits_retained" in source
    assert "numeric_central_difference_retained" in source
    assert "numeric_ratio_retained" in source


def test_raw_protocol_contains_no_inference_or_adaptive_epsilon() -> None:
    source = inspect.getsource(runner).lower()
    assert "scipy" not in source
    assert "ttest" not in source
    assert '"p_value_count_executed": 0' in source
    assert '"row_filtering_performed": false' in source
    assert '"rescue_performed": false' in source
    assert '"adaptive_epsilon_selection_performed": false' in source
    assert '"scientific_conclusion": none' in source
