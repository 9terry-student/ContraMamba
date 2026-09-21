from __future__ import annotations

import inspect

import torch

from scripts import reason_router_gen4_factor2_reference_backend_verifier as runner


def test_protocol_is_bounded_and_outcome_independent() -> None:
    runner.validate_protocol()
    assert runner.SUBSET_SELECTION_SALT == "factor2-reference-backend-v1"
    assert runner.SUBSET_SIZE == 4
    assert runner.SUBSET_PAIRS == (
        "xg1_fact_5726",
        "xg1_fact_5879",
        "xg1_fact_5940",
        "xg1_fact_5975",
    )
    assert runner.EPSILON == 0.03125
    assert runner.TOTAL_ROWS == 16
    assert runner.TOTAL_FORWARDS == 96
    assert runner.TOTAL_BACKWARDS == 32


def test_project_delta_l_sign_matches_definition() -> None:
    g = torch.tensor([1.0, 2.0], dtype=torch.float64)
    delta_control = torch.tensor([3.0, -4.0], dtype=torch.float64)
    # Delta_L = g·(selected-control) = -g·delta_control
    assert runner.project_delta_l(g, delta_control) == 5.0


def test_central_difference_definition() -> None:
    eps = runner.EPSILON
    derivative = 3.25
    native = 7.0
    plus = native - eps * derivative
    minus = native + eps * derivative
    assert abs(runner.central_from_margins(plus, minus) - derivative) < 1e-12


def test_force_slow_uses_transformers_slow_forward_explicitly() -> None:
    source = inspect.getsource(runner.force_all_slow)
    assert "self.slow_forward" in source
    assert "mixer.forward" in source


def test_runner_does_not_adapt_backend_or_epsilon() -> None:
    source = inspect.getsource(runner).lower()
    assert '"adaptive_backend_selection_performed": false' in source
    assert '"adaptive_epsilon_selection_performed": false' in source
    assert "scipy" not in source
    assert "ttest" not in source
    assert '"scientific_conclusion": none' in source
