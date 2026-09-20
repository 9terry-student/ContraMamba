from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts import (
    reason_router_gen4_precursor_v4_analytic_vjp_one_row_equivalence
    as eq,
)


def test_protocol_constants() -> None:
    eq.validate_protocol()
    assert eq.EQUIVALENCE_ATOL == 1e-4
    assert eq.EQUIVALENCE_RTOL == 1e-4
    assert eq.PLANES == ("P3", "P5")
    assert eq.BASIS_NAMES == ("plus", "minus")
    assert eq.DIM == 650
    assert eq.RELATIVE_OFFSET == -4
    assert eq.GENERATED_PREFIX_LENGTH == 5
    assert eq.CPU_MODEL_FORWARD_COUNT == 1
    assert eq.GPU_MODEL_FORWARD_COUNT == 1
    assert eq.CPU_VJP_COUNT == 1
    assert eq.GPU_VJP_COUNT == 1
    assert eq.TOTAL_EQUIVALENCE_FORWARD_COUNT == 2
    assert eq.TOTAL_EQUIVALENCE_VJP_COUNT == 2


def test_historical_row_selection_is_first_decisive_gold_only() -> None:
    rows = [
        {"correct_label": "NOT_ENTITLED", "example_id": "a"},
        {"correct_label": "REFUTE", "example_id": "b"},
        {"correct_label": "SUPPORT", "example_id": "c"},
    ]
    padded = rows + [
        {"correct_label": "NOT_ENTITLED", "example_id": f"x{i}"}
        for i in range(eq.base.N - len(rows))
    ]
    index, row = eq.select_historical_decisive_row(padded)
    assert index == 1
    assert row["example_id"] == "b"


def test_differentiable_gold_margin_gradients() -> None:
    logits = torch.tensor([2.0, 5.0, 7.0], requires_grad=True)
    branch_tokens = {"REFUTE": 0, "SUPPORT": 2}

    support = eq.differentiable_gold_margin(
        logits,
        gold_label="SUPPORT",
        branch_tokens=branch_tokens,
    )
    grad_support, = torch.autograd.grad(support, logits, retain_graph=True)
    assert support.item() == pytest.approx(5.0)
    assert grad_support.tolist() == pytest.approx([-1.0, 0.0, 1.0])

    refute = eq.differentiable_gold_margin(
        logits,
        gold_label="REFUTE",
        branch_tokens=branch_tokens,
    )
    grad_refute, = torch.autograd.grad(refute, logits)
    assert refute.item() == pytest.approx(-5.0)
    assert grad_refute.tolist() == pytest.approx([1.0, 0.0, -1.0])


def test_summarize_directional_values() -> None:
    out = eq.summarize_directional_values({
        "P3": {"plus": 3.0, "minus": 4.0},
        "P5": {"plus": 0.0, "minus": 1.0},
    })
    assert out["chi_p3"] == pytest.approx(5.0)
    assert out["chi_p5"] == pytest.approx(1.0)
    assert out["d_t"] == pytest.approx(4.0)


def test_compare_observations_uses_frozen_tolerance() -> None:
    base = {
        "gold_aligned_f_t": 1.0,
        "directional_derivatives": {
            "P3": {"plus": 0.1, "minus": 0.2},
            "P5": {"plus": 0.05, "minus": 0.08},
        },
        "chi_p3": (0.1**2 + 0.2**2) ** 0.5,
        "chi_p5": (0.05**2 + 0.08**2) ** 0.5,
        "d_t": (0.1**2 + 0.2**2) ** 0.5 - (0.05**2 + 0.08**2) ** 0.5,
    }
    same = {
        **base,
        "gold_aligned_f_t": 1.00005,
        "directional_derivatives": {
            "P3": {"plus": 0.10001, "minus": 0.19999},
            "P5": {"plus": 0.05001, "minus": 0.07999},
        },
    }
    summary = eq.summarize_directional_values(same["directional_derivatives"])
    same = {
        **same,
        "chi_p3": summary["chi_p3"],
        "chi_p5": summary["chi_p5"],
        "d_t": summary["d_t"],
    }
    assert eq.compare_observations(base, same)["result"] == "PASS"

    bad = dict(same)
    bad["gold_aligned_f_t"] = 1.001
    with pytest.raises(eq.PrecursorV4AnalyticVJPError):
        eq.compare_observations(base, bad)


def test_source_has_no_finite_difference_or_fresh_cohort_access() -> None:
    source = Path(eq.__file__).read_text(encoding="utf-8")
    assert "EPSILON" not in source
    assert "epsilon_sign" not in source
    assert "validate_frozen_cohort(" not in source
    assert "precursor_v2_stage_a_cohort.jsonl" not in source
    assert '"fresh_n800_scientific_cohort_accessed": False' in source
    assert '"scientific_model_forward_count": 0' in source
    assert '"parameter_update_executed": False' in source
