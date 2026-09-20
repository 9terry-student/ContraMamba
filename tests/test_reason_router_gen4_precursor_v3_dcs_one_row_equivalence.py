from __future__ import annotations

from pathlib import Path

import pytest

from scripts import (
    reason_router_gen4_precursor_v3_dcs_one_row_equivalence
    as eq,
)


def test_protocol_constants() -> None:
    eq.validate_protocol()
    assert eq.EQUIVALENCE_ATOL == 1e-4
    assert eq.EQUIVALENCE_RTOL == 1e-4
    assert eq.V3_EPSILON == 0.05
    assert eq.dcs.EPSILON == 0.025
    assert eq.RELATIVE_OFFSET == -4
    assert eq.GENERATED_PREFIX_LENGTH == 5
    assert eq.PROBES_PER_BACKEND == 8
    assert eq.CPU_MODEL_FORWARD_COUNT == 8
    assert eq.GPU_MODEL_FORWARD_COUNT == 8
    assert eq.TOTAL_EQUIVALENCE_FORWARD_COUNT == 16


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


def test_compare_scalar_uses_frozen_atol_rtol() -> None:
    assert eq._compare_scalar(1.0, 1.00015, label="ok") == pytest.approx(
        0.00015
    )
    with pytest.raises(eq.PrecursorV3DCSEquivalenceError):
        eq._compare_scalar(1.0, 1.00025, label="fail")


def _probe(
    plane: str,
    basis: str,
    sign: int,
    *,
    shift: float = 0.0,
) -> dict[str, object]:
    base_margin = {
        ("P3", "plus"): 0.10,
        ("P3", "minus"): 0.20,
        ("P5", "plus"): 0.05,
        ("P5", "minus"): 0.08,
    }[(plane, basis)]
    margin = sign * base_margin + shift
    return {
        "plane": plane,
        "basis": basis,
        "epsilon_sign": sign,
        "epsilon": eq.V3_EPSILON,
        "branch_start_logits": {
            "REFUTE": 1.0 + margin,
            "SUPPORT": 2.0 + margin,
        },
        "gold_aligned_decisive_margin": margin,
        "intervention_audit": {
            "applied_delta_l2": eq.V3_EPSILON,
        },
        "equivalence_model_forward_count": 1,
    }


def test_compare_probe_sets_and_derived_dcs() -> None:
    cpu = [
        _probe(plane, basis, sign)
        for plane in eq.dcs.PLANES
        for basis in eq.dcs.BASIS_NAMES
        for sign in eq.dcs.EPSILON_SIGNS
    ]
    gpu = [
        _probe(plane, basis, sign, shift=1e-6)
        for plane in eq.dcs.PLANES
        for basis in eq.dcs.BASIS_NAMES
        for sign in eq.dcs.EPSILON_SIGNS
    ]
    result = eq.compare_probe_sets(cpu, gpu)
    assert result["result"] == "PASS"
    assert result["max_branch_token_logit_abs_diff"] == pytest.approx(1e-6)
    # Equal additive shift on +/- probes cancels in the central difference.
    assert result["d_t_abs_diff"] == pytest.approx(0.0, abs=1e-12)
    assert eq.dcs.EPSILON == 0.025


def test_source_never_reads_new_precursor_v2_cohort() -> None:
    source = Path(eq.__file__).read_text(encoding="utf-8")
    assert "precursor_v2_stage_a_cohort.jsonl" not in source
    assert "validate_frozen_cohort(" not in source
    assert '"new_precursor_v3_scientific_cohort_accessed": False' in source
    assert '"scientific_model_forward_count": 0' in source
    assert '"p_value_count_added": 0' in source
