from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts import (
    reason_router_gen4_precursor_v2_dcs_stage_a_fast_cuda
    as dcs,
)


def test_protocol_constants_and_forward_accounting() -> None:
    dcs.validate_protocol()
    assert dcs.N == 800
    assert dcs.EPSILON == 0.025
    assert dcs.OBSERVATION_OFFSETS == (-4, -3, -2, -1)
    assert dcs.PROBES_PER_OFFSET == 8
    assert dcs.PROBES_PER_ROW == 32
    assert dcs.NATIVE_FORWARDS_PER_ROW == 12
    assert dcs.TOTAL_FORWARDS_PER_ROW == 44
    assert dcs.SCIENTIFIC_FORWARD_BUDGET == 35200


def test_frozen_response_free_cohort_is_exact() -> None:
    rows, manifest = dcs.validate_frozen_cohort()
    assert len(rows) == 800
    assert manifest["cohort"]["source_label_counts"] == {
        "Refuted": 400,
        "Supported": 400,
    }
    assert manifest["cohort"]["correct_label_counts"] == {
        "REFUTE": 400,
        "SUPPORT": 400,
    }
    assert manifest["response_fields_present"] is False
    assert manifest["model_forward_count"] == 0
    assert manifest["scientific_inference_executed"] is False
    assert manifest["p_value_count_added"] == 0


def test_gold_aligned_margin() -> None:
    logits = {"REFUTE": 2.0, "SUPPORT": 5.5}
    assert dcs.gold_aligned_margin("SUPPORT", logits) == pytest.approx(3.5)
    assert dcs.gold_aligned_margin("REFUTE", logits) == pytest.approx(-3.5)


def _probe(
    plane: str,
    basis: str,
    sign: int,
    margin: float,
) -> dict[str, object]:
    return {
        "plane": plane,
        "basis": basis,
        "epsilon_sign": sign,
        "gold_aligned_decisive_margin": margin,
    }


def test_offset_summary_implements_frozen_central_difference() -> None:
    eps = dcs.EPSILON
    probes = [
        _probe("P3", "plus", -1, -eps * 3.0),
        _probe("P3", "plus", 1, eps * 3.0),
        _probe("P3", "minus", -1, -eps * 4.0),
        _probe("P3", "minus", 1, eps * 4.0),
        _probe("P5", "plus", -1, -eps * 1.0),
        _probe("P5", "plus", 1, eps * 1.0),
        _probe("P5", "minus", -1, 0.0),
        _probe("P5", "minus", 1, 0.0),
    ]
    out = dcs.summarize_offset_from_probe_values(probes=probes)
    assert out["derivatives"]["P3"]["plus"] == pytest.approx(3.0)
    assert out["derivatives"]["P3"]["minus"] == pytest.approx(4.0)
    assert out["chi_p3"] == pytest.approx(5.0)
    assert out["chi_p5"] == pytest.approx(1.0)
    assert out["d_t"] == pytest.approx(4.0)


def test_perturbation_hook_changes_only_target_strong_content() -> None:
    output = torch.zeros((1, 3, 8), dtype=torch.float32)
    strong_mask = torch.tensor([True, False, True, False])
    basis = torch.tensor([1.0, 0.0], dtype=torch.float64)
    audit: dict[str, object] = {}

    changed = dcs.perturbation_hook(
        output,
        token_index=1,
        strong_mask=strong_mask,
        basis=basis,
        epsilon_sign=1,
        epsilon=0.025,
        intermediate_size=4,
        dim=2,
        cast_tol=1e-7,
        audit=audit,
    )

    assert torch.equal(output, torch.zeros_like(output))
    assert changed[0, 1, 0].item() == pytest.approx(0.025)
    assert changed[0, 1, 2].item() == pytest.approx(0.0)
    assert torch.count_nonzero(changed[:, :, 4:]).item() == 0
    assert torch.count_nonzero(changed[:, 0, :]).item() == 0
    assert torch.count_nonzero(changed[:, 2, :]).item() == 0
    assert audit["gate_changed"] is False
    assert audit["nonstrong_changed"] is False
    assert audit["other_token_changed"] is False


def test_runner_source_contains_only_corrected_forward_budget() -> None:
    source = Path(dcs.__file__).read_text(encoding="utf-8")
    assert "SCIENTIFIC_FORWARD_BUDGET = N * TOTAL_FORWARDS_PER_ROW" in source
    assert "PROBES_PER_ROW = PROBES_PER_OFFSET * len(OBSERVATION_OFFSETS)" in source
    assert "22400" not in source
    assert "TOTAL_FORWARDS_PER_ROW == 44" in source
    assert "SCIENTIFIC_FORWARD_BUDGET == 35200" in source
