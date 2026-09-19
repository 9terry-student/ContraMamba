from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from scripts import (
    reason_router_gen4_mamba14b_residual_characterization_fast_cuda
    as subject,
)


def _planes():
    planes = {}
    for i, plane in enumerate(subject.PLANE_ORDER):
        plus = torch.zeros(subject.DIM, dtype=torch.float64)
        minus = torch.zeros(subject.DIM, dtype=torch.float64)
        plus[2 * i] = 1.0
        minus[2 * i + 1] = 1.0
        planes[plane] = {"plus": plus, "minus": minus}
    return planes


def test_protocol_and_budget_are_frozen():
    subject.validate_protocol()
    assert subject.PAIR_IDS[0] == "xg1_fact_4501"
    assert subject.PAIR_IDS[-1] == "xg1_fact_4800"
    assert subject.SELECTED_DOMINANT == "P5"
    assert subject.RESIDUAL_PLANES == ("P1", "P2", "P3", "P4")
    assert subject.FORWARDS_PER_PAIR == 240
    assert subject.TOTAL_FORWARD_BUDGET == 72000
    assert subject.DIM == 829


def test_frozen_selection_is_p5():
    selection = subject.load_frozen_selection()
    assert selection["selected_dominant_candidate"] == "P5"
    assert selection["selection_unique"] is True


def test_population_is_exact_and_response_blind():
    facts, rows = subject.load_population()
    assert len(facts) == 300
    assert len(rows) == 1800
    assert facts[0]["pair_id"] == "xg1_fact_4501"
    assert facts[-1]["pair_id"] == "xg1_fact_4800"


def test_neutralization_preserves_selected_p5():
    planes = _planes()
    h = torch.arange(1, subject.DIM + 1, dtype=torch.float64) / 101.0
    before = (
        float(torch.dot(h, planes["P5"]["plus"])),
        float(torch.dot(h, planes["P5"]["minus"])),
    )
    for condition in (
        "p1_neutralized",
        "p2_neutralized",
        "p3_neutralized",
        "p4_neutralized",
        "residual_all_neutralized",
    ):
        result = subject.condition_correction(
            h, condition=condition, planes=planes
        )
        post = h + result["correction"]
        for plane in result["neutralized_planes"]:
            assert abs(float(torch.dot(post, planes[plane]["plus"]))) <= subject.TOL
            assert abs(float(torch.dot(post, planes[plane]["minus"]))) <= subject.TOL
        after = (
            float(torch.dot(post, planes["P5"]["plus"])),
            float(torch.dot(post, planes["P5"]["minus"])),
        )
        assert after == pytest.approx(before, abs=subject.TOL)


def test_c_k_is_mean_branch_energy():
    audits = [
        {"native_residual_coefficient_energy": {
            "P1": 1.0, "P2": 4.0, "P3": 9.0, "P4": 16.0}},
        {"native_residual_coefficient_energy": {
            "P1": 3.0, "P2": 6.0, "P3": 11.0, "P4": 18.0}},
    ]
    assert subject.mean_branch_coefficient_energy(audits) == {
        "P1": 2.0, "P2": 5.0, "P3": 10.0, "P4": 17.0
    }


def _item(i: int):
    vals = {
        "P1": 0.10 + i * 1e-4,
        "P2": -0.20 - i * 2e-4,
        "P3": 0.05 + i * 3e-5,
        "P4": 0.30 + i * 3e-4,
    }
    s_sum = sum(vals.values())
    s_res = s_sum + 0.07
    row = {
        "source_pair_id": subject.PAIR_IDS[i],
        "S_RES": s_res,
        "S_SUM": s_sum,
        "I_RES": 0.07,
        "Q_native": 1.0 + i * 1e-3,
        "Q_residual_all_neutralized": 1.0 + i * 1e-3 - s_res,
    }
    for j, plane in enumerate(subject.RESIDUAL_PLANES, 1):
        row[f"S_{plane}"] = vals[plane]
        row[f"C_{plane}"] = float(j) + i * 1e-3
    return row


def test_summary_is_descriptive_only():
    summary = subject.summarize_items(
        [_item(i) for i in range(subject.PAIR_COUNT)]
    )
    assert summary["interaction_residual"]["mean"] == pytest.approx(0.07)
    assert set(summary["individual_effects"]) == {"P1", "P2", "P3", "P4"}
    assert set(summary["coefficient_energy"]) == {"P1", "P2", "P3", "P4"}
    assert summary["formal_inference_performed"] is False
    assert summary["p_value_count"] == 0
    assert summary["selection_performed"] is False
    assert summary["core_confirmation_inference_accessed"] is False
    assert summary["scientific_conclusion_established"] is False
    assert summary["rescue_performed"] is False


def test_no_confirmation_artifact_or_module_access():
    source = Path(subject.__file__).read_text(encoding="utf-8")
    assert "reason_router_gen4_mamba14b_confirmation_fast_cuda" not in source
    assert "confirmation_inference.json" not in source


def test_identities_and_state_width():
    assert subject.HOLDOUT_FREEZE_COMMIT == "e988cf3e20990a9397b8238df0a1edaf03674522"
    assert subject.GEOMETRY_FREEZE_COMMIT == "f97b597fb4da08a8d360ed07e48c6727e15ae0be"
    assert subject.DISCOVERY_FREEZE_COMMIT == "e7804bcab88a50dd57c2edd86af88daeb90a56d5"
    assert subject.PRIOR_CORE_CONFIRMATION_FREEZE_COMMIT == "693c12fef2194cbc48d17ed8fc563cc5065dcae0"
    assert subject.DISCOVERY_SELECTION_SHA256 == "deeb9a2514503a317fd184c77c963d6fc296dc0cf75ce4d180fd46d319595c09"
    assert subject.SOURCE_SHA256 == "beeba91a246f5a13e2ba1843a87f49bd41efe9dc1c1fea3e9744dd42c892312d"
    assert subject.ROWS_SHA256 == "8e4d330a9b75a4b3a5f16b53ab6f2f0b25ff70cc908eefadbfb1eb9133d8a002"
    assert subject.STRUCTURAL_MANIFEST_SHA256 == "94fcd70d15dfc91b81651da7f2e7fa147a2f4adb230f5d2cad12c211293fc18d"

    state = torch.zeros(
        1,
        subject.geom.INTERMEDIATE_SIZE,
        subject.geom.STATE_SIZE,
        dtype=torch.float32,
    )
    flat = subject.flatten_state(state)
    assert flat.shape == (65536,)
    assert math.isfinite(float(flat.sum()))

def test_script_has_exactly_one_main_entrypoint():
    source = Path(subject.__file__).read_text(encoding="utf-8")
    assert source.count('if __name__ == "__main__":') == 1
    assert source.count("    main()") == 1
