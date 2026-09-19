from __future__ import annotations

import math
import pytest
import torch

from scripts import reason_router_gen4_mamba14b_confirmation_fast_cuda as subject


def _orthogonal_planes():
    planes = {}
    for index, plane in enumerate(subject.PLANE_ORDER):
        plus = torch.zeros(subject.DIM, dtype=torch.float64)
        minus = torch.zeros(subject.DIM, dtype=torch.float64)
        plus[2 * index] = 1.0
        minus[2 * index + 1] = 1.0
        planes[plane] = {"plus": plus, "minus": minus}
    return planes


def _mock_item(index: int, value: float):
    return {"source_pair_id": subject.PAIR_IDS[index], "D_CORE": value}


def test_protocol_and_budget_are_exact():
    subject.validate_protocol()
    assert subject.PAIR_IDS[0] == "xg1_fact_4201"
    assert subject.PAIR_IDS[-1] == "xg1_fact_4500"
    assert subject.SELECTED_PLANE == "P5"
    assert subject.CONTROL_PLANE == "P4"
    assert subject.CONDITION_ORDER == ("dominant_restored", "dominant_control")
    assert subject.DIM == 829
    assert subject.FORWARDS_PER_CONDITION == 40
    assert subject.FORWARDS_PER_PAIR == 80
    assert subject.FORWARDS_PER_SHARD == 12000
    assert subject.TOTAL_FORWARD_BUDGET == 24000


def test_frozen_selection_and_confirmation_population():
    selection = subject.load_frozen_selection()
    assert selection["selected_dominant_candidate"] == "P5"
    assert selection["response_blind_control_plane"] == "P4"
    facts, rows = subject.load_confirmation_population()
    assert len(facts) == 300 and len(rows) == 1800
    assert facts[0]["pair_id"] == "xg1_fact_4201"
    assert facts[-1]["pair_id"] == "xg1_fact_4500"


def test_restored_round_trip_and_p4_matched_control():
    planes = _orthogonal_planes()
    h = torch.arange(1, subject.DIM + 1, dtype=torch.float64) / 100.0
    restored = subject.condition_correction(h, condition="dominant_restored", planes=planes)
    assert torch.equal(restored["correction"], torch.zeros_like(h))
    control = subject.condition_correction(h, condition="dominant_control", planes=planes)
    post = h + control["correction"]
    p5 = planes["P5"]
    assert abs(float(torch.dot(post, p5["plus"]))) <= subject.TOL
    assert abs(float(torch.dot(post, p5["minus"]))) <= subject.TOL
    assert control["dominant_matched_norm_abs_mismatch"] <= subject.TOL


def test_exactly_one_primary_p_value_positive_support():
    items = [_mock_item(i, 1.0 + (i % 7) * 0.01) for i in range(subject.PAIR_COUNT)]
    result = subject.infer_confirmation(items)
    primary = result["primary_test"]
    assert primary["p_value_count"] == 1
    assert primary["test"] == "one_sample_student_t"
    assert primary["tail"] == "greater"
    assert primary["mean"] > 0.0 and primary["p_raw"] < 0.05
    assert result["core_supported"] is True
    assert "H_RES" not in result and "Holm" not in str(result)


def test_negative_effect_not_supported_no_rescue():
    items = [_mock_item(i, -1.0 - (i % 11) * 0.01) for i in range(subject.PAIR_COUNT)]
    result = subject.infer_confirmation(items)
    assert result["core_supported"] is False
    assert result["rescue_performed"] is False
    assert result["additional_p_values_executed"] is False


def test_frozen_identities_and_state_width():
    assert subject.DISCOVERY_FREEZE_COMMIT == "e7804bcab88a50dd57c2edd86af88daeb90a56d5"
    assert subject.DISCOVERY_SELECTION_SHA256 == "deeb9a2514503a317fd184c77c963d6fc296dc0cf75ce4d180fd46d319595c09"
    assert subject.CONFIRMATION_SOURCE_SHA256 == "92c8c4c4c4ed61f33465b609a1466992214b9340822422659b04dafec2ddd6ea"
    assert subject.CONFIRMATION_ROWS_SHA256 == "d9641443f395fe622d37634a3e58e7fcd4e512897264c265249908104c8069ed"
    assert subject.CONFIRMATION_MANIFEST_SHA256 == "dd6f35639e69f6e87fd85eb8f7debb00fc97f803309ef71c178f790ade4ea2a4"
    state = torch.zeros(1, subject.geom.INTERMEDIATE_SIZE, subject.geom.STATE_SIZE, dtype=torch.float32)
    flat = subject.flatten_state(state)
    assert flat.shape == (65536,)
    assert math.isfinite(float(flat.sum()))
