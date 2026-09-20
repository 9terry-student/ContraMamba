from __future__ import annotations

from pathlib import Path

import pytest

from scripts import (
    reason_router_gen4_precursor_v4_analytic_stage_a_fast_cuda
    as v4,
)


def test_protocol_and_accounting() -> None:
    v4.validate_protocol()
    assert v4.N == 800
    assert v4.OBSERVATION_OFFSETS == (-4, -3, -2, -1)
    assert v4.OBSERVATION_PREFIX_LENGTHS == (5, 6, 7, 8)
    assert v4.FULL_MODEL_FORWARDS_BY_EMITTED_LABEL == {
        "REFUTE": 12,
        "SUPPORT": 11,
    }
    assert v4.MIN_FULL_MODEL_FORWARDS_PER_ROW == 11
    assert v4.MAX_FULL_MODEL_FORWARDS_PER_ROW == 12
    assert v4.LOCAL_VJPS_PER_ROW == 4
    assert v4.SCIENTIFIC_FORWARD_BUDGET_MIN == 8800
    assert v4.SCIENTIFIC_FORWARD_BUDGET_MAX == 9600
    assert v4.SCIENTIFIC_LOCAL_VJP_BUDGET == 3200
    assert v4.SHARDS[0]["forward_budget_min"] == 4400
    assert v4.SHARDS[0]["forward_budget_max"] == 4800
    assert v4.SHARDS[1]["forward_budget_min"] == 4400
    assert v4.SHARDS[1]["forward_budget_max"] == 4800
    assert v4.SHARDS[0]["local_vjp_budget"] == 1600
    assert v4.SHARDS[1]["local_vjp_budget"] == 1600


def test_forward_accounting_is_derived_from_frozen_grammar() -> None:
    assert len(v4.forced.FORCED_TOKEN_IDS["REFUTE"]) == 12
    assert len(v4.forced.FORCED_TOKEN_IDS["SUPPORT"]) == 11
    assert v4.FULL_MODEL_FORWARDS_BY_EMITTED_LABEL == {
        label: len(v4.forced.FORCED_TOKEN_IDS[label])
        for label in v4.FORCED_CLASS_ORDER
    }


def test_equivalence_freeze_is_exact_pass() -> None:
    report = v4.validate_equivalence_freeze()
    assert report["result"] == (
        "PASS_PRECURSOR_V4_ANALYTIC_VJP_CPU_CUDA_EQUIVALENCE"
    )
    assert report["comparison"]["result"] == "PASS"
    assert report["measurement"]["epsilon"] is None
    assert (
        report["boundary"]["fresh_n800_scientific_cohort_accessed"]
        is False
    )
    assert report["boundary"]["scientific_inference_executed"] is False
    assert report["boundary"]["p_value_count_added"] == 0


def test_frozen_response_free_cohort_is_still_exact() -> None:
    rows, manifest = v4.cohort_source.validate_frozen_cohort()
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


def test_item_endpoint_is_equal_weight_mean() -> None:
    measurements = [
        {"relative_offset": -4, "d_t": 1.0},
        {"relative_offset": -3, "d_t": 2.0},
        {"relative_offset": -2, "d_t": -1.0},
        {"relative_offset": -1, "d_t": 0.0},
    ]
    assert v4.summarize_item_endpoint(measurements) == pytest.approx(0.5)


def test_item_endpoint_rejects_offset_reordering() -> None:
    measurements = [
        {"relative_offset": -3, "d_t": 1.0},
        {"relative_offset": -4, "d_t": 2.0},
        {"relative_offset": -2, "d_t": -1.0},
        {"relative_offset": -1, "d_t": 0.0},
    ]
    with pytest.raises(v4.PrecursorV4StageAError):
        v4.summarize_item_endpoint(measurements)


def test_runner_source_has_no_finite_difference_or_statistics() -> None:
    source = Path(v4.__file__).read_text(encoding="utf-8")
    assert "EPSILON =" not in source
    assert "epsilon_sign" not in source
    assert "central_difference" not in source
    assert "ttest" not in source.lower()
    assert "welch" not in source.lower()
    assert '"epsilon": None' in source
    assert '"p_value_count_added": 0' in source


def test_observation_forward_is_reused_for_decoding() -> None:
    source = Path(v4.__file__).read_text(encoding="utf-8")
    assert (
        '"observation_vjp_forward_reuses_decoding_forward": True'
        in source
    )
    assert "FULL_MODEL_FORWARDS_BY_EMITTED_LABEL" in source
    assert "MIN_FULL_MODEL_FORWARDS_PER_ROW" in source
    assert "MAX_FULL_MODEL_FORWARDS_PER_ROW" in source
    assert "LOCAL_VJPS_PER_ROW = 4" in source


def test_raw_runner_does_not_create_parameter_gradients() -> None:
    source = Path(v4.__file__).read_text(encoding="utf-8")
    assert "torch.autograd.grad(" in source
    assert "not any(parameter.requires_grad" in source
    assert '"parameter_gradient_created": False' in source
    assert '"parameter_update_executed": False' in source
