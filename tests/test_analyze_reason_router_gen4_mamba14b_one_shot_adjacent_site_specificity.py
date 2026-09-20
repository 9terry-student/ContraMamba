from __future__ import annotations

import inspect

import pytest

from scripts import (
    analyze_reason_router_gen4_mamba14b_one_shot_adjacent_site_specificity
    as a,
)


def test_frozen_identity_constants() -> None:
    assert a.RAW_FREEZE_COMMIT == (
        "d0f7c086cc68476eeee33e7db65c7f21409bf42e"
    )
    assert a.RAW_ITEMS_SHA256 == (
        "49b8087a7928f104a88fca3e2966264f1dbd21a68ccce947e08197fb239686a3"
    )
    assert a.RAW_SUMMARY_SHA256 == (
        "70f4170bb2e1e4847da90026ec5cad7cbc973a847af6d66d306cb701074dccd9"
    )
    assert a.RAW_MANIFEST_SHA256 == (
        "445307d593d9e9560d47e34491fa2875cbce09c103fdb523e8d4dc67c5e01150"
    )
    assert a.RAW_SUMS_SHA256 == (
        "46ed4636ca1c99375f647f1a23d312f1e780aca292127347939bee0062619e09"
    )


def test_primary_protocol_constants() -> None:
    assert a.PAIR_COUNT == 300
    assert a.PAIR_IDS[0] == "xg1_fact_5101"
    assert a.PAIR_IDS[-1] == "xg1_fact_5400"
    assert a.ALPHA == 0.05
    assert a.SUCCESS_LABEL == (
        "MAMBA14B_ONE_SHOT_ADJACENT_SITE_SPECIFICITY_SUPPORTED"
    )
    assert a.FAILURE_LABEL == (
        "MAMBA14B_ONE_SHOT_ADJACENT_SITE_SPECIFICITY_NOT_ESTABLISHED"
    )


def test_descriptive_basic() -> None:
    got = a.descriptive([1.0, 2.0, 3.0, 4.0])
    assert got["n"] == 4
    assert got["finite_count"] == 4
    assert got["nonfinite_count"] == 0
    assert got["mean"] == pytest.approx(2.5)
    assert got["fraction_positive"] == 1.0
    assert got["cohen_dz"] is not None
    assert got["ci95_t_low"] < got["mean"] < got["ci95_t_high"]


def test_exactly_one_primary_inference_call_in_analyze() -> None:
    source = inspect.getsource(a.analyze)
    assert source.count("primary_one_sample_greater(") == 1


def test_primary_function_has_exactly_one_pvalue_tail_call() -> None:
    source = inspect.getsource(a.primary_one_sample_greater).lower()
    assert source.count(".sf(") == 1
    assert "ttest_1samp" not in source
    assert "ttest_rel" not in source


def test_no_alternative_inferential_families() -> None:
    source = inspect.getsource(a).lower()
    forbidden = (
        "mannwhitney",
        "wilcoxon",
        "fisher_exact",
        "chi2",
        "permutation_test",
        "bootstrap",
        "multipletests",
        "holm",
        "bonferroni",
    )
    for token in forbidden:
        assert token not in source


def test_decision_rule_requires_both_gates() -> None:
    source = inspect.getsource(a.analyze)
    assert 'finite(d_can_stats["mean"], "D_CAN_MEAN") > 0.0' in source
    assert 'finite(s_stats["mean"], "S_MEAN") > 0.0' in source
    assert 'finite(primary["p_value"], "PRIMARY_P_VALUE") < ALPHA' in source
    assert "canonical_sign_gate\n        and paired_specificity_gate" in source


def test_canonical_sign_gate_adds_no_pvalue() -> None:
    source = inspect.getsource(a.analyze)
    assert '"adds_p_value": False' in source
    assert '"canonical_sign_gate_adds_p_value": False' in source


def test_forward_accounting_is_frozen() -> None:
    source = inspect.getsource(a.analyze)
    assert '"adjacent_geometry_model_forward_count": 2400' in source
    assert '"canonical_geometry_rerun_count": 0' in source
    assert '"canonical_response_model_forward_count": 24000' in source
    assert '"adjacent_response_model_forward_count": 24000' in source
    assert '"paired_response_model_forward_count": 48000' in source
    assert '"total_new_scientific_model_forward_count": 50400' in source
    assert '"analysis_model_forward_count": 0' in source


def test_analysis_has_no_model_runtime_dependency() -> None:
    source = inspect.getsource(a).lower()
    assert "import torch" not in source
    assert "cuda" not in source
    assert "model.forward" not in source
    assert "backward(" not in source


def test_claim_boundaries_and_no_rescue() -> None:
    source = inspect.getsource(a.analyze)
    assert '"second_adjacent_site_executed": False' in source
    assert '"layer_sweep_executed": False' in source
    assert '"token_sweep_executed": False' in source
    assert '"epsilon_sweep_executed": False' in source
    assert '"alternative_tail_executed": False' in source
    assert '"row_subset_rescue_executed": False' in source
    assert '"rescue_performed": False' in source
    assert '"experiments_1_to_3_rescued": False' in source


def test_adjacent_geometry_reporting_is_required() -> None:
    source = inspect.getsource(a.analyze)
    assert '"strong_dimension": 1205' in source
    assert '"lambda_plus_by_plane": lambda_plus' in source
    assert '"response_blind_control_plane": "P4"' in source
