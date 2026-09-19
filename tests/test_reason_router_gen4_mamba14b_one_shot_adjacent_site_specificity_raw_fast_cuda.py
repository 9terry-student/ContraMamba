from __future__ import annotations

import inspect

from scripts import (
    reason_router_gen4_mamba14b_one_shot_adjacent_site_specificity_raw_fast_cuda
    as m,
)


def test_protocol_constants() -> None:
    assert m.PAIR_FIRST == 5101
    assert m.PAIR_LAST == 5400
    assert m.PAIR_COUNT == 300
    assert m.PAIR_IDS[0] == "xg1_fact_5101"
    assert m.PAIR_IDS[-1] == "xg1_fact_5400"
    assert m.SELECTED_PLANE == "P5"
    assert m.CANONICAL_CONTROL_PLANE == "P4"
    assert m.ADJACENT_CONTROL_PLANE == "P4"
    assert m.EPS == 0.025
    assert m.TARGET_OFFSET == 2


def test_site_identities() -> None:
    assert m.CANONICAL_TRIPLET == (33, 34, 35)
    assert m.ADJACENT_TRIPLET == (34, 35, 36)
    assert m.CANONICAL_DIM == 829
    assert m.ADJACENT_DIM == 1205
    assert m.CANONICAL_STRONG_INDEX_SHA256 == (
        "ceaebe6046c747e09a169e8b5c0e59d68c9d82185dc5a9017373f497cc0a96c7"
    )
    assert m.ADJACENT_STRONG_INDEX_SHA256 == (
        "3c1d39df9b9b7a9acd3200b9b8cb31576bd6cc81fcd4518781f2cbbee3533141"
    )


def test_forward_budget_is_exactly_frozen() -> None:
    assert m.FORWARDS_PER_CONDITION == 40
    assert m.FORWARDS_PER_PAIR_PER_SITE == 80
    assert m.FORWARDS_PER_SITE == 24000
    assert m.TOTAL_FORWARD_BUDGET == 48000


def test_one_gpu_per_site_assignment() -> None:
    assert m.SITE_CONFIG["canonical"]["physical_device"] == 0
    assert m.SITE_CONFIG["adjacent"]["physical_device"] == 1


def test_configure_canonical_site() -> None:
    cfg = m.configure_site("canonical")
    assert tuple(cfg["triplet"]) == (33, 34, 35)
    assert m.geom.SOURCE_BLOCK == 33
    assert m.geom.TARGET_RESIDUAL_LAYER == 34
    assert m.geom.INTERVENTION_LAYER == 35
    assert m.core.DIM == 829
    assert m.core.PAIR_IDS == m.PAIR_IDS
    assert m.core.SELECTED_PLANE == "P5"
    assert m.core.CONTROL_PLANE == "P4"


def test_configure_adjacent_site() -> None:
    cfg = m.configure_site("adjacent")
    assert tuple(cfg["triplet"]) == (34, 35, 36)
    assert m.geom.SOURCE_BLOCK == 34
    assert m.geom.TARGET_RESIDUAL_LAYER == 35
    assert m.geom.INTERVENTION_LAYER == 36
    assert m.core.DIM == 1205
    assert m.core.SELECTED_PLANE == "P5"
    assert m.core.CONTROL_PLANE == "P4"


def test_endpoint_formula_is_paired_and_fixed() -> None:
    source = inspect.getsource(m.merge_site_outputs)
    assert 'd_can = float(canonical["D"])' in source
    assert 'd_adj = float(adjacent["D"])' in source
    assert "s_value = d_can - d_adj" in source
    assert '"D_CAN": d_can' in source
    assert '"D_ADJ": d_adj' in source
    assert '"S": s_value' in source


def test_raw_runner_has_no_inferential_test() -> None:
    source = inspect.getsource(m).lower()

    forbidden = (
        "import scipy",
        "from scipy",
        "ttest",
        "t.sf",
        "p_raw",
        "cohen_dz",
        "holm",
        "multipletests",
    )
    for token in forbidden:
        assert token not in source

    required = (
        '"performed": false',
        '"primary_p_value_computed": false',
        '"scientific_conclusion": none',
        '"second_adjacent_site_executed": false',
        '"rescue_performed": false',
    )
    for token in required:
        assert token in source


def test_no_geometry_rerun_or_response_based_control() -> None:
    source = inspect.getsource(m)
    assert "discovery.load_frozen_geometry()" in source
    assert "load_adjacent_geometry()" in source
    assert "geometry_prepare_fast_cuda.py" not in source
    assert '"response_based_control_selection": False' in source
    assert '"control_selection_uses_xg1_response": False' in source


def test_output_is_raw_response_not_scientific_verdict() -> None:
    assert m.RESULT_PASS == (
        "PASS_MAMBA14B_ONE_SHOT_ADJACENT_SITE_SPECIFICITY_RAW_RESPONSE"
    )
    assert "SUPPORTED" not in m.RESULT_PASS
    assert "NOT_ESTABLISHED" not in m.RESULT_PASS
