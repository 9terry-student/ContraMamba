from __future__ import annotations

import inspect

from scripts import (
    reason_router_gen4_mamba14b_adjacent_geometry_prepare_fast_cuda
    as m,
)


def test_adjacent_protocol_constants() -> None:
    m._configure_canonical_module()
    m.validate_protocol_constants()

    assert m.CANONICAL_TRIPLET == (33, 34, 35)
    assert m.ADJACENT_TRIPLET == (34, 35, 36)
    assert m.SOURCE_BLOCK == 34
    assert m.TARGET_RESIDUAL_LAYER == 35
    assert m.INTERVENTION_LAYER == 36
    assert m.LOCAL_LAYER_OFFSETS == (-2, -1, 0)
    assert m.TARGET_OFFSET == 2
    assert m.TOTAL_FORWARD_BUDGET == 2400
    assert m.XG1_FORWARD_BUDGET == 0
    assert m.SELECTED_CAUSAL_CANDIDATE == "P5"

    assert m.canonical.SOURCE_BLOCK == 34
    assert m.canonical.TARGET_RESIDUAL_LAYER == 35
    assert m.canonical.INTERVENTION_LAYER == 36


def test_response_blind_control_rule() -> None:
    # P5 is fixed and excluded. P4 has the largest remaining lambda.
    control = m.select_response_blind_control(
        [0.1, 0.2, 0.3, 0.9, 0.99]
    )
    assert control == "P4"


def test_response_blind_control_rejects_tie() -> None:
    try:
        m.select_response_blind_control(
            [0.1, 0.8, 0.8, 0.2, 0.99]
        )
    except m.AdjacentGeometryError as exc:
        assert "CONTROL_SELECTION_NOT_UNIQUE" in str(exc)
    else:
        raise AssertionError("tie must block control selection")


def test_worker_reconfigures_before_canonical_execution() -> None:
    source = inspect.getsource(m._adjacent_worker_run)
    assert "_configure_canonical_module()" in source
    assert "canonical.worker_run(**kwargs)" in source


def test_no_xg1_scientific_execution_in_geometry_wrapper() -> None:
    source = inspect.getsource(m)
    assert "XG1_FORWARD_BUDGET = 0" in source
    assert '"xg1_specificity_accessed"] = False' in source
    assert '"xg1_accessed": False' in source
    assert '"response_observed": False' in source
    assert '"plane_selection_performed": False' in source
    assert '"control_selection_uses_response": False' in source


def test_geometry_runner_budget_and_site_are_fixed() -> None:
    source = inspect.getsource(m.run_geometry_preparation)
    assert "canonical.write_output_bundle" in source
    assert "_rewrite_adjacent_bundle" in source
    assert "mamba14b_adjacent_geometry_" in source

    main_source = inspect.getsource(m.main)
    assert "SCIENTIFIC_MODEL_FORWARD_COUNT_THIS_RUN=2400" in main_source
    assert "ADJACENT_TRIPLET=34,35,36" in main_source
    assert "FIXED_CAUSAL_CANDIDATE=P5" in main_source
