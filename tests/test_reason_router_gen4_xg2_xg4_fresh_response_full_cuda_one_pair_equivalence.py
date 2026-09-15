from __future__ import annotations

import inspect
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from scripts import reason_router_gen4_xg2_xg4_fresh_response_full_cuda_one_pair_equivalence as eq


def test_freeze_chain_and_pair_selection():
    assert (
        eq.DESIGN_FREEZE_COMMIT
        == "2074f52d39bf0fca6d63248016ce47c54bff5e06"
    )
    assert (
        eq.STRUCTURAL_FREEZE_COMMIT
        == "4bda8dd4b46d56ffe9bb37cffc91f8506026ca69"
    )
    assert (
        eq.ELIGIBILITY_IMPLEMENTATION_COMMIT
        == "18189ac9532250faec5d3c6642b70e5636b9a0b8"
    )
    assert (
        eq.ELIGIBILITY_FREEZE_COMMIT
        == "309c82ebc9aa6da013633ee8171f73ac8a191ba7"
    )

    assert eq.FAMILY_CONFIG["xg2"]["pair_id"] == "xg2_fact_301"
    assert eq.FAMILY_CONFIG["xg4"]["pair_id"] == "xg4_fact_301"


def test_forward_budget_is_exactly_cpu6_gpu6():
    assert eq.FORWARDS_PER_BACKEND == 6
    assert eq.TOTAL_MODEL_FORWARDS == 12

    assert eq.inherited.BRANCH_LABELS == (
        "baseline_tp",
        "baseline_tm",
        "baseline_rp",
        "baseline_rm",
        "alignment_tp",
        "alignment_tm",
    )


def test_tolerances_are_frozen_at_one_e_minus_four():
    assert eq.STATE_ATOL == 1e-4
    assert eq.STATE_RTOL == 1e-4
    assert eq.GEOMETRY_ATOL == 1e-4
    assert eq.GEOMETRY_RTOL == 1e-4
    assert eq.PE_ATOL == 1e-4


def test_inherited_full_response_has_alignment_and_no_magnitude():
    source = inspect.getsource(
        eq.inherited.run_prospective_pair
    ).lower()

    assert "alignment_delta" in source
    assert "alignment" in source
    assert "magnitude" not in source


def test_inherited_reduction_is_exact_r_align_definition():
    out = eq.inherited._prospective_reductions(
        0.8,
        0.3,
        0.7,
        0.4,
    )

    assert out["delta_baseline"] == pytest.approx(0.5)
    assert out["delta_alignment"] == pytest.approx(0.3)
    assert out["R_ALIGN"] == pytest.approx(-0.2)


def test_pinned_runtime_blob_contract():
    root = Path(__file__).resolve().parents[1]

    for path, expected in eq.PINNED_BLOBS.items():
        observed = eq.git(
            root,
            "rev-parse",
            f"HEAD:{path}",
        )
        assert observed == expected


@pytest.mark.parametrize(
    "family",
    ("xg2", "xg4"),
)
def test_frozen_anchor_artifact_is_exact(
    family,
):
    rows = eq.load_frozen_anchor_manifest(
        family
    )

    assert len(rows) == 1800

    assert Counter(
        row["anchor_name"]
        for row in rows
    ) == Counter({
        "A_IDENTITY": 1200,
        "A_NAME": 600,
    })

    assert all(
        row["post4_eligible"] is True
        for row in rows
    )
    assert all(
        row["exclusion_code"] is None
        for row in rows
    )


@pytest.mark.parametrize(
    "family",
    ("xg2", "xg4"),
)
def test_fresh_input_loader_reconstructs_301_600_population(
    family,
):
    rows, encoded, events = (
        eq.load_family_inputs(
            family,
            None,
        )
    )

    assert len(rows) == 1800
    assert len(events) == 1800

    pairs = eq._pair_order(
        family,
        rows,
    )

    assert pairs[0] == f"{family}_fact_301"
    assert pairs[-1] == f"{family}_fact_600"

    assert tuple(
        encoded["input_ids"].shape
    ) == (1800, 128)

    assert tuple(
        encoded["attention_mask"].shape
    ) == (1800, 128)


def _synthetic_item():
    item = {}

    for field in eq.inherited.EXACT_FIELDS:
        item[field] = f"x:{field}"

    item["schema_version"] = (
        eq.inherited.ITEM_SCHEMA
    )

    for field in eq.inherited.GEOMETRY_FIELDS:
        item[field] = 0.25

    for field in eq.inherited.PE_FIELDS:
        item[field] = 0.5

    return item


def _synthetic_records():
    records = []

    for index, label in enumerate(
        eq.inherited.BRANCH_LABELS
    ):
        states = None

        if label in {
            "baseline_tp",
            "baseline_tm",
            "alignment_tp",
            "alignment_tm",
        }:
            states = [
                np.asarray(
                    [
                        float(index),
                        float(offset),
                    ],
                    dtype=np.float32,
                )
                for offset in range(5)
            ]

        records.append({
            "label": label,
            "anchor": 10,
            "target_abs": 12,
            "states": states,
        })

    return records


def test_inherited_compare_accepts_exact_six_forward_surface():
    item = _synthetic_item()
    records = _synthetic_records()

    result = eq.inherited.compare_pair(
        item,
        dict(item),
        records,
        records,
    )

    assert result == {
        "max_state_abs_diff": 0.0,
        "max_geometry_abs_diff": 0.0,
        "max_pe_abs_diff": 0.0,
    }


def test_compare_fails_closed_above_pe_tolerance():
    cpu = _synthetic_item()
    gpu = dict(cpu)

    gpu["R_ALIGN"] = (
        float(cpu["R_ALIGN"])
        + 1e-3
    )

    records = _synthetic_records()

    with pytest.raises(
        eq.inherited.ProspectiveEquivalenceError
    ):
        eq.inherited.compare_pair(
            cpu,
            gpu,
            records,
            records,
        )


def test_equivalence_report_boundary_is_not_scientific_execution():
    assert "equivalence" in eq.REPORT_SCHEMA
    assert eq.TOTAL_MODEL_FORWARDS == 12

    source = inspect.getsource(
        eq.run_one_pair
    )

    assert (
        '"scientific_budget_forward_count":'
        in source
    )
    assert (
        '"scientific_conclusion":'
        in source
    )
    assert (
        '"endpoint_values_persisted":'
        in source
    )
    assert (
        '"prospective_response_values_persisted":'
        in source
    )


def test_no_training_or_backward_path_added():
    source = inspect.getsource(eq)

    assert ".backward(" not in source
    assert ".train(" not in source
    assert "optimizer.step" not in source
