from __future__ import annotations

import inspect
from collections import Counter

import numpy as np
import pytest

from scripts import reason_router_gen4_generator_family_prevalence_fast_cuda_one_pair_equivalence as eq


def test_exact_scope_and_freezes():
    assert eq.EXPECTED_BRANCH == "gen4-k-xg1-cross-generator-replication"
    assert eq.ELIGIBILITY_FREEZE_COMMIT == (
        "0af44566eaadc324a6aaf5cec19f3972c8e371ef"
    )
    assert eq.ELIGIBILITY_EXECUTION_HEAD == (
        "8df2a5e2cb72f8f4eed33c4996dfba594a1f9687"
    )
    assert set(eq.FAMILY_CONFIG) == {"xg2", "xg4"}
    assert "xg3" not in eq.FAMILY_CONFIG


def test_pair_selection_is_fixed_and_outcome_blind():
    assert eq.FAMILY_CONFIG["xg2"]["pair_id"] == "xg2_fact_001"
    assert eq.FAMILY_CONFIG["xg4"]["pair_id"] == "xg4_fact_001"
    assert eq.SOURCE_PAIR_COUNT == 300


def test_backend_gate_is_baseline_only_four_forwards_per_backend():
    assert eq.FORWARDS_PER_BACKEND == 4
    assert eq.TOTAL_MODEL_FORWARDS == 8
    assert eq.BASELINE_LABELS == (
        "baseline_tp",
        "baseline_tm",
        "baseline_rp",
        "baseline_rm",
    )
    assert all("alignment" not in label for label in eq.BASELINE_LABELS)


def test_backend_tolerances_are_reused_without_relaxation():
    assert eq.STATE_ATOL == eq.xg1_eq.STATE_ATOL == 1e-4
    assert eq.STATE_RTOL == eq.xg1_eq.STATE_RTOL == 1e-4
    assert eq.GEOMETRY_ATOL == eq.xg1_eq.GEOMETRY_ATOL == 1e-4
    assert eq.GEOMETRY_RTOL == eq.xg1_eq.GEOMETRY_RTOL == 1e-4


@pytest.mark.parametrize("family", ("xg2", "xg4"))
def test_frozen_eligibility_manifest_identity_counts_and_target_equality(family):
    rows = eq.load_frozen_anchor_manifest(family)
    assert len(rows) == 1800
    assert Counter(row["anchor_name"] for row in rows) == Counter({
        "A_IDENTITY": 1200,
        "A_NAME": 600,
    })
    assert all(row["post4_eligible"] is True for row in rows)
    assert all(row["exclusion_code"] is None for row in rows)

    lookup = {
        (
            row["source_pair_id"],
            row["contrast_cell_id"],
            row["anchor_name"],
        ): row
        for row in rows
    }
    for index in range(1, 301):
        pair = f"{family}_fact_{index:03d}"
        for cell in ("C0_SHAM", "C2_NAME"):
            assert (
                lookup[(pair, cell, "A_IDENTITY")][
                    "absolute_anchor_token_index"
                ]
                == lookup[(pair, cell, "A_NAME")][
                    "absolute_anchor_token_index"
                ]
            )


def test_authenticate_repo_accepts_detached_exact_head(monkeypatch):
    expected = "f" * 40

    def fake_git(root, *args):
        if args == ("branch", "--show-current"):
            return ""
        if args == ("rev-parse", "HEAD"):
            return expected
        if args == ("status", "--porcelain"):
            return ""
        if args == ("rev-parse", f"HEAD:{eq.ELIGIBILITY_GATE_PATH}"):
            return eq.ELIGIBILITY_GATE_BLOB
        if args == ("rev-parse", f"HEAD:{eq.COHORT_BUILDER_PATH}"):
            return eq.COHORT_BUILDER_BLOB
        raise AssertionError(args)

    monkeypatch.setattr(eq, "git", fake_git)
    monkeypatch.setattr(eq.subprocess, "call", lambda *args, **kwargs: 0)
    eq.authenticate_repo(expected)


def _synthetic_item(family: str = "xg2") -> dict[str, object]:
    item: dict[str, object] = {
        "schema_version":
            "gen4-generator-family-prevalence-baseline-equivalence-item-v1",
        "family_key": family,
        "source_pair_id": f"{family}_fact_001",
        "source_block": 15,
        "target_residual_layer": 16,
        "relative_coordinate": 4,
        "target_plus_cell": "C2_NAME",
        "target_minus_cell": "C0_SHAM",
        "reference_plus_cell": "C5_TITLE_NAME",
        "reference_minus_cell": "C1_TITLE",
        "anchor_name": "A_IDENTITY",
        "target_plus_anchor": 10,
        "target_minus_anchor": 10,
        "reference_plus_anchor": 10,
        "reference_minus_anchor": 10,
        "target_plus_geometry_token": 14,
        "target_minus_geometry_token": 14,
        "reference_plus_geometry_token": 14,
        "reference_minus_geometry_token": 14,
    }
    for field in eq.GEOMETRY_FIELDS:
        item[field] = 0.25
    return item


def _synthetic_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index, label in enumerate(eq.BASELINE_LABELS):
        states = [
            np.asarray([float(index), float(offset)], dtype=np.float32)
            for offset in range(5)
        ]
        records.append({
            "label": label,
            "anchor": 10,
            "target_abs": 14,
            "states": states,
        })
    return records


def test_baseline_comparator_accepts_identical_surface():
    item = _synthetic_item()
    records = _synthetic_records()
    result = eq.compare_baseline_pair(
        item,
        dict(item),
        records,
        records,
    )
    assert result == {
        "max_state_abs_diff": 0.0,
        "max_geometry_abs_diff": 0.0,
    }


def test_baseline_comparator_rejects_geometry_difference_over_tolerance():
    cpu = _synthetic_item()
    gpu = dict(cpu)
    gpu["target_C"] = float(cpu["target_C"]) + 1e-3
    records = _synthetic_records()

    with pytest.raises(
        eq.PrevalenceBaselineEquivalenceError,
        match="EQUIVALENCE_FAILURE",
    ):
        eq.compare_baseline_pair(cpu, gpu, records, records)


def test_baseline_comparator_rejects_state_difference_over_tolerance():
    item = _synthetic_item()
    cpu_records = _synthetic_records()
    gpu_records = _synthetic_records()
    gpu_records[0] = dict(gpu_records[0])
    changed = list(gpu_records[0]["states"])
    changed[0] = np.asarray([1e-2, 0.0], dtype=np.float32)
    gpu_records[0]["states"] = changed

    with pytest.raises(
        eq.PrevalenceBaselineEquivalenceError,
        match="STATE_EQUIVALENCE_FAILURE",
    ):
        eq.compare_baseline_pair(
            item,
            dict(item),
            cpu_records,
            gpu_records,
        )


def test_runner_contains_no_intervention_or_response_execution():
    source = inspect.getsource(eq.run_one_pair)
    baseline_source = inspect.getsource(eq.run_baseline_geometry_pair)

    forbidden = (
        "run_prospective_pair",
        "_run_alignment_pair",
        "alignment_delta(",
        "magnitude_delta",
        "R_ALIGN",
        "delta_alignment",
        "path_efficiency",
    )
    for token in forbidden:
        assert token not in source
        assert token not in baseline_source

    assert "ForwardBudget(FORWARDS_PER_BACKEND)" in source
    assert "capture_states=True" in baseline_source


def test_report_contract_persists_no_geometry_or_response_values():
    source = inspect.getsource(eq.run_one_pair)
    assert '"baseline_only": True' in source
    assert '"alignment_intervention_executed": False' in source
    assert '"magnitude_intervention_executed": False' in source
    assert '"response_endpoints_computed": False' in source
    assert '"geometry_values_persisted": False' in source
    assert '"alignment_shift_value_persisted": False' in source
    assert '"response_values_persisted": False' in source
    assert '"scientific_conclusion": None' in source


def test_cli_does_not_offer_xg3():
    args = eq.parse_args([
        "--family", "xg2",
        "--expected-head", "f" * 40,
        "--model-snapshot", "m",
        "--tokenizer-snapshot", "t",
        "--checkpoint", "c",
        "--output", "o",
    ])
    assert args.family == "xg2"

    with pytest.raises(SystemExit):
        eq.parse_args([
            "--family", "xg3",
            "--expected-head", "f" * 40,
            "--model-snapshot", "m",
            "--tokenizer-snapshot", "t",
            "--checkpoint", "c",
            "--output", "o",
        ])
