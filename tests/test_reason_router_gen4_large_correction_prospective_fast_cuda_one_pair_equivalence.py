from __future__ import annotations

import inspect
from collections import Counter

import numpy as np
import pytest

from scripts import reason_router_gen4_large_correction_prospective_fast_cuda_one_pair_equivalence as eq


def test_exact_prospective_branch_and_prerequisite_freeze():
    assert eq.EXPECTED_BRANCH == "gen4-k-large-correction-prospective-validation"
    assert eq.PREREQUISITE_FREEZE_COMMIT == "3d7ca42005a1f5befae26acfa4e2fd3296768ce7"
    assert eq.ELIGIBILITY_IMPLEMENTATION_COMMIT == "2de39772dd641499ee6fef26fa57d9d0396f0e21"




def test_authenticate_repo_accepts_cm_kaggle_detached_exact_head(monkeypatch):
    expected = "f" * 40

    def fake_git(*args: str) -> str:
        if args == ("branch", "--show-current"):
            return ""
        if args == ("rev-parse", "HEAD"):
            return expected
        if args == ("status", "--porcelain"):
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(eq, "git", fake_git)
    monkeypatch.setattr(eq.subprocess, "call", lambda *args, **kwargs: 0)

    eq.authenticate_repo(expected)


def test_authenticate_repo_rejects_wrong_named_branch(monkeypatch):
    expected = "f" * 40

    def fake_git(*args: str) -> str:
        if args == ("branch", "--show-current"):
            return "main"
        if args == ("rev-parse", "HEAD"):
            return expected
        if args == ("status", "--porcelain"):
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(eq, "git", fake_git)

    with pytest.raises(eq.ProspectiveEquivalenceError, match="BRANCH_MISMATCH:main"):
        eq.authenticate_repo(expected)


def test_frozen_holdout_pair_selection_is_outcome_blind_first_pair():
    assert eq.PROSPECTIVE_PAIR_ID == "generated_fact_301"
    assert eq.SOURCE_PAIR_COUNT == 300


def test_forward_budget_is_exactly_six_per_backend():
    assert eq.FORWARDS_PER_BACKEND == 6
    assert eq.TOTAL_MODEL_FORWARDS == 12
    assert eq.BRANCH_LABELS == (
        "baseline_tp",
        "baseline_tm",
        "baseline_rp",
        "baseline_rm",
        "alignment_tp",
        "alignment_tm",
    )


def test_backend_tolerances_are_not_relaxed():
    assert eq.STATE_ATOL == eq.backend.STATE_ATOL == 1e-4
    assert eq.STATE_RTOL == eq.backend.STATE_RTOL == 1e-4
    assert eq.GEOMETRY_ATOL == eq.backend.GEOMETRY_ATOL == 1e-4
    assert eq.GEOMETRY_RTOL == eq.backend.GEOMETRY_RTOL == 1e-4
    assert eq.PE_ATOL == eq.backend.PE_ATOL == 1e-4


def test_backend_kernel_identity_is_reused_exactly():
    assert eq.backend.KERNELS_VERSION == "0.10.2"
    assert eq.backend.MAMBA_REV == "c8ffc584c147878a6eb978ae0e8db4d116c93a8c"
    assert eq.backend.CONV_REV == "f2651e776f66069cdcf842840db637583def1223"
    assert eq.backend.MAMBA_BINARY_SHA256 == (
        "dc4d76a6323b510e77cfb66b5aa7bb0086c8f5cba238002b9c20bc31ea706587"
    )
    assert eq.backend.CONV_BINARY_SHA256 == (
        "6b013d7b9a033bb9b0a2a714b26470e1aaba4af9bf1b3ec7442c2a53afb6b7b6"
    )


def test_prospective_reductions_match_frozen_r_align_definition():
    result = eq._prospective_reductions(0.8, 0.3, 0.7, 0.4)
    assert result["delta_baseline"] == pytest.approx(0.5)
    assert result["delta_alignment"] == pytest.approx(0.3)
    assert result["R_ALIGN"] == pytest.approx(-0.2)


def test_scalar_gate_accepts_within_tolerance_and_fails_closed():
    assert eq._compare_scalar(
        1.00005,
        1.0,
        atol=1e-4,
        label="ok",
    ) <= 1e-4

    with pytest.raises(eq.ProspectiveEquivalenceError):
        eq._compare_scalar(
            1.001,
            1.0,
            atol=1e-4,
            label="bad",
        )


def test_runner_has_no_magnitude_intervention_path():
    source = inspect.getsource(eq.run_prospective_pair)
    assert "magnitude_delta" not in source
    assert "magnitude" not in source.lower()
    assert "alignment_delta" in source


def test_frozen_anchor_manifest_identity_and_counts():
    rows = eq.load_frozen_anchor_manifest()
    assert len(rows) == 1800
    assert Counter(row["anchor_name"] for row in rows) == Counter({
        "A_IDENTITY": 1200,
        "A_NAME": 600,
    })
    assert all(row["post4_eligible"] is True for row in rows)
    assert all(row["exclusion_code"] is None for row in rows)


def test_frozen_anchor_manifest_has_exact_target_identity_name_equality():
    rows = eq.load_frozen_anchor_manifest()
    lookup = {
        (
            row["source_pair_id"],
            row["contrast_cell_id"],
            row["anchor_name"],
        ): row
        for row in rows
    }
    for pair_index in range(301, 601):
        pair = f"generated_fact_{pair_index}"
        for cell in ("C0_SHAM", "C2_NAME"):
            assert (
                lookup[(pair, cell, "A_IDENTITY")]["absolute_anchor_token_index"]
                == lookup[(pair, cell, "A_NAME")]["absolute_anchor_token_index"]
            )


def test_prospective_input_loader_reconstructs_fixed_population_and_encoding():
    rows, encoded, event_rows = eq.load_prospective_inputs(None)
    assert len(rows) == 1800
    assert len(event_rows) == 1800
    assert eq._pair_order(rows)[0] == "generated_fact_301"
    assert eq._pair_order(rows)[-1] == "generated_fact_600"
    assert tuple(encoded["input_ids"].shape) == (1800, 128)
    assert tuple(encoded["attention_mask"].shape) == (1800, 128)


def _synthetic_item() -> dict[str, object]:
    item: dict[str, object] = {}
    for field in eq.EXACT_FIELDS:
        item[field] = f"x:{field}"
    item["schema_version"] = eq.ITEM_SCHEMA
    item["source_pair_id"] = eq.PROSPECTIVE_PAIR_ID
    for field in eq.GEOMETRY_FIELDS:
        item[field] = 0.25
    for field in eq.PE_FIELDS:
        item[field] = 0.5
    return item


def _synthetic_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index, label in enumerate(eq.BRANCH_LABELS):
        states = None
        if label in {
            "baseline_tp",
            "baseline_tm",
            "alignment_tp",
            "alignment_tm",
        }:
            states = [np.asarray([float(index), float(offset)], dtype=np.float32) for offset in range(5)]
        records.append({
            "label": label,
            "anchor": 10,
            "target_abs": 12,
            "states": states,
        })
    return records


def test_compare_pair_accepts_identical_six_forward_surface():
    item = _synthetic_item()
    records = _synthetic_records()
    result = eq.compare_pair(item, dict(item), records, records)
    assert result == {
        "max_state_abs_diff": 0.0,
        "max_geometry_abs_diff": 0.0,
        "max_pe_abs_diff": 0.0,
    }


def test_compare_pair_rejects_endpoint_difference_over_tolerance():
    cpu = _synthetic_item()
    gpu = dict(cpu)
    gpu["R_ALIGN"] = float(cpu["R_ALIGN"]) + 1e-3
    records = _synthetic_records()
    with pytest.raises(eq.ProspectiveEquivalenceError):
        eq.compare_pair(cpu, gpu, records, records)


def test_report_schema_is_equivalence_only_not_scientific_result_schema():
    assert "equivalence" in eq.REPORT_SCHEMA
    assert "prospective" in eq.REPORT_SCHEMA
    assert eq.TOTAL_MODEL_FORWARDS == 12
