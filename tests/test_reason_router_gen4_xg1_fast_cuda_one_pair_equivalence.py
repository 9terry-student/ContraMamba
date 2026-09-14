from __future__ import annotations

import inspect
from collections import Counter

import numpy as np
import pytest

from scripts import reason_router_gen4_xg1_fast_cuda_one_pair_equivalence as eq


def test_exact_xg1_branch_and_freezes():
    assert eq.EXPECTED_BRANCH == "gen4-k-xg1-cross-generator-replication"
    assert eq.ELIGIBILITY_FREEZE_COMMIT == (
        "ee7c2c10a0cbb4930b78eb0047ec0603b63e0d41"
    )
    assert eq.BACKEND_EQUIVALENCE_FREEZE_COMMIT == (
        "34a958fbc52300f0807b211b1d97f68f6fea7339"
    )


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
    with pytest.raises(eq.XG1EquivalenceError, match="BRANCH_MISMATCH:main"):
        eq.authenticate_repo(expected)


def test_xg1_pair_selection_is_fixed_and_outcome_blind():
    assert eq.XG1_PAIR_ID == "xg1_fact_001"
    assert eq.SOURCE_PAIR_COUNT == 300


def test_forward_budget_is_exactly_six_per_backend():
    assert eq.FORWARDS_PER_BACKEND == eq.base_eq.FORWARDS_PER_BACKEND == 6
    assert eq.TOTAL_MODEL_FORWARDS == eq.base_eq.TOTAL_MODEL_FORWARDS == 12
    assert eq.BRANCH_LABELS == (
        "baseline_tp",
        "baseline_tm",
        "baseline_rp",
        "baseline_rm",
        "alignment_tp",
        "alignment_tm",
    )


def test_backend_tolerances_are_reused_without_relaxation():
    assert eq.STATE_ATOL == eq.base_eq.STATE_ATOL == 1e-4
    assert eq.STATE_RTOL == eq.base_eq.STATE_RTOL == 1e-4
    assert eq.GEOMETRY_ATOL == eq.base_eq.GEOMETRY_ATOL == 1e-4
    assert eq.GEOMETRY_RTOL == eq.base_eq.GEOMETRY_RTOL == 1e-4
    assert eq.PE_ATOL == eq.base_eq.PE_ATOL == 1e-4


def test_backend_kernel_identity_is_reused_exactly():
    assert eq.backend.KERNELS_VERSION == "0.10.2"
    assert eq.backend.MAMBA_REV == (
        "c8ffc584c147878a6eb978ae0e8db4d116c93a8c"
    )
    assert eq.backend.CONV_REV == (
        "f2651e776f66069cdcf842840db637583def1223"
    )
    assert eq.backend.MAMBA_BINARY_SHA256 == (
        "dc4d76a6323b510e77cfb66b5aa7bb0086c8f5cba238002b9c20bc31ea706587"
    )
    assert eq.backend.CONV_BINARY_SHA256 == (
        "6b013d7b9a033bb9b0a2a714b26470e1aaba4af9bf1b3ec7442c2a53afb6b7b6"
    )


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
    for pair_index in range(1, 301):
        pair = f"xg1_fact_{pair_index:03d}"
        for cell in ("C0_SHAM", "C2_NAME"):
            assert (
                lookup[(pair, cell, "A_IDENTITY")][
                    "absolute_anchor_token_index"
                ]
                == lookup[(pair, cell, "A_NAME")][
                    "absolute_anchor_token_index"
                ]
            )


def test_xg1_input_loader_preserves_fixed_population_and_encoding():
    rows, encoded, event_rows = eq.load_xg1_inputs(None)
    assert len(rows) == 1800
    assert len(event_rows) == 1800
    assert eq._pair_order(rows)[0] == "xg1_fact_001"
    assert eq._pair_order(rows)[-1] == "xg1_fact_300"
    assert tuple(encoded["input_ids"].shape) == (1800, 128)
    assert tuple(encoded["attention_mask"].shape) == (1800, 128)


def _synthetic_item() -> dict[str, object]:
    item: dict[str, object] = {}
    for field in eq.EXACT_FIELDS:
        item[field] = f"x:{field}"
    item["schema_version"] = eq.base_eq.ITEM_SCHEMA
    item["source_pair_id"] = eq.XG1_PAIR_ID
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
            states = [
                np.asarray(
                    [float(index), float(offset)],
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


def test_reused_comparator_accepts_identical_six_forward_surface():
    item = _synthetic_item()
    records = _synthetic_records()
    result = eq.base_eq.compare_pair(
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


def test_reused_comparator_rejects_endpoint_difference_over_tolerance():
    cpu = _synthetic_item()
    gpu = dict(cpu)
    gpu["R_ALIGN"] = float(cpu["R_ALIGN"]) + 1e-3
    records = _synthetic_records()

    with pytest.raises(eq.base_eq.ProspectiveEquivalenceError):
        eq.base_eq.compare_pair(
            cpu,
            gpu,
            records,
            records,
        )


def test_runner_reuses_validated_six_forward_causal_computation():
    source = inspect.getsource(eq.run_one_pair)
    assert "base_eq.run_prospective_pair" in source
    assert "base_eq.compare_pair" in source
    assert "magnitude_delta" not in source
    assert "scientific_budget_forward_count" in source


def test_report_schema_is_xg1_equivalence_only():
    assert eq.REPORT_SCHEMA == (
        "gen4-k-xg1-fast-cuda-one-pair-equivalence-v1"
    )
    assert eq.TOTAL_MODEL_FORWARDS == 12


def test_report_contract_persists_no_endpoint_or_response_values():
    source = inspect.getsource(eq.run_one_pair)
    assert '"endpoint_values_persisted": False' in source
    assert '"xg1_response_values_persisted": False' in source
    assert '"scientific_conclusion": None' in source
