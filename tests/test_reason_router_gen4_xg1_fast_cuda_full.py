from __future__ import annotations

import inspect
import math

import pytest

from scripts import reason_router_gen4_xg1_fast_cuda_full as full


def test_exact_xg1_gate_freeze_and_artifact_identity():
    assert full.GATE_FREEZE_COMMIT == "f41aa2abec3fa372699913c387082131b0447051"
    assert full.GATE_EXECUTION_HEAD == "6afb6191d1ef8c0aa39d4c5497a9250fda192110"
    assert (
        full.GATE_ARTIFACT_SHA256
        == "865124fd1804c2198d58ae01f3846319770fb3eda6482363b4036ba35f721877"
    )


def test_xg1_threshold_group_gate_and_forward_budget_are_frozen_unchanged():
    assert full.SOURCE_PAIR_COUNT == 300
    assert full.ALIGNMENT_SHIFT_THRESHOLD == 0.11228626366380845
    assert full.MIN_GROUP_SIZE == 30
    assert full.FAMILY_ALPHA == 0.05
    assert full.BASELINE_FORWARDS_PER_PAIR == 4
    assert full.ALIGNMENT_FORWARDS_PER_PAIR == 2
    assert full.FORWARDS_PER_PAIR == 6
    assert full.BASELINE_FORWARD_BUDGET == 1200
    assert full.ALIGNMENT_FORWARD_BUDGET == 600
    assert full.FULL_FORWARD_BUDGET == 1800


def test_threshold_classification_is_inclusive_only_for_large():
    t = full.ALIGNMENT_SHIFT_THRESHOLD
    assert full.classify_regime(t - 1e-12) == full.REGIME_SMALL
    assert full.classify_regime(t) == full.REGIME_LARGE
    assert full.classify_regime(t + 1e-12) == full.REGIME_LARGE


def _baseline_row(pair: str, shift: float) -> dict[str, object]:
    return {
        "schema_version": full.REGIME_ITEM_SCHEMA,
        "source_pair_id": pair,
        "target_C": 0.1,
        "reference_C": 0.1 + shift,
        "alignment_shift_abs": shift,
        "threshold": full.ALIGNMENT_SHIFT_THRESHOLD,
        "regime": full.classify_regime(shift),
        "classification_frozen_before_alignment": True,
    }


def test_regime_freeze_uses_all_xg1_pairs_and_contains_no_response_fields():
    pairs = tuple(f"xg1_fact_{i:03d}" for i in range(1, 301))
    rows = [
        _baseline_row(pair, 0.2 if index < 75 else 0.05)
        for index, pair in enumerate(pairs)
    ]
    frozen, counts = full.freeze_regimes(pairs, rows)
    assert counts == {full.REGIME_LARGE: 75, full.REGIME_SMALL: 225}
    assert len(frozen) == 300
    assert frozen[0]["source_pair_id"] == "xg1_fact_001"
    assert frozen[-1]["source_pair_id"] == "xg1_fact_300"
    assert all(row["schema_version"] == full.REGIME_ITEM_SCHEMA for row in frozen)
    assert all(full._RESPONSE_FIELDS.isdisjoint(row) for row in frozen)
    assert all(row["classification_frozen_before_alignment"] is True for row in frozen)


def test_regime_freeze_rejects_response_field_leak():
    pairs = tuple(f"xg1_fact_{i:03d}" for i in range(1, 301))
    rows = [_baseline_row(pair, 0.05) for pair in pairs]
    rows[0]["R_ALIGN"] = -0.1
    with pytest.raises(full.XG1FullError, match="FREEZE_RESPONSE_FIELD_LEAK"):
        full.freeze_regimes(pairs, rows)


def test_group_size_gate_is_exactly_30_per_group():
    assert full.group_size_passes({full.REGIME_LARGE: 30, full.REGIME_SMALL: 270})
    assert full.group_size_passes({full.REGIME_LARGE: 270, full.REGIME_SMALL: 30})
    assert not full.group_size_passes({full.REGIME_LARGE: 29, full.REGIME_SMALL: 271})
    assert not full.group_size_passes({full.REGIME_LARGE: 271, full.REGIME_SMALL: 29})


def _analysis_items(large_values, small_values):
    items = []
    for index, value in enumerate(large_values, 1):
        items.append(
            {
                "source_pair_id": f"xg1_fact_{index:03d}",
                "regime": full.REGIME_LARGE,
                "R_ALIGN": float(value),
            }
        )
    offset = len(items)
    for index, value in enumerate(small_values, 1):
        items.append(
            {
                "source_pair_id": f"xg1_fact_{offset + index:03d}",
                "regime": full.REGIME_SMALL,
                "R_ALIGN": float(value),
            }
        )
    return items


def test_analysis_maps_supported_result_to_cross_generator_replication():
    large = [-0.20 + 0.002 * i for i in range(30)]
    small = [0.05 + 0.0005 * (i % 20) for i in range(270)]
    result = full.analyze_responses(_analysis_items(large, small))
    tests = result["inferential_family"]["tests"]
    assert result["inferential_family"]["hypothesis_count"] == 2
    assert {row["hypothesis"] for row in tests} == {"H1", "H2"}
    assert result["xg1_prospective_regime_test"] == "SUPPORTED"
    assert result["scientific_conclusion"] == "CROSS_GENERATOR_ADVERSE_REGIME_REPLICATED"


def test_analysis_maps_failed_confirmation_to_not_established():
    large = [-0.01 + 0.001 * i for i in range(30)]
    small = [-0.10 + 0.0005 * (i % 20) for i in range(270)]
    result = full.analyze_responses(_analysis_items(large, small))
    assert result["scientific_conclusion"] == (
        "CROSS_GENERATOR_ADVERSE_REGIME_NOT_ESTABLISHED"
    )


def test_full_runner_uses_xg1_inputs_and_freezes_before_alignment():
    source = inspect.getsource(full.run_full)
    assert "xg1_eq.load_xg1_inputs" in source
    assert "load_prospective_inputs" not in source
    freeze_pos = source.index("_write_regime_freeze")
    alignment_pos = source.index("_run_alignment_pair")
    assert freeze_pos < alignment_pos
    assert "budget.used == BASELINE_FORWARD_BUDGET" in source
    assert "budget.assert_exact()" in source


def test_full_runner_has_no_magnitude_path_or_extra_inferential_family():
    run_source = inspect.getsource(full.run_full).lower()
    analysis_source = inspect.getsource(full.analyze_responses)
    base_analysis_source = inspect.getsource(full.base.analyze_responses)
    assert "magnitude" not in run_source
    assert '"H1"' in base_analysis_source
    assert '"H2"' in base_analysis_source
    assert '"H3"' not in base_analysis_source
    assert "base.analyze_responses" in analysis_source


def test_output_surface_is_xg1_specific():
    assert full.REGIME_ITEM_SCHEMA == "gen4-k-xg1-cross-generator-regime-item-v1"
    assert full.REGIME_FREEZE_SCHEMA == "gen4-k-xg1-cross-generator-regime-freeze-v1"
    assert full.ITEM_SCHEMA == "gen4-k-xg1-cross-generator-full-item-v1"
    assert full.SUMMARY_SCHEMA == "gen4-k-xg1-cross-generator-full-summary-v1"
    assert full.MANIFEST_SCHEMA == "gen4-k-xg1-cross-generator-full-manifest-v1"
    assert full.REGIME_FILE == "regime_manifest.jsonl"
    assert full.REGIME_FREEZE_FILE == "regime_freeze.json"
    assert full.ITEM_FILE == "item_metrics.jsonl"
    assert full.SUMMARY_FILE == "summary.json"
    assert full.MANIFEST_FILE == "manifest.json"
    assert full.CHECKSUM_FILE == "SHA256SUMS.txt"


def test_gate_validation_requires_frozen_xg1_report(monkeypatch, tmp_path):
    report = {
        "result": "PASS_XG1_FAST_CUDA_ONE_PAIR_EQUIVALENCE",
        "execution_head": full.GATE_EXECUTION_HEAD,
        "source_pair_id": "xg1_fact_001",
        "cpu_model_forward_count": 6,
        "gpu_model_forward_count": 6,
        "total_model_forward_count": 12,
        "scientific_budget_forward_count": 0,
        "scientific_conclusion": None,
        "representative_checkpoint_sha256":
            full.base.extraction.REPRESENTATIVE_CHECKPOINT_SHA256,
        "xg1_rows_sha256": full.xg1_eq.eligibility.EXPECTED_ROWS_SHA256,
        "eligibility_anchor_manifest_sha256":
            full.xg1_eq.ELIGIBILITY_ANCHOR_MANIFEST_SHA256,
        "state_atol": 1e-4,
        "state_rtol": 1e-4,
        "geometry_atol": 1e-4,
        "geometry_rtol": 1e-4,
        "pe_atol": 1e-4,
        "max_state_abs_diff": 5e-5,
        "max_geometry_abs_diff": 2e-6,
        "max_pe_abs_diff": 2e-6,
    }
    artifact = tmp_path / "equivalence_report.json"
    artifact.write_text(__import__("json").dumps(report), encoding="utf-8")
    monkeypatch.setattr(full, "ROOT", tmp_path)
    monkeypatch.setattr(full, "GATE_ARTIFACT_REL", artifact.relative_to(tmp_path))
    monkeypatch.setattr(full, "GATE_ARTIFACT_SHA256", full.base.sha256_file(artifact))
    assert full.validate_gate_artifact()["result"].startswith("PASS_XG1_")


def test_authenticate_repo_accepts_detached_exact_head_and_rejects_wrong_branch(monkeypatch):
    expected = "f" * 40

    def detached_git(*args: str) -> str:
        if args == ("branch", "--show-current"):
            return ""
        if args == ("rev-parse", "HEAD"):
            return expected
        if args == ("status", "--porcelain"):
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(full, "git", detached_git)
    monkeypatch.setattr(full.subprocess, "call", lambda *args, **kwargs: 0)
    monkeypatch.setattr(full, "validate_gate_artifact", lambda: {})
    full.authenticate_repo(expected)

    def wrong_git(*args: str) -> str:
        if args == ("branch", "--show-current"):
            return "main"
        if args == ("rev-parse", "HEAD"):
            return expected
        if args == ("status", "--porcelain"):
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(full, "git", wrong_git)
    with pytest.raises(full.XG1FullError, match="BRANCH_MISMATCH:main"):
        full.authenticate_repo(expected)
