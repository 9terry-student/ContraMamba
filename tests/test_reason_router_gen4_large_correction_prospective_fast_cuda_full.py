from __future__ import annotations

import inspect
import math

import pytest

from scripts import reason_router_gen4_large_correction_prospective_fast_cuda_full as full


def test_exact_gate_freeze_and_artifact_identity():
    assert full.GATE_FREEZE_COMMIT == "34a958fbc52300f0807b211b1d97f68f6fea7339"
    assert full.GATE_EXECUTION_HEAD == "8b4a83314b4a69d8004a68c7ae4a848f6e4ef401"
    assert full.GATE_ARTIFACT_SHA256 == "34392044015cbc9217bbb90535e499be64c13f7ed068d566b9bec2e4a2448061"


def test_exact_prospective_threshold_group_gate_and_forward_budget():
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
        "source_pair_id": pair,
        "target_C": 0.1,
        "reference_C": 0.1 + shift,
        "alignment_shift_abs": shift,
        "regime": full.classify_regime(shift),
    }


def test_regime_freeze_uses_all_300_pairs_and_contains_no_response_fields():
    pairs = tuple(f"generated_fact_{i}" for i in range(301, 601))
    rows = [
        _baseline_row(pair, 0.2 if index < 75 else 0.05)
        for index, pair in enumerate(pairs)
    ]
    frozen, counts = full.freeze_regimes(pairs, rows)
    assert counts == {full.REGIME_LARGE: 75, full.REGIME_SMALL: 225}
    assert len(frozen) == 300
    assert all(full._RESPONSE_FIELDS.isdisjoint(row) for row in frozen)
    assert all(row["classification_frozen_before_alignment"] is True for row in frozen)


def test_regime_freeze_rejects_any_response_field_leak():
    pairs = tuple(f"generated_fact_{i}" for i in range(301, 601))
    rows = [_baseline_row(pair, 0.05) for pair in pairs]
    rows[0]["R_ALIGN"] = -0.1
    with pytest.raises(full.ProspectiveFullError, match="FREEZE_RESPONSE_FIELD_LEAK"):
        full.freeze_regimes(pairs, rows)


def test_group_size_gate_is_exactly_30_per_group():
    assert full.group_size_passes({full.REGIME_LARGE: 30, full.REGIME_SMALL: 270})
    assert full.group_size_passes({full.REGIME_LARGE: 270, full.REGIME_SMALL: 30})
    assert not full.group_size_passes({full.REGIME_LARGE: 29, full.REGIME_SMALL: 271})
    assert not full.group_size_passes({full.REGIME_LARGE: 271, full.REGIME_SMALL: 29})


def test_student_t_cdf_matches_cauchy_closed_form_for_df1():
    for t in (-2.0, -0.5, 0.0, 0.5, 2.0):
        expected = 0.5 + math.atan(t) / math.pi
        assert full.student_t_cdf(t, 1.0) == pytest.approx(expected, abs=1e-12)


def test_one_sample_less_matches_df1_closed_form_example():
    # [-3, -1] => mean=-2, sample sd=sqrt(2), SE=1, t=-2, df=1.
    result = full.one_sample_less([-3.0, -1.0])
    expected_p = 0.5 + math.atan(-2.0) / math.pi
    assert result["t_statistic"] == pytest.approx(-2.0)
    assert result["degrees_of_freedom"] == pytest.approx(1.0)
    assert result["raw_p"] == pytest.approx(expected_p, abs=1e-12)


def test_welch_less_matches_df2_closed_form_example():
    # A=[-3,-1], B=[1,3]: t=-2*sqrt(2), Welch df=2.
    result = full.welch_less([-3.0, -1.0], [1.0, 3.0])
    t = -2.0 * math.sqrt(2.0)
    expected_p = 0.5 + t / (2.0 * math.sqrt(2.0 + t * t))
    assert result["t_statistic"] == pytest.approx(t)
    assert result["degrees_of_freedom"] == pytest.approx(2.0)
    assert result["raw_p"] == pytest.approx(expected_p, abs=1e-12)


def test_holm_family_is_exactly_two_and_step_down_adjusted():
    rows = full.holm_two([
        {"hypothesis": "H1", "raw_p": 0.01},
        {"hypothesis": "H2", "raw_p": 0.04},
    ])
    lookup = {row["hypothesis"]: row for row in rows}
    assert lookup["H1"]["holm_adjusted_p"] == pytest.approx(0.02)
    assert lookup["H2"]["holm_adjusted_p"] == pytest.approx(0.04)
    assert lookup["H1"]["holm_reject"] is True
    assert lookup["H2"]["holm_reject"] is True

    with pytest.raises(full.ProspectiveFullError, match="HOLM_EXACTLY_TWO_TESTS"):
        full.holm_two([{"hypothesis": "H1", "raw_p": 0.01}])


def _analysis_items(large_values, small_values):
    items = []
    for index, value in enumerate(large_values):
        items.append({
            "source_pair_id": f"generated_fact_{301 + index}",
            "regime": full.REGIME_LARGE,
            "R_ALIGN": float(value),
        })
    offset = len(items)
    for index, value in enumerate(small_values):
        items.append({
            "source_pair_id": f"generated_fact_{301 + offset + index}",
            "regime": full.REGIME_SMALL,
            "R_ALIGN": float(value),
        })
    return items


def test_analysis_uses_only_h1_h2_and_support_requires_both():
    # 30 LARGE with a strong negative mean and 270 SMALL around positive values.
    large = [-0.20 + 0.002 * i for i in range(30)]
    small = [0.05 + 0.0005 * (i % 20) for i in range(270)]
    result = full.analyze_responses(_analysis_items(large, small))
    tests = result["inferential_family"]["tests"]
    assert result["inferential_family"]["hypothesis_count"] == 2
    assert {row["hypothesis"] for row in tests} == {"H1", "H2"}
    assert result["prospective_regime_test"] == "SUPPORTED"
    assert result["scientific_conclusion"] == "PROSPECTIVE_ADVERSE_REGIME_SUPPORTED"


def test_full_runner_source_freezes_membership_before_alignment_phase():
    source = inspect.getsource(full.run_full)
    freeze_pos = source.index("_write_regime_freeze")
    alignment_pos = source.index("_run_alignment_pair")
    assert freeze_pos < alignment_pos
    assert "budget.used == BASELINE_FORWARD_BUDGET" in source
    assert "budget.assert_exact()" in source


def test_full_runner_has_no_magnitude_path_or_extra_inferential_tests():
    run_source = inspect.getsource(full.run_full).lower()
    analysis_source = inspect.getsource(full.analyze_responses)
    assert "magnitude" not in run_source
    assert '"H1"' in analysis_source
    assert '"H2"' in analysis_source
    assert '"H3"' not in analysis_source
    assert "holm_two" in analysis_source


def test_output_surface_separates_preintervention_freeze_and_responses():
    assert full.REGIME_FILE == "regime_manifest.jsonl"
    assert full.REGIME_FREEZE_FILE == "regime_freeze.json"
    assert full.ITEM_FILE == "item_metrics.jsonl"
    assert full.SUMMARY_FILE == "summary.json"
    assert full.MANIFEST_FILE == "manifest.json"
    assert full.CHECKSUM_FILE == "SHA256SUMS.txt"


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
    with pytest.raises(full.ProspectiveFullError, match="BRANCH_MISMATCH:main"):
        full.authenticate_repo(expected)
