from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest

from scripts import reason_router_gen4_xg2_xg4_fresh_response_fast_cuda_full_baseline as full


@pytest.mark.parametrize(
    "family,first,last",
    (
        ("xg2", "xg2_fact_301", "xg2_fact_600"),
        ("xg4", "xg4_fact_301", "xg4_fact_600"),
    ),
)
def test_fresh_population_is_exact_301_600(
    family,
    first,
    last,
):
    rows, encoded, events = (
        full.fresh_eq.load_family_inputs(
            family,
            None,
        )
    )

    pairs = full.fresh_eq._pair_order(
        family,
        rows,
    )

    assert len(pairs) == 300
    assert pairs[0] == first
    assert pairs[-1] == last
    assert tuple(
        encoded["input_ids"].shape
    ) == (1800, 128)
    assert len(events) == 1800


def test_phase1_budget_and_threshold_are_frozen():
    assert full.SOURCE_PAIR_COUNT == 300
    assert full.BASELINE_FORWARDS_PER_PAIR == 4
    assert full.FULL_BASELINE_FORWARD_BUDGET == 1200
    assert (
        full.ALIGNMENT_SHIFT_THRESHOLD
        == 0.11228626366380845
    )
    assert full.MIN_GROUP_SIZE == 30


@pytest.mark.parametrize(
    "family",
    ("xg2", "xg4"),
)
def test_frozen_equivalence_artifacts_are_exact(
    family,
):
    report = full.validate_equivalence_artifact(
        family
    )

    assert (
        report["result"]
        == full.fresh_eq.RESULT_PASS
    )
    assert report["family_key"] == family
    assert (
        report["source_pair_id"]
        == f"{family}_fact_301"
    )
    assert (
        report["scientific_budget_forward_count"]
        == 0
    )


def _items(
    n_large: int,
) -> list[dict[str, object]]:
    rows = []

    for index in range(300):
        regime = (
            full.REGIME_LARGE
            if index < n_large
            else full.REGIME_SMALL
        )
        rows.append(
            {
                "source_pair_id": (
                    f"xg2_fact_{index + 301:03d}"
                ),
                "regime": regime,
            }
        )

    return rows


def test_support_gate_ready_at_30_270():
    summary = full.summarize_regimes(
        _items(30)
    )

    assert summary["n_LARGE"] == 30
    assert summary["n_SMALL"] == 270
    assert summary["support_gate_pass"] is True
    assert (
        summary["prospective_regime_test"]
        == full.REGIME_TEST_READY
    )


def test_support_gate_blocks_at_29_271():
    summary = full.summarize_regimes(
        _items(29)
    )

    assert summary["n_LARGE"] == 29
    assert summary["n_SMALL"] == 271
    assert summary["support_gate_pass"] is False
    assert (
        summary["prospective_regime_test"]
        == full.REGIME_TEST_BLOCKED
    )


def test_regime_boundary_is_exact_threshold():
    t = full.ALIGNMENT_SHIFT_THRESHOLD

    assert (
        full.classify_regime(t)
        == full.REGIME_LARGE
    )
    assert (
        full.classify_regime(t - 1e-12)
        == full.REGIME_SMALL
    )


def test_output_contract_is_canonical_lf(
    tmp_path,
):
    items = [
        {
            "source_pair_id": (
                f"xg2_fact_{index + 301:03d}"
            ),
            "regime": (
                full.REGIME_LARGE
                if index < 30
                else full.REGIME_SMALL
            ),
        }
        for index in range(300)
    ]

    summary = {
        "schema_version": full.SUMMARY_SCHEMA,
        "result": full.RESULT_PASS,
    }

    out = tmp_path / "artifact"

    hashes = full._write_outputs(
        out,
        items=items,
        summary=summary,
    )

    assert set(hashes) == {
        full.ITEM_FILE,
        full.SUMMARY_FILE,
        full.MANIFEST_FILE,
    }

    for name in (
        full.ITEM_FILE,
        full.SUMMARY_FILE,
        full.MANIFEST_FILE,
        full.CHECKSUM_FILE,
    ):
        raw = (out / name).read_bytes()
        assert raw.endswith(b"\n")
        assert b"\r\n" not in raw

    loaded = json.loads(
        (out / full.SUMMARY_FILE)
        .read_text(encoding="utf-8")
    )
    assert loaded == summary


def test_execution_surface_is_baseline_only():
    source = inspect.getsource(
        full.run_full_baseline
    )
    tree = ast.parse(source)

    calls = {
        (
            node.func.value.id,
            node.func.attr,
        )
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
    }

    assert (
        "fresh_eq",
        "load_family_inputs",
    ) in calls

    assert (
        "prevalence_eq",
        "run_baseline_geometry_pair",
    ) in calls

    assert (
        "prevalence_eq",
        "run_prospective_pair",
    ) not in calls

    assert "alignment_delta" not in source


def test_no_training_backward_or_task_head_path():
    source = inspect.getsource(full)

    assert ".backward(" not in source
    assert ".train(" not in source
    assert "optimizer.step" not in source
    assert "task_head(" not in source


def test_reused_dependency_blobs_are_exact():
    root = Path(__file__).resolve().parents[1]

    for path, expected in full.REUSED_BLOBS.items():
        observed = full.git(
            "rev-parse",
            f"HEAD:{path}",
        )
        assert observed == expected


def test_design_and_equivalence_freezes_are_exact():
    assert (
        full.DESIGN_FREEZE_COMMIT
        == "2074f52d39bf0fca6d63248016ce47c54bff5e06"
    )
    assert (
        full.EQUIVALENCE_ARTIFACT_FREEZE_COMMIT
        == "9ab93bcbb05fe2ba69e1d8f0c77b62626edd9945"
    )
    assert (
        full.EQUIVALENCE_EXECUTION_HEAD
        == "0385dbe8989c4468b471eef2e2a7196bac20826d"
    )
