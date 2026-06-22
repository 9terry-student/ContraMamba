from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.evaluate_router_ensemble import router_cost_metrics
from scripts.write_stage9a_router_cost_aggregate import main as aggregate_main


def _auditor(example_id: str, gate: float) -> dict:
    return {
        "id": example_id,
        "frame_prob": gate,
        "predicate_coverage_prob": gate,
        "sufficiency_prob": gate,
        "entitlement_prob": gate,
        "polarity_margin": 1.0,
    }


def test_router_cost_metrics_measure_downgrades_recall_and_precision() -> None:
    records = [
        {"id": "a", "gold_final_label": "SUPPORT"},
        {"id": "b", "gold_final_label": "SUPPORT"},
        {"id": "c", "gold_final_label": "NOT_ENTITLED"},
        {"id": "d", "gold_final_label": "REFUTE"},
        {"id": "e", "gold_final_label": "SUPPORT"},
    ]
    classifier = {
        "a": "SUPPORT",
        "b": "SUPPORT",
        "c": "SUPPORT",
        "d": "REFUTE",
        "e": "NOT_ENTITLED",
    }
    routed = {
        "a": "SUPPORT",
        "b": "NOT_ENTITLED",
        "c": "NOT_ENTITLED",
        "d": "REFUTE",
        "e": "NOT_ENTITLED",
    }
    auditors = {
        "a": (_auditor("a", 0.9),),
        "b": (_auditor("b", 0.2),),
        "c": (_auditor("c", 0.2),),
        "d": (_auditor("d", 0.9),),
        "e": (_auditor("e", 0.2),),
    }
    metrics = router_cost_metrics(records, classifier, routed, auditors, 0.5)

    assert metrics["classifier_entitled_count"] == 4
    assert metrics["routed_entitled_count"] == 2
    assert metrics["downgraded_count"] == 2
    assert metrics["downgrade_rate_among_classifier_entitled"] == 0.5
    assert metrics["downgraded_gold_support_count"] == 1
    assert metrics["support_recall_pre_router"] == pytest.approx(2 / 3)
    assert metrics["support_recall_post_router"] == pytest.approx(1 / 3)
    assert metrics["support_recall_drop"] == pytest.approx(1 / 3)
    assert metrics["support_precision_pre_router"] == pytest.approx(2 / 3)
    assert metrics["support_precision_post_router"] == 1.0
    assert metrics["support_precision_gain"] == pytest.approx(1 / 3)
    assert metrics["false_support_removed_count"] == 1
    assert metrics["false_refute_removed_count"] == 0
    assert metrics["pre_router_candidate_gate_fail_count"] == 2
    assert metrics["pre_router_candidate_gate_fail_rate"] == 0.5
    assert metrics["retained_violation_rate"] == 0.0


def test_router_cost_metrics_handle_zero_denominators() -> None:
    records = [{"id": "x", "gold_final_label": "NOT_ENTITLED"}]
    labels = {"x": "NOT_ENTITLED"}
    metrics = router_cost_metrics(
        records, labels, labels, {"x": (_auditor("x", 0.1),)}, 0.5
    )
    for metric in (
        "downgrade_rate_among_classifier_entitled",
        "support_recall_pre_router",
        "support_recall_post_router",
        "support_recall_drop",
        "support_precision_pre_router",
        "support_precision_post_router",
        "support_precision_gain",
        "pre_router_candidate_gate_fail_rate",
        "retained_violation_rate",
    ):
        assert metrics[metric] == 0.0


def _write_seed(path: Path, downgrade_rate: float) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "threshold",
                "system",
                "final_accuracy",
                "final_macro_f1",
                "downgrade_rate_among_classifier_entitled",
                "support_recall_drop",
                "retained_violation_rate",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "threshold": 0.5,
                "system": "conservative_balanced_router",
                "final_accuracy": 0.9,
                "final_macro_f1": 0.85,
                "downgrade_rate_among_classifier_entitled": downgrade_rate,
                "support_recall_drop": 0.1,
                "retained_violation_rate": 0.0,
            }
        )


def test_stage9_aggregate_writes_sample_statistics(tmp_path: Path) -> None:
    seed1, seed2 = tmp_path / "seed1.csv", tmp_path / "seed2.csv"
    output_csv, output_md = tmp_path / "aggregate.csv", tmp_path / "aggregate.md"
    _write_seed(seed1, 0.2)
    _write_seed(seed2, 0.4)
    assert aggregate_main(
        [
            "--input", str(seed1), "--input", str(seed2),
            "--output-csv", str(output_csv), "--output-md", str(output_md),
        ]
    ) == 0
    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    target = next(
        row for row in rows
        if row["metric"] == "downgrade_rate_among_classifier_entitled"
    )
    assert float(target["mean"]) == pytest.approx(0.3)
    assert float(target["std"]) == pytest.approx(0.2 / 2**0.5)
    assert "THRESHOLD 0.5" in output_md.read_text(encoding="utf-8")
