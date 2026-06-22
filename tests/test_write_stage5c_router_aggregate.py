from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.write_stage5c_router_aggregate import SYSTEM_ORDER, main


def _write_seed(path: Path, offset: float) -> None:
    rows: list[dict[str, object]] = []
    for index, system in enumerate(SYSTEM_ORDER):
        macro = 0.70 + offset + index * 0.01
        metrics = (
            ("classification", "final_accuracy", macro + 0.02),
            ("classification", "final_macro_f1", macro),
            ("classification", "NOT_ENTITLED_f1", macro),
            ("classification", "REFUTE_f1", macro - 0.01),
            ("classification", "SUPPORT_f1", macro + 0.01),
            ("pairwise", "paraphrase_preserved", 0.8),
            ("pairwise", "predicate_disentangled", 0.7 + offset),
            ("pairwise", "polarity_flip_preserved_and_reversed", 0.9),
            ("internal_faithfulness", "entitled_output_gate_violation_rate", 0.2 if system == "classifier_only" else 0.0),
            ("internal_faithfulness", "entitled_output_count", 10),
            ("internal_faithfulness", "entitled_output_gate_violations", 2 if system == "classifier_only" else 0),
            ("internal_faithfulness", "polarity_flip_output_ok", 0.9),
            ("internal_faithfulness", "polarity_flip_internal_ok", 0.5 if system == "classifier_only" else 0.9),
            ("internal_faithfulness", "polarity_flip_output_internal_gap", 0.4 if system == "classifier_only" else 0.0),
            ("internal_faithfulness", "polarity_flip_output_ok_but_internal_bad", 4 if system == "classifier_only" else 0),
        )
        rows.extend(
            {"section": section, "system": system, "metric": metric, "value": value}
            for section, metric, value in metrics
        )
        rows.append(
            {
                "section": "classification",
                "system": system,
                "metric": "prediction_distribution",
                "value": '{"SUPPORT": 5}',
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("section", "system", "metric", "value")
        )
        writer.writeheader()
        writer.writerows(rows)


def test_stage5c_router_aggregate_writes_reproducible_reports(tmp_path: Path) -> None:
    seed1 = tmp_path / "seed1.csv"
    seed2 = tmp_path / "seed2.csv"
    output_md = tmp_path / "aggregate.md"
    output_csv = tmp_path / "aggregate.csv"
    _write_seed(seed1, 0.0)
    _write_seed(seed2, 0.1)

    assert (
        main(
            [
                "--input",
                str(seed1),
                "--input",
                str(seed2),
                "--output-md",
                str(output_md),
                "--output-csv",
                str(output_csv),
            ]
        )
        == 0
    )
    assert output_md.exists() and output_csv.exists()

    with output_csv.open(newline="", encoding="utf-8") as handle:
        aggregate = list(csv.DictReader(handle))
    assert list(aggregate[0]) == [
        "section",
        "system",
        "metric",
        "mean",
        "std",
        "n",
        "formatted",
    ]
    macro = next(row for row in aggregate
                 if row["system"] == "classifier_only"
                 and row["metric"] == "final_macro_f1")
    assert float(macro["mean"]) == pytest.approx(0.75)
    assert float(macro["std"]) == pytest.approx(0.1 / 2**0.5)
    assert int(macro["n"]) == 2

    assert not any(row["metric"] == "prediction_distribution" for row in aggregate)
    assert any(row["section"] == "internal_faithfulness"
               and row["metric"] == "polarity_flip_output_internal_gap"
               for row in aggregate)

    markdown = output_md.read_text(encoding="utf-8")
    assert "## ROUTER_INTERNAL_FAITHFULNESS_AGGREGATE" in markdown
    positions = [markdown.index(f"| {system} |") for system in SYSTEM_ORDER]
    assert positions == sorted(positions)
