from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.aggregate_controlled_results import aggregate_results
from scripts.write_stage5b_v2_report import main


PAIRWISE = (
    "paraphrase_preserved",
    "predicate_disentangled",
    "polarity_flip_preserved_and_reversed",
    "deletion_sufficiency_lower",
    "truncation_sufficiency_lower",
    "entity_frame_lower",
    "event_frame_lower",
)


def synthetic_result(macro_f1: float, predicate: float, polarity: float) -> dict:
    checks = {name: {"pass_rate": 1.0, "passed": True} for name in PAIRWISE}
    checks["predicate_disentangled"]["pass_rate"] = predicate
    checks["polarity_flip_preserved_and_reversed"]["pass_rate"] = polarity
    return {
        "best_epoch": 4,
        "final_epoch": 5,
        "best_dev_metrics": {
            "final_accuracy": macro_f1 + 0.02,
            "final_macro_f1": macro_f1,
            "frame_accuracy": 0.8,
            "predicate_accuracy": 0.9,
            "polarity_accuracy_entitled": 1.0,
            "sufficiency_accuracy": 1.0,
            "prediction_distribution": {"NOT_ENTITLED": 10},
        },
        "best_dev_interventions": {},
        "best_dev_pairwise_checks": checks,
    }


def test_stage5b_writer_outputs_required_reports() -> None:
    directory = Path(__file__).parent
    inputs = {
        "v2_full4e": directory / ".stage5b_full.json",
        "v2_no_intervention": directory / ".stage5b_none.json",
        "v2_no_predicate_contrast": directory / ".stage5b_nopred.json",
        "v2_no_polarity_flip": directory / ".stage5b_noflip.json",
    }
    output_md = directory / ".stage5b_report.md"
    output_csv = directory / ".stage5b_report.csv"
    cleanup = [*inputs.values(), output_md, output_csv]
    try:
        inputs["v2_full4e"].write_text(
            json.dumps(synthetic_result(0.75, 0.9, 0.8)), encoding="utf-8"
        )
        inputs["v2_no_intervention"].write_text(
            json.dumps(synthetic_result(0.85, 0.6, 0.2)), encoding="utf-8"
        )
        inputs["v2_no_predicate_contrast"].write_text(
            json.dumps(synthetic_result(0.7, 0.3, 0.7)), encoding="utf-8"
        )
        inputs["v2_no_polarity_flip"].write_text(
            json.dumps(synthetic_result(0.72, 0.8, 0.1)), encoding="utf-8"
        )
        arguments: list[str] = []
        for name, path in inputs.items():
            arguments.extend(["--group", f"{name}={path}"])
        arguments.extend(
            ["--output-md", str(output_md), "--output-csv", str(output_csv)]
        )
        assert main(arguments) == 0

        markdown = output_md.read_text(encoding="utf-8")
        assert markdown.count("## ") == 4
        for heading in (
            "CLASSIFICATION_SUMMARY",
            "PAIRWISE_CONSISTENCY_SUMMARY",
            "KEY_CONTRAST_SUMMARY",
            "INTERPRETATION",
        ):
            assert f"## {heading}" in markdown
        assert "0.750 ± 0.000" in markdown
        assert "diverge across configurations" in markdown
        assert "weakens predicate behavior" in markdown
        assert "collapses polarity-flip behavior" in markdown

        with output_csv.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert rows
        assert set(rows[0]) == {
            "section", "config", "metric", "mean", "std", "formatted"
        }
        assert {row["section"] for row in rows} == {
            "classification", "pairwise", "key_contrast"
        }
        assert {row["config"] for row in rows} == set(inputs)
        assert all("±" in row["formatted"] for row in rows)

        # Existing ungrouped aggregation remains importable and functional.
        assert aggregate_results([inputs["v2_full4e"]])["aggregate"][
            "final_macro_f1"
        ]["std"] == 0.0
    finally:
        for path in cleanup:
            path.unlink(missing_ok=True)

