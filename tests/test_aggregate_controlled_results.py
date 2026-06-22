from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.aggregate_controlled_results import (
    aggregate_groups,
    aggregate_results,
    group_markdown,
    main,
)


CHECK_NAMES = (
    "paraphrase_preserved",
    "predicate_disentangled",
    "polarity_flip_preserved_and_reversed",
    "deletion_sufficiency_lower",
    "truncation_sufficiency_lower",
    "entity_frame_lower",
    "event_frame_lower",
)


def synthetic_report(epoch: int, accuracy: float, macro_f1: float) -> dict:
    return {
        "final_epoch": 10,
        "best_epoch": epoch,
        "best_dev_metrics": {
            "final_accuracy": accuracy,
            "final_macro_f1": macro_f1,
            "frame_accuracy": 0.8,
            "predicate_accuracy": 0.7,
            "polarity_accuracy_entitled": 0.9,
            "sufficiency_accuracy": 1.0,
            "prediction_distribution": {
                "NOT_ENTITLED": 10,
                "REFUTE": 2,
                "SUPPORT": 3,
            },
        },
        "best_dev_interventions": {},
        "best_dev_pairwise_checks": {
            name: {"pass_rate": 0.5, "passed": False} for name in CHECK_NAMES
        },
    }


def write_synthetic(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report), encoding="utf-8")


def test_aggregate_mean_and_sample_std() -> None:
    paths = [
        Path(__file__).parent / ".aggregate_seed1.json",
        Path(__file__).parent / ".aggregate_seed2.json",
    ]
    try:
        write_synthetic(paths[0], synthetic_report(4, 0.7, 0.6))
        write_synthetic(paths[1], synthetic_report(8, 0.8, 0.8))
        result = aggregate_results(paths)
        assert len(result["runs"]) == 2
        assert result["aggregate"]["final_accuracy"]["mean"] == pytest.approx(0.75)
        assert result["aggregate"]["final_accuracy"]["std"] == pytest.approx(
            0.0707106781
        )
        assert result["aggregate"]["best_epoch"]["mean"] == 6.0
        assert result["runs"][0]["prediction_distribution"]["REFUTE"] == 2
    finally:
        for path in paths:
            path.unlink(missing_ok=True)


def test_cli_prints_markdown_and_writes_csv_json(capsys) -> None:
    directory = Path(__file__).parent
    inputs = [directory / ".aggregate_cli1.json", directory / ".aggregate_cli2.json"]
    csv_path = directory / ".aggregate_output.csv"
    json_path = directory / ".aggregate_output.json"
    paths_to_clean = [*inputs, csv_path, json_path]
    try:
        write_synthetic(inputs[0], synthetic_report(3, 0.6, 0.5))
        write_synthetic(inputs[1], synthetic_report(5, 0.9, 0.7))
        exit_code = main(
            [
                "--inputs",
                str(inputs[0]),
                str(inputs[1]),
                "--output-csv",
                str(csv_path),
                "--output-json",
                str(json_path),
            ]
        )
        stdout = capsys.readouterr().out
        assert exit_code == 0
        assert "| file | epoch | acc | macro-F1" in stdout
        assert "| mean |" in stdout
        with csv_path.open(encoding="utf-8", newline="") as handle:
            assert len(list(csv.DictReader(handle))) == 2
        exported = json.loads(json_path.read_text(encoding="utf-8"))
        assert len(exported["runs"]) == 2
        assert exported["aggregate"]["final_macro_f1"]["mean"] == pytest.approx(0.6)
    finally:
        for path in paths_to_clean:
            path.unlink(missing_ok=True)


def test_group_comparison_tables_and_statistics(capsys) -> None:
    directory = Path(__file__).parent
    full_paths = [directory / ".group_full1.json", directory / ".group_full2.json"]
    ablation_paths = [directory / ".group_ablate1.json", directory / ".group_ablate2.json"]
    paths = [*full_paths, *ablation_paths]
    try:
        write_synthetic(full_paths[0], synthetic_report(3, 0.7, 0.6))
        write_synthetic(full_paths[1], synthetic_report(4, 0.8, 0.8))
        write_synthetic(ablation_paths[0], synthetic_report(5, 0.9, 0.85))
        write_synthetic(ablation_paths[1], synthetic_report(6, 0.8, 0.75))
        specs = [
            f"full4e={full_paths[0]},{full_paths[1]}",
            f"no_intervention={ablation_paths[0]},{ablation_paths[1]}",
        ]
        result = aggregate_groups(specs)
        assert result["groups"]["full4e"]["aggregate"]["final_macro_f1"][
            "mean"
        ] == pytest.approx(0.7)
        assert result["groups"]["no_intervention"]["aggregate"][
            "final_accuracy"
        ]["mean"] == pytest.approx(0.85)
        markdown = group_markdown(result)
        assert "CLASSIFICATION_SUMMARY" in markdown
        assert "PAIRWISE_CONSISTENCY_SUMMARY" in markdown
        assert "KEY_CONTRAST_SUMMARY" in markdown
        assert "predicate-pair" in markdown

        assert main(["--group", specs[0], "--group", specs[1]]) == 0
        stdout = capsys.readouterr().out
        assert "full4e" in stdout
        assert "no_intervention" in stdout
    finally:
        for path in paths:
            path.unlink(missing_ok=True)
