"""Aggregate controlled-v5 result JSON files into compact reports."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Sequence


METRIC_FIELDS = (
    "final_accuracy",
    "final_macro_f1",
    "frame_accuracy",
    "predicate_accuracy",
    "polarity_accuracy_entitled",
    "sufficiency_accuracy",
)
PAIRWISE_FIELDS = (
    "paraphrase_preserved",
    "predicate_disentangled",
    "polarity_flip_preserved_and_reversed",
    "deletion_sufficiency_lower",
    "truncation_sufficiency_lower",
    "entity_frame_lower",
    "event_frame_lower",
)
NUMERIC_FIELDS = METRIC_FIELDS + PAIRWISE_FIELDS


def _best_summary(payload: dict[str, Any]) -> dict[str, Any]:
    if "best_dev_metrics" in payload:
        return payload
    runs = payload.get("runs")
    if isinstance(runs, dict) and len(runs) == 1:
        run = next(iter(runs.values()))
        if isinstance(run, dict) and "best_dev_metrics" in run:
            return run
    raise ValueError("result has no unambiguous best_dev_metrics summary")


def extract_run(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = _best_summary(payload)
    metrics = summary["best_dev_metrics"]
    checks = summary["best_dev_pairwise_checks"]
    row: dict[str, Any] = {
        "filename": path.name,
        "best_epoch": int(summary["best_epoch"]),
        "prediction_distribution": metrics.get("prediction_distribution", {}),
    }
    for field in METRIC_FIELDS:
        row[field] = float(metrics[field])
    for field in PAIRWISE_FIELDS:
        value = checks[field]
        row[field] = float(value["pass_rate"] if isinstance(value, dict) else value)
    return row


def aggregate_results(paths: Sequence[Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one input result is required")
    rows = [extract_run(path) for path in paths]
    aggregate: dict[str, dict[str, float]] = {}
    for field in NUMERIC_FIELDS:
        values = [float(row[field]) for row in rows]
        aggregate[field] = {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    best_epochs = [float(row["best_epoch"]) for row in rows]
    aggregate["best_epoch"] = {
        "mean": statistics.fmean(best_epochs),
        "std": statistics.stdev(best_epochs) if len(best_epochs) > 1 else 0.0,
    }
    return {"runs": rows, "aggregate": aggregate}


def _distribution_text(distribution: dict[str, int]) -> str:
    return "/".join(f"{label}:{count}" for label, count in sorted(distribution.items()))


def markdown_table(result: dict[str, Any]) -> str:
    headers = (
        "file", "epoch", "acc", "macro-F1", "frame", "predicate", "polarity",
        "sufficiency", "para", "pred-pair", "flip", "delete", "truncate",
        "entity", "event", "predictions",
    )
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in result["runs"]:
        values = [
            row["filename"],
            str(row["best_epoch"]),
            f"{row['final_accuracy']:.3f}",
            f"{row['final_macro_f1']:.3f}",
            f"{row['frame_accuracy']:.3f}",
            f"{row['predicate_accuracy']:.3f}",
            f"{row['polarity_accuracy_entitled']:.3f}",
            f"{row['sufficiency_accuracy']:.3f}",
            f"{row['paraphrase_preserved']:.3f}",
            f"{row['predicate_disentangled']:.3f}",
            f"{row['polarity_flip_preserved_and_reversed']:.3f}",
            f"{row['deletion_sufficiency_lower']:.3f}",
            f"{row['truncation_sufficiency_lower']:.3f}",
            f"{row['entity_frame_lower']:.3f}",
            f"{row['event_frame_lower']:.3f}",
            _distribution_text(row["prediction_distribution"]),
        ]
        lines.append("| " + " | ".join(values) + " |")

    aggregate = result["aggregate"]
    lines.extend(
        [
            "",
            "| statistic | epoch | acc | macro-F1 | frame | predicate | polarity | sufficiency |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for statistic in ("mean", "std"):
        values = [
            statistic,
            f"{aggregate['best_epoch'][statistic]:.3f}",
            *[
                f"{aggregate[field][statistic]:.3f}"
                for field in METRIC_FIELDS
            ],
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_csv(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["filename", "best_epoch", *NUMERIC_FIELDS, "prediction_distribution"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in result["runs"]:
            serialized = dict(row)
            serialized["prediction_distribution"] = json.dumps(
                row["prediction_distribution"], sort_keys=True, separators=(",", ":")
            )
            writer.writerow(serialized)


def write_json(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)

    result = aggregate_results(args.inputs)
    print(markdown_table(result))
    if args.output_csv:
        write_csv(result, args.output_csv)
    if args.output_json:
        write_json(result, args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

