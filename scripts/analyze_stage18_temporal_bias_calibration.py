"""Analyze Stage18-B1 temporal bias calibration outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "stage18_temporal_bias_calibrated_seed1.json"
DEFAULT_OUTPUT_DIR = ROOT / "results"
ENTITLED = {"REFUTE", "SUPPORT"}
CONTROL_GROUPS = ("temporal_erased", "surface_control", "sufficiency_control")


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def distribution(rows: Sequence[dict[str, Any]], key: str) -> str:
    return json.dumps(dict(sorted(Counter(row.get(key) for row in rows).items())), sort_keys=True)


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    gold_ne = sum(row["gold_final_label"] == "NOT_ENTITLED" for row in rows)
    original_correct = sum(row["original_pred_final_label"] == row["gold_final_label"] for row in rows)
    adjusted_correct = sum(row["adjusted_pred_final_label"] == row["gold_final_label"] for row in rows)
    original_false = sum(
        row["gold_final_label"] == "NOT_ENTITLED"
        and row["original_pred_final_label"] in ENTITLED
        for row in rows
    )
    adjusted_false = sum(
        row["gold_final_label"] == "NOT_ENTITLED"
        and row["adjusted_pred_final_label"] in ENTITLED
        for row in rows
    )
    return {
        "split": rows[0]["split"] if rows else "",
        "stage15_probe_type": rows[0].get("stage15_probe_type") if rows else "",
        "n": n,
        "original_accuracy": safe_div(original_correct, n),
        "adjusted_accuracy": safe_div(adjusted_correct, n),
        "original_false_entitled_count": original_false,
        "adjusted_false_entitled_count": adjusted_false,
        "original_false_entitled_rate": safe_div(original_false, gold_ne),
        "adjusted_false_entitled_rate": safe_div(adjusted_false, gold_ne),
        "changed_count": sum(
            row["original_pred_final_label"] != row["adjusted_pred_final_label"]
            for row in rows
        ),
        "temporal_mismatch_flag_count": sum(int(row["temporal_mismatch_flag"]) for row in rows),
        "original_prediction_distribution": distribution(rows, "original_pred_final_label"),
        "adjusted_prediction_distribution": distribution(rows, "adjusted_pred_final_label"),
    }


def group_metrics(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["split"], str(row.get("stage15_probe_type")))].append(row)
    return [summarize(grouped[key]) for key in sorted(grouped)]


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.3f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def find_metric(metrics: Sequence[dict[str, Any]], split: str, group: str) -> dict[str, Any] | None:
    for row in metrics:
        if row["split"] == split and row["stage15_probe_type"] == group:
            return row
    return None


def controls_preserved(metrics: Sequence[dict[str, Any]]) -> bool:
    for split in ("calibration", "heldout"):
        for group in CONTROL_GROUPS:
            row = find_metric(metrics, split, group)
            if row is not None and int(row["changed_count"]) != 0:
                return False
    return True


def summary_md(metadata: dict[str, Any], metrics: Sequence[dict[str, Any]]) -> str:
    selected = metadata.get("selected_alpha")
    objective = metadata.get("objective")
    cal_temp = find_metric(metrics, "calibration", "temporal_mismatch")
    held_temp = find_metric(metrics, "heldout", "temporal_mismatch")
    columns = [
        "split",
        "stage15_probe_type",
        "n",
        "original_false_entitled_count",
        "adjusted_false_entitled_count",
        "changed_count",
        "temporal_mismatch_flag_count",
        "adjusted_accuracy",
    ]
    lines = [
        "# Stage18-B1 Temporal Bias Calibration Summary",
        "",
        f"Selected alpha: {selected}",
        f"Objective: {objective}",
        f"Controls preserved: {controls_preserved(metrics)}",
        "",
        "Stage18-B1 is prediction-level calibration, not an end-to-end trained model.",
        "",
        "## Temporal mismatch before/after",
        "",
    ]
    if cal_temp:
        lines.append(
            f"- Calibration temporal_mismatch false-entitled: {cal_temp['original_false_entitled_count']} -> {cal_temp['adjusted_false_entitled_count']}"
        )
    if held_temp:
        lines.append(
            f"- Heldout temporal_mismatch false-entitled: {held_temp['original_false_entitled_count']} -> {held_temp['adjusted_false_entitled_count']}"
        )
    lines.extend(["", "## Group metrics", "", markdown_table(metrics, columns), ""])
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrated-preds", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = json.loads(args.calibrated_preds.read_text(encoding="utf-8"))
    rows = payload.get("predictions", [])
    metadata = payload.get("metadata", {})
    if not rows:
        raise ValueError("calibrated prediction file has no predictions")
    metrics = group_metrics(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "stage18_temporal_bias_calibration_group_metrics.csv", metrics)
    write_csv(
        args.output_dir / "stage18_temporal_bias_calibration_alpha_grid.csv",
        metadata.get("calibration_metrics_per_alpha", []),
    )
    write_csv(args.output_dir / "stage18_temporal_bias_calibration_examples.csv", rows)
    (args.output_dir / "stage18_temporal_bias_calibration_summary.md").write_text(
        summary_md(metadata, metrics),
        encoding="utf-8",
    )
    print("STAGE18_TEMPORAL_BIAS_CALIBRATION_ANALYSIS")
    print(f"selected_alpha\t{metadata.get('selected_alpha')}")
    print(f"objective\t{metadata.get('objective')}")
    print(f"controls_preserved\t{controls_preserved(metrics)}")
    print(f"group_metrics\t{args.output_dir / 'stage18_temporal_bias_calibration_group_metrics.csv'}")
    print(f"alpha_grid\t{args.output_dir / 'stage18_temporal_bias_calibration_alpha_grid.csv'}")
    print(f"examples\t{args.output_dir / 'stage18_temporal_bias_calibration_examples.csv'}")
    print(f"summary\t{args.output_dir / 'stage18_temporal_bias_calibration_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

