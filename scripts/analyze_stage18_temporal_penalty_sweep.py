"""Analyze Stage18-A temporal penalty sweep outputs."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_GLOB = str(ROOT / "results" / "stage18_temporal_penalty_p*.json")
DEFAULT_OUTPUT_DIR = ROOT / "results"
ENTITLED = {"REFUTE", "SUPPORT"}
CONTROL_GROUPS = ("temporal_erased", "surface_control", "sufficiency_control")


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def prediction_distribution(rows: Sequence[dict[str, Any]], key: str) -> str:
    return json.dumps(dict(sorted(Counter(row.get(key) for row in rows).items())), sort_keys=True)


def load_penalty_files(pattern: str) -> list[dict[str, Any]]:
    paths = [Path(path) for path in sorted(glob.glob(pattern))]
    if not paths:
        raise ValueError(f"no files matched {pattern!r}")
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        penalty = payload.get("metadata", {}).get("penalty")
        for row in payload.get("predictions", []):
            merged = dict(row)
            merged["penalty"] = float(row.get("penalty", penalty))
            merged["_source_file"] = str(path)
            rows.append(merged)
    if not rows:
        raise ValueError("matched files contain no predictions")
    return rows


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    original_correct = sum(
        row.get("original_pred_final_label") == row.get("gold_final_label")
        for row in rows
    )
    adjusted_correct = sum(
        row.get("adjusted_pred_final_label") == row.get("gold_final_label")
        for row in rows
    )
    gold_ne = sum(row.get("gold_final_label") == "NOT_ENTITLED" for row in rows)
    original_false = sum(
        row.get("gold_final_label") == "NOT_ENTITLED"
        and row.get("original_pred_final_label") in ENTITLED
        for row in rows
    )
    adjusted_false = sum(
        row.get("gold_final_label") == "NOT_ENTITLED"
        and row.get("adjusted_pred_final_label") in ENTITLED
        for row in rows
    )
    changed = sum(
        row.get("original_pred_final_label") != row.get("adjusted_pred_final_label")
        for row in rows
    )
    return {
        "penalty": float(rows[0].get("penalty", 0.0)) if rows else 0.0,
        "stage15_probe_type": rows[0].get("stage15_probe_type") if rows else "",
        "n": n,
        "original_accuracy": safe_div(original_correct, n),
        "adjusted_accuracy": safe_div(adjusted_correct, n),
        "original_false_entitled_count": original_false,
        "adjusted_false_entitled_count": adjusted_false,
        "original_false_entitled_rate": safe_div(original_false, gold_ne),
        "adjusted_false_entitled_rate": safe_div(adjusted_false, gold_ne),
        "changed_count": changed,
        "temporal_mismatch_flag_count": sum(
            int(row.get("temporal_mismatch_flag", 0)) for row in rows
        ),
        "original_prediction_distribution": prediction_distribution(
            rows,
            "original_pred_final_label",
        ),
        "adjusted_prediction_distribution": prediction_distribution(
            rows,
            "adjusted_pred_final_label",
        ),
    }


def group_metrics(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(float(row.get("penalty", 0.0)), str(row.get("stage15_probe_type")))].append(row)
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


def smallest_valid_penalty(metrics: Sequence[dict[str, Any]]) -> float | None:
    by_penalty_group = {
        (float(row["penalty"]), row["stage15_probe_type"]): row for row in metrics
    }
    penalties = sorted({float(row["penalty"]) for row in metrics})
    baseline = by_penalty_group.get((0.0, "temporal_mismatch"))
    if baseline is None:
        return None
    baseline_false = int(baseline["original_false_entitled_count"])
    target_false = 0.1 * baseline_false
    for penalty in penalties:
        temporal = by_penalty_group.get((penalty, "temporal_mismatch"))
        if temporal is None:
            continue
        if int(temporal["adjusted_false_entitled_count"]) > target_false:
            continue
        controls_preserved = True
        for group in CONTROL_GROUPS:
            row = by_penalty_group.get((penalty, group))
            if row is None or int(row["changed_count"]) != 0:
                controls_preserved = False
                break
        if controls_preserved:
            return penalty
    return None


def markdown_table(rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.3f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def summary_markdown(metrics: Sequence[dict[str, Any]], valid_penalty: float | None) -> str:
    key_groups = {
        "temporal_mismatch",
        "temporal_erased",
        "surface_control",
        "sufficiency_control",
    }
    rows = [row for row in metrics if row["stage15_probe_type"] in key_groups]
    columns = [
        "penalty",
        "stage15_probe_type",
        "n",
        "original_false_entitled_count",
        "adjusted_false_entitled_count",
        "changed_count",
        "temporal_mismatch_flag_count",
        "adjusted_accuracy",
    ]
    if valid_penalty is None:
        selection = "No valid soft penalty found under the Stage18-A selection rule."
    else:
        selection = f"Smallest valid soft penalty: {valid_penalty:.3f}."
    return "\n\n".join(
        [
            "# Stage18-A Temporal Penalty Sweep Summary",
            "Stage18-A is a diagnostic soft temporal comparator sweep, not a trained model result.",
            "## Smallest valid penalty",
            selection,
            "## Key group metrics",
            markdown_table(rows, columns),
            "## Selection rule",
            "A valid penalty must reduce temporal_mismatch false-entitled count by at least 90% while making zero prediction changes to temporal_erased, surface_control, and sufficiency_control.",
        ]
    ) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", default=DEFAULT_INPUT_GLOB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = load_penalty_files(args.input_glob)
    metrics = group_metrics(rows)
    valid_penalty = smallest_valid_penalty(metrics)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "stage18_temporal_penalty_sweep_group_metrics.csv", metrics)
    write_csv(args.output_dir / "stage18_temporal_penalty_sweep_examples.csv", rows)
    (args.output_dir / "stage18_temporal_penalty_sweep_summary.md").write_text(
        summary_markdown(metrics, valid_penalty),
        encoding="utf-8",
    )
    print("STAGE18_TEMPORAL_PENALTY_SWEEP_ANALYSIS")
    print(f"rows\t{len(rows)}")
    print(f"group_metric_rows\t{len(metrics)}")
    print(
        "smallest_valid_penalty\t"
        + ("NONE" if valid_penalty is None else f"{valid_penalty:.6g}")
    )
    print(f"group_metrics\t{args.output_dir / 'stage18_temporal_penalty_sweep_group_metrics.csv'}")
    print(f"examples\t{args.output_dir / 'stage18_temporal_penalty_sweep_examples.csv'}")
    print(f"summary\t{args.output_dir / 'stage18_temporal_penalty_sweep_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

