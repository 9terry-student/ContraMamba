"""Analyze Stage18-B2 temporal-bias calibration across seeds."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATED_GLOB = str(ROOT / "results" / "stage18_temporal_bias_calibrated_seed*.json")
DEFAULT_SUMMARY_CSV = ROOT / "results" / "stage18_temporal_bias_multiseed_summary.csv"
DEFAULT_OUTPUT_DIR = ROOT / "results"
ENTITLED = {"REFUTE", "SUPPORT"}


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def sample_std(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def distribution(rows: Sequence[dict[str, Any]], key: str) -> str:
    return json.dumps(dict(sorted(Counter(row.get(key) for row in rows).items())), sort_keys=True)


def load_json_predictions(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("metadata", {}), payload.get("predictions", [])


def summarize_group(seed: Any, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    gold_ne = sum(row["gold_final_label"] == "NOT_ENTITLED" for row in rows)
    original_correct = sum(row["original_pred_final_label"] == row["gold_final_label"] for row in rows)
    adjusted_correct = sum(row["adjusted_pred_final_label"] == row["gold_final_label"] for row in rows)
    original_false = sum(
        row["gold_final_label"] == "NOT_ENTITLED" and row["original_pred_final_label"] in ENTITLED
        for row in rows
    )
    adjusted_false = sum(
        row["gold_final_label"] == "NOT_ENTITLED" and row["adjusted_pred_final_label"] in ENTITLED
        for row in rows
    )
    return {
        "seed": seed,
        "split": rows[0].get("split") if rows else "",
        "stage15_probe_type": rows[0].get("stage15_probe_type") if rows else "",
        "n": n,
        "original_accuracy": safe_div(original_correct, n),
        "adjusted_accuracy": safe_div(adjusted_correct, n),
        "original_false_entitled_count": original_false,
        "adjusted_false_entitled_count": adjusted_false,
        "changed_count": sum(row["original_pred_final_label"] != row["adjusted_pred_final_label"] for row in rows),
        "temporal_mismatch_flag_count": sum(int(row["temporal_mismatch_flag"]) for row in rows),
        "original_prediction_distribution": distribution(rows, "original_pred_final_label"),
        "adjusted_prediction_distribution": distribution(rows, "adjusted_pred_final_label"),
    }


def group_metrics(payloads: Sequence[tuple[dict[str, Any], list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for metadata, rows in payloads:
        seed = metadata.get("seed")
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[(row.get("split"), row.get("stage15_probe_type"))].append(row)
        for key in sorted(grouped):
            metrics.append(summarize_group(seed, grouped[key]))
    return metrics


def read_summary_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
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


def summary_markdown(summary_rows: Sequence[dict[str, Any]], metrics: Sequence[dict[str, Any]]) -> str:
    alphas = [float(row["selected_alpha"]) for row in summary_rows if row.get("selected_alpha") not in {"", None}]
    heldout_adjusted = [
        float(row["heldout_temporal_adjusted_false_entitled"])
        for row in summary_rows
        if row.get("heldout_temporal_adjusted_false_entitled") not in {"", None}
    ]
    controls_all = all(str(row.get("controls_preserved")).lower() == "true" for row in summary_rows) if summary_rows else False
    aggregate_lines = [
        f"Selected alpha mean/std: {statistics.fmean(alphas):.3f} +/- {sample_std(alphas):.3f}" if alphas else "Selected alpha mean/std: unavailable",
        f"Heldout temporal adjusted false-entitled mean/std: {statistics.fmean(heldout_adjusted):.3f} +/- {sample_std(heldout_adjusted):.3f}" if heldout_adjusted else "Heldout temporal adjusted false-entitled mean/std: unavailable",
        f"Controls preserved in all seeds: {controls_all}",
        "Stage18-B2 is prediction-level calibration robustness analysis, not an end-to-end trained model.",
    ]
    summary_columns = [
        "seed",
        "selected_alpha",
        "heldout_temporal_original_false_entitled",
        "heldout_temporal_adjusted_false_entitled",
        "controls_preserved",
    ]
    metric_columns = [
        "seed",
        "split",
        "stage15_probe_type",
        "n",
        "original_false_entitled_count",
        "adjusted_false_entitled_count",
        "changed_count",
        "temporal_mismatch_flag_count",
    ]
    key_metrics = [
        row
        for row in metrics
        if row["stage15_probe_type"] in {"temporal_mismatch", "temporal_erased", "surface_control", "sufficiency_control"}
    ]
    return "\n\n".join(
        [
            "# Stage18-B2 Temporal Bias Multiseed Summary",
            "## Aggregate",
            "\n".join(f"- {line}" for line in aggregate_lines),
            "## Per-seed selected alpha and heldout temporal metrics",
            markdown_table(summary_rows, summary_columns),
            "## Key group metrics",
            markdown_table(key_metrics, metric_columns),
        ]
    ) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrated-glob", default=DEFAULT_CALIBRATED_GLOB)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = [Path(path) for path in sorted(glob.glob(args.calibrated_glob))]
    if not paths:
        raise FileNotFoundError(f"no calibrated files matched: {args.calibrated_glob}")
    payloads = [load_json_predictions(path) for path in paths]
    metrics = group_metrics(payloads)
    examples = []
    for _metadata, rows in payloads:
        examples.extend(rows)
    summary_rows = read_summary_csv(args.summary_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "stage18_temporal_bias_multiseed_group_metrics.csv", metrics)
    write_csv(args.output_dir / "stage18_temporal_bias_multiseed_examples.csv", examples)
    (args.output_dir / "stage18_temporal_bias_multiseed_summary.md").write_text(
        summary_markdown(summary_rows, metrics),
        encoding="utf-8",
    )
    print("STAGE18_TEMPORAL_BIAS_MULTISEED_ANALYSIS")
    print(f"calibrated_files\t{len(paths)}")
    print(f"group_metrics\t{args.output_dir / 'stage18_temporal_bias_multiseed_group_metrics.csv'}")
    print(f"examples\t{args.output_dir / 'stage18_temporal_bias_multiseed_examples.csv'}")
    print(f"summary\t{args.output_dir / 'stage18_temporal_bias_multiseed_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

