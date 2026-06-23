"""Analyze Stage19-B frame-model temporal-bias calibration."""

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
DEFAULT_GLOB = str(ROOT / "results" / "stage19_frame_model_temporal_bias_seed*.json")
DEFAULT_SUMMARY = ROOT / "results" / "stage19_frame_model_temporal_bias_summary.csv"
DEFAULT_OUTPUT_DIR = ROOT / "results"
ENTITLED = {"REFUTE", "SUPPORT"}
CONTROL_GROUPS = ("temporal_erased", "surface_control", "sufficiency_control")


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def sample_std(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def distribution(rows: Sequence[dict[str, Any]], key: str) -> str:
    return json.dumps(dict(sorted(Counter(row.get(key) for row in rows).items())), sort_keys=True)


def summarize(seed: int, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    gold_ne = sum(row["gold_final_label"] == "NOT_ENTITLED" for row in rows)
    original_correct = sum(row["original_pred_final_label"] == row["gold_final_label"] for row in rows)
    adjusted_correct = sum(row["adjusted_pred_final_label"] == row["gold_final_label"] for row in rows)
    original_false = sum(row["gold_final_label"] == "NOT_ENTITLED" and row["original_pred_final_label"] in ENTITLED for row in rows)
    adjusted_false = sum(row["gold_final_label"] == "NOT_ENTITLED" and row["adjusted_pred_final_label"] in ENTITLED for row in rows)
    return {
        "seed": seed,
        "split": rows[0].get("split") if rows else "",
        "stage15_probe_type": rows[0].get("stage15_probe_type") if rows else "",
        "n": n,
        "original_accuracy": safe_div(original_correct, n),
        "adjusted_accuracy": safe_div(adjusted_correct, n),
        "original_false_entitled_count": original_false,
        "adjusted_false_entitled_count": adjusted_false,
        "original_false_entitled_rate": safe_div(original_false, gold_ne),
        "adjusted_false_entitled_rate": safe_div(adjusted_false, gold_ne),
        "changed_count": sum(row["original_pred_final_label"] != row["adjusted_pred_final_label"] for row in rows),
        "temporal_mismatch_flag_count": sum(int(row["temporal_mismatch_flag"]) for row in rows),
        "original_prediction_distribution": distribution(rows, "original_pred_final_label"),
        "adjusted_prediction_distribution": distribution(rows, "adjusted_pred_final_label"),
    }


def group_metrics(payloads: Sequence[tuple[dict[str, Any], list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    metrics = []
    for metadata, rows in payloads:
        seed = int(metadata.get("seed", 0))
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[(row.get("split"), row.get("stage15_probe_type"))].append(row)
        for key in sorted(grouped):
            metrics.append(summarize(seed, grouped[key]))
    return metrics


def read_summary(path: Path) -> list[dict[str, Any]]:
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


def aggregate(summary_rows: Sequence[dict[str, Any]], key: str) -> tuple[float, float]:
    values = [float(row[key]) for row in summary_rows if row.get(key) not in {"", None}]
    return (statistics.fmean(values), sample_std(values)) if values else (0.0, 0.0)


def bool_all(summary_rows: Sequence[dict[str, Any]], key: str) -> bool:
    return bool(summary_rows) and all(str(row.get(key)).lower() == "true" for row in summary_rows)


def summary_md(summary_rows: Sequence[dict[str, Any]], metrics: Sequence[dict[str, Any]]) -> str:
    temporal_mean, temporal_std = aggregate(summary_rows, "heldout_temporal_adjusted_false_entitled")
    frame_loc_mean, _ = aggregate(summary_rows, "frame_location_false_entitled")
    frame_role_mean, _ = aggregate(summary_rows, "frame_role_false_entitled")
    pred_mean, pred_std = aggregate(summary_rows, "predicate_false_entitled")
    controls = bool_all(summary_rows, "controls_preserved")
    compatible = (
        temporal_mean <= 5
        and frame_role_mean <= 5
        and frame_loc_mean <= 10
        and controls
    )
    predicate_unresolved = pred_mean >= 30
    lines = [
        "# Stage19-B Frame-Model Temporal Bias Calibration",
        "",
        "Stage19-B is prediction-level target-model calibration, not an end-to-end trained model.",
        "",
        "## Per-seed summary",
        "",
        markdown_table(
            summary_rows,
            [
                "seed",
                "selected_alpha",
                "valid_guarded_alpha_found",
                "controls_preserved",
                "calibration_temporal_original_false_entitled",
                "calibration_temporal_adjusted_false_entitled",
                "heldout_temporal_original_false_entitled",
                "heldout_temporal_adjusted_false_entitled",
                "frame_location_false_entitled",
                "frame_role_false_entitled",
                "predicate_false_entitled",
            ],
        ),
        "",
        "## Aggregate interpretation",
        "",
        f"- Heldout temporal adjusted false-entitled mean/std: {temporal_mean:.3f} +/- {temporal_std:.3f}",
        f"- Predicate false-entitled mean/std: {pred_mean:.3f} +/- {pred_std:.3f}",
        f"- Temporal + frame fixes compatible by criterion: {compatible}",
        f"- Predicate remains dominant unresolved failure by criterion: {predicate_unresolved}",
        f"- Controls preserved in all seeds: {controls}",
        "",
        "## Group metrics",
        "",
        markdown_table(
            metrics,
            [
                "seed",
                "split",
                "stage15_probe_type",
                "n",
                "original_false_entitled_count",
                "adjusted_false_entitled_count",
                "changed_count",
                "temporal_mismatch_flag_count",
            ],
        ),
        "",
    ]
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrated-glob", default=DEFAULT_GLOB)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = [Path(path) for path in sorted(glob.glob(args.calibrated_glob))]
    if not paths:
        raise FileNotFoundError(f"no calibrated files matched: {args.calibrated_glob}")
    payloads = []
    examples = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payloads.append((payload.get("metadata", {}), payload.get("predictions", [])))
        examples.extend(payload.get("predictions", []))
    metrics = group_metrics(payloads)
    summary_rows = read_summary(args.summary_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "stage19_frame_model_temporal_bias_group_metrics.csv", metrics)
    write_csv(args.output_dir / "stage19_frame_model_temporal_bias_examples.csv", examples)
    (args.output_dir / "stage19_frame_model_temporal_bias_summary.md").write_text(
        summary_md(summary_rows, metrics),
        encoding="utf-8",
    )
    print("STAGE19_FRAME_MODEL_TEMPORAL_BIAS_ANALYSIS")
    print(f"calibrated_files\t{len(paths)}")
    print(f"group_metrics\t{args.output_dir / 'stage19_frame_model_temporal_bias_group_metrics.csv'}")
    print(f"examples\t{args.output_dir / 'stage19_frame_model_temporal_bias_examples.csv'}")
    print(f"summary_md\t{args.output_dir / 'stage19_frame_model_temporal_bias_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

