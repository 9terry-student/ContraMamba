"""Analyze Stage19-A combined temporal-bias + frame-supervised patch outputs."""

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
DEFAULT_ADJUSTED_GLOB = str(ROOT / "results" / "stage19_combined_temporal_frame_seed*.json")
DEFAULT_OUTPUT_DIR = ROOT / "results"
ENTITLED = {"REFUTE", "SUPPORT"}
KEY_GROUPS = (
    "temporal_mismatch",
    "frame_location_mismatch",
    "frame_role_mismatch",
    "predicate_mismatch",
    "temporal_erased",
    "surface_control",
    "sufficiency_control",
)
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


def load_payloads(paths: Sequence[Path]) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    payloads = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payloads.append((payload.get("metadata", {}), payload.get("predictions", [])))
    return payloads


def group_metrics(payloads: Sequence[tuple[dict[str, Any], list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    metrics = []
    for metadata, rows in payloads:
        seed = int(metadata.get("seed", 0))
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row.get("stage15_probe_type"))].append(row)
        for group in sorted(grouped):
            metrics.append(summarize(seed, grouped[group]))
    return metrics


def summary_rows(metrics: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metrics:
        if row["stage15_probe_type"] in KEY_GROUPS:
            grouped[row["stage15_probe_type"]].append(row)
    output = []
    for group in KEY_GROUPS:
        items = grouped.get(group, [])
        adjusted_false = [float(row["adjusted_false_entitled_count"]) for row in items]
        changed = [float(row["changed_count"]) for row in items]
        output.append(
            {
                "stage15_probe_type": group,
                "seeds": len(items),
                "adjusted_false_entitled_mean": statistics.fmean(adjusted_false) if adjusted_false else 0.0,
                "adjusted_false_entitled_std": sample_std(adjusted_false),
                "changed_count_mean": statistics.fmean(changed) if changed else 0.0,
                "changed_count_std": sample_std(changed),
            }
        )
    return output


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


def interpretation(summary: Sequence[dict[str, Any]], metrics: Sequence[dict[str, Any]]) -> list[str]:
    by_group = {row["stage15_probe_type"]: row for row in summary}
    temporal_ok = by_group.get("temporal_mismatch", {}).get("adjusted_false_entitled_mean", 999) <= 1
    frame_ok = (
        by_group.get("frame_location_mismatch", {}).get("adjusted_false_entitled_mean", 999) <= 5
        and by_group.get("frame_role_mismatch", {}).get("adjusted_false_entitled_mean", 999) <= 5
    )
    predicate_bad = by_group.get("predicate_mismatch", {}).get("adjusted_false_entitled_mean", 0) >= 10
    controls_preserved = all(
        row["stage15_probe_type"] not in CONTROL_GROUPS or int(row["changed_count"]) == 0
        for row in metrics
    )
    return [
        f"Temporal and frame fixes compatible: {temporal_ok and frame_ok}.",
        f"Controls preserved: {controls_preserved}.",
        f"Predicate remains a failure mode: {predicate_bad}.",
        "If temporal_mismatch is near zero and frame_location/frame_role remain low, the fixes are compatible.",
        "If predicate_mismatch remains high, Stage20 should implement a predicate guard.",
        "Stage19-A is a diagnostic post-processing analysis, not a trained model.",
    ]


def write_summary_md(path: Path, summary: Sequence[dict[str, Any]], metrics: Sequence[dict[str, Any]]) -> None:
    md = "\n\n".join(
        [
            "# Stage19-A Combined Temporal Bias + Frame-Supervised Diagnostic",
            "## Summary metrics",
            markdown_table(
                summary,
                [
                    "stage15_probe_type",
                    "seeds",
                    "adjusted_false_entitled_mean",
                    "adjusted_false_entitled_std",
                    "changed_count_mean",
                    "changed_count_std",
                ],
            ),
            "## Interpretation",
            "\n".join(f"- {line}" for line in interpretation(summary, metrics)),
        ]
    )
    path.write_text(md + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adjusted-glob", default=DEFAULT_ADJUSTED_GLOB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = [Path(path) for path in sorted(glob.glob(args.adjusted_glob))]
    if not paths:
        raise FileNotFoundError(f"no adjusted files matched: {args.adjusted_glob}")
    payloads = load_payloads(paths)
    metrics = group_metrics(payloads)
    summary = summary_rows(metrics)
    examples = []
    for _metadata, rows in payloads:
        examples.extend(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "stage19_combined_temporal_frame_group_metrics.csv", metrics)
    write_csv(args.output_dir / "stage19_combined_temporal_frame_summary.csv", summary)
    write_csv(args.output_dir / "stage19_combined_temporal_frame_examples.csv", examples)
    write_summary_md(args.output_dir / "stage19_combined_temporal_frame_summary.md", summary, metrics)
    print("STAGE19_COMBINED_TEMPORAL_FRAME_ANALYSIS")
    print(f"adjusted_files\t{len(paths)}")
    print(f"group_metrics\t{args.output_dir / 'stage19_combined_temporal_frame_group_metrics.csv'}")
    print(f"summary_csv\t{args.output_dir / 'stage19_combined_temporal_frame_summary.csv'}")
    print(f"summary_md\t{args.output_dir / 'stage19_combined_temporal_frame_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

