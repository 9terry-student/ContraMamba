"""Analyze Stage17 temporal-comparator adjusted predictions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADJUSTED = ROOT / "results" / "stage17_temporal_comparator_predictions.json"
DEFAULT_OUTPUT_DIR = ROOT / "results"
ENTITLED = {"REFUTE", "SUPPORT"}


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def prediction_distribution(rows: Sequence[dict[str, Any]], key: str) -> str:
    return json.dumps(dict(sorted(Counter(row.get(key) for row in rows).items())), sort_keys=True)


def group_key(row: dict[str, Any]) -> str:
    return str(row.get("stage15_probe_type") or row.get("stage14_probe_type") or "UNKNOWN")


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
    original_false = [
        row
        for row in rows
        if row.get("gold_final_label") == "NOT_ENTITLED"
        and row.get("original_pred_final_label") in ENTITLED
    ]
    adjusted_false = [
        row
        for row in rows
        if row.get("gold_final_label") == "NOT_ENTITLED"
        and row.get("adjusted_pred_final_label") in ENTITLED
    ]
    changed = [
        row
        for row in rows
        if row.get("original_pred_final_label") != row.get("adjusted_pred_final_label")
    ]
    return {
        "group": group_key(rows[0]) if rows else "",
        "n": n,
        "original_accuracy": safe_div(original_correct, n),
        "adjusted_accuracy": safe_div(adjusted_correct, n),
        "original_false_entitled_count": len(original_false),
        "adjusted_false_entitled_count": len(adjusted_false),
        "original_false_entitled_rate": safe_div(
            len(original_false),
            sum(row.get("gold_final_label") == "NOT_ENTITLED" for row in rows),
        ),
        "adjusted_false_entitled_rate": safe_div(
            len(adjusted_false),
            sum(row.get("gold_final_label") == "NOT_ENTITLED" for row in rows),
        ),
        "changed_count": len(changed),
        "original_prediction_distribution": prediction_distribution(
            rows,
            "original_pred_final_label",
        ),
        "adjusted_prediction_distribution": prediction_distribution(
            rows,
            "adjusted_pred_final_label",
        ),
        "temporal_mismatch_flag_count": sum(
            int(row.get("temporal_mismatch_flag", 0)) for row in rows
        ),
    }


def group_metrics(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[group_key(row)].append(row)
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


def interpretation(metrics: Sequence[dict[str, Any]]) -> list[str]:
    by_group = {row["group"]: row for row in metrics}
    lines = [
        "Stage17 is a diagnostic post-processing upper bound, not a trained model result.",
    ]
    temporal = by_group.get("temporal_mismatch")
    erased = by_group.get("temporal_erased")
    surface = by_group.get("surface_control")
    if temporal:
        lines.append(
            "Temporal mismatch flags: "
            f"{temporal['temporal_mismatch_flag_count']}/{temporal['n']}; "
            "adjusted false-entitled "
            f"{temporal['adjusted_false_entitled_count']} vs original "
            f"{temporal['original_false_entitled_count']}."
        )
    if erased:
        lines.append(
            "Temporal-erased flags: "
            f"{erased['temporal_mismatch_flag_count']}/{erased['n']}."
        )
    if surface:
        lines.append(
            "Surface-control flags: "
            f"{surface['temporal_mismatch_flag_count']}/{surface['n']}."
        )
    lines.extend(
        [
            "If hard_override reduces temporal_mismatch false entitlement while preserving temporal_erased and surface_control, explicit temporal comparison is sufficient at diagnostic upper-bound level.",
            "If temporal_erased or surface_control are harmed, the extractor is too broad.",
            "If temporal_mismatch flags are low, the extractor is too narrow and should be audited.",
        ]
    )
    return lines


def write_summary(path: Path, metrics: Sequence[dict[str, Any]]) -> None:
    columns = [
        "group",
        "n",
        "original_accuracy",
        "adjusted_accuracy",
        "original_false_entitled_count",
        "adjusted_false_entitled_count",
        "original_false_entitled_rate",
        "adjusted_false_entitled_rate",
        "changed_count",
        "temporal_mismatch_flag_count",
    ]
    md = "\n\n".join(
        [
            "# Stage17 Temporal Comparator Summary",
            "## Group metrics",
            markdown_table(metrics, columns),
            "## Interpretation",
            "\n".join(f"- {line}" for line in interpretation(metrics)),
        ]
    )
    path.write_text(md + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adjusted-preds", type=Path, default=DEFAULT_ADJUSTED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = json.loads(args.adjusted_preds.read_text(encoding="utf-8"))
    rows = payload.get("predictions", [])
    if not rows:
        raise ValueError(f"no predictions found in {args.adjusted_preds}")
    metrics = group_metrics(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "stage17_temporal_comparator_group_metrics.csv", metrics)
    write_csv(args.output_dir / "stage17_temporal_comparator_examples.csv", rows)
    write_summary(args.output_dir / "stage17_temporal_comparator_summary.md", metrics)
    print("STAGE17_TEMPORAL_COMPARATOR_ANALYSIS")
    print(f"rows\t{len(rows)}")
    print(f"groups\t{len(metrics)}")
    print(f"group_metrics\t{args.output_dir / 'stage17_temporal_comparator_group_metrics.csv'}")
    print(f"examples\t{args.output_dir / 'stage17_temporal_comparator_examples.csv'}")
    print(f"summary\t{args.output_dir / 'stage17_temporal_comparator_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

