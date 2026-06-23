"""Analyze Stage20-B soft predicate penalty sweep."""

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
DEFAULT_GLOB = str(ROOT / "results" / "stage20_predicate_penalty_lexical_predicate_p*.json")
DEFAULT_OUTPUT_DIR = ROOT / "results"
ENTITLED = {"REFUTE", "SUPPORT"}
CONTROL_GROUPS = (
    "surface_control",
    "temporal_erased",
    "sufficiency_control",
    "temporal_mismatch",
    "frame_location_mismatch",
    "frame_role_mismatch",
)


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def sample_std(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def dist(rows: Sequence[dict[str, Any]], key: str) -> str:
    return json.dumps(dict(sorted(Counter(row.get(key) for row in rows).items())), sort_keys=True)


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    gold_ne = sum(row["gold_final_label"] == "NOT_ENTITLED" for row in rows)
    base_correct = sum(row["base_pred_final_label"] == row["gold_final_label"] for row in rows)
    adjusted_correct = sum(row["predicate_penalty_adjusted_pred_final_label"] == row["gold_final_label"] for row in rows)
    base_false = sum(row["gold_final_label"] == "NOT_ENTITLED" and row["base_pred_final_label"] in ENTITLED for row in rows)
    adjusted_false = sum(row["gold_final_label"] == "NOT_ENTITLED" and row["predicate_penalty_adjusted_pred_final_label"] in ENTITLED for row in rows)
    return {
        "detector_mode": rows[0]["predicate_detector_mode"] if rows else "",
        "penalty": float(rows[0]["predicate_penalty"]) if rows else 0.0,
        "seed": rows[0].get("seed", "") if rows else "",
        "stage15_probe_type": rows[0].get("stage15_probe_type") if rows else "",
        "n": n,
        "base_accuracy": safe_div(base_correct, n),
        "adjusted_accuracy": safe_div(adjusted_correct, n),
        "base_false_entitled_count": base_false,
        "adjusted_false_entitled_count": adjusted_false,
        "base_false_entitled_rate": safe_div(base_false, gold_ne),
        "adjusted_false_entitled_rate": safe_div(adjusted_false, gold_ne),
        "changed_count": sum(row["base_pred_final_label"] != row["predicate_penalty_adjusted_pred_final_label"] for row in rows),
        "predicate_mismatch_flag_count": sum(int(row["predicate_mismatch_flag"]) for row in rows),
        "base_prediction_distribution": dist(rows, "base_pred_final_label"),
        "adjusted_prediction_distribution": dist(rows, "predicate_penalty_adjusted_pred_final_label"),
    }


def group_metrics(payloads: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float, Any, str], list[dict[str, Any]]] = defaultdict(list)
    for payload in payloads:
        for row in payload.get("predictions", []):
            grouped[
                (
                    row["predicate_detector_mode"],
                    float(row["predicate_penalty"]),
                    row.get("seed", ""),
                    row.get("stage15_probe_type", ""),
                )
            ].append(row)
    return [summarize(grouped[key]) for key in sorted(grouped)]


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


def summary_rows(metrics: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in metrics:
        grouped[(row["detector_mode"], float(row["penalty"]))].append(row)
    rows = []
    for (detector, penalty), items in sorted(grouped.items()):
        def vals(group: str, field: str) -> list[float]:
            return [float(row[field]) for row in items if row["stage15_probe_type"] == group]

        pred_adjusted = vals("predicate_mismatch", "adjusted_false_entitled_count")
        pred_changed = vals("predicate_mismatch", "changed_count")
        pred_base = vals("predicate_mismatch", "base_false_entitled_count")
        item = {
            "detector_mode": detector,
            "penalty": penalty,
            "predicate_base_false_entitled_mean": statistics.fmean(pred_base) if pred_base else 0.0,
            "predicate_base_false_entitled_std": sample_std(pred_base),
            "predicate_adjusted_false_entitled_mean": statistics.fmean(pred_adjusted) if pred_adjusted else 0.0,
            "predicate_adjusted_false_entitled_std": sample_std(pred_adjusted),
            "predicate_changed_count_mean": statistics.fmean(pred_changed) if pred_changed else 0.0,
            "predicate_changed_count_std": sample_std(pred_changed),
        }
        non_pred_flags = 0.0
        for group in CONTROL_GROUPS:
            changed = vals(group, "changed_count")
            flags = vals(group, "predicate_mismatch_flag_count")
            item[f"{group}_changed_count_mean"] = statistics.fmean(changed) if changed else 0.0
            item[f"{group}_changed_count_std"] = sample_std(changed)
            non_pred_flags += sum(flags)
        item["non_predicate_false_positive_flags"] = non_pred_flags
        rows.append(item)
    return rows


def smallest_valid(summary: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    by_detector: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary:
        by_detector[row["detector_mode"]].append(row)
    candidates = by_detector.get("lexical_predicate") or next(iter(by_detector.values()), [])
    for row in sorted(candidates, key=lambda item: float(item["penalty"])):
        base = float(row["predicate_base_false_entitled_mean"])
        adjusted = float(row["predicate_adjusted_false_entitled_mean"])
        if adjusted > 0.1 * base:
            continue
        if any(float(row[f"{group}_changed_count_mean"]) != 0 for group in CONTROL_GROUPS):
            continue
        return row
    return None


def markdown_table(rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        vals = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                value = f"{value:.3f}"
            vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def summary_md(summary: Sequence[dict[str, Any]], valid: dict[str, Any] | None) -> str:
    columns = [
        "detector_mode",
        "penalty",
        "predicate_base_false_entitled_mean",
        "predicate_adjusted_false_entitled_mean",
        "predicate_changed_count_mean",
        "surface_control_changed_count_mean",
        "temporal_erased_changed_count_mean",
        "sufficiency_control_changed_count_mean",
        "temporal_mismatch_changed_count_mean",
        "frame_location_mismatch_changed_count_mean",
        "frame_role_mismatch_changed_count_mean",
        "non_predicate_false_positive_flags",
    ]
    selection = (
        "No valid soft predicate penalty found."
        if valid is None
        else f"Smallest valid predicate penalty: {float(valid['penalty']):.3f} ({valid['detector_mode']})."
    )
    return "\n\n".join(
        [
            "# Stage20-B Soft Predicate Penalty Sweep",
            "Stage20-B is post-processing diagnostic analysis, not an end-to-end trained model.",
            "## Smallest valid predicate penalty",
            selection,
            "## Penalty sweep table",
            markdown_table(summary, columns),
            "## Interpretation guide",
            "- If lexical_predicate succeeds with unchanged controls, predicate correction does not require hard override.",
            "- If oracle succeeds but lexical fails, detector recall/confidence is the bottleneck.",
            "- If controls change, the guard is unsafe.",
        ]
    ) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adjusted-glob", default=DEFAULT_GLOB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = [Path(path) for path in sorted(glob.glob(args.adjusted_glob))]
    if not paths:
        raise FileNotFoundError(f"no adjusted files matched: {args.adjusted_glob}")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    metrics = group_metrics(payloads)
    summary = summary_rows(metrics)
    valid = smallest_valid(summary)
    examples = [row for payload in payloads for row in payload.get("predictions", [])]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "stage20_predicate_penalty_sweep_group_metrics.csv", metrics)
    write_csv(args.output_dir / "stage20_predicate_penalty_sweep_summary.csv", summary)
    write_csv(args.output_dir / "stage20_predicate_penalty_sweep_examples.csv", examples)
    (args.output_dir / "stage20_predicate_penalty_sweep_summary.md").write_text(
        summary_md(summary, valid),
        encoding="utf-8",
    )
    print("STAGE20_PREDICATE_PENALTY_SWEEP_ANALYSIS")
    print(f"adjusted_files\t{len(paths)}")
    print("smallest_valid_predicate_penalty\t" + ("NONE" if valid is None else str(valid["penalty"])))
    print(f"group_metrics\t{args.output_dir / 'stage20_predicate_penalty_sweep_group_metrics.csv'}")
    print(f"summary_csv\t{args.output_dir / 'stage20_predicate_penalty_sweep_summary.csv'}")
    print(f"summary_md\t{args.output_dir / 'stage20_predicate_penalty_sweep_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

