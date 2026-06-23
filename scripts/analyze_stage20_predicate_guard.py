"""Analyze Stage20-A predicate guard outputs."""

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
DEFAULT_GLOB = str(ROOT / "results" / "stage20_predicate_guard_seed*_*.json")
DEFAULT_OUTPUT_DIR = ROOT / "results"
ENTITLED = {"REFUTE", "SUPPORT"}
KEY_CHANGED_GROUPS = (
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
    adjusted_correct = sum(row["predicate_adjusted_pred_final_label"] == row["gold_final_label"] for row in rows)
    base_false = sum(row["gold_final_label"] == "NOT_ENTITLED" and row["base_pred_final_label"] in ENTITLED for row in rows)
    adjusted_false = sum(row["gold_final_label"] == "NOT_ENTITLED" and row["predicate_adjusted_pred_final_label"] in ENTITLED for row in rows)
    return {
        "detector_mode": rows[0]["predicate_detector_mode"] if rows else "",
        "patch_mode": rows[0]["predicate_patch_mode"] if rows else "",
        "seed": rows[0].get("seed", "") if rows else "",
        "stage15_probe_type": rows[0].get("stage15_probe_type") if rows else "",
        "n": n,
        "base_accuracy": safe_div(base_correct, n),
        "predicate_adjusted_accuracy": safe_div(adjusted_correct, n),
        "base_false_entitled_count": base_false,
        "predicate_adjusted_false_entitled_count": adjusted_false,
        "base_false_entitled_rate": safe_div(base_false, gold_ne),
        "predicate_adjusted_false_entitled_rate": safe_div(adjusted_false, gold_ne),
        "changed_count": sum(row["base_pred_final_label"] != row["predicate_adjusted_pred_final_label"] for row in rows),
        "predicate_mismatch_flag_count": sum(int(row["predicate_mismatch_flag"]) for row in rows),
        "base_prediction_distribution": dist(rows, "base_pred_final_label"),
        "predicate_adjusted_prediction_distribution": dist(rows, "predicate_adjusted_pred_final_label"),
    }


def group_metrics(payloads: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, Any, str], list[dict[str, Any]]] = defaultdict(list)
    for payload in payloads:
        for row in payload.get("predictions", []):
            grouped[
                (
                    row["predicate_detector_mode"],
                    row["predicate_patch_mode"],
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
    keyed: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in metrics:
        keyed[(row["detector_mode"], row["patch_mode"])].append(row)
    output = []
    for (detector, patch), rows in sorted(keyed.items()):
        def values(group: str, field: str) -> list[float]:
            return [float(row[field]) for row in rows if row["stage15_probe_type"] == group]

        pred_false = values("predicate_mismatch", "predicate_adjusted_false_entitled_count")
        base_pred_false = values("predicate_mismatch", "base_false_entitled_count")
        non_pred_flags = [
            float(row["predicate_mismatch_flag_count"])
            for row in rows
            if row["stage15_probe_type"] != "predicate_mismatch"
        ]
        item = {
            "detector_mode": detector,
            "patch_mode": patch,
            "predicate_base_false_entitled_mean": statistics.fmean(base_pred_false) if base_pred_false else 0.0,
            "predicate_base_false_entitled_std": sample_std(base_pred_false),
            "predicate_adjusted_false_entitled_mean": statistics.fmean(pred_false) if pred_false else 0.0,
            "predicate_adjusted_false_entitled_std": sample_std(pred_false),
            "non_predicate_false_positive_flags": sum(non_pred_flags),
        }
        for group in KEY_CHANGED_GROUPS:
            changed = values(group, "changed_count")
            item[f"{group}_changed_count_mean"] = statistics.fmean(changed) if changed else 0.0
            item[f"{group}_changed_count_std"] = sample_std(changed)
        output.append(item)
    return output


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


def interpretation(summary: Sequence[dict[str, Any]]) -> list[str]:
    lines = []
    for row in summary:
        detector = row["detector_mode"]
        adjusted = row["predicate_adjusted_false_entitled_mean"]
        surface_changed = row.get("surface_control_changed_count_mean", 0)
        false_pos = row.get("non_predicate_false_positive_flags", 0)
        if detector == "oracle_probe_type" and adjusted <= 1 and surface_changed == 0:
            lines.append("Oracle hard predicate signal can correct predicate_mismatch without changing controls.")
        if detector == "lexical_predicate":
            if adjusted > 10 and surface_changed == 0:
                lines.append("Lexical predicate detector is conservative/safe but has limited recall; a learned predicate comparator is likely needed.")
            if surface_changed > 0:
                lines.append("Lexical predicate detector changed surface_control and is too broad.")
            if false_pos > 0:
                lines.append(f"Lexical detector produced non-predicate flags: {false_pos}.")
    lines.append("Stage20-A is diagnostic post-processing, not a trained model result.")
    return lines


def write_summary_md(path: Path, summary: Sequence[dict[str, Any]]) -> None:
    columns = [
        "detector_mode",
        "patch_mode",
        "predicate_base_false_entitled_mean",
        "predicate_adjusted_false_entitled_mean",
        "surface_control_changed_count_mean",
        "temporal_erased_changed_count_mean",
        "sufficiency_control_changed_count_mean",
        "temporal_mismatch_changed_count_mean",
        "frame_location_mismatch_changed_count_mean",
        "frame_role_mismatch_changed_count_mean",
        "non_predicate_false_positive_flags",
    ]
    md = "\n\n".join(
        [
            "# Stage20-A Predicate Guard Summary",
            "## Summary metrics",
            markdown_table(summary, columns),
            "## Interpretation",
            "\n".join(f"- {line}" for line in interpretation(summary)),
        ]
    )
    path.write_text(md + "\n", encoding="utf-8")


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
    examples = [row for payload in payloads for row in payload.get("predictions", [])]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "stage20_predicate_guard_group_metrics.csv", metrics)
    write_csv(args.output_dir / "stage20_predicate_guard_summary.csv", summary)
    write_csv(args.output_dir / "stage20_predicate_guard_examples.csv", examples)
    write_summary_md(args.output_dir / "stage20_predicate_guard_summary.md", summary)
    print("STAGE20_PREDICATE_GUARD_ANALYSIS")
    print(f"adjusted_files\t{len(paths)}")
    print(f"group_metrics\t{args.output_dir / 'stage20_predicate_guard_group_metrics.csv'}")
    print(f"summary_csv\t{args.output_dir / 'stage20_predicate_guard_summary.csv'}")
    print(f"summary_md\t{args.output_dir / 'stage20_predicate_guard_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

