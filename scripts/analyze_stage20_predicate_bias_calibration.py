"""Analyze Stage20-C predicate bias calibration outputs."""

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
DEFAULT_GLOB = str(ROOT / "results" / "stage20_predicate_bias_calibrated_seed*.json")
DEFAULT_OUTPUT_DIR = ROOT / "results"
DEFAULT_SUMMARY_CSV = str(ROOT / "results" / "stage20_predicate_bias_summary.csv")

ENTITLED = {"REFUTE", "SUPPORT"}
CONTROL_GROUPS = (
    "surface_control",
    "temporal_erased",
    "sufficiency_control",
    "temporal_mismatch",
    "frame_location_mismatch",
    "frame_role_mismatch",
)


def safe_div(n: float, d: float) -> float:
    return n / d if d else 0.0


def sample_std(vals: Sequence[float]) -> float:
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def dist(rows: Sequence[dict[str, Any]], key: str) -> str:
    return json.dumps(dict(sorted(Counter(r.get(key) for r in rows).items())), sort_keys=True)


def summarize_group(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    gold_ne = sum(r["gold_final_label"] == "NOT_ENTITLED" for r in rows)
    base_correct = sum(r["base_pred_final_label"] == r["gold_final_label"] for r in rows)
    adj_correct = sum(r["adjusted_pred_final_label"] == r["gold_final_label"] for r in rows)
    base_fe = sum(r["gold_final_label"] == "NOT_ENTITLED" and r["base_pred_final_label"] in ENTITLED for r in rows)
    adj_fe = sum(r["gold_final_label"] == "NOT_ENTITLED" and r["adjusted_pred_final_label"] in ENTITLED for r in rows)
    return {
        "detector_mode": rows[0]["predicate_detector_mode"] if rows else "",
        "seed": rows[0].get("seed", "") if rows else "",
        "split": rows[0].get("split", "") if rows else "",
        "stage15_probe_type": rows[0].get("stage15_probe_type") if rows else "",
        "selected_alpha": rows[0].get("selected_alpha", "") if rows else "",
        "n": n,
        "base_accuracy": safe_div(base_correct, n),
        "adjusted_accuracy": safe_div(adj_correct, n),
        "base_false_entitled_count": base_fe,
        "adjusted_false_entitled_count": adj_fe,
        "base_false_entitled_rate": safe_div(base_fe, gold_ne),
        "adjusted_false_entitled_rate": safe_div(adj_fe, gold_ne),
        "changed_count": sum(r["base_pred_final_label"] != r["adjusted_pred_final_label"] for r in rows),
        "predicate_mismatch_flag_count": sum(int(r["predicate_mismatch_flag"]) for r in rows),
        "base_prediction_distribution": dist(rows, "base_pred_final_label"),
        "adjusted_prediction_distribution": dist(rows, "adjusted_pred_final_label"),
    }


def group_metrics(payloads: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for payload in payloads:
        alpha = payload["metadata"].get("stage20c_selected_alpha", 0.0)
        for row in payload.get("predictions", []):
            key = (
                row.get("predicate_detector_mode", ""),
                row.get("seed", ""),
                row.get("split", ""),
                row.get("stage15_probe_type", ""),
            )
            grouped[key].append({**row, "selected_alpha": alpha})
    return [summarize_group(grouped[k]) for k in sorted(grouped)]


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summary_rows(
    payloads: Sequence[dict[str, Any]],
    metrics: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for payload in payloads:
        meta = payload.get("metadata", {})
        seed = meta.get("seed", "")
        alpha = meta.get("stage20c_selected_alpha", "")
        valid = meta.get("stage20c_valid_guarded_alpha_found", None)

        def _vals(split: str, group: str, field: str) -> list[float]:
            return [
                float(m[field])
                for m in metrics
                if m["seed"] == seed and m["split"] == split and m["stage15_probe_type"] == group
            ]

        cal_base = _vals("calibration", "predicate_mismatch", "base_false_entitled_count")
        cal_adj = _vals("calibration", "predicate_mismatch", "adjusted_false_entitled_count")
        heldout_base = _vals("heldout", "predicate_mismatch", "base_false_entitled_count")
        heldout_adj = _vals("heldout", "predicate_mismatch", "adjusted_false_entitled_count")

        controls_preserved = all(
            sum(_vals("heldout", g, "changed_count")) == 0 for g in CONTROL_GROUPS
        )

        # detector recall: flagged vs total predicate examples on heldout
        heldout_pred_rows = [
            r for r in payload.get("predictions", [])
            if r.get("split") == "heldout" and r.get("stage15_probe_type") == "predicate_mismatch"
        ]
        flagged = sum(int(r["predicate_mismatch_flag"]) for r in heldout_pred_rows)
        total_pred = len(heldout_pred_rows)

        rows.append(
            {
                "seed": seed,
                "detector_mode": meta.get("stage20c_detector_mode", ""),
                "selected_alpha": alpha,
                "valid_guarded_alpha_found": valid,
                "calibration_pred_base_false_entitled": sum(cal_base),
                "calibration_pred_adjusted_false_entitled": sum(cal_adj),
                "heldout_pred_base_false_entitled": sum(heldout_base),
                "heldout_pred_adjusted_false_entitled": sum(heldout_adj),
                "heldout_controls_preserved": controls_preserved,
                "heldout_detector_flagged": flagged,
                "heldout_pred_n": total_pred,
                "heldout_detector_recall": flagged / total_pred if total_pred else 0.0,
            }
        )
    return rows


def markdown_table(rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        cells = []
        for col in columns:
            v = row.get(col, "")
            if isinstance(v, float):
                v = f"{v:.3f}"
            cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_summary_md(
    path: Path,
    summary: Sequence[dict[str, Any]],
    metrics: Sequence[dict[str, Any]],
) -> None:
    sum_cols = [
        "seed", "selected_alpha", "valid_guarded_alpha_found",
        "calibration_pred_base_false_entitled", "calibration_pred_adjusted_false_entitled",
        "heldout_pred_base_false_entitled", "heldout_pred_adjusted_false_entitled",
        "heldout_controls_preserved", "heldout_detector_recall",
    ]
    metric_cols = [
        "seed", "split", "stage15_probe_type",
        "n", "base_accuracy", "adjusted_accuracy",
        "base_false_entitled_count", "adjusted_false_entitled_count",
        "changed_count",
    ]
    interp: list[str] = []
    for row in summary:
        alpha = row["selected_alpha"]
        valid = row["valid_guarded_alpha_found"]
        recall = float(row.get("heldout_detector_recall", 0))
        hb = row["heldout_pred_base_false_entitled"]
        ha = row["heldout_pred_adjusted_false_entitled"]
        ctrl = row["heldout_controls_preserved"]
        seed = row["seed"]
        interp.append(f"seed={seed}: selected_alpha={alpha}, valid={valid}")
        if not valid:
            interp.append(f"  WARNING: no valid guarded alpha found for seed={seed}; best-accuracy alpha used.")
        if recall < 1.0:
            interp.append(f"  Detector recall bottleneck: {recall:.1%} of heldout predicate_mismatch flagged; residual false-entitled reflects unflagged examples.")
        if not ctrl:
            interp.append(f"  WARNING: heldout control groups were changed for seed={seed}.")
        if hb > 0:
            pct = 100.0 * (hb - ha) / hb
            interp.append(f"  Heldout predicate false-entitled: {hb} -> {ha} ({pct:.1f}% reduction).")
        else:
            interp.append(f"  Heldout predicate false-entitled: {hb} -> {ha} (base already 0).")

    md = "\n\n".join(
        [
            "# Stage20-C Predicate Bias Calibration Summary",
            "Stage20-C is prediction-level predicate bias calibration, not an end-to-end trained model.",
            "## Per-seed summary",
            markdown_table(summary, sum_cols),
            "## Group metrics (selected alpha)",
            markdown_table(
                [m for m in metrics if m["split"] in ("calibration", "heldout")],
                metric_cols,
            ),
            "## Interpretation",
            "\n".join(f"- {line}" for line in interp),
        ]
    )
    path.write_text(md + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrated-glob", default=DEFAULT_GLOB)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = [Path(p) for p in sorted(glob.glob(args.calibrated_glob))]
    if not paths:
        raise FileNotFoundError(f"no calibrated files matched: {args.calibrated_glob}")
    payloads = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    metrics = group_metrics(payloads)
    summary = summary_rows(payloads, metrics)
    examples = [r for payload in payloads for r in payload.get("predictions", [])]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "stage20_predicate_bias_group_metrics.csv", metrics)
    write_csv(args.summary_csv, summary)
    write_csv(args.output_dir / "stage20_predicate_bias_examples.csv", examples)
    write_summary_md(
        args.output_dir / "stage20_predicate_bias_calibration_summary.md",
        summary,
        metrics,
    )
    print("STAGE20_PREDICATE_BIAS_CALIBRATION_ANALYSIS")
    print(f"calibrated_files\t{len(paths)}")
    for row in summary:
        print(
            f"seed={row['seed']}\tselected_alpha={row['selected_alpha']}"
            f"\tvalid={row['valid_guarded_alpha_found']}"
            f"\theldout_pred_fe={row['heldout_pred_base_false_entitled']}"
            f"->{row['heldout_pred_adjusted_false_entitled']}"
            f"\tcontrols_preserved={row['heldout_controls_preserved']}"
        )
    print(f"group_metrics\t{args.output_dir / 'stage20_predicate_bias_group_metrics.csv'}")
    print(f"summary_csv\t{args.summary_csv}")
    print(f"summary_md\t{args.output_dir / 'stage20_predicate_bias_calibration_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
