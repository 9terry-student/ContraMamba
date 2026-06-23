"""Run Stage18-B2 temporal-bias calibration across prediction seeds.

This is prediction-level calibration robustness analysis. It does not train a
model; it calibrates one scalar temporal alpha per prediction file.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.calibrate_stage18_temporal_bias import (  # noqa: E402
    DEFAULT_GRID,
    CONTROL_GROUPS,
    assign_splits,
    load_rows,
    materialize_rows,
    metrics,
    parse_grid,
    select_alpha,
)


DEFAULT_PROBE = ROOT / "data" / "stage15_slot_sensitivity_probe.jsonl"
DEFAULT_PRED_GLOB = str(ROOT / "results" / "stage15_v5_slot_sensitivity_seed*_v5_seed*_preds.json")
DEFAULT_OUTPUT_DIR = ROOT / "results"


def infer_seed(path: Path, fallback: int) -> int:
    matches = re.findall(r"seed(\d+)", path.name)
    return int(matches[-1]) if matches else fallback


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


def summarize_group(rows: Sequence[dict[str, Any]], split: str, group: str) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row.get("split") == split and row.get("stage15_probe_type") == group
    ]
    if not selected:
        return {
            "original_false_entitled_count": 0,
            "adjusted_false_entitled_count": 0,
            "original_accuracy": 0.0,
            "adjusted_accuracy": 0.0,
            "changed_count": 0,
        }
    original_correct = sum(
        row["original_pred_final_label"] == row["gold_final_label"]
        for row in selected
    )
    adjusted_correct = sum(
        row["adjusted_pred_final_label"] == row["gold_final_label"]
        for row in selected
    )
    original_false = sum(
        row["gold_final_label"] == "NOT_ENTITLED"
        and row["original_pred_final_label"] in {"REFUTE", "SUPPORT"}
        for row in selected
    )
    adjusted_false = sum(
        row["gold_final_label"] == "NOT_ENTITLED"
        and row["adjusted_pred_final_label"] in {"REFUTE", "SUPPORT"}
        for row in selected
    )
    changed = sum(
        row["original_pred_final_label"] != row["adjusted_pred_final_label"]
        for row in selected
    )
    n = len(selected)
    return {
        "original_false_entitled_count": original_false,
        "adjusted_false_entitled_count": adjusted_false,
        "original_accuracy": original_correct / n if n else 0.0,
        "adjusted_accuracy": adjusted_correct / n if n else 0.0,
        "changed_count": changed,
    }


def controls_preserved(rows: Sequence[dict[str, Any]]) -> bool:
    for group in CONTROL_GROUPS:
        selected = [row for row in rows if row.get("stage15_probe_type") == group]
        if any(row["original_pred_final_label"] != row["adjusted_pred_final_label"] for row in selected):
            return False
    return True


def run_one(
    *,
    prediction_file: Path,
    seed: int,
    probe: Path,
    output_dir: Path,
    split_seed: int,
    calibration_frac: float,
    alpha_grid: Sequence[float],
    objective: str,
    eps: float,
    regularization: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = load_rows(probe, prediction_file, eps)
    assign_splits(rows, seed=split_seed, calibration_frac=calibration_frac)
    selected_alpha, grid_rows, valid_found = select_alpha(
        rows,
        alpha_grid,
        objective=objective,
        regularization=regularization,
    )
    materialized = materialize_rows(rows, selected_alpha)
    calibration_temporal = summarize_group(materialized, "calibration", "temporal_mismatch")
    heldout_temporal = summarize_group(materialized, "heldout", "temporal_mismatch")
    summary = {
        "seed": seed,
        "prediction_file": str(prediction_file),
        "selected_alpha": selected_alpha,
        "objective": objective,
        "controls_preserved": controls_preserved(materialized),
        "calibration_temporal_original_false_entitled": calibration_temporal["original_false_entitled_count"],
        "calibration_temporal_adjusted_false_entitled": calibration_temporal["adjusted_false_entitled_count"],
        "heldout_temporal_original_false_entitled": heldout_temporal["original_false_entitled_count"],
        "heldout_temporal_adjusted_false_entitled": heldout_temporal["adjusted_false_entitled_count"],
        "calibration_temporal_accuracy_before": calibration_temporal["original_accuracy"],
        "calibration_temporal_accuracy_after": calibration_temporal["adjusted_accuracy"],
        "heldout_temporal_accuracy_before": heldout_temporal["original_accuracy"],
        "heldout_temporal_accuracy_after": heldout_temporal["adjusted_accuracy"],
        "temporal_erased_changed_count": summarize_group(materialized, "calibration", "temporal_erased")["changed_count"]
        + summarize_group(materialized, "heldout", "temporal_erased")["changed_count"],
        "surface_control_changed_count": summarize_group(materialized, "calibration", "surface_control")["changed_count"]
        + summarize_group(materialized, "heldout", "surface_control")["changed_count"],
        "sufficiency_control_changed_count": summarize_group(materialized, "calibration", "sufficiency_control")["changed_count"]
        + summarize_group(materialized, "heldout", "sufficiency_control")["changed_count"],
        "total_changed_count": sum(
            row["original_pred_final_label"] != row["adjusted_pred_final_label"]
            for row in materialized
        ),
        "valid_guarded_alpha_found": valid_found,
    }
    payload = {
        "metadata": {
            "seed": seed,
            "prediction_file": str(prediction_file),
            "selected_alpha": selected_alpha,
            "objective": objective,
            "calibration_frac": calibration_frac,
            "split_seed": split_seed,
            "alpha_grid": list(alpha_grid),
            "valid_guarded_alpha_found": valid_found,
            "flag_counts_by_group": dict(
                Counter(
                    row["stage15_probe_type"]
                    for row in rows
                    if row["temporal_mismatch_flag"]
                )
            ),
            "note": "Stage18-B2 is prediction-level temporal-bias calibration robustness analysis.",
        },
        "predictions": materialized,
    }
    output_path = output_dir / f"stage18_temporal_bias_calibrated_seed{seed}.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    alpha_grid_rows = []
    for row in grid_rows:
        item = dict(row)
        item["seed"] = seed
        item["prediction_file"] = str(prediction_file)
        alpha_grid_rows.append(item)
    return summary, alpha_grid_rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--pred-glob", default=DEFAULT_PRED_GLOB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--calibration-frac", type=float, default=0.5)
    parser.add_argument("--alpha-grid", type=parse_grid, default=parse_grid(DEFAULT_GRID))
    parser.add_argument(
        "--objective",
        choices=("cross_entropy", "accuracy", "false_entitlement_guarded_accuracy"),
        default="false_entitlement_guarded_accuracy",
    )
    parser.add_argument("--regularization", type=float, default=0.0)
    parser.add_argument("--eps", type=float, default=1e-8)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = [Path(path) for path in sorted(glob.glob(args.pred_glob))]
    if not paths:
        raise FileNotFoundError(f"no prediction files matched: {args.pred_glob}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    alpha_rows: list[dict[str, Any]] = []
    for index, path in enumerate(paths, start=1):
        seed = infer_seed(path, fallback=index)
        summary, alpha_grid_rows = run_one(
            prediction_file=path,
            seed=seed,
            probe=args.probe,
            output_dir=args.output_dir,
            split_seed=args.split_seed,
            calibration_frac=args.calibration_frac,
            alpha_grid=args.alpha_grid,
            objective=args.objective,
            eps=args.eps,
            regularization=args.regularization,
        )
        summary_rows.append(summary)
        alpha_rows.extend(alpha_grid_rows)
    write_csv(args.output_dir / "stage18_temporal_bias_multiseed_summary.csv", summary_rows)
    write_csv(args.output_dir / "stage18_temporal_bias_multiseed_alpha_grid.csv", alpha_rows)
    print("STAGE18_TEMPORAL_BIAS_MULTISEED")
    print(f"prediction_files_found\t{len(paths)}")
    for row in summary_rows:
        print(
            "seed\t{seed}\tselected_alpha\t{selected_alpha}\t"
            "heldout_temporal_false_entitled\t{heldout_temporal_original_false_entitled}->{heldout_temporal_adjusted_false_entitled}\t"
            "controls_preserved\t{controls_preserved}".format(**row)
        )
    print(f"summary_csv\t{args.output_dir / 'stage18_temporal_bias_multiseed_summary.csv'}")
    print(f"alpha_grid_csv\t{args.output_dir / 'stage18_temporal_bias_multiseed_alpha_grid.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

