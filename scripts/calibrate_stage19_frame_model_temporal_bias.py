"""Calibrate temporal bias directly on Stage16 frame-model predictions.

Stage19-B is target-model prediction-level calibration: it learns one temporal
alpha per frame-supervised prediction file and evaluates whether temporal and
frame fixes are compatible without changing model architecture or training.
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
    assign_splits,
    load_rows,
    materialize_rows,
    metrics,
    parse_grid,
    select_alpha,
)


DEFAULT_PROBE = ROOT / "data" / "stage15_slot_sensitivity_probe.jsonl"
DEFAULT_PRED_GLOB = str(ROOT / "results" / "stage16_frame_only_slotloss_multiseed_v5_seed*_preds.json")
DEFAULT_OUTPUT_DIR = ROOT / "results"
DEFAULT_GRID = "0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0,4.0,5.0,6.0"
ENTITLED = {"REFUTE", "SUPPORT"}
CONTROL_GROUPS = ("temporal_erased", "surface_control", "sufficiency_control")


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


def group_rows(rows: Sequence[dict[str, Any]], group: str, split: str | None = None) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("stage15_probe_type") == group
        and (split is None or row.get("split") == split)
    ]


def false_entitled_count(rows: Sequence[dict[str, Any]], key: str = "adjusted_pred_final_label") -> int:
    return sum(
        row["gold_final_label"] == "NOT_ENTITLED" and row[key] in ENTITLED
        for row in rows
    )


def accuracy(rows: Sequence[dict[str, Any]], key: str) -> float:
    return (
        sum(row[key] == row["gold_final_label"] for row in rows) / len(rows)
        if rows
        else 0.0
    )


def changed_count(rows: Sequence[dict[str, Any]]) -> int:
    return sum(row["original_pred_final_label"] != row["adjusted_pred_final_label"] for row in rows)


def controls_preserved(rows: Sequence[dict[str, Any]]) -> bool:
    return all(changed_count(group_rows(rows, group)) == 0 for group in CONTROL_GROUPS)


def run_one(
    *,
    prediction_file: Path,
    seed: int,
    probe: Path,
    output_dir: Path,
    output_prefix: str,
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
    cal_temporal = group_rows(materialized, "temporal_mismatch", "calibration")
    held_temporal = group_rows(materialized, "temporal_mismatch", "heldout")
    frame_location = group_rows(materialized, "frame_location_mismatch")
    frame_role = group_rows(materialized, "frame_role_mismatch")
    predicate = group_rows(materialized, "predicate_mismatch")
    summary = {
        "seed": seed,
        "prediction_file": str(prediction_file),
        "selected_alpha": selected_alpha,
        "objective": objective,
        "valid_guarded_alpha_found": valid_found,
        "controls_preserved": controls_preserved(materialized),
        "calibration_temporal_original_false_entitled": false_entitled_count(cal_temporal, "original_pred_final_label"),
        "calibration_temporal_adjusted_false_entitled": false_entitled_count(cal_temporal),
        "heldout_temporal_original_false_entitled": false_entitled_count(held_temporal, "original_pred_final_label"),
        "heldout_temporal_adjusted_false_entitled": false_entitled_count(held_temporal),
        "calibration_temporal_accuracy_before": accuracy(cal_temporal, "original_pred_final_label"),
        "calibration_temporal_accuracy_after": accuracy(cal_temporal, "adjusted_pred_final_label"),
        "heldout_temporal_accuracy_before": accuracy(held_temporal, "original_pred_final_label"),
        "heldout_temporal_accuracy_after": accuracy(held_temporal, "adjusted_pred_final_label"),
        "frame_location_false_entitled": false_entitled_count(frame_location),
        "frame_role_false_entitled": false_entitled_count(frame_role),
        "predicate_false_entitled": false_entitled_count(predicate),
        "temporal_erased_changed_count": changed_count(group_rows(materialized, "temporal_erased")),
        "surface_control_changed_count": changed_count(group_rows(materialized, "surface_control")),
        "sufficiency_control_changed_count": changed_count(group_rows(materialized, "sufficiency_control")),
        "total_changed_count": changed_count(materialized),
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
            "note": "Stage19-B is target-model prediction-level temporal calibration.",
        },
        "predictions": materialized,
    }
    output_path = output_dir / f"{output_prefix}_seed{seed}.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    alpha_rows = []
    for row in grid_rows:
        item = dict(row)
        item["seed"] = seed
        item["prediction_file"] = str(prediction_file)
        alpha_rows.append(item)
    return summary, alpha_rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--pred-glob", default=DEFAULT_PRED_GLOB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", default="stage19_frame_model_temporal_bias")
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
        seed = infer_seed(path, index)
        summary, grid = run_one(
            prediction_file=path,
            seed=seed,
            probe=args.probe,
            output_dir=args.output_dir,
            output_prefix=args.output_prefix,
            split_seed=args.split_seed,
            calibration_frac=args.calibration_frac,
            alpha_grid=args.alpha_grid,
            objective=args.objective,
            eps=args.eps,
            regularization=args.regularization,
        )
        summary_rows.append(summary)
        alpha_rows.extend(grid)
    write_csv(args.output_dir / f"{args.output_prefix}_summary.csv", summary_rows)
    write_csv(args.output_dir / f"{args.output_prefix}_alpha_grid.csv", alpha_rows)
    print("STAGE19_FRAME_MODEL_TEMPORAL_BIAS_CALIBRATION")
    print(f"prediction_files_processed\t{len(paths)}")
    for row in summary_rows:
        print(
            "seed\t{seed}\tselected_alpha\t{selected_alpha}\t"
            "temporal_heldout\t{heldout_temporal_original_false_entitled}->{heldout_temporal_adjusted_false_entitled}\t"
            "controls_preserved\t{controls_preserved}\t"
            "predicate_false_entitled\t{predicate_false_entitled}".format(**row)
        )
    print(f"summary_csv\t{args.output_dir / (args.output_prefix + '_summary.csv')}")
    print(f"alpha_grid_csv\t{args.output_dir / (args.output_prefix + '_alpha_grid.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

