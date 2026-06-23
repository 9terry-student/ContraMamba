"""Calibrate a learnable Stage18-B1 temporal bias from prediction JSON.

This is prediction-level calibration, not end-to-end model training. It learns a
single scalar alpha on a calibration split and applies the Stage17 temporal
comparator bias to heldout predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.apply_stage17_temporal_comparator import (  # noqa: E402
    LABELS,
    compare_temporal,
    load_jsonl,
    softmax,
)


DEFAULT_PROBE = ROOT / "data" / "stage15_slot_sensitivity_probe.jsonl"
DEFAULT_OUTPUT = ROOT / "results" / "stage18_temporal_bias_calibrated_seed1.json"
DEFAULT_GRID = "0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0,4.0"
ENTITLED = {"REFUTE", "SUPPORT"}
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
CONTROL_GROUPS = {"temporal_erased", "surface_control", "sufficiency_control"}


def parse_grid(value: str) -> list[float]:
    grid = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not grid:
        raise argparse.ArgumentTypeError("alpha grid cannot be empty")
    if any(alpha < 0 for alpha in grid):
        raise argparse.ArgumentTypeError("alpha values must be non-negative")
    return grid


def prediction_logits(pred: dict[str, Any], eps: float) -> list[float]:
    logits = pred.get("logits") or pred.get("final_logits")
    if logits is not None:
        return [float(x) for x in logits]
    probs = pred.get("final_probs")
    if probs is None:
        raise ValueError(f"prediction {pred.get('id')} has no logits or final_probs")
    return [math.log(max(float(prob), eps)) for prob in probs]


def apply_alpha(logits: Sequence[float], *, alpha: float, flag: int) -> list[float]:
    adjusted = [float(x) for x in logits]
    if flag and alpha:
        adjusted[0] -= alpha
        adjusted[1] += alpha
        adjusted[2] -= alpha
    return adjusted


def row_with_probe(pred: dict[str, Any], probe: dict[str, Any], eps: float) -> dict[str, Any]:
    flag, status, claim_exprs, evidence_exprs = compare_temporal(
        probe["claim"], probe["evidence"]
    )
    return {
        "id": pred["id"],
        "pair_id": probe.get("pair_id") or pred.get("pair_id"),
        "stage15_source_id": probe.get("stage15_source_id") or pred["id"],
        "stage15_probe_type": probe.get("stage15_probe_type"),
        "temporal_mismatch_flag": flag,
        "temporal_comparator_status": status,
        "original_pred_final_label": pred.get("pred_final_label"),
        "gold_final_label": pred.get("gold_final_label")
        or probe.get("final_label")
        or probe.get("label"),
        "original_final_probs": pred.get("final_probs"),
        "_base_logits": prediction_logits(pred, eps),
        "claim_temporal_expressions": claim_exprs,
        "evidence_temporal_expressions": evidence_exprs,
    }


def load_rows(probe_path: Path, preds_path: Path, eps: float) -> list[dict[str, Any]]:
    probe_by_id = {row["id"]: row for row in load_jsonl(probe_path)}
    payload = json.loads(preds_path.read_text(encoding="utf-8"))
    rows = []
    for pred in payload.get("predictions", []):
        probe = probe_by_id.get(pred.get("id"))
        if probe is not None:
            rows.append(row_with_probe(pred, probe, eps))
    if not rows:
        raise ValueError("no prediction rows matched probe ids")
    return rows


def assign_splits(rows: list[dict[str, Any]], *, seed: int, calibration_frac: float) -> None:
    if not 0 < calibration_frac < 1:
        raise ValueError("--calibration-frac must be between 0 and 1")
    keys = sorted({str(row.get("stage15_source_id") or row["id"]) for row in rows})
    rng = random.Random(seed)
    rng.shuffle(keys)
    calibration_count = max(1, min(len(keys) - 1, round(len(keys) * calibration_frac)))
    calibration = set(keys[:calibration_count])
    for row in rows:
        key = str(row.get("stage15_source_id") or row["id"])
        row["split"] = "calibration" if key in calibration else "heldout"


def adjusted_prediction(row: dict[str, Any], alpha: float) -> tuple[str, list[float]]:
    logits = apply_alpha(
        row["_base_logits"],
        alpha=alpha,
        flag=int(row["temporal_mismatch_flag"]),
    )
    probs = softmax(logits)
    return LABELS[max(range(3), key=lambda index: probs[index])], probs


def metrics(rows: Sequence[dict[str, Any]], alpha: float) -> dict[str, Any]:
    n = len(rows)
    ce = 0.0
    correct = 0
    original_false = 0
    adjusted_false = 0
    changed = 0
    for row in rows:
        pred, probs = adjusted_prediction(row, alpha)
        gold = row["gold_final_label"]
        gold_id = LABEL_TO_ID[gold]
        ce -= math.log(max(probs[gold_id], 1e-12))
        correct += int(pred == gold)
        original_false += int(gold == "NOT_ENTITLED" and row["original_pred_final_label"] in ENTITLED)
        adjusted_false += int(gold == "NOT_ENTITLED" and pred in ENTITLED)
        changed += int(pred != row["original_pred_final_label"])
    return {
        "n": n,
        "cross_entropy": ce / n if n else 0.0,
        "accuracy": correct / n if n else 0.0,
        "original_false_entitled_count": original_false,
        "adjusted_false_entitled_count": adjusted_false,
        "changed_count": changed,
    }


def grouped_alpha_metrics(rows: Sequence[dict[str, Any]], alpha: float) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("stage15_probe_type"))].append(row)
    return {group: metrics(items, alpha) for group, items in grouped.items()}


def valid_guarded_alpha(grouped: dict[str, dict[str, Any]]) -> bool:
    temporal = grouped.get("temporal_mismatch")
    if temporal is None:
        return False
    baseline = temporal["original_false_entitled_count"]
    if temporal["adjusted_false_entitled_count"] > 0.1 * baseline:
        return False
    for group in CONTROL_GROUPS:
        item = grouped.get(group)
        if item is None or item["changed_count"] != 0:
            return False
    return True


def select_alpha(
    rows: Sequence[dict[str, Any]],
    alpha_grid: Sequence[float],
    *,
    objective: str,
    regularization: float,
) -> tuple[float, list[dict[str, Any]], bool]:
    calibration = [row for row in rows if row["split"] == "calibration"]
    grid_rows: list[dict[str, Any]] = []
    valid_found = False
    for alpha in alpha_grid:
        overall = metrics(calibration, alpha)
        grouped = grouped_alpha_metrics(calibration, alpha)
        is_valid = valid_guarded_alpha(grouped)
        valid_found = valid_found or is_valid
        grid_rows.append(
            {
                "alpha": alpha,
                "objective": objective,
                "calibration_cross_entropy": overall["cross_entropy"],
                "calibration_objective_value": overall["cross_entropy"] + regularization * alpha * alpha,
                "calibration_accuracy": overall["accuracy"],
                "calibration_changed_count": overall["changed_count"],
                "calibration_temporal_original_false_entitled": grouped.get("temporal_mismatch", {}).get("original_false_entitled_count", 0),
                "calibration_temporal_adjusted_false_entitled": grouped.get("temporal_mismatch", {}).get("adjusted_false_entitled_count", 0),
                "valid_guarded_alpha": is_valid,
            }
        )
    if objective == "cross_entropy":
        selected = min(grid_rows, key=lambda row: (row["calibration_objective_value"], row["alpha"]))
    elif objective == "accuracy":
        selected = max(grid_rows, key=lambda row: (row["calibration_accuracy"], -row["alpha"]))
    elif objective == "false_entitlement_guarded_accuracy":
        valid = [row for row in grid_rows if row["valid_guarded_alpha"]]
        selected = min(valid, key=lambda row: row["alpha"]) if valid else max(
            grid_rows, key=lambda row: (row["calibration_accuracy"], -row["alpha"])
        )
    else:
        raise ValueError(f"unknown objective: {objective}")
    return float(selected["alpha"]), grid_rows, valid_found


def materialize_rows(rows: Sequence[dict[str, Any]], selected_alpha: float) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        pred, probs = adjusted_prediction(row, selected_alpha)
        output.append(
            {
                "id": row["id"],
                "split": row["split"],
                "stage15_probe_type": row.get("stage15_probe_type"),
                "temporal_mismatch_flag": row["temporal_mismatch_flag"],
                "temporal_comparator_status": row["temporal_comparator_status"],
                "original_pred_final_label": row["original_pred_final_label"],
                "adjusted_pred_final_label": pred,
                "gold_final_label": row["gold_final_label"],
                "original_final_probs": row["original_final_probs"],
                "adjusted_final_probs": probs,
                "alpha": selected_alpha,
                "selected_alpha": selected_alpha,
                "claim_temporal_expressions": row["claim_temporal_expressions"],
                "evidence_temporal_expressions": row["evidence_temporal_expressions"],
            }
        )
    return output


def write_alpha_grid_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--preds", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--calibration-frac", type=float, default=0.5)
    parser.add_argument("--alpha-grid", type=parse_grid, default=parse_grid(DEFAULT_GRID))
    parser.add_argument(
        "--objective",
        choices=("cross_entropy", "accuracy", "false_entitlement_guarded_accuracy"),
        default="cross_entropy",
    )
    parser.add_argument("--regularization", type=float, default=0.0)
    parser.add_argument("--eps", type=float, default=1e-8)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = load_rows(args.probe, args.preds, args.eps)
    assign_splits(rows, seed=args.split_seed, calibration_frac=args.calibration_frac)
    selected_alpha, grid_rows, valid_found = select_alpha(
        rows,
        args.alpha_grid,
        objective=args.objective,
        regularization=args.regularization,
    )
    materialized = materialize_rows(rows, selected_alpha)
    heldout = [row for row in materialized if row["split"] == "heldout"]
    calibration = [row for row in materialized if row["split"] == "calibration"]
    metadata = {
        "selected_alpha": selected_alpha,
        "objective": args.objective,
        "calibration_frac": args.calibration_frac,
        "split_seed": args.split_seed,
        "alpha_grid": list(args.alpha_grid),
        "regularization": args.regularization,
        "valid_guarded_alpha_found": valid_found,
        "calibration_metrics_per_alpha": grid_rows,
        "heldout_metrics_for_selected_alpha": metrics(
            [row for row in rows if row["split"] == "heldout"], selected_alpha
        ),
        "flag_counts_by_group": dict(
            Counter(
                row["stage15_probe_type"]
                for row in rows
                if row["temporal_mismatch_flag"]
            )
        ),
        "note": "Stage18-B1 is prediction-level calibration, not an end-to-end trained model.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"metadata": metadata, "predictions": materialized}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_alpha_grid_csv(args.output.parent / "stage18_temporal_bias_alpha_grid_metrics.csv", grid_rows)
    print("STAGE18_TEMPORAL_BIAS_CALIBRATION")
    print(f"rows\t{len(rows)}")
    print(f"calibration_rows\t{len(calibration)}")
    print(f"heldout_rows\t{len(heldout)}")
    print(f"selected_alpha\t{selected_alpha}")
    print(f"valid_guarded_alpha_found\t{valid_found}")
    print(f"flag_counts_by_group\t{json.dumps(metadata['flag_counts_by_group'], sort_keys=True)}")
    print(f"wrote\t{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

