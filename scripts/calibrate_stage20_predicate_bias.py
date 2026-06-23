"""Stage20-C: calibrate predicate bias alpha from a calibration split.

Reuses Stage20-A/B lexical predicate detector. Splits examples into
calibration/heldout, sweeps alpha on the calibration half, selects the
smallest alpha that satisfies the guarded-accuracy objective, then applies
it to the full set and writes per-seed calibrated JSON.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.apply_stage17_temporal_comparator import LABELS, load_jsonl, softmax  # noqa: E402
from scripts.apply_stage20_predicate_guard import (  # noqa: E402
    base_prediction,
    base_probs,
    predicate_flag,
)

DEFAULT_PROBE = ROOT / "data" / "stage15_slot_sensitivity_probe.jsonl"
DEFAULT_PRED_GLOB = str(ROOT / "results" / "stage19_frame_model_temporal_bias_seed*.json")
DEFAULT_OUTPUT_DIR = ROOT / "results"
DEFAULT_ALPHA_GRID = "0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0,4.0,5.0,6.0"
EPS = 1e-8

CONTROL_GROUPS = (
    "surface_control",
    "temporal_erased",
    "sufficiency_control",
    "temporal_mismatch",
    "frame_location_mismatch",
    "frame_role_mismatch",
)
ENTITLED = {"REFUTE", "SUPPORT"}


def parse_alpha_grid(value: str) -> list[float]:
    values = [float(x.strip()) for x in value.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("alpha grid cannot be empty")
    return values


def _stable_hash(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def calibration_split(
    ids: Sequence[str],
    probe_by_id: dict[str, dict[str, Any]],
    *,
    calibration_frac: float,
    split_seed: int,
) -> tuple[set[str], set[str]]:
    """Split by stage15_source_id if available, else by id."""
    source_id_of: dict[str, str] = {}
    for id_ in ids:
        probe = probe_by_id.get(id_, {})
        source_id_of[id_] = str(probe.get("stage15_source_id") or id_)

    unique_sources = sorted(set(source_id_of.values()))
    rng_base = split_seed * 2654435761
    shuffled = sorted(unique_sources, key=lambda s: (_stable_hash(s) ^ rng_base) % (2**32))
    n_cal = max(1, int(len(shuffled) * calibration_frac))
    cal_sources = set(shuffled[:n_cal])

    cal_ids: set[str] = set()
    heldout_ids: set[str] = set()
    for id_ in ids:
        if source_id_of[id_] in cal_sources:
            cal_ids.add(id_)
        else:
            heldout_ids.add(id_)
    return cal_ids, heldout_ids


def apply_alpha(
    probs: Sequence[float] | None,
    *,
    flag: int,
    alpha: float,
) -> list[float] | None:
    if probs is None:
        return None
    if not flag or alpha == 0.0:
        return [float(x) for x in probs]
    logits = [math.log(max(float(p), EPS)) for p in probs]
    logits[0] -= alpha   # SUPPORT
    logits[1] += alpha   # NOT_ENTITLED
    logits[2] -= alpha   # REFUTE
    return softmax(logits)


def build_pred_rows(
    payload: dict[str, Any],
    probe_by_id: dict[str, dict[str, Any]],
    *,
    detector_mode: str,
    alpha: float,
    cal_ids: set[str],
    heldout_ids: set[str],
) -> list[dict[str, Any]]:
    seed = int(payload.get("metadata", {}).get("seed", 0))
    rows: list[dict[str, Any]] = []
    for pred in payload.get("predictions", []):
        id_ = pred.get("id", "")
        probe = probe_by_id.get(id_)
        if probe is None:
            continue
        if id_ in cal_ids:
            split = "calibration"
        elif id_ in heldout_ids:
            split = "heldout"
        else:
            continue
        flag, claim_fam, ev_fam, conflict, note = predicate_flag(
            probe=probe, detector_mode=detector_mode
        )
        probs = base_probs(pred)
        adj = apply_alpha(probs, flag=flag, alpha=alpha)
        adj_pred = (
            LABELS[max(range(3), key=lambda i: adj[i])]
            if adj is not None
            else base_prediction(pred)
        )
        gold = pred.get("gold_final_label") or probe.get("final_label") or probe.get("label")
        rows.append(
            {
                "id": id_,
                "seed": seed,
                "split": split,
                "stage15_probe_type": probe.get("stage15_probe_type"),
                "gold_final_label": gold,
                "base_pred_final_label": base_prediction(pred),
                "adjusted_pred_final_label": adj_pred,
                "predicate_mismatch_flag": flag,
                "predicate_detector_mode": detector_mode,
                "selected_alpha": alpha,
                "base_final_probs": probs,
                "adjusted_final_probs": adj,
                "claim_predicate_families": claim_fam,
                "evidence_predicate_families": ev_fam,
                "predicate_conflict_pair": conflict,
                "detector_notes": note,
            }
        )
    return rows


def split_rows(
    rows: Sequence[dict[str, Any]],
    split: str,
) -> list[dict[str, Any]]:
    return [r for r in rows if r["split"] == split]


def false_entitled_count(rows: Sequence[dict[str, Any]], pred_key: str) -> int:
    return sum(
        r["gold_final_label"] == "NOT_ENTITLED" and r[pred_key] in ENTITLED
        for r in rows
    )


def changed_count(rows: Sequence[dict[str, Any]]) -> int:
    return sum(r["base_pred_final_label"] != r["adjusted_pred_final_label"] for r in rows)


def select_alpha(
    payload: dict[str, Any],
    probe_by_id: dict[str, dict[str, Any]],
    *,
    detector_mode: str,
    alpha_grid: list[float],
    cal_ids: set[str],
    heldout_ids: set[str],
    objective: str,
) -> tuple[float, bool, list[dict[str, Any]]]:
    """Return (selected_alpha, valid_found, alpha_grid_rows_for_this_seed)."""
    grid_rows: list[dict[str, Any]] = []
    seed = int(payload.get("metadata", {}).get("seed", 0))

    best_alpha = alpha_grid[0]
    best_acc = -1.0
    valid_found = False

    for alpha in alpha_grid:
        rows = build_pred_rows(
            payload, probe_by_id,
            detector_mode=detector_mode, alpha=alpha,
            cal_ids=cal_ids, heldout_ids=heldout_ids,
        )
        cal = split_rows(rows, "calibration")
        pred_cal = [r for r in cal if r["stage15_probe_type"] == "predicate_mismatch"]

        base_fe = false_entitled_count(pred_cal, "base_pred_final_label")
        adj_fe = false_entitled_count(pred_cal, "adjusted_pred_final_label")
        reduction_ok = base_fe == 0 or adj_fe <= 0.1 * base_fe

        controls_ok = all(
            changed_count([r for r in cal if r["stage15_probe_type"] == g]) == 0
            for g in CONTROL_GROUPS
        )

        n_cal = len(cal)
        acc = sum(r["adjusted_pred_final_label"] == r["gold_final_label"] for r in cal) / n_cal if n_cal else 0.0

        grid_rows.append(
            {
                "seed": seed,
                "alpha": alpha,
                "calibration_pred_base_false_entitled": base_fe,
                "calibration_pred_adjusted_false_entitled": adj_fe,
                "calibration_controls_ok": controls_ok,
                "calibration_reduction_ok": reduction_ok,
                "calibration_accuracy": acc,
                "valid": reduction_ok and controls_ok,
            }
        )

        if reduction_ok and controls_ok and not valid_found:
            best_alpha = alpha
            valid_found = True

        if not valid_found and acc > best_acc:
            best_acc = acc
            best_alpha = alpha

    return best_alpha, valid_found, grid_rows


def calibrate_payload(
    payload: dict[str, Any],
    probe_by_id: dict[str, dict[str, Any]],
    *,
    detector_mode: str,
    alpha_grid: list[float],
    calibration_frac: float,
    split_seed: int,
    objective: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    all_ids = [
        pred["id"]
        for pred in payload.get("predictions", [])
        if pred.get("id") in probe_by_id
    ]
    cal_ids, heldout_ids = calibration_split(
        all_ids, probe_by_id,
        calibration_frac=calibration_frac,
        split_seed=split_seed,
    )

    selected_alpha, valid_found, grid_rows = select_alpha(
        payload, probe_by_id,
        detector_mode=detector_mode,
        alpha_grid=alpha_grid,
        cal_ids=cal_ids,
        heldout_ids=heldout_ids,
        objective=objective,
    )

    rows = build_pred_rows(
        payload, probe_by_id,
        detector_mode=detector_mode, alpha=selected_alpha,
        cal_ids=cal_ids, heldout_ids=heldout_ids,
    )

    seed = int(payload.get("metadata", {}).get("seed", 0))
    out_meta = {
        **payload.get("metadata", {}),
        "stage20c_detector_mode": detector_mode,
        "stage20c_selected_alpha": selected_alpha,
        "stage20c_valid_guarded_alpha_found": valid_found,
        "stage20c_calibration_frac": calibration_frac,
        "stage20c_split_seed": split_seed,
        "stage20c_objective": objective,
        "stage20c_alpha_grid": alpha_grid,
        "note": (
            "Stage20-C is prediction-level predicate bias calibration, "
            "not an end-to-end trained model."
        ),
    }
    return {"metadata": out_meta, "predictions": rows}, grid_rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--pred-glob", default=DEFAULT_PRED_GLOB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--detector-mode",
        choices=("oracle_probe_type", "lexical_predicate"),
        default="lexical_predicate",
    )
    parser.add_argument("--alpha-grid", type=parse_alpha_grid, default=parse_alpha_grid(DEFAULT_ALPHA_GRID))
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--calibration-frac", type=float, default=0.5)
    parser.add_argument(
        "--objective",
        choices=("false_entitlement_guarded_accuracy",),
        default="false_entitlement_guarded_accuracy",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = [Path(p) for p in sorted(glob.glob(args.pred_glob))]
    if not paths:
        raise FileNotFoundError(f"no prediction files matched: {args.pred_glob}")
    probe_by_id = {row["id"]: row for row in load_jsonl(args.probe)}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_grid_rows: list[dict[str, Any]] = []
    print("STAGE20_PREDICATE_BIAS_CALIBRATION")
    print(f"detector_mode\t{args.detector_mode}")
    print(f"objective\t{args.objective}")
    print(f"calibration_frac\t{args.calibration_frac}")
    print(f"split_seed\t{args.split_seed}")
    print(f"prediction_files\t{len(paths)}")

    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        calibrated, grid_rows = calibrate_payload(
            payload, probe_by_id,
            detector_mode=args.detector_mode,
            alpha_grid=args.alpha_grid,
            calibration_frac=args.calibration_frac,
            split_seed=args.split_seed,
            objective=args.objective,
        )
        seed = calibrated["metadata"].get("seed", len(all_grid_rows))
        out_path = args.output_dir / f"stage20_predicate_bias_calibrated_seed{seed}.json"
        out_path.write_text(
            json.dumps(calibrated, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        all_grid_rows.extend(grid_rows)
        alpha = calibrated["metadata"]["stage20c_selected_alpha"]
        valid = calibrated["metadata"]["stage20c_valid_guarded_alpha_found"]
        print(f"seed={seed}\tselected_alpha={alpha}\tvalid_guarded={valid}\tout={out_path.name}")

    # write alpha grid CSV
    import csv
    grid_path = args.output_dir / "stage20_predicate_bias_alpha_grid.csv"
    if all_grid_rows:
        keys = list(all_grid_rows[0].keys())
        with grid_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_grid_rows)
    print(f"alpha_grid_csv\t{grid_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
