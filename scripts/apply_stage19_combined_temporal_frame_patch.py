"""Apply Stage19-A combined temporal-bias patch to frame-supervised predictions.

This is a diagnostic post-processing stage. It applies calibrated temporal
biases to arbitrary prediction JSONs, typically Stage16 frame-supervised runs,
without modifying model architecture or training code.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
import sys
from collections import Counter
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
DEFAULT_ALPHA_CSV = ROOT / "results" / "stage18_temporal_bias_multiseed_summary.csv"
DEFAULT_OUTPUT_DIR = ROOT / "results"
EPS = 1e-8


def infer_seed(path: Path, fallback: int) -> int:
    matches = re.findall(r"seed(\d+)", path.name)
    return int(matches[-1]) if matches else fallback


def load_alpha_map(path: Path) -> dict[int, float]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    mapping = {}
    for row in rows:
        try:
            mapping[int(row["seed"])] = float(row["selected_alpha"])
        except (KeyError, TypeError, ValueError):
            continue
    return mapping


def prediction_logits(prediction: dict[str, Any]) -> list[float]:
    logits = prediction.get("logits") or prediction.get("final_logits")
    if logits is not None:
        return [float(x) for x in logits]
    probs = prediction.get("final_probs")
    if probs is None:
        raise ValueError(f"prediction {prediction.get('id')} lacks logits/final_probs")
    return [math.log(max(float(prob), EPS)) for prob in probs]


def adjusted_probs(prediction: dict[str, Any], *, alpha: float, flag: int, mode: str) -> list[float]:
    if mode != "pseudo_logit_penalty":
        raise ValueError(f"unsupported mode: {mode}")
    logits = prediction_logits(prediction)
    if flag:
        logits[0] -= alpha
        logits[1] += alpha
        logits[2] -= alpha
    return softmax(logits)


def apply_one(
    *,
    prediction_file: Path,
    seed: int,
    alpha: float,
    probe_by_id: dict[str, dict[str, Any]],
    mode: str,
) -> dict[str, Any]:
    payload = json.loads(prediction_file.read_text(encoding="utf-8"))
    output_rows: list[dict[str, Any]] = []
    for prediction in payload.get("predictions", []):
        probe = probe_by_id.get(prediction.get("id"))
        if probe is None:
            continue
        flag, status, claim_exprs, evidence_exprs = compare_temporal(
            probe["claim"],
            probe["evidence"],
        )
        probs = adjusted_probs(prediction, alpha=alpha, flag=flag, mode=mode)
        adjusted_pred = LABELS[max(range(3), key=lambda index: probs[index])]
        output_rows.append(
            {
                "id": prediction["id"],
                "seed": seed,
                "pair_id": probe.get("pair_id") or prediction.get("pair_id"),
                "stage15_probe_type": probe.get("stage15_probe_type"),
                "original_pred_final_label": prediction.get("pred_final_label"),
                "adjusted_pred_final_label": adjusted_pred,
                "gold_final_label": prediction.get("gold_final_label")
                or probe.get("final_label")
                or probe.get("label"),
                "temporal_mismatch_flag": flag,
                "temporal_comparator_status": status,
                "alpha": alpha,
                "original_final_probs": prediction.get("final_probs"),
                "adjusted_final_probs": probs,
                "claim_temporal_expressions": claim_exprs,
                "evidence_temporal_expressions": evidence_exprs,
            }
        )
    if not output_rows:
        raise ValueError(f"no predictions from {prediction_file} matched probe ids")
    return {
        "metadata": {
            "seed": seed,
            "prediction_file": str(prediction_file),
            "selected_alpha": alpha,
            "mode": mode,
            "flag_counts_by_stage15_probe_type": dict(
                Counter(
                    row["stage15_probe_type"]
                    for row in output_rows
                    if row["temporal_mismatch_flag"]
                )
            ),
            "note": (
                "Stage19-A is a combined temporal-bias + frame-supervised "
                "diagnostic patch, not a trained model."
            ),
        },
        "predictions": output_rows,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--pred-glob", required=True)
    parser.add_argument("--alpha-csv", type=Path, default=DEFAULT_ALPHA_CSV)
    parser.add_argument("--default-alpha", type=float, default=1.25)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", default="stage19_combined_temporal_frame")
    parser.add_argument("--mode", default="pseudo_logit_penalty")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    prediction_files = [Path(path) for path in sorted(glob.glob(args.pred_glob))]
    if not prediction_files:
        raise FileNotFoundError(f"no prediction files matched: {args.pred_glob}")
    probe_by_id = {row["id"]: row for row in load_jsonl(args.probe)}
    alpha_map = load_alpha_map(args.alpha_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    used_alphas: dict[int, float] = {}
    for index, path in enumerate(prediction_files, start=1):
        seed = infer_seed(path, fallback=index)
        alpha = alpha_map.get(seed, args.default_alpha)
        used_alphas[seed] = alpha
        payload = apply_one(
            prediction_file=path,
            seed=seed,
            alpha=alpha,
            probe_by_id=probe_by_id,
            mode=args.mode,
        )
        output_path = args.output_dir / f"{args.output_prefix}_seed{seed}.json"
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("STAGE19_COMBINED_TEMPORAL_FRAME_PATCH")
    print(f"prediction_files_processed\t{len(prediction_files)}")
    print(f"alphas_used\t{json.dumps(used_alphas, sort_keys=True)}")
    print(f"output_prefix\t{args.output_dir / args.output_prefix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

