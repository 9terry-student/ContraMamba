"""Sweep soft Stage20-B predicate penalties.

This diagnostic reuses Stage20-A predicate detectors and applies finite
pseudo-logit/probability penalties instead of hard override.
"""

from __future__ import annotations

import argparse
import csv
import glob
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
DEFAULT_PENALTIES = "0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0,4.0,5.0,6.0"
EPS = 1e-8


def parse_penalties(value: str) -> list[float]:
    penalties = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not penalties:
        raise argparse.ArgumentTypeError("penalty list cannot be empty")
    if any(penalty < 0 for penalty in penalties):
        raise argparse.ArgumentTypeError("penalties must be non-negative")
    return penalties


def penalty_tag(value: float) -> str:
    return f"{value:.6g}".replace("-", "m").replace(".", "p")


def pseudo_logits(probs: Sequence[float], eps: float) -> list[float]:
    return [math.log(max(float(prob), eps)) for prob in probs]


def prob_penalty(probs: Sequence[float], penalty: float) -> list[float]:
    adjusted = [float(x) for x in probs]
    transfer_fraction = 1.0 - math.exp(-penalty)
    entitled = adjusted[0] + adjusted[2]
    transfer = transfer_fraction * entitled
    adjusted[0] *= 1.0 - transfer_fraction
    adjusted[2] *= 1.0 - transfer_fraction
    adjusted[1] += transfer
    total = sum(adjusted)
    return [x / total for x in adjusted] if total else [0.0, 1.0, 0.0]


def adjusted_probs(
    probs: Sequence[float] | None,
    *,
    flag: int,
    penalty: float,
    mode: str,
    eps: float,
) -> list[float] | None:
    if probs is None:
        return None
    if not flag or penalty == 0:
        return [float(x) for x in probs]
    if mode == "pseudo_logit_penalty":
        logits = pseudo_logits(probs, eps)
        logits[0] -= penalty
        logits[1] += penalty
        logits[2] -= penalty
        return softmax(logits)
    if mode == "prob_penalty":
        return prob_penalty(probs, penalty)
    raise ValueError(f"unknown mode: {mode}")


def build_rows(
    *,
    payloads: Sequence[dict[str, Any]],
    probe_by_id: dict[str, dict[str, Any]],
    detector_mode: str,
    penalty: float,
    mode: str,
    eps: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        seed = payload.get("metadata", {}).get("seed", 0)
        for pred in payload.get("predictions", []):
            probe = probe_by_id.get(pred.get("id"))
            if probe is None:
                continue
            flag, claim_families, evidence_families, conflict, note = predicate_flag(
                probe=probe,
                detector_mode=detector_mode,
            )
            probs = base_probs(pred)
            adjusted = adjusted_probs(
                probs,
                flag=flag,
                penalty=penalty,
                mode=mode,
                eps=eps,
            )
            adjusted_pred = (
                LABELS[max(range(3), key=lambda index: adjusted[index])]
                if adjusted is not None
                else base_prediction(pred)
            )
            rows.append(
                {
                    "id": pred["id"],
                    "seed": seed,
                    "stage15_probe_type": probe.get("stage15_probe_type"),
                    "base_pred_final_label": base_prediction(pred),
                    "predicate_penalty_adjusted_pred_final_label": adjusted_pred,
                    "gold_final_label": pred.get("gold_final_label")
                    or probe.get("final_label")
                    or probe.get("label"),
                    "predicate_mismatch_flag": flag,
                    "predicate_detector_mode": detector_mode,
                    "predicate_penalty": penalty,
                    "base_final_probs": probs,
                    "predicate_penalty_adjusted_final_probs": adjusted,
                    "claim_predicate_families": claim_families,
                    "evidence_predicate_families": evidence_families,
                    "predicate_conflict_pair": conflict,
                    "detector_notes": note,
                }
            )
    if not rows:
        raise ValueError("no prediction rows matched probe ids")
    return rows


def write_json(path: Path, metadata: dict[str, Any], rows: Sequence[dict[str, Any]]) -> None:
    path.write_text(
        json.dumps({"metadata": metadata, "predictions": list(rows)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_raw_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = [
        "predicate_detector_mode",
        "predicate_penalty",
        "seed",
        "id",
        "stage15_probe_type",
        "gold_final_label",
        "base_pred_final_label",
        "predicate_penalty_adjusted_pred_final_label",
        "predicate_mismatch_flag",
        "predicate_conflict_pair",
        "detector_notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_diagnostics(rows: Sequence[dict[str, Any]]) -> None:
    counts = Counter(row["stage15_probe_type"] for row in rows if row["predicate_mismatch_flag"])
    non_predicate = sum(
        row["predicate_mismatch_flag"] and row["stage15_probe_type"] != "predicate_mismatch"
        for row in rows
    )
    print(f"predicate_flag_counts_by_stage15_probe_type\t{json.dumps(dict(sorted(counts.items())), sort_keys=True)}")
    print(f"non_predicate_false_positive_flags\t{non_predicate}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--pred-glob", default=DEFAULT_PRED_GLOB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", default="stage20_predicate_penalty")
    parser.add_argument(
        "--detector-mode",
        choices=("oracle_probe_type", "lexical_predicate"),
        default="lexical_predicate",
    )
    parser.add_argument("--penalties", type=parse_penalties, default=parse_penalties(DEFAULT_PENALTIES))
    parser.add_argument(
        "--mode",
        choices=("pseudo_logit_penalty", "prob_penalty"),
        default="pseudo_logit_penalty",
    )
    parser.add_argument("--eps", type=float, default=1e-8)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = [Path(path) for path in sorted(glob.glob(args.pred_glob))]
    if not paths:
        raise FileNotFoundError(f"no prediction files matched: {args.pred_glob}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    probe_by_id = {row["id"]: row for row in load_jsonl(args.probe)}
    all_rows: list[dict[str, Any]] = []
    for penalty in args.penalties:
        rows = build_rows(
            payloads=payloads,
            probe_by_id=probe_by_id,
            detector_mode=args.detector_mode,
            penalty=penalty,
            mode=args.mode,
            eps=args.eps,
        )
        all_rows.extend(rows)
        output = args.output_dir / f"{args.output_prefix}_{args.detector_mode}_p{penalty_tag(penalty)}.json"
        write_json(
            output,
            {
                "detector_mode": args.detector_mode,
                "mode": args.mode,
                "penalty": penalty,
                "prediction_files": [str(path) for path in paths],
                "note": "Stage20-B is a soft predicate penalty diagnostic, not a trained model.",
            },
            rows,
        )
    write_raw_csv(args.output_dir / "stage20_predicate_penalty_sweep_raw.csv", all_rows)
    print("STAGE20_PREDICATE_PENALTY_SWEEP")
    print(f"prediction_files_processed\t{len(paths)}")
    print(f"detector_mode\t{args.detector_mode}")
    print(f"penalties\t{','.join(str(x) for x in args.penalties)}")
    print_diagnostics(all_rows[: len(all_rows) // len(args.penalties)])
    print(f"raw_csv\t{args.output_dir / 'stage20_predicate_penalty_sweep_raw.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

