"""Apply Stage18-A soft temporal logit/probability penalty sweep.

This is a diagnostic post-processing experiment. It reuses the Stage17
temporal comparator and tests whether finite soft penalties can recover the
hard-override temporal correction without changing model code or retraining.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
    expand_paths,
    load_jsonl,
    softmax,
)


DEFAULT_PROBE = ROOT / "data" / "stage15_slot_sensitivity_probe.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "results"
EPS = 1e-8


def parse_penalties(value: str) -> list[float]:
    penalties = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not penalties:
        raise argparse.ArgumentTypeError("at least one penalty is required")
    if any(penalty < 0 for penalty in penalties):
        raise argparse.ArgumentTypeError("penalties must be non-negative")
    return penalties


def penalty_tag(penalty: float) -> str:
    return f"{penalty:.6g}".replace("-", "m").replace(".", "p")


def prediction_logits(prediction: dict[str, Any]) -> list[float] | None:
    logits = prediction.get("logits") or prediction.get("final_logits")
    if logits is not None:
        return [float(x) for x in logits]
    probs = prediction.get("final_probs")
    if probs is None:
        return None
    return [math.log(max(float(prob), EPS)) for prob in probs]


def prob_penalty_adjustment(probs: Sequence[float], penalty: float) -> list[float]:
    adjusted = [float(x) for x in probs]
    if len(adjusted) != 3:
        raise ValueError("final_probs must be [REFUTE, NOT_ENTITLED, SUPPORT]")
    if penalty == 0:
        return adjusted
    transfer_fraction = 1.0 - math.exp(-penalty)
    entitled_mass = adjusted[0] + adjusted[2]
    transferred = transfer_fraction * entitled_mass
    adjusted[0] *= 1.0 - transfer_fraction
    adjusted[2] *= 1.0 - transfer_fraction
    adjusted[1] += transferred
    total = sum(adjusted)
    return [x / total for x in adjusted] if total else [0.0, 1.0, 0.0]


def adjusted_probs_for_penalty(
    prediction: dict[str, Any],
    *,
    penalty: float,
    flag: int,
    mode: str,
) -> list[float] | None:
    original_probs = prediction.get("final_probs")
    if not flag or penalty == 0:
        return [float(x) for x in original_probs] if original_probs is not None else None

    if mode == "pseudo_logit_penalty":
        logits = prediction_logits(prediction)
        if logits is None:
            raise ValueError(
                "pseudo_logit_penalty requires logits/final_logits or final_probs"
            )
        if len(logits) != 3:
            raise ValueError("logits/final_probs must contain 3 labels")
        adjusted_logits = list(logits)
        adjusted_logits[0] -= penalty
        adjusted_logits[1] += penalty
        adjusted_logits[2] -= penalty
        return softmax(adjusted_logits)

    if mode == "prob_penalty":
        if original_probs is None:
            raise ValueError("prob_penalty requires final_probs")
        return prob_penalty_adjustment(original_probs, penalty)

    raise ValueError(f"unsupported mode: {mode}")


def load_predictions(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for prediction in payload.get("predictions", []):
            row = dict(prediction)
            row["_prediction_file"] = str(path)
            rows.append(row)
    if not rows:
        raise ValueError("no prediction rows loaded")
    return rows


def build_rows_for_penalty(
    *,
    probe_by_id: dict[str, dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    penalty: float,
    mode: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prediction in predictions:
        probe = probe_by_id.get(prediction.get("id"))
        if probe is None:
            continue
        flag, status, claim_exprs, evidence_exprs = compare_temporal(
            probe["claim"],
            probe["evidence"],
        )
        adjusted_probs = adjusted_probs_for_penalty(
            prediction,
            penalty=penalty,
            flag=flag,
            mode=mode,
        )
        original_pred = prediction.get("pred_final_label")
        if adjusted_probs is None:
            adjusted_pred = original_pred
        else:
            adjusted_pred = LABELS[max(range(3), key=lambda index: adjusted_probs[index])]
        original_entitlement = prediction.get("entitlement_prob")
        adjusted_entitlement = original_entitlement
        if flag and original_entitlement is not None and penalty > 0:
            adjusted_entitlement = max(0.0, float(original_entitlement) * math.exp(-penalty))
        rows.append(
            {
                "id": prediction["id"],
                "pair_id": probe.get("pair_id") or prediction.get("pair_id"),
                "prediction_file": prediction.get("_prediction_file"),
                "stage15_probe_type": probe.get("stage15_probe_type"),
                "original_pred_final_label": original_pred,
                "adjusted_pred_final_label": adjusted_pred,
                "gold_final_label": prediction.get("gold_final_label")
                or probe.get("final_label")
                or probe.get("label"),
                "temporal_mismatch_flag": flag,
                "temporal_comparator_status": status,
                "original_final_probs": prediction.get("final_probs"),
                "adjusted_final_probs": adjusted_probs,
                "original_entitlement_prob": original_entitlement,
                "adjusted_entitlement_prob": adjusted_entitlement,
                "penalty": penalty,
                "claim_temporal_expressions": claim_exprs,
                "evidence_temporal_expressions": evidence_exprs,
            }
        )
    if not rows:
        raise ValueError("no prediction rows matched probe ids")
    return rows


def write_json(path: Path, metadata: dict[str, Any], rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"metadata": metadata, "predictions": list(rows)}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def write_raw_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = [
        "penalty",
        "id",
        "stage15_probe_type",
        "gold_final_label",
        "original_pred_final_label",
        "adjusted_pred_final_label",
        "temporal_mismatch_flag",
        "temporal_comparator_status",
        "original_entitlement_prob",
        "adjusted_entitlement_prob",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_flag_counts(rows: Sequence[dict[str, Any]]) -> None:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[str(row.get("stage15_probe_type"))] += int(row.get("temporal_mismatch_flag", 0))
    print(f"temporal_mismatch_flags_by_stage15_type\t{json.dumps(dict(sorted(counts.items())), sort_keys=True)}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--preds", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--penalties",
        type=parse_penalties,
        default=parse_penalties("0,0.25,0.5,1.0,1.5,2.0,3.0,4.0"),
    )
    parser.add_argument(
        "--mode",
        choices=("pseudo_logit_penalty", "prob_penalty"),
        default="pseudo_logit_penalty",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    probe_rows = load_jsonl(args.probe)
    probe_by_id = {row["id"]: row for row in probe_rows}
    prediction_paths = expand_paths(args.preds)
    predictions = load_predictions(prediction_paths)
    all_rows: list[dict[str, Any]] = []

    for penalty in args.penalties:
        rows = build_rows_for_penalty(
            probe_by_id=probe_by_id,
            predictions=predictions,
            penalty=penalty,
            mode=args.mode,
        )
        all_rows.extend(rows)
        output = args.output_dir / f"stage18_temporal_penalty_p{penalty_tag(penalty)}.json"
        write_json(
            output,
            {
                "probe": str(args.probe),
                "prediction_files": [str(path) for path in prediction_paths],
                "mode": args.mode,
                "penalty": penalty,
                "note": (
                    "Stage18-A is a diagnostic soft temporal comparator sweep, "
                    "not a trained model result."
                ),
            },
            rows,
        )

    write_raw_csv(args.output_dir / "stage18_temporal_penalty_sweep_raw.csv", all_rows)
    print("STAGE18_TEMPORAL_PENALTY_SWEEP_APPLIED")
    print(f"prediction_rows_per_penalty\t{len(all_rows) // len(args.penalties)}")
    print(f"penalties\t{','.join(str(x) for x in args.penalties)}")
    print_flag_counts(all_rows[: len(all_rows) // len(args.penalties)])
    print(f"wrote_raw\t{args.output_dir / 'stage18_temporal_penalty_sweep_raw.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

