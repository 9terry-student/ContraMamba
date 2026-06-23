"""Apply a conservative Stage17 temporal comparator to prediction JSON files.

This is an evaluation-time diagnostic. It does not change the model, retrain
the encoder, or alter the original prediction file.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROBE = ROOT / "data" / "stage15_slot_sensitivity_probe.jsonl"
DEFAULT_OUTPUT = ROOT / "results" / "stage17_temporal_comparator_predictions.json"
LABELS = ["REFUTE", "NOT_ENTITLED", "SUPPORT"]
ENTITLED = {"REFUTE", "SUPPORT"}
WEEKDAYS = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)
MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
RELATIVE_EVENTIVE_WORDS = (
    "before",
    "after",
    "during",
    "previously",
    "later",
    "earlier",
    "upcoming",
    "planned",
    "completed",
    "launched",
    "will",
    "had",
)

TEMPORAL_PATTERN = re.compile(
    r"\b("
    + "|".join(
        re.escape(x.lower())
        for x in [*WEEKDAYS, *MONTHS, *RELATIVE_EVENTIVE_WORDS]
    )
    + r")\b",
    flags=re.IGNORECASE,
)
ANCHOR_PATTERN = re.compile(
    r"\b("
    + "|".join(re.escape(x.lower()) for x in [*WEEKDAYS, *MONTHS])
    + r")\b",
    flags=re.IGNORECASE,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def expand_paths(patterns: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(Path(match) for match in matches) if matches else paths.append(Path(pattern))
    return paths


def normalize_temporal_expression(value: str) -> str:
    return value.strip().lower()


def extract_temporal_expressions(text: str) -> list[str]:
    seen: set[str] = set()
    expressions: list[str] = []
    for match in TEMPORAL_PATTERN.finditer(text):
        value = normalize_temporal_expression(match.group(1))
        if value not in seen:
            seen.add(value)
            expressions.append(value)
    return expressions


def extract_temporal_anchors(text: str) -> list[str]:
    seen: set[str] = set()
    anchors: list[str] = []
    for match in ANCHOR_PATTERN.finditer(text):
        value = normalize_temporal_expression(match.group(1))
        if value not in seen:
            seen.add(value)
            anchors.append(value)
    return anchors


def compare_temporal(claim: str, evidence: str) -> tuple[int, str, list[str], list[str]]:
    claim_exprs = extract_temporal_expressions(claim)
    evidence_exprs = extract_temporal_expressions(evidence)
    claim_anchors = extract_temporal_anchors(claim)
    evidence_anchors = extract_temporal_anchors(evidence)
    # Be conservative: explicit weekday/month anchors dominate weaker eventive
    # words such as "launched". This avoids flagging predicate swaps where the
    # date is identical but one predicate is itself an eventive word.
    if claim_anchors and evidence_anchors:
        if set(claim_anchors) == set(evidence_anchors):
            return 0, "temporal_match", claim_exprs, evidence_exprs
        return 1, "temporal_mismatch", claim_exprs, evidence_exprs
    if not claim_exprs or not evidence_exprs:
        return 0, "insufficient_temporal_info", claim_exprs, evidence_exprs
    if set(claim_exprs) == set(evidence_exprs):
        return 0, "temporal_match", claim_exprs, evidence_exprs
    return 1, "temporal_mismatch", claim_exprs, evidence_exprs


def softmax(logits: Sequence[float]) -> list[float]:
    max_logit = max(logits)
    exps = [math.exp(float(x) - max_logit) for x in logits]
    total = sum(exps)
    return [x / total for x in exps]


def renormalize(probs: Sequence[float]) -> list[float]:
    clipped = [max(0.0, float(x)) for x in probs]
    total = sum(clipped)
    if total <= 0:
        return [0.0, 1.0, 0.0]
    return [x / total for x in clipped]


def adjusted_from_probs(
    final_probs: Sequence[float],
    *,
    support_penalty: float,
) -> tuple[str, list[float]]:
    probs = list(float(x) for x in final_probs)
    if len(probs) != 3:
        raise ValueError("final_probs must contain REFUTE, NOT_ENTITLED, SUPPORT")
    # Treat support_penalty as an odds/log-probability style penalty while
    # staying in probability space when logits are unavailable.
    probs[2] = probs[2] * math.exp(-support_penalty)
    adjusted = renormalize(probs)
    return LABELS[max(range(3), key=lambda index: adjusted[index])], adjusted


def adjusted_from_logits(
    logits: Sequence[float],
    *,
    support_penalty: float,
) -> tuple[str, list[float], list[float]]:
    adjusted_logits = list(float(x) for x in logits)
    if len(adjusted_logits) != 3:
        raise ValueError("logits must contain REFUTE, NOT_ENTITLED, SUPPORT")
    adjusted_logits[2] -= support_penalty
    probs = softmax(adjusted_logits)
    return LABELS[max(range(3), key=lambda index: probs[index])], adjusted_logits, probs


def merge_prediction(
    *,
    pred: dict[str, Any],
    probe: dict[str, Any],
    mode: str,
    support_penalty: float,
    entitlement_threshold: float,
    prediction_file: str,
) -> dict[str, Any]:
    flag, status, claim_exprs, evidence_exprs = compare_temporal(
        probe["claim"],
        probe["evidence"],
    )
    original_pred = pred.get("pred_final_label")
    adjusted_pred = original_pred
    adjusted_probs = pred.get("final_probs")
    adjusted_entitlement = pred.get("entitlement_prob")
    adjusted_logits = None

    if flag:
        if mode == "hard_override":
            if original_pred in ENTITLED:
                adjusted_pred = "NOT_ENTITLED"
                adjusted_entitlement = min(
                    float(adjusted_entitlement)
                    if adjusted_entitlement is not None
                    else entitlement_threshold,
                    entitlement_threshold,
                )
        elif mode == "prob_penalty":
            if pred.get("final_probs") is not None:
                adjusted_pred, adjusted_probs = adjusted_from_probs(
                    pred["final_probs"],
                    support_penalty=support_penalty,
                )
            if adjusted_entitlement is not None:
                adjusted_entitlement = max(
                    0.0,
                    float(adjusted_entitlement) * math.exp(-support_penalty),
                )
        elif mode == "logit_penalty":
            logits = pred.get("logits") or pred.get("final_logits")
            if logits is None:
                raise ValueError(
                    "mode=logit_penalty requires prediction logits/final_logits, "
                    f"but they are absent for id={pred.get('id')}"
                )
            adjusted_pred, adjusted_logits, adjusted_probs = adjusted_from_logits(
                logits,
                support_penalty=support_penalty,
            )
        else:
            raise ValueError(f"unsupported mode: {mode}")

    return {
        "id": pred["id"],
        "pair_id": probe.get("pair_id") or pred.get("pair_id"),
        "prediction_file": prediction_file,
        "original_pred_final_label": original_pred,
        "adjusted_pred_final_label": adjusted_pred,
        "gold_final_label": pred.get("gold_final_label")
        or probe.get("final_label")
        or probe.get("label"),
        "temporal_mismatch_flag": flag,
        "temporal_comparator_status": status,
        "claim_temporal_expressions": claim_exprs,
        "evidence_temporal_expressions": evidence_exprs,
        "original_entitlement_prob": pred.get("entitlement_prob"),
        "adjusted_entitlement_prob": adjusted_entitlement,
        "original_final_probs": pred.get("final_probs"),
        "adjusted_final_probs": adjusted_probs,
        "adjusted_logits": adjusted_logits,
        "claim": probe.get("claim"),
        "evidence": probe.get("evidence"),
        "stage14_probe_type": probe.get("stage14_probe_type"),
        "stage15_probe_type": probe.get("stage15_probe_type"),
        "stage15_source_id": probe.get("stage15_source_id"),
        "stage15_original_probe_type": probe.get("stage15_original_probe_type"),
        "source_intervention_type": probe.get("source_intervention_type"),
        "stage15_expected_behavior": probe.get("stage15_expected_behavior"),
    }


def apply_comparator(
    *,
    probe_rows: Sequence[dict[str, Any]],
    prediction_paths: Sequence[Path],
    mode: str,
    support_penalty: float,
    entitlement_threshold: float,
) -> dict[str, Any]:
    probe_by_id = {row["id"]: row for row in probe_rows}
    outputs: list[dict[str, Any]] = []
    skipped_missing_probe = 0
    for path in prediction_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for pred in payload.get("predictions", []):
            probe = probe_by_id.get(pred.get("id"))
            if probe is None:
                skipped_missing_probe += 1
                continue
            outputs.append(
                merge_prediction(
                    pred=pred,
                    probe=probe,
                    mode=mode,
                    support_penalty=support_penalty,
                    entitlement_threshold=entitlement_threshold,
                    prediction_file=str(path),
                )
            )
    if not outputs:
        raise ValueError("no prediction rows matched probe ids")
    return {
        "metadata": {
            "probe": str(DEFAULT_PROBE),
            "prediction_files": [str(path) for path in prediction_paths],
            "mode": mode,
            "support_penalty": support_penalty,
            "entitlement_threshold": entitlement_threshold,
            "skipped_missing_probe": skipped_missing_probe,
            "note": (
                "Stage17 temporal comparator is a diagnostic post-processing "
                "upper bound, not a trained model result."
            ),
        },
        "predictions": outputs,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--preds", nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--mode",
        choices=("hard_override", "logit_penalty", "prob_penalty"),
        default="hard_override",
    )
    parser.add_argument("--support-penalty", type=float, default=2.0)
    parser.add_argument("--entitlement-threshold", type=float, default=0.5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    probe_rows = load_jsonl(args.probe)
    prediction_paths = expand_paths(args.preds)
    payload = apply_comparator(
        probe_rows=probe_rows,
        prediction_paths=prediction_paths,
        mode=args.mode,
        support_penalty=args.support_penalty,
        entitlement_threshold=args.entitlement_threshold,
    )
    payload["metadata"]["probe"] = str(args.probe)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    counts = {}
    for row in payload["predictions"]:
        key = row.get("stage15_probe_type") or "UNKNOWN"
        counts[key] = counts.get(key, 0) + int(row["temporal_mismatch_flag"])
    print("STAGE17_TEMPORAL_COMPARATOR_APPLIED")
    print(f"rows\t{len(payload['predictions'])}")
    print(f"mode\t{args.mode}")
    print(f"temporal_mismatch_flags_by_stage15_type\t{json.dumps(counts, sort_keys=True)}")
    print(f"wrote\t{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
