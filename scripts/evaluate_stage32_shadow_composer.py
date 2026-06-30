"""Stage32-C: evaluate shadow composer prediction exports.

Diagnostic only. This script reads prediction JSON produced with Stage32 owner-state
export enabled and compares the current final prediction against the shadow composer
label. It must not be used for training, calibration, threshold selection, or
checkpoint selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

LABELS = ["REFUTE", "NOT_ENTITLED", "SUPPORT"]
WRAPPER_KEYS = [
    "predictions",
    "records",
    "examples",
    "items",
    "data",
    "per_example",
    "per_example_predictions",
    "external_predictions",
]
GROUP_FIELD_CANDIDATES = [
    "group",
    "intervention_type",
    "normalized_intervention",
    "primary_failure_type",
]
GOLD_FIELD_CANDIDATES = ["gold_final_label", "final_label", "gold_label", "label"]
CURRENT_PRED_FIELD_CANDIDATES = [
    "pred_final_label",
    "pred_label",
    "prediction",
    "predicted_label",
]
SHADOW_PRED_FIELD_DEFAULT = "stage32_shadow_label"

OWNER_MEAN_FIELDS = [
    "stage32_hard_core_prob",
    "stage32_coverage_entails_support_prob",
    "stage32_coverage_overclaim_ne_prob",
    "stage32_coverage_contradicts_refute_prob",
    "stage32_coverage_confidence",
    "stage32_coverage_v2_top_prob",
    "stage32_coverage_v2_second_prob",
    "stage32_coverage_v2_margin",
    "stage33_structured_coverage_confidence",
    "stage32_polarity_support_prob",
    "stage32_polarity_refute_prob",
]

SUPPORT_ENTAILMENT_GROUPS = {
    "all_to_some_support",
    "specific_to_general_support",
    "only_to_base_support",
    "whole_to_part_support",
}
OVERCLAIM_GROUPS = {
    "some_to_all_not_entitled",
    "general_to_specific_not_entitled",
    "also_to_only_not_entitled",
    "part_to_whole_not_entitled",
}
REFUTE_GROUPS = {
    "none_to_some_refute",
    "some_to_none_refute",
}


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list) and all(isinstance(row, dict) for row in data):
        return data
    if isinstance(data, dict):
        for key in WRAPPER_KEYS:
            value = data.get(key)
            if isinstance(value, list) and all(isinstance(row, dict) for row in value):
                return value
        list_candidates = [
            value
            for value in data.values()
            if isinstance(value, list) and all(isinstance(row, dict) for row in value)
        ]
        if list_candidates:
            return max(list_candidates, key=len)
        raise ValueError(
            "Prediction JSON wrapper did not contain rows under any known key: "
            f"{WRAPPER_KEYS}"
        )
    raise ValueError("Prediction file must be a list of objects or a wrapper object.")


def normalize_label(raw: Any) -> str:
    if raw is None:
        raise ValueError("missing label")
    key = str(raw).strip().upper().replace("-", "_")
    key = " ".join(key.split())
    mapping = {
        "REFUTE": "REFUTE",
        "REFUTES": "REFUTE",
        "0": "REFUTE",
        "NOT_ENTITLED": "NOT_ENTITLED",
        "NE": "NOT_ENTITLED",
        "NOT ENOUGH INFO": "NOT_ENTITLED",
        "NOT_ENOUGH_INFO": "NOT_ENTITLED",
        "1": "NOT_ENTITLED",
        "SUPPORT": "SUPPORT",
        "SUPPORTS": "SUPPORT",
        "2": "SUPPORT",
    }
    if key in mapping:
        return mapping[key]
    raise ValueError(f"Cannot normalize label value {raw!r}")


def detect_field(
    rows: list[dict[str, Any]],
    candidates: list[str],
    explicit: str | None,
    *,
    required: bool,
    field_role: str,
) -> str | None:
    if explicit:
        if required and not any(explicit in row for row in rows):
            raise ValueError(f"Requested {field_role} field {explicit!r} was not found.")
        return explicit
    for candidate in candidates:
        if any(candidate in row for row in rows):
            return candidate
    if required:
        raise ValueError(
            f"Could not auto-detect {field_role} field. Tried {candidates}."
        )
    return None


def require_field_on_all_rows(
    rows: list[dict[str, Any]],
    field: str,
    *,
    field_role: str,
    stage32_hint: bool = False,
) -> None:
    missing = [idx for idx, row in enumerate(rows) if field not in row]
    if missing:
        hint = (
            " Prediction export must be run with Stage32 owner-state export enabled."
            if stage32_hint
            else ""
        )
        raise ValueError(
            f"Required {field_role} field {field!r} is missing on "
            f"{len(missing)} rows; first missing row index={missing[0]}.{hint}"
        )


def safe_float(raw: Any) -> float | None:
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def normalize_bool(raw: Any) -> bool | None:
    if isinstance(raw, bool):
        return raw
    if raw is None or raw == "":
        return None
    val = str(raw).strip().lower()
    if val in {"true", "1", "yes", "y"}:
        return True
    if val in {"false", "0", "no", "n"}:
        return False
    return None


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def parse_float_list(raw: str) -> list[float]:
    values: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    if not values:
        raise ValueError(f"Expected at least one numeric value in {raw!r}")
    return values


def numeric_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "mean": None,
            "min": None,
            "max": None,
            "p10": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
        }
    vals = sorted(values)

    def percentile(pct: float) -> float:
        if len(vals) == 1:
            return vals[0]
        pos = (len(vals) - 1) * pct
        lo = int(pos)
        hi = min(lo + 1, len(vals) - 1)
        frac = pos - lo
        return vals[lo] * (1.0 - frac) + vals[hi] * frac

    return {
        "mean": round(sum(vals) / len(vals), 6),
        "min": round(vals[0], 6),
        "max": round(vals[-1], 6),
        "p10": round(percentile(0.10), 6),
        "p25": round(percentile(0.25), 6),
        "p50": round(percentile(0.50), 6),
        "p75": round(percentile(0.75), 6),
        "p90": round(percentile(0.90), 6),
    }


def macro_f1(golds: list[str], preds: list[str]) -> float:
    scores: list[float] = []
    for label in LABELS:
        tp = sum(g == label and p == label for g, p in zip(golds, preds))
        fp = sum(g != label and p == label for g, p in zip(golds, preds))
        fn = sum(g == label and p != label for g, p in zip(golds, preds))
        prec = safe_div(tp, tp + fp)
        rec = safe_div(tp, tp + fn)
        scores.append(safe_div(2.0 * prec * rec, prec + rec))
    return sum(scores) / len(scores)


def confusion_matrix(golds: list[str], preds: list[str]) -> dict[str, dict[str, int]]:
    matrix = {gold: {pred: 0 for pred in LABELS} for gold in LABELS}
    for gold, pred in zip(golds, preds):
        matrix[gold][pred] += 1
    return matrix


def prediction_metrics(golds: list[str], preds: list[str]) -> dict[str, Any]:
    return {
        "accuracy": round(safe_div(sum(g == p for g, p in zip(golds, preds)), len(golds)), 4),
        "macro_f1": round(macro_f1(golds, preds), 4),
        "confusion_matrix": confusion_matrix(golds, preds),
        "prediction_distribution": dict(Counter(preds)),
    }


def means_for_rows(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for field in OWNER_MEAN_FIELDS:
        values = [safe_float(row.get(field)) for row in rows]
        values = [value for value in values if value is not None]
        out[field] = round(sum(values) / len(values), 5) if values else None
    return out


def group_distribution(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(Counter(str(row.get(field, "MISSING")) for row in rows))


def compute_coverage_v2_summary(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    v2_fields = {
        "stage32_coverage_v2_pred_label",
        "stage32_coverage_v2_route",
        "stage32_coverage_v2_reason",
        "stage32_coverage_v2_abstained",
        "stage32_coverage_v2_margin",
        "stage32_coverage_v2_top_prob",
    }
    if not any(any(field in row for field in v2_fields) for row in rows):
        return None
    abstain_values = [
        normalize_bool(row.get("stage32_coverage_v2_abstained"))
        for row in rows
        if "stage32_coverage_v2_abstained" in row
    ]
    abstain_count = sum(value is True for value in abstain_values)
    margins = [
        value for value in
        (safe_float(row.get("stage32_coverage_v2_margin")) for row in rows)
        if value is not None
    ]
    top_probs = [
        value for value in
        (safe_float(row.get("stage32_coverage_v2_top_prob")) for row in rows)
        if value is not None
    ]
    return {
        "coverage_v2_pred_label_counts": group_distribution(
            rows, "stage32_coverage_v2_pred_label"
        ),
        "coverage_v2_route_counts": group_distribution(
            rows, "stage32_coverage_v2_route"
        ),
        "coverage_v2_reason_counts": group_distribution(
            rows, "stage32_coverage_v2_reason"
        ),
        "coverage_v2_abstain_count": abstain_count,
        "coverage_v2_abstain_rate": round(safe_div(abstain_count, len(rows)), 4),
        "coverage_v2_margin_summary": numeric_summary(margins),
        "coverage_v2_top_prob_summary": numeric_summary(top_probs),
    }


def compute_stage33_structured_summary(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    fields = {
        "stage33_structured_coverage_label",
        "stage33_structured_coverage_route",
        "stage33_structured_coverage_reason",
        "stage33_structured_coverage_confidence",
    }
    if not any(any(field in row for field in fields) for row in rows):
        return None
    confidences = [
        value for value in
        (safe_float(row.get("stage33_structured_coverage_confidence")) for row in rows)
        if value is not None
    ]
    return {
        "stage33_structured_coverage_label_counts": group_distribution(
            rows, "stage33_structured_coverage_label"
        ),
        "stage33_structured_coverage_route_counts": group_distribution(
            rows, "stage33_structured_coverage_route"
        ),
        "stage33_structured_coverage_reason_counts": group_distribution(
            rows, "stage33_structured_coverage_reason"
        ),
        "stage33_structured_coverage_confidence_summary": numeric_summary(confidences),
    }


def compute_group_metrics(
    rows: list[dict[str, Any]],
    group_field: str | None,
    gold_field: str,
    current_field: str,
    shadow_field: str,
) -> dict[str, Any]:
    if group_field is None:
        return {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_field, "MISSING"))].append(row)

    out: dict[str, Any] = {}
    for group, group_rows in sorted(grouped.items()):
        golds = [normalize_label(row[gold_field]) for row in group_rows]
        current = [normalize_label(row[current_field]) for row in group_rows]
        shadow = [normalize_label(row[shadow_field]) for row in group_rows]
        current_acc = safe_div(sum(g == p for g, p in zip(golds, current)), len(golds))
        shadow_acc = safe_div(sum(g == p for g, p in zip(golds, shadow)), len(golds))
        out[group] = {
            "n": len(group_rows),
            "current_accuracy": round(current_acc, 4),
            "shadow_accuracy": round(shadow_acc, 4),
            "shadow_minus_current_accuracy": round(shadow_acc - current_acc, 4),
            "current_pred_distribution": dict(Counter(current)),
            "shadow_pred_distribution": dict(Counter(shadow)),
            "shadow_reason_distribution": group_distribution(
                group_rows, "stage32_shadow_reason"
            ),
            "owner_state_means": means_for_rows(group_rows),
        }
    return out


def compute_change_counts(
    golds: list[str],
    current: list[str],
    shadow: list[str],
) -> dict[str, int]:
    counts = Counter()
    for gold, cur, shad in zip(golds, current, shadow):
        cur_correct = cur == gold
        shad_correct = shad == gold
        if not cur_correct and shad_correct:
            counts["current_wrong_shadow_correct"] += 1
        elif cur_correct and not shad_correct:
            counts["current_correct_shadow_wrong"] += 1
        elif cur_correct and shad_correct:
            counts["both_correct"] += 1
        elif cur == shad:
            counts["both_wrong_same"] += 1
        else:
            counts["both_wrong_different"] += 1
    for key in (
        "current_wrong_shadow_correct",
        "current_correct_shadow_wrong",
        "both_correct",
        "both_wrong_same",
        "both_wrong_different",
    ):
        counts.setdefault(key, 0)
    return dict(counts)


def compute_stage31_diagnostics(
    rows: list[dict[str, Any]],
    group_field: str | None,
    current_field: str,
    shadow_field: str,
) -> dict[str, Any] | None:
    current = [normalize_label(row[current_field]) for row in rows]
    shadow = [normalize_label(row[shadow_field]) for row in rows]
    v2_routes = [str(row.get("stage32_coverage_v2_route", "")) for row in rows]
    structured_routes = [
        str(row.get("stage33_structured_coverage_route", "")) for row in rows
    ]
    return compute_stage31_diagnostics_from_values(
        rows,
        group_field,
        current,
        shadow,
        v2_routes,
        structured_routes,
    )


def compute_stage31_diagnostics_from_values(
    rows: list[dict[str, Any]],
    group_field: str | None,
    current: list[str],
    shadow: list[str],
    v2_routes: list[str] | None = None,
    structured_routes: list[str] | None = None,
) -> dict[str, Any] | None:
    if group_field is None:
        return None
    groups = {str(row.get(group_field, "")) for row in rows}
    if not groups.intersection(SUPPORT_ENTAILMENT_GROUPS | OVERCLAIM_GROUPS | REFUTE_GROUPS):
        return None

    if v2_routes is None:
        v2_routes = [""] * len(rows)
    if structured_routes is None:
        structured_routes = [""] * len(rows)
    diag = Counter()
    for row, current_label, shadow_label, v2_route, structured_route in zip(
        rows, current, shadow, v2_routes, structured_routes
    ):
        group = str(row.get(group_field, ""))
        if group in SUPPORT_ENTAILMENT_GROUPS:
            if structured_route == "ENTAILMENT_PRESERVE":
                diag["support_entailment_stage33_entailment_preserve"] += 1
            if structured_route == "RESIDUAL":
                diag["support_entailment_stage33_unresolved"] += 1
            if v2_route == "ENTAILMENT_PRESERVE":
                diag["support_entailment_v2_entailment_preserve"] += 1
            if v2_route == "RESIDUAL":
                diag["support_entailment_v2_unresolved"] += 1
            if current_label == "NOT_ENTITLED":
                diag["support_entailment_current_ne"] += 1
            if shadow_label == "NOT_ENTITLED":
                diag["support_entailment_shadow_ne"] += 1
                diag["shadow_support_to_ne"] += 1
                diag["stage33_shadow_support_to_ne"] += 1
            if shadow_label == "SUPPORT":
                diag["support_entailment_shadow_support"] += 1
                diag["support_entailment_stage33_shadow_support"] += 1
            if shadow_label == "REFUTE":
                diag["shadow_support_to_refute"] += 1
                diag["stage33_shadow_support_to_refute"] += 1
        elif group in OVERCLAIM_GROUPS:
            if structured_route == "OVERCLAIM_NE":
                diag["overclaim_stage33_overclaim_ne"] += 1
            if structured_route == "RESIDUAL":
                diag["overclaim_stage33_unresolved"] += 1
            if v2_route == "OVERCLAIM_NE":
                diag["overclaim_v2_overclaim_ne"] += 1
            if v2_route == "RESIDUAL":
                diag["overclaim_v2_unresolved"] += 1
            if current_label == "SUPPORT":
                diag["overclaim_current_support"] += 1
            if shadow_label == "SUPPORT":
                diag["overclaim_shadow_support"] += 1
                diag["shadow_overclaim_to_support"] += 1
                diag["overclaim_stage33_shadow_support"] += 1
                diag["stage33_shadow_overclaim_to_support"] += 1
            if shadow_label == "NOT_ENTITLED":
                diag["overclaim_shadow_ne"] += 1
        elif group in REFUTE_GROUPS:
            if structured_route == "CONTRADICTION_REFUTE":
                diag["refute_stage33_contradiction_refute"] += 1
            if structured_route == "RESIDUAL":
                diag["refute_stage33_unresolved"] += 1
            if v2_route == "CONTRADICTION_REFUTE":
                diag["refute_v2_contradiction_refute"] += 1
            if v2_route == "RESIDUAL":
                diag["refute_v2_unresolved"] += 1
            if current_label == "SUPPORT":
                diag["refute_current_support"] += 1
            if shadow_label == "SUPPORT":
                diag["refute_shadow_support"] += 1
                diag["shadow_refute_to_support"] += 1
                diag["stage33_shadow_refute_to_support"] += 1
            if shadow_label == "REFUTE":
                diag["refute_shadow_refute"] += 1
                diag["refute_stage33_shadow_refute"] += 1
            if shadow_label == "NOT_ENTITLED":
                diag["refute_shadow_ne"] += 1

    for key in (
        "support_entailment_current_ne",
        "support_entailment_shadow_ne",
        "support_entailment_shadow_support",
        "overclaim_current_support",
        "overclaim_shadow_support",
        "overclaim_shadow_ne",
        "refute_current_support",
        "refute_shadow_support",
        "refute_shadow_refute",
        "refute_shadow_ne",
        "shadow_overclaim_to_support",
        "shadow_refute_to_support",
        "shadow_support_to_ne",
        "shadow_support_to_refute",
        "support_entailment_v2_entailment_preserve",
        "support_entailment_v2_unresolved",
        "overclaim_v2_overclaim_ne",
        "overclaim_v2_unresolved",
        "refute_v2_contradiction_refute",
        "refute_v2_unresolved",
        "support_entailment_stage33_entailment_preserve",
        "support_entailment_stage33_shadow_support",
        "support_entailment_stage33_unresolved",
        "overclaim_stage33_overclaim_ne",
        "overclaim_stage33_shadow_support",
        "overclaim_stage33_unresolved",
        "refute_stage33_contradiction_refute",
        "refute_stage33_shadow_refute",
        "refute_stage33_unresolved",
        "stage33_shadow_overclaim_to_support",
        "stage33_shadow_refute_to_support",
        "stage33_shadow_support_to_ne",
        "stage33_shadow_support_to_refute",
    ):
        diag.setdefault(key, 0)
    return dict(diag)


def coverage_route_from_label(label: str) -> str:
    if label == "ENTAILS_SUPPORT":
        return "ENTAILMENT_PRESERVE"
    if label == "OVERCLAIM_NOT_ENTITLED":
        return "OVERCLAIM_NE"
    if label == "CONTRADICTS_REFUTE":
        return "CONTRADICTION_REFUTE"
    return "RESIDUAL"


def offline_v2_route_for_row(
    row: dict[str, Any],
    min_confidence: float,
    min_margin: float,
) -> dict[str, Any]:
    probs = [
        ("ENTAILS_SUPPORT", safe_float(row.get("stage32_coverage_entails_support_prob"))),
        ("OVERCLAIM_NOT_ENTITLED", safe_float(row.get("stage32_coverage_overclaim_ne_prob"))),
        ("CONTRADICTS_REFUTE", safe_float(row.get("stage32_coverage_contradicts_refute_prob"))),
    ]
    if any(prob is None for _, prob in probs):
        return {
            "pred_label": "UNRESOLVED_COVERAGE",
            "route": "RESIDUAL",
            "reason": "missing_coverage_probs",
            "top_prob": None,
            "second_prob": None,
            "margin": None,
        }
    sorted_probs = sorted(probs, key=lambda item: item[1] or 0.0, reverse=True)
    top_label, top_prob = sorted_probs[0]
    _, second_prob = sorted_probs[1]
    margin = float(top_prob) - float(second_prob)
    if float(top_prob) < min_confidence:
        return {
            "pred_label": "UNRESOLVED_COVERAGE",
            "route": "RESIDUAL",
            "reason": "low_confidence_abstain",
            "top_prob": float(top_prob),
            "second_prob": float(second_prob),
            "margin": margin,
        }
    if margin < min_margin:
        return {
            "pred_label": "UNRESOLVED_COVERAGE",
            "route": "RESIDUAL",
            "reason": "low_margin_abstain",
            "top_prob": float(top_prob),
            "second_prob": float(second_prob),
            "margin": margin,
        }
    return {
        "pred_label": top_label,
        "route": coverage_route_from_label(top_label),
        "reason": "confident_coverage_prediction",
        "top_prob": float(top_prob),
        "second_prob": float(second_prob),
        "margin": margin,
    }


def offline_shadow_label(row: dict[str, Any], route: str) -> str:
    hard_core_pass = normalize_bool(row.get("stage32_hard_core_pass"))
    polarity = str(row.get("stage32_polarity_pred_label", "NEUTRAL_OR_BLOCKED"))
    if hard_core_pass is False:
        return "NOT_ENTITLED"
    if route == "OVERCLAIM_NE":
        return "NOT_ENTITLED"
    if route == "CONTRADICTION_REFUTE":
        return "REFUTE"
    if route == "ENTAILMENT_PRESERVE" and polarity == "SUPPORT":
        return "SUPPORT"
    if route == "ENTAILMENT_PRESERVE":
        return "NOT_ENTITLED"
    return "NOT_ENTITLED"


def offline_sweep_decision(
    row: dict[str, Any],
    current_macro_f1: float,
) -> str:
    safe = row["shadow_overclaim_to_support"] == 0 and row["shadow_refute_to_support"] == 0
    recovers_support = row["support_entailment_shadow_support"] > 0
    improves_macro = row["delta_macro_f1"] >= 0.05
    if not (safe and recovers_support):
        return "STAGE32_D2_NO_SAFE_SUPPORT_RECOVERY"
    if improves_macro and row["shadow_macro_f1"] >= current_macro_f1:
        return "STAGE32_D2_PROMISING_BUT_STILL_SHADOW_ONLY"
    return "STAGE32_D2_DIAGNOSTIC_THRESHOLD_FOUND_NOT_APPLY"


def compute_offline_sweep(
    rows: list[dict[str, Any]],
    group_field: str | None,
    golds: list[str],
    current: list[str],
    confidences: list[float],
    margins: list[float],
    current_metrics: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    sweep_rows: list[dict[str, Any]] = []
    for conf in confidences:
        for margin_threshold in margins:
            route_infos = [
                offline_v2_route_for_row(row, conf, margin_threshold)
                for row in rows
            ]
            routes = [info["route"] for info in route_infos]
            pred_labels = [info["pred_label"] for info in route_infos]
            reasons = [info["reason"] for info in route_infos]
            shadow = [
                offline_shadow_label(row, route)
                for row, route in zip(rows, routes)
            ]
            shadow_metrics = prediction_metrics(golds, shadow)
            stage31_diag = compute_stage31_diagnostics_from_values(
                rows,
                group_field,
                current,
                shadow,
                routes,
            ) or {}
            unresolved_count = sum(route == "RESIDUAL" for route in routes)
            out = {
                "min_confidence": conf,
                "min_margin": margin_threshold,
                "shadow_accuracy": shadow_metrics["accuracy"],
                "shadow_macro_f1": shadow_metrics["macro_f1"],
                "delta_macro_f1": round(
                    shadow_metrics["macro_f1"] - current_metrics["macro_f1"], 4
                ),
                "shadow_prediction_distribution": dict(Counter(shadow)),
                "route_counts": dict(Counter(routes)),
                "pred_label_counts": dict(Counter(pred_labels)),
                "reason_counts": dict(Counter(reasons)),
                "unresolved_count": unresolved_count,
                "unresolved_rate": round(safe_div(unresolved_count, len(rows)), 4),
                "support_entailment_shadow_support": stage31_diag.get(
                    "support_entailment_shadow_support", 0
                ),
                "support_entailment_v2_entailment_preserve": stage31_diag.get(
                    "support_entailment_v2_entailment_preserve", 0
                ),
                "overclaim_v2_overclaim_ne": stage31_diag.get(
                    "overclaim_v2_overclaim_ne", 0
                ),
                "refute_v2_contradiction_refute": stage31_diag.get(
                    "refute_v2_contradiction_refute", 0
                ),
                "shadow_overclaim_to_support": stage31_diag.get(
                    "shadow_overclaim_to_support", 0
                ),
                "shadow_refute_to_support": stage31_diag.get(
                    "shadow_refute_to_support", 0
                ),
                "shadow_support_to_ne": stage31_diag.get("shadow_support_to_ne", 0),
                "shadow_support_to_refute": stage31_diag.get(
                    "shadow_support_to_refute", 0
                ),
            }
            out["decision_label"] = offline_sweep_decision(
                out, current_metrics["macro_f1"]
            )
            sweep_rows.append(out)
    safe_candidates = [
        row for row in sweep_rows
        if row["shadow_overclaim_to_support"] == 0
        and row["shadow_refute_to_support"] == 0
    ]
    best = (
        max(safe_candidates, key=lambda row: row["shadow_macro_f1"])
        if safe_candidates
        else None
    )
    return sweep_rows, best


def decide(
    current_metrics: dict[str, Any],
    shadow_metrics: dict[str, Any],
    stage31_diag: dict[str, Any] | None,
    coverage_v2_summary: dict[str, Any] | None,
    stage33_summary: dict[str, Any] | None = None,
) -> dict[str, str]:
    macro_delta = shadow_metrics["macro_f1"] - current_metrics["macro_f1"]
    if stage33_summary is not None:
        if stage31_diag:
            stage33_overclaim_support = stage31_diag.get(
                "stage33_shadow_overclaim_to_support", 0
            )
            stage33_refute_support = stage31_diag.get(
                "stage33_shadow_refute_to_support", 0
            )
            stage33_support_recovered = stage31_diag.get(
                "support_entailment_stage33_shadow_support", 0
            )
            stage33_support_ne = stage31_diag.get("stage33_shadow_support_to_ne", 0)
            stage33_support_refute = stage31_diag.get("stage33_shadow_support_to_refute", 0)
        else:
            stage33_overclaim_support = 0
            stage33_refute_support = 0
            stage33_support_recovered = 0
            stage33_support_ne = 0
            stage33_support_refute = 0
        if stage33_overclaim_support > 0 or stage33_refute_support > 0 or macro_delta <= -0.05:
            return {
                "label": "STAGE33_STRUCTURED_OWNER_UNSAFE",
                "reason": (
                    "Structured shadow route is unsafe: safety errors appeared or "
                    "macro-F1 collapsed materially."
                ),
            }
        if (
            stage33_support_recovered > 0
            and stage33_overclaim_support == 0
            and stage33_refute_support == 0
            and macro_delta > -0.02
        ):
            return {
                "label": "STAGE33_STRUCTURED_OWNER_PROMISING",
                "reason": (
                    "Structured owner recovers some SUPPORT without overclaim/refute "
                    "to SUPPORT safety errors and without material macro-F1 drop."
                ),
            }
        return {
            "label": "STAGE33_STRUCTURED_OWNER_DIAGNOSTIC_ONLY",
            "reason": (
                "Structured routes are interpretable, but SUPPORT recovery is still "
                f"weak or unresolved collapse remains (support_to_ne={stage33_support_ne}, "
                f"support_to_refute={stage33_support_refute})."
            ),
        }
    if stage31_diag:
        refute_safety_increase = (
            stage31_diag["shadow_refute_to_support"]
            > stage31_diag["refute_current_support"]
        )
        overclaim_safety_increase = (
            stage31_diag["shadow_overclaim_to_support"]
            > stage31_diag["overclaim_current_support"]
        )
        support_recovery_improved = (
            stage31_diag["support_entailment_shadow_support"]
            >= stage31_diag["support_entailment_current_ne"] + 10
        )
        support_collapse = stage31_diag["shadow_support_to_ne"] >= 40
        v2_reduced_overclaim_routing = (
            coverage_v2_summary is not None
            and stage31_diag.get("overclaim_v2_unresolved", 0) > 0
        )
    else:
        refute_safety_increase = False
        overclaim_safety_increase = False
        support_recovery_improved = False
        support_collapse = False
        v2_reduced_overclaim_routing = False

    if (
        macro_delta >= 0.05
        and not refute_safety_increase
        and not overclaim_safety_increase
        and support_recovery_improved
    ):
        return {
            "label": "STAGE32_SHADOW_PROMISING",
            "reason": (
                "Shadow macro-F1 improves materially without increasing Stage31 "
                "safety errors, and support-entailment recovery improves."
            ),
        }
    if refute_safety_increase or overclaim_safety_increase:
        return {
            "label": "STAGE32_SHADOW_UNSAFE",
            "reason": (
                "Shadow route is not safe to apply: REFUTE->SUPPORT or "
                "OVERCLAIM->SUPPORT safety errors increased."
            ),
        }
    if (macro_delta <= -0.02 or support_collapse) and not v2_reduced_overclaim_routing:
        return {
            "label": "STAGE32_SHADOW_UNSAFE",
            "reason": (
                "Shadow route is not safe to apply: macro-F1 dropped materially "
                "or SUPPORT cases remain collapsed without useful v2 unresolved routing."
            ),
        }
    if coverage_v2_summary is not None and v2_reduced_overclaim_routing:
        return {
            "label": "STAGE32_SHADOW_DIAGNOSTIC_ONLY_CONTINUE",
            "reason": (
                "Coverage Owner v2 exposes unresolved routing, but the shadow composer "
                "is not yet safe to apply until SUPPORT recovery improves."
            ),
        }
    return {
        "label": "STAGE32_SHADOW_DIAGNOSTIC_ONLY_CONTINUE",
        "reason": (
            "Shadow routes provide diagnostic traces, but evidence is insufficient "
            "to apply the composer."
        ),
    }


def format_confusion_matrix(matrix: dict[str, dict[str, int]]) -> str:
    lines = ["| Gold \\ Pred | REFUTE | NOT_ENTITLED | SUPPORT |", "|---|---:|---:|---:|"]
    for gold in LABELS:
        lines.append(
            f"| {gold} | {matrix[gold]['REFUTE']} | "
            f"{matrix[gold]['NOT_ENTITLED']} | {matrix[gold]['SUPPORT']} |"
        )
    return "\n".join(lines)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Stage32-C Shadow Composer Evaluation",
        "",
        "## Purpose",
        "Diagnostic-only comparison of existing final predictions against the Stage32 shadow composer route. The shadow label is not applied to model outputs.",
        "",
        "## Input",
        f"- File: `{report['predictions_file']}`",
        f"- Run name: `{report['run_name']}`",
        f"- Row count: {report['row_count']}",
        "",
        "## Current vs Shadow Metrics",
        "| Metric | Current | Shadow | Delta |",
        "|---|---:|---:|---:|",
        f"| Accuracy | {report['current_metrics']['accuracy']:.4f} | {report['shadow_metrics']['accuracy']:.4f} | {report['delta']['shadow_minus_current_accuracy']:.4f} |",
        f"| Macro-F1 | {report['current_metrics']['macro_f1']:.4f} | {report['shadow_metrics']['macro_f1']:.4f} | {report['delta']['shadow_minus_current_macro_f1']:.4f} |",
        f"| Rows changed |  |  | {report['delta']['rows_changed_by_shadow']} ({report['delta']['rows_changed_by_shadow_rate']:.4f}) |",
        "",
        "## Current Confusion Matrix",
        format_confusion_matrix(report["current_metrics"]["confusion_matrix"]),
        "",
        "## Shadow Confusion Matrix",
        format_confusion_matrix(report["shadow_metrics"]["confusion_matrix"]),
        "",
        "## Change-Type Counts",
        "| Type | Count |",
        "|---|---:|",
    ]
    for key, value in report["change_type_counts"].items():
        lines.append(f"| {key} | {value} |")

    lines.extend(["", "## Shadow Reason Counts", "| Reason | Count |", "|---|---:|"])
    for key, value in sorted(report["shadow_reason_counts"].items()):
        lines.append(f"| {key} | {value} |")

    if report["shadow_priority_trace_counts"]:
        lines.extend(["", "## Shadow Priority Trace Counts", "| Trace | Count |", "|---|---:|"])
        for key, value in sorted(report["shadow_priority_trace_counts"].items(), key=lambda item: (-item[1], item[0]))[:30]:
            lines.append(f"| `{key}` | {value} |")

    if report["coverage_v2_summary"] is not None:
        lines.extend(["", "## Coverage Owner v2 Summary"])
        lines.append(f"- Abstain count: {report['coverage_v2_summary']['coverage_v2_abstain_count']}")
        lines.append(f"- Abstain rate: {report['coverage_v2_summary']['coverage_v2_abstain_rate']:.4f}")
        for title, key in (
            ("v2 Pred Label Counts", "coverage_v2_pred_label_counts"),
            ("v2 Route Counts", "coverage_v2_route_counts"),
            ("v2 Reason Counts", "coverage_v2_reason_counts"),
        ):
            lines.extend(["", f"### {title}", "| Value | Count |", "|---|---:|"])
            for name, count in sorted(report["coverage_v2_summary"][key].items()):
                lines.append(f"| {name} | {count} |")

    if report["stage33_structured_coverage_summary"] is not None:
        lines.extend(["", "## Stage33 Structured Coverage Owner Summary"])
        for title, key in (
            ("Structured Label Counts", "stage33_structured_coverage_label_counts"),
            ("Structured Route Counts", "stage33_structured_coverage_route_counts"),
            ("Structured Reason Counts", "stage33_structured_coverage_reason_counts"),
        ):
            lines.extend(["", f"### {title}", "| Value | Count |", "|---|---:|"])
            for name, count in sorted(report["stage33_structured_coverage_summary"][key].items()):
                lines.append(f"| {name} | {count} |")
        lines.extend(["", "Confidence summary:"])
        for key, value in report["stage33_structured_coverage_summary"][
            "stage33_structured_coverage_confidence_summary"
        ].items():
            lines.append(f"- {key}: {value}")

    if report["group_metrics"]:
        lines.extend(["", "## Group-Level Metrics", "| Group | N | Current Acc | Shadow Acc | Delta |", "|---|---:|---:|---:|---:|"])
        for group, metrics in report["group_metrics"].items():
            lines.append(
                f"| {group} | {metrics['n']} | {metrics['current_accuracy']:.4f} | "
                f"{metrics['shadow_accuracy']:.4f} | {metrics['shadow_minus_current_accuracy']:.4f} |"
            )

    if report["stage31_specific_diagnostics"] is not None:
        lines.extend(["", "## Stage31-Specific Safety Diagnostics", "| Counter | Count |", "|---|---:|"])
        for key, value in sorted(report["stage31_specific_diagnostics"].items()):
            lines.append(f"| {key} | {value} |")

    if report.get("coverage_v2_offline_sweep"):
        lines.extend([
            "",
            "## Coverage Owner v2 Offline Threshold Sweep",
            "| Conf | Margin | Safe | Shadow Macro-F1 | Delta Macro-F1 | Support Recovered | Unresolved Rate | Decision |",
            "|---:|---:|---|---:|---:|---:|---:|---|",
        ])
        sorted_sweep = sorted(
            report["coverage_v2_offline_sweep"],
            key=lambda row: (
                not (
                    row["shadow_overclaim_to_support"] == 0
                    and row["shadow_refute_to_support"] == 0
                ),
                -row["support_entailment_shadow_support"],
                -row["shadow_macro_f1"],
            ),
        )
        for row in sorted_sweep:
            safe = (
                row["shadow_overclaim_to_support"] == 0
                and row["shadow_refute_to_support"] == 0
            )
            lines.append(
                f"| {row['min_confidence']:.2f} | {row['min_margin']:.2f} | "
                f"{safe} | {row['shadow_macro_f1']:.4f} | {row['delta_macro_f1']:.4f} | "
                f"{row['support_entailment_shadow_support']} | {row['unresolved_rate']:.4f} | "
                f"{row['decision_label']} |"
            )
        if report.get("coverage_v2_offline_sweep_best") is not None:
            best = report["coverage_v2_offline_sweep_best"]
            lines.extend([
                "",
                "Best safe candidate:",
                f"- min_confidence={best['min_confidence']:.2f}, min_margin={best['min_margin']:.2f}, shadow_macro_f1={best['shadow_macro_f1']:.4f}",
            ])
        else:
            lines.extend(["", "Best safe candidate: none"])

    lines.extend([
        "",
        "## Decision",
        f"- **{report['decision']['label']}**: {report['decision']['reason']}",
        "",
        "## Leakage Policy",
        report["leakage_policy"],
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage32-C diagnostic evaluation of shadow composer exports."
    )
    parser.add_argument("--predictions-file", required=True)
    parser.add_argument(
        "--output-md",
        default="reports/stage32c_shadow_composer_eval_report.md",
    )
    parser.add_argument(
        "--output-json",
        default="reports/stage32c_shadow_composer_eval_report.json",
    )
    parser.add_argument("--run-name", default="stage32c_shadow_composer_eval")
    parser.add_argument("--group-field", default=None)
    parser.add_argument("--gold-field", default=None)
    parser.add_argument("--current-pred-field", default=None)
    parser.add_argument("--shadow-pred-field", default=SHADOW_PRED_FIELD_DEFAULT)
    parser.add_argument("--coverage-v2-offline-sweep", action="store_true", default=False)
    parser.add_argument(
        "--coverage-v2-sweep-confidences",
        default="0.30,0.35,0.40,0.45,0.50",
    )
    parser.add_argument(
        "--coverage-v2-sweep-margins",
        default="0.00,0.02,0.05,0.08,0.10",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predictions_path = Path(args.predictions_file)
    if not predictions_path.is_absolute():
        predictions_path = REPO_ROOT / predictions_path
    if not predictions_path.exists():
        print(f"ERROR: predictions file not found: {predictions_path}", file=sys.stderr)
        return 1

    try:
        rows = load_rows(predictions_path)
        if not rows:
            raise ValueError("Prediction file contains no rows.")
        gold_field = detect_field(
            rows,
            GOLD_FIELD_CANDIDATES,
            args.gold_field,
            required=True,
            field_role="gold label",
        )
        current_field = detect_field(
            rows,
            CURRENT_PRED_FIELD_CANDIDATES,
            args.current_pred_field,
            required=True,
            field_role="current prediction",
        )
        shadow_field = args.shadow_pred_field
        if not any(shadow_field in row for row in rows):
            raise ValueError(
                f"Missing {shadow_field!r}. Prediction export must be run with "
                "Stage32 owner-state export enabled."
            )
        require_field_on_all_rows(rows, gold_field, field_role="gold label")
        require_field_on_all_rows(rows, current_field, field_role="current prediction")
        require_field_on_all_rows(
            rows,
            shadow_field,
            field_role="shadow prediction",
            stage32_hint=True,
        )
        group_field = detect_field(
            rows,
            GROUP_FIELD_CANDIDATES,
            args.group_field,
            required=False,
            field_role="group",
        )

        golds = [normalize_label(row[gold_field]) for row in rows]
        current = [normalize_label(row[current_field]) for row in rows]
        shadow = [normalize_label(row[shadow_field]) for row in rows]
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    current_metrics = prediction_metrics(golds, current)
    shadow_metrics = prediction_metrics(golds, shadow)
    rows_changed = sum(c != s for c, s in zip(current, shadow))
    stage31_diag = compute_stage31_diagnostics(rows, group_field, current_field, shadow_field)
    coverage_v2_summary = compute_coverage_v2_summary(rows)
    stage33_summary = compute_stage33_structured_summary(rows)
    offline_sweep: list[dict[str, Any]] | None = None
    offline_sweep_best: dict[str, Any] | None = None
    offline_sweep_decision: dict[str, str] | None = None
    if args.coverage_v2_offline_sweep:
        try:
            sweep_confidences = parse_float_list(args.coverage_v2_sweep_confidences)
            sweep_margins = parse_float_list(args.coverage_v2_sweep_margins)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        offline_sweep, offline_sweep_best = compute_offline_sweep(
            rows,
            group_field,
            golds,
            current,
            sweep_confidences,
            sweep_margins,
            current_metrics,
        )
        safe_support_candidates = [
            row for row in offline_sweep
            if row["support_entailment_shadow_support"] > 0
            and row["shadow_overclaim_to_support"] == 0
            and row["shadow_refute_to_support"] == 0
        ]
        if not safe_support_candidates:
            offline_sweep_decision = {
                "label": "STAGE32_D2_NO_SAFE_SUPPORT_RECOVERY",
                "reason": (
                    "No threshold pair recovered SUPPORT while keeping overclaim/refute "
                    "to SUPPORT safety errors at zero."
                ),
            }
        else:
            best_support = max(
                safe_support_candidates,
                key=lambda row: row["shadow_macro_f1"],
            )
            if best_support["delta_macro_f1"] >= 0.05:
                offline_sweep_decision = {
                    "label": "STAGE32_D2_PROMISING_BUT_STILL_SHADOW_ONLY",
                    "reason": (
                        "At least one threshold pair safely recovers SUPPORT and "
                        "improves macro-F1 materially, but this remains shadow-only."
                    ),
                }
            else:
                offline_sweep_decision = {
                    "label": "STAGE32_D2_DIAGNOSTIC_THRESHOLD_FOUND_NOT_APPLY",
                    "reason": (
                        "At least one threshold pair safely recovers SUPPORT, but "
                        "macro-F1 remains too weak for application."
                    ),
                }
    decision = offline_sweep_decision or decide(
        current_metrics,
        shadow_metrics,
        stage31_diag,
        coverage_v2_summary,
        stage33_summary,
    )
    report = {
        "run_name": args.run_name,
        "predictions_file": str(predictions_path),
        "row_count": len(rows),
        "fields": {
            "gold_field": gold_field,
            "current_pred_field": current_field,
            "shadow_pred_field": shadow_field,
            "group_field": group_field,
        },
        "current_metrics": current_metrics,
        "shadow_metrics": shadow_metrics,
        "delta": {
            "shadow_minus_current_accuracy": round(
                shadow_metrics["accuracy"] - current_metrics["accuracy"], 4
            ),
            "shadow_minus_current_macro_f1": round(
                shadow_metrics["macro_f1"] - current_metrics["macro_f1"], 4
            ),
            "rows_changed_by_shadow": rows_changed,
            "rows_changed_by_shadow_rate": round(safe_div(rows_changed, len(rows)), 4),
        },
        "change_type_counts": compute_change_counts(golds, current, shadow),
        "shadow_reason_counts": group_distribution(rows, "stage32_shadow_reason"),
        "shadow_priority_trace_counts": group_distribution(
            rows, "stage32_shadow_priority_trace"
        ),
        "shadow_route_boolean_counts": {
            "stage32_shadow_would_block_support": dict(Counter(
                normalize_bool(row.get("stage32_shadow_would_block_support")) for row in rows
            )),
            "stage32_shadow_would_route_ne": dict(Counter(
                normalize_bool(row.get("stage32_shadow_would_route_ne")) for row in rows
            )),
            "stage32_shadow_would_route_refute": dict(Counter(
                normalize_bool(row.get("stage32_shadow_would_route_refute")) for row in rows
            )),
        },
        "owner_state_means": means_for_rows(rows),
        "coverage_v2_summary": coverage_v2_summary,
        "coverage_v2_pred_label_counts": (
            coverage_v2_summary.get("coverage_v2_pred_label_counts")
            if coverage_v2_summary else None
        ),
        "coverage_v2_route_counts": (
            coverage_v2_summary.get("coverage_v2_route_counts")
            if coverage_v2_summary else None
        ),
        "coverage_v2_reason_counts": (
            coverage_v2_summary.get("coverage_v2_reason_counts")
            if coverage_v2_summary else None
        ),
        "coverage_v2_abstain_count": (
            coverage_v2_summary.get("coverage_v2_abstain_count")
            if coverage_v2_summary else None
        ),
        "coverage_v2_abstain_rate": (
            coverage_v2_summary.get("coverage_v2_abstain_rate")
            if coverage_v2_summary else None
        ),
        "coverage_v2_margin_summary": (
            coverage_v2_summary.get("coverage_v2_margin_summary")
            if coverage_v2_summary else None
        ),
        "coverage_v2_top_prob_summary": (
            coverage_v2_summary.get("coverage_v2_top_prob_summary")
            if coverage_v2_summary else None
        ),
        "stage33_structured_coverage_summary": stage33_summary,
        "stage33_structured_coverage_label_counts": (
            stage33_summary.get("stage33_structured_coverage_label_counts")
            if stage33_summary else None
        ),
        "stage33_structured_coverage_route_counts": (
            stage33_summary.get("stage33_structured_coverage_route_counts")
            if stage33_summary else None
        ),
        "stage33_structured_coverage_reason_counts": (
            stage33_summary.get("stage33_structured_coverage_reason_counts")
            if stage33_summary else None
        ),
        "stage33_structured_coverage_confidence_summary": (
            stage33_summary.get("stage33_structured_coverage_confidence_summary")
            if stage33_summary else None
        ),
        "group_metrics": compute_group_metrics(
            rows, group_field, gold_field, current_field, shadow_field
        ),
        "stage31_specific_diagnostics": stage31_diag,
        "coverage_v2_offline_sweep": offline_sweep,
        "coverage_v2_offline_sweep_best": offline_sweep_best,
        "decision": decision,
        "leakage_policy": (
            "This evaluator is diagnostic-only. It must not be used for training, "
            "calibration, threshold selection, or checkpoint selection. Stage31 "
            "probe remains diagnostic-only."
        ),
    }
    write_markdown(REPO_ROOT / args.output_md, report)
    write_json(REPO_ROOT / args.output_json, report)
    print(f"Markdown report -> {REPO_ROOT / args.output_md}")
    print(f"JSON report     -> {REPO_ROOT / args.output_json}")
    print(f"Decision        -> {report['decision']['label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
