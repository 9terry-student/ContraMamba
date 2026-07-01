"""Evaluate Stage35-A adversarial held-out structured coverage predictions.

Diagnostic only. This evaluator reads prediction exports for the Stage35-A
adversarial probe and must not be used for training, calibration, threshold
selection, loss computation, or checkpoint selection.
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

SUBSET_SUPPORT_GROUPS = {
    "adv_whole_to_part_support_verb_diverse",
    "adv_whole_to_part_support_fronted_modifier",
    "adv_whole_to_part_support_postnominal_modifier",
    "adv_whole_to_part_support_sentence_order_flip",
    "adv_passive_active_support",
    "adv_coordination_support",
    "adv_numeric_subset_support",
}
EXCEPTION_GROUPS = {
    "adv_all_except_subset_not_entitled",
    "adv_all_except_subset_support_for_nonexcluded",
    "adv_no_except_subset_support",
    "adv_no_except_nonexcluded_refute",
}
EXCEPTION_SUPPORT_GROUPS = {
    "adv_all_except_subset_support_for_nonexcluded",
    "adv_no_except_subset_support",
}
NUMERIC_SUPPORT_GROUPS = {"adv_numeric_subset_support"}
REVERSE_GROUP_MARKERS = (
    "part_to_whole",
    "reverse",
    "some_to_all",
    "also_to_only",
    "general_to_specific",
    "exactly_some_to_all",
    "not_all_to_some",
)


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in WRAPPER_KEYS:
            value = data.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"Could not find prediction rows in {path}")


def normalize_label(raw: Any) -> str:
    key = str(raw).strip().upper().replace("-", "_")
    mapping = {
        "0": "REFUTE",
        "REFUTE": "REFUTE",
        "REFUTES": "REFUTE",
        "1": "NOT_ENTITLED",
        "NE": "NOT_ENTITLED",
        "NOT_ENTITLED": "NOT_ENTITLED",
        "NOT ENOUGH INFO": "NOT_ENTITLED",
        "NOT_ENOUGH_INFO": "NOT_ENTITLED",
        "2": "SUPPORT",
        "SUPPORT": "SUPPORT",
        "SUPPORTS": "SUPPORT",
    }
    if key not in mapping:
        raise ValueError(f"Cannot normalize label value {raw!r}")
    return mapping[key]


def normalize_bool(raw: Any) -> bool | None:
    if isinstance(raw, bool):
        return raw
    if raw is None or raw == "":
        return None
    key = str(raw).strip().lower()
    if key in {"true", "1", "yes"}:
        return True
    if key in {"false", "0", "no"}:
        return False
    return None


def detect_field(rows: list[dict[str, Any]], candidates: list[str], role: str) -> str:
    for candidate in candidates:
        if any(candidate in row for row in rows):
            return candidate
    raise ValueError(f"Could not detect {role} field; tried {candidates}")


def first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def get_gold_label(row: dict[str, Any]) -> str:
    value = first_present(row, ("gold_final_label", "gold_label", "final_label", "label"))
    if value is None:
        raise ValueError("Could not find gold label in row")
    return normalize_label(value)


def get_current_label(row: dict[str, Any]) -> str:
    value = first_present(row, ("pred_final_label", "pred_label", "prediction", "predicted_label"))
    if value is None:
        raise ValueError("Could not find current prediction label in row")
    return normalize_label(value)


def get_shadow_label(row: dict[str, Any]) -> str:
    value = first_present(row, ("stage32_shadow_label", "stage33_conditional_shadow_label"))
    if value is None:
        raise ValueError("Could not find shadow prediction label in row")
    return normalize_label(value)


def has_stage36_fields(rows: list[dict[str, Any]]) -> bool:
    """True when any row carries Stage36-A support-safety blocker diagnostics."""
    return any(row.get("stage36_support_blocker_fired") is not None for row in rows)


def has_stage37_fields(rows: list[dict[str, Any]]) -> bool:
    """True when any row carries Stage37-A safe SUPPORT recovery diagnostics."""
    return any(row.get("stage37_safe_recovery_fired") is not None for row in rows)


def resolve_shadow_label(row: dict[str, Any]) -> str:
    """Final shadow label. Priority: stage37_final > stage36_final > stage32 > stage33."""
    value = row.get("stage37_final_shadow_label")
    if value not in (None, ""):
        return normalize_label(value)
    value = row.get("stage36_final_shadow_label")
    if value not in (None, ""):
        return normalize_label(value)
    return get_shadow_label(row)


def resolve_shadow_label_original(row: dict[str, Any]) -> str:
    """Pre-Stage36 shadow label: prefers stage36_original_shadow_label when present."""
    value = row.get("stage36_original_shadow_label")
    if value not in (None, ""):
        return normalize_label(value)
    return get_shadow_label(row)


def resolve_shadow_label_pre_stage37(row: dict[str, Any]) -> str:
    """Pre-Stage37 (post-Stage36) shadow label: prefers stage37_original_shadow_label."""
    value = row.get("stage37_original_shadow_label")
    if value not in (None, ""):
        return normalize_label(value)
    value = row.get("stage36_final_shadow_label")
    if value not in (None, ""):
        return normalize_label(value)
    return get_shadow_label(row)


def get_group(row: dict[str, Any]) -> str:
    value = first_present(row, ("group", "intervention_type", "normalized_intervention", "primary_failure_type"))
    return "UNKNOWN" if value is None else str(value)


def infer_family(group: str) -> str:
    key = group.lower()
    if "exception" in key or "except" in key:
        return "exception"
    if "temporal_scope" in key or "location_scope" in key:
        return "scope"
    if "numeric" in key:
        return "numeric"
    if "coordination" in key:
        return "coordination"
    if "passive_active" in key:
        return "voice"
    if "none_to_any" in key:
        return "negation"
    if "some" in key or "all_to" in key or "not_all" in key:
        return "quantifier"
    if "whole_to_part" in key or "part_to_whole" in key:
        return "whole_part"
    return "unknown_family"


def get_family(row: dict[str, Any]) -> str:
    value = row.get("stage35_family")
    if value not in (None, ""):
        return str(value)
    return infer_family(get_group(row))


def get_perturbation(row: dict[str, Any]) -> str:
    value = row.get("stage35_perturbation")
    if value not in (None, ""):
        return str(value)
    group = get_group(row)
    return group.removeprefix("adv_") if group != "UNKNOWN" else "unknown_perturbation"


def get_expected_route(row: dict[str, Any]) -> str:
    value = row.get("stage35_expected_route")
    if value not in (None, ""):
        return str(value)
    gold = get_gold_label(row)
    if gold == "SUPPORT":
        return "ENTAILMENT_PRESERVE"
    if gold == "REFUTE":
        return "CONTRADICTION_REFUTE"
    return "OVERCLAIM_NE"


def count_values(rows: list[dict[str, Any]], field: str, *, normalize_bools: bool = False) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        if field not in row or row.get(field) in (None, ""):
            counts["MISSING"] += 1
            continue
        value = normalize_bool(row[field]) if normalize_bools else row[field]
        counts[str(value)] += 1
    return dict(counts)


def prediction_metrics(golds: list[str], preds: list[str]) -> dict[str, Any]:
    accuracy = sum(g == p for g, p in zip(golds, preds)) / max(1, len(golds))
    f1s: list[float] = []
    per_class: dict[str, Any] = {}
    for label in LABELS:
        tp = sum(g == label and p == label for g, p in zip(golds, preds))
        fp = sum(g != label and p == label for g, p in zip(golds, preds))
        fn = sum(g == label and p != label for g, p in zip(golds, preds))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        f1s.append(f1)
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(g == label for g in golds),
        }
    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(sum(f1s) / len(f1s), 4),
        "per_class": per_class,
        "prediction_counts": dict(Counter(preds)),
    }


def bucket_metrics(rows: list[dict[str, Any]], bucket_fn) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[bucket_fn(row)].append(row)
    out: dict[str, Any] = {}
    for bucket, bucket_rows in sorted(buckets.items()):
        golds = [get_gold_label(row) for row in bucket_rows]
        current = [get_current_label(row) for row in bucket_rows]
        shadow = [resolve_shadow_label(row) for row in bucket_rows]
        out[bucket] = {
            "n": len(bucket_rows),
            "current_metrics": prediction_metrics(golds, current),
            "shadow_metrics": prediction_metrics(golds, shadow),
        }
    return out


def is_reverse_overclaim_row(row: dict[str, Any]) -> bool:
    group = get_group(row).lower()
    return (
        get_gold_label(row) == "NOT_ENTITLED"
        and (
            get_expected_route(row) == "OVERCLAIM_NE"
            or any(marker in group for marker in REVERSE_GROUP_MARKERS)
        )
    )


def stage35_reverse_overclaim_handling(rows: list[dict[str, Any]]) -> str:
    reverse_rows = [row for row in rows if is_reverse_overclaim_row(row)]
    if any(resolve_shadow_label(row) == "SUPPORT" for row in reverse_rows):
        return "unsafe"
    explicit = [
        row for row in reverse_rows
        if str(row.get("stage33_structured_coverage_route", "")) == "OVERCLAIM_NE"
        or str(row.get("stage33_conditional_override_type", "")) == "high_precision_overclaim"
    ]
    explicit_ids = {id(row) for row in explicit}
    fallback = [
        row for row in reverse_rows
        if resolve_shadow_label(row) == "NOT_ENTITLED" and id(row) not in explicit_ids
    ]
    if explicit and fallback:
        return "mixed"
    if explicit:
        return "explicit_overclaim_route"
    return "fallback_preserved_ne"


def stage35_scope_safety(counters: dict[str, int], row_count: int) -> str:
    scope_errors = (
        counters["adv_exception_to_support_error"]
        + counters["adv_temporal_scope_to_support_error"]
        + counters["adv_location_scope_to_support_error"]
    )
    if scope_errors == 0:
        return "safe"
    if scope_errors >= 5 or scope_errors / max(1, row_count) >= 0.05:
        return "unsafe"
    return "mixed"


def compute_counters(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counters = Counter()
    route_counts = Counter()
    reason_counts = Counter()
    action_counts = Counter()
    relation_counts = Counter()
    match_counts = Counter()
    for row in rows:
        group = get_group(row)
        gold = get_gold_label(row)
        shadow = resolve_shadow_label(row)
        route = str(row.get("stage33_structured_coverage_route", "MISSING"))
        reason = str(row.get("stage33_structured_coverage_reason", "MISSING"))
        action = str(row.get("stage33_conditional_action", "MISSING"))
        relation = str(row.get("stage33_structured_coverage_whole_part_relation", "MISSING"))
        match = str(row.get("stage33_structured_coverage_whole_part_match", ""))
        route_counts[route] += 1
        reason_counts[reason] += 1
        action_counts[action] += 1
        relation_counts[relation] += 1
        match_counts[match if match else "MISSING"] += 1

        if gold == "NOT_ENTITLED" and shadow == "SUPPORT":
            counters["adv_overclaim_to_support"] += 1
        if gold == "REFUTE" and shadow == "SUPPORT":
            counters["adv_refute_to_support"] += 1
        if gold == "SUPPORT" and shadow == "REFUTE":
            counters["adv_support_to_refute"] += 1
        if group in EXCEPTION_GROUPS and gold != "SUPPORT" and shadow == "SUPPORT":
            counters["adv_exception_to_support_error"] += 1
        if group == "adv_temporal_scope_not_entitled" and shadow == "SUPPORT":
            counters["adv_temporal_scope_to_support_error"] += 1
        if group == "adv_location_scope_not_entitled" and shadow == "SUPPORT":
            counters["adv_location_scope_to_support_error"] += 1

        if gold == "SUPPORT":
            if shadow == "SUPPORT":
                counters["adv_support_shadow_support"] += 1
            if shadow == "NOT_ENTITLED":
                counters["adv_support_shadow_ne"] += 1
            if shadow == "REFUTE":
                counters["adv_support_shadow_refute"] += 1
        if group in SUBSET_SUPPORT_GROUPS and shadow == "SUPPORT":
            counters["adv_subset_support_recovered"] += 1
        if group in EXCEPTION_SUPPORT_GROUPS and shadow == "SUPPORT":
            counters["adv_exception_support_recovered"] += 1
        if group in NUMERIC_SUPPORT_GROUPS and shadow == "SUPPORT":
            counters["adv_numeric_support_recovered"] += 1

        if is_reverse_overclaim_row(row) and shadow == "NOT_ENTITLED":
            counters["adv_reverse_ne_preserved"] += 1
            if route == "OVERCLAIM_NE" or str(row.get("stage33_conditional_override_type", "")) == "high_precision_overclaim":
                counters["adv_reverse_explicit_overclaim_route"] += 1
            else:
                counters["adv_reverse_fallback_preserved_ne"] += 1

        if match.startswith("pattern:"):
            counters["adv_pattern_match_count"] += 1
        elif match:
            counters["adv_known_lexicon_match_count"] += 1
        else:
            counters["adv_no_match_count"] += 1
        if normalize_bool(row.get("stage33_whole_part_direct_support_allowed")) is True:
            counters["adv_direct_support_allowed_count"] += 1
        if (
            normalize_bool(row.get("stage33_whole_part_direct_support_allowed")) is True
            and str(row.get("stage33_conditional_action", "")) == "fallback_current_final"
        ):
            counters["adv_allowed_but_fallback_count"] += 1
        if str(row.get("stage33_conditional_override_type", "")) == "whole_part_conditional_safe_direct_support":
            counters["adv_conditional_safe_override_count"] += 1

    for key in (
        "adv_overclaim_to_support",
        "adv_refute_to_support",
        "adv_support_to_refute",
        "adv_exception_to_support_error",
        "adv_temporal_scope_to_support_error",
        "adv_location_scope_to_support_error",
        "adv_support_shadow_support",
        "adv_support_shadow_ne",
        "adv_support_shadow_refute",
        "adv_subset_support_recovered",
        "adv_exception_support_recovered",
        "adv_numeric_support_recovered",
        "adv_reverse_ne_preserved",
        "adv_reverse_explicit_overclaim_route",
        "adv_reverse_fallback_preserved_ne",
        "adv_known_lexicon_match_count",
        "adv_pattern_match_count",
        "adv_no_match_count",
        "adv_direct_support_allowed_count",
        "adv_allowed_but_fallback_count",
        "adv_conditional_safe_override_count",
    ):
        counters.setdefault(key, 0)
    out = dict(counters)
    out["stage33_structured_coverage_route_counts"] = dict(route_counts)
    out["stage33_structured_coverage_reason_counts"] = dict(reason_counts)
    out["stage33_conditional_action_counts"] = dict(action_counts)
    out["stage33_whole_part_relation_counts"] = dict(relation_counts)
    out["stage33_whole_part_match_counts"] = dict(match_counts)
    out["stage33_conditional_override_type_counts"] = count_values(
        rows, "stage33_conditional_override_type"
    )
    out["stage33_structured_coverage_rule_strength_counts"] = count_values(
        rows, "stage33_structured_coverage_rule_strength"
    )
    return out


def recovery_summary(rows: list[dict[str, Any]], counters: dict[str, Any]) -> dict[str, Any]:
    support_total = (
        counters["adv_support_shadow_support"]
        + counters["adv_support_shadow_ne"]
        + counters["adv_support_shadow_refute"]
    )
    subset_total = sum(get_group(row) in SUBSET_SUPPORT_GROUPS for row in rows)
    exception_support_total = sum(get_group(row) in EXCEPTION_SUPPORT_GROUPS for row in rows)
    numeric_support_total = sum(get_group(row) in NUMERIC_SUPPORT_GROUPS for row in rows)
    return {
        "support_total": support_total,
        "support_shadow_support": counters["adv_support_shadow_support"],
        "support_shadow_ne": counters["adv_support_shadow_ne"],
        "support_shadow_refute": counters["adv_support_shadow_refute"],
        "support_shadow_support_rate": round(
            counters["adv_support_shadow_support"] / max(1, support_total), 4
        ),
        "subset_support_total": subset_total,
        "subset_support_recovered": counters["adv_subset_support_recovered"],
        "subset_support_recovery_rate": round(
            counters["adv_subset_support_recovered"] / max(1, subset_total), 4
        ),
        "exception_support_total": exception_support_total,
        "exception_support_recovered": counters["adv_exception_support_recovered"],
        "numeric_support_total": numeric_support_total,
        "numeric_support_recovered": counters["adv_numeric_support_recovered"],
    }


def decide(
    current_metrics: dict[str, Any],
    shadow_metrics: dict[str, Any],
    counters: dict[str, Any],
    recovery: dict[str, Any],
    reverse_handling: str,
    scope_safety: str,
) -> dict[str, str]:
    macro_delta = shadow_metrics["macro_f1"] - current_metrics["macro_f1"]
    unsafe_errors = (
        counters["adv_overclaim_to_support"]
        + counters["adv_refute_to_support"]
        + counters["adv_exception_to_support_error"]
        + counters["adv_temporal_scope_to_support_error"]
        + counters["adv_location_scope_to_support_error"]
    )
    if unsafe_errors >= 5 or unsafe_errors / max(1, recovery["support_total"]) >= 0.05:
        return {
            "label": "STAGE35A_ADVERSARIAL_UNSAFE",
            "reason": (
                "Adversarial overclaim, refute, exception, or scope rows were routed "
                f"to SUPPORT; reverse handling is {reverse_handling}, scope safety is {scope_safety}."
            ),
        }
    if (
        macro_delta >= 0.15
        and recovery["support_shadow_support_rate"] >= 0.80
        and recovery["subset_support_recovery_rate"] >= 0.80
        and unsafe_errors <= 2
        and scope_safety in {"safe", "mixed"}
    ):
        return {
            "label": "STAGE35A_ADVERSARIAL_GENERALIZATION_STRONG",
            "reason": (
                "Shadow predictions strongly improve current predictions while preserving "
                f"adversarial support recovery and safety; reverse handling is {reverse_handling}."
            ),
        }
    if (
        recovery["support_shadow_support_rate"] < 0.65
        or recovery["subset_support_recovery_rate"] < 0.65
    ) and unsafe_errors == 0:
        return {
            "label": "STAGE35A_TEMPLATE_GENERALIZATION_ONLY",
            "reason": (
                "Safety remains acceptable, but adversarial support recovery drops on "
                "non-Stage34-style templates."
            ),
        }
    return {
        "label": "STAGE35A_DIAGNOSTIC_ONLY",
        "reason": (
            "Adversarial behavior is mixed or inconclusive; "
            f"reverse handling is {reverse_handling}, scope safety is {scope_safety}."
        ),
    }


def stage36_blocker_counters(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Stage36-A: top-level counters over rows carrying blocker diagnostics."""
    fired = 0
    exception_fired = 0
    not_all_fired = 0
    location_fired = 0
    temporal_fired = 0
    blocked_to_ne = 0
    reason_counts: Counter = Counter()
    for row in rows:
        if row.get("stage36_support_blocker_fired") is not True:
            continue
        fired += 1
        if row.get("stage36_exception_blocker_fired") is True:
            exception_fired += 1
        if row.get("stage36_not_all_blocker_fired") is True:
            not_all_fired += 1
        if row.get("stage36_location_scope_blocker_fired") is True:
            location_fired += 1
        if row.get("stage36_temporal_scope_blocker_fired") is True:
            temporal_fired += 1
        if row.get("stage36_final_shadow_label") == "NOT_ENTITLED":
            blocked_to_ne += 1
        for reason in row.get("stage36_support_blocker_reasons") or []:
            reason_counts[str(reason)] += 1
    return {
        "stage36_support_blocker_fired_count": fired,
        "stage36_exception_blocker_fired_count": exception_fired,
        "stage36_not_all_blocker_fired_count": not_all_fired,
        "stage36_location_scope_blocker_fired_count": location_fired,
        "stage36_temporal_scope_blocker_fired_count": temporal_fired,
        "stage36_blocked_support_to_ne_count": blocked_to_ne,
        "stage36_blocker_reason_counts": dict(reason_counts),
    }


def stage36_before_after_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Stage36-A: compare unsafe-error and support-recovery counters before/after blocking."""

    def _compute(shadow_fn) -> tuple[int, int, int, int, int]:
        overclaim = 0
        exception_err = 0
        location_err = 0
        temporal_err = 0
        support_shadow_support = 0
        for row in rows:
            gold = get_gold_label(row)
            group = get_group(row)
            shadow = shadow_fn(row)
            if gold == "NOT_ENTITLED" and shadow == "SUPPORT":
                overclaim += 1
            if group in EXCEPTION_GROUPS and gold != "SUPPORT" and shadow == "SUPPORT":
                exception_err += 1
            if group == "adv_location_scope_not_entitled" and shadow == "SUPPORT":
                location_err += 1
            if group == "adv_temporal_scope_not_entitled" and shadow == "SUPPORT":
                temporal_err += 1
            if gold == "SUPPORT" and shadow == "SUPPORT":
                support_shadow_support += 1
        return overclaim, exception_err, location_err, temporal_err, support_shadow_support

    o_overclaim, o_exc, o_loc, o_temp, o_sss = _compute(resolve_shadow_label_original)
    p_overclaim, p_exc, p_loc, p_temp, p_sss = _compute(resolve_shadow_label)
    return {
        "stage36_original_overclaim_to_support": o_overclaim,
        "stage36_post_overclaim_to_support": p_overclaim,
        "stage36_original_exception_to_support_error": o_exc,
        "stage36_post_exception_to_support_error": p_exc,
        "stage36_original_location_scope_to_support_error": o_loc,
        "stage36_post_location_scope_to_support_error": p_loc,
        "stage36_original_temporal_scope_to_support_error": o_temp,
        "stage36_post_temporal_scope_to_support_error": p_temp,
        "stage36_original_support_shadow_support": o_sss,
        "stage36_post_support_shadow_support": p_sss,
    }


def stage36_decide(
    before_after: dict[str, Any],
    counters: dict[str, Any],
) -> dict[str, str]:
    """Stage36-A decision label. Only called when Stage36 fields are present."""
    original_scope_errors = (
        before_after["stage36_original_exception_to_support_error"]
        + before_after["stage36_original_location_scope_to_support_error"]
        + before_after["stage36_original_temporal_scope_to_support_error"]
    )
    post_scope_errors = (
        before_after["stage36_post_exception_to_support_error"]
        + before_after["stage36_post_location_scope_to_support_error"]
        + before_after["stage36_post_temporal_scope_to_support_error"]
    )
    overclaim_reduced = (
        before_after["stage36_original_overclaim_to_support"] > 0
        and before_after["stage36_post_overclaim_to_support"]
        < before_after["stage36_original_overclaim_to_support"]
    )
    scope_reduced = original_scope_errors > 0 and post_scope_errors < original_scope_errors
    refute_to_support_zero = counters.get("adv_refute_to_support", 0) == 0
    original_support = before_after["stage36_original_support_shadow_support"]
    post_support = before_after["stage36_post_support_shadow_support"]
    support_collapsed = original_support > 0 and (post_support / original_support) < 0.35
    unsafe_remaining = post_scope_errors + before_after["stage36_post_overclaim_to_support"]

    if not refute_to_support_zero:
        return {
            "label": "STAGE36A_DIAGNOSTIC_ONLY",
            "reason": (
                "REFUTE-to-SUPPORT leakage detected after Stage36; blockers cannot "
                "be judged safe from this run alone."
            ),
        }
    if unsafe_remaining >= 5:
        return {
            "label": "STAGE36A_SAFETY_BLOCKERS_INEFFECTIVE",
            "reason": (
                f"Unsafe SUPPORT errors remain high after Stage36 blocking "
                f"({unsafe_remaining} residual overclaim/scope-to-SUPPORT errors)."
            ),
        }
    if support_collapsed:
        return {
            "label": "STAGE36A_SAFETY_BLOCKERS_TOO_CONSERVATIVE",
            "reason": (
                "Safety improved but SUPPORT recovery collapsed below a useful "
                f"level ({post_support}/{original_support} SUPPORT rows retained)."
            ),
        }
    if (overclaim_reduced or scope_reduced) and unsafe_remaining <= 2:
        return {
            "label": "STAGE36A_SAFETY_BLOCKERS_EFFECTIVE",
            "reason": (
                "Overclaim and/or exception/location/temporal scope SUPPORT errors "
                "were reduced, REFUTE-to-SUPPORT leakage stayed at zero, and SUPPORT "
                "recovery was not fully collapsed."
            ),
        }
    return {
        "label": "STAGE36A_DIAGNOSTIC_ONLY",
        "reason": "Stage36 blocker impact on this run is mixed or inconclusive.",
    }


def stage37_safe_recovery_counters(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Stage37-A: top-level counters over rows carrying safe SUPPORT recovery diagnostics."""
    fired = 0
    no_except_fired = 0
    coord_fired = 0
    numeric_fired = 0
    from_not_entitled = 0
    from_refute = 0
    blocked_stage36 = 0
    blocked_scope = 0
    blocked_exception = 0
    blocked_not_all = 0
    reason_counts: Counter = Counter()
    for row in rows:
        if row.get("stage37_safe_recovery_fired") is True:
            fired += 1
            if row.get("stage37_no_except_included_subset_fired") is True:
                no_except_fired += 1
            if row.get("stage37_coordination_universal_subset_fired") is True:
                coord_fired += 1
            if row.get("stage37_numeric_universal_subset_fired") is True:
                numeric_fired += 1
            recovered_from = row.get("stage37_recovered_from_label")
            if recovered_from == "NOT_ENTITLED":
                from_not_entitled += 1
            elif recovered_from == "REFUTE":
                from_refute += 1
            for reason in row.get("stage37_safe_recovery_reasons") or []:
                reason_counts[str(reason)] += 1
        if row.get("stage37_blocked_by_stage36") is True:
            blocked_stage36 += 1
        if row.get("stage37_blocked_by_scope_hazard") is True:
            blocked_scope += 1
        if row.get("stage37_blocked_by_exception_hazard") is True:
            blocked_exception += 1
        if row.get("stage37_blocked_by_not_all_hazard") is True:
            blocked_not_all += 1
    return {
        "stage37_safe_recovery_fired_count": fired,
        "stage37_no_except_included_subset_fired_count": no_except_fired,
        "stage37_coordination_universal_subset_fired_count": coord_fired,
        "stage37_numeric_universal_subset_fired_count": numeric_fired,
        "stage37_recovered_from_not_entitled_count": from_not_entitled,
        "stage37_recovered_from_refute_count": from_refute,
        "stage37_blocked_by_stage36_count": blocked_stage36,
        "stage37_blocked_by_scope_hazard_count": blocked_scope,
        "stage37_blocked_by_exception_hazard_count": blocked_exception,
        "stage37_blocked_by_not_all_hazard_count": blocked_not_all,
        "stage37_safe_recovery_reason_counts": dict(reason_counts),
    }


def _stage37_diagnostic_snapshot(rows: list[dict[str, Any]], shadow_fn) -> dict[str, int]:
    overclaim = 0
    exception_err = 0
    location_err = 0
    refute_to_support = 0
    support_shadow_support = 0
    subset_recovered = 0
    numeric_recovered = 0
    exception_recovered = 0
    for row in rows:
        gold = get_gold_label(row)
        group = get_group(row)
        shadow = shadow_fn(row)
        if gold == "NOT_ENTITLED" and shadow == "SUPPORT":
            overclaim += 1
        if gold == "REFUTE" and shadow == "SUPPORT":
            refute_to_support += 1
        if group in EXCEPTION_GROUPS and gold != "SUPPORT" and shadow == "SUPPORT":
            exception_err += 1
        if group == "adv_location_scope_not_entitled" and shadow == "SUPPORT":
            location_err += 1
        if gold == "SUPPORT" and shadow == "SUPPORT":
            support_shadow_support += 1
        if group in SUBSET_SUPPORT_GROUPS and shadow == "SUPPORT":
            subset_recovered += 1
        if group in NUMERIC_SUPPORT_GROUPS and shadow == "SUPPORT":
            numeric_recovered += 1
        if group in EXCEPTION_SUPPORT_GROUPS and shadow == "SUPPORT":
            exception_recovered += 1
    return {
        "overclaim_to_support": overclaim,
        "exception_to_support_error": exception_err,
        "location_scope_to_support_error": location_err,
        "refute_to_support": refute_to_support,
        "support_shadow_support": support_shadow_support,
        "subset_support_recovered": subset_recovered,
        "numeric_support_recovered": numeric_recovered,
        "exception_support_recovered": exception_recovered,
    }


def stage37_before_after_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Stage37-A: compare unsafe-error and support-recovery counters before/after safe recovery."""
    before = _stage37_diagnostic_snapshot(rows, resolve_shadow_label_pre_stage37)
    after = _stage37_diagnostic_snapshot(rows, resolve_shadow_label)
    return {
        "stage37_original_support_shadow_support": before["support_shadow_support"],
        "stage37_post_support_shadow_support": after["support_shadow_support"],
        "stage37_original_subset_support_recovered": before["subset_support_recovered"],
        "stage37_post_subset_support_recovered": after["subset_support_recovered"],
        "stage37_original_numeric_support_recovered": before["numeric_support_recovered"],
        "stage37_post_numeric_support_recovered": after["numeric_support_recovered"],
        "stage37_original_exception_support_recovered": before["exception_support_recovered"],
        "stage37_post_exception_support_recovered": after["exception_support_recovered"],
        "stage37_original_overclaim_to_support": before["overclaim_to_support"],
        "stage37_post_overclaim_to_support": after["overclaim_to_support"],
        "stage37_original_exception_to_support_error": before["exception_to_support_error"],
        "stage37_post_exception_to_support_error": after["exception_to_support_error"],
        "stage37_original_location_scope_to_support_error": before["location_scope_to_support_error"],
        "stage37_post_location_scope_to_support_error": after["location_scope_to_support_error"],
        "stage37_original_refute_to_support": before["refute_to_support"],
        "stage37_post_refute_to_support": after["refute_to_support"],
    }


def stage37_decide(before_after: dict[str, Any]) -> dict[str, str]:
    """Stage37-A decision label. Only called when Stage37 fields are present."""
    support_increased = (
        before_after["stage37_post_support_shadow_support"]
        > before_after["stage37_original_support_shadow_support"]
    )
    subset_increased = (
        before_after["stage37_post_subset_support_recovered"]
        > before_after["stage37_original_subset_support_recovered"]
    )
    numeric_improved = (
        before_after["stage37_post_numeric_support_recovered"]
        > before_after["stage37_original_numeric_support_recovered"]
    )
    exception_improved = (
        before_after["stage37_post_exception_support_recovered"]
        > before_after["stage37_original_exception_support_recovered"]
    )
    overclaim_zero = before_after["stage37_post_overclaim_to_support"] == 0
    exception_err_zero = before_after["stage37_post_exception_to_support_error"] == 0
    location_err_zero = before_after["stage37_post_location_scope_to_support_error"] == 0
    refute_not_increased = (
        before_after["stage37_post_refute_to_support"]
        <= before_after["stage37_original_refute_to_support"]
    )
    unsafe_reappeared = (
        before_after["stage37_post_overclaim_to_support"] > 0
        or before_after["stage37_post_exception_to_support_error"] > 0
        or before_after["stage37_post_location_scope_to_support_error"] > 0
        or before_after["stage37_post_refute_to_support"]
        > before_after["stage37_original_refute_to_support"]
    )

    if unsafe_reappeared and (support_increased or subset_increased):
        return {
            "label": "STAGE37A_RECOVERY_TOO_UNSAFE",
            "reason": (
                "SUPPORT recovery improved but unsafe SUPPORT errors (overclaim, "
                "exception, location-scope, or REFUTE-to-SUPPORT leakage) reappeared."
            ),
        }
    if (
        support_increased
        and (subset_increased or numeric_improved or exception_improved)
        and overclaim_zero
        and exception_err_zero
        and location_err_zero
        and refute_not_increased
    ):
        return {
            "label": "STAGE37A_SAFE_SUPPORT_RECOVERY_EFFECTIVE",
            "reason": (
                "SUPPORT recovery increased (subset, numeric, and/or exception/"
                "no-except support improved) while overclaim, exception, and "
                "location-scope SUPPORT errors remained zero and REFUTE-to-"
                "SUPPORT leakage did not increase."
            ),
        }
    if (
        overclaim_zero
        and exception_err_zero
        and location_err_zero
        and refute_not_increased
        and not support_increased
        and not subset_increased
        and not numeric_improved
        and not exception_improved
    ):
        return {
            "label": "STAGE37A_RECOVERY_TOO_WEAK",
            "reason": "Safety remained good after Stage37, but SUPPORT recovery barely improved.",
        }
    return {
        "label": "STAGE37A_DIAGNOSTIC_ONLY",
        "reason": "Stage37 safe SUPPORT recovery impact on this run is mixed or inconclusive.",
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    recovery = report["support_recovery_summary"]
    lines = [
        "# Stage35-A Adversarial Coverage Evaluation",
        "",
        f"- Run: `{report['run_name']}`",
        f"- Rows: {report['row_count']}",
        f"- Decision: `{report['decision']['label']}`",
        f"- Decision reason: {report['decision']['reason']}",
        "",
        "## Aggregate Metrics",
        f"- Current accuracy: {report['current_metrics']['accuracy']:.4f}",
        f"- Current macro-F1: {report['current_metrics']['macro_f1']:.4f}",
        f"- Shadow accuracy: {report['shadow_metrics']['accuracy']:.4f}",
        f"- Shadow macro-F1: {report['shadow_metrics']['macro_f1']:.4f}",
        f"- Delta macro-F1: {report['delta']['shadow_minus_current_macro_f1']:.4f}",
        "",
        "## Support Recovery",
        "| Metric | Value |",
        "|---|---:|",
        f"| Support rows | {recovery['support_total']} |",
        f"| Shadow SUPPORT | {recovery['support_shadow_support']} |",
        f"| Shadow NOT_ENTITLED | {recovery['support_shadow_ne']} |",
        f"| Shadow REFUTE | {recovery['support_shadow_refute']} |",
        f"| Support recovery rate | {recovery['support_shadow_support_rate']:.4f} |",
        "",
        "## Subset/Whole-Part Adversarial Recovery",
        "| Metric | Value |",
        "|---|---:|",
        f"| Subset support recovered | {recovery['subset_support_recovered']} / {recovery['subset_support_total']} |",
        f"| Subset recovery rate | {recovery['subset_support_recovery_rate']:.4f} |",
        f"| Numeric support recovered | {recovery['numeric_support_recovered']} / {recovery['numeric_support_total']} |",
        "",
        "## Exception Stress Results",
        f"- Exception support recovered: {recovery['exception_support_recovered']} / {recovery['exception_support_total']}",
        f"- Exception-to-SUPPORT errors: {report['adv_exception_to_support_error']}",
        "",
        "## Temporal/Location Scope Stress Results",
        f"- Scope safety: `{report['stage35_scope_safety']}`",
        f"- Temporal scope SUPPORT errors: {report['adv_temporal_scope_to_support_error']}",
        f"- Location scope SUPPORT errors: {report['adv_location_scope_to_support_error']}",
        "",
        "## Reverse Overclaim Handling",
        f"- Handling: `{report['stage35_reverse_overclaim_handling']}`",
        f"- Reverse NE preserved: {report['adv_reverse_ne_preserved']}",
        f"- Explicit overclaim route: {report['adv_reverse_explicit_overclaim_route']}",
        f"- Fallback preserved NE: {report['adv_reverse_fallback_preserved_ne']}",
        "",
        "## Pattern Vs Lexicon Match Count",
        "| Match Type | Count |",
        "|---|---:|",
        f"| Known lexicon | {report['adv_known_lexicon_match_count']} |",
        f"| Pattern | {report['adv_pattern_match_count']} |",
        f"| No match | {report['adv_no_match_count']} |",
        "",
        "## Safety Counters",
        "| Counter | Value |",
        "|---|---:|",
    ]
    for key, value in sorted(report["adv_counters"].items()):
        if isinstance(value, int):
            lines.append(f"| {key} | {value} |")
    for title, key in (
        ("Route Counts", "stage33_structured_coverage_route_counts"),
        ("Reason Counts", "stage33_structured_coverage_reason_counts"),
        ("Conditional Action Counts", "stage33_conditional_action_counts"),
        ("Whole/Part Relation Counts", "stage33_whole_part_relation_counts"),
        ("Whole/Part Match Counts", "stage33_whole_part_match_counts"),
    ):
        lines.extend(["", f"## {title}", "| Value | Count |", "|---|---:|"])
        for name, count in sorted(report[key].items()):
            lines.append(f"| `{name}` | {count} |")
    if report.get("stage36_fields_present"):
        lines.extend([
            "",
            "## Stage36-A Support Safety Blockers",
            f"- Decision: `{report['stage36_decision']['label']}`",
            f"- Decision reason: {report['stage36_decision']['reason']}",
            "",
            "| Counter | Value |",
            "|---|---:|",
            f"| Blockers fired | {report['stage36_support_blocker_fired_count']} |",
            f"| Exception blocker fired | {report['stage36_exception_blocker_fired_count']} |",
            f"| Not-all blocker fired | {report['stage36_not_all_blocker_fired_count']} |",
            f"| Location scope blocker fired | {report['stage36_location_scope_blocker_fired_count']} |",
            f"| Temporal scope blocker fired | {report['stage36_temporal_scope_blocker_fired_count']} |",
            f"| Blocked SUPPORT -> NOT_ENTITLED | {report['stage36_blocked_support_to_ne_count']} |",
            "",
            "| Before/After | Original | Post-Stage36 |",
            "|---|---:|---:|",
            f"| Overclaim to SUPPORT | {report['stage36_original_overclaim_to_support']} | {report['stage36_post_overclaim_to_support']} |",
            f"| Exception to SUPPORT error | {report['stage36_original_exception_to_support_error']} | {report['stage36_post_exception_to_support_error']} |",
            f"| Location scope to SUPPORT error | {report['stage36_original_location_scope_to_support_error']} | {report['stage36_post_location_scope_to_support_error']} |",
            f"| Temporal scope to SUPPORT error | {report['stage36_original_temporal_scope_to_support_error']} | {report['stage36_post_temporal_scope_to_support_error']} |",
            f"| SUPPORT rows shadow==SUPPORT | {report['stage36_original_support_shadow_support']} | {report['stage36_post_support_shadow_support']} |",
            "",
            "### Blocker Reason Counts",
            "| Reason | Count |",
            "|---|---:|",
        ])
        for name, count in sorted(report["stage36_blocker_reason_counts"].items()):
            lines.append(f"| `{name}` | {count} |")
    if report.get("stage37_fields_present"):
        lines.extend([
            "",
            "## Stage37-A Safe SUPPORT Recovery",
            f"- Decision: `{report['stage37_decision']['label']}`",
            f"- Decision reason: {report['stage37_decision']['reason']}",
            "",
            "| Counter | Value |",
            "|---|---:|",
            f"| Safe recovery fired | {report['stage37_safe_recovery_fired_count']} |",
            f"| No-except included-subset fired | {report['stage37_no_except_included_subset_fired_count']} |",
            f"| Coordination universal-subset fired | {report['stage37_coordination_universal_subset_fired_count']} |",
            f"| Numeric universal-subset fired | {report['stage37_numeric_universal_subset_fired_count']} |",
            f"| Recovered from NOT_ENTITLED | {report['stage37_recovered_from_not_entitled_count']} |",
            f"| Recovered from REFUTE | {report['stage37_recovered_from_refute_count']} |",
            f"| Blocked by Stage36 blocker | {report['stage37_blocked_by_stage36_count']} |",
            f"| Blocked by scope hazard | {report['stage37_blocked_by_scope_hazard_count']} |",
            f"| Blocked by exception hazard | {report['stage37_blocked_by_exception_hazard_count']} |",
            f"| Blocked by not-all hazard | {report['stage37_blocked_by_not_all_hazard_count']} |",
            "",
            "| Before/After | Post-Stage36 | Post-Stage37 |",
            "|---|---:|---:|",
            f"| SUPPORT rows shadow==SUPPORT | {report['stage37_original_support_shadow_support']} | {report['stage37_post_support_shadow_support']} |",
            f"| Subset SUPPORT recovered | {report['stage37_original_subset_support_recovered']} | {report['stage37_post_subset_support_recovered']} |",
            f"| Numeric SUPPORT recovered | {report['stage37_original_numeric_support_recovered']} | {report['stage37_post_numeric_support_recovered']} |",
            f"| Exception SUPPORT recovered | {report['stage37_original_exception_support_recovered']} | {report['stage37_post_exception_support_recovered']} |",
            f"| Overclaim to SUPPORT | {report['stage37_original_overclaim_to_support']} | {report['stage37_post_overclaim_to_support']} |",
            f"| Exception to SUPPORT error | {report['stage37_original_exception_to_support_error']} | {report['stage37_post_exception_to_support_error']} |",
            f"| Location scope to SUPPORT error | {report['stage37_original_location_scope_to_support_error']} | {report['stage37_post_location_scope_to_support_error']} |",
            f"| REFUTE to SUPPORT | {report['stage37_original_refute_to_support']} | {report['stage37_post_refute_to_support']} |",
            "",
            "### Safe Recovery Reason Counts",
            "| Reason | Count |",
            "|---|---:|",
        ])
        for name, count in sorted(report["stage37_safe_recovery_reason_counts"].items()):
            lines.append(f"| `{name}` | {count} |")
    lines.extend([
        "",
        "## Caution",
        "Stage35-A is adversarial diagnostic only. It must not be used for training, calibration, threshold selection, loss, checkpoint selection, or Kaggle selection.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-file", required=True)
    parser.add_argument("--output-json", default="reports/stage35a_adversarial_coverage_eval.json")
    parser.add_argument("--output-md", default="reports/stage35a_adversarial_coverage_eval.md")
    parser.add_argument("--run-name", default="stage35a_adversarial")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = load_rows(Path(args.predictions_file))
    if not rows:
        print("ERROR: prediction file has no rows", file=sys.stderr)
        return 1
    gold_field = detect_field(rows, ["gold_final_label", "gold_label", "final_label", "label"], "gold label")
    current_field = detect_field(rows, ["pred_final_label", "pred_label", "prediction", "predicted_label"], "current prediction")
    shadow_field = detect_field(
        rows,
        [
            "stage37_final_shadow_label",
            "stage36_final_shadow_label",
            "stage32_shadow_label",
            "stage33_conditional_shadow_label",
        ],
        "shadow prediction",
    )
    golds = [get_gold_label(row) for row in rows]
    current = [get_current_label(row) for row in rows]
    shadow = [resolve_shadow_label(row) for row in rows]
    current_metrics = prediction_metrics(golds, current)
    shadow_metrics = prediction_metrics(golds, shadow)
    counters = compute_counters(rows)
    recovery = recovery_summary(rows, counters)
    reverse_handling = stage35_reverse_overclaim_handling(rows)
    scope_safety = stage35_scope_safety(counters, len(rows))
    scalar_counters = {
        key: value for key, value in counters.items() if isinstance(value, int)
    }
    stage36_present = has_stage36_fields(rows)
    stage37_present = has_stage37_fields(rows)
    report = {
        "run_name": args.run_name,
        "predictions_file": args.predictions_file,
        "row_count": len(rows),
        "fields": {
            "gold_field": gold_field,
            "current_pred_field": current_field,
            "shadow_pred_field": shadow_field,
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
        },
        "group_metrics": bucket_metrics(rows, get_group),
        "family_metrics": bucket_metrics(rows, get_family),
        "perturbation_metrics": bucket_metrics(rows, get_perturbation),
        "support_recovery_summary": recovery,
        "stage35_reverse_overclaim_handling": reverse_handling,
        "stage35_scope_safety": scope_safety,
        "adv_counters": counters,
        **{
            key: value for key, value in counters.items()
            if key.startswith("stage33_")
        },
        **scalar_counters,
        "decision": decide(
            current_metrics,
            shadow_metrics,
            counters,
            recovery,
            reverse_handling,
            scope_safety,
        ),
        "stage36_fields_present": stage36_present,
        "stage37_fields_present": stage37_present,
        "leakage_policy": (
            "Diagnostic-only. Do not use for training, calibration, threshold "
            "selection, loss, or checkpoint selection."
        ),
    }
    if stage36_present:
        stage36_counters = stage36_blocker_counters(rows)
        stage36_before_after = stage36_before_after_diagnostics(rows)
        report.update(stage36_counters)
        report.update(stage36_before_after)
        report["stage36_decision"] = stage36_decide(stage36_before_after, counters)
    if stage37_present:
        stage37_counters = stage37_safe_recovery_counters(rows)
        stage37_before_after = stage37_before_after_diagnostics(rows)
        report.update(stage37_counters)
        report.update(stage37_before_after)
        report["stage37_decision"] = stage37_decide(stage37_before_after)
    write_json(REPO_ROOT / args.output_json, report)
    write_markdown(REPO_ROOT / args.output_md, report)
    print(f"JSON report -> {REPO_ROOT / args.output_json}")
    print(f"Markdown report -> {REPO_ROOT / args.output_md}")
    print(f"Decision -> {report['decision']['label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
