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
        shadow = [get_shadow_label(row) for row in bucket_rows]
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
    if any(get_shadow_label(row) == "SUPPORT" for row in reverse_rows):
        return "unsafe"
    explicit = [
        row for row in reverse_rows
        if str(row.get("stage33_structured_coverage_route", "")) == "OVERCLAIM_NE"
        or str(row.get("stage33_conditional_override_type", "")) == "high_precision_overclaim"
    ]
    explicit_ids = {id(row) for row in explicit}
    fallback = [
        row for row in reverse_rows
        if get_shadow_label(row) == "NOT_ENTITLED" and id(row) not in explicit_ids
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
        shadow = get_shadow_label(row)
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
        rows, ["stage32_shadow_label", "stage33_conditional_shadow_label"], "shadow prediction"
    )
    golds = [get_gold_label(row) for row in rows]
    current = [get_current_label(row) for row in rows]
    shadow = [get_shadow_label(row) for row in rows]
    current_metrics = prediction_metrics(golds, current)
    shadow_metrics = prediction_metrics(golds, shadow)
    counters = compute_counters(rows)
    recovery = recovery_summary(rows, counters)
    reverse_handling = stage35_reverse_overclaim_handling(rows)
    scope_safety = stage35_scope_safety(counters, len(rows))
    scalar_counters = {
        key: value for key, value in counters.items() if isinstance(value, int)
    }
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
        "leakage_policy": (
            "Diagnostic-only. Do not use for training, calibration, threshold "
            "selection, loss, or checkpoint selection."
        ),
    }
    write_json(REPO_ROOT / args.output_json, report)
    write_markdown(REPO_ROOT / args.output_md, report)
    print(f"JSON report -> {REPO_ROOT / args.output_json}")
    print(f"Markdown report -> {REPO_ROOT / args.output_md}")
    print(f"Decision -> {report['decision']['label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
