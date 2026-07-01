"""Evaluate Stage34-A held-out structured coverage predictions.

Diagnostic only. This evaluator reads prediction exports for the Stage34-A
held-out probe and must not be used for training, calibration, threshold
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
SUPPORT_GROUPS = {
    "heldout_all_to_some_support",
    "heldout_only_to_base_support",
    "heldout_specific_to_general_support",
    "heldout_whole_to_part_support",
    "heldout_collection_to_member_support",
    "heldout_region_to_subregion_support",
    "heldout_category_to_subcategory_support",
    "heldout_role_to_specialized_role_support",
    "heldout_material_to_variant_support",
}
REFUTE_GROUPS = {
    "heldout_none_to_some_refute",
    "heldout_some_to_none_refute",
}
WHOLE_PART_SUPPORT_GROUPS = {
    "heldout_whole_to_part_support",
    "heldout_collection_to_member_support",
    "heldout_region_to_subregion_support",
    "heldout_category_to_subcategory_support",
    "heldout_role_to_specialized_role_support",
    "heldout_material_to_variant_support",
}
WHOLE_PART_NE_GROUPS = {
    "heldout_part_to_whole_not_entitled",
    "heldout_member_to_collection_not_entitled",
    "heldout_subregion_to_region_not_entitled",
    "heldout_subcategory_to_category_not_entitled",
    "heldout_specialized_role_to_role_not_entitled",
    "heldout_variant_to_material_not_entitled",
}
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
    for key in (
        "group",
        "intervention_type",
        "normalized_intervention",
        "primary_failure_type",
    ):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return "UNKNOWN"


def resolve_group(row: dict[str, Any]) -> str:
    return get_group(row)


def infer_stage34_family(group: str) -> str:
    key = group.lower()
    if any(token in key for token in (
        "whole_to_part",
        "part_to_whole",
        "collection",
        "member",
        "region",
        "subregion",
        "category",
        "subcategory",
        "role",
        "specialized_role",
        "material",
        "variant",
    )):
        return "whole_part_family"
    if any(token in key for token in (
        "all_to_some",
        "some_to_all",
        "none_to_some",
        "some_to_none",
        "only_to_base",
        "also_to_only",
    )):
        return "logical_quantifier_family"
    if "specific_to_general" in key or "general_to_specific" in key:
        return "specific_general_family"
    return "unknown_family"


def resolve_stage34_family(row: dict[str, Any], group: str) -> str:
    value = row.get("stage34_family")
    if value not in (None, ""):
        return str(value)
    return infer_stage34_family(group)


def get_family(row: dict[str, Any]) -> str:
    return resolve_stage34_family(row, get_group(row))


def infer_stage34_relation(group: str) -> str:
    key = group.lower()
    if key.startswith("heldout_"):
        key = key[len("heldout_"):]
    for suffix in ("_support", "_not_entitled", "_refute"):
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return "unknown_relation"


def resolve_stage34_relation(row: dict[str, Any], group: str) -> str:
    value = row.get("stage34_relation")
    if value not in (None, ""):
        return str(value)
    return infer_stage34_relation(group)


def infer_expected_route(group: str, gold: str | None = None) -> str:
    key = group.lower()
    if key.endswith("_support"):
        return "ENTAILMENT_PRESERVE"
    if key.endswith("_refute"):
        return "CONTRADICTION_REFUTE"
    if key.endswith("_not_entitled"):
        return "OVERCLAIM_NE"
    if gold == "SUPPORT":
        return "ENTAILMENT_PRESERVE"
    if gold == "REFUTE":
        return "CONTRADICTION_REFUTE"
    if gold == "NOT_ENTITLED":
        return "OVERCLAIM_NE"
    return "UNKNOWN"


def resolve_expected_route(row: dict[str, Any], group: str) -> str:
    value = row.get("stage34_expected_route")
    if value not in (None, ""):
        return str(value)
    gold = None
    try:
        gold = get_gold_label(row)
    except ValueError:
        pass
    return infer_expected_route(group, gold)


def get_expected_route(row: dict[str, Any]) -> str:
    return resolve_expected_route(row, get_group(row))


def metadata_status(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = [resolve_group(row) for row in rows]
    unknown_count = sum(group == "UNKNOWN" for group in groups)
    has_any_stage34 = any(
        row.get("stage34_family") not in (None, "")
        or row.get("stage34_relation") not in (None, "")
        or row.get("stage34_expected_route") not in (None, "")
        or resolve_group(row) != "UNKNOWN"
        for row in rows
    )
    return {
        "metadata_available": bool(has_any_stage34 and unknown_count < len(rows)),
        "unknown_group_count": unknown_count,
        "unknown_group_rate": round(unknown_count / max(1, len(rows)), 4),
    }


def prediction_metrics(golds: list[str], preds: list[str]) -> dict[str, Any]:
    accuracy = sum(g == p for g, p in zip(golds, preds)) / max(1, len(golds))
    per_class: dict[str, dict[str, float]] = {}
    f1s: list[float] = []
    for label in LABELS:
        tp = sum(g == label and p == label for g, p in zip(golds, preds))
        fp = sum(g != label and p == label for g, p in zip(golds, preds))
        fn = sum(g == label and p != label for g, p in zip(golds, preds))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(g == label for g in golds),
        }
        f1s.append(f1)
    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(sum(f1s) / len(f1s), 4),
        "per_class": per_class,
        "prediction_counts": dict(Counter(preds)),
    }


def count_values(rows: list[dict[str, Any]], field: str, *, normalize_bools: bool = False) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        if field not in row or row.get(field) in (None, ""):
            counts["MISSING"] += 1
            continue
        value = normalize_bool(row.get(field)) if normalize_bools else row.get(field)
        counts[str(value)] += 1
    return dict(counts)


def is_reverse_overclaim_group(group: str) -> bool:
    key = group.lower()
    return any(token in key for token in (
        "to_whole",
        "to_collection",
        "to_region",
        "to_category",
        "to_role",
        "to_material",
        "some_to_all",
        "also_to_only",
        "general_to_specific",
    ))


def is_whole_part_support_group(group: str) -> bool:
    return group in WHOLE_PART_SUPPORT_GROUPS


def is_reverse_whole_part_group(group: str) -> bool:
    return group in WHOLE_PART_NE_GROUPS


def is_reverse_overclaim_row(row: dict[str, Any]) -> bool:
    group = get_group(row)
    return (
        get_gold_label(row) == "NOT_ENTITLED"
        and (
            get_expected_route(row) == "OVERCLAIM_NE"
            or is_reverse_overclaim_group(group)
        )
    )


def reverse_overclaim_handling(rows: list[dict[str, Any]]) -> str:
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


def group_metrics(
    rows: list[dict[str, Any]],
    gold_field: str,
    current_field: str,
    shadow_field: str,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[resolve_group(row)].append(row)
    out: dict[str, Any] = {}
    for group, group_rows in sorted(grouped.items()):
        golds = [get_gold_label(row) for row in group_rows]
        current = [get_current_label(row) for row in group_rows]
        shadow = [get_shadow_label(row) for row in group_rows]
        out[group] = {
            "n": len(group_rows),
            "current_metrics": prediction_metrics(golds, current),
            "shadow_metrics": prediction_metrics(golds, shadow),
            "shadow_reason_counts": dict(Counter(
                str(row.get("stage32_shadow_reason", "MISSING")) for row in group_rows
            )),
        }
    return out


def heldout_group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[get_group(row)].append(row)
    summary: dict[str, Any] = {}
    for group, group_rows in sorted(grouped.items()):
        match_counts = Counter(
            str(row.get("stage33_structured_coverage_whole_part_match", ""))
            for row in group_rows
            if row.get("stage33_structured_coverage_whole_part_match", "") not in (None, "")
        )
        summary[group] = {
            "n": len(group_rows),
            "gold_counts": dict(Counter(get_gold_label(row) for row in group_rows)),
            "current_counts": dict(Counter(get_current_label(row) for row in group_rows)),
            "shadow_counts": dict(Counter(get_shadow_label(row) for row in group_rows)),
            "stage33_reason_counts": count_values(group_rows, "stage33_structured_coverage_reason"),
            "stage33_route_counts": count_values(group_rows, "stage33_structured_coverage_route"),
            "whole_part_relation_counts": count_values(
                group_rows, "stage33_structured_coverage_whole_part_relation"
            ),
            "whole_part_match_counts_top10": dict(match_counts.most_common(10)),
            "conditional_action_counts": count_values(group_rows, "stage33_conditional_action"),
            "conditional_override_type_counts": count_values(
                group_rows, "stage33_conditional_override_type"
            ),
        }
    return summary


def stage33_top_level_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        "stage33_structured_coverage_route_counts": count_values(
            rows, "stage33_structured_coverage_route"
        ),
        "stage33_structured_coverage_reason_counts": count_values(
            rows, "stage33_structured_coverage_reason"
        ),
        "stage33_structured_coverage_rule_strength_counts": count_values(
            rows, "stage33_structured_coverage_rule_strength"
        ),
        "stage33_conditional_action_counts": count_values(
            rows, "stage33_conditional_action"
        ),
        "stage33_conditional_override_type_counts": count_values(
            rows, "stage33_conditional_override_type"
        ),
        "stage33_whole_part_relation_counts": count_values(
            rows, "stage33_structured_coverage_whole_part_relation"
        ),
        "stage33_whole_part_match_counts": count_values(
            rows, "stage33_structured_coverage_whole_part_match"
        ),
        "stage33_whole_part_direct_support_allowed_counts": count_values(
            rows, "stage33_whole_part_direct_support_allowed", normalize_bools=True
        ),
        "stage33_whole_part_direct_support_candidate_counts": count_values(
            rows, "stage33_whole_part_direct_support_candidate", normalize_bools=True
        ),
    }


def recovery_summary(rows: list[dict[str, Any]], counters: dict[str, Any]) -> dict[str, Any]:
    support_total = (
        counters["heldout_support_shadow_support"]
        + counters["heldout_support_shadow_ne"]
        + counters["heldout_support_shadow_refute"]
    )
    whole_part_support_total = sum(
        get_gold_label(row) == "SUPPORT" and is_whole_part_support_group(get_group(row))
        for row in rows
    )
    recovered = counters["heldout_whole_to_part_support_recovered"]
    return {
        "support_total": support_total,
        "support_shadow_support": counters["heldout_support_shadow_support"],
        "support_shadow_ne": counters["heldout_support_shadow_ne"],
        "support_shadow_refute": counters["heldout_support_shadow_refute"],
        "support_shadow_support_rate": round(
            counters["heldout_support_shadow_support"] / max(1, support_total), 4
        ),
        "whole_part_support_recovered": recovered,
        "whole_part_support_recovery_rate": None
        if whole_part_support_total == 0
        else round(recovered / whole_part_support_total, 4),
    }


def compute_stage34_counters(
    rows: list[dict[str, Any]],
    gold_field: str,
    current_field: str,
    shadow_field: str,
) -> dict[str, Any]:
    counters = Counter()
    block_reasons = Counter()
    route_counts = Counter()
    reason_counts = Counter()
    match_counts = Counter()
    family_counts = Counter()
    relation_counts = Counter()
    expected_route_counts = Counter()
    for row in rows:
        group = get_group(row)
        family = get_family(row)
        relation = resolve_stage34_relation(row, group)
        expected_route = get_expected_route(row)
        gold = get_gold_label(row)
        current = get_current_label(row)
        shadow = get_shadow_label(row)
        route = str(row.get("stage33_structured_coverage_route", "MISSING"))
        reason = str(row.get("stage33_structured_coverage_reason", "MISSING"))
        match = str(row.get("stage33_structured_coverage_whole_part_match", ""))
        route_counts[route] += 1
        reason_counts[reason] += 1
        if match:
            match_counts[match] += 1
        family_counts[family] += 1
        relation_counts[relation] += 1
        expected_route_counts[expected_route] += 1

        if gold == "NOT_ENTITLED" and shadow == "SUPPORT":
            counters["heldout_overclaim_to_support"] += 1
        if gold == "REFUTE" and shadow == "SUPPORT":
            counters["heldout_refute_to_support"] += 1
        if gold == "SUPPORT" and shadow == "REFUTE":
            counters["heldout_support_to_refute"] += 1
        if gold == "SUPPORT":
            if shadow == "SUPPORT":
                counters["heldout_support_shadow_support"] += 1
            if shadow == "NOT_ENTITLED":
                counters["heldout_support_shadow_ne"] += 1
            if shadow == "REFUTE":
                counters["heldout_support_shadow_refute"] += 1
            if current == "SUPPORT":
                counters["heldout_support_current_support"] += 1
        if family == "whole_part_family" and (
            route == "RESIDUAL" or reason == "no_structured_rule_fired"
        ):
            counters["heldout_whole_part_unresolved"] += 1
        if is_whole_part_support_group(group):
            if shadow == "SUPPORT":
                counters["heldout_whole_to_part_support_recovered"] += 1
        if is_reverse_whole_part_group(group) and gold == "NOT_ENTITLED" and shadow == "NOT_ENTITLED":
            counters["heldout_part_to_whole_ne_preserved"] += 1

        if match.startswith("pattern:"):
            counters["heldout_pattern_match_count"] += 1
        elif match:
            counters["heldout_known_lexicon_match_count"] += 1
        else:
            counters["heldout_no_match_count"] += 1
        if normalize_bool(row.get("stage33_whole_part_direct_support_allowed")) is True:
            counters["heldout_direct_support_allowed_count"] += 1
        if (
            normalize_bool(row.get("stage33_whole_part_direct_support_allowed")) is True
            and str(row.get("stage33_conditional_action", "")) == "fallback_current_final"
        ):
            counters["heldout_allowed_but_fallback_count"] += 1
        if str(row.get("stage33_conditional_override_type", "")) == "whole_part_conditional_safe_direct_support":
            counters["heldout_conditional_safe_override_count"] += 1
        block_reason = str(row.get("stage33_whole_part_direct_support_block_reason", ""))
        if block_reason:
            block_reasons[block_reason] += 1

    for key in (
        "heldout_overclaim_to_support",
        "heldout_refute_to_support",
        "heldout_support_to_refute",
        "heldout_support_shadow_support",
        "heldout_support_shadow_ne",
        "heldout_support_shadow_refute",
        "heldout_support_current_support",
        "heldout_whole_to_part_support_recovered",
        "heldout_part_to_whole_ne_preserved",
        "heldout_whole_part_unresolved",
        "heldout_known_lexicon_match_count",
        "heldout_pattern_match_count",
        "heldout_no_match_count",
        "heldout_direct_support_allowed_count",
        "heldout_allowed_but_fallback_count",
        "heldout_conditional_safe_override_count",
    ):
        counters.setdefault(key, 0)
    out = dict(counters)
    out["stage33_route_counts"] = dict(route_counts)
    out["stage33_reason_counts"] = dict(reason_counts)
    out["stage33_whole_part_match_counts"] = dict(match_counts)
    out["stage33_whole_part_direct_support_block_reason_counts"] = dict(block_reasons)
    out["stage33_structured_coverage_route_counts"] = dict(route_counts)
    out["stage33_structured_coverage_reason_counts"] = dict(reason_counts)
    out["stage34_family_counts"] = dict(family_counts)
    out["stage34_relation_counts"] = dict(relation_counts)
    out["stage34_expected_route_counts"] = dict(expected_route_counts)
    return out


def decide(
    current_metrics: dict[str, Any],
    shadow_metrics: dict[str, Any],
    counters: dict[str, Any],
    metadata: dict[str, Any],
    reverse_handling: str,
) -> dict[str, Any]:
    macro_delta = shadow_metrics["macro_f1"] - current_metrics["macro_f1"]
    if not metadata["metadata_available"]:
        return {
            "label": "STAGE34A_METADATA_MISSING_DIAGNOSTIC_INVALID",
            "reason": (
                "Held-out metadata is absent from prediction rows; aggregate metrics "
                "are available but group-specific generalization cannot be claimed."
            ),
            "aggregate_shadow_macro_f1": shadow_metrics["macro_f1"],
            "aggregate_delta_macro_f1": round(macro_delta, 4),
        }
    support_gain = (
        counters["heldout_support_shadow_support"]
        - counters["heldout_support_current_support"]
    )
    safety_errors = (
        counters["heldout_overclaim_to_support"]
        + counters["heldout_refute_to_support"]
    )
    if safety_errors > 0:
        return {
            "label": "STAGE34A_HELDOUT_UNSAFE",
            "reason": (
                "Held-out overclaim/refute rows were routed to SUPPORT; "
                f"reverse overclaim handling is {reverse_handling}."
            ),
        }
    if (
        support_gain >= 20
        and counters["heldout_whole_to_part_support_recovered"] >= 20
        and counters["heldout_part_to_whole_ne_preserved"] >= 20
        and macro_delta >= -0.02
    ):
        return {
            "label": "STAGE34A_HELDOUT_GENERALIZATION_PROMISING",
            "reason": (
                "Held-out SUPPORT recovery improves without safety errors or collapse; "
                f"reverse overclaim handling is {reverse_handling}."
            ),
        }
    if (
        counters["heldout_whole_to_part_support_recovered"] < 6
        or counters["heldout_whole_part_unresolved"] > 30
    ):
        return {
            "label": "STAGE34A_HELDOUT_SYMBOLIC_MEMORIZATION_RISK",
            "reason": (
                "Held-out whole/part support remains low or unresolved; "
                f"reverse overclaim handling is {reverse_handling}."
            ),
        }
    return {
        "label": "STAGE34A_HELDOUT_DIAGNOSTIC_ONLY",
        "reason": (
            "Held-out behavior is inconclusive; "
            f"reverse overclaim handling is {reverse_handling}."
        ),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    support_summary = report["support_recovery_summary"]
    whole_part_rate = support_summary["whole_part_support_recovery_rate"]
    whole_part_rate_text = "n/a" if whole_part_rate is None else f"{whole_part_rate:.4f}"
    lines = [
        "# Stage34-A Held-Out Structured Coverage Evaluation",
        "",
        f"- Run: `{report['run_name']}`",
        f"- Rows: {report['row_count']}",
        f"- Decision: `{report['decision']['label']}`",
        f"- Decision reason: {report['decision']['reason']}",
        f"- Metadata available: {report['metadata_status']['metadata_available']}",
        f"- Unknown group rows: {report['metadata_status']['unknown_group_count']}",
        "",
        "## Metrics",
        f"- Current accuracy: {report['current_metrics']['accuracy']:.4f}",
        f"- Current macro-F1: {report['current_metrics']['macro_f1']:.4f}",
        f"- Shadow accuracy: {report['shadow_metrics']['accuracy']:.4f}",
        f"- Shadow macro-F1: {report['shadow_metrics']['macro_f1']:.4f}",
        f"- Delta macro-F1: {report['delta']['shadow_minus_current_macro_f1']:.4f}",
        "",
        "## Support Recovery Summary",
        "| Metric | Value |",
        "|---|---:|",
        f"| Support rows | {support_summary['support_total']} |",
        f"| Shadow SUPPORT on support rows | {support_summary['support_shadow_support']} |",
        f"| Shadow NOT_ENTITLED on support rows | {support_summary['support_shadow_ne']} |",
        f"| Shadow REFUTE on support rows | {support_summary['support_shadow_refute']} |",
        f"| Support recovery rate | {support_summary['support_shadow_support_rate']:.4f} |",
        "",
        "## Whole/Part-Family Support Recovery",
        "| Metric | Value |",
        "|---|---:|",
        f"| Recovered whole/part support rows | {support_summary['whole_part_support_recovered']} |",
        f"| Whole/part support recovery rate | {whole_part_rate_text} |",
        f"| Whole/part unresolved rows | {report['heldout_whole_part_unresolved']} |",
        "",
        "## Reverse Overclaim Handling",
        f"- Handling: `{report['stage34_reverse_overclaim_handling']}`",
        f"- Reverse whole/part NE preserved: {report['heldout_part_to_whole_ne_preserved']}",
        "",
        "## Safety Counters",
        "| Counter | Value |",
        "|---|---:|",
    ]
    scalar_counters = {
        key: value
        for key, value in report["stage34_counters"].items()
        if isinstance(value, int)
    }
    for key, value in sorted(scalar_counters.items()):
        lines.append(f"| {key} | {value} |")
    lines.extend([
        "",
        "## Pattern Vs Lexicon Match Count",
        "| Match Type | Count |",
        "|---|---:|",
        f"| Known lexicon | {report['heldout_known_lexicon_match_count']} |",
        f"| Pattern | {report['heldout_pattern_match_count']} |",
        f"| No match | {report['heldout_no_match_count']} |",
    ])
    for title, key in (
        ("Stage33 Route Counts", "stage33_structured_coverage_route_counts"),
        ("Stage33 Reason Counts", "stage33_structured_coverage_reason_counts"),
        ("Stage33 Rule Strength Counts", "stage33_structured_coverage_rule_strength_counts"),
        ("Conditional Action Counts", "stage33_conditional_action_counts"),
        ("Conditional Override Type Counts", "stage33_conditional_override_type_counts"),
        ("Whole/Part Relation Counts", "stage33_whole_part_relation_counts"),
        ("Whole/Part Match Counts", "stage33_whole_part_match_counts"),
        ("Stage34 Family Counts", "stage34_family_counts"),
        ("Stage34 Relation Counts", "stage34_relation_counts"),
        ("Stage34 Expected Route Counts", "stage34_expected_route_counts"),
    ):
        lines.extend(["", f"## {title}", "| Value | Count |", "|---|---:|"])
        source = report if key in report else report["stage34_counters"]
        for name, count in sorted(source[key].items()):
            lines.append(f"| `{name}` | {count} |")
    lines.extend([
        "",
        "## Caution",
        "Support-oriented held-out subset relations generalize strongly when the promising decision fires. Reverse overclaim safety is mostly fallback-preserved unless an explicit `OVERCLAIM_NE` route is observed.",
        "",
        "## Leakage Policy",
        "This held-out probe is diagnostic-only and must not be used for training, calibration, threshold selection, loss, or checkpoint selection.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-file", required=True)
    parser.add_argument("--output-json", default="reports/stage34a_heldout_coverage_eval.json")
    parser.add_argument("--output-md", default="reports/stage34a_heldout_coverage_eval.md")
    parser.add_argument("--run-name", default="stage34a_heldout")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = load_rows(Path(args.predictions_file))
    if not rows:
        print("ERROR: prediction file has no rows", file=sys.stderr)
        return 1
    gold_field = detect_field(rows, ["gold_label", "gold_final_label", "final_label", "label"], "gold label")
    current_field = detect_field(rows, ["pred_final_label", "pred_label", "prediction", "predicted_label"], "current prediction")
    shadow_field = detect_field(
        rows, ["stage32_shadow_label", "stage33_conditional_shadow_label"], "shadow prediction"
    )
    golds = [get_gold_label(row) for row in rows]
    current = [get_current_label(row) for row in rows]
    shadow = [get_shadow_label(row) for row in rows]
    current_metrics = prediction_metrics(golds, current)
    shadow_metrics = prediction_metrics(golds, shadow)
    counters = compute_stage34_counters(rows, gold_field, current_field, shadow_field)
    metadata = metadata_status(rows)
    top_level_counts = stage33_top_level_counts(rows)
    reverse_handling = reverse_overclaim_handling(rows)
    summary = recovery_summary(rows, counters)
    group_summary = heldout_group_summary(rows)
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
        "group_metrics": group_metrics(rows, gold_field, current_field, shadow_field),
        "heldout_group_summary": group_summary,
        "support_recovery_summary": summary,
        "stage34_reverse_overclaim_handling": reverse_handling,
        "stage34_counters": counters,
        **top_level_counts,
        **scalar_counters,
        "metadata_status": metadata,
        "decision": decide(
            current_metrics, shadow_metrics, counters, metadata, reverse_handling
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
