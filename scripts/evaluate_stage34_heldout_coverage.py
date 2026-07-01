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


def group_metrics(
    rows: list[dict[str, Any]],
    gold_field: str,
    current_field: str,
    shadow_field: str,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("group", "MISSING"))].append(row)
    out: dict[str, Any] = {}
    for group, group_rows in sorted(grouped.items()):
        golds = [normalize_label(row[gold_field]) for row in group_rows]
        current = [normalize_label(row[current_field]) for row in group_rows]
        shadow = [normalize_label(row[shadow_field]) for row in group_rows]
        out[group] = {
            "n": len(group_rows),
            "current_metrics": prediction_metrics(golds, current),
            "shadow_metrics": prediction_metrics(golds, shadow),
            "shadow_reason_counts": dict(Counter(
                str(row.get("stage32_shadow_reason", "MISSING")) for row in group_rows
            )),
        }
    return out


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
    for row in rows:
        group = str(row.get("group", ""))
        gold = normalize_label(row[gold_field])
        current = normalize_label(row[current_field])
        shadow = normalize_label(row[shadow_field])
        route = str(row.get("stage33_structured_coverage_route", "MISSING"))
        reason = str(row.get("stage33_structured_coverage_reason", "MISSING"))
        match = str(row.get("stage33_structured_coverage_whole_part_match", ""))
        route_counts[route] += 1
        reason_counts[reason] += 1
        if match:
            match_counts[match] += 1

        if gold == "NOT_ENTITLED" and shadow == "SUPPORT":
            counters["heldout_overclaim_to_support"] += 1
        if gold == "REFUTE" and shadow == "SUPPORT":
            counters["heldout_refute_to_support"] += 1
        if gold == "SUPPORT" and shadow == "REFUTE":
            counters["heldout_support_to_refute"] += 1
        if group in SUPPORT_GROUPS:
            if shadow == "SUPPORT":
                counters["heldout_support_shadow_support"] += 1
            if shadow == "NOT_ENTITLED":
                counters["heldout_support_shadow_ne"] += 1
            if shadow == "REFUTE":
                counters["heldout_support_shadow_refute"] += 1
            if current == "SUPPORT":
                counters["heldout_support_current_support"] += 1
        if group in WHOLE_PART_SUPPORT_GROUPS:
            if shadow == "SUPPORT":
                counters["heldout_whole_to_part_support_recovered"] += 1
            if route == "RESIDUAL":
                counters["heldout_whole_part_unresolved"] += 1
        if group in WHOLE_PART_NE_GROUPS and shadow == "NOT_ENTITLED":
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
    return out


def decide(current_metrics: dict[str, Any], shadow_metrics: dict[str, Any], counters: dict[str, Any]) -> dict[str, str]:
    macro_delta = shadow_metrics["macro_f1"] - current_metrics["macro_f1"]
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
            "reason": "Held-out overclaim/refute rows were routed to SUPPORT.",
        }
    if support_gain >= 20 and macro_delta >= -0.02:
        return {
            "label": "STAGE34A_HELDOUT_GENERALIZATION_PROMISING",
            "reason": "Held-out SUPPORT recovery improves without safety errors or collapse.",
        }
    if (
        counters["heldout_whole_to_part_support_recovered"] < 6
        or counters["heldout_whole_part_unresolved"] > 30
    ):
        return {
            "label": "STAGE34A_HELDOUT_SYMBOLIC_MEMORIZATION_RISK",
            "reason": "Held-out whole/part support remains low or unresolved.",
        }
    return {
        "label": "STAGE34A_HELDOUT_DIAGNOSTIC_ONLY",
        "reason": "Held-out behavior is inconclusive.",
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Stage34-A Held-Out Structured Coverage Evaluation",
        "",
        f"- Run: `{report['run_name']}`",
        f"- Rows: {report['row_count']}",
        f"- Decision: `{report['decision']['label']}`",
        f"- Decision reason: {report['decision']['reason']}",
        "",
        "## Metrics",
        f"- Current accuracy: {report['current_metrics']['accuracy']:.4f}",
        f"- Current macro-F1: {report['current_metrics']['macro_f1']:.4f}",
        f"- Shadow accuracy: {report['shadow_metrics']['accuracy']:.4f}",
        f"- Shadow macro-F1: {report['shadow_metrics']['macro_f1']:.4f}",
        f"- Delta macro-F1: {report['delta']['shadow_minus_current_macro_f1']:.4f}",
        "",
        "## Safety And Recovery Counters",
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
    for title, key in (
        ("Stage33 Route Counts", "stage33_route_counts"),
        ("Stage33 Reason Counts", "stage33_reason_counts"),
        ("Whole/Part Match Counts", "stage33_whole_part_match_counts"),
    ):
        lines.extend(["", f"## {title}", "| Value | Count |", "|---|---:|"])
        for name, count in sorted(report["stage34_counters"][key].items()):
            lines.append(f"| `{name}` | {count} |")
    lines.extend([
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
    shadow_field = detect_field(rows, ["stage32_shadow_label"], "shadow prediction")
    golds = [normalize_label(row[gold_field]) for row in rows]
    current = [normalize_label(row[current_field]) for row in rows]
    shadow = [normalize_label(row[shadow_field]) for row in rows]
    current_metrics = prediction_metrics(golds, current)
    shadow_metrics = prediction_metrics(golds, shadow)
    counters = compute_stage34_counters(rows, gold_field, current_field, shadow_field)
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
        "stage34_counters": counters,
        "decision": decide(current_metrics, shadow_metrics, counters),
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
