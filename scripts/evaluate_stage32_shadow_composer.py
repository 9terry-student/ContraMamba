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
    if group_field is None:
        return None
    groups = {str(row.get(group_field, "")) for row in rows}
    if not groups.intersection(SUPPORT_ENTAILMENT_GROUPS | OVERCLAIM_GROUPS | REFUTE_GROUPS):
        return None

    diag = Counter()
    for row in rows:
        group = str(row.get(group_field, ""))
        current = normalize_label(row[current_field])
        shadow = normalize_label(row[shadow_field])
        if group in SUPPORT_ENTAILMENT_GROUPS:
            if current == "NOT_ENTITLED":
                diag["support_entailment_current_ne"] += 1
            if shadow == "NOT_ENTITLED":
                diag["support_entailment_shadow_ne"] += 1
                diag["shadow_support_to_ne"] += 1
            if shadow == "SUPPORT":
                diag["support_entailment_shadow_support"] += 1
            if shadow == "REFUTE":
                diag["shadow_support_to_refute"] += 1
        elif group in OVERCLAIM_GROUPS:
            if current == "SUPPORT":
                diag["overclaim_current_support"] += 1
            if shadow == "SUPPORT":
                diag["overclaim_shadow_support"] += 1
                diag["shadow_overclaim_to_support"] += 1
            if shadow == "NOT_ENTITLED":
                diag["overclaim_shadow_ne"] += 1
        elif group in REFUTE_GROUPS:
            if current == "SUPPORT":
                diag["refute_current_support"] += 1
            if shadow == "SUPPORT":
                diag["refute_shadow_support"] += 1
                diag["shadow_refute_to_support"] += 1
            if shadow == "REFUTE":
                diag["refute_shadow_refute"] += 1
            if shadow == "NOT_ENTITLED":
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
    ):
        diag.setdefault(key, 0)
    return dict(diag)


def decide(
    current_metrics: dict[str, Any],
    shadow_metrics: dict[str, Any],
    stage31_diag: dict[str, Any] | None,
) -> dict[str, str]:
    macro_delta = shadow_metrics["macro_f1"] - current_metrics["macro_f1"]
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
    else:
        refute_safety_increase = False
        overclaim_safety_increase = False
        support_recovery_improved = False
        support_collapse = False

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
    if macro_delta <= -0.02 or refute_safety_increase or overclaim_safety_increase or support_collapse:
        return {
            "label": "STAGE32_SHADOW_UNSAFE",
            "reason": (
                "Shadow route is not safe to apply: macro-F1 dropped materially, "
                "safety errors increased, or SUPPORT cases collapsed to NOT_ENTITLED."
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
        "group_metrics": compute_group_metrics(
            rows, group_field, gold_field, current_field, shadow_field
        ),
        "stage31_specific_diagnostics": stage31_diag,
        "decision": decide(current_metrics, shadow_metrics, stage31_diag),
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
