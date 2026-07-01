"""Import-safe Stage43-C0 external fact-verification evaluation utilities.

Pure reporting/metrics helpers for the post-training hook in
scripts/train_controlled_v6b_minimal.py. This module does not import torch,
does not train, does not select checkpoints, and does not modify composer
behavior.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

LABELS = ["SUPPORT", "REFUTE", "NOT_ENTITLED"]
VALID_LABELS = set(LABELS)
SAMPLE_LIMIT = 20

LEAKAGE_POLICY = (
    "Stage43-C0 external fact-verification data is evaluation-only. It is not "
    "used for training, calibration, threshold selection, checkpoint selection, "
    "loss design, model selection, or composer behavior changes."
)


def normalize_label(raw: Any) -> str | None:
    if raw is None:
        return None
    key = str(raw).strip().upper().replace("-", "_")
    mapping = {
        "SUPPORT": "SUPPORT",
        "SUPPORTS": "SUPPORT",
        "REFUTE": "REFUTE",
        "REFUTES": "REFUTE",
        "NOT_ENTITLED": "NOT_ENTITLED",
        "NOT ENOUGH INFO": "NOT_ENTITLED",
        "NOT_ENOUGH_INFO": "NOT_ENTITLED",
        "NEI": "NOT_ENTITLED",
    }
    return mapping.get(key)


def json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return json_safe(value)


def load_stage43_jsonl(path: Path, max_rows: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if max_rows is not None and len(rows) >= max_rows:
                break
            text = line.strip()
            if not text:
                continue
            obj = json.loads(text)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}: line {line_no} is not a JSON object")
            rows.append(obj)
    return rows


def validate_stage43_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    required = ("id", "claim", "evidence", "label", "source_dataset", "source_label", "stage43_split", "metadata")
    for index, row in enumerate(rows):
        row_errors = []
        for field in required:
            if field not in row:
                row_errors.append(f"missing_{field}")
        if normalize_label(row.get("label")) not in VALID_LABELS:
            row_errors.append("invalid_label")
        if not str(row.get("claim") or "").strip():
            row_errors.append("empty_claim")
        if not str(row.get("evidence") or "").strip():
            row_errors.append("empty_evidence")
        if row_errors:
            errors.append(
                {
                    "row_index": index,
                    "id": row.get("id"),
                    "errors": row_errors,
                }
            )
            if len(errors) >= SAMPLE_LIMIT:
                break
    label_counts = Counter(normalize_label(row.get("label")) for row in rows)
    label_counts.pop(None, None)
    if len(label_counts) < 2:
        errors.append(
            {
                "row_index": None,
                "id": None,
                "errors": ["fewer_than_two_labels_represented"],
            }
        )
    return errors


def stage43_rows_to_controlled_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        label = normalize_label(row.get("label"))
        if label is None:
            continue
        row_id = str(row.get("id") or f"stage43_row_{index}")
        records.append(
            {
                **row,
                "id": row_id,
                "pair_id": str(row.get("pair_id") or row_id),
                "claim": str(row.get("claim") or ""),
                "evidence": str(row.get("evidence") or ""),
                "final_label": label,
                "intervention_type": "stage43_external_factver",
                "normalized_intervention": "stage43_external_factver",
                "source_intervention_type": "",
                "primary_failure_type": "stage43_external_factver",
                "frame_compatible_label": 0,
                "predicate_covered_label": 0,
                "sufficiency_label": 0,
                "polarity_label": "NONE",
            }
        )
    return records


def _prediction_label(row: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        if key in row:
            label = normalize_label(row.get(key))
            if label is not None:
                return label
    return None


def get_base_prediction(pred_row: dict[str, Any]) -> str | None:
    return _prediction_label(
        pred_row,
        (
            "stage39_original_final_label",
            "stage39_original_pred_final_label",
            "pred_final_label",
            "pred_label",
            "prediction",
            "predicted_label",
        ),
    )


def get_composed_prediction(pred_row: dict[str, Any]) -> tuple[str | None, bool, str | None]:
    composed = _prediction_label(pred_row, ("stage39_composed_final_label", "composed_prediction"))
    if composed is not None:
        return composed, True, str(pred_row.get("stage39_composer_reason") or "")
    base = get_base_prediction(pred_row)
    return base, False, None


def empty_confusion() -> dict[str, dict[str, int]]:
    return {gold: {pred: 0 for pred in LABELS} for gold in LABELS}


def compute_metrics(golds: list[str], preds: list[str]) -> dict[str, Any]:
    confusion = empty_confusion()
    for gold, pred in zip(golds, preds):
        confusion[gold][pred] += 1
    total = len(golds)
    correct = sum(gold == pred for gold, pred in zip(golds, preds))
    per_label: dict[str, Any] = {}
    f1s: list[float] = []
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[gold][label] for gold in LABELS if gold != label)
        fn = sum(confusion[label][pred] for pred in LABELS if pred != label)
        support = sum(confusion[label].values())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
        f1s.append(f1)
        per_label[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": support,
        }
    return {
        "accuracy": round(correct / total, 6) if total else None,
        "macro_f1": round(sum(f1s) / len(f1s), 6) if f1s else None,
        "per_label": per_label,
        "confusion_matrix": confusion,
        "prediction_counts": dict(Counter(preds)),
    }


def _incomplete_report(
    *,
    input_jsonl: Path,
    run_name: str,
    output_predictions_path: str | None,
    rows: list[dict[str, Any]],
    sample_errors: list[dict[str, Any]],
    reason: str,
) -> dict[str, Any]:
    gold_counts = Counter(normalize_label(row.get("label")) for row in rows)
    gold_counts.pop(None, None)
    return {
        "decision": "STAGE43C0_EXTERNAL_FACTVER_INCOMPLETE",
        "input_jsonl": str(input_jsonl),
        "run_name": run_name,
        "row_count": len(rows),
        "gold_label_counts": dict(gold_counts),
        "base_prediction_counts": {},
        "composed_prediction_counts": {},
        "base_accuracy": None,
        "base_macro_f1": None,
        "composed_accuracy": None,
        "composed_macro_f1": None,
        "delta_accuracy": None,
        "delta_macro_f1": None,
        "base_per_label": {},
        "composed_per_label": {},
        "base_confusion_matrix": empty_confusion(),
        "composed_confusion_matrix": empty_confusion(),
        "changed_row_count": 0,
        "changed_to_SUPPORT_count": 0,
        "changed_to_REFUTE_count": 0,
        "changed_to_NOT_ENTITLED_count": 0,
        "introduced_unsafe_SUPPORT_count": 0,
        "introduced_REFUTE_to_SUPPORT_count": 0,
        "introduced_SUPPORT_to_REFUTE_count": 0,
        "total_composed_wrong_SUPPORT_count": 0,
        "total_composed_SUPPORT_to_REFUTE_count": 0,
        "blocker_fired_count": 0,
        "recovery_fired_count": 0,
        "composer_mode": "safe_structured_v2",
        "prediction_source": "post_training_in_memory_best_state",
        "stage43_external_eval_timing": "post_training_after_best_state_restore",
        "used_for_training": False,
        "used_for_checkpoint_selection": False,
        "used_for_threshold_selection": False,
        "output_predictions_jsonl": output_predictions_path,
        "sample_changed_rows": [],
        "sample_error_rows": sample_errors[:SAMPLE_LIMIT],
        "risks": [reason],
        "recommendation": reason,
        "leakage_policy": LEAKAGE_POLICY,
    }


def analyze_stage43_predictions(
    *,
    input_jsonl: Path,
    run_name: str,
    rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    output_predictions_path: str | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    validation_errors = validate_stage43_rows(rows)
    if validation_errors:
        return _incomplete_report(
            input_jsonl=input_jsonl,
            run_name=run_name,
            output_predictions_path=output_predictions_path,
            rows=rows,
            sample_errors=validation_errors,
            reason="Input schema is invalid or has fewer than two labels represented.",
        ), []

    pred_by_id = {str(row.get("id")): row for row in prediction_rows}
    clean_golds: list[str] = []
    base_preds: list[str] = []
    composed_preds: list[str] = []
    output_rows: list[dict[str, Any]] = []
    sample_errors: list[dict[str, Any]] = []
    sample_changed: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    composer_unavailable_count = 0

    for index, row in enumerate(rows):
        row_id = str(row.get("id"))
        pred_row = pred_by_id.get(row_id)
        gold = normalize_label(row.get("label"))
        if pred_row is None or gold is None:
            if len(sample_errors) < SAMPLE_LIMIT:
                sample_errors.append(
                    {
                        "row_index": index,
                        "id": row_id,
                        "error": "missing_prediction_or_gold_label",
                    }
                )
            continue
        base = get_base_prediction(pred_row)
        composed, composer_available, composer_reason = get_composed_prediction(pred_row)
        if base is None or composed is None:
            if len(sample_errors) < SAMPLE_LIMIT:
                sample_errors.append(
                    {
                        "row_index": index,
                        "id": row_id,
                        "error": "missing_base_or_composed_prediction",
                    }
                )
            continue
        if not composer_available or composer_reason == "missing_source_shadow_label":
            composer_unavailable_count += 1

        clean_golds.append(gold)
        base_preds.append(base)
        composed_preds.append(composed)
        changed = base != composed
        if changed:
            counters["changed_row_count"] += 1
            counters[f"changed_to_{composed}_count"] += 1
            if len(sample_changed) < SAMPLE_LIMIT:
                sample_changed.append(
                    {
                        "id": row_id,
                        "gold_label": gold,
                        "base_prediction": base,
                        "composed_prediction": composed,
                        "composer_reason": composer_reason,
                    }
                )
        if changed and base == "REFUTE" and composed == "SUPPORT":
            counters["introduced_REFUTE_to_SUPPORT_count"] += 1
        if changed and base == "SUPPORT" and composed == "REFUTE":
            counters["introduced_SUPPORT_to_REFUTE_count"] += 1
        if changed and composed == "SUPPORT" and gold != "SUPPORT" and base != "SUPPORT":
            counters["introduced_unsafe_SUPPORT_count"] += 1
        if composed == "SUPPORT" and gold != "SUPPORT":
            counters["total_composed_wrong_SUPPORT_count"] += 1
        if gold == "SUPPORT" and composed == "REFUTE":
            counters["total_composed_SUPPORT_to_REFUTE_count"] += 1
        if bool(pred_row.get("stage36_support_blocker_fired") or pred_row.get("stage39_blocked_by_stage36")):
            counters["blocker_fired_count"] += 1
        if bool(pred_row.get("stage37_safe_recovery_fired")):
            counters["recovery_fired_count"] += 1

        output_rows.append(
            {
                "id": row_id,
                "claim": row.get("claim"),
                "evidence": row.get("evidence"),
                "gold_label": gold,
                "base_prediction": base,
                "composed_prediction": composed,
                "changed": changed,
                "composer_mode": "safe_structured_v2",
                "composer_reason": composer_reason,
                "blocker_fired": bool(pred_row.get("stage36_support_blocker_fired") or pred_row.get("stage39_blocked_by_stage36")),
                "recovery_fired": bool(pred_row.get("stage37_safe_recovery_fired")),
                "source_dataset": row.get("source_dataset"),
                "metadata": row.get("metadata") or {},
            }
        )

    if not output_rows:
        return _incomplete_report(
            input_jsonl=input_jsonl,
            run_name=run_name,
            output_predictions_path=output_predictions_path,
            rows=rows,
            sample_errors=sample_errors,
            reason="Model predictions could not be produced for Stage43 external rows.",
        ), []

    if len(output_rows) != len(rows):
        sample_errors.append(
            {
                "error": "partial_prediction_coverage",
                "detail": f"Only {len(output_rows)} of {len(rows)} rows had usable predictions.",
            }
        )

    base_metrics = compute_metrics(clean_golds, base_preds)
    composed_metrics = compute_metrics(clean_golds, composed_preds)
    gold_counts = dict(Counter(clean_golds))
    report = {
        "decision": "STAGE43C0_EXTERNAL_FACTVER_INCOMPLETE",
        "input_jsonl": str(input_jsonl),
        "run_name": run_name,
        "row_count": len(output_rows),
        "gold_label_counts": gold_counts,
        "base_prediction_counts": base_metrics["prediction_counts"],
        "composed_prediction_counts": composed_metrics["prediction_counts"],
        "base_accuracy": base_metrics["accuracy"],
        "base_macro_f1": base_metrics["macro_f1"],
        "composed_accuracy": composed_metrics["accuracy"],
        "composed_macro_f1": composed_metrics["macro_f1"],
        "delta_accuracy": round(composed_metrics["accuracy"] - base_metrics["accuracy"], 6),
        "delta_macro_f1": round(composed_metrics["macro_f1"] - base_metrics["macro_f1"], 6),
        "base_per_label": base_metrics["per_label"],
        "composed_per_label": composed_metrics["per_label"],
        "base_confusion_matrix": base_metrics["confusion_matrix"],
        "composed_confusion_matrix": composed_metrics["confusion_matrix"],
        "changed_row_count": counters["changed_row_count"],
        "changed_to_SUPPORT_count": counters["changed_to_SUPPORT_count"],
        "changed_to_REFUTE_count": counters["changed_to_REFUTE_count"],
        "changed_to_NOT_ENTITLED_count": counters["changed_to_NOT_ENTITLED_count"],
        "introduced_unsafe_SUPPORT_count": counters["introduced_unsafe_SUPPORT_count"],
        "introduced_REFUTE_to_SUPPORT_count": counters["introduced_REFUTE_to_SUPPORT_count"],
        "introduced_SUPPORT_to_REFUTE_count": counters["introduced_SUPPORT_to_REFUTE_count"],
        "total_composed_wrong_SUPPORT_count": counters["total_composed_wrong_SUPPORT_count"],
        "total_composed_SUPPORT_to_REFUTE_count": counters["total_composed_SUPPORT_to_REFUTE_count"],
        "blocker_fired_count": counters["blocker_fired_count"],
        "recovery_fired_count": counters["recovery_fired_count"],
        "composer_mode": "safe_structured_v2",
        "prediction_source": "post_training_in_memory_best_state",
        "stage43_external_eval_timing": "post_training_after_best_state_restore",
        "used_for_training": False,
        "used_for_checkpoint_selection": False,
        "used_for_threshold_selection": False,
        "output_predictions_jsonl": output_predictions_path,
        "sample_changed_rows": sample_changed,
        "sample_error_rows": sample_errors[:SAMPLE_LIMIT],
        "risks": [
            "Stage43 external data is evaluation-only and must not guide training or selection.",
            "Climate-FEVER failures are cross-domain limitations, not training signals.",
        ],
        "recommendation": "",
        "leakage_policy": LEAKAGE_POLICY,
    }
    decision, recommendation = decide_stage43c0(report, len(rows), composer_unavailable_count)
    report["decision"] = decision
    report["recommendation"] = recommendation
    if composer_unavailable_count == len(output_rows):
        report["risks"].append("safe_structured_v2 composer output was unavailable for every evaluated row.")
    return report, output_rows


def decide_stage43c0(report: dict[str, Any], input_row_count: int, composer_unavailable_count: int) -> tuple[str, str]:
    if report["row_count"] != input_row_count:
        return (
            "STAGE43C0_EXTERNAL_FACTVER_INCOMPLETE",
            "Model predictions were not produced for every external row.",
        )
    if composer_unavailable_count == report["row_count"]:
        return (
            "STAGE43C0_EXTERNAL_FACTVER_INCOMPLETE",
            "safe_structured_v2 could not be applied for the evaluated rows.",
        )
    if report["row_count"] < 100 or len(report["gold_label_counts"]) < 2:
        return (
            "STAGE43C0_EXTERNAL_FACTVER_INCOMPLETE",
            "Input has fewer than 100 rows or fewer than two labels represented.",
        )
    unsafe = (
        report["introduced_unsafe_SUPPORT_count"] > 0
        or report["introduced_REFUTE_to_SUPPORT_count"] > 0
        or report["introduced_SUPPORT_to_REFUTE_count"] > 0
    )
    if unsafe:
        return (
            "STAGE43C0_EXTERNAL_FACTVER_UNSAFE",
            "Composition introduced unsafe SUPPORT or SUPPORT/REFUTE transition(s).",
        )
    if (
        all(label in report["gold_label_counts"] for label in LABELS)
        and report["composed_macro_f1"] is not None
        and report["base_macro_f1"] is not None
        and report["composed_macro_f1"] >= report["base_macro_f1"]
    ):
        return (
            "STAGE43C0_EXTERNAL_FACTVER_PASS",
            "All three labels are represented, composed macro-F1 is at least base macro-F1, and no introduced safety transitions were observed.",
        )
    return (
        "STAGE43C0_EXTERNAL_FACTVER_SAFE_BUT_NO_GAIN",
        "No introduced safety transitions were observed, but composed macro-F1 did not improve over base or not all labels were represented.",
    )


def render_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Stage43-C0 External Fact-Verification Evaluation",
        "",
        "## 1. Decision",
        "",
        f"- Run name: `{report['run_name']}`",
        f"- Decision: `{report['decision']}`",
        f"- Recommendation: {report['recommendation']}",
        "",
        "## 2. Dataset/input summary",
        "",
        f"- Input JSONL: `{report['input_jsonl']}`",
        f"- Row count: {report['row_count']}",
        f"- Prediction source: `{report['prediction_source']}`",
        "",
        "## 3. Evaluation timing",
        "",
        f"`{report['stage43_external_eval_timing']}`",
        "",
        "## 4. Base vs composed metrics",
        "",
        "| Metric | Base | Composed | Delta |",
        "|---|---:|---:|---:|",
        f"| accuracy | {report['base_accuracy']} | {report['composed_accuracy']} | {report['delta_accuracy']} |",
        f"| macro-F1 | {report['base_macro_f1']} | {report['composed_macro_f1']} | {report['delta_macro_f1']} |",
        "",
        "## 5. Label distributions",
        "",
        "```json",
        json.dumps(
            {
                "gold": report["gold_label_counts"],
                "base_predictions": report["base_prediction_counts"],
                "composed_predictions": report["composed_prediction_counts"],
            },
            indent=2,
        ),
        "```",
        "",
        "## 6. Confusion matrices",
        "",
        "```json",
        json.dumps(
            {
                "base": report["base_confusion_matrix"],
                "composed": report["composed_confusion_matrix"],
            },
            indent=2,
        ),
        "```",
        "",
        "## 7. Safety counters",
        "",
        "```json",
        json.dumps(
            {
                "changed_row_count": report["changed_row_count"],
                "changed_to_SUPPORT_count": report["changed_to_SUPPORT_count"],
                "changed_to_REFUTE_count": report["changed_to_REFUTE_count"],
                "changed_to_NOT_ENTITLED_count": report["changed_to_NOT_ENTITLED_count"],
                "introduced_unsafe_SUPPORT_count": report["introduced_unsafe_SUPPORT_count"],
                "introduced_REFUTE_to_SUPPORT_count": report["introduced_REFUTE_to_SUPPORT_count"],
                "introduced_SUPPORT_to_REFUTE_count": report["introduced_SUPPORT_to_REFUTE_count"],
                "total_composed_wrong_SUPPORT_count": report["total_composed_wrong_SUPPORT_count"],
                "total_composed_SUPPORT_to_REFUTE_count": report["total_composed_SUPPORT_to_REFUTE_count"],
                "blocker_fired_count": report["blocker_fired_count"],
                "recovery_fired_count": report["recovery_fired_count"],
            },
            indent=2,
        ),
        "```",
        "",
        "## 8. Changed-row analysis",
        "",
    ]
    if report["sample_changed_rows"]:
        lines.extend(["```json", json.dumps(report["sample_changed_rows"], indent=2), "```"])
    else:
        lines.append("None.")
    lines.extend(["", "## 9. Error analysis", ""])
    if report["sample_error_rows"]:
        lines.extend(["```json", json.dumps(report["sample_error_rows"], indent=2), "```"])
    else:
        lines.append("None.")
    lines.extend(["", "## 10. Risks", ""])
    lines.extend(f"- {risk}" for risk in report["risks"])
    lines.extend(
        [
            "",
            "## 11. Recommendation",
            "",
            report["recommendation"],
            "",
            "## 12. Leakage policy",
            "",
            report["leakage_policy"],
            "",
        ]
    )
    return "\n".join(lines)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(to_jsonable(row), ensure_ascii=False))
            fh.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_aggregate_report(run_prefix: str, reports: list[dict[str, Any]]) -> dict[str, Any]:
    decisions = {Path(r["input_jsonl"]).stem: r["decision"] for r in reports}
    total_rows = sum(int(r.get("row_count") or 0) for r in reports)
    total_unsafe_support = sum(int(r.get("introduced_unsafe_SUPPORT_count") or 0) for r in reports)
    total_refute_to_support = sum(int(r.get("introduced_REFUTE_to_SUPPORT_count") or 0) for r in reports)
    total_support_to_refute = sum(int(r.get("introduced_SUPPORT_to_REFUTE_count") or 0) for r in reports)
    any_incomplete = any(r["decision"].endswith("_INCOMPLETE") for r in reports)
    any_unsafe = any(r["decision"].endswith("_UNSAFE") for r in reports)
    all_pass = bool(reports) and all(r["decision"] == "STAGE43C0_EXTERNAL_FACTVER_PASS" for r in reports)
    all_safe = bool(reports) and not any_incomplete and not any_unsafe
    any_improved = any((r.get("delta_macro_f1") or 0) > 0 for r in reports)
    if any_incomplete:
        decision = "STAGE43C0_EXTERNAL_FACTVER_AGGREGATE_INCOMPLETE"
        recommendation = "At least one external dataset evaluation is incomplete; do not make external safety claims yet."
    elif any_unsafe:
        decision = "STAGE43C0_EXTERNAL_FACTVER_AGGREGATE_UNSAFE"
        recommendation = "At least one external dataset has introduced safety transitions; keep external composer claims rejected."
    elif all_pass:
        decision = "STAGE43C0_EXTERNAL_FACTVER_AGGREGATE_PASS"
        recommendation = "All evaluated external datasets passed; keep results evaluation-only and do not use them for selection."
    else:
        decision = "STAGE43C0_EXTERNAL_FACTVER_AGGREGATE_SAFE_MIXED"
        recommendation = "All evaluated datasets are safe, but at least one had no gain; report conservatively."
    return {
        "decision": decision,
        "run_prefix": run_prefix,
        "input_reports": [r.get("output_report_json") for r in reports],
        "per_dataset_decisions": decisions,
        "total_row_count": total_rows,
        "total_introduced_unsafe_SUPPORT": total_unsafe_support,
        "total_introduced_REFUTE_to_SUPPORT": total_refute_to_support,
        "total_introduced_SUPPORT_to_REFUTE": total_support_to_refute,
        "all_evaluated_datasets_safe": all_safe,
        "any_evaluated_dataset_improved": any_improved,
        "recommendation": recommendation,
        "leakage_policy": LEAKAGE_POLICY,
    }


def render_aggregate_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Stage43-C0 External Fact-Verification Aggregate Report",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Total rows: {report['total_row_count']}",
        f"- Total introduced unsafe SUPPORT: {report['total_introduced_unsafe_SUPPORT']}",
        f"- Total introduced REFUTE-to-SUPPORT: {report['total_introduced_REFUTE_to_SUPPORT']}",
        f"- Total introduced SUPPORT-to-REFUTE: {report['total_introduced_SUPPORT_to_REFUTE']}",
        f"- All evaluated datasets safe: {report['all_evaluated_datasets_safe']}",
        f"- Any evaluated dataset improved: {report['any_evaluated_dataset_improved']}",
        "",
        "## Per-Dataset Decisions",
        "",
    ]
    for name, decision in report["per_dataset_decisions"].items():
        lines.append(f"- `{name}`: `{decision}`")
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            report["recommendation"],
            "",
            "## Leakage Policy",
            "",
            report["leakage_policy"],
            "",
        ]
    )
    return "\n".join(lines)
