"""Stage43-B2 external fact-verification final-composer evaluation scaffold.

Evaluation only. This script reads Stage43-B1 external fact-verification
JSONL rows and reports base-vs-composed metrics/safety counters. It does not
train, tune thresholds, select checkpoints, alter model behavior, or modify
Stage39-C composer logic.

Prediction policy:
- ``--prediction-source existing_model`` expects rows to already carry an
  existing ContraMamba prediction/export schema, or a future import-safe
  checkpoint inference adapter to be added. If predictions cannot be obtained,
  the script writes an INCOMPLETE report.
- ``--prediction-source heuristic_only`` requires ``--allow-heuristic-only``.
  It is diagnostic-only and always reports INCOMPLETE, never PASS.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LABELS = ["SUPPORT", "REFUTE", "NOT_ENTITLED"]
VALID_LABELS = set(LABELS)
SAMPLE_LIMIT = 20

BASE_PREDICTION_KEYS = (
    "base_prediction",
    "stage39_original_final_label",
    "stage39_original_pred_final_label",
    "pred_final_label",
    "pred_label",
    "prediction",
    "predicted_label",
)

COMPOSED_PREDICTION_KEYS = (
    "composed_prediction",
    "stage39_composed_final_label",
)

LEAKAGE_POLICY = (
    "Stage43-B2 is external-evaluation-only. Stage43-B1 data and Stage43-B2 "
    "outcomes must not be used for training, calibration, threshold tuning, "
    "checkpoint selection, loss design, or model/composer behavior changes."
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


def read_jsonl(path: Path, max_rows: int | None = None) -> list[dict[str, Any]]:
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
                raise ValueError(f"Line {line_no} is not a JSON object.")
            rows.append(obj)
    return rows


def validate_stage43_rows(rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    required = ("id", "claim", "evidence", "label", "source_dataset", "source_label", "stage43_split", "metadata")
    for index, row in enumerate(rows):
        for field in required:
            if field not in row:
                errors.append(f"row {index}: missing required field {field}")
        label = normalize_label(row.get("label"))
        if label not in VALID_LABELS:
            errors.append(f"row {index}: invalid label {row.get('label')!r}")
        if not str(row.get("claim") or "").strip():
            errors.append(f"row {index}: empty claim")
        if not str(row.get("evidence") or "").strip():
            errors.append(f"row {index}: empty evidence")
        if errors and len(errors) >= SAMPLE_LIMIT:
            break
    return errors


def first_label(row: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        if key in row:
            label = normalize_label(row.get(key))
            if label is not None:
                return label
    return None


def token_set(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-z0-9]+", text.lower()) if len(tok) > 2}


def heuristic_prediction(row: dict[str, Any]) -> tuple[str, str]:
    """Transparent diagnostic-only fallback, never a model evaluation."""
    claim = str(row.get("claim") or "")
    evidence = str(row.get("evidence") or "")
    claim_tokens = token_set(claim)
    evidence_tokens = token_set(evidence)
    if not claim_tokens or not evidence_tokens:
        return "NOT_ENTITLED", "heuristic_empty_or_low_token_signal"

    overlap = len(claim_tokens & evidence_tokens) / max(1, len(claim_tokens))
    claim_has_negation = any(tok in claim.lower() for tok in (" not ", " never ", " no "))
    evidence_has_negation = any(tok in evidence.lower() for tok in (" not ", " never ", " no "))
    if overlap >= 0.45 and claim_has_negation != evidence_has_negation:
        return "REFUTE", "heuristic_overlap_with_negation_mismatch"
    if overlap >= 0.35:
        return "SUPPORT", "heuristic_token_overlap_support"
    return "NOT_ENTITLED", "heuristic_low_overlap_not_entitled"


def try_import_stage39_composer():
    try:
        from scripts.train_controlled_v6b_minimal import compute_stage39_final_composer

        return compute_stage39_final_composer, None
    except Exception as exc:
        return None, str(exc)


def build_stage39_args(composer_mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        stage39_use_final_composer_opt_in=(composer_mode == "safe_structured_v2"),
        stage39_final_composer_export=True,
        stage39_final_composer_policy="safe_structured_v2",
        stage39_final_composer_output_mode="export_only",
        stage39_final_composer_source="stage37_final_shadow_label",
        stage39_disallow_refute_to_support=True,
        stage39_require_stage36_safety_clear=True,
        stage39_require_stage37_not_from_refute=True,
    )


def build_stage39_row(row: dict[str, Any], base_prediction: str) -> dict[str, Any]:
    return {
        "pred_final_label": base_prediction,
        "stage37_final_shadow_label": row.get("stage37_final_shadow_label"),
        "stage36_final_shadow_label": row.get("stage36_final_shadow_label"),
        "stage32_shadow_label": row.get("stage32_shadow_label"),
        "stage36_support_blocker_fired": row.get("stage36_support_blocker_fired", False),
        "stage37_recovered_from_label": row.get("stage37_recovered_from_label"),
        "stage33_structured_coverage_reason": row.get("stage33_structured_coverage_reason"),
        "stage33_structured_coverage_route": row.get("stage33_structured_coverage_route"),
        "stage33_structured_coverage_label": row.get("stage33_structured_coverage_label"),
        "stage33_conditional_override_type": row.get("stage33_conditional_override_type"),
        "stage36_conditional_override_type": row.get("stage36_conditional_override_type"),
        "stage37_conditional_override_type": row.get("stage37_conditional_override_type"),
        "gold_final_label": row.get("label"),
    }


def get_base_prediction(row: dict[str, Any], args: argparse.Namespace) -> tuple[str | None, str | None]:
    label = first_label(row, BASE_PREDICTION_KEYS)
    if label is not None:
        return label, "row_prediction_field"
    if args.prediction_source == "heuristic_only" and args.allow_heuristic_only:
        return heuristic_prediction(row)
    return None, None


def get_composed_prediction(
    row: dict[str, Any],
    base_prediction: str,
    args: argparse.Namespace,
    composer_func: Any,
) -> tuple[str, dict[str, Any]]:
    if args.composer_mode == "off":
        return base_prediction, {
            "composer_reason": "composer_mode_off",
            "blocker_fired": False,
            "recovery_fired": False,
        }

    existing = first_label(row, COMPOSED_PREDICTION_KEYS)
    if existing is not None:
        return existing, {
            "composer_reason": str(row.get("stage39_composer_reason") or "existing_composed_prediction_field"),
            "blocker_fired": bool(row.get("stage36_support_blocker_fired") or row.get("stage39_blocked_by_stage36")),
            "recovery_fired": bool(row.get("stage37_safe_recovery_fired")),
        }

    if composer_func is None:
        return base_prediction, {
            "composer_reason": "stage39_composer_import_unavailable",
            "blocker_fired": bool(row.get("stage36_support_blocker_fired")),
            "recovery_fired": bool(row.get("stage37_safe_recovery_fired")),
        }

    result = composer_func(build_stage39_row(row, base_prediction), build_stage39_args(args.composer_mode))
    label = normalize_label(result.get("stage39_composed_final_label")) or base_prediction
    return label, {
        "composer_reason": result.get("stage39_composer_reason"),
        "composer_action": result.get("stage39_composer_action"),
        "blocker_fired": bool(result.get("stage39_blocked_by_stage36") or row.get("stage36_support_blocker_fired")),
        "recovery_fired": bool(row.get("stage37_safe_recovery_fired")),
        "stage39": result,
    }


def empty_confusion() -> dict[str, dict[str, int]]:
    return {gold: {pred: 0 for pred in LABELS} for gold in LABELS}


def compute_metrics(golds: list[str], preds: list[str]) -> dict[str, Any]:
    confusion = empty_confusion()
    for gold, pred in zip(golds, preds):
        confusion[gold][pred] += 1

    total = len(golds)
    correct = sum(gold == pred for gold, pred in zip(golds, preds))
    per_label: dict[str, dict[str, float | int]] = {}
    f1s: list[float] = []
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[gold][label] for gold in LABELS if gold != label)
        fn = sum(confusion[label][pred] for pred in LABELS if pred != label)
        support = sum(confusion[label].values())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
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


def make_prediction_record(
    row: dict[str, Any],
    gold: str,
    base: str,
    composed: str,
    composer_info: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "id": str(row.get("id")),
        "claim": row.get("claim"),
        "evidence": row.get("evidence"),
        "gold_label": gold,
        "base_prediction": base,
        "composed_prediction": composed,
        "changed": base != composed,
        "composer_mode": args.composer_mode,
        "composer_reason": composer_info.get("composer_reason"),
        "blocker_fired": bool(composer_info.get("blocker_fired")),
        "recovery_fired": bool(composer_info.get("recovery_fired")),
        "source_dataset": row.get("source_dataset"),
        "metadata": row.get("metadata") or {},
    }


def incomplete_report(
    args: argparse.Namespace,
    row_count: int,
    gold_label_counts: dict[str, int],
    reason: str,
    sample_errors: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "decision": "STAGE43B2_EXTERNAL_FACTVER_INCOMPLETE",
        "run_name": args.run_name,
        "input_jsonl": str(args.input_jsonl),
        "row_count": row_count,
        "gold_label_counts": gold_label_counts,
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
        "composer_mode": args.composer_mode,
        "prediction_source": args.prediction_source,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "sample_changed_rows": [],
        "sample_error_rows": sample_errors[:SAMPLE_LIMIT],
        "risks": [
            reason,
            "No PASS decision is possible until real model predictions are available.",
        ],
        "recommendation": reason,
        "leakage_policy": LEAKAGE_POLICY,
    }


def decide(
    report: dict[str, Any],
    heuristic_only: bool,
    partial_prediction_coverage: bool,
    composer_not_applied_all: bool,
) -> tuple[str, str]:
    if heuristic_only:
        return (
            "STAGE43B2_EXTERNAL_FACTVER_INCOMPLETE",
            "Heuristic-only output is diagnostic-only and is not a model evaluation.",
        )
    if partial_prediction_coverage:
        return (
            "STAGE43B2_EXTERNAL_FACTVER_INCOMPLETE",
            "Model predictions were not available for every input row.",
        )
    if composer_not_applied_all:
        return (
            "STAGE43B2_EXTERNAL_FACTVER_INCOMPLETE",
            "safe_structured_v2 could not be applied because composer/source owner diagnostics were unavailable for every evaluated row.",
        )
    if report["row_count"] < 100 or len(report["gold_label_counts"]) < 2:
        return (
            "STAGE43B2_EXTERNAL_FACTVER_INCOMPLETE",
            "Input is too small or lacks at least two represented labels.",
        )
    unsafe = (
        report["introduced_unsafe_SUPPORT_count"] > 0
        or report["introduced_REFUTE_to_SUPPORT_count"] > 0
        or report["introduced_SUPPORT_to_REFUTE_count"] > 0
    )
    if unsafe:
        return (
            "STAGE43B2_EXTERNAL_FACTVER_UNSAFE",
            "Composition introduced unsafe SUPPORT or SUPPORT/REFUTE transition(s).",
        )
    all_three = all(label in report["gold_label_counts"] for label in LABELS)
    if (
        all_three
        and report["composed_macro_f1"] is not None
        and report["base_macro_f1"] is not None
        and report["composed_macro_f1"] >= report["base_macro_f1"]
    ):
        return (
            "STAGE43B2_EXTERNAL_FACTVER_PASS",
            "All three labels are represented, composed macro-F1 is at least base macro-F1, and no introduced safety transitions were observed.",
        )
    return (
        "STAGE43B2_EXTERNAL_FACTVER_SAFE_BUT_NO_GAIN",
        "No introduced safety transitions were observed, but composed macro-F1 did not improve over base or not all labels were represented.",
    )


def evaluate(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = read_jsonl(args.input_jsonl, args.max_rows)
    validation_errors = validate_stage43_rows(rows)
    golds = [normalize_label(row.get("label")) for row in rows]
    gold_label_counts = dict(Counter(label for label in golds if label))

    if validation_errors:
        sample_errors = [{"error": error} for error in validation_errors]
        return incomplete_report(
            args,
            len(rows),
            gold_label_counts,
            "Input schema is invalid; fix Stage43 JSONL before evaluation.",
            sample_errors,
        ), []

    if args.prediction_source == "heuristic_only" and not args.allow_heuristic_only:
        return incomplete_report(
            args,
            len(rows),
            gold_label_counts,
            "--prediction-source heuristic_only requires --allow-heuristic-only.",
            [{"error": "heuristic_only_without_explicit_allow"}],
        ), []

    if args.prediction_source == "existing_model" and args.checkpoint:
        # Reserved for a future import-safe checkpoint inference adapter. This
        # scaffold intentionally does not call the training script as a hidden
        # evaluator or silently switch to heuristics.
        checkpoint_note = (
            "Checkpoint argument was supplied, but this scaffold does not yet "
            "have an import-safe Stage43 external inference adapter. Provide "
            "rows with existing prediction/export fields or add a dedicated "
            "read-only inference helper before running model evaluation."
        )
    else:
        checkpoint_note = ""

    composer_func, composer_import_error = try_import_stage39_composer()
    prediction_rows: list[dict[str, Any]] = []
    base_preds: list[str] = []
    composed_preds: list[str] = []
    clean_golds: list[str] = []
    sample_errors: list[dict[str, Any]] = []
    sample_changed: list[dict[str, Any]] = []
    counters = Counter()
    composer_not_applied_count = 0

    for index, row in enumerate(rows):
        gold = normalize_label(row.get("label"))
        base, base_reason = get_base_prediction(row, args)
        if gold is None or base is None:
            if len(sample_errors) < SAMPLE_LIMIT:
                sample_errors.append(
                    {
                        "id": row.get("id"),
                        "row_index": index,
                        "error": "missing_model_prediction",
                        "detail": checkpoint_note or "No recognized prediction field was present.",
                    }
                )
            continue

        composed, composer_info = get_composed_prediction(row, base, args, composer_func)
        reason_text = str(composer_info.get("composer_reason") or "")
        if args.composer_mode == "safe_structured_v2" and reason_text in {
            "stage39_composer_import_unavailable",
            "missing_source_shadow_label",
        }:
            composer_not_applied_count += 1
        record = make_prediction_record(row, gold, base, composed, composer_info, args)
        record["prediction_source_reason"] = base_reason
        prediction_rows.append(record)
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
                        "id": row.get("id"),
                        "gold_label": gold,
                        "base_prediction": base,
                        "composed_prediction": composed,
                        "composer_reason": composer_info.get("composer_reason"),
                    }
                )
        if changed and composed == "SUPPORT" and gold != "SUPPORT" and base != "SUPPORT":
            counters["introduced_unsafe_SUPPORT_count"] += 1
        if changed and base == "REFUTE" and composed == "SUPPORT":
            counters["introduced_REFUTE_to_SUPPORT_count"] += 1
        if changed and base == "SUPPORT" and composed == "REFUTE":
            counters["introduced_SUPPORT_to_REFUTE_count"] += 1
        if composed == "SUPPORT" and gold != "SUPPORT":
            counters["total_composed_wrong_SUPPORT_count"] += 1
        if gold == "SUPPORT" and composed == "REFUTE":
            counters["total_composed_SUPPORT_to_REFUTE_count"] += 1
        if composer_info.get("blocker_fired"):
            counters["blocker_fired_count"] += 1
        if composer_info.get("recovery_fired"):
            counters["recovery_fired_count"] += 1

    if not prediction_rows:
        reason = checkpoint_note or (
            "Model predictions could not be obtained. Supply rows with existing "
            "prediction/export fields, or use --prediction-source heuristic_only "
            "--allow-heuristic-only for diagnostic-only output."
        )
        if composer_import_error:
            reason += f" Stage39 composer import note: {composer_import_error}"
        return incomplete_report(args, len(rows), gold_label_counts, reason, sample_errors), []

    partial_prediction_coverage = len(prediction_rows) < len(rows)
    if partial_prediction_coverage and args.prediction_source == "existing_model":
        sample_errors.append(
            {
                "error": "partial_prediction_coverage",
                "detail": f"Only {len(prediction_rows)} of {len(rows)} rows had usable predictions.",
            }
        )

    base_metrics = compute_metrics(clean_golds, base_preds)
    composed_metrics = compute_metrics(clean_golds, composed_preds)

    report: dict[str, Any] = {
        "decision": "STAGE43B2_EXTERNAL_FACTVER_INCOMPLETE",
        "run_name": args.run_name,
        "input_jsonl": str(args.input_jsonl),
        "row_count": len(prediction_rows),
        "gold_label_counts": dict(Counter(clean_golds)),
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
        "composer_mode": args.composer_mode,
        "prediction_source": args.prediction_source,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "sample_changed_rows": sample_changed,
        "sample_error_rows": sample_errors[:SAMPLE_LIMIT],
        "composer_not_applied_count": composer_not_applied_count,
        "risks": [
            "External fact-verification data is evaluation-only and must not guide training or checkpoint selection.",
            "Climate-FEVER failures should be interpreted as cross-domain limitations, not training signals.",
        ],
        "recommendation": "",
        "leakage_policy": LEAKAGE_POLICY,
    }
    if composer_import_error:
        report["risks"].append(f"Stage39 composer import unavailable; existing composed fields were used when present. Error: {composer_import_error}")
    if args.prediction_source == "heuristic_only":
        report["risks"].append("Heuristic-only predictions are diagnostic-only and not model evaluation.")

    decision, recommendation = decide(
        report,
        args.prediction_source == "heuristic_only",
        partial_prediction_coverage and args.prediction_source == "existing_model",
        args.composer_mode == "safe_structured_v2"
        and composer_not_applied_count == len(prediction_rows),
    )
    report["decision"] = decision
    report["recommendation"] = recommendation
    return report, prediction_rows


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Stage43-B2 External Fact-Verification Evaluation",
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
        f"- Evaluated row count: {report['row_count']}",
        f"- Composer mode: `{report['composer_mode']}`",
        f"- Prediction source: `{report['prediction_source']}`",
        f"- Checkpoint: `{report.get('checkpoint')}`",
        "",
        "## 3. Base vs composed metrics",
        "",
        "| Metric | Base | Composed | Delta |",
        "|---|---:|---:|---:|",
        f"| accuracy | {report['base_accuracy']} | {report['composed_accuracy']} | {report['delta_accuracy']} |",
        f"| macro-F1 | {report['base_macro_f1']} | {report['composed_macro_f1']} | {report['delta_macro_f1']} |",
        "",
        "## 4. Label distributions",
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
        "## 5. Confusion matrices",
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
        "## 6. Safety counters",
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
        "## 7. Changed-row analysis",
        "",
    ]
    if report["sample_changed_rows"]:
        lines.extend(["```json", json.dumps(report["sample_changed_rows"], indent=2), "```"])
    else:
        lines.append("None.")
    lines.extend(["", "## 8. Error analysis", ""])
    if report["sample_error_rows"]:
        lines.extend(["```json", json.dumps(report["sample_error_rows"], indent=2), "```"])
    else:
        lines.append("None.")
    lines.extend(["", "## 9. Risks", ""])
    lines.extend(f"- {risk}" for risk in report["risks"])
    lines.extend(
        [
            "",
            "## 10. Recommendation",
            "",
            report["recommendation"],
            "",
            "## 11. Leakage policy",
            "",
            report["leakage_policy"],
            "",
        ]
    )
    return "\n".join(lines)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(to_jsonable(row), ensure_ascii=False))
            fh.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--composer-mode",
        choices=["off", "safe_structured_v2"],
        default="safe_structured_v2",
    )
    parser.add_argument(
        "--prediction-source",
        choices=["existing_model", "heuristic_only"],
        default="existing_model",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--write-predictions-jsonl", type=Path, default=None)
    parser.add_argument("--allow-heuristic-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report, prediction_rows = evaluate(args)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as fh:
        json.dump(to_jsonable(report), fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(report), encoding="utf-8")

    if args.write_predictions_jsonl is not None and prediction_rows:
        write_jsonl(args.write_predictions_jsonl, prediction_rows)

    print(f"Decision: {report['decision']}")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    if args.write_predictions_jsonl is not None and prediction_rows:
        print(f"Wrote {args.write_predictions_jsonl}")
    return 0 if report["decision"] != "STAGE43B2_EXTERNAL_FACTVER_INCOMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
