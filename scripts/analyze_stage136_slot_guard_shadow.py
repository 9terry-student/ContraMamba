"""Stage136-A: export-only slot mismatch shadow guard analysis.

This script reads existing prediction JSONL exports containing a
slot_mismatch_prob field and evaluates thresholded shadow policies. It is
analysis-only: it does not import model or training code, does not run a model,
does not write modified prediction files, and does not change final logits,
checkpoint selection, or any existing prediction export.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any


LABELS = ["REFUTE", "NOT_ENTITLED", "SUPPORT"]
PRED_FIELD_CANDIDATES = [
    "prediction",
    "pred_label",
    "final_prediction",
    "final_pred",
    "pred",
    "label_pred",
]
GOLD_FIELD_CANDIDATES = [
    "gold_label",
    "label",
    "target_label",
    "true_label",
    "final_label",
]
GROUP_FIELDS_DEFAULT = [
    "source_intervention_type",
    "intervention_type",
    "stage133_case_type",
    "slot_type",
    "family",
]
SLOT_TARGET_FIELD = "slot_mismatch_target"
SLOT_TARGET_VALID_FIELD = "slot_mismatch_target_valid"

DECISION_NO_SUPPORT = "STAGE136A_SHADOW_GUARD_UNINFORMATIVE_NO_SUPPORT_PREDICTIONS"
DECISION_NO_GOLD = "STAGE136A_SHADOW_GUARD_COUNT_ONLY_NO_GOLD"
DECISION_SAFE = "STAGE136A_SHADOW_GUARD_CANDIDATE_SAFE"
DECISION_TRADEOFF = "STAGE136A_SHADOW_GUARD_TRADEOFF"
DECISION_NOT_USEFUL = "STAGE136A_SHADOW_GUARD_NOT_USEFUL"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate slot_mismatch_prob as an export-only shadow guard."
    )
    parser.add_argument(
        "--input-jsonl",
        action="append",
        default=[],
        help="Prediction JSONL file. May be passed more than once.",
    )
    parser.add_argument(
        "--input-glob",
        default=None,
        help='Glob pattern for prediction JSONL files, e.g. "reports/**/predictions*.jsonl".',
    )
    parser.add_argument("--output-dir", required=True, help="Directory for Stage136-A outputs.")
    parser.add_argument(
        "--thresholds",
        default="0.50,0.60,0.70,0.80,0.90",
        help="Comma-separated slot_mismatch_prob thresholds.",
    )
    parser.add_argument(
        "--policy",
        default="support_to_ne",
        choices=["support_to_ne"],
        help="Shadow policy to evaluate.",
    )
    parser.add_argument(
        "--slot-prob-field",
        default="slot_mismatch_prob",
        help="Field containing slot mismatch probability.",
    )
    parser.add_argument(
        "--pred-field",
        default="auto",
        help="Prediction field name, or auto.",
    )
    parser.add_argument(
        "--gold-field",
        default="auto",
        help="Gold label field name, or auto.",
    )
    parser.add_argument(
        "--group-fields",
        default=",".join(GROUP_FIELDS_DEFAULT),
        help="Comma-separated fields for group breakdowns.",
    )
    return parser.parse_args()


def parse_thresholds(raw: str) -> list[float]:
    values: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError as exc:
            raise ValueError(f"Invalid threshold value {part!r}") from exc
    if not values:
        raise ValueError("At least one threshold is required.")
    return values


def discover_input_files(input_jsonl: list[str], input_glob: str | None) -> list[Path]:
    if not input_jsonl and not input_glob:
        raise ValueError("Provide at least one --input-jsonl or an --input-glob pattern.")

    paths: list[Path] = []
    for item in input_jsonl:
        paths.append(Path(item))
    if input_glob:
        paths.extend(Path(p) for p in glob.glob(input_glob, recursive=True))

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve())
        if key not in seen:
            deduped.append(path)
            seen.add(key)

    if not deduped:
        raise ValueError("No input files matched.")
    missing = [str(path) for path in deduped if not path.exists()]
    if missing:
        raise ValueError(f"Input file(s) not found: {missing}")
    return deduped


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: invalid JSON on line {idx + 1}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}: line {idx + 1} is not a JSON object.")
            row["_stage136_source_file"] = str(path)
            row["_stage136_row_index"] = idx
            rows.append(row)
    return rows


def infer_field(
    rows: list[dict[str, Any]],
    requested: str,
    candidates: list[str],
    role: str,
    *,
    required: bool,
) -> str | None:
    if requested != "auto":
        if required and not any(requested in row for row in rows):
            raise ValueError(f"Requested {role} field {requested!r} was not found.")
        return requested
    for candidate in candidates:
        if any(candidate in row for row in rows):
            return candidate
    if required:
        raise ValueError(f"Could not infer {role} field. Tried: {candidates}")
    return None


def normalize_label(raw: Any) -> str | None:
    if raw is None:
        return None
    key = str(raw).strip().upper().replace("-", "_")
    key = "_".join(key.split())
    mapping = {
        "REFUTE": "REFUTE",
        "REFUTES": "REFUTE",
        "CONTRADICT": "REFUTE",
        "CONTRADICTION": "REFUTE",
        "0": "REFUTE",
        "NOT_ENTITLED": "NOT_ENTITLED",
        "NOT_ENOUGH_INFO": "NOT_ENTITLED",
        "NOTENOUGHINFO": "NOT_ENTITLED",
        "NEI": "NOT_ENTITLED",
        "NE": "NOT_ENTITLED",
        "NONE": "NOT_ENTITLED",
        "1": "NOT_ENTITLED",
        "SUPPORT": "SUPPORT",
        "SUPPORTS": "SUPPORT",
        "ENTAILMENT": "SUPPORT",
        "ENTAILS": "SUPPORT",
        "2": "SUPPORT",
    }
    return mapping.get(key)


def safe_float(raw: Any) -> float | None:
    if raw is None or raw == "":
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def truthy_valid(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return False
    value = str(raw).strip().lower()
    return value in {"1", "true", "yes", "y", "valid"}


def safe_div(num: float, den: float) -> float | None:
    return num / den if den else None


def count_predictions(labels: list[str | None]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for label in labels:
        if label in LABELS:
            counter[label] += 1
    return counter


def shadow_label(prediction: str | None, prob: float | None, threshold: float) -> str | None:
    if prediction == "SUPPORT" and prob is not None and prob >= threshold:
        return "NOT_ENTITLED"
    return prediction


def class_precision_recall(
    golds: list[str],
    preds: list[str],
    label: str,
) -> tuple[float | None, float | None]:
    tp = sum(1 for gold, pred in zip(golds, preds) if gold == label and pred == label)
    fp = sum(1 for gold, pred in zip(golds, preds) if gold != label and pred == label)
    fn = sum(1 for gold, pred in zip(golds, preds) if gold == label and pred != label)
    return safe_div(tp, tp + fp), safe_div(tp, tp + fn)


def macro_f1(golds: list[str], preds: list[str]) -> float | None:
    if not golds:
        return None
    f1s: list[float] = []
    for label in LABELS:
        precision, recall = class_precision_recall(golds, preds, label)
        if precision is None and recall is None:
            f1s.append(0.0)
        elif not precision or not recall:
            f1s.append(0.0)
        else:
            f1s.append(2.0 * precision * recall / (precision + recall))
    return sum(f1s) / len(f1s)


def supervised_metrics(golds: list[str], preds: list[str]) -> dict[str, float | int | None]:
    if not golds:
        result: dict[str, float | int | None] = {
            "accuracy": None,
            "macro_f1": None,
            "false_support": None,
            "false_ne": None,
        }
        for label in LABELS:
            key = label.lower()
            result[f"{key}_precision"] = None
            result[f"{key}_recall"] = None
        return result

    result = {
        "accuracy": safe_div(sum(g == p for g, p in zip(golds, preds)), len(golds)),
        "macro_f1": macro_f1(golds, preds),
        "false_support": sum(1 for gold, pred in zip(golds, preds) if pred == "SUPPORT" and gold != "SUPPORT"),
        "false_ne": sum(1 for gold, pred in zip(golds, preds) if pred == "NOT_ENTITLED" and gold != "NOT_ENTITLED"),
    }
    for label in LABELS:
        precision, recall = class_precision_recall(golds, preds, label)
        key = label.lower()
        result[f"{key}_precision"] = precision
        result[f"{key}_recall"] = recall
    return result


def prefixed_metrics(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    metric_names = [
        "accuracy",
        "macro_f1",
        "support_precision",
        "support_recall",
        "not_entitled_precision",
        "not_entitled_recall",
        "refute_precision",
        "refute_recall",
        "false_support",
        "false_ne",
    ]
    for name in metric_names:
        out[f"{name}_before"] = before.get(name)
        out[f"{name}_after"] = after.get(name)
    out["delta_accuracy"] = delta(after.get("accuracy"), before.get("accuracy"))
    out["delta_macro_f1"] = delta(after.get("macro_f1"), before.get("macro_f1"))
    out["delta_false_support"] = delta(after.get("false_support"), before.get("false_support"))
    out["delta_false_ne"] = delta(after.get("false_ne"), before.get("false_ne"))
    out["delta_support_recall"] = delta(after.get("support_recall"), before.get("support_recall"))
    return out


def delta(after: Any, before: Any) -> Any:
    if after is None or before is None:
        return None
    return after - before


def compute_slot_target_metrics(rows: list[dict[str, Any]], slot_prob_field: str) -> dict[str, Any]:
    y_true: list[int] = []
    y_score: list[float] = []
    for row in rows:
        if not truthy_valid(row.get(SLOT_TARGET_VALID_FIELD)):
            continue
        prob = safe_float(row.get(slot_prob_field))
        if prob is None:
            continue
        target = row.get(SLOT_TARGET_FIELD)
        if str(target).strip() not in {"0", "1", "0.0", "1.0", "False", "True", "false", "true"}:
            continue
        target_int = 1 if str(target).strip().lower() in {"1", "1.0", "true"} else 0
        y_true.append(target_int)
        y_score.append(prob)

    pos = sum(y_true)
    neg = len(y_true) - pos
    metrics = {
        "slot_target_auc": None,
        "slot_target_auprc": None,
        "slot_target_positive_count": pos,
        "slot_target_negative_count": neg,
    }
    if pos == 0 or neg == 0:
        return metrics

    try:
        from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore
    except Exception:
        return metrics

    metrics["slot_target_auc"] = float(roc_auc_score(y_true, y_score))
    metrics["slot_target_auprc"] = float(average_precision_score(y_true, y_score))
    return metrics


def build_threshold_row(
    rows: list[dict[str, Any]],
    threshold: float,
    pred_field: str,
    gold_field: str | None,
    slot_prob_field: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions_before: list[str | None] = []
    predictions_after: list[str | None] = []
    golds: list[str] = []
    supervised_before: list[str] = []
    supervised_after: list[str] = []
    changed_rows: list[dict[str, Any]] = []
    n_with_slot_prob = 0

    for row in rows:
        pred = normalize_label(row.get(pred_field))
        prob = safe_float(row.get(slot_prob_field))
        after = shadow_label(pred, prob, threshold)
        predictions_before.append(pred)
        predictions_after.append(after)
        if prob is not None:
            n_with_slot_prob += 1
        if pred != after:
            changed_rows.append(row)
        if gold_field:
            gold = normalize_label(row.get(gold_field))
            if gold in LABELS and pred in LABELS and after in LABELS:
                golds.append(gold)
                supervised_before.append(pred)
                supervised_after.append(after)

    before_counts = count_predictions(predictions_before)
    after_counts = count_predictions(predictions_after)
    before_supervised = supervised_metrics(golds, supervised_before)
    after_supervised = supervised_metrics(golds, supervised_after)

    row_out: dict[str, Any] = {
        "threshold": threshold,
        "n_total": len(rows),
        "n_with_slot_prob": n_with_slot_prob,
        "n_missing_slot_prob": len(rows) - n_with_slot_prob,
        "n_changed_total": len(changed_rows),
        "n_support_to_ne": len(changed_rows),
        "support_pred_before": before_counts["SUPPORT"],
        "support_pred_after": after_counts["SUPPORT"],
        "not_entitled_pred_before": before_counts["NOT_ENTITLED"],
        "not_entitled_pred_after": after_counts["NOT_ENTITLED"],
        "refute_pred_before": before_counts["REFUTE"],
        "refute_pred_after": after_counts["REFUTE"],
    }
    row_out.update(prefixed_metrics(before_supervised, after_supervised))
    row_out.update(compute_slot_target_metrics(rows, slot_prob_field))
    return row_out, changed_rows


def build_group_row(
    rows: list[dict[str, Any]],
    threshold: float,
    group_field: str,
    group_value: Any,
    pred_field: str,
    gold_field: str | None,
    slot_prob_field: str,
) -> dict[str, Any]:
    threshold_row, _ = build_threshold_row(rows, threshold, pred_field, gold_field, slot_prob_field)
    return {
        "group_field": group_field,
        "group_value": group_value,
        "threshold": threshold,
        "n_total": threshold_row["n_total"],
        "n_changed_total": threshold_row["n_changed_total"],
        "support_pred_before": threshold_row["support_pred_before"],
        "support_pred_after": threshold_row["support_pred_after"],
        "false_support_before": threshold_row["false_support_before"],
        "false_support_after": threshold_row["false_support_after"],
        "false_ne_before": threshold_row["false_ne_before"],
        "false_ne_after": threshold_row["false_ne_after"],
        "macro_f1_before": threshold_row["macro_f1_before"],
        "macro_f1_after": threshold_row["macro_f1_after"],
    }


def choose_decision(rows: list[dict[str, Any]], threshold_rows: list[dict[str, Any]], pred_field: str, gold_field: str | None) -> str:
    support_count = sum(1 for row in rows if normalize_label(row.get(pred_field)) == "SUPPORT")
    if support_count == 0:
        return DECISION_NO_SUPPORT
    if not gold_field or all(row.get("false_support_before") is None for row in threshold_rows):
        return DECISION_NO_GOLD

    any_false_support_decrease = False
    total = len(rows)
    for row in threshold_rows:
        delta_false_support = row.get("delta_false_support")
        delta_macro_f1 = row.get("delta_macro_f1")
        delta_false_ne = row.get("delta_false_ne")
        if delta_false_support is not None and delta_false_support < 0:
            any_false_support_decrease = True
            macro_cost_ok = delta_macro_f1 is not None and delta_macro_f1 >= -0.01
            false_ne_ok = delta_false_ne is not None and delta_false_ne <= 0.05 * total
            if macro_cost_ok and false_ne_ok:
                return DECISION_SAFE
    if any_false_support_decrease:
        return DECISION_TRADEOFF
    return DECISION_NOT_USEFUL


def select_best_threshold(threshold_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        row
        for row in threshold_rows
        if row.get("delta_false_support") is not None and row.get("delta_false_support") < 0
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda row: (
            row.get("delta_false_support", 0),
            -(row.get("delta_macro_f1") if row.get("delta_macro_f1") is not None else -999.0),
            row["threshold"],
        ),
    )[0]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = []
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def example_record(
    row: dict[str, Any],
    threshold: float,
    pred_field: str,
    gold_field: str | None,
    slot_prob_field: str,
    group_fields: list[str],
) -> dict[str, Any]:
    pred = normalize_label(row.get(pred_field))
    out: dict[str, Any] = {
        "source_file": row.get("_stage136_source_file"),
        "row_index": row.get("_stage136_row_index"),
        "prediction_before": pred,
        "shadow_prediction": shadow_label(pred, safe_float(row.get(slot_prob_field)), threshold),
        "slot_mismatch_prob": safe_float(row.get(slot_prob_field)),
        "threshold": threshold,
    }
    for key in ["id", "pair_id"]:
        if key in row:
            out[key] = row[key]
    if gold_field and gold_field in row:
        out["gold_label"] = row[gold_field]
    for key in group_fields:
        if key in row:
            out[key] = row[key]
    for key in ["claim", "evidence"]:
        if key in row:
            out[key] = row[key]
    return out


def markdown_table(rows: list[dict[str, Any]], columns: list[str], max_rows: int | None = None) -> str:
    selected = rows[:max_rows] if max_rows is not None else rows
    if not selected:
        return "_No rows._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in selected:
        values = [format_md_value(row.get(col)) for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def format_md_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")


def write_markdown_report(
    path: Path,
    decision: str,
    input_files: list[Path],
    pred_field: str,
    gold_field: str | None,
    slot_prob_field: str,
    threshold_rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
    changed_examples_count: int,
    best_threshold: dict[str, Any] | None,
    warnings: list[str],
) -> None:
    threshold_columns = [
        "threshold",
        "n_total",
        "n_with_slot_prob",
        "n_changed_total",
        "support_pred_before",
        "support_pred_after",
        "false_support_before",
        "false_support_after",
        "false_ne_before",
        "false_ne_after",
        "macro_f1_before",
        "macro_f1_after",
        "delta_macro_f1",
    ]
    group_columns = [
        "group_field",
        "group_value",
        "threshold",
        "n_total",
        "n_changed_total",
        "false_support_before",
        "false_support_after",
        "false_ne_before",
        "false_ne_after",
        "macro_f1_before",
        "macro_f1_after",
    ]

    best_text = "_No threshold reduced false SUPPORT._"
    if best_threshold:
        best_text = json.dumps(best_threshold, indent=2, sort_keys=True)

    input_lines = "\n".join(f"- {path}" for path in input_files)
    warning_lines = "\n".join(f"- {warning}" for warning in warnings) if warnings else "- None"
    text = f"""# Stage136-A Slot Guard Shadow Report

## 1. Summary decision

Decision: `{decision}`

This is export-only analysis of a shadow policy. It does not modify training, model
forward behavior, final logits, checkpoint selection, or prediction JSONL files.

Detected fields:

- Prediction field: `{pred_field}`
- Gold field: `{gold_field if gold_field else 'none'}`
- Slot probability field: `{slot_prob_field}`

## 2. Input files

{input_lines}

## 3. Threshold table

{markdown_table(threshold_rows, threshold_columns)}

## 4. Best threshold if any

```json
{best_text}
```

## 5. Group breakdown summary

{markdown_table(group_rows, group_columns, max_rows=40)}

## 6. Changed example summary

Changed example records written: `{changed_examples_count}`

## 7. Limitations

- This script only analyzes existing exports passed by the user.
- Missing gold labels produce count-only metrics with supervised metrics set to null.
- Missing slot targets produce null AUROC/AUPRC slot-target diagnostics.
- Thresholds are analysis-only and must not be treated as checkpoint or model selection.

Warnings:

{warning_lines}

## 8. Recommendation

Use this report only to decide whether a future, separately reviewed guard experiment is
worth proposing. Do not route slot mismatch probabilities into final logits based on this
script alone.
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    try:
        args = parse_args()
        thresholds = parse_thresholds(args.thresholds)
        group_fields = [field.strip() for field in args.group_fields.split(",") if field.strip()]
        input_files = discover_input_files(args.input_jsonl, args.input_glob)

        rows: list[dict[str, Any]] = []
        for path in input_files:
            rows.extend(load_jsonl(path))
        if not rows:
            raise ValueError("Input files contained no JSONL rows.")

        pred_field = infer_field(rows, args.pred_field, PRED_FIELD_CANDIDATES, "prediction", required=True)
        assert pred_field is not None
        gold_field = infer_field(rows, args.gold_field, GOLD_FIELD_CANDIDATES, "gold", required=False)

        warnings: list[str] = []
        n_with_slot_prob_any = sum(1 for row in rows if safe_float(row.get(args.slot_prob_field)) is not None)
        if n_with_slot_prob_any == 0:
            warnings.append(f"No rows contain usable {args.slot_prob_field!r} values.")
            print(f"WARNING: no rows contain usable {args.slot_prob_field!r} values.", file=sys.stderr)
        if gold_field is None:
            warnings.append("No gold label field was detected; supervised metrics are null.")
        if not any(SLOT_TARGET_FIELD in row and SLOT_TARGET_VALID_FIELD in row for row in rows):
            warnings.append("Slot target fields were not detected; slot-target AUROC/AUPRC are null.")

        threshold_rows: list[dict[str, Any]] = []
        all_changed_examples: list[dict[str, Any]] = []
        for threshold in thresholds:
            threshold_row, changed_rows = build_threshold_row(
                rows,
                threshold,
                pred_field,
                gold_field,
                args.slot_prob_field,
            )
            threshold_rows.append(threshold_row)
            all_changed_examples.extend(
                example_record(row, threshold, pred_field, gold_field, args.slot_prob_field, group_fields)
                for row in changed_rows
            )

        group_rows: list[dict[str, Any]] = []
        for group_field in group_fields:
            if not any(group_field in row for row in rows):
                continue
            values = sorted({str(row.get(group_field)) for row in rows if group_field in row})
            for value in values:
                group_subset = [row for row in rows if str(row.get(group_field)) == value]
                for threshold in thresholds:
                    group_rows.append(
                        build_group_row(
                            group_subset,
                            threshold,
                            group_field,
                            value,
                            pred_field,
                            gold_field,
                            args.slot_prob_field,
                        )
                    )

        decision = choose_decision(rows, threshold_rows, pred_field, gold_field)
        best_threshold = select_best_threshold(threshold_rows)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "stage136a_slot_guard_shadow_report.json"
        md_path = output_dir / "stage136a_slot_guard_shadow_report.md"
        threshold_csv_path = output_dir / "stage136a_threshold_metrics.csv"
        group_csv_path = output_dir / "stage136a_group_metrics.csv"
        examples_path = output_dir / "stage136a_shadow_examples.jsonl"

        report = {
            "stage": "Stage136-A",
            "decision": decision,
            "policy": args.policy,
            "export_only": True,
            "safety": {
                "imports_model_code": False,
                "runs_training": False,
                "runs_model_eval": False,
                "modifies_prediction_files": False,
                "modifies_training_code": False,
                "modifies_model_code": False,
                "routes_slot_mismatch_into_final_logits": False,
                "changes_checkpoint_selection": False,
            },
            "input_files": [str(path) for path in input_files],
            "fields": {
                "prediction": pred_field,
                "gold": gold_field,
                "slot_probability": args.slot_prob_field,
                "slot_target": SLOT_TARGET_FIELD,
                "slot_target_valid": SLOT_TARGET_VALID_FIELD,
                "group_fields_requested": group_fields,
            },
            "threshold_metrics": threshold_rows,
            "group_metrics_rows": len(group_rows),
            "changed_examples_rows": len(all_changed_examples),
            "best_threshold": best_threshold,
            "warnings": warnings,
        }
        json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        write_csv(threshold_csv_path, threshold_rows)
        write_csv(group_csv_path, group_rows)
        with examples_path.open("w", encoding="utf-8") as fh:
            for record in all_changed_examples:
                fh.write(json.dumps(record, sort_keys=True) + "\n")
        write_markdown_report(
            md_path,
            decision,
            input_files,
            pred_field,
            gold_field,
            args.slot_prob_field,
            threshold_rows,
            group_rows,
            len(all_changed_examples),
            best_threshold,
            warnings,
        )

        print(f"decision={decision}")
        print(f"wrote={json_path}")
        print(f"wrote={md_path}")
        print(f"wrote={threshold_csv_path}")
        print(f"wrote={group_csv_path}")
        print(f"wrote={examples_path}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
