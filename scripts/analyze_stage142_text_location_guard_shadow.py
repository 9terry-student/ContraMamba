"""Stage142-A: reusable text-location shadow guard analyzer.

This script evaluates the deployable ``text_loc_disjoint`` policy as a
diagnostic-only shadow analyzer. It reads existing prediction JSONL files,
computes a shadow prediction for audit purposes, and writes reports. It does
not modify source prediction files, model behavior, training, checkpoints,
final logits, or final predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


POLICY_NAME = "text_loc_disjoint"
STAGE = "Stage142-A"
LABELS_DEFAULT = ["REFUTE", "NOT_ENTITLED", "SUPPORT"]

PRED_FIELD_CANDIDATES = [
    "prediction",
    "pred_label",
    "final_prediction",
    "final_pred",
    "pred",
    "label_pred",
    "composed_prediction",
    "base_prediction",
]
GOLD_FIELD_CANDIDATES = [
    "gold_label",
    "label",
    "target_label",
    "true_label",
    "final_label",
]
CLAIM_FIELD_CANDIDATES = ["claim", "core_claim"]
EVIDENCE_FIELD_CANDIDATES = ["evidence", "core_evidence"]
GROUP_FIELDS_DEFAULT = ["intervention_type", "stage122_family", "family"]

LOCATION_PREP_RE = re.compile(
    r"\b(?:in|at|near|from|to|inside|outside|around|across|through)\s+"
    r"([A-Z][A-Za-z]*(?:[\s\-][A-Z][A-Za-z]*){0,3})"
)
SPAN_CLEAN_RE = re.compile(r"[^A-Za-z\-\s]+")
SPACE_RE = re.compile(r"\s+")

EXCLUDED_TIME_WORDS = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "morning",
    "afternoon",
    "evening",
    "night",
    "today",
    "yesterday",
    "tomorrow",
}
EXCLUDED_LEADING_WORDS = {
    "dr",
    "captain",
    "coach",
    "mayor",
    "professor",
    "director",
    "chief",
    "officer",
    "minister",
    "president",
    "prime",
    "the",
    "a",
    "an",
    "during",
    "before",
    "after",
}

SAFETY_POLICY = {
    "shadow_only": True,
    "diagnostic_only": True,
    "final_logits_modified": False,
    "final_predictions_modified": False,
    "training_modified": False,
    "checkpoint_selection_modified": False,
    "stage128_guard_enabled": False,
    "stage15_used": False,
    "external_data_used_for_training": False,
    "threshold_used_for_model_selection": False,
}

POLICY_INPUT_SAFETY = {
    "uses_claim_text": True,
    "uses_evidence_text": True,
    "uses_original_prediction": True,
    "uses_intervention_type": False,
    "uses_slot_mismatch_target": False,
    "uses_gold_label_for_policy": False,
    "uses_diagnostic_family_for_policy": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the Stage142-A text_loc_disjoint shadow-only guard."
    )
    parser.add_argument(
        "--input-jsonl",
        action="append",
        required=True,
        help="Prediction JSONL file. May be passed more than once.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for Stage142-A outputs.")
    parser.add_argument("--prediction-field", default="auto", help="Prediction field name, or auto.")
    parser.add_argument("--gold-field", default="auto", help="Gold field name, or auto.")
    parser.add_argument("--claim-field", default="auto", help="Claim text field name, or auto.")
    parser.add_argument("--evidence-field", default="auto", help="Evidence text field name, or auto.")
    parser.add_argument(
        "--group-fields",
        default=",".join(GROUP_FIELDS_DEFAULT),
        help="Comma-separated audit-only group fields.",
    )
    parser.add_argument("--max-examples", type=int, default=200, help="Maximum changed examples to write.")
    parser.add_argument(
        "--write-shadow-jsonl",
        action="store_true",
        help="Write full per-row shadow JSONL with Stage142 fields added.",
    )
    parser.add_argument(
        "--label-set",
        default=",".join(LABELS_DEFAULT),
        help="Comma-separated label set for supervised metrics.",
    )
    return parser.parse_args()


def parse_label_set(raw: str) -> list[str]:
    labels = [normalize_label(part) for part in raw.split(",") if part.strip()]
    clean = [label for label in labels if label]
    return clean or list(LABELS_DEFAULT)


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


def discover_input_files(input_jsonl: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for item in input_jsonl:
        path = Path(item)
        key = str(path.resolve())
        if key not in seen:
            paths.append(path)
            seen.add(key)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise ValueError(f"Input file(s) not found: {missing}")
    return paths


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                errors.append({"path": str(path), "line": line_no, "error": str(exc)})
                continue
            if not isinstance(row, dict):
                errors.append({"path": str(path), "line": line_no, "error": "JSON value is not an object"})
                continue
            row["_stage142_source_file"] = str(path)
            row["_stage142_line_number"] = line_no
            rows.append(row)
    return rows, errors


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


def normalize_span(span: str) -> str | None:
    cleaned = SPAN_CLEAN_RE.sub(" ", span)
    cleaned = SPACE_RE.sub(" ", cleaned).strip()
    if not cleaned:
        return None
    words = cleaned.split()
    leading = words[0].lower().rstrip(".")
    if leading in EXCLUDED_LEADING_WORDS:
        return None
    if any(word.lower().strip("-") in EXCLUDED_TIME_WORDS for word in words):
        return None
    return " ".join(words)


def extract_location_like_spans(text: Any) -> list[str]:
    if text is None:
        return []
    spans: list[str] = []
    seen: set[str] = set()
    for match in LOCATION_PREP_RE.finditer(str(text)):
        normalized = normalize_span(match.group(1))
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        spans.append(normalized)
        seen.add(key)
    return spans


def compute_location_features(claim_text: Any, evidence_text: Any) -> dict[str, Any]:
    claim_spans = extract_location_like_spans(claim_text)
    evidence_spans = extract_location_like_spans(evidence_text)
    claim_set = {span.lower() for span in claim_spans}
    evidence_set = {span.lower() for span in evidence_spans}
    intersection = claim_set & evidence_set
    union = claim_set | evidence_set
    both_nonempty = bool(claim_set and evidence_set)
    loc_disjoint = both_nonempty and not intersection
    return {
        "claim_location_spans": claim_spans,
        "evidence_location_spans": evidence_spans,
        "loc_disjoint": loc_disjoint,
        "loc_any_diff": bool((claim_set ^ evidence_set) if both_nonempty else False),
        "loc_jaccard": (len(intersection) / len(union)) if union else None,
    }


def apply_shadow_policy(
    prediction: str | None,
    claim_text: Any,
    evidence_text: Any,
) -> dict[str, Any]:
    features = compute_location_features(claim_text, evidence_text)
    triggered = prediction == "SUPPORT" and bool(features["loc_disjoint"])
    shadow_prediction = "NOT_ENTITLED" if triggered else prediction
    return {
        "shadow_prediction": shadow_prediction,
        "stage142_original_prediction": prediction,
        "stage142_policy": POLICY_NAME,
        "stage142_policy_triggered": triggered,
        "stage142_claim_location_spans": features["claim_location_spans"],
        "stage142_evidence_location_spans": features["evidence_location_spans"],
        "stage142_loc_disjoint": features["loc_disjoint"],
        "stage142_loc_any_diff": features["loc_any_diff"],
        "stage142_loc_jaccard": features["loc_jaccard"],
        "stage142_diagnostic_only": True,
    }


def safe_div(num: float, den: float) -> float | None:
    return num / den if den else None


def delta(after: Any, before: Any) -> Any:
    if after is None or before is None:
        return None
    return after - before


def count_predictions(labels: list[str | None], label_set: list[str]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for label in labels:
        if label in label_set:
            counter[label] += 1
    return counter


def class_precision_recall(
    golds: list[str],
    preds: list[str],
    label: str,
) -> tuple[float | None, float | None]:
    tp = sum(1 for gold, pred in zip(golds, preds) if gold == label and pred == label)
    fp = sum(1 for gold, pred in zip(golds, preds) if gold != label and pred == label)
    fn = sum(1 for gold, pred in zip(golds, preds) if gold == label and pred != label)
    return safe_div(tp, tp + fp), safe_div(tp, tp + fn)


def macro_f1(golds: list[str], preds: list[str], label_set: list[str]) -> float | None:
    if not golds:
        return None
    f1s: list[float] = []
    for label in label_set:
        precision, recall = class_precision_recall(golds, preds, label)
        if precision is None and recall is None:
            f1s.append(0.0)
        elif not precision or not recall:
            f1s.append(0.0)
        else:
            f1s.append(2.0 * precision * recall / (precision + recall))
    return sum(f1s) / len(f1s) if label_set else None


def supervised_metrics(
    golds: list[str],
    preds: list[str],
    label_set: list[str],
) -> dict[str, float | int | None]:
    result: dict[str, float | int | None] = {
        "accuracy": None,
        "macro_f1": None,
        "false_support": None,
        "false_ne": None,
        "support_precision": None,
        "support_recall": None,
        "refute_recall": None,
        "not_entitled_recall": None,
    }
    if not golds:
        return result
    result["accuracy"] = safe_div(sum(g == p for g, p in zip(golds, preds)), len(golds))
    result["macro_f1"] = macro_f1(golds, preds, label_set)
    result["false_support"] = sum(
        1 for gold, pred in zip(golds, preds) if pred == "SUPPORT" and gold != "SUPPORT"
    )
    result["false_ne"] = sum(
        1
        for gold, pred in zip(golds, preds)
        if pred == "NOT_ENTITLED" and gold != "NOT_ENTITLED"
    )
    support_precision, support_recall = class_precision_recall(golds, preds, "SUPPORT")
    _, refute_recall = class_precision_recall(golds, preds, "REFUTE")
    _, not_entitled_recall = class_precision_recall(golds, preds, "NOT_ENTITLED")
    result["support_precision"] = support_precision
    result["support_recall"] = support_recall
    result["refute_recall"] = refute_recall
    result["not_entitled_recall"] = not_entitled_recall
    return result


def prefixed_supervised_metrics(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in [
        "accuracy",
        "macro_f1",
        "false_support",
        "false_ne",
        "support_precision",
        "support_recall",
        "refute_recall",
        "not_entitled_recall",
    ]:
        out[f"{name}_before"] = before.get(name)
        out[f"{name}_after"] = after.get(name)
    out["delta_false_support"] = delta(after.get("false_support"), before.get("false_support"))
    out["delta_false_ne"] = delta(after.get("false_ne"), before.get("false_ne"))
    out["delta_macro_f1"] = delta(after.get("macro_f1"), before.get("macro_f1"))
    return out


def compute_metrics(
    rows: list[dict[str, Any]],
    pred_field: str | None,
    gold_field: str | None,
    claim_field: str | None,
    evidence_field: str | None,
    label_set: list[str],
) -> dict[str, Any]:
    predictions_before: list[str | None] = []
    predictions_after: list[str | None] = []
    golds: list[str] = []
    supervised_before: list[str] = []
    supervised_after: list[str] = []
    n_with_prediction = 0
    n_with_claim_evidence = 0
    n_changed_total = 0
    n_support_to_ne = 0
    support_rows_with_loc_disjoint = 0
    rows_claim_locs_nonempty = 0
    rows_evidence_locs_nonempty = 0
    rows_both_locs_nonempty = 0
    rows_loc_disjoint = 0
    feature_false_support_tp = 0
    feature_correct_support_fp = 0
    feature_false_support_fn = 0

    for row in rows:
        pred = normalize_label(row.get(pred_field)) if pred_field else None
        after = row.get("shadow_prediction")
        if after is not None:
            after = normalize_label(after)
        if pred in label_set:
            n_with_prediction += 1
        if claim_field and evidence_field and row.get(claim_field) is not None and row.get(evidence_field) is not None:
            n_with_claim_evidence += 1
        predictions_before.append(pred)
        predictions_after.append(after)
        claim_locs = row.get("stage142_claim_location_spans") or []
        evidence_locs = row.get("stage142_evidence_location_spans") or []
        loc_disjoint = bool(row.get("stage142_loc_disjoint"))
        if claim_locs:
            rows_claim_locs_nonempty += 1
        if evidence_locs:
            rows_evidence_locs_nonempty += 1
        if claim_locs and evidence_locs:
            rows_both_locs_nonempty += 1
        if loc_disjoint:
            rows_loc_disjoint += 1
        if pred == "SUPPORT" and loc_disjoint:
            support_rows_with_loc_disjoint += 1
        if pred != after:
            n_changed_total += 1
            if pred == "SUPPORT" and after == "NOT_ENTITLED":
                n_support_to_ne += 1
        gold = normalize_label(row.get(gold_field)) if gold_field else None
        if gold in label_set and pred in label_set and after in label_set:
            golds.append(gold)
            supervised_before.append(pred)
            supervised_after.append(after)
            if pred == "SUPPORT":
                triggered = bool(row.get("stage142_policy_triggered"))
                false_support = gold != "SUPPORT"
                if triggered and false_support:
                    feature_false_support_tp += 1
                elif triggered and not false_support:
                    feature_correct_support_fp += 1
                elif not triggered and false_support:
                    feature_false_support_fn += 1

    before_counts = count_predictions(predictions_before, label_set)
    after_counts = count_predictions(predictions_after, label_set)
    before_supervised = supervised_metrics(golds, supervised_before, label_set)
    after_supervised = supervised_metrics(golds, supervised_after, label_set)

    metrics: dict[str, Any] = {
        "n_rows": len(rows),
        "n_valid_rows": len(rows),
        "n_with_prediction": n_with_prediction,
        "n_with_claim_evidence": n_with_claim_evidence,
        "n_changed_total": n_changed_total,
        "n_support_to_ne": n_support_to_ne,
        "support_rows_with_loc_disjoint": support_rows_with_loc_disjoint,
        "rows_claim_locs_nonempty": rows_claim_locs_nonempty,
        "rows_evidence_locs_nonempty": rows_evidence_locs_nonempty,
        "rows_both_locs_nonempty": rows_both_locs_nonempty,
        "rows_loc_disjoint": rows_loc_disjoint,
        "prediction_counts_before": dict(before_counts),
        "prediction_counts_after": dict(after_counts),
        "feature_false_support_tp": feature_false_support_tp if golds else None,
        "feature_correct_support_fp": feature_correct_support_fp if golds else None,
        "feature_false_support_fn": feature_false_support_fn if golds else None,
        "feature_precision_for_false_support_among_support_preds": safe_div(
            feature_false_support_tp, feature_false_support_tp + feature_correct_support_fp
        )
        if golds
        else None,
        "feature_recall_for_false_support_among_support_preds": safe_div(
            feature_false_support_tp, feature_false_support_tp + feature_false_support_fn
        )
        if golds
        else None,
    }
    for label in label_set:
        key = label.lower()
        metrics[f"{key}_pred_before"] = before_counts[label]
        metrics[f"{key}_pred_after"] = after_counts[label]
    metrics.update(prefixed_supervised_metrics(before_supervised, after_supervised))
    return metrics


def add_metrics_rows(
    rows: list[dict[str, Any]],
    pred_field: str | None,
    gold_field: str | None,
    claim_field: str | None,
    evidence_field: str | None,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        pred = normalize_label(row.get(pred_field)) if pred_field else None
        gold = normalize_label(row.get(gold_field)) if gold_field else None
        claim = row.get(claim_field) if claim_field else None
        evidence = row.get(evidence_field) if evidence_field else None
        policy_fields = apply_shadow_policy(pred, claim, evidence)
        out = dict(row)
        out["_stage142_normalized_prediction"] = pred
        out["_stage142_normalized_gold"] = gold
        out.update(policy_fields)
        enriched.append(out)
    return enriched


def process_file(
    path: Path,
    args: argparse.Namespace,
    label_set: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows, errors = load_jsonl(path)
    pred_field = infer_field(rows, args.prediction_field, PRED_FIELD_CANDIDATES, "prediction", required=False)
    gold_field = infer_field(rows, args.gold_field, GOLD_FIELD_CANDIDATES, "gold", required=False)
    claim_field = infer_field(rows, args.claim_field, CLAIM_FIELD_CANDIDATES, "claim", required=False)
    evidence_field = infer_field(rows, args.evidence_field, EVIDENCE_FIELD_CANDIDATES, "evidence", required=False)
    enriched = add_metrics_rows(rows, pred_field, gold_field, claim_field, evidence_field)
    metrics = compute_metrics(enriched, pred_field, gold_field, claim_field, evidence_field, label_set)
    metrics.update(
        {
            "path": str(path),
            "prediction_field": pred_field,
            "gold_field": gold_field,
            "claim_field": claim_field,
            "evidence_field": evidence_field,
            "n_malformed_rows": len(errors),
            "has_usable_gold": metrics.get("accuracy_before") is not None,
        }
    )
    changed = [row for row in enriched if row.get("stage142_policy_triggered")]
    return metrics, changed, enriched, errors


def aggregate_metrics(
    rows: list[dict[str, Any]],
    file_metrics: list[dict[str, Any]],
    label_set: list[str],
    malformed_count: int,
) -> dict[str, Any]:
    aggregate = compute_metrics(
        rows,
        "_stage142_normalized_prediction",
        "_stage142_normalized_gold",
        None,
        None,
        label_set,
    )
    aggregate["n_malformed_rows"] = malformed_count
    aggregate["n_with_claim_evidence"] = sum(metrics.get("n_with_claim_evidence", 0) for metrics in file_metrics)
    gold_files = [metrics for metrics in file_metrics if metrics.get("has_usable_gold")]
    if not gold_files:
        for key in [
            "accuracy_before",
            "accuracy_after",
            "macro_f1_before",
            "macro_f1_after",
            "false_support_before",
            "false_support_after",
            "false_ne_before",
            "false_ne_after",
            "delta_false_support",
            "delta_false_ne",
            "delta_macro_f1",
            "support_precision_before",
            "support_precision_after",
            "support_recall_before",
            "support_recall_after",
            "refute_recall_before",
            "refute_recall_after",
            "not_entitled_recall_before",
            "not_entitled_recall_after",
        ]:
            aggregate[key] = None
        aggregate["feature_false_support_tp"] = None
        aggregate["feature_correct_support_fp"] = None
        aggregate["feature_false_support_fn"] = None
        aggregate["feature_precision_for_false_support_among_support_preds"] = None
        aggregate["feature_recall_for_false_support_among_support_preds"] = None
        aggregate["min_per_file_delta_macro_f1"] = None
    else:
        aggregate["min_per_file_delta_macro_f1"] = min(
            (m.get("delta_macro_f1") for m in gold_files if m.get("delta_macro_f1") is not None),
            default=None,
        )
    aggregate["n_files"] = len(file_metrics)
    aggregate["n_files_with_usable_gold"] = len(gold_files)
    return aggregate


def choose_decision(aggregate: dict[str, Any]) -> str:
    if aggregate.get("n_valid_rows", 0) == 0:
        return "STAGE142_TEXT_LOCATION_GUARD_NO_VALID_INPUTS"
    if aggregate.get("n_files_with_usable_gold", 0) == 0:
        return "STAGE142_TEXT_LOCATION_GUARD_COUNT_ONLY_NO_GOLD"
    delta_false_support = aggregate.get("delta_false_support")
    delta_false_ne = aggregate.get("delta_false_ne")
    min_delta_macro_f1 = aggregate.get("min_per_file_delta_macro_f1")
    if (
        delta_false_support is not None
        and delta_false_support < 0
        and delta_false_ne is not None
        and delta_false_ne <= 5
        and min_delta_macro_f1 is not None
        and min_delta_macro_f1 >= -0.01
    ):
        return "STAGE142_TEXT_LOCATION_GUARD_SHADOW_CANDIDATE_ROBUST"
    if delta_false_support is not None and delta_false_support < 0:
        return "STAGE142_TEXT_LOCATION_GUARD_SHADOW_MIXED"
    if aggregate.get("n_changed_total", 0) > 0:
        return "STAGE142_TEXT_LOCATION_GUARD_SHADOW_HARM_OR_NO_FS_GAIN"
    return "STAGE142_TEXT_LOCATION_GUARD_SHADOW_NO_EFFECT"


def group_metrics(
    rows_by_file: dict[str, list[dict[str, Any]]],
    file_metrics_by_path: dict[str, dict[str, Any]],
    group_fields: list[str],
    label_set: list[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path, rows in rows_by_file.items():
        fields = file_metrics_by_path[path]
        gold_field = fields.get("gold_field")
        if not gold_field:
            continue
        for group_field in group_fields:
            if not any(group_field in row for row in rows):
                continue
            values = sorted({str(row.get(group_field)) for row in rows if group_field in row})
            for value in values:
                subset = [row for row in rows if str(row.get(group_field)) == value]
                metrics = compute_metrics(
                    subset,
                    fields.get("prediction_field"),
                    gold_field,
                    fields.get("claim_field"),
                    fields.get("evidence_field"),
                    label_set,
                )
                out.append(
                    {
                        "path": path,
                        "audit_group_field": group_field,
                        "audit_group_value": value,
                        "n_total": metrics["n_rows"],
                        "n_changed_total": metrics["n_changed_total"],
                        "false_support_before": metrics["false_support_before"],
                        "false_support_after": metrics["false_support_after"],
                        "delta_false_support": metrics["delta_false_support"],
                        "false_ne_before": metrics["false_ne_before"],
                        "false_ne_after": metrics["false_ne_after"],
                        "delta_false_ne": metrics["delta_false_ne"],
                        "macro_f1_before": metrics["macro_f1_before"],
                        "macro_f1_after": metrics["macro_f1_after"],
                        "delta_macro_f1": metrics["delta_macro_f1"],
                        "support_pred_before": metrics.get("support_pred_before"),
                        "support_pred_after": metrics.get("support_pred_after"),
                    }
                )
    return out


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


def write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def markdown_table(rows: list[dict[str, Any]], columns: list[str], max_rows: int = 20) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for row in rows[:max_rows]:
        values = [str(row.get(column, "")) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    if len(rows) > max_rows:
        lines.append(f"| ... | ... | ... | ... |")
    return "\n".join(lines)


def build_markdown_report(report: dict[str, Any]) -> str:
    aggregate = report["aggregate_metrics"]
    lines = [
        "# Stage142-A Text Location Guard Shadow Analyzer",
        "",
        f"## Summary decision",
        "",
        f"`{report['decision']}`",
        "",
        "## Policy definition",
        "",
        "Policy `text_loc_disjoint`: for SUPPORT predictions only, extract location-like spans from claim and evidence text; if both sides contain non-empty disjoint location sets, shadow SUPPORT to NOT_ENTITLED.",
        "",
        "## Policy input safety",
        "",
        "The policy uses only claim text, evidence text, and the original prediction label. Gold labels, diagnostic metadata, group fields, row identifiers, and file paths are not used by the policy.",
        "",
        "## Aggregate metrics",
        "",
        markdown_table([aggregate], [
            "n_valid_rows",
            "n_changed_total",
            "n_support_to_ne",
            "false_support_before",
            "false_support_after",
            "delta_false_support",
            "false_ne_before",
            "false_ne_after",
            "delta_false_ne",
            "macro_f1_before",
            "macro_f1_after",
            "delta_macro_f1",
        ]),
        "",
        "## Output pointers",
        "",
        f"- Per-file metrics: `{report['per_file_metrics_path']}`",
        f"- Group audit: `{report['group_metrics_path']}`",
        f"- Changed examples: `{report['changed_examples_path']}`",
        "",
        "## Safety policy",
        "",
        "This script is shadow-only. It does not modify source predictions. It must not be interpreted as final model integration.",
        "",
        "```json",
        json.dumps(report["safety_policy"], indent=2, sort_keys=True),
        "```",
        "",
        "## Interpretation",
        "",
        report["interpretation"],
        "",
    ]
    if report.get("shadow_predictions_path"):
        lines.insert(-4, f"- Shadow predictions: `{report['shadow_predictions_path']}`")
    return "\n".join(lines)


def trim_changed_row(row: dict[str, Any]) -> dict[str, Any]:
    internal = {
        "_stage142_source_file",
        "_stage142_line_number",
        "_stage142_normalized_prediction",
        "_stage142_normalized_gold",
    }
    out = {key: value for key, value in row.items() if key not in internal}
    out["stage142_source_file"] = row.get("_stage142_source_file")
    out["stage142_line_number"] = row.get("_stage142_line_number")
    return out


def main() -> None:
    args = parse_args()
    label_set = parse_label_set(args.label_set)
    input_files = discover_input_files(args.input_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    group_fields = [field.strip() for field in args.group_fields.split(",") if field.strip()]

    file_metrics: list[dict[str, Any]] = []
    all_changed: list[dict[str, Any]] = []
    all_enriched: list[dict[str, Any]] = []
    all_shadow: list[dict[str, Any]] = []
    all_errors: list[dict[str, Any]] = []
    rows_by_file: dict[str, list[dict[str, Any]]] = {}

    for path in input_files:
        metrics, changed, enriched, errors = process_file(path, args, label_set)
        file_metrics.append(metrics)
        all_changed.extend(trim_changed_row(row) for row in changed)
        all_enriched.extend(enriched)
        if args.write_shadow_jsonl:
            all_shadow.extend(trim_changed_row(row) for row in enriched)
        all_errors.extend(errors)
        rows_by_file[str(path)] = enriched

    file_metrics_by_path = {metrics["path"]: metrics for metrics in file_metrics}
    group_rows = group_metrics(rows_by_file, file_metrics_by_path, group_fields, label_set)
    aggregate = aggregate_metrics(all_enriched, file_metrics, label_set, len(all_errors))
    decision = choose_decision(aggregate)

    file_metrics_path = output_dir / "stage142_file_metrics.csv"
    aggregate_metrics_path = output_dir / "stage142_aggregate_metrics.json"
    group_metrics_path = output_dir / "stage142_group_metrics.csv"
    changed_examples_path = output_dir / "stage142_changed_examples.jsonl"
    shadow_predictions_path = output_dir / "stage142_shadow_predictions.jsonl"
    report_json_path = output_dir / "stage142_text_location_guard_shadow_report.json"
    report_md_path = output_dir / "stage142_text_location_guard_shadow_report.md"

    write_csv(file_metrics_path, file_metrics)
    write_json(aggregate_metrics_path, aggregate)
    write_csv(group_metrics_path, group_rows)
    write_jsonl(changed_examples_path, all_changed[: max(args.max_examples, 0)])
    if args.write_shadow_jsonl:
        write_jsonl(shadow_predictions_path, all_shadow)

    interpretation = (
        "This is a reusable diagnostic shadow analyzer for text_loc_disjoint. "
        "It can quantify candidate behavior on prediction JSONL exports, but it "
        "does not justify or perform final-logit or final-prediction integration."
    )
    report = {
        "stage": STAGE,
        "decision": decision,
        "policy": {
            "name": POLICY_NAME,
            "description": "For SUPPORT predictions only, disjoint non-empty extracted claim/evidence location sets shadow SUPPORT to NOT_ENTITLED.",
        },
        "input_files": [str(path) for path in input_files],
        "output_dir": str(output_dir),
        "aggregate_metrics": aggregate,
        "per_file_metrics_path": str(file_metrics_path),
        "group_metrics_path": str(group_metrics_path),
        "changed_examples_path": str(changed_examples_path),
        "shadow_predictions_path": str(shadow_predictions_path) if args.write_shadow_jsonl else None,
        "policy_input_safety": POLICY_INPUT_SAFETY,
        "safety_policy": SAFETY_POLICY,
        "interpretation": interpretation,
        "jsonl_read_errors": all_errors,
        "label_set": label_set,
        "group_fields_requested": group_fields,
    }
    write_json(report_json_path, report)
    report_md_path.write_text(build_markdown_report(report), encoding="utf-8")

    print(f"decision={decision}")
    print(f"wrote={report_json_path}")
    print(f"wrote={report_md_path}")
    print(f"wrote={file_metrics_path}")
    print(f"wrote={aggregate_metrics_path}")
    print(f"wrote={group_metrics_path}")
    print(f"wrote={changed_examples_path}")
    if args.write_shadow_jsonl:
        print(f"wrote={shadow_predictions_path}")


if __name__ == "__main__":
    main()




