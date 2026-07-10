"""Stage145-A: refined text-location shadow guard analyzer v2.

This script evaluates ``text_loc_disjoint_v2`` as a diagnostic-only shadow
analyzer. It improves the Stage142 raw location-disjoint policy with
deterministic alias canonicalization and a conservative organization-like span
block. It does not mutate source prediction files, model behavior, training,
checkpoints, final logits, or final predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


STAGE = "Stage145-A"
POLICY_NAME_DEFAULT = "text_loc_disjoint_v2"
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
GROUP_FIELDS_DEFAULT = [
    "intervention_type",
    "stage122_family",
    "family",
    "stage144_case_type",
    "stage144_expected_policy_behavior",
]

LOCATION_PREP_RE = re.compile(
    r"\b(?:in|at|near|from|to|inside|outside|around|across|through)\s+"
    r"([A-Z][A-Za-z]*(?:[\s\-][A-Z][A-Za-z]*){0,3})"
)
SPAN_CLEAN_RE = re.compile(r"[^A-Za-z\-\s]+")
CANONICAL_CLEAN_RE = re.compile(r"[^a-z\-\s]+")
SPACE_RE = re.compile(r"\s+")

KNOWN_LOWERCASE_ALIASES = [
    "new york city",
    "san francisco",
    "los angeles",
    "beacon harbor",
    "elm valley",
    "falcon ridge",
    "aurora city",
    "new york",
    "cambridge",
    "dublin",
    "busan",
    "paris",
    "cork",
    "seoul",
    "lyon",
    "oxford",
    "nyc",
    "la",
    "sf",
]
KNOWN_LOWERCASE_RE = re.compile(
    r"\b(?:in|at|near|from|to|inside|outside|around|across|through)\s+("
    + "|".join(re.escape(alias) for alias in KNOWN_LOWERCASE_ALIASES)
    + r")\b",
    flags=re.IGNORECASE,
)

ALIAS_MAP = {
    "nyc": "new york",
    "new york city": "new york",
    "new york": "new york",
    "la": "los angeles",
    "los angeles": "los angeles",
    "sf": "san francisco",
    "san francisco": "san francisco",
}
KNOWN_CANONICAL_LOCATIONS = {
    "dublin",
    "cork",
    "beacon harbor",
    "elm valley",
    "falcon ridge",
    "aurora city",
    "seoul",
    "busan",
    "paris",
    "lyon",
    "oxford",
    "cambridge",
    "new york",
    "los angeles",
    "san francisco",
}

ORG_LIKE_WORDS = {
    "lab",
    "labs",
    "laboratory",
    "laboratories",
    "center",
    "centre",
    "institute",
    "initiative",
    "university",
    "college",
    "school",
    "hospital",
    "clinic",
    "company",
    "corporation",
    "corp",
    "inc",
    "ltd",
    "llc",
    "foundation",
    "agency",
    "department",
    "office",
    "council",
    "committee",
    "museum",
    "group",
    "team",
    "authority",
    "administration",
    "ministry",
    "bureau",
}

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

POLICY_INPUT_SAFETY = {
    "uses_claim_text": True,
    "uses_evidence_text": True,
    "uses_original_prediction": True,
    "uses_deterministic_alias_rules": True,
    "uses_deterministic_org_like_rules": True,
    "uses_intervention_type": False,
    "uses_slot_mismatch_target": False,
    "uses_gold_label_for_policy": False,
    "uses_diagnostic_family_for_policy": False,
    "uses_file_path_heuristics": False,
    "uses_row_id_heuristics": False,
}

SAFETY_POLICY = {
    "shadow_only": True,
    "diagnostic_only": True,
    "final_logits_modified": False,
    "final_predictions_modified": False,
    "source_predictions_mutated": False,
    "training_modified": False,
    "checkpoint_selection_modified": False,
    "stage128_guard_enabled": False,
    "stage15_used": False,
    "external_data_used_for_training": False,
    "threshold_used_for_model_selection": False,
}

GOLD_METRIC_KEYS = [
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the Stage145-A text_loc_disjoint_v2 shadow-only guard."
    )
    parser.add_argument(
        "--input-jsonl",
        action="append",
        required=True,
        help="Prediction JSONL file. May be passed more than once.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for Stage145-A outputs.")
    parser.add_argument("--prediction-field", default="auto", help="Prediction field name, or auto.")
    parser.add_argument("--gold-field", default="auto", help="Gold field name, or auto.")
    parser.add_argument("--claim-field", default="auto", help="Claim text field name, or auto.")
    parser.add_argument("--evidence-field", default="auto", help="Evidence text field name, or auto.")
    parser.add_argument(
        "--group-fields",
        default=",".join(GROUP_FIELDS_DEFAULT),
        help="Comma-separated audit-only group fields.",
    )
    parser.add_argument("--max-examples", type=int, default=300, help="Maximum changed examples to write.")
    parser.add_argument(
        "--write-shadow-jsonl",
        action="store_true",
        help="Write full per-row shadow JSONL with Stage145 fields added.",
    )
    parser.add_argument(
        "--label-set",
        default=",".join(LABELS_DEFAULT),
        help="Comma-separated label set for supervised metrics.",
    )
    parser.add_argument(
        "--disable-built-in-aliases",
        action="store_true",
        help="Disable built-in alias/canonicalization mappings.",
    )
    parser.add_argument(
        "--disable-organization-block",
        action="store_true",
        help="Disable the conservative organization-like span block.",
    )
    parser.add_argument("--policy-name", default=POLICY_NAME_DEFAULT, help="Policy name to report.")
    return parser.parse_args()


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


def parse_label_set(raw: str) -> list[str]:
    labels = [normalize_label(part) for part in raw.split(",") if part.strip()]
    clean = [label for label in labels if label]
    return clean or list(LABELS_DEFAULT)


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
            row["_stage145_source_file"] = str(path)
            row["_stage145_line_number"] = line_no
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


def append_unique(spans: list[str], seen: set[str], span: str) -> None:
    key = span.lower()
    if key not in seen:
        spans.append(span)
        seen.add(key)


def extract_known_lowercase_location_spans(text: Any) -> list[str]:
    if text is None:
        return []
    spans: list[str] = []
    seen: set[str] = set()
    for match in KNOWN_LOWERCASE_RE.finditer(str(text)):
        normalized = normalize_span(match.group(1))
        if normalized:
            append_unique(spans, seen, normalized)
    return spans


def extract_location_like_spans(text: Any) -> list[str]:
    if text is None:
        return []
    spans: list[str] = []
    seen: set[str] = set()
    text_s = str(text)
    for match in LOCATION_PREP_RE.finditer(text_s):
        normalized = normalize_span(match.group(1))
        if normalized:
            append_unique(spans, seen, normalized)
    for span in extract_known_lowercase_location_spans(text_s):
        append_unique(spans, seen, span)
    return spans


def canonicalize_location_span(span: str, *, use_aliases: bool = True) -> str | None:
    cleaned = CANONICAL_CLEAN_RE.sub(" ", str(span).lower())
    cleaned = SPACE_RE.sub(" ", cleaned).strip()
    if not cleaned:
        return None
    if use_aliases:
        cleaned = ALIAS_MAP.get(cleaned, cleaned)
    if cleaned in KNOWN_CANONICAL_LOCATIONS:
        return cleaned
    return cleaned


def is_organization_like_span(span: str) -> bool:
    cleaned = CANONICAL_CLEAN_RE.sub(" ", str(span).lower())
    tokens = [token.strip("-") for token in SPACE_RE.sub(" ", cleaned).split()]
    return any(token in ORG_LIKE_WORDS for token in tokens)


def unique_canonical_spans(spans: list[str], *, use_aliases: bool) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for span in spans:
        canonical = canonicalize_location_span(span, use_aliases=use_aliases)
        if canonical and canonical not in seen:
            out.append(canonical)
            seen.add(canonical)
    return out


def compute_location_features_v2(
    claim_text: Any,
    evidence_text: Any,
    *,
    use_aliases: bool = True,
    organization_block_enabled: bool = True,
) -> dict[str, Any]:
    claim_spans = extract_location_like_spans(claim_text)
    evidence_spans = extract_location_like_spans(evidence_text)
    claim_canonical = unique_canonical_spans(claim_spans, use_aliases=use_aliases)
    evidence_canonical = unique_canonical_spans(evidence_spans, use_aliases=use_aliases)

    claim_raw_set = {span.lower() for span in claim_spans}
    evidence_raw_set = {span.lower() for span in evidence_spans}
    claim_canonical_set = set(claim_canonical)
    evidence_canonical_set = set(evidence_canonical)

    claim_org_like = [span for span in claim_spans if is_organization_like_span(span)]
    evidence_org_like = [span for span in evidence_spans if is_organization_like_span(span)]
    blocked = organization_block_enabled and bool(claim_org_like or evidence_org_like)

    raw_both_nonempty = bool(claim_raw_set and evidence_raw_set)
    canonical_both_nonempty = bool(claim_canonical_set and evidence_canonical_set)
    raw_overlap = claim_raw_set & evidence_raw_set
    canonical_overlap = claim_canonical_set & evidence_canonical_set

    return {
        "claim_location_spans_raw": claim_spans,
        "evidence_location_spans_raw": evidence_spans,
        "claim_location_spans_canonical": claim_canonical,
        "evidence_location_spans_canonical": evidence_canonical,
        "claim_org_like_spans": claim_org_like,
        "evidence_org_like_spans": evidence_org_like,
        "trigger_blocked_by_org_like_span": blocked,
        "loc_disjoint_raw": raw_both_nonempty and not raw_overlap,
        "loc_disjoint_canonical": canonical_both_nonempty and not canonical_overlap,
        "loc_any_overlap_canonical": bool(canonical_overlap),
    }


def apply_shadow_policy_v2(
    prediction: str | None,
    claim_text: Any,
    evidence_text: Any,
    *,
    policy_name: str,
    use_aliases: bool = True,
    organization_block_enabled: bool = True,
) -> dict[str, Any]:
    features = compute_location_features_v2(
        claim_text,
        evidence_text,
        use_aliases=use_aliases,
        organization_block_enabled=organization_block_enabled,
    )
    triggered = (
        prediction == "SUPPORT"
        and not features["trigger_blocked_by_org_like_span"]
        and bool(features["loc_disjoint_canonical"])
    )
    shadow_prediction = "NOT_ENTITLED" if triggered else prediction
    return {
        "shadow_prediction": shadow_prediction,
        "stage145_original_prediction": prediction,
        "stage145_policy": policy_name,
        "stage145_policy_triggered": triggered,
        "stage145_claim_location_spans_raw": features["claim_location_spans_raw"],
        "stage145_evidence_location_spans_raw": features["evidence_location_spans_raw"],
        "stage145_claim_location_spans_canonical": features["claim_location_spans_canonical"],
        "stage145_evidence_location_spans_canonical": features["evidence_location_spans_canonical"],
        "stage145_claim_org_like_spans": features["claim_org_like_spans"],
        "stage145_evidence_org_like_spans": features["evidence_org_like_spans"],
        "stage145_trigger_blocked_by_org_like_span": features["trigger_blocked_by_org_like_span"],
        "stage145_loc_disjoint_raw": features["loc_disjoint_raw"],
        "stage145_loc_disjoint_canonical": features["loc_disjoint_canonical"],
        "stage145_loc_any_overlap_canonical": features["loc_any_overlap_canonical"],
        "stage145_diagnostic_only": True,
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


def class_precision_recall(golds: list[str], preds: list[str], label: str) -> tuple[float | None, float | None]:
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


def supervised_metrics(golds: list[str], preds: list[str], label_set: list[str]) -> dict[str, float | int | None]:
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
    support_rows_with_raw_loc_disjoint = 0
    support_rows_with_canonical_loc_disjoint = 0
    support_rows_blocked_by_org_like_span = 0
    rows_claim_locs_nonempty_raw = 0
    rows_evidence_locs_nonempty_raw = 0
    rows_claim_locs_nonempty_canonical = 0
    rows_evidence_locs_nonempty_canonical = 0
    rows_both_locs_nonempty_canonical = 0
    rows_loc_disjoint_canonical = 0
    rows_org_like_span_present = 0
    feature_false_support_tp = 0
    feature_correct_support_fp = 0
    feature_false_support_fn = 0

    for row in rows:
        pred = normalize_label(row.get(pred_field)) if pred_field else None
        after = normalize_label(row.get("shadow_prediction"))
        predictions_before.append(pred)
        predictions_after.append(after)
        if pred in label_set:
            n_with_prediction += 1
        if claim_field and evidence_field and row.get(claim_field) is not None and row.get(evidence_field) is not None:
            n_with_claim_evidence += 1

        claim_raw = row.get("stage145_claim_location_spans_raw") or []
        evidence_raw = row.get("stage145_evidence_location_spans_raw") or []
        claim_canonical = row.get("stage145_claim_location_spans_canonical") or []
        evidence_canonical = row.get("stage145_evidence_location_spans_canonical") or []
        raw_disjoint = bool(row.get("stage145_loc_disjoint_raw"))
        canonical_disjoint = bool(row.get("stage145_loc_disjoint_canonical"))
        org_blocked = bool(row.get("stage145_trigger_blocked_by_org_like_span"))
        org_present = bool(row.get("stage145_claim_org_like_spans") or row.get("stage145_evidence_org_like_spans"))

        if claim_raw:
            rows_claim_locs_nonempty_raw += 1
        if evidence_raw:
            rows_evidence_locs_nonempty_raw += 1
        if claim_canonical:
            rows_claim_locs_nonempty_canonical += 1
        if evidence_canonical:
            rows_evidence_locs_nonempty_canonical += 1
        if claim_canonical and evidence_canonical:
            rows_both_locs_nonempty_canonical += 1
        if canonical_disjoint:
            rows_loc_disjoint_canonical += 1
        if org_present:
            rows_org_like_span_present += 1
        if pred == "SUPPORT" and raw_disjoint:
            support_rows_with_raw_loc_disjoint += 1
        if pred == "SUPPORT" and canonical_disjoint:
            support_rows_with_canonical_loc_disjoint += 1
        if pred == "SUPPORT" and org_blocked:
            support_rows_blocked_by_org_like_span += 1
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
                triggered = bool(row.get("stage145_policy_triggered"))
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
        "prediction_counts_before": dict(before_counts),
        "prediction_counts_after": dict(after_counts),
        "support_rows_with_raw_loc_disjoint": support_rows_with_raw_loc_disjoint,
        "support_rows_with_canonical_loc_disjoint": support_rows_with_canonical_loc_disjoint,
        "support_rows_blocked_by_org_like_span": support_rows_blocked_by_org_like_span,
        "rows_claim_locs_nonempty_raw": rows_claim_locs_nonempty_raw,
        "rows_evidence_locs_nonempty_raw": rows_evidence_locs_nonempty_raw,
        "rows_claim_locs_nonempty_canonical": rows_claim_locs_nonempty_canonical,
        "rows_evidence_locs_nonempty_canonical": rows_evidence_locs_nonempty_canonical,
        "rows_both_locs_nonempty_canonical": rows_both_locs_nonempty_canonical,
        "rows_loc_disjoint_canonical": rows_loc_disjoint_canonical,
        "rows_org_like_span_present": rows_org_like_span_present,
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


def add_policy_fields(
    rows: list[dict[str, Any]],
    pred_field: str | None,
    gold_field: str | None,
    claim_field: str | None,
    evidence_field: str | None,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        pred = normalize_label(row.get(pred_field)) if pred_field else None
        gold = normalize_label(row.get(gold_field)) if gold_field else None
        claim = row.get(claim_field) if claim_field else None
        evidence = row.get(evidence_field) if evidence_field else None
        policy_fields = apply_shadow_policy_v2(
            pred,
            claim,
            evidence,
            policy_name=args.policy_name,
            use_aliases=not args.disable_built_in_aliases,
            organization_block_enabled=not args.disable_organization_block,
        )
        out = dict(row)
        out["_stage145_normalized_prediction"] = pred
        out["_stage145_normalized_gold"] = gold
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
    enriched = add_policy_fields(rows, pred_field, gold_field, claim_field, evidence_field, args)
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
    changed = [row for row in enriched if row.get("stage145_policy_triggered")]
    return metrics, changed, enriched, errors


def aggregate_metrics(
    rows: list[dict[str, Any]],
    file_metrics: list[dict[str, Any]],
    label_set: list[str],
    malformed_count: int,
) -> dict[str, Any]:
    aggregate = compute_metrics(
        rows,
        "_stage145_normalized_prediction",
        "_stage145_normalized_gold",
        None,
        None,
        label_set,
    )
    aggregate["n_malformed_rows"] = malformed_count
    aggregate["n_with_claim_evidence"] = sum(metrics.get("n_with_claim_evidence", 0) for metrics in file_metrics)
    gold_files = [metrics for metrics in file_metrics if metrics.get("has_usable_gold")]
    if not gold_files:
        for key in GOLD_METRIC_KEYS:
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
        return "STAGE145_TEXT_LOCATION_GUARD_V2_NO_VALID_INPUTS"
    if aggregate.get("n_files_with_usable_gold", 0) == 0:
        return "STAGE145_TEXT_LOCATION_GUARD_V2_COUNT_ONLY_NO_GOLD"
    delta_false_support = aggregate.get("delta_false_support")
    delta_false_ne = aggregate.get("delta_false_ne")
    min_delta_macro_f1 = aggregate.get("min_per_file_delta_macro_f1")
    if (
        delta_false_support is not None
        and delta_false_support < 0
        and delta_false_ne == 0
        and min_delta_macro_f1 is not None
        and min_delta_macro_f1 >= 0
    ):
        return "STAGE145_TEXT_LOCATION_GUARD_V2_SHADOW_CANDIDATE_PASS"
    if (
        delta_false_support is not None
        and delta_false_support < 0
        and delta_false_ne is not None
        and delta_false_ne <= 3
        and min_delta_macro_f1 is not None
        and min_delta_macro_f1 >= -0.01
    ):
        return "STAGE145_TEXT_LOCATION_GUARD_V2_SHADOW_CANDIDATE_MILD_TRADEOFF"
    if delta_false_support is not None and delta_false_support < 0:
        return "STAGE145_TEXT_LOCATION_GUARD_V2_SHADOW_MIXED"
    if aggregate.get("n_changed_total", 0) > 0:
        return "STAGE145_TEXT_LOCATION_GUARD_V2_HARM_OR_NO_FS_GAIN"
    return "STAGE145_TEXT_LOCATION_GUARD_V2_NO_EFFECT"


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
        if not gold_field or not fields.get("has_usable_gold"):
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
                        "support_rows_blocked_by_org_like_span": metrics[
                            "support_rows_blocked_by_org_like_span"
                        ],
                        "support_rows_with_canonical_loc_disjoint": metrics[
                            "support_rows_with_canonical_loc_disjoint"
                        ],
                    }
                )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
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


def format_md_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")


def markdown_table(rows: list[dict[str, Any]], columns: list[str], max_rows: int = 20) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for row in rows[:max_rows]:
        values = [format_md_value(row.get(column)) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    if len(rows) > max_rows:
        lines.append("| ... | ... | ... | ... |")
    return "\n".join(lines)


def build_markdown_report(report: dict[str, Any]) -> str:
    aggregate = report["aggregate_metrics"]
    lines = [
        "# Stage145-A Text Location Guard Shadow Analyzer v2",
        "",
        "## Summary decision",
        "",
        f"`{report['decision']}`",
        "",
        "## Policy definition",
        "",
        "Policy `text_loc_disjoint_v2`: for SUPPORT predictions only, extract location-like spans from claim and evidence text, canonicalize deterministic aliases, block rows with organization-like extracted spans, and shadow SUPPORT to NOT_ENTITLED only when both canonical location sets are non-empty and disjoint.",
        "",
        "## Refinements vs Stage142",
        "",
        "- Adds deterministic alias canonicalization for NYC/New York, LA/Los Angeles, and SF/San Francisco.",
        "- Adds conservative known lowercase location aliases without broad lowercase phrase extraction.",
        "- Adds an organization-like span block for labs, centers, institutes, universities, companies, agencies, and related suffixes.",
        "",
        "## Policy input safety",
        "",
        "The policy uses only claim text, evidence text, the original prediction label, deterministic alias rules, and deterministic organization-like suffix rules. Gold labels, diagnostic metadata, group fields, row identifiers, and file paths are not used by the policy.",
        "",
        "```json",
        json.dumps(report["policy_input_safety"], indent=2, sort_keys=True),
        "```",
        "",
        "## Aggregate metrics",
        "",
        markdown_table(
            [aggregate],
            [
                "n_valid_rows",
                "n_changed_total",
                "n_support_to_ne",
                "support_rows_with_canonical_loc_disjoint",
                "support_rows_blocked_by_org_like_span",
                "false_support_before",
                "false_support_after",
                "delta_false_support",
                "false_ne_before",
                "false_ne_after",
                "delta_false_ne",
                "macro_f1_before",
                "macro_f1_after",
                "delta_macro_f1",
            ],
        ),
        "",
        "## Output pointers",
        "",
        f"- Per-file metrics: `{report['per_file_metrics_path']}`",
        f"- Group audit: `{report['group_metrics_path']}`",
        f"- Changed examples: `{report['changed_examples_path']}`",
        "",
        "## Safety policy",
        "",
        "This script is shadow-only. It does not modify source predictions. It must not be interpreted as final model integration. It was created because Stage144-A exposed alias and organization-like false-positive risk.",
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
        insert_at = lines.index("## Safety policy") - 1
        lines.insert(insert_at, f"- Shadow predictions: `{report['shadow_predictions_path']}`")
    return "\n".join(lines)


def trim_output_row(row: dict[str, Any]) -> dict[str, Any]:
    internal = {
        "_stage145_source_file",
        "_stage145_line_number",
        "_stage145_normalized_prediction",
        "_stage145_normalized_gold",
    }
    out = {key: value for key, value in row.items() if key not in internal}
    out["stage145_source_file"] = row.get("_stage145_source_file")
    out["stage145_line_number"] = row.get("_stage145_line_number")
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
        all_changed.extend(trim_output_row(row) for row in changed)
        all_enriched.extend(enriched)
        if args.write_shadow_jsonl:
            all_shadow.extend(trim_output_row(row) for row in enriched)
        all_errors.extend(errors)
        rows_by_file[str(path)] = enriched

    file_metrics_by_path = {metrics["path"]: metrics for metrics in file_metrics}
    group_rows = group_metrics(rows_by_file, file_metrics_by_path, group_fields, label_set)
    aggregate = aggregate_metrics(all_enriched, file_metrics, label_set, len(all_errors))
    decision = choose_decision(aggregate)

    file_metrics_path = output_dir / "stage145_file_metrics.csv"
    aggregate_metrics_path = output_dir / "stage145_aggregate_metrics.json"
    group_metrics_path = output_dir / "stage145_group_metrics.csv"
    changed_examples_path = output_dir / "stage145_changed_examples.jsonl"
    shadow_predictions_path = output_dir / "stage145_shadow_predictions.jsonl"
    report_json_path = output_dir / "stage145_text_location_guard_shadow_v2_report.json"
    report_md_path = output_dir / "stage145_text_location_guard_shadow_v2_report.md"

    write_csv(file_metrics_path, file_metrics)
    write_json(aggregate_metrics_path, aggregate)
    write_csv(group_metrics_path, group_rows)
    write_jsonl(changed_examples_path, all_changed[: max(args.max_examples, 0)])
    if args.write_shadow_jsonl:
        write_jsonl(shadow_predictions_path, all_shadow)

    interpretation = (
        "Stage145-A is a diagnostic refinement of the Stage142 text-location shadow guard. "
        "It keeps the explicit-location mismatch signal while reducing known alias and "
        "organization-like false-positive risk observed in Stage144-A. Count-only inputs "
        "are supported when gold labels are unavailable."
    )
    refinement_summary = {
        "baseline": "Stage142 text_loc_disjoint",
        "policy": args.policy_name,
        "alias_canonicalization": not args.disable_built_in_aliases,
        "known_lowercase_alias_extraction": True,
        "organization_like_span_block": not args.disable_organization_block,
        "stage144a_risk_addressed": [
            "New York City vs New York and NYC aliases",
            "Los Angeles vs LA aliases",
            "San Francisco vs SF aliases",
            "organization-like spans such as labs, centers, institutes, and initiatives",
        ],
    }
    report = {
        "stage": STAGE,
        "decision": decision,
        "policy": {
            "name": args.policy_name,
            "description": "For SUPPORT predictions only, canonical disjoint non-empty extracted claim/evidence location sets shadow SUPPORT to NOT_ENTITLED unless organization-like spans are present.",
        },
        "input_files": [str(path) for path in input_files],
        "output_dir": str(output_dir),
        "aggregate_metrics": aggregate,
        "per_file_metrics_path": str(file_metrics_path),
        "group_metrics_path": str(group_metrics_path),
        "changed_examples_path": str(changed_examples_path),
        "shadow_predictions_path": str(shadow_predictions_path) if args.write_shadow_jsonl else None,
        "policy_input_safety": POLICY_INPUT_SAFETY,
        "refinement_summary": refinement_summary,
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
