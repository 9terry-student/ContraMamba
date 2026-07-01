"""Stage38-A integrated regression audit for Stage33-F + Stage36 + Stage37.

Diagnostic only. This script reads already-exported prediction files and
already-computed Stage34/Stage35/Stage36/Stage37 evaluator reports; it does
not train, evaluate the model, run Kaggle, or change any final classifier
logits/predictions/shadow behavior. It must not be used for training,
calibration, threshold selection, loss computation, or checkpoint selection.

It audits four things:
  1. clean dev final prediction identity (Stage33-F/36/37 must not touch the
     final classifier's predictions on the clean dev set)
  2. Stage34 held-out structured coverage preservation
  3. Stage35/Stage37 adversarial safety and recovery
  4. Stage36/Stage37 shadow-only blocker/recovery behavior confirmation

All inputs are optional except --run-name/--output-json/--output-md. Any
missing or unreadable input degrades its section to "missing" rather than
raising, and any unavailable metric is reported as null rather than raising.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

LABELS = ["REFUTE", "NOT_ENTITLED", "SUPPORT"]

PREDICTION_WRAPPER_KEYS = [
    "predictions",
    "records",
    "examples",
    "items",
    "data",
    "per_example",
    "per_example_predictions",
    "external_predictions",
]

FINAL_PREDICTION_KEY_PRIORITY = (
    "pred_final_label",
    "pred_label",
    "prediction",
    "predicted_label",
    "final_prediction",
    "current_final_label",
)

ID_KEY_PRIORITY = ("id", "pair_id", "example_id")

# Group name sets mirrored from scripts/evaluate_stage34_heldout_coverage.py
# and scripts/evaluate_stage35_adversarial_coverage.py so this audit script
# can classify rows the same way when only raw prediction exports (rather
# than pre-computed evaluator reports) are supplied. Read-only reuse; the
# evaluator scripts themselves are never imported or modified.
STAGE34_WHOLE_PART_SUPPORT_GROUPS = {
    "heldout_whole_to_part_support",
    "heldout_collection_to_member_support",
    "heldout_region_to_subregion_support",
    "heldout_category_to_subcategory_support",
    "heldout_role_to_specialized_role_support",
    "heldout_material_to_variant_support",
}
STAGE34_WHOLE_PART_NE_GROUPS = {
    "heldout_part_to_whole_not_entitled",
    "heldout_member_to_collection_not_entitled",
    "heldout_subregion_to_region_not_entitled",
    "heldout_subcategory_to_category_not_entitled",
    "heldout_specialized_role_to_role_not_entitled",
    "heldout_variant_to_material_not_entitled",
}

STAGE35_SUBSET_SUPPORT_GROUPS = {
    "adv_whole_to_part_support_verb_diverse",
    "adv_whole_to_part_support_fronted_modifier",
    "adv_whole_to_part_support_postnominal_modifier",
    "adv_whole_to_part_support_sentence_order_flip",
    "adv_passive_active_support",
    "adv_coordination_support",
    "adv_numeric_subset_support",
}
STAGE35_EXCEPTION_GROUPS = {
    "adv_all_except_subset_not_entitled",
    "adv_all_except_subset_support_for_nonexcluded",
    "adv_no_except_subset_support",
    "adv_no_except_nonexcluded_refute",
}
STAGE35_EXCEPTION_SUPPORT_GROUPS = {
    "adv_all_except_subset_support_for_nonexcluded",
    "adv_no_except_subset_support",
}
STAGE35_NUMERIC_SUPPORT_GROUPS = {"adv_numeric_subset_support"}


# ---------------------------------------------------------------------------
# Generic, crash-proof IO helpers
# ---------------------------------------------------------------------------


def safe_read_json(path_str: str | None) -> Any | None:
    """Load JSON from path_str. Returns None (never raises) if missing/unreadable."""
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists() or not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None


def safe_read_jsonl_or_json(path_str: str | None) -> Any | None:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists() or not path.is_file():
        return None
    try:
        if path.suffix == ".jsonl":
            rows = []
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            return rows
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None


def as_rows(data: Any) -> list[dict[str, Any]] | None:
    """Normalize a loaded JSON/JSONL payload into a list of row dicts.

    Handles: bare list, dict-with-predictions/records/items/etc wrapper. Any
    other shape (e.g. a report-style dict) returns None rather than raising.
    """
    if data is None:
        return None
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for key in PREDICTION_WRAPPER_KEYS:
            value = data.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    return None


def as_report(data: Any) -> dict[str, Any] | None:
    """Normalize a loaded JSON payload into a flat report dict.

    Accepts a bare report dict directly, or one nested under a "report" key.
    Returns None if the payload does not look like a report-style dict.
    """
    if data is None:
        return None
    if isinstance(data, dict):
        nested = data.get("report")
        if isinstance(nested, dict):
            return nested
        return data
    return None


def load_prediction_rows(path_str: str | None) -> list[dict[str, Any]] | None:
    data = safe_read_jsonl_or_json(path_str)
    return as_rows(data)


def load_report(path_str: str | None) -> dict[str, Any] | None:
    data = safe_read_json(path_str)
    return as_report(data)


def get_nested(report: dict[str, Any] | None, *path: str) -> Any:
    """Safely walk a chain of dict keys, returning None if anything is missing."""
    if report is None:
        return None
    current: Any = report
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def first_non_null(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


# ---------------------------------------------------------------------------
# Label / row helpers (mirrors normalize_label used by Stage34/35 evaluators)
# ---------------------------------------------------------------------------


def normalize_label_safe(raw: Any) -> str | None:
    if raw is None or raw == "":
        return None
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
    return mapping.get(key)


def get_row_id(row: dict[str, Any], index: int) -> str:
    for key in ID_KEY_PRIORITY:
        value = row.get(key)
        if value not in (None, ""):
            return f"{key}:{value}"
    return f"__index__:{index}"


def get_final_prediction_raw(row: dict[str, Any]) -> Any:
    for key in FINAL_PREDICTION_KEY_PRIORITY:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def get_gold_label(row: dict[str, Any]) -> str | None:
    value = first_non_null(
        row.get("gold_final_label"), row.get("gold_label"), row.get("final_label"), row.get("label")
    )
    return normalize_label_safe(value)


def get_current_label(row: dict[str, Any]) -> str | None:
    return normalize_label_safe(get_final_prediction_raw(row))


def resolve_shadow_label(row: dict[str, Any]) -> str | None:
    """Priority: stage37_final > stage36_final > stage32 > stage33 conditional."""
    value = first_non_null(
        row.get("stage37_final_shadow_label"),
        row.get("stage36_final_shadow_label"),
        row.get("stage32_shadow_label"),
        row.get("stage33_conditional_shadow_label"),
    )
    return normalize_label_safe(value)


def get_group(row: dict[str, Any]) -> str:
    value = first_non_null(
        row.get("group"),
        row.get("intervention_type"),
        row.get("normalized_intervention"),
        row.get("primary_failure_type"),
    )
    return "UNKNOWN" if value is None else str(value)


def prediction_metrics_safe(golds: list[str], preds: list[str]) -> dict[str, Any] | None:
    """Accuracy + macro-F1 over rows where both gold and pred normalized cleanly."""
    pairs = [(g, p) for g, p in zip(golds, preds) if g is not None and p is not None]
    if not pairs:
        return None
    accuracy = sum(g == p for g, p in pairs) / len(pairs)
    f1s = []
    for label in LABELS:
        tp = sum(g == label and p == label for g, p in pairs)
        fp = sum(g != label and p == label for g, p in pairs)
        fn = sum(g == label and p != label for g, p in pairs)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        f1s.append(f1)
    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(sum(f1s) / len(f1s), 4),
        "n": len(pairs),
    }


# ---------------------------------------------------------------------------
# Part B: dev final prediction identity
# ---------------------------------------------------------------------------


def dev_identity_audit(baseline_path: str | None, candidate_path: str | None) -> dict[str, Any]:
    section: dict[str, Any] = {
        "status": "missing",
        "dev_row_count_baseline": None,
        "dev_row_count_candidate": None,
        "dev_overlap_count": None,
        "dev_final_prediction_mismatch_count": None,
        "dev_final_prediction_mismatch_rate": None,
        "dev_final_prediction_identity_pass": None,
        "dev_baseline_prediction_counts": None,
        "dev_candidate_prediction_counts": None,
        "dev_first_mismatches": [],
    }
    baseline_rows = load_prediction_rows(baseline_path)
    candidate_rows = load_prediction_rows(candidate_path)
    if baseline_rows is None and candidate_rows is None:
        section["note"] = "Both --dev-baseline-predictions and --dev-candidate-predictions are missing or unreadable."
        return section
    if baseline_rows is None or candidate_rows is None:
        section["status"] = "partial"
        section["dev_row_count_baseline"] = None if baseline_rows is None else len(baseline_rows)
        section["dev_row_count_candidate"] = None if candidate_rows is None else len(candidate_rows)
        section["note"] = "Only one of the two dev prediction files was readable; identity cannot be compared."
        return section

    baseline_by_id: dict[str, Any] = {}
    for index, row in enumerate(baseline_rows):
        baseline_by_id[get_row_id(row, index)] = get_final_prediction_raw(row)
    candidate_by_id: dict[str, Any] = {}
    for index, row in enumerate(candidate_rows):
        candidate_by_id[get_row_id(row, index)] = get_final_prediction_raw(row)

    overlap_ids = [key for key in baseline_by_id if key in candidate_by_id]
    mismatches: list[dict[str, Any]] = []
    baseline_counts: dict[str, int] = {}
    candidate_counts: dict[str, int] = {}
    for key in overlap_ids:
        base_val = baseline_by_id[key]
        cand_val = candidate_by_id[key]
        base_norm = normalize_label_safe(base_val)
        cand_norm = normalize_label_safe(cand_val)
        base_display = base_norm if base_norm is not None else base_val
        cand_display = cand_norm if cand_norm is not None else cand_val
        baseline_counts[str(base_display)] = baseline_counts.get(str(base_display), 0) + 1
        candidate_counts[str(cand_display)] = candidate_counts.get(str(cand_display), 0) + 1
        is_mismatch = base_display != cand_display
        if is_mismatch and len(mismatches) < 20:
            mismatches.append({
                "row_id": key,
                "baseline_final_prediction": base_display,
                "candidate_final_prediction": cand_display,
            })
        elif is_mismatch:
            pass

    mismatch_count = sum(
        1 for key in overlap_ids
        if (normalize_label_safe(baseline_by_id[key]) or baseline_by_id[key])
        != (normalize_label_safe(candidate_by_id[key]) or candidate_by_id[key])
    )

    section.update({
        "status": "ok",
        "dev_row_count_baseline": len(baseline_rows),
        "dev_row_count_candidate": len(candidate_rows),
        "dev_overlap_count": len(overlap_ids),
        "dev_final_prediction_mismatch_count": mismatch_count,
        "dev_final_prediction_mismatch_rate": (
            round(mismatch_count / len(overlap_ids), 4) if overlap_ids else None
        ),
        "dev_final_prediction_identity_pass": (
            mismatch_count == 0 if overlap_ids else None
        ),
        "dev_baseline_prediction_counts": baseline_counts,
        "dev_candidate_prediction_counts": candidate_counts,
        "dev_first_mismatches": mismatches,
    })
    return section


# ---------------------------------------------------------------------------
# Part C: Stage34 preservation audit
# ---------------------------------------------------------------------------


def _stage34_metrics_from_report(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if report is None:
        return None
    return {
        "shadow_accuracy": get_nested(report, "shadow_metrics", "accuracy"),
        "shadow_macro_f1": get_nested(report, "shadow_metrics", "macro_f1"),
        "support_shadow_support": report.get("heldout_support_shadow_support"),
        "whole_to_part_support_recovered": report.get("heldout_whole_to_part_support_recovered"),
        "part_to_whole_ne_preserved": report.get("heldout_part_to_whole_ne_preserved"),
        "overclaim_to_support": report.get("heldout_overclaim_to_support"),
        "refute_to_support": report.get("heldout_refute_to_support"),
        "support_to_refute": report.get("heldout_support_to_refute"),
    }


def _stage34_metrics_from_rows(rows: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    if not rows:
        return None
    golds = [get_gold_label(row) for row in rows]
    shadow = [resolve_shadow_label(row) for row in rows]
    metrics = prediction_metrics_safe(
        [g for g in golds if g is not None], [s for s in shadow if s is not None]
    )
    support_shadow_support = 0
    whole_to_part_recovered = 0
    part_to_whole_ne_preserved = 0
    overclaim_to_support = 0
    refute_to_support = 0
    support_to_refute = 0
    for row, gold, shadow_label in zip(rows, golds, shadow):
        group = get_group(row)
        if gold == "SUPPORT" and shadow_label == "SUPPORT":
            support_shadow_support += 1
        if gold == "NOT_ENTITLED" and shadow_label == "SUPPORT":
            overclaim_to_support += 1
        if gold == "REFUTE" and shadow_label == "SUPPORT":
            refute_to_support += 1
        if gold == "SUPPORT" and shadow_label == "REFUTE":
            support_to_refute += 1
        if group in STAGE34_WHOLE_PART_SUPPORT_GROUPS and shadow_label == "SUPPORT":
            whole_to_part_recovered += 1
        if group in STAGE34_WHOLE_PART_NE_GROUPS and gold == "NOT_ENTITLED" and shadow_label == "NOT_ENTITLED":
            part_to_whole_ne_preserved += 1
    return {
        "shadow_accuracy": metrics["accuracy"] if metrics else None,
        "shadow_macro_f1": metrics["macro_f1"] if metrics else None,
        "support_shadow_support": support_shadow_support,
        "whole_to_part_support_recovered": whole_to_part_recovered,
        "part_to_whole_ne_preserved": part_to_whole_ne_preserved,
        "overclaim_to_support": overclaim_to_support,
        "refute_to_support": refute_to_support,
        "support_to_refute": support_to_refute,
    }


def stage34_preservation_audit(args: argparse.Namespace) -> dict[str, Any]:
    section: dict[str, Any] = {
        "status": "missing",
        "stage34_baseline_shadow_accuracy": None,
        "stage34_candidate_shadow_accuracy": None,
        "stage34_baseline_shadow_macro_f1": None,
        "stage34_candidate_shadow_macro_f1": None,
        "stage34_shadow_macro_f1_delta": None,
        "stage34_support_shadow_support": None,
        "stage34_whole_to_part_support_recovered": None,
        "stage34_part_to_whole_ne_preserved": None,
        "stage34_overclaim_to_support": None,
        "stage34_refute_to_support": None,
        "stage34_support_to_refute": None,
        "stage34_preservation_pass": None,
    }

    baseline_report = load_report(args.stage34_baseline_report)
    candidate_report = load_report(args.stage34_candidate_report)
    baseline = _stage34_metrics_from_report(baseline_report)
    candidate = _stage34_metrics_from_report(candidate_report)
    if baseline is None:
        baseline = _stage34_metrics_from_rows(load_prediction_rows(args.stage34_baseline_predictions))
    if candidate is None:
        candidate = _stage34_metrics_from_rows(load_prediction_rows(args.stage34_candidate_predictions))

    if baseline is None and candidate is None:
        section["note"] = "No Stage34 baseline or candidate report/predictions were supplied or readable."
        return section

    if baseline is not None:
        section["stage34_baseline_shadow_accuracy"] = baseline["shadow_accuracy"]
        section["stage34_baseline_shadow_macro_f1"] = baseline["shadow_macro_f1"]

    if candidate is None:
        section["status"] = "baseline_only"
        section["note"] = "Only Stage34 baseline metrics are available; candidate is missing."
        section["stage34_support_shadow_support"] = baseline["support_shadow_support"] if baseline else None
        section["stage34_whole_to_part_support_recovered"] = (
            baseline["whole_to_part_support_recovered"] if baseline else None
        )
        section["stage34_part_to_whole_ne_preserved"] = (
            baseline["part_to_whole_ne_preserved"] if baseline else None
        )
        section["stage34_overclaim_to_support"] = baseline["overclaim_to_support"] if baseline else None
        section["stage34_refute_to_support"] = baseline["refute_to_support"] if baseline else None
        section["stage34_support_to_refute"] = baseline["support_to_refute"] if baseline else None
        section["stage34_preservation_pass"] = None
        return section

    section["status"] = "ok"
    section["stage34_candidate_shadow_accuracy"] = candidate["shadow_accuracy"]
    section["stage34_candidate_shadow_macro_f1"] = candidate["shadow_macro_f1"]
    if baseline and baseline["shadow_macro_f1"] is not None and candidate["shadow_macro_f1"] is not None:
        section["stage34_shadow_macro_f1_delta"] = round(
            candidate["shadow_macro_f1"] - baseline["shadow_macro_f1"], 4
        )
    section["stage34_support_shadow_support"] = candidate["support_shadow_support"]
    section["stage34_whole_to_part_support_recovered"] = candidate["whole_to_part_support_recovered"]
    section["stage34_part_to_whole_ne_preserved"] = candidate["part_to_whole_ne_preserved"]
    section["stage34_overclaim_to_support"] = candidate["overclaim_to_support"]
    section["stage34_refute_to_support"] = candidate["refute_to_support"]
    section["stage34_support_to_refute"] = candidate["support_to_refute"]

    macro_f1 = candidate["shadow_macro_f1"]
    overclaim = candidate["overclaim_to_support"]
    refute_to_support = candidate["refute_to_support"]
    support_to_refute = candidate["support_to_refute"]
    whole_to_part = candidate["whole_to_part_support_recovered"]
    baseline_whole_to_part = baseline["whole_to_part_support_recovered"] if baseline else None

    checks_available = all(
        value is not None for value in (macro_f1, overclaim, refute_to_support)
    )
    if not checks_available:
        section["stage34_preservation_pass"] = None
    else:
        whole_to_part_ok = True
        if whole_to_part is not None and baseline_whole_to_part is not None and baseline_whole_to_part > 0:
            whole_to_part_ok = whole_to_part >= 0.8 * baseline_whole_to_part
        support_to_refute_ok = support_to_refute is None or support_to_refute <= 1
        section["stage34_preservation_pass"] = bool(
            macro_f1 >= 0.94
            and overclaim == 0
            and refute_to_support == 0
            and support_to_refute_ok
            and whole_to_part_ok
        )
    return section


# ---------------------------------------------------------------------------
# Part D: Stage35/37 adversarial audit
# ---------------------------------------------------------------------------


def _adv_report_source(args: argparse.Namespace) -> tuple[dict[str, Any] | None, str | None]:
    """Prefer the most complete report: Stage37 > Stage36 > Stage35 baseline."""
    stage37 = load_report(args.stage37_report)
    if stage37 is not None:
        return stage37, "stage37_report"
    stage36 = load_report(args.stage36_report)
    if stage36 is not None:
        return stage36, "stage36_report"
    stage35 = load_report(args.stage35_baseline_report)
    if stage35 is not None:
        return stage35, "stage35_baseline_report"
    return None, None


def _adv_metrics_from_rows(rows: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    if not rows:
        return None
    golds = [get_gold_label(row) for row in rows]
    currents = [get_current_label(row) for row in rows]
    shadow = [resolve_shadow_label(row) for row in rows]
    metrics = prediction_metrics_safe(
        [g for g in golds if g is not None], [s for s in shadow if s is not None]
    )
    current_metrics = prediction_metrics_safe(
        [g for g in golds if g is not None], [c for c in currents if c is not None]
    )
    counters = {
        "adv_overclaim_to_support": 0,
        "adv_exception_to_support_error": 0,
        "adv_location_scope_to_support_error": 0,
        "adv_temporal_scope_to_support_error": 0,
        "adv_refute_to_support": 0,
        "adv_support_to_refute": 0,
        "adv_support_shadow_support": 0,
        "adv_subset_support_recovered": 0,
        "adv_numeric_support_recovered": 0,
        "adv_exception_support_recovered": 0,
    }
    for row, gold, shadow_label in zip(rows, golds, shadow):
        group = get_group(row)
        if gold == "NOT_ENTITLED" and shadow_label == "SUPPORT":
            counters["adv_overclaim_to_support"] += 1
        if gold == "REFUTE" and shadow_label == "SUPPORT":
            counters["adv_refute_to_support"] += 1
        if gold == "SUPPORT" and shadow_label == "REFUTE":
            counters["adv_support_to_refute"] += 1
        if group in STAGE35_EXCEPTION_GROUPS and gold != "SUPPORT" and shadow_label == "SUPPORT":
            counters["adv_exception_to_support_error"] += 1
        if group == "adv_location_scope_not_entitled" and shadow_label == "SUPPORT":
            counters["adv_location_scope_to_support_error"] += 1
        if group == "adv_temporal_scope_not_entitled" and shadow_label == "SUPPORT":
            counters["adv_temporal_scope_to_support_error"] += 1
        if gold == "SUPPORT" and shadow_label == "SUPPORT":
            counters["adv_support_shadow_support"] += 1
        if group in STAGE35_SUBSET_SUPPORT_GROUPS and shadow_label == "SUPPORT":
            counters["adv_subset_support_recovered"] += 1
        if group in STAGE35_NUMERIC_SUPPORT_GROUPS and shadow_label == "SUPPORT":
            counters["adv_numeric_support_recovered"] += 1
        if group in STAGE35_EXCEPTION_SUPPORT_GROUPS and shadow_label == "SUPPORT":
            counters["adv_exception_support_recovered"] += 1
    scope_errors = (
        counters["adv_exception_to_support_error"]
        + counters["adv_temporal_scope_to_support_error"]
        + counters["adv_location_scope_to_support_error"]
    )
    scope_safety = "safe" if scope_errors == 0 else ("unsafe" if scope_errors >= 5 else "mixed")
    return {
        "shadow_accuracy": metrics["accuracy"] if metrics else None,
        "shadow_macro_f1": metrics["macro_f1"] if metrics else None,
        "delta_macro_f1": (
            round(metrics["macro_f1"] - current_metrics["macro_f1"], 4)
            if metrics and current_metrics else None
        ),
        "stage35_scope_safety": scope_safety,
        "stage35_reverse_overclaim_handling": None,
        **counters,
    }


def stage35_37_adversarial_audit(args: argparse.Namespace) -> dict[str, Any]:
    section: dict[str, Any] = {
        "status": "missing",
        "source": None,
        "stage37_shadow_accuracy": None,
        "stage37_shadow_macro_f1": None,
        "stage37_delta_macro_f1": None,
        "adv_support_shadow_support": None,
        "adv_subset_support_recovered": None,
        "adv_numeric_support_recovered": None,
        "adv_exception_support_recovered": None,
        "adv_overclaim_to_support": None,
        "adv_exception_to_support_error": None,
        "adv_location_scope_to_support_error": None,
        "adv_temporal_scope_to_support_error": None,
        "adv_refute_to_support": None,
        "adv_support_to_refute": None,
        "stage35_scope_safety": None,
        "stage35_reverse_overclaim_handling": None,
        "adversarial_safety_pass": None,
        "adversarial_recovery_pass": None,
    }

    report, source = _adv_report_source(args)
    if report is not None:
        section["status"] = "ok"
        section["source"] = source
        section["stage37_shadow_accuracy"] = get_nested(report, "shadow_metrics", "accuracy")
        section["stage37_shadow_macro_f1"] = get_nested(report, "shadow_metrics", "macro_f1")
        section["stage37_delta_macro_f1"] = get_nested(report, "delta", "shadow_minus_current_macro_f1")
        section["adv_support_shadow_support"] = report.get("adv_support_shadow_support")
        section["adv_subset_support_recovered"] = report.get("adv_subset_support_recovered")
        section["adv_numeric_support_recovered"] = report.get("adv_numeric_support_recovered")
        section["adv_exception_support_recovered"] = report.get("adv_exception_support_recovered")
        section["adv_overclaim_to_support"] = report.get("adv_overclaim_to_support")
        section["adv_exception_to_support_error"] = report.get("adv_exception_to_support_error")
        section["adv_location_scope_to_support_error"] = report.get("adv_location_scope_to_support_error")
        section["adv_temporal_scope_to_support_error"] = report.get("adv_temporal_scope_to_support_error")
        section["adv_refute_to_support"] = report.get("adv_refute_to_support")
        section["adv_support_to_refute"] = report.get("adv_support_to_refute")
        section["stage35_scope_safety"] = report.get("stage35_scope_safety")
        section["stage35_reverse_overclaim_handling"] = report.get("stage35_reverse_overclaim_handling")
    else:
        rows = load_prediction_rows(args.stage37_predictions)
        source_label = "stage37_predictions"
        if rows is None:
            rows = load_prediction_rows(args.stage35_baseline_predictions)
            source_label = "stage35_baseline_predictions"
        computed = _adv_metrics_from_rows(rows)
        if computed is None:
            section["note"] = (
                "No Stage35/36/37 report or raw predictions were supplied or readable."
            )
            return section
        section["status"] = "ok"
        section["source"] = source_label
        section["stage37_shadow_accuracy"] = computed["shadow_accuracy"]
        section["stage37_shadow_macro_f1"] = computed["shadow_macro_f1"]
        section["stage37_delta_macro_f1"] = computed["delta_macro_f1"]
        section["adv_support_shadow_support"] = computed["adv_support_shadow_support"]
        section["adv_subset_support_recovered"] = computed["adv_subset_support_recovered"]
        section["adv_numeric_support_recovered"] = computed["adv_numeric_support_recovered"]
        section["adv_exception_support_recovered"] = computed["adv_exception_support_recovered"]
        section["adv_overclaim_to_support"] = computed["adv_overclaim_to_support"]
        section["adv_exception_to_support_error"] = computed["adv_exception_to_support_error"]
        section["adv_location_scope_to_support_error"] = computed["adv_location_scope_to_support_error"]
        section["adv_temporal_scope_to_support_error"] = computed["adv_temporal_scope_to_support_error"]
        section["adv_refute_to_support"] = computed["adv_refute_to_support"]
        section["adv_support_to_refute"] = computed["adv_support_to_refute"]
        section["stage35_scope_safety"] = computed["stage35_scope_safety"]
        section["stage35_reverse_overclaim_handling"] = computed["stage35_reverse_overclaim_handling"]

    safety_fields = (
        section["adv_overclaim_to_support"],
        section["adv_exception_to_support_error"],
        section["adv_location_scope_to_support_error"],
        section["adv_temporal_scope_to_support_error"],
        section["adv_refute_to_support"],
    )
    if all(value is not None for value in safety_fields):
        section["adversarial_safety_pass"] = all(value == 0 for value in safety_fields)

    recovery_fields = (
        section["adv_support_shadow_support"],
        section["adv_subset_support_recovered"],
        section["adv_numeric_support_recovered"],
        section["stage37_shadow_macro_f1"],
    )
    if all(value is not None for value in recovery_fields):
        section["adversarial_recovery_pass"] = bool(
            section["adv_support_shadow_support"] >= 140
            and section["adv_subset_support_recovered"] >= 95
            and section["adv_numeric_support_recovered"] >= 20
            and section["stage37_shadow_macro_f1"] >= 0.70
        )
    return section


# ---------------------------------------------------------------------------
# Part E: Stage36/37 behavior audit
# ---------------------------------------------------------------------------


def stage36_37_behavior_audit(args: argparse.Namespace) -> dict[str, Any]:
    section: dict[str, Any] = {
        "status": "missing",
        "stage36_support_blocker_fired_count": None,
        "stage36_exception_blocker_fired_count": None,
        "stage36_not_all_blocker_fired_count": None,
        "stage36_location_scope_blocker_fired_count": None,
        "stage36_blocked_support_to_ne_count": None,
        "stage37_safe_recovery_fired_count": None,
        "stage37_no_except_included_subset_fired_count": None,
        "stage37_coordination_universal_subset_fired_count": None,
        "stage37_numeric_universal_subset_fired_count": None,
        "stage37_recovered_from_not_entitled_count": None,
        "stage37_recovered_from_refute_count": None,
        "stage37_blocked_by_stage36_count": None,
        "stage36_37_behavior_pass": None,
    }
    stage37_report = load_report(args.stage37_report)
    stage36_report = load_report(args.stage36_report)
    report = stage37_report or stage36_report
    if report is None:
        section["note"] = "No Stage36/Stage37 report was supplied or readable."
        return section

    section["status"] = "ok"
    section["stage36_support_blocker_fired_count"] = report.get("stage36_support_blocker_fired_count")
    section["stage36_exception_blocker_fired_count"] = report.get("stage36_exception_blocker_fired_count")
    section["stage36_not_all_blocker_fired_count"] = report.get("stage36_not_all_blocker_fired_count")
    section["stage36_location_scope_blocker_fired_count"] = report.get(
        "stage36_location_scope_blocker_fired_count"
    )
    section["stage36_blocked_support_to_ne_count"] = report.get("stage36_blocked_support_to_ne_count")
    section["stage37_safe_recovery_fired_count"] = report.get("stage37_safe_recovery_fired_count")
    section["stage37_no_except_included_subset_fired_count"] = report.get(
        "stage37_no_except_included_subset_fired_count"
    )
    section["stage37_coordination_universal_subset_fired_count"] = report.get(
        "stage37_coordination_universal_subset_fired_count"
    )
    section["stage37_numeric_universal_subset_fired_count"] = report.get(
        "stage37_numeric_universal_subset_fired_count"
    )
    section["stage37_recovered_from_not_entitled_count"] = report.get(
        "stage37_recovered_from_not_entitled_count"
    )
    section["stage37_recovered_from_refute_count"] = report.get("stage37_recovered_from_refute_count")
    section["stage37_blocked_by_stage36_count"] = report.get("stage37_blocked_by_stage36_count")

    stage36_fired = section["stage36_support_blocker_fired_count"]
    stage37_fired = section["stage37_safe_recovery_fired_count"]
    recovered_from_refute = section["stage37_recovered_from_refute_count"]
    blocked_by_stage36 = section["stage37_blocked_by_stage36_count"]

    if stage36_fired is not None and stage37_fired is not None and recovered_from_refute is not None:
        no_override_proof = blocked_by_stage36 is not None and blocked_by_stage36 > 0
        section["stage36_37_behavior_pass"] = bool(
            stage36_fired > 0
            and stage37_fired > 0
            and recovered_from_refute == 0
            and no_override_proof
        )
    return section


# ---------------------------------------------------------------------------
# Part F: integrated decision
# ---------------------------------------------------------------------------


def integrated_decision(
    dev_identity: dict[str, Any],
    stage34: dict[str, Any],
    adversarial: dict[str, Any],
    behavior: dict[str, Any],
) -> dict[str, Any]:
    dev_pass = dev_identity.get("dev_final_prediction_identity_pass")
    dev_mismatch_count = dev_identity.get("dev_final_prediction_mismatch_count")

    if dev_mismatch_count is not None and dev_mismatch_count > 0:
        return {
            "label": "STAGE38A_DEV_FINAL_CHANGED",
            "reason": (
                f"{dev_mismatch_count} dev row(s) had a different final classifier "
                "prediction between baseline and candidate exports."
            ),
        }

    stage34_pass = stage34.get("stage34_preservation_pass")
    if stage34_pass is False:
        return {
            "label": "STAGE38A_STAGE34_REGRESSION",
            "reason": (
                "Stage34 candidate held-out shadow metrics regressed below threshold "
                "or introduced an overclaim/refute/support-to-refute safety error."
            ),
        }

    safety_pass = adversarial.get("adversarial_safety_pass")
    if safety_pass is False:
        return {
            "label": "STAGE38A_ADVERSARIAL_SAFETY_REGRESSION",
            "reason": (
                "One or more unsafe-SUPPORT counters (overclaim, exception, "
                "location-scope, temporal-scope, or refute-to-support) is nonzero."
            ),
        }

    recovery_pass = adversarial.get("adversarial_recovery_pass")
    if safety_pass is True and recovery_pass is False:
        return {
            "label": "STAGE38A_RECOVERY_TOO_WEAK",
            "reason": (
                "Adversarial safety holds, but Stage37 SUPPORT recovery counters fall "
                "below the required thresholds."
            ),
        }

    behavior_pass = behavior.get("stage36_37_behavior_pass")

    required_sections_present = (
        dev_pass is not None
        and safety_pass is not None
        and recovery_pass is not None
        and behavior_pass is not None
    )
    stage34_contradicted = stage34_pass is False
    stage34_ok_or_unsupplied = stage34_pass is True or stage34.get("status") in ("missing",)

    if (
        required_sections_present
        and dev_pass
        and stage34_ok_or_unsupplied
        and not stage34_contradicted
        and safety_pass
        and recovery_pass
        and behavior_pass
    ):
        return {
            "label": "STAGE38A_INTEGRATED_REGRESSION_PASS",
            "reason": (
                "Dev final predictions are identical, Stage34 preservation is intact "
                "or not contradicted, Stage35/37 adversarial safety and recovery pass "
                "thresholds, and Stage36/37 shadow-only behavior is confirmed."
            ),
        }

    return {
        "label": "STAGE38A_INCOMPLETE_AUDIT",
        "reason": (
            "Required input files are missing or their sections could not be fully "
            "evaluated, and no hard failure was observed."
        ),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def collect_missing_inputs(args: argparse.Namespace) -> list[str]:
    optional_flag_names = [
        "dev_baseline_predictions",
        "dev_candidate_predictions",
        "stage34_baseline_report",
        "stage34_candidate_report",
        "stage34_baseline_predictions",
        "stage34_candidate_predictions",
        "stage35_baseline_report",
        "stage36_report",
        "stage37_report",
        "stage35_baseline_predictions",
        "stage37_predictions",
    ]
    missing = []
    for name in optional_flag_names:
        value = getattr(args, name, None)
        flag = "--" + name.replace("_", "-")
        if not value:
            missing.append(flag)
            continue
        path = Path(value)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.exists():
            missing.append(flag)
    return missing


def build_key_metrics(
    dev_identity: dict[str, Any],
    stage34: dict[str, Any],
    adversarial: dict[str, Any],
    behavior: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dev_final_prediction_mismatch_count": dev_identity.get("dev_final_prediction_mismatch_count"),
        "dev_final_prediction_mismatch_rate": dev_identity.get("dev_final_prediction_mismatch_rate"),
        "stage34_candidate_shadow_macro_f1": stage34.get("stage34_candidate_shadow_macro_f1"),
        "stage34_shadow_macro_f1_delta": stage34.get("stage34_shadow_macro_f1_delta"),
        "stage37_shadow_macro_f1": adversarial.get("stage37_shadow_macro_f1"),
        "adv_support_shadow_support": adversarial.get("adv_support_shadow_support"),
        "adv_subset_support_recovered": adversarial.get("adv_subset_support_recovered"),
        "adv_numeric_support_recovered": adversarial.get("adv_numeric_support_recovered"),
        "stage36_support_blocker_fired_count": behavior.get("stage36_support_blocker_fired_count"),
        "stage37_safe_recovery_fired_count": behavior.get("stage37_safe_recovery_fired_count"),
    }


def build_risks(
    dev_identity: dict[str, Any],
    stage34: dict[str, Any],
    adversarial: dict[str, Any],
    behavior: dict[str, Any],
    missing_inputs: list[str],
) -> list[str]:
    risks = []
    if missing_inputs:
        risks.append(
            "Some optional input files were not supplied or not found: "
            + ", ".join(missing_inputs)
            + ". Sections built from missing inputs cannot contribute to a pass verdict."
        )
    if dev_identity.get("status") != "ok":
        risks.append(
            "Dev final prediction identity could not be verified because baseline "
            "and/or candidate dev prediction files were not both readable."
        )
    if stage34.get("status") == "missing":
        risks.append(
            "Stage34 held-out preservation was not audited; a regression there "
            "would not be caught by this run."
        )
    if stage34.get("status") == "baseline_only":
        risks.append(
            "Only Stage34 baseline metrics are available; candidate preservation "
            "is unverified."
        )
    if adversarial.get("status") != "ok":
        risks.append(
            "Stage35/37 adversarial safety and recovery could not be audited; "
            "no Stage35/36/37 report or raw predictions were readable."
        )
    if behavior.get("status") != "ok":
        risks.append(
            "Stage36/37 shadow-only blocker/recovery behavior could not be "
            "confirmed from the supplied reports."
        )
    if adversarial.get("stage37_recovered_from_refute_count", 0):
        risks.append("Stage37 recovered at least one row from REFUTE, which requires extra scrutiny.")
    if not risks:
        risks.append(
            "This audit only inspects already-exported diagnostic fields and reports; "
            "it cannot detect regressions in prediction pipelines that never populate "
            "stage3x_* fields."
        )
    return risks


def build_recommendation(decision_label: str) -> str:
    if decision_label == "STAGE38A_INTEGRATED_REGRESSION_PASS":
        return (
            "Stage33-F + Stage36 + Stage37 is safe as a shadow diagnostic owner; "
            "final composer still requires separate explicit opt-in validation."
        )
    if decision_label == "STAGE38A_DEV_FINAL_CHANGED":
        return "Reject as shadow-only regression."
    if decision_label == "STAGE38A_STAGE34_REGRESSION":
        return "Fix Stage37 interaction with Stage34 before continuing."
    if decision_label == "STAGE38A_ADVERSARIAL_SAFETY_REGRESSION":
        return "Reject Stage37 recovery rules or tighten hazards."
    if decision_label == "STAGE38A_RECOVERY_TOO_WEAK":
        return (
            "Safety holds, but Stage37 recovery is too weak to justify promoting it; "
            "keep as diagnostic-only until recovery thresholds are met."
        )
    return (
        "Supply the missing Stage34/35/36/37 reports or prediction exports and re-run "
        "this audit before drawing any conclusion."
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    dev = report["sections"]["dev_identity"]
    stage34 = report["sections"]["stage34_preservation"]
    adversarial = report["sections"]["stage35_37_adversarial"]
    behavior = report["sections"]["stage36_37_behavior"]
    pass_flags = report["pass_flags"]

    def fmt(value: Any) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    lines = [
        "# Stage38-A Integrated Regression Audit (Stage33-F + Stage36 + Stage37)",
        "",
        f"- Run: `{report['run_name']}`",
        f"- Decision: `{report['decision']['label']}`",
        f"- Decision reason: {report['decision']['reason']}",
        "",
        "## Input Files",
        "| Input | Status |",
        "|---|---|",
    ]
    input_status = {
        "--dev-baseline-predictions": dev.get("dev_row_count_baseline") is not None,
        "--dev-candidate-predictions": dev.get("dev_row_count_candidate") is not None,
        "--stage34 (report or predictions)": stage34.get("status") != "missing",
        "--stage35/36/37 (report or predictions)": adversarial.get("status") != "missing",
        "--stage36/37 report": behavior.get("status") != "missing",
    }
    for name, present in input_status.items():
        lines.append(f"| {name} | {'present' if present else 'missing'} |")
    if report["missing_inputs"]:
        lines.extend([
            "",
            "Missing/unreadable optional inputs: "
            + ", ".join(f"`{item}`" for item in report["missing_inputs"]),
        ])

    lines.extend([
        "",
        "## 1. Dev Final Prediction Identity",
        f"- Status: `{dev.get('status')}`",
        f"- Baseline rows: {fmt(dev.get('dev_row_count_baseline'))}",
        f"- Candidate rows: {fmt(dev.get('dev_row_count_candidate'))}",
        f"- Overlap rows: {fmt(dev.get('dev_overlap_count'))}",
        f"- Mismatch count: {fmt(dev.get('dev_final_prediction_mismatch_count'))}",
        f"- Mismatch rate: {fmt(dev.get('dev_final_prediction_mismatch_rate'))}",
        f"- Identity pass: {fmt(dev.get('dev_final_prediction_identity_pass'))}",
    ])
    if dev.get("dev_first_mismatches"):
        lines.extend([
            "",
            "| Row ID | Baseline | Candidate |",
            "|---|---|---|",
        ])
        for item in dev["dev_first_mismatches"]:
            lines.append(
                f"| {item['row_id']} | {item['baseline_final_prediction']} | {item['candidate_final_prediction']} |"
            )

    lines.extend([
        "",
        "## 2. Stage34 Held-Out Preservation",
        f"- Status: `{stage34.get('status')}`",
        f"- Baseline shadow accuracy: {fmt(stage34.get('stage34_baseline_shadow_accuracy'))}",
        f"- Candidate shadow accuracy: {fmt(stage34.get('stage34_candidate_shadow_accuracy'))}",
        f"- Baseline shadow macro-F1: {fmt(stage34.get('stage34_baseline_shadow_macro_f1'))}",
        f"- Candidate shadow macro-F1: {fmt(stage34.get('stage34_candidate_shadow_macro_f1'))}",
        f"- Shadow macro-F1 delta: {fmt(stage34.get('stage34_shadow_macro_f1_delta'))}",
        f"- Support shadow==SUPPORT: {fmt(stage34.get('stage34_support_shadow_support'))}",
        f"- Whole->part support recovered: {fmt(stage34.get('stage34_whole_to_part_support_recovered'))}",
        f"- Part->whole NE preserved: {fmt(stage34.get('stage34_part_to_whole_ne_preserved'))}",
        f"- Overclaim to SUPPORT: {fmt(stage34.get('stage34_overclaim_to_support'))}",
        f"- Refute to SUPPORT: {fmt(stage34.get('stage34_refute_to_support'))}",
        f"- Support to REFUTE: {fmt(stage34.get('stage34_support_to_refute'))}",
        f"- Preservation pass: {fmt(stage34.get('stage34_preservation_pass'))}",
    ])

    lines.extend([
        "",
        "## 3. Stage35/37 Adversarial Audit",
        f"- Status: `{adversarial.get('status')}` (source: {adversarial.get('source')})",
        f"- Shadow accuracy: {fmt(adversarial.get('stage37_shadow_accuracy'))}",
        f"- Shadow macro-F1: {fmt(adversarial.get('stage37_shadow_macro_f1'))}",
        f"- Delta macro-F1: {fmt(adversarial.get('stage37_delta_macro_f1'))}",
        f"- adv_support_shadow_support: {fmt(adversarial.get('adv_support_shadow_support'))}",
        f"- adv_subset_support_recovered: {fmt(adversarial.get('adv_subset_support_recovered'))}",
        f"- adv_numeric_support_recovered: {fmt(adversarial.get('adv_numeric_support_recovered'))}",
        f"- adv_exception_support_recovered: {fmt(adversarial.get('adv_exception_support_recovered'))}",
        f"- adv_overclaim_to_support: {fmt(adversarial.get('adv_overclaim_to_support'))}",
        f"- adv_exception_to_support_error: {fmt(adversarial.get('adv_exception_to_support_error'))}",
        f"- adv_location_scope_to_support_error: {fmt(adversarial.get('adv_location_scope_to_support_error'))}",
        f"- adv_temporal_scope_to_support_error: {fmt(adversarial.get('adv_temporal_scope_to_support_error'))}",
        f"- adv_refute_to_support: {fmt(adversarial.get('adv_refute_to_support'))}",
        f"- adv_support_to_refute: {fmt(adversarial.get('adv_support_to_refute'))}",
        f"- stage35_scope_safety: `{adversarial.get('stage35_scope_safety')}`",
        f"- stage35_reverse_overclaim_handling: `{adversarial.get('stage35_reverse_overclaim_handling')}`",
        f"- Adversarial safety pass: {fmt(adversarial.get('adversarial_safety_pass'))}",
        f"- Adversarial recovery pass: {fmt(adversarial.get('adversarial_recovery_pass'))}",
    ])

    lines.extend([
        "",
        "## 4. Stage36/37 Behavior Audit",
        f"- Status: `{behavior.get('status')}`",
        f"- Stage36 support blocker fired: {fmt(behavior.get('stage36_support_blocker_fired_count'))}",
        f"- Stage36 exception blocker fired: {fmt(behavior.get('stage36_exception_blocker_fired_count'))}",
        f"- Stage36 not-all blocker fired: {fmt(behavior.get('stage36_not_all_blocker_fired_count'))}",
        f"- Stage36 location-scope blocker fired: {fmt(behavior.get('stage36_location_scope_blocker_fired_count'))}",
        f"- Stage36 blocked SUPPORT->NE: {fmt(behavior.get('stage36_blocked_support_to_ne_count'))}",
        f"- Stage37 safe recovery fired: {fmt(behavior.get('stage37_safe_recovery_fired_count'))}",
        f"- Stage37 no-except included-subset fired: {fmt(behavior.get('stage37_no_except_included_subset_fired_count'))}",
        f"- Stage37 coordination universal-subset fired: {fmt(behavior.get('stage37_coordination_universal_subset_fired_count'))}",
        f"- Stage37 numeric universal-subset fired: {fmt(behavior.get('stage37_numeric_universal_subset_fired_count'))}",
        f"- Stage37 recovered from NOT_ENTITLED: {fmt(behavior.get('stage37_recovered_from_not_entitled_count'))}",
        f"- Stage37 recovered from REFUTE: {fmt(behavior.get('stage37_recovered_from_refute_count'))}",
        f"- Stage37 blocked by Stage36: {fmt(behavior.get('stage37_blocked_by_stage36_count'))}",
        f"- Stage36/37 behavior pass: {fmt(behavior.get('stage36_37_behavior_pass'))}",
    ])

    lines.extend([
        "",
        "## Pass/Fail Table",
        "| Check | Result |",
        "|---|---:|",
        f"| dev_final_prediction_identity_pass | {fmt(pass_flags['dev_final_prediction_identity_pass'])} |",
        f"| stage34_preservation_pass | {fmt(pass_flags['stage34_preservation_pass'])} |",
        f"| adversarial_safety_pass | {fmt(pass_flags['adversarial_safety_pass'])} |",
        f"| adversarial_recovery_pass | {fmt(pass_flags['adversarial_recovery_pass'])} |",
        f"| stage36_37_behavior_pass | {fmt(pass_flags['stage36_37_behavior_pass'])} |",
        "",
        "## Remaining Risks",
    ])
    for risk in report["risks"]:
        lines.append(f"- {risk}")
    lines.extend([
        "",
        "## Recommendation",
        report["recommendation"],
        "",
        "## Leakage Policy",
        "Diagnostic-only. This audit must not be used for training, calibration, "
        "threshold selection, loss, checkpoint selection, or Kaggle selection.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)

    parser.add_argument("--dev-baseline-predictions", default=None)
    parser.add_argument("--dev-candidate-predictions", default=None)

    parser.add_argument("--stage34-baseline-report", default=None)
    parser.add_argument("--stage34-candidate-report", default=None)
    parser.add_argument("--stage34-baseline-predictions", default=None)
    parser.add_argument("--stage34-candidate-predictions", default=None)

    parser.add_argument("--stage35-baseline-report", default=None)
    parser.add_argument("--stage36-report", default=None)
    parser.add_argument("--stage37-report", default=None)
    parser.add_argument("--stage35-baseline-predictions", default=None)
    parser.add_argument("--stage37-predictions", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    dev_identity = dev_identity_audit(args.dev_baseline_predictions, args.dev_candidate_predictions)
    stage34 = stage34_preservation_audit(args)
    adversarial = stage35_37_adversarial_audit(args)
    behavior = stage36_37_behavior_audit(args)

    decision = integrated_decision(dev_identity, stage34, adversarial, behavior)
    missing_inputs = collect_missing_inputs(args)

    pass_flags = {
        "dev_final_prediction_identity_pass": dev_identity.get("dev_final_prediction_identity_pass"),
        "stage34_preservation_pass": stage34.get("stage34_preservation_pass"),
        "adversarial_safety_pass": adversarial.get("adversarial_safety_pass"),
        "adversarial_recovery_pass": adversarial.get("adversarial_recovery_pass"),
        "stage36_37_behavior_pass": behavior.get("stage36_37_behavior_pass"),
    }

    report = {
        "run_name": args.run_name,
        "decision": decision,
        "sections": {
            "dev_identity": dev_identity,
            "stage34_preservation": stage34,
            "stage35_37_adversarial": adversarial,
            "stage36_37_behavior": behavior,
        },
        "pass_flags": pass_flags,
        "key_metrics": build_key_metrics(dev_identity, stage34, adversarial, behavior),
        "missing_inputs": missing_inputs,
        "risks": build_risks(dev_identity, stage34, adversarial, behavior, missing_inputs),
        "recommendation": build_recommendation(decision["label"]),
        "leakage_policy": (
            "Diagnostic-only. Do not use for training, calibration, threshold "
            "selection, loss, checkpoint selection, or Kaggle selection."
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
