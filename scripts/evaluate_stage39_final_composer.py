"""Stage39-A/C opt-in final composer evaluator.

Diagnostic only. This script reads an already-exported prediction file (one
run of scripts/train_controlled_v6b_minimal.py with Stage39 flags on) and
reports whether the Stage39 opt-in final composer is safe to use as an
explicit final-prediction replacement. It does not train, evaluate the
model, run Kaggle, or change any prediction file it reads.

Must not be used for training, calibration, threshold selection, loss
computation, or checkpoint selection. Stage39 itself is off by default in
the training script; this evaluator only inspects what a Stage39-enabled
export already produced.

Stage39-C update: the JSON report exposes flat top-level keys (run_name,
decision, original_metrics, composed_metrics, exported_metrics, delta,
change_counts, safety_counters, introduced_safety_counters,
stage39_behavior_counters, first_changed_rows, first_unsafe_rows,
recommendation) instead of nesting everything under an "analysis" key, and
the decision logic distinguishes total residual errors (safety_counters,
inherited from the original model) from errors actually introduced by
Stage39 composition (introduced_safety_counters).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

LABELS = ["REFUTE", "NOT_ENTITLED", "SUPPORT"]

# Dict wrapper keys tried, in order, when the top-level JSON payload is a
# dict rather than a bare list of row dicts.
DICT_WRAPPER_KEYS = ("predictions", "records", "items", "data")

ORIGINAL_FINAL_KEY_PRIORITY = (
    "stage39_original_final_label",
    "stage39_original_pred_final_label",
    "pred_final_label",
    "pred_label",
    "prediction",
    "predicted_label",
)

EXPORTED_FINAL_KEY_PRIORITY = (
    "pred_final_label",
    "pred_label",
    "prediction",
    "predicted_label",
)

GOLD_KEY_PRIORITY = ("gold_final_label", "gold_label", "final_label", "label")

# Stage35 adversarial group name substrings used to classify unsafe SUPPORT
# leakage. Mirrors scripts/evaluate_stage35_adversarial_coverage.py; read-only
# reuse of the naming convention, not an import.
STAGE35_OVERCLAIM_GROUP_SUBSTRING = "not_entitled"
STAGE35_EXCEPTION_GROUP_SUBSTRING = "all_except_subset_not_entitled"
STAGE35_LOCATION_GROUP_SUBSTRING = "location_scope_not_entitled"
STAGE35_TEMPORAL_GROUP_SUBSTRING = "temporal_scope_not_entitled"

# Decision thresholds (documented here, not tuned against Stage34/35 gold
# labels -- these are generic regression/improvement guardrails applied
# uniformly regardless of which probe is being evaluated).
MATERIAL_REGRESSION_THRESHOLD = -0.01
MATERIAL_IMPROVEMENT_THRESHOLD = 0.03


# ---------------------------------------------------------------------------
# Generic, crash-proof IO helpers
# ---------------------------------------------------------------------------


def _resolve_path(path_str: "str | None") -> "Path | None":
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def safe_read_json(path_str: "str | None") -> Any:
    path = _resolve_path(path_str)
    if path is None or not path.exists() or not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None


def _first_list_of_dicts(data: dict[str, Any]) -> "list[dict[str, Any]] | None":
    for key in DICT_WRAPPER_KEYS:
        value = data.get(key)
        if isinstance(value, list) and (not value or isinstance(value[0], dict)):
            return [row for row in value if isinstance(row, dict)]
    for value in data.values():
        if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
            return list(value)
    return None


def as_rows(data: Any) -> "list[dict[str, Any]] | None":
    """Normalize a loaded JSON payload into a list of row dicts.

    Handles: bare list JSON, dict with predictions/records/items/data, and
    dict with any other list-of-dicts field.
    """
    if data is None:
        return None
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        return _first_list_of_dicts(data)
    return None


def load_prediction_rows(path_str: "str | None") -> "list[dict[str, Any]] | None":
    return as_rows(safe_read_json(path_str))


def first_non_null(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


# ---------------------------------------------------------------------------
# Label / row helpers
# ---------------------------------------------------------------------------


def normalize_label_safe(raw: Any) -> "str | None":
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


def get_original_final_label(row: dict[str, Any]) -> "str | None":
    for key in ORIGINAL_FINAL_KEY_PRIORITY:
        value = row.get(key)
        if value not in (None, ""):
            return normalize_label_safe(value)
    return None


def get_composed_final_label(row: dict[str, Any]) -> "str | None":
    value = row.get("stage39_composed_final_label")
    if value not in (None, ""):
        return normalize_label_safe(value)
    for key in EXPORTED_FINAL_KEY_PRIORITY:
        value = row.get(key)
        if value not in (None, ""):
            return normalize_label_safe(value)
    return None


def get_exported_final_label(row: dict[str, Any]) -> "str | None":
    for key in EXPORTED_FINAL_KEY_PRIORITY:
        value = row.get(key)
        if value not in (None, ""):
            return normalize_label_safe(value)
    return None


def get_gold_label(row: dict[str, Any]) -> "str | None":
    for key in GOLD_KEY_PRIORITY:
        value = row.get(key)
        if value not in (None, ""):
            return normalize_label_safe(value)
    return None


def get_group(row: dict[str, Any]) -> str:
    value = first_non_null(
        row.get("group"),
        row.get("intervention_type"),
        row.get("normalized_intervention"),
        row.get("primary_failure_type"),
    )
    return "" if value is None else str(value)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def prediction_counts(labels: "list[str | None]") -> dict[str, int]:
    counter = Counter(label for label in labels if label is not None)
    return dict(sorted(counter.items()))


def compute_metrics_block(
    golds: "list[str | None]", preds: "list[str | None]"
) -> "dict[str, Any] | None":
    """accuracy/macro_f1/n/prediction_counts/per_class over cleanly-paired rows."""
    pairs = [(g, p) for g, p in zip(golds, preds) if g is not None and p is not None]
    if not pairs:
        return None
    accuracy = sum(g == p for g, p in pairs) / len(pairs)
    per_class: dict[str, Any] = {}
    f1s = []
    for label in LABELS:
        tp = sum(g == label and p == label for g, p in pairs)
        fp = sum(g != label and p == label for g, p in pairs)
        fn = sum(g == label and p != label for g, p in pairs)
        support = sum(g == label for g, _ in pairs)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
        f1s.append(f1)
    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(sum(f1s) / len(f1s), 4),
        "n": len(pairs),
        "prediction_counts": prediction_counts(preds),
        "per_class": per_class,
    }


def _metric_delta(key: str, before: "dict[str, Any] | None", after: "dict[str, Any] | None") -> "float | None":
    if before is None or after is None:
        return None
    return round(after[key] - before[key], 4)


# ---------------------------------------------------------------------------
# Row-level analysis
# ---------------------------------------------------------------------------


def empty_change_counts() -> dict[str, Any]:
    return {
        "changed_row_count": 0,
        "changed_row_rate": 0.0,
        "changed_to_SUPPORT_count": 0,
        "changed_to_REFUTE_count": 0,
        "changed_to_NOT_ENTITLED_count": 0,
        "REFUTE_to_SUPPORT_count": 0,
        "NOT_ENTITLED_to_SUPPORT_count": 0,
        "SUPPORT_to_NOT_ENTITLED_count": 0,
        "SUPPORT_to_REFUTE_count": 0,
    }


def empty_safety_counters() -> dict[str, Any]:
    return {
        "overclaim_to_SUPPORT": 0,
        "exception_to_SUPPORT_error": 0,
        "location_scope_to_SUPPORT_error": 0,
        "temporal_scope_to_SUPPORT_error": 0,
        "refute_to_SUPPORT": 0,
        "support_to_refute": 0,
    }


def empty_introduced_safety_counters() -> dict[str, Any]:
    return {
        "introduced_overclaim_to_SUPPORT": 0,
        "introduced_exception_to_SUPPORT_error": 0,
        "introduced_location_scope_to_SUPPORT_error": 0,
        "introduced_temporal_scope_to_SUPPORT_error": 0,
        "introduced_refute_to_SUPPORT": 0,
        "introduced_support_to_REFUTE": 0,
        "introduced_unsafe_SUPPORT_total": 0,
    }


def empty_behavior_counters() -> dict[str, Any]:
    return {
        "stage39_enabled_count": 0,
        "stage39_changed_count": 0,
        "stage39_support_composed_count": 0,
        "stage39_refute_composed_count": 0,
        "stage39_not_entitled_composed_count": 0,
        "stage39_blocked_by_stage36_count": 0,
        "stage39_blocked_by_refute_to_support_guard_count": 0,
        "stage39_blocked_by_stage37_from_refute_guard_count": 0,
        "stage39_missing_source_count": 0,
        "stage39_composer_action_counts": {},
        "stage39_composer_reason_counts": {},
    }


def analyze_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    golds = [get_gold_label(row) for row in rows]
    originals = [get_original_final_label(row) for row in rows]
    composed = [get_composed_final_label(row) for row in rows]
    exported = [get_exported_final_label(row) for row in rows]

    original_metrics = compute_metrics_block(golds, originals)
    composed_metrics = compute_metrics_block(golds, composed)
    exported_metrics = compute_metrics_block(golds, exported)

    delta = {
        "composed_minus_original_accuracy": _metric_delta(
            "accuracy", original_metrics, composed_metrics
        ),
        "composed_minus_original_macro_f1": _metric_delta(
            "macro_f1", original_metrics, composed_metrics
        ),
        "exported_minus_original_accuracy": _metric_delta(
            "accuracy", original_metrics, exported_metrics
        ),
        "exported_minus_original_macro_f1": _metric_delta(
            "macro_f1", original_metrics, exported_metrics
        ),
    }

    row_count = len(rows)
    changed_rows: list[dict[str, Any]] = []
    unsafe_rows: list[dict[str, Any]] = []
    introduced_unsafe_rows: list[dict[str, Any]] = []

    change_counts = empty_change_counts()
    safety_counters = empty_safety_counters()
    introduced_safety_counters = empty_introduced_safety_counters()
    behavior_counters = empty_behavior_counters()
    composer_action_counts: Counter[str] = Counter()
    composer_reason_counts: Counter[str] = Counter()

    for index, row in enumerate(rows):
        gold = golds[index]
        original = originals[index]
        candidate = composed[index]
        group = get_group(row)

        if bool(row.get("stage39_final_composer_enabled")):
            behavior_counters["stage39_enabled_count"] += 1
        if bool(row.get("stage39_blocked_by_stage36")):
            behavior_counters["stage39_blocked_by_stage36_count"] += 1
        if bool(row.get("stage39_blocked_by_refute_to_support_guard")):
            behavior_counters["stage39_blocked_by_refute_to_support_guard_count"] += 1
        if bool(row.get("stage39_blocked_by_stage37_from_refute_guard")):
            behavior_counters["stage39_blocked_by_stage37_from_refute_guard_count"] += 1
        if bool(row.get("stage39_blocked_by_missing_source")):
            behavior_counters["stage39_missing_source_count"] += 1
        action = row.get("stage39_composer_action")
        if action not in (None, ""):
            composer_action_counts[str(action)] += 1
        reason = row.get("stage39_composer_reason")
        if reason not in (None, ""):
            composer_reason_counts[str(reason)] += 1

        is_changed = bool(row.get("stage39_final_label_changed"))
        if is_changed is False and original is not None and candidate is not None:
            is_changed = original != candidate

        if is_changed:
            behavior_counters["stage39_changed_count"] += 1
            change_counts["changed_row_count"] += 1
            if candidate == "SUPPORT":
                change_counts["changed_to_SUPPORT_count"] += 1
                behavior_counters["stage39_support_composed_count"] += 1
            elif candidate == "REFUTE":
                change_counts["changed_to_REFUTE_count"] += 1
                behavior_counters["stage39_refute_composed_count"] += 1
            elif candidate == "NOT_ENTITLED":
                change_counts["changed_to_NOT_ENTITLED_count"] += 1
                behavior_counters["stage39_not_entitled_composed_count"] += 1

            if original == "REFUTE" and candidate == "SUPPORT":
                change_counts["REFUTE_to_SUPPORT_count"] += 1
            if original == "NOT_ENTITLED" and candidate == "SUPPORT":
                change_counts["NOT_ENTITLED_to_SUPPORT_count"] += 1
            if original == "SUPPORT" and candidate == "NOT_ENTITLED":
                change_counts["SUPPORT_to_NOT_ENTITLED_count"] += 1
            if original == "SUPPORT" and candidate == "REFUTE":
                change_counts["SUPPORT_to_REFUTE_count"] += 1

            row_summary = {
                "row_id": str(row.get("stable_id") or row.get("id") or index),
                "gold": gold,
                "original_final_label": original,
                "composed_final_label": candidate,
                "composer_action": row.get("stage39_composer_action"),
                "composer_reason": row.get("stage39_composer_reason"),
            }
            if len(changed_rows) < 30:
                changed_rows.append(row_summary)

        # Total (residual-inclusive) safety counters: may include errors the
        # original model already made before Stage39 ever ran.
        is_unsafe = False
        if STAGE35_OVERCLAIM_GROUP_SUBSTRING in group and candidate == "SUPPORT":
            safety_counters["overclaim_to_SUPPORT"] += 1
            is_unsafe = True
        if STAGE35_EXCEPTION_GROUP_SUBSTRING in group and candidate == "SUPPORT":
            safety_counters["exception_to_SUPPORT_error"] += 1
            is_unsafe = True
        if STAGE35_LOCATION_GROUP_SUBSTRING in group and candidate == "SUPPORT":
            safety_counters["location_scope_to_SUPPORT_error"] += 1
            is_unsafe = True
        if STAGE35_TEMPORAL_GROUP_SUBSTRING in group and candidate == "SUPPORT":
            safety_counters["temporal_scope_to_SUPPORT_error"] += 1
            is_unsafe = True
        if gold == "REFUTE" and candidate == "SUPPORT":
            safety_counters["refute_to_SUPPORT"] += 1
            is_unsafe = True
        if gold == "SUPPORT" and candidate == "REFUTE":
            safety_counters["support_to_refute"] += 1
            is_unsafe = True

        if is_unsafe and len(unsafe_rows) < 30:
            unsafe_rows.append(
                {
                    "row_id": str(row.get("stable_id") or row.get("id") or index),
                    "group": group,
                    "gold": gold,
                    "original_final_label": original,
                    "composed_final_label": candidate,
                    "composer_action": row.get("stage39_composer_action"),
                    "composer_reason": row.get("stage39_composer_reason"),
                }
            )

        # Introduced-only safety counters: only rows Stage39 itself changed
        # can have introduced a harmful transition. A row that was already
        # unsafe before Stage39 ran (is_changed == False) is a residual
        # error inherited from the original model, not something Stage39
        # introduced.
        if is_changed:
            introduced_unsafe_support = False
            if STAGE35_OVERCLAIM_GROUP_SUBSTRING in group and candidate == "SUPPORT":
                introduced_safety_counters["introduced_overclaim_to_SUPPORT"] += 1
                introduced_unsafe_support = True
            if STAGE35_EXCEPTION_GROUP_SUBSTRING in group and candidate == "SUPPORT":
                introduced_safety_counters["introduced_exception_to_SUPPORT_error"] += 1
                introduced_unsafe_support = True
            if STAGE35_LOCATION_GROUP_SUBSTRING in group and candidate == "SUPPORT":
                introduced_safety_counters[
                    "introduced_location_scope_to_SUPPORT_error"
                ] += 1
                introduced_unsafe_support = True
            if STAGE35_TEMPORAL_GROUP_SUBSTRING in group and candidate == "SUPPORT":
                introduced_safety_counters[
                    "introduced_temporal_scope_to_SUPPORT_error"
                ] += 1
                introduced_unsafe_support = True
            if gold == "REFUTE" and candidate == "SUPPORT":
                introduced_safety_counters["introduced_refute_to_SUPPORT"] += 1
                introduced_unsafe_support = True
            introduced_support_to_refute = gold == "SUPPORT" and candidate == "REFUTE"
            if introduced_support_to_refute:
                introduced_safety_counters["introduced_support_to_REFUTE"] += 1

            if introduced_unsafe_support:
                introduced_safety_counters["introduced_unsafe_SUPPORT_total"] += 1
                if len(introduced_unsafe_rows) < 30:
                    introduced_unsafe_rows.append(
                        {
                            "row_id": str(row.get("stable_id") or row.get("id") or index),
                            "group": group,
                            "gold": gold,
                            "original_final_label": original,
                            "composed_final_label": candidate,
                            "composer_action": row.get("stage39_composer_action"),
                            "composer_reason": row.get("stage39_composer_reason"),
                        }
                    )
            elif introduced_support_to_refute and len(introduced_unsafe_rows) < 30:
                introduced_unsafe_rows.append(
                    {
                        "row_id": str(row.get("stable_id") or row.get("id") or index),
                        "group": group,
                        "gold": gold,
                        "original_final_label": original,
                        "composed_final_label": candidate,
                        "composer_action": row.get("stage39_composer_action"),
                        "composer_reason": row.get("stage39_composer_reason"),
                    }
                )

    change_counts["changed_row_rate"] = (
        round(change_counts["changed_row_count"] / row_count, 4) if row_count else 0.0
    )
    behavior_counters["stage39_composer_action_counts"] = dict(composer_action_counts)
    behavior_counters["stage39_composer_reason_counts"] = dict(composer_reason_counts)

    return {
        "row_count": row_count,
        "original_metrics": original_metrics,
        "composed_metrics": composed_metrics,
        "exported_metrics": exported_metrics,
        "delta": delta,
        "change_counts": change_counts,
        "safety_counters": safety_counters,
        "introduced_safety_counters": introduced_safety_counters,
        "stage39_behavior_counters": behavior_counters,
        "first_changed_rows": changed_rows,
        "first_unsafe_rows": unsafe_rows,
        "first_introduced_unsafe_rows": introduced_unsafe_rows,
    }


# ---------------------------------------------------------------------------
# Baseline comparison (optional)
# ---------------------------------------------------------------------------


def compare_to_baseline(
    baseline_rows: "list[dict[str, Any]] | None", candidate_rows: list[dict[str, Any]]
) -> "dict[str, Any] | None":
    if not baseline_rows:
        return None
    baseline_by_id: dict[str, Any] = {}
    for index, row in enumerate(baseline_rows):
        row_id = str(row.get("stable_id") or row.get("id") or index)
        baseline_by_id[row_id] = get_exported_final_label(row)
    unexpected_baseline_diffs = 0
    overlap = 0
    for index, row in enumerate(candidate_rows):
        row_id = str(row.get("stable_id") or row.get("id") or index)
        if row_id not in baseline_by_id:
            continue
        overlap += 1
        original = get_original_final_label(row)
        baseline_label = baseline_by_id[row_id]
        if original is not None and baseline_label is not None and original != baseline_label:
            unexpected_baseline_diffs += 1
    return {
        "baseline_row_count": len(baseline_rows),
        "overlap_row_count": overlap,
        "unexpected_original_vs_baseline_diff_count": unexpected_baseline_diffs,
        "original_matches_baseline": (
            unexpected_baseline_diffs == 0 if overlap else None
        ),
    }


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------


def _primary_metrics(analysis: dict[str, Any]) -> "dict[str, Any] | None":
    """Composed metrics when available (the diagnostic composition signal),
    falling back to exported metrics (e.g. when composed==exported or the
    run only carries exported labels)."""
    return analysis["composed_metrics"] if analysis["composed_metrics"] is not None else analysis["exported_metrics"]


def decide(profile: str, analysis: dict[str, Any]) -> dict[str, Any]:
    introduced = analysis["introduced_safety_counters"]
    unsafe_support_total = introduced.get("introduced_unsafe_SUPPORT_total") or 0
    introduced_refute_to_support = introduced.get("introduced_refute_to_SUPPORT") or 0
    introduced_support_to_refute = introduced.get("introduced_support_to_REFUTE") or 0

    # Global regression check: any run, any profile. Only counts errors that
    # Stage39 composition itself introduced -- pre-existing model errors
    # (safety_counters, including total support_to_refute) are reported
    # separately as residual, never used to fail here.
    if unsafe_support_total > 0 or introduced_refute_to_support > 0:
        return {
            "label": "STAGE39A_FINAL_COMPOSER_SAFETY_REGRESSION",
            "reason": (
                "Stage39 composition introduced at least one unsafe SUPPORT "
                "transition (overclaim/exception/location-scope/temporal-"
                "scope/REFUTE-to-SUPPORT) that was not present before "
                "composition. introduced_unsafe_SUPPORT_total="
                f"{unsafe_support_total}, introduced_refute_to_SUPPORT="
                f"{introduced_refute_to_support}."
            ),
        }

    original_metrics = analysis["original_metrics"]
    metrics = _primary_metrics(analysis)
    delta = analysis["delta"]
    change_counts = analysis["change_counts"]
    changed_row_count = change_counts.get("changed_row_count", 0)

    macro_delta = delta.get("composed_minus_original_macro_f1")
    if macro_delta is None:
        macro_delta = delta.get("exported_minus_original_macro_f1")
    accuracy_delta = delta.get("composed_minus_original_accuracy")
    if accuracy_delta is None:
        accuracy_delta = delta.get("exported_minus_original_accuracy")

    if profile == "dev":
        if changed_row_count == 0:
            return {
                "label": "STAGE39A_FINAL_COMPOSER_TOO_WEAK",
                "reason": (
                    "safety_profile=dev: no row's final label was changed by "
                    "composition, so there is nothing to validate for a "
                    "production replace_pred_final_label decision yet."
                ),
            }
        material_regression = (
            macro_delta is not None and macro_delta < MATERIAL_REGRESSION_THRESHOLD
        ) or (accuracy_delta is not None and accuracy_delta < MATERIAL_REGRESSION_THRESHOLD)
        if material_regression:
            return {
                "label": "STAGE39A_FINAL_COMPOSER_DEV_REGRESSION",
                "reason": (
                    "safety_profile=dev and composed/exported macro-F1 or "
                    "accuracy regressed materially versus the original final "
                    "label."
                ),
            }
        return {
            "label": "STAGE39A_FINAL_COMPOSER_OPT_IN_PASS",
            "reason": (
                "dev profile: no material accuracy/macro-F1 regression and "
                "no unsafe SUPPORT/REFUTE transitions were introduced by "
                "Stage39 composition."
            ),
        }

    if profile == "stage34":
        mf1 = metrics.get("macro_f1") if metrics else None
        acc = metrics.get("accuracy") if metrics else None
        if (
            mf1 is not None
            and mf1 >= 0.90
            and acc is not None
            and acc >= 0.90
            and unsafe_support_total == 0
            and introduced_refute_to_support == 0
            and introduced_support_to_refute == 0
            and changed_row_count > 0
        ):
            return {
                "label": "STAGE39C_SAFE_STRUCTURED_V2_PASS",
                "reason": (
                    "Stage34 composed/exported macro-F1 and accuracy both "
                    "meet the >=0.90 target, Stage39 introduced no unsafe "
                    "SUPPORT or SUPPORT<->REFUTE transitions, and at least "
                    "one row was changed by composition."
                ),
            }
        material_improvement = (
            macro_delta is not None and macro_delta >= MATERIAL_IMPROVEMENT_THRESHOLD
        )
        if (
            material_improvement
            and unsafe_support_total == 0
            and introduced_support_to_refute == 0
        ):
            return {
                "label": "STAGE39C_SAFE_STRUCTURED_V2_PARTIAL",
                "reason": (
                    "Stage34 macro-F1 improved materially over the original "
                    "final label with no unsafe transitions introduced by "
                    "Stage39, but the >=0.90 macro-F1/accuracy PASS "
                    "threshold was not fully met."
                ),
            }
        return {
            "label": "STAGE39A_FINAL_COMPOSER_DIAGNOSTIC_ONLY",
            "reason": (
                "Stage34 profile thresholds are not conclusively met (macro-"
                "F1/accuracy, introduced-safety, or changed-row-count); "
                "treat as diagnostic only until re-validated."
            ),
        }

    if profile == "stage35":
        mf1 = metrics.get("macro_f1") if metrics else None
        acc = metrics.get("accuracy") if metrics else None
        changed_to_support = change_counts.get("changed_to_SUPPORT_count", 0)
        if (
            mf1 is not None
            and mf1 >= 0.70
            and acc is not None
            and acc >= 0.75
            and changed_to_support >= 140
            and unsafe_support_total == 0
            and introduced_refute_to_support == 0
            and introduced_support_to_refute == 0
        ):
            return {
                "label": "STAGE39C_SAFE_STRUCTURED_V2_PASS",
                "reason": (
                    "Stage35 composed/exported macro-F1 >=0.70, accuracy "
                    ">=0.75, >=140 rows recovered to SUPPORT, and Stage39 "
                    "introduced no unsafe SUPPORT or SUPPORT<->REFUTE "
                    "transitions."
                ),
            }
        material_improvement = (
            macro_delta is not None and macro_delta >= MATERIAL_IMPROVEMENT_THRESHOLD
        )
        if (
            material_improvement
            and unsafe_support_total == 0
            and introduced_support_to_refute == 0
        ):
            return {
                "label": "STAGE39C_SAFE_STRUCTURED_V2_PARTIAL",
                "reason": (
                    "Stage35 macro-F1 improved materially with no unsafe "
                    "transitions introduced by Stage39, but the full "
                    "recovery thresholds (macro-F1 >=0.70, accuracy >=0.75, "
                    ">=140 SUPPORT recoveries) were not fully met."
                ),
            }
        return {
            "label": "STAGE39A_FINAL_COMPOSER_DIAGNOSTIC_ONLY",
            "reason": (
                "Stage35 profile thresholds are not conclusively met (macro-"
                "F1/accuracy, SUPPORT recovery count, or introduced-safety); "
                "treat as diagnostic only until re-validated."
            ),
        }

    # generic profile: report metrics without the Stage34/35-specific
    # recovery thresholds; still gated on no unsafe transitions introduced.
    if (
        original_metrics is not None
        and metrics is not None
        and macro_delta is not None
        and macro_delta >= 0
        and introduced_support_to_refute == 0
    ):
        return {
            "label": "STAGE39A_FINAL_COMPOSER_OPT_IN_PASS",
            "reason": (
                "generic profile: composed/exported macro-F1 does not "
                "regress versus original and Stage39 introduced no unsafe "
                "SUPPORT/REFUTE transitions."
            ),
        }
    return {
        "label": "STAGE39A_FINAL_COMPOSER_DIAGNOSTIC_ONLY",
        "reason": (
            "generic profile: insufficient fields or mixed evidence to "
            "reach a strict pass verdict."
        ),
    }


def build_recommendation(decision_label: str) -> str:
    if decision_label in (
        "STAGE39A_FINAL_COMPOSER_OPT_IN_PASS",
        "STAGE39C_SAFE_STRUCTURED_V2_PASS",
    ):
        return (
            "The Stage39 opt-in final composer is safe to use as an "
            "explicit final-prediction replacement under the evaluated "
            "policy/profile. It remains off by default; enabling it in "
            "production requires an explicit --stage39-use-final-composer-"
            "opt-in (and, to replace predictions, --stage39-final-composer-"
            "output-mode replace_pred_final_label) decision outside this "
            "script."
        )
    if decision_label == "STAGE39C_SAFE_STRUCTURED_V2_PARTIAL":
        return (
            "Composition is safe (no unsafe SUPPORT or SUPPORT<->REFUTE "
            "transitions were introduced by Stage39) and materially "
            "improves macro-F1, but does not yet meet the full recovery "
            "threshold for this safety profile. Useful as a diagnostic "
            "improvement; do not treat as a final production PASS until "
            "thresholds are met."
        )
    if decision_label == "STAGE39A_FINAL_COMPOSER_SAFETY_REGRESSION":
        return (
            "Reject this composer configuration: Stage39 composition "
            "introduced unsafe SUPPORT leakage. Tighten the policy/guards "
            "or keep Stage39 export_only."
        )
    if decision_label == "STAGE39A_FINAL_COMPOSER_DEV_REGRESSION":
        return (
            "Reject this composer configuration for dev: it materially "
            "regresses accuracy/macro-F1 versus the original final label."
        )
    if decision_label == "STAGE39A_FINAL_COMPOSER_TOO_WEAK":
        return (
            "Composition is safe but did not change any row; there is no "
            "benefit yet to justify enabling replace_pred_final_label."
        )
    return (
        "Insufficient or mixed evidence. Keep Stage39 in export_only "
        "diagnostic mode and re-run this evaluator with more complete "
        "predictions/gold labels before considering replace_pred_final_label."
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _metrics_table_rows(original: dict, composed: dict, exported: dict) -> list[str]:
    rows = [
        "| Metric | Original | Composed | Exported |",
        "|---|---:|---:|---:|",
        f"| accuracy | {_fmt(original.get('accuracy'))} | {_fmt(composed.get('accuracy'))} | {_fmt(exported.get('accuracy'))} |",
        f"| macro_f1 | {_fmt(original.get('macro_f1'))} | {_fmt(composed.get('macro_f1'))} | {_fmt(exported.get('macro_f1'))} |",
        f"| n | {_fmt(original.get('n'))} | {_fmt(composed.get('n'))} | {_fmt(exported.get('n'))} |",
    ]
    return rows


def _row_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> list[str]:
    if not rows:
        return ["None."]
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    lines = [header, sep]
    for row in rows:
        cells = [_fmt(row.get(key)) for key, _ in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    original_metrics = report["original_metrics"] or {}
    composed_metrics = report["composed_metrics"] or {}
    exported_metrics = report["exported_metrics"] or {}
    delta = report["delta"]
    change_counts = report["change_counts"]
    safety = report["safety_counters"]
    introduced_safety = report["introduced_safety_counters"]
    behavior = report["stage39_behavior_counters"]

    lines = [
        "# Stage39 Final Composer Opt-In Validation",
        "",
        "## 1. Overall Decision",
        f"- Run: `{report['run_name']}`",
        f"- Safety profile: `{report['safety_profile']}`",
        f"- Decision: `{report['decision']['label']}`",
        f"- Decision reason: {report['decision']['reason']}",
        "",
        "## Input File",
        f"- Predictions file: `{report['predictions_file']}`",
        f"- Row count: {report['row_count']}",
        f"- Baseline predictions file: `{report.get('baseline_predictions_file') or 'n/a'}`",
        f"- Probe name: `{report.get('probe_name') or 'n/a'}`",
        f"- Expected split: `{report.get('expected_split') or 'n/a'}`",
    ]

    baseline = report.get("baseline_comparison")
    if baseline is not None:
        lines.extend([
            "",
            "### Baseline Comparison",
            f"- Baseline rows: {_fmt(baseline.get('baseline_row_count'))}",
            f"- Overlap rows: {_fmt(baseline.get('overlap_row_count'))}",
            f"- Unexpected original-vs-baseline diffs: {_fmt(baseline.get('unexpected_original_vs_baseline_diff_count'))}",
            f"- Original matches baseline: {_fmt(baseline.get('original_matches_baseline'))}",
        ])

    lines.extend([
        "",
        "## 2. Original vs Composed vs Exported Final Metrics",
        *_metrics_table_rows(original_metrics, composed_metrics, exported_metrics),
        "",
        f"- composed - original accuracy: {_fmt(delta.get('composed_minus_original_accuracy'))}",
        f"- composed - original macro_f1: {_fmt(delta.get('composed_minus_original_macro_f1'))}",
        f"- exported - original accuracy: {_fmt(delta.get('exported_minus_original_accuracy'))}",
        f"- exported - original macro_f1: {_fmt(delta.get('exported_minus_original_macro_f1'))}",
        f"- Original prediction counts: {original_metrics.get('prediction_counts')}",
        f"- Composed prediction counts: {composed_metrics.get('prediction_counts')}",
        f"- Exported prediction counts: {exported_metrics.get('prediction_counts')}",
    ])

    lines.extend([
        "",
        "## 3. Change Counts",
        "| Counter | Value |",
        "|---|---:|",
    ])
    for key, value in change_counts.items():
        lines.append(f"| {key} | {value} |")

    lines.extend([
        "",
        "## 4. Total Safety Counters (may include residual pre-Stage39 errors)",
        "| Counter | Value |",
        "|---|---:|",
    ])
    for key, value in safety.items():
        lines.append(f"| {key} | {value} |")

    lines.extend([
        "",
        "## 5. Introduced Safety Counters (errors caused by Stage39 composition only)",
        "| Counter | Value |",
        "|---|---:|",
    ])
    for key, value in introduced_safety.items():
        lines.append(f"| {key} | {value} |")

    lines.extend([
        "",
        "## 6. Stage39 Behavior Counters",
        "| Counter | Value |",
        "|---|---:|",
    ])
    for key, value in behavior.items():
        if key in ("stage39_composer_action_counts", "stage39_composer_reason_counts"):
            continue
        lines.append(f"| {key} | {value} |")
    lines.extend([
        "",
        f"- stage39_composer_action_counts: {behavior.get('stage39_composer_action_counts')}",
        f"- stage39_composer_reason_counts: {behavior.get('stage39_composer_reason_counts')}",
    ])

    changed_columns = [
        ("row_id", "Row ID"),
        ("gold", "Gold"),
        ("original_final_label", "Original"),
        ("composed_final_label", "Composed"),
        ("composer_action", "Action"),
        ("composer_reason", "Reason"),
    ]
    unsafe_columns = [
        ("row_id", "Row ID"),
        ("group", "Group"),
        ("gold", "Gold"),
        ("original_final_label", "Original"),
        ("composed_final_label", "Composed"),
        ("composer_action", "Action"),
        ("composer_reason", "Reason"),
    ]

    lines.extend(["", "## 7. First 30 Changed Rows"])
    lines.extend(_row_table(report["first_changed_rows"], changed_columns))

    lines.extend(["", "## 8. First 30 Unsafe Rows (total, including residual)"])
    lines.extend(_row_table(report["first_unsafe_rows"], unsafe_columns))

    lines.extend(["", "## 9. First 30 Introduced-Unsafe Rows (Stage39-caused only)"])
    lines.extend(_row_table(report["first_introduced_unsafe_rows"], unsafe_columns))

    lines.extend([
        "",
        "## 10. Recommendation",
        report["recommendation"],
        "",
        "## Leakage Policy",
        "Diagnostic-only. This evaluator must not be used for training, "
        "calibration, threshold selection, loss, checkpoint selection, or "
        "Kaggle selection.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--predictions-file", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--baseline-predictions", default=None)
    parser.add_argument("--probe-name", default=None)
    parser.add_argument("--expected-split", default=None)
    parser.add_argument(
        "--safety-profile",
        choices=("dev", "stage34", "stage35", "generic"),
        default="generic",
    )
    return parser


def _missing_file_report(args: argparse.Namespace) -> dict[str, Any]:
    decision = {
        "label": "STAGE39A_FINAL_COMPOSER_DIAGNOSTIC_ONLY",
        "reason": (
            f"--predictions-file '{args.predictions_file}' was missing, "
            "unreadable, or not in a recognized row format."
        ),
    }
    return {
        "run_name": args.run_name,
        "predictions_file": args.predictions_file,
        "baseline_predictions_file": args.baseline_predictions,
        "probe_name": args.probe_name,
        "expected_split": args.expected_split,
        "safety_profile": args.safety_profile,
        "row_count": 0,
        "decision": decision,
        "original_metrics": None,
        "composed_metrics": None,
        "exported_metrics": None,
        "delta": {
            "composed_minus_original_accuracy": None,
            "composed_minus_original_macro_f1": None,
            "exported_minus_original_accuracy": None,
            "exported_minus_original_macro_f1": None,
        },
        "change_counts": empty_change_counts(),
        "safety_counters": empty_safety_counters(),
        "introduced_safety_counters": empty_introduced_safety_counters(),
        "stage39_behavior_counters": empty_behavior_counters(),
        "first_changed_rows": [],
        "first_unsafe_rows": [],
        "first_introduced_unsafe_rows": [],
        "baseline_comparison": None,
        "recommendation": build_recommendation(decision["label"]),
        "leakage_policy": (
            "Diagnostic-only. Do not use for training, calibration, threshold "
            "selection, loss, checkpoint selection, or Kaggle selection."
        ),
    }


def main() -> int:
    args = build_parser().parse_args()

    rows = load_prediction_rows(args.predictions_file)
    if rows is None:
        report = _missing_file_report(args)
        write_json(REPO_ROOT / args.output_json, report)
        write_markdown(REPO_ROOT / args.output_md, report)
        print(f"JSON report -> {REPO_ROOT / args.output_json}")
        print(f"Markdown report -> {REPO_ROOT / args.output_md}")
        print(f"Decision -> {report['decision']['label']}")
        return 0

    analysis = analyze_rows(rows)
    baseline_rows = (
        load_prediction_rows(args.baseline_predictions) if args.baseline_predictions else None
    )
    baseline_comparison = compare_to_baseline(baseline_rows, rows)
    decision = decide(args.safety_profile, analysis)

    report = {
        "run_name": args.run_name,
        "predictions_file": args.predictions_file,
        "baseline_predictions_file": args.baseline_predictions,
        "probe_name": args.probe_name,
        "expected_split": args.expected_split,
        "safety_profile": args.safety_profile,
        "row_count": analysis["row_count"],
        "decision": decision,
        "original_metrics": analysis["original_metrics"],
        "composed_metrics": analysis["composed_metrics"],
        "exported_metrics": analysis["exported_metrics"],
        "delta": analysis["delta"],
        "change_counts": analysis["change_counts"],
        "safety_counters": analysis["safety_counters"],
        "introduced_safety_counters": analysis["introduced_safety_counters"],
        "stage39_behavior_counters": analysis["stage39_behavior_counters"],
        "first_changed_rows": analysis["first_changed_rows"],
        "first_unsafe_rows": analysis["first_unsafe_rows"],
        "first_introduced_unsafe_rows": analysis["first_introduced_unsafe_rows"],
        "baseline_comparison": baseline_comparison,
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
