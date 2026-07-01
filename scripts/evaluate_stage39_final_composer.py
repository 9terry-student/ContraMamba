"""Stage39-A opt-in final composer evaluator.

Diagnostic only. This script reads an already-exported prediction file (one
run of scripts/train_controlled_v6b_minimal.py with Stage39 flags on) and
reports whether the Stage39-A opt-in final composer is safe to use as an
explicit final-prediction replacement. It does not train, evaluate the
model, run Kaggle, or change any prediction file it reads.

Must not be used for training, calibration, threshold selection, loss
computation, or checkpoint selection. Stage39 itself is off by default in
the training script; this evaluator only inspects what a Stage39-enabled
export already produced.
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


def prediction_metrics_safe(
    golds: "list[str | None]", preds: "list[str | None]"
) -> "dict[str, Any] | None":
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


def prediction_counts(labels: "list[str | None]") -> dict[str, int]:
    counter = Counter(label for label in labels if label is not None)
    return dict(sorted(counter.items()))


# ---------------------------------------------------------------------------
# Row-level analysis
# ---------------------------------------------------------------------------


def analyze_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    golds = [get_gold_label(row) for row in rows]
    originals = [get_original_final_label(row) for row in rows]
    composed = [get_composed_final_label(row) for row in rows]
    exported = [get_exported_final_label(row) for row in rows]

    original_metrics = prediction_metrics_safe(golds, originals)
    composed_metrics = prediction_metrics_safe(golds, composed)
    exported_metrics = prediction_metrics_safe(golds, exported)

    macro_f1_delta_composed_original = None
    accuracy_delta_composed_original = None
    if original_metrics is not None and composed_metrics is not None:
        macro_f1_delta_composed_original = round(
            composed_metrics["macro_f1"] - original_metrics["macro_f1"], 4
        )
        accuracy_delta_composed_original = round(
            composed_metrics["accuracy"] - original_metrics["accuracy"], 4
        )

    changed_rows: list[dict[str, Any]] = []
    unsafe_rows: list[dict[str, Any]] = []

    change_counts = {
        "changed_row_count": 0,
        "changed_to_SUPPORT": 0,
        "changed_to_REFUTE": 0,
        "changed_to_NOT_ENTITLED": 0,
        "REFUTE_to_SUPPORT": 0,
        "NOT_ENTITLED_to_SUPPORT": 0,
        "SUPPORT_to_NOT_ENTITLED": 0,
        "SUPPORT_to_REFUTE": 0,
    }
    safety_counters = {
        "overclaim_to_SUPPORT": 0,
        "exception_to_SUPPORT_error": 0,
        "location_scope_to_SUPPORT_error": 0,
        "temporal_scope_to_SUPPORT_error": 0,
        "refute_to_SUPPORT": 0,
        "support_to_refute": 0,
    }
    behavior_counters = {
        "stage39_enabled_count": 0,
        "stage39_changed_count": 0,
        "stage39_support_composed_count": 0,
        "stage39_refute_composed_count": 0,
        "stage39_not_entitled_composed_count": 0,
        "stage39_blocked_by_stage36_count": 0,
        "stage39_blocked_by_refute_to_support_guard_count": 0,
        "stage39_blocked_by_stage37_from_refute_guard_count": 0,
        "stage39_missing_source_count": 0,
    }
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
                change_counts["changed_to_SUPPORT"] += 1
                behavior_counters["stage39_support_composed_count"] += 1
            elif candidate == "REFUTE":
                change_counts["changed_to_REFUTE"] += 1
                behavior_counters["stage39_refute_composed_count"] += 1
            elif candidate == "NOT_ENTITLED":
                change_counts["changed_to_NOT_ENTITLED"] += 1
                behavior_counters["stage39_not_entitled_composed_count"] += 1

            if original == "REFUTE" and candidate == "SUPPORT":
                change_counts["REFUTE_to_SUPPORT"] += 1
            if original == "NOT_ENTITLED" and candidate == "SUPPORT":
                change_counts["NOT_ENTITLED_to_SUPPORT"] += 1
            if original == "SUPPORT" and candidate == "NOT_ENTITLED":
                change_counts["SUPPORT_to_NOT_ENTITLED"] += 1
            if original == "SUPPORT" and candidate == "REFUTE":
                change_counts["SUPPORT_to_REFUTE"] += 1

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

    behavior_counters["stage39_composer_action_counts"] = dict(composer_action_counts)
    behavior_counters["stage39_composer_reason_counts"] = dict(composer_reason_counts)

    return {
        "row_count": len(rows),
        "original_final_metrics": original_metrics,
        "composed_final_metrics": composed_metrics,
        "exported_final_metrics": exported_metrics,
        "macro_f1_delta_composed_original": macro_f1_delta_composed_original,
        "accuracy_delta_composed_original": accuracy_delta_composed_original,
        "original_prediction_counts": prediction_counts(originals),
        "composed_prediction_counts": prediction_counts(composed),
        "exported_prediction_counts": prediction_counts(exported),
        "change_counts": change_counts,
        "safety_counters": safety_counters,
        "stage39_behavior_counters": behavior_counters,
        "changed_rows": changed_rows,
        "unsafe_rows": unsafe_rows,
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


def decide(
    profile: str,
    analysis: dict[str, Any],
) -> dict[str, Any]:
    safety = analysis["safety_counters"]
    unsafe_support_leak = any(
        (safety.get(key) or 0) > 0
        for key in (
            "overclaim_to_SUPPORT",
            "exception_to_SUPPORT_error",
            "location_scope_to_SUPPORT_error",
            "temporal_scope_to_SUPPORT_error",
            "refute_to_SUPPORT",
        )
    )
    if unsafe_support_leak:
        return {
            "label": "STAGE39A_FINAL_COMPOSER_SAFETY_REGRESSION",
            "reason": (
                "One or more unsafe-SUPPORT leakage counters (overclaim, "
                "exception, location-scope, temporal-scope, or refute-to-"
                "SUPPORT) is nonzero after composition."
            ),
        }

    original_metrics = analysis["original_final_metrics"]
    composed_metrics = analysis["composed_final_metrics"]
    exported_metrics = analysis["exported_final_metrics"]
    macro_delta = analysis["macro_f1_delta_composed_original"]
    accuracy_delta = analysis["accuracy_delta_composed_original"]

    if profile == "dev":
        material_regression = (macro_delta is not None and macro_delta < -0.01) or (
            accuracy_delta is not None and accuracy_delta < -0.01
        )
        if material_regression:
            return {
                "label": "STAGE39A_FINAL_COMPOSER_DEV_REGRESSION",
                "reason": (
                    "safety_profile=dev and composed/exported macro-F1 or "
                    "accuracy regressed materially versus the original final "
                    "label."
                ),
            }

    behavior = analysis["stage39_behavior_counters"]
    changed_count = behavior.get("stage39_changed_count", 0)
    enabled_count = behavior.get("stage39_enabled_count", 0)
    if enabled_count > 0 and changed_count == 0:
        return {
            "label": "STAGE39A_FINAL_COMPOSER_TOO_WEAK",
            "reason": (
                "Stage39 was enabled on at least one row, but no row's final "
                "label was actually changed by composition."
            ),
        }

    explainable_actions = (
        behavior.get("stage39_support_composed_count", 0)
        + behavior.get("stage39_refute_composed_count", 0)
        + behavior.get("stage39_not_entitled_composed_count", 0)
    )
    changed_rows_explainable = explainable_actions == changed_count

    support_to_refute = safety.get("support_to_refute") or 0
    support_to_refute_ok = support_to_refute == 0
    if profile == "stage35":
        support_to_refute_ok = support_to_refute <= 1

    if profile == "stage34":
        mf1 = (composed_metrics or exported_metrics or {}).get("macro_f1")
        safety_zero = all((safety.get(key) or 0) == 0 for key in safety)
        if (
            mf1 is not None
            and mf1 >= 0.94
            and safety_zero
            and changed_rows_explainable
        ):
            return {
                "label": "STAGE39A_FINAL_COMPOSER_OPT_IN_PASS",
                "reason": (
                    "Stage34 composed/exported macro-F1 meets the >=0.94 "
                    "target, all safety counters are zero, and changed rows "
                    "are fully explained by Stage39 composer actions."
                ),
            }
        return {
            "label": "STAGE39A_FINAL_COMPOSER_DIAGNOSTIC_ONLY",
            "reason": (
                "Stage34 profile thresholds are not conclusively met (macro-"
                "F1, safety counters, or explainability); treat as diagnostic "
                "only until re-validated."
            ),
        }

    if profile == "stage35":
        mf1 = (composed_metrics or exported_metrics or {}).get("macro_f1")
        support_shadow_support = (
            behavior.get("stage39_support_composed_count")
            if enabled_count
            else None
        )
        recovery_ok = support_shadow_support is None or support_shadow_support >= 140
        if (
            mf1 is not None
            and mf1 >= 0.70
            and recovery_ok
            and support_to_refute_ok
            and changed_rows_explainable
        ):
            return {
                "label": "STAGE39A_FINAL_COMPOSER_OPT_IN_PASS",
                "reason": (
                    "Stage35 composed/exported macro-F1 meets the >=0.70 "
                    "target, SUPPORT recovery meets the >=140 target when "
                    "available, unsafe SUPPORT counters are zero, and "
                    "changed rows are explainable."
                ),
            }
        return {
            "label": "STAGE39A_FINAL_COMPOSER_DIAGNOSTIC_ONLY",
            "reason": (
                "Stage35 profile thresholds are not conclusively met (macro-"
                "F1, recovery, or explainability); treat as diagnostic only "
                "until re-validated."
            ),
        }

    if profile == "dev":
        if (
            (macro_delta is None or macro_delta >= -0.01)
            and (accuracy_delta is None or accuracy_delta >= -0.01)
            and (safety.get("refute_to_SUPPORT") or 0) == 0
            and support_to_refute_ok
            and changed_rows_explainable
        ):
            return {
                "label": "STAGE39A_FINAL_COMPOSER_OPT_IN_PASS",
                "reason": (
                    "dev profile: no material accuracy/macro-F1 regression, "
                    "refute-to-SUPPORT is zero, and changed rows are "
                    "explainable."
                ),
            }
        return {
            "label": "STAGE39A_FINAL_COMPOSER_DIAGNOSTIC_ONLY",
            "reason": (
                "dev profile thresholds are not conclusively met; treat as "
                "diagnostic only until re-validated."
            ),
        }

    # generic profile: report metrics without strict pass/fail unless enough
    # fields exist.
    if (
        original_metrics is not None
        and composed_metrics is not None
        and macro_delta is not None
        and macro_delta >= 0
        and support_to_refute_ok
        and (safety.get("refute_to_SUPPORT") or 0) == 0
        and changed_rows_explainable
    ):
        return {
            "label": "STAGE39A_FINAL_COMPOSER_OPT_IN_PASS",
            "reason": (
                "generic profile: composed macro-F1 does not regress versus "
                "original, unsafe SUPPORT/REFUTE counters are zero, and "
                "changed rows are explainable."
            ),
        }
    return {
        "label": "STAGE39A_FINAL_COMPOSER_DIAGNOSTIC_ONLY",
        "reason": (
            "generic profile: insufficient fields or mixed evidence to reach "
            "a strict pass verdict."
        ),
    }


def build_recommendation(decision_label: str) -> str:
    if decision_label == "STAGE39A_FINAL_COMPOSER_OPT_IN_PASS":
        return (
            "The Stage39-A opt-in final composer is safe to use as an "
            "explicit final-prediction replacement under the evaluated "
            "policy/profile. It remains off by default; enabling it in "
            "production requires an explicit --stage39-use-final-composer-"
            "opt-in (and, to replace predictions, --stage39-final-composer-"
            "output-mode replace_pred_final_label) decision outside this "
            "script."
        )
    if decision_label == "STAGE39A_FINAL_COMPOSER_SAFETY_REGRESSION":
        return (
            "Reject this composer configuration: unsafe SUPPORT leakage was "
            "detected. Tighten the policy/guards or keep Stage39 export_only."
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


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    analysis = report["analysis"]
    original_metrics = analysis["original_final_metrics"] or {}
    composed_metrics = analysis["composed_final_metrics"] or {}
    exported_metrics = analysis["exported_final_metrics"] or {}
    change_counts = analysis["change_counts"]
    safety = analysis["safety_counters"]
    behavior = analysis["stage39_behavior_counters"]

    lines = [
        "# Stage39-A Final Composer Opt-In Validation",
        "",
        "## 1. Overall Decision",
        f"- Run: `{report['run_name']}`",
        f"- Safety profile: `{report['safety_profile']}`",
        f"- Decision: `{report['decision']['label']}`",
        f"- Decision reason: {report['decision']['reason']}",
        "",
        "## 2. Input File",
        f"- Predictions file: `{report['predictions_file']}`",
        f"- Row count: {analysis['row_count']}",
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
        "## 3. Original vs Composed vs Exported Final Metrics",
        "| Metric | Original | Composed | Exported |",
        "|---|---:|---:|---:|",
        f"| accuracy | {_fmt(original_metrics.get('accuracy'))} | {_fmt(composed_metrics.get('accuracy'))} | {_fmt(exported_metrics.get('accuracy'))} |",
        f"| macro_f1 | {_fmt(original_metrics.get('macro_f1'))} | {_fmt(composed_metrics.get('macro_f1'))} | {_fmt(exported_metrics.get('macro_f1'))} |",
        f"| n | {_fmt(original_metrics.get('n'))} | {_fmt(composed_metrics.get('n'))} | {_fmt(exported_metrics.get('n'))} |",
        "",
        f"- macro_f1 delta (composed-original): {_fmt(analysis['macro_f1_delta_composed_original'])}",
        f"- accuracy delta (composed-original): {_fmt(analysis['accuracy_delta_composed_original'])}",
        f"- Original prediction counts: {analysis['original_prediction_counts']}",
        f"- Composed prediction counts: {analysis['composed_prediction_counts']}",
        f"- Exported prediction counts: {analysis['exported_prediction_counts']}",
    ])

    lines.extend([
        "",
        "## 4. Change Counts",
        "| Counter | Value |",
        "|---|---:|",
    ])
    for key, value in change_counts.items():
        lines.append(f"| {key} | {value} |")

    lines.extend([
        "",
        "## 5. Safety Counters",
        "| Counter | Value |",
        "|---|---:|",
    ])
    for key, value in safety.items():
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

    lines.extend(["", "## 7. First 30 Changed Rows"])
    changed_rows = analysis["changed_rows"]
    if changed_rows:
        lines.extend([
            "| Row ID | Gold | Original | Composed | Action | Reason |",
            "|---|---|---|---|---|---|",
        ])
        for row in changed_rows:
            lines.append(
                f"| {row['row_id']} | {_fmt(row['gold'])} | {_fmt(row['original_final_label'])} | "
                f"{_fmt(row['composed_final_label'])} | {_fmt(row['composer_action'])} | {_fmt(row['composer_reason'])} |"
            )
    else:
        lines.append("None.")

    lines.extend(["", "## 8. First 30 Unsafe Rows"])
    unsafe_rows = analysis["unsafe_rows"]
    if unsafe_rows:
        lines.extend([
            "| Row ID | Group | Gold | Original | Composed | Action | Reason |",
            "|---|---|---|---|---|---|---|",
        ])
        for row in unsafe_rows:
            lines.append(
                f"| {row['row_id']} | {_fmt(row['group'])} | {_fmt(row['gold'])} | {_fmt(row['original_final_label'])} | "
                f"{_fmt(row['composed_final_label'])} | {_fmt(row['composer_action'])} | {_fmt(row['composer_reason'])} |"
            )
    else:
        lines.append("None.")

    lines.extend([
        "",
        "## 9. Recommendation",
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


def main() -> int:
    args = build_parser().parse_args()

    rows = load_prediction_rows(args.predictions_file)
    if rows is None:
        report = {
            "run_name": args.run_name,
            "predictions_file": args.predictions_file,
            "baseline_predictions_file": args.baseline_predictions,
            "probe_name": args.probe_name,
            "expected_split": args.expected_split,
            "safety_profile": args.safety_profile,
            "decision": {
                "label": "STAGE39A_FINAL_COMPOSER_DIAGNOSTIC_ONLY",
                "reason": (
                    f"--predictions-file '{args.predictions_file}' was missing, "
                    "unreadable, or not in a recognized row format."
                ),
            },
            "analysis": None,
            "baseline_comparison": None,
            "recommendation": build_recommendation("STAGE39A_FINAL_COMPOSER_DIAGNOSTIC_ONLY"),
        }
        write_json(REPO_ROOT / args.output_json, report)
        Path(REPO_ROOT / args.output_md).parent.mkdir(parents=True, exist_ok=True)
        (REPO_ROOT / args.output_md).write_text(
            "# Stage39-A Final Composer Opt-In Validation\n\n"
            f"Predictions file `{args.predictions_file}` was missing, unreadable, "
            "or not in a recognized row format. No analysis was performed.\n",
            encoding="utf-8",
        )
        print(f"JSON report -> {REPO_ROOT / args.output_json}")
        print(f"Markdown report -> {REPO_ROOT / args.output_md}")
        print(f"Decision -> {report['decision']['label']}")
        return 0

    analysis = analyze_rows(rows)
    baseline_rows = load_prediction_rows(args.baseline_predictions) if args.baseline_predictions else None
    baseline_comparison = compare_to_baseline(baseline_rows, rows)
    decision = decide(args.safety_profile, analysis)

    report = {
        "run_name": args.run_name,
        "predictions_file": args.predictions_file,
        "baseline_predictions_file": args.baseline_predictions,
        "probe_name": args.probe_name,
        "expected_split": args.expected_split,
        "safety_profile": args.safety_profile,
        "decision": decision,
        "analysis": analysis,
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
