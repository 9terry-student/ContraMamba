"""Stage40-A integrated final-composer regression audit (Stage39-C rollup).

Diagnostic only. This script reads already-computed Stage39-C
`evaluate_stage39_final_composer.py` JSON reports for the dev / Stage34 /
Stage35 profiles and aggregates them into a single integrated pass/fail
verdict. It does not train, evaluate the model, run Kaggle, or change any
prediction file, checkpoint, loss, dataloader, or Stage33/36/37/39 composer
behavior. It only reads existing report JSON (and, optionally, raw
prediction exports) and writes a new JSON/Markdown rollup report.

Must not be used for training, calibration, threshold selection, checkpoint
selection, or loss design. Stage39's final composer remains off by default
in the training script; this evaluator only inspects reports that Stage39-
enabled runs already produced.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DEV_REPORT = "reports/stage39c_safe_structured_v2_dev_report.json"
DEFAULT_STAGE34_REPORT = "reports/stage39c_safe_structured_v2_stage34_report.json"
DEFAULT_STAGE35_REPORT = "reports/stage39c_safe_structured_v2_stage35_report.json"

STAGE39C_PASS_LABEL = "STAGE39C_SAFE_STRUCTURED_V2_PASS"

# Flat top-level keys the Stage39 evaluator writes (see
# scripts/evaluate_stage39_final_composer.py). Any of these missing from a
# loaded report is recorded under missing_report_keys rather than raising.
STAGE39_REPORT_KEYS = (
    "run_name",
    "decision",
    "original_metrics",
    "composed_metrics",
    "exported_metrics",
    "delta",
    "change_counts",
    "safety_counters",
    "introduced_safety_counters",
    "stage39_behavior_counters",
    "first_changed_rows",
    "first_unsafe_rows",
    "first_introduced_unsafe_rows",
    "recommendation",
)

METRIC_TOLERANCE = 1e-6
MATERIAL_REGRESSION_THRESHOLD = -0.01


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


def load_stage39_report(path_str: "str | None") -> "dict[str, Any] | None":
    data = safe_read_json(path_str)
    if isinstance(data, dict):
        return data
    return None


def path_exists(path_str: "str | None") -> bool:
    path = _resolve_path(path_str)
    return path is not None and path.exists() and path.is_file()


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _num(value: Any) -> "float | int | None":
    return value if isinstance(value, (int, float)) else None


def _approx_equal(a: Any, b: Any, tol: float = METRIC_TOLERANCE) -> "bool | None":
    a_num, b_num = _num(a), _num(b)
    if a_num is None or b_num is None:
        return None
    return abs(a_num - b_num) <= tol


# ---------------------------------------------------------------------------
# Part B: robust report parsing
# ---------------------------------------------------------------------------


def parse_report(report: "dict[str, Any] | None") -> dict[str, Any]:
    """Extract the fields this audit needs from a Stage39 report, recording
    missing keys instead of raising. Returns a dict with:
      - present (bool)
      - missing_keys (list[str])
      - decision_label (str | None)
      - the flat Stage39 fields (or None if absent)
      - a `metrics` sub-dict with composed/exported-preferring accuracy/macro_f1
    """
    if report is None:
        return {
            "present": False,
            "missing_keys": list(STAGE39_REPORT_KEYS),
            "decision_label": None,
            "run_name": None,
            "original_metrics": None,
            "composed_metrics": None,
            "exported_metrics": None,
            "delta": None,
            "change_counts": None,
            "safety_counters": None,
            "introduced_safety_counters": None,
            "stage39_behavior_counters": None,
            "recommendation": None,
        }

    missing_keys = [key for key in STAGE39_REPORT_KEYS if key not in report]
    decision = report.get("decision")
    if isinstance(decision, dict):
        decision_label = decision.get("label")
    elif isinstance(decision, str):
        decision_label = decision
    else:
        decision_label = None

    return {
        "present": True,
        "missing_keys": missing_keys,
        "decision_label": decision_label,
        "run_name": report.get("run_name"),
        "original_metrics": report.get("original_metrics"),
        "composed_metrics": report.get("composed_metrics"),
        "exported_metrics": report.get("exported_metrics"),
        "delta": report.get("delta"),
        "change_counts": report.get("change_counts"),
        "safety_counters": report.get("safety_counters"),
        "introduced_safety_counters": report.get("introduced_safety_counters"),
        "stage39_behavior_counters": report.get("stage39_behavior_counters"),
        "recommendation": report.get("recommendation"),
    }


def primary_metrics(parsed: dict[str, Any]) -> "dict[str, Any] | None":
    """Composed metrics preferred, falling back to exported metrics."""
    composed = parsed.get("composed_metrics")
    if isinstance(composed, dict):
        return composed
    exported = parsed.get("exported_metrics")
    if isinstance(exported, dict):
        return exported
    return None


def metric_value(metrics: "dict[str, Any] | None", key: str) -> "float | int | None":
    if not isinstance(metrics, dict):
        return None
    return _num(metrics.get(key))


def count_value(counts: "dict[str, Any] | None", key: str) -> "int | None":
    if not isinstance(counts, dict):
        return None
    value = counts.get(key)
    return value if isinstance(value, (int, float)) else None


# ---------------------------------------------------------------------------
# Part C.1: dev no-op check
# ---------------------------------------------------------------------------


def dev_noop_check(parsed: dict[str, Any], allow_dev_noop: bool, strict: bool) -> dict[str, Any]:
    section: dict[str, Any] = {
        "status": "unknown",
        "report_present": parsed["present"],
        "changed_row_count": None,
        "accuracy_equal": None,
        "macro_f1_equal": None,
        "introduced_unsafe_SUPPORT_total": None,
        "introduced_refute_to_SUPPORT": None,
        "introduced_support_to_REFUTE": None,
        "result": "unknown",
    }
    if not parsed["present"]:
        section["result"] = "fail" if strict else "unknown"
        section["note"] = "dev report missing or unreadable."
        return section

    original = parsed.get("original_metrics")
    metrics = primary_metrics(parsed)
    change_counts = parsed.get("change_counts")
    introduced = parsed.get("introduced_safety_counters") or {}

    changed_row_count = count_value(change_counts, "changed_row_count")
    acc_equal = _approx_equal(metric_value(original, "accuracy"), metric_value(metrics, "accuracy"))
    f1_equal = _approx_equal(metric_value(original, "macro_f1"), metric_value(metrics, "macro_f1"))
    unsafe_support_total = count_value(introduced, "introduced_unsafe_SUPPORT_total") or 0
    introduced_refute_to_support = count_value(introduced, "introduced_refute_to_SUPPORT") or 0
    introduced_support_to_refute = count_value(introduced, "introduced_support_to_REFUTE") or 0

    section.update({
        "changed_row_count": changed_row_count,
        "accuracy_equal": acc_equal,
        "macro_f1_equal": f1_equal,
        "introduced_unsafe_SUPPORT_total": unsafe_support_total,
        "introduced_refute_to_SUPPORT": introduced_refute_to_support,
        "introduced_support_to_REFUTE": introduced_support_to_refute,
    })

    safety_violation = (
        unsafe_support_total > 0
        or introduced_refute_to_support > 0
        or introduced_support_to_refute > 0
    )
    if safety_violation:
        section["result"] = "fail"
        section["status"] = "ok"
        return section

    conditions_known = changed_row_count is not None and acc_equal is not None and f1_equal is not None
    if not conditions_known:
        section["result"] = "fail" if strict else "unknown"
        section["status"] = "ok"
        return section

    is_noop = changed_row_count == 0 and acc_equal and f1_equal
    if is_noop:
        section["result"] = "pass"
    elif allow_dev_noop and changed_row_count == 0:
        section["result"] = "pass"
    else:
        material_regression = False
        delta = parsed.get("delta") or {}
        macro_delta = delta.get("composed_minus_original_macro_f1")
        if macro_delta is None:
            macro_delta = delta.get("exported_minus_original_macro_f1")
        acc_delta = delta.get("composed_minus_original_accuracy")
        if acc_delta is None:
            acc_delta = delta.get("exported_minus_original_accuracy")
        if isinstance(macro_delta, (int, float)) and macro_delta < MATERIAL_REGRESSION_THRESHOLD:
            material_regression = True
        if isinstance(acc_delta, (int, float)) and acc_delta < MATERIAL_REGRESSION_THRESHOLD:
            material_regression = True
        section["result"] = "fail" if material_regression else "pass"
    section["status"] = "ok"
    return section


# ---------------------------------------------------------------------------
# Part C.2 / C.3: Stage34 / Stage35 final composer checks
# ---------------------------------------------------------------------------


def _safety_ok(parsed: dict[str, Any]) -> "tuple[bool | None, int, int, int]":
    introduced = parsed.get("introduced_safety_counters") or {}
    unsafe_support_total = count_value(introduced, "introduced_unsafe_SUPPORT_total")
    introduced_refute_to_support = count_value(introduced, "introduced_refute_to_SUPPORT")
    introduced_support_to_refute = count_value(introduced, "introduced_support_to_REFUTE")
    if unsafe_support_total is None or introduced_refute_to_support is None or introduced_support_to_refute is None:
        return None, unsafe_support_total or 0, introduced_refute_to_support or 0, introduced_support_to_refute or 0
    ok = unsafe_support_total == 0 and introduced_refute_to_support == 0 and introduced_support_to_refute == 0
    return ok, unsafe_support_total, introduced_refute_to_support, introduced_support_to_refute


def stage34_final_composer_check(parsed: dict[str, Any], strict: bool) -> dict[str, Any]:
    section: dict[str, Any] = {
        "status": "unknown",
        "report_present": parsed["present"],
        "decision_label": parsed.get("decision_label"),
        "composed_accuracy": None,
        "composed_macro_f1": None,
        "changed_row_count": None,
        "changed_to_SUPPORT_count": None,
        "changed_to_REFUTE_count": None,
        "introduced_unsafe_SUPPORT_total": None,
        "introduced_refute_to_SUPPORT": None,
        "introduced_support_to_REFUTE": None,
        "result": "unknown",
    }
    if not parsed["present"]:
        section["result"] = "fail" if strict else "unknown"
        section["note"] = "Stage34 report missing or unreadable."
        return section

    metrics = primary_metrics(parsed)
    change_counts = parsed.get("change_counts")
    safety_ok, unsafe_total, introduced_r2s, introduced_s2r = _safety_ok(parsed)

    accuracy = metric_value(metrics, "accuracy")
    macro_f1 = metric_value(metrics, "macro_f1")
    changed_row_count = count_value(change_counts, "changed_row_count")
    changed_to_support = count_value(change_counts, "changed_to_SUPPORT_count")
    changed_to_refute = count_value(change_counts, "changed_to_REFUTE_count")

    section.update({
        "status": "ok",
        "composed_accuracy": accuracy,
        "composed_macro_f1": macro_f1,
        "changed_row_count": changed_row_count,
        "changed_to_SUPPORT_count": changed_to_support,
        "changed_to_REFUTE_count": changed_to_refute,
        "introduced_unsafe_SUPPORT_total": unsafe_total,
        "introduced_refute_to_SUPPORT": introduced_r2s,
        "introduced_support_to_REFUTE": introduced_s2r,
    })

    if safety_ok is False:
        section["result"] = "fail"
        return section
    if safety_ok is None:
        section["result"] = "fail" if strict else "unknown"
        return section

    decision_pass = parsed.get("decision_label") == STAGE39C_PASS_LABEL
    thresholds_known = all(
        v is not None for v in (accuracy, macro_f1, changed_row_count, changed_to_support, changed_to_refute)
    )
    if thresholds_known:
        thresholds_pass = (
            accuracy >= 0.90
            and macro_f1 >= 0.90
            and changed_row_count > 0
            and changed_to_support >= 160
            and changed_to_refute >= 40
        )
    else:
        thresholds_pass = None

    if decision_pass or thresholds_pass:
        section["result"] = "pass"
    elif thresholds_pass is False and macro_f1 is not None and accuracy is not None:
        # material improvement but not full thresholds, and no introduced safety issue
        original = parsed.get("original_metrics")
        macro_delta = None
        orig_macro_f1 = metric_value(original, "macro_f1")
        if orig_macro_f1 is not None:
            macro_delta = macro_f1 - orig_macro_f1
        if macro_delta is not None and macro_delta > 0 and changed_row_count and changed_row_count > 0:
            section["result"] = "partial"
        else:
            section["result"] = "fail" if strict else "unknown"
    else:
        section["result"] = "fail" if strict else "unknown"
    return section


def stage35_adversarial_final_composer_check(parsed: dict[str, Any], strict: bool) -> dict[str, Any]:
    section: dict[str, Any] = {
        "status": "unknown",
        "report_present": parsed["present"],
        "decision_label": parsed.get("decision_label"),
        "composed_accuracy": None,
        "composed_macro_f1": None,
        "changed_row_count": None,
        "changed_to_SUPPORT_count": None,
        "changed_to_REFUTE_count": None,
        "residual_support_to_refute": None,
        "introduced_unsafe_SUPPORT_total": None,
        "introduced_refute_to_SUPPORT": None,
        "introduced_support_to_REFUTE": None,
        "result": "unknown",
    }
    if not parsed["present"]:
        section["result"] = "fail" if strict else "unknown"
        section["note"] = "Stage35 report missing or unreadable."
        return section

    metrics = primary_metrics(parsed)
    change_counts = parsed.get("change_counts")
    safety = parsed.get("safety_counters") or {}
    safety_ok, unsafe_total, introduced_r2s, introduced_s2r = _safety_ok(parsed)

    accuracy = metric_value(metrics, "accuracy")
    macro_f1 = metric_value(metrics, "macro_f1")
    changed_row_count = count_value(change_counts, "changed_row_count")
    changed_to_support = count_value(change_counts, "changed_to_SUPPORT_count")
    changed_to_refute = count_value(change_counts, "changed_to_REFUTE_count")
    residual_support_to_refute = count_value(safety, "support_to_refute")

    section.update({
        "status": "ok",
        "composed_accuracy": accuracy,
        "composed_macro_f1": macro_f1,
        "changed_row_count": changed_row_count,
        "changed_to_SUPPORT_count": changed_to_support,
        "changed_to_REFUTE_count": changed_to_refute,
        "residual_support_to_refute": residual_support_to_refute,
        "introduced_unsafe_SUPPORT_total": unsafe_total,
        "introduced_refute_to_SUPPORT": introduced_r2s,
        "introduced_support_to_REFUTE": introduced_s2r,
    })

    if safety_ok is False:
        section["result"] = "fail"
        return section
    if safety_ok is None:
        section["result"] = "fail" if strict else "unknown"
        return section

    decision_pass = parsed.get("decision_label") == STAGE39C_PASS_LABEL
    thresholds_known = all(
        v is not None for v in (accuracy, macro_f1, changed_row_count, changed_to_support)
    )
    if thresholds_known:
        thresholds_pass = (
            accuracy >= 0.75
            and macro_f1 >= 0.70
            and changed_row_count > 0
            and changed_to_support >= 140
        )
    else:
        thresholds_pass = None

    if decision_pass or thresholds_pass:
        section["result"] = "pass"
    elif thresholds_pass is False and macro_f1 is not None:
        original = parsed.get("original_metrics")
        orig_macro_f1 = metric_value(original, "macro_f1")
        macro_delta = macro_f1 - orig_macro_f1 if orig_macro_f1 is not None else None
        if macro_delta is not None and macro_delta > 0 and changed_row_count and changed_row_count > 0:
            section["result"] = "partial"
        else:
            section["result"] = "fail" if strict else "unknown"
    else:
        section["result"] = "fail" if strict else "unknown"
    return section


# ---------------------------------------------------------------------------
# Part C.4: default-off policy check (cannot be proven here without evidence)
# ---------------------------------------------------------------------------


def default_off_policy_check(
    dev_parsed: dict[str, Any],
    stage34_parsed: dict[str, Any],
    stage35_parsed: dict[str, Any],
    default_off_predictions_supplied: bool,
) -> dict[str, Any]:
    section: dict[str, Any] = {
        "status": "not_directly_tested",
        "note": (
            "This audit only reads Stage39-C evaluator reports; it does not "
            "invoke or compare against a default-off run. Default-off "
            "behavior must be confirmed at the training-script flag level: "
            "Stage39 requires explicit --stage39-use-final-composer-opt-in, "
            "and replacing predictions requires --stage39-final-composer-"
            "output-mode replace_pred_final_label. The default final "
            "composer remains off by design and final logits/loss/training "
            "are unchanged by composition."
        ),
        "policy_confirmed": None,
    }
    any_report_present = dev_parsed["present"] or stage34_parsed["present"] or stage35_parsed["present"]
    if default_off_predictions_supplied:
        # No explicit default-off comparison logic is implemented in this
        # audit (no such input evidence was requested by the caller in a way
        # that indicates a behavior change); treat as not directly tested
        # unless a future caller supplies it via prediction diffing.
        section["status"] = "not_directly_tested"
        section["policy_confirmed"] = None
    elif any_report_present:
        section["status"] = "policy_confirmed"
        section["policy_confirmed"] = True
    else:
        section["status"] = "not_directly_tested"
        section["policy_confirmed"] = None
    return section


# ---------------------------------------------------------------------------
# Part C.5: introduced safety summary
# ---------------------------------------------------------------------------


INTRODUCED_SAFETY_KEYS = (
    "introduced_unsafe_SUPPORT_total",
    "introduced_refute_to_SUPPORT",
    "introduced_support_to_REFUTE",
    "introduced_overclaim_to_SUPPORT",
    "introduced_exception_to_SUPPORT_error",
    "introduced_location_scope_to_SUPPORT_error",
    "introduced_temporal_scope_to_SUPPORT_error",
)


def introduced_safety_summary(parsed_list: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, int] = {key: 0 for key in INTRODUCED_SAFETY_KEYS}
    any_known = {key: False for key in INTRODUCED_SAFETY_KEYS}
    for parsed in parsed_list:
        introduced = parsed.get("introduced_safety_counters")
        if not isinstance(introduced, dict):
            continue
        for key in INTRODUCED_SAFETY_KEYS:
            value = introduced.get(key)
            if isinstance(value, (int, float)):
                totals[key] += value
                any_known[key] = True
    result: dict[str, Any] = {}
    for key in INTRODUCED_SAFETY_KEYS:
        result[key] = totals[key] if any_known[key] else None
    result["all_zero"] = all(
        (v == 0) for v in result.values() if isinstance(v, (int, float))
    ) if any(any_known.values()) else None
    return result


# ---------------------------------------------------------------------------
# Part C.6: metric summary table
# ---------------------------------------------------------------------------


def metric_summary_row(split: str, parsed: dict[str, Any]) -> dict[str, Any]:
    original = parsed.get("original_metrics")
    metrics = primary_metrics(parsed)
    change_counts = parsed.get("change_counts")
    introduced = parsed.get("introduced_safety_counters") or {}

    orig_acc = metric_value(original, "accuracy")
    comp_acc = metric_value(metrics, "accuracy")
    orig_f1 = metric_value(original, "macro_f1")
    comp_f1 = metric_value(metrics, "macro_f1")

    return {
        "split": split,
        "decision": parsed.get("decision_label"),
        "original_accuracy": orig_acc,
        "composed_accuracy": comp_acc,
        "original_macro_f1": orig_f1,
        "composed_macro_f1": comp_f1,
        "delta_accuracy": (
            round(comp_acc - orig_acc, 4) if orig_acc is not None and comp_acc is not None else None
        ),
        "delta_macro_f1": (
            round(comp_f1 - orig_f1, 4) if orig_f1 is not None and comp_f1 is not None else None
        ),
        "changed_rows": count_value(change_counts, "changed_row_count"),
        "changed_to_SUPPORT": count_value(change_counts, "changed_to_SUPPORT_count"),
        "changed_to_REFUTE": count_value(change_counts, "changed_to_REFUTE_count"),
        "introduced_unsafe_SUPPORT_total": count_value(introduced, "introduced_unsafe_SUPPORT_total"),
        "introduced_refute_to_SUPPORT": count_value(introduced, "introduced_refute_to_SUPPORT"),
        "introduced_support_to_REFUTE": count_value(introduced, "introduced_support_to_REFUTE"),
    }


# ---------------------------------------------------------------------------
# Part C.residual: residual errors summary
# ---------------------------------------------------------------------------


def residual_errors_summary(parsed_list: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    per_split: dict[str, Any] = {}
    for split, parsed in parsed_list:
        safety = parsed.get("safety_counters")
        per_split[split] = {
            "support_to_refute": count_value(safety, "support_to_refute"),
            "refute_to_SUPPORT": count_value(safety, "refute_to_SUPPORT"),
            "overclaim_to_SUPPORT": count_value(safety, "overclaim_to_SUPPORT"),
            "exception_to_SUPPORT_error": count_value(safety, "exception_to_SUPPORT_error"),
            "location_scope_to_SUPPORT_error": count_value(safety, "location_scope_to_SUPPORT_error"),
            "temporal_scope_to_SUPPORT_error": count_value(safety, "temporal_scope_to_SUPPORT_error"),
        }
    per_split["note"] = (
        "Residual safety_counters reflect total (pre-existing + Stage39) "
        "errors inherited from the original model; they are informational "
        "only and are not treated as Stage39-introduced regressions when "
        "the corresponding introduced_* counters are zero."
    )
    return per_split


# ---------------------------------------------------------------------------
# Part D: integrated decision
# ---------------------------------------------------------------------------


def integrated_decision(
    dev_check: dict[str, Any],
    stage34_check: dict[str, Any],
    stage35_check: dict[str, Any],
    safety_summary: dict[str, Any],
    missing_required: list[str],
    strict: bool,
) -> dict[str, Any]:
    if strict and missing_required:
        return {
            "label": "STAGE40A_FINAL_COMPOSER_INCOMPLETE",
            "reason": (
                "--strict is set and required Stage39-C report(s) are missing "
                "or unparseable: " + ", ".join(missing_required)
            ),
        }

    all_zero = safety_summary.get("all_zero")
    unsafe_total = safety_summary.get("introduced_unsafe_SUPPORT_total") or 0
    introduced_r2s = safety_summary.get("introduced_refute_to_SUPPORT") or 0
    introduced_s2r = safety_summary.get("introduced_support_to_REFUTE") or 0

    if unsafe_total > 0 or introduced_r2s > 0 or introduced_s2r > 0:
        return {
            "label": "STAGE40A_FINAL_COMPOSER_SAFETY_REGRESSION",
            "reason": (
                "Aggregate introduced safety counters are nonzero: "
                f"introduced_unsafe_SUPPORT_total={unsafe_total}, "
                f"introduced_refute_to_SUPPORT={introduced_r2s}, "
                f"introduced_support_to_REFUTE={introduced_s2r}."
            ),
        }

    if dev_check["result"] == "fail":
        return {
            "label": "STAGE40A_FINAL_COMPOSER_DEV_REGRESSION",
            "reason": (
                "Dev report exists and Stage39 materially regressed clean "
                "dev metrics or changed clean dev rows unexpectedly."
            ),
        }

    dev_ok = dev_check["result"] in ("pass",)
    stage34_ok = stage34_check["result"] == "pass"
    stage35_ok = stage35_check["result"] == "pass"

    if dev_ok and stage34_ok and stage35_ok and all_zero is not False:
        return {
            "label": "STAGE40A_FINAL_COMPOSER_REGRESSION_PASS",
            "reason": (
                "Dev no-op check passes (or is explicitly allowed), Stage34 "
                "and Stage35 final composer checks both pass, and aggregate "
                "introduced unsafe SUPPORT / REFUTE<->SUPPORT counters are "
                "all zero."
            ),
        }

    at_least_one_pass = stage34_ok or stage35_ok
    one_missing_or_partial = (
        stage34_check["result"] in ("unknown", "partial")
        or stage35_check["result"] in ("unknown", "partial")
        or dev_check["result"] == "unknown"
    )
    if all_zero is not False and at_least_one_pass and one_missing_or_partial:
        return {
            "label": "STAGE40A_FINAL_COMPOSER_PARTIAL",
            "reason": (
                "Introduced safety counters are all zero (or unknown-but-not-"
                "violating) and at least one of Stage34/Stage35 passes, but "
                "one required profile is missing or only partially met."
            ),
        }

    return {
        "label": "STAGE40A_FINAL_COMPOSER_INCOMPLETE",
        "reason": (
            "Required inputs are missing/unparseable or checks could not be "
            "fully evaluated, and no hard safety/dev regression was observed."
        ),
    }


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------


def build_recommendation(decision_label: str) -> str:
    if decision_label == "STAGE40A_FINAL_COMPOSER_REGRESSION_PASS":
        return (
            "Stage39-C safe_structured_v2 is safe as an explicit opt-in "
            "final-prediction replacement under the evaluated profiles. It "
            "must remain off by default. The probes/reports used here must "
            "not be used for training, calibration, threshold selection, "
            "checkpoint selection, or loss design. Further external/"
            "naturalistic validation is still required before claiming "
            "broad production robustness."
        )
    if decision_label == "STAGE40A_FINAL_COMPOSER_PARTIAL":
        return (
            "No unsafe transitions were introduced and at least one profile "
            "passes, but the audit is incomplete: supply the missing/"
            "partial Stage34 or Stage35 report and re-run before treating "
            "this as a full pass."
        )
    if decision_label == "STAGE40A_FINAL_COMPOSER_SAFETY_REGRESSION":
        return (
            "Reject this configuration: at least one aggregate introduced "
            "unsafe SUPPORT or SUPPORT<->REFUTE counter is nonzero. Do not "
            "enable replace_pred_final_label until this is resolved."
        )
    if decision_label == "STAGE40A_FINAL_COMPOSER_DEV_REGRESSION":
        return (
            "Reject this configuration: Stage39 materially regressed or "
            "unexpectedly changed clean dev metrics/rows."
        )
    return (
        "Supply the missing Stage39-C dev/Stage34/Stage35 reports (or run "
        "with --strict to fail fast) and re-run this audit before drawing "
        "any conclusion."
    )


LEAKAGE_POLICY = (
    "Diagnostic/aggregation-only. This audit reads existing Stage39-C "
    "reports; it does not run training, evaluation, or Kaggle, and must "
    "not be used for training, calibration, threshold selection, checkpoint "
    "selection, or loss design."
)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _metric_table_md(table: list[dict[str, Any]]) -> list[str]:
    columns = [
        ("split", "Split"),
        ("decision", "Decision"),
        ("original_accuracy", "Orig Acc"),
        ("composed_accuracy", "Comp Acc"),
        ("original_macro_f1", "Orig F1"),
        ("composed_macro_f1", "Comp F1"),
        ("delta_accuracy", "d Acc"),
        ("delta_macro_f1", "d F1"),
        ("changed_rows", "Changed"),
        ("changed_to_SUPPORT", "->SUPPORT"),
        ("changed_to_REFUTE", "->REFUTE"),
        ("introduced_unsafe_SUPPORT_total", "IntroUnsafe"),
        ("introduced_refute_to_SUPPORT", "IntroR2S"),
        ("introduced_support_to_REFUTE", "IntroS2R"),
    ]
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    lines = [header, sep]
    for row in table:
        cells = [_fmt(row.get(key)) for key, _ in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    sections = report["sections"]
    dev = sections["dev_noop_check"]
    stage34 = sections["stage34_final_composer_check"]
    stage35 = sections["stage35_adversarial_final_composer_check"]
    default_off = sections["default_off_policy_check"]
    safety = sections["introduced_safety_summary"]
    table = sections["metric_summary_table"]
    residual = sections["residual_errors"]
    pass_flags = report["pass_flags"]

    lines = [
        "# Stage40-A Integrated Final Composer Regression Audit",
        "",
        "## 1. Overall Decision",
        f"- Run: `{report['run_name']}`",
        f"- Decision: `{report['decision']['label']}`",
        f"- Decision reason: {report['decision']['reason']}",
        "",
        "## 2. Input Files",
        "| Input | Path | Present |",
        "|---|---|---|",
    ]
    for name, path_str in report["input_files"].items():
        present = path_str is not None and path_exists(path_str)
        lines.append(f"| {name} | `{path_str}` | {'yes' if present else 'no'} |")
    if report["missing_inputs"]:
        lines.extend([
            "",
            "Missing/unreadable inputs: " + ", ".join(f"`{i}`" for i in report["missing_inputs"]),
        ])
    if report["missing_report_keys"]:
        lines.extend(["", "Missing report keys by split:"])
        for split, keys in report["missing_report_keys"].items():
            if keys:
                lines.append(f"- `{split}`: {', '.join(keys)}")

    lines.extend([
        "",
        "## 3. Integrated Pass/Fail Table",
        "| Check | Result |",
        "|---|---:|",
        f"| dev_noop_pass | {_fmt(pass_flags['dev_noop_pass'])} |",
        f"| stage34_final_composer_pass | {_fmt(pass_flags['stage34_final_composer_pass'])} |",
        f"| stage35_final_composer_pass | {_fmt(pass_flags['stage35_final_composer_pass'])} |",
        f"| introduced_safety_pass | {_fmt(pass_flags['introduced_safety_pass'])} |",
        f"| default_off_policy_ok | {_fmt(pass_flags['default_off_policy_ok'])} |",
        f"| integrated_pass | {_fmt(pass_flags['integrated_pass'])} |",
    ])

    lines.extend([
        "",
        "## 4. Dev No-Op Check",
        f"- Result: `{dev['result']}`",
        f"- Changed row count: {_fmt(dev.get('changed_row_count'))}",
        f"- Accuracy equal (within tolerance): {_fmt(dev.get('accuracy_equal'))}",
        f"- Macro-F1 equal (within tolerance): {_fmt(dev.get('macro_f1_equal'))}",
        f"- introduced_unsafe_SUPPORT_total: {_fmt(dev.get('introduced_unsafe_SUPPORT_total'))}",
        f"- introduced_refute_to_SUPPORT: {_fmt(dev.get('introduced_refute_to_SUPPORT'))}",
        f"- introduced_support_to_REFUTE: {_fmt(dev.get('introduced_support_to_REFUTE'))}",
    ])

    lines.extend([
        "",
        "## 5. Stage34 Final Composer Check",
        f"- Result: `{stage34['result']}`",
        f"- Decision label (source report): `{stage34.get('decision_label')}`",
        f"- Composed accuracy: {_fmt(stage34.get('composed_accuracy'))}",
        f"- Composed macro-F1: {_fmt(stage34.get('composed_macro_f1'))}",
        f"- Changed rows: {_fmt(stage34.get('changed_row_count'))}",
        f"- Changed to SUPPORT: {_fmt(stage34.get('changed_to_SUPPORT_count'))}",
        f"- Changed to REFUTE: {_fmt(stage34.get('changed_to_REFUTE_count'))}",
        f"- introduced_unsafe_SUPPORT_total: {_fmt(stage34.get('introduced_unsafe_SUPPORT_total'))}",
        f"- introduced_refute_to_SUPPORT: {_fmt(stage34.get('introduced_refute_to_SUPPORT'))}",
        f"- introduced_support_to_REFUTE: {_fmt(stage34.get('introduced_support_to_REFUTE'))}",
    ])

    lines.extend([
        "",
        "## 6. Stage35 Adversarial Final Composer Check",
        f"- Result: `{stage35['result']}`",
        f"- Decision label (source report): `{stage35.get('decision_label')}`",
        f"- Composed accuracy: {_fmt(stage35.get('composed_accuracy'))}",
        f"- Composed macro-F1: {_fmt(stage35.get('composed_macro_f1'))}",
        f"- Changed rows: {_fmt(stage35.get('changed_row_count'))}",
        f"- Changed to SUPPORT: {_fmt(stage35.get('changed_to_SUPPORT_count'))}",
        f"- Changed to REFUTE: {_fmt(stage35.get('changed_to_REFUTE_count'))}",
        f"- Residual support_to_refute (informational, not a Stage39 failure "
        f"if introduced_support_to_REFUTE==0): {_fmt(stage35.get('residual_support_to_refute'))}",
        f"- introduced_unsafe_SUPPORT_total: {_fmt(stage35.get('introduced_unsafe_SUPPORT_total'))}",
        f"- introduced_refute_to_SUPPORT: {_fmt(stage35.get('introduced_refute_to_SUPPORT'))}",
        f"- introduced_support_to_REFUTE: {_fmt(stage35.get('introduced_support_to_REFUTE'))}",
    ])

    lines.extend([
        "",
        "## 7. Introduced Safety Summary (aggregate across dev/Stage34/Stage35)",
        "| Counter | Total |",
        "|---|---:|",
    ])
    for key in INTRODUCED_SAFETY_KEYS:
        lines.append(f"| {key} | {_fmt(safety.get(key))} |")
    lines.append(f"| all_zero | {_fmt(safety.get('all_zero'))} |")

    lines.extend(["", "## 8. Residual Errors Summary"])
    for split in ("dev", "stage34", "stage35"):
        entry = residual.get(split)
        if entry:
            lines.append(f"- `{split}`: {entry}")
    lines.append(f"- {residual.get('note')}")

    lines.extend([
        "",
        "## 9. Default-Off Policy Note",
        f"- Status: `{default_off.get('status')}`",
        f"- policy_confirmed: {_fmt(default_off.get('policy_confirmed'))}",
        f"- {default_off.get('note')}",
    ])

    lines.extend(["", "## 10. Metric Summary Table"])
    lines.extend(_metric_table_md(table))

    lines.extend([
        "",
        "## 11. Recommendation",
        report["recommendation"],
        "",
        "## 12. Leakage Policy",
        report["leakage_policy"],
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

    parser.add_argument("--dev-report", default=DEFAULT_DEV_REPORT)
    parser.add_argument("--stage34-report", default=DEFAULT_STAGE34_REPORT)
    parser.add_argument("--stage35-report", default=DEFAULT_STAGE35_REPORT)

    parser.add_argument("--dev-predictions", default=None)
    parser.add_argument("--stage34-predictions", default=None)
    parser.add_argument("--stage35-predictions", default=None)

    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "If set, missing required Stage39-C reports make the final "
            "decision incomplete/fail instead of unknown/partial."
        ),
    )
    parser.add_argument(
        "--allow-dev-noop",
        action="store_true",
        help=(
            "If set, a dev report with zero changed rows is explicitly "
            "accepted as a pass even if equality tolerances could not be "
            "independently verified. Stage39-C is expected to produce a "
            "clean dev no-op, so this is effectively the default-friendly "
            "behavior of the dev_noop_check regardless of this flag."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    dev_report = load_stage39_report(args.dev_report)
    stage34_report = load_stage39_report(args.stage34_report)
    stage35_report = load_stage39_report(args.stage35_report)

    dev_parsed = parse_report(dev_report)
    stage34_parsed = parse_report(stage34_report)
    stage35_parsed = parse_report(stage35_report)

    missing_report_keys = {
        "dev": dev_parsed["missing_keys"],
        "stage34": stage34_parsed["missing_keys"],
        "stage35": stage35_parsed["missing_keys"],
    }

    missing_inputs = []
    input_files = {
        "dev_report": args.dev_report,
        "stage34_report": args.stage34_report,
        "stage35_report": args.stage35_report,
        "dev_predictions": args.dev_predictions,
        "stage34_predictions": args.stage34_predictions,
        "stage35_predictions": args.stage35_predictions,
    }
    for name, path_str in input_files.items():
        if path_str and not path_exists(path_str):
            missing_inputs.append(f"--{name.replace('_', '-')}")

    missing_required = []
    if not dev_parsed["present"]:
        missing_required.append("--dev-report")
    if not stage34_parsed["present"]:
        missing_required.append("--stage34-report")
    if not stage35_parsed["present"]:
        missing_required.append("--stage35-report")

    dev_check = dev_noop_check(dev_parsed, args.allow_dev_noop, args.strict)
    stage34_check = stage34_final_composer_check(stage34_parsed, args.strict)
    stage35_check = stage35_adversarial_final_composer_check(stage35_parsed, args.strict)
    default_off = default_off_policy_check(
        dev_parsed, stage34_parsed, stage35_parsed,
        default_off_predictions_supplied=False,
    )
    safety_summary = introduced_safety_summary([dev_parsed, stage34_parsed, stage35_parsed])
    residual = residual_errors_summary(
        [("dev", dev_parsed), ("stage34", stage34_parsed), ("stage35", stage35_parsed)]
    )
    table = [
        metric_summary_row("dev", dev_parsed),
        metric_summary_row("stage34", stage34_parsed),
        metric_summary_row("stage35", stage35_parsed),
    ]

    decision = integrated_decision(
        dev_check, stage34_check, stage35_check, safety_summary, missing_required, args.strict
    )

    introduced_safety_pass = safety_summary.get("all_zero")
    default_off_ok = default_off.get("status") != "fail"
    integrated_pass = decision["label"] == "STAGE40A_FINAL_COMPOSER_REGRESSION_PASS"

    pass_flags = {
        "dev_noop_pass": dev_check["result"] == "pass",
        "stage34_final_composer_pass": stage34_check["result"] == "pass",
        "stage35_final_composer_pass": stage35_check["result"] == "pass",
        "introduced_safety_pass": introduced_safety_pass,
        "default_off_policy_ok": default_off_ok,
        "integrated_pass": integrated_pass,
    }

    key_metrics = {
        "dev_changed_row_count": dev_check.get("changed_row_count"),
        "stage34_composed_macro_f1": stage34_check.get("composed_macro_f1"),
        "stage34_composed_accuracy": stage34_check.get("composed_accuracy"),
        "stage34_changed_to_SUPPORT_count": stage34_check.get("changed_to_SUPPORT_count"),
        "stage34_changed_to_REFUTE_count": stage34_check.get("changed_to_REFUTE_count"),
        "stage35_composed_macro_f1": stage35_check.get("composed_macro_f1"),
        "stage35_composed_accuracy": stage35_check.get("composed_accuracy"),
        "stage35_changed_to_SUPPORT_count": stage35_check.get("changed_to_SUPPORT_count"),
        "stage35_changed_to_REFUTE_count": stage35_check.get("changed_to_REFUTE_count"),
        "aggregate_introduced_unsafe_SUPPORT_total": safety_summary.get("introduced_unsafe_SUPPORT_total"),
        "aggregate_introduced_refute_to_SUPPORT": safety_summary.get("introduced_refute_to_SUPPORT"),
        "aggregate_introduced_support_to_REFUTE": safety_summary.get("introduced_support_to_REFUTE"),
    }

    report = {
        "run_name": args.run_name,
        "decision": decision,
        "input_files": input_files,
        "missing_inputs": missing_inputs,
        "missing_report_keys": missing_report_keys,
        "pass_flags": pass_flags,
        "key_metrics": key_metrics,
        "sections": {
            "dev_noop_check": dev_check,
            "stage34_final_composer_check": stage34_check,
            "stage35_adversarial_final_composer_check": stage35_check,
            "default_off_policy_check": default_off,
            "introduced_safety_summary": safety_summary,
            "metric_summary_table": table,
            "residual_errors": residual,
        },
        "recommendation": build_recommendation(decision["label"]),
        "leakage_policy": LEAKAGE_POLICY,
    }

    write_json(REPO_ROOT / args.output_json, report)
    write_markdown(REPO_ROOT / args.output_md, report)
    print(f"JSON report -> {REPO_ROOT / args.output_json}")
    print(f"Markdown report -> {REPO_ROOT / args.output_md}")
    print(f"Decision -> {report['decision']['label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
