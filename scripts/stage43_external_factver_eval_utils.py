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
    "Stage43-B1 external fact-verification data is evaluation-only. It is not "
    "used for training, calibration, threshold selection, checkpoint selection, "
    "loss design, model selection, or composer behavior changes."
)

STAGE43C1_REQUIRED_COMPOSER_FIELDS = (
    "stage37_final_shadow_label",
    "stage36_final_shadow_label",
    "stage32_shadow_label",
    "stage36_support_blocker_fired",
    "stage37_recovered_from_label",
    "stage33_structured_coverage_reason",
    "stage33_structured_coverage_route",
    "stage33_structured_coverage_label",
    "stage33_conditional_override_type",
    "stage36_conditional_override_type",
    "stage37_conditional_override_type",
)

STAGE43C1_FIELD_PRESENCE_KEYS = (
    "stage32_shadow_label",
    "stage32_shadow_reason",
    "stage33_structured_coverage_reason",
    "stage33_structured_coverage_route",
    "stage33_structured_coverage_label",
    "stage33_conditional_override_type",
    "stage36_final_shadow_label",
    "stage36_support_blocker_fired",
    "stage36_support_blocker_reasons",
    "stage36_conditional_override_type",
    "stage37_final_shadow_label",
    "stage37_safe_recovery_fired",
    "stage37_safe_recovery_reasons",
    "stage37_recovered_from_label",
    "stage37_conditional_override_type",
    "stage39_source_shadow_label",
    "stage39_composed_final_label",
    "stage39_composer_action",
    "stage39_composer_reason",
    "stage39_blocked_by_missing_source",
)

STAGE43C2_REQUIRED_SHADOW_FIELDS = (
    "stage32_shadow_label",
    "stage32_shadow_reason",
    "stage33_structured_coverage_label",
    "stage33_structured_coverage_reason",
    "stage33_structured_coverage_route",
    "stage36_final_shadow_label",
    "stage36_support_blocker_fired",
    "stage36_support_blocker_reasons",
    "stage37_final_shadow_label",
    "stage37_safe_recovery_fired",
    "stage37_safe_recovery_reasons",
    "stage37_recovered_from_label",
    "stage39_source_shadow_label",
    "stage39_composed_final_label",
    "stage39_composer_action",
    "stage39_composer_reason",
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


def _percentile(sorted_values: list[float], percentile: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return round(float(sorted_values[0]), 6)
    pos = (len(sorted_values) - 1) * percentile
    lower = int(pos)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = pos - lower
    value = sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
    return round(float(value), 6)


def summarize_numeric(values: list[float | int]) -> dict[str, float | int | None]:
    clean = sorted(float(value) for value in values if value is not None)
    if not clean:
        return {
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    return {
        "min": round(clean[0], 6),
        "p25": _percentile(clean, 0.25),
        "median": _percentile(clean, 0.50),
        "p75": _percentile(clean, 0.75),
        "p90": _percentile(clean, 0.90),
        "p95": _percentile(clean, 0.95),
        "max": round(clean[-1], 6),
    }


def _has_present_value(row: dict[str, Any], key: str) -> bool:
    return key in row and row.get(key) is not None


def _build_stage43c2_diagnostics(
    *,
    prediction_rows: list[dict[str, Any]],
    path_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    enabled = bool(path_diagnostics.get("stage43c2_shadow_export_enabled", False))
    row_count = len(prediction_rows)
    present_counts = {key: 0 for key in STAGE43C2_REQUIRED_SHADOW_FIELDS}
    non_null_counts = {key: 0 for key in STAGE43C2_REQUIRED_SHADOW_FIELDS}
    blocked_reasons: Counter[str] = Counter()
    application_count = 0
    blocked_count = 0
    for row in prediction_rows:
        for key in present_counts:
            if key in row:
                present_counts[key] += 1
            if key in row and row.get(key) is not None:
                non_null_counts[key] += 1
        action = str(row.get("stage39_composer_action") or "")
        reason = str(row.get("stage39_composer_reason") or "")
        if action.startswith("composed_to_"):
            application_count += 1
        elif action.startswith("blocked_by_"):
            blocked_count += 1
            blocked_reasons[reason or action] += 1

    missing_dependencies: list[str] = []
    if not enabled:
        missing_dependencies.append("stage43_external_enable_shadow_export_disabled")
    if enabled and row_count == 0:
        missing_dependencies.append("no_prediction_rows_available_for_shadow_export")
    if enabled and row_count:
        for key, count in present_counts.items():
            if count < row_count:
                missing_dependencies.append(f"missing:{key}")
        if non_null_counts.get("stage39_source_shadow_label", 0) < row_count:
            missing_dependencies.append("missing_or_null:stage39_source_shadow_label")
        if present_counts.get("stage32_shadow_label", 0) < row_count:
            missing_dependencies.append("stage32_owner_state_export_or_shadow_owner_state_unavailable")
        if present_counts.get("stage33_structured_coverage_label", 0) < row_count:
            missing_dependencies.append("stage33_structured_coverage_owner_export_unavailable")
        if present_counts.get("stage36_final_shadow_label", 0) < row_count:
            missing_dependencies.append("stage36_support_safety_export_unavailable")
        if present_counts.get("stage37_final_shadow_label", 0) < row_count:
            missing_dependencies.append("stage37_safe_support_export_unavailable")
        if present_counts.get("stage39_source_shadow_label", 0) < row_count:
            missing_dependencies.append("stage39_source_shadow_label_unavailable")

    shadow_available = bool(
        enabled
        and row_count > 0
        and present_counts.get("stage37_final_shadow_label", 0) == row_count
        and non_null_counts.get("stage39_source_shadow_label", 0) == row_count
    )
    if not enabled:
        conclusion = (
            "Stage43-C2 shadow export was not requested; safe_structured_v2 may remain unavailable if source shadow labels are absent."
        )
        next_action = "Run with --stage43-external-enable-shadow-export for diagnostic-only external shadow/composer export."
    elif shadow_available:
        conclusion = (
            "Stage43-C2 shadow export produced Stage37 source shadow labels for every prediction row using the internal prediction_records_v6b export path."
        )
        next_action = (
            "Inspect base-vs-composed metrics and safety counters; do not use Stage43-B1 labels for tuning, calibration, or selection."
        )
    else:
        conclusion = (
            "Stage43-C2 shadow export did not make safe_structured_v2 fully available; inspect missing dependencies before interpreting composed metrics."
        )
        next_action = (
            "Enable or repair only the missing export-only shadow dependencies; do not fake fields or tune on external labels."
        )

    return {
        "stage43c2_shadow_export_enabled": enabled,
        "stage43c2_shadow_export_available": shadow_available,
        "stage43c2_shadow_export_missing_dependencies": sorted(set(missing_dependencies)),
        "stage43c2_required_shadow_fields_present_counts": present_counts,
        "stage43c2_composer_application_count": application_count,
        "stage43c2_composer_blocked_count": blocked_count,
        "stage43c2_composer_blocked_reasons": dict(blocked_reasons),
        "stage43c2_reused_internal_export_path": path_diagnostics.get("stage43c2_reused_internal_export_path"),
        "stage43c2_forced_eval_only_exports": path_diagnostics.get("stage43c2_forced_eval_only_exports", []),
        "stage43c2_conclusion": conclusion,
        "stage43c2_next_action": next_action,
    }


def _sample_row(
    *,
    source_row: dict[str, Any],
    pred_row: dict[str, Any] | None,
    gold: str | None,
    base: str | None,
    row_index: int,
) -> dict[str, Any]:
    sample = {
        "row_index": row_index,
        "id": source_row.get("id"),
        "gold_label": gold,
        "base_prediction": base,
        "claim": source_row.get("claim"),
        "evidence": source_row.get("evidence"),
        "source_dataset": source_row.get("source_dataset"),
    }
    if pred_row is not None:
        sample["max_probability"] = pred_row.get("stage43c1_max_probability")
        sample["prediction_entropy"] = pred_row.get("stage43c1_prediction_entropy")
        sample["composer_reason"] = pred_row.get("stage39_composer_reason")
    return sample


def _build_stage43c1_diagnostics(
    *,
    rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    base_predictions_by_id: dict[str, str],
    token_diagnostics: dict[str, Any] | None,
    path_diagnostics: dict[str, Any] | None,
) -> dict[str, Any]:
    token_diagnostics = token_diagnostics or {}
    path_diagnostics = path_diagnostics or {}
    pred_by_id = {str(row.get("id")): row for row in prediction_rows}
    per_gold_prediction_distribution = {
        gold: {pred: 0 for pred in LABELS} for gold in LABELS
    }
    sample_rows_by_gold_and_prediction: dict[str, dict[str, list[dict[str, Any]]]] = {
        gold: {pred: [] for pred in LABELS} for gold in LABELS
    }
    sample_not_entitled_collapse_rows: list[dict[str, Any]] = []
    sample_high_confidence_errors: list[dict[str, Any]] = []
    max_probs: list[float] = []
    entropies: list[float] = []
    missing_claim_count = 0
    missing_evidence_count = 0

    for index, row in enumerate(rows):
        row_id = str(row.get("id"))
        gold = normalize_label(row.get("label"))
        base = base_predictions_by_id.get(row_id)
        pred_row = pred_by_id.get(row_id)
        if not str(row.get("claim") or "").strip():
            missing_claim_count += 1
        if not str(row.get("evidence") or "").strip():
            missing_evidence_count += 1
        if gold in LABELS and base in LABELS:
            per_gold_prediction_distribution[gold][base] += 1
            samples = sample_rows_by_gold_and_prediction[gold][base]
            if len(samples) < 3:
                samples.append(
                    _sample_row(
                        source_row=row,
                        pred_row=pred_row,
                        gold=gold,
                        base=base,
                        row_index=index,
                    )
                )
            if base == "NOT_ENTITLED" and gold != "NOT_ENTITLED" and len(sample_not_entitled_collapse_rows) < SAMPLE_LIMIT:
                sample_not_entitled_collapse_rows.append(
                    _sample_row(
                        source_row=row,
                        pred_row=pred_row,
                        gold=gold,
                        base=base,
                        row_index=index,
                    )
                )
            max_prob = pred_row.get("stage43c1_max_probability") if pred_row else None
            entropy = pred_row.get("stage43c1_prediction_entropy") if pred_row else None
            if isinstance(max_prob, (int, float)):
                max_probs.append(float(max_prob))
                if gold != base and max_prob >= 0.80 and len(sample_high_confidence_errors) < SAMPLE_LIMIT:
                    sample_high_confidence_errors.append(
                        _sample_row(
                            source_row=row,
                            pred_row=pred_row,
                            gold=gold,
                            base=base,
                            row_index=index,
                        )
                    )
            if isinstance(entropy, (int, float)):
                entropies.append(float(entropy))

    composer_available_row_count = 0
    composer_unavailable_reasons: Counter[str] = Counter()
    required_field_counts: dict[str, int] = {key: 0 for key in STAGE43C1_REQUIRED_COMPOSER_FIELDS}
    stage_field_counts: dict[str, int] = {key: 0 for key in STAGE43C1_FIELD_PRESENCE_KEYS}
    for pred_row in prediction_rows:
        action = str(pred_row.get("stage39_composer_action") or "")
        reason = str(pred_row.get("stage39_composer_reason") or "")
        if action.startswith("composed_to_") or action == "no_change":
            composer_available_row_count += 1
        else:
            composer_unavailable_reasons[reason or action or "unknown"] += 1
        for key in required_field_counts:
            if _has_present_value(pred_row, key):
                required_field_counts[key] += 1
        for key in stage_field_counts:
            if _has_present_value(pred_row, key):
                stage_field_counts[key] += 1

    row_count = len(prediction_rows)
    ne_count = sum(1 for value in base_predictions_by_id.values() if value == "NOT_ENTITLED")
    ne_rate = (ne_count / row_count) if row_count else None
    truncation_count = int(token_diagnostics.get("external_truncation_count") or 0)
    truncation_rate = token_diagnostics.get("external_truncation_rate")
    if not isinstance(truncation_rate, (int, float)):
        truncation_rate = None

    collapse_factors: list[str] = []
    if truncation_rate is not None and truncation_rate >= 0.25:
        collapse_factors.append("excessive token truncation is present")
    else:
        collapse_factors.append("excessive token truncation is not indicated by the token audit")
    if not bool(path_diagnostics.get("external_uses_same_prediction_path_as_dev", False)):
        collapse_factors.append("external prediction path differs from the controlled dev export path")
    else:
        collapse_factors.append("external prediction path matches the controlled dev encode/forward/export path")
    if not bool(path_diagnostics.get("external_uses_same_label_mapping_as_dev", False)):
        collapse_factors.append("label mapping mismatch is possible")
    else:
        collapse_factors.append("label mapping matches the controlled dev mapping")
    if missing_claim_count or missing_evidence_count:
        collapse_factors.append("missing claim/evidence fields are present")
    else:
        collapse_factors.append("missing claim/evidence fields are not indicated")
    if max_probs:
        collapse_factors.append("model confidence is available for collapse inspection")
    else:
        collapse_factors.append("model confidence probabilities are unavailable")
    if ne_rate is not None and ne_rate >= 0.80:
        collapse_factors.append("observed predictions are dominated by NOT_ENTITLED, consistent with learned controlled-model behavior under this external distribution")

    composer_unavailable_row_count = row_count - composer_available_row_count
    missing_required = [key for key, count in required_field_counts.items() if count < row_count]
    if composer_unavailable_row_count == row_count and missing_required:
        composer_conclusion = (
            "safe_structured_v2 composer output is unavailable because the external export rows do not carry "
            "the required Stage32/36/37/39 intermediate structures for every row: "
            + ", ".join(missing_required)
        )
    elif composer_unavailable_row_count:
        composer_conclusion = "safe_structured_v2 composer output is partially unavailable; inspect composer_unavailable_reasons and required_composer_fields_present_counts."
    else:
        composer_conclusion = "safe_structured_v2 composer diagnostics are available for all exported rows."

    diagnostic_conclusion = (
        "Diagnostic-only audit: "
        + "; ".join(collapse_factors)
        + ". "
        + composer_conclusion
        + " Decisions remain Stage43-C0 decisions and diagnostics do not convert INCOMPLETE into PASS."
    )
    diagnostic_next_action = (
        "For a future non-diagnostic stage, compare external formatting with controlled dev examples and enable/export the required Stage32/36/37 shadow structures before expecting safe_structured_v2 composition; do not tune thresholds or calibrate on Stage43-B1 labels."
    )
    stage43c2_diagnostics = _build_stage43c2_diagnostics(
        prediction_rows=prediction_rows,
        path_diagnostics=path_diagnostics,
    )

    return {
        "stage43c1_diagnostic_enabled": True,
        "external_input_template": path_diagnostics.get("external_input_template"),
        "controlled_dev_input_template": path_diagnostics.get("controlled_dev_input_template"),
        "external_uses_same_prediction_path_as_dev": bool(path_diagnostics.get("external_uses_same_prediction_path_as_dev", False)),
        "external_uses_same_label_mapping_as_dev": bool(path_diagnostics.get("external_uses_same_label_mapping_as_dev", False)),
        "label_id_to_name": path_diagnostics.get("label_id_to_name", {}),
        "name_to_label_id": path_diagnostics.get("name_to_label_id", {}),
        "external_tokenizer_source": path_diagnostics.get("tokenizer_source"),
        "controlled_dev_tokenizer_source": path_diagnostics.get("tokenizer_source"),
        "external_max_length": path_diagnostics.get("max_length"),
        "controlled_dev_max_length": path_diagnostics.get("max_length"),
        "external_token_length_summary": token_diagnostics.get("external_token_length_summary", summarize_numeric([])),
        "external_truncation_count": truncation_count,
        "external_truncation_rate": round(float(truncation_rate), 6) if isinstance(truncation_rate, (int, float)) else None,
        "prediction_entropy_summary": summarize_numeric(entropies) if entropies else None,
        "max_probability_summary": summarize_numeric(max_probs) if max_probs else None,
        "per_gold_prediction_distribution": per_gold_prediction_distribution,
        "sample_rows_by_gold_and_prediction": sample_rows_by_gold_and_prediction,
        "sample_high_confidence_errors": sample_high_confidence_errors if max_probs else None,
        "sample_not_entitled_collapse_rows": sample_not_entitled_collapse_rows,
        "composer_availability_summary": {
            "requested_composer_mode": "safe_structured_v2",
            "composer_available_row_count": composer_available_row_count,
            "composer_unavailable_row_count": composer_unavailable_row_count,
            "composer_unavailable_reasons": dict(composer_unavailable_reasons),
            "required_composer_fields_present_counts": required_field_counts,
        },
        "stage36_stage37_stage39_field_presence": stage_field_counts,
        "diagnostic_conclusion": diagnostic_conclusion,
        "diagnostic_next_action": diagnostic_next_action,
        "diagnostic_non_leakage_statement": LEAKAGE_POLICY,
        **stage43c2_diagnostics,
    }


def _incomplete_report(
    *,
    input_jsonl: Path,
    run_name: str,
    output_predictions_path: str | None,
    rows: list[dict[str, Any]],
    sample_errors: list[dict[str, Any]],
    reason: str,
    prediction_rows: list[dict[str, Any]] | None = None,
    token_diagnostics: dict[str, Any] | None = None,
    path_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gold_counts = Counter(normalize_label(row.get("label")) for row in rows)
    gold_counts.pop(None, None)
    diagnostics = _build_stage43c1_diagnostics(
        rows=rows,
        prediction_rows=prediction_rows or [],
        base_predictions_by_id={},
        token_diagnostics=token_diagnostics,
        path_diagnostics=path_diagnostics,
    )
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
        **diagnostics,
    }


def analyze_stage43_predictions(
    *,
    input_jsonl: Path,
    run_name: str,
    rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    output_predictions_path: str | None,
    token_diagnostics: dict[str, Any] | None = None,
    path_diagnostics: dict[str, Any] | None = None,
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
            prediction_rows=prediction_rows,
            token_diagnostics=token_diagnostics,
            path_diagnostics=path_diagnostics,
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
            prediction_rows=prediction_rows,
            token_diagnostics=token_diagnostics,
            path_diagnostics=path_diagnostics,
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
    report.update(
        _build_stage43c1_diagnostics(
            rows=rows,
            prediction_rows=prediction_rows,
            base_predictions_by_id={row["id"]: row["base_prediction"] for row in output_rows},
            token_diagnostics=token_diagnostics,
            path_diagnostics=path_diagnostics,
        )
    )
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
    base_counts = report.get("base_prediction_counts") or {}
    not_entitled_rate = (
        int(base_counts.get("NOT_ENTITLED") or 0) / report["row_count"]
        if report["row_count"]
        else 0.0
    )
    if not_entitled_rate >= 0.80:
        return (
            "STAGE43C0_EXTERNAL_FACTVER_INCOMPLETE",
            "Diagnostic audit found prediction collapse dominated by NOT_ENTITLED; do not report this external result as PASS.",
        )
    if (
        report.get("stage43c2_shadow_export_enabled")
        and report.get("stage43c2_shadow_export_available")
        and int(report.get("stage43c2_composer_application_count") or 0) == 0
    ):
        return (
            "STAGE43C0_EXTERNAL_FACTVER_INCOMPLETE",
            "Stage43-C2 shadow fields were exported, but safe_structured_v2 did not compose any row; do not report this external result as PASS.",
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
    if report.get("stage43c1_diagnostic_enabled"):
        diag_payload = {
            "prediction_collapse_summary": {
                "base_prediction_counts": report.get("base_prediction_counts"),
                "per_gold_prediction_distribution": report.get("per_gold_prediction_distribution"),
                "max_probability_summary": report.get("max_probability_summary"),
                "prediction_entropy_summary": report.get("prediction_entropy_summary"),
            },
            "token_length_truncation_summary": {
                "external_token_length_summary": report.get("external_token_length_summary"),
                "external_truncation_count": report.get("external_truncation_count"),
                "external_truncation_rate": report.get("external_truncation_rate"),
            },
            "input_template_path_audit": {
                "external_input_template": report.get("external_input_template"),
                "controlled_dev_input_template": report.get("controlled_dev_input_template"),
                "external_uses_same_prediction_path_as_dev": report.get("external_uses_same_prediction_path_as_dev"),
                "external_tokenizer_source": report.get("external_tokenizer_source"),
                "controlled_dev_tokenizer_source": report.get("controlled_dev_tokenizer_source"),
                "external_max_length": report.get("external_max_length"),
                "controlled_dev_max_length": report.get("controlled_dev_max_length"),
            },
            "label_mapping_audit": {
                "external_uses_same_label_mapping_as_dev": report.get("external_uses_same_label_mapping_as_dev"),
                "label_id_to_name": report.get("label_id_to_name"),
                "name_to_label_id": report.get("name_to_label_id"),
            },
            "composer_availability_audit": report.get("composer_availability_summary"),
            "stage36_stage37_stage39_field_presence": report.get("stage36_stage37_stage39_field_presence"),
            "sample_collapsed_rows": report.get("sample_not_entitled_collapse_rows"),
            "sample_high_confidence_errors": report.get("sample_high_confidence_errors"),
        }
        lines.extend(
            [
                "",
                "## Stage43-C1 Diagnostic Audit",
                "",
                "### Prediction collapse summary",
                "",
                "```json",
                json.dumps(diag_payload["prediction_collapse_summary"], indent=2),
                "```",
                "",
                "### Token length/truncation summary",
                "",
                "```json",
                json.dumps(diag_payload["token_length_truncation_summary"], indent=2),
                "```",
                "",
                "### Input template/path audit",
                "",
                "```json",
                json.dumps(diag_payload["input_template_path_audit"], indent=2),
                "```",
                "",
                "### Label mapping audit",
                "",
                "```json",
                json.dumps(diag_payload["label_mapping_audit"], indent=2),
                "```",
                "",
                "### Composer availability audit",
                "",
                "```json",
                json.dumps(
                    {
                        "composer_availability_summary": diag_payload["composer_availability_audit"],
                        "stage36_stage37_stage39_field_presence": diag_payload["stage36_stage37_stage39_field_presence"],
                    },
                    indent=2,
                ),
                "```",
                "",
                "### Sample collapsed rows",
                "",
                "```json",
                json.dumps(diag_payload["sample_collapsed_rows"], indent=2),
                "```",
                "",
                "### Diagnostic conclusion",
                "",
                str(report.get("diagnostic_conclusion")),
                "",
                "### Non-leakage statement",
                "",
                str(report.get("diagnostic_non_leakage_statement") or report.get("leakage_policy")),
            ]
        )
    if "stage43c2_shadow_export_enabled" in report:
        lines.extend(
            [
                "",
                "## Stage43-C2 Shadow Export Audit",
                "",
                "```json",
                json.dumps(
                    {
                        "stage43c2_shadow_export_enabled": report.get("stage43c2_shadow_export_enabled"),
                        "stage43c2_shadow_export_available": report.get("stage43c2_shadow_export_available"),
                        "stage43c2_shadow_export_missing_dependencies": report.get("stage43c2_shadow_export_missing_dependencies"),
                        "stage43c2_required_shadow_fields_present_counts": report.get("stage43c2_required_shadow_fields_present_counts"),
                        "stage43c2_composer_application_count": report.get("stage43c2_composer_application_count"),
                        "stage43c2_composer_blocked_count": report.get("stage43c2_composer_blocked_count"),
                        "stage43c2_composer_blocked_reasons": report.get("stage43c2_composer_blocked_reasons"),
                        "stage43c2_reused_internal_export_path": report.get("stage43c2_reused_internal_export_path"),
                        "stage43c2_forced_eval_only_exports": report.get("stage43c2_forced_eval_only_exports"),
                    },
                    indent=2,
                ),
                "```",
                "",
                "### Stage43-C2 conclusion",
                "",
                str(report.get("stage43c2_conclusion")),
                "",
                "### Stage43-C2 next action",
                "",
                str(report.get("stage43c2_next_action")),
            ]
        )
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
