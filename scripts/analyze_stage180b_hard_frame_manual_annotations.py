"""Analyze completed Stage180 manual annotations without model execution.

The analyzer restores identities from the hidden key, measures review quality
and descriptive taxonomy contrasts, and selects one predeclared Stage180-B
gate.  It never applies a label or recommended action to the dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


STAGE = "Stage180-B"
READY = "STAGE180A_HARD_FRAME_MANUAL_REVIEW_PACKET_READY"
BLOCKED = "STAGE180B_HARD_FRAME_MANUAL_TAXONOMY_BLOCKED"
DECISION_DATA = "STAGE180B_FRAME_LABEL_OR_DATA_DESIGN_ISSUE_IDENTIFIED"
DECISION_MODEL = "STAGE180B_VALID_FRAME_LABELS_WITH_MODEL_FAILURE_TAXONOMY_IDENTIFIED"
DECISION_HARD = "STAGE180B_GENUINELY_HARD_FRAME_SEMANTIC_SUBSET_IDENTIFIED"
DECISION_MIXED = "STAGE180B_HARD_FRAME_MANUAL_TAXONOMY_MIXED"

PASS1_VALUES = {"compatible", "incompatible", "ambiguous", "insufficient_context"}
GOLD_VALUES = {"gold_consistent", "gold_questionable", "gold_likely_incorrect", "cannot_determine"}
VALIDITY_VALUES = {"clean_single_axis_edit", "valid_but_multi_axis_edit", "weak_or_ineffective_edit",
                   "unnatural_or_broken_text", "canonical_control", "cannot_determine"}
PHENOMENON_VALUES = {"entity_identity", "event_identity", "location_scope", "role_relation",
                     "title_name_identity", "predicate_scope", "polarity", "temporal_scope",
                     "referent_resolution", "evidence_sufficiency_interaction", "world_knowledge_dependency",
                     "lexical_or_surface_artifact", "other", "cannot_determine"}
LOCUS_VALUES = {"input_representation_insensitivity", "head_direction_or_readout", "downstream_final_boundary",
                "data_or_intervention_design", "label_semantics_or_annotation", "genuinely_hard_semantic_case",
                "mixed", "cannot_determine"}
ACTION_VALUES = {"keep_unchanged", "manual_adjudication", "revise_frame_label", "rewrite_claim_or_evidence",
                 "redesign_intervention", "add_minimal_counterpart", "exclude_from_training",
                 "retain_as_diagnostic_only", "cannot_determine"}
OUTPUTS = [
    "stage180b_manual_taxonomy_report.json", "stage180b_manual_taxonomy_report.md",
    "stage180b_item_level_adjudication.csv", "stage180b_pass1_gold_agreement.csv",
    "stage180b_repeat_consistency.csv", "stage180b_gold_assessment_summary.csv",
    "stage180b_intervention_validity_summary.csv", "stage180b_failure_locus_summary.csv",
    "stage180b_recommended_action_summary.csv", "stage180b_hard_control_comparison.csv",
    "stage180b_beneficial_harmful_taxonomy.csv", "stage180b_high_confidence_review_queue.csv",
    "stage180b_reviewer_disagreement_queue.csv", "stage180b_decision_evidence.csv",
]
ITEM_COLUMNS = ["row_id", "item_role", "stage176_cohort", "intervention_type", "native_frame_label",
                "gold_final_label", "match_link_id", "matched_hard_row_id", "match_level",
                "independent_frame_judgment", "gold_frame_assessment", "intervention_validity",
                "primary_semantic_phenomenon", "diagnostic_failure_locus", "recommended_data_action",
                "reviewer_count", "pass1_confidence_min", "pass2_confidence_min", "independent_gold_agreement"]
PASS1_GOLD_COLUMNS = ["item_role", "stage176_cohort", "native_frame_label", "rows", "binary_judgments",
                      "agreement_count", "agreement_rate", "ambiguous_rate", "insufficient_context_rate",
                      "high_confidence_disagreement_count"]
REPEAT_COLUMNS = ["reviewer_id", "repeat_group_id", "row_id", "item_role", "instance_count", "exact_agreement",
                  "judgment_flip", "adjacent_confidence_difference", "judgments", "confidences"]
GOLD_SUMMARY_COLUMNS = ["item_role", "intervention_type", "axis", "category", "count", "denominator", "rate"]
SUMMARY_COLUMNS = ["item_role", "axis", "category", "count", "denominator", "rate"]
COMPARISON_COLUMNS = ["axis", "category", "hard_count", "hard_total", "hard_rate", "control_count",
                      "control_total", "control_rate", "risk_difference", "risk_ratio", "fisher_exact_p",
                      "benjamini_hochberg_q"]
COHORT_COLUMNS = ["stage176_cohort", "intervention_type", "axis", "category", "count", "denominator", "rate"]
HIGH_QUEUE_COLUMNS = ["review_instance_id", "source_item_id", "hard_or_control", "stage176_cohort",
                      "independent_frame_judgment", "gold_frame_assessment", "intervention_validity",
                      "diagnostic_failure_locus", "recommended_data_action", "confidence", "rationale", "queue_reason"]
DISAGREEMENT_COLUMNS = ["review_instance_id", "source_item_id", "axis", "reviewer_ids", "reviewer_values",
                        "agreement_status", "notes"]
DECISION_COLUMNS = ["gate", "passed", "high_confidence_hard_issue_count", "hard_issue_rate",
                    "control_issue_rate", "risk_difference", "repeat_consistency", "label_valid_rate",
                    "high_confidence_data_issue_count", "hard_model_locus_rate", "control_model_locus_rate",
                    "hard_data_issue_rate", "hard_semantic_rate", "control_semantic_rate",
                    "semantic_issue_intervention_family_count", "selected_by_priority"]
CSV_SCHEMAS = {
    OUTPUTS[2]: ITEM_COLUMNS, OUTPUTS[3]: PASS1_GOLD_COLUMNS, OUTPUTS[4]: REPEAT_COLUMNS,
    OUTPUTS[5]: GOLD_SUMMARY_COLUMNS, OUTPUTS[6]: SUMMARY_COLUMNS, OUTPUTS[7]: SUMMARY_COLUMNS,
    OUTPUTS[8]: SUMMARY_COLUMNS, OUTPUTS[9]: COMPARISON_COLUMNS, OUTPUTS[10]: COHORT_COLUMNS,
    OUTPUTS[11]: HIGH_QUEUE_COLUMNS, OUTPUTS[12]: DISAGREEMENT_COLUMNS, OUTPUTS[13]: DECISION_COLUMNS,
}


class AnnotationBlocked(ValueError):
    """An annotation, packet, or provenance contract failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnnotationBlocked(message)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnnotationBlocked(f"cannot read JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root must be object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    except OSError as exc:
        raise AnnotationBlocked(f"cannot read CSV {path}: {exc}") from exc
    require(bool(rows), f"CSV is empty: {path}")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def csv_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    require(fields is not None and bool(fields), f"fixed CSV schema is required: {path.name}")
    require(len(fields) == len(set(fields)), f"duplicate CSV schema column: {path.name}")
    require(path.name in CSV_SCHEMAS and fields == CSV_SCHEMAS[path.name], f"unexpected CSV schema: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({field: csv_value(row.get(field, "")) for field in fields} for row in rows)


def integer(value: Any, field: str) -> int:
    try:
        result = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise AnnotationBlocked(f"{field} must be an integer: {value!r}") from exc
    return result


def boolean(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    require(lowered in {"true", "false", "1", "0"}, f"{field} must be boolean")
    return lowered in {"true", "1"}


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def validate_annotations(rows: list[dict[str, str]], expected_ids: set[str], pass_number: int,
                         minimum: float) -> dict[tuple[str, str], dict[str, Any]]:
    specifications = ({"independent_frame_judgment": PASS1_VALUES},
                      {"gold_frame_assessment": GOLD_VALUES, "intervention_validity": VALIDITY_VALUES,
                       "primary_semantic_phenomenon": PHENOMENON_VALUES, "diagnostic_failure_locus": LOCUS_VALUES,
                       "recommended_data_action": ACTION_VALUES})[pass_number - 1]
    confidence_field = f"pass{pass_number}_confidence"
    rationale_field = f"pass{pass_number}_rationale"
    result: dict[tuple[str, str], dict[str, Any]] = {}
    reviewer_ids: dict[str, set[str]] = defaultdict(set)
    for number, source in enumerate(rows, 2):
        rid = str(source.get("review_instance_id", "")).strip()
        reviewer = str(source.get("reviewer_id", "")).strip()
        require(rid in expected_ids, f"Pass {pass_number} row {number} has unknown instance {rid!r}")
        require(reviewer, f"Pass {pass_number} row {number} has empty reviewer_id")
        key = (reviewer, rid)
        require(key not in result, f"duplicate Pass {pass_number} annotation: {key}")
        item: dict[str, Any] = dict(source)
        for field, allowed in specifications.items():
            value = str(source.get(field, "")).strip()
            require(value in allowed, f"invalid Pass {pass_number} {field} at {key}: {value!r}")
            item[field] = value
        confidence = integer(source.get(confidence_field), confidence_field)
        require(1 <= confidence <= 5, f"{confidence_field} outside 1..5 at {key}")
        rationale = str(source.get(rationale_field, "")).strip()
        require(rationale, f"empty {rationale_field} at {key}")
        item[confidence_field], item[rationale_field] = confidence, rationale
        result[key] = item
        reviewer_ids[reviewer].add(rid)
    require(reviewer_ids, f"Pass {pass_number} has no reviewers")
    for reviewer, ids in reviewer_ids.items():
        completion = len(ids) / len(expected_ids)
        require(completion >= minimum, f"Pass {pass_number} reviewer {reviewer} completion {completion:.6f} below {minimum}")
        if minimum == 1.0:
            require(ids == expected_ids, f"Pass {pass_number} reviewer {reviewer} instance set mismatch")
    require(set().union(*reviewer_ids.values()) == expected_ids, f"Pass {pass_number} annotations do not cover exact packet ID set")
    return result


def categorical_agreement(matrix: list[list[str]], categories: list[str]) -> dict[str, Any]:
    if not matrix or len(matrix[0]) < 2:
        return {"computed": False, "reason": "single_reviewer_provisional"}
    pair_agreements = []
    for ratings in matrix:
        pairs = [(ratings[i], ratings[j]) for i in range(len(ratings)) for j in range(i + 1, len(ratings))]
        pair_agreements.extend(a == b for a, b in pairs)
    raw = mean(float(value) for value in pair_agreements)
    if len(matrix[0]) == 2:
        left, right = [row[0] for row in matrix], [row[1] for row in matrix]
        observed = mean(float(a == b) for a, b in zip(left, right)) or 0.0
        p_left = Counter(left); p_right = Counter(right); n = len(matrix)
        expected = sum((p_left[c] / n) * (p_right[c] / n) for c in categories)
        kappa = (observed - expected) / (1 - expected) if expected < 1 else None
        return {"computed": True, "method": "cohen_kappa", "categorical_agreement": raw, "kappa": kappa}
    n_raters = len(matrix[0]); n_items = len(matrix)
    p_bar = mean((sum(count * count for count in Counter(row).values()) - n_raters) /
                 (n_raters * (n_raters - 1)) for row in matrix) or 0.0
    totals = Counter(value for row in matrix for value in row)
    p_e = sum((totals[c] / (n_items * n_raters)) ** 2 for c in categories)
    kappa = (p_bar - p_e) / (1 - p_e) if p_e < 1 else None
    return {"computed": True, "method": "fleiss_kappa", "categorical_agreement": raw, "kappa": kappa}


def consensus(values: list[Any]) -> Any:
    """Conservative item consensus; intentionally not a majority vote."""
    return values[0] if values and all(value == values[0] for value in values) else "disagreement"


def build_observations(hidden: list[dict[str, str]], pass1: dict[tuple[str, str], dict[str, Any]],
                       pass2: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    key_by_id = {row["review_instance_id"]: row for row in hidden}
    result = []
    for reviewer, rid in sorted(pass1):
        if (reviewer, rid) not in pass2:  # allowed only when minimum completion is below one
            continue
        key = key_by_id[rid]
        p1, p2 = pass1[(reviewer, rid)], pass2[(reviewer, rid)]
        native = integer(key["native_frame_label"], "native_frame_label")
        judgment = p1["independent_frame_judgment"]
        binary = 1 if judgment == "compatible" else 0 if judgment == "incompatible" else None
        result.append({**key, "reviewer_id": reviewer, **p1, **p2,
                       "native_frame_label": native, "is_repeat": boolean(key["is_repeat"], "is_repeat"),
                       "pass1_binary_judgment": binary,
                       "independent_gold_agreement": binary == native if binary is not None else None,
                       "high_confidence_pass1_disagreement": binary is not None and binary != native and p1["pass1_confidence"] >= 4})
    return result


def semantic_consensus(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    originals = [row for row in observations if not row["is_repeat"]]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in originals:
        groups[row["row_id"]].append(row)
    fields = ["independent_frame_judgment", "gold_frame_assessment", "intervention_validity",
              "primary_semantic_phenomenon", "diagnostic_failure_locus", "recommended_data_action"]
    result = []
    for source_id, rows in sorted(groups.items()):
        base = {key: rows[0][key] for key in ("row_id", "item_role", "stage176_cohort", "intervention_type",
                                               "native_frame_label", "gold_final_label", "match_link_id",
                                               "matched_hard_row_id", "match_level")}
        for field in fields:
            base[field] = consensus([row[field] for row in rows])
        base["reviewer_count"] = len(rows)
        base["pass1_confidence_min"] = min(row["pass1_confidence"] for row in rows)
        base["pass2_confidence_min"] = min(row["pass2_confidence"] for row in rows)
        binary = 1 if base["independent_frame_judgment"] == "compatible" else 0 if base["independent_frame_judgment"] == "incompatible" else None
        base["independent_gold_agreement"] = binary == base["native_frame_label"] if binary is not None else None
        result.append(base)
    return result


def summary_by(rows: list[dict[str, Any]], field: str, values: Iterable[str], axes: tuple[str, ...] = ("item_role",)) -> list[dict[str, Any]]:
    result = []
    group_keys = sorted({tuple(row[axis] for axis in axes) for row in rows})
    for key in group_keys:
        group = [row for row in rows if tuple(row[axis] for axis in axes) == key]
        for value in values:
            count = sum(row[field] == value for row in group)
            result.append({**dict(zip(axes, key)), "axis": field, "category": value,
                           "count": count, "denominator": len(group), "rate": rate(count, len(group))})
    return result


def pass1_gold_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    axes = [("item_role",), ("item_role", "stage176_cohort"), ("item_role", "native_frame_label")]
    for fields in axes:
        for key in sorted({tuple(row[field] for field in fields) for row in rows}):
            group = [row for row in rows if tuple(row[field] for field in fields) == key]
            decided = [row for row in group if row["independent_gold_agreement"] is not None]
            result.append({**dict(zip(fields, key)), "rows": len(group), "binary_judgments": len(decided),
                           "agreement_count": sum(row["independent_gold_agreement"] is True for row in decided),
                           "agreement_rate": mean(float(row["independent_gold_agreement"]) for row in decided),
                           "ambiguous_rate": mean(row["independent_frame_judgment"] == "ambiguous" for row in group),
                           "insufficient_context_rate": mean(row["independent_frame_judgment"] == "insufficient_context" for row in group),
                           "high_confidence_disagreement_count": sum(row["high_confidence_pass1_disagreement"] for row in group)})
    return result


def repeat_rows(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        groups[(row["reviewer_id"], row["repeat_group_id"])].append(row)
    result = []
    for (reviewer, group_id), rows in sorted(groups.items()):
        if len(rows) < 2:
            continue
        judgments = [row["independent_frame_judgment"] for row in rows]
        confidences = [row["pass1_confidence"] for row in rows]
        result.append({"reviewer_id": reviewer, "repeat_group_id": group_id, "row_id": rows[0]["row_id"],
                       "item_role": rows[0]["item_role"], "instance_count": len(rows),
                       "exact_agreement": len(set(judgments)) == 1, "judgment_flip": len(set(judgments)) > 1,
                       "adjacent_confidence_difference": max(confidences) - min(confidences),
                       "judgments": judgments, "confidences": confidences})
    return result


def log_comb(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    row1, row2, col1, total = a + b, c + d, a + c, a + b + c + d
    if total == 0:
        return 1.0
    low, high = max(0, col1 - row2), min(row1, col1)
    def probability(x: int) -> float:
        return math.exp(log_comb(col1, x) + log_comb(total - col1, row1 - x) - log_comb(total, row1))
    observed = probability(a)
    return min(1.0, sum(probability(x) for x in range(low, high + 1) if probability(x) <= observed + 1e-12))


def hard_control(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    axes = {"independent_frame_judgment": PASS1_VALUES, "gold_frame_assessment": GOLD_VALUES,
            "intervention_validity": VALIDITY_VALUES, "primary_semantic_phenomenon": PHENOMENON_VALUES,
            "diagnostic_failure_locus": LOCUS_VALUES, "recommended_data_action": ACTION_VALUES}
    hard = [row for row in rows if row["item_role"] == "hard"]
    control = [row for row in rows if row["item_role"] == "control"]
    result = []
    for axis, values in axes.items():
        for value in sorted(values):
            a, c = sum(row[axis] == value for row in hard), sum(row[axis] == value for row in control)
            b, d = len(hard) - a, len(control) - c
            hard_rate, control_rate = rate(a, len(hard)), rate(c, len(control))
            result.append({"axis": axis, "category": value, "hard_count": a, "hard_total": len(hard),
                           "hard_rate": hard_rate, "control_count": c, "control_total": len(control),
                           "control_rate": control_rate,
                           "risk_difference": (hard_rate - control_rate) if hard_rate is not None and control_rate is not None else None,
                           "risk_ratio": (hard_rate / control_rate) if control_rate not in (None, 0) else None,
                           "fisher_exact_p": fisher_exact(a, b, c, d)})
    ordered = sorted(range(len(result)), key=lambda index: result[index]["fisher_exact_p"])
    adjusted = [1.0] * len(result); running = 1.0
    for rank_index in range(len(ordered) - 1, -1, -1):
        index = ordered[rank_index]; rank = rank_index + 1
        running = min(running, result[index]["fisher_exact_p"] * len(result) / rank)
        adjusted[index] = running
    for row, value in zip(result, adjusted):
        row["benjamini_hochberg_q"] = value
    return result


def high_confidence_queue(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in observations:
        if row["is_repeat"] or row["pass2_confidence"] < 4:
            continue
        reasons = []
        if row["gold_frame_assessment"] != "gold_consistent":
            reasons.append("gold_frame_requires_review")
        if row["recommended_data_action"] != "keep_unchanged":
            reasons.append("non_keep_data_action")
        if not reasons:
            continue
        result.append({"review_instance_id": row["review_instance_id"], "source_item_id": row["row_id"],
                       "hard_or_control": row["item_role"], "stage176_cohort": row["stage176_cohort"],
                       "independent_frame_judgment": row["independent_frame_judgment"],
                       "gold_frame_assessment": row["gold_frame_assessment"],
                       "intervention_validity": row["intervention_validity"],
                       "diagnostic_failure_locus": row["diagnostic_failure_locus"],
                       "recommended_data_action": row["recommended_data_action"],
                       "confidence": row["pass2_confidence"], "rationale": row["pass2_rationale"],
                       "queue_reason": ";".join(reasons)})
    return result


def disagreements(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = ["independent_frame_judgment", "gold_frame_assessment", "intervention_validity",
              "primary_semantic_phenomenon", "diagnostic_failure_locus", "recommended_data_action"]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        if not row["is_repeat"]:
            groups[row["row_id"]].append(row)
    result = []
    for source_id, rows in groups.items():
        for field in fields:
            values = {row[field] for row in rows}
            if len(values) > 1:
                result.append({"review_instance_id": ";".join(sorted(row["review_instance_id"] for row in rows)),
                               "source_item_id": source_id, "axis": field,
                               "reviewer_ids": ";".join(sorted(row["reviewer_id"] for row in rows)),
                               "reviewer_values": {row["reviewer_id"]: row[field] for row in rows},
                               "agreement_status": "disagreement",
                               "notes": "Descriptive queue only; no automatic majority relabeling."})
    return result


def choose_decision(rows: list[dict[str, Any]], repeat_rate: float) -> tuple[str, str, list[dict[str, Any]], dict[str, Any]]:
    hard = [row for row in rows if row["item_role"] == "hard"]
    controls = [row for row in rows if row["item_role"] == "control"]
    require(len(hard) == 39, f"decision requires 39 unique hard rows, got {len(hard)}")
    data_validity = {"valid_but_multi_axis_edit", "weak_or_ineffective_edit", "unnatural_or_broken_text"}
    def data_issue(row: dict[str, Any]) -> bool:
        return row["gold_frame_assessment"] == "gold_likely_incorrect" or row["intervention_validity"] in data_validity
    high_data = [row for row in hard if row["pass2_confidence_min"] >= 4 and data_issue(row)]
    hard_data_rate = mean(data_issue(row) for row in hard) or 0.0
    control_data_rate = mean(data_issue(row) for row in controls) or 0.0
    data_gate = len(high_data) >= 5 and hard_data_rate - control_data_rate >= .20 and repeat_rate >= .80
    pass1_valid = mean(row["independent_gold_agreement"] is True for row in hard if row["independent_gold_agreement"] is not None) or 0.0
    gold_valid = mean(row["gold_frame_assessment"] == "gold_consistent" for row in hard) or 0.0
    label_valid = max(pass1_valid, gold_valid)
    model_loci = {"input_representation_insensitivity", "head_direction_or_readout", "downstream_final_boundary"}
    hard_model = mean(row["diagnostic_failure_locus"] in model_loci for row in hard) or 0.0
    control_model = mean(row["diagnostic_failure_locus"] in model_loci for row in controls) or 0.0
    model_gate = label_valid >= .80 and len(high_data) <= 3 and hard_model >= .60 and hard_model - control_model >= .20 and repeat_rate >= .80
    hard_semantic_values = {"genuinely_hard_semantic_case", "cannot_determine"}
    hard_semantic = mean(row["diagnostic_failure_locus"] in hard_semantic_values for row in hard) or 0.0
    control_semantic = mean(row["diagnostic_failure_locus"] in hard_semantic_values for row in controls) or 0.0
    issue_families = {row["intervention_type"] for row in hard if row["diagnostic_failure_locus"] in hard_semantic_values}
    hard_gate = label_valid >= .80 and hard_data_rate < .20 and hard_semantic >= .50 and hard_semantic - control_semantic >= .20 and len(issue_families) >= 2
    evidence = [
        {"gate": "frame_label_or_data_design", "passed": data_gate, "high_confidence_hard_issue_count": len(high_data),
         "hard_issue_rate": hard_data_rate, "control_issue_rate": control_data_rate,
         "risk_difference": hard_data_rate - control_data_rate, "repeat_consistency": repeat_rate},
        {"gate": "valid_labels_model_failure", "passed": model_gate, "label_valid_rate": label_valid,
         "high_confidence_data_issue_count": len(high_data), "hard_model_locus_rate": hard_model,
         "control_model_locus_rate": control_model, "risk_difference": hard_model - control_model,
         "repeat_consistency": repeat_rate},
        {"gate": "genuinely_hard_semantic_subset", "passed": hard_gate, "label_valid_rate": label_valid,
         "hard_data_issue_rate": hard_data_rate, "hard_semantic_rate": hard_semantic,
         "control_semantic_rate": control_semantic, "risk_difference": hard_semantic - control_semantic,
         "semantic_issue_intervention_family_count": len(issue_families)},
    ]
    passed = [row for row in evidence if row["passed"]]
    # Priority is explicit so exactly one decision is emitted even if gates overlap.
    if data_gate:
        selected, next_stage = DECISION_DATA, "STAGE181_FRAME_DATASET_ADJUDICATION_AND_REDESIGN_SPEC"
    elif model_gate:
        selected, next_stage = DECISION_MODEL, "STAGE181_FRAME_FAILURE_LOCALIZATION_DESIGN_AUDIT"
    elif hard_gate:
        selected, next_stage = DECISION_HARD, "STAGE181_HARD_FRAME_SEMANTIC_BENCHMARK_DESIGN"
    else:
        selected, next_stage = DECISION_MIXED, "STAGE181_STRATIFIED_FRAME_DATA_AND_MODEL_ROADMAP"
    evidence.append({"gate": "mixed", "passed": not passed, "selected_by_priority": selected == DECISION_MIXED})
    metrics = {"high_confidence_data_issue_count": len(high_data), "hard_data_issue_rate": hard_data_rate,
               "control_data_issue_rate": control_data_rate, "label_valid_rate": label_valid,
               "hard_model_locus_rate": hard_model, "control_model_locus_rate": control_model,
               "hard_semantic_rate": hard_semantic, "control_semantic_rate": control_semantic,
               "repeat_consistency": repeat_rate, "simultaneously_passing_gate_count": len(passed),
               "priority_order": [DECISION_DATA, DECISION_MODEL, DECISION_HARD, DECISION_MIXED]}
    return selected, next_stage, evidence, metrics


def markdown(report: dict[str, Any]) -> str:
    return "\n".join([
        "# Stage180-B hard-frame manual taxonomy", "", f"**Decision:** `{report['decision']}`", "",
        f"Review status: `{report['reviewer_protocol']['status']}`. Unique hard rows: "
        f"{report['completion']['unique_hard_rows']}; unique controls: {report['completion']['unique_control_rows']}.", "",
        f"Repeat exact agreement was `{report['repeat_consistency']['exact_agreement_rate']}`. "
        "All hard/control comparisons are descriptive; Fisher exact p-values use Benjamini-Hochberg correction.", "",
        f"Authorized next stage: `{report['stage181_gate']['next_stage']}`.", "",
        "No recommendation was applied to the dataset. No automatic majority relabeling, model forward, training, fitting, or checkpoint modification occurred.", "",
    ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--hidden-item-key", type=Path, required=True)
    parser.add_argument("--pass1-annotations", type=Path, required=True)
    parser.add_argument("--pass2-annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-completion-rate", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args(); output_dir = args.output_dir.resolve(); current = "argument_validation"
    try:
        require(0 < args.minimum_completion_rate <= 1, "minimum-completion-rate must be in (0, 1]")
        manifest = read_json(args.manifest.resolve())
        require(manifest.get("decision") == READY, f"manifest decision must be {READY}")
        current = "hidden_key_validation"
        hidden_path = args.hidden_item_key.resolve(); hidden = read_csv(hidden_path)
        expected_digest = (manifest.get("input_validation") or {}).get("hidden_item_key_sha256")
        require(expected_digest and sha256(hidden_path) == expected_digest, "hidden item key SHA-256 does not match manifest")
        expected_ids = {row["review_instance_id"] for row in hidden}
        require(len(expected_ids) == len(hidden), "hidden key has duplicate review instance IDs")
        require(len({row["row_id"] for row in hidden if not boolean(row["is_repeat"], "is_repeat") and row["item_role"] == "hard"}) == 39,
                "hidden key does not contain all 39 original hard rows")
        hard_ids = {row["row_id"] for row in hidden if row["item_role"] == "hard"}
        control_ids = {row["row_id"] for row in hidden if row["item_role"] == "control"}
        require(not hard_ids & control_ids, "hidden key hard/control overlap")
        current = "annotation_validation"
        p1 = validate_annotations(read_csv(args.pass1_annotations.resolve()), expected_ids, 1, args.minimum_completion_rate)
        p2 = validate_annotations(read_csv(args.pass2_annotations.resolve()), expected_ids, 2, args.minimum_completion_rate)
        require(set(p1) == set(p2), "Pass 1 and Pass 2 reviewer/instance alignment mismatch")
        reviewers = sorted({key[0] for key in p1})
        observations = build_observations(hidden, p1, p2)
        consensus_rows = semantic_consensus(observations)
        require(len([row for row in consensus_rows if row["item_role"] == "hard"]) == 39, "original hard rows not preserved")
        current = "taxonomy_analysis"
        pass1_rows = pass1_gold_rows([row for row in observations if not row["is_repeat"]])
        repeats = repeat_rows(observations)
        repeat_rate = mean(row["exact_agreement"] for row in repeats)
        require(repeat_rate is not None, "no analyzable repeat instances")
        gold_summary = summary_by(consensus_rows, "gold_frame_assessment", sorted(GOLD_VALUES))
        gold_summary += summary_by(consensus_rows, "gold_frame_assessment", sorted(GOLD_VALUES), ("item_role", "intervention_type"))
        validity_summary = summary_by(consensus_rows, "intervention_validity", sorted(VALIDITY_VALUES))
        locus_summary = summary_by(consensus_rows, "diagnostic_failure_locus", sorted(LOCUS_VALUES))
        action_summary = summary_by(consensus_rows, "recommended_data_action", sorted(ACTION_VALUES))
        comparisons = hard_control(consensus_rows)
        cohort_rows = summary_by([row for row in consensus_rows if row["item_role"] == "hard"],
                                 "diagnostic_failure_locus", sorted(LOCUS_VALUES), ("stage176_cohort", "intervention_type"))
        high_queue = high_confidence_queue(observations); disagreement_queue = disagreements(observations)
        decision_value, next_stage, decision_rows, diagnosis = choose_decision(consensus_rows, repeat_rate)
        current = "reviewer_agreement"
        agreement_fields = ["independent_frame_judgment", "gold_frame_assessment", "intervention_validity",
                            "primary_semantic_phenomenon", "diagnostic_failure_locus", "recommended_data_action"]
        agreement: dict[str, Any] = {}
        originals = [row for row in observations if not row["is_repeat"]]
        for field in agreement_fields:
            by_source: dict[str, dict[str, str]] = defaultdict(dict)
            for row in originals:
                by_source[row["row_id"]][row["reviewer_id"]] = row[field]
            matrix = [[mapping[reviewer] for reviewer in reviewers] for mapping in by_source.values()
                      if all(reviewer in mapping for reviewer in reviewers)]
            categories = sorted({value for row in matrix for value in row})
            agreement[field] = categorical_agreement(matrix, categories)
        weighted = mean((1.0 if len({row[field] for row in group}) == 1 else 0.0) *
                        mean((row["pass1_confidence"] if field == "independent_frame_judgment" else row["pass2_confidence"]) / 5 for row in group)
                        for field in agreement_fields for group in
                        [[row for row in originals if row["row_id"] == source] for source in {row["row_id"] for row in originals}])
        status = "single_reviewer_provisional" if len(reviewers) == 1 else "multi_reviewer_descriptive_agreement"
        output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(output_dir / OUTPUTS[2], consensus_rows, ITEM_COLUMNS)
        write_csv(output_dir / OUTPUTS[3], pass1_rows, PASS1_GOLD_COLUMNS)
        write_csv(output_dir / OUTPUTS[4], repeats, REPEAT_COLUMNS)
        write_csv(output_dir / OUTPUTS[5], gold_summary, GOLD_SUMMARY_COLUMNS)
        write_csv(output_dir / OUTPUTS[6], validity_summary, SUMMARY_COLUMNS)
        write_csv(output_dir / OUTPUTS[7], locus_summary, SUMMARY_COLUMNS)
        write_csv(output_dir / OUTPUTS[8], action_summary, SUMMARY_COLUMNS)
        write_csv(output_dir / OUTPUTS[9], comparisons, COMPARISON_COLUMNS)
        write_csv(output_dir / OUTPUTS[10], cohort_rows, COHORT_COLUMNS)
        write_csv(output_dir / OUTPUTS[11], high_queue, HIGH_QUEUE_COLUMNS)
        write_csv(output_dir / OUTPUTS[12], disagreement_queue, DISAGREEMENT_COLUMNS)
        write_csv(output_dir / OUTPUTS[13], decision_rows, DECISION_COLUMNS)
        report = {
            "stage": STAGE, "decision": decision_value,
            "scope": {"clean_controlled_manual_annotations_only": True, "model_forward": False, "training": False},
            "input_validation": {"status": "passed", "manifest_decision": READY, "hidden_key_sha256_match": True,
                                 "review_instance_id_set_validated": True, "hard_control_overlap": False,
                                 "original_hard_rows": 39, "pass2_read_only_extra_columns_allowed": True,
                                 "matched_control_context_used_for_annotation_vote": False,
                                 "hidden_key_remains_identity_authority": True,
                                 "output_schemas": CSV_SCHEMAS},
            "reviewer_protocol": {"status": status, "reviewer_count": len(reviewers), "reviewers": reviewers,
                                  "agreement": agreement, "confidence_weighted_descriptive_agreement": weighted,
                                  "automatic_majority_relabeling": False, "consensus_rule": "unanimity_only"},
            "completion": {"minimum_required_rate": args.minimum_completion_rate,
                           "review_instances": len(expected_ids), "annotation_pairs": len(observations),
                           "unique_hard_rows": 39, "unique_control_rows": len(control_ids), "missing_or_duplicate": False},
            "pass1_gold_agreement": {"rows": len(pass1_rows), "high_confidence_disagreements": sum(row["high_confidence_pass1_disagreement"] for row in originals)},
            "repeat_consistency": {"comparisons": len(repeats), "exact_agreement_rate": repeat_rate,
                                   "judgment_flips": sum(row["judgment_flip"] for row in repeats),
                                   "mean_confidence_difference": mean(row["adjacent_confidence_difference"] for row in repeats),
                                   "by_role": {role: mean(row["exact_agreement"] for row in repeats if row["item_role"] == role)
                                               for role in ("hard", "control")}},
            "gold_assessment": {"summary_rows": len(gold_summary), "high_confidence_queue_rows": len(high_queue)},
            "intervention_validity": {"summary_rows": len(validity_summary), "hard_control_rate_differences_in_comparison": True},
            "failure_locus_taxonomy": {"summary_rows": len(locus_summary),
                                       "stage179_comparison": "descriptive manual taxonomy versus mixed/insufficient automated localization; no causal equivalence asserted"},
            "recommended_actions": {"summary_rows": len(action_summary), "applied_to_dataset": False},
            "hard_control_comparison": {"rows": len(comparisons), "statistics": ["risk_difference", "risk_ratio", "Fisher exact", "Benjamini-Hochberg"],
                                        "interpretation": "observational/provisional" if len(reviewers) == 1 else "descriptive multi-reviewer"},
            "beneficial_harmful_attribution": {"summary_rows": len(cohort_rows), "beneficial_expected": 25, "harmful_expected": 14,
                                               "families_reported": sorted({row["intervention_type"] for row in consensus_rows if row["item_role"] == "hard")},
                                               "high_confidence_action_queue_rows": len(high_queue)},
            "diagnosis": diagnosis,
            "stage181_gate": {"decision": decision_value, "next_stage": next_stage,
                              "provisional": len(reviewers) == 1, "training_authorized": False,
                              "automatic_relabeling_authorized": False},
            "limitations": ["Manual taxonomy is observational and does not prove a causal model locus.",
                            "Single-reviewer results are provisional; multi-reviewer consensus uses unanimity, not majority relabeling.",
                            "Matched controls are deterministic comparisons rather than randomized experimental controls."],
            "safety_policy": {"model_forward": False, "training": False, "optimizer": False, "backward": False,
                              "calibration": False, "threshold_fitting": False, "fitted_probe": False,
                              "external_evaluation": False, "external_labels": False, "time_swap": False,
                              "relabeling": False, "dataset_modification": False, "checkpoint_modification": False,
                              "automatic_adjudication": False, "architecture_implementation": False, "loss_implementation": False},
        }
        write_json(output_dir / OUTPUTS[0], report)
        (output_dir / OUTPUTS[1]).write_text(markdown(report), encoding="utf-8")
        return 0
    except Exception as error:  # a structured blocked result is mandatory
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / OUTPUTS[0], {"stage": STAGE, "decision": BLOCKED, "error_type": type(error).__name__,
                                            "error": str(error), "failure_stage": current, "traceback": traceback.format_exc()})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
