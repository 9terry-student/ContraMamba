"""Build the Stage181-A unique-item stratified roadmap without model execution.

This analyzer joins frozen Stage176--180 artifacts by source ``row_id``.  It is
descriptive and specification-only: it never imports a model, changes data,
fits a statistic, or applies an annotation recommendation.
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


STAGE = "Stage181-A"
S180A_READY = "STAGE180A_HARD_FRAME_MANUAL_REVIEW_PACKET_READY"
S180B_MIXED = "STAGE180B_HARD_FRAME_MANUAL_TAXONOMY_MIXED"
BLOCKED = "STAGE181A_STRATIFIED_FRAME_ROADMAP_BLOCKED"
DUAL = "STAGE181A_PROVISIONAL_DUAL_TRACK_ROADMAP_READY"
DATA_FIRST = "STAGE181A_PROVISIONAL_DATA_REPAIR_ROADMAP_READY"
MODEL_FIRST = "STAGE181A_PROVISIONAL_MODEL_FAILURE_ROADMAP_READY"
ADJUDICATE = "STAGE181A_SECOND_REVIEW_OR_HUMAN_ADJUDICATION_REQUIRED"

A = "DATA_INTERVENTION_REPAIR_CANDIDATE"
B = "CLEAN_LABEL_MODEL_FAILURE_CANDIDATE"
C = "GENUINELY_HARD_SEMANTIC_CANDIDATE"
D = "ADJUDICATION_HOLD"
E = "CLEAN_CONTROL_REFERENCE"
F = "CONTROL_ANOMALY"
G = "MIXED_EVIDENCE"
STRATA = (A, B, C, D, E, F, G)

DATA_VALIDITY = {"valid_but_multi_axis_edit", "weak_or_ineffective_edit", "unnatural_or_broken_text"}
CLEAN_VALIDITY = {"clean_single_axis_edit", "canonical_control"}
BROKEN_WEAK = {"weak_or_ineffective_edit", "unnatural_or_broken_text"}
DATA_ACTIONS = {"manual_adjudication", "revise_frame_label", "rewrite_claim_or_evidence",
                "redesign_intervention", "exclude_from_training", "retain_as_diagnostic_only"}
MODEL_LOCI = {"input_representation_insensitivity", "head_direction_or_readout", "downstream_final_boundary"}
HARD_LOCI = {"genuinely_hard_semantic_case", "cannot_determine"}
UNCERTAIN_JUDGMENTS = {"ambiguous", "insufficient_context"}

OUTPUTS = [
    "stage181a_stratified_frame_roadmap_report.json",
    "stage181a_stratified_frame_roadmap_report.md",
    "stage181a_unique_item_roadmap.csv",
    "stage181a_stratum_summary.csv",
    "stage181a_data_intervention_repair_queue.csv",
    "stage181a_clean_model_failure_queue.csv",
    "stage181a_genuinely_hard_semantic_queue.csv",
    "stage181a_adjudication_hold_queue.csv",
    "stage181a_control_reference_and_anomaly.csv",
    "stage181a_matched_control_contrast.csv",
    "stage181a_beneficial_harmful_stratification.csv",
    "stage181a_intervention_family_roadmap.csv",
    "stage181a_priority_ranking.csv",
    "stage181a_decision_evidence.csv",
]

UNIQUE_COLUMNS = [
    "source_row_id", "item_role", "primary_stratum", "secondary_tags", "stage176_cohort",
    "intervention_family", "native_frame_label", "native_frame_prediction", "native_frame_correct",
    "independent_frame_judgment", "pass1_confidence", "gold_frame_assessment",
    "intervention_validity", "primary_semantic_phenomenon", "diagnostic_failure_locus",
    "recommended_data_action", "pass2_confidence", "repeat_consistency",
    "matched_control_source_id", "matched_control_primary_stratum", "matched_control_taxonomy",
    "stage179_diagnostic_class", "stage179_centroid_prediction", "stage179_centroid_correct",
    "stage179_frame_logit", "stage179_head_direction_projection", "data_priority_score",
    "model_priority_score", "classification_reason",
]
SUMMARY_COLUMNS = ["primary_stratum", "item_role", "count", "denominator", "rate"]
CONTRAST_COLUMNS = [
    "hard_source_row_id", "control_source_row_id", "same_pass1_judgment", "same_gold_assessment",
    "same_intervention_validity", "same_semantic_phenomenon", "same_failure_locus",
    "hard_only_issue", "control_shared_issue", "matched_control_anomaly", "interpretation",
]
COHORT_COLUMNS = [
    "view", "stage176_cohort", "category", "count", "denominator", "rate",
    "other_cohort_count", "other_cohort_denominator", "other_cohort_rate", "risk_difference",
    "fisher_exact_p", "interpretation",
]
FAMILY_COLUMNS = [
    "intervention_family", "hard_count", "control_count", "data_repair_candidate_count",
    "clean_model_failure_candidate_count", "hard_semantic_candidate_count", "hold_mixed_count",
    "control_anomaly_count", "high_confidence_count", "high_confidence_rate",
    "beneficial_count", "harmful_count", "support_below_three", "priority_gate_used",
]
PRIORITY_COLUMNS = [
    "rank", "track", "source_row_id", "primary_stratum", "priority_score",
    "pass2_confidence", "intervention_family", "semantic_phenomenon", "matched_control_clean",
    "score_basis", "score_is_fitted_metric",
]
DECISION_COLUMNS = [
    "gate", "passed", "selected", "data_candidate_count", "data_candidate_rate",
    "model_candidate_count", "model_candidate_rate", "hold_mixed_count", "hold_mixed_rate",
    "control_anomaly_count", "control_anomaly_rate", "label_data_issue_count",
    "matched_controls_clean_for_model_candidates", "controls_explain_contrast", "reason",
]
CSV_SCHEMAS = {
    OUTPUTS[2]: UNIQUE_COLUMNS, OUTPUTS[3]: SUMMARY_COLUMNS,
    OUTPUTS[4]: UNIQUE_COLUMNS, OUTPUTS[5]: UNIQUE_COLUMNS, OUTPUTS[6]: UNIQUE_COLUMNS,
    OUTPUTS[7]: UNIQUE_COLUMNS, OUTPUTS[8]: UNIQUE_COLUMNS, OUTPUTS[9]: CONTRAST_COLUMNS,
    OUTPUTS[10]: COHORT_COLUMNS, OUTPUTS[11]: FAMILY_COLUMNS,
    OUTPUTS[12]: PRIORITY_COLUMNS, OUTPUTS[13]: DECISION_COLUMNS,
}


class RoadmapBlocked(ValueError):
    """A required frozen-artifact contract failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RoadmapBlocked(message)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RoadmapBlocked(f"cannot read JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def read_csv(path: Path, required: Iterable[str], allow_empty: bool = False) -> tuple[list[dict[str, str]], list[str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            header = list(reader.fieldnames or [])
            rows = [dict(row) for row in reader]
    except OSError as exc:
        raise RoadmapBlocked(f"cannot read CSV {path}: {exc}") from exc
    missing = sorted(set(required) - set(header))
    require(not missing, f"schema mismatch in {path}: missing columns {missing}")
    require(allow_empty or bool(rows), f"CSV has no rows: {path}")
    return rows, header


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def integer(value: Any, field: str) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise RoadmapBlocked(f"{field} must be an integer: {value!r}") from exc


def boolean(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    require(text in {"true", "false", "1", "0"}, f"{field} must be boolean: {value!r}")
    return text in {"true", "1"}


def optional_bool(value: Any) -> bool | None:
    if value is None or str(value).strip() == "":
        return None
    return boolean(value, "optional boolean")


def optional_number(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def csv_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    require(fields == CSV_SCHEMAS[path.name], f"unexpected output schema: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({field: csv_value(row.get(field, "")) for field in fields} for row in rows)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def two_sided_fisher(a: int, b: int, c: int, d: int) -> float:
    row1, row2, col1, total = a + b, c + d, a + c, a + b + c + d
    if total == 0:
        return 1.0
    low, high = max(0, col1 - row2), min(row1, col1)

    def probability(x: int) -> float:
        logp = (math.lgamma(col1 + 1) - math.lgamma(x + 1) - math.lgamma(col1 - x + 1)
                + math.lgamma(total - col1 + 1) - math.lgamma(row1 - x + 1)
                - math.lgamma(total - col1 - row1 + x + 1)
                - math.lgamma(total + 1) + math.lgamma(row1 + 1) + math.lgamma(total - row1 + 1))
        return math.exp(logp)

    observed = probability(a)
    return min(1.0, sum(probability(x) for x in range(low, high + 1)
                        if probability(x) <= observed + 1e-12))


def unique_index(rows: list[dict[str, Any]], field: str, name: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    duplicates = []
    for row in rows:
        key = str(row.get(field, "")).strip()
        require(key, f"empty {field} in {name}")
        if key in result:
            duplicates.append(key)
        result[key] = row
    require(not duplicates, f"duplicate IDs in {name}: {sorted(set(duplicates))}")
    return result


def report_status(report: dict[str, Any]) -> str:
    return str((report.get("reviewer_protocol") or {}).get("status")
               or report.get("review_status") or "")


def repeat_rate(report: dict[str, Any]) -> float | None:
    value = (report.get("repeat_consistency") or {}).get("exact_agreement_rate")
    if value is None:
        value = (report.get("scope") or {}).get("repeat_exact_agreement")
    return optional_number(value)


def artifact_issue(row: dict[str, Any]) -> bool:
    return (row["intervention_validity"] in DATA_VALIDITY
            or row["gold_frame_assessment"] in {"gold_questionable", "gold_likely_incorrect"}
            or row["diagnostic_failure_locus"] in {"data_or_intervention_design", "label_semantics_or_annotation"})


def model_issue(row: dict[str, Any]) -> bool:
    return row["diagnostic_failure_locus"] in MODEL_LOCI


def taxonomy_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in ("independent_frame_judgment", "gold_frame_assessment",
                                      "intervention_validity", "primary_semantic_phenomenon",
                                      "diagnostic_failure_locus", "recommended_data_action")}


def secondary_tags(row: dict[str, Any], low_confidence: bool) -> list[str]:
    tags: set[str] = {"PROVISIONAL_AI_REVIEW"}
    validity, phenomenon, locus = (row["intervention_validity"], row["primary_semantic_phenomenon"],
                                   row["diagnostic_failure_locus"])
    if validity == "unnatural_or_broken_text" or phenomenon == "lexical_or_surface_artifact":
        tags.add("DATA_GRAMMAR_ARTIFACT")
    if phenomenon == "polarity" and artifact_issue(row):
        tags.add("DATA_POLARITY_LEAK")
    if validity == "valid_but_multi_axis_edit":
        tags.add("DATA_MULTI_AXIS_EDIT")
    if validity == "weak_or_ineffective_edit":
        tags.add("DATA_WEAK_EDIT")
    if row["gold_frame_assessment"] != "gold_consistent" or row["recommended_data_action"] in DATA_ACTIONS:
        tags.add("LABEL_ADJUDICATION_REQUIRED")
    tags.update({
        "input_representation_insensitivity": "MODEL_REPRESENTATION_CANDIDATE",
        "head_direction_or_readout": "MODEL_READOUT_CANDIDATE",
        "downstream_final_boundary": "MODEL_FINAL_BOUNDARY_CANDIDATE",
    }.get(locus, "") for _ in [0])
    tags.discard("")
    semantic = {"referent_resolution": "HARD_REFERENT_RESOLUTION", "predicate_scope": "HARD_PREDICATE_SCOPE",
                "polarity": "HARD_POLARITY", "temporal_scope": "HARD_TEMPORAL",
                "event_identity": "HARD_EVENT_IDENTITY"}
    if phenomenon in semantic:
        tags.add(semantic[phenomenon])
    if row["item_role"] == "control" and (artifact_issue(row) or model_issue(row)):
        tags.add("CONTROL_CONTAMINATION")
    if low_confidence:
        tags.add("LOW_CONFIDENCE")
    return sorted(tags)


def classify(row: dict[str, Any], minimum: int, repeat_consistent: bool) -> tuple[str, str]:
    p1, p2 = row["pass1_confidence"], row["pass2_confidence"]
    low = p1 < minimum or p2 < minimum
    questionable_unclear = (row["gold_frame_assessment"] == "gold_questionable"
                             and row["recommended_data_action"] in {"keep_unchanged", "cannot_determine"})
    missing_axis = any(not str(row.get(field, "")).strip() for field in (
        "independent_frame_judgment", "gold_frame_assessment", "intervention_validity",
        "primary_semantic_phenomenon", "diagnostic_failure_locus", "recommended_data_action"))
    if not repeat_consistent or low or questionable_unclear or missing_axis:
        reasons = []
        if not repeat_consistent: reasons.append("repeat_annotation_inconsistent")
        if low: reasons.append("confidence_below_minimum")
        if questionable_unclear: reasons.append("questionable_gold_without_clear_action")
        if missing_axis: reasons.append("incomplete_annotation_schema")
        return D, ";".join(reasons)

    if row["item_role"] == "control":
        clean = (row["native_frame_correct"] is True
                 and row["gold_frame_assessment"] == "gold_consistent"
                 and row["intervention_validity"] in CLEAN_VALIDITY
                 and row["independent_frame_judgment"] not in UNCERTAIN_JUDGMENTS
                 and not artifact_issue(row) and not model_issue(row))
        return (E, "clean_correct_control") if clean else (F, "control_has_data_model_or_semantic_anomaly")

    data_match = (p2 >= minimum
                  and (row["gold_frame_assessment"] in {"gold_questionable", "gold_likely_incorrect"}
                       or row["intervention_validity"] in DATA_VALIDITY)
                  and row["recommended_data_action"] in DATA_ACTIONS)
    model_match = (row["gold_frame_assessment"] == "gold_consistent"
                   and row["intervention_validity"] in CLEAN_VALIDITY
                   and p2 >= minimum and model_issue(row)
                   and row["recommended_data_action"] not in {"revise_frame_label", "rewrite_claim_or_evidence",
                                                               "redesign_intervention", "exclude_from_training"})
    semantic_match = (row["gold_frame_assessment"] in {"gold_consistent", "cannot_determine"}
                      and row["intervention_validity"] not in BROKEN_WEAK
                      and (row["diagnostic_failure_locus"] in HARD_LOCI
                           or row["independent_frame_judgment"] in UNCERTAIN_JUDGMENTS))
    matched = [name for name, value in ((A, data_match), (B, model_match), (C, semantic_match)) if value]
    if (artifact_issue(row) and model_issue(row)) or len(matched) > 1:
        return G, "conflicting_data_model_or_multiple_stratum_evidence"
    if len(matched) == 1:
        return matched[0], "predeclared_stratum_conditions_met"
    return D, "no_stable_predeclared_stratum"


def score_priorities(row: dict[str, Any], control: dict[str, Any] | None,
                     phenomenon_count: Counter[str]) -> tuple[int, list[str], int, list[str]]:
    data_score, data_basis, model_score, model_basis = 0, [], 0, []
    for condition, points, label in (
        (row["intervention_validity"] == "unnatural_or_broken_text", 3, "broken_text:+3"),
        (row["gold_frame_assessment"] == "gold_likely_incorrect", 3, "likely_bad_gold:+3"),
        (row["intervention_validity"] == "valid_but_multi_axis_edit", 2, "multi_axis:+2"),
        (row["intervention_validity"] == "weak_or_ineffective_edit", 2, "weak_edit:+2"),
        (row["recommended_data_action"] == "redesign_intervention", 2, "redesign:+2"),
        (row["recommended_data_action"] == "rewrite_claim_or_evidence", 2, "rewrite:+2"),
        (row["recommended_data_action"] == "manual_adjudication", 1, "manual_adjudication:+1"),
        (row["pass2_confidence"] == 5, 1, "confidence_5:+1"),
        (control is not None and artifact_issue(row) and artifact_issue(control), 1, "control_shares_artifact:+1"),
    ):
        if condition:
            data_score += points; data_basis.append(label)
    clean = row["gold_frame_assessment"] == "gold_consistent" and row["intervention_validity"] in CLEAN_VALIDITY
    for condition, points, label in (
        (clean and row["diagnostic_failure_locus"] == "input_representation_insensitivity", 3, "clean_representation_failure:+3"),
        (clean and row["diagnostic_failure_locus"] == "downstream_final_boundary", 3, "clean_boundary_failure:+3"),
        (row["diagnostic_failure_locus"] == "head_direction_or_readout", 2, "readout_candidate:+2"),
        (row["pass2_confidence"] == 5, 2, "confidence_5:+2"),
        (control is not None and control["primary_stratum"] == E, 2, "matched_control_clean:+2"),
        (row["pass2_confidence"] == 4, 1, "confidence_4:+1"),
        (phenomenon_count[row["primary_semantic_phenomenon"]] >= 3, 1, "phenomenon_support_ge_3:+1"),
    ):
        if condition:
            model_score += points; model_basis.append(label)
    return data_score, data_basis, model_score, model_basis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage180a-manifest", type=Path, required=True)
    parser.add_argument("--stage180a-hidden-item-key", type=Path, required=True)
    parser.add_argument("--stage180b-report", type=Path, required=True)
    parser.add_argument("--stage180b-item-level-adjudication", type=Path, required=True)
    parser.add_argument("--stage180b-hard-control-comparison", type=Path, required=True)
    parser.add_argument("--stage180b-beneficial-harmful-taxonomy", type=Path, required=True)
    parser.add_argument("--stage180b-high-confidence-review-queue", type=Path, required=True)
    parser.add_argument("--stage179a-hard39-attribution", type=Path, required=True)
    parser.add_argument("--stage176a-row-transitions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-high-confidence", type=int, default=4)
    return parser.parse_args()


def decision_gate(items: list[dict[str, Any]], contrasts: list[dict[str, Any]]) -> tuple[str, list[str], list[dict[str, Any]], dict[str, Any]]:
    hard = [row for row in items if row["item_role"] == "hard"]
    controls = [row for row in items if row["item_role"] == "control"]
    data_count = sum(row["primary_stratum"] == A for row in hard)
    model_count = sum(row["primary_stratum"] == B for row in hard)
    hold_mixed = sum(row["primary_stratum"] in {D, G} for row in hard)
    anomalies = sum(row["primary_stratum"] == F for row in controls)
    data_rate, model_rate = data_count / 39, model_count / 39
    hold_rate, anomaly_rate = hold_mixed / 39, anomalies / 39
    label_data = sum(artifact_issue(row) for row in hard)
    model_ids = {row["source_row_id"] for row in hard if row["primary_stratum"] == B}
    relevant = [row for row in contrasts if row["hard_source_row_id"] in model_ids]
    model_controls_clean = bool(relevant) and all(not row["matched_control_anomaly"] for row in relevant)
    controls_explain = (anomaly_rate >= max(data_rate, model_rate)) if max(data_rate, model_rate) else False
    unstable = hold_rate >= .30 or anomaly_rate >= .20
    dual_pass = data_count >= 5 and model_count >= 5 and data_rate >= .15 and model_rate >= .15 and not controls_explain
    data_pass = data_count >= 8 and data_rate - model_rate >= .20 and data_rate - anomaly_rate >= .20
    model_pass = (model_count >= 8 and model_rate - data_rate >= .20 and model_controls_clean and label_data <= 3)
    if unstable:
        selected, next_stages = ADJUDICATE, ["STAGE182_SECOND_REVIEW_PACKET_OR_HUMAN_ADJUDICATION"]
        selected_reason = "adjudication safety override: hold/mixed or control-anomaly rate exceeded"
    elif dual_pass:
        selected = DUAL
        next_stages = ["STAGE182A_FRAME_DATA_REPAIR_ADJUDICATION_SPEC",
                       "STAGE182B_CLEAN_FRAME_MODEL_FAILURE_LOCALIZATION_SPEC"]
        selected_reason = "both provisional tracks met minimum count/rate and controls did not explain contrast"
    elif data_pass:
        selected, next_stages = DATA_FIRST, ["STAGE182A_FRAME_DATA_REPAIR_ADJUDICATION_SPEC"]
        selected_reason = "data-repair count and rate-difference gate passed"
    elif model_pass:
        selected = MODEL_FIRST
        next_stages = ["STAGE182A_CLEAN_FRAME_MODEL_FAILURE_LOCALIZATION_SPEC"]
        selected_reason = "clean model-failure count, contrast, control, and label/data gates passed"
    else:
        selected, next_stages = ADJUDICATE, ["STAGE182_SECOND_REVIEW_PACKET_OR_HUMAN_ADJUDICATION"]
        selected_reason = "no provisional roadmap branch was stable under a single reviewer"
    common = {"data_candidate_count": data_count, "data_candidate_rate": data_rate,
              "model_candidate_count": model_count, "model_candidate_rate": model_rate,
              "hold_mixed_count": hold_mixed, "hold_mixed_rate": hold_rate,
              "control_anomaly_count": anomalies, "control_anomaly_rate": anomaly_rate,
              "label_data_issue_count": label_data,
              "matched_controls_clean_for_model_candidates": model_controls_clean,
              "controls_explain_contrast": controls_explain}
    gates = [
        {"gate": "adjudication_safety_override", "passed": unstable, "reason": "hold/mixed >= .30 or control anomaly >= .20"},
        {"gate": "dual_track", "passed": dual_pass and not unstable, "reason": "both counts >= 5, rates >= .15, controls do not explain"},
        {"gate": "data_first", "passed": data_pass and not unstable, "reason": "data count >= 8 and rate advantages >= .20"},
        {"gate": "model_first", "passed": model_pass and not unstable, "reason": "model count >= 8, rate advantage >= .20, clean controls, data issues <= 3"},
        {"gate": "fallback_adjudication", "passed": selected == ADJUDICATE and not unstable, "reason": "no stable provisional branch"},
    ]
    for row in gates:
        row.update(common); row["selected"] = (row["gate"] == "adjudication_safety_override" and unstable
                                                or row["gate"] == "dual_track" and selected == DUAL
                                                or row["gate"] == "data_first" and selected == DATA_FIRST
                                                or row["gate"] == "model_first" and selected == MODEL_FIRST
                                                or row["gate"] == "fallback_adjudication" and selected == ADJUDICATE and not unstable)
    return selected, next_stages, gates, {**common, "reason": selected_reason}


def markdown(report: dict[str, Any]) -> str:
    counts = report["stratum_counts"]
    gate = report["stage182_gate"]
    return "\n".join([
        "# Stage181-A stratified frame data-and-model roadmap", "",
        f"**Decision:** `{report['decision']}`", "",
        "## Provisional interpretation", "",
        "This result is based on a single AI reviewer and is provisional. High confidence does not confirm a label, data edit, or causal model locus. Repeat agreement of 1.0 is reviewer self-consistency, not inter-rater reliability.", "",
        "## Unique-item normalization", "",
        f"The analysis used {report['unique_item_topology']['unique_hard']} hard and {report['unique_item_topology']['unique_controls']} control source items. Hidden repeats validated consistency only and did not increase item counts. `row_id` was the cross-artifact identity; `stable_row_index` was not.", "",
        "## Data/model separation", "",
        f"Hard-item strata: data repair {counts.get(A, 0)}, clean model failure {counts.get(B, 0)}, genuinely hard semantic {counts.get(C, 0)}, adjudication hold {counts.get(D, 0)}, and mixed evidence {counts.get(G, 0)}. These are roadmap queues, not applied actions or causal findings.", "",
        "## Matched controls", "",
        f"There were {report['matched_control_analysis']['control_anomalies']} control anomalies. A clean matched control strengthens a hard-only model candidate; a shared artifact redirects interpretation toward pair/template construction; an anomalous control weakens the contrast.", "",
        "## Beneficial/harmful cohorts", "",
        "Beneficial-25 and harmful-14 differences, risk differences, and Fisher exact tests are descriptive only. No causal claim follows from a single reviewer or these selected cohorts.", "",
        "## Intervention-family roadmap", "",
        "Families are summarized descriptively. Support below three is flagged but never used as a priority gate.", "",
        "## Priority ranking", "",
        "Data and model priority scores use fixed rules solely for sorting. They are not fitted metrics, probabilities, or scientific effect estimates.", "",
        "## Authorization boundary", "",
        "No dataset row was edited or relabeled. No filtering, exclusion, training, loss or architecture change, model forward, checkpoint operation, calibration, threshold search, external evaluation, time-swap, or multi-seed work was performed or authorized.", "",
        "## Stage182 route", "",
        f"Authorized specification/adjudication route(s): {', '.join(f'`{value}`' for value in gate['authorized_next_stages'])}. Stage182 remains specification or adjudication only.", "",
        "## Limitations", "",
        "The taxonomy is observational, matched controls are deterministic rather than randomized, and a second reviewer or human adjudication is required before dataset-level or paper-level human claims.", "",
    ])


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    current = "argument_validation"
    diagnostics: dict[str, Any] = {"missing_ids": [], "extra_ids": [], "duplicate_ids": [], "schema_mismatch": []}
    try:
        require(1 <= args.minimum_high_confidence <= 5, "minimum-high-confidence must be in 1..5")
        paths = {name.replace("-", "_"): getattr(args, name.replace("-", "_")) for name in (
            "stage180a-manifest", "stage180a-hidden-item-key", "stage180b-report",
            "stage180b-item-level-adjudication", "stage180b-hard-control-comparison",
            "stage180b-beneficial-harmful-taxonomy", "stage180b-high-confidence-review-queue",
            "stage179a-hard39-attribution", "stage176a-row-transitions")}
        paths = {key: value.resolve() for key, value in paths.items()}
        manifest, stage180_report = read_json(paths["stage180a_manifest"]), read_json(paths["stage180b_report"])
        require(manifest.get("decision") == S180A_READY, f"Stage180-A decision must be {S180A_READY}")
        require(stage180_report.get("decision") == S180B_MIXED, f"Stage180-B decision must be {S180B_MIXED}")
        require(report_status(stage180_report) == "single_reviewer_provisional", "review status must be single_reviewer_provisional")
        observed_repeat = repeat_rate(stage180_report)
        require(observed_repeat == 1.0, f"repeat exact agreement must be 1.0, got {observed_repeat}")

        current = "input_schema_validation"
        hidden, _ = read_csv(paths["stage180a_hidden_item_key"],
                             ["review_instance_id", "row_id", "item_role", "stage176_cohort", "intervention_type",
                              "native_frame_label", "match_link_id", "matched_hard_row_id", "is_repeat", "repeat_group_id"])
        expected_digest = (manifest.get("input_validation") or {}).get("hidden_item_key_sha256")
        require(expected_digest and sha256(paths["stage180a_hidden_item_key"]) == expected_digest,
                "hidden item key SHA-256 mismatch")
        item_rows, _ = read_csv(paths["stage180b_item_level_adjudication"],
                                ["row_id", "item_role", "stage176_cohort", "intervention_type", "native_frame_label",
                                 "independent_frame_judgment", "gold_frame_assessment", "intervention_validity",
                                 "primary_semantic_phenomenon", "diagnostic_failure_locus", "recommended_data_action",
                                 "pass1_confidence_min", "pass2_confidence_min"])
        comparison_rows, _ = read_csv(paths["stage180b_hard_control_comparison"],
                                      ["axis", "category", "hard_count", "control_count"])
        cohort_taxonomy, _ = read_csv(paths["stage180b_beneficial_harmful_taxonomy"],
                                      ["stage176_cohort", "intervention_type", "axis", "category", "count"])
        high_queue, _ = read_csv(paths["stage180b_high_confidence_review_queue"],
                                 ["source_item_id", "hard_or_control", "confidence", "queue_reason"], allow_empty=True)
        stage179_rows, _ = read_csv(paths["stage179a_hard39_attribution"],
                                    ["row_id", "stage176_cohort", "frame_prediction", "frame_correct", "diagnostic_class",
                                     "centroid_prediction", "centroid_correct"])
        stage176_rows, _ = read_csv(paths["stage176a_row_transitions"],
                                    ["row_id", "intervention_type", "correctness_transition"])
        require(bool(comparison_rows) and bool(cohort_taxonomy), "Stage180-B aggregate evidence inputs must be non-empty")

        current = "identity_topology_validation"
        originals = [row for row in hidden if not boolean(row["is_repeat"], "is_repeat")]
        repeat_source_ids = {row["row_id"] for row in hidden if boolean(row["is_repeat"], "is_repeat")}
        hidden_by_id = unique_index(originals, "row_id", "Stage180-A hidden originals")
        items_by_id = unique_index(item_rows, "row_id", "Stage180-B item adjudication")
        hard_ids = {key for key, row in hidden_by_id.items() if row["item_role"] == "hard"}
        control_ids = {key for key, row in hidden_by_id.items() if row["item_role"] == "control"}
        require(len(hard_ids) == 39, f"unique hard topology must be 39, got {len(hard_ids)}")
        require(len(control_ids) == 39, f"unique control topology must be 39, got {len(control_ids)}")
        require(not hard_ids & control_ids, "hard/control overlap is non-empty")
        diagnostics["missing_ids"] = sorted((hard_ids | control_ids) - set(items_by_id))
        diagnostics["extra_ids"] = sorted(set(items_by_id) - (hard_ids | control_ids))
        require(not diagnostics["missing_ids"] and not diagnostics["extra_ids"],
                f"Stage180 item identity mismatch: missing={diagnostics['missing_ids']}, extra={diagnostics['extra_ids']}")
        require(all(items_by_id[key]["item_role"] == hidden_by_id[key]["item_role"] for key in hidden_by_id),
                "Stage180 item role conflicts with hidden key")

        beneficial_ids = {row["row_id"] for row in stage176_rows if row["correctness_transition"] == "incorrect_to_correct"}
        harmful_ids = {row["row_id"] for row in stage176_rows if row["correctness_transition"] == "correct_to_incorrect"}
        require(len(beneficial_ids) == 25 and len(harmful_ids) == 14,
                f"Stage176 topology must be beneficial-25/harmful-14, got {len(beneficial_ids)}/{len(harmful_ids)}")
        diagnostics["missing_ids"] = sorted(hard_ids - (beneficial_ids | harmful_ids))
        diagnostics["extra_ids"] = sorted((beneficial_ids | harmful_ids) - hard_ids)
        require(not diagnostics["missing_ids"] and not diagnostics["extra_ids"],
                f"Stage176/Stage180 hard ID mismatch: {diagnostics}")
        stage179_by_id = unique_index(stage179_rows, "row_id", "Stage179 hard-39 attribution")
        diagnostics["missing_ids"] = sorted(hard_ids - set(stage179_by_id))
        diagnostics["extra_ids"] = sorted(set(stage179_by_id) - hard_ids)
        require(not diagnostics["missing_ids"] and not diagnostics["extra_ids"],
                f"Stage179/Stage180 hard ID mismatch: {diagnostics}")
        require({items_by_id[key]["stage176_cohort"] for key in hard_ids} == {"beneficial_correction", "harmful_regression"},
                "Stage180 hard cohort values are invalid")
        require(sum(items_by_id[key]["stage176_cohort"] == "beneficial_correction" for key in hard_ids) == 25,
                "Stage180 beneficial cohort count is not 25")
        require(sum(items_by_id[key]["stage176_cohort"] == "harmful_regression" for key in hard_ids) == 14,
                "Stage180 harmful cohort count is not 14")

        links: dict[str, dict[str, str]] = defaultdict(dict)
        for key, row in hidden_by_id.items():
            link = str(row["match_link_id"]).strip()
            require(link, f"empty match_link_id for {key}")
            require(row["item_role"] not in links[link], f"duplicate {row['item_role']} match link {link}")
            links[link][row["item_role"]] = key
        require(len(links) == 39 and all(set(value) == {"hard", "control"} for value in links.values()),
                "matched hard/control topology is not 39 one-to-one pairs")
        control_for_hard = {value["hard"]: value["control"] for value in links.values()}

        current = "unique_item_normalization"
        repeat_consistent = {key: observed_repeat == 1.0 for key in repeat_source_ids}
        items: list[dict[str, Any]] = []
        for source_id in sorted(hidden_by_id):
            source, hidden_row = items_by_id[source_id], hidden_by_id[source_id]
            role = hidden_row["item_role"]
            p1 = integer(source["pass1_confidence_min"], "pass1_confidence_min")
            p2 = integer(source["pass2_confidence_min"], "pass2_confidence_min")
            require(1 <= p1 <= 5 and 1 <= p2 <= 5, f"confidence outside 1..5 for {source_id}")
            native = integer(source["native_frame_label"], "native_frame_label")
            if role == "hard":
                diagnostic = stage179_by_id[source_id]
                prediction = integer(diagnostic["frame_prediction"], "frame_prediction")
                correct = boolean(diagnostic["frame_correct"], "frame_correct")
            else:
                diagnostic = {}
                require((manifest.get("control_selection") or {}).get("native_frame_correct_only") is True,
                        "control prediction cannot be inferred without native_frame_correct_only manifest contract")
                prediction, correct = native, True
            row: dict[str, Any] = {
                "source_row_id": source_id, "item_role": role,
                "stage176_cohort": source["stage176_cohort"], "intervention_family": source["intervention_type"],
                "native_frame_label": native, "native_frame_prediction": prediction, "native_frame_correct": correct,
                "independent_frame_judgment": source["independent_frame_judgment"], "pass1_confidence": p1,
                "gold_frame_assessment": source["gold_frame_assessment"],
                "intervention_validity": source["intervention_validity"],
                "primary_semantic_phenomenon": source["primary_semantic_phenomenon"],
                "diagnostic_failure_locus": source["diagnostic_failure_locus"],
                "recommended_data_action": source["recommended_data_action"], "pass2_confidence": p2,
                "repeat_consistency": "exact" if repeat_consistent.get(source_id, True) else "inconsistent",
                "matched_control_source_id": control_for_hard.get(source_id, ""),
                "stage179_diagnostic_class": diagnostic.get("diagnostic_class", "control_selected_native_correct"),
                "stage179_centroid_prediction": diagnostic.get("centroid_prediction", ""),
                "stage179_centroid_correct": diagnostic.get("centroid_correct", ""),
                "stage179_frame_logit": diagnostic.get("frame_logit", ""),
                "stage179_head_direction_projection": diagnostic.get("head_direction_projection", ""),
            }
            primary, reason = classify(row, args.minimum_high_confidence, repeat_consistent.get(source_id, True))
            row["primary_stratum"], row["classification_reason"] = primary, reason
            row["secondary_tags"] = secondary_tags(row, p1 < args.minimum_high_confidence or p2 < args.minimum_high_confidence)
            items.append(row)
        by_id = {row["source_row_id"]: row for row in items}
        for hard_id, control_id in control_for_hard.items():
            by_id[hard_id]["matched_control_primary_stratum"] = by_id[control_id]["primary_stratum"]
            by_id[hard_id]["matched_control_taxonomy"] = taxonomy_snapshot(by_id[control_id])
        for row in items:
            row.setdefault("matched_control_primary_stratum", "")
            row.setdefault("matched_control_taxonomy", "")
        require(all("PROVISIONAL_AI_REVIEW" in row["secondary_tags"] for row in items),
                "mandatory provisional tag was lost")

        current = "matched_control_analysis"
        contrasts = []
        for hard_id, control_id in sorted(control_for_hard.items()):
            hard, control = by_id[hard_id], by_id[control_id]
            hard_only = artifact_issue(hard) and not artifact_issue(control) or model_issue(hard) and not model_issue(control)
            shared = artifact_issue(hard) and artifact_issue(control)
            anomaly = control["primary_stratum"] == F
            interpretation = ("matched_control_anomaly_weakens_contrast" if anomaly else
                              "pair_or_template_construction_evidence" if shared else
                              "hard_only_model_failure_evidence" if model_issue(hard) else
                              "hard_only_intervention_construction_evidence" if artifact_issue(hard) else
                              "no_axis_specific_contrast")
            contrasts.append({"hard_source_row_id": hard_id, "control_source_row_id": control_id,
                              "same_pass1_judgment": hard["independent_frame_judgment"] == control["independent_frame_judgment"],
                              "same_gold_assessment": hard["gold_frame_assessment"] == control["gold_frame_assessment"],
                              "same_intervention_validity": hard["intervention_validity"] == control["intervention_validity"],
                              "same_semantic_phenomenon": hard["primary_semantic_phenomenon"] == control["primary_semantic_phenomenon"],
                              "same_failure_locus": hard["diagnostic_failure_locus"] == control["diagnostic_failure_locus"],
                              "hard_only_issue": hard_only, "control_shared_issue": shared,
                              "matched_control_anomaly": anomaly, "interpretation": interpretation})

        current = "priority_and_aggregation"
        phenomena = Counter(row["primary_semantic_phenomenon"] for row in items if row["item_role"] == "hard")
        priority = []
        for row in items:
            if row["item_role"] != "hard":
                row["data_priority_score"] = row["model_priority_score"] = 0
                continue
            control = by_id[control_for_hard[row["source_row_id"]]]
            ds, db, ms, mb = score_priorities(row, control, phenomena)
            row["data_priority_score"], row["model_priority_score"] = ds, ms
            for track, score, basis in (("data", ds, db), ("model", ms, mb)):
                priority.append({"track": track, "source_row_id": row["source_row_id"],
                                 "primary_stratum": row["primary_stratum"], "priority_score": score,
                                 "pass2_confidence": row["pass2_confidence"],
                                 "intervention_family": row["intervention_family"],
                                 "semantic_phenomenon": row["primary_semantic_phenomenon"],
                                 "matched_control_clean": control["primary_stratum"] == E,
                                 "score_basis": basis, "score_is_fitted_metric": False})
        priority.sort(key=lambda row: (row["track"], -row["priority_score"], -row["pass2_confidence"], row["source_row_id"]))
        ranks = defaultdict(int)
        for row in priority:
            ranks[row["track"]] += 1; row["rank"] = ranks[row["track"]]

        summaries = []
        for role in ("hard", "control"):
            group = [row for row in items if row["item_role"] == role]
            for stratum in STRATA:
                count = sum(row["primary_stratum"] == stratum for row in group)
                summaries.append({"primary_stratum": stratum, "item_role": role, "count": count,
                                  "denominator": len(group), "rate": rate(count, len(group))})

        cohort_rows = []
        hard = [row for row in items if row["item_role"] == "hard"]
        beneficial = [row for row in hard if row["stage176_cohort"] == "beneficial_correction"]
        harmful = [row for row in hard if row["stage176_cohort"] == "harmful_regression"]
        categories = [("roadmap", A), ("roadmap", B), ("roadmap", C), ("roadmap", "HOLD_OR_MIXED")]
        categories += [("semantic_phenomenon", value) for value in sorted({row["primary_semantic_phenomenon"] for row in hard})]
        categories += [("intervention_family", value) for value in sorted({row["intervention_family"] for row in hard})]
        for view, category in categories:
            def matches(row: dict[str, Any]) -> bool:
                if view == "roadmap":
                    return row["primary_stratum"] in {D, G} if category == "HOLD_OR_MIXED" else row["primary_stratum"] == category
                return row[view] == category
            a, c = sum(matches(row) for row in beneficial), sum(matches(row) for row in harmful)
            p1, p2 = a / 25, c / 14
            fisher = two_sided_fisher(a, 25 - a, c, 14 - c)
            for name, count, denom, own_rate, other_count, other_denom, other_rate, rd in (
                ("beneficial_correction", a, 25, p1, c, 14, p2, p1 - p2),
                ("harmful_regression", c, 14, p2, a, 25, p1, p2 - p1)):
                cohort_rows.append({"view": view, "stage176_cohort": name, "category": category,
                                    "count": count, "denominator": denom, "rate": own_rate,
                                    "other_cohort_count": other_count, "other_cohort_denominator": other_denom,
                                    "other_cohort_rate": other_rate, "risk_difference": rd,
                                    "fisher_exact_p": fisher, "interpretation": "descriptive_only_no_causal_claim"})

        family_rows = []
        for family in sorted({row["intervention_family"] for row in items}):
            group = [row for row in items if row["intervention_family"] == family]
            hard_group = [row for row in group if row["item_role"] == "hard"]
            high = sum(row["pass2_confidence"] >= args.minimum_high_confidence for row in group)
            family_rows.append({"intervention_family": family, "hard_count": len(hard_group),
                                "control_count": sum(row["item_role"] == "control" for row in group),
                                "data_repair_candidate_count": sum(row["primary_stratum"] == A for row in hard_group),
                                "clean_model_failure_candidate_count": sum(row["primary_stratum"] == B for row in hard_group),
                                "hard_semantic_candidate_count": sum(row["primary_stratum"] == C for row in hard_group),
                                "hold_mixed_count": sum(row["primary_stratum"] in {D, G} for row in hard_group),
                                "control_anomaly_count": sum(row["primary_stratum"] == F for row in group),
                                "high_confidence_count": high, "high_confidence_rate": rate(high, len(group)),
                                "beneficial_count": sum(row["stage176_cohort"] == "beneficial_correction" for row in hard_group),
                                "harmful_count": sum(row["stage176_cohort"] == "harmful_regression" for row in hard_group),
                                "support_below_three": len(hard_group) < 3, "priority_gate_used": False})

        selected, next_stages, decision_rows, diagnosis = decision_gate(items, contrasts)
        stratum_counts = dict(Counter(row["primary_stratum"] for row in items))
        input_hashes = {key: sha256(path) for key, path in paths.items()}
        report = {
            "stage": STAGE, "decision": selected,
            "scope": {"unit": "unique_source_item", "clean_controlled_frozen_artifacts_only": True,
                      "analysis_only": True, "minimum_high_confidence": args.minimum_high_confidence},
            "input_validation": {"status": "passed", "stage180a_decision": S180A_READY,
                                 "stage180b_decision": S180B_MIXED, "input_sha256": input_hashes,
                                 "missing_ids": [], "extra_ids": [], "duplicate_ids": [], "schema_mismatch": [],
                                 "stage180b_aggregate_comparison_rows": len(comparison_rows),
                                 "stage180b_cohort_taxonomy_rows": len(cohort_taxonomy),
                                 "stage180b_high_confidence_queue_rows": len(high_queue)},
            "reviewer_status": {"status": "single_reviewer_provisional", "ai_assisted": True,
                                "human_annotation": False, "repeat_exact_agreement": observed_repeat,
                                "repeat_is_inter_rater_reliability": False},
            "unique_item_topology": {"unique_hard": 39, "unique_controls": 39, "beneficial": 25,
                                     "harmful": 14, "hard_control_overlap": 0,
                                     "repeat_source_items": len(repeat_source_ids),
                                     "repeats_counted_as_items": False, "identity_key": "row_id",
                                     "stable_row_index_is_identity": False},
            "stratification_policy": {"primary_strata": list(STRATA), "exactly_one_primary": True,
                                      "secondary_tags": sorted({tag for row in items for tag in row["secondary_tags"]}),
                                      "mandatory_tag": "PROVISIONAL_AI_REVIEW"},
            "stratum_counts": stratum_counts,
            "matched_control_analysis": {"pairs": len(contrasts),
                                         "hard_only_issue_pairs": sum(row["hard_only_issue"] for row in contrasts),
                                         "control_shared_issue_pairs": sum(row["control_shared_issue"] for row in contrasts),
                                         "control_anomalies": sum(row["matched_control_anomaly"] for row in contrasts)},
            "beneficial_harmful_analysis": {"beneficial_rows": 25, "harmful_rows": 14,
                                            "summary_rows": len(cohort_rows), "fisher_exact": "two_sided",
                                            "interpretation": "descriptive_only"},
            "intervention_family_roadmap": {"families": len(family_rows), "rows": family_rows,
                                            "minimum_support_priority_gate": False},
            "priority_queues": {"data_rows": sum(row["track"] == "data" for row in priority),
                                "model_rows": sum(row["track"] == "model" for row in priority),
                                "rule_based_sorting_only": True, "fitted_metric": False},
            "diagnosis": diagnosis,
            "stage182_gate": {"decision": selected, "authorized_next_stages": next_stages,
                              "specification_or_adjudication_only": True, "execution_authorized": False},
            "limitations": ["Single AI reviewer annotations are provisional and are not human annotation.",
                            "Repeat agreement is self-consistency, not inter-rater reliability.",
                            "Matched controls are deterministic, not randomized.",
                            "Failure loci and cohort contrasts do not establish causality."],
            "safety_policy": {"dataset_modification": False, "automatic_relabeling": False,
                              "filtering": False, "exclusion": False, "training_subset_construction": False,
                              "model_import": False, "model_forward": False, "training": False,
                              "loss_modification": False, "architecture_modification": False,
                              "checkpoint_load_or_modification": False, "fitting": False,
                              "calibration": False, "threshold_search": False,
                              "external_evaluation": False, "time_swap": False, "multi_seed": False},
        }

        current = "output_write"
        output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(output_dir / OUTPUTS[2], items, UNIQUE_COLUMNS)
        write_csv(output_dir / OUTPUTS[3], summaries, SUMMARY_COLUMNS)
        write_csv(output_dir / OUTPUTS[4], [row for row in items if row["primary_stratum"] == A], UNIQUE_COLUMNS)
        write_csv(output_dir / OUTPUTS[5], [row for row in items if row["primary_stratum"] == B], UNIQUE_COLUMNS)
        write_csv(output_dir / OUTPUTS[6], [row for row in items if row["primary_stratum"] == C], UNIQUE_COLUMNS)
        write_csv(output_dir / OUTPUTS[7], [row for row in items if row["primary_stratum"] in {D, G}], UNIQUE_COLUMNS)
        write_csv(output_dir / OUTPUTS[8], [row for row in items if row["item_role"] == "control"], UNIQUE_COLUMNS)
        write_csv(output_dir / OUTPUTS[9], contrasts, CONTRAST_COLUMNS)
        write_csv(output_dir / OUTPUTS[10], cohort_rows, COHORT_COLUMNS)
        write_csv(output_dir / OUTPUTS[11], family_rows, FAMILY_COLUMNS)
        write_csv(output_dir / OUTPUTS[12], priority, PRIORITY_COLUMNS)
        write_csv(output_dir / OUTPUTS[13], decision_rows, DECISION_COLUMNS)
        write_json(output_dir / OUTPUTS[0], report)
        (output_dir / OUTPUTS[1]).write_text(markdown(report), encoding="utf-8")
        return 0
    except Exception as error:
        output_dir.mkdir(parents=True, exist_ok=True)
        blocked = {"stage": STAGE, "decision": BLOCKED, "error_type": type(error).__name__,
                   "error": str(error), "failure_stage": current, "traceback": traceback.format_exc(),
                   "missing_ids": diagnostics["missing_ids"], "extra_ids": diagnostics["extra_ids"],
                   "duplicate_ids": diagnostics["duplicate_ids"], "schema_mismatch": diagnostics["schema_mismatch"]}
        write_json(output_dir / OUTPUTS[0], blocked)
        (output_dir / OUTPUTS[1]).write_text(
            "# Stage181-A blocked\n\n"
            f"**Decision:** `{BLOCKED}`\n\nFailure stage: `{current}`\n\nError: `{error}`\n",
            encoding="utf-8")
        for filename, fields in CSV_SCHEMAS.items():
            write_csv(output_dir / filename, [], fields)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

