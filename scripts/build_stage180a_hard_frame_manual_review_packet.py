"""Build the blinded Stage180-A hard-frame manual-review packet.

Data packaging only: this module never imports a model, loads a checkpoint, or
performs a forward pass.  Native labels are copied only to the unblinded packet
and hidden key; intervention metadata never determines a target.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


STAGE = "Stage180-A"
READY = "STAGE180A_HARD_FRAME_MANUAL_REVIEW_PACKET_READY"
BLOCKED = "STAGE180A_HARD_FRAME_MANUAL_REVIEW_PACKET_BLOCKED"
EXPECTED = {
    "stage176a": "STAGE176A_CLEAN_DEV_SUPPORT_BOUNDARY_ATTRIBUTION_COMPLETE",
    "stage177a": "STAGE177A_FRAME_PAIRWISE_SIGNAL_PRESENT_ABSOLUTE_DISCRIMINATION_WEAK",
    "stage177e": "STAGE177E_FRAME_PAIRWISE_OBJECTIVE_REDUNDANT_PATH_CLOSED",
    "stage178a": "STAGE178A_PAIR_OFFSET_EXPLANATION_WEAK_PATH_CLOSED",
    "stage179a": "STAGE179A_FRAME_SEMANTICS_REPRESENTATION_CAUSE_MIXED_OR_INSUFFICIENT",
}
OUTPUTS = [
    "stage180a_manual_review_manifest.json",
    "stage180a_manual_review_manifest.md",
    "stage180a_blinded_pass1_packet.csv",
    "stage180a_pass1_annotation_template.csv",
    "stage180a_unblinded_pass2_packet.csv",
    "stage180a_pass2_annotation_template.csv",
    "stage180a_hidden_item_key.csv",
    "stage180a_control_matching_summary.csv",
    "stage180a_packet_item_summary.csv",
    "stage180a_review_instructions.md",
]
BLIND_COLUMNS = ["review_instance_id", "claim", "evidence", "claim_segment", "evidence_segment", "reviewer_notes"]
PASS1_TEMPLATE_COLUMNS = ["review_instance_id", "claim", "evidence", "claim_segment", "evidence_segment",
                          "reviewer_id", "independent_frame_judgment", "pass1_confidence", "pass1_rationale"]
MATCHED_CONTROL_COLUMNS = [
    "matched_control_review_instance_id", "matched_control_match_level", "matched_control_claim",
    "matched_control_evidence", "matched_control_native_frame_label", "matched_control_final_gold_label",
    "matched_control_frame_prediction", "matched_control_frame_logit", "matched_control_frame_prob",
]
UNBLIND_COLUMNS = [
    "review_instance_id", "dataset_native_frame_label", "intervention_type", "canonical_none_claim",
    "canonical_none_evidence", "stage176_cohort", "native_frame_prediction", "native_frame_logit",
    "native_frame_probability", "final_prediction", "gold_final_label", "frame_head_projection",
    "representation_movement_from_none", "stage179_centroid_prediction", "stage179_centroid_correct",
] + MATCHED_CONTROL_COLUMNS
PASS2_TEMPLATE_COLUMNS = ["review_instance_id", "reviewer_id", "gold_frame_assessment", "intervention_validity",
                          "primary_semantic_phenomenon", "diagnostic_failure_locus", "recommended_data_action",
                          "pass2_confidence", "pass2_rationale"]
HIDDEN_COLUMNS = ["review_instance_id", "stable_row_index", "row_id", "pair_id", "item_role", "stage176_cohort",
                  "intervention_type", "native_frame_label", "gold_final_label", "match_link_id",
                  "matched_hard_row_id", "match_level", "is_repeat", "repeat_group_id"]
MATCHING_COLUMNS = ["match_link_id", "hard_row_id", "control_row_id", "match_level", "control_reused",
                    "text_length_difference", "model_confidence_used"]
ITEM_SUMMARY_COLUMNS = ["row_id", "item_role", "stage176_cohort", "review_instance_count", "match_link_id",
                        "has_direct_matched_control", "hard_items_with_direct_matched_control",
                        "hard_items_missing_direct_matched_control", "pass2_matched_control_context_included"]
CSV_SCHEMAS = {
    OUTPUTS[2]: BLIND_COLUMNS, OUTPUTS[3]: PASS1_TEMPLATE_COLUMNS, OUTPUTS[4]: UNBLIND_COLUMNS,
    OUTPUTS[5]: PASS2_TEMPLATE_COLUMNS, OUTPUTS[6]: HIDDEN_COLUMNS, OUTPUTS[7]: MATCHING_COLUMNS,
    OUTPUTS[8]: ITEM_SUMMARY_COLUMNS,
}
_MISSING = object()
STAGE176_REPORT_DEV_COUNT_PATH = ("split", "dev_rows")
STAGE176_TRANSITION_IDENTITY_COLUMN = "row_id"
STAGE179_DEV_IDENTITY_COLUMN = "row_id"


class PacketBlocked(ValueError):
    """A Stage180-A contract or provenance validation failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PacketBlocked(message)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PacketBlocked(f"cannot read JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    except OSError as exc:
        raise PacketBlocked(f"cannot read CSV {path}: {exc}") from exc
    require(bool(rows), f"CSV is empty: {path}")
    return rows


def read_data(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            value = json.loads(text)
            rows = value if isinstance(value, list) else value.get("records", value.get("data"))
    except (OSError, json.JSONDecodeError, AttributeError) as exc:
        raise PacketBlocked(f"cannot read controlled data {path}: {exc}") from exc
    require(isinstance(rows, list) and rows, f"controlled data has no row list: {path}")
    require(all(isinstance(row, dict) for row in rows), "controlled data rows must be objects")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise PacketBlocked(f"cannot hash {path}: {exc}") from exc
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


def decision(report: dict[str, Any], stage: str) -> str | None:
    if stage == "stage176a":
        return (report.get("closure") or {}).get("attribution_completion_decision") or report.get("decision")
    return report.get("decision") or report.get("execution_decision")


def as_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if str(value).strip().lower() in {"true", "1"}:
        return True
    if str(value).strip().lower() in {"false", "0", ""}:
        return False
    raise PacketBlocked(f"invalid boolean {field}: {value!r}")


def as_int(value: Any, field: str) -> int:
    try:
        result = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise PacketBlocked(f"invalid integer {field}: {value!r}") from exc
    return result


def row_id(row: dict[str, Any]) -> str:
    value = row.get("row_id", row.get("id"))
    require(value is not None and str(value).strip(), "row lacks stable row_id/id")
    return str(value)


def _get_path(mapping: dict[str, Any], path: tuple[str, ...], default: Any = _MISSING) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, dict) or key not in current:
            if default is _MISSING:
                raise KeyError(".".join(path))
            return default
        current = current[key]
    return current


def validate_dev_artifacts(stage176_rows: list[dict[str, str]], stage179_rows: list[dict[str, str]],
                           stage176_report: dict[str, Any], diagnostic: dict[str, Any]) -> dict[str, Any]:
    stage176_columns = list(stage176_rows[0]) if stage176_rows else []
    stage179_columns = list(stage179_rows[0]) if stage179_rows else []
    diagnostic.update({
        "stage176a_dev_csv_row_count": len(stage176_rows),
        "stage179a_dev_csv_row_count": len(stage179_rows),
        "stage176a_candidate_identity_columns": [column for column in ("row_id", "stable_row_index")
                                                  if column in stage176_columns],
        "stage179a_candidate_identity_columns": [column for column in ("row_id", "stable_row_index")
                                                  if column in stage179_columns],
        "stage176a_selected_identity_column": STAGE176_TRANSITION_IDENTITY_COLUMN,
        "stage179a_selected_identity_column": STAGE179_DEV_IDENTITY_COLUMN,
        "stage176a_report_dev_count_path": ".".join(STAGE176_REPORT_DEV_COUNT_PATH),
        "dev_count_authoritative_source": "stage176a_row_transitions_csv",
    })
    try:
        report_count_value = _get_path(stage176_report, STAGE176_REPORT_DEV_COUNT_PATH)
        report_count_present = True
    except KeyError:
        report_count_value = None
        report_count_present = False
    diagnostic["stage176a_report_dev_count_present"] = report_count_present
    diagnostic["stage176a_report_dev_count_raw_value"] = report_count_value
    report_count = None if report_count_value is None else as_int(report_count_value, "Stage176-A split.dev_rows")
    diagnostic["stage176a_report_dev_count"] = report_count
    require(STAGE176_TRANSITION_IDENTITY_COLUMN in stage176_columns,
            f"Stage176-A transitions lack writer identity column {STAGE176_TRANSITION_IDENTITY_COLUMN!r}")
    require(STAGE179_DEV_IDENTITY_COLUMN in stage179_columns,
            f"Stage179-A dev analysis lacks writer identity column {STAGE179_DEV_IDENTITY_COLUMN!r}")
    stage176_ids = [str(row[STAGE176_TRANSITION_IDENTITY_COLUMN]) for row in stage176_rows]
    stage179_ids = [str(row[STAGE179_DEV_IDENTITY_COLUMN]) for row in stage179_rows]
    require(all(value.strip() for value in stage176_ids), "Stage176-A transitions contain empty stable row IDs")
    require(all(value.strip() for value in stage179_ids), "Stage179-A dev analysis contains empty stable row IDs")
    stage176_set, stage179_set = set(stage176_ids), set(stage179_ids)
    only_stage176 = sorted(stage176_set - stage179_set)
    only_stage179 = sorted(stage179_set - stage176_set)
    diagnostic.update({
        "stage176a_dev_unique_row_count": len(stage176_set),
        "stage179a_dev_unique_row_count": len(stage179_set),
        "stage176a_duplicate_stable_row_id_count": len(stage176_ids) - len(stage176_set),
        "stage179a_duplicate_stable_row_id_count": len(stage179_ids) - len(stage179_set),
        "stage176a_only_identity_count": len(only_stage176),
        "stage179a_only_identity_count": len(only_stage179),
        "stage176a_only_identities": only_stage176,
        "stage179a_only_identities": only_stage179,
        "stage176a_stage179a_dev_identity_match": stage176_set == stage179_set,
    })
    require(len(stage176_rows) == 720, f"Stage176-A transition CSV must contain 720 rows, got {len(stage176_rows)}")
    require(len(stage179_rows) == 720, f"Stage179-A dev analysis CSV must contain 720 rows, got {len(stage179_rows)}")
    require(len(stage176_set) == 720, f"Stage176-A transition CSV must contain 720 unique row IDs, got {len(stage176_set)}")
    require(len(stage179_set) == 720, f"Stage179-A dev analysis CSV must contain 720 unique row IDs, got {len(stage179_set)}")
    require(stage176_set == stage179_set,
            f"Stage176-A/Stage179-A dev identity mismatch: only176={len(only_stage176)}, only179={len(only_stage179)}")
    if report_count is not None:
        require(report_count == 720, f"Stage176-A report split.dev_rows must be 720 when present, got {report_count}")
        require(report_count == len(stage176_rows), "Stage176-A report/transition CSV dev counts differ")
    return diagnostic


def cohort_from_transition(row: dict[str, Any]) -> str:
    explicit = str(row.get("stage176_cohort", row.get("cohort", ""))).strip()
    if explicit in {"beneficial_correction", "harmful_regression"}:
        return explicit
    gold = str(row.get("gold_final_label", ""))
    before = str(row.get("baseline_predicted_label", ""))
    after = str(row.get("treatment_predicted_label", ""))
    if gold == "NOT_ENTITLED" and before == "SUPPORT" and after == "NOT_ENTITLED":
        return "beneficial_correction"
    if gold == "SUPPORT" and before == "SUPPORT" and after == "NOT_ENTITLED":
        return "harmful_regression"
    return "none"


def validate_authority(codebook: Path) -> dict[str, Any]:
    try:
        text = codebook.read_text(encoding="utf-8")
    except OSError as exc:
        raise PacketBlocked(f"cannot read codebook {codebook}: {exc}") from exc
    markers = ["frame_compatible_label", "scripts/build_controlled_v5.py", "predicate_swap",
               "evidence_deletion", "polarity_flip", "entity_swap", "irrelevant_evidence"]
    require(all(marker in text for marker in markers), "codebook lacks the repository-backed authoritative frame definition")
    generator = Path(__file__).resolve().with_name("build_controlled_v5.py")
    require(generator.is_file(), "authoritative generator scripts/build_controlled_v5.py is missing")
    source = generator.read_text(encoding="utf-8")
    required_source = ["frame_compatible_label", 'fact, "predicate_swap"', 'fact, "evidence_deletion"',
                       'fact, "polarity_flip"', 'fact, "entity_swap"', 'fact, "irrelevant_evidence"']
    require(all(marker in source for marker in required_source), "authoritative generator no longer matches codebook source anchors")
    return {"status": "passed", "target_field": "frame_compatible_label",
            "schema_source": "scripts/build_controlled_v5.py::_record/_build_records",
            "codebook_sha256": sha256(codebook), "conflicting_definition_found": False}


def merge_dev(data: list[dict[str, Any]], analysis: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for item in data:
        key = row_id(item)
        require(key not in by_id, f"non-unique stable data id: {key}")
        by_id[key] = item
    ordered = sorted(analysis, key=lambda row: as_int(row.get("stable_row_index"), "stable_row_index"))
    require(len(ordered) == 720, f"dev analysis must contain 720 rows, got {len(ordered)}")
    require([as_int(row["stable_row_index"], "stable_row_index") for row in ordered] == list(range(720)),
            "dev stable_row_index must be exactly 0..719")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for diagnostic in ordered:
        key = row_id(diagnostic)
        require(key in by_id, f"dev row absent from controlled data: {key}")
        require(key not in seen, f"duplicate dev identity: {key}")
        source = by_id[key]
        for field in ("claim", "evidence", "pair_id", "intervention_type", "final_label", "frame_compatible_label"):
            require(field in source and source[field] not in (None, ""), f"dev row {key} lacks {field}")
        frame = as_int(source["frame_compatible_label"], f"{key}.frame_compatible_label")
        require(frame in (0, 1), f"non-binary frame target for {key}")
        require(str(source["intervention_type"]) != "time_swap", f"time_swap is forbidden: {key}")
        merged = {**source, **diagnostic, "row_id": key,
                  "stable_row_index": as_int(diagnostic["stable_row_index"], "stable_row_index"),
                  "frame_compatible_label": frame}
        result.append(merged)
        seen.add(key)
    return result


def validate_hard(transitions: list[dict[str, str]], attribution: list[dict[str, str]],
                  dev: list[dict[str, Any]], diagnostic: dict[str, Any]
                  ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    dev_by_id = {row_id(row): row for row in dev}
    transition_hard_rows: list[dict[str, Any]] = []
    for row in transitions:
        cohort = cohort_from_transition(row)
        if cohort in {"beneficial_correction", "harmful_regression"}:
            transition_hard_rows.append({**row, "stage176_cohort": cohort})
    transition_ids = [row_id(row) for row in transition_hard_rows]
    attribution_ids = [row_id(row) for row in attribution]
    transition_counts, attribution_counts = Counter(transition_ids), Counter(attribution_ids)
    transition_duplicates = sorted(key for key, count in transition_counts.items() if count > 1)
    attribution_duplicates = sorted(key for key, count in attribution_counts.items() if count > 1)
    transition_set, attribution_set = set(transition_ids), set(attribution_ids)
    missing_ids = sorted(transition_set - attribution_set)
    extra_ids = sorted(attribution_set - transition_set)
    outside_dev = sorted((transition_set | attribution_set) - set(dev_by_id))
    cohort_counts = Counter(row["stage176_cohort"] for row in transition_hard_rows)
    diagnostic.update({
        "cross_artifact_identity_key": "row_id",
        "row_id_cross_artifact_authoritative": True,
        "stable_index_cross_artifact_authoritative": False,
        "stage176a_hard_row_count": len(transition_ids),
        "stage179a_hard_row_count": len(attribution_ids),
        "stage176a_hard_unique_row_count": len(transition_set),
        "stage179a_hard_unique_row_count": len(attribution_set),
        "stage176a_duplicate_hard_row_ids": transition_duplicates,
        "stage179a_duplicate_hard_row_ids": attribution_duplicates,
        "missing_hard_row_ids": missing_ids,
        "extra_hard_row_ids": extra_ids,
        "hard_row_ids_outside_dev": outside_dev,
        "hard_row_identity_set_match": transition_set == attribution_set,
        "beneficial_hard_row_count": cohort_counts["beneficial_correction"],
        "harmful_hard_row_count": cohort_counts["harmful_regression"],
    })

    def local_stable(rows: list[dict[str, Any]], artifact: str) -> tuple[dict[str, int], list[str], list[dict[str, Any]], int]:
        values: dict[str, int] = {}
        missing: list[str] = []
        invalid: list[dict[str, Any]] = []
        positions: Counter[int] = Counter()
        for row in rows:
            key = row_id(row)
            raw = row.get("stable_row_index")
            if raw is None or str(raw).strip() == "":
                missing.append(key)
                continue
            try:
                value = int(str(raw).strip())
            except (TypeError, ValueError):
                invalid.append({"row_id": key, "value": raw})
                continue
            values[key] = value
            positions[value] += 1
        duplicate_count = sum(count - 1 for count in positions.values() if count > 1)
        diagnostic[f"{artifact}_hard_stable_index_missing_ids"] = sorted(missing)
        diagnostic[f"{artifact}_hard_stable_index_invalid"] = invalid
        diagnostic[f"{artifact}_hard_stable_index_duplicate_count"] = duplicate_count
        return values, missing, invalid, duplicate_count

    stage176_stable, stage176_missing, stage176_invalid, stage176_duplicate_indexes = local_stable(
        transition_hard_rows, "stage176a")
    stage179_stable, stage179_missing, stage179_invalid, stage179_duplicate_indexes = local_stable(
        attribution, "stage179a")
    stable_matches = 0
    stable_mismatches: list[dict[str, Any]] = []
    stable_missing = 0
    for key in sorted(transition_set & attribution_set):
        if key not in stage176_stable or key not in stage179_stable:
            stable_missing += 1
        elif stage176_stable[key] == stage179_stable[key]:
            stable_matches += 1
        else:
            stable_mismatches.append({"row_id": key,
                                      "stage176a_stable_row_index": stage176_stable[key],
                                      "stage179a_stable_row_index": stage179_stable[key]})
    diagnostic.update({
        "hard_stable_index_match_count": stable_matches,
        "hard_stable_index_mismatch_count": len(stable_mismatches),
        "hard_stable_index_missing_count": stable_missing,
        "stable_index_mismatch_examples": stable_mismatches[:20],
    })

    require(len(transition_ids) == 39 and len(transition_set) == 39,
            f"Stage176-A hard cohort must contain 39 unique row IDs, rows={len(transition_ids)}, unique={len(transition_set)}")
    require(len(attribution_ids) == 39 and len(attribution_set) == 39,
            f"Stage179-A hard attribution must contain 39 unique row IDs, rows={len(attribution_ids)}, unique={len(attribution_set)}")
    require(not transition_duplicates and not attribution_duplicates,
            f"duplicate hard row IDs: stage176={transition_duplicates}, stage179={attribution_duplicates}")
    require(not missing_ids and not extra_ids,
            f"hard row-ID sets differ: missing={missing_ids}, extra={extra_ids}")
    require(not outside_dev, f"hard row IDs are outside canonical dev-720 identity set: {outside_dev}")
    require(cohort_counts == Counter({"beneficial_correction": 25, "harmful_regression": 14}),
            f"hard cohort must be beneficial=25/harmful=14, got {dict(cohort_counts)}")
    require(not stage176_invalid and not stage179_invalid,
            f"non-integer artifact-local stable indexes: stage176={stage176_invalid}, stage179={stage179_invalid}")
    require(stage176_duplicate_indexes == 0 and stage179_duplicate_indexes == 0,
            f"duplicate artifact-local hard stable indexes: stage176={stage176_duplicate_indexes}, stage179={stage179_duplicate_indexes}")

    transition_by_id = {row_id(row): row for row in transition_hard_rows}
    attribution_by_id = {row_id(row): row for row in attribution}
    semantic_fields = ("pair_id", "intervention_type", "gold_frame_label", "gold_final_label", "claim", "evidence")
    semantic_mismatches: list[dict[str, Any]] = []
    checked_fields: set[str] = set()
    for key in sorted(transition_set):
        left, right = transition_by_id[key], attribution_by_id[key]
        for field in semantic_fields:
            if field in left and field in right:
                checked_fields.add(field)
                if str(left[field]) != str(right[field]):
                    semantic_mismatches.append({"row_id": key, "semantic_mismatch_field": field,
                                                "stage176a_value": left[field], "stage179a_value": right[field]})
        if "stage176_cohort" in right:
            checked_fields.add("stage176_cohort")
            if str(left["stage176_cohort"]) != str(right["stage176_cohort"]):
                semantic_mismatches.append({"row_id": key, "semantic_mismatch_field": "stage176_cohort",
                                            "stage176a_value": left["stage176_cohort"],
                                            "stage179a_value": right["stage176_cohort"]})
    diagnostic["hard_semantic_fields_checked"] = sorted(checked_fields)
    diagnostic["hard_semantic_mismatches"] = semantic_mismatches[:20]
    diagnostic["hard_semantic_field_consistency"] = not semantic_mismatches
    require(not semantic_mismatches,
            f"hard semantic mismatch: {semantic_mismatches[0] if semantic_mismatches else None}")

    hard: dict[str, dict[str, Any]] = {}
    for key, row in transition_by_id.items():
        hard[key] = {**dev_by_id[key], **row, "stage176_cohort": row["stage176_cohort"]}
    for key, row in attribution_by_id.items():
        hard[key].update({f"stage179_{field}": value for field, value in row.items()
                          if field not in {"row_id", "id"}})
    hard_check = {"status": "passed", "beneficial": 25, "harmful": 14, "total": 39,
                  "cross_artifact_identity_key": "row_id", "stage176a_hard_unique_row_count": 39,
                  "stage179a_hard_unique_row_count": 39, "hard_row_identity_set_match": True,
                  "hard_semantic_field_consistency": True, "row_id_cross_artifact_authoritative": True,
                  "stable_index_cross_artifact_authoritative": False,
                  "hard_stable_index_match_count": stable_matches,
                  "hard_stable_index_mismatch_count": len(stable_mismatches),
                  "hard_stable_index_missing_count": stable_missing,
                  "stable_index_mismatch_examples": stable_mismatches[:20]}
    return hard, hard_check


def text_length(row: dict[str, Any]) -> int:
    return len(str(row.get("claim", ""))) + len(str(row.get("evidence", "")))


def select_controls(hard: dict[str, dict[str, Any]], dev: list[dict[str, Any]], per_hard: int
                    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    require(per_hard >= 1, "controls-per-hard-row must be at least 1")
    hard_ids = set(hard)
    pool = [row for row in dev if row_id(row) not in hard_ids and as_bool(row.get("frame_correct"), "frame_correct")]
    require(pool, "native-frame-correct control pool is empty")
    unused = {row_id(row) for row in pool}
    selected: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    levels = [
        ("same_intervention_frame_final", lambda h, c: c["intervention_type"] == h["intervention_type"] and c["frame_compatible_label"] == h["frame_compatible_label"] and c["final_label"] == h["final_label"]),
        ("same_intervention_frame", lambda h, c: c["intervention_type"] == h["intervention_type"] and c["frame_compatible_label"] == h["frame_compatible_label"]),
        ("same_frame_final", lambda h, c: c["frame_compatible_label"] == h["frame_compatible_label"] and c["final_label"] == h["final_label"]),
        ("same_frame", lambda h, c: c["frame_compatible_label"] == h["frame_compatible_label"]),
        ("all_correct", lambda _h, _c: True),
    ]
    for hard_row in sorted(hard.values(), key=row_id):
        for ordinal in range(per_hard):
            chosen: dict[str, Any] | None = None
            chosen_level = ""
            reused = False
            for allow_reuse in (False, True):
                for level, predicate in levels:
                    candidates = [row for row in pool if predicate(hard_row, row) and
                                  (allow_reuse or row_id(row) in unused)]
                    if candidates:
                        chosen = min(candidates, key=lambda row: (abs(text_length(row) - text_length(hard_row)), row_id(row)))
                        chosen_level, reused = level, allow_reuse
                        break
                if chosen is not None:
                    break
            require(chosen is not None, f"no control available for {row_id(hard_row)}")
            unused.discard(row_id(chosen))
            link = f"M180-{len(selected) + 1:04d}"
            item = {**chosen, "item_role": "control", "matched_hard_row_id": row_id(hard_row),
                    "match_link_id": link, "match_level": chosen_level, "control_reused": reused,
                    "control_ordinal": ordinal + 1}
            selected.append(item)
            summaries.append({"match_link_id": link, "hard_row_id": row_id(hard_row),
                              "control_row_id": row_id(chosen), "match_level": chosen_level,
                              "control_reused": reused, "text_length_difference": abs(text_length(chosen) - text_length(hard_row)),
                              "model_confidence_used": False})
    require(not (hard_ids & {row_id(row) for row in selected}), "hard/control identity overlap")
    return selected, summaries


def canonical_none(dev: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dev:
        if str(row["intervention_type"]) == "none":
            groups[str(row["pair_id"])].append(row)
    require(all(len(rows) == 1 for rows in groups.values()), "each dev pair must have exactly one canonical none row")
    return {pair: rows[0] for pair, rows in groups.items()}


def arrange_instances(items: list[dict[str, Any]], repeat_count: int, seed: int) -> list[dict[str, Any]]:
    require(0 <= repeat_count <= len(items), "repeat-instance-count is outside valid range")
    rng = random.Random(seed)
    originals = [{**item, "is_repeat": False, "repeat_group_id": f"G180-{index + 1:04d}"}
                 for index, item in enumerate(items)]
    by_role = {role: [item for item in originals if item["item_role"] == role] for role in ("hard", "control")}
    repeat_sources: list[dict[str, Any]] = []
    targets = {"hard": (repeat_count + 1) // 2, "control": repeat_count // 2}
    for role in ("hard", "control"):
        candidates = list(by_role[role])
        rng.shuffle(candidates)
        repeat_sources.extend(candidates[:targets[role]])
    repeats = [{**item, "is_repeat": True} for item in repeat_sources]
    pending = originals + repeats
    rng.shuffle(pending)
    # Greedy reordering prevents identical semantic sources from becoming adjacent.
    ordered: list[dict[str, Any]] = []
    while pending:
        choices = [item for item in pending if not ordered or row_id(item) != row_id(ordered[-1])]
        choice = choices[0] if choices else pending[0]
        ordered.append(choice)
        pending.remove(choice)
    for index, item in enumerate(ordered, 1):
        item["review_instance_id"] = f"R{seed}-{index:04d}"
    require(len({item["review_instance_id"] for item in ordered}) == len(ordered), "duplicate review instance ID")
    return ordered


def first(row: dict[str, Any], *names: str) -> Any:
    return next((row[name] for name in names if row.get(name) not in (None, "")), "")


def attach_direct_control_context(instances: list[dict[str, Any]], matching: list[dict[str, Any]],
                                  hard_ids: set[str]) -> dict[str, Any]:
    matches: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in matching:
        matches[str(row["hard_row_id"])].append(row)
    require(set(matches) == hard_ids, "direct matched-control hard identity set mismatch")
    require(all(len(rows) == 1 for rows in matches.values()), "each hard item must have exactly one matched control")
    original_controls: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in instances:
        if item["item_role"] == "control" and not item["is_repeat"]:
            original_controls[row_id(item)].append(item)
    require(all(len(rows) == 1 for rows in original_controls.values()), "control source must have one original review instance")
    for item in instances:
        if item["item_role"] != "hard":
            item["direct_matched_control"] = {}
            continue
        source_id = row_id(item)
        match = matches[source_id][0]
        control_id = str(match["control_row_id"])
        require(control_id != source_id and control_id not in hard_ids, f"invalid hard/control self-match: {source_id}")
        require(control_id in original_controls, f"matched control lacks public original instance: {control_id}")
        control = original_controls[control_id][0]
        require(as_bool(control.get("frame_correct"), "matched control frame_correct"),
                f"matched control native frame prediction is not correct: {control_id}")
        predicted = first(control, "frame_prediction", "baseline_frame_prediction")
        require(predicted != "" and as_int(predicted, "matched control frame_prediction") == int(control["frame_compatible_label"]),
                f"matched control prediction/label mismatch: {control_id}")
        require(str(control.get("match_link_id", "")) == str(match["match_link_id"]),
                f"hidden control match link mismatch: {control_id}")
        item["match_link_id"] = match["match_link_id"]
        item["match_level"] = match["match_level"]
        item["direct_matched_control"] = {
            "matched_control_review_instance_id": control["review_instance_id"],
            "matched_control_match_level": match["match_level"],
            "matched_control_claim": control["claim"],
            "matched_control_evidence": control["evidence"],
            "matched_control_native_frame_label": control["frame_compatible_label"],
            "matched_control_final_gold_label": control["final_label"],
            "matched_control_frame_prediction": predicted,
            "matched_control_frame_logit": first(control, "frame_logit", "baseline_frame_logit", "baseline_frame_score"),
            "matched_control_frame_prob": first(control, "frame_prob", "baseline_frame_prob"),
        }
    linked_sources = {row_id(item) for item in instances if item["item_role"] == "hard" and not item["is_repeat"]
                      and item.get("direct_matched_control")}
    require(linked_sources == hard_ids and len(linked_sources) == 39,
            "all 39 original hard items must have direct matched-control context")
    return {"hard_items_with_direct_matched_control": len(linked_sources),
            "hard_items_missing_direct_matched_control": len(hard_ids - linked_sources),
            "pass2_matched_control_context_included": True}


def build_rows(instances: list[dict[str, Any]], none_rows: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    blind, unblind, hidden = [], [], []
    for item in instances:
        rid = item["review_instance_id"]
        pair = str(item["pair_id"])
        anchor = none_rows[pair]
        blind.append({"review_instance_id": rid, "claim": item["claim"], "evidence": item["evidence"],
                      "claim_segment": item["claim"], "evidence_segment": item["evidence"], "reviewer_notes": ""})
        matched_context = item.get("direct_matched_control", {}) if item["item_role"] == "hard" else {}
        unblind.append({
            "review_instance_id": rid,
            "dataset_native_frame_label": item["frame_compatible_label"],
            "intervention_type": item["intervention_type"],
            "canonical_none_claim": anchor["claim"], "canonical_none_evidence": anchor["evidence"],
            "stage176_cohort": item.get("stage176_cohort", "none"),
            "native_frame_prediction": first(item, "frame_prediction", "baseline_frame_prediction"),
            "native_frame_logit": first(item, "frame_logit", "baseline_frame_logit", "baseline_frame_score"),
            "native_frame_probability": first(item, "frame_prob", "baseline_frame_prob"),
            "final_prediction": first(item, "baseline_predicted_label", "baseline_final_prediction"),
            "gold_final_label": item["final_label"],
            "frame_head_projection": first(item, "head_direction_projection", "stage179_head_direction_projection"),
            "representation_movement_from_none": first(item, "representation_displacement_from_none", "stage179_representation_displacement_from_none"),
            "stage179_centroid_prediction": first(item, "centroid_prediction", "stage179_centroid_prediction"),
            "stage179_centroid_correct": first(item, "centroid_correct", "stage179_centroid_correct"),
            **{column: matched_context.get(column, "") for column in MATCHED_CONTROL_COLUMNS},
        })
        hidden.append({
            "review_instance_id": rid, "stable_row_index": item["stable_row_index"], "row_id": row_id(item),
            "pair_id": pair, "item_role": item["item_role"], "stage176_cohort": item.get("stage176_cohort", "none"),
            "intervention_type": item["intervention_type"], "native_frame_label": item["frame_compatible_label"],
            "gold_final_label": item["final_label"], "match_link_id": item.get("match_link_id", ""),
            "matched_hard_row_id": item.get("matched_hard_row_id", ""), "match_level": item.get("match_level", ""),
            "is_repeat": item["is_repeat"], "repeat_group_id": item["repeat_group_id"],
        })
    return blind, unblind, hidden


def render_manifest(report: dict[str, Any]) -> str:
    return "\n".join([
        "# Stage180-A manual review packet", "", f"**Decision:** `{report['decision']}`", "",
        f"The packet contains {report['packet_counts']['semantic_items']} unique semantic items and "
        f"{report['packet_counts']['review_instances']} blinded review instances, including "
        f"{report['repeat_instances']['count']} hidden repeats.", "",
        "Pass 1 exposes text only. Pass 2 reveals native labels and diagnostics only after Pass 1 is frozen. "
        "The hidden key is not a reviewer artifact and its SHA-256 digest is recorded in this manifest.", "",
        "No model forward, fitting, training, relabeling, or dataset modification is performed.", "",
    ])


def instructions() -> str:
    return """# Stage180 manual review instructions

1. Read the codebook before annotation.
2. Complete and freeze `stage180a_pass1_annotation_template.csv` using only the blinded packet. Do not open the pass-2 packet or hidden key first.
3. Use one allowed judgment, integer confidence 1–5, a non-empty reviewer ID, and a non-empty rationale for every review instance.
4. After Pass 1 is frozen, open the unblinded packet and complete the Pass 2 template. Never alter Pass 1 in response to unblinded context.
5. Keep the hidden key unavailable to reviewers. It exists only to restore hard/control, matching, and repeat identity during analysis.
6. Recommendations are queues for human adjudication. They do not authorize automatic relabeling or data edits.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "stage176a-report", "stage176a-row-transitions", "stage177a-report",
                 "stage177e-report", "stage178a-report", "stage179a-report", "stage179a-dev-row-analysis",
                 "stage179a-hard39-attribution", "codebook", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--random-seed", type=int, default=180)
    parser.add_argument("--controls-per-hard-row", type=int, default=1)
    parser.add_argument("--repeat-instance-count", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    current = "argument_validation"
    dev_validation_diagnostic: dict[str, Any] = {
        "stage176a_dev_csv_row_count": None, "stage179a_dev_csv_row_count": None,
        "stage176a_dev_unique_row_count": None, "stage179a_dev_unique_row_count": None,
        "stage176a_stage179a_dev_identity_match": None,
        "stage176a_report_dev_count_present": False, "stage176a_report_dev_count": None,
        "dev_count_authoritative_source": "stage176a_row_transitions_csv",
    }
    try:
        require(args.controls_per_hard_row >= 1, "controls-per-hard-row must be positive")
        require(args.repeat_instance_count >= 0, "repeat-instance-count must be non-negative")
        paths = {key: value.resolve() for key, value in vars(args).items() if isinstance(value, Path)}
        reports = {name: read_json(paths[f"{name}_report"]) for name in EXPECTED}
        current = "upstream_decision_validation"
        actual = {name: decision(reports[name], name) for name in EXPECTED}
        for name, expected in EXPECTED.items():
            require(actual[name] == expected, f"{name} decision mismatch: {actual[name]!r}")
        current = "dev_csv_identity_validation"
        stage176_transition_rows = read_csv(paths["stage176a_row_transitions"])
        stage179_dev_analysis_rows = read_csv(paths["stage179a_dev_row_analysis"])
        validate_dev_artifacts(stage176_transition_rows, stage179_dev_analysis_rows,
                               reports["stage176a"], dev_validation_diagnostic)
        for name, report in reports.items():
            encoded = json.dumps(report, sort_keys=True).lower()
            require('"external_evaluation": true' not in encoded and '"external_labels": true' not in encoded,
                    f"{name} enables external evaluation/labels")
            require('"time_swap": true' not in encoded and '"time_swap_authorized": true' not in encoded,
                    f"{name} enables time_swap")
        current = "authoritative_definition_validation"
        authority = validate_authority(paths["codebook"])
        current = "dev_and_hard_identity_validation"
        dev = merge_dev(read_data(paths["data"]), stage179_dev_analysis_rows)
        hard, hard_check = validate_hard(stage176_transition_rows,
                                         read_csv(paths["stage179a_hard39_attribution"]), dev,
                                         dev_validation_diagnostic)
        current = "control_selection"
        controls, matching = select_controls(hard, dev, args.controls_per_hard_row)
        hard_items = [{**row, "item_role": "hard", "matched_hard_row_id": row_id(row),
                       "match_link_id": "", "match_level": "", "control_reused": False}
                      for row in hard.values()]
        semantic_items = hard_items + controls
        current = "repeat_and_blinding_construction"
        instances = arrange_instances(semantic_items, args.repeat_instance_count, args.random_seed)
        direct_control = attach_direct_control_context(instances, matching, set(hard))
        blind, unblind, hidden = build_rows(instances, canonical_none(dev))
        require(all(not ({"item_role", "native_frame_label", "gold_final_label", "pair_id", "intervention_type"} & set(row)) for row in blind),
                "blinded packet leaks protected metadata")
        pass2_ids = {row["review_instance_id"] for row in unblind}
        hidden_by_instance = {row["review_instance_id"]: row for row in hidden}
        for row in unblind:
            matched_id = row["matched_control_review_instance_id"]
            if matched_id:
                require(matched_id in pass2_ids, f"direct matched-control instance is absent from Pass 2: {matched_id}")
                matched_hidden = hidden_by_instance[matched_id]
                require(matched_hidden["item_role"] == "control" and not as_bool(matched_hidden["is_repeat"], "is_repeat"),
                        f"direct matched-control link must target an original control: {matched_id}")
                require(matched_hidden["match_link_id"] == hidden_by_instance[row["review_instance_id"]]["match_link_id"],
                        f"direct matched-control hidden mapping mismatch: {row['review_instance_id']}")
        output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(output_dir / OUTPUTS[2], blind, BLIND_COLUMNS)
        pass1 = [{**row, "reviewer_id": "", "independent_frame_judgment": "", "pass1_confidence": "", "pass1_rationale": ""} for row in blind]
        write_csv(output_dir / OUTPUTS[3], pass1, PASS1_TEMPLATE_COLUMNS)
        write_csv(output_dir / OUTPUTS[4], unblind, UNBLIND_COLUMNS)
        write_csv(output_dir / OUTPUTS[5], [{field: row["review_instance_id"] if field == "review_instance_id" else ""
                                            for field in PASS2_TEMPLATE_COLUMNS} for row in hidden], PASS2_TEMPLATE_COLUMNS)
        write_csv(output_dir / OUTPUTS[6], hidden, HIDDEN_COLUMNS)
        write_csv(output_dir / OUTPUTS[7], matching, MATCHING_COLUMNS)
        direct_hard_ids = {row_id(item) for item in instances if item["item_role"] == "hard"
                           and not item["is_repeat"] and item.get("direct_matched_control")}
        item_summary = [{"row_id": row_id(row), "item_role": row["item_role"],
                         "stage176_cohort": row.get("stage176_cohort", "none"),
                         "review_instance_count": sum(row_id(item) == row_id(row) for item in instances),
                         "match_link_id": next((item.get("match_link_id", "") for item in instances
                                                if row_id(item) == row_id(row) and not item["is_repeat"]), ""),
                         "has_direct_matched_control": row["item_role"] == "hard" and row_id(row) in direct_hard_ids,
                         **direct_control} for row in semantic_items]
        write_csv(output_dir / OUTPUTS[8], item_summary, ITEM_SUMMARY_COLUMNS)
        (output_dir / OUTPUTS[9]).write_text(instructions(), encoding="utf-8")
        hidden_digest = sha256(output_dir / OUTPUTS[6])
        report = {
            "stage": STAGE, "decision": READY,
            "scope": {"clean_controlled_data_only": True, "seed": args.random_seed, "dev_rows": 720,
                      "evaluation_and_data_packaging_only": True},
            "input_validation": {"status": "passed", "decisions": actual, "hard39": hard_check,
                                 "external_evaluation": False, "external_labels": False, "time_swap": False,
                                 "stable_row_identity": True, "hidden_item_key_sha256": hidden_digest,
                                 **dev_validation_diagnostic},
            "authoritative_frame_definition": authority,
            "hard_cohort": {"beneficial": 25, "harmful": 14, "total": 39},
            "control_selection": {"controls_per_hard_row": args.controls_per_hard_row, "selected": len(controls),
                                  "without_replacement_preferred": True,
                                  "reused": sum(as_bool(row["control_reused"], "control_reused") for row in matching),
                                  "native_frame_correct_only": True, "model_confidence_used": False,
                                  "hierarchy": [level for level, _ in [
                                      ("same_intervention_frame_final", None), ("same_intervention_frame", None),
                                      ("same_frame_final", None), ("same_frame", None), ("all_correct", None)]]},
            "repeat_instances": {"count": args.repeat_instance_count, "balanced_hard_control": True,
                                 "hidden_from_pass1": True, "semantic_item_count_unchanged": True},
            "blinding_contract": {"pass1_text_only": True, "hard_control_hidden": True, "native_labels_hidden": True,
                                  "model_fields_hidden": True, "pair_family_cohort_hidden": True,
                                  "pass1_frozen_before_pass2": True},
            "packet_counts": {"hard_items": 39, "control_items": len(controls), "semantic_items": len(semantic_items),
                              "review_instances": len(instances), "repeat_instances": args.repeat_instance_count,
                              **direct_control},
            "output_files": OUTPUTS, "output_schemas": CSV_SCHEMAS,
            "next_step": "Complete blinded Pass 1, freeze it, then complete Pass 2 and run Stage180-B.",
            "limitations": ["Manual judgments are observational taxonomy, not causal proof.",
                            "Matched controls are selected deterministically and are not randomized experimental controls."],
            "safety_policy": {"model_forward": False, "training": False, "optimizer": False, "backward": False,
                              "calibration": False, "threshold_fitting": False, "fitted_probe": False,
                              "external_evaluation": False, "external_labels": False, "time_swap": False,
                              "relabeling": False, "dataset_modification": False, "checkpoint_modification": False,
                              "automatic_adjudication": False},
        }
        write_json(output_dir / OUTPUTS[0], report)
        (output_dir / OUTPUTS[1]).write_text(render_manifest(report), encoding="utf-8")
        return 0
    except Exception as error:  # blocked artifact is part of the external contract
        output_dir.mkdir(parents=True, exist_ok=True)
        blocked = {"stage": STAGE, "decision": BLOCKED, "error_type": type(error).__name__, "error": str(error),
                   "failure_stage": current, "traceback": traceback.format_exc(),
                   **dev_validation_diagnostic}
        write_json(output_dir / OUTPUTS[0], blocked)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
