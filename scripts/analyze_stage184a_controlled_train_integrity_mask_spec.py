#!/usr/bin/env python3
"""Stage184-A deterministic controlled-train integrity sidecar feasibility audit.

The analyzer is read-only with respect to repository inputs.  It imports no
project module, model, or torch package and performs no model/checkpoint work.
It specifies a future Stage185 sidecar; it does not build row eligibility.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


STAGE = "Stage184-A"
READY_DECISION = "STAGE184A_DETERMINISTIC_FAIL_CLOSED_INTEGRITY_SIDECAR_SPEC_READY"
READY_NEXT = "STAGE185_CONTROLLED_TRAIN_INTEGRITY_SIDECAR_BUILDER"
BLOCKED_DECISION = "STAGE184A_CONTROLLED_TRAIN_INTEGRITY_MASK_SPEC_BLOCKED"
AUTHORITATIVE_RELATIVE = Path("data/controlled_v5_v3_without_time_swap.jsonl")
RULE_VERSION = "stage184a_v1"

BASE_FIELDS = {
    "id", "pair_id", "claim", "evidence", "final_label",
    "frame_compatible_label", "predicate_covered_label", "sufficiency_label",
    "polarity_label", "primary_failure_type", "intervention_type",
}
BINARY_FIELDS = {"frame_compatible_label", "predicate_covered_label", "sufficiency_label"}
FINAL_LABELS = {"SUPPORT", "NOT_ENTITLED", "REFUTE"}
POLARITY_LABELS = {"SUPPORT", "NONE", "REFUTE"}
PRIMARY_FAILURE_TYPES = {"none", "frame", "predicate", "sufficiency", "polarity"}

ALL_AXES = {"title", "name", "role", "predicate", "object", "time", "location", "polarity"}
FAMILY_CONTRACTS: dict[str, dict[str, Any]] = {
    "none": {
        "changed": [], "preserved": sorted(ALL_AXES),
        "labels": "canonical final/polarity; frame=1 predicate=1 sufficiency=1",
        "branch_token": 'fact, "none"', "evidence_relation": "evidence is canonical rendering",
    },
    "paraphrase": {
        "changed": ["realization"], "preserved": sorted(ALL_AXES),
        "labels": "same final/polarity as canonical; frame=1 predicate=1 sufficiency=1",
        "branch_token": 'fact, "paraphrase"', "evidence_relation": "deterministic paraphrase of canonical structured fact",
    },
    "entity_swap": {
        "changed": ["name"], "preserved": sorted(ALL_AXES - {"name"}),
        "labels": "NOT_ENTITLED; frame=0 predicate=0 sufficiency=1 polarity=NONE",
        "branch_token": 'fact, "entity_swap"', "evidence_relation": "alternate_name replaces canonical name",
    },
    "event_swap": {
        "changed": ["object"], "preserved": sorted(ALL_AXES - {"object"}),
        "labels": "NOT_ENTITLED; frame=0 predicate=0 sufficiency=1 polarity=NONE",
        "branch_token": 'fact, "event_swap"', "evidence_relation": "alternate_object replaces canonical object",
    },
    "time_swap": {
        "changed": ["time"], "preserved": sorted(ALL_AXES - {"time"}),
        "labels": "NOT_ENTITLED; frame=0 predicate=1 sufficiency=1 polarity=NONE",
        "branch_token": 'fact, "time_swap"', "evidence_relation": "alternate_time replaces canonical time",
    },
    "location_swap": {
        "changed": ["location"], "preserved": sorted(ALL_AXES - {"location"}),
        "labels": "NOT_ENTITLED; frame=0 predicate=1 sufficiency=1 polarity=NONE",
        "branch_token": 'fact, "location_swap"', "evidence_relation": "alternate_location replaces canonical location",
    },
    "role_swap": {
        "changed": ["role"], "preserved": sorted(ALL_AXES - {"role"}),
        "labels": "NOT_ENTITLED; frame=0 predicate=1 sufficiency=1 polarity=NONE",
        "branch_token": 'fact, "role_swap"', "evidence_relation": "alternate_role replaces canonical role",
    },
    "title_name_swap": {
        "changed": ["title", "name"], "preserved": sorted(ALL_AXES - {"title", "name"}),
        "labels": "NOT_ENTITLED; frame=0 predicate=0 sufficiency=1 polarity=NONE",
        "branch_token": 'fact, "title_name_swap"', "evidence_relation": "alternate_title and alternate_name replace canonical values",
    },
    "predicate_swap": {
        "changed": ["predicate"], "preserved": sorted(ALL_AXES - {"predicate"}),
        "labels": "NOT_ENTITLED; frame=1 predicate=0 sufficiency=1 polarity=NONE",
        "branch_token": 'fact, "predicate_swap"', "evidence_relation": "alternate_predicate replaces canonical predicate",
    },
    "evidence_deletion": {
        "changed": ["content_deletion"], "preserved": ["claim", "canonical_linkage", "polarity"],
        "labels": "NOT_ENTITLED; frame=1 predicate=1 sufficiency=0 polarity=NONE",
        "branch_token": 'fact, "evidence_deletion"', "evidence_relation": "fixed deletion message",
    },
    "evidence_truncation": {
        "changed": ["content_truncation"], "preserved": ["claim", "canonical_linkage", "polarity"],
        "labels": "NOT_ENTITLED; frame=1 predicate=1 sufficiency=0 polarity=NONE",
        "branch_token": 'fact, "evidence_truncation"', "evidence_relation": "fixed canonical title/name truncation",
    },
    "irrelevant_evidence": {
        "changed": ["content_replacement"], "preserved": ["claim", "canonical_linkage", "polarity"],
        "labels": "NOT_ENTITLED; frame=0 predicate=0 sufficiency=0 polarity=NONE",
        "branch_token": 'fact, "irrelevant_evidence"', "evidence_relation": "fixed unrelated evidence",
    },
    "polarity_flip": {
        "changed": ["polarity"], "preserved": sorted(ALL_AXES - {"polarity"}),
        "labels": "canonical SUPPORT/REFUTE flipped; frame=1 predicate=1 sufficiency=1",
        "branch_token": 'fact, "polarity_flip"', "evidence_relation": "deterministic negation inversion",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=174)
    parser.add_argument("--dev-ratio", type=float, default=0.2)
    parser.add_argument("--minimum-safe-positive-rate", type=float, default=0.0)
    args = parser.parse_args()
    if not 0.0 < args.dev_ratio < 1.0:
        parser.error("--dev-ratio must satisfy 0 < value < 1")
    if not 0.0 <= args.minimum_safe_positive_rate <= 1.0:
        parser.error("--minimum-safe-positive-rate must satisfy 0 <= value <= 1")
    return args


def resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if isinstance(value, bool):
        return str(value).lower()
    return "" if value is None else value


def write_csv(path: Path, headers: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({header: csv_value(row.get(header)) for header in headers})


def line_evidence(path: Path, token: str) -> str:
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if token in line:
            return f"{path.as_posix()}:{number}: {line.strip()}"
    return f"{path.as_posix()}: token not found: {token}"


def split_rows(rows: list[dict[str, Any]], seed: int, dev_ratio: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    pair_ids = sorted({str(row.get("pair_id")) for row in rows})
    random.Random(seed).shuffle(pair_ids)
    dev_count = min(len(pair_ids) - 1, max(1, round(len(pair_ids) * dev_ratio)))
    dev_ids = set(pair_ids[:dev_count])
    train = [row for row in rows if str(row.get("pair_id")) not in dev_ids]
    dev = [row for row in rows if str(row.get("pair_id")) in dev_ids]
    return train, dev, dev_ids


def base_schema_errors(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = sorted(BASE_FIELDS - set(row))
    errors.extend(f"missing:{field}" for field in missing)
    for field in ("id", "pair_id", "claim", "evidence", "final_label", "polarity_label", "primary_failure_type", "intervention_type"):
        if field in row and (not isinstance(row[field], str) or not row[field].strip()):
            errors.append(f"invalid_string:{field}")
    for field in BINARY_FIELDS:
        if field in row and (isinstance(row[field], bool) or not isinstance(row[field], int) or row[field] not in (0, 1)):
            errors.append(f"invalid_binary:{field}")
    if row.get("final_label") not in FINAL_LABELS:
        errors.append("invalid_enum:final_label")
    if row.get("polarity_label") not in POLARITY_LABELS:
        errors.append("invalid_enum:polarity_label")
    if row.get("primary_failure_type") not in PRIMARY_FAILURE_TYPES:
        errors.append("invalid_enum:primary_failure_type")
    return errors


def identity_audit(
    rows: list[dict[str, Any]], train: list[dict[str, Any]], dev: list[dict[str, Any]],
    data_path: Path, authoritative_path: Path, expected: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], bool]:
    pair_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pair_groups[str(row.get("pair_id"))].append(row)
    row_ids = [str(row.get("id")) for row in rows]
    semantic_keys = [(str(row.get("pair_id")), str(row.get("intervention_type"))) for row in rows]
    row_id_duplicates = sorted(key for key, count in Counter(row_ids).items() if count > 1)
    semantic_duplicates = sorted(key for key, count in Counter(semantic_keys).items() if count > 1)
    rows_per_pair = Counter(len(group) for group in pair_groups.values())
    canonical_per_pair = Counter(
        sum(row.get("intervention_type") == "none" for row in group)
        for group in pair_groups.values()
    )
    families = sorted({str(row.get("intervention_type")) for row in rows})
    missing_base = sum(bool(BASE_FIELDS - set(row)) for row in rows)
    schema_error_rows = sum(bool(base_schema_errors(row)) for row in rows)
    train_labels = Counter(row.get("frame_compatible_label") for row in train)
    dev_labels = Counter(row.get("frame_compatible_label") for row in dev)
    actual = {
        "dataset_path": str(data_path),
        "dataset_sha256": digest(data_path),
        "row_count": len(rows),
        "pair_count": len(pair_groups),
        "intervention_family_count": len(families),
        "intervention_families": families,
        "unique_row_id_count": len(set(row_ids)),
        "row_id_duplicate_count": len(row_id_duplicates),
        "row_id_duplicates": row_id_duplicates,
        "unique_pair_id_count": len(pair_groups),
        "rows_per_pair_distribution": dict(sorted(rows_per_pair.items())),
        "canonical_none_count_per_pair_distribution": dict(sorted(canonical_per_pair.items())),
        "train_pair_count": len({row.get("pair_id") for row in train}),
        "dev_pair_count": len({row.get("pair_id") for row in dev}),
        "train_row_count": len(train),
        "dev_row_count": len(dev),
        "train_frame_compatible": train_labels.get(1, 0),
        "train_frame_incompatible": train_labels.get(0, 0),
        "dev_frame_compatible": dev_labels.get(1, 0),
        "dev_frame_incompatible": dev_labels.get(0, 0),
        "time_swap_count": sum(row.get("intervention_type") == "time_swap" for row in rows),
        "rows_missing_required_base_fields": missing_base,
        "rows_with_base_schema_errors": schema_error_rows,
        "duplicate_semantic_identity_key_count": len(semantic_duplicates),
        "duplicate_semantic_identity_keys": [list(key) for key in semantic_duplicates],
        "authoritative_path_exact": data_path == authoritative_path,
    }
    expected_map = {
        "row_count": expected.get("all_rows"), "pair_count": expected.get("all_pairs"),
        "train_row_count": expected.get("train_rows"), "train_pair_count": expected.get("train_pairs"),
        "dev_row_count": expected.get("dev_rows"), "dev_pair_count": expected.get("dev_pairs"),
        "train_frame_compatible": expected.get("train_frame_compatible"),
        "train_frame_incompatible": expected.get("train_frame_incompatible"),
    }
    audit_rows: list[dict[str, Any]] = []
    for metric, value in actual.items():
        if metric in expected_map:
            expected_value = expected_map[metric]
            passed = value == expected_value
            source = "Stage183-A closure frozen_topology"
        elif metric == "authoritative_path_exact":
            expected_value, passed, source = True, bool(value), "exact resolved path"
        elif metric in {"row_id_duplicate_count", "time_swap_count", "rows_missing_required_base_fields", "rows_with_base_schema_errors", "duplicate_semantic_identity_key_count"}:
            expected_value, passed, source = 0, value == 0, "Stage184 fail-closed identity contract"
        elif metric == "canonical_none_count_per_pair_distribution":
            expected_value, passed, source = {1: len(pair_groups)}, value == {1: len(pair_groups)}, "exactly one none row per pair"
        else:
            expected_value, passed, source = "observed", True, "computed from authoritative JSONL"
        audit_rows.append({"metric": metric, "observed": value, "expected": expected_value, "passed": passed, "evidence_source": source})
    blocking_metrics = {
        "row_count", "pair_count", "train_row_count", "train_pair_count", "dev_row_count", "dev_pair_count",
        "train_frame_compatible", "train_frame_incompatible", "authoritative_path_exact", "row_id_duplicate_count",
        "time_swap_count", "rows_missing_required_base_fields", "rows_with_base_schema_errors",
        "duplicate_semantic_identity_key_count", "canonical_none_count_per_pair_distribution",
    }
    passed = all(row["passed"] for row in audit_rows if row["metric"] in blocking_metrics)
    return audit_rows, actual, passed


def criterion_rows(generator: Path, stage182: Path, policy182: Path) -> list[dict[str, Any]]:
    generator_text = generator.read_text(encoding="utf-8")
    stage182_text = stage182.read_text(encoding="utf-8")
    evidence = {
        "facts": line_evidence(generator, "_FACT_FIELDS ="),
        "record": line_evidence(generator, "def _record("),
        "branches": line_evidence(generator, "def _build_records("),
        "schema": line_evidence(generator, "def validate_record("),
        "axes": line_evidence(stage182, "INTENDED_AXES:"),
        "changed": line_evidence(stage182, "def changed_axes("),
        "grammar": line_evidence(stage182, "def grammar_anomaly("),
        "polarity": line_evidence(stage182, "polarity_leak ="),
        "canonical": line_evidence(stage182, 'none_id = f"{pair_id}__none"'),
        "policy": line_evidence(policy182, "Generator equality is provenance"),
    }
    capabilities = {
        "fact_args": "_FACT_FIELDS" in generator_text and "alternate_predicate" in generator_text,
        "branches": "def _build_records" in generator_text,
        "schema": "def validate_record" in generator_text,
        "axis_rules": "INTENDED_AXES" in stage182_text and "def changed_axes" in stage182_text,
        "grammar_rule": "DID_NOT_INFLECTED_PREDICATE" in stage182_text,
        "polarity_rule": "NON_POLARITY_INTERVENTION_POLARITY_CHANGE" in stage182_text,
        "canonical_rule": 'none_id = f"{pair_id}__none"' in stage182_text,
    }
    specs = [
        ("grammar_valid", "generator structured arguments + deterministic template/morphology contract", "known-template PASS/FAIL; otherwise UNRESOLVED", ["predicate", "alternate_predicate", "rendered evidence"], capabilities["fact_args"] and capabilities["grammar_rule"], True, True, False, False, False, True, True, True, evidence["grammar"], "general English grammaticality is never inferred"),
        ("intervention_contract_exact", "generator branch + structured arguments + Stage182 intended-axis contract", "exact structured-axis delta relative to canonical none", ["family branch", "original/alternate axes", "canonical row", "labels"], capabilities["branches"] and capabilities["axis_rules"], True, True, False, False, False, True, True, True, evidence["axes"] + " | " + evidence["changed"], "family name and generator equality alone cannot pass"),
        ("polarity_contamination_absent", "structured polarity labels + deterministic rendered polarity + canonical row", "declared polarity delta and realization consistency", ["intervention_type", "polarity_label", "canonical polarity", "rendered evidence"], capabilities["polarity_rule"] and capabilities["fact_args"], True, True, False, False, False, True, True, True, evidence["polarity"], "no sentiment model or heuristic lexicon"),
        ("schema_resolved", "generator REQUIRED_FIELDS/enums/types + sidecar linkage schema", "static type/nullability/enum/identity validation", sorted(BASE_FIELDS), capabilities["schema"], True, True, False, False, False, True, True, True, evidence["schema"], "structural validity only"),
        ("canonical_row_valid", "same-pair none row + reconstructed canonical fact/linkage", "unique structural canonical validation", ["pair_id", "none row", "canonical labels", "claim/evidence linkage"], capabilities["canonical_rule"] and capabilities["fact_args"], True, True, False, False, False, True, True, True, evidence["canonical"], "does not assert real-world truth"),
        ("time_swap_absent", "authoritative JSONL intervention_type", "exact count/equality", ["intervention_type"], True, True, True, False, False, False, True, False, True, "exact row field", "time_swap rows are excluded"),
        ("authoritative_main_dataset", "resolved path + dataset SHA-256", "exact path/hash contract", ["source path", "SHA-256"], True, True, True, False, False, False, True, False, True, "dataset identity audit", "same content at an unapproved source cannot silently replace canonical provenance"),
        ("not_external_or_stage34_35", "authoritative path/hash allowlist", "exact source allowlist", ["source path", "SHA-256"], True, True, True, False, False, False, True, False, True, "authoritative dataset identity", "no family/text inference"),
        ("required_base_schema_present", "generator REQUIRED_FIELDS and type/enum contract", "static schema validation", sorted(BASE_FIELDS), capabilities["schema"], True, True, False, False, False, True, False, True, evidence["schema"], "missing fields fail closed"),
        ("frame_label_compatible", "row frame_compatible_label", "exact binary value", ["frame_compatible_label", "split"], True, True, True, False, False, False, True, False, True, "authoritative row label", "loss scope, not semantic cleanliness"),
    ]
    headers = ["criterion", "required", "authoritative_source", "derivation_mode", "structured_inputs", "deterministic", "exhaustive", "text_heuristic_required", "model_required", "manual_annotation_required", "can_fail_closed", "unresolved_possible", "implementation_ready", "evidence", "limitation"]
    rows = [
        dict(
            zip(
                headers,
                (
                    name,
                    required,
                    source,
                    mode,
                    inputs,
                    deterministic,
                    exhaustive,
                    heuristic,
                    model,
                    manual,
                    fail_closed,
                    unresolved,
                    ready,
                    ev,
                    limitation,
                ),
            )
        )
        for (
            name,
            source,
            mode,
            inputs,
            required,
            deterministic,
            exhaustive,
            heuristic,
            model,
            manual,
            fail_closed,
            unresolved,
            ready,
            ev,
            limitation,
        ) in specs
    ]

    if len(headers) != 15:
        raise ValueError(
            f"criterion_rows expected 15 headers, got {len(headers)}"
        )

    if len(rows) != 10:
        raise ValueError(
            f"criterion_rows expected 10 criteria, got {len(rows)}"
        )

    expected_keys = set(headers)
    criteria = [row[headers[0]] for row in rows]

    if len(set(criteria)) != len(criteria):
        raise ValueError(
            f"Duplicate integrity criteria: {criteria}"
        )

    boolean_indexes = (1, 5, 6, 7, 8, 9, 10, 11, 12)
    required_text_indexes = (0, 2, 3, 13, 14)

    for row in rows:
        if set(row) != expected_keys:
            raise ValueError(
                "Criterion row key mismatch: "
                f"expected={sorted(expected_keys)}, "
                f"actual={sorted(row)}"
            )

        for field_index in boolean_indexes:
            field = headers[field_index]
            if not isinstance(row[field], bool):
                raise ValueError(
                    f"Criterion {row[headers[0]]} field "
                    f"{field} must be bool, got "
                    f"{type(row[field]).__name__}"
                )

        for field_index in required_text_indexes:
            field = headers[field_index]
            if not str(row[field]).strip():
                raise ValueError(
                    f"Criterion {row[headers[0]]} has empty "
                    f"required field {field}"
                )

    return rows


def family_rows(rows: list[dict[str, Any]], generator: Path) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pair_none = {str(row.get("pair_id")) for row in rows if row.get("intervention_type") == "none"}
    for row in rows:
        groups[str(row.get("intervention_type"))].append(row)
    result: list[dict[str, Any]] = []
    for family in sorted(groups):
        members = groups[family]
        contract = FAMILY_CONTRACTS.get(family)
        pairs = {str(row.get("pair_id")) for row in members}
        branch_found = bool(contract and contract["branch_token"] in generator.read_text(encoding="utf-8"))
        structured = bool(contract and branch_found)
        result.append({
            "family": family,
            "row_count": len(members),
            "pair_coverage": len(pairs),
            "canonical_counterpart_available": len(pairs & pair_none) == len(pairs),
            "intended_changed_axes": contract["changed"] if contract else [],
            "intended_preserved_axes": contract["preserved"] if contract else [],
            "expected_label_transition": contract["labels"] if contract else "UNRESOLVED",
            "canonical_counterpart_requirement": "exactly one same-pair none row",
            "evidence_text_relation_requirement": contract["evidence_relation"] if contract else "UNRESOLVED",
            "reason_codes": ["INTERVENTION_CONTRACT_FAIL", "INTERVENTION_CONTRACT_UNRESOLVED"],
            "generator_branch_location": line_evidence(generator, contract["branch_token"]) if contract else "not found",
            "structured_provenance_available": structured,
            "deterministic_contract_check_possible": structured,
            "polarity_check_possible": structured,
            "grammar_check_possible": structured,
            "unresolved_risk": "general grammar and any missing exact generator provenance remain UNRESOLVED",
            "sidecar_implementation_readiness": "ready_fail_closed" if structured else "unresolved_only",
        })
    return result


def pair_invariants() -> list[dict[str, Any]]:
    return [
        {"invariant": "unique_pair_id", "level": "dataset", "validation": "stable nonempty pair_id grouping", "failure_scope": "dataset", "failure_status": "BLOCKED", "reason_code": "PAIR_TOPOLOGY_INVALID", "stage185_action": "stop artifact"},
        {"invariant": "unique_row_id", "level": "dataset", "validation": "id unique across all rows", "failure_scope": "dataset", "failure_status": "BLOCKED", "reason_code": "DUPLICATE_ROW_ID", "stage185_action": "stop; one-to-one join impossible"},
        {"invariant": "exactly_one_canonical_none", "level": "pair", "validation": "one intervention_type=none row", "failure_scope": "pair", "failure_status": "UNRESOLVED", "reason_code": "CANONICAL_ROW_MISSING|CANONICAL_ROW_DUPLICATE", "stage185_action": "mark entire pair unresolved"},
        {"invariant": "expected_family_coverage", "level": "pair", "validation": "same observed family set for every pair", "failure_scope": "pair", "failure_status": "UNRESOLVED", "reason_code": "PAIR_TOPOLOGY_INVALID", "stage185_action": "mark affected pair unresolved"},
        {"invariant": "canonical_claim_linkage", "level": "pair", "validation": "every intervention claim equals same-pair canonical claim", "failure_scope": "pair", "failure_status": "UNRESOLVED", "reason_code": "CANONICAL_LINKAGE_INVALID", "stage185_action": "mark entire pair unresolved"},
        {"invariant": "canonical_evidence_linkage", "level": "pair", "validation": "none is canonical evidence; non-none relation follows family contract", "failure_scope": "row_or_pair", "failure_status": "INELIGIBLE_OR_UNRESOLVED", "reason_code": "CANONICAL_LINKAGE_INVALID", "stage185_action": "pair unresolved if anchor ambiguous; otherwise row ineligible"},
        {"invariant": "allowed_axis_delta_topology", "level": "row", "validation": "observed structured delta equals family contract", "failure_scope": "row", "failure_status": "INELIGIBLE", "reason_code": "INTERVENTION_CONTRACT_FAIL", "stage185_action": "exclude row"},
        {"invariant": "label_topology_consistency", "level": "row", "validation": "labels equal branch contract", "failure_scope": "row", "failure_status": "INELIGIBLE", "reason_code": "INVALID_ENUM|INTERVENTION_CONTRACT_FAIL", "stage185_action": "exclude row"},
        {"invariant": "no_unintended_polarity_delta", "level": "row", "validation": "only polarity_flip may change declared polarity", "failure_scope": "row", "failure_status": "INELIGIBLE", "reason_code": "NON_POLARITY_INTERVENTION_POLARITY_CHANGE", "stage185_action": "exclude row"},
        {"invariant": "no_duplicate_intervention_row", "level": "pair", "validation": "one row per pair_id+intervention_type", "failure_scope": "pair", "failure_status": "BLOCKED", "reason_code": "PAIR_TOPOLOGY_INVALID", "stage185_action": "stop if identity ambiguous"},
        {"invariant": "no_cross_pair_canonical_leakage", "level": "pair", "validation": "canonical_row_id belongs to same pair", "failure_scope": "pair", "failure_status": "UNRESOLVED", "reason_code": "CROSS_PAIR_CANONICAL_LEAKAGE", "stage185_action": "mark entire pair unresolved"},
        {"invariant": "no_time_swap", "level": "dataset", "validation": "time_swap count is zero", "failure_scope": "dataset", "failure_status": "BLOCKED", "reason_code": "TIME_SWAP_EXCLUDED", "stage185_action": "block authoritative-main sidecar build"},
        {"invariant": "deterministic_split_assignment", "level": "pair", "validation": "sorted pair IDs + seed 174 + ratio 0.2", "failure_scope": "dataset", "failure_status": "BLOCKED", "reason_code": "SPLIT_ASSIGNMENT_MISMATCH", "stage185_action": "stop artifact"},
    ]


def sidecar_schema() -> list[dict[str, Any]]:
    fields = [
        ("row_id", "string", False, "authoritative JSONL id", "exact one-to-one join key"),
        ("pair_id", "string", False, "authoritative JSONL pair_id", "frozen pair identity"),
        ("split", "enum(train,dev)", False, "deterministic frozen pair split", "not independently assigned"),
        ("intervention_type", "string enum", False, "authoritative row", "must match source"),
        ("frame_compatible_label", "integer enum(0,1)", False, "authoritative row", "loss-scope label"),
        ("grammar_status", "enum(PASS,FAIL,UNRESOLVED,NOT_APPLICABLE)", False, "integrity rules", "criterion status"),
        ("intervention_contract_status", "enum(PASS,FAIL,UNRESOLVED,NOT_APPLICABLE)", False, "axis contract", "criterion status"),
        ("polarity_contamination_status", "enum(PASS,FAIL,UNRESOLVED,NOT_APPLICABLE)", False, "structured polarity contract", "criterion status"),
        ("schema_status", "enum(PASS,FAIL,UNRESOLVED,NOT_APPLICABLE)", False, "base/sidecar schema", "criterion status"),
        ("canonical_status", "enum(PASS,FAIL,UNRESOLVED,NOT_APPLICABLE)", False, "same-pair canonical validation", "criterion status"),
        ("time_swap_status", "enum(PASS,FAIL,UNRESOLVED,NOT_APPLICABLE)", False, "intervention_type", "PASS means absent/not time_swap"),
        ("dataset_source_status", "enum(PASS,FAIL,UNRESOLVED,NOT_APPLICABLE)", False, "path/hash allowlist", "criterion status"),
        ("integrity_status", "enum(ELIGIBLE,INELIGIBLE,UNRESOLVED)", False, "fail-closed composition", "exactly one"),
        ("eligible_for_positive_margin", "boolean", False, "exact eligibility expression", "true only for eligible train compatible rows"),
        ("reason_codes", "sorted unique string list", False, "rule outputs", "JSON array; CSV semicolon-delimited"),
        ("canonical_row_id", "string", False, "same-pair none identity", "must exist in source"),
        ("family_contract_id", "string", False, "rule catalog", "stable family contract version"),
        ("rule_version", "string", False, RULE_VERSION, "fixed contract version"),
        ("source_dataset_path", "string", False, "canonical resolved path", "provenance"),
        ("source_dataset_sha256", "sha256", False, "source bytes", "mismatch blocks all"),
        ("generator_source_sha256", "sha256", False, "generator bytes", "mismatch blocks build/use"),
        ("integrity_builder_sha256", "sha256", False, "builder bytes", "provenance"),
        ("created_at", "UTC ISO-8601", False, "builder clock", "excluded from semantic hash"),
    ]
    return [{"field": f, "type": t, "nullable": n, "source": s, "contract": c} for f, t, n, s, c in fields]


def reason_codes() -> list[dict[str, Any]]:
    items = [
        ("ELIGIBLE_CLEAN_COMPATIBLE", "row", "ELIGIBLE", "composed criterion statuses + train/frame label", "positive-margin eligible", "include only after every gate passes"),
        ("FRAME_LABEL_NOT_COMPATIBLE", "row", "INELIGIBLE", "frame_compatible_label", "frame label is 0", "exclude row"),
        ("DEV_SPLIT_EXCLUDED", "row", "INELIGIBLE", "frozen split", "row is dev", "exclude row"),
        ("TIME_SWAP_EXCLUDED", "row/dataset", "INELIGIBLE_OR_BLOCKED", "intervention_type", "time-swap is out of scope", "exclude; block authoritative dataset if present"),
        ("EXTERNAL_DATA_EXCLUDED", "dataset", "BLOCKED", "path/hash allowlist", "source is external or Stage34/35", "reject artifact"),
        ("MISSING_REQUIRED_BASE_FIELD", "row", "UNRESOLVED", "base schema", "required source field missing", "exclude row; block if identity field"),
        ("INVALID_FIELD_TYPE", "row", "INELIGIBLE", "base schema", "field has wrong type/nullability", "exclude row"),
        ("INVALID_ENUM", "row", "INELIGIBLE", "base schema enums", "label/type enum invalid", "exclude row"),
        ("DUPLICATE_ROW_ID", "dataset", "BLOCKED", "row identity counter", "one-to-one join impossible", "stop artifact"),
        ("PAIR_TOPOLOGY_INVALID", "pair", "UNRESOLVED_OR_BLOCKED", "pair/family matrix", "pair shape is invalid", "pair unresolved; block if identity ambiguous"),
        ("CANONICAL_ROW_MISSING", "pair", "UNRESOLVED", "same-pair none lookup", "canonical row absent", "entire pair unresolved"),
        ("CANONICAL_ROW_DUPLICATE", "pair", "UNRESOLVED", "same-pair none lookup", "canonical row ambiguous", "entire pair unresolved"),
        ("CANONICAL_LINKAGE_INVALID", "pair", "UNRESOLVED", "claim/evidence linkage", "canonical linkage failed", "entire pair unresolved"),
        ("CROSS_PAIR_CANONICAL_LEAKAGE", "pair", "UNRESOLVED", "pair_id/canonical_row_id", "canonical belongs to another pair", "entire pair unresolved"),
        ("INTERVENTION_CONTRACT_FAIL", "row", "INELIGIBLE", "exact structured-axis delta", "declared branch contract failed", "exclude row"),
        ("INTERVENTION_CONTRACT_UNRESOLVED", "row", "UNRESOLVED", "missing structured evidence", "contract cannot be decided", "exclude row"),
        ("NON_POLARITY_INTERVENTION_POLARITY_CHANGE", "row", "INELIGIBLE", "canonical/rendered structured polarity delta", "undeclared polarity change", "exclude row"),
        ("POLARITY_REALIZATION_MISMATCH", "row", "INELIGIBLE", "structured label vs template realization", "polarity label and realization disagree", "exclude row"),
        ("DID_NOT_INFLECTED_PREDICATE", "row", "INELIGIBLE", "generator predicate morphology rule", "did not precedes inflected predicate", "exclude row"),
        ("GRAMMAR_TEMPLATE_FAIL", "row", "INELIGIBLE", "deterministic template grammar", "known template grammar defect", "exclude row"),
        ("GRAMMAR_GENERAL_UNRESOLVED", "row", "UNRESOLVED", "absence of authorized general grammar source", "general grammar not decided", "exclude row"),
        ("GENERATOR_PROVENANCE_MISSING", "dataset/row", "UNRESOLVED_OR_BLOCKED", "generator path/hash/reconstruction", "structured provenance unavailable", "block reconstruction or mark affected rows unresolved"),
        ("DATASET_SHA_MISMATCH", "dataset", "BLOCKED", "SHA-256", "sidecar/source mismatch", "reject entire sidecar"),
        ("GENERATOR_SHA_MISMATCH", "dataset", "BLOCKED", "SHA-256", "sidecar/generator mismatch", "reject build/use"),
        ("SIDECAR_ROW_MISSING", "dataset", "BLOCKED", "row-id set equality", "source row has no sidecar row", "reject entire sidecar"),
        ("SIDECAR_ROW_EXTRA", "dataset", "BLOCKED", "row-id set equality", "sidecar has unknown row", "reject entire sidecar"),
        ("SPLIT_ASSIGNMENT_MISMATCH", "dataset", "BLOCKED", "frozen split recomputation", "split differs from seed/ratio contract", "reject artifact"),
    ]
    return [{"reason_code": c, "level": l, "status_effect": s, "criterion": d, "deterministic_source": src, "fail_closed_behavior": f, "human_interpretation": h} for c, l, s, src, h, f in items for d in [criterion_for_code(c)]]


def criterion_for_code(code: str) -> str:
    if "GRAMMAR" in code or code == "DID_NOT_INFLECTED_PREDICATE": return "grammar_valid"
    if "POLARITY" in code: return "polarity_contamination_absent"
    if "CANONICAL" in code: return "canonical_row_valid"
    if "CONTRACT" in code: return "intervention_contract_exact"
    if "SCHEMA" in code or "FIELD" in code or "ENUM" in code: return "schema_resolved"
    if "SHA" in code or "EXTERNAL" in code or "SIDECAR" in code: return "authoritative_main_dataset"
    if "TIME_SWAP" in code: return "time_swap_absent"
    if "FRAME_LABEL" in code: return "frame_label_compatible"
    return "integrity_composition"


def fail_closed_rows() -> list[dict[str, Any]]:
    return [
        {"condition": "all required criteria PASS; train; compatible; authoritative; no time_swap", "scope": "row", "result": "ELIGIBLE", "positive_margin_eligible": True, "propagation": "row only", "rationale": "exact conjunction"},
        {"condition": "any deterministic criterion FAIL", "scope": "row", "result": "INELIGIBLE", "positive_margin_eligible": False, "propagation": "row unless canonical/pair identity affected", "rationale": "known failure"},
        {"condition": "any required criterion UNRESOLVED and none FAIL", "scope": "row", "result": "UNRESOLVED", "positive_margin_eligible": False, "propagation": "row or pair per invariant", "rationale": "missing evidence never passes"},
        {"condition": "canonical missing/duplicate/linkage invalid", "scope": "pair", "result": "UNRESOLVED", "positive_margin_eligible": False, "propagation": "entire pair", "rationale": "all intervention deltas depend on anchor"},
        {"condition": "duplicate row id or dataset/sidecar SHA mismatch", "scope": "dataset", "result": "BLOCKED", "positive_margin_eligible": False, "propagation": "entire artifact", "rationale": "one-to-one provenance impossible"},
        {"condition": "dev or frame-incompatible row", "scope": "row", "result": "INELIGIBLE_FOR_POSITIVE_MARGIN", "positive_margin_eligible": False, "propagation": "row", "rationale": "loss scope exclusion independent of cleanliness"},
        {"condition": "external/Stage34/35/time_swap source", "scope": "row_or_dataset", "result": "INELIGIBLE_OR_BLOCKED", "positive_margin_eligible": False, "propagation": "exclude or reject artifact", "rationale": "explicit safety boundary"},
    ]


def markdown(report: dict[str, Any]) -> str:
    identity = report["dataset_identity"]
    coverage = report["coverage_feasibility"]
    return f"""# Stage184-A controlled-train integrity mask specification report

## Decision

`{report['decision']}`

Authorized next: `{report['stage185_gate']['authorized_next_stage']}`. Stage185 is limited to deterministic sidecar generation and static audit. Loss implementation and training remain unauthorized.

## Stage183-A closure

Stage183-A required an integrity mask before the contingent compatible-positive absolute-margin hinge. Train frame labels are balanced at 1,440/1,440, so positive reweighting has no imbalance basis. The hinge remains distinct from Stage175's final SUPPORT anchor and Stage177's pair ordering, but its margin and nonzero weight remain unset.

## Authoritative dataset identity

- Path: `{identity['dataset_path']}`
- SHA-256: `{identity['dataset_sha256']}`
- Rows/pairs: {identity['row_count']} / {identity['pair_count']}
- Families: {identity['intervention_family_count']}
- Train/dev rows: {identity['train_row_count']} / {identity['dev_row_count']}
- Train compatible/incompatible: {identity['train_frame_compatible']} / {identity['train_frame_incompatible']}
- Time-swap rows: {identity['time_swap_count']}

Topology is cross-checked against the Stage183-A closure. Any mismatch blocks the audit.

## Complete mask

Complete means every source row receives exactly one of `ELIGIBLE`, `INELIGIBLE`, or `UNRESOLVED`; it does not mean every row is clean. `UNRESOLVED` is fail-closed and never enters the positive-margin loss. Dev, frame-incompatible, time-swap, external, and Stage34/35 rows are excluded independently.

## Derivability and generator cleanliness boundary

The generator retains structured original/alternate fact arguments and exact intervention branches. Stage182-A supplies deterministic same-pair canonical, structured-axis, polarity-leak, and known morphology rules. This is enough to build a reproducible fail-closed sidecar without rewriting the JSONL. General grammar or missing provenance becomes `UNRESOLVED`; no model, LLM, annotation, text classifier, or family-name shortcut is allowed.

Exact generator equality proves provenance only. A buggy generator can exactly reproduce `did not` plus an inflected predicate or unintended non-polarity polarity changes, so grammar, contract, and polarity remain independent gates.

## Family and pair contracts

All families are enumerated from the actual JSONL, then joined to generator branch evidence. Each contract specifies changed/preserved axes, expected labels, canonical counterpart, evidence relation, and fail-closed reason codes. Unknown families remain unresolved.

Missing/duplicate canonical rows or invalid canonical linkage make the whole pair unresolved. A single deterministic grammar, polarity, schema, or intervention-contract failure makes the affected row ineligible. Duplicate row identity, SHA mismatch, or an impossible one-to-one join blocks the artifact.

## Sidecar and join contract

The sidecar is one-to-one on `row_id`, uses enum criterion statuses, records sorted stable reason codes, canonical/family/rule identity, source/generator/builder hashes, and the frozen pair split. Missing/extra/duplicate sidecar rows or dataset SHA mismatch block use. Generator SHA mismatch also blocks rebuilding or use under a different provenance contract.

## Coverage feasibility

- Decision coverage: `{coverage['decision_coverage']}`
- Positive eligible coverage: `{coverage['positive_eligible_coverage']}`
- Expected exact eligible count: `{coverage['expected_exact_eligible_count']}`
- Scientific usability: `{coverage['scientific_usability']}`

Stage184-A does not estimate semantic cleanliness or fit a coverage threshold. Stage185 must compute exact counts by split, frame label, family, and reason code; family concentration or very low coverage remains a separate scientific risk.

## Safety

Static specification/feasibility audit only. No dataset/JSONL/generator modification, model or Torch import, checkpoint load, forward, loss implementation, training, smoke, annotation, LLM labeling, text classifier, learned parser/probe, threshold fitting, calibration, external evaluation, time-swap use, multi-seed run, or hyperparameter sweep is authorized.
"""


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    data_path = resolve(root, args.data)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    authoritative = (root / AUTHORITATIVE_RELATIVE).resolve()
    generator = root / "scripts/build_controlled_v5.py"
    stage182 = root / "scripts/analyze_stage182a_controlled_intervention_integrity.py"
    policy182 = root / "reports/stage182a_controlled_intervention_integrity_policy.md"
    closure183 = root / "reports/stage183a_controlled_train_integrity_mask_required_closure.json"
    required = [data_path, generator, stage182, policy182, closure183,
                root / "reports/stage182a_data_contamination_clean_failure_set_closure.json",
                root / "reports/stage182b_compatible_positive_margin_collapse_closure.json"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing Stage184-A inputs: " + ", ".join(missing))

    closure = read_json(closure183)
    expected = closure.get("frozen_topology") or {}
    rows = read_jsonl(data_path)
    train, dev, _ = split_rows(rows, args.split_seed, args.dev_ratio)
    identity_csv, identity, topology_passed = identity_audit(rows, train, dev, data_path, authoritative, expected)
    criteria = criterion_rows(generator, stage182, policy182)
    families = family_rows(rows, generator)
    invariants = pair_invariants()
    schema = sidecar_schema()
    codes = reason_codes()
    fail_policy = fail_closed_rows()
    criterion_ready = all(bool(row["implementation_ready"]) and bool(row["can_fail_closed"]) for row in criteria)
    family_ready = all(row["sidecar_implementation_readiness"] in {"ready_fail_closed", "unresolved_only"} for row in families)
    join_ready = identity["row_id_duplicate_count"] == 0 and identity["duplicate_semantic_identity_key_count"] == 0
    ready = topology_passed and criterion_ready and family_ready and join_ready
    decision = READY_DECISION if ready else BLOCKED_DECISION
    next_stage = READY_NEXT if ready else "STAGE185_CONTROLLED_GENERATOR_PROVENANCE_RECOVERY_AUDIT"

    coverage_rows = [
        {"coverage_kind": "decision_coverage", "population": "all authoritative rows", "denominator": len(rows), "value": len(rows) if ready else 0, "status": "structurally computable at Stage185" if ready else "blocked", "threshold_used": None, "risk": "UNRESOLVED may dominate but never enters loss"},
        {"coverage_kind": "positive_eligible_coverage", "population": "train frame-compatible rows", "denominator": identity["train_frame_compatible"], "value": None, "status": "unavailable before builder", "threshold_used": args.minimum_safe_positive_rate, "risk": "eligible count may be small or family-concentrated"},
        {"coverage_kind": "scientific_usability", "population": "eligible train positives", "denominator": None, "value": None, "status": "requires Stage185 family/reason-code distribution", "threshold_used": None, "risk": "technical sidecar completeness does not imply useful training coverage"},
    ]
    gate_rows = [
        {"gate": "authoritative_dataset_identity_and_topology", "required": True, "passed": topology_passed, "evidence": "Stage183 closure cross-check + identity audit", "authorization": next_stage},
        {"gate": "all_criteria_have_deterministic_PASS_FAIL_UNRESOLVED_rules", "required": True, "passed": criterion_ready, "evidence": "criterion derivability matrix", "authorization": next_stage},
        {"gate": "one_to_one_row_id_sidecar_join", "required": True, "passed": join_ready, "evidence": "no duplicate row or pair-family identity", "authorization": next_stage},
        {"gate": "no_model_llm_manual_annotation_required", "required": True, "passed": all(not row["model_required"] and not row["manual_annotation_required"] for row in criteria), "evidence": "criterion derivability matrix", "authorization": next_stage},
        {"gate": "original_jsonl_modification_required", "required": False, "passed": True, "evidence": "external row-id sidecar contract", "authorization": "modification forbidden"},
        {"gate": "loss_or_training_authorized", "required": False, "passed": False, "evidence": "Stage184 scope", "authorization": "not authorized"},
    ]
    optional_inputs = [
        root / "reports/stage183a_compatible_positive_preservation_policy.md",
        root / "reports/stage183a_positive_preservation_design_report.json",
    ]
    input_files = required + [path for path in optional_inputs if path.exists()]
    report = {
        "stage": STAGE,
        "decision": decision,
        "scope": {"static_specification_and_feasibility_only": True, "sidecar_built": False, "dataset_modified": False, "loss_implemented": False, "training": False},
        "input_validation": {
            "status": "passed" if topology_passed else "blocked",
            "required_files": [{"path": str(path), "sha256": digest(path)} for path in input_files],
            "optional_stage183_runtime_report_available": optional_inputs[1].exists(),
            "stage183_fallback_used": not optional_inputs[1].exists(),
            "fallback_sources": [str(closure183), str(root / "reports/stage184a_controlled_train_integrity_mask_policy.md")],
            "split_seed": args.split_seed, "dev_ratio": args.dev_ratio,
            "minimum_safe_positive_rate": args.minimum_safe_positive_rate,
            "minimum_safe_positive_rate_is_reporting_only": True,
        },
        "dataset_identity": identity,
        "generator_provenance": {
            "source": str(generator), "sha256": digest(generator),
            "structured_fact_arguments_available": True,
            "original_and_alternate_axes_available": True,
            "deterministic_family_branches_available": True,
            "row_metadata_embeds_structured_arguments": False,
            "reconstruction_requires_exact_generator_and_dataset_identity": True,
            "generator_equality_is_cleanliness": False,
            "evidence": [line_evidence(generator, "_FACT_FIELDS ="), line_evidence(generator, "def _build_records("), line_evidence(stage182, "def reconstruct_generator(")],
        },
        "criterion_derivability": criteria,
        "family_contracts": families,
        "pair_invariants": invariants,
        "sidecar_schema": schema,
        "reason_codes": codes,
        "fail_closed_policy": fail_policy,
        "coverage_feasibility": {
            "decision_coverage": "all rows classifiable as ELIGIBLE/INELIGIBLE/UNRESOLVED at Stage185" if ready else "blocked",
            "positive_eligible_coverage": "unavailable before builder",
            "expected_exact_eligible_count": None,
            "scientific_usability": "unknown until family/reason-code coverage is materialized",
            "minimum_safe_positive_rate": args.minimum_safe_positive_rate,
            "threshold_fitted": False,
            "rows": coverage_rows,
        },
        "implementation_readiness": {
            "dataset_topology_ready": topology_passed,
            "criteria_ready_fail_closed": criterion_ready,
            "family_contract_join_ready": family_ready,
            "row_id_sidecar_join_ready": join_ready,
            "generator_provenance_augmentation_required_first": False if ready else None,
            "stage185_sidecar_builder_ready": ready,
        },
        "stage185_gate": {"authorized_next_stage": next_stage, "sidecar_generation_and_static_audit_only": ready, "loss_implementation_authorized": False, "training_authorized": False, "rows": gate_rows},
        "limitations": [
            "Stage184-A does not build or count ELIGIBLE rows.",
            "General English grammaticality is outside deterministic template rules and may remain UNRESOLVED.",
            "Exact generator reproduction proves provenance, not cleanliness.",
            "Eligible positive coverage and family concentration are unknown before Stage185.",
            "Canonical structural validity does not assert real-world truth or complete semantic quality.",
        ],
        "safety_policy": {
            "no_dataset_modification": True, "no_jsonl_rewrite": True, "no_generator_modification": True,
            "no_model_import": True, "no_torch_import": True, "no_checkpoint_load": True, "no_forward": True,
            "no_loss_implementation": True, "no_training": True, "no_smoke": True, "no_annotation": True,
            "no_llm_labeling": True, "no_text_classifier": True, "no_learned_parser_or_probe": True,
            "no_threshold_fitting": True, "no_calibration": True, "no_external_evaluation": True,
            "no_time_swap": True, "no_multi_seed": True, "no_hyperparameter_sweep": True,
        },
    }

    write_json(output_dir / "stage184a_controlled_train_integrity_mask_spec_report.json", report)
    (output_dir / "stage184a_controlled_train_integrity_mask_spec_report.md").write_text(markdown(report), encoding="utf-8")
    write_csv(output_dir / "stage184a_dataset_identity_audit.csv", ["metric", "observed", "expected", "passed", "evidence_source"], identity_csv)
    write_csv(output_dir / "stage184a_integrity_criterion_derivability.csv", ["criterion", "required", "authoritative_source", "derivation_mode", "structured_inputs", "deterministic", "exhaustive", "text_heuristic_required", "model_required", "manual_annotation_required", "can_fail_closed", "unresolved_possible", "implementation_ready", "evidence", "limitation"], criteria)
    write_csv(output_dir / "stage184a_family_contract_matrix.csv", ["family", "row_count", "pair_coverage", "canonical_counterpart_available", "intended_changed_axes", "intended_preserved_axes", "expected_label_transition", "canonical_counterpart_requirement", "evidence_text_relation_requirement", "reason_codes", "generator_branch_location", "structured_provenance_available", "deterministic_contract_check_possible", "polarity_check_possible", "grammar_check_possible", "unresolved_risk", "sidecar_implementation_readiness"], families)
    write_csv(output_dir / "stage184a_pair_invariant_spec.csv", ["invariant", "level", "validation", "failure_scope", "failure_status", "reason_code", "stage185_action"], invariants)
    write_csv(output_dir / "stage184a_sidecar_schema.csv", ["field", "type", "nullable", "source", "contract"], schema)
    write_csv(output_dir / "stage184a_reason_code_catalog.csv", ["reason_code", "level", "status_effect", "criterion", "deterministic_source", "fail_closed_behavior", "human_interpretation"], codes)
    write_csv(output_dir / "stage184a_fail_closed_policy.csv", ["condition", "scope", "result", "positive_margin_eligible", "propagation", "rationale"], fail_policy)
    write_csv(output_dir / "stage184a_coverage_feasibility.csv", ["coverage_kind", "population", "denominator", "value", "status", "threshold_used", "risk"], coverage_rows)
    write_csv(output_dir / "stage184a_stage185_gate.csv", ["gate", "required", "passed", "evidence", "authorization"], gate_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
