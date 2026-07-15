#!/usr/bin/env python3
"""Build the Stage185-A deterministic controlled-train integrity sidecar.

This builder imports no model or torch package, never rewrites source data, and
uses Stage182 only as a regression oracle.  All uncertain evidence fails closed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable


STAGE = "Stage185-A"
STAGE184_DECISION = "STAGE184A_DETERMINISTIC_FAIL_CLOSED_INTEGRITY_SIDECAR_SPEC_READY"
STAGE184_ROUTE = "STAGE185_CONTROLLED_TRAIN_INTEGRITY_SIDECAR_BUILDER"
STAGE182_DECISION = "STAGE182A_DATA_CONTAMINATION_CONFIRMED_AND_CLEAN_MODEL_FAILURE_SET_READY"
SUCCESS_DECISION = "STAGE185A_INTEGRITY_SIDECAR_BUILT_AND_POSITIVE_ELIGIBILITY_MATERIALIZED"
ZERO_DECISION = "STAGE185A_INTEGRITY_SIDECAR_BUILT_WITH_ZERO_POSITIVE_ELIGIBILITY"
BLOCKED_DECISION = "STAGE185A_CONTROLLED_TRAIN_INTEGRITY_SIDECAR_BUILD_BLOCKED"
SUCCESS_NEXT = "STAGE186_COMPATIBLE_POSITIVE_MARGIN_FIXED_SPEC_AUDIT"
ZERO_NEXT = "STAGE186_GENERATOR_PROVENANCE_RECOVERY_SPEC"
AUTHORITATIVE_DATA = Path("data/controlled_v5_v3_without_time_swap.jsonl")
AUTHORITATIVE_SHA256 = "f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640"
STATUS_ENUM = {"PASS", "FAIL", "UNRESOLVED", "NOT_APPLICABLE"}
INTEGRITY_ENUM = {"ELIGIBLE", "INELIGIBLE", "UNRESOLVED"}
CRITERION_FIELDS = (
    "grammar_status", "intervention_contract_status",
    "polarity_contamination_status", "schema_status", "canonical_status",
    "time_swap_status", "dataset_source_status",
)
BASE_FIELDS = {
    "id", "pair_id", "claim", "evidence", "final_label",
    "frame_compatible_label", "predicate_covered_label", "sufficiency_label",
    "polarity_label", "primary_failure_type", "intervention_type",
}
BINARY_FIELDS = {"frame_compatible_label", "predicate_covered_label", "sufficiency_label"}
FINAL_LABELS = {"SUPPORT", "NOT_ENTITLED", "REFUTE"}
POLARITY_LABELS = {"SUPPORT", "NONE", "REFUTE"}
PRIMARY_FAILURE_TYPES = {"none", "frame", "predicate", "sufficiency", "polarity"}
SLOT_AXES = {"title", "name", "role", "predicate", "object", "location", "time"}
CONTENT_OPERATIONS = {"evidence_deletion", "evidence_truncation", "irrelevant_evidence"}
CONTAMINATION_CODES = {
    "NON_POLARITY_INTERVENTION_POLARITY_CHANGE", "DID_NOT_INFLECTED_PREDICATE",
}

STAGE184_FILES = (
    "stage184a_controlled_train_integrity_mask_spec_report.json",
    "stage184a_controlled_train_integrity_mask_spec_report.md",
    "stage184a_dataset_identity_audit.csv",
    "stage184a_integrity_criterion_derivability.csv",
    "stage184a_family_contract_matrix.csv",
    "stage184a_pair_invariant_spec.csv",
    "stage184a_sidecar_schema.csv",
    "stage184a_reason_code_catalog.csv",
    "stage184a_fail_closed_policy.csv",
    "stage184a_coverage_feasibility.csv",
    "stage184a_stage185_gate.csv",
)
STAGE182_FILES = (
    "stage182a_controlled_intervention_integrity_report.json",
    "stage182a_unique_item_integrity.csv",
    "stage182a_intervention_contract_audit.csv",
    "stage182a_grammar_integrity_audit.csv",
    "stage182a_canonical_control_audit.csv",
    "stage182a_structured_axis_delta.csv",
    "stage182a_data_intervention_contamination_queue.csv",
    "stage182a_control_anomaly_queue.csv",
)
OUTPUT_FILES = (
    "stage185a_controlled_train_integrity_sidecar_report.json",
    "stage185a_controlled_train_integrity_sidecar_report.md",
    "stage185a_controlled_train_integrity_sidecar.jsonl",
    "stage185a_controlled_train_integrity_sidecar.csv",
    "stage185a_dataset_join_audit.csv",
    "stage185a_criterion_status_summary.csv",
    "stage185a_integrity_status_summary.csv",
    "stage185a_positive_eligibility_summary.csv",
    "stage185a_family_coverage.csv",
    "stage185a_pair_integrity_audit.csv",
    "stage185a_reason_code_counts.csv",
    "stage185a_stage182_overlap_regression.csv",
    "stage185a_eligible_positive_rows.csv",
    "stage185a_ineligible_rows.csv",
    "stage185a_unresolved_rows.csv",
    "stage185a_provenance.json",
    "stage185a_stage186_gate.csv",
)
SIDECAR_FIELDS = [
    "row_id", "pair_id", "split", "intervention_type", "frame_compatible_label",
    "grammar_status", "intervention_contract_status",
    "polarity_contamination_status", "schema_status", "canonical_status",
    "time_swap_status", "dataset_source_status", "integrity_status",
    "eligible_for_positive_margin", "reason_codes", "canonical_row_id",
    "family_contract_id", "rule_version", "source_dataset_path",
    "source_dataset_sha256", "generator_source_path", "generator_source_sha256",
    "stage182a_report_sha256", "stage184a_report_sha256",
    "integrity_builder_sha256", "created_at", "audit_changed_axes",
    "audit_preserved_axes", "audit_expected_axes", "audit_pair_failure_scope",
]


class BuildBlocked(ValueError):
    """A dataset, provenance, regression, or output invariant blocked the build."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BuildBlocked(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--stage184a-dir", type=Path, required=True)
    parser.add_argument("--stage182a-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generator-source", type=Path, default=Path("scripts/build_controlled_v5.py"))
    parser.add_argument("--stage182a-analyzer-source", type=Path, default=Path("scripts/analyze_stage182a_controlled_intervention_integrity.py"))
    parser.add_argument("--split-seed", type=int, default=174)
    parser.add_argument("--dev-ratio", type=float, default=0.2)
    parser.add_argument("--rule-version", default="stage185a_v1")
    args = parser.parse_args()
    if not 0.0 < args.dev_ratio < 1.0:
        parser.error("--dev-ratio must satisfy 0 < value < 1")
    if not str(args.rule_version).strip():
        parser.error("--rule-version must be nonempty")
    return args


def resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            require(isinstance(value, dict), f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def read_csv(path: Path, required: set[str] | None = None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if required:
            require(required <= fields, f"{path}: missing columns {sorted(required - fields)}")
        return [dict(row) for row in reader]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def csv_cell(value: Any) -> Any:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "" if value is None else value


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]], *, sidecar: bool = False) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            encoded = {field: csv_cell(row.get(field)) for field in fields}
            if sidecar:
                encoded["reason_codes"] = "|".join(row.get("reason_codes", []))
                for field in ("audit_changed_axes", "audit_preserved_axes", "audit_expected_axes"):
                    encoded[field] = "|".join(row.get(field, []))
            writer.writerow(encoded)


def parse_json_cell(value: str) -> Any:
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


def bool_cell(value: str) -> bool:
    return str(value).strip().lower() == "true"


def decision_value(report: dict[str, Any]) -> Any:
    return report.get("decision") or report.get("official_decision")


def require_files(directory: Path, names: Iterable[str]) -> dict[str, Path]:
    paths = {name: directory / name for name in names}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    require(not missing, "missing required artifacts: " + ", ".join(missing))
    return paths


def safe_load_generator(path: Path, root: Path) -> ModuleType:
    labels_path = root / "src/contramamba/labels.py"
    require(labels_path.is_file(), f"label schema missing: {labels_path}")
    labels_spec = importlib.util.spec_from_file_location("contramamba.labels", labels_path)
    require(labels_spec is not None and labels_spec.loader is not None, "cannot load label schema spec")
    labels_module = importlib.util.module_from_spec(labels_spec)
    package = ModuleType("contramamba")
    package.__path__ = [str(labels_path.parent)]  # type: ignore[attr-defined]
    sentinel = object()
    old_package = sys.modules.get("contramamba", sentinel)
    old_labels = sys.modules.get("contramamba.labels", sentinel)
    try:
        sys.modules["contramamba"] = package
        sys.modules["contramamba.labels"] = labels_module
        labels_spec.loader.exec_module(labels_module)
        spec = importlib.util.spec_from_file_location("stage185_controlled_generator", path)
        require(spec is not None and spec.loader is not None, "cannot load generator spec")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if old_package is sentinel:
            sys.modules.pop("contramamba", None)
        else:
            sys.modules["contramamba"] = old_package  # type: ignore[assignment]
        if old_labels is sentinel:
            sys.modules.pop("contramamba.labels", None)
        else:
            sys.modules["contramamba.labels"] = old_labels  # type: ignore[assignment]


def exact_map(rows: Iterable[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row.get(key, ""))
        require(bool(value), f"missing identity {key}")
        require(value not in result, f"duplicate {key}: {value}")
        result[value] = row
    return result


def split_by_pair(rows: list[dict[str, Any]], seed: int, ratio: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    pair_ids = sorted({str(row["pair_id"]) for row in rows})
    random.Random(seed).shuffle(pair_ids)
    count = min(len(pair_ids) - 1, max(1, round(len(pair_ids) * ratio)))
    dev_ids = set(pair_ids[:count])
    train = [row for row in rows if str(row["pair_id"]) not in dev_ids]
    dev = [row for row in rows if str(row["pair_id"]) in dev_ids]
    return train, dev, dev_ids


def schema_errors(row: dict[str, Any]) -> list[str]:
    errors = [f"MISSING_REQUIRED_BASE_FIELD:{field}" for field in sorted(BASE_FIELDS - set(row))]
    for field in ("id", "pair_id", "claim", "evidence", "final_label", "polarity_label", "primary_failure_type", "intervention_type"):
        if field in row and (not isinstance(row[field], str) or not row[field].strip()):
            errors.append(f"INVALID_FIELD_TYPE:{field}")
    for field in BINARY_FIELDS:
        value = row.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
            errors.append(f"INVALID_FIELD_TYPE:{field}")
    if row.get("final_label") not in FINAL_LABELS:
        errors.append("INVALID_ENUM:final_label")
    if row.get("polarity_label") not in POLARITY_LABELS:
        errors.append("INVALID_ENUM:polarity_label")
    if row.get("primary_failure_type") not in PRIMARY_FAILURE_TYPES:
        errors.append("INVALID_ENUM:primary_failure_type")
    return errors


def polarity(text: str) -> str:
    return "negative" if re.search(r"\bdid\s+not\b", text, re.IGNORECASE) else "positive"


def semantic_state(fact: dict[str, Any], evidence: str) -> dict[str, str]:
    state: dict[str, str] = {}
    for axis in SLOT_AXES:
        original = str(fact[axis])
        alternate = str(fact.get(f"alternate_{axis}", ""))
        original_present = original in evidence
        alternate_present = bool(alternate) and alternate in evidence
        if original_present and not alternate_present:
            state[axis] = f"original:{original}"
        elif alternate_present and not original_present:
            state[axis] = f"alternate:{alternate}"
        elif original_present and alternate_present:
            state[axis] = f"ambiguous_both:{original}|{alternate}"
        else:
            state[axis] = "absent"
    state["polarity"] = polarity(evidence)
    return state


def changed_axes(row: dict[str, Any], canonical: dict[str, Any], fact: dict[str, Any], intended: set[str]) -> set[str]:
    family = str(row["intervention_type"])
    if family in CONTENT_OPERATIONS:
        changed = set(intended)
        if polarity(str(row["evidence"])) != polarity(str(canonical["evidence"])):
            changed.add("polarity")
        return changed
    base = semantic_state(fact, str(canonical["evidence"]))
    current = semantic_state(fact, str(row["evidence"]))
    changed = {axis for axis in base if base[axis] != current[axis]}
    if family == "paraphrase" and row["evidence"] != canonical["evidence"]:
        changed.add("realization")
    return changed


def grammar_anomaly(row: dict[str, Any], fact: dict[str, Any]) -> bool:
    predicates = {str(fact["predicate"]), str(fact["alternate_predicate"])}
    text = str(row["evidence"])
    return any(re.search(rf"\bdid\s+not\s+{re.escape(value)}\b", text, re.IGNORECASE) for value in predicates)


def labels_match(row: dict[str, Any], expected: dict[str, Any]) -> bool:
    fields = (
        "final_label", "frame_compatible_label", "predicate_covered_label",
        "sufficiency_label", "polarity_label", "primary_failure_type",
    )
    return all(row.get(field) == expected.get(field) for field in fields)


def stage184_identity(report: dict[str, Any]) -> dict[str, Any]:
    identity = report.get("dataset_identity") or {}
    require(isinstance(identity, dict), "Stage184 dataset_identity is not an object")
    return identity


def recorded_sha(report: dict[str, Any], suffix: str) -> str | None:
    provenance = report.get("generator_provenance") or {}
    if suffix == "build_controlled_v5.py" and isinstance(provenance, dict):
        value = provenance.get("sha256")
        if isinstance(value, str):
            return value
    validation = report.get("input_validation") or {}
    for item in validation.get("required_files", []) if isinstance(validation, dict) else []:
        if str(item.get("path", "")).replace("\\", "/").endswith(suffix):
            return item.get("sha256")
    return None


def validate_inputs(
    root: Path, data_path: Path, generator_path: Path, analyzer_path: Path,
    stage184_dir: Path, stage182_dir: Path,
) -> tuple[dict[str, Path], dict[str, Path], dict[str, Any], dict[str, Any]]:
    require(data_path == (root / AUTHORITATIVE_DATA).resolve(), "data path is not authoritative main data")
    require(data_path.is_file(), f"dataset missing: {data_path}")
    require(generator_path.is_file(), f"generator missing: {generator_path}")
    require(analyzer_path.is_file(), f"Stage182 analyzer missing: {analyzer_path}")
    stage184 = require_files(stage184_dir, STAGE184_FILES)
    stage182 = require_files(stage182_dir, STAGE182_FILES)
    report184 = read_json(stage184[STAGE184_FILES[0]])
    report182 = read_json(stage182[STAGE182_FILES[0]])
    require(decision_value(report184) == STAGE184_DECISION, "Stage184 decision mismatch")
    gate = report184.get("stage185_gate") or {}
    require(gate.get("authorized_next_stage") == STAGE184_ROUTE, "Stage184 authorized route mismatch")
    require(decision_value(report182) == STAGE182_DECISION, "Stage182 decision mismatch")
    generator_recorded = recorded_sha(report184, "scripts/build_controlled_v5.py")
    if generator_recorded:
        require(generator_recorded == sha256(generator_path), "generator SHA mismatch against Stage184")
    analyzer_recorded = recorded_sha(report184, "scripts/analyze_stage182a_controlled_intervention_integrity.py")
    if analyzer_recorded:
        require(analyzer_recorded == sha256(analyzer_path), "Stage182 analyzer SHA mismatch against Stage184")
    return stage184, stage182, report184, report182


def validate_dataset(
    rows: list[dict[str, Any]], report184: dict[str, Any], seed: int, ratio: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    require(sha256_value := bool(rows), "dataset is empty")
    del sha256_value
    errors = [(str(row.get("id", index)), schema_errors(row)) for index, row in enumerate(rows)]
    require(not any(value for _, value in errors), f"base schema errors: {[item for item in errors if item[1]][:5]}")
    row_ids = [str(row["id"]) for row in rows]
    require(len(row_ids) == len(set(row_ids)), "duplicate row_id")
    semantic = [(str(row["pair_id"]), str(row["intervention_type"])) for row in rows]
    require(len(semantic) == len(set(semantic)), "duplicate pair_id + intervention_type")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["pair_id"])].append(row)
    families = sorted({str(row["intervention_type"]) for row in rows})
    family_set = set(families)
    require(all({str(row["intervention_type"]) for row in group} == family_set for group in groups.values()), "nonrectangular pair/family topology")
    require(all(sum(row["intervention_type"] == "none" for row in group) == 1 for group in groups.values()), "canonical none count is not exactly one per pair")
    require(sum(row["intervention_type"] == "time_swap" for row in rows) == 0, "time_swap rows present")
    train, dev, dev_ids = split_by_pair(rows, seed, ratio)
    train_labels = Counter(row["frame_compatible_label"] for row in train)
    identity = {
        "dataset_sha256": AUTHORITATIVE_SHA256,
        "row_count": len(rows), "pair_count": len(groups),
        "intervention_family_count": len(families), "intervention_families": families,
        "rows_per_pair_distribution": dict(Counter(len(group) for group in groups.values())),
        "canonical_none_per_pair_distribution": dict(Counter(sum(row["intervention_type"] == "none" for row in group) for group in groups.values())),
        "train_pair_count": len({row["pair_id"] for row in train}),
        "dev_pair_count": len({row["pair_id"] for row in dev}),
        "train_row_count": len(train), "dev_row_count": len(dev),
        "train_frame_compatible": train_labels.get(1, 0),
        "train_frame_incompatible": train_labels.get(0, 0),
        "time_swap_count": 0,
    }
    observed_sha = report184.get("dataset_identity", {}).get("dataset_sha256")
    require(observed_sha == AUTHORITATIVE_SHA256, "Stage184 dataset SHA mismatch")
    expected = stage184_identity(report184)
    aliases = {
        "row_count": "row_count", "pair_count": "pair_count",
        "train_pair_count": "train_pair_count", "dev_pair_count": "dev_pair_count",
        "train_row_count": "train_row_count", "dev_row_count": "dev_row_count",
        "train_frame_compatible": "train_frame_compatible",
        "train_frame_incompatible": "train_frame_incompatible", "time_swap_count": "time_swap_count",
    }
    for actual_key, report_key in aliases.items():
        require(identity[actual_key] == expected.get(report_key), f"Stage184 topology mismatch: {actual_key}")
    require(set(families) == set(expected.get("intervention_families", [])), "Stage184 family set mismatch")
    return identity, train, dev, dev_ids


def load_contracts(path: Path, families: set[str]) -> dict[str, dict[str, Any]]:
    rows = read_csv(path, {"family", "intended_changed_axes", "intended_preserved_axes", "structured_provenance_available", "sidecar_implementation_readiness"})
    contracts = exact_map(rows, "family")
    require(set(contracts) == families, "Stage184 family contract set mismatch")
    for family, row in contracts.items():
        row["changed_axes"] = parse_json_cell(row["intended_changed_axes"])
        row["preserved_axes"] = parse_json_cell(row["intended_preserved_axes"])
        require(isinstance(row["changed_axes"], list), f"invalid changed axes for {family}")
        require(bool_cell(row["structured_provenance_available"]), f"structured provenance unavailable for {family}")
    return contracts


def reconstruct(
    module: ModuleType, rows: list[dict[str, Any]], families: set[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    pair_ids = sorted({str(row["pair_id"]) for row in rows})
    generated = module.build_controlled_records(len(pair_ids))
    expected_rows = [row for row in generated if str(row["intervention_type"]) in families]
    expected = exact_map(expected_rows, "id")
    observed = exact_map(rows, "id")
    require(set(expected) == set(observed), "generator/dataset row identity mismatch")
    mismatches = [row_id for row_id in sorted(expected) if expected[row_id] != observed[row_id]]
    require(not mismatches, f"generator/dataset exact equality mismatch: {mismatches[:10]}")
    facts = exact_map(module.fact_templates_for_count(len(pair_ids)), "pair_id")
    require(set(facts) == set(pair_ids), "generator fact/pair mismatch")
    return expected, facts


def build_sidecar(
    rows: list[dict[str, Any]], contracts: dict[str, dict[str, Any]],
    expected: dict[str, dict[str, Any]], facts: dict[str, dict[str, Any]],
    dev_ids: set[str], provenance: dict[str, str], rule_version: str, created_at: str,
) -> list[dict[str, Any]]:
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[str(row["pair_id"])].append(row)
    canonical = {
        pair_id: next(row for row in group if row["intervention_type"] == "none")
        for pair_id, group in by_pair.items()
    }
    canonical_defect = {
        pair_id: grammar_anomaly(row, facts[pair_id])
        for pair_id, row in canonical.items()
    }
    result: list[dict[str, Any]] = []
    non_polarity = set(contracts) - {"none", "polarity_flip"}
    for row in rows:
        row_id = str(row["id"])
        pair_id = str(row["pair_id"])
        family = str(row["intervention_type"])
        fact = facts[pair_id]
        anchor = canonical[pair_id]
        contract = contracts[family]
        intended = set(contract["changed_axes"])
        observed = changed_axes(row, anchor, fact, intended)
        unexpected = observed - intended
        missing = intended - observed
        reasons: set[str] = set()

        grammar_bad = grammar_anomaly(row, fact)
        grammar_status = "FAIL" if grammar_bad else "PASS"
        if grammar_bad:
            reasons.update({"DID_NOT_INFLECTED_PREDICATE", "GRAMMAR_TEMPLATE_FAIL"})

        contract_bad = bool(unexpected or missing or not labels_match(row, expected[row_id]))
        intervention_status = "FAIL" if contract_bad else "PASS"
        if contract_bad:
            reasons.add("INTERVENTION_CONTRACT_FAIL")

        polarity_bad = family in non_polarity and "polarity" in observed
        if family == "polarity_flip" and "polarity" not in observed:
            polarity_bad = True
        polarity_status = "FAIL" if polarity_bad else "PASS"
        if polarity_bad:
            reasons.add("NON_POLARITY_INTERVENTION_POLARITY_CHANGE" if family in non_polarity else "POLARITY_REALIZATION_MISMATCH")

        schema_status = "PASS"
        canonical_status = "PASS"
        pair_failure_scope = "none"
        if row["claim"] != anchor["claim"] or anchor["pair_id"] != pair_id:
            canonical_status = "UNRESOLVED"
            pair_failure_scope = "pair"
            reasons.add("CANONICAL_LINKAGE_INVALID")
        elif canonical_defect[pair_id] and row_id != anchor["id"]:
            canonical_status = "UNRESOLVED"
            pair_failure_scope = "pair"
            reasons.add("CANONICAL_ROW_KNOWN_GENERATOR_DEFECT")

        time_status = "PASS" if family != "time_swap" else "FAIL"
        if time_status == "FAIL":
            reasons.add("TIME_SWAP_EXCLUDED")
        source_status = "PASS"
        statuses = [grammar_status, intervention_status, polarity_status, schema_status, canonical_status, time_status, source_status]
        require(all(value in STATUS_ENUM for value in statuses), f"invalid criterion status for {row_id}")
        if "FAIL" in statuses:
            integrity = "INELIGIBLE"
        elif "UNRESOLVED" in statuses or "NOT_APPLICABLE" in statuses:
            integrity = "UNRESOLVED"
        else:
            integrity = "ELIGIBLE"
        split = "dev" if pair_id in dev_ids else "train"
        eligible = (
            integrity == "ELIGIBLE" and split == "train"
            and row["frame_compatible_label"] == 1
            and time_status == "PASS" and source_status == "PASS"
        )
        if eligible:
            reasons.add("ELIGIBLE_CLEAN_COMPATIBLE")
        else:
            if split == "dev": reasons.add("DEV_SPLIT_EXCLUDED")
            if row["frame_compatible_label"] != 1: reasons.add("FRAME_LABEL_NOT_COMPATIBLE")
            if source_status != "PASS": reasons.add("EXTERNAL_DATA_EXCLUDED")
        result.append({
            "row_id": row_id, "pair_id": pair_id, "split": split,
            "intervention_type": family,
            "frame_compatible_label": row["frame_compatible_label"],
            "grammar_status": grammar_status,
            "intervention_contract_status": intervention_status,
            "polarity_contamination_status": polarity_status,
            "schema_status": schema_status, "canonical_status": canonical_status,
            "time_swap_status": time_status, "dataset_source_status": source_status,
            "integrity_status": integrity, "eligible_for_positive_margin": eligible,
            "reason_codes": sorted(reasons), "canonical_row_id": str(anchor["id"]),
            "family_contract_id": f"{rule_version}:{family}", "rule_version": rule_version,
            **provenance, "created_at": created_at,
            "audit_changed_axes": sorted(observed),
            "audit_preserved_axes": sorted(set(contract["preserved_axes"]) - observed),
            "audit_expected_axes": sorted(intended),
            "audit_pair_failure_scope": pair_failure_scope,
        })
    return result


def overlap_regression(
    sidecar: list[dict[str, Any]], unique_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    side_by_id = exact_map(sidecar, "row_id")
    unique_by_id = exact_map(unique_rows, "row_id")
    audit: list[dict[str, Any]] = []
    expected_polarity = expected_grammar = recovered_polarity = recovered_grammar = 0
    contaminated_count = contaminated_eligible = clean_contamination_code = 0
    for row_id, oracle in sorted(unique_by_id.items()):
        require(row_id in side_by_id, f"Stage182 overlap row absent from sidecar: {row_id}")
        observed = side_by_id[row_id]
        raw_codes = parse_json_cell(oracle.get("anomaly_codes", "[]"))
        oracle_codes = set(raw_codes if isinstance(raw_codes, list) else [])
        expected_codes = oracle_codes & CONTAMINATION_CODES
        observed_codes = set(observed["reason_codes"])
        is_contaminated = oracle.get("integrity_status") == "CONTAMINATED_CONSTRUCTION"
        is_clean = oracle.get("integrity_status") == "CLEAN_SINGLE_AXIS_CONSTRUCTION"
        if "NON_POLARITY_INTERVENTION_POLARITY_CHANGE" in expected_codes: expected_polarity += 1
        if "DID_NOT_INFLECTED_PREDICATE" in expected_codes: expected_grammar += 1
        if "NON_POLARITY_INTERVENTION_POLARITY_CHANGE" in observed_codes and "NON_POLARITY_INTERVENTION_POLARITY_CHANGE" in expected_codes: recovered_polarity += 1
        if "DID_NOT_INFLECTED_PREDICATE" in observed_codes and "DID_NOT_INFLECTED_PREDICATE" in expected_codes: recovered_grammar += 1
        if is_contaminated:
            contaminated_count += 1
            contaminated_eligible += int(observed["integrity_status"] == "ELIGIBLE")
        if is_clean and observed_codes & CONTAMINATION_CODES:
            clean_contamination_code += 1
        passed = (
            (not is_contaminated or observed["integrity_status"] == "INELIGIBLE")
            and expected_codes <= observed_codes
            and not (is_clean and observed_codes & CONTAMINATION_CODES)
        )
        audit.append({
            "row_id": row_id, "stage182_integrity_status": oracle.get("integrity_status"),
            "stage182_contamination_codes": sorted(expected_codes),
            "stage185_integrity_status": observed["integrity_status"],
            "stage185_contamination_codes": sorted(observed_codes & CONTAMINATION_CODES),
            "stage185_eligible_for_positive_margin": observed["eligible_for_positive_margin"],
            "passed": passed,
        })
    summary = {
        "overlap_count": len(audit), "duplicate_overlap_count": 0,
        "deterministic_contaminated_count": contaminated_count,
        "contaminated_classified_eligible": contaminated_eligible,
        "expected_polarity_contamination": expected_polarity,
        "recovered_polarity_contamination": recovered_polarity,
        "expected_grammar_contamination": expected_grammar,
        "recovered_grammar_contamination": recovered_grammar,
        "clean_rows_with_contamination_code": clean_contamination_code,
    }
    summary["passed"] = (
        all(row["passed"] for row in audit)
        and contaminated_count == 22 and contaminated_eligible == 0
        and expected_polarity == recovered_polarity == 21
        and expected_grammar == recovered_grammar == 1
        and clean_contamination_code == 0
    )
    return audit, summary


def coverage(sidecar: list[dict[str, Any]]) -> dict[str, Any]:
    integrity = Counter(row["integrity_status"] for row in sidecar)
    criterion = {field: Counter(row[field] for row in sidecar) for field in CRITERION_FIELDS}
    family_rows: list[dict[str, Any]] = []
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sidecar: by_family[row["intervention_type"]].append(row)
    for family in sorted(by_family):
        rows = by_family[family]
        eligible = [row for row in rows if row["eligible_for_positive_margin"]]
        reason_counts = Counter(code for row in rows for code in row["reason_codes"])
        compatible = sum(row["frame_compatible_label"] == 1 for row in rows)
        family_rows.append({
            "family": family, "total": len(rows),
            "train": sum(row["split"] == "train" for row in rows),
            "compatible": compatible, "eligible_positive": len(eligible),
            "ineligible": sum(row["integrity_status"] == "INELIGIBLE" for row in rows),
            "unresolved": sum(row["integrity_status"] == "UNRESOLVED" for row in rows),
            "eligible_positive_rate": len(eligible) / compatible if compatible else 0.0,
            "dominant_reason_codes": [code for code, count in reason_counts.most_common() if count == reason_counts.most_common(1)[0][1]][:5] if reason_counts else [],
        })
    eligible = [row for row in sidecar if row["eligible_for_positive_margin"]]
    train_compatible = [row for row in sidecar if row["split"] == "train" and row["frame_compatible_label"] == 1]
    eligible_families = Counter(row["intervention_type"] for row in eligible)
    largest_share = max(eligible_families.values(), default=0) / len(eligible) if eligible else 0.0
    warnings: list[str] = []
    if not eligible: warnings.append("ZERO_ELIGIBLE_POSITIVES")
    if len(eligible_families) == 1: warnings.append("SINGLE_FAMILY_ELIGIBILITY")
    if largest_share > 0.80: warnings.append("FAMILY_CONCENTRATION_WARNING")
    unresolved_positive = [row for row in train_compatible if row["integrity_status"] == "UNRESOLVED"]
    ineligible_positive = [row for row in train_compatible if row["integrity_status"] == "INELIGIBLE"]
    return {
        "integrity": integrity, "criterion": criterion, "family_rows": family_rows,
        "eligible_rows": eligible, "train_compatible": train_compatible,
        "eligible_count": len(eligible),
        "eligible_rate": len(eligible) / len(train_compatible) if train_compatible else 0.0,
        "eligible_family_count": len(eligible_families), "eligible_family_counts": eligible_families,
        "largest_family_share": largest_share,
        "families_with_zero_eligible": sorted(set(by_family) - set(eligible_families)),
        "unresolved_positive_count": len(unresolved_positive),
        "top_unresolved_reason": top_reason(unresolved_positive),
        "top_ineligible_reason": top_reason(ineligible_positive), "warnings": warnings,
    }


def top_reason(rows: list[dict[str, Any]]) -> str | None:
    counts = Counter(code for row in rows for code in row["reason_codes"] if not code.endswith("_EXCLUDED"))
    return counts.most_common(1)[0][0] if counts else None


def pair_audit(sidecar: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sidecar: groups[row["pair_id"]].append(row)
    result = []
    for pair_id in sorted(groups):
        rows = groups[pair_id]
        statuses = Counter(row["integrity_status"] for row in rows)
        result.append({
            "pair_id": pair_id, "row_count": len(rows),
            "all_eligible": statuses.get("ELIGIBLE", 0) == len(rows),
            "mixed_status": len(statuses) > 1,
            "unresolved_pair": statuses.get("UNRESOLVED", 0) > 0,
            "deterministic_failure_pair": statuses.get("INELIGIBLE", 0) > 0,
            "canonical_invalid_pair": any(row["canonical_status"] != "PASS" for row in rows),
            "eligible_positive_count": sum(row["eligible_for_positive_margin"] for row in rows),
            "status_counts": dict(statuses),
        })
    return result


def semantic_sidecar_sha(sidecar: list[dict[str, Any]]) -> str:
    excluded = {"created_at"}
    canonical = [{key: row[key] for key in sorted(row) if key not in excluded} for row in sidecar]
    return canonical_sha(canonical)


def markdown_report(report: dict[str, Any]) -> str:
    identity = report["dataset_identity"]
    positive = report["positive_eligibility"]
    risk = report["scientific_coverage_risk"]
    overlap = report["stage182_overlap_regression"]
    return f"""# Stage185-A controlled-train integrity sidecar report

## Decision

`{report['decision']}`

Authorized next: `{report['stage186_gate']['authorized_next_stage']}`. Loss implementation and training remain unauthorized.

## Stage184-A closure and authoritative identity

Stage184-A authorized deterministic sidecar construction only. The source is `{identity['path']}`, SHA-256 `{identity['sha256']}`, with {identity['rows']} rows, {identity['pairs']} pairs, and {identity['families']} families.

## Complete sidecar and statuses

Complete means one exact sidecar row for every source row, not that every row is clean. `ELIGIBLE` requires every integrity criterion to pass; any deterministic failure yields `INELIGIBLE`; otherwise unresolved evidence yields `UNRESOLVED`. Unresolved rows never enter positive-margin eligibility.

Integrity status and loss eligibility are distinct. Dev and frame-incompatible rows may be integrity-eligible while loss-ineligible. Positive eligibility additionally requires train split, compatible frame label, and passing time/source gates.

## Criterion coverage

```json
{json.dumps(report['criterion_coverage'], indent=2, sort_keys=True)}
```

## Stage182 regression

Overlap rows: {overlap['overlap_count']}; deterministic contaminated: {overlap['deterministic_contaminated_count']}; recovered polarity/grammar: {overlap['recovered_polarity_contamination']}/{overlap['recovered_grammar_contamination']}; passed: {overlap['passed']}.

The Stage182 subset is a regression oracle only, never a whitelist.

## Positive and family coverage

Eligible train-compatible positives: {positive['eligible_count']} / {positive['train_compatible_count']} ({positive['eligible_rate']:.6f}). Eligible families: {risk['eligible_positive_family_count']}; largest family share: {risk['largest_family_share']:.6f}. Unresolved positives: {risk['unresolved_positive_count']}.

Warnings: {', '.join(report['scientific_coverage_risk']['warnings']) or 'none'}.

## Join and hashes

The sidecar is an exact source-order, one-to-one row-ID join. JSONL SHA-256: `{report['sidecar_hashes']['jsonl_sha256']}`; CSV SHA-256: `{report['sidecar_hashes']['csv_sha256']}`; semantic SHA-256: `{report['sidecar_hashes']['semantic_sha256']}`.

## Safety

No source JSONL, generator, trainer, model, loss, checkpoint, or annotation was modified. No model/Torch/checkpoint/forward/training, LLM labeling, grammar model, text classifier, learned probe, threshold fitting, calibration, or external evaluation is used.
"""


def run(args: argparse.Namespace) -> int:
    root = args.repo_root.resolve()
    data_path = resolve(root, args.data)
    stage184_dir = resolve(root, args.stage184a_dir)
    stage182_dir = resolve(root, args.stage182a_dir)
    output_dir = args.output_dir.resolve()
    generator_path = resolve(root, args.generator_source)
    analyzer_path = resolve(root, args.stage182a_analyzer_source)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage184, stage182, report184, report182 = validate_inputs(
        root, data_path, generator_path, analyzer_path, stage184_dir, stage182_dir
    )
    require(sha256(data_path) == AUTHORITATIVE_SHA256, "authoritative dataset SHA mismatch")
    rows = read_jsonl(data_path)
    identity, train, dev, dev_ids = validate_dataset(rows, report184, args.split_seed, args.dev_ratio)
    families = set(identity["intervention_families"])
    contracts = load_contracts(stage184["stage184a_family_contract_matrix.csv"], families)
    module = safe_load_generator(generator_path, root)
    expected, facts = reconstruct(module, rows, families)

    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    builder_path = Path(__file__).resolve()
    report184_path = stage184["stage184a_controlled_train_integrity_mask_spec_report.json"]
    report182_path = stage182["stage182a_controlled_intervention_integrity_report.json"]
    provenance_fields = {
        "source_dataset_path": AUTHORITATIVE_DATA.as_posix(),
        "source_dataset_sha256": sha256(data_path),
        "generator_source_path": str(args.generator_source).replace("\\", "/"),
        "generator_source_sha256": sha256(generator_path),
        "stage182a_report_sha256": sha256(report182_path),
        "stage184a_report_sha256": sha256(report184_path),
        "integrity_builder_sha256": sha256(builder_path),
    }
    sidecar = build_sidecar(
        rows, contracts, expected, facts, dev_ids, provenance_fields,
        str(args.rule_version), created_at,
    )
    require(len(sidecar) == len(rows) == 3600, "sidecar/source row count mismatch")
    require([row["row_id"] for row in sidecar] == [row["id"] for row in rows], "sidecar row order/join mismatch")
    require(len({row["row_id"] for row in sidecar}) == len(sidecar), "duplicate sidecar row_id")
    require(all(row["integrity_status"] in INTEGRITY_ENUM for row in sidecar), "invalid integrity enum")

    unique182 = read_csv(stage182["stage182a_unique_item_integrity.csv"], {"row_id", "integrity_status", "anomaly_codes"})
    overlap_rows, overlap = overlap_regression(sidecar, unique182)
    require(overlap["passed"], f"Stage182 overlap regression failed: {overlap}")
    cov = coverage(sidecar)
    pairs = pair_audit(sidecar)

    jsonl_path = output_dir / "stage185a_controlled_train_integrity_sidecar.jsonl"
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in sidecar:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")
    csv_path = output_dir / "stage185a_controlled_train_integrity_sidecar.csv"
    write_csv(csv_path, SIDECAR_FIELDS, sidecar, sidecar=True)
    require(sum(1 for _ in jsonl_path.open("r", encoding="utf-8")) == 3600, "JSONL output row count mismatch")
    require(sum(1 for _ in csv_path.open("r", encoding="utf-8")) - 1 == 3600, "CSV output row count mismatch")
    hashes = {
        "jsonl_sha256": sha256(jsonl_path), "csv_sha256": sha256(csv_path),
        "semantic_sha256": semantic_sidecar_sha(sidecar),
    }
    require(bool(hashes["semantic_sha256"]), "semantic sidecar SHA unavailable")

    decision = SUCCESS_DECISION if cov["eligible_count"] > 0 else ZERO_DECISION
    next_stage = SUCCESS_NEXT if cov["eligible_count"] > 0 else ZERO_NEXT
    criterion_summary = [
        {"criterion": field, "status": status, "count": cov["criterion"][field].get(status, 0)}
        for field in CRITERION_FIELDS for status in ("PASS", "FAIL", "UNRESOLVED", "NOT_APPLICABLE")
    ]
    integrity_summary = [
        {"dimension": "overall", "group": "all", "integrity_status": status, "count": cov["integrity"].get(status, 0)}
        for status in ("ELIGIBLE", "INELIGIBLE", "UNRESOLVED")
    ]
    for split in ("train", "dev"):
        counts = Counter(row["integrity_status"] for row in sidecar if row["split"] == split)
        integrity_summary.extend({"dimension": "split", "group": split, "integrity_status": status, "count": counts.get(status, 0)} for status in INTEGRITY_ENUM)
    positive_summary = [
        {"metric": "eligible_train_compatible", "value": cov["eligible_count"], "denominator": len(cov["train_compatible"]), "rate": cov["eligible_rate"]},
        {"metric": "ineligible_train_compatible", "value": sum(row["integrity_status"] == "INELIGIBLE" for row in cov["train_compatible"]), "denominator": len(cov["train_compatible"]), "rate": None},
        {"metric": "unresolved_train_compatible", "value": cov["unresolved_positive_count"], "denominator": len(cov["train_compatible"]), "rate": None},
    ]
    reason_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sidecar:
        for code in row["reason_codes"]: reason_groups[code].append(row)
    reason_rows = [{
        "reason_code": code, "row_count": len(items),
        "pair_count": len({row["pair_id"] for row in items}),
        "split_count": len({row["split"] for row in items}),
        "family_count": len({row["intervention_type"] for row in items}),
        "split_distribution": dict(Counter(row["split"] for row in items)),
        "family_distribution": dict(Counter(row["intervention_type"] for row in items)),
    } for code, items in sorted(reason_groups.items())]
    join_rows = [
        {"check": "source_row_count", "observed": len(rows), "expected": 3600, "passed": len(rows) == 3600},
        {"check": "sidecar_row_count", "observed": len(sidecar), "expected": len(rows), "passed": len(sidecar) == len(rows)},
        {"check": "row_id_set_equal", "observed": len({row['row_id'] for row in sidecar}), "expected": len(rows), "passed": {row['row_id'] for row in sidecar} == {row['id'] for row in rows}},
        {"check": "source_order_equal", "observed": True, "expected": True, "passed": [row['row_id'] for row in sidecar] == [row['id'] for row in rows]},
        {"check": "duplicate_sidecar_row_id", "observed": len(sidecar) - len({row['row_id'] for row in sidecar}), "expected": 0, "passed": len({row['row_id'] for row in sidecar}) == len(sidecar)},
    ]
    pair_summary = {
        "all_eligible": sum(row["all_eligible"] for row in pairs),
        "mixed_status": sum(row["mixed_status"] for row in pairs),
        "unresolved_pair": sum(row["unresolved_pair"] for row in pairs),
        "deterministic_failure_pair": sum(row["deterministic_failure_pair"] for row in pairs),
        "canonical_invalid_pair": sum(row["canonical_invalid_pair"] for row in pairs),
    }
    provenance = {
        **provenance_fields, "stage184a_report_path": str(report184_path),
        "stage182a_report_path": str(report182_path),
        "stage184a_report_sha256": sha256(report184_path),
        "stage182a_report_sha256": sha256(report182_path),
        "stage182a_analyzer_path": str(args.stage182a_analyzer_source).replace("\\", "/"),
        "stage182a_analyzer_sha256": sha256(analyzer_path),
        "rule_version": args.rule_version, "created_at": created_at,
        "sidecar_hashes": hashes,
    }
    risk = {
        "eligible_positive_count": cov["eligible_count"],
        "eligible_positive_rate": cov["eligible_rate"],
        "eligible_positive_family_count": cov["eligible_family_count"],
        "largest_family_share": cov["largest_family_share"],
        "families_with_zero_eligible_positives": cov["families_with_zero_eligible"],
        "unresolved_positive_count": cov["unresolved_positive_count"],
        "top_unresolved_reason": cov["top_unresolved_reason"],
        "top_ineligible_reason": cov["top_ineligible_reason"],
        "warnings": cov["warnings"], "threshold_fitted": False,
    }
    gate_rows = [
        {"gate": "sidecar_rows_3600", "passed": len(sidecar) == 3600, "observed": len(sidecar), "required": 3600},
        {"gate": "exact_one_to_one_join", "passed": all(row["passed"] for row in join_rows), "observed": "passed", "required": "passed"},
        {"gate": "all_rows_classified", "passed": sum(cov["integrity"].values()) == 3600, "observed": sum(cov["integrity"].values()), "required": 3600},
        {"gate": "blocked_invariants", "passed": True, "observed": 0, "required": 0},
        {"gate": "stage182_regression", "passed": overlap["passed"], "observed": overlap["passed"], "required": True},
        {"gate": "eligible_positive_count", "passed": cov["eligible_count"] > 0, "observed": cov["eligible_count"], "required": ">0 for Decision1"},
        {"gate": "semantic_sha", "passed": bool(hashes["semantic_sha256"]), "observed": hashes["semantic_sha256"], "required": "nonempty"},
    ]
    report = {
        "stage": STAGE, "decision": decision,
        "scope": {"deterministic_sidecar_generation_only": True, "source_modified": False, "loss_implemented": False, "training": False},
        "input_validation": {"stage184_decision": decision_value(report184), "stage182_decision": decision_value(report182), "required_stage184_artifacts": list(STAGE184_FILES), "required_stage182_artifacts": list(STAGE182_FILES), "status": "passed"},
        "dataset_identity": {"path": AUTHORITATIVE_DATA.as_posix(), "sha256": sha256(data_path), "rows": len(rows), "pairs": identity["pair_count"], "families": identity["intervention_family_count"]},
        "split_topology": {key: identity[key] for key in ("train_pair_count", "dev_pair_count", "train_row_count", "dev_row_count", "train_frame_compatible", "train_frame_incompatible")},
        "generator_provenance": provenance,
        "rule_version": args.rule_version,
        "family_contracts": cov["family_rows"],
        "criterion_coverage": {field: dict(cov["criterion"][field]) for field in CRITERION_FIELDS},
        "integrity_status": dict(cov["integrity"]),
        "positive_eligibility": {"eligible_count": cov["eligible_count"], "train_compatible_count": len(cov["train_compatible"]), "eligible_rate": cov["eligible_rate"], "formula": "ELIGIBLE and train and compatible and time/source PASS"},
        "family_coverage": cov["family_rows"], "pair_coverage": pair_summary,
        "reason_code_coverage": reason_rows,
        "stage182_overlap_regression": overlap,
        "sidecar_contract": {"source_order_preserved": True, "one_to_one_row_id_join": True, "row_count": len(sidecar), "reason_codes_sorted_unique": True, "claim_evidence_duplicated": False},
        "sidecar_hashes": hashes, "scientific_coverage_risk": risk,
        "implementation_readiness": {"sidecar_complete": True, "positive_eligibility_materialized": True, "loss_implementation_authorized": False, "training_authorized": False},
        "stage186_gate": {"authorized_next_stage": next_stage, "loss_implementation_authorized": False, "training_authorized": False, "rows": gate_rows},
        "limitations": ["Template PASS is not general-English quality proof.", "Stage182 overlap is a regression oracle, not a whitelist.", "Coverage warnings are descriptive and do not authorize training."],
        "safety_policy": {"no_jsonl_modification": True, "no_generator_modification": True, "no_model_or_torch": True, "no_checkpoint_or_forward": True, "no_loss_implementation": True, "no_training": True, "no_llm_or_text_classifier": True, "no_threshold_fitting": True, "no_external_evaluation": True},
    }

    write_csv(output_dir / "stage185a_dataset_join_audit.csv", ["check", "observed", "expected", "passed"], join_rows)
    write_csv(output_dir / "stage185a_criterion_status_summary.csv", ["criterion", "status", "count"], criterion_summary)
    write_csv(output_dir / "stage185a_integrity_status_summary.csv", ["dimension", "group", "integrity_status", "count"], integrity_summary)
    write_csv(output_dir / "stage185a_positive_eligibility_summary.csv", ["metric", "value", "denominator", "rate"], positive_summary)
    write_csv(output_dir / "stage185a_family_coverage.csv", ["family", "total", "train", "compatible", "eligible_positive", "ineligible", "unresolved", "eligible_positive_rate", "dominant_reason_codes"], cov["family_rows"])
    write_csv(output_dir / "stage185a_pair_integrity_audit.csv", ["pair_id", "row_count", "all_eligible", "mixed_status", "unresolved_pair", "deterministic_failure_pair", "canonical_invalid_pair", "eligible_positive_count", "status_counts"], pairs)
    write_csv(output_dir / "stage185a_reason_code_counts.csv", ["reason_code", "row_count", "pair_count", "split_count", "family_count", "split_distribution", "family_distribution"], reason_rows)
    write_csv(output_dir / "stage185a_stage182_overlap_regression.csv", ["row_id", "stage182_integrity_status", "stage182_contamination_codes", "stage185_integrity_status", "stage185_contamination_codes", "stage185_eligible_for_positive_margin", "passed"], overlap_rows)
    write_csv(output_dir / "stage185a_eligible_positive_rows.csv", SIDECAR_FIELDS, cov["eligible_rows"], sidecar=True)
    write_csv(output_dir / "stage185a_ineligible_rows.csv", SIDECAR_FIELDS, [row for row in sidecar if row["integrity_status"] == "INELIGIBLE"], sidecar=True)
    write_csv(output_dir / "stage185a_unresolved_rows.csv", SIDECAR_FIELDS, [row for row in sidecar if row["integrity_status"] == "UNRESOLVED"], sidecar=True)
    write_json(output_dir / "stage185a_provenance.json", provenance)
    write_csv(output_dir / "stage185a_stage186_gate.csv", ["gate", "passed", "observed", "required"], gate_rows)
    write_json(output_dir / "stage185a_controlled_train_integrity_sidecar_report.json", report)
    (output_dir / "stage185a_controlled_train_integrity_sidecar_report.md").write_text(markdown_report(report), encoding="utf-8")
    require(set(path.name for path in output_dir.iterdir() if path.name in OUTPUT_FILES) == set(OUTPUT_FILES), "17-file output contract failure")
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except BuildBlocked as exc:
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        blocked = {
            "stage": STAGE, "decision": BLOCKED_DECISION,
            "scope": {"sidecar_built": False, "loss_implemented": False, "training": False},
            "blocked_reason": str(exc),
            "stage186_gate": {"authorized_next_stage": None, "loss_implementation_authorized": False, "training_authorized": False},
            "safety_policy": {"fail_closed": True},
        }
        write_json(output_dir / "stage185a_controlled_train_integrity_sidecar_report.json", blocked)
        (output_dir / "stage185a_controlled_train_integrity_sidecar_report.md").write_text(
            f"# Stage185-A build blocked\n\n`{BLOCKED_DECISION}`\n\nReason: {exc}\n",
            encoding="utf-8",
        )
        write_csv(output_dir / "stage185a_stage186_gate.csv", ["gate", "passed", "observed", "required"], [{"gate": "build_blocked", "passed": False, "observed": str(exc), "required": "no blocked invariant"}])
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
