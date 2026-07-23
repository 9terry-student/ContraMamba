#!/usr/bin/env python3
"""Diagnose exact Stage196-B2-B6P3 action-response safety-gate geometry.

This artifact-only stage joins the frozen P2 composer geometry to P0 safety
targets for evaluation.  It does not train, fit, promote, or integrate a gate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


STAGE = "Stage196-B2-B6P3"
SEEDS = (183, 184, 185)
CANDIDATE_MASKS = ("00100000000000", "01000000000000", "10000000000000")
TARGETS = ("MUST_ALLOW", "MUST_BLOCK", "OPTIONAL")
P2_DECISION = "STAGE196B2B6P2_ACTION_CONDITIONAL_COMPOSER_MARGIN_EXPORT_COMPLETE"
P2_NEXT = "STAGE196B2B6P3_ACTION_RESPONSE_SAFETY_GATE_DIAGNOSTIC"
P0_DECISION = "STAGE196B2B6P0_CURRENT_SAFETY_OBSERVABILITY_INSUFFICIENT"
P0_NEXT = "STAGE196B2B6P1_ADDITIONAL_SAFETY_STATE_OBSERVABILITY_DESIGN"
BLOCKED = "STAGE196B2B6P3_BLOCKED_CONTRACT_FAILURE"
BLOCKED_NEXT = "STAGE196B2B6P3_REPAIR_CONTRACT"
DECISIONS = (
    ("STAGE196B2B6P3_SHARED_NATURAL_BOUNDARY_GATE_IDENTIFIED",
     "STAGE196B2B6P4_SHARED_GATE_MINIMAL_INTERVENTION_DESIGN"),
    ("STAGE196B2B6P3_CANDIDATE_SPECIFIC_NATURAL_BOUNDARY_GATES_IDENTIFIED",
     "STAGE196B2B6P4_CANDIDATE_SPECIFIC_GATE_INTERVENTION_DESIGN"),
    ("STAGE196B2B6P3_POSTHOC_THRESHOLD_SIGNAL_ONLY",
     "STAGE196B2B6P4_THRESHOLD_MECHANISM_DESIGN"),
    ("STAGE196B2B6P3_CROSS_SEED_UNSTABLE_ACTION_RESPONSE_SIGNAL",
     "STAGE196B2B6P4_ACTION_RESPONSE_STABILITY_STATE_DESIGN"),
    ("STAGE196B2B6P3_CURRENT_ACTION_RESPONSE_STATE_INSUFFICIENT",
     "STAGE196B2B7_SELECTOR_INTERVENTION_RETHINK"),
)
OUTPUTS = (
    "stage196b2b6p3_analysis.json",
    "stage196b2b6p3_report.md",
    "stage196b2b6p3_source_closure.csv",
    "stage196b2b6p3_joined_action_response_safety_rows.csv",
    "stage196b2b6p3_feature_dictionary.csv",
    "stage196b2b6p3_natural_boundary_gate_summary.csv",
    "stage196b2b6p3_state_conflict_audit.csv",
    "stage196b2b6p3_continuous_threshold_envelope.csv",
    "stage196b2b6p3_cross_seed_transfer_audit.csv",
    "stage196b2b6p3_decision_gate.csv",
    "stage196b2b6p3_contract.csv",
)
P2_OUTPUTS = (
    "stage196b2b6p2_analysis.json",
    "stage196b2b6p2_report.md",
    "stage196b2b6p2_source_closure.csv",
    "stage196b2b6p2_native_composer_scores.csv",
    "stage196b2b6p2_candidate_action_composer_scores.csv",
    "stage196b2b6p2_action_response_margin_rows.csv",
    "stage196b2b6p2_coverage_and_reproduction_summary.csv",
    "stage196b2b6p2_decision_gate.csv",
    "stage196b2b6p2_contract.csv",
)
P0_OUTPUTS = (
    "stage196b2b6p0_analysis.json",
    "stage196b2b6p0_report.md",
    "stage196b2b6p0_safety_feature_dictionary.csv",
    "stage196b2b6p0_row_safety_targets.csv",
    "stage196b2b6p0_single_state_signature_rows.csv",
    "stage196b2b6p0_single_state_gate_summary.csv",
    "stage196b2b6p0_diagnostic_gate_summary.csv",
    "stage196b2b6p0_gated_policy_audit.csv",
    "stage196b2b6p0_contract.csv",
)

PRIMARY_NUMERIC = (
    "native_margin_support_minus_not_entitled",
    "native_margin_support_minus_refute",
    "native_margin_refute_minus_not_entitled",
    "native_top1_runner_up_margin",
    "counterfactual_margin_support_minus_not_entitled",
    "counterfactual_margin_support_minus_refute",
    "counterfactual_margin_refute_minus_not_entitled",
    "counterfactual_top1_runner_up_margin",
    "delta_support_minus_not_entitled",
    "delta_support_minus_refute",
    "delta_refute_minus_not_entitled",
    "delta_top1_runner_up_margin",
    "delta_score_support",
    "delta_score_not_entitled",
    "delta_score_refute",
)
SECONDARY_SCORES = (
    "native_score_support", "native_score_not_entitled", "native_score_refute",
    "counterfactual_score_support", "counterfactual_score_not_entitled",
    "counterfactual_score_refute",
)
NORMS = ("action_response_l1", "action_response_l2", "action_response_linf")
BOOLEAN_FIELDS = (
    "prediction_changed", "entitlement_branch_changed",
    "polarity_branch_changed", "polarity_direction_preserved",
)
PREDICTION_FIELDS = ("native_prediction", "counterfactual_prediction")
AUTHORIZED_NUMERIC = PRIMARY_NUMERIC + SECONDARY_SCORES + NORMS
AUTHORIZED_FIELDS = AUTHORIZED_NUMERIC + BOOLEAN_FIELDS + PREDICTION_FIELDS
PROVENANCE_FIELDS = (
    "seed", "stable_row_id", "data_identity", "candidate_mask",
    "candidate_action_key",
)
PROHIBITED_FIELDS = (
    "seed", "stable_row_id", "data_identity", "id", "source_row_id",
    "dev_position", "candidate_mask", "candidate_feature_subset_mask",
    "candidate_action_key", "raw_text", "gold_label", "population",
    "primary_case", "transition_role", "safety_target",
    "safety_target_reason", "joint_prediction", "selector_prediction",
    "joint_correct", "selector_correct", "correct_to_incorrect",
    "incorrect_to_correct", "discovery_membership", "signature_support",
)
RAW_SCORE_FIELDS = frozenset(SECONDARY_SCORES)
CONTRADICTORY_PAIRS = frozenset({
    frozenset(("prediction_relation:eq", "prediction_changed:true")),
    frozenset(("prediction_relation:eq", "entitlement_branch_changed:true")),
    frozenset(("prediction_relation:eq", "polarity_branch_changed:true")),
    frozenset(("prediction_relation:ne", "prediction_changed:false")),
    frozenset(("prediction_relation:ne", "polarity_direction_preserved:true")),
    frozenset(("prediction_changed:false", "entitlement_branch_changed:true")),
    frozenset(("prediction_changed:false", "polarity_branch_changed:true")),
    frozenset(("prediction_changed:true", "polarity_direction_preserved:true")),
    frozenset(("entitlement_branch_changed:true", "polarity_branch_changed:true")),
    frozenset(("entitlement_branch_changed:true", "polarity_direction_preserved:true")),
    frozenset(("polarity_branch_changed:true", "polarity_direction_preserved:true")),
    frozenset(("polarity_branch_changed:true", "native_prediction:NOT_ENTITLED")),
    frozenset(("polarity_branch_changed:true", "counterfactual_prediction:NOT_ENTITLED")),
    frozenset(("polarity_direction_preserved:true", "native_prediction:NOT_ENTITLED")),
    frozenset(("polarity_direction_preserved:true", "counterfactual_prediction:NOT_ENTITLED")),
})
SIGNATURE_NUMERIC = PRIMARY_NUMERIC
CLASSES = ("SUPPORT", "NOT_ENTITLED", "REFUTE")

CONTRACT_H = ("scope", "run", "gate", "required", "observed", "passed", "blocking_reason")
SOURCE_H = (
    "stage", "artifact", "path", "required", "loaded", "row_count",
    "columns", "sha256", "purpose",
)
JOIN_H = (
    "candidate_mask", "seed", "stable_row_id", "data_identity", "id",
    "source_row_id", "dev_position", "candidate_action_key", "safety_target",
    "safety_target_reason", "population", "primary_case", "transition_role",
    "native_prediction", "counterfactual_prediction",
) + AUTHORIZED_NUMERIC + BOOLEAN_FIELDS
FEATURE_H = (
    "field", "feature_family", "authorization", "gate_input_authorized",
    "natural_predicate_family", "continuous_threshold_diagnostic",
    "formula_or_semantics", "prohibition_reason",
)
GATE_H = (
    "scope", "candidate_mask", "gate_id", "predicate_count",
    "canonical_formula", "constituent_predicates", "allow_vector_sha256",
    "must_allow_count", "must_allow_allowed", "must_allow_blocked",
    "must_block_count", "must_block_allowed", "must_block_blocked",
    "optional_count", "optional_allowed", "optional_blocked",
    "optional_activation_rate", "candidate_specific_optional_activation",
    "seed_specific_optional_activation", "constrained_error_count", "feasible",
    "inclusion_minimal", "seed183_feasible", "seed184_feasible",
    "seed185_feasible", "all_constrained_seeds_feasible", "rank",
)
CONFLICT_H = (
    "scope", "candidate_mask", "audit_family", "signature_count",
    "signatures_containing_must_allow", "signatures_containing_must_block",
    "conflicting_signatures_containing_both",
    "must_allow_rows_in_conflicting_signatures",
    "must_block_rows_in_conflicting_signatures", "signature_fields",
    "exact_serialized_values_used",
)
THRESHOLD_H = (
    "scope", "candidate_mask", "field", "orientation", "authorization",
    "pooled_feasible", "pooled_interval", "pooled_representative_threshold",
    "seed183_feasible", "seed183_interval", "seed184_feasible",
    "seed184_interval", "seed185_feasible", "seed185_interval",
    "pooled_constrained_error_count", "pooled_must_allow_blocked",
    "pooled_must_block_allowed", "pooled_optional_allowed",
    "pooled_optional_activation_rate", "candidate_specific_optional_activation",
    "seed_specific_optional_activation",
)
TRANSFER_H = (
    "scope", "candidate_mask", "field", "orientation", "authorization",
    "seed184_derived_interval", "seed185_derived_interval",
    "seed184_interval_tested_unchanged_on_seed185",
    "seed185_interval_tested_unchanged_on_seed184",
    "overlapping_nonempty_threshold_interval",
    "bidirectional_transfer_success", "seed183_audit_only",
)
DECISION_H = (
    "order", "decision", "condition", "observed", "reached",
    "recommended_next_stage", "scientific_authorization",
)


class Predicate:
    def __init__(
        self, key: str, formula: str, underlying: str, family: str,
        raw_score: bool, test: Callable[[dict[str, Any]], bool],
    ) -> None:
        self.key = key
        self.formula = formula
        self.underlying = underlying
        self.family = family
        self.raw_score = raw_score
        self.test = test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--stage196b2b6p2-analysis-json", required=True, type=Path)
    parser.add_argument("--stage196b2b6p0-analysis-json", required=True, type=Path)
    parser.add_argument("--current-git-commit", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def cell(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def boolean(value: Any, label: str) -> bool:
    value = cell(value)
    if type(value) is not bool:
        raise ValueError(f"{label}: exact boolean required")
    return value


def integer(value: Any, label: str) -> int:
    value = cell(value)
    if type(value) is not int:
        raise ValueError(f"{label}: exact integer required")
    return value


def number(value: Any, label: str) -> float:
    value = cell(value)
    if type(value) not in (int, float):
        raise ValueError(f"{label}: numeric value required")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label}: finite value required")
    return result


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON object required")
    return value


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return rows, list(reader.fieldnames or ())


def require_columns(columns: Sequence[str], required: Sequence[str], label: str) -> None:
    missing = sorted(set(required) - set(columns))
    if missing:
        raise ValueError(f"{label}: missing columns {missing}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vector_hash(values: Iterable[bool]) -> str:
    return hashlib.sha256(bytes(int(value) for value in values)).hexdigest()


def git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--verify", "HEAD^{commit}"],
        check=False, capture_output=True, text=True, timeout=30,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def gate(
    rows: list[dict[str, Any]], scope: str, run: str, name: str,
    required: Any, observed: Any, passed: bool, reason: str,
    fatal: bool = True,
) -> None:
    rows.append({
        "scope": scope, "run": run, "gate": name, "required": required,
        "observed": observed, "passed": bool(passed),
        "blocking_reason": "" if passed else reason,
    })
    if fatal and not passed:
        raise ValueError(f"{name}: {reason}")


def validate_decision(
    doc: dict[str, Any], expected_decision: str, expected_next: str,
    label: str, gates: list[dict[str, Any]],
) -> None:
    required = {
        "decision": expected_decision,
        "recommended_next_stage": expected_next,
        "blocking_reasons": [],
    }
    observed = {key: doc.get(key) for key in required}
    gate(
        gates, "authority", label, f"{label.lower()}_decision_closure",
        required, observed, observed == required, f"{label} decision closure changed",
    )


def exact_directory_closure(
    directory: Path, names: Sequence[str], label: str,
    gates: list[dict[str, Any]],
) -> None:
    observed = sorted(path.name for path in directory.iterdir() if path.is_file())
    required = sorted(names)
    gate(
        gates, "source", label, f"{label.lower()}_exact_file_closure",
        required, observed, observed == required, f"{label} exact artifact closure changed",
    )


def all_contract_pass(path: Path, label: str) -> tuple[list[dict[str, str]], list[str]]:
    rows, columns = read_csv(path)
    require_columns(columns, CONTRACT_H, f"{label} contract")
    if not rows or any(not boolean(row["passed"], f"{label} contract passed") for row in rows):
        raise ValueError(f"{label} upstream contract is not fully passing")
    return rows, columns


def source_row(stage: str, path: Path, purpose: str, rows: int | str, columns: Sequence[str]) -> dict[str, Any]:
    return {
        "stage": stage, "artifact": path.name, "path": str(path),
        "required": True, "loaded": True, "row_count": rows,
        "columns": list(columns), "sha256": sha256(path), "purpose": purpose,
    }


def p0_identity(row: dict[str, str]) -> str:
    return canonical({
        "id": row["id"], "source_row_id": row["source_row_id"],
        "dev_position": integer(row["dev_position"], "P0 dev_position"),
    })


def valid_serialized_data_identity(value: str) -> bool:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return False
    return (
        isinstance(parsed, dict)
        and set(parsed) == {"id", "source_row_id", "dev_position"}
        and isinstance(parsed["id"], str)
        and isinstance(parsed["source_row_id"], str)
        and type(parsed["dev_position"]) is int
        and canonical(parsed) == value
    )


def make_feature_dictionary() -> list[dict[str, Any]]:
    semantics = {
        "prediction_changed": "native_prediction != counterfactual_prediction",
        "entitlement_branch_changed": "transition between NOT_ENTITLED and {SUPPORT,REFUTE}",
        "polarity_branch_changed": "SUPPORT/REFUTE class changes while both outputs are entitled",
        "polarity_direction_preserved": "SUPPORT/REFUTE class is unchanged while both outputs are entitled",
        "native_prediction": "exact native model-output class state",
        "counterfactual_prediction": "exact counterfactual model-output class state",
    }
    rows: list[dict[str, Any]] = []
    for field in PRIMARY_NUMERIC:
        rows.append({
            "field": field, "feature_family": "PRIMARY_MECHANISTIC",
            "authorization": "PRIMARY_DIAGNOSTIC_GATE_STATE",
            "gate_input_authorized": True,
            "natural_predicate_family": "SIGNED_EXACT_ZERO",
            "continuous_threshold_diagnostic": True,
            "formula_or_semantics": "exact P2 signed score/margin geometry",
            "prohibition_reason": "",
        })
    for field in SECONDARY_SCORES:
        rows.append({
            "field": field, "feature_family": "SECONDARY_EXPLICIT",
            "authorization": "SECONDARY_DIAGNOSTIC_ONLY",
            "gate_input_authorized": True,
            "natural_predicate_family": "SIGNED_EXACT_ZERO",
            "continuous_threshold_diagnostic": True,
            "formula_or_semantics": "raw absolute class-score coordinate; cannot replace signed margins",
            "prohibition_reason": "",
        })
    for field in NORMS:
        rows.append({
            "field": field, "feature_family": "SECONDARY_EXPLICIT",
            "authorization": "SECONDARY_DIAGNOSTIC_ONLY",
            "gate_input_authorized": True,
            "natural_predicate_family": "NORM_EXACT_ZERO",
            "continuous_threshold_diagnostic": True,
            "formula_or_semantics": "action-response norm; cannot replace signed margins",
            "prohibition_reason": "",
        })
    for field in BOOLEAN_FIELDS + PREDICTION_FIELDS:
        rows.append({
            "field": field, "feature_family": "PRIMARY_MECHANISTIC",
            "authorization": "PRIMARY_DIAGNOSTIC_GATE_STATE",
            "gate_input_authorized": True,
            "natural_predicate_family": (
                "BOOLEAN_STATE" if field in BOOLEAN_FIELDS else "PREDICTION_STATE"
            ),
            "continuous_threshold_diagnostic": False,
            "formula_or_semantics": semantics[field], "prohibition_reason": "",
        })
    for field in PROHIBITED_FIELDS:
        rows.append({
            "field": field, "feature_family": "PROVENANCE_OR_EVALUATION_ONLY",
            "authorization": "PROHIBITED_GATE_INPUT",
            "gate_input_authorized": False, "natural_predicate_family": "NONE",
            "continuous_threshold_diagnostic": False, "formula_or_semantics": "",
            "prohibition_reason": (
                "grouping/provenance only" if field in PROVENANCE_FIELDS
                else "target, outcome, correctness, membership, text, or label leakage"
            ),
        })
    return rows


def natural_predicates() -> list[Predicate]:
    predicates: list[Predicate] = []
    for field in PRIMARY_NUMERIC + SECONDARY_SCORES:
        for operator, symbol, test in (
            ("lt", "<", lambda value: value < 0.0),
            ("eq", "==", lambda value: value == 0.0),
            ("gt", ">", lambda value: value > 0.0),
        ):
            predicates.append(Predicate(
                f"{field}:{operator}", f"{field} {symbol} 0", field,
                "SIGNED_EXACT_ZERO", field in RAW_SCORE_FIELDS,
                lambda row, f=field, fn=test: fn(row[f]),
            ))
    for field in NORMS:
        for operator, symbol, test in (
            ("eq", "==", lambda value: value == 0.0),
            ("gt", ">", lambda value: value > 0.0),
        ):
            predicates.append(Predicate(
                f"{field}:{operator}", f"{field} {symbol} 0", field,
                "NORM_EXACT_ZERO", False,
                lambda row, f=field, fn=test: fn(row[f]),
            ))
    for field in BOOLEAN_FIELDS:
        for value in (False, True):
            predicates.append(Predicate(
                f"{field}:{str(value).lower()}",
                f"{field} == {str(value).lower()}", field, "BOOLEAN_STATE", False,
                lambda row, f=field, v=value: row[f] is v,
            ))
    for field in PREDICTION_FIELDS:
        for value in CLASSES:
            predicates.append(Predicate(
                f"{field}:{value}", f"{field} == {value}", field,
                "PREDICTION_STATE", False,
                lambda row, f=field, v=value: row[f] == v,
            ))
    predicates.extend((
        Predicate(
            "prediction_relation:eq",
            "native_prediction == counterfactual_prediction",
            "prediction_relation", "PREDICTION_STATE", False,
            lambda row: row["native_prediction"] == row["counterfactual_prediction"],
        ),
        Predicate(
            "prediction_relation:ne",
            "native_prediction != counterfactual_prediction",
            "prediction_relation", "PREDICTION_STATE", False,
            lambda row: row["native_prediction"] != row["counterfactual_prediction"],
        ),
    ))
    return sorted(predicates, key=lambda item: item.formula)


def valid_pair(left: Predicate, right: Predicate) -> bool:
    return (
        left.key != right.key
        and left.underlying != right.underlying
        and frozenset((left.key, right.key)) not in CONTRADICTORY_PAIRS
        and int(left.raw_score) + int(right.raw_score) <= 1
    )


def count_metrics(rows: Sequence[dict[str, Any]], allowed: Sequence[bool]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for target, prefix in (
        ("MUST_ALLOW", "must_allow"),
        ("MUST_BLOCK", "must_block"),
        ("OPTIONAL", "optional"),
    ):
        indices = [index for index, row in enumerate(rows) if row["safety_target"] == target]
        active = sum(allowed[index] for index in indices)
        metrics[f"{prefix}_count"] = len(indices)
        metrics[f"{prefix}_allowed"] = active
        metrics[f"{prefix}_blocked"] = len(indices) - active
    metrics["constrained_error_count"] = (
        metrics["must_allow_blocked"] + metrics["must_block_allowed"]
    )
    metrics["feasible"] = metrics["constrained_error_count"] == 0
    metrics["optional_activation_rate"] = (
        metrics["optional_allowed"] / metrics["optional_count"]
        if metrics["optional_count"] else 0.0
    )
    return metrics


def seed_feasibility(rows: Sequence[dict[str, Any]], allowed: Sequence[bool]) -> dict[int, bool]:
    result: dict[int, bool] = {}
    for seed in SEEDS:
        indices = [
            index for index, row in enumerate(rows)
            if row["seed"] == seed and row["safety_target"] != "OPTIONAL"
        ]
        result[seed] = all(
            (allowed[index] if rows[index]["safety_target"] == "MUST_ALLOW"
             else not allowed[index])
            for index in indices
        )
    return result


def optional_breakdown(rows: Sequence[dict[str, Any]], allowed: Sequence[bool]) -> tuple[dict[str, int], dict[str, int]]:
    by_candidate: Counter[str] = Counter()
    by_seed: Counter[str] = Counter()
    for row, active in zip(rows, allowed):
        if active and row["safety_target"] == "OPTIONAL":
            by_candidate[row["candidate_mask"]] += 1
            by_seed[str(row["seed"])] += 1
    return (
        {mask: by_candidate[mask] for mask in CANDIDATE_MASKS},
        {str(seed): by_seed[str(seed)] for seed in SEEDS},
    )


def enumerate_gates(
    all_rows: Sequence[dict[str, Any]], predicates: Sequence[Predicate],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    scope_success: dict[str, Any] = {
        "shared": False, "candidate": {}, "any_natural_signal": False,
    }
    scopes = [("SHARED", "", list(all_rows))]
    scopes.extend(
        ("CANDIDATE_SPECIFIC", mask,
         [row for row in all_rows if row["candidate_mask"] == mask])
        for mask in CANDIDATE_MASKS
    )
    for scope, mask, rows in scopes:
        predicate_vectors = {
            predicate.key: bytes(predicate.test(row) for row in rows)
            for predicate in predicates
        }
        constrained_by_seed = {
            seed: [
                index for index, row in enumerate(rows)
                if row["seed"] == seed and row["safety_target"] != "OPTIONAL"
            ]
            for seed in SEEDS
        }

        def passes_any_constrained_seed(allowed: bytes) -> bool:
            return any(
                bool(indices) and all(
                    (allowed[index] if rows[index]["safety_target"] == "MUST_ALLOW"
                     else not allowed[index])
                    for index in indices
                )
                for indices in constrained_by_seed.values()
            )

        single_feasible = {
            predicate.key for predicate in predicates
            if count_metrics(rows, predicate_vectors[predicate.key])["feasible"]
        }
        any_seed_signal = any(
            passes_any_constrained_seed(predicate_vectors[predicate.key])
            for predicate in predicates
        )
        representatives: dict[str, tuple[tuple[Predicate, ...], bytes]] = {}

        def retain(parts: tuple[Predicate, ...], allowed: bytes) -> None:
            nonlocal any_seed_signal
            any_seed_signal |= passes_any_constrained_seed(allowed)
            digest = vector_hash(allowed)
            current = representatives.get(digest)
            formula = " AND ".join(part.formula for part in parts)
            if current is None:
                representatives[digest] = (parts, allowed)
                return
            current_formula = " AND ".join(part.formula for part in current[0])
            if (len(parts), formula) < (len(current[0]), current_formula):
                representatives[digest] = (parts, allowed)

        for predicate in predicates:
            retain((predicate,), predicate_vectors[predicate.key])
        for left, right in combinations(predicates, 2):
            if not valid_pair(left, right):
                continue
            allowed = bytes(
                left_value and right_value
                for left_value, right_value in zip(
                    predicate_vectors[left.key], predicate_vectors[right.key]
                )
            )
            retain((left, right), allowed)
        scope_rows: list[dict[str, Any]] = []
        for digest, (parts, allowed) in representatives.items():
            metrics = count_metrics(rows, allowed)
            seeds = seed_feasibility(rows, allowed)
            by_candidate, by_seed = optional_breakdown(rows, allowed)
            minimal = metrics["feasible"] and (
                len(parts) == 1
                or not any(part.key in single_feasible for part in parts)
            )
            formula = " AND ".join(part.formula for part in parts)
            scope_rows.append({
                "scope": scope, "candidate_mask": mask,
                "gate_id": hashlib.sha256(formula.encode("utf-8")).hexdigest()[:16],
                "predicate_count": len(parts), "canonical_formula": formula,
                "constituent_predicates": [part.key for part in parts],
                "allow_vector_sha256": digest, **metrics,
                "candidate_specific_optional_activation": by_candidate,
                "seed_specific_optional_activation": by_seed,
                "inclusion_minimal": minimal,
                "seed183_feasible": seeds[183], "seed184_feasible": seeds[184],
                "seed185_feasible": seeds[185],
                "all_constrained_seeds_feasible": all(seeds.values()),
                "rank": 0,
            })
        scope_rows.sort(key=lambda row: (
            not row["feasible"], row["predicate_count"],
            row["constrained_error_count"], row["must_block_allowed"],
            row["must_allow_blocked"], row["optional_allowed"],
            row["canonical_formula"],
        ))
        for rank, row in enumerate(scope_rows, 1):
            row["rank"] = rank
        summaries.extend(scope_rows)
        success = any(
            row["feasible"] and row["inclusion_minimal"]
            and row["all_constrained_seeds_feasible"]
            for row in scope_rows
        )
        if scope == "SHARED":
            scope_success["shared"] = success
        else:
            scope_success["candidate"][mask] = success
        scope_success["any_natural_signal"] |= success or any_seed_signal
    return summaries, scope_success


def sign_token(value: float) -> str:
    if value < 0.0:
        return "NEGATIVE"
    if value > 0.0:
        return "POSITIVE"
    return "ZERO"


def conflict_record(
    scope: str, mask: str, family: str, rows: Sequence[dict[str, Any]],
    signature: Callable[[dict[str, Any]], tuple[Any, ...]], fields: Sequence[str],
) -> dict[str, Any]:
    buckets: defaultdict[tuple[Any, ...], Counter[str]] = defaultdict(Counter)
    for row in rows:
        buckets[signature(row)][row["safety_target"]] += 1
    allow_signatures = sum(bucket["MUST_ALLOW"] > 0 for bucket in buckets.values())
    block_signatures = sum(bucket["MUST_BLOCK"] > 0 for bucket in buckets.values())
    conflicts = [
        bucket for bucket in buckets.values()
        if bucket["MUST_ALLOW"] and bucket["MUST_BLOCK"]
    ]
    return {
        "scope": scope, "candidate_mask": mask, "audit_family": family,
        "signature_count": len(buckets),
        "signatures_containing_must_allow": allow_signatures,
        "signatures_containing_must_block": block_signatures,
        "conflicting_signatures_containing_both": len(conflicts),
        "must_allow_rows_in_conflicting_signatures": sum(x["MUST_ALLOW"] for x in conflicts),
        "must_block_rows_in_conflicting_signatures": sum(x["MUST_BLOCK"] for x in conflicts),
        "signature_fields": list(fields), "exact_serialized_values_used": True,
    }


def conflict_audit(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    scopes = [("SHARED", "", list(rows))]
    scopes.extend(
        ("CANDIDATE_SPECIFIC", mask,
         [row for row in rows if row["candidate_mask"] == mask])
        for mask in CANDIDATE_MASKS
    )
    categorical_fields = SIGNATURE_NUMERIC + BOOLEAN_FIELDS + PREDICTION_FIELDS
    for scope, mask, selected in scopes:
        result.append(conflict_record(
            scope, mask, "PRECOMMITTED_CATEGORICAL_STATE", selected,
            lambda row: tuple(sign_token(row[field]) for field in SIGNATURE_NUMERIC)
            + tuple(row[field] for field in BOOLEAN_FIELDS + PREDICTION_FIELDS),
            categorical_fields,
        ))
        result.append(conflict_record(
            scope, mask, "EXACT_AUTHORIZED_NUMERIC_STATE", selected,
            lambda row: tuple(row["_serialized_numeric"][field] for field in AUTHORIZED_NUMERIC),
            AUTHORIZED_NUMERIC,
        ))
    return result


def threshold_interval(
    rows: Sequence[dict[str, Any]], field: str, orientation: str,
) -> dict[str, Any]:
    allow = [row[field] for row in rows if row["safety_target"] == "MUST_ALLOW"]
    block = [row[field] for row in rows if row["safety_target"] == "MUST_BLOCK"]
    if not allow or not block:
        return {"feasible": False, "lower": None, "lower_closed": False,
                "upper": None, "upper_closed": False}
    if orientation == "<=":
        lower, upper = max(allow), min(block)
        feasible = lower < upper
        return {"feasible": feasible, "lower": lower, "lower_closed": True,
                "upper": upper, "upper_closed": False}
    lower, upper = max(block), min(allow)
    feasible = lower < upper
    return {"feasible": feasible, "lower": lower, "lower_closed": False,
            "upper": upper, "upper_closed": True}


def interval_text(interval: dict[str, Any]) -> str:
    if not interval["feasible"]:
        return ""
    return (
        ("[" if interval["lower_closed"] else "(")
        + f"{interval['lower']:.17g},{interval['upper']:.17g}"
        + ("]" if interval["upper_closed"] else ")")
    )


def interval_intersection(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    if not left["feasible"] or not right["feasible"]:
        return {"feasible": False}
    lower = max(left["lower"], right["lower"])
    upper = min(left["upper"], right["upper"])
    lower_closed = all(
        item["lower"] < lower or item["lower_closed"] for item in (left, right)
    )
    upper_closed = all(
        item["upper"] > upper or item["upper_closed"] for item in (left, right)
    )
    feasible = lower < upper or (lower == upper and lower_closed and upper_closed)
    return {
        "feasible": feasible, "lower": lower, "lower_closed": lower_closed,
        "upper": upper, "upper_closed": upper_closed,
    }


def representative(interval: dict[str, Any]) -> float | None:
    if not interval["feasible"]:
        return None
    return interval["lower"] if interval["lower_closed"] else interval["upper"]


def threshold_allowed(value: float, orientation: str, threshold: float) -> bool:
    return value <= threshold if orientation == "<=" else value >= threshold



def best_threshold_metrics(
    rows: Sequence[dict[str, Any]], field: str, orientation: str,
) -> tuple[float, dict[str, Any]]:
    grouped: defaultdict[float, Counter[str]] = defaultdict(Counter)
    totals = Counter(row["safety_target"] for row in rows)
    for row in rows:
        grouped[row[field]][row["safety_target"]] += 1
    active = Counter() if orientation == "<=" else totals.copy()
    ranked: list[tuple[tuple[Any, ...], float, dict[str, Any]]] = []
    for threshold in sorted(grouped):
        group = grouped[threshold]
        if orientation == "<=":
            active.update(group)
        metrics = {
            "must_allow_blocked": totals["MUST_ALLOW"] - active["MUST_ALLOW"],
            "must_block_allowed": active["MUST_BLOCK"],
            "optional_allowed": active["OPTIONAL"],
            "optional_activation_rate": (
                active["OPTIONAL"] / totals["OPTIONAL"]
                if totals["OPTIONAL"] else 0.0
            ),
        }
        metrics["constrained_error_count"] = (
            metrics["must_allow_blocked"] + metrics["must_block_allowed"]
        )
        rank = (
            metrics["constrained_error_count"], metrics["must_block_allowed"],
            metrics["must_allow_blocked"], metrics["optional_allowed"], threshold,
        )
        ranked.append((rank, threshold, metrics))
        if orientation == ">=":
            active.subtract(group)
    _, threshold, metrics = min(ranked)
    return threshold, metrics


def threshold_diagnostics(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, bool]]:
    envelopes: list[dict[str, Any]] = []
    transfers: list[dict[str, Any]] = []
    signal = {"pooled": False, "seed_specific": False, "transferable": False}
    scopes = [("SHARED", "", list(rows))]
    scopes.extend(
        ("CANDIDATE_SPECIFIC", mask,
         [row for row in rows if row["candidate_mask"] == mask])
        for mask in CANDIDATE_MASKS
    )
    for scope, mask, selected in scopes:
        for field in AUTHORIZED_NUMERIC:
            for orientation in ("<=", ">="):
                pooled = threshold_interval(selected, field, orientation)
                by_seed = {
                    seed: threshold_interval(
                        [row for row in selected if row["seed"] == seed],
                        field, orientation,
                    )
                    for seed in SEEDS
                }
                threshold = representative(pooled)
                if threshold is None:
                    threshold, metrics = best_threshold_metrics(
                        selected, field, orientation
                    )
                else:
                    metrics = count_metrics(
                        selected,
                        [threshold_allowed(row[field], orientation, threshold) for row in selected],
                    )
                errors = {
                    key: metrics[key] for key in (
                        "constrained_error_count", "must_allow_blocked",
                        "must_block_allowed", "optional_allowed",
                        "optional_activation_rate",
                    )
                }
                threshold_vector = [
                    threshold_allowed(row[field], orientation, threshold)
                    for row in selected
                ]
                by_candidate, by_seed_optional = optional_breakdown(
                    selected, threshold_vector
                )
                envelopes.append({
                    "scope": scope, "candidate_mask": mask, "field": field,
                    "orientation": orientation,
                    "authorization": "POSTHOC_DIAGNOSTIC_ONLY",
                    "pooled_feasible": pooled["feasible"],
                    "pooled_interval": interval_text(pooled),
                    "pooled_representative_threshold": threshold,
                    "seed183_feasible": by_seed[183]["feasible"],
                    "seed183_interval": interval_text(by_seed[183]),
                    "seed184_feasible": by_seed[184]["feasible"],
                    "seed184_interval": interval_text(by_seed[184]),
                    "seed185_feasible": by_seed[185]["feasible"],
                    "seed185_interval": interval_text(by_seed[185]),
                    "pooled_constrained_error_count": errors["constrained_error_count"],
                    "pooled_must_allow_blocked": errors["must_allow_blocked"],
                    "pooled_must_block_allowed": errors["must_block_allowed"],
                    "pooled_optional_allowed": errors["optional_allowed"],
                    "pooled_optional_activation_rate": errors["optional_activation_rate"],
                    "candidate_specific_optional_activation": by_candidate,
                    "seed_specific_optional_activation": by_seed_optional,
                })
                overlap = interval_intersection(by_seed[184], by_seed[185])
                bidirectional = overlap.get("feasible", False)
                transfers.append({
                    "scope": scope, "candidate_mask": mask, "field": field,
                    "orientation": orientation,
                    "authorization": "POSTHOC_DIAGNOSTIC_ONLY",
                    "seed184_derived_interval": interval_text(by_seed[184]),
                    "seed185_derived_interval": interval_text(by_seed[185]),
                    "seed184_interval_tested_unchanged_on_seed185": bidirectional,
                    "seed185_interval_tested_unchanged_on_seed184": bidirectional,
                    "overlapping_nonempty_threshold_interval": interval_text(overlap),
                    "bidirectional_transfer_success": bidirectional,
                    "seed183_audit_only": interval_text(by_seed[183]),
                })
                signal["pooled"] |= pooled["feasible"]
                signal["seed_specific"] |= any(item["feasible"] for item in by_seed.values())
                signal["transferable"] |= bidirectional
    return envelopes, transfers, signal


def decision_rows(
    shared: bool, candidate_specific: bool, any_natural_signal: bool,
    signal: dict[str, bool],
) -> tuple[str, str, list[dict[str, Any]]]:
    observations = (
        shared,
        (not shared) and candidate_specific,
        (not shared) and (not candidate_specific) and signal["transferable"],
        (not shared) and (not candidate_specific) and (not signal["transferable"])
        and (any_natural_signal or signal["pooled"] or signal["seed_specific"]),
        (not shared) and (not candidate_specific) and (not signal["transferable"])
        and not (any_natural_signal or signal["pooled"] or signal["seed_specific"]),
    )
    conditions = (
        "one identical inclusion-minimal natural gate passes every candidate and constrained seed",
        "no shared gate; every candidate has an inclusion-minimal natural gate passing every constrained seed",
        "no natural gate; a one-dimensional threshold interval transfers bidirectionally across seeds 184 and 185",
        "no natural or transferable threshold gate; pooled or one-seed signal exists",
        "no natural gate and no transferable one-dimensional threshold signal exists",
    )
    reached_index = next(index for index, observed in enumerate(observations) if observed)
    rows = []
    for index, ((decision, next_stage), condition, observed) in enumerate(
        zip(DECISIONS, conditions, observations), 1
    ):
        rows.append({
            "order": index, "decision": decision, "condition": condition,
            "observed": observed, "reached": index - 1 == reached_index,
            "recommended_next_stage": next_stage,
            "scientific_authorization": (
                "POSTHOC_DIAGNOSTIC_ONLY; DIRECT_INTEGRATION_PROHIBITED"
                if index == 3 else "DIAGNOSTIC_ONLY; NO_GATE_PROMOTION_OR_INTEGRATION"
            ),
        })
    return DECISIONS[reached_index][0], DECISIONS[reached_index][1], rows


def analyze(
    ns: argparse.Namespace, gates: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    root = ns.repo_root.resolve()
    p2_path = ns.stage196b2b6p2_analysis_json.resolve()
    p0_path = ns.stage196b2b6p0_analysis_json.resolve()
    output = ns.output_dir.resolve()
    invocation_ok = (
        ns.repo_root.is_absolute() and ns.stage196b2b6p2_analysis_json.is_absolute()
        and ns.stage196b2b6p0_analysis_json.is_absolute()
        and ns.output_dir.is_absolute() and root.is_dir()
        and p2_path.is_file() and p0_path.is_file()
        and p2_path.name == P2_OUTPUTS[0] and p0_path.name == P0_OUTPUTS[0]
        and bool(re.fullmatch(r"[0-9a-f]{40}", ns.current_git_commit))
    )
    gate(
        gates, "invocation", "", "explicit_path_and_commit_closure",
        {"absolute_exact_paths": True, "commit": "40 lowercase hex"},
        {"absolute_exact_paths": invocation_ok, "commit": ns.current_git_commit},
        invocation_ok, "CLI paths or commit format are invalid",
    )
    gate(
        gates, "invocation", "", "output_directory_nonexistence",
        False, output.exists(), not output.exists(), "output directory already exists",
    )
    observed_head = git_head(root)
    gate(
        gates, "source", "", "current_commit_identity",
        ns.current_git_commit, observed_head, observed_head == ns.current_git_commit,
        "repository HEAD differs from supplied commit",
    )

    p2 = read_json(p2_path)
    p0 = read_json(p0_path)
    validate_decision(p2, P2_DECISION, P2_NEXT, "P2", gates)
    validate_decision(p0, P0_DECISION, P0_NEXT, "P0", gates)
    exact_directory_closure(p2_path.parent, P2_OUTPUTS, "P2", gates)
    exact_directory_closure(p0_path.parent, P0_OUTPUTS, "P0", gates)

    p2_paths = {name: p2_path.parent / name for name in P2_OUTPUTS}
    p0_paths = {name: p0_path.parent / name for name in P0_OUTPUTS}
    native, native_h = read_csv(p2_paths[P2_OUTPUTS[3]])
    candidate, candidate_h = read_csv(p2_paths[P2_OUTPUTS[4]])
    response, response_h = read_csv(p2_paths[P2_OUTPUTS[5]])
    reproduction, reproduction_h = read_csv(p2_paths[P2_OUTPUTS[6]])
    p2_contract, p2_contract_h = all_contract_pass(p2_paths[P2_OUTPUTS[8]], "P2")
    targets, targets_h = read_csv(p0_paths[P0_OUTPUTS[3]])
    p0_contract, p0_contract_h = all_contract_pass(p0_paths[P0_OUTPUTS[8]], "P0")

    require_columns(native_h, (
        "seed", "stable_row_id", "data_identity", "native_prediction",
        "native_top1_runner_up_margin", "native_score_support",
        "native_score_not_entitled", "native_score_refute",
        "native_margin_support_minus_not_entitled",
        "native_margin_support_minus_refute",
        "native_margin_refute_minus_not_entitled",
    ), "P2 native")
    require_columns(candidate_h, (
        "seed", "stable_row_id", "data_identity", "candidate_mask",
        "candidate_action_key", "native_prediction", "counterfactual_prediction",
        "counterfactual_top1_runner_up_margin", "counterfactual_score_support",
        "counterfactual_score_not_entitled", "counterfactual_score_refute",
        "counterfactual_margin_support_minus_not_entitled",
        "counterfactual_margin_support_minus_refute",
        "counterfactual_margin_refute_minus_not_entitled",
    ), "P2 candidate")
    require_columns(response_h, (
        "seed", "stable_row_id", "data_identity", "candidate_mask",
        "candidate_action_key", *PRIMARY_NUMERIC[8:], *BOOLEAN_FIELDS, *NORMS,
    ), "P2 response")
    require_columns(targets_h, (
        "candidate_feature_subset_mask", "seed", "stable_row_id", "id",
        "source_row_id", "dev_position", "population", "primary_case",
        "transition_role", "safety_target", "safety_target_reason",
    ), "P0 targets")

    population = p2.get("population_closure", {})
    reproduction_doc = p2.get("prediction_reproduction", {})
    p2_observed = {
        "native_rows": len(native), "candidate_action_rows": len(candidate),
        "native_prediction_disagreements":
            reproduction_doc.get("native_prediction_disagreements"),
        "counterfactual_prediction_disagreements":
            reproduction_doc.get("counterfactual_prediction_disagreements"),
        "categorical_response_disagreements":
            reproduction_doc.get("categorical_response_disagreements"),
    }
    p2_required = {
        "native_rows": 2160, "candidate_action_rows": 6480,
        "native_prediction_disagreements": 0,
        "counterfactual_prediction_disagreements": 0,
        "categorical_response_disagreements": 0,
    }
    gate(
        gates, "authority", "P2", "p2_native_row_count",
        2160, p2_observed["native_rows"], p2_observed["native_rows"] == 2160,
        "P2 native row count changed",
    )
    gate(
        gates, "authority", "P2", "p2_candidate_action_row_count",
        6480, p2_observed["candidate_action_rows"],
        p2_observed["candidate_action_rows"] == 6480,
        "P2 candidate-action row count changed",
    )
    zero_reproduction = {
        key: p2_observed[key] for key in (
            "native_prediction_disagreements",
            "counterfactual_prediction_disagreements",
            "categorical_response_disagreements",
        )
    }
    gate(
        gates, "authority", "P2", "p2_zero_reproduction_disagreements",
        {key: 0 for key in zero_reproduction}, zero_reproduction,
        all(value == 0 for value in zero_reproduction.values()),
        "P2 prediction reproduction is no longer exact",
    )
    controlled = p2.get("controlled_row_cross_check", {})
    controlled_required = {
        "matched_controlled_rows": 48,
        "score_coordinate_disagreements": 0,
        "margin_disagreements": 0,
        "prediction_disagreements": 0,
        "maximum_absolute_numeric_difference": 0.0,
    }
    controlled_observed = {key: controlled.get(key) for key in controlled_required}
    gate(
        gates, "authority", "P2", "p2_controlled_exact_reproduction_closure",
        controlled_required, controlled_observed,
        controlled_observed == controlled_required,
        "P2 controlled-row exact reproduction closure changed",
    )
    gate(
        gates, "authority", "P2", "p2_population_and_zero_reproduction_closure",
        p2_required, p2_observed,
        p2_observed == p2_required
        and population.get("native_rows") == 2160
        and population.get("candidate_action_rows") == 6480,
        "P2 row counts or exact reproduction closure changed",
    )
    summary_map = {
        row["metric"]: integer(row["observed"], f"P2 summary {row['metric']}")
        for row in reproduction
        if row.get("metric") in {
            "native rows", "candidate-action rows",
            "native prediction disagreements",
            "counterfactual prediction disagreements",
            "categorical response disagreements",
        }
    }
    gate(
        gates, "authority", "P2", "p2_summary_reproduction_closure",
        {
            "native rows": 2160, "candidate-action rows": 6480,
            "native prediction disagreements": 0,
            "counterfactual prediction disagreements": 0,
            "categorical response disagreements": 0,
        },
        summary_map,
        summary_map == {
            "native rows": 2160, "candidate-action rows": 6480,
            "native prediction disagreements": 0,
            "counterfactual prediction disagreements": 0,
            "categorical response disagreements": 0,
        },
        "P2 coverage/reproduction summary changed",
    )

    mask_sets = {
        "P2 candidate": sorted({row["candidate_mask"] for row in candidate}),
        "P2 response": sorted({row["candidate_mask"] for row in response}),
        "P0 target": sorted({row["candidate_feature_subset_mask"] for row in targets}),
    }
    seed_sets = {
        "P2 native": sorted({integer(row["seed"], "P2 native seed") for row in native}),
        "P2 candidate": sorted({integer(row["seed"], "P2 candidate seed") for row in candidate}),
        "P2 response": sorted({integer(row["seed"], "P2 response seed") for row in response}),
        "P0 target": sorted({integer(row["seed"], "P0 target seed") for row in targets}),
    }
    gate(
        gates, "population", "", "exact_candidate_masks",
        {name: list(CANDIDATE_MASKS) for name in mask_sets}, mask_sets,
        all(value == list(CANDIDATE_MASKS) for value in mask_sets.values()),
        "candidate mask closure changed",
    )
    gate(
        gates, "population", "", "exact_seeds",
        {name: list(SEEDS) for name in seed_sets}, seed_sets,
        all(value == list(SEEDS) for value in seed_sets.values()),
        "seed closure changed",
    )
    target_counts = Counter(row["safety_target"] for row in targets)
    required_counts = Counter({"MUST_ALLOW": 21, "MUST_BLOCK": 171, "OPTIONAL": 6288})
    gate(
        gates, "authority", "P0", "p0_exact_target_counts",
        {"rows": 6480, "counts": dict(required_counts)},
        {"rows": len(targets), "counts": dict(target_counts)},
        len(targets) == 6480 and target_counts == required_counts,
        "P0 target closure changed",
    )

    native_index: dict[tuple[int, str], dict[str, str]] = {}
    for row in native:
        key = (integer(row["seed"], "native seed"), row["data_identity"])
        if key in native_index:
            raise ValueError("duplicate P2 native key")
        native_index[key] = row
    candidate_index: dict[tuple[str, int, str, str], dict[str, str]] = {}
    response_index: dict[tuple[str, int, str, str], dict[str, str]] = {}
    for source, index, label in (
        (candidate, candidate_index, "candidate"),
        (response, response_index, "response"),
    ):
        for row in source:
            key = (
                row["candidate_mask"], integer(row["seed"], f"{label} seed"),
                row["stable_row_id"], row["data_identity"],
            )
            if key in index:
                raise ValueError(f"duplicate P2 {label} key")
            index[key] = row
    p0_index: dict[tuple[str, int, str, str], dict[str, str]] = {}
    identity_disagreements = sum(
        not valid_serialized_data_identity(row["data_identity"])
        for row in candidate
    )
    component_disagreements = Counter()
    for row in targets:
        identity = p0_identity(row)
        key = (
            row["candidate_feature_subset_mask"],
            integer(row["seed"], "P0 seed"), row["stable_row_id"], identity,
        )
        if key in p0_index:
            raise ValueError("duplicate P0 semantic target key")
        p0_index[key] = row
    p0_keys, p2_keys = set(p0_index), set(candidate_index)
    joined_keys = sorted(p0_keys & p2_keys)
    join_observed = {
        "p0_unique_target_keys": len(p0_keys),
        "p2_unique_candidate_action_keys": len(p2_keys),
        "joined_rows": len(joined_keys),
        "unmatched_p0_rows": len(p0_keys - p2_keys),
        "unmatched_p2_rows": len(p2_keys - p0_keys),
        "duplicate_keys": (len(targets) - len(p0_keys)) + (len(candidate) - len(p2_keys)),
        "identity_disagreements": identity_disagreements,
    }
    join_required = {
        "p0_unique_target_keys": 6480, "p2_unique_candidate_action_keys": 6480,
        "joined_rows": 6480, "unmatched_p0_rows": 0, "unmatched_p2_rows": 0,
        "duplicate_keys": 0, "identity_disagreements": 0,
    }
    gate(
        gates, "join", "", "p0_unique_key_closure",
        6480, join_observed["p0_unique_target_keys"],
        join_observed["p0_unique_target_keys"] == 6480,
        "P0 unique target-key closure failed",
    )
    gate(
        gates, "join", "", "p2_unique_key_closure",
        6480, join_observed["p2_unique_candidate_action_keys"],
        join_observed["p2_unique_candidate_action_keys"] == 6480,
        "P2 unique candidate-action-key closure failed",
    )
    gate(
        gates, "join", "", "exact_6480_row_join",
        6480, join_observed["joined_rows"],
        join_observed["joined_rows"] == 6480,
        "P0/P2 join does not contain exactly 6,480 rows",
    )
    gate(
        gates, "join", "", "zero_unmatched_rows",
        {"P0": 0, "P2": 0},
        {"P0": join_observed["unmatched_p0_rows"],
         "P2": join_observed["unmatched_p2_rows"]},
        join_observed["unmatched_p0_rows"] == 0
        and join_observed["unmatched_p2_rows"] == 0,
        "P0 or P2 has unmatched semantic keys",
    )
    gate(
        gates, "join", "", "zero_duplicate_rows",
        0, join_observed["duplicate_keys"], join_observed["duplicate_keys"] == 0,
        "duplicate P0/P2 semantic keys exist",
    )
    gate(
        gates, "join", "", "zero_identity_disagreements",
        0, join_observed["identity_disagreements"],
        join_observed["identity_disagreements"] == 0,
        "P2 serialized data identity is invalid or disagrees",
    )
    gate(
        gates, "join", "", "exact_semantic_join_closure",
        join_required, join_observed, join_observed == join_required,
        "P0/P2 exact semantic join failed",
    )
    for position, component in enumerate(
        ("candidate_mask", "seed", "stable_row_id", "data_identity")
    ):
        p0_components = Counter(key[position] for key in p0_keys)
        p2_components = Counter(key[position] for key in p2_keys)
        component_disagreements[component] = (
            sum((p0_components - p2_components).values())
            + sum((p2_components - p0_components).values())
        )
        gate(
            gates, "join", component, f"join_component_{component}_closure",
            0, component_disagreements[component],
            component_disagreements[component] == 0,
            f"semantic join component {component} disagrees",
        )
    gate(
        gates, "join", "", "four_join_components_independently_validated",
        {name: 0 for name in component_disagreements},
        dict(component_disagreements),
        not any(component_disagreements.values()),
        "one or more semantic join components disagree",
    )
    if set(response_index) != p2_keys:
        raise ValueError("P2 response keys do not equal P2 candidate keys")

    joined: list[dict[str, Any]] = []
    for key in joined_keys:
        target = p0_index[key]
        candidate_row = candidate_index[key]
        response_row = response_index[key]
        native_row = native_index[(key[1], key[3])]
        if (
            candidate_row["candidate_action_key"] != response_row["candidate_action_key"]
            or candidate_row["native_prediction"] != native_row["native_prediction"]
        ):
            raise ValueError("P2 internal identity or prediction disagreement")
        row: dict[str, Any] = {
            "candidate_mask": key[0], "seed": key[1], "stable_row_id": key[2],
            "data_identity": key[3], "id": target["id"],
            "source_row_id": target["source_row_id"],
            "dev_position": integer(target["dev_position"], "joined dev_position"),
            "candidate_action_key": candidate_row["candidate_action_key"],
            "safety_target": target["safety_target"],
            "safety_target_reason": target["safety_target_reason"],
            "population": target["population"],
            "primary_case": boolean(target["primary_case"], "primary_case"),
            "transition_role": target["transition_role"],
            "native_prediction": native_row["native_prediction"],
            "counterfactual_prediction": candidate_row["counterfactual_prediction"],
            "_serialized_numeric": {},
        }
        for field in AUTHORIZED_NUMERIC:
            source = (
                native_row if field.startswith("native_")
                else candidate_row if field.startswith("counterfactual_")
                else response_row
            )
            row["_serialized_numeric"][field] = source[field]
            row[field] = number(source[field], field)
        for field in BOOLEAN_FIELDS:
            row[field] = boolean(response_row[field], field)
        joined.append(row)

    finite = all(
        math.isfinite(row[field]) for row in joined for field in AUTHORIZED_NUMERIC
    )
    gate(
        gates, "features", "", "finite_numeric_fields",
        True, finite, finite, "authorized numeric state contains NaN or infinity",
    )
    margin_ok = all(
        row["native_margin_support_minus_not_entitled"]
        == row["native_score_support"] - row["native_score_not_entitled"]
        and row["native_margin_support_minus_refute"]
        == row["native_score_support"] - row["native_score_refute"]
        and row["native_margin_refute_minus_not_entitled"]
        == row["native_score_refute"] - row["native_score_not_entitled"]
        and row["counterfactual_margin_support_minus_not_entitled"]
        == row["counterfactual_score_support"] - row["counterfactual_score_not_entitled"]
        and row["counterfactual_margin_support_minus_refute"]
        == row["counterfactual_score_support"] - row["counterfactual_score_refute"]
        and row["counterfactual_margin_refute_minus_not_entitled"]
        == row["counterfactual_score_refute"] - row["counterfactual_score_not_entitled"]
        for row in joined
    )
    delta_ok = all(
        row["delta_score_support"]
        == row["counterfactual_score_support"] - row["native_score_support"]
        and row["delta_score_not_entitled"]
        == row["counterfactual_score_not_entitled"] - row["native_score_not_entitled"]
        and row["delta_score_refute"]
        == row["counterfactual_score_refute"] - row["native_score_refute"]
        and row["delta_support_minus_not_entitled"]
        == row["counterfactual_margin_support_minus_not_entitled"]
        - row["native_margin_support_minus_not_entitled"]
        and row["delta_support_minus_refute"]
        == row["counterfactual_margin_support_minus_refute"]
        - row["native_margin_support_minus_refute"]
        and row["delta_refute_minus_not_entitled"]
        == row["counterfactual_margin_refute_minus_not_entitled"]
        - row["native_margin_refute_minus_not_entitled"]
        for row in joined
    )
    gate(
        gates, "numeric", "", "margin_arithmetic_spot_closure",
        {"disagreements": 0}, {"disagreements": 0 if margin_ok else 1},
        margin_ok, "exact margin arithmetic failed",
    )
    gate(
        gates, "numeric", "", "delta_arithmetic_spot_closure",
        {"disagreements": 0}, {"disagreements": 0 if delta_ok else 1},
        delta_ok, "exact delta arithmetic failed",
    )

    dictionary = make_feature_dictionary()
    authorized_names = {
        row["field"] for row in dictionary if row["gate_input_authorized"]
    }
    prohibited_authorized = sorted(set(PROHIBITED_FIELDS) & authorized_names)
    gate(
        gates, "features", "", "authorized_feature_dictionary_closure",
        sorted(AUTHORIZED_FIELDS), sorted(authorized_names),
        authorized_names == set(AUTHORIZED_FIELDS),
        "feature dictionary authorization changed",
    )
    gate(
        gates, "features", "", "prohibited_feature_exclusion",
        [], prohibited_authorized, not prohibited_authorized,
        "prohibited field entered gate authorization",
    )

    predicates = natural_predicates()
    expected_predicates = len(PRIMARY_NUMERIC + SECONDARY_SCORES) * 3 + len(NORMS) * 2 + len(BOOLEAN_FIELDS) * 2 + 8
    gate(
        gates, "natural", "", "natural_predicate_precommit_closure",
        {
            "exact_zero": True, "epsilon": None,
            "signed_predicates_per_field": 3, "norm_predicates_per_field": 2,
            "boolean_predicates_per_field": 2, "prediction_predicates": 8,
            "predicate_count": expected_predicates,
        },
        {
            "exact_zero": True, "epsilon": None,
            "predicate_count": len(predicates),
        },
        len(predicates) == expected_predicates,
        "natural predicate language is incomplete",
    )
    natural_summary, natural_success = enumerate_gates(joined, predicates)
    feasibility_ok = all(
        row["constrained_error_count"]
        == row["must_allow_blocked"] + row["must_block_allowed"]
        and row["feasible"] == (row["constrained_error_count"] == 0)
        and row["must_allow_allowed"] + row["must_allow_blocked"]
        == row["must_allow_count"]
        and row["must_block_allowed"] + row["must_block_blocked"]
        == row["must_block_count"]
        and row["optional_allowed"] + row["optional_blocked"]
        == row["optional_count"]
        for row in natural_summary
    )
    gate(
        gates, "natural", "", "natural_gate_feasibility_definition_closure",
        "must_allow_blocked + must_block_allowed == 0",
        "must_allow_blocked + must_block_allowed == 0" if feasibility_ok else "INVALID",
        feasibility_ok, "natural gate feasibility accounting changed",
    )
    scope_semantics_ok = all(
        (row["scope"] == "SHARED" and row["candidate_mask"] == "")
        or (row["scope"] == "CANDIDATE_SPECIFIC"
            and row["candidate_mask"] in CANDIDATE_MASKS)
        for row in natural_summary
    )
    gate(
        gates, "natural", "", "shared_candidate_specific_scope_closure",
        True, scope_semantics_ok, scope_semantics_ok,
        "shared and candidate-specific gate scopes were pooled",
    )
    canonical_ok = len({
        (row["scope"], row["candidate_mask"], row["allow_vector_sha256"])
        for row in natural_summary
    }) == len(natural_summary)
    gate(
        gates, "natural", "", "natural_gate_canonicalization_closure",
        True, canonical_ok, canonical_ok,
        "logically equivalent allow vectors were not deduplicated",
    )
    predicate_by_key = {predicate.key: predicate for predicate in predicates}
    pair_language_ok = all(
        row["predicate_count"] == 1
        or valid_pair(
            predicate_by_key[row["constituent_predicates"][0]],
            predicate_by_key[row["constituent_predicates"][1]],
        )
        for row in natural_summary
    )
    complexity_ok = all(row["predicate_count"] in (1, 2) for row in natural_summary)
    gate(
        gates, "natural", "", "two_predicate_maximum_complexity_closure",
        2, max((row["predicate_count"] for row in natural_summary), default=0),
        complexity_ok, "natural gate exceeds two predicates",
    )
    gate(
        gates, "natural", "", "two_predicate_language_restriction_closure",
        {
            "contradictory": False, "duplicate": False,
            "same_underlying_scalar": False, "raw_score_predicates_maximum": 1,
        },
        {
            "valid_representatives": pair_language_ok,
            "precommitted_valid_pair_count": sum(
                valid_pair(left, right)
                for left, right in combinations(predicates, 2)
            ),
        },
        pair_language_ok,
        "two-predicate language restrictions were violated",
    )

    conflicts = conflict_audit(joined)
    gate(
        gates, "audit", "", "state_conflict_audit_closure",
        {"scopes": 4, "families_per_scope": 2, "rows": 8},
        {
            "scopes": len({(row["scope"], row["candidate_mask"]) for row in conflicts}),
            "families": len(conflicts), "rows": len(conflicts),
        },
        len(conflicts) == 8, "state-conflict audit is incomplete",
    )
    envelopes, transfers, threshold_signal = threshold_diagnostics(joined)
    threshold_only = all(
        row["authorization"] == "POSTHOC_DIAGNOSTIC_ONLY" for row in envelopes + transfers
    )
    gate(
        gates, "threshold", "", "threshold_diagnostic_only_closure",
        "POSTHOC_DIAGNOSTIC_ONLY", "POSTHOC_DIAGNOSTIC_ONLY" if threshold_only else "INVALID",
        threshold_only, "post-hoc threshold was marked integration-authorized",
    )
    transfer_complete = len(transfers) == 4 * len(AUTHORIZED_NUMERIC) * 2
    gate(
        gates, "threshold", "", "cross_seed_transfer_closure",
        4 * len(AUTHORIZED_NUMERIC) * 2, len(transfers),
        transfer_complete, "bidirectional threshold transfer audit is incomplete",
    )

    shared_success = natural_success["shared"]
    candidate_success = all(
        natural_success["candidate"].get(mask, False) for mask in CANDIDATE_MASKS
    )
    decision, next_stage, decisions = decision_rows(
        shared_success, candidate_success,
        natural_success["any_natural_signal"], threshold_signal,
    )
    gate(
        gates, "decision", "", "decision_hierarchy_reachability",
        {"exactly_one_reached": True, "decision_in_hierarchy": True},
        {
            "reached": sum(row["reached"] for row in decisions),
            "decision": decision,
        },
        sum(row["reached"] for row in decisions) == 1
        and decision in {item[0] for item in DECISIONS},
        "ordered decision hierarchy is unreachable or ambiguous",
    )
    gate(
        gates, "output", "", "exact_eleven_file_output_closure",
        sorted(OUTPUTS), sorted(OUTPUTS),
        len(OUTPUTS) == len(set(OUTPUTS)) == 11,
        "output declaration is not exactly eleven files",
    )

    source = [
        source_row("P2", p2_path, "frozen P2 analysis authority", "", ()),
        source_row("P2", p2_paths[P2_OUTPUTS[3]], "native composer scores", len(native), native_h),
        source_row("P2", p2_paths[P2_OUTPUTS[4]], "candidate-action composer scores", len(candidate), candidate_h),
        source_row("P2", p2_paths[P2_OUTPUTS[5]], "action-response margins and flags", len(response), response_h),
        source_row("P2", p2_paths[P2_OUTPUTS[6]], "coverage and reproduction summary", len(reproduction), reproduction_h),
        source_row("P2", p2_paths[P2_OUTPUTS[8]], "fully passing P2 contract", len(p2_contract), p2_contract_h),
        source_row("P0", p0_path, "frozen P0 analysis authority", "", ()),
        source_row("P0", p0_paths[P0_OUTPUTS[3]], "evaluation-only safety targets", len(targets), targets_h),
        source_row("P0", p0_paths[P0_OUTPUTS[8]], "fully passing P0 contract", len(p0_contract), p0_contract_h),
    ]
    analysis = {
        "stage": STAGE, "decision": decision,
        "recommended_next_stage": next_stage, "blocking_reasons": [],
        "current_git_commit": ns.current_git_commit,
        "p2_authority": {
            "decision": P2_DECISION, "exact_composer_reproduction": True,
            "native_rows": 2160, "candidate_action_rows": 6480,
            "prediction_disagreements": 0,
        },
        "p0_authority": {
            "decision": P0_DECISION, "safety_targets_are_evaluation_only": True,
            "target_counts": dict(required_counts),
        },
        "join_closure": join_observed,
        "join_key": [
            "candidate_feature_subset_mask == candidate_mask", "seed == seed",
            "stable_row_id == stable_row_id",
            "canonical(id,source_row_id,dev_position) == data_identity",
        ],
        "feature_authorization": {
            "primary": list(PRIMARY_NUMERIC + BOOLEAN_FIELDS + PREDICTION_FIELDS),
            "secondary": list(SECONDARY_SCORES + NORMS),
            "provenance_and_evaluation_only": list(PROHIBITED_FIELDS),
        },
        "natural_search": {
            "predicate_count": len(predicates),
            "valid_two_predicate_candidate_count": sum(
                valid_pair(left, right)
                for left, right in combinations(predicates, 2)
            ),
            "maximum_conjunction_size": 2,
            "disjunctions_evaluated": False, "arbitrary_epsilon_used": False,
            "shared_gate_feasible": shared_success,
            "candidate_specific_gate_feasible": natural_success["candidate"],
        },
        "feasibility": {
            "definition": "must_allow_blocked + must_block_allowed == 0",
            "optional_rows_define_feasibility": False,
        },
        "state_conflict_audit": conflicts,
        "threshold_signal": {
            **threshold_signal, "authorization": "POSTHOC_DIAGNOSTIC_ONLY",
            "direct_integration_authorized": False,
        },
        "decision_hierarchy": decisions, "exact_outputs": list(OUTPUTS),
        "scope": {
            "diagnostic_only": True, "model_or_checkpoint_loaded": False,
            "training_or_classifier_fitting_performed": False,
            "gate_promoted_or_integrated": False, "external_evaluation_run": False,
            "deployability_claimed": False,
        },
    }
    return analysis, {
        "source": source, "joined": joined, "dictionary": dictionary,
        "natural": natural_summary, "conflicts": conflicts,
        "thresholds": envelopes, "transfers": transfers, "decisions": decisions,
    }


def report(analysis: dict[str, Any]) -> str:
    blocked = analysis.get("blocking_reasons", [])
    return f"""# Stage196-B2-B6P3 Action-Response Safety-Gate Diagnostic

## Decision

`{analysis["decision"]}`

Recommended next stage: `{analysis["recommended_next_stage"]}`.

Blocking reasons: `{canonical(blocked)}`.

## Scientific scope

P2 score geometry exactly reproduces the existing composer. P3 is the first
stage joining that exact, gold-independent geometry to P0 safety targets.
Those safety targets are evaluation-only and never enter a gate predicate.
Seed, row identity, data identity, candidate mask, and candidate action key are
grouping or provenance only, not gate features.

Natural-boundary gates are explicitly distinguished from post-hoc continuous
thresholds. Natural rules use exact-zero signs, exact boolean states, model
prediction states, and conjunctions of at most two predicates. No disjunction
or conjunction larger than two is evaluated.

## Feasibility and scope semantics

Allowed means the predicate is true; blocked means it is false. Feasibility
requires every MUST_ALLOW row to be allowed and every MUST_BLOCK row to be
blocked. OPTIONAL rows never define feasibility and their activation is only a
descriptive audit.

A shared gate is one identical predicate for all candidate masks. A
candidate-specific result requires an independently feasible rule for every
frozen candidate and is never reported as shared. Every constrained seed is
audited separately.

## Exact-state conflict audit

Precommitted signatures combine signed margin/delta state, transition flags,
native prediction, and counterfactual prediction. Exact full numeric-vector
duplicates use unrounded serialized P2 values. Conflicts identify authorized
states containing both MUST_ALLOW and MUST_BLOCK targets.

## Continuous thresholds

One-dimensional `field <= threshold` and `field >= threshold` envelopes are
diagnostic-only. Reported intervals use exact observed endpoints and contain the strict midpoint
boundaries between adjacent distinct values. For non-strict one-dimensional
rules, an observed endpoint is the canonical representative of the same row
partition as its adjacent strict midpoint. No multi-feature threshold,
conjunction, weighted score, or learned classifier is evaluated. Seed-184 intervals are tested
unchanged on seed 185 and vice versa. Bidirectional success requires the same
orientation and a nonempty overlapping interval. Seed 183 is contrast-only
and is never used to tune a threshold.

Any threshold selected after observing safety labels is
`POSTHOC_DIAGNOSTIC_ONLY` and cannot be directly integrated.

## Scientific limits

No gate is promoted, applied to production, or integrated. No training,
external/OOD improvement, deployability, formal causal mediation, or
unfrozen-Mamba validity is claimed. A negative diagnostic result is
scientifically valid when all contracts pass. The next stage follows only from
the precommitted ordered decision hierarchy.
"""


def csv_value(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite CSV value")
        return f"{value:.17g}"
    if isinstance(value, (dict, list, tuple)):
        return canonical(value)
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return ""
    return value


def render_csv(header: Sequence[str], rows: Iterable[dict[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=list(header), extrasaction="raise", lineterminator="\n"
    )
    writer.writeheader()
    for raw in rows:
        row = {key: raw.get(key, "") for key in header}
        writer.writerow({key: csv_value(value) for key, value in row.items()})
    return buffer.getvalue()


def blocked_payload(
    ns: argparse.Namespace, error: BaseException,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    reason = f"{type(error).__name__}: {error}"
    decisions = [{
        "order": 0, "decision": BLOCKED,
        "condition": "source, join, schema, numerical, or contract failure",
        "observed": True, "reached": True,
        "recommended_next_stage": BLOCKED_NEXT,
        "scientific_authorization": "NONE",
    }]
    analysis = {
        "stage": STAGE, "decision": BLOCKED,
        "recommended_next_stage": BLOCKED_NEXT,
        "blocking_reasons": [reason],
        "current_git_commit": ns.current_git_commit,
        "p2_authority": {}, "p0_authority": {}, "join_closure": {},
        "join_key": [], "feature_authorization": {},
        "natural_search": {}, "feasibility": {},
        "state_conflict_audit": [], "threshold_signal": {},
        "decision_hierarchy": decisions, "exact_outputs": list(OUTPUTS),
        "scope": {"diagnostic_only": True, "gate_promoted_or_integrated": False},
    }
    return analysis, {
        "source": [], "joined": [], "dictionary": [], "natural": [],
        "conflicts": [], "thresholds": [], "transfers": [],
        "decisions": decisions,
    }


def payloads(
    analysis: dict[str, Any], tables: dict[str, list[dict[str, Any]]],
    gates: list[dict[str, Any]],
) -> dict[str, str]:
    joined_rows = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in tables["joined"]
    ]
    contract_rows = [{
        **row, "required": canonical(row["required"]),
        "observed": canonical(row["observed"]),
    } for row in gates]
    return {
        OUTPUTS[0]: json.dumps(analysis, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        OUTPUTS[1]: report(analysis),
        OUTPUTS[2]: render_csv(SOURCE_H, tables["source"]),
        OUTPUTS[3]: render_csv(JOIN_H, joined_rows),
        OUTPUTS[4]: render_csv(FEATURE_H, tables["dictionary"]),
        OUTPUTS[5]: render_csv(GATE_H, tables["natural"]),
        OUTPUTS[6]: render_csv(CONFLICT_H, tables["conflicts"]),
        OUTPUTS[7]: render_csv(THRESHOLD_H, tables["thresholds"]),
        OUTPUTS[8]: render_csv(TRANSFER_H, tables["transfers"]),
        OUTPUTS[9]: render_csv(DECISION_H, tables["decisions"]),
        OUTPUTS[10]: render_csv(CONTRACT_H, contract_rows),
    }


def atomic_write(output: Path, data: dict[str, str]) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output}")
    if set(data) != set(OUTPUTS):
        raise ValueError("exact eleven-file payload closure failed")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for name in OUTPUTS:
            temporary = staging / f".{name}.tmp"
            with temporary.open("x", encoding="utf-8", newline="") as handle:
                handle.write(data[name])
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, staging / name)
        if sorted(path.name for path in staging.iterdir()) != sorted(OUTPUTS):
            raise RuntimeError("staged eleven-file closure failed")
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    ns = parse_args()
    gates: list[dict[str, Any]] = []
    try:
        analysis, tables = analyze(ns, gates)
        if any(not row["passed"] for row in gates):
            raise ValueError("contract contains a failed gate")
    except Exception as error:
        if not any(not row["passed"] for row in gates):
            gate(
                gates, "exception", "", "unhandled_exception", None,
                f"{type(error).__name__}: {error}", False,
                "unhandled analyzer exception", fatal=False,
            )
        analysis, tables = blocked_payload(ns, error)
    try:
        atomic_write(ns.output_dir.resolve(), payloads(analysis, tables, gates))
    except FileExistsError:
        pass
    return 0 if analysis["blocking_reasons"] == [] else 2


if __name__ == "__main__":
    raise SystemExit(main())
