#!/usr/bin/env python3
"""Design label-blind Stage196-B2-B6P4 action-response stability state.

The analyzer reconstructs the frozen three-action final-composer response at
epochs 18, 19, and 20.  It does not read P0 targets, search a safety gate or
threshold, train or load a model, change the selector, or use outcome labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Sequence


STAGE = "Stage196-B2-B6P4"
SEEDS = (183, 184, 185)
EPOCHS = (18, 19, 20)
MODES = ("joint", "frame_local_only")
RUNS = tuple(f"seed{seed}_{mode}" for seed in SEEDS for mode in MODES)
CANDIDATE_MASKS = ("00100000000000", "01000000000000", "10000000000000")
INTERNAL_CLASS_ORDER = ("REFUTE", "NOT_ENTITLED", "SUPPORT")
SEMANTIC_CLASS_ORDER = ("SUPPORT", "NOT_ENTITLED", "REFUTE")
RUNTIME_COMMIT = "fa16787efa84bb15d832b6d9382fafd77016c4e2"
PROVENANCE = "PROVENANCE_ONLY_NOT_FEATURE_AUTHORIZED"
DIAGNOSTIC = "DIAGNOSTIC_ONLY_NOT_INFERENCE_AUTHORIZED"
INFERENCE_CANDIDATE = "POTENTIALLY_INTEGRATION_AUTHORIZED_CANDIDATE_STATE_ONLY"
SERIALIZED_PRECISION = "%.17g"

P3_DECISION = "STAGE196B2B6P3_CROSS_SEED_UNSTABLE_ACTION_RESPONSE_SIGNAL"
P3_NEXT = "STAGE196B2B6P4_ACTION_RESPONSE_STABILITY_STATE_DESIGN"
P2_DECISION = "STAGE196B2B6P2_ACTION_CONDITIONAL_COMPOSER_MARGIN_EXPORT_COMPLETE"
B6_DECISION = "STAGE196B2B6_PRIMARY_EXACT_CLEAN_DEV_UNSAFE"
B5_DECISION = "STAGE196B2B5_CROSS_SEED_RECIPIENT_SELECTOR_LOCALIZED"
B4_DECISION = "STAGE196B2B4_ROW_SPECIFIC_PRIMITIVE_INTERACTION"

DECISIONS = (
    ("STAGE196B2B6P4_CANDIDATE_RELATIVE_INVARIANT_STATE_IDENTIFIED",
     "STAGE196B2B6P5_CANDIDATE_RELATIVE_STATE_SAFETY_DIAGNOSTIC"),
    ("STAGE196B2B6P4_STABLE_TRAJECTORY_TOPOLOGY_WITH_ENDPOINT_SHIFT",
     "STAGE196B2B6P5_CENTERED_RESPONSE_STATE_DESIGN"),
    ("STAGE196B2B6P4_ACTION_RESPONSE_TOPOLOGY_UNSTABLE",
     "STAGE196B2B6P5_TRAINING_SIDE_RESPONSE_STABILITY_INTERVENTION_DESIGN"),
    ("STAGE196B2B6P4_EXACT_TAIL_ACTION_RESPONSE_EXPORT_REQUIRED",
     "STAGE196B2B6P5_EXACT_TAIL_ACTION_RESPONSE_EXPORT"),
    ("STAGE196B2B6P4_TAIL_COMPOSER_INSTRUMENTATION_REQUIRED",
     "STAGE196B2B6P5_TAIL_COMPOSER_INSTRUMENTATION"),
)
BLOCKED = "STAGE196B2B6P4_BLOCKED_CONTRACT_FAILURE"
BLOCKED_NEXT = "STAGE196B2B6P4_REPAIR_CONTRACT"

OUTPUTS = (
    "stage196b2b6p4_analysis.json",
    "stage196b2b6p4_report.md",
    "stage196b2b6p4_source_closure.csv",
    "stage196b2b6p4_tail3_action_response_rows.csv",
    "stage196b2b6p4_stability_state_dictionary.csv",
    "stage196b2b6p4_trajectory_topology_audit.csv",
    "stage196b2b6p4_candidate_relative_order_audit.csv",
    "stage196b2b6p4_endpoint_shift_audit.csv",
    "stage196b2b6p4_leakage_boundary.csv",
    "stage196b2b6p4_decision_gate.csv",
    "stage196b2b6p4_contract.csv",
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
P3_OUTPUTS = (
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

SIGNED_FIELDS = (
    "delta_score_support",
    "delta_score_not_entitled",
    "delta_score_refute",
    "delta_top1_runner_up_margin",
    "delta_support_minus_not_entitled",
    "delta_support_minus_refute",
    "delta_refute_minus_not_entitled",
)
SCORE_FIELDS = ("support", "not_entitled", "refute")
MARGIN_FIELDS = (
    "top1_runner_up_margin",
    "support_minus_not_entitled",
    "support_minus_refute",
    "refute_minus_not_entitled",
)
BOOL_FIELDS = (
    "prediction_changed",
    "entitlement_transition",
    "polarity_transition",
    "polarity_direction_preserved",
)

CONTRACT_H = ("scope", "run", "gate", "required", "observed", "passed", "blocking_reason")
SOURCE_H = (
    "stage", "artifact", "path", "sha256", "row_count", "purpose",
    "feature_authorization",
)
TAIL_H = (
    "seed", "epoch", "stable_row_id", "data_identity", "candidate_mask",
    "candidate_action_key", "provenance_authorization", "tail_state_authorization",
    "native_prediction",
    "counterfactual_prediction",
    "native_score_support", "native_score_not_entitled", "native_score_refute",
    "counterfactual_score_support", "counterfactual_score_not_entitled",
    "counterfactual_score_refute", "native_top1_runner_up_margin",
    "native_margin_support_minus_not_entitled",
    "native_margin_support_minus_refute",
    "native_margin_refute_minus_not_entitled",
    "counterfactual_top1_runner_up_margin",
    "counterfactual_margin_support_minus_not_entitled",
    "counterfactual_margin_support_minus_refute",
    "counterfactual_margin_refute_minus_not_entitled",
    *SIGNED_FIELDS, *BOOL_FIELDS,
    "native_entitlement_reserve", "counterfactual_entitlement_reserve",
    "entitlement_reserve_change", "native_polarity_reserve",
    "counterfactual_polarity_reserve", "polarity_reserve_change",
    "entitlement_reserve_change_sign", "entitlement_reserve_direction",
    "polarity_reserve_change_sign", "polarity_reserve_direction",
    *(f"{field}_sign_relative_to_native" for field in SIGNED_FIELDS),
    *(f"{field}_centered_across_candidates" for field in SIGNED_FIELDS),
    *(f"{field}_candidate_rank" for field in SIGNED_FIELDS),
    *(f"{field}_pairwise_candidate_ordering" for field in SIGNED_FIELDS),
    "score_source_boundary", "native_score_dtype", "counterfactual_score_dtype",
)
DICTIONARY_H = (
    "field_or_state", "family", "authorization", "inference_time_available",
    "formula_or_definition", "exact_zero", "allowed_as_gate_feature",
)
TOPOLOGY_H = (
    "seed", "stable_row_id", "data_identity", "candidate_mask",
    "candidate_action_key", "signed_response_coordinate",
    "epoch18_value", "epoch19_value", "epoch20_value",
    "tail_minimum", "tail_maximum", "tail_range", "tail_mean",
    "epoch20_minus_epoch19", "epoch19_minus_epoch18",
    "epoch20_minus_tail_mean", "epoch18_sign", "epoch19_sign", "epoch20_sign",
    "sign_persistence_count", "zero_crossing_count", "sign_reversal_count",
    "monotonic_direction", "final_step_direction",
    "native_prediction_sequence", "counterfactual_prediction_sequence",
    "counterfactual_winner_persistence", "prediction_changed_sequence",
    "prediction_changed_persistence", "entitlement_transition_sequence",
    "entitlement_transition_persistence", "polarity_transition_sequence",
    "polarity_transition_persistence", "polarity_direction_preserved_sequence",
    "polarity_direction_persistence", "decision_boundary_crossing_count",
    "provenance_authorization", "state_authorization",
)
ORDER_H = (
    "seed", "stable_row_id", "data_identity", "signed_response_coordinate",
    "candidate_rank_epoch18", "candidate_rank_epoch19", "candidate_rank_epoch20",
    "pairwise_ordering_epoch18", "pairwise_ordering_epoch19",
    "pairwise_ordering_epoch20", "rank_persistence_across_tail",
    "pairwise_ordering_persistence_across_tail", "cross_seed_rank_agreement",
    "cross_seed_pairwise_order_agreement", "three_way_exact_candidate_order_agreement",
    "provenance_authorization", "state_authorization",
)
ENDPOINT_H = (
    "stable_row_id", "data_identity", "candidate_mask", "candidate_action_keys",
    "signed_response_coordinate", "seed183_endpoint", "seed184_endpoint",
    "seed185_endpoint", "endpoint_values_all_equal",
    "tail_sign_sequences", "same_sign_sequence_but_different_endpoint_value",
    "monotonic_directions",
    "same_monotonic_direction_but_different_endpoint_value",
    "categorical_sequences",
    "same_categorical_sequence_but_different_endpoint_margin",
    "tail_sign_sequence_agreement", "monotonic_direction_agreement",
    "counterfactual_winner_sequence_agreement", "transition_sequence_agreement",
    "candidate_rank_agreement", "pairwise_order_agreement",
    "provenance_authorization", "state_authorization",
)
LEAKAGE_H = (
    "field_or_category", "authorization", "present_in_row_outputs",
    "permitted_use", "enforcement",
)
DECISION_H = (
    "order", "decision", "condition", "observed", "reached",
    "recommended_next_stage", "scientific_authorization",
)

PROHIBITED = (
    "gold_label", "correctness", "recovery", "harm", "MUST_ALLOW", "MUST_BLOCK",
    "OPTIONAL", "safety_target", "transition_role", "primary_case",
    "discovery_membership", "raw_text",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, kind in (
        ("repo-root", Path),
        ("stage196b2b6p3-analysis-json", Path),
        ("stage196b2b6p2-analysis-json", Path),
        ("stage196b2b6-analysis-json", Path),
        ("stage196b2b5-analysis-json", Path),
        ("stage196b2b4-analysis-json", Path),
        ("stage196b2b3p0-run-root", Path),
        ("stage196b2b3p0-runtime-git-commit", str),
        ("stage196b2b5-recipient-signature-rows-csv", Path),
        ("current-git-commit", str),
        ("output-dir", Path),
    ):
        parser.add_argument(f"--{name}", required=True, type=kind)
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


def integer(value: Any, label: str) -> int:
    value = cell(value)
    if type(value) is not int:
        raise ValueError(f"{label}: integer required")
    return value


def optional_integer(value: Any, label: str) -> int | None:
    value = cell(value)
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if type(value) is not int:
        raise ValueError(f"{label}: integer or null required")
    return value


def boolean(value: Any, label: str) -> bool:
    value = cell(value)
    if type(value) is not bool:
        raise ValueError(f"{label}: boolean required")
    return value


def number(value: Any, label: str) -> float:
    value = cell(value)
    if type(value) not in (int, float):
        raise ValueError(f"{label}: numeric required")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label}: finite numeric required")
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
        return list(reader), list(reader.fieldnames or ())


def projected_csv(path: Path, fields: Sequence[str]) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or ())
        require_columns(header, fields, str(path))
        return [{field: row[field] for field in fields} for row in reader], header


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL row")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: JSON object required")
            rows.append(row)
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_columns(columns: Sequence[str], required: Sequence[str], label: str) -> None:
    missing = sorted(set(required) - set(columns))
    if missing:
        raise ValueError(f"{label}: missing columns {missing}")


def gate(
    rows: list[dict[str, Any]], scope: str, run: str, name: str,
    required: Any, observed: Any, passed: bool, reason: str, fatal: bool = True,
) -> None:
    rows.append({
        "scope": scope, "run": run, "gate": name, "required": required,
        "observed": observed, "passed": bool(passed),
        "blocking_reason": "" if passed else reason,
    })
    if fatal and not passed:
        raise ValueError(f"{name}: {reason}")


def source_row(stage: str, path: Path, count: Any, purpose: str) -> dict[str, Any]:
    return {
        "stage": stage, "artifact": path.name, "path": str(path),
        "sha256": sha256(path), "row_count": count, "purpose": purpose,
        "feature_authorization": PROVENANCE,
    }


def exact_files(path: Path, expected: Sequence[str], gates: list[dict[str, Any]], name: str) -> None:
    observed = sorted(item.name for item in path.iterdir() if item.is_file()) if path.is_dir() else []
    gate(
        gates, "source", "", name, sorted(expected), observed,
        observed == sorted(expected), f"{name}: exact file closure failed",
    )


def contract_pass(path: Path, label: str) -> tuple[list[dict[str, str]], list[str]]:
    rows, header = read_csv(path)
    require_columns(header, CONTRACT_H, label)
    if not rows or any(not boolean(row["passed"], f"{label} passed") or row["blocking_reason"].strip() for row in rows):
        raise ValueError(f"{label}: frozen contract is not fully passing")
    return rows, header


def import_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import deterministic source boundary {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def data_identity(identity: tuple[str, str, int]) -> str:
    return canonical({"id": identity[0], "source_row_id": identity[1], "dev_position": identity[2]})


def identity_from_row(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(row["id"]), str(row["source_row_id"]),
        integer(row["dev_position"], "dev_position"),
    )


def sign(value: float) -> str:
    return "ZERO" if value == 0 else "POSITIVE" if value > 0 else "NEGATIVE"


def direction(left: float, right: float) -> str:
    return "CONSTANT" if left == right else "INCREASING" if right > left else "DECREASING"


def monotonic(values: Sequence[float]) -> str:
    if values[0] == values[1] == values[2]:
        return "CONSTANT"
    if values[0] < values[1] < values[2]:
        return "STRICTLY_INCREASING"
    if values[0] <= values[1] <= values[2]:
        return "NONDECREASING"
    if values[0] > values[1] > values[2]:
        return "STRICTLY_DECREASING"
    if values[0] >= values[1] >= values[2]:
        return "NONINCREASING"
    return "NON_MONOTONIC"


def reversal_count(signs: Sequence[str]) -> int:
    nonzero = [item for item in signs if item != "ZERO"]
    return sum(left != right for left, right in zip(nonzero, nonzero[1:]))


def crossing_count(signs: Sequence[str]) -> int:
    return sum(left != right for left, right in zip(signs, signs[1:]))


def persistence_count(values: Sequence[Any]) -> int:
    return max(Counter(values).values())


def geometry_from_p2(module: ModuleType, scores: dict[str, float]) -> dict[str, Any]:
    """Rename fields from the exact P2 geometry function without recomputation."""
    value = module.geometry(scores)
    return {
        "prediction": value["prediction"],
        "top1_runner_up_margin": value["top1_margin"],
        "support_minus_not_entitled": value["support_minus_not_entitled"],
        "support_minus_refute": value["support_minus_refute"],
        "refute_minus_not_entitled": value["refute_minus_not_entitled"],
    }


def branch_flags_from_p2(
    module: ModuleType, native: str, counterfactual: str,
) -> dict[str, bool]:
    """Rename fields from the exact P2 branch-state function."""
    value = module.branch_flags(native, counterfactual)
    return {
        "prediction_changed": native != counterfactual,
        "entitlement_transition": value["entitlement_branch_changed"],
        "polarity_transition": value["polarity_branch_changed"],
        "polarity_direction_preserved": value["polarity_direction_preserved"],
    }


def rank_and_pairs(values: dict[str, float]) -> tuple[list[list[str]], dict[str, str], dict[str, int]]:
    groups: dict[float, list[str]] = defaultdict(list)
    for mask in CANDIDATE_MASKS:
        groups[values[mask]].append(mask)
    ranks = [groups[value] for value in sorted(groups, reverse=True)]
    numeric_rank = {
        mask: index + 1 for index, group in enumerate(ranks) for mask in group
    }
    pairs: dict[str, str] = {}
    for index, left in enumerate(CANDIDATE_MASKS):
        for right in CANDIDATE_MASKS[index + 1:]:
            relation = "TIE" if values[left] == values[right] else (
                "GREATER" if values[left] > values[right] else "LESS"
            )
            pairs[f"{left}__VS__{right}"] = relation
    return ranks, pairs, numeric_rank


def exact_authority_hash(authority: dict[str, Any], path: Path) -> str:
    normalized = str(path).replace("\\", "/")
    matches = [
        str(value) for key, value in authority.items()
        if normalized.endswith(str(key).replace("\\", "/"))
        or str(key).replace("\\", "/").endswith("/".join(path.parts[-4:]).replace("\\", "/"))
    ]
    return matches[0] if len(set(matches)) == 1 else ""


def normalize_sha256_authority(value: Any) -> str | None:
    """Return a canonical SHA256 only for valid, nonempty digest authority."""
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized if re.fullmatch(r"[0-9a-f]{64}", normalized) else None


def artifact_basename(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    return value.strip().replace("\\", "/").rsplit("/", 1)[-1]


def recipient_artifact_row(row: dict[str, Any]) -> bool:
    target = "stage196b2b5_recipient_signature_rows.csv"
    values = (
        row.get("artifact"), row.get("path"), row.get("gate"),
        row.get("purpose"), row.get("source_role"),
    )
    return any(
        target in str(value)
        or "recipient_signature" in str(value).lower()
        or "recipient-signature" in str(value).lower()
        or "b2b5_large_file" in str(value).lower()
        for value in values
    )


def unique_valid_hash(
    candidates: Iterable[tuple[str, Any]],
) -> tuple[str | None, str, list[dict[str, Any]], bool]:
    evidence: list[dict[str, Any]] = []
    valid: list[tuple[str, str]] = []
    for source, raw in candidates:
        normalized = normalize_sha256_authority(raw)
        evidence.append({
            "source": source, "raw_available": raw is not None,
            "normalized_sha256": normalized,
            "status": (
                "HASH_AUTHORITY_AVAILABLE"
                if normalized is not None else "HASH_AUTHORITY_UNAVAILABLE"
            ),
        })
        if normalized is not None:
            valid.append((source, normalized))
    distinct = sorted({value for _, value in valid})
    conflict = len(distinct) > 1
    if len(distinct) != 1:
        return None, "", evidence, conflict
    sources = sorted(source for source, value in valid if value == distinct[0])
    return distinct[0], "+".join(sources), evidence, False


def named_hash_candidates(source: str, value: Any) -> list[tuple[str, Any]]:
    candidates: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child_source = f"{source}.{key}"
            if "sha256" in str(key).lower():
                candidates.append((child_source, item))
            candidates.extend(named_hash_candidates(child_source, item))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            candidates.extend(named_hash_candidates(f"{source}[{index}]", item))
    return candidates


def decision_rows(
    full_relative: bool, topology_fully_stable: bool, endpoints_differ: bool,
    exact_export_available: bool, instrumentation_available: bool,
    instability: dict[str, int],
) -> tuple[str, str, list[dict[str, Any]]]:
    observed = (
        full_relative,
        not full_relative and topology_fully_stable and endpoints_differ,
        not full_relative and not topology_fully_stable,
        not exact_export_available and instrumentation_available,
        not instrumentation_available,
    )
    conditions = (
        "at least one signed coordinate has exact candidate ordering stable across epochs, seeds, and the full population",
        "no full relative invariant; all response topology is stable and exact endpoint values differ",
        "sign, winner, transition, monotonic, or candidate-order topology changes within tail or across seeds",
        "source state exists but exact full tail action response cannot be materialized",
        "no existing source boundary exposes epoch-level composer coordinates",
    )
    reached = next((index for index, value in enumerate(observed) if value), 3)
    rows: list[dict[str, Any]] = []
    for index, ((decision, next_stage), condition, value) in enumerate(zip(DECISIONS, conditions, observed), 1):
        rows.append({
            "order": index, "decision": decision, "condition": condition,
            "observed": value if index != 3 else {"triggered": value, "modes": instability},
            "reached": index - 1 == reached, "recommended_next_stage": next_stage,
            "scientific_authorization": "STATE_FAMILY_IDENTIFICATION_ONLY; NO_SAFETY_GATE_EVALUATED",
        })
    return DECISIONS[reached][0], DECISIONS[reached][1], rows


def dictionary_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    def add(field: str, family: str, authorization: str, available: bool, definition: str) -> None:
        rows.append({
            "field_or_state": field, "family": family, "authorization": authorization,
            "inference_time_available": available, "formula_or_definition": definition,
            "exact_zero": True, "allowed_as_gate_feature": False,
        })
    for field in SIGNED_FIELDS:
        add(field, "A_ABSOLUTE_TAIL_RESPONSE", DIAGNOSTIC, False,
            "counterfactual coordinate or margin minus native coordinate or margin; export epochs 18/19/20, min, max, range, mean, adjacent differences, and endpoint-minus-mean")
        add(field, "B_TRAJECTORY_TOPOLOGY", DIAGNOSTIC, False,
            "exact-zero sign sequence, persistence, crossings, reversals, monotonic direction, and final-step direction")
        add(f"{field}_sign_relative_to_native", "E_NATIVE_RELATIVE_ACTION_GEOMETRY",
            INFERENCE_CANDIDATE, True,
            "exact ZERO, POSITIVE, or NEGATIVE sign of counterfactual-minus-native response; no epsilon")
        add(f"{field}_centered_across_candidates", "E_NATIVE_RELATIVE_ACTION_GEOMETRY",
            INFERENCE_CANDIDATE, True,
            "candidate delta minus exact arithmetic mean of the three candidate deltas within the same row and checkpoint")
        add(f"{field}_candidate_rank", "D_CANDIDATE_RELATIVE_ACTION_ORDERING",
            INFERENCE_CANDIDATE, True,
            "descending exact-value tie-group rank among the three candidate actions; ties remain explicit")
        add(f"{field}_pairwise_candidate_ordering", "D_CANDIDATE_RELATIVE_ACTION_ORDERING",
            INFERENCE_CANDIDATE, True,
            "exact GREATER, LESS, or TIE relation for each precommitted candidate pair")
    add("monotonic_direction", "B_TRAJECTORY_TOPOLOGY", DIAGNOSTIC, False,
        "STRICTLY_INCREASING iff x18<x19<x20; NONDECREASING iff not strict and x18<=x19<=x20; STRICTLY_DECREASING iff x18>x19>x20; NONINCREASING iff not strict and x18>=x19>=x20; CONSTANT iff all equal; otherwise NON_MONOTONIC")
    add("categorical_action_response_persistence", "C_CATEGORICAL_ACTION_RESPONSE_PERSISTENCE",
        DIAGNOSTIC, False, "exact native/counterfactual prediction and model-output transition sequences over epochs 18,19,20")
    add("epoch20_model_output_transition_flags", "C_CATEGORICAL_ACTION_RESPONSE_PERSISTENCE",
        INFERENCE_CANDIDATE, True, "prediction, entitlement-branch, polarity-branch, and polarity-direction flags from model outputs only")
    add("native_relative_entitlement_reserve_change", "E_NATIVE_RELATIVE_ACTION_GEOMETRY",
        INFERENCE_CANDIDATE, True, "(counterfactual max(SUPPORT,REFUTE)-NOT_ENTITLED) minus its native value")
    add("native_relative_polarity_reserve_change", "E_NATIVE_RELATIVE_ACTION_GEOMETRY",
        INFERENCE_CANDIDATE, True, "counterfactual SUPPORT-minus-REFUTE minus native SUPPORT-minus-REFUTE")
    add("native_relative_reserve_change_direction", "E_NATIVE_RELATIVE_ACTION_GEOMETRY",
        INFERENCE_CANDIDATE, True, "exact INCREASES, DECREASES, or UNCHANGED state plus exact-zero sign for entitlement and polarity reserves")
    add("cross_seed_mechanism_agreement", "F_CROSS_SEED_MECHANISM_AGREEMENT",
        DIAGNOSTIC, False, "exact sequence, topology, candidate-rank, and pairwise-order agreement grouped across seeds")
    for field in ("seed", "epoch", "stable_row_id", "data_identity"):
        add(field, "PROVENANCE", PROVENANCE, False, "grouping and source-closure key only")
    return rows


def analyze(
    ns: argparse.Namespace, gates: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    root = ns.repo_root.resolve()
    output = ns.output_dir.resolve()
    authority_paths = {
        "p3": ns.stage196b2b6p3_analysis_json.resolve(),
        "p2": ns.stage196b2b6p2_analysis_json.resolve(),
        "b6": ns.stage196b2b6_analysis_json.resolve(),
        "b5": ns.stage196b2b5_analysis_json.resolve(),
        "b4": ns.stage196b2b4_analysis_json.resolve(),
        "runtime": ns.stage196b2b3p0_run_root.resolve(),
        "b5_large": ns.stage196b2b5_recipient_signature_rows_csv.resolve(),
    }
    explicit = (
        root.is_dir() and all(path.is_absolute() for path in authority_paths.values())
        and ns.repo_root.is_absolute() and ns.output_dir.is_absolute()
        and all(re.fullmatch(r"[0-9a-f]{40}", value) for value in (
            ns.stage196b2b3p0_runtime_git_commit, ns.current_git_commit,
        ))
    )
    gate(gates, "invocation", "", "exact_explicit_path_and_commit_contract", True, explicit,
         explicit, "all source paths and commits must be explicit")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output}")
    expected_basenames = {
        "p3": P3_OUTPUTS[0], "p2": P2_OUTPUTS[0],
        "b6": "stage196b2b6_analysis.json", "b5": "stage196b2b5_analysis.json",
        "b4": "stage196b2b4_analysis.json",
        "b5_large": "stage196b2b5_recipient_signature_rows.csv",
    }
    for key, name in expected_basenames.items():
        if authority_paths[key].name != name:
            raise ValueError(f"{key}: exact basename {name} required")

    actual_commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], check=False,
        capture_output=True, text=True, timeout=30,
    ).stdout.strip()
    gate(gates, "source", "", "current_commit_identity", ns.current_git_commit,
         actual_commit, actual_commit == ns.current_git_commit,
         "current Git commit disagrees with explicit authority")

    p3, p2, b6, b5, b4 = (
        read_json(authority_paths[key]) for key in ("p3", "p2", "b6", "b5", "b4")
    )
    required_decisions = {
        "p3": {"decision": P3_DECISION, "recommended_next_stage": P3_NEXT, "blocking_reasons": []},
        "p2": {"decision": P2_DECISION, "blocking_reasons": []},
        "b6": {"decision": B6_DECISION},
        "b5": {"decision": B5_DECISION},
        "b4": {"decision": B4_DECISION},
    }
    for key, required in required_decisions.items():
        document = {"p3": p3, "p2": p2, "b6": b6, "b5": b5, "b4": b4}[key]
        observed = {name: document.get(name) for name in required}
        gate(gates, "authority", key.upper(), f"{key}_decision_closure", required,
             observed, observed == required, f"{key} frozen decision changed")

    p3_dir, p2_dir = authority_paths["p3"].parent, authority_paths["p2"].parent
    exact_files(p3_dir, P3_OUTPUTS, gates, "p3_exact_eleven_file_closure")
    exact_files(p2_dir, P2_OUTPUTS, gates, "p2_exact_nine_file_closure")
    p3_contract, _ = contract_pass(p3_dir / P3_OUTPUTS[-1], "P3 contract")
    p2_contract, _ = contract_pass(p2_dir / P2_OUTPUTS[-1], "P2 contract")
    p2_source, p2_source_h = read_csv(p2_dir / P2_OUTPUTS[2])
    require_columns(
        p2_source_h,
        ("stage", "artifact", "path", "required", "loaded", "row_count", "sha256", "purpose"),
        "P2 source closure",
    )
    p2_recipient_source = [row for row in p2_source if recipient_artifact_row(row)]
    p2_recipient_contract = [row for row in p2_contract if recipient_artifact_row(row)]
    p2_hash_candidates: list[tuple[str, Any]] = [
        (f"P2_SOURCE_CLOSURE[{index}].sha256", row.get("sha256"))
        for index, row in enumerate(p2_recipient_source)
    ]
    for index, row in enumerate(p2_recipient_contract):
        p2_hash_candidates.extend(named_hash_candidates(
            f"P2_CONTRACT[{index}]", {
                "required": cell(row.get("required")),
                "observed": cell(row.get("observed")),
            },
        ))
    p2_hash_candidates.extend(named_hash_candidates(
        "P2_ANALYSIS.recipient_artifact",
        {
            "recipient_signature_authority": p2.get("recipient_signature_authority"),
            "b2b5_large_file_validation": p2.get("b2b5_large_file_validation"),
        },
    ))
    p2_expected_hash, p2_hash_source, p2_hash_evidence, p2_hash_conflict = unique_valid_hash(
        p2_hash_candidates
    )
    natural, natural_h = projected_csv(p3_dir / P3_OUTPUTS[5], ("feasible",))
    transfers, transfer_h = projected_csv(
        p3_dir / P3_OUTPUTS[8], ("bidirectional_transfer_success",)
    )
    require_columns(natural_h, ("feasible",), "P3 natural gates")
    require_columns(transfer_h, ("bidirectional_transfer_success",), "P3 transfers")
    feasible_count = sum(boolean(row["feasible"], "P3 feasible") for row in natural)
    transfer_count = sum(boolean(row["bidirectional_transfer_success"], "P3 transfer") for row in transfers)
    gate(gates, "authority", "P3", "p3_zero_natural_feasible_gates", 0,
         feasible_count, feasible_count == 0, "P3 natural feasible-gate count changed")
    gate(gates, "authority", "P3", "p3_zero_bidirectional_threshold_transfers",
         {"rows": 192, "successes": 0},
         {"rows": len(transfers), "successes": transfer_count},
         len(transfers) == 192 and transfer_count == 0,
         "P3 transfer audit closure or success count changed")

    p2_native, p2_native_h = read_csv(p2_dir / P2_OUTPUTS[3])
    p2_candidate, p2_candidate_h = read_csv(p2_dir / P2_OUTPUTS[4])
    p2_response, p2_response_h = read_csv(p2_dir / P2_OUTPUTS[5])
    required_repro = {
        "native_rows": 2160, "candidate_action_rows": 6480,
        "native_prediction_disagreements": 0,
        "counterfactual_prediction_disagreements": 0,
        "categorical_response_disagreements": 0,
    }
    repro = p2.get("prediction_reproduction", {})
    observed_repro = {
        "native_rows": len(p2_native), "candidate_action_rows": len(p2_candidate),
        "native_prediction_disagreements": repro.get("native_prediction_disagreements"),
        "counterfactual_prediction_disagreements": repro.get("counterfactual_prediction_disagreements"),
        "categorical_response_disagreements": repro.get("categorical_response_disagreements"),
    }
    gate(gates, "authority", "P2", "p2_endpoint_authority_closure",
         required_repro, observed_repro, observed_repro == required_repro,
         "P2 endpoint reproduction authority changed")

    observed_masks = sorted(row.get("feature_subset_mask", "") for row in b6.get("candidate_feature_subsets", []))
    gate(gates, "authority", "B2-B6", "b2b6_exact_candidate_masks",
         list(CANDIDATE_MASKS), observed_masks, observed_masks == list(CANDIDATE_MASKS),
         "B2-B6 candidate masks changed")
    gate(gates, "authority", "B2-B3P0", "runtime_commit_identity",
         RUNTIME_COMMIT,
         {"explicit": ns.stage196b2b3p0_runtime_git_commit},
         ns.stage196b2b3p0_runtime_git_commit == RUNTIME_COMMIT,
         "B2-B3P0 runtime commit changed")

    b5_required_schema = (
        "feature_family", "feature_subset_mask", "feature_subset_size",
        "feature_subset_members", "seed", "stable_row_id", "signature",
        "acceptable_coalitions", "signature_action_intersection",
        "signature_feasible", "source_seed", "target_seed", "transfer_status",
    )
    b5_fields = (
        "feature_family", "feature_subset_mask", "feature_subset_members", "seed",
        "stable_row_id", "signature", "acceptable_coalitions",
        "signature_action_intersection", "signature_feasible", "transfer_status",
        "source_seed", "target_seed",
    )
    explicit_b5_path = authority_paths["b5_large"]
    explicit_b5_exists = explicit_b5_path.exists()
    explicit_b5_is_file = explicit_b5_path.is_file()
    b5_large, b5_large_header = projected_csv(explicit_b5_path, b5_fields)
    actual_large_hash = normalize_sha256_authority(sha256(explicit_b5_path))
    if actual_large_hash is None:
        raise ValueError("computed recipient-signature SHA256 is invalid")
    actual_large_rows = len(b5_large)

    b5_dir = authority_paths["b5"].parent
    b5_contract_path = b5_dir / "stage196b2b5_contract.csv"
    b5_contract, _ = contract_pass(b5_contract_path, "B2-B5 contract")
    b5_hash_candidates: list[tuple[str, Any]] = []
    b5_source_closure_path = b5_dir / "stage196b2b5_source_closure.csv"
    b5_source_closure_rows: list[dict[str, str]] = []
    if p2_expected_hash is None:
        for key, value in b5.get("source_hashes", {}).items():
            if artifact_basename(key) == explicit_b5_path.name:
                b5_hash_candidates.append((f"B2B5_ANALYSIS.source_hashes[{key}]", value))
        b5_recipient_contract = [row for row in b5_contract if recipient_artifact_row(row)]
        for index, row in enumerate(b5_recipient_contract):
            b5_hash_candidates.extend(named_hash_candidates(
                f"B2B5_CONTRACT[{index}]", {
                    "required": cell(row.get("required")),
                    "observed": cell(row.get("observed")),
                },
            ))
        if b5_source_closure_path.is_file():
            b5_source_closure_rows, b5_source_closure_h = read_csv(b5_source_closure_path)
            require_columns(
                b5_source_closure_h, ("artifact", "path", "sha256"),
                "B2-B5 source closure",
            )
            for index, row in enumerate(b5_source_closure_rows):
                if recipient_artifact_row(row):
                    b5_hash_candidates.append(
                        (f"B2B5_SOURCE_CLOSURE[{index}].sha256", row.get("sha256"))
                    )
    b5_expected_hash, b5_hash_source, b5_hash_evidence, b5_hash_conflict = unique_valid_hash(
        b5_hash_candidates
    )

    if p2_expected_hash is not None:
        authority_mode = "P2_BYTE_HASH"
        authority_source = p2_hash_source or "P2_CONSUMED_ARTIFACT"
        expected_large_hash: str | None = p2_expected_hash
        selected_hash_conflict = p2_hash_conflict
    elif b5_expected_hash is not None:
        authority_mode = "B2B5_BYTE_HASH"
        authority_source = b5_hash_source or "B2B5_STORED_ARTIFACT"
        expected_large_hash = b5_expected_hash
        selected_hash_conflict = b5_hash_conflict
    else:
        authority_mode = "SEMANTIC_CLOSURE"
        authority_source = "EXPLICIT_CSV_PLUS_P2_B2B5_B2B6_SEMANTIC_AUTHORITIES"
        expected_large_hash = None
        selected_hash_conflict = p2_hash_conflict or b5_hash_conflict
    expected_sha256_available = expected_large_hash is not None
    byte_hash_match = (
        expected_sha256_available
        and actual_large_hash is not None
        and expected_large_hash == actual_large_hash
    )
    hash_authority_status = (
        "HASH_AUTHORITY_MISMATCH"
        if expected_sha256_available and actual_large_hash is not None and not byte_hash_match
        else "HASH_AUTHORITY_AVAILABLE"
        if expected_sha256_available
        else "HASH_AUTHORITY_UNAVAILABLE"
    )

    p2_recorded_names = sorted({
        artifact_basename(row.get("artifact") or row.get("path"))
        for row in p2_recipient_source
        if artifact_basename(row.get("artifact") or row.get("path"))
    })
    p2_recorded_counts: list[int] = []
    for row in p2_recipient_source:
        raw_count = cell(row.get("row_count"))
        if type(raw_count) is int:
            p2_recorded_counts.append(raw_count)
        elif isinstance(raw_count, str) and raw_count.strip().isdigit():
            p2_recorded_counts.append(int(raw_count.strip()))
    p2_recorded_roles = sorted({
        str(row.get("purpose", "")) for row in p2_recipient_source
        if str(row.get("purpose", "")).strip()
    })
    p2_explicit_status = all(
        boolean(row.get("required"), "P2 recipient required")
        and boolean(row.get("loaded"), "P2 recipient loaded")
        for row in p2_recipient_source
    ) if p2_recipient_source else False
    p2_provenance_present = bool(
        p2_recipient_source or p2_recipient_contract or p2_hash_candidates
    )
    p2_provenance_ok = True
    if p2_provenance_present:
        p2_provenance_ok = (
            not p2_hash_conflict
            and (not p2_recorded_names or p2_recorded_names == [explicit_b5_path.name])
            and (not p2_recorded_counts or set(p2_recorded_counts) == {actual_large_rows})
            and (p2_expected_hash is None or p2_expected_hash == actual_large_hash)
            and bool(p2_recipient_source)
            and p2_explicit_status
            and any("external" in role.lower() for role in p2_recorded_roles)
        )
        gate(
            gates, "authority", "P2", "p2_recipient_signature_provenance_closure",
            {
                "semantic_artifact": explicit_b5_path.name,
                "recorded_hash_when_available_matches_actual": True,
                "recorded_row_count_when_available": 524256,
                "source_role": "explicit external artifact",
                "required_and_loaded": True,
                "machine_specific_root_equality_required": False,
            },
            {
                "recorded_filenames": p2_recorded_names,
                "recorded_sha256": p2_expected_hash,
                "recorded_sha256_available": p2_expected_hash is not None,
                "recorded_row_counts": p2_recorded_counts,
                "source_roles": p2_recorded_roles,
                "required_and_loaded": p2_explicit_status,
                "hash_evidence": p2_hash_evidence,
            },
            p2_provenance_ok,
            "P2 recipient-signature consumed-artifact provenance changed",
            fatal=False,
        )

    b2b5_artifact_seed_set = sorted({integer(row["seed"], "recipient-signature seed") for row in b5_large})
    semantic_keys = {
        (
            row["feature_family"], row["feature_subset_mask"],
            integer(row["seed"], "recipient-signature key seed"),
            row["stable_row_id"], canonical(cell(row["feature_subset_members"])),
            canonical(cell(row["signature"])),
            canonical(cell(row["signature_action_intersection"])),
            row["transfer_status"], row["source_seed"], row["target_seed"],
        )
        for row in b5_large
    }
    b5_committed_candidate_members = {
        row.get("feature_subset_mask"): tuple(row.get("feature_subset_members", ()))
        for row in b5.get("recipient_inclusion_minimal_feasible_subsets", ())
        if row.get("feature_subset_mask") in CANDIDATE_MASKS
    }
    b6_committed_candidate_members = {
        row.get("feature_subset_mask"): tuple(row.get("feature_subset_members", ()))
        for row in b6.get("candidate_feature_subsets", ())
        if row.get("feature_subset_mask") in CANDIDATE_MASKS
    }
    external_candidate_members = {
        mask: {
            tuple(cell(row["feature_subset_members"]))
            for row in b5_large if row["feature_subset_mask"] == mask
        }
        for mask in CANDIDATE_MASKS
    }
    candidate_semantic_agreement = all(
        external_candidate_members[mask]
        == {b5_committed_candidate_members.get(mask)}
        == {b6_committed_candidate_members.get(mask)}
        and None not in external_candidate_members[mask]
        for mask in CANDIDATE_MASKS
    )
    external_schema_ok = set(b5_required_schema) <= set(b5_large_header)
    # Pooled intersections are signature-level constants; transfer intersections
    # are emitted once per target identity and are target-row-specific.
    authorized_transfer_directions = {(184, 185), (185, 184)}
    authorized_transfer_statuses = {"UNSEEN", "COMPATIBLE", "INCOMPATIBLE"}
    pooled_evidence_by_signature: dict[
        tuple[str, str], set[tuple[tuple[str, ...], bool]]
    ] = defaultdict(set)
    pooled_primary_identities: set[tuple[int, str]] = set()
    pooled_identity_key_counts: Counter[tuple[str, int, str]] = Counter()
    pooled_identity_row_count = pooled_empty_intersection_count = 0
    pooled_feasibility_semantic_disagreement_count = 0
    transfer_evidence_by_target: dict[
        tuple[str, int, int, str],
        set[tuple[str, str, tuple[str, ...], tuple[str, ...]]],
    ] = defaultdict(set)
    transfer_target_key_counts: Counter[tuple[str, int, int, str]] = Counter()
    transfer_records: list[dict[str, Any]] = []
    transfer_status_counts: Counter[str] = Counter()
    transfer_row_semantic_disagreements = 0
    selected_seed_identities: set[tuple[int, str]] = set()

    for row in b5_large:
        transfer_status = str(cell(row["transfer_status"]))
        source_seed = optional_integer(row["source_seed"], "B2-B5 source_seed")
        target_seed = optional_integer(row["target_seed"], "B2-B5 target_seed")
        row_seed = integer(row["seed"], "B2-B5 signature row seed")
        if transfer_status == "POOLED":
            if source_seed is not None or target_seed is not None:
                raise ValueError("B2-B5 pooled context has non-null transfer seeds")
        else:
            if transfer_status not in authorized_transfer_statuses:
                raise ValueError("B2-B5 transfer context has an unknown transfer status")
            if (source_seed, target_seed) not in authorized_transfer_directions:
                raise ValueError("B2-B5 transfer context has an unauthorized direction")
            if row_seed != target_seed:
                raise ValueError("B2-B5 transfer row seed does not equal target_seed")

        intersection = cell(row["signature_action_intersection"])
        acceptable = cell(row["acceptable_coalitions"])
        for values, label in (
            (intersection, "signature action intersection"),
            (acceptable, "acceptable coalitions"),
        ):
            if not isinstance(values, list):
                raise ValueError(f"B2-B5 {label} must be a list")
            if any(
                not isinstance(action, str)
                or re.fullmatch(r"[01]{5}", action) is None
                for action in values
            ):
                raise ValueError(f"B2-B5 {label} contains an invalid primitive action")
            if len(values) != len(set(values)):
                raise ValueError(f"B2-B5 {label} contains duplicate serialized primitive actions")
        signature_feasible = boolean(row["signature_feasible"], "B2-B5 signature_feasible")

        if row["feature_family"] != "recipient_local" or row["feature_subset_mask"] not in CANDIDATE_MASKS:
            continue
        if row_seed not in (184, 185):
            raise ValueError("selected B2-B5 recipient candidate row has a non-primary seed")
        mask = row["feature_subset_mask"]
        signature = canonical(cell(row["signature"]))
        stable_row_id = str(row["stable_row_id"])
        if not stable_row_id:
            raise ValueError("selected B2-B5 row has an empty stable_row_id")
        canonical_intersection = tuple(sorted(intersection))
        canonical_acceptable = tuple(sorted(acceptable))
        selected_seed_identities.add((row_seed, stable_row_id))

        if transfer_status == "POOLED":
            pooled_identity_row_count += 1
            pooled_primary_identities.add((row_seed, stable_row_id))
            pooled_identity_key_counts[(mask, row_seed, stable_row_id)] += 1
            pooled_evidence_by_signature[(mask, signature)].add(
                (canonical_intersection, signature_feasible)
            )
            pooled_empty_intersection_count += not canonical_intersection
            pooled_feasibility_semantic_disagreement_count += (
                signature_feasible != bool(canonical_intersection)
            )
            continue

        target_key = (mask, source_seed, target_seed, stable_row_id)
        transfer_target_key_counts[target_key] += 1
        transfer_evidence_by_target[target_key].add((
            signature, transfer_status, canonical_intersection, canonical_acceptable,
        ))
        transfer_status_counts[transfer_status] += 1
        semantic_ok = (
            set(canonical_intersection) <= set(canonical_acceptable)
            and signature_feasible == bool(canonical_intersection)
            and (
                (transfer_status == "COMPATIBLE" and bool(canonical_intersection))
                or (transfer_status in {"INCOMPATIBLE", "UNSEEN"} and not canonical_intersection)
            )
        )
        transfer_row_semantic_disagreements += not semantic_ok
        transfer_records.append({
            "candidate_mask": mask,
            "source_seed": source_seed,
            "target_seed": target_seed,
            "stable_row_id": stable_row_id,
            "signature": signature,
            "transfer_status": transfer_status,
            "intersection": canonical_intersection,
            "acceptable_coalitions": canonical_acceptable,
            "signature_feasible": signature_feasible,
        })

    pooled_duplicate_identity_key_count = sum(
        count - 1 for count in pooled_identity_key_counts.values() if count > 1
    )
    pooled_identity_mask_closure = (
        len(pooled_identity_key_counts) == 48
        and all(
            sum(key[1:] == identity for key in pooled_identity_key_counts) == 3
            for identity in pooled_primary_identities
        )
    )
    pooled_within_signature_conflicting_set_count = sum(
        len({entry[0] for entry in evidence}) > 1
        for evidence in pooled_evidence_by_signature.values()
    )
    pooled_within_signature_conflicting_feasible_count = sum(
        len({entry[1] for entry in evidence}) > 1
        for evidence in pooled_evidence_by_signature.values()
    )
    b5_pooled_actions_by_signature = {
        key: frozenset(next(iter(evidence))[0])
        for key, evidence in pooled_evidence_by_signature.items()
        if len({entry[0] for entry in evidence}) == 1
        and len({entry[1] for entry in evidence}) == 1
        and next(iter(evidence))[0]
    }
    transfer_identity_row_count = len(transfer_records)
    transfer_duplicate_target_row_key_count = sum(
        count - 1 for count in transfer_target_key_counts.values() if count > 1
    )
    transfer_within_target_row_disagreement_count = sum(
        len(evidence) > 1 for evidence in transfer_evidence_by_target.values()
    )

    variation_intersections: dict[
        tuple[str, str, int, int, str], set[tuple[str, ...]]
    ] = defaultdict(set)
    variation_row_counts: Counter[tuple[str, str, int, int, str]] = Counter()
    for record in transfer_records:
        variation_key = (
            record["candidate_mask"], record["signature"], record["source_seed"],
            record["target_seed"], record["transfer_status"],
        )
        variation_intersections[variation_key].add(record["intersection"])
        variation_row_counts[variation_key] += 1
    varying_transfer_groups = {
        key for key, intersections in variation_intersections.items()
        if len(intersections) > 1
    }
    transfer_variation_evidence = {
        "transfer_rows_accounted_for": sum(variation_row_counts.values()),
        "group_count": len(variation_intersections),
        "groups_with_one_intersection": sum(
            len(intersections) == 1 for intersections in variation_intersections.values()
        ),
        "groups_with_multiple_target_row_intersections": len(varying_transfer_groups),
        "maximum_distinct_intersections_within_group": max(
            (len(intersections) for intersections in variation_intersections.values()),
            default=0,
        ),
        "target_rows_in_varying_intersection_groups": sum(
            variation_row_counts[key] for key in varying_transfer_groups
        ),
        "transfer_target_row_specific_intersection_variation_count": len(varying_transfer_groups),
        "variation_is_contract_failure": False,
    }

    selected_primary_seed_counts = Counter(seed for seed, _ in selected_seed_identities)
    selected_candidate_masks = (
        {record["candidate_mask"] for record in transfer_records}
        | {key[0] for key in pooled_evidence_by_signature}
    )
    transfer_direction_counts = {
        "184_to_185": sum(
            record["source_seed"] == 184 and record["target_seed"] == 185
            for record in transfer_records
        ),
        "185_to_184": sum(
            record["source_seed"] == 185 and record["target_seed"] == 184
            for record in transfer_records
        ),
    }
    gate(
        gates, "authority", "B2-B5", "b2b5_selected_population_closure",
        {
            "candidate_masks": 3, "primary_identities": 16,
            "primary_seed_counts": {"184": 11, "185": 5},
            "pooled_rows": 48, "transfer_rows": 48, "selected_rows": 96,
            "seed183_selector_rows": 0,
        },
        {
            "candidate_masks": len(selected_candidate_masks),
            "primary_identities": len(selected_seed_identities),
            "primary_seed_counts": {
                "184": selected_primary_seed_counts[184],
                "185": selected_primary_seed_counts[185],
            },
            "pooled_rows": pooled_identity_row_count,
            "transfer_rows": transfer_identity_row_count,
            "selected_rows": pooled_identity_row_count + transfer_identity_row_count,
            "seed183_selector_rows": sum(seed == 183 for seed, _ in selected_seed_identities),
        },
        selected_candidate_masks == set(CANDIDATE_MASKS)
        and len(selected_seed_identities) == 16
        and selected_primary_seed_counts == Counter({184: 11, 185: 5})
        and pooled_identity_row_count == 48
        and transfer_identity_row_count == 48,
        "selected B2-B5 recipient candidate population does not close at 48 pooled plus 48 transfer rows",
    )
    gate(
        gates, "authority", "B2-B5", "b2b5_pooled_signature_intersection_closure",
        {
            "pooled_identity_row_count": 48,
            "pooled_unique_identity_keys": 48,
            "pooled_duplicate_identity_keys": 0,
            "pooled_identity_mask_closure": True,
            "pooled_within_signature_conflicting_set_count": 0,
            "pooled_within_signature_conflicting_feasible_count": 0,
            "pooled_feasibility_semantic_disagreements": 0,
            "pooled_empty_intersection_count": 0,
        },
        {
            "pooled_identity_row_count": pooled_identity_row_count,
            "pooled_unique_identity_keys": len(pooled_identity_key_counts),
            "pooled_duplicate_identity_keys": pooled_duplicate_identity_key_count,
            "pooled_identity_mask_closure": pooled_identity_mask_closure,
            "pooled_signature_key_count": len(pooled_evidence_by_signature),
            "pooled_within_signature_conflicting_set_count": pooled_within_signature_conflicting_set_count,
            "pooled_within_signature_conflicting_feasible_count": pooled_within_signature_conflicting_feasible_count,
            "pooled_feasibility_semantic_disagreements": pooled_feasibility_semantic_disagreement_count,
            "pooled_empty_intersection_count": pooled_empty_intersection_count,
            "candidate_masks": sorted({key[0] for key in pooled_evidence_by_signature}),
        },
        pooled_identity_row_count == 48
        and len(pooled_identity_key_counts) == 48
        and pooled_duplicate_identity_key_count == 0
        and pooled_identity_mask_closure
        and pooled_within_signature_conflicting_set_count == 0
        and pooled_within_signature_conflicting_feasible_count == 0
        and pooled_feasibility_semantic_disagreement_count == 0
        and pooled_empty_intersection_count == 0
        and {key[0] for key in pooled_evidence_by_signature} == set(CANDIDATE_MASKS),
        "B2-B5 pooled signature-level intersection authority does not close",
    )
    gate(
        gates, "authority", "B2-B5", "b2b5_transfer_target_row_closure",
        {
            "transfer_target_rows": 48,
            "unique_target_row_keys": 48,
            "duplicate_target_row_keys": 0,
            "within_target_row_evidence_disagreements": 0,
            "direction_row_counts": {"184_to_185": 15, "185_to_184": 33},
            "target_row_key_fields": [
                "feature_subset_mask", "source_seed", "target_seed", "stable_row_id",
            ],
        },
        {
            "transfer_target_rows": transfer_identity_row_count,
            "unique_target_row_keys": len(transfer_target_key_counts),
            "duplicate_target_row_keys": transfer_duplicate_target_row_key_count,
            "within_target_row_evidence_disagreements": transfer_within_target_row_disagreement_count,
            "direction_row_counts": transfer_direction_counts,
        },
        transfer_identity_row_count == 48
        and len(transfer_target_key_counts) == 48
        and transfer_duplicate_target_row_key_count == 0
        and transfer_within_target_row_disagreement_count == 0
        and transfer_direction_counts == {"184_to_185": 15, "185_to_184": 33},
        "B2-B5 transfer target-row identity closure failed",
    )

    b5_candidate_transfer_summary = {
        row.get("feature_subset_mask"): row
        for row in b5.get("recipient_inclusion_minimal_feasible_subsets", ())
        if row.get("feature_subset_mask") in CANDIDATE_MASKS
    }
    candidate_summary_bidirectional_full_pass = (
        set(b5_candidate_transfer_summary) == set(CANDIDATE_MASKS)
        and all(
            row.get("bidirectional_cross_seed_full_pass") is True
            for row in b5_candidate_transfer_summary.values()
        )
    )
    transfer_status_required = (
        {"COMPATIBLE": 48, "INCOMPATIBLE": 0, "UNSEEN": 0}
        if candidate_summary_bidirectional_full_pass else None
    )
    observed_transfer_status_counts = {
        status: transfer_status_counts[status]
        for status in ("COMPATIBLE", "INCOMPATIBLE", "UNSEEN")
    }
    gate(
        gates, "authority", "B2-B5", "b2b5_transfer_status_semantics",
        {
            "candidate_summary_bidirectional_full_pass": True,
            "transfer_rows": 48,
            "status_counts": {"COMPATIBLE": 48, "INCOMPATIBLE": 0, "UNSEEN": 0},
            "transfer_row_semantic_disagreements": 0,
        },
        {
            "candidate_summary_bidirectional_full_pass": candidate_summary_bidirectional_full_pass,
            "summary_by_candidate_mask": {
                mask: row.get("bidirectional_cross_seed_full_pass")
                for mask, row in sorted(b5_candidate_transfer_summary.items())
            },
            "transfer_rows": transfer_identity_row_count,
            "status_counts": observed_transfer_status_counts,
            "transfer_row_semantic_disagreements": transfer_row_semantic_disagreements,
        },
        candidate_summary_bidirectional_full_pass
        and transfer_status_required == observed_transfer_status_counts
        and transfer_identity_row_count == 48
        and transfer_row_semantic_disagreements == 0,
        "B2-B5 selected transfer status or target-row intersection semantics changed",
    )
    gate(
        gates, "diagnostic", "B2-B5",
        "b2b5_transfer_target_specific_variation_audit",
        {
            "transfer_rows_accounted_for": 48,
            "target_row_specific_variation_is_failure": False,
        },
        transfer_variation_evidence,
        sum(variation_row_counts.values()) == 48,
        "B2-B5 transfer target-row variation audit is incomplete",
    )
    b6_dir, b4_dir = authority_paths["b6"].parent, authority_paths["b4"].parent
    action_path = b6_dir / "stage196b2b6_clean_dev_signature_audit.csv"
    action_fields = (
        "feature_subset_mask", "seed", "stable_row_id", "id", "source_row_id",
        "dev_position", "signature", "assigned_action_set",
    )
    action_rows, action_header = projected_csv(action_path, action_fields)
    actions: dict[tuple[str, int, tuple[str, str, int]], tuple[str, str]] = {}
    action_signatures: dict[tuple[str, int, tuple[str, str, int]], str] = {}
    actions_by_stable: dict[tuple[str, int, str], str] = {}
    for row in action_rows:
        mask = row["feature_subset_mask"]
        seed = integer(row["seed"], "action seed")
        identity = identity_from_row(row)
        assigned = cell(row["assigned_action_set"])
        if mask not in CANDIDATE_MASKS or seed not in SEEDS or not isinstance(assigned, list) or len(assigned) != 1:
            raise ValueError("invalid deterministic B2-B6 action mapping")
        action = assigned[0]
        if not isinstance(action, str) or re.fullmatch(r"[01]{5}", action) is None:
            raise ValueError("invalid primitive action mask")
        key = (mask, seed, identity)
        if key in actions:
            raise ValueError("duplicate deterministic candidate action")
        actions[key] = (action, row["stable_row_id"])
        action_signatures[key] = canonical(cell(row["signature"]))
        stable_key = (mask, seed, row["stable_row_id"])
        if stable_key in actions_by_stable:
            raise ValueError("duplicate B2-B6 candidate action stable-row key")
        actions_by_stable[stable_key] = action

    action_seed_counts = Counter(seed for _, seed, _ in actions)
    action_mask_counts = Counter(mask for mask, _, _ in actions)
    deterministic_action_population_ok = (
        len(action_rows) == len(actions) == 6480
        and action_seed_counts == Counter({183: 2160, 184: 2160, 185: 2160})
        and set(action_mask_counts) == set(CANDIDATE_MASKS)
        and all(action_mask_counts[mask] == 2160 for mask in CANDIDATE_MASKS)
    )

    transfer_action_membership = {
        "transfer_rows_checked": 0,
        "assigned_action_in_transfer_intersection": 0,
        "assigned_action_outside_transfer_intersection": 0,
        "empty_transfer_intersections": 0,
        "missing_deterministic_target_actions": 0,
        "diagnostic_only": True,
        "deterministic_assignment_redefined": False,
    }
    for record in transfer_records:
        action = actions_by_stable.get((
            record["candidate_mask"], record["target_seed"], record["stable_row_id"],
        ))
        if action is None:
            transfer_action_membership["missing_deterministic_target_actions"] += 1
            continue
        transfer_action_membership["transfer_rows_checked"] += 1
        if not record["intersection"]:
            transfer_action_membership["empty_transfer_intersections"] += 1
        if action in record["intersection"]:
            transfer_action_membership["assigned_action_in_transfer_intersection"] += 1
        else:
            transfer_action_membership["assigned_action_outside_transfer_intersection"] += 1
    pooled_primary_matched_rows = pooled_primary_action_disagreements = 0
    pooled_primary_missing_authority = 0
    for key, (action, stable_row_id) in actions.items():
        mask, seed, _ = key
        if (seed, stable_row_id) not in pooled_primary_identities:
            continue
        acceptable_actions = b5_pooled_actions_by_signature.get((mask, action_signatures[key]))
        if acceptable_actions is None:
            pooled_primary_missing_authority += 1
            continue
        pooled_primary_matched_rows += 1
        pooled_primary_action_disagreements += action not in acceptable_actions
    gate(
        gates, "authority", "B2-B5/B2-B6",
        "b2b5_primary_pooled_action_membership_cross_check",
        {
            "primary_seeds": [184, 185], "primary_identities": 16,
            "candidate_masks": 3, "pooled_primary_matched_rows": 48,
            "pooled_primary_action_disagreements": 0,
        },
        {
            "primary_seeds": sorted({seed for seed, _ in pooled_primary_identities}),
            "primary_identities": len(pooled_primary_identities),
            "candidate_masks": len(CANDIDATE_MASKS),
            "pooled_primary_matched_rows": pooled_primary_matched_rows,
            "pooled_primary_missing_authority": pooled_primary_missing_authority,
            "pooled_primary_action_disagreements": pooled_primary_action_disagreements,
            "seed183_full_population_B2B5_membership_required": False,
        },
        len(pooled_primary_identities) == 16
        and pooled_primary_matched_rows == 48
        and pooled_primary_missing_authority == 0
        and pooled_primary_action_disagreements == 0,
        "B2-B5 pooled primary acceptable-action membership does not reproduce B2-B6 assignments",
    )

    p2_action_index: dict[tuple[str, int, str], tuple[str, str]] = {}
    for row in p2_candidate:
        mask = row["candidate_mask"]
        seed = integer(row["seed"], "P2 candidate seed")
        action = row["candidate_action_key"]
        if mask not in CANDIDATE_MASKS or seed not in SEEDS:
            raise ValueError("invalid P2 candidate action provenance")
        if not isinstance(action, str) or re.fullmatch(r"[01]{5}", action) is None:
            raise ValueError("invalid P2 primitive action key")
        key = (mask, seed, row["data_identity"])
        if key in p2_action_index:
            raise ValueError("duplicate P2 candidate-action provenance key")
        p2_action_index[key] = (action, row["stable_row_id"])

    p2_matched_rows = p2_action_disagreements = p2_stable_row_disagreements = 0
    p2_missing_rows = 0
    b6_p2_keys: set[tuple[str, int, str]] = set()
    for (mask, seed, identity), (action, stable_row_id) in actions.items():
        p2_key = (mask, seed, data_identity(identity))
        b6_p2_keys.add(p2_key)
        p2_value = p2_action_index.get(p2_key)
        if p2_value is None:
            p2_missing_rows += 1
            continue
        p2_matched_rows += 1
        p2_action_disagreements += p2_value[0] != action
        p2_stable_row_disagreements += p2_value[1] != stable_row_id
    p2_extra_rows = len(set(p2_action_index) - b6_p2_keys)
    p2_mapping_disagreements = (
        p2_missing_rows + p2_extra_rows
        + p2_action_disagreements + p2_stable_row_disagreements
    )
    gate(
        gates, "authority", "B2-B6/P2",
        "b2b6_p2_full_population_action_mapping_closure",
        {
            "B2B6_rows": 6480, "P2_rows": 6480, "matched_rows": 6480,
            "action_disagreements": 0, "stable_row_disagreements": 0,
            "missing_rows": 0, "extra_rows": 0,
        },
        {
            "B2B6_rows": len(actions), "P2_rows": len(p2_action_index),
            "matched_rows": p2_matched_rows,
            "action_disagreements": p2_action_disagreements,
            "stable_row_disagreements": p2_stable_row_disagreements,
            "missing_rows": p2_missing_rows, "extra_rows": p2_extra_rows,
        },
        deterministic_action_population_ok
        and len(p2_action_index) == 6480
        and p2_matched_rows == 6480
        and p2_action_disagreements == 0
        and p2_stable_row_disagreements == 0
        and p2_missing_rows == 0
        and p2_extra_rows == 0,
        "B2-B6 deterministic full-population actions do not reproduce P2 provenance",
    )

    committed_coalitions = set(b5.get("action_set_definition", {}).get("coalition_masks", ()))
    explicit_action_set_values = {
        action
        for evidence in pooled_evidence_by_signature.values()
        for intersection, _ in evidence
        for action in intersection
    } | {
        action for record in transfer_records for action in record["intersection"]
    }
    action_semantic_agreement = (
        committed_coalitions == {f"{value:05b}" for value in range(32)}
        and explicit_action_set_values <= committed_coalitions
        and {action for action, _ in actions.values()} <= committed_coalitions
    )
    runtime_for_semantic = authority_paths["runtime"]
    semantic_run_names = sorted(
        path.name for path in runtime_for_semantic.iterdir() if path.is_dir()
    ) if runtime_for_semantic.is_dir() else []
    prohibited_reconstruction_inputs = sorted(
        (set(b5_fields) | set(action_fields))
        & (set(PROHIBITED) - {"seed"})
    )
    actual_sha256_valid_nonempty_pass = (
        actual_large_hash is not None
        and normalize_sha256_authority(actual_large_hash) == actual_large_hash
    )
    file_identity_pass = (
        explicit_b5_exists
        and explicit_b5_is_file
        and actual_large_rows == 524256
    )
    schema_and_key_pass = (
        external_schema_ok
        and len(semantic_keys) == 524256
        and len(semantic_keys) == actual_large_rows
    )
    b2b5_population_pass = (
        pooled_identity_row_count + transfer_identity_row_count == 96
        and pooled_identity_row_count == 48
        and transfer_identity_row_count == 48
    )
    b2b5_pooled_pass = (
        pooled_duplicate_identity_key_count == 0
        and pooled_within_signature_conflicting_set_count == 0
        and pooled_within_signature_conflicting_feasible_count == 0
        and pooled_feasibility_semantic_disagreement_count == 0
        and pooled_empty_intersection_count == 0
        and pooled_primary_matched_rows == 48
        and pooled_primary_action_disagreements == 0
    )
    b2b5_transfer_pass = (
        transfer_duplicate_target_row_key_count == 0
        and transfer_row_semantic_disagreements == 0
        and transfer_within_target_row_disagreement_count == 0
        and transfer_variation_evidence["transfer_rows_accounted_for"] == 48
        and observed_transfer_status_counts
            == {"COMPATIBLE": 48, "INCOMPATIBLE": 0, "UNSEEN": 0}
    )
    b2b5_seed_closure_pass = (
        b2b5_artifact_seed_set == [184, 185]
        and sorted({seed for seed, _ in selected_seed_identities}) == [184, 185]
    )
    full_population_seed_closure_pass = (
        action_seed_counts == Counter({183: 2160, 184: 2160, 185: 2160})
    )
    b2b6_p2_mapping_pass = (
        len(actions) == 6480
        and len(p2_action_index) == 6480
        and p2_matched_rows == 6480
        and p2_action_disagreements == 0
        and p2_stable_row_disagreements == 0
    )
    candidate_semantics_pass = (
        selected_candidate_masks == set(CANDIDATE_MASKS)
        and {mask for mask, _, _ in actions} == set(CANDIDATE_MASKS)
        and set(b5_committed_candidate_members) == set(CANDIDATE_MASKS)
        and set(b6_committed_candidate_members) == set(CANDIDATE_MASKS)
        and b5_committed_candidate_members == b6_committed_candidate_members
        and candidate_semantic_agreement
        and action_semantic_agreement
    )
    six_run_pass = semantic_run_names == sorted(RUNS)
    leakage_pass = prohibited_reconstruction_inputs == []
    semantic_subconditions = {
        "actual_sha256_valid_nonempty_pass": actual_sha256_valid_nonempty_pass,
        "file_identity_pass": file_identity_pass,
        "schema_and_key_pass": schema_and_key_pass,
        "b2b5_population_pass": b2b5_population_pass,
        "b2b5_pooled_pass": b2b5_pooled_pass,
        "b2b5_transfer_pass": b2b5_transfer_pass,
        "b2b5_seed_closure_pass": b2b5_seed_closure_pass,
        "full_population_seed_closure_pass": full_population_seed_closure_pass,
        "b2b6_p2_mapping_pass": b2b6_p2_mapping_pass,
        "candidate_semantics_pass": candidate_semantics_pass,
        "six_run_pass": six_run_pass,
        "leakage_pass": leakage_pass,
    }
    failed_semantic_subconditions = [
        name for name, passed in semantic_subconditions.items() if not passed
    ]
    semantic_closure_passed = all(semantic_subconditions.values())
    semantic_closure_evidence = {
        "semantic_subconditions": semantic_subconditions,
        "failed_semantic_subconditions": failed_semantic_subconditions,
        "explicit_path_exists": explicit_b5_exists,
        "explicit_path_is_regular_file": explicit_b5_is_file,
        "actual_sha256_valid_nonempty": actual_sha256_valid_nonempty_pass,
        "actual_rows": actual_large_rows,
        "required_rows": 524256,
        "required_schema_present": external_schema_ok,
        "B2B5_selector_primary_seed_set": b2b5_artifact_seed_set,
        "semantic_key_count": len(semantic_keys),
        "semantic_keys_unique": len(semantic_keys) == actual_large_rows,
        "six_run_names": semantic_run_names,
        "B2B5_selected_rows": pooled_identity_row_count + transfer_identity_row_count,
        "B2B5_pooled_identity_rows": pooled_identity_row_count,
        "B2B5_pooled_signature_key_count": len(pooled_evidence_by_signature),
        "B2B5_pooled_duplicate_identity_keys": pooled_duplicate_identity_key_count,
        "B2B5_pooled_identity_mask_closure": pooled_identity_mask_closure,
        "B2B5_pooled_within_signature_conflicting_set_count": pooled_within_signature_conflicting_set_count,
        "B2B5_pooled_within_signature_conflicting_feasible_count": pooled_within_signature_conflicting_feasible_count,
        "B2B5_pooled_feasibility_semantic_disagreements": pooled_feasibility_semantic_disagreement_count,
        "B2B5_pooled_empty_intersection_count": pooled_empty_intersection_count,
        "B2B5_transfer_target_rows": transfer_identity_row_count,
        "B2B5_transfer_duplicate_target_row_keys": transfer_duplicate_target_row_key_count,
        "B2B5_transfer_within_target_row_disagreements": transfer_within_target_row_disagreement_count,
        "B2B5_transfer_row_semantic_disagreements": transfer_row_semantic_disagreements,
        "B2B5_transfer_status_counts": observed_transfer_status_counts,
        "B2B5_transfer_direction_counts": transfer_direction_counts,
        "B2B5_candidate_summary_bidirectional_full_pass": candidate_summary_bidirectional_full_pass,
        "B2B5_transfer_target_specific_variation": transfer_variation_evidence,
        "B2B5_pooled_primary_matched_rows": pooled_primary_matched_rows,
        "B2B5_pooled_primary_action_disagreements": pooled_primary_action_disagreements,
        "B2B5_transfer_action_membership_diagnostic": transfer_action_membership,
        "candidate_action_rows": len(actions),
        "candidate_action_seed_counts": {
            str(seed): action_seed_counts[seed] for seed in SEEDS
        },
        "candidate_masks": sorted({mask for mask, _, _ in actions}),
        "candidate_semantic_agreement": candidate_semantic_agreement,
        "B2B5_candidate_members": b5_committed_candidate_members,
        "B2B6_candidate_members": b6_committed_candidate_members,
        "committed_action_semantic_agreement": action_semantic_agreement,
        "P2_candidate_action_rows": len(p2_action_index),
        "P2_candidate_action_matched_rows": p2_matched_rows,
        "P2_candidate_action_disagreements": p2_action_disagreements,
        "P2_stable_row_disagreements": p2_stable_row_disagreements,
        "prohibited_reconstruction_inputs": prohibited_reconstruction_inputs,
    }
    semantic_required = {name: True for name in semantic_subconditions}
    semantic_failure_reason = (
        "explicit recipient-signature semantic authority closure failed: "
        + ", ".join(failed_semantic_subconditions)
        if failed_semantic_subconditions
        else ""
    )
    gate(
        gates, "authority", "B2-B5/P2/B2-B6",
        "recipient_signature_semantic_closure",
        semantic_required,
        semantic_closure_evidence,
        semantic_closure_passed,
        semantic_failure_reason,
        fatal=False,
    )

    if authority_mode == "P2_BYTE_HASH":
        recipient_signature_authority_passed = (
            explicit_b5_exists and byte_hash_match and semantic_closure_passed
        )
    elif authority_mode == "B2B5_BYTE_HASH":
        recipient_signature_authority_passed = (
            explicit_b5_exists and byte_hash_match and semantic_closure_passed
        )
    else:
        recipient_signature_authority_passed = (
            explicit_b5_exists
            and authority_mode == "SEMANTIC_CLOSURE"
            and not selected_hash_conflict
            and semantic_closure_passed
        )
    recipient_action_authority = {
        "full_population_assignment": "B2-B6 singleton assigned_action_set",
        "full_population_reproduction": "P2 candidate_action_key",
        "pooled_signature_semantics": "B2-B5 pooled signature_action_intersection",
        "transfer_semantics": "B2-B5 target-row-specific source-signature/target-acceptable intersection",
        "pooled_key_fields": ["feature_subset_mask", "canonical_signature"],
        "transfer_row_key_fields": [
            "feature_subset_mask", "source_seed", "target_seed", "stable_row_id",
        ],
        "transfer_intersection_is_signature_constant": False,
        "selected_rows": pooled_identity_row_count + transfer_identity_row_count,
        "pooled_rows": pooled_identity_row_count,
        "transfer_rows": transfer_identity_row_count,
        "pooled_primary_cross_check_rows": pooled_primary_matched_rows,
        "transfer_target_row_specific_intersection_variation_count": (
            transfer_variation_evidence["transfer_target_row_specific_intersection_variation_count"]
        ),
        "transfer_action_membership_diagnostic": transfer_action_membership,
        "B2B6_P2_matched_rows": p2_matched_rows,
        "B2B6_P2_action_disagreements": p2_action_disagreements,
        "B2B6_P2_stable_row_disagreements": p2_stable_row_disagreements,
    }
    recipient_signature_authority = {
        "authority_mode": authority_mode,
        "authority_source": authority_source,
        "explicit_path": str(explicit_b5_path),
        "actual_sha256": actual_large_hash,
        "expected_sha256": expected_large_hash,
        "expected_sha256_available": expected_sha256_available,
        "byte_hash_match": byte_hash_match,
        "hash_authority_status": hash_authority_status,
        "hash_authority_evidence": {
            "P2": p2_hash_evidence, "B2B5": b5_hash_evidence,
            "selected_hash_conflict": selected_hash_conflict,
        },
        "row_count": actual_large_rows,
        "semantic_closure_passed": semantic_closure_passed,
        "semantic_subconditions": semantic_subconditions,
        "failed_semantic_subconditions": failed_semantic_subconditions,
        "recipient_signature_authority_passed": recipient_signature_authority_passed,
    }
    gate(
        gates, "authority", "B2-B5/P2",
        "explicit_recipient_signature_authority",
        {
            "explicit_path_required": True,
            "authority_mode": authority_mode,
        },
        {
            "explicit_path": ns.stage196b2b5_recipient_signature_rows_csv.is_absolute(),
            "explicit_path_exists": explicit_b5_exists,
            "explicit_path_is_regular_file": explicit_b5_is_file,
            "resolved_path": str(explicit_b5_path),
            "actual_rows": actual_large_rows,
            "actual_sha256": actual_large_hash,
            "expected_sha256": expected_large_hash,
            "expected_sha256_available": expected_sha256_available,
            "authority_source": authority_source,
            "authority_mode": authority_mode,
            "byte_hash_match": byte_hash_match,
            "semantic_closure_passed": semantic_closure_passed,
            "semantic_subconditions": semantic_subconditions,
            "failed_semantic_subconditions": failed_semantic_subconditions,
            "recipient_signature_authority_passed": recipient_signature_authority_passed,
        },
        recipient_signature_authority_passed,
        (
            "recipient-signature semantic authority closure failed: "
            + ", ".join(failed_semantic_subconditions)
            if failed_semantic_subconditions
            else "recipient-signature byte hash authority mismatch"
            if hash_authority_status == "HASH_AUTHORITY_MISMATCH"
            else "recipient-signature authority closure failed"
        ),
    )

    p2_script = root / "scripts" / "export_stage196b2b6p2_action_conditional_composer_margins.py"
    b6_script = root / "scripts" / "analyze_stage196b2b6_minimal_selector_intervention.py"
    p2_module = import_module(p2_script, "_stage196b2b6p2_exact_boundary")
    b6_module = import_module(b6_script, "_stage196b2b6_exact_boundary")
    source_boundary_ok = (
        tuple(b6_module.LABELS) == INTERNAL_CLASS_ORDER
        and p2_module.SOURCE_BOUNDARY == p2.get("composer_source_boundary", {}).get("identity")
        and p2_module.score_map_from_native.__module__ == p2_module.__name__
        and p2_module.score_map_from_reconstruction.__module__ == p2_module.__name__
        and p2_module.geometry.__module__ == p2_module.__name__
        and p2_module.branch_flags.__module__ == p2_module.__name__
    )
    gate(gates, "source", "", "exact_p2_composer_boundary_reuse",
         {"reused_P2_functions": True, "class_order": list(INTERNAL_CLASS_ORDER)},
         {"reused_P2_functions": source_boundary_ok, "class_order": list(b6_module.LABELS)},
         source_boundary_ok, "P2 exact composer implementation cannot be reused")
    gate(gates, "source", "", "class_order_authority", list(INTERNAL_CLASS_ORDER),
         list(b6_module.LABELS), tuple(b6_module.LABELS) == INTERNAL_CLASS_ORDER,
         "class-coordinate order changed")
    expected_primitives = (
        "FRAME", "PREDICATE", "SUFFICIENCY", "POSITIVE_ENERGY", "NEGATIVE_ENERGY",
    )
    observed_b4_primitives = tuple(b4.get("primitive_lattice", {}).get("canonical_order", ()))
    primitive_semantics_ok = (
        tuple(b6_module.PRIMITIVES) == expected_primitives
        and observed_b4_primitives == expected_primitives
        and set(b6_module.PRIMITIVE_FIELDS) == set(expected_primitives)
    )
    gate(gates, "source", "B2-B4/B2-B6", "primitive_mask_semantic_authority",
         {"primitive_order": list(expected_primitives), "mask_width": 5},
         {"B2-B4_order": list(observed_b4_primitives),
          "B2-B6_order": list(b6_module.PRIMITIVES), "mask_width": 5},
         primitive_semantics_ok, "frozen primitive mask semantics changed")

    runtime = authority_paths["runtime"]
    states: dict[tuple[int, int, tuple[str, str, int]], dict[str, dict[str, Any]]] = {}
    source: list[dict[str, Any]] = [
        source_row("P3", authority_paths["p3"], "", "motivation and source closure only; joined safety rows not loaded"),
        source_row("P2", authority_paths["p2"], "", "epoch-20 exact endpoint reproduction authority"),
        source_row("B2-B6", authority_paths["b6"], "", "candidate-mask and action semantics"),
        source_row("B2-B5", authority_paths["b5"], "", "contextual recipient acceptable-action-set semantics"),
        source_row("B2-B5", authority_paths["b5_large"], len(b5_large), "explicit contextual signature acceptable-action-set authority; not deterministic assignment"),
        source_row("B2-B4", authority_paths["b4"], "", "controlled composer semantic and numeric authority"),
        source_row("P3", p3_dir / P3_OUTPUTS[5], len(natural), "projected feasible-count closure only; safety fields not loaded"),
        source_row("P3", p3_dir / P3_OUTPUTS[8], len(transfers), "projected bidirectional-transfer closure only"),
        source_row("P3", p3_dir / P3_OUTPUTS[-1], len(p3_contract), "fully passing P3 contract"),
        source_row("P2", p2_dir / P2_OUTPUTS[2], len(p2_source), "consumed-artifact provenance and preferred recipient-signature hash authority"),
        source_row("P2", p2_dir / P2_OUTPUTS[3], len(p2_native), "epoch-20 native score authority"),
        source_row("P2", p2_dir / P2_OUTPUTS[4], len(p2_candidate), "epoch-20 counterfactual score authority"),
        source_row("P2", p2_dir / P2_OUTPUTS[5], len(p2_response), "epoch-20 margin and transition authority"),
        source_row("P2", p2_dir / P2_OUTPUTS[-1], len(p2_contract), "fully passing P2 contract"),
        source_row("B2-B5", b5_contract_path, len(b5_contract), "fully passing B2-B5 semantic contract"),
        source_row("B2-B6", action_path, len(action_rows), "projected deterministic row/action mapping"),
        source_row("source", p2_script, "", "reused exact P2 score, geometry, and branch-state boundary"),
        source_row("source", b6_script, "", "reused deterministic reconstruct/apply_mask boundary"),
    ]
    composer_files = trajectory_files = manifest_files = 0
    observed_runs = sorted(path.name for path in runtime.iterdir() if path.is_dir())
    gate(gates, "runtime", "", "six_run_directory_closure", sorted(RUNS),
         observed_runs, observed_runs == sorted(RUNS),
         "B2-B3P0 run-root directory closure failed")
    runtime_hashes = b6.get("source_hashes", {})
    for seed in SEEDS:
        for mode in MODES:
            run = f"seed{seed}_{mode}"
            composer_dir = runtime / run / "composer_inputs"
            trajectory_dir = runtime / run / "trajectory"
            if not trajectory_dir.is_dir():
                trajectory_dir = runtime / run / "trajectories"
            manifest_path = composer_dir / "stage196b2b3p0_composer_input_manifest.json"
            manifest = read_json(manifest_path)
            manifest_files += 1
            source.append(source_row("B2-B3P0", manifest_path, "", "exact runtime manifest authority"))
            expected_manifest_hash = exact_authority_hash(runtime_hashes, manifest_path)
            manifest_hash_ok = bool(expected_manifest_hash) and sha256(manifest_path) == expected_manifest_hash
            manifest_ok = (
                manifest_hash_ok
                and manifest.get("current_git_commit") == RUNTIME_COMMIT
                and integer(manifest.get("seed"), "manifest seed") == seed
                and manifest.get("gradient_ownership_mode") == mode
                and manifest.get("completed") is True
            )
            gate(gates, "runtime", run, "manifest_runtime_identity", True,
                 {"runtime_commit": manifest.get("current_git_commit"),
                  "seed": manifest.get("seed"), "mode": manifest.get("gradient_ownership_mode")},
                 manifest_ok, "runtime manifest identity changed")
            composer_names = tuple(f"stage196b2b3p0_epoch_composer_inputs_{epoch:03d}.jsonl" for epoch in range(1, 21))
            trajectory_names = tuple(f"stage196b2p0_epoch_channels_{epoch:03d}.jsonl" for epoch in range(1, 21))
            observed_composer = tuple(sorted(
                path.name for path in composer_dir.iterdir()
                if re.fullmatch(r"stage196b2b3p0_epoch_composer_inputs_[0-9]{3}\.jsonl", path.name)
            ))
            observed_trajectory = tuple(sorted(
                path.name for path in trajectory_dir.iterdir()
                if re.fullmatch(r"stage196b2p0_epoch_channels_[0-9]{3}\.jsonl", path.name)
            ))
            if observed_composer != composer_names or observed_trajectory != trajectory_names:
                raise ValueError(f"{run}: exact 20-epoch sidecar namespace failed")
            composer_files += len(observed_composer)
            trajectory_files += len(observed_trajectory)
            for epoch in EPOCHS:
                composer_path = composer_dir / f"stage196b2b3p0_epoch_composer_inputs_{epoch:03d}.jsonl"
                trajectory_path = trajectory_dir / f"stage196b2p0_epoch_channels_{epoch:03d}.jsonl"
                expected_hash = manifest.get("sidecar_sha256", {}).get(composer_path.name)
                expected_trajectory_hash = exact_authority_hash(runtime_hashes, trajectory_path)
                if not expected_hash or sha256(composer_path) != expected_hash:
                    raise ValueError(f"{run}:{epoch}: composer sidecar hash mismatch")
                if not expected_trajectory_hash or sha256(trajectory_path) != expected_trajectory_hash:
                    raise ValueError(f"{run}:{epoch}: trajectory sidecar hash mismatch")
                composer_rows = read_jsonl(composer_path)
                trajectory_rows = read_jsonl(trajectory_path)
                if len(composer_rows) != 720 or len(trajectory_rows) != 720:
                    raise ValueError(f"{run}:{epoch}: expected 720 rows per sidecar")
                trajectory_index = {identity_from_row(row): row for row in trajectory_rows}
                if len(trajectory_index) != 720:
                    raise ValueError(f"{run}:{epoch}: duplicate trajectory identity")
                seen: set[tuple[str, str, int]] = set()
                for row in composer_rows:
                    identity = identity_from_row(row)
                    if identity in seen or identity not in trajectory_index:
                        raise ValueError(f"{run}:{epoch}: identity join failure")
                    seen.add(identity)
                    if (
                        integer(row["seed"], "composer seed") != seed
                        or integer(row["epoch"], "composer epoch") != epoch
                        or row["gradient_ownership_mode"] != mode
                        or tuple(row["native_logit_order"]) != INTERNAL_CLASS_ORDER
                        or trajectory_index[identity].get("prediction") != row["final_native_prediction"]
                    ):
                        raise ValueError(f"{run}:{epoch}: composer provenance mismatch")
                    for field, value in row.items():
                        if type(value) in (int, float) and (
                            field.endswith(("_logit", "_prob", "_energy", "_bias"))
                            or field.startswith(("final_", "reconstructed_final_"))
                        ) and not math.isfinite(float(value)):
                            raise ValueError(f"{run}:{epoch}:{field}: non-finite")
                    states.setdefault((seed, epoch, identity), {})[mode] = row
                source.extend((
                    source_row("B2-B3P0", composer_path, len(composer_rows), "exact tail composer inputs"),
                    source_row("B2-B3P0", trajectory_path, len(trajectory_rows), "exact tail prediction identity authority"),
                ))
    gate(gates, "runtime", "", "six_run_and_sidecar_closure",
         {"runs": list(RUNS), "composer": 120, "trajectory": 120, "manifests": 6},
         {"runs": list(RUNS), "composer": composer_files, "trajectory": trajectory_files,
          "manifests": manifest_files},
         composer_files == trajectory_files == 120 and manifest_files == 6,
         "six-run sidecar closure failed")

    identity_sets = {
        (seed, epoch): {identity for state_seed, state_epoch, identity in states
                        if state_seed == seed and state_epoch == epoch}
        for seed in SEEDS for epoch in EPOCHS
    }
    state_ok = (
        len(states) == 6480
        and all(len(value) == 720 for value in identity_sets.values())
        and all(set(arms) == set(MODES) for arms in states.values())
        and len({frozenset(value) for value in identity_sets.values()}) == 1
    )
    gate(gates, "population", "", "tail_native_epoch_population",
         {"epochs": list(EPOCHS), "rows": 6480, "rows_per_seed_epoch": 720,
          "same_cross_seed_identity_set": True},
         {"epochs": sorted({key[1] for key in states}), "rows": len(states),
          "rows_per_seed_epoch": {f"{seed}:{epoch}": len(identity_sets[(seed, epoch)])
                                  for seed in SEEDS for epoch in EPOCHS},
          "same_cross_seed_identity_set": len({frozenset(value) for value in identity_sets.values()}) == 1},
         state_ok, "tail native epoch population is incomplete")

    tail_rows: list[dict[str, Any]] = []
    native_epoch: dict[tuple[int, int, str], dict[str, Any]] = {}
    candidate_epoch: dict[tuple[int, int, str, str], dict[str, Any]] = {}
    for key in sorted(states):
        seed, epoch, identity = key
        joint = states[key]["joint"]
        native_scores = p2_module.score_map_from_native(joint)
        native_geo = geometry_from_p2(p2_module, native_scores)
        native_prediction = str(joint["final_native_prediction"])
        if native_geo["prediction"] != native_prediction:
            raise ValueError("native score/prediction disagreement")
        stable = str(joint["stable_row_id"])
        identity_text = data_identity(identity)
        native_epoch[(seed, epoch, identity_text)] = {
            "scores": native_scores, "geo": native_geo, "prediction": native_prediction,
            "stable_row_id": stable,
        }
        epoch_candidates: dict[str, dict[str, Any]] = {}
        for mask in CANDIDATE_MASKS:
            action_key = (mask, seed, identity)
            if action_key not in actions:
                raise ValueError("missing candidate action")
            action, action_stable = actions[action_key]
            if action_stable != stable:
                raise ValueError("cross-seed or row-identity action confusion")
            reconstructed = b6_module.apply_mask(joint, states[key]["frame_local_only"], action)
            cf_scores = p2_module.score_map_from_reconstruction(reconstructed)
            cf_geo = geometry_from_p2(p2_module, cf_scores)
            cf_prediction = str(reconstructed.get("prediction", ""))
            if cf_geo["prediction"] != cf_prediction:
                raise ValueError("counterfactual score/prediction disagreement")
            delta = {
                "delta_score_support": cf_scores["SUPPORT"] - native_scores["SUPPORT"],
                "delta_score_not_entitled": cf_scores["NOT_ENTITLED"] - native_scores["NOT_ENTITLED"],
                "delta_score_refute": cf_scores["REFUTE"] - native_scores["REFUTE"],
                "delta_top1_runner_up_margin": cf_geo["top1_runner_up_margin"] - native_geo["top1_runner_up_margin"],
                "delta_support_minus_not_entitled": cf_geo["support_minus_not_entitled"] - native_geo["support_minus_not_entitled"],
                "delta_support_minus_refute": cf_geo["support_minus_refute"] - native_geo["support_minus_refute"],
                "delta_refute_minus_not_entitled": cf_geo["refute_minus_not_entitled"] - native_geo["refute_minus_not_entitled"],
            }
            native_entitlement = max(native_scores["SUPPORT"], native_scores["REFUTE"]) - native_scores["NOT_ENTITLED"]
            cf_entitlement = max(cf_scores["SUPPORT"], cf_scores["REFUTE"]) - cf_scores["NOT_ENTITLED"]
            flags = branch_flags_from_p2(p2_module, native_prediction, cf_prediction)
            epoch_candidates[mask] = {
                "action": action, "cf_scores": cf_scores, "cf_geo": cf_geo,
                "cf_prediction": cf_prediction, "delta": delta, "flags": flags,
                "native_entitlement": native_entitlement,
                "cf_entitlement": cf_entitlement,
            }
        order_state: dict[str, tuple[list[list[str]], dict[str, str], dict[str, int]]] = {}
        for field in SIGNED_FIELDS:
            order_state[field] = rank_and_pairs({
                mask: epoch_candidates[mask]["delta"][field] for mask in CANDIDATE_MASKS
            })
        for mask in CANDIDATE_MASKS:
            item = epoch_candidates[mask]
            row: dict[str, Any] = {
                "seed": seed, "epoch": epoch, "stable_row_id": stable,
                "data_identity": identity_text, "candidate_mask": mask,
                "candidate_action_key": item["action"],
                "provenance_authorization": PROVENANCE,
                "tail_state_authorization": (
                    INFERENCE_CANDIDATE if epoch == 20 else DIAGNOSTIC
                ),
                "native_prediction": native_prediction,
                "counterfactual_prediction": item["cf_prediction"],
                "native_score_support": native_scores["SUPPORT"],
                "native_score_not_entitled": native_scores["NOT_ENTITLED"],
                "native_score_refute": native_scores["REFUTE"],
                "counterfactual_score_support": item["cf_scores"]["SUPPORT"],
                "counterfactual_score_not_entitled": item["cf_scores"]["NOT_ENTITLED"],
                "counterfactual_score_refute": item["cf_scores"]["REFUTE"],
                **{f"native_{field}": native_geo[field] for field in MARGIN_FIELDS},
                **{f"counterfactual_{field}": item["cf_geo"][field] for field in MARGIN_FIELDS},
                **item["delta"], **item["flags"],
                "native_entitlement_reserve": item["native_entitlement"],
                "counterfactual_entitlement_reserve": item["cf_entitlement"],
                "entitlement_reserve_change": item["cf_entitlement"] - item["native_entitlement"],
                "native_polarity_reserve": native_geo["support_minus_refute"],
                "counterfactual_polarity_reserve": item["cf_geo"]["support_minus_refute"],
                "polarity_reserve_change": item["delta"]["delta_support_minus_refute"],
                "entitlement_reserve_change_sign": sign(item["cf_entitlement"] - item["native_entitlement"]),
                "entitlement_reserve_direction": (
                    "UNCHANGED" if item["cf_entitlement"] == item["native_entitlement"] else
                    "INCREASES" if item["cf_entitlement"] > item["native_entitlement"] else "DECREASES"
                ),
                "polarity_reserve_change_sign": sign(item["delta"]["delta_support_minus_refute"]),
                "polarity_reserve_direction": (
                    "UNCHANGED" if item["delta"]["delta_support_minus_refute"] == 0 else
                    "INCREASES" if item["delta"]["delta_support_minus_refute"] > 0 else "DECREASES"
                ),
                **{f"{field}_sign_relative_to_native": sign(item["delta"][field]) for field in SIGNED_FIELDS},
                "score_source_boundary": p2_module.SOURCE_BOUNDARY,
                "native_score_dtype": p2_module.NATIVE_SOURCE_DTYPE,
                "counterfactual_score_dtype": p2_module.COUNTERFACTUAL_SOURCE_DTYPE,
            }
            for field in SIGNED_FIELDS:
                mean = sum(epoch_candidates[candidate]["delta"][field] for candidate in CANDIDATE_MASKS) / 3.0
                row[f"{field}_centered_across_candidates"] = item["delta"][field] - mean
                row[f"{field}_candidate_rank"] = order_state[field][2][mask]
                row[f"{field}_pairwise_candidate_ordering"] = order_state[field][1]
            tail_rows.append(row)
            candidate_epoch[(seed, epoch, identity_text, mask)] = row

    trajectories: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in tail_rows:
        trajectories[(row["seed"], row["data_identity"], row["candidate_mask"])].append(row)
    trajectory_ok = (
        len(tail_rows) == 19440 and len(trajectories) == 6480
        and all([row["epoch"] for row in sorted(values, key=lambda item: item["epoch"])] == list(EPOCHS)
                for values in trajectories.values())
    )
    gate(gates, "population", "", "candidate_action_tail_trajectory_closure",
         {"candidate_epoch_rows": 19440, "unique_trajectories": 6480,
          "epochs_per_trajectory": list(EPOCHS)},
         {"candidate_epoch_rows": len(tail_rows), "unique_trajectories": len(trajectories),
          "bad_trajectory_epoch_sets": sum(
              {row["epoch"] for row in values} != set(EPOCHS) for values in trajectories.values())},
         trajectory_ok, "candidate-action trajectory population is incomplete")

    topology_rows: list[dict[str, Any]] = []
    topology_index: dict[tuple[int, str, str, str], dict[str, Any]] = {}
    for trajectory_key in sorted(trajectories):
        values = sorted(trajectories[trajectory_key], key=lambda row: row["epoch"])
        first = values[0]
        native_sequence = [row["native_prediction"] for row in values]
        winner_sequence = [row["counterfactual_prediction"] for row in values]
        bool_sequences = {field: [row[field] for row in values] for field in BOOL_FIELDS}
        decision_crossings = sum(
            left != right for left, right in zip(winner_sequence, winner_sequence[1:])
        )
        for field in SIGNED_FIELDS:
            numeric = [float(row[field]) for row in values]
            signs = [sign(value) for value in numeric]
            row = {
                "seed": first["seed"], "stable_row_id": first["stable_row_id"],
                "data_identity": first["data_identity"],
                "candidate_mask": first["candidate_mask"],
                "candidate_action_key": first["candidate_action_key"],
                "signed_response_coordinate": field,
                "epoch18_value": numeric[0], "epoch19_value": numeric[1],
                "epoch20_value": numeric[2], "tail_minimum": min(numeric),
                "tail_maximum": max(numeric), "tail_range": max(numeric) - min(numeric),
                "tail_mean": sum(numeric) / 3.0,
                "epoch20_minus_epoch19": numeric[2] - numeric[1],
                "epoch19_minus_epoch18": numeric[1] - numeric[0],
                "epoch20_minus_tail_mean": numeric[2] - sum(numeric) / 3.0,
                "epoch18_sign": signs[0], "epoch19_sign": signs[1],
                "epoch20_sign": signs[2],
                "sign_persistence_count": persistence_count(signs),
                "zero_crossing_count": crossing_count(signs),
                "sign_reversal_count": reversal_count(signs),
                "monotonic_direction": monotonic(numeric),
                "final_step_direction": direction(numeric[1], numeric[2]),
                "native_prediction_sequence": native_sequence,
                "counterfactual_prediction_sequence": winner_sequence,
                "counterfactual_winner_persistence": persistence_count(winner_sequence),
                "prediction_changed_sequence": bool_sequences["prediction_changed"],
                "prediction_changed_persistence": persistence_count(bool_sequences["prediction_changed"]),
                "entitlement_transition_sequence": bool_sequences["entitlement_transition"],
                "entitlement_transition_persistence": persistence_count(bool_sequences["entitlement_transition"]),
                "polarity_transition_sequence": bool_sequences["polarity_transition"],
                "polarity_transition_persistence": persistence_count(bool_sequences["polarity_transition"]),
                "polarity_direction_preserved_sequence": bool_sequences["polarity_direction_preserved"],
                "polarity_direction_persistence": persistence_count(bool_sequences["polarity_direction_preserved"]),
                "decision_boundary_crossing_count": decision_crossings,
                "provenance_authorization": PROVENANCE,
                "state_authorization": DIAGNOSTIC,
            }
            topology_rows.append(row)
            topology_index[(first["seed"], first["data_identity"], first["candidate_mask"], field)] = row

    order_rows: list[dict[str, Any]] = []
    order_by_identity: dict[tuple[str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    common_identities = sorted(identity_sets[(SEEDS[0], EPOCHS[0])])
    for seed in SEEDS:
        for identity in common_identities:
            identity_text = data_identity(identity)
            stable = native_epoch[(seed, 18, identity_text)]["stable_row_id"]
            for field in SIGNED_FIELDS:
                ranks, pairs = [], []
                for epoch in EPOCHS:
                    values = {
                        mask: float(candidate_epoch[(seed, epoch, identity_text, mask)][field])
                        for mask in CANDIDATE_MASKS
                    }
                    rank, pair, _ = rank_and_pairs(values)
                    ranks.append(rank)
                    pairs.append(pair)
                row = {
                    "seed": seed, "stable_row_id": stable, "data_identity": identity_text,
                    "signed_response_coordinate": field,
                    "candidate_rank_epoch18": ranks[0], "candidate_rank_epoch19": ranks[1],
                    "candidate_rank_epoch20": ranks[2],
                    "pairwise_ordering_epoch18": pairs[0],
                    "pairwise_ordering_epoch19": pairs[1],
                    "pairwise_ordering_epoch20": pairs[2],
                    "rank_persistence_across_tail": ranks[0] == ranks[1] == ranks[2],
                    "pairwise_ordering_persistence_across_tail": pairs[0] == pairs[1] == pairs[2],
                    "cross_seed_rank_agreement": False,
                    "cross_seed_pairwise_order_agreement": False,
                    "three_way_exact_candidate_order_agreement": False,
                    "provenance_authorization": PROVENANCE,
                    "state_authorization": DIAGNOSTIC,
                }
                order_rows.append(row)
                order_by_identity[(identity_text, field)][seed] = row
    for bucket in order_by_identity.values():
        rank_agreement = len(bucket) == 3 and len({
            canonical([row[f"candidate_rank_epoch{epoch}"] for epoch in EPOCHS])
            for row in bucket.values()
        }) == 1
        pair_agreement = len(bucket) == 3 and len({
            canonical([row[f"pairwise_ordering_epoch{epoch}"] for epoch in EPOCHS])
            for row in bucket.values()
        }) == 1
        full_order = rank_agreement and pair_agreement and all(
            row["rank_persistence_across_tail"]
            and row["pairwise_ordering_persistence_across_tail"]
            for row in bucket.values()
        )
        for row in bucket.values():
            row["cross_seed_rank_agreement"] = rank_agreement
            row["cross_seed_pairwise_order_agreement"] = pair_agreement
            row["three_way_exact_candidate_order_agreement"] = full_order
            row["state_authorization"] = DIAGNOSTIC

    endpoint_rows: list[dict[str, Any]] = []
    endpoint_groups: dict[tuple[str, str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for (seed, identity_text, mask, field), row in topology_index.items():
        endpoint_groups[(identity_text, mask, field)][seed] = row
    order_lookup = {(row["seed"], row["data_identity"], row["signed_response_coordinate"]): row for row in order_rows}
    for (identity_text, mask, field), bucket in sorted(endpoint_groups.items()):
        if set(bucket) != set(SEEDS):
            raise ValueError("cross-seed endpoint identity confusion")
        endpoints = [bucket[seed]["epoch20_value"] for seed in SEEDS]
        sign_sequences = {
            str(seed): [bucket[seed][f"epoch{epoch}_sign"] for epoch in EPOCHS]
            for seed in SEEDS
        }
        monotonic_sequences = {str(seed): bucket[seed]["monotonic_direction"] for seed in SEEDS}
        categorical_sequences = {
            str(seed): {
                "native": bucket[seed]["native_prediction_sequence"],
                "winner": bucket[seed]["counterfactual_prediction_sequence"],
                "prediction_changed": bucket[seed]["prediction_changed_sequence"],
                "entitlement": bucket[seed]["entitlement_transition_sequence"],
                "polarity": bucket[seed]["polarity_transition_sequence"],
                "polarity_preserved": bucket[seed]["polarity_direction_preserved_sequence"],
            } for seed in SEEDS
        }
        endpoint_equal = endpoints[0] == endpoints[1] == endpoints[2]
        sign_agree = len({canonical(value) for value in sign_sequences.values()}) == 1
        monotonic_agree = len(set(monotonic_sequences.values())) == 1
        winner_agree = len({
            canonical(value["winner"]) for value in categorical_sequences.values()
        }) == 1
        transition_agree = len({
            canonical({key: value for key, value in item.items() if key not in ("native", "winner")})
            for item in categorical_sequences.values()
        }) == 1
        categorical_agree = len({canonical(item) for item in categorical_sequences.values()}) == 1
        ranks = [order_lookup[(seed, identity_text, field)] for seed in SEEDS]
        rank_agree = all(row["cross_seed_rank_agreement"] for row in ranks)
        pair_agree = all(row["cross_seed_pairwise_order_agreement"] for row in ranks)
        endpoint_rows.append({
            "stable_row_id": canonical({str(seed): bucket[seed]["stable_row_id"] for seed in SEEDS}),
            "data_identity": identity_text, "candidate_mask": mask,
            "candidate_action_keys": canonical({str(seed): bucket[seed]["candidate_action_key"] for seed in SEEDS}),
            "signed_response_coordinate": field,
            "seed183_endpoint": endpoints[0], "seed184_endpoint": endpoints[1],
            "seed185_endpoint": endpoints[2],
            "endpoint_values_all_equal": endpoint_equal,
            "tail_sign_sequences": sign_sequences,
            "same_sign_sequence_but_different_endpoint_value": sign_agree and not endpoint_equal,
            "monotonic_directions": monotonic_sequences,
            "same_monotonic_direction_but_different_endpoint_value": monotonic_agree and not endpoint_equal,
            "categorical_sequences": categorical_sequences,
            "same_categorical_sequence_but_different_endpoint_margin":
                field in SIGNED_FIELDS[3:]
                and categorical_agree and not endpoint_equal,
            "tail_sign_sequence_agreement": sign_agree,
            "monotonic_direction_agreement": monotonic_agree,
            "counterfactual_winner_sequence_agreement": winner_agree,
            "transition_sequence_agreement": transition_agree,
            "candidate_rank_agreement": rank_agree,
            "pairwise_order_agreement": pair_agree,
            "provenance_authorization": PROVENANCE,
            "state_authorization": DIAGNOSTIC,
        })

    p2_native_index = {(integer(row["seed"], "P2 seed"), row["data_identity"]): row for row in p2_native}
    p2_candidate_index = {
        (integer(row["seed"], "P2 seed"), row["data_identity"], row["candidate_mask"]): row
        for row in p2_candidate
    }
    p2_response_index = {
        (integer(row["seed"], "P2 seed"), row["data_identity"], row["candidate_mask"]): row
        for row in p2_response
    }
    native_score_bad = candidate_score_bad = margin_bad = prediction_bad = categorical_bad = identity_bad = 0
    for row in tail_rows:
        if row["epoch"] != 20:
            continue
        native = p2_native_index[(row["seed"], row["data_identity"])]
        candidate = p2_candidate_index[(row["seed"], row["data_identity"], row["candidate_mask"])]
        response = p2_response_index[(row["seed"], row["data_identity"], row["candidate_mask"])]
        native_score_bad += any(
            float(row[f"native_score_{field}"]) != number(native[f"native_score_{field}"], "P2 native")
            for field in SCORE_FIELDS
        )
        candidate_score_bad += any(
            float(row[f"counterfactual_score_{field}"]) != number(candidate[f"counterfactual_score_{field}"], "P2 cf")
            for field in SCORE_FIELDS
        )
        margin_bad += any(
            float(row[field]) != number(response[field], "P2 response")
            for field in SIGNED_FIELDS
        )
        identity_bad += (
            row["stable_row_id"] != native["stable_row_id"]
            or row["stable_row_id"] != candidate["stable_row_id"]
            or row["candidate_action_key"] != candidate["candidate_action_key"]
        )
        prediction_bad += (
            row["native_prediction"] != native["native_prediction"]
            or row["counterfactual_prediction"] != candidate["counterfactual_prediction"]
        )
        categorical_bad += (
            row["prediction_changed"] != boolean(response["prediction_changed"], "P2 prediction changed")
            or row["entitlement_transition"] != boolean(response["entitlement_branch_changed"], "P2 entitlement transition")
            or row["polarity_transition"] != boolean(response["polarity_branch_changed"], "P2 polarity transition")
            or row["polarity_direction_preserved"] != boolean(response["polarity_direction_preserved"], "P2 polarity preserved")
        )
    endpoint_reproduction = {
        "native_score_disagreements": native_score_bad,
        "counterfactual_score_disagreements": candidate_score_bad,
        "margin_disagreements": margin_bad,
        "prediction_disagreements": prediction_bad,
        "categorical_response_disagreements": categorical_bad,
        "identity_or_action_disagreements": identity_bad,
    }
    endpoint_ok = all(value == 0 for value in endpoint_reproduction.values())
    gate(gates, "reproduction", "P2", "epoch20_exact_endpoint_reproduction",
         {key: 0 for key in endpoint_reproduction}, endpoint_reproduction,
         endpoint_ok, "P4 epoch-20 state differs from P2")

    b4_controlled_path = b4_dir / "stage196b2b4_primitive_coalition_rows.csv"
    controlled, controlled_h = projected_csv(b4_controlled_path, (
        "seed", "epoch", "stable_row_id", "direction", "coalition_mask",
        "counterfactual_refute_logit", "counterfactual_not_entitled_logit",
        "counterfactual_support_logit", "counterfactual_margin", "counterfactual_prediction",
    ))
    controlled_index = {
        (integer(row["seed"], "B4 seed"), integer(row["epoch"], "B4 epoch"),
         row["stable_row_id"], row["coalition_mask"]): row
        for row in controlled
        if row["direction"] == "JOINT_RECIPIENT_FRAME_LOCAL_ONLY_DONOR"
        and integer(row["epoch"], "B4 epoch") in EPOCHS
    }
    controlled_matched = controlled_bad = 0
    for row in tail_rows:
        control = controlled_index.get((
            row["seed"], row["epoch"], row["stable_row_id"], row["candidate_action_key"],
        ))
        if control is None:
            continue
        controlled_matched += 1
        controlled_bad += (
            row["counterfactual_score_refute"] != number(control["counterfactual_refute_logit"], "B4 refute")
            or row["counterfactual_score_not_entitled"] != number(control["counterfactual_not_entitled_logit"], "B4 not entitled")
            or row["counterfactual_score_support"] != number(control["counterfactual_support_logit"], "B4 support")
            or row["counterfactual_margin_support_minus_not_entitled"] != number(control["counterfactual_margin"], "B4 margin")
            or row["counterfactual_prediction"] != control["counterfactual_prediction"]
        )
    gate(gates, "cross-check", "B2-B4", "controlled_tail_numeric_cross_check",
         {"matched_rows_nonzero": True, "disagreements": 0},
         {"matched_rows": controlled_matched, "disagreements": controlled_bad},
         controlled_matched > 0 and controlled_bad == 0,
         "B2-B4 controlled tail coordinates disagree")
    source.append(source_row("B2-B4", b4_controlled_path, len(controlled), "exact controlled coalition cross-check"))

    dictionary = dictionary_rows()
    dictionary_ok = (
        {row["family"] for row in dictionary} >= {
            "A_ABSOLUTE_TAIL_RESPONSE", "B_TRAJECTORY_TOPOLOGY",
            "C_CATEGORICAL_ACTION_RESPONSE_PERSISTENCE",
            "D_CANDIDATE_RELATIVE_ACTION_ORDERING",
            "E_NATIVE_RELATIVE_ACTION_GEOMETRY",
            "F_CROSS_SEED_MECHANISM_AGREEMENT",
        }
        and all(row["allowed_as_gate_feature"] is False for row in dictionary)
    )
    gate(gates, "state", "", "stability_state_dictionary_closure", True,
         {"rows": len(dictionary), "families": sorted({row["family"] for row in dictionary})},
         dictionary_ok, "precommitted stability-state dictionary is incomplete")
    finite_fields = (
        tuple(f"native_score_{field}" for field in SCORE_FIELDS)
        + tuple(f"counterfactual_score_{field}" for field in SCORE_FIELDS)
        + tuple(f"native_{field}" for field in MARGIN_FIELDS)
        + tuple(f"counterfactual_{field}" for field in MARGIN_FIELDS)
        + SIGNED_FIELDS
    )
    finite_ok = all(math.isfinite(float(row[field])) for row in tail_rows for field in finite_fields)
    margin_ok = all(
        row["native_margin_support_minus_not_entitled"] == row["native_score_support"] - row["native_score_not_entitled"]
        and row["native_margin_support_minus_refute"] == row["native_score_support"] - row["native_score_refute"]
        and row["native_margin_refute_minus_not_entitled"] == row["native_score_refute"] - row["native_score_not_entitled"]
        and row["counterfactual_margin_support_minus_not_entitled"] == row["counterfactual_score_support"] - row["counterfactual_score_not_entitled"]
        and row["counterfactual_margin_support_minus_refute"] == row["counterfactual_score_support"] - row["counterfactual_score_refute"]
        and row["counterfactual_margin_refute_minus_not_entitled"] == row["counterfactual_score_refute"] - row["counterfactual_score_not_entitled"]
        for row in tail_rows
    )
    delta_ok = all(
        row["delta_score_support"] == row["counterfactual_score_support"] - row["native_score_support"]
        and row["delta_score_not_entitled"] == row["counterfactual_score_not_entitled"] - row["native_score_not_entitled"]
        and row["delta_score_refute"] == row["counterfactual_score_refute"] - row["native_score_refute"]
        and row["delta_support_minus_not_entitled"] == row["counterfactual_margin_support_minus_not_entitled"] - row["native_margin_support_minus_not_entitled"]
        and row["delta_support_minus_refute"] == row["counterfactual_margin_support_minus_refute"] - row["native_margin_support_minus_refute"]
        and row["delta_refute_minus_not_entitled"] == row["counterfactual_margin_refute_minus_not_entitled"] - row["native_margin_refute_minus_not_entitled"]
        for row in tail_rows
    )
    for name, observed in (
        ("finite_exact_score_fields", finite_ok),
        ("margin_arithmetic_closure", margin_ok),
        ("delta_arithmetic_closure", delta_ok),
    ):
        gate(gates, "numeric", "", name, True, observed, observed, f"{name} failed")

    trajectory_closure = len(topology_rows) == 6480 * len(SIGNED_FIELDS)
    order_closure = len(order_rows) == 2160 * len(SIGNED_FIELDS)
    endpoint_closure = len(endpoint_rows) == 720 * len(CANDIDATE_MASKS) * len(SIGNED_FIELDS)
    gate(gates, "audit", "", "trajectory_topology_audit_closure",
         6480 * len(SIGNED_FIELDS), len(topology_rows), trajectory_closure,
         "trajectory topology audit is incomplete")
    gate(gates, "audit", "", "candidate_relative_order_audit_closure",
         2160 * len(SIGNED_FIELDS), len(order_rows), order_closure,
         "candidate relative order audit is incomplete")
    gate(gates, "audit", "", "endpoint_shift_audit_closure",
         720 * len(CANDIDATE_MASKS) * len(SIGNED_FIELDS), len(endpoint_rows),
         endpoint_closure, "endpoint shift audit is incomplete")

    leakage = [
        {"field_or_category": field, "authorization": PROVENANCE,
         "present_in_row_outputs": field in ("seed", "epoch", "stable_row_id", "data_identity"),
         "permitted_use": "join, grouping, audit, and provenance only",
         "enforcement": "never enters decision state as a feature"}
        for field in ("seed", "epoch", "stable_row_id", "data_identity")
    ] + [
        {"field_or_category": field, "authorization": "PROHIBITED",
         "present_in_row_outputs": False, "permitted_use": "none",
         "enforcement": "not loaded from P0; projected reads discard non-authorized authority columns"}
        for field in PROHIBITED
    ] + [
        {"field_or_category": "epoch20 deterministic action-response state",
         "authorization": INFERENCE_CANDIDATE, "present_in_row_outputs": True,
         "permitted_use": "candidate state definition only; no gate evaluated or promoted",
         "enforcement": "one checkpoint plus deterministic simulation of three candidates"},
        {"field_or_category": "tail trajectory and cross-seed state",
         "authorization": DIAGNOSTIC, "present_in_row_outputs": True,
         "permitted_use": "mechanism diagnosis only",
         "enforcement": "not inference-time feature authorized"},
    ]
    output_headers = set(TAIL_H + DICTIONARY_H + TOPOLOGY_H + ORDER_H + ENDPOINT_H)
    prohibited_header_overlap = sorted(set(PROHIBITED) & output_headers)
    authorization_ok = (
        not prohibited_header_overlap
        and all(row["authorization"] != INFERENCE_CANDIDATE
                for row in leakage if row["field_or_category"] == "tail trajectory and cross-seed state")
    )
    gate(gates, "leakage", "", "authorization_boundary_closure", True,
         {"prohibited_output_headers": prohibited_header_overlap,
          "p0_safety_target_loaded": False, "gate_or_threshold_search": False},
         authorization_ok, "authorization boundary failed")
    gate(gates, "leakage", "", "safety_target_nondependency",
         {"P0 loaded": False, "safety labels used": False},
         {"P0 loaded": False, "safety labels used": False},
         True, "safety-target nondependency failed")
    gate(gates, "leakage", "", "prohibited_field_exclusion", [],
         prohibited_header_overlap, not prohibited_header_overlap,
         "prohibited field appears in output schema")

    instability = {
        "within_tail_sign_reversal": sum(row["sign_reversal_count"] > 0 for row in topology_rows),
        "within_tail_winner_change": sum(
            row["decision_boundary_crossing_count"] > 0
            for row in topology_rows
            if row["signed_response_coordinate"] == SIGNED_FIELDS[0]
        ),
        "within_tail_transition_flag_change": sum(
            row["prediction_changed_persistence"] < 3
            or row["entitlement_transition_persistence"] < 3
            or row["polarity_transition_persistence"] < 3
            or row["polarity_direction_persistence"] < 3
            for row in topology_rows
            if row["signed_response_coordinate"] == SIGNED_FIELDS[0]
        ),
        "cross_seed_sign_sequence_disagreement": sum(
            not row["tail_sign_sequence_agreement"] for row in endpoint_rows
        ),
        "cross_seed_winner_sequence_disagreement": sum(
            not row["counterfactual_winner_sequence_agreement"]
            for row in endpoint_rows
            if row["signed_response_coordinate"] == SIGNED_FIELDS[0]
        ),
        "cross_seed_transition_sequence_disagreement": sum(
            not row["transition_sequence_agreement"]
            for row in endpoint_rows
            if row["signed_response_coordinate"] == SIGNED_FIELDS[0]
        ),
        "cross_seed_monotonic_direction_disagreement": sum(
            not row["monotonic_direction_agreement"] for row in endpoint_rows
        ),
        "within_or_cross_seed_candidate_order_disagreement": sum(
            not row["three_way_exact_candidate_order_agreement"] for row in order_rows
        ),
    }
    relative_by_field = {
        field: all(
            row["three_way_exact_candidate_order_agreement"]
            for row in order_rows if row["signed_response_coordinate"] == field
        )
        for field in SIGNED_FIELDS
    }
    dominant_instability_count = max(instability.values(), default=0)
    dominant_instability_modes = sorted(
        name for name, count in instability.items()
        if count == dominant_instability_count and count > 0
    )
    full_relative = any(relative_by_field.values())
    topology_fully_stable = all(value == 0 for value in instability.values())
    endpoints_differ = any(not row["endpoint_values_all_equal"] for row in endpoint_rows)
    decision, next_stage, decisions = decision_rows(
        full_relative, topology_fully_stable, endpoints_differ, True, True, instability,
    )
    gate(gates, "decision", "", "decision_hierarchy_reachability",
         {"exactly_one_reached": True, "decision_in_hierarchy": True},
         {"reached": sum(bool(row["reached"]) for row in decisions), "decision": decision},
         sum(bool(row["reached"]) for row in decisions) == 1
         and decision in {item[0] for item in DECISIONS},
         "decision hierarchy is unreachable")
    gate(gates, "output", "", "exact_eleven_file_output_closure",
         sorted(OUTPUTS), sorted(OUTPUTS),
         len(OUTPUTS) == len(set(OUTPUTS)) == 11,
         "output declaration is not exactly eleven files")

    mechanism_audits = {
        "endpoint_shift": {
            "same_sign_sequence_different_endpoint": sum(
                row["same_sign_sequence_but_different_endpoint_value"] for row in endpoint_rows),
            "same_monotonic_direction_different_endpoint": sum(
                row["same_monotonic_direction_but_different_endpoint_value"] for row in endpoint_rows),
            "same_categorical_sequence_different_endpoint_margin": sum(
                row["same_categorical_sequence_but_different_endpoint_margin"] for row in endpoint_rows),
        },
        "topology_instability": {
            "counts": instability,
            "dominant_count": dominant_instability_count,
            "dominant_modes": dominant_instability_modes,
        },
        "candidate_relative_invariance": {
            "rank_stable_trajectories": sum(row["rank_persistence_across_tail"] for row in order_rows),
            "pairwise_order_stable_trajectories": sum(
                row["pairwise_ordering_persistence_across_tail"] for row in order_rows),
            "cross_seed_rank_agreement": sum(row["cross_seed_rank_agreement"] for row in order_rows),
            "cross_seed_pairwise_order_agreement": sum(
                row["cross_seed_pairwise_order_agreement"] for row in order_rows),
            "three_way_exact_candidate_order_agreement": sum(
                row["three_way_exact_candidate_order_agreement"] for row in order_rows),
            "full_population_invariant_by_coordinate": relative_by_field,
        },
        "endpoint_authority": endpoint_reproduction,
    }
    analysis = {
        "stage": STAGE, "decision": decision,
        "recommended_next_stage": next_stage, "blocking_reasons": [],
        "current_git_commit": ns.current_git_commit,
        "scientific_scope": {
            "label_blind": True, "P0_safety_targets_loaded": False,
            "safety_gate_evaluated": False, "threshold_search_performed": False,
            "training_performed": False, "selector_changed": False,
            "model_or_checkpoint_loaded": False,
        },
        "p3_closure": {
            "decision": P3_DECISION, "natural_feasible_gate_count": feasible_count,
            "bidirectional_transfer_success_count": transfer_count,
            "joined_safety_rows_loaded": False,
        },
        "p2_endpoint_authority": {
            "decision": P2_DECISION, "required_rows": required_repro,
            "epoch20_reproduction": endpoint_reproduction,
        },
        "recipient_signature_authority": recipient_signature_authority,
        "recipient_action_authority": recipient_action_authority,
        "tail_population": {
            "epochs": list(EPOCHS), "seeds": list(SEEDS),
            "native_epoch_rows": len(states), "candidate_action_epoch_rows": len(tail_rows),
            "unique_trajectories": len(trajectories), "observations_per_trajectory": 3,
        },
        "composer_source_boundary": {
            "identity": p2_module.SOURCE_BOUNDARY,
            "reused_p2_score_mapping": True,
            "reused_b2b6_reconstruct_apply_mask": True,
            "internal_class_order": list(INTERNAL_CLASS_ORDER),
            "serialized_precision": SERIALIZED_PRECISION,
        },
        "stability_state_hierarchy": {
            "A": "absolute tail response; diagnostic only",
            "B": "exact-zero trajectory topology; diagnostic causal state",
            "C": "categorical model-output persistence",
            "D": "candidate-relative exact rank and pairwise ordering",
            "E": "native-relative reserve and centered action geometry",
            "F": "cross-seed mechanism agreement; diagnostic only",
        },
        "mechanism_audits": mechanism_audits,
        "authorization_boundary": {
            "potentially_integration_authorized": [
                "epoch20 signed action deltas", "epoch20 native-relative reserve changes",
                "epoch20 candidate-relative centered deltas", "epoch20 candidate ranks",
                "epoch20 pairwise candidate ordering", "model-output transition flags",
            ],
            "diagnostic_only": [
                "epoch18 and epoch19 values", "tail means and ranges",
                "trajectory sign sequences", "monotonic directions", "tail persistence",
                "cross-seed agreement", "runtime seed", "training epoch",
            ],
            "prohibited": list(PROHIBITED),
            "provenance_marker": PROVENANCE,
        },
        "decision_hierarchy": decisions,
        "exact_outputs": list(OUTPUTS),
    }
    return analysis, {
        "source": source, "tail": tail_rows, "dictionary": dictionary,
        "topology": topology_rows, "order": order_rows, "endpoint": endpoint_rows,
        "leakage": leakage, "decisions": decisions,
    }


def csv_value(value: Any) -> Any:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite CSV value")
        return SERIALIZED_PRECISION % value
    if isinstance(value, (dict, list, tuple)):
        return canonical(value)
    if value is None:
        return ""
    return value


def render_csv(header: Sequence[str], rows: Iterable[dict[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(header), extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: csv_value(row.get(field)) for field in header})
    return buffer.getvalue()


def render_report(analysis: dict[str, Any]) -> str:
    return f"""# Stage196-B2-B6P4 Action-Response Stability-State Design

## Decision

`{analysis["decision"]}`

Recommended next stage: `{analysis["recommended_next_stage"]}`.

Blocking reasons: `{canonical(analysis.get("blocking_reasons", []))}`.

## Scientific scope

P3 found no transferable absolute threshold: it closed with zero feasible
natural-boundary gates and zero bidirectionally transferable one-dimensional
thresholds. P4 does not search another safety gate. P4 is label-blind and does
not load P0 safety targets, correctness, recovery, harm, or any safety-target
category.

P4 reconstructs the exact action-conditioned final composer at epochs 18, 19,
and 20 for seeds 183, 184, and 185. Epoch-20 reconstruction exactly reproduces
P2:

`{canonical(analysis.get("p2_endpoint_authority", {}).get("epoch20_reproduction", {}))}`

## Recipient-signature authority

The explicit external CSV always has its actual SHA256 computed. P2
consumed-artifact provenance is preferred when it supplies a valid nonempty
digest; B2-B5 is secondary; otherwise exact semantic closure is required.
Unavailable hashes are never compared as empty expected digests.

`{canonical(analysis.get("recipient_signature_authority", {}))}`

B2-B6 is the deterministic 6,480-row action authority and P2 independently
reproduces that mapping. B2-B5 pooled intersections are signature-level
constants and provide the strict 48-row primary membership check. B2-B5
transfer intersections are target-row-specific diagnostics keyed by target
`stable_row_id`; same-signature variation across target rows is expected and
never assigns or redefines an action.

`{canonical(analysis.get("recipient_action_authority", {}))}`

## Stability mechanism

Tail trajectory values, signs, monotonic directions, persistence, and
cross-seed agreement are diagnostic-only. Exact-zero is used without epsilon.
Monotonic categories are STRICTLY_INCREASING, NONDECREASING,
STRICTLY_DECREASING, NONINCREASING, CONSTANT, and NON_MONOTONIC, using the
definitions recorded in the state dictionary.

Candidate-relative endpoint state may be inference-authorized when computed
from one checkpoint and deterministic simulation of all three candidates.
Ranks are exact descending tie groups and pairwise relations are GREATER, LESS,
or TIE; ties are never broken by mask order. Candidate-relative ordering is
scientifically distinct from an absolute threshold.

Endpoint shift is distinguished from topology instability by exact cross-seed
comparisons of endpoint values conditional on sign, monotonic, winner, and
transition-sequence agreement:

`{canonical(analysis.get("mechanism_audits", {}))}`

Seed and training epoch are grouping/provenance dimensions, not gate features.
Stable row ID and data identity are also provenance-only. No selector or
training change is made.

## Authorization boundary

Epoch-20 signed deltas, native-relative reserve changes, centered three-action
deltas, candidate ranks/orderings, and model-output transition flags are
candidate state definitions only. P4 evaluates and promotes no safety gate.
Earlier tail state and every cross-seed agreement quantity remain
diagnostic-only.

## Next stage

The recommended next stage follows from the exact stability mechanism found by
the precommitted hierarchy. The result identifies at most a state family, not
a safety gate, safety claim, deployment rule, or selector change.
"""


def blocked_payload(
    ns: argparse.Namespace, gates: list[dict[str, Any]], reason: str,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    analysis = {
        "stage": STAGE, "decision": BLOCKED,
        "recommended_next_stage": BLOCKED_NEXT,
        "blocking_reasons": [reason], "current_git_commit": ns.current_git_commit,
        "scientific_scope": {
            "label_blind": True, "P0_safety_targets_loaded": False,
            "safety_gate_evaluated": False, "threshold_search_performed": False,
            "training_performed": False, "selector_changed": False,
            "model_or_checkpoint_loaded": False,
        },
        "p3_closure": {}, "p2_endpoint_authority": {},
        "recipient_signature_authority": {},
        "recipient_action_authority": {},
        "tail_population": {}, "composer_source_boundary": {},
        "stability_state_hierarchy": {}, "mechanism_audits": {},
        "authorization_boundary": {
            "prohibited": list(PROHIBITED), "provenance_marker": PROVENANCE,
        },
        "decision_hierarchy": [{
            "order": 0, "decision": BLOCKED,
            "condition": "source, schema, identity, reconstruction, or contract failure",
            "observed": reason, "reached": True,
            "recommended_next_stage": BLOCKED_NEXT,
            "scientific_authorization": "NONE",
        }],
        "exact_outputs": list(OUTPUTS),
    }
    leakage = [
        {"field_or_category": field, "authorization": "PROHIBITED",
         "present_in_row_outputs": False, "permitted_use": "none",
         "enforcement": "blocked output preserves nondependency"}
        for field in PROHIBITED
    ]
    return analysis, {
        "source": [], "tail": [], "dictionary": dictionary_rows(),
        "topology": [], "order": [], "endpoint": [], "leakage": leakage,
        "decisions": analysis["decision_hierarchy"],
    }


def write_outputs(
    output: Path, analysis: dict[str, Any],
    tables: dict[str, list[dict[str, Any]]], gates: list[dict[str, Any]],
) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        payloads = {
            OUTPUTS[0]: json.dumps(analysis, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            OUTPUTS[1]: render_report(analysis),
            OUTPUTS[2]: render_csv(SOURCE_H, tables["source"]),
            OUTPUTS[3]: render_csv(TAIL_H, tables["tail"]),
            OUTPUTS[4]: render_csv(DICTIONARY_H, tables["dictionary"]),
            OUTPUTS[5]: render_csv(TOPOLOGY_H, tables["topology"]),
            OUTPUTS[6]: render_csv(ORDER_H, tables["order"]),
            OUTPUTS[7]: render_csv(ENDPOINT_H, tables["endpoint"]),
            OUTPUTS[8]: render_csv(LEAKAGE_H, tables["leakage"]),
            OUTPUTS[9]: render_csv(DECISION_H, tables["decisions"]),
            OUTPUTS[10]: render_csv(CONTRACT_H, gates),
        }
        if set(payloads) != set(OUTPUTS):
            raise ValueError("exact eleven-file payload closure failed")
        for name, content in payloads.items():
            temporary = staging / f".{name}.tmp"
            temporary.write_text(content, encoding="utf-8", newline="")
            os.replace(temporary, staging / name)
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
        write_outputs(ns.output_dir.resolve(), analysis, tables, gates)
        return 0
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        if not any(not row["passed"] for row in gates):
            gate(gates, "exception", "", "unhandled_exception", None, reason,
                 False, reason, fatal=False)
        analysis, tables = blocked_payload(ns, gates, reason)
        try:
            write_outputs(ns.output_dir.resolve(), analysis, tables, gates)
        except FileExistsError:
            pass
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
