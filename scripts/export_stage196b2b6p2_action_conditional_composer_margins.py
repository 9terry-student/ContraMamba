#!/usr/bin/env python3
"""Export exact Stage196-B2-B6P2 action-conditional composer score geometry.

This stage is export-only.  It reuses the frozen B2-B6 primitive application
implementation, never reads the P0 row-safety target CSV, and evaluates no
safety gate, threshold, feature subset, correctness, recovery, or harm.
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


STAGE = "Stage196-B2-B6P2"
SEEDS = (183, 184, 185)
CANDIDATE_MASKS = ("00100000000000", "01000000000000", "10000000000000")
INTERNAL_CLASS_ORDER = ("REFUTE", "NOT_ENTITLED", "SUPPORT")
SEMANTIC_CLASS_ORDER = ("SUPPORT", "NOT_ENTITLED", "REFUTE")
NATIVE_SOURCE_DTYPE = "torch.float32"
COUNTERFACTUAL_SOURCE_DTYPE = "python_float_binary64"
SERIALIZED_PRECISION = "%.17g"
SOURCE_BOUNDARY = (
    "scripts/analyze_stage196b2b6_minimal_selector_intervention.py:"
    "reconstruct/apply_mask over src/contramamba/modeling_v6b_minimal.py:"
    "forward output['logits']=final_logits"
)
SUCCESS = "STAGE196B2B6P2_ACTION_CONDITIONAL_COMPOSER_MARGIN_EXPORT_COMPLETE"
BLOCKED_SOURCE = "STAGE196B2B6P2_BLOCKED_SOURCE_OR_ARTIFACT_FAILURE"
BLOCKED_REPRODUCTION = "STAGE196B2B6P2_BLOCKED_SCORE_REPRODUCTION_FAILURE"
NEXT_SUCCESS = "STAGE196B2B6P3_ACTION_RESPONSE_SAFETY_GATE_DIAGNOSTIC"
NEXT_SOURCE = "STAGE196B2B6P2_REPAIR_SOURCE_OR_ARTIFACT"
NEXT_REPRODUCTION = "STAGE196B2B6P2_REPAIR_COMPOSER_SCORE_EXPORT"

OUTPUTS = (
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
CONTRACT_H = ("scope", "run", "gate", "required", "observed", "passed", "blocking_reason")
SOURCE_H = (
    "stage", "artifact", "path", "required", "loaded", "row_count", "columns",
    "sha256", "purpose", "provenance_authorization",
)
NATIVE_H = (
    "seed", "stable_row_id", "data_identity", "native_prediction", "native_top1_class",
    "native_runner_up_class", "native_top1_runner_up_margin", "native_score_support",
    "native_score_not_entitled", "native_score_refute",
    "native_margin_support_minus_not_entitled", "native_margin_support_minus_refute",
    "native_margin_refute_minus_not_entitled", "score_dtype", "score_source_boundary",
)
CANDIDATE_H = (
    "seed", "stable_row_id", "data_identity", "candidate_mask", "candidate_action_key",
    "native_prediction", "counterfactual_prediction", "counterfactual_top1_class",
    "counterfactual_runner_up_class", "counterfactual_top1_runner_up_margin",
    "counterfactual_score_support", "counterfactual_score_not_entitled",
    "counterfactual_score_refute", "counterfactual_margin_support_minus_not_entitled",
    "counterfactual_margin_support_minus_refute",
    "counterfactual_margin_refute_minus_not_entitled", "score_dtype",
    "score_source_boundary",
)
RESPONSE_H = (
    "seed", "stable_row_id", "data_identity", "candidate_mask", "candidate_action_key",
    "delta_score_support", "delta_score_not_entitled", "delta_score_refute",
    "delta_top1_runner_up_margin", "delta_support_minus_not_entitled",
    "delta_support_minus_refute", "delta_refute_minus_not_entitled",
    "prediction_changed", "entitlement_branch_changed", "polarity_branch_changed",
    "polarity_direction_preserved", "action_response_l1", "action_response_l2",
    "action_response_linf",
)
SUMMARY_H = ("scope", "metric", "required", "observed", "passed", "details")
DECISION_H = ("order", "decision", "condition", "observed", "reached", "recommended_next_stage")

REQUIRED_B5_FILES = (
    "stage196b2b5_analysis.json",
    "stage196b2b5_feature_dictionary.csv",
    "stage196b2b5_row_action_sets.csv",
    "stage196b2b5_recipient_selector_summary.csv",
)
REQUIRED_B6_FILES = (
    "stage196b2b6_analysis.json",
    "stage196b2b6_candidate_feature_subsets.csv",
    "stage196b2b6_signature_action_map.csv",
    "stage196b2b6_clean_dev_signature_audit.csv",
    "stage196b2b6_clean_dev_application_summary.csv",
    "stage196b2b6_contract.csv",
)
PROVENANCE_MARKER = "PROVENANCE_ONLY_NOT_FEATURE_AUTHORIZED"
PROHIBITED_FEATURES = (
    "seed", "stable_row_id", "data_identity", "raw text", "gold label", "correctness",
    "recovery", "harm", "MUST_ALLOW", "MUST_BLOCK", "OPTIONAL",
    "discovery membership", "primary-case membership", "candidate outcome frequency",
)
AUTHORIZED_FIELDS = (
    "native class scores", "native margins", "counterfactual class scores",
    "counterfactual margins", "exact score deltas", "exact margin deltas",
    "model-output transition flags", "explicit action-response norms",
)


class ReproductionFailure(ValueError):
    """Authoritative scores were emitted but did not reproduce predictions."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, kind in (
        ("repo-root", Path),
        ("stage196b2b6p1-analysis-json", Path),
        ("stage196b2b6p0-analysis-json", Path),
        ("stage196b2b6-analysis-json", Path),
        ("stage196b2b5-analysis-json", Path),
        ("stage196b2b4-analysis-json", Path),
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


def boolean(value: Any, name: str) -> bool:
    value = cell(value)
    if type(value) is not bool:
        raise ValueError(f"{name}: boolean required")
    return value


def integer(value: Any, name: str) -> int:
    value = cell(value)
    if type(value) is not int:
        raise ValueError(f"{name}: integer required")
    return value


def number(value: Any, name: str) -> float:
    value = cell(value)
    if type(value) not in (int, float):
        raise ValueError(f"{name}: numeric value required")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name}: finite numeric value required")
    return result


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON object required")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL row")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSON object required")
            rows.append(value)
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_header_count(path: Path) -> tuple[list[str], int]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"{path}: empty CSV") from exc
        return header, sum(1 for _ in reader)


def require_columns(columns: Sequence[str], required: Sequence[str], label: str) -> None:
    missing = sorted(set(required) - set(columns))
    if missing:
        raise ValueError(f"{label}: missing columns {missing}")


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


def git_value(root: Path, args: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args], check=False, capture_output=True,
        text=True, timeout=30,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def authority_hash(source_hashes: dict[str, Any], path: Path, suffix_parts: int = 1) -> str:
    suffix = "/".join(path.parts[-suffix_parts:]).replace("\\", "/")
    matches = [
        str(value) for key, value in source_hashes.items()
        if str(key).replace("\\", "/").endswith(suffix)
    ]
    return matches[0] if len(set(matches)) == 1 else ""


def source_row(
    stage: str, path: Path, purpose: str, provenance: str,
    row_count: int | str = "", columns: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "stage": stage, "artifact": path.name, "path": str(path), "required": True,
        "loaded": True, "row_count": row_count, "columns": list(columns),
        "sha256": sha256(path), "purpose": purpose,
        "provenance_authorization": provenance,
    }


def validate_decision(
    doc: dict[str, Any], decision: str, label: str,
    gates: list[dict[str, Any]], recommended: str | None = None,
) -> None:
    required: dict[str, Any] = {"decision": decision, "blocking_reasons": []}
    if recommended is not None:
        required["recommended_next_stage"] = recommended
    observed = {key: doc.get(key) for key in required}
    gate(
        gates, "authority", label, f"{label.lower()}_decision_closure",
        required, observed, observed == required, f"{label} frozen decision changed",
    )


def import_b6_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("_stage196b2b6_authority", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot construct deterministic B2-B6 source import")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("reconstruct", "apply_mask", "LABELS", "PRIMITIVES", "PRIMITIVE_FIELDS"):
        if not hasattr(module, name):
            raise ValueError(f"B2-B6 source boundary missing {name}")
    return module


def resolve_runtime_root(root: Path, b6: dict[str, Any]) -> Path:
    recorded = Path(str(b6.get("source_paths", {}).get("stage196b2b3p0_run_root", "")))
    candidates = [recorded]
    if recorded.name:
        candidates.append(root / "reports" / recorded.name)
    existing = [candidate.resolve() for candidate in candidates if candidate.is_dir()]
    if len(set(existing)) != 1:
        raise ValueError("frozen B2-B6 runtime root cannot be resolved uniquely")
    return existing[0]


def validate_large_b5(
    path: Path, authority: dict[str, Any], gates: list[dict[str, Any]],
) -> tuple[list[str], int, dict[str, Any]]:
    if not path.is_absolute() or path.name != "stage196b2b5_recipient_signature_rows.csv":
        raise ValueError("large B2-B5 authority must be the exact absolute CLI path")
    header, count = csv_header_count(path)
    required = (
        "feature_family", "feature_subset_mask", "feature_subset_members", "seed",
        "stable_row_id", "signature", "transfer_status",
    )
    require_columns(header, required, "B2-B5 recipient-signature authority")
    expected_hash = authority_hash(authority.get("source_hashes", {}), path)
    actual_hash = sha256(path)
    gate(
        gates, "authority", "B2-B5", "b2b5_large_file_identity",
        {"basename": path.name, "sha256": expected_hash, "explicit_cli_path": True},
        {"basename": path.name, "sha256": actual_hash, "explicit_cli_path": True},
        bool(expected_hash) and actual_hash == expected_hash,
        "explicit B2-B5 recipient-signature CSV hash disagrees with frozen authority",
    )
    seeds: set[int] = set()
    masks: set[str] = set()
    identities: set[tuple[int, str]] = set()
    pooled: Counter[str] = Counter()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["feature_family"] != "recipient_local":
                continue
            seed = integer(row["seed"], "B2-B5 signature seed")
            seeds.add(seed)
            masks.add(row["feature_subset_mask"])
            identities.add((seed, row["stable_row_id"]))
            if row["transfer_status"] == "POOLED":
                pooled[row["feature_subset_mask"]] += 1
    evidence = {
        "rows": count, "recipient_seeds": sorted(seeds),
        "recipient_unique_identities": len(identities),
        "candidate_input_masks_present": sorted(set(CANDIDATE_MASKS) & masks),
        "pooled_rows_by_candidate": {mask: pooled[mask] for mask in CANDIDATE_MASKS},
    }
    passed = (
        seeds == {184, 185} and len(identities) == 16
        and set(CANDIDATE_MASKS) <= masks
        and all(pooled[mask] == 16 for mask in CANDIDATE_MASKS)
    )
    gate(
        gates, "authority", "B2-B5", "b2b5_large_file_schema_provenance_seed_action_closure",
        {
            "recipient_seeds": [184, 185], "recipient_unique_identities": 16,
            "candidate_input_masks_present": list(CANDIDATE_MASKS),
            "pooled_rows_per_candidate": 16,
        },
        evidence, passed, "B2-B5 large recipient-signature authority is incomplete",
    )
    return header, count, evidence


def load_audit_actions(path: Path) -> tuple[dict[tuple[str, int, str, str, int], tuple[str, str]], list[str], int]:
    required = (
        "feature_subset_mask", "seed", "stable_row_id", "id", "source_row_id",
        "dev_position", "assigned_action_set",
    )
    actions: dict[tuple[str, int, str, str, int], tuple[str, str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        require_columns(columns, required, "B2-B6 clean-dev signature audit")
        for row in reader:
            candidate = row["feature_subset_mask"]
            seed = integer(row["seed"], "audit seed")
            position = integer(row["dev_position"], "audit dev_position")
            assigned = cell(row["assigned_action_set"])
            if (
                candidate not in CANDIDATE_MASKS or seed not in SEEDS
                or not isinstance(assigned, list) or len(assigned) != 1
                or not isinstance(assigned[0], str)
                or not re.fullmatch(r"[01]{5}", assigned[0])
            ):
                raise ValueError("B2-B6 candidate-action input is invalid")
            key = (candidate, seed, row["id"], row["source_row_id"], position)
            if key in actions:
                raise ValueError("duplicate B2-B6 candidate-action row")
            actions[key] = (assigned[0], row["stable_row_id"])
    return actions, columns, len(actions)


def validate_composer_row(row: dict[str, Any], seed: int, mode: str) -> tuple[str, str, int]:
    required = (
        "seed", "epoch", "stable_row_id", "id", "source_row_id", "dev_position",
        "gradient_ownership_mode", "native_logit_order", "final_native_prediction",
        "final_refute_logit", "final_not_entitled_logit", "final_support_logit",
    )
    missing = sorted(set(required) - set(row))
    if missing:
        raise ValueError(f"composer row missing fields {missing}")
    identity = (str(row["id"]), str(row["source_row_id"]), integer(row["dev_position"], "dev_position"))
    if (
        integer(row["seed"], "seed") != seed or integer(row["epoch"], "epoch") != 20
        or row["gradient_ownership_mode"] != mode
        or tuple(row["native_logit_order"]) != INTERNAL_CLASS_ORDER
    ):
        raise ValueError("composer row provenance or class order changed")
    for field, value in row.items():
        if (
            field.endswith(("_logit", "_prob", "_energy", "_bias"))
            or field.startswith(("final_", "reconstructed_final_"))
        ) and type(value) in (int, float) and not math.isfinite(float(value)):
            raise ValueError(f"{field}: non-finite composer input")
    return identity


def score_map_from_native(row: dict[str, Any]) -> dict[str, float]:
    return {
        "SUPPORT": number(row["final_support_logit"], "native support"),
        "NOT_ENTITLED": number(row["final_not_entitled_logit"], "native not-entitled"),
        "REFUTE": number(row["final_refute_logit"], "native refute"),
    }


def score_map_from_reconstruction(value: dict[str, Any]) -> dict[str, float]:
    final = value.get("final")
    if not isinstance(final, (tuple, list)) or len(final) != 3:
        raise ValueError("B2-B6 source boundary did not return three final coordinates")
    scores = {label: number(final[index], f"counterfactual {label}") for index, label in enumerate(INTERNAL_CLASS_ORDER)}
    return scores


def geometry(scores: dict[str, float]) -> dict[str, Any]:
    ordered = sorted(INTERNAL_CLASS_ORDER, key=lambda label: (-scores[label], INTERNAL_CLASS_ORDER.index(label)))
    top, runner = ordered[:2]
    return {
        "prediction": top, "top1": top, "runner_up": runner,
        "top1_margin": scores[top] - scores[runner],
        "support_minus_not_entitled": scores["SUPPORT"] - scores["NOT_ENTITLED"],
        "support_minus_refute": scores["SUPPORT"] - scores["REFUTE"],
        "refute_minus_not_entitled": scores["REFUTE"] - scores["NOT_ENTITLED"],
    }


def data_identity(identity: tuple[str, str, int]) -> str:
    return canonical({
        "id": identity[0], "source_row_id": identity[1], "dev_position": identity[2],
    })


def branch_flags(native: str, counterfactual: str) -> dict[str, bool]:
    entitled = {"SUPPORT", "REFUTE"}
    both_entitled = native in entitled and counterfactual in entitled
    return {
        "entitlement_branch_changed": (native in entitled) != (counterfactual in entitled),
        "polarity_branch_changed": both_entitled and native != counterfactual,
        "polarity_direction_preserved": both_entitled and native == counterfactual,
    }


def floats_finite(rows: Iterable[dict[str, Any]], fields: Sequence[str]) -> bool:
    return all(math.isfinite(float(row[field])) for row in rows for field in fields)


def compare_controlled(
    path: Path, candidate_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    controlled = read_csv(path)
    required = (
        "seed", "epoch", "stable_row_id", "direction", "coalition_mask",
        "counterfactual_refute_logit", "counterfactual_not_entitled_logit",
        "counterfactual_support_logit", "counterfactual_margin",
        "counterfactual_prediction",
    )
    if not controlled:
        return {
            "matched_controlled_rows": 0, "score_coordinate_disagreements": 0,
            "margin_disagreements": 0, "prediction_disagreements": 0,
            "maximum_absolute_numeric_difference": 0.0,
            "text_serialization_tolerance": 1e-15,
        }
    require_columns(controlled[0].keys(), required, "B2-B4 controlled rows")
    index: dict[tuple[int, str, str], dict[str, str]] = {}
    for row in controlled:
        if integer(row["epoch"], "controlled epoch") == 20 and row["direction"] == "JOINT_RECIPIENT_FRAME_LOCAL_ONLY_DONOR":
            key = (integer(row["seed"], "controlled seed"), row["stable_row_id"], row["coalition_mask"])
            if key in index:
                raise ValueError("duplicate controlled B2-B4 score row")
            index[key] = row
    matched = score_bad = margin_bad = prediction_bad = 0
    maximum = 0.0
    tolerance = 1e-15
    for row in candidate_rows:
        control = index.get((row["seed"], row["stable_row_id"], row["candidate_action_key"]))
        if control is None:
            continue
        matched += 1
        pairs = (
            (row["counterfactual_score_refute"], number(control["counterfactual_refute_logit"], "controlled refute")),
            (row["counterfactual_score_not_entitled"], number(control["counterfactual_not_entitled_logit"], "controlled not-entitled")),
            (row["counterfactual_score_support"], number(control["counterfactual_support_logit"], "controlled support")),
        )
        differences = [abs(float(left) - right) for left, right in pairs]
        maximum = max([maximum, *differences])
        score_bad += any(value > tolerance for value in differences)
        margin_difference = abs(
            float(row["counterfactual_margin_support_minus_not_entitled"])
            - number(control["counterfactual_margin"], "controlled margin")
        )
        maximum = max(maximum, margin_difference)
        margin_bad += margin_difference > tolerance
        prediction_bad += row["counterfactual_prediction"] != control["counterfactual_prediction"]
    return {
        "matched_controlled_rows": matched,
        "score_coordinate_disagreements": score_bad,
        "margin_disagreements": margin_bad,
        "prediction_disagreements": prediction_bad,
        "maximum_absolute_numeric_difference": maximum,
        "text_serialization_tolerance": tolerance,
    }


def analyze(
    ns: argparse.Namespace, gates: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    root = ns.repo_root.resolve()
    output = ns.output_dir.resolve()
    authority_paths = {
        "p1": ns.stage196b2b6p1_analysis_json.resolve(),
        "p0": ns.stage196b2b6p0_analysis_json.resolve(),
        "b6": ns.stage196b2b6_analysis_json.resolve(),
        "b5": ns.stage196b2b5_analysis_json.resolve(),
        "b4": ns.stage196b2b4_analysis_json.resolve(),
        "b5_large": ns.stage196b2b5_recipient_signature_rows_csv.resolve(),
    }
    explicit = (
        root.is_dir() and all(path.is_absolute() for path in authority_paths.values())
        and all(path.is_file() for path in authority_paths.values())
        and ns.repo_root.is_absolute() and ns.output_dir.is_absolute()
        and bool(re.fullmatch(r"[0-9a-f]{40}", ns.current_git_commit))
    )
    gate(
        gates, "invocation", "", "explicit_authority_paths",
        {"absolute_existing_inputs": True, "commit": "40 lowercase hex"},
        {"absolute_existing_inputs": explicit, "current_git_commit": ns.current_git_commit},
        explicit, "all authorities must be explicit absolute files",
    )
    gate(
        gates, "invocation", "", "output_directory_nonexistence",
        {"exists": False}, {"exists": output.exists()}, not output.exists(),
        "output directory already exists",
    )
    repo_head = git_value(root, ("rev-parse", "--verify", "HEAD^{commit}"))
    gate(
        gates, "source", "", "current_commit_identity",
        ns.current_git_commit, repo_head, repo_head == ns.current_git_commit,
        "repository HEAD differs from --current-git-commit",
    )

    p1, p0, b6, b5, b4 = (
        read_json(authority_paths[name]) for name in ("p1", "p0", "b6", "b5", "b4")
    )
    validate_decision(
        p1, "STAGE196B2B6P1_ACTION_CONDITIONAL_COMPOSER_MARGIN_EXPORT_REQUIRED",
        "B2-B6P1", gates,
        "STAGE196B2B6P2_ACTION_CONDITIONAL_COMPOSER_MARGIN_EXPORT",
    )
    validate_decision(
        p0, "STAGE196B2B6P0_CURRENT_SAFETY_OBSERVABILITY_INSUFFICIENT",
        "B2-B6P0", gates,
    )
    validate_decision(
        b6, "STAGE196B2B6_PRIMARY_EXACT_CLEAN_DEV_UNSAFE", "B2-B6", gates,
    )
    validate_decision(
        b5, "STAGE196B2B5_CROSS_SEED_RECIPIENT_SELECTOR_LOCALIZED", "B2-B5", gates,
    )
    validate_decision(
        b4, "STAGE196B2B4_ROW_SPECIFIC_PRIMITIVE_INTERACTION", "B2-B4", gates,
    )

    source: list[dict[str, Any]] = []
    for label, path in authority_paths.items():
        if label == "b5_large":
            continue
        source.append(source_row(label.upper(), path, "frozen analysis authority", PROVENANCE_MARKER))

    p1_dir, b5_dir, b6_dir, b4_dir, p0_dir = (
        authority_paths["p1"].parent, authority_paths["b5"].parent,
        authority_paths["b6"].parent, authority_paths["b4"].parent,
        authority_paths["p0"].parent,
    )
    p1_inventory_path = p1_dir / "stage196b2b6p1_existing_observability_inventory.csv"
    p1_dictionary_path = p1_dir / "stage196b2b6p1_candidate_state_dictionary.csv"
    p1_matrix_path = p1_dir / "stage196b2b6p1_action_response_authority_matrix.csv"
    p1_inventory = read_csv(p1_inventory_path)
    p1_dictionary = read_csv(p1_dictionary_path)
    p1_matrix = read_csv(p1_matrix_path)
    require_columns(
        p1_inventory[0].keys() if p1_inventory else (),
        ("artifact", "field", "numeric_authority", "action_conditional"),
        "P1 observability inventory",
    )
    require_columns(
        p1_dictionary[0].keys() if p1_dictionary else (),
        ("state_family", "state_name", "authority_status", "integration_status"),
        "P1 state dictionary",
    )
    require_columns(
        p1_matrix[0].keys() if p1_matrix else (),
        (
            "state_family", "source_boundary_identified", "requires_new_export",
            "requires_new_instrumentation",
        ),
        "P1 action-response authority matrix",
    )
    p1_primary_families = {row["state_family"] for row in p1_dictionary} & {"A", "B", "C"}
    p1_matrix_rows = [row for row in p1_matrix if row["state_family"] in {"A", "B", "C"}]
    p1_state_ok = (
        p1_primary_families == {"A", "B", "C"} and len(p1_matrix_rows) == 3
        and all(boolean(row["source_boundary_identified"], "P1 source boundary") for row in p1_matrix_rows)
        and all(boolean(row["requires_new_export"], "P1 export requirement") for row in p1_matrix_rows)
        and not any(boolean(row["requires_new_instrumentation"], "P1 instrumentation") for row in p1_matrix_rows)
    )
    gate(
        gates, "authority", "B2-B6P1", "p1_inventory_and_action_response_authority_closure",
        {"primary_families": ["A", "B", "C"], "export_required": True, "instrumentation_required": False},
        {"primary_families": sorted(p1_primary_families), "matrix_rows": len(p1_matrix_rows)},
        p1_state_ok, "P1 observability inventory or authority matrix changed",
    )
    for p1_path, purpose, rows in (
        (p1_inventory_path, "P1 observability inventory", p1_inventory),
        (p1_dictionary_path, "P1 required state families", p1_dictionary),
        (p1_matrix_path, "P1 action-response authority matrix", p1_matrix),
    ):
        source.append(source_row(
            "B2-B6P1", p1_path, purpose, PROVENANCE_MARKER,
            len(rows), rows[0].keys() if rows else (),
        ))

    for name in REQUIRED_B5_FILES:
        path = b5_dir / name
        if not path.is_file():
            raise ValueError(f"missing B2-B5 authority {name}")
        expected_hash = authority_hash(b6.get("source_hashes", {}), path)
        if not expected_hash or sha256(path) != expected_hash:
            raise ValueError(f"B2-B5 companion hash mismatch: {name}")
        header, count = (csv_header_count(path) if path.suffix == ".csv" else ([], ""))
        source.append(source_row("B2-B5", path, "frozen B2-B5 companion", PROVENANCE_MARKER, count, header))
    b5_header, b5_count, b5_large_evidence = validate_large_b5(authority_paths["b5_large"], b6, gates)
    source.append(source_row(
        "B2-B5", authority_paths["b5_large"], "explicit external large-file authority",
        PROVENANCE_MARKER, b5_count, b5_header,
    ))

    for name in REQUIRED_B6_FILES:
        path = b6_dir / name
        if not path.is_file():
            raise ValueError(f"missing B2-B6 authority {name}")
        expected_hash = authority_hash(p0.get("source_hashes", {}), path)
        if not expected_hash or sha256(path) != expected_hash:
            raise ValueError(f"B2-B6 companion hash mismatch: {name}")
        header, count = (csv_header_count(path) if path.suffix == ".csv" else ([], ""))
        source.append(source_row("B2-B6", path, "candidate-action application authority", PROVENANCE_MARKER, count, header))
    b4_controlled_path = b4_dir / "stage196b2b4_primitive_coalition_rows.csv"
    b4_feature_path = b4_dir / "stage196b2b4_primitive_tail_summary.csv"
    for path in (b4_controlled_path, b4_feature_path):
        expected_hash = authority_hash(b6.get("source_hashes", {}), path)
        if not expected_hash or sha256(path) != expected_hash:
            raise ValueError(f"B2-B4 companion hash mismatch: {path.name}")
        header, count = csv_header_count(path)
        source.append(source_row("B2-B4", path, "primitive/action semantic authority", PROVENANCE_MARKER, count, header))

    p0_contract_path = p0_dir / "stage196b2b6p0_contract.csv"
    p0_contract = read_csv(p0_contract_path)
    count_gate = next((row for row in p0_contract if row.get("gate") == "p0_count_closure"), None)
    observed_count = cell(count_gate["observed"]) if count_gate else {}
    p0_population = {
        "candidate-row applications": observed_count.get("tail paired states"),
        "native recipient rows": (
            b6.get("clean_dev_population_semantics", {}).get("all_clean_dev_count_per_seed", 0)
            * len(SEEDS)
        ),
        "candidate count": len(b6.get("candidate_feature_subsets", [])),
    }
    gate(
        gates, "population", "B2-B6P0", "p0_population_closure",
        {
            "candidate-row applications": 6480,
            "native recipient rows": 2160,
            "candidate count": 3,
        },
        p0_population,
        p0_population == {
            "candidate-row applications": 6480,
            "native recipient rows": 2160,
            "candidate count": 3,
        },
        "P0 frozen population facts changed",
    )
    source.append(source_row("B2-B6P0", p0_contract_path, "population facts only; target CSV not opened", PROVENANCE_MARKER, len(p0_contract), p0_contract[0].keys()))

    observed_masks = tuple(sorted(
        row.get("feature_subset_mask", "") for row in b6.get("candidate_feature_subsets", [])
    ))
    gate(
        gates, "action", "B2-B6", "candidate_mask_closure",
        list(CANDIDATE_MASKS), list(observed_masks),
        observed_masks == CANDIDATE_MASKS,
        "B2-B6 candidate-mask set changed",
    )
    b5_dictionary = read_csv(b5_dir / "stage196b2b5_feature_dictionary.csv")
    dictionary_names = {row.get("feature_name") for row in b5_dictionary}
    candidate_members = {
        row["feature_subset_mask"]: tuple(row.get("feature_subset_members", []))
        for row in b6.get("candidate_feature_subsets", [])
    }
    semantic_ok = all(
        len(candidate_members.get(mask, ())) == 1
        and candidate_members[mask][0] in dictionary_names
        for mask in CANDIDATE_MASKS
    )
    gate(
        gates, "action", "B2-B5/B2-B6", "candidate_action_semantic_closure",
        {"candidate_masks": list(CANDIDATE_MASKS), "members_from_b2b5_dictionary": True},
        {"candidate_masks": sorted(candidate_members), "members": candidate_members},
        semantic_ok, "candidate masks no longer resolve through B2-B5 semantics",
    )

    model_path = root / "src" / "contramamba" / "modeling_v6b_minimal.py"
    head_path = root / "src" / "contramamba" / "heads" / "entitlement_decision.py"
    b6_script = root / "scripts" / "analyze_stage196b2b6_minimal_selector_intervention.py"
    for path in (model_path, head_path, b6_script):
        if not path.is_file():
            raise ValueError(f"source boundary file missing: {path}")
        source.append(source_row("source", path, "exact composer/action implementation", "SOURCE_AUTHORITY"))
    model_text = model_path.read_text(encoding="utf-8")
    head_text = head_path.read_text(encoding="utf-8")
    source_identity_ok = all((
        'base_logits = decision["logits"]' in model_text,
        "final_logits = base_logits" in model_text,
        '"logits": final_logits' in model_text,
        "final_logits.argmax" in model_text,
        "[refute_logit, not_entitled_logit, support_logit]" in head_text,
    ))
    p1_boundary = p1.get("composer_source_boundary", {})
    p1_boundary_ok = p1_boundary.get("final_scores_computed_and_returned") is True
    gate(
        gates, "source", "", "source_boundary_identity",
        {
            "p1_final_scores_computed_and_returned": True,
            "internal_class_order": list(INTERNAL_CLASS_ORDER),
            "boundary": SOURCE_BOUNDARY,
        },
        {
            "p1_final_scores_computed_and_returned": p1_boundary_ok,
            "internal_class_order": list(INTERNAL_CLASS_ORDER),
            "static_source_matches": source_identity_ok,
            "boundary": SOURCE_BOUNDARY,
        },
        p1_boundary_ok and source_identity_ok,
        "exact final composer source boundary cannot be identified",
    )
    module = import_b6_module(b6_script)
    class_ok = tuple(module.LABELS) == INTERNAL_CLASS_ORDER
    gate(
        gates, "source", "", "class_order_authority",
        {
            "internal": list(INTERNAL_CLASS_ORDER),
            "semantic_export_fields": list(SEMANTIC_CLASS_ORDER),
        },
        {
            "b2b6_labels": list(module.LABELS),
            "semantic_export_fields": list(SEMANTIC_CLASS_ORDER),
        },
        class_ok, "B2-B6 class-coordinate order changed",
    )
    expected_primitives = (
        "FRAME", "PREDICATE", "SUFFICIENCY", "POSITIVE_ENERGY", "NEGATIVE_ENERGY",
    )
    primitive_ok = (
        tuple(module.PRIMITIVES) == expected_primitives
        and set(module.PRIMITIVE_FIELDS) == set(expected_primitives)
    )
    gate(
        gates, "action", "B2-B4/B2-B6", "b2b4_action_semantic_closure",
        {"primitive_order": list(expected_primitives), "mask_width": 5},
        {"primitive_order": list(module.PRIMITIVES), "mask_width": 5},
        primitive_ok, "frozen primitive order or action fields changed",
    )

    runtime_root = resolve_runtime_root(root, b6)
    actions, audit_columns, action_count = load_audit_actions(
        b6_dir / "stage196b2b6_clean_dev_signature_audit.csv"
    )
    gate(
        gates, "population", "", "candidate_action_unique_key_closure",
        {"rows": 6480, "unique_keys": 6480},
        {"rows": action_count, "unique_keys": len(actions)},
        action_count == len(actions) == 6480,
        "candidate-action authority does not have 6,480 unique rows",
    )

    states: dict[tuple[int, tuple[str, str, int]], dict[str, dict[str, Any]]] = {}
    source_hashes = b6.get("source_hashes", {})
    for seed in SEEDS:
        for mode in ("joint", "frame_local_only"):
            path = (
                runtime_root / f"seed{seed}_{mode}" / "composer_inputs"
                / "stage196b2b3p0_epoch_composer_inputs_020.jsonl"
            )
            rows = read_jsonl(path)
            expected_hash = authority_hash(source_hashes, path, 4)
            actual_hash = sha256(path)
            if not expected_hash or expected_hash != actual_hash:
                raise ValueError(f"{path}: frozen B2-B6 runtime hash mismatch")
            if len(rows) != 720:
                raise ValueError(f"{path}: expected 720 composer rows")
            seen: set[tuple[str, str, int]] = set()
            for row in rows:
                identity = validate_composer_row(row, seed, mode)
                if identity in seen:
                    raise ValueError(f"{path}: duplicate data identity")
                seen.add(identity)
                states.setdefault((seed, identity), {})[mode] = row
            source.append(source_row(
                "B2-B3P0", path, "epoch-20 exact composer input authority",
                PROVENANCE_MARKER, len(rows), rows[0].keys(),
            ))
    per_seed = Counter(seed for seed, _ in states)
    cross_seed_sets = {
        seed: {identity for row_seed, identity in states if row_seed == seed}
        for seed in SEEDS
    }
    state_ok = (
        len(states) == 2160 and per_seed == Counter({183: 720, 184: 720, 185: 720})
        and all(set(value) == {"joint", "frame_local_only"} for value in states.values())
        and cross_seed_sets[183] == cross_seed_sets[184] == cross_seed_sets[185]
    )
    gate(
        gates, "population", "", "native_unique_key_closure",
        {"rows": 2160, "per_seed": {"183": 720, "184": 720, "185": 720}, "same_identity_set": True},
        {
            "rows": len(states), "per_seed": {str(key): value for key, value in sorted(per_seed.items())},
            "same_identity_set": cross_seed_sets[183] == cross_seed_sets[184] == cross_seed_sets[185],
        },
        state_ok, "native composer population is not exactly 3 x 720",
    )

    native_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []
    native_by_key: dict[tuple[int, tuple[str, str, int]], tuple[dict[str, float], dict[str, Any]]] = {}
    native_disagreements = 0
    for key in sorted(states):
        seed, identity = key
        joint = states[key]["joint"]
        scores = score_map_from_native(joint)
        geo = geometry(scores)
        authoritative = str(joint["final_native_prediction"])
        native_disagreements += geo["prediction"] != authoritative
        stable = str(joint["stable_row_id"])
        native_by_key[key] = (scores, geo)
        native_rows.append({
            "seed": seed, "stable_row_id": stable, "data_identity": data_identity(identity),
            "native_prediction": authoritative, "native_top1_class": geo["top1"],
            "native_runner_up_class": geo["runner_up"],
            "native_top1_runner_up_margin": geo["top1_margin"],
            "native_score_support": scores["SUPPORT"],
            "native_score_not_entitled": scores["NOT_ENTITLED"],
            "native_score_refute": scores["REFUTE"],
            "native_margin_support_minus_not_entitled": geo["support_minus_not_entitled"],
            "native_margin_support_minus_refute": geo["support_minus_refute"],
            "native_margin_refute_minus_not_entitled": geo["refute_minus_not_entitled"],
            "score_dtype": NATIVE_SOURCE_DTYPE, "score_source_boundary": SOURCE_BOUNDARY,
        })

    counterfactual_disagreements = 0
    categorical_disagreements = 0
    masks_per_recipient: defaultdict[tuple[int, tuple[str, str, int]], set[str]] = defaultdict(set)
    for action_key in sorted(actions):
        candidate_mask, seed, identity_name, source_id, position = action_key
        identity = (identity_name, source_id, position)
        state_key = (seed, identity)
        if state_key not in states:
            raise ValueError("B2-B6 action row has no exact recipient composer state")
        joint, donor = states[state_key]["joint"], states[state_key]["frame_local_only"]
        if str(joint["stable_row_id"]) == "":
            raise ValueError("empty stable_row_id")
        primitive_action, audit_stable_row_id = actions[action_key]
        if audit_stable_row_id != str(joint["stable_row_id"]):
            raise ValueError("B2-B6 action row stable_row_id disagrees with composer state")
        reconstructed = module.apply_mask(joint, donor, primitive_action)
        cf_scores = score_map_from_reconstruction(reconstructed)
        cf_geo = geometry(cf_scores)
        native_scores, native_geo = native_by_key[state_key]
        authoritative_cf = str(reconstructed.get("prediction", ""))
        counterfactual_disagreements += cf_geo["prediction"] != authoritative_cf
        native_prediction = str(joint["final_native_prediction"])
        prediction_changed = native_prediction != authoritative_cf
        categorical_disagreements += prediction_changed != (native_prediction != cf_geo["prediction"])
        stable = str(joint["stable_row_id"])
        masks_per_recipient[state_key].add(candidate_mask)
        candidate_rows.append({
            "seed": seed, "stable_row_id": stable, "data_identity": data_identity(identity),
            "candidate_mask": candidate_mask, "candidate_action_key": primitive_action,
            "native_prediction": native_prediction,
            "counterfactual_prediction": authoritative_cf,
            "counterfactual_top1_class": cf_geo["top1"],
            "counterfactual_runner_up_class": cf_geo["runner_up"],
            "counterfactual_top1_runner_up_margin": cf_geo["top1_margin"],
            "counterfactual_score_support": cf_scores["SUPPORT"],
            "counterfactual_score_not_entitled": cf_scores["NOT_ENTITLED"],
            "counterfactual_score_refute": cf_scores["REFUTE"],
            "counterfactual_margin_support_minus_not_entitled": cf_geo["support_minus_not_entitled"],
            "counterfactual_margin_support_minus_refute": cf_geo["support_minus_refute"],
            "counterfactual_margin_refute_minus_not_entitled": cf_geo["refute_minus_not_entitled"],
            "score_dtype": COUNTERFACTUAL_SOURCE_DTYPE, "score_source_boundary": SOURCE_BOUNDARY,
        })
        deltas = {label: cf_scores[label] - native_scores[label] for label in SEMANTIC_CLASS_ORDER}
        flags = branch_flags(native_prediction, authoritative_cf)
        response_rows.append({
            "seed": seed, "stable_row_id": stable, "data_identity": data_identity(identity),
            "candidate_mask": candidate_mask, "candidate_action_key": primitive_action,
            "delta_score_support": deltas["SUPPORT"],
            "delta_score_not_entitled": deltas["NOT_ENTITLED"],
            "delta_score_refute": deltas["REFUTE"],
            "delta_top1_runner_up_margin": cf_geo["top1_margin"] - native_geo["top1_margin"],
            "delta_support_minus_not_entitled": (
                cf_geo["support_minus_not_entitled"] - native_geo["support_minus_not_entitled"]
            ),
            "delta_support_minus_refute": cf_geo["support_minus_refute"] - native_geo["support_minus_refute"],
            "delta_refute_minus_not_entitled": (
                cf_geo["refute_minus_not_entitled"] - native_geo["refute_minus_not_entitled"]
            ),
            "prediction_changed": prediction_changed, **flags,
            "action_response_l1": sum(abs(value) for value in deltas.values()),
            "action_response_l2": math.sqrt(sum(value * value for value in deltas.values())),
            "action_response_linf": max(abs(value) for value in deltas.values()),
        })

    three_masks_ok = (
        len(masks_per_recipient) == 2160
        and all(value == set(CANDIDATE_MASKS) for value in masks_per_recipient.values())
    )
    gate(
        gates, "population", "", "three_exact_masks_per_recipient",
        {"recipients": 2160, "masks": list(CANDIDATE_MASKS)},
        {
            "recipients": len(masks_per_recipient),
            "bad_recipients": sum(value != set(CANDIDATE_MASKS) for value in masks_per_recipient.values()),
        },
        three_masks_ok, "recipient candidate-mask coverage is incomplete",
    )

    native_score_fields = (
        "native_score_support", "native_score_not_entitled", "native_score_refute",
        "native_top1_runner_up_margin", "native_margin_support_minus_not_entitled",
        "native_margin_support_minus_refute", "native_margin_refute_minus_not_entitled",
    )
    candidate_score_fields = (
        "counterfactual_score_support", "counterfactual_score_not_entitled",
        "counterfactual_score_refute", "counterfactual_top1_runner_up_margin",
        "counterfactual_margin_support_minus_not_entitled",
        "counterfactual_margin_support_minus_refute",
        "counterfactual_margin_refute_minus_not_entitled",
    )
    response_numeric_fields = tuple(
        name for name in RESPONSE_H
        if name.startswith("delta_") or name.startswith("action_response_")
    )
    native_complete = len(native_rows) == 2160 and all(
        all(field in row for field in native_score_fields) for row in native_rows
    )
    candidate_complete = len(candidate_rows) == 6480 and all(
        all(field in row for field in candidate_score_fields) for row in candidate_rows
    )
    gate(
        gates, "numeric", "", "native_score_completeness",
        {"rows": 2160, "complete": True},
        {"rows": len(native_rows), "complete": native_complete},
        native_complete, "native score export is incomplete",
    )
    gate(
        gates, "numeric", "", "counterfactual_score_completeness",
        {"rows": 6480, "complete": True},
        {"rows": len(candidate_rows), "complete": candidate_complete},
        candidate_complete, "counterfactual score export is incomplete",
    )
    finite_ok = (
        floats_finite(native_rows, native_score_fields)
        and floats_finite(candidate_rows, candidate_score_fields)
        and floats_finite(response_rows, response_numeric_fields)
    )
    gate(
        gates, "numeric", "", "finite_numeric_values",
        True, finite_ok, finite_ok, "NaN or infinity found in exported numeric state",
    )
    margin_ok = all(
        row["native_margin_support_minus_not_entitled"]
        == row["native_score_support"] - row["native_score_not_entitled"]
        and row["native_margin_support_minus_refute"]
        == row["native_score_support"] - row["native_score_refute"]
        and row["native_margin_refute_minus_not_entitled"]
        == row["native_score_refute"] - row["native_score_not_entitled"]
        for row in native_rows
    ) and all(
        row["counterfactual_margin_support_minus_not_entitled"]
        == row["counterfactual_score_support"] - row["counterfactual_score_not_entitled"]
        and row["counterfactual_margin_support_minus_refute"]
        == row["counterfactual_score_support"] - row["counterfactual_score_refute"]
        and row["counterfactual_margin_refute_minus_not_entitled"]
        == row["counterfactual_score_refute"] - row["counterfactual_score_not_entitled"]
        for row in candidate_rows
    )
    gate(
        gates, "numeric", "", "margin_arithmetic_closure",
        {"disagreements": 0}, {"disagreements": 0 if margin_ok else 1},
        margin_ok, "named margin arithmetic does not close",
    )
    candidate_index = {
        (row["candidate_mask"], row["seed"], row["data_identity"]): row
        for row in candidate_rows
    }
    native_index = {(row["seed"], row["data_identity"]): row for row in native_rows}
    delta_ok = True
    for row in response_rows:
        candidate = candidate_index[(row["candidate_mask"], row["seed"], row["data_identity"])]
        native = native_index[(row["seed"], row["data_identity"])]
        for suffix in ("support", "not_entitled", "refute"):
            delta_ok &= (
                row[f"delta_score_{suffix}"]
                == candidate[f"counterfactual_score_{suffix}"] - native[f"native_score_{suffix}"]
            )
    gate(
        gates, "numeric", "", "delta_arithmetic_closure",
        {"disagreements": 0}, {"disagreements": 0 if delta_ok else 1},
        delta_ok, "class-coordinate delta arithmetic does not close",
    )

    controlled = compare_controlled(b4_controlled_path, candidate_rows)
    controlled_ok = (
        controlled["score_coordinate_disagreements"] == 0
        and controlled["margin_disagreements"] == 0
        and controlled["prediction_disagreements"] == 0
    )
    gate(
        gates, "cross-check", "B2-B4", "controlled_row_score_cross_check",
        {
            "score_coordinate_disagreements": 0,
            "margin_disagreements": 0,
            "prediction_disagreements": 0,
        },
        controlled, controlled_ok,
        "controlled B2-B4 score coordinates disagree with P2 export",
        fatal=False,
    )
    if not controlled_ok:
        raise ReproductionFailure(canonical(controlled))

    stored_applications = read_csv(
        b6_dir / "stage196b2b6_clean_dev_application_summary.csv"
    )
    require_columns(
        stored_applications[0].keys() if stored_applications else (),
        (
            "feature_subset_mask", "seed", "population", "policy_action_mode",
            "row_count", "prediction_change_count",
        ),
        "B2-B6 application summary",
    )
    stored_change_counts = {
        (row["feature_subset_mask"], integer(row["seed"], "application seed")):
        integer(row["prediction_change_count"], "stored prediction-change count")
        for row in stored_applications
        if row["feature_subset_mask"] in CANDIDATE_MASKS
        and row["population"] == "ALL_720"
        and row["policy_action_mode"] == "UNIQUE_DETERMINISTIC"
    }
    exported_change_counts = Counter(
        (row["candidate_mask"], row["seed"])
        for row in response_rows if row["prediction_changed"]
    )
    expected_application_keys = {
        (candidate, seed) for candidate in CANDIDATE_MASKS for seed in SEEDS
    }
    categorical_disagreements += sum(
        stored_change_counts.get(key) != exported_change_counts.get(key, 0)
        for key in expected_application_keys
    )
    categorical_disagreements += len(set(stored_change_counts) - expected_application_keys)
    reproduction = {
        "native_prediction_disagreements": native_disagreements,
        "counterfactual_prediction_disagreements": counterfactual_disagreements,
        "categorical_response_disagreements": categorical_disagreements,
        "stored_b2b6_prediction_change_groups": len(stored_change_counts),
    }
    for name, value in reproduction.items():
        required_value = 9 if name == "stored_b2b6_prediction_change_groups" else 0
        gate(
            gates, "reproduction", "", name, required_value, value,
            value == required_value, f"{name} disagrees", fatal=False,
        )
    if (
        native_disagreements or counterfactual_disagreements
        or categorical_disagreements or len(stored_change_counts) != 9
    ):
        raise ReproductionFailure(canonical(reproduction))

    leakage_ok = (
        not any(field in NATIVE_H + CANDIDATE_H + RESPONSE_H for field in (
            "gold_label", "correctness", "recovery", "harm",
        ))
        and set(("seed", "stable_row_id", "data_identity")) <= set(NATIVE_H)
    )
    gate(
        gates, "leakage", "", "leakage_boundary_closure",
        {
            "provenance_marker": PROVENANCE_MARKER,
            "authorized": list(AUTHORIZED_FIELDS),
            "prohibited": list(PROHIBITED_FEATURES),
            "safety_target_csv_loaded": False,
        },
        {
            "provenance_marker": PROVENANCE_MARKER,
            "authorized": list(AUTHORIZED_FIELDS),
            "prohibited": list(PROHIBITED_FEATURES),
            "safety_target_csv_loaded": False,
        },
        leakage_ok, "leakage boundary is incomplete",
    )
    gate(
        gates, "scope", "", "no_safety_analysis",
        {
            "safety_gate_evaluated": False, "threshold_fitted": False,
            "feature_subset_enumeration_performed": False,
            "p0_row_safety_target_csv_loaded": False,
        },
        {
            "safety_gate_evaluated": False, "threshold_fitted": False,
            "feature_subset_enumeration_performed": False,
            "p0_row_safety_target_csv_loaded": False,
        },
        True, "P2 scope expanded into safety analysis",
    )
    gate(
        gates, "output", "", "exact_nine_file_output_closure",
        sorted(OUTPUTS), sorted(OUTPUTS),
        len(OUTPUTS) == len(set(OUTPUTS)) == 9,
        "output declaration is not exactly nine files",
    )

    summaries = [
        {"scope": "population", "metric": "native rows", "required": 2160, "observed": len(native_rows), "passed": len(native_rows) == 2160, "details": "3 seeds x 720"},
        {"scope": "population", "metric": "candidate-action rows", "required": 6480, "observed": len(candidate_rows), "passed": len(candidate_rows) == 6480, "details": "2160 recipients x 3 masks"},
        {"scope": "population", "metric": "seeds", "required": list(SEEDS), "observed": sorted(per_seed), "passed": sorted(per_seed) == list(SEEDS), "details": ""},
        {"scope": "reproduction", "metric": "native prediction disagreements", "required": 0, "observed": native_disagreements, "passed": native_disagreements == 0, "details": "argmax exact native scores"},
        {"scope": "reproduction", "metric": "counterfactual prediction disagreements", "required": 0, "observed": counterfactual_disagreements, "passed": counterfactual_disagreements == 0, "details": "argmax exact counterfactual scores"},
        {"scope": "reproduction", "metric": "categorical response disagreements", "required": 0, "observed": categorical_disagreements, "passed": categorical_disagreements == 0, "details": "prediction_changed"},
        {"scope": "controlled", "metric": "matched controlled rows", "required": "reported; zero overlap allowed", "observed": controlled["matched_controlled_rows"], "passed": controlled_ok, "details": controlled},
    ]
    analysis = {
        "stage": STAGE, "decision": SUCCESS, "recommended_next_stage": NEXT_SUCCESS,
        "blocking_reasons": [], "current_git_commit": ns.current_git_commit,
        "population_closure": {
            "seeds": list(SEEDS), "recipient_rows_per_seed": 720,
            "native_rows": len(native_rows), "candidate_masks": list(CANDIDATE_MASKS),
            "candidate_action_rows": len(candidate_rows),
        },
        "composer_source_boundary": {
            "identity": SOURCE_BOUNDARY,
            "model_function": "ContraMambaV6BMinimal.forward",
            "returned_coordinate": "output['logits'] == final_logits",
            "counterfactual_application_functions": [
                "analyze_stage196b2b6_minimal_selector_intervention.reconstruct",
                "analyze_stage196b2b6_minimal_selector_intervention.apply_mask",
            ],
            "new_instrumentation_required": False,
        },
        "class_coordinate_authority": {
            "internal_order": list(INTERNAL_CLASS_ORDER),
            "semantic_export_field_order": list(SEMANTIC_CLASS_ORDER),
            "mapping": {"0": "REFUTE", "1": "NOT_ENTITLED", "2": "SUPPORT"},
        },
        "numeric_fidelity": {
            "native_source_dtype": NATIVE_SOURCE_DTYPE,
            "counterfactual_source_dtype": COUNTERFACTUAL_SOURCE_DTYPE,
            "serialized_precision": SERIALIZED_PRECISION,
            "exactness_definition": "native coordinates preserve serialized torch.float32 final_logits; counterfactual coordinates are exact relative to the binary64 B2-B6 deterministic action-composer boundary; both use lossless 17-significant-digit CSV serialization",
            "float64_to_float32_cast_performed": False,
            "bitwise_exactness_across_dtype_conversion_claimed": False,
        },
        "definitions": {
            "delta_score": "counterfactual score - native score",
            "support_minus_not_entitled": "score_support - score_not_entitled",
            "support_minus_refute": "score_support - score_refute",
            "refute_minus_not_entitled": "score_refute - score_not_entitled",
            "delta_top1_runner_up_margin": "counterfactual top1-runner-up margin - native top1-runner-up margin",
            "entitlement_branch_changed": "native is in {SUPPORT,REFUTE} XOR counterfactual is in {SUPPORT,REFUTE}",
            "polarity_branch_changed": "both outputs are in {SUPPORT,REFUTE} and their classes differ",
            "polarity_direction_preserved": "both outputs are in {SUPPORT,REFUTE} and their classes are equal",
            "action_response_l1": "sum of absolute class-coordinate deltas",
            "action_response_l2": "Euclidean norm of the three class-coordinate deltas",
            "action_response_linf": "maximum absolute class-coordinate delta",
        },
        "prediction_reproduction": reproduction,
        "controlled_row_cross_check": controlled,
        "b2b5_large_file_validation": b5_large_evidence,
        "leakage_boundary": {
            "provenance_fields": {
                "seed": PROVENANCE_MARKER, "stable_row_id": PROVENANCE_MARKER,
                "data_identity": PROVENANCE_MARKER,
            },
            "integration_authorized": list(AUTHORIZED_FIELDS),
            "prohibited_feature_inputs": list(PROHIBITED_FEATURES),
        },
        "scope_prohibitions": {
            "p0_row_safety_target_csv_loaded": False, "gold_labels_used": False,
            "safety_gate_evaluated": False, "threshold_fitted": False,
            "feature_subset_enumeration_performed": False, "training_performed": False,
            "model_or_checkpoint_loaded": False,
        },
        "exact_outputs": list(OUTPUTS),
    }
    decisions = [
        {"order": 1, "decision": BLOCKED_SOURCE, "condition": "authoritative input or exact source boundary missing", "observed": False, "reached": False, "recommended_next_stage": NEXT_SOURCE},
        {"order": 2, "decision": BLOCKED_REPRODUCTION, "condition": "scores emitted but prediction reproduction disagrees", "observed": False, "reached": False, "recommended_next_stage": NEXT_REPRODUCTION},
        {"order": 3, "decision": SUCCESS, "condition": "population, source, score, margin, and reproduction contracts pass", "observed": True, "reached": True, "recommended_next_stage": NEXT_SUCCESS},
    ]
    return analysis, {
        "source": source, "native": native_rows, "candidate": candidate_rows,
        "response": response_rows, "summary": summaries, "decision": decisions,
    }


def csv_value(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite value cannot be serialized")
        return SERIALIZED_PRECISION % value
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
    writer = csv.DictWriter(buffer, fieldnames=list(header), extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: csv_value(row.get(name)) for name in header})
    return buffer.getvalue()


def render_report(analysis: dict[str, Any]) -> str:
    return f"""# Stage196-B2-B6P2 Action-Conditional Composer Margin Export

## Decision

`{analysis["decision"]}`

Recommended next stage: `{analysis["recommended_next_stage"]}`.

## Scope and result

B2-B6P1 identified an existing export boundary, not a new instrumentation
requirement. P2 exports exact full-population action-conditioned score geometry:
2,160 native recipient rows and 6,480 candidate-action rows. No safety labels
were used, no safety gate was evaluated, and no threshold was fitted.

Exported-score argmax must reproduce every authoritative native and
counterfactual prediction, and categorical response must reproduce exactly; any
disagreement blocks the run. The observed audit is:

`{canonical(analysis.get("prediction_reproduction", {}))}`

## Composer and class-coordinate authority

The reused source boundary is `{analysis.get("composer_source_boundary", {}).get("identity", "")}`.
The validated internal class order is `REFUTE, NOT_ENTITLED, SUPPORT`, mapped
explicitly to semantic export fields `SUPPORT, NOT_ENTITLED, REFUTE`; coordinates
are never silently reordered.

## Exact geometry

For every class, delta score is counterfactual minus native. Pairwise margins
are SUPPORT-minus-NOT_ENTITLED, SUPPORT-minus-REFUTE, and
REFUTE-minus-NOT_ENTITLED. The top-1 delta is the counterfactual top1-minus-
runner-up margin minus its native counterpart. L1, L2, and L-infinity norms are
supplemental diagnostics and do not replace the three coordinate deltas.

Entitlement transition means movement between NOT_ENTITLED and the entitled
set {{SUPPORT, REFUTE}}. Polarity transition means differing SUPPORT/REFUTE
classes when both outputs are entitled. Polarity direction is preserved when
both are entitled and their SUPPORT/REFUTE class is unchanged. These flags use
model outputs only.

## Controlled-row cross-check

`{canonical(analysis.get("controlled_row_cross_check", {}))}`

Controlled overlap is reported but is not required to cover the full
population.

## Numerical fidelity

`{canonical(analysis.get("numeric_fidelity", {}))}`

Numeric fields are serialized with 17 significant digits and are never rounded
before serialization. NaN and infinity fail closed.

## Leakage boundary

`seed`, `stable_row_id`, and `data_identity` are
`{PROVENANCE_MARKER}`. They are not feature-authorized. Raw text, gold labels,
correctness, recovery, harm, safety-target categories, discovery/primary
membership, and outcome frequencies are prohibited feature inputs.

Only model-internal, gold-independent scores, margins, exact deltas,
model-output transition flags, and explicit response norms are
integration-authorized by this export.

## Next stage

The next stage is diagnostic only after exact export closure. This report does
not authorize a gate, threshold, promotion, or deployment.
"""


def blocked_payload(
    decision: str, next_stage: str, reason: str, ns: argparse.Namespace,
    gates: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    analysis = {
        "stage": STAGE, "decision": decision, "recommended_next_stage": next_stage,
        "blocking_reasons": [reason], "current_git_commit": ns.current_git_commit,
        "population_closure": {}, "composer_source_boundary": {},
        "class_coordinate_authority": {
            "internal_order": list(INTERNAL_CLASS_ORDER),
            "semantic_export_field_order": list(SEMANTIC_CLASS_ORDER),
        },
        "numeric_fidelity": {
            "native_source_dtype": NATIVE_SOURCE_DTYPE,
            "counterfactual_source_dtype": COUNTERFACTUAL_SOURCE_DTYPE,
            "serialized_precision": SERIALIZED_PRECISION,
        },
        "prediction_reproduction": {},
        "controlled_row_cross_check": {},
        "leakage_boundary": {
            "provenance_marker": PROVENANCE_MARKER,
            "prohibited_feature_inputs": list(PROHIBITED_FEATURES),
        },
        "scope_prohibitions": {
            "p0_row_safety_target_csv_loaded": False, "gold_labels_used": False,
            "safety_gate_evaluated": False, "threshold_fitted": False,
            "feature_subset_enumeration_performed": False, "training_performed": False,
        },
        "exact_outputs": list(OUTPUTS),
    }
    decisions = [
        {"order": 1, "decision": BLOCKED_SOURCE, "condition": "authoritative input or exact source boundary missing", "observed": decision == BLOCKED_SOURCE, "reached": decision == BLOCKED_SOURCE, "recommended_next_stage": NEXT_SOURCE},
        {"order": 2, "decision": BLOCKED_REPRODUCTION, "condition": "scores emitted but prediction reproduction disagrees", "observed": decision == BLOCKED_REPRODUCTION, "reached": decision == BLOCKED_REPRODUCTION, "recommended_next_stage": NEXT_REPRODUCTION},
        {"order": 3, "decision": SUCCESS, "condition": "all contracts pass", "observed": False, "reached": False, "recommended_next_stage": NEXT_SUCCESS},
    ]
    summary = [{
        "scope": "blocked", "metric": "blocking reason", "required": "",
        "observed": reason, "passed": False, "details": decision,
    }]
    return analysis, {
        "source": [], "native": [], "candidate": [], "response": [],
        "summary": summary, "decision": decisions,
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
            OUTPUTS[3]: render_csv(NATIVE_H, tables["native"]),
            OUTPUTS[4]: render_csv(CANDIDATE_H, tables["candidate"]),
            OUTPUTS[5]: render_csv(RESPONSE_H, tables["response"]),
            OUTPUTS[6]: render_csv(SUMMARY_H, tables["summary"]),
            OUTPUTS[7]: render_csv(DECISION_H, tables["decision"]),
            OUTPUTS[8]: render_csv(CONTRACT_H, gates),
        }
        if set(payloads) != set(OUTPUTS):
            raise ValueError("exact nine-file payload closure failed")
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
        decision = BLOCKED_REPRODUCTION if isinstance(exc, ReproductionFailure) else BLOCKED_SOURCE
        next_stage = NEXT_REPRODUCTION if decision == BLOCKED_REPRODUCTION else NEXT_SOURCE
        reason = f"{type(exc).__name__}: {exc}"
        if not any(not row["passed"] for row in gates):
            gate(
                gates, "exception", "", "unhandled_exception",
                {"exception": None}, {"exception": reason}, False, reason, fatal=False,
            )
        analysis, tables = blocked_payload(decision, next_stage, reason, ns, gates)
        try:
            write_outputs(ns.output_dir.resolve(), analysis, tables, gates)
        except FileExistsError:
            pass
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
