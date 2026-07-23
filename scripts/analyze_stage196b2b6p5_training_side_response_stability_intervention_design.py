#!/usr/bin/env python3
"""Design Stage196-B2-B6P5 training-side response-stability interventions.

This is a source-feasibility and frozen-artifact analysis.  It does not import
torch, load a model/checkpoint, train, evaluate OOD data, or change behavior.
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
from pathlib import Path
from typing import Any, Iterable, Sequence


STAGE = "Stage196-B2-B6P5"
SEEDS = (183, 184, 185)
EPOCHS = (18, 19, 20)
CANDIDATE_MASKS = ("00100000000000", "01000000000000", "10000000000000")
P4_SIGNED_COORDINATES = (
    "delta_score_support",
    "delta_score_not_entitled",
    "delta_score_refute",
    "delta_top1_runner_up_margin",
    "delta_support_minus_not_entitled",
    "delta_support_minus_refute",
    "delta_refute_minus_not_entitled",
)
PRIMARY_COORDINATES = (
    "delta_support_minus_not_entitled",
    "delta_support_minus_refute",
    "delta_refute_minus_not_entitled",
    "delta_top1_runner_up_margin",
)
PAIR_NAMES = tuple(
    (left, right)
    for index, left in enumerate(CANDIDATE_MASKS)
    for right in CANDIDATE_MASKS[index + 1 :]
)
P4_DECISION = "STAGE196B2B6P4_ACTION_RESPONSE_TOPOLOGY_UNSTABLE"
P4_NEXT = "STAGE196B2B6P5_TRAINING_SIDE_RESPONSE_STABILITY_INTERVENTION_DESIGN"
P3_DECISION = "STAGE196B2B6P3_CROSS_SEED_UNSTABLE_ACTION_RESPONSE_SIGNAL"
P2_DECISION = "STAGE196B2B6P2_ACTION_CONDITIONAL_COMPOSER_MARGIN_EXPORT_COMPLETE"
MAIN_DATA = "data/controlled_v5_v3_without_time_swap.jsonl"
SERIALIZED_PRECISION = "%.17g"

P4_OUTPUTS = (
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
OUTPUTS = (
    "stage196b2b6p5_analysis.json",
    "stage196b2b6p5_report.md",
    "stage196b2b6p5_source_feasibility_audit.csv",
    "stage196b2b6p5_instability_localization.csv",
    "stage196b2b6p5_pairwise_gap_audit.csv",
    "stage196b2b6p5_intervention_designs.csv",
    "stage196b2b6p5_trial_manifest.json",
    "stage196b2b6p5_decision_gate.csv",
    "stage196b2b6p5_contract.csv",
)
FROZEN_COUNTS = {
    "trajectory_topology_rows": 45360,
    "candidate_relative_rows": 15120,
    "endpoint_shift_rows": 15120,
    "non_monotonic": 26575,
    "within_tail_sign_reversal": 16630,
    "within_tail_winner_change": 1002,
    "within_tail_transition_change": 996,
    "within_tail_candidate_order_disagreement": 337,
    "cross_seed_candidate_order_disagreement": 2331,
    "cross_seed_sign_sequence_disagreement": 12448,
    "cross_seed_monotonic_direction_disagreement": 13031,
    "cross_seed_winner_sequence_disagreement": 4032,
    "cross_seed_transition_sequence_disagreement": 4074,
    "endpoint_values_all_equal": 0,
}
P2_REPRODUCTION_KEYS = (
    "native_score_disagreements",
    "counterfactual_score_disagreements",
    "native_margin_disagreements",
    "counterfactual_margin_disagreements",
    "score_delta_disagreements",
    "margin_delta_disagreements",
    "native_prediction_disagreements",
    "counterfactual_prediction_disagreements",
    "categorical_response_disagreements",
)
PROHIBITED = (
    "gold correctness",
    "recovery",
    "harm",
    "MUST_ALLOW",
    "MUST_BLOCK",
    "OPTIONAL",
    "safety target",
    "seed feature",
    "row identity feature",
)

SOURCE_H = (
    "audit_item", "classification", "source_path", "source_lines_or_symbol",
    "finding", "gradient_connected", "independently_addable", "estimated_cost",
    "evidence",
)
LOCALIZATION_H = (
    "seed", "candidate_mask", "candidate_action_key",
    "signed_response_coordinate", "instability_family", "trajectory_count",
    "non_monotonic_count", "sign_reversal_count", "zero_crossing_count",
    "winner_change_count", "transition_change_count",
    "within_tail_candidate_order_disagreement",
    "cross_seed_candidate_order_disagreement",
    "cross_seed_sign_sequence_disagreement",
    "cross_seed_monotonic_direction_disagreement",
)
PAIRWISE_H = (
    "seed", "stable_row_id", "data_identity", "epoch",
    "signed_response_coordinate", "candidate_mask_a", "candidate_action_key_a",
    "candidate_mask_b", "candidate_action_key_b", "candidate_response_a",
    "candidate_response_b", "pairwise_gap_a_minus_b", "absolute_pairwise_gap",
    "exact_tie", "epoch18_gap", "epoch19_gap", "epoch20_gap",
    "pairwise_sign_sequence", "pairwise_order_reversal",
    "cross_seed_pairwise_order_disagreement",
)
DESIGN_H = (
    "variant", "family", "enabled", "single_mechanistic_hypothesis",
    "exact_loss_inputs", "teacher", "mathematical_loss", "gradient_path",
    "activation_scope", "loss_normalization", "initial_coefficient",
    "ablation_flag", "expected_affected_metrics", "failure_criteria",
    "rollback_criteria", "independently_disableable",
)
DECISION_H = (
    "order", "decision", "condition", "observed", "reached",
    "recommended_next_stage",
)
CONTRACT_H = (
    "scope", "run", "gate", "required", "observed", "passed", "blocking_reason",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--stage196b2b6p4-analysis-json", type=Path, required=True)
    parser.add_argument("--stage196b2b6p3-analysis-json", type=Path, required=True)
    parser.add_argument("--stage196b2b6p2-analysis-json", type=Path, required=True)
    parser.add_argument("--current-git-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def cell(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if stripped in ("true", "false"):
        return stripped == "true"
    if stripped and stripped[0] in "[{\"" and stripped[-1] in "]}\"":
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value
    return value


def boolean(value: Any, name: str) -> bool:
    value = cell(value)
    if type(value) is not bool:
        raise ValueError(f"{name}: exact boolean required")
    return value


def integer(value: Any, name: str) -> int:
    value = cell(value)
    if type(value) is not int:
        if isinstance(value, str) and re.fullmatch(r"-?[0-9]+", value):
            return int(value)
        raise ValueError(f"{name}: integer required")
    return value


def number(value: Any, name: str) -> float:
    value = cell(value)
    if type(value) not in (int, float):
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError as exc:
                raise ValueError(f"{name}: numeric required") from exc
        else:
            raise ValueError(f"{name}: numeric required")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name}: finite numeric required")
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True,
        check=False, timeout=30,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


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


def require_columns(rows: list[dict[str, Any]], fields: Sequence[str], label: str) -> None:
    columns = set(rows[0]) if rows else set()
    missing = sorted(set(fields) - columns)
    if missing:
        raise ValueError(f"{label}: missing columns {missing}")


def contract_closed(rows: list[dict[str, str]]) -> bool:
    return bool(rows) and all(
        boolean(row.get("passed"), "contract passed")
        and not str(row.get("blocking_reason", "")).strip()
        for row in rows
    )


def exact_directory(path: Path, expected: Sequence[str]) -> tuple[list[str], list[str], list[str]]:
    observed = sorted(item.name for item in path.iterdir() if item.is_file())
    return observed, sorted(set(expected) - set(observed)), sorted(set(observed) - set(expected))


def sign(value: float) -> str:
    return "ZERO" if value == 0.0 else "POSITIVE" if value > 0.0 else "NEGATIVE"


def quantile(sorted_values: Sequence[float], probability: float) -> float | None:
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def quantile_summary(values: Iterable[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    probabilities = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)
    return {
        "count": len(ordered),
        "minimum": ordered[0] if ordered else None,
        "maximum": ordered[-1] if ordered else None,
        "quantiles": {
            SERIALIZED_PRECISION % probability: quantile(ordered, probability)
            for probability in probabilities
        },
    }


def source_feasibility(root: Path, gates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trainer = root / "scripts" / "train_controlled_v6b_minimal.py"
    model = root / "src" / "contramamba" / "modeling_v6b_minimal.py"
    p4_script = root / "scripts" / "analyze_stage196b2b6p4_action_response_stability_state_design.py"
    b6_script = root / "scripts" / "analyze_stage196b2b6_minimal_selector_intervention.py"
    for path in (trainer, model, p4_script, b6_script):
        if not path.is_file():
            raise ValueError(f"required static source missing: {path}")
    trainer_text = trainer.read_text(encoding="utf-8")
    model_text = model.read_text(encoding="utf-8")
    p4_text = p4_script.read_text(encoding="utf-8")
    b6_text = b6_script.read_text(encoding="utf-8")
    checks = {
        "entry_point": "def run_training_v6b(" in trainer_text and "def main(" in trainer_text,
        "native_scores": "final_logits = base_logits" in model_text and '"logits": final_logits' in model_text,
        "diagnostic_detach": "detached diagnostic witnesses" in model_text and ".detach()" in model_text,
        "optimizer_step": "optimizer.step()" in trainer_text and "grad_scaler.step(optimizer)" in trainer_text,
        "p4_recomposition": "b6_module.apply_mask" in p4_text and "materialize_composer_geometry" in p4_text,
        "row_specific_action": "def apply_mask(" in b6_text and "PRIMITIVE_FIELDS" in b6_text,
        "main_data": MAIN_DATA in trainer_text,
        "no_ema": not re.search(r"\b(?:EMA|ExponentialMovingAverage|AveragedModel)\b", trainer_text),
    }
    gate(
        gates, "source", "", "source_gradient_feasibility_closure",
        {key: True for key in checks}, checks, all(checks.values()),
        "static training/composer source boundary changed",
    )
    rows = [
        {
            "audit_item": "exact_training_entry_point",
            "classification": "TRAINING_GRADIENT_PATH_AVAILABLE",
            "source_path": str(trainer),
            "source_lines_or_symbol": "main -> nested run_training_v6b",
            "finding": "The active v6b path defines run_training_v6b inside main and calls the model once for the native training output before assembling total_loss.",
            "gradient_connected": True, "independently_addable": True,
            "estimated_cost": "existing native forward",
            "evidence": "def run_training_v6b; output = _vnext_forward_maybe_batched",
        },
        {
            "audit_item": "native_final_composer_scores",
            "classification": "TRAINING_GRADIENT_PATH_AVAILABLE",
            "source_path": str(model),
            "source_lines_or_symbol": "ContraMambaV6BMinimal.forward: final_logits",
            "finding": "Native final scores are produced as differentiable torch tensors from decision-head base logits plus optional final-logit modulation and returned as output['logits'].",
            "gradient_connected": True, "independently_addable": True,
            "estimated_cost": "no additional forward",
            "evidence": "final_logits = base_logits; losses consume final_logits; logits returned",
        },
        {
            "audit_item": "p4_counterfactual_recomposition",
            "classification": "DIAGNOSTIC_RECOMPOSITION_ONLY",
            "source_path": str(p4_script),
            "source_lines_or_symbol": "b6_module.apply_mask; materialize_composer_geometry",
            "finding": "P4 recomposes Python floats from exported joint and separately trained frame_local_only arm state. It is not a torch training graph.",
            "gradient_connected": False, "independently_addable": False,
            "estimated_cost": "post-hoc only; zero training cost",
            "evidence": "exported sidecars, float geometry, diagnostic observability tensors detached",
        },
        {
            "audit_item": "counterfactual_candidate_actions_with_gradients",
            "classification": "MINIMAL_GRADIENT_INSTRUMENTATION_REQUIRED",
            "source_path": str(trainer),
            "source_lines_or_symbol": "_install_framegate_gradient_ownership; run_training_v6b",
            "finding": "No live operator maps the exact row-specific 5-bit candidate action keys onto differentiable composer primitives. The two P4 arms are separate trained runs, not two values emitted by one training model.",
            "gradient_connected": False, "independently_addable": False,
            "estimated_cost": "at least one additional counterpart-arm full forward when active",
            "evidence": "frame_local_only is an installed gradient hook/run mode; P4 action substitution is external",
        },
        {
            "audit_item": "three_candidates_in_one_training_step",
            "classification": "MINIMAL_GRADIENT_INSTRUMENTATION_REQUIRED",
            "source_path": str(b6_script),
            "source_lines_or_symbol": "apply_mask",
            "finding": "After live joint/counterpart primitive tensors and exact action keys exist, all three candidates can be vectorized in one step; three separate backbone forwards are not intrinsically required.",
            "gradient_connected": False, "independently_addable": False,
            "estimated_cost": "one counterpart full forward plus O(batch x 3 x 4) composer arithmetic",
            "evidence": "each action only substitutes a subset of five primitive fields before the same analytic composer",
        },
        {
            "audit_item": "final_composer_margin_autograd",
            "classification": "TRAINING_GRADIENT_PATH_AVAILABLE",
            "source_path": str(model),
            "source_lines_or_symbol": "output['logits']",
            "finding": "Signed class-score differences formed directly from output['logits'] retain autograd connectivity; argmax/top1 identity does not, and the exact top1-runner-up coordinate requires a piecewise gather/topk definition.",
            "gradient_connected": True, "independently_addable": True,
            "estimated_cost": "negligible tensor arithmetic",
            "evidence": "no detach on returned final_logits",
        },
        {
            "audit_item": "optimizer_step_location",
            "classification": "TRAINING_GRADIENT_PATH_AVAILABLE",
            "source_path": str(trainer),
            "source_lines_or_symbol": "run_training_v6b after total_loss assembly",
            "finding": "The intervention loss must be added to total_loss before loss_for_backward; optimizer.step and GradScaler.step occur immediately after backward and clipping.",
            "gradient_connected": True, "independently_addable": True,
            "estimated_cost": "no extra optimizer step",
            "evidence": "loss_for_backward.backward(); clip_grad_norm_; optimizer.step",
        },
        {
            "audit_item": "ema_or_frozen_anchor_teacher",
            "classification": "MINIMAL_GRADIENT_INSTRUMENTATION_REQUIRED",
            "source_path": str(trainer),
            "source_lines_or_symbol": "run_training_v6b state management",
            "finding": "No EMA teacher or response anchor exists. Either can be added as isolated state, but neither is source-authorized as the scientific teacher until live candidate geometry and teacher provenance are instrumented.",
            "gradient_connected": False, "independently_addable": True,
            "estimated_cost": "EMA update O(parameters); teacher evaluation one no-grad full forward per active step",
            "evidence": "no EMA/AveragedModel path; existing state capture is checkpoint/SWA machinery, not a teacher",
        },
        {
            "audit_item": "full_counterfactual_forward",
            "classification": "FULL_COUNTERFACTUAL_FORWARD_REQUIRED",
            "source_path": str(trainer),
            "source_lines_or_symbol": "separate joint/frame_local_only run construction",
            "finding": "Exact P4 donor-arm primitive values cannot be recovered from the native joint forward. A counterpart model/anchor forward is required unless future instrumentation co-emits an equivalent state.",
            "gradient_connected": False, "independently_addable": False,
            "estimated_cost": "minimum +1 full counterpart forward on active steps; +1 teacher forward if teacher is separate",
            "evidence": "P4 joins sidecars from separately trained gradient-ownership arms",
        },
    ]
    summary = {
        "exact_training_entry_point": "scripts/train_controlled_v6b_minimal.py::main::<locals>.run_training_v6b",
        "native_final_scores": "src/contramamba/modeling_v6b_minimal.py::ContraMambaV6BMinimal.forward::output['logits']",
        "counterfactual_gradient_path": "MINIMAL_GRADIENT_INSTRUMENTATION_REQUIRED",
        "three_candidates_vectorizable": True,
        "three_candidates_currently_available": False,
        "final_margin_autograd": True,
        "top1_runner_up_piecewise": True,
        "optimizer_step_boundary": "after total_loss assembly/backward/clip in run_training_v6b",
        "teacher_status": "UNAVAILABLE_WITHOUT_ADDITIONAL_INSTRUMENTATION",
        "estimated_incremental_cost": {
            "candidate_geometry": "minimum one counterpart-arm full forward plus negligible three-candidate composer arithmetic",
            "separate_teacher": "one additional no-grad teacher full forward plus EMA update",
        },
    }
    return rows, summary


def p4_observed_counts(
    topology: list[dict[str, str]], order: list[dict[str, str]],
    endpoint: list[dict[str, str]],
) -> dict[str, int]:
    winner_coordinate = P4_SIGNED_COORDINATES[0]
    return {
        "trajectory_topology_rows": len(topology),
        "candidate_relative_rows": len(order),
        "endpoint_shift_rows": len(endpoint),
        "non_monotonic": sum(row["monotonic_direction"] == "NON_MONOTONIC" for row in topology),
        "within_tail_sign_reversal": sum(integer(row["sign_reversal_count"], "sign reversal") > 0 for row in topology),
        "within_tail_winner_change": sum(
            integer(row["decision_boundary_crossing_count"], "winner crossings") > 0
            for row in topology if row["signed_response_coordinate"] == winner_coordinate
        ),
        "within_tail_transition_change": sum(
            integer(row["prediction_changed_persistence"], "prediction persistence") < 3
            or integer(row["entitlement_transition_persistence"], "entitlement persistence") < 3
            or integer(row["polarity_transition_persistence"], "polarity persistence") < 3
            or integer(row["polarity_direction_persistence"], "polarity direction persistence") < 3
            for row in topology if row["signed_response_coordinate"] == winner_coordinate
        ),
        "within_tail_candidate_order_disagreement": sum(
            not boolean(row["pairwise_ordering_persistence_across_tail"], "pairwise persistence")
            for row in order
        ),
        "cross_seed_candidate_order_disagreement": sum(
            not boolean(row["cross_seed_pairwise_order_agreement"], "cross-seed pairwise agreement")
            for row in order
        ),
        "cross_seed_sign_sequence_disagreement": sum(
            not boolean(row["tail_sign_sequence_agreement"], "sign sequence agreement")
            for row in endpoint
        ),
        "cross_seed_monotonic_direction_disagreement": sum(
            not boolean(row["monotonic_direction_agreement"], "monotonic agreement")
            for row in endpoint
        ),
        "cross_seed_winner_sequence_disagreement": sum(
            not boolean(row["counterfactual_winner_sequence_agreement"], "winner agreement")
            for row in endpoint if row["signed_response_coordinate"] == winner_coordinate
        ),
        "cross_seed_transition_sequence_disagreement": sum(
            not boolean(row["transition_sequence_agreement"], "transition agreement")
            for row in endpoint if row["signed_response_coordinate"] == winner_coordinate
        ),
        "endpoint_values_all_equal": sum(
            boolean(row["endpoint_values_all_equal"], "endpoint equality") for row in endpoint
        ),
    }


def build_localization(
    topology: list[dict[str, str]], order: list[dict[str, str]],
    endpoint: list[dict[str, str]],
) -> list[dict[str, Any]]:
    order_index = {
        (integer(row["seed"], "order seed"), row["data_identity"], row["signed_response_coordinate"]): row
        for row in order
    }
    endpoint_index = {
        (row["data_identity"], row["candidate_mask"], row["signed_response_coordinate"]): row
        for row in endpoint
    }
    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in topology:
        coordinate = row["signed_response_coordinate"]
        if coordinate not in PRIMARY_COORDINATES:
            continue
        seed = integer(row["seed"], "topology seed")
        key = (seed, row["candidate_mask"], row["candidate_action_key"], coordinate)
        if key not in groups:
            groups[key] = {
                "seed": seed, "candidate_mask": row["candidate_mask"],
                "candidate_action_key": row["candidate_action_key"],
                "signed_response_coordinate": coordinate, "trajectory_count": 0,
                "non_monotonic_count": 0, "sign_reversal_count": 0,
                "zero_crossing_count": 0, "winner_change_count": 0,
                "transition_change_count": 0,
                "within_tail_candidate_order_disagreement": 0,
                "cross_seed_candidate_order_disagreement": 0,
                "cross_seed_sign_sequence_disagreement": 0,
                "cross_seed_monotonic_direction_disagreement": 0,
            }
        target = groups[key]
        target["trajectory_count"] += 1
        target["non_monotonic_count"] += row["monotonic_direction"] == "NON_MONOTONIC"
        target["sign_reversal_count"] += integer(row["sign_reversal_count"], "sign reversal") > 0
        target["zero_crossing_count"] += integer(row["zero_crossing_count"], "zero crossing") > 0
        target["winner_change_count"] += integer(row["decision_boundary_crossing_count"], "winner crossing") > 0
        target["transition_change_count"] += (
            integer(row["prediction_changed_persistence"], "prediction persistence") < 3
            or integer(row["entitlement_transition_persistence"], "entitlement persistence") < 3
            or integer(row["polarity_transition_persistence"], "polarity persistence") < 3
            or integer(row["polarity_direction_persistence"], "polarity direction persistence") < 3
        )
        order_row = order_index[(seed, row["data_identity"], coordinate)]
        endpoint_row = endpoint_index[(row["data_identity"], row["candidate_mask"], coordinate)]
        target["within_tail_candidate_order_disagreement"] += not boolean(
            order_row["pairwise_ordering_persistence_across_tail"], "within-tail order"
        )
        target["cross_seed_candidate_order_disagreement"] += not boolean(
            order_row["cross_seed_pairwise_order_agreement"], "cross-seed order"
        )
        target["cross_seed_sign_sequence_disagreement"] += not boolean(
            endpoint_row["tail_sign_sequence_agreement"], "cross-seed sign"
        )
        target["cross_seed_monotonic_direction_disagreement"] += not boolean(
            endpoint_row["monotonic_direction_agreement"], "cross-seed monotonic"
        )
    result: list[dict[str, Any]] = []
    for key in sorted(groups):
        row = groups[key]
        families: list[str] = []
        if any(row[field] for field in (
            "non_monotonic_count", "sign_reversal_count", "zero_crossing_count",
            "winner_change_count", "transition_change_count",
        )):
            families.append("WITHIN_TAIL_TOPOLOGY")
        if row["within_tail_candidate_order_disagreement"] or row["cross_seed_candidate_order_disagreement"]:
            families.append("CANDIDATE_RELATIVE_ORDER")
        if row["cross_seed_sign_sequence_disagreement"] or row["cross_seed_monotonic_direction_disagreement"]:
            families.append("CROSS_SEED_TOPOLOGY")
        row["instability_family"] = "+".join(families) if families else "STABLE_UNDER_AUDITED_RELATIONS"
        result.append(row)
    return result


def build_pairwise(
    tail: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    values: dict[tuple[int, str, str, int, str], tuple[float, str, str]] = {}
    stable_by_data: dict[tuple[int, str], str] = {}
    for row in tail:
        coordinate_values = {
            coordinate: number(row[coordinate], coordinate)
            for coordinate in PRIMARY_COORDINATES
        }
        seed = integer(row["seed"], "tail seed")
        epoch = integer(row["epoch"], "tail epoch")
        data = row["data_identity"]
        stable_by_data[(seed, data)] = row["stable_row_id"]
        for coordinate, value in coordinate_values.items():
            key = (seed, data, coordinate, epoch, row["candidate_mask"])
            if key in values:
                raise ValueError(f"duplicate tail candidate key: {key}")
            values[key] = (value, row["candidate_action_key"], row["stable_row_id"])
    expected_values = len(SEEDS) * 720 * len(PRIMARY_COORDINATES) * len(EPOCHS) * len(CANDIDATE_MASKS)
    if len(values) != expected_values:
        raise ValueError(f"primary candidate value population {len(values)} != {expected_values}")
    trajectories: dict[tuple[int, str, str, str, str], dict[str, Any]] = {}
    data_identities = sorted({key[1] for key in values})
    for seed in SEEDS:
        for data in data_identities:
            for coordinate in PRIMARY_COORDINATES:
                for left, right in PAIR_NAMES:
                    gaps: list[float] = []
                    actions: dict[str, str] = {}
                    for epoch in EPOCHS:
                        left_value, left_action, _ = values[(seed, data, coordinate, epoch, left)]
                        right_value, right_action, _ = values[(seed, data, coordinate, epoch, right)]
                        gaps.append(left_value - right_value)
                        actions[left] = left_action
                        actions[right] = right_action
                    signs = tuple(sign(gap) for gap in gaps)
                    trajectories[(seed, data, coordinate, left, right)] = {
                        "gaps": gaps, "signs": signs, "actions": actions,
                        "reversal": "POSITIVE" in signs and "NEGATIVE" in signs,
                        "tie_contact": "ZERO" in signs,
                    }
    cross_disagreement: dict[tuple[str, str, str, str], bool] = {}
    for data in data_identities:
        for coordinate in PRIMARY_COORDINATES:
            for left, right in PAIR_NAMES:
                sequences = {
                    trajectories[(seed, data, coordinate, left, right)]["signs"]
                    for seed in SEEDS
                }
                cross_disagreement[(data, coordinate, left, right)] = len(sequences) != 1
    rows: list[dict[str, Any]] = []
    for key in sorted(trajectories):
        seed, data, coordinate, left, right = key
        trajectory = trajectories[key]
        for epoch_index, epoch in enumerate(EPOCHS):
            left_value, left_action, stable = values[(seed, data, coordinate, epoch, left)]
            right_value, right_action, _ = values[(seed, data, coordinate, epoch, right)]
            gap_value = trajectory["gaps"][epoch_index]
            rows.append({
                "seed": seed, "stable_row_id": stable, "data_identity": data,
                "epoch": epoch, "signed_response_coordinate": coordinate,
                "candidate_mask_a": left, "candidate_action_key_a": left_action,
                "candidate_mask_b": right, "candidate_action_key_b": right_action,
                "candidate_response_a": left_value, "candidate_response_b": right_value,
                "pairwise_gap_a_minus_b": gap_value,
                "absolute_pairwise_gap": abs(gap_value),
                "exact_tie": gap_value == 0.0,
                "epoch18_gap": trajectory["gaps"][0],
                "epoch19_gap": trajectory["gaps"][1],
                "epoch20_gap": trajectory["gaps"][2],
                "pairwise_sign_sequence": list(trajectory["signs"]),
                "pairwise_order_reversal": trajectory["reversal"],
                "cross_seed_pairwise_order_disagreement":
                    cross_disagreement[(data, coordinate, left, right)],
            })
    absolute = [row["absolute_pairwise_gap"] for row in rows]
    trajectory_values = list(trajectories.values())
    strict_reversals = sum(item["reversal"] and not item["tie_contact"] for item in trajectory_values)
    tie_contact_reversals = sum(item["reversal"] and item["tie_contact"] for item in trajectory_values)
    if strict_reversals > tie_contact_reversals:
        mechanism = "STRICT_PAIRWISE_REVERSAL_COUNT_DOMINATES_TIE_CONTACT_REVERSAL_COUNT"
    elif tie_contact_reversals > strict_reversals:
        mechanism = "EXACT_TIE_CONTACT_REVERSAL_COUNT_DOMINATES_STRICT_REVERSAL_COUNT"
    else:
        mechanism = "STRICT_AND_TIE_CONTACT_REVERSAL_COUNTS_EQUAL"
    by_coordinate = {}
    for coordinate in PRIMARY_COORDINATES:
        subset = [row for row in rows if row["signed_response_coordinate"] == coordinate]
        by_coordinate[coordinate] = {
            "row_count": len(subset),
            "exact_tie_count": sum(row["exact_tie"] for row in subset),
            "minimum_absolute_pairwise_gap": min(row["absolute_pairwise_gap"] for row in subset),
            "absolute_gap_distribution": quantile_summary(row["absolute_pairwise_gap"] for row in subset),
            "sign_sequence_distribution": dict(sorted(Counter(
                canonical(row["pairwise_sign_sequence"]) for row in subset if row["epoch"] == 18
            ).items())),
            "pairwise_order_reversal_trajectories": sum(
                row["pairwise_order_reversal"] for row in subset if row["epoch"] == 18
            ),
            "cross_seed_pairwise_order_disagreement_rows": sum(
                row["cross_seed_pairwise_order_disagreement"] for row in subset
            ),
        }
    summary = {
        "population_rows": len(rows),
        "expected_population_rows": len(SEEDS) * 720 * len(PRIMARY_COORDINATES) * len(PAIR_NAMES) * len(EPOCHS),
        "trajectory_count": len(trajectories),
        "exact_tie_count": sum(row["exact_tie"] for row in rows),
        "minimum_absolute_pairwise_gap": min(absolute),
        "absolute_pairwise_gap_distribution": quantile_summary(absolute),
        "pairwise_sign_sequence_distribution": dict(sorted(Counter(
            canonical(item["signs"]) for item in trajectory_values
        ).items())),
        "pairwise_order_reversal_trajectories": sum(item["reversal"] for item in trajectory_values),
        "strict_reversal_trajectories_without_exact_tie_contact": strict_reversals,
        "reversal_trajectories_with_exact_tie_contact": tie_contact_reversals,
        "cross_seed_pairwise_order_disagreement_trajectories": sum(cross_disagreement.values()),
        "mechanism_without_magnitude_threshold": mechanism,
        "threshold_selected": False,
        "by_coordinate": by_coordinate,
    }
    return rows, summary


def intervention_designs() -> list[dict[str, Any]]:
    direction_math = (
        "For row i, candidate a and coordinate k, let r^S_iak be the student "
        "counterfactual-minus-native signed margin response and t_iak=sgn(r^T_iak) "
        "from a stop-gradient teacher. Ignore r^T_iak=0 exactly. "
        "L_A=mean_{eligible i,a,k} softplus(-t_iak*r^S_iak)."
    )
    order_math = (
        "Let c^S_iak=r^S_iak-(1/3)sum_b r^S_ibk. For each unordered candidate "
        "pair (a,b), q_iabk=sgn(c^T_iak-c^T_ibk). Ignore exact teacher ties. "
        "L_B=mean_{eligible i,k,a<b} softplus(-q_iabk*(c^S_iak-c^S_ibk))."
    )
    common_failure = (
        "Any precommitted clean-dev primary regression or adverse change in support "
        "recall, false entitlement, false not-entitled, or polarity errors; failure "
        "to preserve P2 exact composer reproduction; or no improvement in its targeted "
        "P4 topology metrics across the three fixed seeds."
    )
    common_rollback = (
        "Disable the single family flag and restore the baseline code path/checkpoint "
        "if its failure criterion is met, gradients are nonfinite, exact ties are not "
        "masked, or the main-data boundary changes."
    )
    return [
        {
            "variant": "baseline", "family": "NONE", "enabled": False,
            "single_mechanistic_hypothesis": "Reference training behavior without stability regularization.",
            "exact_loss_inputs": "none", "teacher": "none", "mathematical_loss": "L=0",
            "gradient_path": "existing native classification path only",
            "activation_scope": "all epochs under existing training configuration",
            "loss_normalization": "not applicable", "initial_coefficient": 0.0,
            "ablation_flag": "both response-stability flags false",
            "expected_affected_metrics": "none by intervention",
            "failure_criteria": "baseline run or contract does not reproduce its frozen configuration",
            "rollback_criteria": "not applicable", "independently_disableable": True,
        },
        {
            "variant": "direction-consistency only",
            "family": "A_SIGNED_RESPONSE_DIRECTION", "enabled": True,
            "single_mechanistic_hypothesis": "Tail sign reversals arise because signed candidate responses lack a temporal direction anchor.",
            "exact_loss_inputs": "student and stop-gradient teacher responses for the four primary signed margin coordinates and exact three candidate actions",
            "teacher": "UNAVAILABLE_WITHOUT_ADDITIONAL_INSTRUMENTATION; EMA is the minimal conceptual teacher, but no teacher is selected for implementation at P5",
            "mathematical_loss": direction_math,
            "gradient_path": "teacher target stop-gradient; student signed margins to live final logits/composer primitives; currently missing exact live candidate operator",
            "activation_scope": "last N epochs with N=3 as a semantic tail-length parameter, not hard-coded epoch numbers; deterministic for any total epoch count",
            "loss_normalization": "mean over non-tied teacher targets across rows, three candidates, and four coordinates; zero loss if none eligible",
            "initial_coefficient": 0.1,
            "ablation_flag": "--use-response-direction-consistency",
            "expected_affected_metrics": "P4 sign reversals, zero crossings, monotonic-direction disagreement; candidate order is not the target",
            "failure_criteria": common_failure, "rollback_criteria": common_rollback,
            "independently_disableable": True,
        },
        {
            "variant": "candidate-order-consistency only",
            "family": "B_CANDIDATE_RELATIVE_ORDER", "enabled": True,
            "single_mechanistic_hypothesis": "Residual instability is caused by changes in within-row pairwise ordering of the exact three candidate responses.",
            "exact_loss_inputs": "centered student and stop-gradient teacher responses for the four primary coordinates and all three unordered candidate pairs",
            "teacher": "UNAVAILABLE_WITHOUT_ADDITIONAL_INSTRUMENTATION; no fixed action order and no mask lexical target",
            "mathematical_loss": order_math,
            "gradient_path": "teacher pair sign stop-gradient; student centered pair gap to live candidate composer; currently missing exact live candidate operator",
            "activation_scope": "last N epochs with N=3 as a semantic tail-length parameter, not hard-coded epoch numbers; deterministic for any total epoch count",
            "loss_normalization": "mean over non-tied teacher pair signs across rows, four coordinates, and three unordered pairs; zero loss if none eligible",
            "initial_coefficient": 0.1,
            "ablation_flag": "--use-candidate-order-consistency",
            "expected_affected_metrics": "within-tail and cross-seed candidate-order disagreement and pairwise reversals; absolute response coordinates are not targets",
            "failure_criteria": common_failure, "rollback_criteria": common_rollback,
            "independently_disableable": True,
        },
    ]


def decision_rows(
    direction_ready: bool, order_ready: bool, detached_state: bool,
    simple_insufficient: bool,
) -> tuple[str, str, list[dict[str, Any]]]:
    definitions = (
        (
            "STAGE196B2B6P5_TWO_SEPARABLE_STABILITY_INTERVENTIONS_READY",
            "STAGE196B2B6P6_SEPARATE_STABILITY_INTERVENTION_IMPLEMENTATION",
            direction_ready and order_ready,
            "both families have distinct valid gradient paths",
        ),
        (
            "STAGE196B2B6P5_DIRECTION_STABILITY_INTERVENTION_READY",
            "STAGE196B2B6P6_DIRECTION_STABILITY_INTERVENTION_IMPLEMENTATION",
            direction_ready and not order_ready,
            "only signed direction consistency has a valid gradient path",
        ),
        (
            "STAGE196B2B6P5_CANDIDATE_ORDER_STABILITY_INTERVENTION_READY",
            "STAGE196B2B6P6_CANDIDATE_ORDER_STABILITY_INTERVENTION_IMPLEMENTATION",
            order_ready and not direction_ready,
            "only candidate-order consistency has a valid gradient path",
        ),
        (
            "STAGE196B2B6P5_GRADIENT_PATH_INSTRUMENTATION_REQUIRED",
            "STAGE196B2B6P6_MINIMAL_GRADIENT_PATH_INSTRUMENTATION",
            detached_state,
            "necessary composer/candidate state exists diagnostically but is detached from training",
        ),
        (
            "STAGE196B2B6P5_SIMPLE_STABILITY_REGULARIZATION_INSUFFICIENT",
            "STAGE196B2B7_SELECTOR_MECHANISM_RETHINK",
            simple_insufficient,
            "strict hard reversals dominate and neither simple family has a non-arbitrary target",
        ),
    )
    reached = False
    rows: list[dict[str, Any]] = []
    selected: tuple[str, str] | None = None
    for order, (decision, next_stage, condition, description) in enumerate(definitions, 1):
        take = bool(condition and not reached)
        rows.append({
            "order": order, "decision": decision, "condition": description,
            "observed": bool(condition), "reached": take,
            "recommended_next_stage": next_stage,
        })
        if take:
            selected = (decision, next_stage)
            reached = True
    if selected is None:
        raise ValueError("scientific decision hierarchy is unreachable")
    return selected[0], selected[1], rows


def analyze(ns: argparse.Namespace, gates: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    root = ns.repo_root.resolve()
    p4_path = ns.stage196b2b6p4_analysis_json.resolve()
    p3_path = ns.stage196b2b6p3_analysis_json.resolve()
    p2_path = ns.stage196b2b6p2_analysis_json.resolve()
    output = ns.output_dir.resolve()
    paths = (root, p4_path, p3_path, p2_path, output)
    paths_ok = all(path.is_absolute() and (path == root or root in path.parents) for path in paths)
    gate(
        gates, "invocation", "", "explicit_path_boundary",
        {"absolute_and_under_repo_root": True}, {"passed": paths_ok}, paths_ok,
        "all paths must be explicit and under repo root",
    )
    gate(
        gates, "invocation", "", "analysis_basenames",
        {
            "p4": P4_OUTPUTS[0], "p3": "stage196b2b6p3_analysis.json",
            "p2": P2_OUTPUTS[0],
        },
        {"p4": p4_path.name, "p3": p3_path.name, "p2": p2_path.name},
        p4_path.name == P4_OUTPUTS[0]
        and p3_path.name == "stage196b2b6p3_analysis.json"
        and p2_path.name == P2_OUTPUTS[0],
        "analysis JSON basename changed",
    )
    commit_format = re.fullmatch(r"[0-9a-f]{40}", ns.current_git_commit) is not None
    head = git_value(root, "rev-parse", "HEAD")
    gate(
        gates, "source", "", "current_commit_identity",
        {"format": "lowercase 40-hex", "equals_HEAD": True},
        {"value": ns.current_git_commit, "HEAD": head},
        commit_format and head == ns.current_git_commit,
        "current commit identity failed",
    )
    observed, missing, unexpected = exact_directory(p4_path.parent, P4_OUTPUTS)
    gate(
        gates, "p4", "", "p4_exact_artifact_closure", sorted(P4_OUTPUTS),
        {"files": observed, "missing": missing, "unexpected": unexpected},
        observed == sorted(P4_OUTPUTS), "P4 exact eleven-file closure failed",
    )
    p4 = read_json(p4_path)
    p3 = read_json(p3_path)
    p2 = read_json(p2_path)
    gate(
        gates, "p4", "", "p4_decision_closure",
        {"decision": P4_DECISION, "recommended_next_stage": P4_NEXT, "blocking_reasons": []},
        {key: p4.get(key) for key in ("decision", "recommended_next_stage", "blocking_reasons")},
        p4.get("decision") == P4_DECISION
        and p4.get("recommended_next_stage") == P4_NEXT
        and p4.get("blocking_reasons") == [],
        "P4 decision closure changed",
    )
    gate(
        gates, "p3", "", "p3_analysis_closure", P3_DECISION, p3.get("decision"),
        p3.get("decision") == P3_DECISION and p3.get("blocking_reasons") == [],
        "P3 analysis closure changed",
    )
    gate(
        gates, "p2", "", "p2_analysis_closure", P2_DECISION, p2.get("decision"),
        p2.get("decision") == P2_DECISION and p2.get("blocking_reasons") == [],
        "P2 analysis closure changed",
    )
    p4_contract = read_csv(p4_path.parent / P4_OUTPUTS[-1])
    gate(
        gates, "p4", "", "p4_zero_failed_contracts", 0,
        sum(not boolean(row.get("passed"), "P4 contract passed") for row in p4_contract),
        contract_closed(p4_contract), "P4 contract contains a failure",
    )
    topology = read_csv(p4_path.parent / P4_OUTPUTS[5])
    order = read_csv(p4_path.parent / P4_OUTPUTS[6])
    endpoint = read_csv(p4_path.parent / P4_OUTPUTS[7])
    tail = read_csv(p4_path.parent / P4_OUTPUTS[3])
    dictionary = read_csv(p4_path.parent / P4_OUTPUTS[4])
    leakage = read_csv(p4_path.parent / P4_OUTPUTS[8])
    require_columns(topology, (
        "seed", "data_identity", "candidate_mask", "candidate_action_key",
        "signed_response_coordinate", "monotonic_direction", "sign_reversal_count",
        "zero_crossing_count", "decision_boundary_crossing_count",
        "prediction_changed_persistence", "entitlement_transition_persistence",
        "polarity_transition_persistence", "polarity_direction_persistence",
    ), "P4 topology")
    require_columns(order, (
        "seed", "data_identity", "signed_response_coordinate",
        "pairwise_ordering_persistence_across_tail",
        "cross_seed_pairwise_order_agreement",
    ), "P4 order")
    require_columns(endpoint, (
        "data_identity", "candidate_mask", "signed_response_coordinate",
        "tail_sign_sequence_agreement", "monotonic_direction_agreement",
        "counterfactual_winner_sequence_agreement", "transition_sequence_agreement",
        "endpoint_values_all_equal",
    ), "P4 endpoint")
    require_columns(tail, (
        "seed", "epoch", "stable_row_id", "data_identity", "candidate_mask",
        "candidate_action_key", *PRIMARY_COORDINATES,
    ), "P4 tail")
    observed_counts = p4_observed_counts(topology, order, endpoint)
    gate(
        gates, "p4", "", "frozen_instability_evidence",
        FROZEN_COUNTS, observed_counts, observed_counts == FROZEN_COUNTS,
        "P4 frozen instability evidence changed",
    )
    seeds = sorted({integer(row["seed"], "topology seed") for row in topology})
    masks = sorted({row["candidate_mask"] for row in topology})
    coordinates = sorted({row["signed_response_coordinate"] for row in topology})
    gate(
        gates, "population", "", "exact_seed_mask_coordinate_set",
        {"seeds": list(SEEDS), "candidate_masks": list(CANDIDATE_MASKS),
         "p4_signed_coordinates": sorted(P4_SIGNED_COORDINATES),
         "intervention_coordinates": sorted(PRIMARY_COORDINATES)},
        {"seeds": seeds, "candidate_masks": masks, "p4_signed_coordinates": coordinates,
         "intervention_coordinates": sorted(PRIMARY_COORDINATES)},
        seeds == list(SEEDS) and masks == list(CANDIDATE_MASKS)
        and coordinates == sorted(P4_SIGNED_COORDINATES),
        "seed, candidate-mask, or signed-coordinate set changed",
    )
    p2_repro = p4.get("p2_endpoint_authority", {}).get("epoch20_reproduction", {})
    reproduction = {key: p2_repro.get(key) for key in P2_REPRODUCTION_KEYS}
    gate(
        gates, "p2", "", "p2_endpoint_reproduction_closure",
        {key: 0 for key in P2_REPRODUCTION_KEYS}, reproduction,
        reproduction == {key: 0 for key in P2_REPRODUCTION_KEYS},
        "P2 epoch-20 exact reproduction changed",
    )
    localization = build_localization(topology, order, endpoint)
    expected_local_trajectories = len(SEEDS) * 720 * len(CANDIDATE_MASKS) * len(PRIMARY_COORDINATES)
    localization_trajectories = sum(row["trajectory_count"] for row in localization)
    gate(
        gates, "localization", "", "instability_localization_closure",
        {"trajectory_count": expected_local_trajectories, "coordinates": sorted(PRIMARY_COORDINATES)},
        {"trajectory_count": localization_trajectories,
         "coordinates": sorted({row["signed_response_coordinate"] for row in localization})},
        localization_trajectories == expected_local_trajectories
        and {row["signed_response_coordinate"] for row in localization} == set(PRIMARY_COORDINATES),
        "instability localization population failed",
    )
    pairwise_rows, pairwise_summary = build_pairwise(tail)
    gate(
        gates, "pairwise", "", "pairwise_gap_population_closure",
        pairwise_summary["expected_population_rows"], pairwise_summary["population_rows"],
        pairwise_summary["population_rows"] == pairwise_summary["expected_population_rows"],
        "pairwise gap population failed",
    )
    source_rows, feasibility = source_feasibility(root, gates)
    designs = intervention_designs()
    variants = [row["variant"] for row in designs]
    independent = (
        variants == ["baseline", "direction-consistency only", "candidate-order-consistency only"]
        and all(row["independently_disableable"] for row in designs)
        and not any("combined" in variant.lower() for variant in variants)
    )
    gate(
        gates, "design", "", "intervention_independence_closure",
        ["baseline", "direction-consistency only", "candidate-order-consistency only"],
        variants, independent, "intervention variants are not independent",
    )
    gate(
        gates, "design", "", "no_combined_first_stage_variant", False,
        any("combined" in variant.lower() for variant in variants),
        not any("combined" in variant.lower() for variant in variants),
        "combined first-stage variant is prohibited",
    )
    serialized_design = canonical(designs)
    forbidden_hits = sorted(term for term in PROHIBITED if term.lower() in serialized_design.lower())
    gate(
        gates, "boundary", "", "gold_safety_target_nondependency", [],
        forbidden_hits, not forbidden_hits,
        "a prohibited target or feature entered intervention design",
    )
    data_boundary = {
        "main_classification_data": MAIN_DATA,
        "time_swap_in_main_classifier_label_training": False,
        "external_or_ood_stability_data": False,
        "stability_inputs": "same in-scope main classification rows only",
    }
    gate(
        gates, "boundary", "", "time_swap_exclusion",
        {"main_data": MAIN_DATA, "time_swap_in_main": False, "external_or_ood": False},
        {"main_data": data_boundary["main_classification_data"],
         "time_swap_in_main": data_boundary["time_swap_in_main_classifier_label_training"],
         "external_or_ood": data_boundary["external_or_ood_stability_data"]},
        data_boundary["main_classification_data"] == MAIN_DATA
        and not data_boundary["time_swap_in_main_classifier_label_training"]
        and not data_boundary["external_or_ood_stability_data"],
        "main-data boundary changed",
    )
    detached = feasibility["counterfactual_gradient_path"] == "MINIMAL_GRADIENT_INSTRUMENTATION_REQUIRED"
    direction_ready = False
    order_ready = False
    simple_insufficient = False
    decision, next_stage, decisions = decision_rows(
        direction_ready, order_ready, detached, simple_insufficient
    )
    gate(
        gates, "decision", "", "decision_hierarchy_reachability",
        {"exactly_one_reached": True}, {"reached": sum(row["reached"] for row in decisions),
                                        "decision": decision},
        sum(row["reached"] for row in decisions) == 1,
        "decision hierarchy is unreachable",
    )
    gate(
        gates, "output", "", "exact_nine_file_closure", sorted(OUTPUTS),
        sorted(OUTPUTS), len(OUTPUTS) == len(set(OUTPUTS)) == 9,
        "output declaration is not exactly nine files",
    )
    trial_manifest = {
        "stage": STAGE,
        "precommitted_variants": variants,
        "combined_variant_in_first_causal_experiment": False,
        "shared_fixed_configuration": {
            "seeds": list(SEEDS),
            "main_classification_data": MAIN_DATA,
            "activation": {"rule": "last_N_epochs", "N": 3,
                           "hard_coded_epoch_numbers": False},
            "evaluation": [
                "clean-dev primary metrics", "P2 exact composer reproduction",
                "P4 topology metrics", "support recall", "false entitlement",
                "false not-entitled", "polarity errors",
            ],
            "topology_stability_guarantees_classifier_safety": False,
            "coefficient_sweep": False,
            "optimization_against_p4_safety_outcomes": False,
        },
        "variants": designs,
        "teacher_precommitment": {
            "status": "UNAVAILABLE_WITHOUT_ADDITIONAL_INSTRUMENTATION",
            "conceptual_preference": "stop-gradient EMA teacher",
            "implementation_authorized": False,
            "reason": "EMA follows the training trajectory without hard-coding a pre-tail checkpoint, but exact live candidate geometry and counterpart-arm provenance do not yet exist.",
            "frozen_pre_tail_anchor_selected": False,
        },
    }
    analysis = {
        "stage": STAGE, "decision": decision,
        "recommended_next_stage": next_stage, "blocking_reasons": [],
        "current_git_commit": ns.current_git_commit,
        "scientific_scope": {
            "design_and_source_feasibility_only": True, "intervention_implemented": False,
            "training_performed": False, "model_behavior_changed": False,
            "safety_targets_evaluated": False, "model_or_checkpoint_loaded": False,
        },
        "source_paths": {
            "p4_analysis": str(p4_path), "p3_analysis": str(p3_path),
            "p2_analysis": str(p2_path),
        },
        "source_hashes": {
            name: sha256(p4_path.parent / name) for name in P4_OUTPUTS
        } | {
            "stage196b2b6p3_analysis.json": sha256(p3_path),
            "stage196b2b6p2_analysis.json": sha256(p2_path),
        },
        "p4_evidence_consumed": observed_counts,
        "p4_required_artifacts_loaded": [
            P4_OUTPUTS[0], P4_OUTPUTS[3], P4_OUTPUTS[5], P4_OUTPUTS[6],
            P4_OUTPUTS[7], P4_OUTPUTS[4], P4_OUTPUTS[8], P4_OUTPUTS[10],
        ],
        "p4_dictionary_rows_loaded": len(dictionary),
        "p4_leakage_rows_loaded": len(leakage),
        "p2_endpoint_reproduction": reproduction,
        "instability_localization": {
            "row_count": len(localization),
            "trajectory_count": localization_trajectories,
            "dimensions": [
                "seed", "candidate mask", "candidate action key",
                "signed response coordinate", "instability family",
            ],
            "coordinates": list(PRIMARY_COORDINATES),
            "pooled_mechanism_claim": False,
        },
        "pairwise_gap_audit": pairwise_summary,
        "source_feasibility": feasibility,
        "intervention_families": {
            "A": designs[1], "B": designs[2],
            "separation_rule": "The first causal experiment contains no combined variant.",
        },
        "activation_scope": {
            "selected": "last N epochs", "N": 3,
            "rationale": "P4 evidence is tail-local and the loop exposes total epochs deterministically; hard-coded epoch 18-20 would not generalize.",
            "classification_loss_stabilization_trigger_selected": False,
            "reason_not_metric_trigger": "No deterministic precommitted stabilization detector exists in current source.",
        },
        "data_boundary": data_boundary,
        "trial_design": trial_manifest,
        "decision_hierarchy": decisions,
        "exact_outputs": list(OUTPUTS),
        "remaining_risks": [
            "Exact live three-candidate geometry is absent from the training graph.",
            "The counterpart frame_local_only arm requires explicit provenance and at least one extra full forward.",
            "No EMA/frozen-anchor teacher is currently implemented or scientifically authorized.",
            "The top1-runner-up response is piecewise because winner identity can change.",
            "Pairwise gap distributions describe magnitudes exactly but select no magnitude threshold.",
            "Improved topology would not by itself establish classifier safety.",
        ],
    }
    return analysis, {
        "source": source_rows, "localization": localization,
        "pairwise": pairwise_rows, "designs": designs,
        "trial": trial_manifest, "decisions": decisions,
    }


def render_report(analysis: dict[str, Any]) -> str:
    return f"""# Stage196-B2-B6P5 Training-Side Response-Stability Intervention Design

## Decision

`{analysis["decision"]}`

Recommended next stage: `{analysis["recommended_next_stage"]}`.

Blocking reasons: `{canonical(analysis["blocking_reasons"])}`.

## Frozen evidence and localization

P5 consumes and preserves the exact P4 instability counts:

`{canonical(analysis.get("p4_evidence_consumed", {}))}`

Localization is not pooled into one mechanism. It is partitioned by seed,
candidate mask, row-specific candidate action key, each of the four primary
signed response coordinates, and the exact combination of within-tail,
candidate-relative, and cross-seed instability families.

## Pairwise-gap audit

Every row, seed, epoch, primary coordinate, and unordered candidate pair is
audited. Exact gaps, exact ties, epoch-18/19/20 gaps, sign sequences, strict
order reversals, cross-seed order disagreement, distributions, and quantiles
are reported. No magnitude threshold is selected and no gap is called small or
large. The exact tie-contact versus strict-sign-reversal result is:

`{canonical(analysis.get("pairwise_gap_audit", {}))}`

## Training source feasibility

`{canonical(analysis.get("source_feasibility", {}))}`

Native final logits and their signed margins retain autograd connectivity.
P4's exact response geometry does not: it recomposes exported Python values
from separately trained joint and frame-local-only arms. The three candidate
compositions can eventually be vectorized, but a live counterpart-arm state
and exact row-specific action application boundary must first be instrumented.

## Intervention A: signed response direction

Family A compares a stop-gradient teacher sign with the student's signed
margin response, ignores exact teacher ties, and uses a logistic sign loss. It
matches no raw score and uses no labels, row identity, seed feature, safety
target, or global threshold.

## Intervention B: candidate-relative order

Family B centers each candidate response within the exact three-candidate row
geometry, derives teacher pair signs, ignores exact teacher ties, and applies a
logistic pairwise topology loss. It imposes no fixed universal action order
and performs no lexical tie-breaking.

## Separation, activation, and data

The precommitted variants are baseline, direction-consistency only, and
candidate-order-consistency only. They remain separate because A targets
temporal sign topology for each action response, whereas B targets within-row
relative action geometry. A combined variant is deferred until both have
independent causal results.

Activation is the deterministic last `N=3` epochs, not literal epochs 18-20.
This follows the tail-local P4 evidence and is implementable for any configured
training length. A classification-loss-stability trigger is not selected
because the source has no precommitted deterministic detector.

Main classification data remains
`data/controlled_v5_v3_without_time_swap.jsonl`. `time_swap`, external data,
and OOD data do not define or supervise either stability loss.

## Trial and interpretation boundary

One coefficient (`0.1`) is precommitted per independently run family; there is
no sweep. Evaluation includes clean-dev primary metrics, P2 reproduction, P4
topology metrics, support recall, false entitlement, false not-entitled, and
polarity errors. Topology stability alone is not a classifier-safety claim.

P5 is design-only. It implements no intervention, trains nothing, loads no
model/checkpoint, changes no behavior, and evaluates no safety target.
"""


def csv_value(value: Any) -> Any:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite CSV value")
        return SERIALIZED_PRECISION % value
    if isinstance(value, (dict, list, tuple)):
        return canonical(value)
    return "" if value is None else value


def render_csv(header: Sequence[str], rows: Iterable[dict[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=list(header), extrasaction="raise", lineterminator="\n"
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({field: csv_value(row.get(field)) for field in header})
    return buffer.getvalue()


def blocked_payload(ns: argparse.Namespace, reason: str) -> tuple[dict[str, Any], dict[str, Any]]:
    decisions = [{
        "order": 0, "decision": "STAGE196B2B6P5_BLOCKED_CONTRACT_FAILURE",
        "condition": "source, upstream, population, boundary, or output contract failure",
        "observed": reason, "reached": True,
        "recommended_next_stage": "STAGE196B2B6P5_REPAIR_CONTRACT",
    }]
    analysis = {
        "stage": STAGE, "decision": "STAGE196B2B6P5_BLOCKED_CONTRACT_FAILURE",
        "recommended_next_stage": "STAGE196B2B6P5_REPAIR_CONTRACT",
        "blocking_reasons": [reason], "current_git_commit": ns.current_git_commit,
        "scientific_scope": {
            "design_and_source_feasibility_only": True,
            "intervention_implemented": False, "training_performed": False,
            "model_behavior_changed": False, "safety_targets_evaluated": False,
            "model_or_checkpoint_loaded": False,
        },
        "source_paths": {}, "source_hashes": {}, "p4_evidence_consumed": {},
        "p4_required_artifacts_loaded": [], "p4_dictionary_rows_loaded": 0,
        "p4_leakage_rows_loaded": 0, "p2_endpoint_reproduction": {},
        "instability_localization": {}, "pairwise_gap_audit": {},
        "source_feasibility": {}, "intervention_families": {},
        "activation_scope": {}, "data_boundary": {}, "trial_design": {},
        "decision_hierarchy": decisions, "exact_outputs": list(OUTPUTS),
        "remaining_risks": ["Repair the failed contract before scientific interpretation."],
    }
    return analysis, {
        "source": [], "localization": [], "pairwise": [],
        "designs": [], "trial": {}, "decisions": decisions,
    }


def payloads(
    analysis: dict[str, Any], tables: dict[str, Any],
    gates: list[dict[str, Any]],
) -> dict[str, str]:
    result = {
        OUTPUTS[0]: json.dumps(analysis, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        OUTPUTS[1]: render_report(analysis),
        OUTPUTS[2]: render_csv(SOURCE_H, tables["source"]),
        OUTPUTS[3]: render_csv(LOCALIZATION_H, tables["localization"]),
        OUTPUTS[4]: render_csv(PAIRWISE_H, tables["pairwise"]),
        OUTPUTS[5]: render_csv(DESIGN_H, tables["designs"]),
        OUTPUTS[6]: json.dumps(tables["trial"], indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        OUTPUTS[7]: render_csv(DECISION_H, tables["decisions"]),
        OUTPUTS[8]: render_csv(CONTRACT_H, gates),
    }
    if set(result) != set(OUTPUTS):
        raise ValueError("exact nine-file payload closure failed")
    return result


def atomic_write(output: Path, contents: dict[str, str]) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for name, content in contents.items():
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
        if ns.output_dir.resolve().exists():
            return 2
        analysis, tables = analyze(ns, gates)
        if any(not row["passed"] for row in gates):
            raise ValueError("contract contains a failed gate")
        atomic_write(ns.output_dir.resolve(), payloads(analysis, tables, gates))
        return 0
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        if not any(not row["passed"] for row in gates):
            gate(
                gates, "exception", "", "unhandled_exception", None, reason,
                False, reason, fatal=False,
            )
        analysis, tables = blocked_payload(ns, reason)
        try:
            atomic_write(ns.output_dir.resolve(), payloads(analysis, tables, gates))
        except FileExistsError:
            pass
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
