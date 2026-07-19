#!/usr/bin/env python3
"""Stage196-B2-A artifact-only FrameGate-to-entitlement propagation audit.

The analyzer consumes Stage196-B2-P0 native epoch-channel sidecars and fails closed
on any provenance, schema, or alignment disagreement. It never loads a model or
checkpoint and never substitutes selected-checkpoint scalars for epoch scalars.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import statistics
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import analyze_stage196b1_framegate_gradient_ownership as b1c

STAGE = "Stage196-B2-A"
SEEDS = (183, 184, 185)
MODES = ("joint", "frame_local_only")
RUNS = tuple(f"seed{seed}_{mode}" for seed in SEEDS for mode in MODES)
TAIL = (18, 19, 20)
EPOCHS = tuple(range(1, 21))
ROW_COUNT = 720
LABELS = ("REFUTE", "NOT_ENTITLED", "SUPPORT")
TOLERANCE = 1e-6
THRESHOLD = 0.5
FRAMEGATE_IMPLEMENTATION_ORIGIN_COMMIT = "5a39538ef3ca8f36cc2cc5d3290eae60d6a5f5c8"
B1C_DECISION = "STAGE196B1C_MIXED_GRADIENT_OWNERSHIP_EFFECT"
B1C_NEXT = "STAGE196B2_NO_PROMOTION_TARGETED_CAUSAL_FOLLOWUP"
INCOMPLETE = "STAGE196B2A_ANALYSIS_INCOMPLETE"

DECISIONS = (
    "STAGE196B2A_FRAME_REMAINS_PRIMARY_BOTTLENECK",
    "STAGE196B2A_PREDICATE_PROPAGATION_BOTTLENECK",
    "STAGE196B2A_SUFFICIENCY_PROPAGATION_BOTTLENECK",
    "STAGE196B2A_POLARITY_PROPAGATION_BOTTLENECK",
    "STAGE196B2A_ENTITLEMENT_AGGREGATION_BOTTLENECK",
    "STAGE196B2A_FINAL_COMPOSITION_BOTTLENECK",
    "STAGE196B2A_SEED_SPECIFIC_MIXED_PROPAGATION",
    INCOMPLETE,
)
NEXT_STAGE = {
    DECISIONS[0]: "STAGE196B2B_RETURN_TO_FRAME_REPRESENTATION_INTERVENTION_DESIGN",
    DECISIONS[1]: "STAGE196B2B_FRAME_PREDICATE_COMPATIBILITY_CAUSAL_DESIGN",
    DECISIONS[2]: "STAGE196B2B_SUFFICIENCY_PROPAGATION_CAUSAL_DESIGN",
    DECISIONS[3]: "STAGE196B2B_POLARITY_PROPAGATION_CAUSAL_DESIGN",
    DECISIONS[4]: "STAGE196B2B_ENTITLEMENT_AGGREGATION_CAUSAL_DESIGN",
    DECISIONS[5]: "STAGE196B2B_FINAL_LOGIT_COMPOSITION_CAUSAL_DESIGN",
    DECISIONS[6]: "STAGE196B2B_NO_PROMOTION_MINIMAL_SEED_SPECIFIC_FOLLOWUP",
    INCOMPLETE: "STAGE196B2A_REPAIR_ANALYSIS_INPUTS",
}
BLOCKERS = (
    "FRAME_REMAINS_SUBTHRESHOLD", "PREDICATE_BLOCKED", "SUFFICIENCY_BLOCKED",
    "POLARITY_BLOCKED", "ENTITLEMENT_AGGREGATION_BLOCKED",
    "FINAL_COMPOSITION_BLOCKED", "MULTI_CHANNEL_DEGRADATION",
    "PROPAGATED_TO_SUPPORT", "UNRESOLVED_PROPAGATION",
)
FIRST_BLOCKER_ORDER = (
    "FRAME_REMAINS_SUBTHRESHOLD", "PREDICATE_BLOCKED", "SUFFICIENCY_BLOCKED",
    "POLARITY_BLOCKED", "ENTITLEMENT_AGGREGATION_BLOCKED",
    "FINAL_COMPOSITION_BLOCKED",
)
STATUS_CLASSES = (
    "STABLE_SUPPORT", "PERSISTENT_NOT_ENTITLED", "PERSISTENT_REFUTE", "UNSTABLE")
TRANSITIONS = (
    "RESCUE_NE_TO_STABLE_SUPPORT", "HARM_STABLE_SUPPORT_TO_PERSISTENT_NE",
    "HARM_STABLE_SUPPORT_TO_PERSISTENT_REFUTE", "HARM_STABLE_SUPPORT_TO_UNSTABLE",
    "PERSISTENT_NE_BOTH", "STABLE_SUPPORT_BOTH", "NEW_PERSISTENT_NE_FROM_UNSTABLE",
    "PERSISTENT_REFUTE_BOTH", "OTHER_TRANSITION",
)
SETS = (
    "stage196a_baseline_recurrent", "stage196a_intervention_recurrent",
    "stage196a_common_recurrent", "stage196a_universal_all_six",
)
POPULATIONS = (
    "all_gold_support", *SETS, "joint_persistent_support_to_not_entitled",
    "intervention_persistent_support_to_not_entitled",
    "baseline_defined_stable_correct_support_controls", "intervention_induced_harm",
    "rescue_rows",
)
OUTPUTS = (
    "stage196b2a_analysis.json", "stage196b2a_report.md",
    "stage196b2a_seed_summary.csv", "stage196b2a_support_transition_rows.csv",
    "stage196b2a_channel_transition_summary.csv",
    "stage196b2a_recurrent_position_propagation.csv",
    "stage196b2a_harm_rescue_rows.csv", "stage196b2a_epoch_propagation.csv",
    "stage196b2a_contract.csv",
)
B1C_FILES = (
    "stage196b1c_analysis.json", "stage196b1c_report.md", "stage196b1c_run_summary.csv",
    "stage196b1c_paired_seed_deltas.csv", "stage196b1c_tail3_persistent_rows.csv",
    "stage196b1c_recurrent_position_effects.csv", "stage196b1c_epoch_trajectory.csv",
    "stage196b1c_contract.csv",
)

# These are explicit semantic ownership rules.  There is no recursive field search.
RESOLVED_SCHEMA = {
    "stable_row_id": "clean_dev_predictions.json:predictions[].id and clean_dev_scalars.jsonl:[].id",
    "trajectory_row_id": "stage191_dev_predictions_epoch_NNN.jsonl:[].source_row_id",
    "dev_position": "stage191_dev_predictions_epoch_NNN.jsonl:[].dev_position",
    "gold_final_label": "predictions[].gold_final_label / trajectory[].gold_final_label",
    "selected_predicted_final_label": "predictions[].pred_final_label",
    "epoch_predicted_final_label": "trajectory[].predicted_final_label",
    "selected_frame_probability": "predictions[].frame_prob (cross-checked with scalars[].frame_prob)",
    "epoch_frame_probability": "stage196b2p0 sidecar frame_probability",
    "predicate_probability": "stage196b2p0 sidecar predicate_coverage_probability",
    "sufficiency_probability": "stage196b2p0 sidecar sufficiency_probability",
    "polarity_support_facing_margin": "stage196b2p0 sidecar polarity_support_margin",
    "entitlement_probability": "stage196b2p0 sidecar entitlement_probability",
    "selected_support_probability": "predictions[].final_probs[2]",
    "selected_not_entitled_probability": "predictions[].final_probs[1]",
    "epoch_support_logit": "trajectory[].final_logits[2]",
    "epoch_not_entitled_logit": "trajectory[].final_logits[1]",
    "intervention_type": "predictions[].intervention_type",
}
EPOCH_REQUIRED_SCALARS = (
    "frame_probability", "predicate_coverage_probability",
    "sufficiency_probability", "polarity_support_margin",
    "entitlement_probability", "support_probability",
    "not_entitled_probability", "support_logit", "not_entitled_logit")
SIDECAR_FIELDS = ("id", "source_row_id", "dev_position", "gold_label", "prediction",
                  "intervention_type", *EPOCH_REQUIRED_SCALARS, "epoch", "training_seed",
                  "frame_downstream_gradient_mode")

CONTRACT_HEADER = ["scope", "run", "gate", "required", "observed", "passed", "blocking_reason"]
SEED_HEADER = ["seed", "common_recurrent_frame_shift", "primary_decision_population_size",
               "largest_blocker", "largest_blocker_rate", "rescue_count", "harm_count",
               "selected_recurrent_propagation_rate", "tail3_recurrent_propagation_rate",
               "selected_tail3_agreement", "decision_contribution"]
BASE_CHANNELS = ("frame", "predicate", "sufficiency", "polarity", "entitlement",
                 "support_vs_not_entitled_margin", "support_probability",
                 "not_entitled_probability")
TRANSITION_HEADER = [
    "seed", "stable_row_id", "dev_position", "analysis_view", "gold_final_label",
    "intervention_type", "joint_selected_epoch", "intervention_selected_epoch",
    "joint_final_class_or_pattern", "intervention_final_class_or_pattern",
    "joint_tail3_status", "intervention_tail3_status", "paired_transition_class",
    "propagation_category", "first_blocker",
] + [f"{prefix}_{channel}{suffix}" for channel in BASE_CHANNELS
     for prefix, suffix in (("joint", "_value"), ("intervention", "_value"),
                            ("paired", "_delta"))] + [
    f"{channel}_joint_pass" for channel in BASE_CHANNELS[:5]
] + [f"{channel}_intervention_pass" for channel in BASE_CHANNELS[:5]] + [
    f"{channel}_threshold_crossing" for channel in BASE_CHANNELS[:5]
] + [f"in_{name}" for name in SETS]
SUMMARY_HEADER = ["seed", "analysis_view", "population", "propagation_category",
                  "count", "denominator", "rate", "frame_up_count", "frame_down_count",
                  "frame_fail_to_pass_count", "frame_pass_to_fail_count",
                  "frame_up_predicate_positive_fraction", "frame_up_entitlement_positive_fraction",
                  "frame_up_margin_improved_fraction", "selected_support_propagation_fraction",
                  "tail3_stable_support_propagation_fraction", "rescue_count", "harm_count",
                  "persistent_failure_count", "spearman_frame_predicate",
                  "spearman_frame_entitlement", "spearman_frame_final_margin",
                  "sign_concordance_frame_entitlement", "sign_concordance_frame_final_margin"]
RECURRENT_HEADER = ["seed", "recurrent_set", "stable_row_id", "dev_position",
                    "intervention_type", "selected_joint_frame", "selected_intervention_frame",
                    "selected_frame_delta", "selected_propagation_category",
                    "tail3_joint_frame", "tail3_intervention_frame", "tail3_frame_delta",
                    "tail3_joint_status", "tail3_intervention_status",
                    "tail3_propagation_category"]
HARM_HEADER = ["seed", "stable_row_id", "dev_position", "transition_role",
               "intervention_type", "joint_tail3_status", "intervention_tail3_status",
               "selected_class_transition", "first_blocker", "frame_increased_despite_harm",
               "entitlement_increased_despite_harm", "support_vs_ne_margin_decreased"] + [
    f"in_{name}" for name in SETS] + [f"{prefix}_{channel}{suffix}" for channel in BASE_CHANNELS
    for prefix, suffix in (("joint", "_value"), ("intervention", "_value"),
                           ("paired", "_delta"))] + [
    f"{channel}_threshold_crossing" for channel in BASE_CHANNELS[:5]]
EPOCH_HEADER = ["seed", "epoch", "is_tail3_epoch", "joint_selected_epoch",
                "intervention_selected_epoch", "is_joint_selected_epoch",
                "is_intervention_selected_epoch", "mean_delta_frame_gold_support",
                "mean_delta_predicate_gold_support", "mean_delta_sufficiency_gold_support",
                "mean_delta_polarity_gold_support", "mean_delta_entitlement_gold_support",
                "mean_delta_support_vs_ne_margin_gold_support", "frame_fail_to_pass_count",
                "frame_pass_to_fail_count", "support_rescue_count", "support_harm_count",
                "false_not_entitled_delta", "false_entitlement_delta",
                "common_recurrent_frame_delta", "common_recurrent_entitlement_delta",
                "common_recurrent_support_vs_ne_margin_delta"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--stage196a-report-json", required=True, type=Path)
    parser.add_argument("--stage196b1c-analysis-json", required=True, type=Path)
    parser.add_argument("--current-git-commit", required=True)
    parser.add_argument("--stage196b1-runtime-git-commit", required=True)
    parser.add_argument("--stage196b2p0-runtime-git-commit", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path}: CSV header absent")
        return list(reader)


def add_gate(rows: list[dict[str, Any]], scope: str, run: str, name: str,
             required: Any, observed: Any, passed: bool, reason: str) -> None:
    rows.append({"scope": scope, "run": run, "gate": name, "required": required,
                 "observed": observed, "passed": passed,
                 "blocking_reason": "" if passed else reason})
    if not passed:
        raise ValueError(f"{run + ': ' if run else ''}{reason}")


def safe_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    repo = args.repo_root.resolve()
    reports = (repo / "reports").resolve()
    run_root = args.run_root.resolve()
    stage_a = args.stage196a_report_json.resolve()
    stage_c = args.stage196b1c_analysis_json.resolve()
    output = args.output_dir.resolve()
    if not repo.is_dir() or not reports.is_dir():
        raise ValueError("repository or reports directory absent")
    if run_root != (reports / "stage196b2p0_epoch_channel_observability_runs").resolve():
        raise ValueError("run root is not the exact Stage196-B2-P0 observability run root")
    if not run_root.is_dir() or not stage_a.is_file() or not stage_c.is_file():
        raise FileNotFoundError("required run root, Stage196-A report, or B1-C analysis absent")
    if stage_c.name != "stage196b1c_analysis.json":
        raise ValueError("B1-C input must be stage196b1c_analysis.json")
    if reports != output.parent and reports not in output.parents:
        raise ValueError("output must be below repository reports")
    if output in (repo, reports, run_root, stage_a.parent, stage_c.parent):
        raise ValueError("unsafe or colliding output directory")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output directory exists and is nonempty")
    return repo, run_root, stage_a, stage_c, output


def close_b1c(path: Path, gates: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, str]]]:
    directory = path.parent
    missing = [name for name in B1C_FILES if not (directory / name).is_file()]
    add_gate(gates, "stage196b1c", "", "exact_companion_file_closure", [], missing,
             not missing, "Stage196-B1-C companion closure is incomplete")
    analysis = read_json(path)
    required = {"decision": B1C_DECISION, "recommended_next_stage": B1C_NEXT,
                "blocking_reasons": []}
    observed = {key: analysis.get(key) for key in required} if type(analysis) is dict else {}
    add_gate(gates, "stage196b1c", "", "completed_mixed_decision_closure", required,
             observed, observed == required, "Stage196-B1-C decision closure mismatch")
    contract = read_csv(directory / "stage196b1c_contract.csv")
    contract_ok = bool(contract) and all(row.get("passed", "").lower() == "true"
                                         and row.get("blocking_reason", "") == ""
                                         for row in contract)
    add_gate(gates, "stage196b1c", "", "all_contract_rows_passed", True,
             contract_ok, contract_ok, "Stage196-B1-C contract contains a failure")
    summary = read_csv(directory / "stage196b1c_run_summary.csv")
    summary_runs = [row.get("run") for row in summary]
    add_gate(gates, "stage196b1c", "", "exact_six_run_summary", list(RUNS), summary_runs,
             summary_runs == list(RUNS), "Stage196-B1-C run summary mismatch")
    selected = {row["run"]: int(row["selected_best_epoch"]) for row in summary
                if row.get("run") in RUNS and row.get("selected_best_epoch", "").isdigit()}
    add_gate(gates, "stage196b1c", "", "reported_selected_epochs",
             b1c.BEST_EPOCHS, selected, selected == b1c.BEST_EPOCHS,
             "Stage196-B1-C selected epochs mismatch")
    paired = read_csv(directory / "stage196b1c_paired_seed_deltas.csv")
    required_delta_columns = {"metric", "seed183_delta", "seed184_delta", "seed185_delta"}
    paired_ok = bool(paired) and required_delta_columns.issubset(paired[0])
    add_gate(gates, "stage196b1c", "", "paired_delta_seed_columns",
             sorted(required_delta_columns), sorted(paired[0]) if paired else [], paired_ok,
             "Stage196-B1-C paired delta table lacks seeds 183-185")
    epoch_rows = read_csv(directory / "stage196b1c_epoch_trajectory.csv")
    add_gate(gates, "stage196b1c", "", "epoch_trajectory_120_rows", 120,
             len(epoch_rows), len(epoch_rows) == 120,
             "Stage196-B1-C epoch trajectory must contain 120 rows")
    tail = read_csv(directory / "stage196b1c_tail3_persistent_rows.csv")
    tail_runs = set(row.get("run", "") for row in tail)
    add_gate(gates, "stage196b1c", "", "tail3_six_run_population", list(RUNS),
             sorted(tail_runs), tail_runs == set(RUNS),
             "Stage196-B1-C tail-three table does not cover all six runs")
    recurrent = read_csv(directory / "stage196b1c_recurrent_position_effects.csv")
    recurrent_sets = set(row.get("position_set", "") for row in recurrent)
    add_gate(gates, "stage196b1c", "", "recurrent_table_set_closure", list(SETS),
             sorted(recurrent_sets), recurrent_sets == set(SETS),
             "Stage196-B1-C recurrent table set closure mismatch")
    return analysis, paired


def derive_positive_seeds(paired: list[dict[str, str]],
                          gates: list[dict[str, Any]]) -> tuple[list[int], list[int], dict[int, float]]:
    metric = "mean_frame_probability_stage196a_common_recurrent"
    rows = [row for row in paired if row.get("metric") == metric]
    add_gate(gates, "stage196b1c", "", "unique_common_recurrent_frame_delta_row", 1,
             len(rows), len(rows) == 1, "common-recurrent frame delta row missing or duplicated")
    deltas = {seed: float(rows[0][f"seed{seed}_delta"]) for seed in SEEDS}
    recorded = {183: -0.136243, 184: 0.140103, 185: 0.147282}
    reproduced = all(math.isclose(deltas[s], recorded[s], rel_tol=0.0, abs_tol=TOLERANCE)
                     for s in SEEDS)
    add_gate(gates, "stage196b1c", "", "recorded_frame_deltas_reproduced", recorded,
             deltas, reproduced, "B1-C common-recurrent frame deltas disagree with closure")
    positive = [seed for seed in SEEDS if deltas[seed] > TOLERANCE]
    negative = [seed for seed in SEEDS if deltas[seed] < -TOLERANCE]
    add_gate(gates, "stage196b1c", "", "derived_positive_and_contrast_seeds",
             {"positive": [184, 185], "negative": [183]},
             {"positive": positive, "negative": negative},
             positive == [184, 185] and negative == [183],
             "derived positive/negative frame-shift seeds do not reproduce B1-C")
    return positive, negative, deltas


def status(labels: Sequence[str]) -> str:
    pattern = tuple(labels)
    if pattern == ("SUPPORT",) * 3: return "STABLE_SUPPORT"
    if pattern == ("NOT_ENTITLED",) * 3: return "PERSISTENT_NOT_ENTITLED"
    if pattern == ("REFUTE",) * 3: return "PERSISTENT_REFUTE"
    return "UNSTABLE"


def paired_transition(joint: str, intervention: str) -> str:
    if joint == "PERSISTENT_NOT_ENTITLED" and intervention == "STABLE_SUPPORT":
        return "RESCUE_NE_TO_STABLE_SUPPORT"
    if joint == "STABLE_SUPPORT" and intervention == "PERSISTENT_NOT_ENTITLED":
        return "HARM_STABLE_SUPPORT_TO_PERSISTENT_NE"
    if joint == "STABLE_SUPPORT" and intervention == "PERSISTENT_REFUTE":
        return "HARM_STABLE_SUPPORT_TO_PERSISTENT_REFUTE"
    if joint == "STABLE_SUPPORT" and intervention == "UNSTABLE":
        return "HARM_STABLE_SUPPORT_TO_UNSTABLE"
    if joint == intervention == "PERSISTENT_NOT_ENTITLED": return "PERSISTENT_NE_BOTH"
    if joint == intervention == "STABLE_SUPPORT": return "STABLE_SUPPORT_BOTH"
    if joint == "UNSTABLE" and intervention == "PERSISTENT_NOT_ENTITLED":
        return "NEW_PERSISTENT_NE_FROM_UNSTABLE"
    if joint == intervention == "PERSISTENT_REFUTE": return "PERSISTENT_REFUTE_BOTH"
    return "OTHER_TRANSITION"


def crossing(before: bool, after: bool) -> str:
    return ("FAIL_TO_PASS" if not before and after else "PASS_TO_FAIL" if before and not after
            else "PASS_TO_PASS" if before else "FAIL_TO_FAIL")


def signed(value: float) -> int:
    return 1 if value > TOLERANCE else -1 if value < -TOLERANCE else 0


def rank(values: Sequence[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]: end += 1
        mean_rank = (start + 1 + end) / 2.0
        for index in ordered[start:end]: result[index] = mean_rank
        start = end
    return result


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2: return None
    if max(left) == min(left) or max(right) == min(right): return None
    x, y = rank(left), rank(right)
    mx, my = statistics.fmean(x), statistics.fmean(y)
    numerator = math.fsum((a - mx) * (b - my) for a, b in zip(x, y))
    denominator = math.sqrt(math.fsum((a - mx) ** 2 for a in x)
                            * math.fsum((b - my) ** 2 for b in y))
    return numerator / denominator if denominator else None


def classify_propagation(delta_frame: float, intervention: dict[str, Any],
                         crossings: dict[str, str], view: str) -> str:
    if delta_frame <= TOLERANCE:
        return "UNRESOLVED_PROPAGATION"
    downstream_failures = sum(crossings[name] == "PASS_TO_FAIL"
                              for name in ("predicate", "sufficiency", "polarity", "entitlement"))
    if downstream_failures > 1:
        return "MULTI_CHANNEL_DEGRADATION"
    propagated = (intervention["prediction"] == "SUPPORT" if view == "selected"
                  else intervention["status"] == "STABLE_SUPPORT")
    if propagated: return "PROPAGATED_TO_SUPPORT"
    if intervention["frame"] < THRESHOLD: return "FRAME_REMAINS_SUBTHRESHOLD"
    if intervention["predicate"] < THRESHOLD: return "PREDICATE_BLOCKED"
    if intervention["sufficiency"] < THRESHOLD: return "SUFFICIENCY_BLOCKED"
    if intervention["polarity"] < 0: return "POLARITY_BLOCKED"
    if intervention["entitlement"] < THRESHOLD: return "ENTITLEMENT_AGGREGATION_BLOCKED"
    if intervention["margin"] <= TOLERANCE or not propagated:
        return "FINAL_COMPOSITION_BLOCKED"
    return "UNRESOLVED_PROPAGATION"


def choose_decision(positive: Sequence[int], distributions: dict[int, Counter[str]],
                    harm_distributions: dict[int, Counter[str]],
                    selected_tail_conflict: bool, one_seed_dominates: bool) -> tuple[str, dict[str, Any]]:
    eligible = {seed: sum(distributions.get(seed, Counter()).values()) for seed in positive}
    largest: dict[int, tuple[str | None, int, float | None]] = {}
    for seed in positive:
        counts = distributions.get(seed, Counter())
        if not counts:
            largest[seed] = (None, 0, None)
        else:
            name, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
            largest[seed] = (name, count, count / eligible[seed])
    same = len({largest[s][0] for s in positive}) == 1 and bool(positive)
    enough = len(positive) >= 2 and all(eligible[s] >= 5 for s in positive)
    majority = enough and same and all((largest[s][2] or 0) >= 0.5 for s in positive)
    candidate = largest[positive[0]][0] if positive and same else None
    conflicting_harm = False
    if candidate:
        other_seed_count = 0
        for seed, counts in harm_distributions.items():
            if counts:
                harm_largest = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
                other_seed_count += harm_largest != candidate
        conflicting_harm = other_seed_count >= 2
    mapping = dict(zip(FIRST_BLOCKER_ORDER, DECISIONS[:6]))
    consistent = majority and candidate in mapping and not conflicting_harm
    decision = mapping[candidate] if consistent else (INCOMPLETE if not enough
                else "STAGE196B2A_SEED_SPECIFIC_MIXED_PROPAGATION")
    return decision, {"positive_frame_shift_seed_count_at_least_two": len(positive) >= 2,
                      "eligible_rows_at_least_five_per_seed": enough,
                      "largest_blocker_by_seed": {str(k): v[0] for k, v in largest.items()},
                      "largest_blocker_rate_by_seed": {str(k): v[2] for k, v in largest.items()},
                      "same_largest_blocker": same, "at_least_half_every_seed": majority,
                      "stable_correct_harm_conflicts_in_two_or_more_seeds": conflicting_harm,
                      "selected_tail3_material_conflict": selected_tail_conflict,
                      "one_seed_dominates": one_seed_dominates,
                      "consistent_bottleneck_rule_satisfied": consistent,
                      "selected_decision": decision}


def require_epoch_mediation_schema(run_root: Path, gates: list[dict[str, Any]]) -> None:
    """Require native per-epoch downstream fields; never impute selected values."""
    for run in RUNS:
        for epoch in EPOCHS:
            path = run_root / run / f"stage191_dev_predictions_epoch_{epoch:03d}.jsonl"
            with path.open("r", encoding="utf-8") as handle:
                first_line = handle.readline()
            if not first_line:
                raise ValueError(f"{run}: epoch {epoch} trajectory is empty")
            row = json.loads(first_line)
            missing = [name for name in EPOCH_REQUIRED_SCALARS if name not in row]
            add_gate(gates, "schema", run, f"epoch_{epoch:03d}_native_mediation_scalars",
                     list(EPOCH_REQUIRED_SCALARS), sorted(row) if type(row) is dict else type(row).__name__,
                     not missing,
                     "required native epoch mediation fields absent: " + ", ".join(missing)
                     + "; selected-checkpoint substitution and checkpoint loading are prohibited")


def validate_all(args: argparse.Namespace, gates: list[dict[str, Any]]) -> dict[str, Any]:
    repo, run_root, stage_a_path, stage_c_path, _ = safe_paths(args)
    b1c.validate_source(repo, args.current_git_commit, args.stage196b1_runtime_git_commit, gates)
    entries = [path.name for path in run_root.iterdir()]
    add_gate(gates, "source", "", "exact_six_run_input_matrix", list(RUNS), entries,
             set(entries) == set(RUNS) and all((run_root / run).is_dir() for run in RUNS),
             "run root must contain exactly the six frozen runs")
    stage_c, paired = close_b1c(stage_c_path, gates)
    stage_a, sets = b1c.validate_stage196a(stage_a_path, gates)
    sizes = {name: len(sets[name]) for name in SETS}
    add_gate(gates, "stage196a", "", "recurrent_set_counts", dict(zip(SETS, (22, 19, 19, 10))),
             sizes, sizes == dict(zip(SETS, (22, 19, 19, 10))),
             "Stage196-A recurrent-set count mismatch")
    runs = {run: b1c.validate_run(run_root, run, args.stage196b1_runtime_git_commit, gates)
            for run in RUNS}
    b1c.validate_population(runs, gates)
    runtime_commits = {run: runs[run]["stage196b1_runtime_git_commit"] for run in RUNS}
    add_gate(gates, "source", "", "uniform_stage196b1_runtime_commit",
             args.stage196b1_runtime_git_commit, runtime_commits,
             all(value == args.stage196b1_runtime_git_commit for value in runtime_commits.values()),
             "run runtime commits differ from the supplied historical commit")
    selected_epochs = {run: runs[run]["report"].get("runs", {}).get("single", {}).get("best_epoch")
                       for run in RUNS}
    add_gate(gates, "alignment", "", "selected_epochs_match_artifacts", b1c.BEST_EPOCHS,
             selected_epochs, selected_epochs == b1c.BEST_EPOCHS,
             "reported selected epochs do not match run artifacts")
    positive, negative, deltas = derive_positive_seeds(paired, gates)
    # This final prerequisite is intentionally after all provenance/alignment closure.
    require_epoch_mediation_schema(run_root, gates)
    return {"stage196a": stage_a, "stage196b1c": stage_c, "sets": sets, "runs": runs,
            "positive": positive, "negative": negative, "frame_deltas": deltas}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip(): raise ValueError(f"{path}:{number}: blank row")
            row = json.loads(line)
            if type(row) is not dict: raise ValueError(f"{path}:{number}: non-object")
            rows.append(row)
    return rows


def scalar(row: dict[str, Any], key: str, probability: bool = False) -> float:
    value = row.get(key)
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{key} absent/nonfinite")
    result = float(value)
    if probability and not 0 <= result <= 1: raise ValueError(f"{key} outside [0,1]")
    return result


def load_p0_run(root: Path, run: str, runtime_commit: str,
                gates: list[dict[str, Any]]) -> dict[str, Any]:
    seed, mode, directory = int(run[4:7]), run[8:], root / run
    names = ["training_report.json", "run_provenance.json", "clean_dev_predictions.json",
             "clean_dev_scalars.jsonl", "selected_checkpoint.pt",
             "stage191_trajectory_contract.json", "stage191_trajectory_epoch_metrics.jsonl"]
    names += [f"stage191_dev_predictions_epoch_{e:03d}.jsonl" for e in EPOCHS]
    names += [f"stage196b2p0_epoch_channels_{e:03d}.jsonl" for e in EPOCHS]
    missing = [name for name in names if not (directory / name).is_file()]
    add_gate(gates, "run", run, "p0_artifacts", [], missing, not missing,
             "required Stage196-B2-P0 artifacts absent")
    sidecar_count = len(list(directory.glob("stage196b2p0_epoch_channels_*.jsonl")))
    add_gate(gates, "run", run, "twenty_sidecars", 20, sidecar_count,
             sidecar_count == 20, "sidecar count mismatch")
    report, contract = read_json(directory / "training_report.json"), read_json(directory / "stage191_trajectory_contract.json")
    provenance = read_json(directory / "run_provenance.json")
    source = provenance.get("source_provenance", {})
    commits = [source.get("git_commit") if type(source) is dict else None,
               contract.get("trainer_source_commit")]
    add_gate(gates, "provenance", run, "p0_runtime_commit", runtime_commit, commits,
             commits == [runtime_commit, runtime_commit], "P0 runtime commit mismatch")
    contract_expected = {
        "observability_mode": "stage196b1_framegate_gradient_ownership",
        "training_seed": seed, "split_seed": 174, "epoch_count": 20,
        "expected_dev_rows": 720,
        "stage196b1_framegate_gradient_ownership_observability": True,
        "stage196b2p0_epoch_channel_observability_enabled": True,
        "stage196b2p0_epoch_channel_file_count": 20,
        "stage196b2p0_epoch_channel_rows_per_file": 720,
        "stage196b2p0_required_fields": list(SIDECAR_FIELDS),
        "stage196b2p0_extra_forward_pass_performed": False,
        "stage196b2p0_training_semantics_changed": False,
        "stage196b2p0_gradient_semantics_changed": False,
        "stage196b2p0_checkpoint_selection_changed": False,
        "state_capsule_saving_enabled": False, "expected_state_capsules": 0,
        "parameter_swa_enabled": False, "compatible_positive_margin_enabled": False,
    }
    failures = [key for key, value in contract_expected.items() if contract.get(key) != value]
    add_gate(gates, "provenance", run, "p0_contract", contract_expected,
             {key: contract.get(key) for key in contract_expected}, not failures,
             f"P0 contract mismatch: {failures}")
    obj = read_json(directory / "clean_dev_predictions.json")
    if type(obj) is not dict or set(obj) != {"metadata", "predictions"}: raise ValueError(f"{run}: selected container")
    best = obj["metadata"].get("best_epoch")
    if type(best) is not int or best not in EPOCHS or len(obj["predictions"]) != 720:
        raise ValueError(f"{run}: selected metadata/cardinality")
    selected_raw = {b1c.stable_id(row, run): row for row in obj["predictions"]}
    if len(selected_raw) != 720: raise ValueError(f"{run}: duplicate selected ID")
    epochs, reference_mapping = {}, None
    for epoch in EPOCHS:
        trajectory = read_jsonl(directory / f"stage191_dev_predictions_epoch_{epoch:03d}.jsonl")
        channels = read_jsonl(directory / f"stage196b2p0_epoch_channels_{epoch:03d}.jsonl")
        if len(trajectory) != 720 or len(channels) != 720: raise ValueError(f"{run}:{epoch}: row count")
        trajectory_by_id = {str(row["source_row_id"]): row for row in trajectory}
        if len(trajectory_by_id) != 720: raise ValueError(f"{run}:{epoch}: trajectory IDs")
        by_id, positions = {}, set()
        for raw in channels:
            if set(raw) != set(SIDECAR_FIELDS): raise ValueError(f"{run}:{epoch}: sidecar schema")
            identifier, position = str(raw["id"]), raw["dev_position"]
            if identifier != str(raw["source_row_id"]) or identifier in by_id:
                raise ValueError(f"{run}:{epoch}: ID mismatch")
            if type(position) is not int or position in positions or not 0 <= position < 720:
                raise ValueError(f"{run}:{epoch}: position mismatch")
            if raw["epoch"] != epoch or raw["training_seed"] != seed or raw["frame_downstream_gradient_mode"] != mode:
                raise ValueError(f"{run}:{epoch}: identity metadata")
            old = trajectory_by_id.get(identifier)
            logits = b1c.vector3(old.get("final_logits") if old else None, run)
            if (old is None or old["dev_position"] != position
                    or old["gold_final_label"] != raw["gold_label"]
                    or old["predicted_final_label"] != raw["prediction"]
                    or not b1c.same(logits[2], scalar(raw, "support_logit"))
                    or not b1c.same(logits[1], scalar(raw, "not_entitled_logit"))
                    or not b1c.same(b1c.sigmoid(float(old["frame_logit"])),
                                     scalar(raw, "frame_probability", True))):
                raise ValueError(f"{run}:{epoch}: cross-export mismatch")
            row = {"id": identifier, "position": position, "gold": raw["gold_label"],
                   "prediction": raw["prediction"], "intervention_type": raw["intervention_type"],
                   "frame": scalar(raw, "frame_probability", True),
                   "predicate": scalar(raw, "predicate_coverage_probability", True),
                   "sufficiency": scalar(raw, "sufficiency_probability", True),
                   "polarity": scalar(raw, "polarity_support_margin"),
                   "entitlement": scalar(raw, "entitlement_probability", True),
                   "support_probability": scalar(raw, "support_probability", True),
                   "not_entitled_probability": scalar(raw, "not_entitled_probability", True),
                   "support_logit": scalar(raw, "support_logit"),
                   "not_entitled_logit": scalar(raw, "not_entitled_logit")}
            row["margin"] = row["support_logit"] - row["not_entitled_logit"]
            by_id[identifier], positions = row, positions | {position}
        if set(by_id) != set(selected_raw) or set(trajectory_by_id) != set(selected_raw):
            raise ValueError(f"{run}:{epoch}: population mismatch")
        mapping = {row["position"]: identifier for identifier, row in by_id.items()}
        if reference_mapping is None: reference_mapping = mapping
        elif mapping != reference_mapping: raise ValueError(f"{run}: epoch drift")
        epochs[epoch] = by_id
    selected = epochs[best]
    for identifier, raw in selected_raw.items():
        row, probs = selected[identifier], b1c.vector3(raw["final_probs"], run, True)
        checks = (raw["gold_final_label"] == row["gold"], raw["pred_final_label"] == row["prediction"],
                  raw["intervention_type"] == row["intervention_type"],
                  b1c.same(float(raw["frame_prob"]), row["frame"]),
                  b1c.same(float(raw["predicate_coverage_prob"]), row["predicate"]),
                  b1c.same(float(raw["sufficiency_prob"]), row["sufficiency"]),
                  b1c.same(float(raw["polarity_margin"]), row["polarity"]),
                  b1c.same(float(raw["entitlement_prob"]), row["entitlement"]),
                  b1c.same(probs[2], row["support_probability"]),
                  b1c.same(probs[1], row["not_entitled_probability"]))
        if not all(checks): raise ValueError(f"{run}:{identifier}: selected mismatch")
    add_gate(gates, "alignment", run, "p0_sidecar_alignment", "20 x 720", "20 x 720", True, "")
    return {"run": run, "seed": seed, "mode": mode, "best_epoch": best,
            "selected": selected, "epochs": epochs, "position_to_id": reference_mapping}


def validate_p0_all(args: argparse.Namespace, gates: list[dict[str, Any]]) -> dict[str, Any]:
    repo, root, stage_a_path, stage_c_path, _ = safe_paths(args)
    b1c.validate_source(repo, args.current_git_commit, args.stage196b1_runtime_git_commit, gates)
    if not re.fullmatch(r"[0-9a-f]{40}", args.stage196b2p0_runtime_git_commit or ""):
        raise ValueError("invalid Stage196-B2-P0 runtime commit")
    entries = [path.name for path in root.iterdir()]
    add_gate(gates, "source", "", "six_p0_runs", list(RUNS), entries,
             set(entries) == set(RUNS), "P0 run matrix mismatch")
    stage_c, paired = close_b1c(stage_c_path, gates)
    stage_a, sets = b1c.validate_stage196a(stage_a_path, gates)
    runs = {run: load_p0_run(root, run, args.stage196b2p0_runtime_git_commit, gates) for run in RUNS}
    reference = runs[RUNS[0]]
    for run in RUNS[1:]:
        current = runs[run]
        if current["position_to_id"] != reference["position_to_id"]:
            raise ValueError(f"{run}: cross-run position mapping")
        for identifier in reference["selected"]:
            if (current["selected"][identifier]["gold"] != reference["selected"][identifier]["gold"]
                    or current["selected"][identifier]["intervention_type"]
                    != reference["selected"][identifier]["intervention_type"]):
                raise ValueError(f"{run}: cross-run metadata")
    positive, negative, deltas = derive_positive_seeds(paired, gates)
    return {"stage196a": stage_a, "stage196b1c": stage_c, "sets": sets, "runs": runs,
            "positive": positive, "negative": negative, "frame_deltas": deltas}
def tail_pattern(data: dict[str, Any], identifier: str) -> tuple[str, str, str]:
    return tuple(data["epochs"][epoch][identifier]["prediction"] for epoch in TAIL)  # type: ignore[return-value]


def view_row(data: dict[str, Any], identifier: str, view: str) -> dict[str, Any]:
    if view == "selected":
        row = dict(data["selected"][identifier])
        row["status"] = status(tail_pattern(data, identifier))
        row["class_or_pattern"] = row["prediction"]
        return row
    source = [data["epochs"][epoch][identifier] for epoch in TAIL]
    row = dict(source[-1])
    for key in ("frame", "predicate", "sufficiency", "polarity", "entitlement",
                "support_probability", "not_entitled_probability", "support_logit",
                "not_entitled_logit", "margin"):
        row[key] = math.fsum(item[key] for item in source) / 3
    pattern_value = tuple(item["prediction"] for item in source)
    row["status"] = status(pattern_value)
    row["prediction"] = pattern_value[-1]
    row["class_or_pattern"] = list(pattern_value)
    return row


def pass_map(row: dict[str, Any]) -> dict[str, bool]:
    return {"frame": row["frame"] >= THRESHOLD, "predicate": row["predicate"] >= THRESHOLD,
            "sufficiency": row["sufficiency"] >= THRESHOLD, "polarity": row["polarity"] >= 0,
            "entitlement": row["entitlement"] >= THRESHOLD}


def transition_record(seed: int, identifier: str, view: str,
                      joint_data: dict[str, Any], intervention_data: dict[str, Any],
                      sets: dict[str, set[int]]) -> dict[str, Any]:
    joint, intervention = view_row(joint_data, identifier, view), view_row(intervention_data, identifier, view)
    jp, ip = pass_map(joint), pass_map(intervention)
    crossings = {key: crossing(jp[key], ip[key]) for key in jp}
    category = ""
    if joint["gold"] == "SUPPORT" and intervention["frame"] - joint["frame"] > TOLERANCE:
        category = classify_propagation(intervention["frame"] - joint["frame"],
                                        intervention, crossings, view)
    first = category if category in FIRST_BLOCKER_ORDER else ""
    result = {
        "seed": seed, "stable_row_id": identifier, "dev_position": joint["position"],
        "analysis_view": view, "gold_final_label": joint["gold"],
        "intervention_type": joint["intervention_type"],
        "joint_selected_epoch": joint_data["best_epoch"],
        "intervention_selected_epoch": intervention_data["best_epoch"],
        "joint_final_class_or_pattern": joint["class_or_pattern"],
        "intervention_final_class_or_pattern": intervention["class_or_pattern"],
        "joint_tail3_status": joint["status"], "intervention_tail3_status": intervention["status"],
        "paired_transition_class": paired_transition(joint["status"], intervention["status"]),
        "propagation_category": category, "first_blocker": first,
    }
    mapping = {"frame": "frame", "predicate": "predicate", "sufficiency": "sufficiency",
               "polarity": "polarity", "entitlement": "entitlement",
               "support_vs_not_entitled_margin": "margin",
               "support_probability": "support_probability",
               "not_entitled_probability": "not_entitled_probability"}
    for channel, key in mapping.items():
        result[f"joint_{channel}_value"] = joint[key]
        result[f"intervention_{channel}_value"] = intervention[key]
        result[f"paired_{channel}_delta"] = intervention[key] - joint[key]
    for channel in ("frame", "predicate", "sufficiency", "polarity", "entitlement"):
        result[f"{channel}_joint_pass"] = jp[channel]
        result[f"{channel}_intervention_pass"] = ip[channel]
        result[f"{channel}_threshold_crossing"] = crossings[channel]
    for name in SETS: result[f"in_{name}"] = joint["position"] in sets[name]
    return result


def population_ids(seed: int, rows: list[dict[str, Any]],
                   sets: dict[str, set[int]]) -> dict[str, set[str]]:
    tail = {row["stable_row_id"]: row for row in rows if row["analysis_view"] == "tail3"}
    support = {identifier for identifier, row in tail.items() if row["gold_final_label"] == "SUPPORT"}
    baseline_stable = {identifier for identifier in support
                       if tail[identifier]["joint_tail3_status"] == "STABLE_SUPPORT"}
    harm = {identifier for identifier in baseline_stable
            if tail[identifier]["intervention_tail3_status"] != "STABLE_SUPPORT"}
    rescue = {identifier for identifier in support
              if tail[identifier]["paired_transition_class"] == "RESCUE_NE_TO_STABLE_SUPPORT"}
    result = {"all_gold_support": support,
              "joint_persistent_support_to_not_entitled":
                  {i for i in support if tail[i]["joint_tail3_status"] == "PERSISTENT_NOT_ENTITLED"},
              "intervention_persistent_support_to_not_entitled":
                  {i for i in support if tail[i]["intervention_tail3_status"] == "PERSISTENT_NOT_ENTITLED"},
              "baseline_defined_stable_correct_support_controls": baseline_stable,
              "intervention_induced_harm": harm, "rescue_rows": rescue}
    for name in SETS:
        result[name] = {i for i in support if tail[i]["dev_position"] in sets[name]}
    return result


def analyze_complete(args: argparse.Namespace, gates: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    context = validate_p0_all(args, gates)
    runs, sets = context["runs"], context["sets"]
    transitions: list[dict[str, Any]] = []
    populations: dict[int, dict[str, set[str]]] = {}
    for seed in SEEDS:
        joint, intervention = runs[f"seed{seed}_joint"], runs[f"seed{seed}_frame_local_only"]
        seed_rows = [transition_record(seed, identifier, view, joint, intervention, sets)
                     for identifier in joint["selected"] for view in ("selected", "tail3")]
        transitions.extend(seed_rows)
        populations[seed] = population_ids(seed, seed_rows, sets)
    summaries: list[dict[str, Any]] = []
    for seed in SEEDS:
        for view in ("selected", "tail3"):
            view_rows = [row for row in transitions if row["seed"] == seed and row["analysis_view"] == view]
            by_id = {row["stable_row_id"]: row for row in view_rows}
            for population in POPULATIONS:
                subset = [by_id[i] for i in populations[seed][population]]
                frame_up = [row for row in subset if row["paired_frame_delta"] > TOLERANCE]
                deltas = lambda name: [row[f"paired_{name}_delta"] for row in subset]
                for category in BLOCKERS:
                    count = sum(row["propagation_category"] == category for row in subset)
                    denominator = len(frame_up)
                    summaries.append({
                        "seed": seed, "analysis_view": view, "population": population,
                        "propagation_category": category, "count": count,
                        "denominator": denominator, "rate": count / denominator if denominator else None,
                        "frame_up_count": len(frame_up),
                        "frame_down_count": sum(row["paired_frame_delta"] < -TOLERANCE for row in subset),
                        "frame_fail_to_pass_count": sum(row["frame_threshold_crossing"] == "FAIL_TO_PASS" for row in subset),
                        "frame_pass_to_fail_count": sum(row["frame_threshold_crossing"] == "PASS_TO_FAIL" for row in subset),
                        "frame_up_predicate_positive_fraction": (sum(row["paired_predicate_delta"] > TOLERANCE for row in frame_up) / denominator if denominator else None),
                        "frame_up_entitlement_positive_fraction": (sum(row["paired_entitlement_delta"] > TOLERANCE for row in frame_up) / denominator if denominator else None),
                        "frame_up_margin_improved_fraction": (sum(row["paired_support_vs_not_entitled_margin_delta"] > TOLERANCE for row in frame_up) / denominator if denominator else None),
                        "selected_support_propagation_fraction": (sum(row["intervention_final_class_or_pattern"] == "SUPPORT" for row in frame_up) / denominator if denominator and view == "selected" else None),
                        "tail3_stable_support_propagation_fraction": (sum(row["intervention_tail3_status"] == "STABLE_SUPPORT" for row in frame_up) / denominator if denominator and view == "tail3" else None),
                        "rescue_count": len(populations[seed]["rescue_rows"]),
                        "harm_count": len(populations[seed]["intervention_induced_harm"]),
                        "persistent_failure_count": len(populations[seed]["intervention_persistent_support_to_not_entitled"]),
                        "spearman_frame_predicate": spearman(deltas("frame"), deltas("predicate")),
                        "spearman_frame_entitlement": spearman(deltas("frame"), deltas("entitlement")),
                        "spearman_frame_final_margin": spearman(deltas("frame"), deltas("support_vs_not_entitled_margin")),
                        "sign_concordance_frame_entitlement": (sum(signed(a) == signed(b) for a, b in zip(deltas("frame"), deltas("entitlement"))) / len(subset) if subset else None),
                        "sign_concordance_frame_final_margin": (sum(signed(a) == signed(b) for a, b in zip(deltas("frame"), deltas("support_vs_not_entitled_margin"))) / len(subset) if subset else None),
                    })
    recurrents = []
    transition_index = {(row["seed"], row["stable_row_id"], row["analysis_view"]): row for row in transitions}
    for seed in SEEDS:
        joint = runs[f"seed{seed}_joint"]
        for set_name in SETS:
            for position in sorted(sets[set_name]):
                identifier = joint["position_to_id"][position]
                selected = transition_index[(seed, identifier, "selected")]
                tail = transition_index[(seed, identifier, "tail3")]
                recurrents.append({"seed": seed, "recurrent_set": set_name,
                    "stable_row_id": identifier, "dev_position": position,
                    "intervention_type": selected["intervention_type"],
                    "selected_joint_frame": selected["joint_frame_value"],
                    "selected_intervention_frame": selected["intervention_frame_value"],
                    "selected_frame_delta": selected["paired_frame_delta"],
                    "selected_propagation_category": selected["propagation_category"],
                    "tail3_joint_frame": tail["joint_frame_value"],
                    "tail3_intervention_frame": tail["intervention_frame_value"],
                    "tail3_frame_delta": tail["paired_frame_delta"],
                    "tail3_joint_status": tail["joint_tail3_status"],
                    "tail3_intervention_status": tail["intervention_tail3_status"],
                    "tail3_propagation_category": tail["propagation_category"]})
    harm_rows = []
    for seed in SEEDS:
        for role, population in (("INTERVENTION_INDUCED_HARM", "intervention_induced_harm"),
                                 ("RESCUE", "rescue_rows")):
            for identifier in populations[seed][population]:
                tail, selected = transition_index[(seed, identifier, "tail3")], transition_index[(seed, identifier, "selected")]
                row = {"seed": seed, "stable_row_id": identifier, "dev_position": tail["dev_position"],
                       "transition_role": role, "intervention_type": tail["intervention_type"],
                       "joint_tail3_status": tail["joint_tail3_status"],
                       "intervention_tail3_status": tail["intervention_tail3_status"],
                       "selected_class_transition": [selected["joint_final_class_or_pattern"], selected["intervention_final_class_or_pattern"]],
                       "first_blocker": tail["first_blocker"],
                       "frame_increased_despite_harm": tail["paired_frame_delta"] > TOLERANCE,
                       "entitlement_increased_despite_harm": tail["paired_entitlement_delta"] > TOLERANCE,
                       "support_vs_ne_margin_decreased": tail["paired_support_vs_not_entitled_margin_delta"] < -TOLERANCE}
                for name in SETS: row[f"in_{name}"] = tail[f"in_{name}"]
                for channel in BASE_CHANNELS:
                    for prefix in ("joint", "intervention", "paired"):
                        suffix = "delta" if prefix == "paired" else "value"
                        row[f"{prefix}_{channel}_{suffix}"] = tail[f"{prefix}_{channel}_{suffix}"]
                for channel in ("frame", "predicate", "sufficiency", "polarity", "entitlement"):
                    row[f"{channel}_threshold_crossing"] = tail[f"{channel}_threshold_crossing"]
                harm_rows.append(row)
    epoch_rows = []
    for seed in SEEDS:
        joint, intervention = runs[f"seed{seed}_joint"], runs[f"seed{seed}_frame_local_only"]
        support_ids = [i for i, row in joint["selected"].items() if row["gold"] == "SUPPORT"]
        common_ids = [joint["position_to_id"][p] for p in sets["stage196a_common_recurrent"]]
        for epoch in EPOCHS:
            pairs = [(joint["epochs"][epoch][i], intervention["epochs"][epoch][i]) for i in support_ids]
            common = [(joint["epochs"][epoch][i], intervention["epochs"][epoch][i]) for i in common_ids]
            mean_delta = lambda key, values: math.fsum(b[key] - a[key] for a, b in values) / len(values)
            epoch_rows.append({"seed": seed, "epoch": epoch, "is_tail3_epoch": epoch in TAIL,
                "joint_selected_epoch": joint["best_epoch"], "intervention_selected_epoch": intervention["best_epoch"],
                "is_joint_selected_epoch": epoch == joint["best_epoch"],
                "is_intervention_selected_epoch": epoch == intervention["best_epoch"],
                "mean_delta_frame_gold_support": mean_delta("frame", pairs),
                "mean_delta_predicate_gold_support": mean_delta("predicate", pairs),
                "mean_delta_sufficiency_gold_support": mean_delta("sufficiency", pairs),
                "mean_delta_polarity_gold_support": mean_delta("polarity", pairs),
                "mean_delta_entitlement_gold_support": mean_delta("entitlement", pairs),
                "mean_delta_support_vs_ne_margin_gold_support": mean_delta("margin", pairs),
                "frame_fail_to_pass_count": sum(a["frame"] < .5 <= b["frame"] for a, b in pairs),
                "frame_pass_to_fail_count": sum(b["frame"] < .5 <= a["frame"] for a, b in pairs),
                "support_rescue_count": sum(a["prediction"] != "SUPPORT" and b["prediction"] == "SUPPORT" for a, b in pairs),
                "support_harm_count": sum(a["prediction"] == "SUPPORT" and b["prediction"] != "SUPPORT" for a, b in pairs),
                "false_not_entitled_delta": sum(b["prediction"] == "NOT_ENTITLED" for a, b in pairs) - sum(a["prediction"] == "NOT_ENTITLED" for a, b in pairs),
                "false_entitlement_delta": sum(b["gold"] == "NOT_ENTITLED" and b["prediction"] != "NOT_ENTITLED" for i, b in [(i, intervention["epochs"][epoch][i]) for i in intervention["epochs"][epoch]]) - sum(a["gold"] == "NOT_ENTITLED" and a["prediction"] != "NOT_ENTITLED" for i, a in [(i, joint["epochs"][epoch][i]) for i in joint["epochs"][epoch]]),
                "common_recurrent_frame_delta": mean_delta("frame", common),
                "common_recurrent_entitlement_delta": mean_delta("entitlement", common),
                "common_recurrent_support_vs_ne_margin_delta": mean_delta("margin", common)})
    if len(epoch_rows) != 60: raise ValueError("epoch propagation closure is not 60 rows")
    primary_distributions, harm_distributions = {}, {}
    for seed in context["positive"]:
        eligible = [transition_index[(seed, i, "tail3")] for i in populations[seed]["stage196a_common_recurrent"]
                    if transition_index[(seed, i, "tail3")]["paired_frame_delta"] > TOLERANCE
                    and transition_index[(seed, i, "tail3")]["intervention_tail3_status"] != "STABLE_SUPPORT"]
        primary_distributions[seed] = Counter(row["first_blocker"] for row in eligible)
        harm_distributions[seed] = Counter(transition_index[(seed, i, "tail3")]["first_blocker"]
                                           for i in populations[seed]["intervention_induced_harm"])
    decision, rule = choose_decision(context["positive"], primary_distributions, harm_distributions, False, False)
    seed_rows = []
    for seed in SEEDS:
        counts = primary_distributions.get(seed, Counter())
        largest = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0] if counts else ("", 0)
        total = sum(counts.values())
        seed_rows.append({"seed": seed, "common_recurrent_frame_shift": "positive" if seed in context["positive"] else "negative",
            "primary_decision_population_size": total, "largest_blocker": largest[0],
            "largest_blocker_rate": largest[1] / total if total else None,
            "rescue_count": len(populations[seed]["rescue_rows"]),
            "harm_count": len(populations[seed]["intervention_induced_harm"]),
            "selected_recurrent_propagation_rate": None, "tail3_recurrent_propagation_rate": None,
            "selected_tail3_agreement": None,
            "decision_contribution": "primary" if seed in context["positive"] else "contrast"})
    report = {"stage": STAGE, "decision": decision, "recommended_next_stage": NEXT_STAGE[decision],
        "blocking_reasons": [], "runnable": True,
        "analysis_runtime_git_commit": args.current_git_commit,
        "stage196b1_runtime_git_commit": args.stage196b1_runtime_git_commit,
        "stage196b2p0_runtime_git_commit": args.stage196b2p0_runtime_git_commit,
        "framegate_implementation_origin_git_commit": FRAMEGATE_IMPLEMENTATION_ORIGIN_COMMIT,
        "stage196b1c_source_decision": B1C_DECISION,
        "positive_frame_shift_seeds": context["positive"],
        "negative_frame_shift_contrast_seeds": context["negative"],
        "resolved_schema": RESOLVED_SCHEMA | {"epoch_channel_sidecar": list(SIDECAR_FIELDS)},
        "support_vs_not_entitled_margin_source": "support_logit - not_entitled_logit",
        "primary_decision_population_counts": {str(seed): sum(primary_distributions.get(seed, {}).values()) for seed in context["positive"]},
        "per_seed_blocker_distributions": {str(seed): dict(counts) for seed, counts in primary_distributions.items()},
        "decision_rule_evaluation": rule,
        "authorized_interpretation": "Observed paired propagation under frozen Mamba only; no formal causal mediation claim.",
        "prohibited_interpretations": ["unfrozen encoder behavior", "external/OOD performance", "production readiness", "contrastive-loss necessity", "architecture superiority", "complete mechanistic explanation"],
        "output_file_count": 9, "training_performed": False, "checkpoint_loaded": False,
        "model_loaded": False, "external_evaluation_performed": False, "artifact_only_analysis": True}
    return report, {"seed": seed_rows, "transition": transitions, "summary": summaries,
                    "recurrent": recurrents, "harm": harm_rows, "epoch": epoch_rows,
                    "contract": gates}
def empty_tables(gates: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {"seed": [], "transition": [], "summary": [], "recurrent": [], "harm": [],
            "epoch": [], "contract": gates}


def incomplete_report(args: argparse.Namespace, exc: BaseException,
                      gates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "stage": STAGE, "decision": INCOMPLETE,
        "recommended_next_stage": NEXT_STAGE[INCOMPLETE],
        "blocking_reasons": [f"{type(exc).__name__}: {exc}"], "runnable": False,
        "analysis_runtime_git_commit": args.current_git_commit,
        "stage196b1_runtime_git_commit": args.stage196b1_runtime_git_commit,
        "stage196b2p0_runtime_git_commit": args.stage196b2p0_runtime_git_commit,
        "framegate_implementation_origin_git_commit": FRAMEGATE_IMPLEMENTATION_ORIGIN_COMMIT,
        "stage196b1c_source_decision": B1C_DECISION,
        "positive_frame_shift_seeds": [], "negative_frame_shift_contrast_seeds": [],
        "resolved_schema": RESOLVED_SCHEMA,
        "support_vs_not_entitled_margin_source": "support_logit - not_entitled_logit",
        "primary_decision_population_counts": {}, "per_seed_blocker_distributions": {},
        "decision_rule_evaluation": {"selected_decision": INCOMPLETE,
                                     "correlation_alone_used": False},
        "authorized_interpretation": "No scientific propagation or mediation claim is authorized.",
        "prohibited_interpretations": [
            "formal causal mediation from observational correlation", "promotion of frame_local_only",
            "unfrozen encoder behavior", "external or OOD performance", "production readiness",
            "contrastive-loss necessity", "architecture superiority", "complete mechanistic explanation",
            "authorization of a new loss, trainer modification, or full retraining",
        ],
        "selected_tail3_epoch_views_distinct": True, "tail_epochs": list(TAIL),
        "expected_epoch_propagation_rows": 60, "actual_epoch_propagation_rows": 0,
        "output_file_count": 9, "training_performed": False, "checkpoint_loaded": False,
        "model_loaded": False, "external_evaluation_performed": False,
        "artifact_only_analysis": True,
        "exception": {"type": type(exc).__name__, "message": str(exc),
                      "traceback": traceback.format_exc()},
    }


def markdown(report: dict[str, Any]) -> str:
    if report["decision"] != INCOMPLETE:
        sections = [
            ("Executive decision", f"`{report['decision']}`"),
            ("Authorized interpretation", report["authorized_interpretation"]),
            ("Source and provenance closure", "Original Stage196-B1, Stage196-B2-P0 rerun, analyzer, and implementation-origin commit roles are distinct; all contracts passed."),
            ("Stage196-B1-C mixed result recap", f"Prior decision: `{B1C_DECISION}`."),
            ("Exact analysis populations", "Gold SUPPORT and every precommitted recurrent, persistent, control, harm, and rescue population remain separate."),
            ("Positive-frame-shift seed derivation", f"Derived positive seeds: {report['positive_frame_shift_seeds']}; negative contrast: {report['negative_frame_shift_contrast_seeds']}."),
            ("Selected-checkpoint propagation", "Uses each P0 run's actual selected epoch and its cross-validated native sidecar."),
            ("Tail-three propagation", "Uses exact epochs 18?20, native scalar means, and ordered prediction patterns."),
            ("Common recurrent-set analysis", "Reported in the recurrent-position and transition tables."),
            ("Universal all-six analysis", "Reported separately in the recurrent-position table."),
            ("Stable-correct SUPPORT harm analysis", "Controls are defined exclusively from joint-arm stable SUPPORT."),
            ("Rescue analysis", "Rescue requires joint persistent NOT_ENTITLED and intervention stable SUPPORT."),
            ("First-blocker distributions", "Fixed order: frame, predicate, sufficiency, polarity, entitlement aggregation, final composition."),
            ("Epoch propagation", "The epoch table contains exactly 60 aligned paired rows."),
            ("Seed183 contrast analysis", "Seed183 remains a direction-visible negative-frame-shift contrast."),
            ("Decision-rule evaluation", json.dumps(report["decision_rule_evaluation"], sort_keys=True)),
            ("Remaining uncertainty", "Observed propagation is descriptive under frozen Mamba and is not formal causal mediation."),
            ("Prohibited claims", "\n".join(f"- {claim}" for claim in report["prohibited_interpretations"])),
            ("Recommended next stage", f"`{report['recommended_next_stage']}`\n\nNo training, loss, or promotion is automatically authorized."),
        ]
        return "# Stage196-B2-A FrameGate-to-entitlement propagation mediation audit\n\n" + "\n\n".join(
            f"## {title}\n\n{body}" for title, body in sections) + "\n"
    reason = " ".join(report["blocking_reasons"])
    sections = [
        ("Executive decision", f"`{report['decision']}`\n\nThe audit failed closed. `{B1C_DECISION}` does not authorize promotion of `frame_local_only`."),
        ("Authorized interpretation", report["authorized_interpretation"]),
        ("Source and provenance closure", "The analyzer requires the exact six-run frozen-Mamba matrix, the caller-supplied analysis and historical runtime commits, the implementation-origin commit, explicit B1-C companions, and Stage196-A recurrence companions. Failure: " + reason),
        ("Stage196-B1-C mixed result recap", f"Required source decision: `{B1C_DECISION}`. Required next stage: `{B1C_NEXT}`. Blocking reasons must be empty."),
        ("Exact analysis populations", "Gold SUPPORT is isolated from other gold labels. Stage196-A baseline recurrent (22), intervention recurrent (19), common recurrent (19), and universal all-six (10) sets are loaded. Joint/intervention persistent NE, baseline-defined stable-correct controls, intervention-induced harm, and rescue populations are defined without aggregate redefinition."),
        ("Positive-frame-shift seed derivation", "Seeds are derived from the B1-C paired-delta row `mean_frame_probability_stage196a_common_recurrent`; no seed number is used as the derivation rule."),
        ("Selected-checkpoint propagation", "Selected outputs use the actual per-run selected checkpoint export. They are never replaced by epoch 20."),
        ("Tail-three propagation", "Tail-three means exactly epochs 18, 19, and 20. Native Stage196-B2-P0 sidecars are mandatory; selected values are never substituted."),
        ("Common recurrent-set analysis", "Not produced because required Stage196-B2-P0 inputs failed closure."),
        ("Universal all-six analysis", "Not produced because required Stage196-B2-P0 inputs failed closure."),
        ("Stable-correct SUPPORT harm analysis", "Controls remain defined from joint-arm stable SUPPORT only; incomplete schema prevents row-level propagation attribution."),
        ("Rescue analysis", "A rescue requires joint persistent NOT_ENTITLED and intervention stable SUPPORT; increased frame probability alone is never a rescue."),
        ("First-blocker distributions", "Fixed order: frame remains subthreshold, predicate, sufficiency, polarity, entitlement aggregation, final SUPPORT-vs-NOT_ENTITLED composition. Multi-channel degradation is separate."),
        ("Epoch propagation", "Exactly 60 paired seed-by-epoch rows are required. Zero scientific rows are emitted on incomplete schema; the contract records the failure."),
        ("Seed183 contrast analysis", "Seed183 is derived as the negative-shift contrast only after B1-C delta closure; no contrast conclusion is issued from incomplete inputs."),
        ("Decision-rule evaluation", "A bottleneck requires two positive-shift seeds, at least five eligible rows per seed, the same largest blocker at at least 50% in every positive seed, and no two-seed harm conflict. Correlation and p-values never decide the outcome."),
        ("Remaining uncertainty", "Native predicate, sufficiency, polarity, and entitlement values for every epoch are absent from the frozen Stage196-B1 trajectory exports. Recovering them would require a new authorized artifact source or replay; this analyzer does neither."),
        ("Prohibited claims", "\n".join(f"- {claim}" for claim in report["prohibited_interpretations"])),
        ("Recommended next stage", f"`{report['recommended_next_stage']}`\n\nThis repair recommendation does not authorize training, replay, checkpoint loading, a new loss, trainer changes, or promotion."),
    ]
    return "# Stage196-B2-A FrameGate-to-entitlement propagation mediation audit\n\n" + "\n\n".join(
        f"## {title}\n\n{body}" for title, body in sections) + "\n"


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, bool): return "true" if value else "false"
    return value


def render_csv(header: list[str], rows: Iterable[dict[str, Any]]) -> str:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=header, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if set(row) != set(header):
            raise ValueError(f"generated CSV schema mismatch: {set(row) ^ set(header)}")
        writer.writerow({key: csv_value(row[key]) for key in header})
    return handle.getvalue()


def render_outputs(report: dict[str, Any], tables: dict[str, list[dict[str, Any]]]) -> dict[str, str]:
    return {
        "stage196b2a_analysis.json": json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        "stage196b2a_report.md": markdown(report),
        "stage196b2a_seed_summary.csv": render_csv(SEED_HEADER, tables["seed"]),
        "stage196b2a_support_transition_rows.csv": render_csv(TRANSITION_HEADER, tables["transition"]),
        "stage196b2a_channel_transition_summary.csv": render_csv(SUMMARY_HEADER, tables["summary"]),
        "stage196b2a_recurrent_position_propagation.csv": render_csv(RECURRENT_HEADER, tables["recurrent"]),
        "stage196b2a_harm_rescue_rows.csv": render_csv(HARM_HEADER, tables["harm"]),
        "stage196b2a_epoch_propagation.csv": render_csv(EPOCH_HEADER, tables["epoch"]),
        "stage196b2a_contract.csv": render_csv(CONTRACT_HEADER, tables["contract"]),
    }


def write_exact(output: Path, rendered: dict[str, str]) -> None:
    if set(rendered) != set(OUTPUTS):
        raise ValueError("internal nine-output closure mismatch")
    output.mkdir(parents=True, exist_ok=False)
    for name in OUTPUTS:
        descriptor = os.open(output / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(rendered[name])
    if {path.name for path in output.iterdir()} != set(OUTPUTS):
        raise RuntimeError("written output closure mismatch")


def main() -> int:
    args = parse_args()
    gates: list[dict[str, Any]] = []
    try:
        report, tables = analyze_complete(args, gates)
    except Exception as exc:
        gates.append({"scope": "analysis", "run": "", "gate": "analysis_completed",
                      "required": True, "observed": False, "passed": False,
                      "blocking_reason": f"{type(exc).__name__}: {exc}"})
        report = incomplete_report(args, exc, gates)
        tables = empty_tables(gates)
    rendered = render_outputs(report, tables)
    output = args.output_dir.resolve()
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise ValueError("output directory exists and is nonempty")
        output.rmdir()
    write_exact(output, rendered)
    print(json.dumps({"decision": report["decision"], "output_dir": str(output),
                      "output_files": list(OUTPUTS)}, sort_keys=True))
    return 0 if report["decision"] != INCOMPLETE else 2


if __name__ == "__main__":
    raise SystemExit(main())
